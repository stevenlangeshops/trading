"""
backfill_predictions.py
══════════════════════════════════════════════════════════════════════════════
Einmaliges Backfill-Skript: Füllt live_predictions_history.csv mit
historischen Modell-Vorhersagen auf, damit update_ic.py die IC-Lücke
seit dem letzten Kaggle-Lauf schließen kann.

Strategie (effizient, kein Look-Ahead):
  1. OHLCV einmalig via yfinance für den gesamten Zeitraum laden.
  2. Feature-Panel einmalig aufbauen (sector-neutral, identisch zum Training).
     Die Sektor-neutrale Z-Score-Normalisierung ist per-Tag cross-sectional
     → kein Look-Ahead über Tage.
  3. Alle Backfill-Daten in einer Schleife durch das Modell jagen (schnell,
     da Panel bereits gebaut ist).
  4. Predictions an live_predictions_history.csv anhängen (Duplikate überspringen).

Verwendung:
    python backfill_predictions.py                        # Standardlauf
    python backfill_predictions.py --start 2026-02-10     # Abweichendes Startdatum
    python backfill_predictions.py --dry-run              # Zeigt Aktionen ohne zu schreiben
    python backfill_predictions.py --device cuda          # GPU nutzen
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_REPO_ROOT = Path(__file__).parent.resolve()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ── Produktions-Defaults (müssen mit live_inference.py übereinstimmen) ───────
_DEFAULT_START    = "2026-02-10"   # Erster Tag der IC-Lücke
_HORIZON          = 7              # Vorhersage-Horizont (Handelstage)
_SEQ_LEN          = 64             # LSTM-Lookback-Fenster
_DOWNLOAD_DAYS    = 420            # SMA200(200) + seq_len(64) + Backfill-Puffer(156)
_CKPT_DIR         = _REPO_ROOT / "checkpoints" / "v2_7d"
_WALK_JSON        = _CKPT_DIR / "v2_7d_walk_forward.json"
_SECTOR_MAP       = _REPO_ROOT / "features" / "sector_map.json"
_ASSET_MAP        = _REPO_ROOT / "checkpoints" / "v2_7d" / "asset_map.json"
_PREDICTIONS_CSV  = _REPO_ROOT / "live_predictions_history.csv"


# ══════════════════════════════════════════════════════════════════════════════
# Hilfs-Importe aus live_inference.py
# ══════════════════════════════════════════════════════════════════════════════

def _import_live_helpers():
    """Importiert Hilfsfunktionen aus live_inference.py."""
    from live_inference import (
        download_ohlcv,
        build_live_features,
        _find_artifact,
        _find_ckpt_dir,
        _load_asset_map,
        save_predictions,
        _latest_trading_day,
    )
    return (download_ohlcv, build_live_features, _find_artifact,
            _find_ckpt_dir, _load_asset_map, save_predictions,
            _latest_trading_day)


# ══════════════════════════════════════════════════════════════════════════════
# Handelstage-Hilfsfunktionen
# ══════════════════════════════════════════════════════════════════════════════

def _trading_days_in_range(
    start: pd.Timestamp,
    end:   pd.Timestamp,
) -> list[pd.Timestamp]:
    """Gibt alle Werktage (Mo-Fr) zwischen start und end zurück.

    Args:
        start: Erster Tag (einschließlich).
        end:   Letzter Tag (einschließlich).

    Returns:
        Liste aufsteigend sortierter Werktage.
    """
    return list(pd.bdate_range(start, end))


def _trading_days_ago(ref: pd.Timestamp, n: int) -> pd.Timestamp:
    """Gibt den Handelstag n Werktage vor ref zurück."""
    ts = ref
    counted = 0
    while counted < n:
        ts -= pd.Timedelta(days=1)
        if ts.weekday() < 5:
            counted += 1
    return ts


# ══════════════════════════════════════════════════════════════════════════════
# Modell laden
# ══════════════════════════════════════════════════════════════════════════════

def load_model_for_backfill(ckpt_dir: Path, walk_json: Path, device: str):
    """Lädt den neuesten Fold-Checkpoint (identisch zu live_inference.py).

    Wir verwenden fold_11_best.pt (val_end=2026-02-09), d.h. das Modell
    hat nie Daten nach diesem Datum gesehen → kein Look-Ahead für den
    Backfill-Zeitraum ab 2026-02-10.

    Args:
        ckpt_dir:  Verzeichnis mit ``fold_*_best.pt``-Dateien.
        walk_json: Pfad zu ``v2_7d_walk_forward.json``.
        device:    Torch-Device (``"cpu"`` oder ``"cuda"``).

    Returns:
        Tuple ``(model, latest_fold_meta)``.
    """
    import json
    from backtest_v2_single_horizon import load_fold_model

    data  = json.loads(walk_json.read_text())
    folds = data.get("fold_summary") or data.get("fold_results", [])

    def _val_end(f):
        return pd.Timestamp(f.get("val_end", "1970-01-01"))

    latest = sorted(folds, key=_val_end)[-1]
    fname  = Path(latest.get("ckpt_path", "")).name or \
             f"fold_{latest.get('fold_id', 0)}_best.pt"
    ckpt   = ckpt_dir / fname

    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint nicht gefunden: {ckpt}")

    print(f"  Checkpoint: {fname}  (val_end={_val_end(latest).date()})")
    model, _ = load_fold_model(str(ckpt), device)
    model.eval()
    return model, latest


# ══════════════════════════════════════════════════════════════════════════════
# Kern-Logik: Panel einmal aufbauen, alle Tage scoren
# ══════════════════════════════════════════════════════════════════════════════

def build_backfill_panel(
    backfill_end:  pd.Timestamp,
    download_days: int,
    sector_map:    dict,
    tickers:       list[str],
) -> pd.DataFrame:
    """Baut den Feature-Panel für den gesamten Backfill-Zeitraum.

    Der Panel endet am ``backfill_end``-Datum und deckt ausreichend
    Warmup-Historie für SMA200 + seq_len ab.

    Args:
        backfill_end:  Letzter Backfill-Tag (= heute - 7 Handelstage).
        download_days: Anzahl Handelstage Rückblick für den Download.
        sector_map:    GICS-Sektorzuordnung ``{ticker → sektor}``.
        tickers:       Vollständige Ticker-Liste.

    Returns:
        MultiIndex-DataFrame ``(date, asset) × FEATURE_COLS``,
        sektor-neutral normalisiert.
    """
    (download_ohlcv, build_live_features, *_) = _import_live_helpers()

    print(f"  OHLCV Download: {download_days} Handelstage Rückblick ...")
    ohlcv_map = download_ohlcv(
        tickers  = tickers,
        days     = download_days,
        end_date = backfill_end.date(),
    )
    if not ohlcv_map:
        raise RuntimeError("Kein OHLCV-Daten erhalten.")

    print(f"  Feature-Panel aufbauen (Sektor-Neutral) ...")
    panel, valid = build_live_features(
        ohlcv_map   = ohlcv_map,
        sector_map  = sector_map,
        target_date = backfill_end,
        min_history = 220,
    )
    if panel.empty:
        raise RuntimeError("Feature-Panel ist leer.")

    panel_dates = panel.index.get_level_values("date").unique()
    print(f"  Panel: {len(valid)} Assets  |  "
          f"{panel_dates.min().date()} bis {panel_dates.max().date()}")
    return panel


def score_all_backfill_dates(
    panel:          pd.DataFrame,
    model,
    asset_map:      dict[str, int],
    backfill_dates: list[pd.Timestamp],
    seq_len:        int,
    device:         str,
) -> dict[pd.Timestamp, pd.Series]:
    """Scort alle Backfill-Daten in einer Schleife.

    Für jeden Tag wird ``predict_cross_section()`` aufgerufen.  Das Panel
    ist bereits vollständig aufgebaut, daher ist dieser Schritt schnell
    (~1–3 Sekunden pro Tag auf CPU).

    Args:
        panel:          Feature-Panel aus ``build_backfill_panel()``.
        model:          Geladenes Modell im eval()-Modus.
        asset_map:      ``{ticker → modell_id}``-Mapping.
        backfill_dates: Zu scornde Handelstage (aufsteigend sortiert).
        seq_len:        LSTM-Lookback-Fenster.
        device:         Torch-Device.

    Returns:
        Dict ``{date → pd.Series(ticker → score)}``.
    """
    import torch
    from backtest_v2_single_horizon import predict_cross_section

    panel_dates = set(panel.index.get_level_values("date").unique())

    results: dict[pd.Timestamp, pd.Series] = {}
    n_skip = 0

    for date in backfill_dates:
        # Nächsten verfügbaren Panel-Tag finden
        if date not in panel_dates:
            candidates = [d for d in panel_dates if d <= date]
            if not candidates:
                n_skip += 1
                continue
            effective_date = max(candidates)
        else:
            effective_date = date

        with torch.no_grad():
            scores = predict_cross_section(
                model     = model,
                features  = panel,
                asset_map = asset_map,
                date      = effective_date,
                seq_len   = seq_len,
                device    = device,
            )

        if scores.empty:
            n_skip += 1
            continue

        results[date] = scores   # unter dem Vorhersage-Datum speichern

    if n_skip:
        print(f"  [WARN] {n_skip} Tage übersprungen (kein Panel-Eintrag / zu kurz)")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Bereits vorhandene Predictions laden (Idempotenz)
# ══════════════════════════════════════════════════════════════════════════════

def _existing_pred_dates(csv_path: Path) -> set[pd.Timestamp]:
    """Gibt alle Daten zurück, für die bereits Predictions existieren."""
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path, usecols=["date"], parse_dates=["date"])
    return set(df["date"].dt.normalize())


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Historische Predictions für live_predictions_history.csv aufbauen.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--start",         default=_DEFAULT_START,
                   help="Erstes Backfill-Datum (YYYY-MM-DD)")
    p.add_argument("--end",           default=None,
                   help="Letztes Backfill-Datum (Standard: heute - 7 Handelstage)")
    p.add_argument("--download-days", type=int, default=_DOWNLOAD_DAYS,
                   help="OHLCV-Rückblick in Handelstagen (inkl. Warmup-Puffer)")
    p.add_argument("--ckpt-dir",      default=str(_CKPT_DIR),
                   help="Verzeichnis mit fold_*_best.pt Checkpoints")
    p.add_argument("--walk-json",     default=str(_WALK_JSON))
    p.add_argument("--asset-map",     default=str(_ASSET_MAP))
    p.add_argument("--sector-map",    default=str(_SECTOR_MAP))
    p.add_argument("--pred-csv",      default=str(_PREDICTIONS_CSV),
                   help="Ziel-CSV: live_predictions_history.csv")
    p.add_argument("--seq-len",       type=int, default=_SEQ_LEN)
    p.add_argument("--device",        default=None,
                   help="cuda oder cpu (auto wenn leer)")
    p.add_argument("--dry-run",       action="store_true",
                   help="Zeigt was gespeichert würde, ohne zu schreiben")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # ── Device ───────────────────────────────────────────────────────────────
    device = args.device
    if not device:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    # ── Datums-Berechnung ─────────────────────────────────────────────────────
    (_, _, _find_artifact, _find_ckpt_dir,
     _load_asset_map, save_predictions,
     _latest_trading_day) = _import_live_helpers()

    today           = _latest_trading_day()
    backfill_start  = pd.Timestamp(args.start).normalize()
    backfill_end    = (
        pd.Timestamp(args.end).normalize() if args.end
        else _trading_days_ago(today, _HORIZON)
    )

    if backfill_end < backfill_start:
        print(f"[FEHLER] End-Datum {backfill_end.date()} < Start-Datum {backfill_start.date()}")
        return 1

    all_dates     = _trading_days_in_range(backfill_start, backfill_end)
    existing_preds = _existing_pred_dates(Path(args.pred_csv))
    pending_dates  = [d for d in all_dates if pd.Timestamp(d).normalize() not in existing_preds]

    print("=" * 60)
    print(f"  Backfill Predictions  |  Device={device}")
    if args.dry_run:
        print("  [DRY-RUN] – es wird nichts geschrieben")
    print("=" * 60)
    print(f"\n  Zeitraum     : {backfill_start.date()} bis {backfill_end.date()}")
    print(f"  Handelstage  : {len(all_dates)}")
    print(f"  Bereits haben: {len(all_dates) - len(pending_dates)}")
    print(f"  Zu berechnen : {len(pending_dates)}")

    if not pending_dates:
        print("\n  Alles aktuell – nichts zu tun.")
        return 0

    # ── Metadaten laden ───────────────────────────────────────────────────────
    print("\n[1/4] Metadaten laden ...")
    from features.engineer import load_sector_map
    sector_map = load_sector_map(Path(args.sector_map))
    asset_map  = _load_asset_map(Path(args.asset_map), sector_map_path=Path(args.sector_map))
    tickers    = sorted(asset_map.keys())
    print(f"  {len(tickers)} Ticker  |  {len(sector_map)} in sector_map")

    # ── OHLCV laden & Feature-Panel aufbauen ──────────────────────────────────
    print("\n[2/4] OHLCV laden & Feature-Panel aufbauen ...")
    panel = build_backfill_panel(
        backfill_end  = backfill_end,
        download_days = args.download_days,
        sector_map    = sector_map,
        tickers       = tickers,
    )

    # ── Modell laden ──────────────────────────────────────────────────────────
    print("\n[3/4] Modell laden ...")
    ckpt_dir  = _find_ckpt_dir(Path(args.ckpt_dir)) or Path(args.ckpt_dir)
    walk_json = _find_artifact(Path(args.walk_json), "v2_7d_walk_forward.json") or Path(args.walk_json)
    model, fold_meta = load_model_for_backfill(ckpt_dir, walk_json, device)

    # ── Alle Tage scoren ──────────────────────────────────────────────────────
    print(f"\n[4/4] {len(pending_dates)} Tage scoren ...")
    scored = score_all_backfill_dates(
        panel          = panel,
        model          = model,
        asset_map      = asset_map,
        backfill_dates = [pd.Timestamp(d) for d in pending_dates],
        seq_len        = args.seq_len,
        device         = device,
    )

    if not scored:
        print("  [FEHLER] Keine Scores berechnet.")
        return 1

    print(f"  {len(scored)}/{len(pending_dates)} Tage erfolgreich gescort")

    # ── Predictions speichern ─────────────────────────────────────────────────
    pred_csv = Path(args.pred_csv)
    n_written = 0

    if not args.dry_run:
        for date, scores in sorted(scored.items()):
            save_predictions(scores, date, pred_csv)
            n_written += 1

        print(f"\n  Gespeichert: {n_written} Tage -> {pred_csv.name}")

        # Zusammenfassung
        df_check = pd.read_csv(pred_csv, parse_dates=["date"])
        unique_dates = sorted(df_check["date"].dt.normalize().unique())
        print(f"  CSV enthält jetzt {len(unique_dates)} Vorhersage-Tage:")
        for d in unique_dates[-5:]:
            n = (df_check["date"].dt.normalize() == d).sum()
            top = df_check[df_check["date"].dt.normalize() == d].nlargest(3, "score")["ticker"].tolist()
            print(f"    {pd.Timestamp(d).date()}  {n} Ticker  Top-3: {', '.join(top)}")

    else:
        print(f"\n  [DRY-RUN] Würde {len(scored)} Tage schreiben:")
        for date, scores in sorted(scored.items()):
            top3 = scores.head(3).index.tolist()
            print(f"    {date.date()}  {len(scores)} Ticker  "
                  f"Top-3: {', '.join(top3)}")

    print(f"\n  Naechster Schritt: python update_ic.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
