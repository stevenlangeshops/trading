"""
live_inference.py
══════════════════════════════════════════════════════════════════════════════
Tägliche Live/Paper-Inference für den v2 Single-Horizon Trading Bot.

Ablauf:
  1. OHLCV-Daten der letzten 250 Tage herunterladen  (→ yfinance Placeholder)
  2. Sektor-Neutrale Features berechnen (engineer.py)
  3. Letzten Fold-Checkpoint laden und Cross-Section-Scores berechnen
  4. IC_roll_40 der letzten 40 Tage prüfen (A3-Policy)
  5. Ziel-Allokation auf die Konsole ausgeben

Platzhalter-Funktionen sind mit ``# TODO`` markiert und können Schritt für
Schritt durch echte API-Calls (yfinance, Alpaca, Interactive Brokers) ersetzt
werden.

Verwendung:
    python live_inference.py
    python live_inference.py --ckpt-dir checkpoints/v2_7d --top-n 5

Konfiguration:
    Alle Pfade und Parameter befinden sich im ``LiveConfig``-Block am Anfang.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Eigene Module (Repo-Root muss im Python-Path sein) ───────────────────────
# Bei Ausführung aus dem Repo-Root: python live_inference.py   → direkt OK.
# Bei Ausführung aus einem anderen Verzeichnis:
#   sys.path.insert(0, "/pfad/zum/repo")
_REPO_ROOT = Path(__file__).parent.resolve()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ══════════════════════════════════════════════════════════════════════════════
# Produktions-Konfiguration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LiveConfig:
    """Alle Pfade und Parameter für den täglichen Inference-Lauf.

    Passe die Felder an deine lokale Verzeichnisstruktur an, bevor du das
    Skript das erste Mal ausführst.  Alle Pfade können auch über CLI-Argumente
    überschrieben werden (siehe ``parse_args()``).

    Attributes:
        ckpt_dir:         Verzeichnis mit ``fold_*_best.pt`` Checkpoint-Dateien.
        walk_json:        Pfad zu ``v2_7d_walk_forward.json`` (Fold-Metadaten).
        asset_map_json:   Pfad zu ``asset_map.json`` (Ticker → Modell-ID).
        ic_history_csv:   Pfad zu ``rolling_ic_v2_7d.csv`` (historische IC-Werte).
        sector_map_json:  Pfad zu ``features/sector_map.json`` (GICS-Sektoren).
        horizon:          Vorhersage-Horizont des geladenen Modells (Standard: 7).
        seq_len:          LSTM-Lookback-Fenster in Handelstagen (Standard: 64).
        download_days:    Wie viele Tage OHLCV-Historie heruntergeladen werden.
        top_n:            Anzahl der Top-Assets im Output (= n_max).
        a3_policy_window: Fenster für IC_roll_40-Berechnung (A3-Policy).
        a3_reduced_n:     n_max wenn A3-Policy aktiv.
        device:           ``"cuda"`` oder ``"cpu"`` (``None`` = auto-detect).
        tickers:          Vollständige Ticker-Liste. ``None`` → aus asset_map laden.
    """

    ckpt_dir:         Path    = Path("checkpoints/v2_7d")
    walk_json:        Path    = Path("checkpoints/v2_7d/v2_7d_walk_forward.json")
    asset_map_json:   Path    = Path("asset_map.json")
    ic_history_csv:   Path    = Path("rolling_ic_v2_7d.csv")
    sector_map_json:  Path    = Path("features/sector_map.json")

    horizon:          int     = 7
    seq_len:          int     = 64
    download_days:    int     = 250     # > seq_len + Indikator-Warmup (~200)
    top_n:            int     = 5       # = PROD_N_MAX
    a3_policy_window: int     = 40      # IC_roll_40
    a3_reduced_n:     int     = 3       # n_max im Defensiv-Modus

    device:           Optional[str]   = None
    tickers:          Optional[list]  = field(default=None)


# ══════════════════════════════════════════════════════════════════════════════
# Schritt 1: OHLCV-Download via yfinance
# ══════════════════════════════════════════════════════════════════════════════

# Pflicht-Spalten die compute_indicators() erwartet (alle lowercase)
_REQUIRED_COLS = ["open", "high", "low", "close", "volume"]


def _extract_ticker_df(
    raw:    pd.DataFrame,
    ticker: str,
) -> Optional[pd.DataFrame]:
    """Extrahiert den OHLCV-Subtable für einen Ticker aus dem yfinance-Ergebnis.

    yfinance 1.x liefert bei Batch-Downloads immer einen MultiIndex:
        Level 0 = Preis-Typ (Close, High, Low, Open, Volume)
        Level 1 = Ticker-Symbol

    Bei einem einzelnen Ticker oder String-Input kommt ebenfalls ein
    MultiIndex (Level 1 enthält den Ticker einmalig).

    Args:
        raw:    Roh-DataFrame von ``yf.download()``.
        ticker: Ticker-Symbol (z.B. ``"AAPL"``).

    Returns:
        Subtable mit Spalten ``Close``, ``High``, ``Low``, ``Open``, ``Volume``
        oder ``None`` wenn Ticker nicht im DataFrame enthalten.
    """
    if not isinstance(raw.columns, pd.MultiIndex):
        # Kein MultiIndex → Einzelticker direkt zurückgeben
        return raw.copy()

    # Level 1 enthält Ticker-Symbole (yfinance 1.x Standard)
    try:
        return raw.xs(ticker, axis=1, level=1).copy()
    except KeyError:
        return None


def download_ohlcv(
    tickers:  list[str],
    days:     int = 250,
    end_date: Optional[date] = None,
) -> dict[str, pd.DataFrame]:
    """Lädt OHLCV-Tagesdaten für alle Ticker via yfinance (Batch-Download).

    Gibt ein Dict ``{ticker → DataFrame}`` zurück, das direkt als Eingabe
    für ``build_live_features()`` → ``compute_indicators()`` dient.

    Jeder DataFrame hat:
    - ``DatetimeIndex``  (tz-naive, nur vollständige Handelstage)
    - Spalten:           ``open``, ``high``, ``low``, ``close``, ``volume`` (lowercase)

    Fehlerbehandlung:
    - Delisted / nicht gefundene Ticker → Warnung, Ticker wird übersprungen.
    - >80 % NaN in ``close`` → delisted, wird übersprungen.
    - Weniger als 50 valide Zeilen → übersprungen (zu wenig Warm-up).
    - Netzwerk-/API-Fehler → Exception wird gefangen, leeres Dict wird zurückgegeben.

    Alternative Datenquellen (Adapter-Muster):
        Um auf Alpaca oder Interactive Brokers umzusteigen, ersetze nur diese
        Funktion – die Signatur ``(tickers, days, end_date) → dict[str, DataFrame]``
        bleibt identisch:

        .. code-block:: python

            # Alpaca (alpaca-py)
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
            request = StockBarsRequest(
                symbol_or_symbols=tickers, timeframe=TimeFrame.Day,
                start=start_ts, end=end_ts,
            )
            bars = client.get_stock_bars(request).df
            # ... Umformatierung auf unser Dict-Format

    Args:
        tickers:  Liste von Ticker-Symbolen (z.B. Keys aus ``asset_map.json``).
        days:     Gewünschte Anzahl **Handelstage** Rückblick (inkl. ``end_date``).
                  Der Download wird mit einem 1,5×-Puffer in Kalendertagen
                  angefragt, damit auch nach langen Feiertags-Perioden genug
                  Daten vorhanden sind.
        end_date: Letzter einzuschließender Tag.  ``None`` = heute.

    Returns:
        ``dict[str, pd.DataFrame]``, bereit für ``build_live_features()``.
        Ticker mit zu wenig Daten oder Fehlern werden nicht zurückgegeben.

    Raises:
        ImportError: Wenn ``yfinance`` nicht installiert ist.
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError(
            "yfinance nicht installiert. Ausführen: pip install yfinance"
        ) from e

    end_ts   = pd.Timestamp(end_date or date.today()).normalize()
    # 1.5× Puffer: Handelstage ≈ 252/365 Kalendertage → sicher 50 % Puffer
    start_ts = end_ts - pd.Timedelta(days=int(days * 1.5))
    # yfinance: end-Parameter ist exklusiv → +1 Kalendertag
    end_dl   = (end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    start_dl = start_ts.strftime("%Y-%m-%d")

    print(f"  yfinance Download: {start_dl} bis {end_ts.date()}"
          f"  ({len(tickers)} Ticker, ~{days} Handelstage) ...")

    # yfinance-interne Warnungen unterdrücken (delisted-Meldungen kommen
    # trotzdem als stderr-Zeilen; wir ersetzen sie durch eigene Warnings).
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    try:
        raw = yf.download(
            tickers     = tickers,
            start       = start_dl,
            end         = end_dl,
            auto_adjust = True,    # Split/Dividend-bereinigt; keine Adj-Close-Spalte
            progress    = False,   # Keine tqdm-Fortschrittsanzeige
        )
    except Exception as exc:
        print(f"  [ERROR] yfinance Download fehlgeschlagen: {exc}")
        return {}

    if raw.empty:
        print("  [WARN] yfinance lieferte leeres DataFrame – Netzwerk-Problem?")
        return {}

    result: dict[str, pd.DataFrame] = {}
    n_skip_nan   = 0
    n_skip_short = 0
    n_skip_cols  = 0

    for tkr in tickers:
        # ── Ticker-Subtable extrahieren ───────────────────────────────────────
        df = _extract_ticker_df(raw, tkr)
        if df is None:
            n_skip_nan += 1
            continue

        # ── Spalten-Namen normalisieren (Lowercase) ───────────────────────────
        df.columns = [c.lower() for c in df.columns]

        missing = set(_REQUIRED_COLS) - set(df.columns)
        if missing:
            n_skip_cols += 1
            continue

        df = df[_REQUIRED_COLS].copy()

        # ── Index bereinigen (tz-strip, Datumsfilter) ─────────────────────────
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df[df.index <= end_ts]

        # ── Delisted-Erkennung: >80 % NaN im Close ───────────────────────────
        nan_ratio = df["close"].isna().mean()
        if nan_ratio > 0.8:
            print(f"  [WARN] {tkr}: {nan_ratio:.0%} NaN in close – möglicherweise delisted")
            n_skip_nan += 1
            continue

        # ── Zeilen mit fehlenden Kern-Daten entfernen ─────────────────────────
        df = df.dropna(subset=["open", "high", "low", "close"])

        # ── Mindest-Länge prüfen (Warm-up für SMA200 + seq_len) ───────────────
        if len(df) < 50:
            n_skip_short += 1
            continue

        result[tkr] = df

    # ── Zusammenfassung ───────────────────────────────────────────────────────
    n_ok   = len(result)
    n_days = len(next(iter(result.values()))) if result else 0
    print(f"  OK: {n_ok}/{len(tickers)} Ticker geladen  ({n_days} Handelstage)")
    if n_skip_nan > 0:
        print(f"     {n_skip_nan} Ticker: delisted / nicht gefunden / kein MultiIndex")
    if n_skip_short > 0:
        print(f"     {n_skip_short} Ticker: zu wenig Daten (<50 Zeilen)")
    if n_skip_cols > 0:
        print(f"     {n_skip_cols} Ticker: fehlende Pflicht-Spalten")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Schritt 2: Feature Engineering (sektor-neutral)
# ══════════════════════════════════════════════════════════════════════════════

def build_live_features(
    ohlcv_map:      dict[str, pd.DataFrame],
    sector_map:     dict[str, str],
    target_date:    pd.Timestamp,
    min_history:    int = 220,
) -> tuple[pd.DataFrame, list[str]]:
    """Berechnet den sektor-neutralen Feature-Panel für den ``target_date``.

    Jagt die rohen OHLCV-Daten durch dieselbe Pipeline wie beim Training:
    ``compute_indicators()`` → ``sector_neutral_zscore()``.

    Für die Inference wird anschließend ausschließlich der Lookback-Block bis
    ``target_date`` benötigt (``seq_len`` letzte Zeilen je Asset).

    Args:
        ohlcv_map:   Dict ``{ticker → OHLCV-DataFrame}`` aus ``download_ohlcv()``.
        sector_map:  Dict ``{ticker → GICS-Sektor}`` aus ``load_sector_map()``.
        target_date: Tag für den Signale berechnet werden sollen (Handelstag).
        min_history: Mindest-Zeilen pro Asset vor Feature-Berechnung.

    Returns:
        Tuple ``(features_panel, valid_tickers)``:
            - ``features_panel``: MultiIndex-DataFrame ``(date, asset) × FEATURE_COLS``
              sektor-neutral z-Score normalisiert, enthält alle Daten bis *inklusive*
              ``target_date``.
            - ``valid_tickers``: Liste der Ticker mit ausreichend Daten.
    """
    from features.engineer import compute_indicators, sector_neutral_zscore, FEATURE_COLS

    all_features: dict[str, pd.DataFrame] = {}

    for ticker, df in ohlcv_map.items():
        if len(df) < min_history:
            continue
        try:
            # Nur Daten bis zum Zieltag (kein Look-Ahead)
            df_hist = df[df.index <= target_date].copy()
            if len(df_hist) < min_history:
                continue
            feats = compute_indicators(df_hist)
            feats = feats.dropna()
            if len(feats) < 2:
                continue
            all_features[ticker] = feats
        except Exception as exc:
            print(f"  [WARN] {ticker}: Feature-Berechnung fehlgeschlagen – {exc}")

    if not all_features:
        return pd.DataFrame(), []

    # MultiIndex Panel (date, asset) aufbauen
    panel = pd.concat(all_features, names=["asset", "date"]).swaplevel().sort_index()

    # Sektor-Neutrale Z-Score Normalisierung (identisch zum Training)
    panel = sector_neutral_zscore(panel, sector_map)

    valid_tickers = list(all_features.keys())
    return panel, valid_tickers


# ══════════════════════════════════════════════════════════════════════════════
# Schritt 3: Modell laden und Scores berechnen
# ══════════════════════════════════════════════════════════════════════════════

def load_latest_checkpoint(
    ckpt_dir:   Path,
    walk_json:  Path,
    device:     str,
) -> tuple:
    """Lädt den zeitlich neuesten Fold-Checkpoint.

    Im Walk-Forward-Setup deckt jeder Fold einen anderen OOS-Zeitraum ab.
    Für Live-Inference verwenden wir den Fold mit dem spätesten ``val_end``
    – das ist das Modell, das mit den aktuellsten Daten trainiert wurde.

    Args:
        ckpt_dir:  Verzeichnis mit ``fold_*_best.pt`` Dateien.
        walk_json: Walk-Forward-JSON mit Fold-Metadaten.
        device:    Torch-Device (``"cpu"`` oder ``"cuda"``).

    Returns:
        Tuple ``(model, ckpt_meta, fold_info)`` oder ``(None, None, None)`` bei
        Fehler.
    """
    from backtest_v2_single_horizon import load_fold_model

    if not walk_json.exists():
        print(f"[ERROR] Walk-Forward-JSON nicht gefunden: {walk_json}")
        return None, None, None

    data  = json.loads(walk_json.read_text())
    folds = data.get("fold_summary", data.get("fold_results", []))
    if not folds:
        print(f"[ERROR] Keine Fold-Daten in {walk_json}")
        return None, None, None

    # Neuesten Fold (spätestes val_end) auswählen
    def _val_end(f: dict) -> pd.Timestamp:
        return pd.Timestamp(f.get("val_end", "1970-01-01"))

    folds_sorted = sorted(folds, key=_val_end)
    latest_fold  = folds_sorted[-1]

    # Checkpoint-Pfad auf aktuelles Verzeichnis zeigen lassen
    ckpt_fname = Path(latest_fold.get("ckpt_path", "")).name
    if not ckpt_fname:
        ckpt_fname = f"fold_{latest_fold.get('fold_id', 0)}_best.pt"
    ckpt_path = ckpt_dir / ckpt_fname

    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint nicht gefunden: {ckpt_path}")
        return None, None, None

    print(f"  Lade Checkpoint: {ckpt_path.name}"
          f"  (val_end={_val_end(latest_fold).date()})")
    model, ckpt_meta = load_fold_model(str(ckpt_path), device)
    return model, ckpt_meta, latest_fold


@pd.core.common.contextlib.contextmanager  # type: ignore[attr-defined]
def _no_grad_context():
    """Kontextmanager-Wrapper für torch.no_grad() ohne Top-Level-Import."""
    import torch
    with torch.no_grad():
        yield


def score_universe(
    model,
    features_panel: pd.DataFrame,
    asset_map:      dict[str, int],
    target_date:    pd.Timestamp,
    seq_len:        int,
    device:         str,
) -> pd.Series:
    """Berechnet LSTM-Scores für alle Assets am ``target_date``.

    Für jedes Asset werden die letzten ``seq_len`` Feature-Zeilen bis
    ``target_date`` als Input-Sequenz verwendet.

    Args:
        model:          Geladenes ``SingleHorizonRankModel`` (im ``eval()``-Modus).
        features_panel: MultiIndex-DataFrame ``(date, asset) × FEATURE_COLS``.
        asset_map:      Dict ``{ticker → Modell-ID}``.
        target_date:    Handelstag für den Signale berechnet werden.
        seq_len:        LSTM-Lookback-Fenster.
        device:         Torch-Device.

    Returns:
        ``pd.Series`` ``{ticker → score}``, absteigend sortiert.
        Höherer Score = stärkeres Kauf-Signal.
    """
    import torch
    from backtest_v2_single_horizon import predict_cross_section

    model.eval()
    with torch.no_grad():
        scores = predict_cross_section(
            model=model,
            features=features_panel,
            asset_map=asset_map,
            date=target_date,
            seq_len=seq_len,
            device=device,
        )

    return scores


# ══════════════════════════════════════════════════════════════════════════════
# Schritt 4: IC_roll_40 laden und A3-Policy prüfen
# ══════════════════════════════════════════════════════════════════════════════

def load_ic_history(ic_csv: Path, window: int = 40) -> Optional[float]:
    """Lädt den aktuellen IC_roll_40 aus der gespeicherten IC-Zeitreihe.

    Die Datei ``rolling_ic_v2_7d.csv`` wird beim Backtest von
    ``save_ic_artifacts()`` erzeugt und enthält tägliche IC-Werte und alle
    Rolling-IC-Fenster.

    Im Live-Betrieb muss diese Datei nach jedem neuen Abschluss-Tag aktualisiert
    werden – entweder durch einen nächtlichen Batch-Prozess (der IC für den
    abgelaufenen Handelstag berechnet) oder durch das Persistence-Dataset auf
    Kaggle.

    Args:
        ic_csv: Pfad zur ``rolling_ic_v2_{h}d.csv``.
        window: Rolling-IC-Fenster (Standard: 40 → A3-Policy).

    Returns:
        Letzter verfügbarer ``ic_roll_{window}``-Wert oder ``None`` wenn
        die Datei nicht existiert oder der Wert fehlt.
    """
    col = f"ic_roll_{window}"

    if not ic_csv.exists():
        print(f"  [WARN] IC-History-CSV nicht gefunden: {ic_csv}")
        print(f"         A3-Policy kann nicht geprueft werden (kein IC_roll_{window}).")
        return None

    try:
        df  = pd.read_csv(ic_csv, parse_dates=["date"]).sort_values("date")
        if col not in df.columns:
            print(f"  [WARN] Spalte '{col}' nicht in {ic_csv.name}")
            return None
        last_row = df.dropna(subset=[col]).iloc[-1]
        val      = float(last_row[col])
        age_days = (pd.Timestamp.today() - pd.Timestamp(last_row["date"])).days
        print(f"  IC_roll_{window} = {val:+.4f}  "
              f"(letzter Eintrag: {last_row['date'].date()}, {age_days} Tage alt)")
        return val
    except Exception as exc:
        print(f"  [WARN] IC-History lesen fehlgeschlagen: {exc}")
        return None


def check_a3_policy(ic_roll_40: Optional[float]) -> bool:
    """Prüft ob die A3-Policy (IC_roll_40 < 0) aktiv ist.

    Die A3-Policy reduziert ``n_max`` von 5 auf 3, wenn das 40-Tage-gleitende
    Mittel des Information Coefficient negativ wird.  Das signalisiert, dass
    das Modell aktuell schlechte Vorhersagequalität hat und weniger Positionen
    das Drawdown-Risiko reduzieren.

    Args:
        ic_roll_40: Aktueller IC_roll_40-Wert (``None`` = unbekannt → Policy inaktiv).

    Returns:
        ``True`` wenn A3 aktiv (ic_roll_40 < 0), sonst ``False``.
    """
    if ic_roll_40 is None:
        return False
    return ic_roll_40 < 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Schritt 5: Allokations-Output
# ══════════════════════════════════════════════════════════════════════════════

def format_allocation(
    scores:        pd.Series,
    top_n:         int,
    a3_active:     bool,
    a3_reduced_n:  int,
    ic_roll_40:    Optional[float],
    target_date:   pd.Timestamp,
) -> str:
    """Formatiert die Ziel-Allokation als lesbaren Konsolen-Report.

    Args:
        scores:       Sortierte Scores aus ``score_universe()``.
        top_n:        Standard-Anzahl Positionen (``PROD_N_MAX``).
        a3_active:    Ob die A3-Policy aktiv ist.
        a3_reduced_n: Reduzierte Positionsanzahl bei aktiver A3-Policy.
        ic_roll_40:   Aktueller IC_roll_40-Wert (für Anzeige).
        target_date:  Datum des Signals.

    Returns:
        Formatierter Multi-Zeilen-String für ``print()``.
    """
    n_eff  = a3_reduced_n if a3_active else top_n
    policy = "AKTIV (Defensiv-Modus)" if a3_active else "Inaktiv"
    top    = scores.head(n_eff)
    sep    = "=" * 60
    sep_s  = "-" * 60

    lines = [
        "",
        sep,
        f"  TAEGLICHE ZIEL-ALLOKATION  [{target_date.date()}]",
        sep,
        f"  A3-Policy (IC_roll_40 < 0): {policy}",
    ]

    if ic_roll_40 is not None:
        lines.append(f"  IC_roll_40              : {ic_roll_40:+.4f}")

    lines += [
        f"  n_max aktiv             : {n_eff}  "
        f"({'reduziert' if a3_active else 'Standard'})",
        sep_s,
        f"  Kaufe (Rang 1-{n_eff}):",
    ]

    for rank, (ticker, score) in enumerate(top.items(), start=1):
        lines.append(f"    {rank}. {ticker:<8}  Score={score:+.4f}")

    lines += [
        sep_s,
        f"  Watchlist (Rang {n_eff+1}-{min(n_eff+5, len(scores))}):",
    ]
    for rank, (ticker, score) in enumerate(scores.iloc[n_eff:n_eff+5].items(),
                                            start=n_eff+1):
        lines.append(f"    {rank}. {ticker:<8}  Score={score:+.4f}")

    lines.append(sep)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Hilfsfunktionen
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_device(device: Optional[str]) -> str:
    """Wählt CUDA wenn verfügbar und nicht explizit überschrieben."""
    if device:
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _load_asset_map(path: Path) -> dict[str, int]:
    """Lädt ``asset_map.json`` → ``{ticker: id}``."""
    if not path.exists():
        raise FileNotFoundError(f"asset_map.json nicht gefunden: {path}")
    return json.loads(path.read_text())


def _latest_trading_day() -> pd.Timestamp:
    """Gibt den letzten Handelstag zurück (grobe Näherung: Mo–Fr, nicht Feiertage).

    TODO: Ersetze durch eine echte Marktkalender-Abfrage, z.B.
        ``import pandas_market_calendars as mcal``
        ``cal = mcal.get_calendar('NYSE')``
    """
    today = pd.Timestamp.today().normalize()
    while today.weekday() >= 5:   # Samstag=5, Sonntag=6
        today -= pd.Timedelta(days=1)
    return today


# ══════════════════════════════════════════════════════════════════════════════
# CLI-Argumente
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tägliche Live/Paper-Inference – generiert Kauf-Signale für heute.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt-dir",      default=str(LiveConfig.ckpt_dir),
                   help="Verzeichnis mit fold_*_best.pt Checkpoints")
    p.add_argument("--walk-json",     default=str(LiveConfig.walk_json),
                   help="Pfad zu v2_7d_walk_forward.json")
    p.add_argument("--asset-map",     default=str(LiveConfig.asset_map_json),
                   help="Pfad zu asset_map.json")
    p.add_argument("--ic-csv",        default=str(LiveConfig.ic_history_csv),
                   help="Pfad zu rolling_ic_v2_7d.csv")
    p.add_argument("--sector-map",    default=str(LiveConfig.sector_map_json),
                   help="Pfad zu sector_map.json")
    p.add_argument("--top-n",         type=int, default=LiveConfig.top_n,
                   help="Anzahl Ziel-Positionen (n_max)")
    p.add_argument("--horizon",       type=int, default=LiveConfig.horizon,
                   help="Vorhersage-Horizont des geladenen Modells")
    p.add_argument("--seq-len",       type=int, default=LiveConfig.seq_len,
                   help="LSTM-Lookback-Fenster")
    p.add_argument("--download-days", type=int, default=LiveConfig.download_days,
                   help="Anzahl Handelstage Rückblick für Download")
    p.add_argument("--device",        default=None,
                   help="cuda oder cpu (auto wenn leer)")
    p.add_argument("--date",          default=None,
                   help="Inference-Datum YYYY-MM-DD (Standard: letzter Handelstag)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Haupt-Orchestrierung
# ══════════════════════════════════════════════════════════════════════════════

def run_live_inference(cfg: LiveConfig, target_date: Optional[pd.Timestamp] = None) -> dict:
    """Führt einen vollständigen Inference-Lauf durch.

    Kann sowohl direkt aus ``main()`` als auch aus anderen Skripten oder einem
    Cron-Job aufgerufen werden.

    Args:
        cfg:         Konfiguration (Pfade, Parameter).
        target_date: Datum für das Signale berechnet werden.
                     ``None`` = letzter Handelstag.

    Returns:
        Dict mit ``target_date``, ``scores``, ``top_tickers``,
        ``a3_active``, ``n_eff``, ``ic_roll_40``.
    """
    if target_date is None:
        target_date = _latest_trading_day()

    device = _resolve_device(cfg.device)
    print(f"\n{'='*60}")
    print(f"  Live-Inference  |  {target_date.date()}  |  Device={device}")
    print(f"{'='*60}")

    # ── Schritt 0: Metadaten laden ────────────────────────────────────────────
    print("\n[1/5] Metadaten laden ...")
    asset_map = _load_asset_map(cfg.asset_map_json)
    tickers   = cfg.tickers or sorted(asset_map.keys())
    print(f"  {len(tickers)} Ticker aus asset_map.json")

    from features.engineer import load_sector_map
    sector_map = load_sector_map(cfg.sector_map_json)
    if sector_map:
        print(f"  {len(sector_map)} Ticker in sector_map.json")
    else:
        print("  [WARN] sector_map.json leer – Fallback auf Cross-Sectional Z-Score")

    # ── Schritt 1: OHLCV herunterladen ────────────────────────────────────────
    print(f"\n[2/5] OHLCV-Daten herunterladen ({cfg.download_days} Tage) ...")
    ohlcv_map = download_ohlcv(
        tickers=tickers,
        days=cfg.download_days,
        end_date=target_date.date(),
    )

    if not ohlcv_map:
        print("\n[ABBRUCH] Keine OHLCV-Daten erhalten.")
        print("  Bitte download_ohlcv() mit yfinance oder Alpaca implementieren,")
        print("  dann erneut ausfuehren.\n")
        # Demo-Modus: Zeige Policy-Status ohne echte Scores
        ic_roll_40 = load_ic_history(cfg.ic_history_csv, cfg.a3_policy_window)
        a3_active  = check_a3_policy(ic_roll_40)
        n_eff      = cfg.a3_reduced_n if a3_active else cfg.top_n
        print(format_allocation(
            scores       = pd.Series({"[kein_download]": 0.0}),
            top_n        = cfg.top_n,
            a3_active    = a3_active,
            a3_reduced_n = cfg.a3_reduced_n,
            ic_roll_40   = ic_roll_40,
            target_date  = target_date,
        ))
        return {
            "target_date": target_date, "scores": pd.Series(dtype=float),
            "top_tickers": [], "a3_active": a3_active,
            "n_eff": n_eff, "ic_roll_40": ic_roll_40,
        }

    print(f"  {len(ohlcv_map)} Ticker heruntergeladen")

    # ── Schritt 2: Sektor-neutrale Features ───────────────────────────────────
    print("\n[3/5] Sektor-Neutrale Features berechnen ...")
    features_panel, valid_tickers = build_live_features(
        ohlcv_map=ohlcv_map,
        sector_map=sector_map,
        target_date=target_date,
    )

    if features_panel.empty:
        print("[ABBRUCH] Feature-Panel ist leer – zu wenig Daten?")
        return {"target_date": target_date, "scores": pd.Series(dtype=float),
                "top_tickers": [], "a3_active": False, "n_eff": 0, "ic_roll_40": None}

    print(f"  {len(valid_tickers)} Assets mit gültigen Features")

    # ── Schritt 3: Modell laden und Scores berechnen ──────────────────────────
    print("\n[4/5] Modell laden und Scores berechnen ...")
    model, ckpt_meta, fold_info = load_latest_checkpoint(
        ckpt_dir=cfg.ckpt_dir,
        walk_json=cfg.walk_json,
        device=device,
    )

    if model is None:
        print("[ABBRUCH] Kein Checkpoint geladen.")
        return {"target_date": target_date, "scores": pd.Series(dtype=float),
                "top_tickers": [], "a3_active": False, "n_eff": 0, "ic_roll_40": None}

    scores = score_universe(
        model=model,
        features_panel=features_panel,
        asset_map=asset_map,
        target_date=target_date,
        seq_len=cfg.seq_len,
        device=device,
    )
    print(f"  {len(scores)} Assets gescort")

    if scores.empty:
        print("[ABBRUCH] Keine Scores berechnet – zu wenig Features für seq_len?")
        return {"target_date": target_date, "scores": pd.Series(dtype=float),
                "top_tickers": [], "a3_active": False, "n_eff": 0, "ic_roll_40": None}

    # ── Schritt 4: A3-Policy prüfen ───────────────────────────────────────────
    print("\n[5/5] A3-Policy (IC_roll_40) prüfen ...")
    ic_roll_40 = load_ic_history(cfg.ic_history_csv, cfg.a3_policy_window)
    a3_active  = check_a3_policy(ic_roll_40)
    n_eff      = cfg.a3_reduced_n if a3_active else cfg.top_n

    # ── Schritt 5: Output ─────────────────────────────────────────────────────
    output = format_allocation(
        scores       = scores,
        top_n        = cfg.top_n,
        a3_active    = a3_active,
        a3_reduced_n = cfg.a3_reduced_n,
        ic_roll_40   = ic_roll_40,
        target_date  = target_date,
    )
    print(output)

    top_tickers = list(scores.head(n_eff).index)
    return {
        "target_date": target_date,
        "scores":      scores,
        "top_tickers": top_tickers,
        "a3_active":   a3_active,
        "n_eff":       n_eff,
        "ic_roll_40":  ic_roll_40,
    }


def main() -> int:
    args = parse_args()

    cfg = LiveConfig(
        ckpt_dir        = Path(args.ckpt_dir),
        walk_json       = Path(args.walk_json),
        asset_map_json  = Path(args.asset_map),
        ic_history_csv  = Path(args.ic_csv),
        sector_map_json = Path(args.sector_map),
        top_n           = args.top_n,
        horizon         = args.horizon,
        seq_len         = args.seq_len,
        download_days   = args.download_days,
        device          = args.device,
    )

    target_date = (
        pd.Timestamp(args.date) if args.date else None
    )

    run_live_inference(cfg, target_date=target_date)
    return 0


if __name__ == "__main__":
    sys.exit(main())
