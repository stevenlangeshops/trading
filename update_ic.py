"""
update_ic.py
══════════════════════════════════════════════════════════════════════════════
Täglicher IC-Updater für den v2 Single-Horizon Trading Bot.

Workflow (täglich VOR live_inference.py ausführen):

    python update_ic.py           # aktualisiert rolling_ic_v2_7d.csv
    python live_inference.py      # liest frisch aktualisierten IC, schreibt Predictions

Ablauf:
  1. Bestimme alle Vorhersage-Daten in live_predictions_history.csv, für die
     der 7-Tage-Horizont bereits abgelaufen ist (pred_date + 7 Tage <= heute)
     und noch kein IC-Eintrag in rolling_ic_v2_7d.csv existiert.
  2. Lade via yfinance die Close-Preise für pred_date und heute.
  3. Berechne actual_return = close_heute / close_pred_date - 1.
  4. Berechne Spearman-IC zwischen gespeicherten Scores und actual_returns.
  5. Hänge neue IC-Zeilen an rolling_ic_v2_7d.csv und berechne alle Rolling-
     Fenster neu (ic_roll_5, ic_roll_10, ..., ic_roll_60).

Verwendung:
    python update_ic.py
    python update_ic.py --pred-csv pfad/zu/live_predictions_history.csv
    python update_ic.py --ic-csv   pfad/zu/rolling_ic_v2_7d.csv
    python update_ic.py --horizon  7
    python update_ic.py --dry-run        # Zeigt was berechnet würde, schreibt nichts
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

# ── Pfade ────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent.resolve()

# ── Rolling-Fenster (müssen mit rolling_ic_v2_7d.csv übereinstimmen) ────────
_ROLLING_WINDOWS = [5, 10, 15, 20, 30, 40, 50, 60]


# ══════════════════════════════════════════════════════════════════════════════
# Datums-Hilfsfunktionen
# ══════════════════════════════════════════════════════════════════════════════

def _is_trading_day(ts: pd.Timestamp) -> bool:
    """Gibt True zurück wenn ts ein Werktag (Mo-Fr) ist.

    Hinweis: US-Feiertage werden hier nicht berücksichtigt.  Für Produktions-
    einsatz empfehlen wir ``pandas_market_calendars`` (NYSE).
    """
    return ts.weekday() < 5


def _latest_trading_day(ref: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    """Gibt den letzten Handelstag zurück (Näherung: Mo-Fr)."""
    ts = (ref or pd.Timestamp.today()).normalize()
    while not _is_trading_day(ts):
        ts -= pd.Timedelta(days=1)
    return ts


def _trading_days_ago(ref: pd.Timestamp, n: int) -> pd.Timestamp:
    """Gibt den Handelstag zurück, der ``n`` Handelstage vor ``ref`` liegt.

    Zählt rückwärts und überspringt Samstage und Sonntage.

    Args:
        ref: Referenzdatum (einschließlich).
        n:   Anzahl Handelstage zurück.

    Returns:
        Datum n Handelstage vor ref.
    """
    ts = ref - pd.Timedelta(days=1)
    counted = 0
    while counted < n:
        if _is_trading_day(ts):
            counted += 1
            if counted < n:
                ts -= pd.Timedelta(days=1)
        else:
            ts -= pd.Timedelta(days=1)
    while not _is_trading_day(ts):
        ts -= pd.Timedelta(days=1)
    return ts


# ══════════════════════════════════════════════════════════════════════════════
# Predictions laden
# ══════════════════════════════════════════════════════════════════════════════

def load_predictions(csv_path: Path) -> pd.DataFrame:
    """Lädt live_predictions_history.csv.

    Args:
        csv_path: Pfad zu ``live_predictions_history.csv``.

    Returns:
        DataFrame mit Spalten ``date`` (pd.Timestamp), ``ticker``, ``score``.

    Raises:
        FileNotFoundError: Wenn die Datei nicht existiert.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"live_predictions_history.csv nicht gefunden: {csv_path}\n"
            "Starte zuerst live_inference.py, um die erste Vorhersage zu speichern."
        )
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    return df


def get_pending_dates(
    predictions:   pd.DataFrame,
    ic_csv:        Path,
    horizon:       int,
    today:         pd.Timestamp,
) -> list[pd.Timestamp]:
    """Gibt alle Vorhersage-Daten zurück, für die noch kein IC berechnet wurde.

    Bedingung: pred_date + horizon Handelstage <= heute (Horizont abgelaufen)
    UND pred_date nicht in rolling_ic_v2_7d.csv vorhanden.

    Args:
        predictions: DataFrame aus ``load_predictions()``.
        ic_csv:      Pfad zu ``rolling_ic_v2_7d.csv``.
        horizon:     Vorhersage-Horizont in Handelstagen (Standard: 7).
        today:       Heutiges Datum.

    Returns:
        Liste von pd.Timestamps (aufsteigend sortiert).
    """
    pred_dates = predictions["date"].dt.normalize().unique()

    # Bereits vorhandene IC-Daten laden
    existing_ic_dates: set[pd.Timestamp] = set()
    if ic_csv.exists():
        ic_df = pd.read_csv(ic_csv, parse_dates=["date"])
        ic_df["date"] = pd.to_datetime(ic_df["date"]).dt.tz_localize(None).dt.normalize()
        existing_ic_dates = set(ic_df["date"])

    pending = []
    for pred_date in sorted(pred_dates):
        if pred_date in existing_ic_dates:
            continue
        # Prüfen ob der Horizont abgelaufen ist
        # (pred_date + horizon Handelstage <= today)
        horizon_end = pred_date
        days_counted = 0
        check = pred_date + pd.Timedelta(days=1)
        while days_counted < horizon:
            if _is_trading_day(check):
                days_counted += 1
            check += pd.Timedelta(days=1)
        horizon_end = check - pd.Timedelta(days=1)

        if horizon_end <= today:
            pending.append(pred_date)

    return pending


# ══════════════════════════════════════════════════════════════════════════════
# Preise laden & IC berechnen
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_closes(
    tickers:    list[str],
    start_date: pd.Timestamp,
    end_date:   pd.Timestamp,
) -> pd.DataFrame:
    """Lädt tägliche Close-Preise für eine Ticker-Liste via yfinance.

    Args:
        tickers:    Liste von Ticker-Symbolen.
        start_date: Erster benötigter Tag (einschließlich).
        end_date:   Letzter benötigter Tag (einschließlich).

    Returns:
        DataFrame ``(date × ticker)`` mit Close-Preisen.
        Nicht gefundene Ticker fehlen als Spalten.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("yfinance fehlt. Installieren: pip install yfinance") from exc

    import logging
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    # 1 Puffertag auf beiden Seiten damit start/end sicher enthalten sind
    dl_start = (start_date - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    dl_end   = (end_date   + pd.Timedelta(days=3)).strftime("%Y-%m-%d")

    raw = yf.download(
        tickers     = tickers,
        start       = dl_start,
        end         = dl_end,
        auto_adjust = True,
        progress    = False,
    )
    if raw.empty:
        return pd.DataFrame()

    # Close-Preise extrahieren (MultiIndex oder flach)
    if isinstance(raw.columns, pd.MultiIndex):
        try:
            closes = raw.xs("Close", axis=1, level=0)
        except KeyError:
            closes = raw.xs("close", axis=1, level=0)
    else:
        closes = raw[["Close"]].rename(columns={"Close": tickers[0]}) if "Close" in raw.columns else raw

    closes.index = pd.to_datetime(closes.index).tz_localize(None).normalize()
    return closes


def compute_ic_for_date(
    pred_date:   pd.Timestamp,
    predictions: pd.DataFrame,
    horizon:     int,
    today:       pd.Timestamp,
) -> Optional[tuple[pd.Timestamp, float, int]]:
    """Berechnet den Tages-IC für eine Vorhersage vom pred_date.

    Lädt Close-Preise für pred_date und pred_date+horizon via yfinance,
    berechnet actual_return und Spearman-IC mit den gespeicherten Scores.

    Args:
        pred_date:   Datum der ursprünglichen Vorhersage.
        predictions: Alle Predictions aus live_predictions_history.csv.
        horizon:     Vorhersage-Horizont in Handelstagen.
        today:       Heutiges Datum (= Obergrenze für den Actual-Return).

    Returns:
        Tuple ``(ic_date, daily_ic, n_assets)`` oder ``None`` bei Fehler.
        ``ic_date`` = pred_date (so wie in rolling_ic_v2_7d.csv gespeichert).
    """
    from scipy.stats import spearmanr

    # Vorhersagen für pred_date filtern
    day_preds = predictions[predictions["date"].dt.normalize() == pred_date].copy()
    if day_preds.empty:
        print(f"  [WARN] Keine Predictions für {pred_date.date()}")
        return None

    tickers = day_preds["ticker"].tolist()

    # Actual-Return-Datum: erster Handelstag >= pred_date + horizon
    target_end = pred_date
    counted = 0
    check = pred_date + pd.Timedelta(days=1)
    while counted < horizon:
        if _is_trading_day(check):
            counted += 1
        if counted < horizon:
            check += pd.Timedelta(days=1)
    target_end = check
    # Sicherstellen dass target_end ein Handelstag und <= today
    while not _is_trading_day(target_end):
        target_end -= pd.Timedelta(days=1)
    if target_end > today:
        print(f"  [WARN] {pred_date.date()}: Horizont-Ende {target_end.date()} "
              f"liegt in der Zukunft (heute={today.date()})")
        return None

    # Preise laden (2 Daten benötigt: pred_date und target_end)
    closes = _fetch_closes(tickers, pred_date, target_end)
    if closes.empty:
        print(f"  [WARN] Keine Preise für {pred_date.date()} – {target_end.date()}")
        return None

    # Nächsten verfügbaren Handelstag für pred_date und target_end finden
    def _nearest_date(index: pd.Index, ts: pd.Timestamp) -> Optional[pd.Timestamp]:
        """Gibt den nächsten verfügbaren Index-Eintrag >= ts zurück."""
        avail = index[index >= ts]
        return avail[0] if len(avail) > 0 else None

    close_start_date = _nearest_date(closes.index, pred_date)
    close_end_date   = _nearest_date(closes.index, target_end)

    if close_start_date is None or close_end_date is None:
        print(f"  [WARN] {pred_date.date()}: pred_date oder target_end nicht in Preis-Index")
        return None
    if close_start_date == close_end_date:
        print(f"  [WARN] {pred_date.date()}: Start- und End-Datum identisch – kein Return berechenbar")
        return None

    close_start = closes.loc[close_start_date]
    close_end   = closes.loc[close_end_date]

    # Actual Return pro Ticker
    actual_returns = (close_end / close_start - 1).dropna()

    # Scores und Returns auf gemeinsame Ticker beschränken
    common = list(set(day_preds["ticker"]) & set(actual_returns.index))
    if len(common) < 10:
        print(f"  [WARN] {pred_date.date()}: Nur {len(common)} gemeinsame Ticker – IC übersprungen")
        return None

    scores_aligned  = day_preds.set_index("ticker").loc[common, "score"]
    returns_aligned = actual_returns.loc[common]

    ic_val, pvalue = spearmanr(scores_aligned.values, returns_aligned.values)
    if np.isnan(ic_val):
        print(f"  [WARN] {pred_date.date()}: IC ist NaN")
        return None

    return pred_date, float(ic_val), len(common)


# ══════════════════════════════════════════════════════════════════════════════
# IC-CSV updaten
# ══════════════════════════════════════════════════════════════════════════════

def _recompute_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Berechnet alle Rolling-IC-Fenster neu.

    Args:
        df: DataFrame mit mindestens Spalten ``date`` und ``ic``,
            sortiert nach ``date`` (aufsteigend).

    Returns:
        DataFrame mit aktualisierten ``ic_roll_*``-Spalten.
    """
    df = df.sort_values("date").reset_index(drop=True)
    for w in _ROLLING_WINDOWS:
        df[f"ic_roll_{w}"] = df["ic"].rolling(w, min_periods=w).mean()
    return df


def append_ic_rows(
    new_rows: list[tuple[pd.Timestamp, float, int]],
    ic_csv:   Path,
    dry_run:  bool = False,
) -> pd.DataFrame:
    """Hängt neue IC-Zeilen an rolling_ic_v2_7d.csv an und berechnet Rolling neu.

    Args:
        new_rows: Liste von ``(date, daily_ic, n_assets)``-Tuples.
        ic_csv:   Pfad zu ``rolling_ic_v2_7d.csv``.
        dry_run:  Wenn True, nur anzeigen ohne zu schreiben.

    Returns:
        Aktualisierter IC-DataFrame (sortiert nach Datum).
    """
    # Bestehende Datei laden oder leeren DataFrame anlegen
    if ic_csv.exists():
        df = pd.read_csv(ic_csv, parse_dates=["date"])
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    else:
        cols = ["date", "ic"] + [f"ic_roll_{w}" for w in _ROLLING_WINDOWS]
        df = pd.DataFrame(columns=cols)

    # Neue Zeilen einfügen (nur Datum + IC, Rolling wird neu berechnet)
    add_rows = pd.DataFrame({
        "date": [r[0] for r in new_rows],
        "ic":   [r[1] for r in new_rows],
    })
    df = pd.concat([df[["date", "ic"]], add_rows], ignore_index=True)
    df = df.drop_duplicates(subset="date", keep="last")

    # Rolling neu berechnen
    df = _recompute_rolling(df)

    if dry_run:
        print("\n  [DRY-RUN] Folgende Zeilen würden geschrieben:")
        print(df[df["date"].isin([r[0] for r in new_rows])][
            ["date", "ic", "ic_roll_20", "ic_roll_40"]
        ].to_string(index=False))
        return df

    df.to_csv(ic_csv, index=False, date_format="%Y-%m-%d")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Täglicher IC-Updater – berechnet Spearman-IC für abgelaufene Vorhersagen.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pred-csv",  default=str(_REPO_ROOT / "live_predictions_history.csv"),
                   help="Pfad zu live_predictions_history.csv")
    p.add_argument("--ic-csv",    default=str(_REPO_ROOT / "rolling_ic_v2_7d.csv"),
                   help="Pfad zu rolling_ic_v2_7d.csv")
    p.add_argument("--horizon",   type=int, default=7,
                   help="Vorhersage-Horizont in Handelstagen")
    p.add_argument("--dry-run",   action="store_true",
                   help="Berechne IC, schreibe aber nichts")
    p.add_argument("--date",      default=None,
                   help="Referenzdatum YYYY-MM-DD (Standard: letzter Handelstag)")
    return p.parse_args()


def main() -> int:
    args    = parse_args()
    pred_csv = Path(args.pred_csv)
    ic_csv   = Path(args.ic_csv)
    horizon  = args.horizon
    dry_run  = args.dry_run

    today = _latest_trading_day(pd.Timestamp(args.date) if args.date else None)

    print("=" * 60)
    print(f"  IC-Update  |  {today.date()}  |  Horizont={horizon}d")
    if dry_run:
        print("  [DRY-RUN] – es wird nichts geschrieben")
    print("=" * 60)

    # ── 1. Predictions laden ──────────────────────────────────────────────────
    print(f"\n[1/4] Predictions laden: {pred_csv.name} ...")
    try:
        predictions = load_predictions(pred_csv)
    except FileNotFoundError as exc:
        print(f"  [FEHLER] {exc}")
        return 1
    print(f"  {len(predictions)} Zeilen, "
          f"{predictions['date'].nunique()} Vorhersage-Tage")

    # ── 2. Offene Daten bestimmen ─────────────────────────────────────────────
    print(f"\n[2/4] Offene IC-Daten bestimmen ...")
    pending = get_pending_dates(predictions, ic_csv, horizon, today)
    if not pending:
        print("  Kein Update nötig – alle Vorhersagen sind bereits berechnet "
              "oder Horizont noch nicht abgelaufen.")
        return 0
    print(f"  {len(pending)} Vorhersage-Datum/Daten ohne IC-Eintrag:")
    for d in pending:
        print(f"    {d.date()}")

    # ── 3. IC pro Datum berechnen ─────────────────────────────────────────────
    print(f"\n[3/4] IC berechnen ...")
    new_rows: list[tuple[pd.Timestamp, float, int]] = []
    for pred_date in pending:
        result = compute_ic_for_date(pred_date, predictions, horizon, today)
        if result is not None:
            ic_date, ic_val, n_assets = result
            sign = "+" if ic_val >= 0 else ""
            print(f"  {ic_date.date()}  IC={sign}{ic_val:.4f}  "
                  f"({n_assets} Assets)")
            new_rows.append(result)

    if not new_rows:
        print("  Kein gültiger IC berechnet.")
        return 0

    # ── 4. CSV updaten ────────────────────────────────────────────────────────
    print(f"\n[4/4] rolling_ic_v2_7d.csv updaten ...")
    updated = append_ic_rows(new_rows, ic_csv, dry_run=dry_run)

    if not dry_run:
        last = updated.sort_values("date").iloc[-1]
        print(f"  Gespeichert: {ic_csv.name}  ({len(updated)} Zeilen)")
        print(f"  Neuester Eintrag: {last['date'].date()}")
        print(f"    ic={last['ic']:+.4f}  "
              f"ic_roll_20={last['ic_roll_20']:+.4f}  "
              f"ic_roll_40={last['ic_roll_40']:+.4f}")

        # Kurze IC-Statistik der letzten 40 Tage
        recent = updated.tail(40)
        print(f"\n  IC-Statistik (letzte {len(recent)} Tage):")
        print(f"    Mean    = {recent['ic'].mean():+.4f}")
        print(f"    Std     = {recent['ic'].std():.4f}")
        print(f"    % > 0   = {(recent['ic'] > 0).mean():.0%}")

    print("\n  A3-Policy-Status (ic_roll_40):", end="  ")
    last_row = updated.dropna(subset=["ic_roll_40"]).iloc[-1] if not updated.dropna(subset=["ic_roll_40"]).empty else None
    if last_row is not None:
        val = last_row["ic_roll_40"]
        status = "AKTIV (Defensiv)" if val < 0 else "Inaktiv"
        print(f"{val:+.4f}  ->  {status}")
    else:
        print("Nicht berechenbar (zu wenig Daten)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
