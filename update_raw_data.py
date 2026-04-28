"""
update_raw_data.py
══════════════════════════════════════════════════════════════════════════════
Aktualisiert alle 260 S&P-500 Parquet-Dateien in data/raw/ mit yfinance-Daten
bis zum heutigen Tag.

Modi:
  --mode fresh   (Standard) Lädt alle Ticker komplett neu (10 Jahre).
                 Sicher, sauber, unabhängig von bestehenden Dateien.
  --mode update  Liest bestehende Parquet-Dateien und hängt nur neue Zeilen
                 an (schneller, nutzt lokale Daten).

Nach dem Lauf:
  Kaggle-Dataset aktualisieren:
      kaggle datasets version -p data/raw -m "Data updated to YYYY-MM-DD" \\
          --dir-mode zip

Verwendung:
    python update_raw_data.py                  # fresh (Standard)
    python update_raw_data.py --mode update    # inkrementell
    python update_raw_data.py --years 12       # mehr Historie
    python update_raw_data.py --ticker AAPL    # einzelner Ticker (Test)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Pfade ────────────────────────────────────────────────────────────────────
_REPO_ROOT  = Path(__file__).parent.resolve()
_OUT_DIR    = _REPO_ROOT / "data" / "raw"
_SECTOR_MAP = _REPO_ROOT / "features" / "sector_map.json"

# Pflicht-Spalten (müssen mit engineer.py übereinstimmen)
_COLS = ["open", "high", "low", "close", "volume"]


# ══════════════════════════════════════════════════════════════════════════════
# Ticker-Liste
# ══════════════════════════════════════════════════════════════════════════════

def load_tickers(sector_map_path: Path) -> list[str]:
    """Lädt die Ticker-Liste aus sector_map.json.

    Args:
        sector_map_path: Pfad zu ``features/sector_map.json``.

    Returns:
        Alphabetisch sortierte Liste der 260 Produktions-Ticker.
    """
    raw = json.loads(sector_map_path.read_text())
    return sorted(k for k in raw if not k.startswith("_"))


# ══════════════════════════════════════════════════════════════════════════════
# Download-Logik
# ══════════════════════════════════════════════════════════════════════════════

def _extract_ticker_df(raw: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """Extrahiert einen einzelnen Ticker aus dem yfinance-MultiIndex-Ergebnis."""
    if isinstance(raw.columns, pd.MultiIndex):
        try:
            df = raw.xs(ticker, axis=1, level=1).copy()
        except KeyError:
            return None
    else:
        df = raw.copy()

    df.columns = [c.lower() for c in df.columns]
    return df


def download_batch(
    tickers:    list[str],
    start:      str,
    end:        str,
    retries:    int = 3,
    retry_wait: int = 15,
    min_rows:   int = 50,
) -> dict[str, pd.DataFrame]:
    """Lädt alle Ticker in einem einzigen yfinance-Batch-Call.

    Args:
        tickers:    Liste von Ticker-Symbolen.
        start:      Startdatum "YYYY-MM-DD".
        end:        Enddatum "YYYY-MM-DD" (exklusiv in yfinance).
        retries:    Anzahl Wiederholungsversuche bei Rate-Limit-Fehlern.
        retry_wait: Wartezeit in Sekunden zwischen den Versuchen.

    Returns:
        Dict ``{ticker → OHLCV-DataFrame}`` mit tz-naivem DatetimeIndex.
    """
    import logging
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    print(f"  yfinance Download: {start} bis {end}  ({len(tickers)} Ticker) ...")
    raw = pd.DataFrame()
    for attempt in range(1, retries + 1):
        raw = yf.download(
            tickers     = tickers,
            start       = start,
            end         = end,
            auto_adjust = True,
            progress    = False,
        )
        if not raw.empty:
            break
        print(f"  [WARN] Leeres Ergebnis (Versuch {attempt}/{retries}) – "
              f"warte {retry_wait}s ...")
        time.sleep(retry_wait)

    if raw.empty:
        print("  [FEHLER] Kein Ergebnis von yfinance nach allen Versuchen.")
        return {}

    result: dict[str, pd.DataFrame] = {}
    for tkr in tickers:
        df = _extract_ticker_df(raw, tkr)
        if df is None:
            continue

        missing = set(_COLS) - set(df.columns)
        if missing:
            continue

        df = df[_COLS].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df = df.dropna(subset=["close"])

        if len(df) >= min_rows:
            result[tkr] = df

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Modi: fresh vs. update
# ══════════════════════════════════════════════════════════════════════════════

def run_fresh(tickers: list[str], years: int, today: date) -> None:
    """Lädt alle Ticker komplett neu (years Jahre Historie).

    Führt einen einzigen großen Batch-Download durch und überschreibt
    bestehende Parquet-Dateien.

    Args:
        tickers: Alle Produktions-Ticker.
        years:   Anzahl Jahre zurück.
        today:   Heutiges Datum (exklusives End-Datum für yfinance).
    """
    start = (today - timedelta(days=365 * years)).strftime("%Y-%m-%d")
    end   = (today + timedelta(days=1)).strftime("%Y-%m-%d")

    data = download_batch(tickers, start, end)

    ok, fail = 0, 0
    for tkr in tickers:
        fname = _OUT_DIR / f"{tkr.replace('.', '_')}_1d.parquet"
        if tkr in data:
            df = data[tkr]
            df.to_parquet(fname)
            print(f"  OK   {tkr:<12}  {len(df):5d} Zeilen  "
                  f"{df.index[0].date()} – {df.index[-1].date()}")
            ok += 1
        else:
            print(f"  SKIP {tkr:<12}  nicht in yfinance-Ergebnis")
            fail += 1

    print(f"\n  Gespeichert: {ok}/{len(tickers)} Ticker  ({fail} fehlgeschlagen)")


def run_update(tickers: list[str], today: date) -> None:
    """Hängt nur neue Zeilen an bestehende Parquet-Dateien an.

    Für Ticker ohne lokale Datei wird ein Volldownload (10 Jahre) gemacht.
    Neue Ticker und bestehende werden in getrennten Batches heruntergeladen.

    Args:
        tickers: Alle Produktions-Ticker.
        today:   Heutiges Datum.
    """
    today_ts   = pd.Timestamp(today).normalize()
    need_full  = []   # Kein lokales Parquet vorhanden
    need_delta = {}   # {ticker: last_date} → nur neue Daten

    for tkr in tickers:
        fname = _OUT_DIR / f"{tkr.replace('.', '_')}_1d.parquet"
        if not fname.exists():
            need_full.append(tkr)
        else:
            df_ex = pd.read_parquet(fname)
            last  = pd.to_datetime(df_ex.index).normalize().max()
            # Tz-Strip: alte Parquets haben UTC-aware Index → tz-naiv machen
            if getattr(last, "tzinfo", None) is not None:
                last = last.tz_localize(None)
            if last < today_ts - pd.Timedelta(days=1):
                need_delta[tkr] = last
            else:
                print(f"  AKTUELL {tkr:<12}  letzte Zeile: {last.date()}")

    # ── Volldownload für neue Ticker ──────────────────────────────────────────
    if need_full:
        print(f"\n  {len(need_full)} neue Ticker (Volldownload 10 Jahre) ...")
        start = (today - timedelta(days=365 * 10)).strftime("%Y-%m-%d")
        end   = (today + timedelta(days=1)).strftime("%Y-%m-%d")
        data  = download_batch(need_full, start, end)
        for tkr in need_full:
            fname = _OUT_DIR / f"{tkr.replace('.', '_')}_1d.parquet"
            if tkr in data:
                data[tkr].to_parquet(fname)
                print(f"  NEU  {tkr:<12}  {len(data[tkr]):5d} Zeilen")
            else:
                print(f"  FAIL {tkr:<12}  nicht gefunden")

    # ── Delta-Download für bestehende Ticker ──────────────────────────────────
    if need_delta:
        earliest_last = min(need_delta.values())
        # 5 Tage Überlappung für Korrekturen (Splits, Dividenden)
        delta_start = (earliest_last - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        delta_end   = (today_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        print(f"\n  {len(need_delta)} Ticker benötigen Delta ab ~{earliest_last.date()} ...")
        delta_tickers = list(need_delta.keys())
        # min_rows=1: Delta-Zeitraum hat nur wenige Wochen, also kein 50-Zeilen-Filter
        new_data = download_batch(delta_tickers, delta_start, delta_end, min_rows=1)

        ok, skip, fail = 0, 0, 0
        for tkr in delta_tickers:
            fname = _OUT_DIR / f"{tkr.replace('.', '_')}_1d.parquet"
            if tkr not in new_data:
                print(f"  FAIL {tkr:<12}  kein Delta von yfinance")
                fail += 1
                continue

            # Bestehend laden und zusammenführen
            df_ex  = pd.read_parquet(fname)
            df_ex.index = pd.to_datetime(df_ex.index).tz_localize(None).normalize()
            df_ex.columns = [c.lower() for c in df_ex.columns]

            df_new = new_data[tkr]
            combined = pd.concat([df_ex[_COLS], df_new[_COLS]])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
            combined = combined.dropna(subset=["close"])

            n_added = len(combined) - len(df_ex)
            combined.to_parquet(fname)

            if n_added > 0:
                print(f"  UPD  {tkr:<12}  +{n_added:3d} neue Zeilen  "
                      f"bis {combined.index[-1].date()}")
                ok += 1
            else:
                print(f"  OK   {tkr:<12}  bereits aktuell")
                skip += 1

        print(f"\n  Delta: {ok} aktualisiert, {skip} bereits aktuell, {fail} Fehler")


# ══════════════════════════════════════════════════════════════════════════════
# Verifikation
# ══════════════════════════════════════════════════════════════════════════════

def verify(tickers: list[str], today: date) -> None:
    """Gibt eine kurze Übersicht über alle lokalen Parquet-Dateien aus."""
    today_ts = pd.Timestamp(today).normalize()
    ok, stale, missing = 0, 0, 0

    print(f"\n{'Ticker':<14} {'Zeilen':>6}  {'Von':>12}  {'Bis':>12}  Status")
    print("-" * 55)

    for tkr in tickers:
        fname = _OUT_DIR / f"{tkr.replace('.', '_')}_1d.parquet"
        if not fname.exists():
            print(f"{tkr:<14}  {'–':>6}  {'–':>12}  {'–':>12}  FEHLT")
            missing += 1
            continue

        df   = pd.read_parquet(fname)
        last = pd.to_datetime(df.index).tz_localize(None).normalize().max()
        gap  = (today_ts - last).days

        status = "OK" if gap <= 5 else f"VERALTET ({gap}d)"
        print(f"{tkr:<14}  {len(df):>6}  "
              f"{pd.to_datetime(df.index).min().date()!s:>12}  "
              f"{last.date()!s:>12}  {status}")

        if gap <= 5:
            ok += 1
        else:
            stale += 1

    print("-" * 55)
    print(f"OK: {ok}  |  Veraltet: {stale}  |  Fehlt: {missing}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Update data/raw/*.parquet mit neuesten yfinance-Daten.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode",   choices=["fresh", "update"], default="fresh",
                   help="fresh=Neuladen (sicher) | update=Delta anhängen (schnell)")
    p.add_argument("--years",  type=int, default=10,
                   help="Anzahl Jahre Historie beim Volldownload")
    p.add_argument("--ticker", default=None,
                   help="Einzelner Ticker für Test-Lauf (leer = alle 260)")
    p.add_argument("--verify-only", action="store_true",
                   help="Nur Status-Übersicht anzeigen, nichts herunterladen")
    p.add_argument("--no-upload", action="store_true",
                   help="Kaggle-Upload-Befehl nicht anzeigen")
    return p.parse_args()


def main() -> int:
    args    = parse_args()
    today   = date.today()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Ticker laden
    if not _SECTOR_MAP.exists():
        print(f"[FEHLER] sector_map.json nicht gefunden: {_SECTOR_MAP}")
        return 1
    tickers = load_tickers(_SECTOR_MAP)
    if args.ticker:
        if args.ticker not in tickers:
            print(f"[WARN] {args.ticker} nicht in sector_map – fahre trotzdem fort")
        tickers = [args.ticker]

    print("=" * 60)
    print(f"  update_raw_data.py  |  {today}  |  Modus={args.mode}")
    print(f"  {len(tickers)} Ticker  ->  {_OUT_DIR}")
    print("=" * 60)

    if args.verify_only:
        verify(tickers, today)
        return 0

    # Download
    if args.mode == "fresh":
        print(f"\n[1/2] Fresh-Download ({args.years} Jahre) ...")
        run_fresh(tickers, args.years, today)
    else:
        print(f"\n[1/2] Update (Delta) ...")
        run_update(tickers, today)

    # Verifikation
    print(f"\n[2/2] Verifikation ...")
    verify(tickers, today)

    # Kaggle-Upload Anleitung
    if not args.no_upload:
        print(f"""
{'='*60}
  NAECHSTER SCHRITT: Kaggle-Dataset updaten
{'='*60}
  Option A – Kaggle CLI (empfohlen):
      kaggle datasets version \\
          -p "{_OUT_DIR.resolve()}" \\
          -m "Raw data updated to {today}" \\
          --dir-mode zip

  Option B – manuell:
      1. https://www.kaggle.com/datasets aufrufen
      2. Dataset "trading-raw-data" -> "New Version"
      3. data/raw/ Ordner hochladen

  Danach auf Kaggle:
      python kaggle_full_run.py   (Schritt 4-9, Training + neue Folds)
{'='*60}""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
