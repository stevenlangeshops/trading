"""
data/download_stocks.py
────────────────────────
Lädt historische OHLCV-Daten für Aktien und ETFs via yfinance.
Speichert jeden Asset als eigene Parquet-Datei in data/raw/.

Aufruf:
    # Standard (79 Assets)
    python data/download_stocks.py --timeframe 1d --years 10

    # S&P 500 Universum (~200 Titel)
    python data/download_stocks.py --asset-file data/asset_list_sp500.txt --timeframe 1d --years 10

    # Einzelner Ticker
    python data/download_stocks.py --ticker AAPL --timeframe 1d --years 10

    # Parallel (schneller bei vielen Assets)
    python data/download_stocks.py --asset-file data/asset_list_sp500.txt --workers 8

Hinweis yfinance Limits:
    1h  → max. 730 Tage Historie
    1d  → unbegrenzt (Jahre)
    Bei >50 parallelen Requests: Rate-Limit-Fehler möglich → workers<=8 empfohlen
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock

import pandas as pd
import yfinance as yf
from loguru import logger

RAW_DIR    = Path(__file__).parent / "raw"
ASSET_LIST = Path(__file__).parent / "asset_list.txt"
RAW_DIR.mkdir(parents=True, exist_ok=True)

_print_lock = Lock()   # verhindert vermischte Log-Ausgaben bei parallelem Download


def load_asset_list(path: Path | None = None) -> list[str]:
    """Liest eine Asset-Liste — ignoriert Kommentare (#) und leere Zeilen."""
    src = path or ASSET_LIST
    tickers = []
    with open(src) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                tickers.append(line)
    return tickers


def fetch_ticker(
    ticker:    str,
    timeframe: str = "1d",
    years:     int = 10,
) -> pd.DataFrame | None:
    """
    Lädt OHLCV-Daten für einen einzelnen Ticker via yfinance.

    Gibt None zurück wenn keine Daten verfügbar oder Pflicht-Spalten fehlen.
    Thread-sicher: kann parallel von mehreren Threads aufgerufen werden.
    """
    end   = datetime.utcnow()
    start = end - timedelta(days=365 * years)

    tf_map = {"1h": "60m"}
    yf_tf  = tf_map.get(timeframe, timeframe)

    try:
        df = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=yf_tf,
            auto_adjust=True,
            progress=False,
        )
    except Exception as e:
        logger.warning(f"  {ticker}: Download-Fehler: {e}")
        return None

    if df is None or df.empty:
        return None

    # MultiIndex-Spalten flachen.
    # yfinance 0.2.x: (Price, Ticker), yfinance 1.x: ebenfalls oder anders.
    # Nach dem Flattening können Duplikate entstehen → deduplizieren.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0).str.lower()
    else:
        df.columns = [c.lower() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    df = df.rename(columns={"adj close": "close"})

    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        logger.warning(f"  {ticker}: Fehlende Spalten {required - set(df.columns)}")
        return None

    df = df[list(required)].copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df.dropna(inplace=True)

    # Mindestlänge: 200 Handelstage (sonst zu wenig für Feature-Engineering)
    if len(df) < 200:
        logger.warning(f"  {ticker}: Zu wenig Daten ({len(df)} Kerzen < 200)")
        return None

    return df


def save_ticker(df: pd.DataFrame, ticker: str, timeframe: str) -> Path:
    fname = RAW_DIR / f"{ticker.replace('.', '_')}_{timeframe}.parquet"
    df.to_parquet(fname)
    return fname


def _download_one(
    ticker:    str,
    timeframe: str,
    years:     int,
    idx:       int,
    total:     int,
    fresh_secs: int = 6 * 3600,
) -> tuple[str, str]:
    """
    Lädt einen einzelnen Ticker und gibt (ticker, status) zurück.
    status: 'success' | 'skip' | 'fail'
    Wird sowohl sequenziell als auch parallel aufgerufen.
    """
    fname = RAW_DIR / f"{ticker.replace('.', '_')}_{timeframe}.parquet"

    if fname.exists() and (time.time() - fname.stat().st_mtime) < fresh_secs:
        with _print_lock:
            logger.info(f"  [{idx:3d}/{total}] {ticker:<14} SKIP (aktuell)")
        return ticker, "skip"

    df = fetch_ticker(ticker, timeframe, years)
    if df is not None:
        save_ticker(df, ticker, timeframe)
        with _print_lock:
            logger.success(
                f"  [{idx:3d}/{total}] {ticker:<14} {len(df):5d} Tage  "
                f"({df.index[0].date()} → {df.index[-1].date()})"
            )
        return ticker, "success"
    else:
        with _print_lock:
            logger.warning(f"  [{idx:3d}/{total}] {ticker:<14} FEHLER")
        return ticker, "fail"


def download_all(
    timeframe:   str = "1d",
    years:       int = 10,
    tickers:     list[str] | None = None,
    asset_file:  Path | None = None,
    workers:     int = 1,
) -> dict[str, int]:
    """
    Lädt alle Assets herunter — sequenziell (workers=1) oder parallel.

    Bei workers > 1 werden mehrere Ticker gleichzeitig von yfinance geladen.
    Empfehlung: workers=4..8 für schnellen Download ohne Rate-Limit-Probleme.
    """
    if tickers is None:
        tickers = load_asset_list(asset_file)

    total   = len(tickers)
    results = {"success": 0, "failed": 0, "skipped": 0}

    logger.info(f"Download: {total} Assets  [{timeframe}  {years}J  workers={workers}]")
    logger.info("─" * 60)

    if workers <= 1:
        # Sequenziell — einfacher, für kleine Listen ausreichend
        for i, ticker in enumerate(tickers, 1):
            _, status = _download_one(ticker, timeframe, years, i, total)
            results["success" if status == "success" else
                    "skipped" if status == "skip" else "failed"] += 1
            if status != "skip":
                time.sleep(0.3)  # Rate-Limit respektieren
    else:
        # Parallel mit ThreadPoolExecutor
        # Jeder Thread wartet 0.1s vor dem Request um Burst-Limits zu vermeiden
        def _worker(args):
            ticker, i = args
            time.sleep(i % workers * 0.1)   # gestaffelter Start
            return _download_one(ticker, timeframe, years, i, total)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_worker, (t, i)): t
                       for i, t in enumerate(tickers, 1)}
            for fut in as_completed(futures):
                _, status = fut.result()
                results["success" if status == "success" else
                        "skipped" if status == "skip" else "failed"] += 1

    logger.info("─" * 60)
    logger.success(
        f"Fertig: {results['success']} geladen  "
        f"{results['skipped']} uebersprungen  "
        f"{results['failed']} fehlgeschlagen"
    )
    if results["failed"] > 0:
        logger.info("Tipp: Fehlgeschlagene Ticker moeglicherweise nicht bei yfinance verfuegbar.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OHLCV-Daten via yfinance herunterladen")
    parser.add_argument("--timeframe",  default="1d",
                        choices=["1m", "5m", "15m", "30m", "1h", "1d", "1wk"])
    parser.add_argument("--years",      type=int, default=10,
                        help="Anzahl Jahre Historiendate (Standard: 10)")
    parser.add_argument("--ticker",     default=None,
                        help="Einzelner Ticker (statt Asset-Liste)")
    parser.add_argument("--asset-file", default=None,
                        help="Pfad zur Asset-Liste (Standard: data/asset_list.txt)")
    parser.add_argument("--workers",    type=int, default=4,
                        help="Parallele Downloads (Standard: 4, max empfohlen: 8)")
    args = parser.parse_args()

    asset_file = Path(args.asset_file) if args.asset_file else None

    if args.ticker:
        df = fetch_ticker(args.ticker, args.timeframe, args.years)
        if df is not None:
            path = save_ticker(df, args.ticker, args.timeframe)
            logger.success(f"Gespeichert: {path}  ({len(df)} Kerzen)")
        else:
            logger.error(f"Keine Daten fuer {args.ticker}")
    else:
        download_all(
            timeframe=args.timeframe,
            years=args.years,
            asset_file=asset_file,
            workers=args.workers,
        )
