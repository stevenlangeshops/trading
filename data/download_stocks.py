"""
data/download_stocks.py
────────────────────────
Lädt historische OHLCV-Daten für Aktien und ETFs via yfinance.
Speichert jeden Asset als eigene Parquet-Datei in data/raw/.

Aufruf:
    python data/download_stocks.py --timeframe 1h --years 2
    python data/download_stocks.py --timeframe 1d --years 5
    python data/download_stocks.py --ticker AAPL --timeframe 1h --years 2

Unterstützte Timeframes:
    1m, 2m, 5m, 15m, 30m, 60m (=1h), 90m
    1h, 1d, 5d, 1wk, 1mo, 3mo

Hinweis yfinance Limits:
    1h  → max. 730 Tage Historie
    1d  → unbegrenzt (Jahre)
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from loguru import logger

RAW_DIR      = Path(__file__).parent / "raw"
ASSET_LIST   = Path(__file__).parent / "asset_list.txt"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def load_asset_list() -> list[str]:
    """Liest asset_list.txt — ignoriert Kommentare und leere Zeilen."""
    tickers = []
    with open(ASSET_LIST) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                tickers.append(line)
    return tickers


def fetch_ticker(
    ticker: str,
    timeframe: str = "1h",
    years: int = 2,
) -> pd.DataFrame | None:
    """
    Lädt OHLCV-Daten für einen einzelnen Ticker via yfinance.
    Gibt None zurück wenn keine Daten verfügbar.
    """
    end   = datetime.utcnow()
    start = end - timedelta(days=365 * years)

    # yfinance nutzt "60m" statt "1h"
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
        logger.warning(f"  {ticker}: Keine Daten verfügbar")
        return None

    # MultiIndex-Spalten flachen (yfinance >= 0.2.x gibt (Price, Ticker) zurück)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"adj close": "close"})
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        logger.warning(f"  {ticker}: Fehlende Spalten {required - set(df.columns)}")
        return None

    df = df[list(required)].copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df.dropna(inplace=True)

    return df


def save_ticker(df: pd.DataFrame, ticker: str, timeframe: str) -> Path:
    fname = RAW_DIR / f"{ticker.replace('.', '_')}_{timeframe}.parquet"
    df.to_parquet(fname)
    return fname


def download_all(
    timeframe: str = "1h",
    years: int = 2,
    tickers: list[str] | None = None,
) -> dict[str, int]:
    """Lädt alle Assets aus asset_list.txt herunter."""
    if tickers is None:
        tickers = load_asset_list()

    results = {"success": 0, "failed": 0, "skipped": 0}
    logger.info(f"Lade {len(tickers)} Assets  [{timeframe}  {years}J]")
    logger.info("─" * 50)

    for i, ticker in enumerate(tickers, 1):
        fname = RAW_DIR / f"{ticker.replace('.', '_')}_{timeframe}.parquet"

        # Überspringen wenn bereits vorhanden und frisch (< 6h alt)
        if fname.exists():
            age_hours = (time.time() - fname.stat().st_mtime) / 3600
            if age_hours < 6:
                logger.info(f"  [{i:2d}/{len(tickers)}] {ticker:<12} SKIP (frisch, {age_hours:.1f}h alt)")
                results["skipped"] += 1
                continue

        df = fetch_ticker(ticker, timeframe, years)
        if df is not None:
            save_ticker(df, ticker, timeframe)
            logger.success(f"  [{i:2d}/{len(tickers)}] {ticker:<12} {len(df):5d} Kerzen  "
                           f"({df.index[0].date()} → {df.index[-1].date()})")
            results["success"] += 1
        else:
            results["failed"] += 1

        time.sleep(0.3)   # yfinance Rate-Limit respektieren

    logger.info("─" * 50)
    logger.success(f"Fertig: {results['success']} geladen, "
                   f"{results['skipped']} übersprungen, "
                   f"{results['failed']} fehlgeschlagen")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe", default="1h",
                        choices=["1m","5m","15m","30m","1h","1d","1wk"])
    parser.add_argument("--years",     type=int, default=10)
    parser.add_argument("--ticker",    default=None,
                        help="Einzelner Ticker statt asset_list.txt")
    args = parser.parse_args()

    if args.ticker:
        df = fetch_ticker(args.ticker, args.timeframe, args.years)
        if df is not None:
            path = save_ticker(df, args.ticker, args.timeframe)
            logger.success(f"Gespeichert: {path}  ({len(df)} Kerzen)")
    else:
        download_all(args.timeframe, args.years)
