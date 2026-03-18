"""
data/download.py
────────────────
Lädt historische OHLCV-Daten via CCXT (Binance) und speichert sie als Parquet.

Aufruf:
    python data/download.py --symbol BTC/USDT --timeframe 1h --limit 5000
"""

import argparse
import time
from pathlib import Path

import ccxt
import pandas as pd
from loguru import logger

RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def fetch_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    limit: int = 5000,
    exchange_id: str = "binance",
) -> pd.DataFrame:
    """Holt bis zu `limit` Kerzen via CCXT (paginiert)."""
    exchange: ccxt.Exchange = getattr(ccxt, exchange_id)(
        {"enableRateLimit": True}
    )

    all_candles: list[list] = []
    since: int | None = None
    batch_size = 1000  # max per request for most exchanges

    logger.info(f"Lade {symbol} [{timeframe}] von {exchange_id} …")

    while len(all_candles) < limit:
        remaining = limit - len(all_candles)
        fetch_count = min(batch_size, remaining)

        candles = exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, since=since, limit=fetch_count
        )
        if not candles:
            break

        all_candles.extend(candles)
        since = candles[-1][0] + 1  # timestamp der nächsten Kerze
        logger.debug(f"  {len(all_candles)} Kerzen geladen …")

        if len(candles) < fetch_count:
            break
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(
        all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep="first")].sort_index()

    logger.success(f"Geladen: {len(df)} Kerzen ({df.index[0]} → {df.index[-1]})")
    return df


def save(df: pd.DataFrame, symbol: str, timeframe: str) -> Path:
    fname = RAW_DIR / f"{symbol.replace('/', '_')}_{timeframe}.parquet"
    df.to_parquet(fname)
    logger.success(f"Gespeichert: {fname}")
    return fname


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--exchange", default="binance")
    args = parser.parse_args()

    df = fetch_ohlcv(args.symbol, args.timeframe, args.limit, args.exchange)
    save(df, args.symbol, args.timeframe)
