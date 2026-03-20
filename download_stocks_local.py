"""
download_stocks_local.py
─────────────────────────
Lokales Windows-Script: Lädt alle 79 Assets (10 Jahre, täglich) via yfinance
und speichert sie als Parquet-Dateien.

Aufruf:
    python download_stocks_local.py
    python download_stocks_local.py --years 10
    python download_stocks_local.py --ticker AAPL

Danach: Parquet-Dateien per WinSCP/MobaXterm hochladen nach /root/trading_v2/data/raw/
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

# Ausgabeverzeichnis (lokal)
OUT_DIR = Path(__file__).parent / "data" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ASSETS = [
    # USA Large Cap
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AMD","JPM","V",
    "UNH","XOM","BRK-B","LLY","AVGO","WMT","COST","HD","BAC","MA",
    "PG","JNJ","ABBV","MRK","KO","PEP","NFLX","CRM","ORCL","MU",
    # USA ETFs
    "SPY","QQQ","IWM","GLD","TLT","XLK","XLF","XLE","XLV","XLI","EEM","EFA",
    # Europa Deutschland
    "SAP.DE","SIE.DE","ALV.DE","MBG.DE","BMW.DE","BAS.DE","DTE.DE","DBK.DE","VOW3.DE",
    # Europa Frankreich
    "AIR.PA","MC.PA","OR.PA","TTE.PA","BNP.PA","SAN.PA",
    # Europa Niederlande/Schweiz
    "ASML.AS","HEIA.AS","NESN.SW","NOVN.SW","ROG.SW",
    # Europa UK
    "SHEL.L","AZN.L","HSBA.L","BP.L","GSK.L",
    # Asien Japan
    "7203.T","6758.T","9984.T","7974.T",
    # Asien China (NYSE)
    "BABA","BIDU","JD","PDD","TSM",
    # Asien ETFs
    "EWJ","MCHI","EWY",
]


def fetch(ticker: str, years: int = 10) -> pd.DataFrame | None:
    end   = datetime.utcnow()
    start = end - timedelta(days=365 * years)
    try:
        df = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
    except Exception as e:
        print(f"  FEHLER {ticker}: {e}")
        return None

    if df is None or df.empty:
        print(f"  KEINE DATEN: {ticker}")
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    df = df.rename(columns={"adj close": "close"})
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        print(f"  FEHLENDE SPALTEN {ticker}: {required - set(df.columns)}")
        return None

    df = df[list(required)].copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df.dropna(inplace=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years",  type=int, default=10)
    parser.add_argument("--ticker", type=str, default=None,
                        help="Einzelner Ticker (leer = alle)")
    args = parser.parse_args()

    tickers = [args.ticker] if args.ticker else ASSETS
    ok, skip, fail = 0, 0, 0

    print(f"\n=== Download: {len(tickers)} Assets, {args.years} Jahre ===\n")

    for i, t in enumerate(tickers, 1):
        fname = OUT_DIR / f"{t.replace('.', '_')}_1d.parquet"

        # Bereits vorhanden überspringen
        if fname.exists():
            rows = pd.read_parquet(fname).shape[0]
            print(f"[{i:3d}/{len(tickers)}] SKIP  {t:<15} ({rows} Kerzen bereits vorhanden)")
            skip += 1
            continue

        print(f"[{i:3d}/{len(tickers)}] Lade  {t:<15} ...", end=" ", flush=True)
        df = fetch(t, args.years)

        if df is not None and len(df) > 50:
            df.to_parquet(fname)
            print(f"OK  ({len(df)} Kerzen, {df.index[0].date()} – {df.index[-1].date()})")
            ok += 1
        else:
            print("FEHLER")
            fail += 1

        time.sleep(0.5)  # kurze Pause gegen Rate-Limit

    print(f"\n=== Fertig: {ok} geladen, {skip} übersprungen, {fail} Fehler ===")
    print(f"Dateien in: {OUT_DIR.resolve()}")
    print("\nNächster Schritt: Dateien per WinSCP/MobaXterm hochladen nach /root/trading_v2/data/raw/")


if __name__ == "__main__":
    main()
