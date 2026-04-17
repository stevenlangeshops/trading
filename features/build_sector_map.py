"""
features/build_sector_map.py
─────────────────────────────
Utility-Script: aktualisiert features/sector_map.json mit GICS-Sektoren
aus yfinance.  Einmalig lokal ausführen, Ergebnis committen.

Verwendung:
    python features/build_sector_map.py
    python features/build_sector_map.py --tickers AAPL MSFT NVDA    # nur einzelne prüfen
    python features/build_sector_map.py --asset-map checkpoints/asset_map.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

HERE        = Path(__file__).parent
SECTOR_FILE = HERE / "sector_map.json"

SECTOR_FALLBACK = "Unknown"


def fetch_sector(ticker: str) -> str:
    """Fragt yfinance nach dem GICS-Sektor eines Tickers."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        sector = info.get("sector") or info.get("sectorKey") or SECTOR_FALLBACK
        return sector
    except Exception:
        return SECTOR_FALLBACK


def build_or_update(tickers: list[str], overwrite: bool = False) -> None:
    # Bestehende Map laden
    if SECTOR_FILE.exists():
        with open(SECTOR_FILE) as f:
            existing = json.load(f)
    else:
        existing = {"_meta": {
            "source": "yfinance GICS",
            "update_script": "features/build_sector_map.py",
        }}

    updated = 0
    for i, ticker in enumerate(tickers):
        if not overwrite and ticker in existing:
            continue  # bereits vorhanden → überspringen

        sector = fetch_sector(ticker)
        existing[ticker] = sector
        updated += 1
        print(f"  [{i+1:3d}/{len(tickers)}] {ticker:<12} → {sector}")

        # Rate-Limit: yfinance erlaubt ~2 req/s
        time.sleep(0.6)

    with open(SECTOR_FILE, "w") as f:
        json.dump(existing, f, indent=2, sort_keys=True)

    print(f"\n✓ {SECTOR_FILE} aktualisiert  ({updated} neue Einträge)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tickers",    nargs="+", default=None,
                   help="Explizite Ticker-Liste (Standard: alle aus asset_map)")
    p.add_argument("--asset-map",  default=None,
                   help="Pfad zur asset_map.json (für automatische Ticker-Extraktion)")
    p.add_argument("--overwrite",  action="store_true",
                   help="Bestehende Einträge mit aktuellen yfinance-Daten überschreiben")
    args = p.parse_args()

    if args.tickers:
        tickers = args.tickers
    elif args.asset_map:
        with open(args.asset_map) as f:
            asset_map = json.load(f)
        # asset_map ist entweder {ticker: idx} oder {idx: ticker}
        if all(isinstance(k, str) for k in asset_map):
            tickers = list(asset_map.keys())
        else:
            tickers = list(asset_map.values())
    else:
        # Alle Tickers aus bestehender sector_map, die noch "Unknown" sind
        if SECTOR_FILE.exists():
            with open(SECTOR_FILE) as f:
                sm = json.load(f)
            tickers = [k for k, v in sm.items()
                       if not k.startswith("_") and v == SECTOR_FALLBACK]
            print(f"Tickers mit Fallback-Sektor: {len(tickers)}")
        else:
            print("Keine Ticker angegeben und keine sector_map.json gefunden.")
            return

    if not tickers:
        print("Nichts zu tun – alle Tickers bereits gemappt.")
        return

    print(f"Verarbeite {len(tickers)} Tickers ...")
    build_or_update(tickers, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
