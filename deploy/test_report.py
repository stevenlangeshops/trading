"""Debuggt den Portfolio-Report isoliert."""
import os, sys, traceback
sys.path.insert(0, "/opt/trading")

# .env laden
env_path = "/opt/trading/.env"
if os.path.exists(env_path):
    for line in open(env_path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()

import pandas as pd
from notifier import (
    _credentials, _alpaca_client, _get_portfolio_history,
    _download_msci, _make_chart, _post_photo, _post_text
)
from datetime import datetime, timezone, timedelta

token, chat_id = _credentials()
print(f"Token OK: {bool(token)}  Chat-ID: {chat_id}")

# 1. Alpaca-Daten
client = _alpaca_client()
if client is None:
    print("FEHLER: Alpaca-Client nicht verfuegbar")
    sys.exit(1)

account = client.get_account()
positions = client.get_all_positions()
print(f"Account equity: {account.equity}")
print(f"Positionen: {[p.symbol for p in positions]}")

# 2. Portfolio-History
port_history = None
try:
    port_history = _get_portfolio_history(client)
    print(f"Portfolio-History: {len(port_history) if port_history is not None else 'KEINE'} Eintraege")
except Exception as e:
    print(f"Portfolio-History FEHLER: {e}")
    traceback.print_exc()

# 3. MSCI World
now = datetime.now(tz=timezone.utc)
msci = _download_msci(now - timedelta(days=400), now)
print(f"MSCI World: {len(msci) if msci is not None else 'KEINE'} Eintraege")

# 4. Chart generieren
try:
    img = _make_chart(
        portfolio_hist=port_history,
        msci_series=msci,
        positions=list(positions),
        sold_symbols=[],
        title_date="2026-05-05",
    )
    print(f"Chart generiert: {len(img)} Bytes")

    # Testbild senden
    _post_photo(token, chat_id, img, "Test-Chart vom Trading-Bot")
    print("sendPhoto: gesendet")

except Exception as e:
    print(f"Chart/sendPhoto FEHLER: {e}")
    traceback.print_exc()
