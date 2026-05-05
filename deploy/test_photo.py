"""Sendet ein Testbild und gibt die vollständige API-Antwort aus."""
import os, sys, base64, requests, io
sys.path.insert(0, "/opt/trading")

env_path = "/opt/trading/.env"
for line in open(env_path):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip()

token   = os.environ["TELEGRAM_TOKEN"]
chat_id = os.environ["TELEGRAM_CHAT_ID"]
print(f"Token:   {token[:20]}...")
print(f"Chat-ID: {chat_id}")

# Test 1: Minimaltest mit 1x1 PNG
print("\n--- Test 1: Minimal-PNG ---")
png_tiny = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQI12NgAAIABQ"
    "AABjkB6QAAAABJRU5ErkJggg=="
)
r = requests.post(
    f"https://api.telegram.org/bot{token}/sendPhoto",
    data={"chat_id": chat_id, "caption": "Test 1: Minimal-PNG"},
    files={"photo": ("test.png", png_tiny, "image/png")},
    timeout=15,
)
print(f"Status: {r.status_code}")
print(f"Antwort: {r.text[:400]}")

# Test 2: Echten Chart aus notifier generieren und senden
print("\n--- Test 2: Echter Chart aus notifier ---")
from notifier import _make_chart, _download_msci, _alpaca_client
from datetime import datetime, timezone, timedelta

client = _alpaca_client()
positions = list(client.get_all_positions()) if client else []
print(f"Positionen: {[p.symbol for p in positions]}")

now   = datetime.now(tz=timezone.utc)
msci  = _download_msci(now - timedelta(days=400), now)
print(f"MSCI: {len(msci) if msci is not None else 0} Eintraege")

img = _make_chart(
    portfolio_hist=None,
    msci_series=msci,
    positions=positions,
    sold_symbols=[],
    title_date="2026-05-05",
)
print(f"Chart-Groesse: {len(img)} Bytes ({len(img)//1024} KB)")

# Bild direkt senden (ohne parse_mode)
r2 = requests.post(
    f"https://api.telegram.org/bot{token}/sendPhoto",
    data={"chat_id": chat_id, "caption": "Test 2: Echter Chart"},
    files={"photo": ("chart.png", img, "image/png")},
    timeout=30,
)
print(f"Status: {r2.status_code}")
print(f"Antwort: {r2.text[:500]}")

# Test 3: Mit HTML caption
print("\n--- Test 3: Chart mit HTML-Caption ---")
r3 = requests.post(
    f"https://api.telegram.org/bot{token}/sendPhoto",
    data={"chat_id": chat_id, "caption": "<b>Test 3</b>: Chart mit HTML", "parse_mode": "HTML"},
    files={"photo": ("chart.png", io.BytesIO(img), "image/png")},
    timeout=30,
)
print(f"Status: {r3.status_code}")
print(f"Antwort: {r3.text[:500]}")
