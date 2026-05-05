import os, sys, requests
sys.path.insert(0, "/opt/trading")

for line in open("/opt/trading/.env"):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip()

from notifier import _make_chart, _download_msci, _alpaca_client
from datetime import datetime, timezone, timedelta

token   = os.environ["TELEGRAM_TOKEN"]
chat_id = os.environ["TELEGRAM_CHAT_ID"]
now_str = datetime.now().strftime("%H:%M:%S")

# Schritt 1: Text-Ping (damit User weiss wohin schauen)
r = requests.post(
    f"https://api.telegram.org/bot{token}/sendMessage",
    data={"chat_id": chat_id,
          "text": f"JETZT {now_str} – Portfolio-Chart kommt gleich darunter..."},
    timeout=10,
)
print(f"Text-Ping: {r.status_code} ok={r.json().get('ok')}")

# Schritt 2: Chart generieren
client          = _alpaca_client()
positions       = list(client.get_all_positions()) if client else []
account         = client.get_account() if client else None
portfolio_start = getattr(account, "created_at", None) if account else None
print(f"Depot-Start: {portfolio_start}")

now  = datetime.now(tz=timezone.utc)
msci = _download_msci(now - timedelta(days=400), now)

img = _make_chart(
    portfolio_hist=None,
    msci_series=msci,
    positions=positions,
    sold_symbols=[],
    title_date=f"Test {now_str}",
    portfolio_start=portfolio_start,
)
print(f"Chart: {len(img)//1024} KB")

# Schritt 3: Bild senden
r2 = requests.post(
    f"https://api.telegram.org/bot{token}/sendPhoto",
    data={"chat_id": chat_id,
          "caption": f"Portfolio-Chart | {now_str}",
          "parse_mode": "HTML"},
    files={"photo": ("chart.png", img, "image/png")},
    timeout=30,
)
result = r2.json()
print(f"Bild senden: {r2.status_code} ok={result.get('ok')}")
if not result.get("ok"):
    print(f"FEHLER: {result.get('description')}")
else:
    print(f"message_id={result['result']['message_id']}  – Bild erfolgreich!")
