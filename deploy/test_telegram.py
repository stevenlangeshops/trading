import os, requests, sys

token   = os.environ.get("TELEGRAM_TOKEN", "")
chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

print(f"Token:   {token[:25]}...")
print(f"Chat-ID: {chat_id}")

if not token or not chat_id:
    print("FEHLER: TELEGRAM_TOKEN oder TELEGRAM_CHAT_ID fehlt!")
    sys.exit(1)

# Schritt 1: Bot-Info prüfen
r = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=10)
print(f"\ngetMe Status: {r.status_code}")
print(f"getMe Antwort: {r.text[:200]}")

# Schritt 2: Nachricht senden
r2 = requests.post(
    f"https://api.telegram.org/bot{token}/sendMessage",
    data={"chat_id": chat_id, "text": "Trading-Bot Test vom Hetzner Server - funktioniert!"},
    timeout=10,
)
print(f"\nsendMessage Status: {r2.status_code}")
print(f"sendMessage Antwort: {r2.text[:400]}")
