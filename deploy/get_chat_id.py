import os, requests, sys

token = os.environ.get("TELEGRAM_TOKEN", "")
if not token:
    sys.exit("TELEGRAM_TOKEN fehlt!")

r = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=10)
data = r.json()
updates = data.get("result", [])

if not updates:
    print("KEINE UPDATES gefunden.")
    print("Bitte diese Schritte ausfuehren:")
    print("  1. Telegram oeffnen")
    print("  2. Bot suchen: @Stevens_trading_bot")
    print("  3. Auf 'Start' druecken oder '/start' schreiben")
    print("  4. Dieses Skript erneut ausfuehren")
    sys.exit(0)

print("Gefundene Chats:")
seen = set()
for u in updates:
    msg  = u.get("message") or u.get("channel_post") or u.get("my_chat_member", {}).get("chat") or {}
    chat = msg.get("chat", msg) if isinstance(msg, dict) else {}
    cid  = chat.get("id")
    if cid and cid not in seen:
        seen.add(cid)
        name = chat.get("first_name","") or chat.get("title","")
        print(f"  Chat-ID: {cid}  Typ: {chat.get('type','')}  Name: {name}")

print("\nTrage die korrekte Chat-ID in /opt/trading/.env ein:")
print("  TELEGRAM_CHAT_ID=<id von oben>")
