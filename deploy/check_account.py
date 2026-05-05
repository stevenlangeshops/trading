import sys, os
sys.path.insert(0, "/opt/trading")
for line in open("/opt/trading/.env"):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip()

from notifier import _alpaca_client
c = _alpaca_client()
a = c.get_account()
print("created_at:", a.created_at)
print("equity:    ", a.equity)
