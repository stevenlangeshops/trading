"""
daily_scan_report.py
==============================================================================
Taeglicher VCP-Scanner mit Telegram-Benachrichtigung.

Ablauf:
  1. Rohdaten aktualisieren  (update_raw_data --mode update)
  2. VCP-Signale scannen     (alle 260 Aktien, gleiche Logik wie v8-Champion)
  3. Marktbreite berechnen   (% Aktien > SMA50 = Makro-Ampel)
  4. Telegram-Nachricht senden

Wird taeglich um 09:00 Uhr (Berlin) per Cron ausgefuehrt.
US-Boersen schliessen um 22:00 Uhr Berlin -> Vortages-Daten sind verfuegbar.

Verwendung (manuell):
    python daily_scan_report.py               # Scan + Telegram
    python daily_scan_report.py --dry-run     # Nur Konsolen-Ausgabe, kein Telegram
    python daily_scan_report.py --no-update   # Kein Daten-Update (Debugging)

Umgebungsvariablen (in .env oder Shell):
    TELEGRAM_TOKEN    Bot-Token von @BotFather
    TELEGRAM_CHAT_ID  Ziel-Chat-ID
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_here    = Path(__file__).parent
_RAW_DIR = _here / "data" / "raw"

sys.path.insert(0, str(_here))
from backtest_v6 import _load_tickers, _atr

# .env laden (python-dotenv, falls vorhanden)
try:
    from dotenv import load_dotenv
    load_dotenv(_here / ".env")
except ImportError:
    pass


# ==============================================================================
# Konfiguration
# ==============================================================================
BB_PERIOD     = 20
BB_STD        = 2.0
BB_SQUEEZE    = 0.10
VOL_MULT      = 1.5
ATR_INIT      = 2.0      # Stop-Loss = Close - 2x ATR
BREADTH_RED   = 0.40     # Marktbreite < 40% = ROT
DEFAULT_YEARS = 7.0


# ==============================================================================
# 1. Daten aktualisieren
# ==============================================================================

def update_data(tickers: list[str]) -> bool:
    """Fuehrt inkrementellen Daten-Update durch. Gibt True bei Erfolg zurueck."""
    from update_raw_data import run_update
    today = date.today()
    print(f"[1/3] Daten-Update ({len(tickers)} Ticker) ...")
    t0 = time.time()
    try:
        run_update(tickers, today)
        print(f"      Update fertig in {time.time()-t0:.1f}s")
        return True
    except Exception as exc:
        print(f"      [WARN] Update fehlgeschlagen: {exc}")
        return False


# ==============================================================================
# 2. Scanner
# ==============================================================================

def load_data(tickers: list[str], years: float) -> dict[str, pd.DataFrame]:
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=int(years * 365.25))
    result = {}
    for fpath in sorted(_RAW_DIR.glob("*_1d.parquet")):
        ticker = fpath.stem.replace("_1d", "")
        if ticker not in set(tickers):
            continue
        try:
            df = pd.read_parquet(fpath)
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.columns = [c.lower() for c in df.columns]
            df = df[df.index >= cutoff].sort_index()
            if len(df) >= 260:
                result[ticker] = df
        except Exception:
            pass
    return result


def run_scan(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.Timestamp, dict]:
    """Scannt alle Ticker auf VCP-Signale am letzten Handelstag."""
    all_last = [df.index[-1] for df in data.values()]
    scan_date = max(all_last)

    rows: list[dict] = []
    breadth_values: list[int] = []

    for ticker, df in data.items():
        if scan_date not in df.index:
            continue

        c   = df["close"]
        h   = df["high"]
        vol = df.get("volume")

        sma50  = c.rolling(50).mean()
        sma200 = c.rolling(200).mean()
        sma20  = c.rolling(BB_PERIOD).mean()
        std20  = c.rolling(BB_PERIOD).std()
        atr14  = _atr(df, 14)
        sma20v = vol.rolling(20).mean() if vol is not None else None

        if sma200.isna().at[scan_date] or atr14.isna().at[scan_date]:
            continue

        if pd.notna(sma50.at[scan_date]):
            breadth_values.append(1 if c.at[scan_date] > sma50.at[scan_date] else 0)

        high50_prev = h.shift(1).rolling(50).max()
        bb_upper    = sma20 + BB_STD * std20
        bb_lower    = sma20 - BB_STD * std20
        bb_width    = (bb_upper - bb_lower) / c.replace(0, np.nan)

        cl   = float(c.at[scan_date])
        atr  = float(atr14.at[scan_date])
        h50p = float(high50_prev.at[scan_date]) if pd.notna(high50_prev.at[scan_date]) else math.nan
        bb_w = float(bb_width.at[scan_date])    if pd.notna(bb_width.at[scan_date])    else math.nan
        s200 = float(sma200.at[scan_date])
        s50  = float(sma50.at[scan_date])       if pd.notna(sma50.at[scan_date])       else math.nan
        v    = float(vol.at[scan_date])          if vol is not None else math.nan

        vol_spike = math.nan
        vol_ok    = False
        if sma20v is not None and pd.notna(sma20v.at[scan_date]) and float(sma20v.at[scan_date]) > 0:
            vol_spike = v / float(sma20v.at[scan_date])
            vol_ok    = vol_spike > VOL_MULT

        # Frischer Ausbruch (gestern noch nicht drueber)
        breakout = (not math.isnan(h50p)) and cl > h50p
        loc = df.index.get_loc(scan_date)
        if loc > 0:
            pd_ = df.index[loc - 1]
            pc  = float(c.at[pd_])
            ph  = float(high50_prev.at[pd_]) if pd.notna(high50_prev.at[pd_]) else math.nan
            prev_above = (not math.isnan(ph)) and pc > ph
        else:
            prev_above = False

        new_breakout = breakout and not prev_above
        squeeze      = (not math.isnan(bb_w)) and bb_w < BB_SQUEEZE
        trend_ok     = cl > s200
        signal       = new_breakout and squeeze and vol_ok and trend_ok

        # RSI-14
        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rsi   = math.nan
        if pd.notna(loss.at[scan_date]) and float(loss.at[scan_date]) != 0:
            rs  = float(gain.at[scan_date]) / float(loss.at[scan_date])
            rsi = 100 - 100 / (1 + rs)

        dist200 = (cl - s200) / s200 * 100 if s200 > 0 else math.nan
        dist50  = (cl - s50)  / s50  * 100 if (not math.isnan(s50) and s50 > 0) else math.nan
        atr_pct = atr / cl * 100
        stop    = cl - ATR_INIT * atr

        rows.append({
            "ticker":       ticker,
            "signal":       signal,
            "close":        round(cl, 2),
            "stop_loss":    round(stop, 2),
            "atr_pct":      round(atr_pct, 2),
            "bb_width_pct": round(bb_w * 100, 1) if not math.isnan(bb_w) else math.nan,
            "dist_sma200":  round(dist200, 1),
            "dist_sma50":   round(dist50,  1),
            "rsi14":        round(rsi, 1) if not math.isnan(rsi) else math.nan,
            "vol_spike":    round(vol_spike, 2) if not math.isnan(vol_spike) else math.nan,
            "new_breakout": new_breakout,
            "squeeze":      squeeze,
            "vol_ok":       vol_ok,
            "trend_ok":     trend_ok,
            "cond_count":   int(new_breakout) + int(squeeze) + int(vol_ok) + int(trend_ok),
        })

    df_out    = pd.DataFrame(rows) if rows else pd.DataFrame()
    n_above   = sum(breadth_values)
    n_total   = len(breadth_values)
    breadth   = {
        "above": n_above,
        "total": n_total,
        "pct":   n_above / n_total if n_total > 0 else 0.0,
    }
    return df_out, scan_date, breadth


# ==============================================================================
# 3. Telegram-Nachricht formatieren
# ==============================================================================

def _fmt_float(v, fmt=".2f") -> str:
    if isinstance(v, float) and math.isnan(v):
        return "n/a"
    return format(v, fmt)


def build_telegram_message(
    df:        pd.DataFrame,
    scan_date: pd.Timestamp,
    breadth:   dict,
    data_age:  int,
) -> str:
    """Erstellt die HTML-formatierte Telegram-Nachricht."""

    br_pct   = breadth["pct"] * 100
    is_green = br_pct >= BREADTH_RED * 100

    lines: list[str] = []

    # Header
    lines.append(f"<b>VCP Signal Scanner | {scan_date.strftime('%d.%m.%Y')}</b>")
    lines.append("")

    # Datenalter-Warnung
    if data_age > 3:
        lines.append(f"&#9888; Daten {data_age} Tage alt – Update pruefen!")
        lines.append("")

    # Marktbreite
    ampel = "&#9989; GRUEN" if is_green else "&#128308; ROT – Kaufstopp!"
    lines.append(f"<b>Marktbreite</b> (Aktien &gt; SMA50): {br_pct:.1f}%")
    lines.append(f"Makro-Ampel: {ampel}")
    lines.append(f"<i>({breadth['above']}/{breadth['total']} Aktien im Aufwaertstrend)</i>")
    lines.append("")

    if df.empty:
        lines.append("Keine Daten verfuegbar.")
        return "\n".join(lines)

    signals   = df[df["signal"]].sort_values("dist_sma200", ascending=False)
    watchlist = df[(df["cond_count"] == 3)].sort_values("dist_sma200", ascending=False).head(5)

    n_sig = len(signals)
    lines.append(f"Universum: {len(df)} Aktien | <b>{n_sig} Signal{'e' if n_sig != 1 else ''}</b>")
    lines.append("&#8212;" * 17)

    # Signale
    if n_sig == 0:
        lines.append("")
        lines.append("Kein VCP-Ausbruch heute.")
        lines.append("Markt konsolidiert – abwarten.")
    else:
        lines.append("")
        fire = "&#128293;" if n_sig >= 2 else "&#128308;"
        lines.append(f"{fire} <b>{n_sig} VCP-SIGNAL{'E' if n_sig != 1 else ''} HEUTE:</b>")
        lines.append("")
        for i, (_, r) in enumerate(signals.head(5).iterrows(), 1):
            d200 = _fmt_float(r["dist_sma200"], "+.1f")
            d50  = _fmt_float(r["dist_sma50"],  "+.1f")
            rsi  = _fmt_float(r["rsi14"],        ".1f")
            vs   = _fmt_float(r["vol_spike"],    ".1f")
            bb   = _fmt_float(r["bb_width_pct"], ".1f")
            lines.append(
                f"<b>{i}. {r['ticker']}</b>  Kurs: {r['close']:.2f}"
            )
            lines.append(
                f"   BB: {bb}%  |  +{d200}% &gt; SMA200  |  SMA50: {d50}%"
            )
            lines.append(
                f"   RSI: {rsi}  |  Vol: {vs}x  |  ATR: {r['atr_pct']:.2f}%"
            )
            lines.append(
                f"   &#9660; Stop-Loss (2x ATR): <b>{r['stop_loss']:.2f}</b>"
            )
            lines.append("")

    # Watchlist
    if len(watchlist) > 0:
        lines.append("&#128065; <b>Watchlist</b> (3/4 Bedingungen):")
        for _, r in watchlist.iterrows():
            missing = []
            if not r["new_breakout"]: missing.append("Ausbruch")
            if not r["squeeze"]:      missing.append("Squeeze")
            if not r["vol_ok"]:       missing.append("Volumen")
            if not r["trend_ok"]:     missing.append("Trend&gt;SMA200")
            d200 = _fmt_float(r["dist_sma200"], "+.1f")
            lines.append(
                f"  {r['ticker']}  {r['close']:.2f}"
                f"  |  {d200}% SMA200"
                f"  |  fehlt: {', '.join(missing)}"
            )
        lines.append("")

    # Strategie-Info
    lines.append("&#8212;" * 17)
    lines.append("<i>VCP Champion | Max 2 Slots | Stop 2x ATR | Fee 20 EUR</i>")

    if not is_green and n_sig > 0:
        lines.append("")
        lines.append("&#9888; <b>Ampel ROT – kein Kauf trotz Signal empfohlen!</b>")

    return "\n".join(lines)


# ==============================================================================
# 4. Telegram senden
# ==============================================================================

def send_message(text: str, dry_run: bool) -> None:
    if dry_run:
        print("\n" + "=" * 60)
        print("  [DRY-RUN] Telegram-Nachricht:")
        print("=" * 60)
        print(text)
        print("=" * 60 + "\n")
        return

    token   = os.getenv("TELEGRAM_TOKEN",  "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not token or not chat_id:
        print("[FEHLER] TELEGRAM_TOKEN oder TELEGRAM_CHAT_ID nicht gesetzt!")
        print("  Setze Variablen in .env oder als Shell-Exports.")
        print("\nNachrichteninhalt (nicht gesendet):")
        print(text)
        return

    import requests
    url  = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    try:
        r = requests.post(url, data=data, timeout=15)
        if r.ok:
            print("  [OK] Telegram-Nachricht gesendet.")
        else:
            print(f"  [FEHLER] Telegram: {r.status_code} – {r.text[:200]}")
    except Exception as exc:
        print(f"  [FEHLER] Telegram-Request: {exc}")


# ==============================================================================
# 5. Main
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Taeglicher VCP-Scanner mit Telegram-Report")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Kein Telegram senden, nur Konsolen-Ausgabe")
    parser.add_argument("--no-update", action="store_true",
                        help="Daten-Update ueberspringen (nur Debugging)")
    parser.add_argument("--years",     type=float, default=DEFAULT_YEARS)
    args = parser.parse_args()

    print("=" * 60)
    print(f"  VCP Daily Scan Report  |  {date.today()}")
    print("=" * 60)

    tickers = _load_tickers()

    # Schritt 1: Daten aktualisieren
    if not args.no_update:
        update_data(tickers)
    else:
        print("[1/3] Daten-Update uebersprungen (--no-update)")

    # Schritt 2: Daten laden + scannen
    print(f"\n[2/3] Lade Daten und scanne Signale ...")
    t0   = time.time()
    data = load_data(tickers, args.years)

    if not data:
        print("[FEHLER] Keine Parquet-Dateien gefunden!")
        send_message("&#9888; VCP Scanner: Keine Daten gefunden – Update fehlgeschlagen!", args.dry_run)
        sys.exit(1)

    df, scan_date, breadth = run_scan(data)
    print(f"      {len(data)} Ticker | {len(df[df['signal']])} Signale | "
          f"Marktbreite {breadth['pct']*100:.1f}% | {time.time()-t0:.1f}s")

    # Datenalter pruefen
    today     = pd.Timestamp.today().normalize()
    data_age  = (today - scan_date).days

    # Schritt 3: Nachricht senden
    print(f"\n[3/3] Telegram-Nachricht erstellen ...")
    msg = build_telegram_message(df, scan_date, breadth, data_age)
    send_message(msg, args.dry_run)

    print("\nFERTIG.")


if __name__ == "__main__":
    main()
