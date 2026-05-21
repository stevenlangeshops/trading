"""
scan_today.py
==============================================================================
VCP-Champion Signal Scanner  -  "Welche Aktien sind HEUTE ein Kauf?"

Gibt alle Aktien aus, die am LETZTEN verfuegbaren Handelstag ein
VCP-Breakout-Signal ausgeloest haben:

  VCP-Einstieg:
    Close  > High_50d_prev   (50-Tage Ausbruch)
    Close  > SMA_200         (Langfristiger Aufwaertstrend)
    BB_Width < 10%           (Squeeze / Kontraktion)
    Volume > 1.5x SMA20_Vol  (Volumen-Bestaetigung)

Optional:
  --regime   Zeigt zusaetzlich die aktuelle Marktbreite (% > SMA50)
             und warnt bei ROT-Regime (< 40%)

Verwendung:
  # Daten zuerst aktualisieren:
  python update_raw_data.py --mode update

  # Dann scannen:
  python scan_today.py
  python scan_today.py --regime
  python scan_today.py --years 5 --top 20
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
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


# ==============================================================================
# Konfiguration (muss mit Champion-Parametern uebereinstimmen)
# ==============================================================================
BB_PERIOD        = 20
BB_STD           = 2.0
BB_SQUEEZE       = 0.10     # < 10% = Squeeze
VOL_MULT         = 1.5      # Volume > 1.5x SMA20
BREADTH_RED      = 0.40     # Marktbreite < 40% = Warnung
DEFAULT_YEARS    = 7.0
DEFAULT_TOP      = 20       # Top-N nach Trend-Staerke


# ==============================================================================
# Daten laden
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
            if len(df) < 260:
                continue
            result[ticker] = df
        except Exception:
            pass
    return result


# ==============================================================================
# Scanner-Kern
# ==============================================================================

def scan(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.Timestamp, dict]:
    """
    Prueft fuer jeden Ticker, ob am letzten Handelstag ein VCP-Signal vorlag.
    Gibt DataFrame mit Signalen, dem Scan-Datum und Marktbreiten-Info zurueck.
    """

    # Letzten gemeinsamen Handelstag bestimmen
    all_last = [df.index[-1] for df in data.values() if len(df) > 0]
    if not all_last:
        raise ValueError("Keine Daten gefunden. Bitte zuerst update_raw_data.py ausfuehren.")
    scan_date = max(all_last)

    rows: list[dict] = []
    breadth_above_sma50 = []

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

        # Marktbreite zaehlen (fuer alle Ticker mit gueltiger SMA50)
        if pd.notna(sma50.at[scan_date]):
            breadth_above_sma50.append(
                1 if c.at[scan_date] > sma50.at[scan_date] else 0)

        # VCP-Bedingungen
        high50_prev = h.shift(1).rolling(50).max()
        bb_upper    = sma20 + BB_STD * std20
        bb_lower    = sma20 - BB_STD * std20
        bb_width    = (bb_upper - bb_lower) / c.replace(0, np.nan)

        cl  = float(c.at[scan_date])
        op  = float(df["open"].at[scan_date]) if "open" in df.columns else cl
        hi  = float(h.at[scan_date])
        lo  = float(df["low"].at[scan_date]) if "low" in df.columns else cl
        v   = float(vol.at[scan_date]) if vol is not None else math.nan
        atr = float(atr14.at[scan_date])

        h50p_val   = float(high50_prev.at[scan_date]) if pd.notna(high50_prev.at[scan_date]) else math.nan
        sma200_val = float(sma200.at[scan_date])
        bb_w_val   = float(bb_width.at[scan_date]) if pd.notna(bb_width.at[scan_date]) else math.nan
        sma200_20  = float(sma20.at[scan_date])    if pd.notna(sma20.at[scan_date])    else math.nan
        sma50_val  = float(sma50.at[scan_date])    if pd.notna(sma50.at[scan_date])    else math.nan

        vol_ok = False
        if sma20v is not None and pd.notna(sma20v.at[scan_date]) and sma20v.at[scan_date] > 0:
            vol_ok = v > VOL_MULT * float(sma20v.at[scan_date])

        breakout50 = (not math.isnan(h50p_val)) and cl > h50p_val
        # Signal: NEUER Ausbruch (gestern kein Ausbruch)
        if df.index.get_loc(scan_date) > 0:
            prev_date = df.index[df.index.get_loc(scan_date) - 1]
            prev_cl   = float(c.at[prev_date])
            prev_h50p = float(high50_prev.at[prev_date]) if pd.notna(high50_prev.at[prev_date]) else math.nan
            prev_was_above = (not math.isnan(prev_h50p)) and prev_cl > prev_h50p
        else:
            prev_was_above = False

        new_breakout = breakout50 and not prev_was_above
        squeeze      = (not math.isnan(bb_w_val)) and bb_w_val < BB_SQUEEZE
        trend_ok     = cl > sma200_val

        signal = new_breakout and squeeze and vol_ok and trend_ok

        # Zusatz-Metriken immer berechnen
        dist_sma200 = (cl - sma200_val) / sma200_val * 100 if sma200_val > 0 else math.nan
        dist_sma50  = (cl - sma50_val)  / sma50_val  * 100 if sma50_val  > 0 else math.nan
        atr_pct     = atr / cl * 100

        # RSI-14
        delta   = c.diff()
        gain    = delta.clip(lower=0).rolling(14).mean()
        loss    = (-delta.clip(upper=0)).rolling(14).mean()
        rs_val  = gain.at[scan_date] / loss.at[scan_date] if loss.at[scan_date] != 0 else math.nan
        rsi     = 100 - 100 / (1 + rs_val) if not math.isnan(rs_val) else math.nan

        rows.append({
            "Ticker":       ticker,
            "Signal":       signal,
            "Close":        round(cl, 2),
            "Open":         round(op, 2),
            "BB_Width_%":   round(bb_w_val * 100, 2) if not math.isnan(bb_w_val) else math.nan,
            "Dist_SMA200_%":round(dist_sma200, 1),
            "Dist_SMA50_%": round(dist_sma50, 1),
            "ATR_%":        round(atr_pct, 2),
            "RSI_14":       round(rsi, 1) if not math.isnan(rsi) else math.nan,
            "Vol_Spike":    round(v / float(sma20v.at[scan_date]), 2)
                            if (sma20v is not None and pd.notna(sma20v.at[scan_date])
                                and sma20v.at[scan_date] > 0) else math.nan,
            "New_Breakout": new_breakout,
            "Squeeze":      squeeze,
            "Vol_OK":       vol_ok,
            "Trend_OK":     trend_ok,
        })

    df_out = pd.DataFrame(rows)
    breadth_info = {
        "above_sma50": sum(breadth_above_sma50),
        "total":       len(breadth_above_sma50),
        "pct":         sum(breadth_above_sma50) / len(breadth_above_sma50)
                       if breadth_above_sma50 else 0.0,
    }
    return df_out, scan_date, breadth_info


# ==============================================================================
# Ausgabe
# ==============================================================================

def print_results(
    df:          pd.DataFrame,
    scan_date:   pd.Timestamp,
    breadth:     dict,
    show_regime: bool,
    top_n:       int,
) -> None:
    sep  = "=" * 80
    line = "-" * 80

    print(f"\n{sep}")
    print(f"  VCP SIGNAL SCANNER  |  Scan-Datum: {scan_date.date()}")
    print(sep)

    # Marktbreite
    if show_regime:
        br_pct   = breadth["pct"] * 100
        regime   = "GRUEN  (Kaufbereit)" if br_pct >= BREADTH_RED * 100 else "ROT    (Krisenmodus!)"
        sym      = "[OK]" if br_pct >= BREADTH_RED * 100 else "[!!]"
        print(f"\n  Marktbreite (% > SMA50):  {br_pct:.1f}%  "
              f"({breadth['above_sma50']}/{breadth['total']} Aktien)")
        print(f"  Makro-Ampel:              {sym} {regime}")
        if br_pct < BREADTH_RED * 100:
            print(f"\n  WARNUNG: Marktbreite unter {BREADTH_RED*100:.0f}%!")
            print(f"  Neue Positionen sind laut Regime-Strategie (v13) nicht empfohlen.")

    # Signale
    signals = df[df["Signal"] == True].copy()
    total   = len(df)
    n_sig   = len(signals)

    print(f"\n  Universum:  {total} Aktien geprueft")
    print(f"  Signale:    {n_sig} VCP-Ausbrueche heute")
    print(line)

    if n_sig == 0:
        print("\n  Kein VCP-Signal heute. Gruende koennen sein:")
        print("    - Markt in Konsolidierung (kein Volumen)")
        print("    - Zu wenige Aktien im Squeeze")
        print("    - Alle 50d-Hochs wurden bereits gebrochen")

        # Zeige "Fast-Signale": alles ausser Signal = True, aber 3/4 Bedingungen erfuellt
        df["conditions_met"] = (
            df["New_Breakout"].astype(int) +
            df["Squeeze"].astype(int) +
            df["Vol_OK"].astype(int) +
            df["Trend_OK"].astype(int)
        )
        near = df[df["conditions_met"] == 3].sort_values(
            "Dist_SMA200_%", ascending=False).head(10)
        if len(near) > 0:
            print(f"\n  WATCHLIST: {len(near)} Aktien mit 3/4 Bedingungen erfuellt")
            print(f"  {'Ticker':<8} {'Close':>8} {'BB%':>7} {'D-200%':>7} "
                  f"{'D-50%':>7} {'ATR%':>6} {'RSI':>6}  Fehlt")
            print("  " + line)
            for _, r in near.iterrows():
                missing = []
                if not r["New_Breakout"]: missing.append("Ausbruch")
                if not r["Squeeze"]:      missing.append("Squeeze")
                if not r["Vol_OK"]:       missing.append("Volumen")
                if not r["Trend_OK"]:     missing.append("Trend>SMA200")
                print(f"  {r['Ticker']:<8} {r['Close']:>8.2f} "
                      f"{r['BB_Width_%']:>6.1f}% "
                      f"{r['Dist_SMA200_%']:>+6.1f}% "
                      f"{r['Dist_SMA50_%']:>+6.1f}% "
                      f"{r['ATR_%']:>5.2f}% "
                      f"{r['RSI_14']:>6.1f}  "
                      f"[{', '.join(missing)}]")
        return

    # Sortieren: beste Signale nach Trend-Staerke (Dist_SMA200)
    signals = signals.sort_values("Dist_SMA200_%", ascending=False).head(top_n)

    print(f"\n  {'#':<3} {'Ticker':<8} {'Close':>8} {'BB%':>7} "
          f"{'D-200%':>8} {'D-50%':>7} {'ATR%':>6} "
          f"{'RSI':>6} {'VolSpike':>9}")
    print("  " + line)

    for i, (_, r) in enumerate(signals.iterrows(), 1):
        bb  = f"{r['BB_Width_%']:.1f}%" if not math.isnan(r['BB_Width_%']) else "  n/a"
        d2  = f"{r['Dist_SMA200_%']:>+.1f}%" if not math.isnan(r['Dist_SMA200_%']) else "  n/a"
        d5  = f"{r['Dist_SMA50_%']:>+.1f}%"  if not math.isnan(r['Dist_SMA50_%'])  else "  n/a"
        at  = f"{r['ATR_%']:.2f}%"            if not math.isnan(r['ATR_%'])          else "  n/a"
        rs  = f"{r['RSI_14']:.1f}"            if not math.isnan(r['RSI_14'])         else " n/a"
        vs  = f"{r['Vol_Spike']:.1f}x"        if not math.isnan(r['Vol_Spike'])      else "  n/a"
        print(f"  {i:<3} {r['Ticker']:<8} {r['Close']:>8.2f} "
              f"{bb:>7} {d2:>8} {d5:>7} {at:>6} {rs:>6} {vs:>9}")

    # Legende / Erklaerung
    print(f"\n  Spalten-Erklaerung:")
    print(f"    BB%      Bollinger-Band-Breite (< {BB_SQUEEZE*100:.0f}% = Squeeze)")
    print(f"    D-200%   Abstand zur SMA200 (Aufwaertstrend-Staerke)")
    print(f"    D-50%    Abstand zur SMA50")
    print(f"    ATR%     Average True Range in % (Volatilitaet)")
    print(f"    RSI      RSI-14 (> 70 = potenziell ueberkauft)")
    print(f"    VolSpike Volume heute / SMA20 Volume (> {VOL_MULT}x = Bestaetigung)")

    # Auftrag-Hinweis
    print(f"\n  AUFTRAGSSTRATEGIE (VCP Champion):")
    print(f"    Stop-Loss:  Einstieg - 2.0 x ATR")
    print(f"    Earned:     wenn Gewinn > 2.0 x ATR  ->  Stop auf 3.5 x ATR nachziehen")
    print(f"    Max Slots:  2 gleichzeitige Positionen")
    print(f"    Fee:        20 EUR / Trade")

    if show_regime and breadth["pct"] * 100 < BREADTH_RED * 100:
        print(f"\n  [WARNUNG] Trotz Signal: Marktbreite ROT -> kein Kauf laut v13!")

    print(f"\n{sep}\n")


# ==============================================================================
# Main
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="VCP Signal Scanner – heutige Kaufkandidaten")
    parser.add_argument("--years",   type=float, default=DEFAULT_YEARS,
                        help="Jahre Datenbasis (default 7)")
    parser.add_argument("--top",     type=int,   default=DEFAULT_TOP,
                        help=f"Top-N Signale anzeigen (default {DEFAULT_TOP})")
    parser.add_argument("--regime",  action="store_true",
                        help="Marktbreite (Makro-Ampel) berechnen und anzeigen")
    args = parser.parse_args()

    print("  Lade Daten aus data/raw/ ...")
    tickers = _load_tickers()
    data    = load_data(tickers, args.years)

    if not data:
        print("\n  [FEHLER] Keine Parquet-Dateien gefunden!")
        print("  Bitte zuerst ausfuehren:")
        print("    python update_raw_data.py --mode update")
        sys.exit(1)

    print(f"  {len(data)} Ticker geladen. Scanne Signale ...")
    df, scan_date, breadth = scan(data)

    # Datenfrischwarn: letzte Zeile aelter als 3 Tage?
    today = pd.Timestamp.today().normalize()
    delta = (today - scan_date).days
    if delta > 3:
        print(f"\n  [WARNUNG] Daten sind {delta} Tage alt (letzter Tag: {scan_date.date()})!")
        print(f"  Bitte aktualisieren: python update_raw_data.py --mode update\n")

    print_results(df, scan_date, breadth, args.regime, args.top)


if __name__ == "__main__":
    main()
