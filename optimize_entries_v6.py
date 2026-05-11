"""
optimize_entries_v6.py
====================================================================================
Signal-Optimierung  |  v6.3 Trade-Manager × 5 Entry-Regeln × 2 Gebühren-Szenarien

Fragestellung:
    Welches Entry-Signal liefert mit dem v6.3 Trade-Manager die beste Rendite,
    Hit-Rate und Gebühreneffizienz? Schlägt ein anderes Signal den Breakout_50?

Motor (unveränderter v6.3 Trade-Manager aus backtest_v6.py):
    ✓ Asymmetrischer "Earned" ATR-Stop (2.0× Tight → 3.5× Earned)
    ✓ Free-Ride Pyramidisieren  (Stop auf Break-Even nach Aufstockung)
    ✓ Rotations-Bremse (2.0×, nur profitable Positionen)
    ✓ Max 5 Slots, Startkapital 10.000 €

Getestete Entry-Signale (alle mit _transition – nur Einstiegs-Tag):
    1. Breakout_50      Close > High_50d_prev  + SMA200  (Baseline)
    2. Breakout_100     Close > High_100d_prev + SMA200  (seltener, höher qualitativ)
    3. RSI_Dip_Bull     RSI14 fällt unter 40   + SMA200  (Pullback im Aufwärtstrend)
    4. Double_Oversold  RSI14 < 30 UND Close < BB_lower  (Panik-Kauf, Mean-Reversion)
    5. MACD_Crossover   MACD-Hist dreht > 0    + SMA200  (frühes Momentum-Signal)

Gebühren-Szenarien:
    A  20.00 € pro Order  (Bank-Broker, Classic)
    B   2.00 € pro Order  (Neo-Broker: Trade Republic, IBKR)

Verwendung:
    python optimize_entries_v6.py
    python optimize_entries_v6.py --years 7 --capital 10000 --no-pyramid
    python optimize_entries_v6.py --atr-init 2.5 --atr-trail 4.0
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Engine aus backtest_v6 importieren ──────────────────────────────────────
_here = Path(__file__).parent
sys.path.insert(0, str(_here))

from backtest_v6 import (
    INITIAL_CAPITAL, ORDER_FEE,
    DEFAULT_MAX_POS, DEFAULT_ATR_INIT, DEFAULT_ATR_TRAIL,
    DEFAULT_ROTATION_FACTOR, DEFAULT_PYRAMID_THRESHOLD, DEFAULT_MAX_PYRAMIDS,
    _load_tickers, load_universe, _atr,
    run_backtest, compute_metrics,
)

# ── Signal-Definitionen ──────────────────────────────────────────────────────
SIGNAL_NAMES: dict[str, str] = {
    "Breakout_50":     "Close > High_50d + SMA200  (Baseline)",
    "Breakout_100":    "Close > High_100d + SMA200 (seltener)",
    "RSI_Dip_Bull":    "RSI14 < 40 (↓Transition) + SMA200",
    "Double_Oversold": "RSI14 < 30 AND Close < BB_lower",
    "MACD_Crossover":  "MACD-Hist > 0 (↑Transition) + SMA200",
}

FEE_SCENARIOS: list[float] = [20.0, 2.0]


# ==============================================================================
# TECHNISCHE INDIKATOREN  (reine Pandas – kein talib benötigt)
# ==============================================================================

def _transition(cond: pd.Series) -> pd.Series:
    """True NUR am Tag, an dem die Bedingung von False auf True wechselt."""
    return cond & ~cond.shift(1).fillna(False)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder-Smoothing via EWM)."""
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _bb_lower(close: pd.Series, period: int = 20, n_std: float = 2.0) -> pd.Series:
    """Unteres Bollinger Band."""
    return close.rolling(period).mean() - n_std * close.rolling(period).std()


def _macd_hist(close: pd.Series,
               fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD-Histogramm (MACD-Linie minus Signal-Linie)."""
    macd_line   = close.ewm(span=fast, adjust=False).mean() \
                - close.ewm(span=slow, adjust=False).mean()
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


def _entry_signal(df: pd.DataFrame, signal_name: str,
                  sma200: pd.Series) -> pd.Series:
    """Berechnet das boolesche Entry-Signal für einen Ticker.

    Alle Signale nutzen _transition → True nur am Einstiegs-Tag.
    """
    c = df["close"]
    h = df["high"]

    if signal_name == "Breakout_50":
        high50_prev = h.shift(1).rolling(50).max()
        raw = (c > high50_prev) & (c > sma200)
        return _transition(raw)

    elif signal_name == "Breakout_100":
        high100_prev = h.shift(1).rolling(100).max()
        raw = (c > high100_prev) & (c > sma200)
        return _transition(raw)

    elif signal_name == "RSI_Dip_Bull":
        # Pullback-Kauf: RSI fällt unter 40 während Close > SMA200
        rsi = _rsi(c)
        return _transition(rsi < 40) & (c > sma200)

    elif signal_name == "Double_Oversold":
        # Panik-Kauf: RSI extrem überverkauft UND unter unterem BB
        # Kein SMA200-Filter – intentional Mean-Reversion
        rsi   = _rsi(c)
        bb_lo = _bb_lower(c)
        return _transition((rsi < 30) & (c < bb_lo))

    elif signal_name == "MACD_Crossover":
        # MACD-Hist dreht ins Positive (Momentum-Frühindikator) + Trendfilter
        hist = _macd_hist(c)
        return _transition(hist > 0) & (c > sma200)

    else:
        raise ValueError(f"Unbekanntes Signal: {signal_name}")


# ==============================================================================
# PANEL-AUFBAU  (pro Signal)
# ==============================================================================

def build_signal_panel(data: dict[str, pd.DataFrame],
                       signal_name: str,
                       atr_period: int = 14) -> dict[str, pd.DataFrame]:
    """Baut Date×Ticker Pivot-Tabellen für ein spezifisches Entry-Signal.

    Gibt exakt dieselbe Struktur zurück wie build_pivot_panel in backtest_v6.
    """
    opens, closes, highs, atrs = {}, {}, {}, {}
    entry_sigs, trend_strs     = {}, {}

    for ticker, df in data.items():
        c      = df["close"]
        h      = df["high"]
        sma200 = c.rolling(200).mean()
        atr14  = _atr(df, atr_period)
        ts     = (c - sma200) / sma200

        # Gültigkeitsmaske: SMA200 + ATR müssen existieren
        valid  = sma200.notna() & atr14.notna()

        # Entry-Signal (ggf. mit NaN → False für erste Perioden)
        sig = _entry_signal(df, signal_name, sma200)
        sig = sig.reindex(c[valid].index).fillna(False)

        opens[ticker]      = df["open"][valid]
        closes[ticker]     = c[valid]
        highs[ticker]      = h[valid]
        atrs[ticker]       = atr14[valid]
        entry_sigs[ticker] = sig
        trend_strs[ticker] = ts[valid]

    return {
        "open":      pd.DataFrame(opens),
        "close":     pd.DataFrame(closes),
        "high":      pd.DataFrame(highs),
        "atr14":     pd.DataFrame(atrs),
        "entry_sig": pd.DataFrame(entry_sigs).fillna(False),
        "trend_str": pd.DataFrame(trend_strs),
    }


# ==============================================================================
# VERGLEICHS-SCHLEIFE
# ==============================================================================

def run_comparison(
    data:               dict[str, pd.DataFrame],
    fee_scenarios:      list[float],
    initial_cap:        float,
    max_pos:            int,
    atr_init:           float,
    atr_trail:          float,
    rotation_factor:    float,
    pyramid_threshold:  float,
    max_pyramids:       int,
    enable_pyramid:     bool,
) -> pd.DataFrame:
    """Führt alle Signale × alle Fee-Szenarien durch und sammelt Metriken."""
    results = []

    for signal_name in SIGNAL_NAMES:
        t0 = time.time()
        print(f"\n  [{signal_name}]  Panel aufbauen ...", end="", flush=True)
        pivots = build_signal_panel(data, signal_name)
        dates  = pivots["open"].index

        # Anzahl Signale pro Jahr (für Diagnostik)
        n_signals_total = int(pivots["entry_sig"].sum().sum())
        n_years         = (dates[-1] - dates[0]).days / 365.25
        sig_per_year    = n_signals_total / n_years if n_years > 0 else 0

        print(f"  {len(dates)} Tage  |  {n_signals_total:,} Signale "
              f"({sig_per_year:.0f}/Jahr)", flush=True)

        for fee in fee_scenarios:
            eq_df, completed, _ = run_backtest(
                pivots             = pivots,
                initial_cap        = initial_cap,
                fee                = fee,
                max_pos            = max_pos,
                atr_init           = atr_init,
                atr_trail          = atr_trail,
                rotation_factor    = rotation_factor,
                pyramid_threshold  = pyramid_threshold,
                max_pyramids       = max_pyramids,
                enable_pyramid     = enable_pyramid,
                market_filter      = None,
                verbose            = False,   # kein Inline-Log in der Schleife
            )
            m = compute_metrics(eq_df, completed, initial_cap, fee)

            # Earned vs. Tight Aufteilung
            n_earned = sum(1 for t in completed if t.get("earned_mode", False))
            n_tight  = len(completed) - n_earned

            results.append({
                "Signal":       signal_name,
                "Fee_€":        fee,
                "Rendite_%":    m["total_ret_%"],
                "CAGR_%":       m["cagr_%"],
                "MaxDD_%":      m["max_dd_%"],
                "Sharpe":       m["sharpe"],
                "N_Trades":     m["n_trades"],
                "N_Earned":     n_earned,
                "N_Tight":      n_tight,
                "Hit_%":        m["hit_%"],
                "Payoff":       m["payoff"],
                "Expect_%":     m["expect_%"],
                "AvgHold_d":    m["avg_hold_d"],
                "MaxWin_%":     m["max_win_%"],
                "Fees_€":       m["total_fees"],
                "Endkap_€":     m["end_equity"],
                "Sig/Jahr":     round(sig_per_year),
            })
            print(f"    Fee={fee:>5.0f}€:  {m['total_ret_%']:>+7.2f}%  "
                  f"({m['n_trades']:>3} Trades, Hit {m['hit_%']:.0f}%, "
                  f"Payoff {m['payoff']:.2f}, Fees {m['total_fees']:>7,.0f}€)",
                  flush=True)

        elapsed = time.time() - t0
        print(f"    [{signal_name}] fertig in {elapsed:.1f}s", flush=True)

    return pd.DataFrame(results)


# ==============================================================================
# AUSGABE
# ==============================================================================

def print_comparison(df: pd.DataFrame) -> None:
    """Druckt die Vergleichstabelle in zwei Blöcken (pro Fee-Szenario)."""
    pd.set_option("display.width", 260)
    pd.set_option("display.float_format", "{:.2f}".format)
    pd.set_option("display.max_columns", 20)

    # ── Block 1: Beide Fee-Szenarien nebeneinander ──────────────────────────
    fee_a = df[df["Fee_€"] == 20.0].set_index("Signal")
    fee_b = df[df["Fee_€"] ==  2.0].set_index("Signal")

    cols = ["Rendite_%", "CAGR_%", "MaxDD_%", "Sharpe",
            "N_Trades", "Hit_%", "Payoff", "AvgHold_d", "Fees_€", "Endkap_€"]

    merged = fee_a[cols].copy()
    merged.columns = [f"F20_{c}" for c in cols]
    for c in cols:
        merged[f"F02_{c}"] = fee_b[c]

    print("\n\n" + "=" * 120)
    print("  SIGNAL-VERGLEICH  |  v6.3 Trade-Manager  |  Startkapital: "
          f"{df['Endkap_€'].iloc[0] and 10000:.0f} EUR")
    print("=" * 120)
    print(f"\n  {'Signal':<20}  "
          f"{'━━━ FEE 20€/Order (Bank) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━':55}  "
          f"{'━━━ FEE 2€/Order (Neo-Broker) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━':55}")
    print(f"  {'':20}  "
          f"{'Rendite%':>9}  {'CAGR%':>7}  {'MaxDD%':>7}  {'Sharpe':>7}  "
          f"{'Trades':>7}  {'Hit%':>5}  {'Payoff':>7}  {'Hold-d':>7}  "
          f"{'Fees€':>8}  {'Endkap€':>9}  "
          f"{'Rendite%':>9}  {'CAGR%':>7}  {'MaxDD%':>7}  {'Sharpe':>7}  "
          f"{'Trades':>7}  {'Hit%':>5}  {'Payoff':>7}  {'Hold-d':>7}  "
          f"{'Fees€':>8}  {'Endkap€':>9}")
    print("  " + "─" * 183)

    # Nach F02 Rendite sortieren für übersichtlicheres Ranking
    order = fee_b["Rendite_%"].sort_values(ascending=False).index
    for sig in order:
        a = fee_a.loc[sig]
        b = fee_b.loc[sig]
        marker_a = " ★" if a["Rendite_%"] == fee_a["Rendite_%"].max() else "  "
        marker_b = " ★" if b["Rendite_%"] == fee_b["Rendite_%"].max() else "  "
        print(f"  {sig:<20}{marker_a}"
              f"  {a['Rendite_%']:>+8.2f}%"
              f"  {a['CAGR_%']:>+6.2f}%"
              f"  {a['MaxDD_%']:>+6.1f}%"
              f"  {a['Sharpe']:>7.2f}"
              f"  {int(a['N_Trades']):>7}"
              f"  {a['Hit_%']:>5.1f}%"
              f"  {a['Payoff']:>7.2f}"
              f"  {a['AvgHold_d']:>6.1f}d"
              f"  {a['Fees_€']:>8,.0f}"
              f"  {a['Endkap_€']:>9,.0f}"
              f"  {marker_b}"
              f"  {b['Rendite_%']:>+8.2f}%"
              f"  {b['CAGR_%']:>+6.2f}%"
              f"  {b['MaxDD_%']:>+6.1f}%"
              f"  {b['Sharpe']:>7.2f}"
              f"  {int(b['N_Trades']):>7}"
              f"  {b['Hit_%']:>5.1f}%"
              f"  {b['Payoff']:>7.2f}"
              f"  {b['AvgHold_d']:>6.1f}d"
              f"  {b['Fees_€']:>8,.0f}"
              f"  {b['Endkap_€']:>9,.0f}")

    print("  " + "─" * 183)
    print(f"  ★ = Bestes Ergebnis in diesem Szenario")

    # ── Block 2: Signal-Diagnose ──────────────────────────────────────────────
    print(f"\n\n  SIGNAL-DIAGNOSE  (Earned/Tight Aufteilung, Signale/Jahr)")
    print("  " + "─" * 90)
    print(f"  {'Signal':<20}  {'Sig/Jahr':>9}  "
          f"{'Trades(F20)':>12}  {'Earned(F20)':>12}  {'Tight(F20)':>11}  "
          f"{'MaxWin(F20)':>12}  {'ExpVal(F20)':>12}")
    print("  " + "─" * 90)
    f20 = df[df["Fee_€"] == 20.0].set_index("Signal")
    for sig in order:
        r = f20.loc[sig]
        print(f"  {sig:<20}  {int(r['Sig/Jahr']):>9,}  "
              f"  {int(r['N_Trades']):>12}  "
              f"  {int(r['N_Earned']):>12}  "
              f"  {int(r['N_Tight']):>11}  "
              f"  {r['MaxWin_%']:>+11.1f}%  "
              f"  {r['Expect_%']:>+11.2f}%")

    # ── Block 3: Break-Even Analyse ───────────────────────────────────────────
    print(f"\n\n  BREAK-EVEN ANALYSE  (ab welchem Kapital wird das Signal profitabel?)")
    print("  " + "─" * 70)
    print(f"  {'Signal':<20}  {'F20-Rendite%':>13}  {'F02-Rendite%':>13}  "
          f"{'Gebühr-Delta':>13}  {'Schätzung Min-Kapital':>22}")
    print("  " + "─" * 70)

    f02 = df[df["Fee_€"] == 2.0].set_index("Signal")
    for sig in order:
        r20 = f20.loc[sig]
        r02 = f02.loc[sig]
        delta = r02["Rendite_%"] - r20["Rendite_%"]
        # Grobe Schätzung: Min-Kapital so dass Gebühren < 2% der Anlage
        n_trades  = max(r20["N_Trades"], 1)
        min_cap   = n_trades * 20.0 * 2 / 0.02  # 2% Gebührenanteil als Grenzwert
        print(f"  {sig:<20}  {r20['Rendite_%']:>+12.2f}%  {r02['Rendite_%']:>+12.2f}%  "
              f"  {delta:>+12.2f}%  "
              f"  ca. {min_cap:>12,.0f} €")

    print()


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Signal-Optimierung für v6.3 Trade-Manager")
    parser.add_argument("--years",             type=float, default=7.0)
    parser.add_argument("--capital",           type=float, default=INITIAL_CAPITAL)
    parser.add_argument("--max-pos",           type=int,   default=DEFAULT_MAX_POS)
    parser.add_argument("--atr-init",          type=float, default=DEFAULT_ATR_INIT)
    parser.add_argument("--atr-trail",         type=float, default=DEFAULT_ATR_TRAIL)
    parser.add_argument("--rotation-factor",   type=float, default=DEFAULT_ROTATION_FACTOR)
    parser.add_argument("--pyramid-threshold", type=float, default=DEFAULT_PYRAMID_THRESHOLD)
    parser.add_argument("--max-pyramids",      type=int,   default=DEFAULT_MAX_PYRAMIDS)
    parser.add_argument("--no-pyramid",        action="store_true")
    parser.add_argument("--save-csv",          action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("  SIGNAL-OPTIMIERUNG v6.3  |  5 Signale × 2 Gebühren-Szenarien")
    print("=" * 70)
    print(f"""
  Trade-Manager (fest, unveränderlich):
    Startkapital:      {args.capital:>10,.0f} €
    Max. Slots:        {args.max_pos:>10}
    Stop Phase 1:      {args.atr_init:>9.1f}× ATR14  (Tight)
    Stop Phase 2:      {args.atr_trail:>9.1f}× ATR14  (Earned Trail)
    Rotations-Faktor:  {args.rotation_factor:>9.1f}× (nur profitable Positionen)
    Pyramidisieren:    {'DEAKTIVIERT' if args.no_pyramid else f'ab +{args.pyramid_threshold:.0%} | max {args.max_pyramids}× | Free-Ride'}

  Signals:  {', '.join(SIGNAL_NAMES.keys())}
  Fees:     {', '.join(f'{f:.0f}€' for f in FEE_SCENARIOS)}
  Gesamt:   {len(SIGNAL_NAMES) * len(FEE_SCENARIOS)} Backtest-Läufe
""")

    # 1. Daten einmalig laden
    tickers = _load_tickers()
    print(f"[1/3] Lade {len(tickers)} Ticker ({args.years:.0f} Jahre)...")
    data = load_universe(tickers, args.years)
    print(f"  {len(data)} Ticker geladen.\n")

    # 2. Vergleichs-Schleife
    print("[2/3] Vergleichs-Läufe starten:")
    print("  " + "─" * 70)
    total_t0  = time.time()
    results   = run_comparison(
        data               = data,
        fee_scenarios      = FEE_SCENARIOS,
        initial_cap        = args.capital,
        max_pos            = args.max_pos,
        atr_init           = args.atr_init,
        atr_trail          = args.atr_trail,
        rotation_factor    = args.rotation_factor,
        pyramid_threshold  = args.pyramid_threshold,
        max_pyramids       = args.max_pyramids,
        enable_pyramid     = not args.no_pyramid,
    )
    print(f"\n  Alle Läufe in {time.time()-total_t0:.1f}s abgeschlossen.")

    # 3. Ausgabe
    print("\n[3/3] Ergebnisse:")
    print_comparison(results)

    # CSV speichern
    if args.save_csv:
        csv_path = _here / "signal_comparison_v6.csv"
        results.to_csv(csv_path, index=False)
        print(f"  Ergebnisse gespeichert: {csv_path}")
    else:
        # Immer speichern für einfache Weiterverarbeitung
        csv_path = _here / "signal_comparison_v6.csv"
        results.to_csv(csv_path, index=False)
        print(f"  Vergleich gespeichert:  {csv_path}")


if __name__ == "__main__":
    main()
