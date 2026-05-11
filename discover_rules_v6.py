"""
discover_rules_v6.py  (Architektur v6.5)
====================================================================================
Rule Discovery Engine mit Trigger-Filter-Matrix  |  v6.3 Trade-Manager  |  260 US

ARCHITEKTUR:
    Klare Trennung zwischen zwei konzeptionell verschiedenen Typen:

    POOL A – TRIGGER (Events):
        Feuern exakt an dem Tag, an dem die Bedingung von False → True wechselt.
        Beispiel: Close kreuzt heute den 50-Tage-Höchstkurs → genau 1 Signal.

    POOL B – FILTER (States):
        Müssen am selben Tag EBENFALLS True sein, aber erzeugen keinen Trigger.
        Beispiel: Close > SMA_200  →  Nur in etabliertem Aufwärtstrend kaufen.

    Strategie = [genau 1 Trigger] AND [0, 1 oder 2 Filter]

    → Damit ist jede Strategie ökonomisch sauber interpretierbar:
      "Kaufe WENN <Ereignis> passiert, ABER NUR wenn <Marktbedingungen> stimmen."

FILTER (Signal-Zähler Grenzen):
    Minimum:  40 Signale  (statistisch signifikant)
    Maximum: 3500 Signale (Gebühren-Schutz – bei 20€/Order mit ≤5 Positionen)

Hardcoded (identisch zu backtest_v6.py):
    INITIAL_CAPITAL = 10.000 €
    ORDER_FEE       = 20 € / Order  (Round-Trip = 40 €!)
    ATR Phase 1:    2.0×  (Tight Stop – Fake-Breakout-Schutz)
    ATR Phase 2:    3.5×  (Earned Trail – nach Profit-Nachweis)

Verwendung:
    python discover_rules_v6.py
    python discover_rules_v6.py --years 5
    python discover_rules_v6.py --no-pyramid --min-signals 20
"""

from __future__ import annotations

import argparse
import itertools
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
    _load_tickers, _atr,
    run_backtest, compute_metrics,
)

# ── Discovery-Konstanten ─────────────────────────────────────────────────────
MIN_SIGNALS   = 40     # statistisch zu dünn darunter
MAX_SIGNALS   = 3500   # ≈ 500 pro Jahr auf 260 Aktien → noch beherrschbar
TOP_N         = 15
DEFAULT_YEARS = 7.0

# ==============================================================================
# GENE POOL  –  klar in TRIGGER und FILTER getrennt
# ==============================================================================

TRIGGER_POOL: dict[str, str] = {
    "Trig_B50":       "Close kreuzt  High_50d_prev  nach oben",
    "Trig_B100":      "Close kreuzt  High_100d_prev nach oben",
    "Trig_MACDcross": "MACD-Hist kreuzt 0 nach oben",
    "Trig_RSI30":     "RSI_14 kreuzt 30 nach oben  (Erholung beginnt)",
}

FILTER_POOL: dict[str, str] = {
    "Filt_SMA200":    "Close > SMA_200               (Langfristiger Aufwärtstrend)",
    "Filt_VolSpike":  "Volume > SMA_Vol_20 × 1.5     (Volumen-Bestätigung)",
    "Filt_TrendAlign":"SMA_20 > SMA_50 > SMA_200     (Vollständige Ausrichtung)",
    "Filt_RSIbull":   "RSI_14 > 55                   (Momentum positiv)",
    "Filt_BBsqueeze": "(BB_hi - BB_lo) / Close < 0.10 (Bollinger-Kompression)",
}


# ==============================================================================
# 1. DATEN LADEN  (OHLCV)
# ==============================================================================

def _load_universe(tickers: list[str], years: float) -> dict[str, pd.DataFrame]:
    raw_dir = _here / "data" / "raw"
    files   = sorted(raw_dir.glob("*_1d.parquet"))
    if not files:
        raise FileNotFoundError(f"Keine Parquet-Dateien in {raw_dir}")

    cutoff = pd.Timestamp.today() - pd.Timedelta(days=int(years * 365.25))
    data: dict[str, pd.DataFrame] = {}

    for fpath in files:
        ticker = fpath.stem.replace("_1d", "")
        if ticker not in tickers:
            continue
        try:
            df = pd.read_parquet(fpath)
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.columns = [c.lower() for c in df.columns]
            df = df[df.index >= cutoff]
            if len(df) < 260:
                continue
            needed = [c for c in ["open", "high", "low", "close", "volume"]
                      if c in df.columns]
            if not {"open", "high", "low", "close"}.issubset(df.columns):
                continue
            data[ticker] = df[needed]
        except Exception:
            pass
    return data


# ==============================================================================
# 2. INDIKATOREN
# ==============================================================================

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd_hist(close: pd.Series,
               fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    macd = (close.ewm(span=fast, adjust=False).mean()
           - close.ewm(span=slow, adjust=False).mean())
    return macd - macd.ewm(span=signal, adjust=False).mean()


def _transition(series: pd.Series) -> pd.Series:
    """True genau an den Tagen, an denen series von False → True wechselt."""
    return series & ~series.shift(1).fillna(False)


# ==============================================================================
# 3. PIVOT-TABELLEN VORBERECHNEN  (einmalig – für alle Strategien)
# ==============================================================================

def build_all_pivots(
    data: dict[str, pd.DataFrame],
) -> tuple[
    dict[str, pd.DataFrame],   # base_pivots (open, close, high, atr14, trend_str)
    dict[str, pd.DataFrame],   # trigger_pivots  (Boolean Event)
    dict[str, pd.DataFrame],   # filter_pivots   (Boolean State)
]:
    base_cols: dict[str, dict] = {
        "open": {}, "close": {}, "high": {}, "atr14": {}, "trend_str": {}
    }
    trig_raw: dict[str, dict] = {k: {} for k in TRIGGER_POOL}
    filt_raw: dict[str, dict] = {k: {} for k in FILTER_POOL}

    has_volume = False

    for ticker, df in data.items():
        c      = df["close"]
        h      = df["high"]
        vol    = df.get("volume")
        sma20  = c.rolling(20).mean()
        sma50  = c.rolling(50).mean()
        sma200 = c.rolling(200).mean()
        atr14  = _atr(df, 14)
        rsi14  = _rsi(c)
        mhist  = _macd_hist(c)
        bb_std = c.rolling(20).std()
        bb_lo  = sma20 - 2 * bb_std
        bb_hi  = sma20 + 2 * bb_std

        valid = sma200.notna() & atr14.notna()
        idx   = c[valid].index

        # ── Basis ────────────────────────────────────────────────────────────
        base_cols["open"][ticker]      = df["open"][valid]
        base_cols["close"][ticker]     = c[valid]
        base_cols["high"][ticker]      = h[valid]
        base_cols["atr14"][ticker]     = atr14[valid]
        base_cols["trend_str"][ticker] = ((c - sma200) / sma200)[valid]

        # ── Trigger (Events – Transitions) ──────────────────────────────────
        trig_raw["Trig_B50"][ticker]       = _transition(
            c > h.shift(1).rolling(50).max()).reindex(idx).fillna(False)
        trig_raw["Trig_B100"][ticker]      = _transition(
            c > h.shift(1).rolling(100).max()).reindex(idx).fillna(False)
        trig_raw["Trig_MACDcross"][ticker] = _transition(
            mhist > 0).reindex(idx).fillna(False)
        trig_raw["Trig_RSI30"][ticker]     = _transition(
            rsi14 > 30).reindex(idx).fillna(False)

        # ── Filter (States – keine Transition) ───────────────────────────────
        filt_raw["Filt_SMA200"][ticker]     = (c > sma200).reindex(idx).fillna(False)
        filt_raw["Filt_TrendAlign"][ticker] = (
            (sma20 > sma50) & (sma50 > sma200)).reindex(idx).fillna(False)
        filt_raw["Filt_RSIbull"][ticker]    = (rsi14 > 55).reindex(idx).fillna(False)
        filt_raw["Filt_BBsqueeze"][ticker]  = (
            (bb_hi - bb_lo) / c.replace(0, np.nan) < 0.10
        ).reindex(idx).fillna(False)

        if vol is not None:
            has_volume = True
            vol_sma = vol.rolling(20).mean()
            filt_raw["Filt_VolSpike"][ticker] = (
                vol > vol_sma * 1.5).reindex(idx).fillna(False)
        else:
            filt_raw["Filt_VolSpike"][ticker] = pd.Series(False, index=idx)

    base = {k: pd.DataFrame(v) for k, v in base_cols.items()}
    trigs = {k: pd.DataFrame(v).fillna(False).astype(bool) for k, v in trig_raw.items()}
    filts = {k: pd.DataFrame(v).fillna(False).astype(bool) for k, v in filt_raw.items()}

    return base, trigs, filts, has_volume


# ==============================================================================
# 4. KOMBINATORIK  (1 Trigger + 0/1/2 Filter)
# ==============================================================================

def _build_strategy_name(trigger: str, filters: tuple[str, ...]) -> str:
    tname = trigger.replace("Trig_", "")
    if not filters:
        return tname
    fnames = " + ".join(f.replace("Filt_", "") for f in filters)
    return f"{tname}  |  {fnames}"


def run_discovery(
    base:    dict[str, pd.DataFrame],
    trigs:   dict[str, pd.DataFrame],
    filts:   dict[str, pd.DataFrame],
    min_sig: int,
    max_sig: int,
    fee:     float,
    **engine_kwargs,
) -> pd.DataFrame:
    """Testet alle [1 Trigger] × [0-2 Filter] Kombinationen."""

    # Alle Filter-Kombinationen (Größe 0, 1, 2)
    filter_combos: list[tuple[str, ...]] = [()]
    for size in (1, 2):
        filter_combos.extend(itertools.combinations(FILTER_POOL.keys(), size))

    strategies = [
        (trig, fcombo)
        for trig    in TRIGGER_POOL.keys()
        for fcombo  in filter_combos
    ]
    n_total   = len(strategies)
    results   = []
    n_tested  = 0
    n_too_few = 0
    n_too_many= 0
    t_start   = time.time()

    print(f"\n  Trigger:              {len(TRIGGER_POOL)} Stück")
    print(f"  Filter-Kombinationen: {len(filter_combos)} "
          f"(0 Filter + {len(FILTER_POOL)} Einzel + C({len(FILTER_POOL)},2) Paare)")
    print(f"  Gesamtstrategien:     {n_total}")
    print(f"  Vorfilter:            {min_sig} ≤ Signale ≤ {max_sig}")
    print(f"  {'─' * 65}")

    for idx, (trig, fcombo) in enumerate(strategies):

        if idx > 0 and idx % 10 == 0:
            elapsed   = time.time() - t_start
            speed     = elapsed / idx
            remaining = speed * (n_total - idx)
            print(f"  {idx:>3}/{n_total}  "
                  f"({idx/n_total*100:>4.1f}%)  "
                  f"Getestet: {n_tested:>2}  |  "
                  f"Zu wenig: {n_too_few:>2}  Zu viel: {n_too_many:>2}  |  "
                  f"ETA: {remaining/60:.1f} min",
                  flush=True)

        # ── Eintrittssignal aufbauen ─────────────────────────────────────────
        entry_sig = trigs[trig].copy()
        for fname in fcombo:
            entry_sig = entry_sig & filts[fname]

        n_sig = int(entry_sig.values.sum())

        if n_sig < min_sig:
            n_too_few += 1
            continue
        if n_sig > max_sig:
            n_too_many += 1
            continue

        n_tested += 1

        # ── Vollständige Simulation ──────────────────────────────────────────
        try:
            eq_df, completed, _ = run_backtest(
                pivots   = {**base, "entry_sig": entry_sig},
                fee      = fee,
                verbose  = False,
                **engine_kwargs,
            )
        except Exception as e:
            print(f"  [WARN] {_build_strategy_name(trig, fcombo)}: {e}")
            continue

        m = compute_metrics(eq_df, completed, engine_kwargs["initial_cap"], fee)

        n_earned   = sum(1 for t in completed if t.get("earned_mode", False))
        earned_pct = n_earned / len(completed) * 100 if completed else 0.0

        results.append({
            "Strategie":   _build_strategy_name(trig, fcombo),
            "Trigger":     trig.replace("Trig_", ""),
            "Filter":      " + ".join(f.replace("Filt_", "") for f in fcombo) or "—",
            "N_Filter":    len(fcombo),
            "Rendite_%":   m["total_ret_%"],
            "CAGR_%":      m["cagr_%"],
            "MaxDD_%":     m["max_dd_%"],
            "Sharpe":      m["sharpe"],
            "N_Trades":    m["n_trades"],
            "Hit_%":       m["hit_%"],
            "Payoff":      m["payoff"],
            "Expect_%":    m["expect_%"],
            "AvgHold_d":   m["avg_hold_d"],
            "MaxWin_%":    m["max_win_%"],
            "Earned_%":    round(earned_pct, 1),
            "Fees_€":      m["total_fees"],
            "Endkap_€":    m["end_equity"],
            "N_Signals":   n_sig,
        })

    elapsed_total = time.time() - t_start
    print(f"\n  {'─' * 65}")
    print(f"  Fertig!   Gesamt: {n_total}  |  Getestet: {n_tested}  |  "
          f"Zu wenig Signale: {n_too_few}  |  Zu viele: {n_too_many}  |  "
          f"Zeit: {elapsed_total/60:.1f} min")

    return pd.DataFrame(results)


# ==============================================================================
# 5. AUSGABE
# ==============================================================================

def print_results(df: pd.DataFrame, fee: float, top_n: int = TOP_N) -> None:
    if df.empty:
        print("\n  Keine Ergebnisse – alle Strategien gefiltert.")
        print("  Tipp: --min-signals kleiner oder --max-signals größer setzen.")
        return

    df_sorted = df.sort_values("Rendite_%", ascending=False).reset_index(drop=True)
    top        = df_sorted.head(top_n)

    pd.set_option("display.width", 300)
    pd.set_option("display.float_format", "{:.2f}".format)

    hdr = (f"  {'Rg':>3}  {'Strategie':<40}  {'Nfil':>4}  "
           f"{'Rendite%':>9}  {'CAGR%':>7}  {'MaxDD%':>7}  "
           f"{'Sharpe':>7}  {'Trades':>6}  {'Hit%':>5}  "
           f"{'Payoff':>7}  {'Hold-d':>7}  {'Earned%':>8}  "
           f"{'Fees€':>7}  {'Endkap€':>9}")
    sep = f"  {'─' * 125}"

    print(f"\n{'=' * 130}")
    print(f"  TOP {top_n} STRATEGIEN  |  Fee = {fee:.0f}€/Order  |  "
          f"Startkapital {INITIAL_CAPITAL:,.0f}€  |  "
          f"ATR-Stop: {DEFAULT_ATR_INIT}× Tight → {DEFAULT_ATR_TRAIL}× Earned")
    print(f"{'=' * 130}")
    print(hdr)
    print(sep)

    for rank, row in top.iterrows():
        marker = " ★" if rank == 0 else "  "
        strat  = row["Strategie"]
        if len(strat) > 40:
            strat = strat[:38] + ".."
        print(f"  {rank+1:>3}{marker}  {strat:<40}  {int(row['N_Filter']):>4}  "
              f"  {row['Rendite_%']:>+8.2f}%"
              f"  {row['CAGR_%']:>+6.2f}%"
              f"  {row['MaxDD_%']:>+6.1f}%"
              f"  {row['Sharpe']:>7.2f}"
              f"  {int(row['N_Trades']):>6}"
              f"  {row['Hit_%']:>5.1f}%"
              f"  {row['Payoff']:>7.2f}"
              f"  {row['AvgHold_d']:>6.1f}d"
              f"  {row['Earned_%']:>7.1f}%"
              f"  {row['Fees_€']:>7,.0f}"
              f"  {row['Endkap_€']:>9,.0f}")

    print(sep)

    # ── Statistiken nach Trigger ─────────────────────────────────────────────
    print(f"\n  PERFORMANCE NACH TRIGGER:")
    print(f"  {'─' * 80}")
    for trig, grp in df_sorted.groupby("Trigger"):
        pos    = (grp["Rendite_%"] > 0).sum()
        best   = grp["Rendite_%"].max()
        avg    = grp["Rendite_%"].mean()
        print(f"  {trig:<15}  {len(grp):>3} Strat.  |  "
              f"Avg: {avg:>+7.2f}%  |  Best: {best:>+7.2f}%  |  "
              f"Positiv: {pos}/{len(grp)}")

    # ── Statistiken nach Anzahl Filter ──────────────────────────────────────
    print(f"\n  PERFORMANCE NACH ANZAHL FILTER:")
    print(f"  {'─' * 80}")
    for nf, grp in df_sorted.groupby("N_Filter"):
        pos    = (grp["Rendite_%"] > 0).sum()
        best   = grp["Rendite_%"].max()
        avg    = grp["Rendite_%"].mean()
        label  = {0: "kein Filter (roh)", 1: "1 Filter", 2: "2 Filter"}.get(nf, str(nf))
        print(f"  {label:<18}  {len(grp):>3} Strat.  |  "
              f"Avg: {avg:>+7.2f}%  |  Best: {best:>+7.2f}%  |  "
              f"Positiv: {pos}/{len(grp)}")

    # ── Flop-Tabelle (nur wenn genug Ergebnisse) ─────────────────────────────
    if len(df_sorted) > top_n:
        flop = df_sorted.tail(5)
        print(f"\n  FLOP 5 (schlechteste Strategien):")
        print(f"  {'─' * 80}")
        for _, row in flop.iterrows():
            strat = row["Strategie"]
            if len(strat) > 40:
                strat = strat[:38] + ".."
            print(f"  {strat:<42}  {row['Rendite_%']:>+8.2f}%  "
                  f"Trades: {int(row['N_Trades']):>4}  "
                  f"Fees: {row['Fees_€']:>7,.0f}€")

    # ── Ingredient Ranking ───────────────────────────────────────────────────
    n_rank = min(30, len(df_sorted))
    if n_rank > 3:
        print(f"\n  INGREDIENT RANKING  (Häufigkeit in den Top-{n_rank} Strategien):")
        print(f"  {'─' * 70}")
        counts: dict[str, int] = {}
        for _, row in df_sorted.head(n_rank).iterrows():
            for part in row["Trigger"].split():
                counts[part] = counts.get(part, 0) + 1
            for f in (row["Filter"].split(" + ") if row["Filter"] != "—" else []):
                counts[f] = counts.get(f, 0) + 1
        for name, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            pct = cnt / n_rank * 100
            bar = "█" * int(pct / 5)
            print(f"  {name:<15}  {cnt:>3}× ({pct:>4.0f}%)  {bar}")

    print()


# ==============================================================================
# 6. MAIN
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rule Discovery Engine v6.5  (Trigger-Filter-Architektur)")
    parser.add_argument("--years",             type=float, default=DEFAULT_YEARS)
    parser.add_argument("--capital",           type=float, default=INITIAL_CAPITAL)
    parser.add_argument("--fee",               type=float, default=ORDER_FEE)
    parser.add_argument("--max-pos",           type=int,   default=DEFAULT_MAX_POS)
    parser.add_argument("--atr-init",          type=float, default=DEFAULT_ATR_INIT)
    parser.add_argument("--atr-trail",         type=float, default=DEFAULT_ATR_TRAIL)
    parser.add_argument("--rotation-factor",   type=float, default=DEFAULT_ROTATION_FACTOR)
    parser.add_argument("--pyramid-threshold", type=float, default=DEFAULT_PYRAMID_THRESHOLD)
    parser.add_argument("--max-pyramids",      type=int,   default=DEFAULT_MAX_PYRAMIDS)
    parser.add_argument("--no-pyramid",        action="store_true")
    parser.add_argument("--min-signals",       type=int,   default=MIN_SIGNALS)
    parser.add_argument("--max-signals",       type=int,   default=MAX_SIGNALS)
    args = parser.parse_args()

    print("=" * 70)
    print("  RULE DISCOVERY ENGINE v6.5  |  Trigger-Filter-Matrix")
    print("=" * 70)
    print(f"""
  Trade-Manager (fest):
    Startkapital:      {args.capital:>10,.0f} €
    Order-Fee:         {args.fee:>10.0f} €  (Round-Trip: {args.fee*2:.0f}€)
    ATR Phase 1:       {args.atr_init:>9.1f}× ATR14  (Tight)
    ATR Phase 2:       {args.atr_trail:>9.1f}× ATR14  (Earned Trail)
    Rotation-Faktor:   {args.rotation_factor:>9.1f}× (nur profitable Positionen)
    Pyramidisieren:    {'DEAKTIVIERT' if args.no_pyramid else f'ab +{args.pyramid_threshold:.0%}  (Free-Ride-Schutz)'}
    Max. Positionen:   {args.max_pos}

  Discovery-Parameter:
    Datenzeitraum:     {args.years:.0f} Jahre
    Signal-Minimum:    {args.min_signals}   (statistisch signifikant)
    Signal-Maximum:    {args.max_signals}   (Gebühren-Schutz)
""")

    print("  TRIGGER POOL:")
    for k, v in TRIGGER_POOL.items():
        print(f"    {k:<18}  {v}")
    print()
    print("  FILTER POOL:")
    for k, v in FILTER_POOL.items():
        print(f"    {k:<18}  {v}")
    print()

    # 1. Tickers + Daten
    tickers = _load_tickers()
    print(f"[1/3] Lade {len(tickers)} Ticker ({args.years:.0f} Jahre)...")
    t0   = time.time()
    data = _load_universe(tickers, args.years)
    has_vol = any("volume" in df.columns for df in data.values())
    print(f"  {len(data)} Ticker geladen in {time.time()-t0:.1f}s  "
          f"| Volume: {'vorhanden ✓' if has_vol else 'fehlt – Filt_VolSpike immer False'}")

    # 2. Alle Pivots einmalig bauen
    print(f"\n[2/3] Berechne Basis- und Indikator-Pivots (einmalig)...")
    t0 = time.time()
    base, trigs, filts, _ = build_all_pivots(data)
    dates = base["open"].index
    n_tick = len(base["open"].columns)
    print(f"  Zeitraum: {dates[0].date()} → {dates[-1].date()}  "
          f"({len(dates):,} Handelstage, {n_tick} Ticker)")
    print(f"  Pivot-Aufbau: {time.time()-t0:.1f}s")

    # Trigger-Diagnose
    years_actual = (dates[-1] - dates[0]).days / 365.25
    print(f"\n  TRIGGER-DIAGNOSE (Signals über {years_actual:.1f} Jahre):")
    for tname, tpivot in trigs.items():
        n = int(tpivot.values.sum())
        print(f"    {tname:<20}  {n:>7,} Signals  "
              f"({n/years_actual:>6.0f}/Jahr   "
              f"{'OK ✓' if args.min_signals <= n <= args.max_signals else 'REIN als Einzel gefiltert'})")

    # 3. Discovery
    print(f"\n[3/3] Starte Discovery-Schleife...")
    results = run_discovery(
        base       = base,
        trigs      = trigs,
        filts      = filts,
        min_sig    = args.min_signals,
        max_sig    = args.max_signals,
        fee        = args.fee,
        initial_cap        = args.capital,
        max_pos            = args.max_pos,
        atr_init           = args.atr_init,
        atr_trail          = args.atr_trail,
        rotation_factor    = args.rotation_factor,
        pyramid_threshold  = args.pyramid_threshold,
        max_pyramids       = args.max_pyramids,
        enable_pyramid     = not args.no_pyramid,
        market_filter      = None,
    )

    if results.empty:
        return

    # 4. Ausgabe + CSV
    print_results(results, args.fee, TOP_N)

    csv_path = _here / "rule_discovery_v65.csv"
    results.sort_values("Rendite_%", ascending=False).to_csv(csv_path, index=False)
    print(f"  {len(results)} Ergebnisse exportiert: {csv_path}\n")

    # 5. Fazit
    best = results.loc[results["Rendite_%"].idxmax()]
    print(f"  FAZIT:")
    print(f"  {'─' * 65}")
    print(f"  Beste Strategie:   '{best['Strategie']}'")
    print(f"  Rendite:           {best['Rendite_%']:>+.2f}%  "
          f"(CAGR {best['CAGR_%']:>+.2f}%  |  MaxDD {best['MaxDD_%']:>+.1f}%)")
    print(f"  Sharpe:            {best['Sharpe']:.2f}")
    print(f"  Trades:            {int(best['N_Trades'])}  "
          f"| Hit: {best['Hit_%']:.1f}%  | Payoff: {best['Payoff']:.2f}")
    print(f"  Endkapital:        {best['Endkap_€']:,.0f} €  "
          f"(Start: {args.capital:,.0f} €)")
    print()


if __name__ == "__main__":
    main()
