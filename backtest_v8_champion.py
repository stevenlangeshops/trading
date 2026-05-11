"""
backtest_v8_champion.py
====================================================================================
Champion Run  |  VCP v8.3  |  Breadth=0.0  |  Rotation=1.5×  |  Diamond Hands

Aus der 2D Grid-Search (optimize_v8_gridsearch.py) als beste Kombination ermittelt.

Hardcoded Champion-Parameter:
    INITIAL_CAPITAL   = 10.000 €
    ORDER_FEE         = 20 €  (Round-Trip = 40 €)
    MAX_POSITIONS     = 2
    BREADTH_THRESHOLD = 0.0  (kein Filter, volle Kaufbereitschaft)
    ROTATION_FACTOR   = 1.5  (+ Diamond Hands: Earned-Mode Positionen sind geschützt)

Neu gegenüber v8.2:
    ► Erweiterte Trade-Lifecycle-Analyse:
        - Avg Hold Time: Winners vs. Losers getrennt
        - Rotation-Exit-Analyse: Was haben wir im Durchschnitt bezahlt?
        - Max Unrealized Profit pro Trade (Peak-Profit vor Exit)
        - Earned-Mode Conversion Rate
    ► Hochauflösender Chart (champion_equity.png):
        - Panel 1: Equity-Kurve + rote Drawdown-Zonen + Annotation Max-DD
        - Panel 2: Gantt-Chart (2 Slots) mit Ticker-Labels + Return-Färbung
        - Panel 3: Jahresrenditen als Bar-Chart
    ► champion_trades.csv: Alle Trades mit vollständigen Metadaten

Verwendung:
    python backtest_v8_champion.py
    python backtest_v8_champion.py --years 7
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_here = Path(__file__).parent
sys.path.insert(0, str(_here))

from backtest_v6 import _load_tickers, _atr

# ── Champion-Parameter (hardcoded, unveränderlich) ───────────────────────────
INITIAL_CAPITAL     = 10_000.0
ORDER_FEE           = 20.0
MAX_POSITIONS       = 2
ATR_INIT            = 2.0
ATR_TRAIL           = 3.5
BB_PERIOD           = 20
BB_STD              = 2.0
BB_SQUEEZE_THRESH   = 0.10
VOL_MULTIPLIER      = 1.5
PYRAMID_THRESHOLD   = 0.20
MAX_PYRAMIDS        = 1
MIN_SHARES          = 5
BREADTH_THRESHOLD   = 0.0   # Champion: kein Breadth-Filter
ROTATION_FACTOR     = 1.5   # Champion: Rotation mit Diamond Hands
DEFAULT_YEARS       = 7.0

_RAW_DIR   = _here / "data" / "raw"
_OUT_PNG   = _here / "champion_equity.png"
_OUT_CSV   = _here / "champion_trades.csv"


# ==============================================================================
# 1. DATEN LADEN
# ==============================================================================

def _load_ohlcv(tickers: list[str], years: float) -> dict[str, pd.DataFrame]:
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=int(years * 365.25))
    tset   = set(tickers)
    data:  dict[str, pd.DataFrame] = {}
    for fpath in sorted(_RAW_DIR.glob("*_1d.parquet")):
        ticker = fpath.stem.replace("_1d", "")
        if ticker not in tset:
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
            if not {"open", "high", "low", "close"}.issubset(df.columns):
                continue
            keep = [c for c in ["open","high","low","close","volume"]
                    if c in df.columns]
            data[ticker] = df[keep]
        except Exception:
            pass
    return data


# ==============================================================================
# 2. PIVOTS
# ==============================================================================

def build_panels(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    cols: dict[str, dict] = {
        k: {} for k in ["open","close","high","low","atr14","trend_str","entry_sig"]
    }
    for ticker, df in data.items():
        c   = df["close"]
        h   = df["high"]
        vol = df.get("volume")

        sma200    = c.rolling(200).mean()
        sma20_vol = vol.rolling(20).mean() if vol is not None else None
        atr14     = _atr(df, 14)
        sma_bb    = c.rolling(BB_PERIOD).mean()
        std_bb    = c.rolling(BB_PERIOD).std()
        bb_w      = (sma_bb + BB_STD*std_bb
                     - (sma_bb - BB_STD*std_bb)) / c.replace(0, np.nan)

        valid = sma200.notna() & atr14.notna()
        idx   = c[valid].index

        cols["open"][ticker]      = df["open"][valid]
        cols["close"][ticker]     = c[valid]
        cols["high"][ticker]      = h[valid]
        cols["low"][ticker]       = df["low"][valid]
        cols["atr14"][ticker]     = atr14[valid]
        cols["trend_str"][ticker] = ((c - sma200) / sma200)[valid]

        high50_prev  = h.shift(1).rolling(50).max()
        b50_raw      = c > high50_prev
        trig_b50     = b50_raw & ~b50_raw.shift(1).fillna(False)
        squeeze_prev = bb_w.shift(1) < BB_SQUEEZE_THRESH
        vol_ok       = (vol > sma20_vol * VOL_MULTIPLIER
                        if (vol is not None and sma20_vol is not None)
                        else pd.Series(False, index=c.index))
        trend_ok     = c > sma200
        sig          = trig_b50 & squeeze_prev & vol_ok & trend_ok
        cols["entry_sig"][ticker] = sig.reindex(idx).fillna(False).astype(bool)

    return {k: pd.DataFrame(v) for k, v in cols.items()}


# ==============================================================================
# 3. SIMULATION ENGINE (erweitertes Trade-Tracking)
# ==============================================================================

def _update_stop(pos: dict, today_high: float, today_atr: float) -> None:
    if math.isnan(today_high) or math.isnan(today_atr) or today_atr <= 0:
        return
    if today_high > pos["max_high"]:
        pos["max_high"] = today_high
    if not pos["earned_mode"]:
        if pos["max_high"] >= pos["entry_price"] + ATR_INIT * pos["atr_at_entry"]:
            pos["earned_mode"] = True
    new_stop = (pos["max_high"] - ATR_TRAIL * today_atr
                if pos["earned_mode"]
                else pos["entry_price"] - ATR_INIT * pos["atr_at_entry"])
    pos["trailing_stop"] = max(pos["trailing_stop"], new_stop)


def run_backtest(
    pivots: dict[str, pd.DataFrame],
) -> tuple[pd.Series, list[dict], int]:
    piv     = pivots
    dates   = piv["open"].index
    tickers = list(piv["open"].columns)

    cash            = INITIAL_CAPITAL
    portfolio:      dict[str, dict] = {}
    free_slots      = {1, 2}
    ticker_to_slot: dict[str, int]  = {}
    completed:      list[dict]      = []
    equity_log:     dict            = {}
    invest_days     = 0

    def _safe(df: pd.DataFrame, date, ticker) -> float:
        try:
            v = df.at[date, ticker]
            return float(v) if pd.notna(v) else math.nan
        except Exception:
            return math.nan

    def _open(ticker: str, buy_date, buy_px: float,
              atr_e: float, slot: int) -> bool:
        nonlocal cash
        mkt    = sum(p["shares"] * p["avg_entry_price"] for p in portfolio.values())
        target = min((cash + mkt) / MAX_POSITIONS, cash - ORDER_FEE)
        shares = int((target - ORDER_FEE) / buy_px) if buy_px > 0 else 0
        if shares < MIN_SHARES or shares * buy_px < ORDER_FEE * 20:
            return False
        cost = shares * buy_px + ORDER_FEE
        if cost > cash:
            return False
        cash -= cost
        portfolio[ticker] = {
            "slot":            slot,
            "entry_date":      buy_date,
            "entry_price":     buy_px,
            "shares":          shares,
            "cost":            cost,
            "atr_at_entry":    atr_e,
            "trailing_stop":   buy_px - ATR_INIT * atr_e,
            "max_high":        buy_px,
            "earned_mode":     False,
            "earned_date":     None,
            "pyramid_count":   0,
            "avg_entry_price": buy_px,
            "days_held":       0,
            "max_unreal_pct":  0.0,   # Peak unrealized profit während des Trades
        }
        ticker_to_slot[ticker] = slot
        free_slots.discard(slot)
        return True

    def _close(ticker: str, sell_date, sell_px: float,
               exit_reason: str) -> None:
        nonlocal cash
        pos      = portfolio[ticker]
        slot     = pos["slot"]
        proceeds = pos["shares"] * sell_px - ORDER_FEE
        pnl      = proceeds - pos["cost"]
        ret_pct  = pnl / pos["cost"] * 100
        cash    += proceeds
        completed.append({
            "ticker":          ticker,
            "slot":            slot,
            "entry_date":      pos["entry_date"],
            "exit_date":       sell_date,
            "entry_price":     round(pos["entry_price"], 2),
            "exit_price":      round(sell_px, 2),
            "shares":          pos["shares"],
            "pnl_€":           round(pnl, 2),
            "ret_%":           round(ret_pct, 2),
            "hold_d":          pos["days_held"],
            "earned_mode":     pos["earned_mode"],
            "earned_date":     pos["earned_date"],
            "pyramid_count":   pos["pyramid_count"],
            "max_unreal_%":    round(pos["max_unreal_pct"], 2),
            "exit_reason":     exit_reason,
            "is_rotation":     exit_reason == "Rotation",
        })
        free_slots.add(slot)
        del portfolio[ticker]
        ticker_to_slot.pop(ticker, None)

    # ── Hauptschleife ────────────────────────────────────────────────────────
    for day_i in range(len(dates) - 1):
        today    = dates[day_i]
        tomorrow = dates[day_i + 1]

        # A. Stops prüfen + Max-Unrealized aktualisieren
        exits: list[str] = []
        for ticker, pos in list(portfolio.items()):
            tc = _safe(piv["close"], today, ticker)
            th = _safe(piv["high"],  today, ticker)
            ta = _safe(piv["atr14"], today, ticker)
            if math.isnan(tc):
                continue
            pos["days_held"] += 1
            _update_stop(pos, th, ta)

            # Earned-Date merken (erster Tag im Earned Mode)
            if pos["earned_mode"] and pos["earned_date"] is None:
                pos["earned_date"] = today

            # Peak-Unrealized aktualisieren
            unreal = (tc - pos["avg_entry_price"]) / pos["avg_entry_price"] * 100
            if unreal > pos["max_unreal_pct"]:
                pos["max_unreal_pct"] = unreal

            if tc < pos["trailing_stop"]:
                exits.append(ticker)

        for ticker in exits:
            sell_px = _safe(piv["open"], tomorrow, ticker)
            if not math.isnan(sell_px):
                _close(ticker, tomorrow, sell_px, exit_reason="ATR")

        # B. Kandidaten (Breadth=0.0 → immer erlaubt)
        candidates = [
            t for t in tickers
            if t not in portfolio
            and _safe(piv["entry_sig"], today, t) == 1.0
        ]
        if candidates:
            candidates.sort(
                key=lambda t: _safe(piv["trend_str"], today, t),
                reverse=True
            )
            for cand in candidates:
                buy_px = _safe(piv["open"],  tomorrow, cand)
                atr_e  = _safe(piv["atr14"], today,    cand)
                if math.isnan(buy_px) or math.isnan(atr_e) or atr_e <= 0:
                    continue

                if len(portfolio) < MAX_POSITIONS and free_slots:
                    slot = min(free_slots)
                    _open(cand, tomorrow, buy_px, atr_e, slot)

                elif len(portfolio) >= MAX_POSITIONS:
                    # Diamond Hands: nur nicht-earned Positionen rotierbar
                    rotatable = {
                        t: p for t, p in portfolio.items()
                        if not p["earned_mode"]
                    }
                    if not rotatable:
                        continue

                    weakest_t   = min(
                        rotatable.keys(),
                        key=lambda t: _safe(piv["trend_str"], today, t)
                    )
                    weakest_str = _safe(piv["trend_str"], today, weakest_t)
                    cand_str    = _safe(piv["trend_str"], today, cand)

                    if (not math.isnan(cand_str)
                            and not math.isnan(weakest_str)
                            and cand_str > ROTATION_FACTOR * weakest_str):
                        rot_px = _safe(piv["open"], tomorrow, weakest_t)
                        if not math.isnan(rot_px):
                            freed = portfolio[weakest_t]["slot"]
                            _close(weakest_t, tomorrow, rot_px,
                                   exit_reason="Rotation")
                            _open(cand, tomorrow, buy_px, atr_e, freed)
                            break

        # C. Pyramidisieren
        if len(portfolio) == 1 and free_slots:
            for ticker, pos in list(portfolio.items()):
                if pos["pyramid_count"] >= MAX_PYRAMIDS:
                    continue
                tc = _safe(piv["close"], today, ticker)
                if math.isnan(tc):
                    continue
                unreal = (tc - pos["avg_entry_price"]) / pos["avg_entry_price"]
                if unreal >= PYRAMID_THRESHOLD:
                    buy_px = _safe(piv["open"],  tomorrow, ticker)
                    atr_e  = _safe(piv["atr14"], today,    ticker)
                    if math.isnan(buy_px) or math.isnan(atr_e) or atr_e <= 0:
                        continue
                    add_sh = int((cash - ORDER_FEE) / buy_px)
                    if add_sh < MIN_SHARES or add_sh * buy_px < ORDER_FEE * 20:
                        continue
                    cost_add = add_sh * buy_px + ORDER_FEE
                    if cost_add > cash:
                        continue
                    old_v   = pos["shares"] * pos["avg_entry_price"]
                    new_avg = (old_v + add_sh * buy_px) / (pos["shares"] + add_sh)
                    cash   -= cost_add
                    pos["cost"]           += cost_add
                    pos["shares"]         += add_sh
                    pos["avg_entry_price"] = new_avg
                    pos["pyramid_count"]  += 1
                    pos["trailing_stop"]   = max(pos["trailing_stop"], new_avg)

        # D. Equity loggen
        equity = cash
        n_held = 0
        for ticker, pos in portfolio.items():
            cp      = _safe(piv["close"], today, ticker)
            equity += pos["shares"] * (cp if not math.isnan(cp)
                                       else pos["avg_entry_price"])
            n_held += 1
        equity_log[today] = equity
        if n_held > 0:
            invest_days += 1

    # Offene Positionen zum letzten Kurs schliessen
    for ticker in list(portfolio.keys()):
        lp = _safe(piv["close"], dates[-1], ticker)
        if not math.isnan(lp):
            _close(ticker, dates[-1], lp, exit_reason="End")

    return pd.Series(equity_log).sort_index(), completed, invest_days


# ==============================================================================
# 4. METRIKEN
# ==============================================================================

def compute_metrics(
    equity:      pd.Series,
    trades:      list[dict],
    invest_days: int,
) -> dict:
    eq    = equity.ffill().bfill()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.1)
    ret   = (eq.iloc[-1] / INITIAL_CAPITAL - 1) * 100
    cagr  = ((eq.iloc[-1] / INITIAL_CAPITAL) ** (1/years) - 1) * 100
    peak  = eq.cummax()
    dd    = (eq - peak) / peak * 100
    dr    = eq.pct_change().dropna()
    sharpe = (dr.mean() / dr.std() * 252**0.5) if dr.std() > 0 else 0

    rets   = [t["ret_%"] for t in trades]
    wins   = [t for t in trades if t["ret_%"] > 0]
    losses = [t for t in trades if t["ret_%"] <= 0]
    atr_exits  = [t for t in trades if t["exit_reason"] == "ATR"]
    rot_exits  = [t for t in trades if t["exit_reason"] == "Rotation"]
    end_exits  = [t for t in trades if t["exit_reason"] == "End"]
    earned_t   = [t for t in trades if t["earned_mode"]]

    hit    = len(wins) / len(rets)  * 100 if rets else 0
    avg_w  = float(np.mean([t["ret_%"] for t in wins]))   if wins   else 0.0
    avg_l  = float(np.mean([t["ret_%"] for t in losses])) if losses else 0.0
    payoff = abs(avg_w / avg_l) if avg_l else float("inf")
    pf     = (sum(t["ret_%"] for t in wins) /
              abs(sum(t["ret_%"] for t in losses))) if losses else float("inf")
    ev     = hit/100 * avg_w + (1 - hit/100) * avg_l

    # Lifecycle
    hold_w   = float(np.mean([t["hold_d"] for t in wins]))   if wins   else 0.0
    hold_l   = float(np.mean([t["hold_d"] for t in losses])) if losses else 0.0
    avg_peak = float(np.mean([t["max_unreal_%"] for t in trades])) if trades else 0.0
    avg_peak_w = float(np.mean([t["max_unreal_%"] for t in wins])) if wins else 0.0

    rot_pnl = float(np.mean([t["ret_%"] for t in rot_exits])) if rot_exits else 0.0
    rot_wins = sum(1 for t in rot_exits if t["ret_%"] > 0)

    earned_rate = len(earned_t) / len(trades) * 100 if trades else 0
    earned_ret  = float(np.mean([t["ret_%"] for t in earned_t])) if earned_t else 0.0
    fresh_ret   = float(np.mean([t["ret_%"] for t in trades
                                 if not t["earned_mode"]])) if trades else 0.0

    # Jährliche Rendite
    annual = {}
    for year, grp in eq.groupby(eq.index.year):
        yr  = (grp.iloc[-1] / grp.iloc[0] - 1) * 100
        annual[year] = round(yr, 1)

    # Max Drawdown Zeitraum
    dd_min_idx = dd.idxmin()
    # Peak vor dem trough
    peak_before_dd = eq[:dd_min_idx].idxmax()

    return {
        # Kernmetriken
        "ret":          round(ret,    2),
        "cagr":         round(cagr,   2),
        "maxdd":        round(dd.min(), 1),
        "maxdd_date":   dd_min_idx,
        "maxdd_peak":   peak_before_dd,
        "sharpe":       round(sharpe, 2),
        "n_trades":     len(trades),
        "n_atr":        len(atr_exits),
        "n_rot":        len(rot_exits),
        "n_end":        len(end_exits),
        "fees_total":   len(trades) * ORDER_FEE * 2,
        "invest_pct":   round(invest_days / len(equity) * 100, 1),
        # Hit/Payoff
        "hit":          round(hit,    1),
        "avg_win":      round(avg_w,  2),
        "avg_loss":     round(avg_l,  2),
        "payoff":       round(payoff, 2),
        "pf":           round(pf,     2),
        "ev":           round(ev,     2),
        # Lifecycle
        "hold_w":       round(hold_w,  1),
        "hold_l":       round(hold_l,  1),
        "avg_peak":     round(avg_peak, 1),
        "avg_peak_w":   round(avg_peak_w, 1),
        # Rotation-Analyse
        "rot_avg_pnl":  round(rot_pnl,  2),
        "rot_hit":      round(rot_wins / len(rot_exits) * 100, 1) if rot_exits else 0,
        # Earned-Mode
        "earned_rate":  round(earned_rate, 1),
        "earned_ret":   round(earned_ret,  2),
        "fresh_ret":    round(fresh_ret,   2),
        # Jahresrenditen
        "annual":       annual,
        "years":        round(years, 1),
        "end_cap":      round(eq.iloc[-1], 0),
        # Drawdown-Kurve für Plot
        "_dd_series":   dd,
        "_eq_series":   eq,
        "_peak_series": peak,
    }


def print_report(m: dict, trades: list[dict]) -> None:
    sep  = "=" * 72
    line = "─" * 72

    print(f"\n{sep}")
    print(f"  CHAMPION RUN  |  VCP v8.3  |  Breadth=0.0  |  Rotation=1.5×")
    print(sep)

    print(f"""
  ┌─ PORTFOLIO ÜBERSICHT {'─' * 47}
  │  Zeitraum:       {m['years']:.1f} Jahre
  │  Gesamtrendite:  {m['ret']:>+.2f}%   (CAGR {m['cagr']:>+.2f}%)
  │  End-Kapital:    {m['end_cap']:>10,.0f} €   (Start: {INITIAL_CAPITAL:,.0f} €)
  │  Sharpe Ratio:   {m['sharpe']:.2f}
  │  Max Drawdown:   {m['maxdd']:>+.1f}%
  │    └─ Trough:    {m['maxdd_date'].date()}  (Peak: {m['maxdd_peak'].date()})
  │  Investitionsq.: {m['invest_pct']:.1f}%  der Handelstage
  │  Gezahlte Geb.:  {m['fees_total']:,.0f} €
  └{'─' * 57}

  ┌─ TRADE-STATISTIK {'─' * 50}
  │  Gesamt-Trades:  {m['n_trades']}
  │    ATR-Exits:    {m['n_atr']}  ({m['n_atr']/m['n_trades']*100:.0f}%)
  │    Rotationen:   {m['n_rot']}  ({m['n_rot']/m['n_trades']*100:.0f}%)
  │    Offen/Ablauf: {m['n_end']}  ({m['n_end']/m['n_trades']*100:.0f}%)
  │
  │  Hit-Rate:       {m['hit']:.1f}%
  │  Avg Win:       +{m['avg_win']:.2f}%  |  Avg Loss: {m['avg_loss']:.2f}%
  │  Payoff Ratio:   {m['payoff']:.2f}    |  Profit Factor: {m['pf']:.2f}
  │  EV/Trade:      {m['ev']:>+.2f}%
  └{'─' * 57}

  ┌─ TRADE LIFECYCLE ANALYSE {'─' * 43}
  │  Ø Haltedauer Winners:  {m['hold_w']:.1f} Tage
  │  Ø Haltedauer Losers:   {m['hold_l']:.1f} Tage
  │
  │  Ø Peak-Unrealized:     {m['avg_peak']:>+.1f}%  (alle Trades)
  │  Ø Peak-Unrealized:     {m['avg_peak_w']:>+.1f}%  (nur Winners)
  │
  │  Rotation-Exit Analyse (n={m['n_rot']}):
  │    Ø PnL bei Rotation:  {m['rot_avg_pnl']:>+.2f}%
  │    Hit-Rate Rotations:  {m['rot_hit']:.1f}%  (% die im Plus waren)
  │
  │  Earned-Mode Analyse:
  │    Conversion Rate:     {m['earned_rate']:.1f}%  (Trades die "Earned" erreichten)
  │    Ø Return Earned:    {m['earned_ret']:>+.2f}%
  │    Ø Return Fresh:     {m['fresh_ret']:>+.2f}%
  └{'─' * 57}

  ┌─ JÄHRLICHE RENDITEN {'─' * 48}""")

    bar_max = max(abs(v) for v in m["annual"].values()) if m["annual"] else 1
    for yr, yret in sorted(m["annual"].items()):
        bar_len = int(abs(yret) / bar_max * 28)
        bar     = ("█" * bar_len) if yret >= 0 else ("▓" * bar_len)
        sign    = "+" if yret >= 0 else ""
        side    = "←" if yret < 0 else " →"
        print(f"  │  {yr}:  {sign}{yret:>6.1f}%  {side} {bar}")

    print(f"  └{'─' * 57}\n")


# ==============================================================================
# 5. VISUALISIERUNG
# ==============================================================================

def plot_champion(
    equity:  pd.Series,
    trades:  list[dict],
    m:       dict,
    out_png: Path,
) -> None:
    eq        = m["_eq_series"]
    dd        = m["_dd_series"]
    peak      = m["_peak_series"]
    annual    = m["annual"]

    # Farb-Palette
    C_EQ      = "#1a6fc4"
    C_PEAK    = "#4caf50"
    C_DD      = "#f44336"
    C_WIN     = "#2e7d32"
    C_LOSS    = "#c62828"
    C_ROT     = "#f57c00"
    C_END     = "#6a1b9a"
    C_BG      = "#f9f9f9"
    C_GRID    = "#e0e0e0"

    fig = plt.figure(figsize=(20, 16), dpi=150, facecolor=C_BG)
    fig.suptitle(
        "VCP v8.3  |  Champion Run  |  Breadth=0.0  |  Rotation=1.5×  |  Diamond Hands",
        fontsize=14, fontweight="bold", y=0.98, color="#212121"
    )
    gs = fig.add_gridspec(
        3, 1,
        height_ratios=[3, 2, 1.2],
        hspace=0.08,
        left=0.06, right=0.97,
        top=0.95, bottom=0.05,
    )

    # ── Panel 1: Equity + Drawdown-Zonen ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(C_BG)

    # Drawdown-Fläche (rot, hinter der Kurve)
    ax1.fill_between(eq.index, eq.values, peak.values,
                     where=(eq < peak),
                     color=C_DD, alpha=0.18, label="_dd_fill")

    # Peak-Linie (dünn, grün gepunktet)
    ax1.plot(peak.index, peak.values, color=C_PEAK,
             linewidth=0.7, linestyle="--", alpha=0.6, label="Laufender Peak")

    # Equity-Kurve
    ax1.plot(eq.index, eq.values, color=C_EQ,
             linewidth=1.8, label=f"Portfolio ({m['ret']:>+.1f}%  |  CAGR {m['cagr']:>+.1f}%)")

    # Startlinie
    ax1.axhline(INITIAL_CAPITAL, color="#9e9e9e", linewidth=0.8,
                linestyle=":", alpha=0.8, label=f"Startkapital ({INITIAL_CAPITAL:,.0f}€)")

    # Max Drawdown annotieren
    dd_val  = m["maxdd"]
    dd_date = m["maxdd_date"]
    dd_y    = float(eq.at[dd_date])
    ax1.annotate(
        f"Max DD\n{dd_val:.1f}%",
        xy=(dd_date, dd_y),
        xytext=(dd_date, dd_y * 0.88),
        arrowprops=dict(arrowstyle="->", color=C_DD, lw=1.2),
        fontsize=8, color=C_DD, fontweight="bold",
        ha="center",
    )

    # End-Kapital annotieren
    ax1.annotate(
        f"  {m['end_cap']:,.0f}€",
        xy=(eq.index[-1], float(eq.iloc[-1])),
        fontsize=9, color=C_EQ, fontweight="bold", va="center",
    )

    ax1.set_ylabel("Kapital (€)", fontsize=9)
    ax1.tick_params(axis="x", labelbottom=False)
    ax1.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}€"))
    ax1.grid(True, color=C_GRID, linewidth=0.5)
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.7)

    # Metriken als Text-Box (rechts oben)
    info_txt = (
        f"Hit-Rate:  {m['hit']:.1f}%\n"
        f"Payoff:    {m['payoff']:.2f}\n"
        f"EV/Trade: {m['ev']:>+.2f}%\n"
        f"Trades:    {m['n_trades']}\n"
        f"  ATR:     {m['n_atr']}\n"
        f"  Rot:     {m['n_rot']}\n"
        f"Sharpe:    {m['sharpe']:.2f}\n"
        f"Invest:    {m['invest_pct']:.0f}%"
    )
    ax1.text(
        0.985, 0.97, info_txt,
        transform=ax1.transAxes,
        fontsize=7.5, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="#cccccc", alpha=0.85),
        family="monospace",
    )

    # ── Panel 2: Gantt-Chart ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.set_facecolor(C_BG)
    ax2.set_ylim(-0.6, 2.6)
    ax2.set_yticks([0.5, 1.5])
    ax2.set_yticklabels(["Slot 2", "Slot 1"], fontsize=8)
    ax2.set_ylabel("Portfolio\nSlots", fontsize=9)

    for t in trades:
        slot  = t["slot"]
        y_bot = (slot - 1)            # Slot 1 → y=0, Slot 2 → y=1
        x0    = mdates.date2num(t["entry_date"])
        x1    = mdates.date2num(t["exit_date"])
        w     = max(x1 - x0, 0.5)

        reason = t["exit_reason"]
        ret    = t["ret_%"]
        if reason == "Rotation":
            color  = C_ROT
            ec     = "#e65100"
        elif reason == "End":
            color  = C_END
            ec     = "#4a148c"
        elif ret > 0:
            color  = C_WIN
            ec     = "#1b5e20"
        else:
            color  = C_LOSS
            ec     = "#b71c1c"

        rect = mpatches.FancyBboxPatch(
            (x0, y_bot + 0.08), w, 0.84,
            boxstyle="round,pad=0",
            facecolor=color, edgecolor=ec,
            linewidth=0.5, alpha=0.85,
        )
        ax2.add_patch(rect)

        # Label (nur bei breiteren Bars, sonst zu gedrängt)
        mid = (x0 + x1) / 2
        if w > 3:
            sign  = "+" if ret >= 0 else ""
            label = f"{t['ticker']} {sign}{ret:.0f}%"
            ax2.text(mid, y_bot + 0.5, label,
                     ha="center", va="center",
                     fontsize=5.5, color="white", fontweight="bold",
                     clip_on=True)

    ax2.set_xlim(
        mdates.date2num(eq.index[0]) - 5,
        mdates.date2num(eq.index[-1]) + 5,
    )
    ax2.grid(True, color=C_GRID, linewidth=0.4, axis="x")
    ax2.tick_params(axis="x", labelbottom=False)

    # Legende Gantt
    legend_patches = [
        mpatches.Patch(color=C_WIN,  label="Winner (ATR-Stop)"),
        mpatches.Patch(color=C_LOSS, label="Loser (ATR-Stop)"),
        mpatches.Patch(color=C_ROT,  label="Rotation Exit"),
        mpatches.Patch(color=C_END,  label="Offen bei Backtest-Ende"),
    ]
    ax2.legend(handles=legend_patches, loc="upper left",
               fontsize=7, framealpha=0.7, ncol=4)

    # ── Panel 3: Jahresrenditen ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor(C_BG)

    years_list = sorted(annual.keys())
    yrets      = [annual[y] for y in years_list]
    colors_y   = [C_WIN if r >= 0 else C_LOSS for r in yrets]
    bars = ax3.bar(years_list, yrets, color=colors_y,
                   edgecolor="#424242", linewidth=0.6, alpha=0.85, width=0.6)
    ax3.axhline(0, color="#424242", linewidth=0.8)
    ax3.set_ylabel("Jahres-\nrendite (%)", fontsize=9)
    ax3.set_xlabel("Jahr", fontsize=9)
    ax3.grid(True, color=C_GRID, linewidth=0.4, axis="y")

    for bar, val in zip(bars, yrets):
        sign = "+" if val >= 0 else ""
        va   = "bottom" if val >= 0 else "top"
        offset = 0.3 if val >= 0 else -0.3
        ax3.text(bar.get_x() + bar.get_width()/2, val + offset,
                 f"{sign}{val:.1f}%",
                 ha="center", va=va, fontsize=7.5,
                 color="#212121", fontweight="bold")

    ax3.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"  Chart gespeichert: {out_png}")


# ==============================================================================
# 6. MAIN
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Champion Run  |  VCP v8.3  |  Breadth=0.0  |  Rot=1.5×")
    parser.add_argument("--years", type=float, default=DEFAULT_YEARS)
    args = parser.parse_args()

    print("=" * 72)
    print("  VCP CHAMPION RUN v8.3  |  Diamond Hands  |  Deep Dive Analyse")
    print("=" * 72)
    print(f"""
  Parameter:
    INITIAL_CAPITAL   = {INITIAL_CAPITAL:,.0f} €
    ORDER_FEE         = {ORDER_FEE:.0f} €  (Round-Trip = {ORDER_FEE*2:.0f} €)
    MAX_POSITIONS     = {MAX_POSITIONS}
    BREADTH_THRESHOLD = {BREADTH_THRESHOLD:.1f}  (kein Filter)
    ROTATION_FACTOR   = {ROTATION_FACTOR:.1f}×  (+Diamond Hands)
    ATR Trail:        {ATR_INIT:.1f}× → {ATR_TRAIL:.1f}×  (Earned Mode)
    BB Squeeze:       < {BB_SQUEEZE_THRESH*100:.0f}%
    Zeitraum:         {args.years:.0f} Jahre
""")

    # 1. Daten laden
    print("[1/4] Lade Daten...")
    import time
    t0      = time.time()
    tickers = _load_tickers()
    data    = _load_ohlcv(tickers, args.years)
    print(f"  {len(data)} Ticker geladen in {time.time()-t0:.1f}s")

    # 2. Pivots
    print("\n[2/4] Pivots aufbauen...")
    t0     = time.time()
    pivots = build_panels(data)
    dates  = pivots["open"].index
    n_sig  = int(pivots["entry_sig"].fillna(False).values.sum())
    print(f"  Zeitraum:  {dates[0].date()} → {dates[-1].date()}")
    print(f"  Ticker:    {len(pivots['open'].columns)}")
    print(f"  VCP-Signale (gesamt):  {n_sig:,}")
    print(f"  Pivot-Aufbau:  {time.time()-t0:.1f}s")

    # 3. Backtest
    print("\n[3/4] Simulation läuft...")
    t0 = time.time()
    equity, trades, invest_days = run_backtest(pivots)
    print(f"  Fertig in {time.time()-t0:.1f}s  |  {len(trades)} Trades")

    # 4. Metriken + Report
    print("\n[4/4] Auswertung...")
    m = compute_metrics(equity, trades, invest_days)
    print_report(m, trades)

    # 5. CSV speichern
    df_trades = pd.DataFrame(trades)
    df_trades.to_csv(_OUT_CSV, index=False)
    print(f"  Trades gespeichert:  {_OUT_CSV}  ({len(trades)} Zeilen)")

    # 6. Chart
    print(f"  Erstelle Chart...")
    plot_champion(equity, trades, m, _OUT_PNG)

    # Top Winner / Loser
    df_trades_sorted = df_trades.sort_values("ret_%", ascending=False)
    print(f"\n  TOP 5 WINNER:")
    for _, row in df_trades_sorted.head(5).iterrows():
        earned_mark = " ★Earned" if row["earned_mode"] else ""
        print(f"    {row['ticker']:<6}  {row['entry_date'].date()} → "
              f"{row['exit_date'].date()}  ({int(row['hold_d'])}d)  "
              f"  +{row['ret_%']:.1f}%{earned_mark}")

    print(f"\n  TOP 5 LOSER:")
    for _, row in df_trades_sorted.tail(5).iterrows():
        print(f"    {row['ticker']:<6}  {row['entry_date'].date()} → "
              f"{row['exit_date'].date()}  ({int(row['hold_d'])}d)  "
              f"  {row['ret_%']:.1f}%")

    print(f"\n  FERTIG.\n")


if __name__ == "__main__":
    main()
