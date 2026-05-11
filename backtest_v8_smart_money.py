"""
backtest_v8_smart_money.py
====================================================================================
Portfolio-Backtester v8  |  "Smart Money VCP"  |  2-Slot Konzentration

KERNTHESE:
    v7 kaufte "Explosions-Ausbrüche" (Amplitude ≥5%) — das war systematisches
    Erschöpfungs-Gap-Chasing. Wir kauften den Jubel, nicht die Welle.

    v8 kauft "Stille vor dem Sturm":
    Das Volatility Contraction Pattern (VCP) identifiziert Aktien, die sich
    in einem engen Bollinger-Band-Squeeze befinden, bevor sie ausbrechen.
    Wenn dann Smart Money den Squeeze mit einem Volumen-Spike durchbricht
    → das ist eine qualitativ andere, stärkere Welle.

Entry-Signal (VCP-Breakout):
    ① Breakout_50:    Close > High_50d_prev             (Ausbruch heute)
    ② BB-Squeeze:     BB_width GESTERN < 10%            (Coiling Energy)
                      (BB_upper − BB_lower) / Close < 0.10  [t-1]
    ③ Volumen:        Volume > SMA_Vol_20 × 1.5          (Smart-Money-Bestätigung)
    ④ Trend:          Close > SMA_200                    (Langzeit-Aufwärtstrend)
    ⑤ Transition:     Nur der erste Tag des Ausbruchs    (Kein Rauschen)
    ⑥ Bärenmarkt-     SPY_Close > SPY_SMA_200 UND        (Gesamtmarkt-Filter)
       Filter:        SPY_Close > SPY_SMA_50

Wichtig: KEINE Amplitude-Regel! Eine enge BB-Squeeze + Volumen ist präziser.

Exit-Management:
    Nur Asymmetrischer ATR-Trail (kein Stall-Stop!):
    Phase 1 (Tight):  Stop = Entry − 2.0× ATR14    (schützt vor Fake-Breakout)
    Phase 2 (Earned): Stop = MaxHigh − 3.5× ATR14  (sobald Max_High > Entry + 2.0×ATR)
    Stop niemals senken!

    Predator-Rotation: neuer Kandidat > 1.5× stärker → rotiere
    Free-Ride Pyramidisierung: ≥+20% unrealized → aufstocken + Stop auf AvgEntry

Warum kein Stall-Stop:
    Der natürliche Pullback nach VCP-Ausbrüchen kann 3-10 Tage dauern.
    Einen 5-Tage-Stall-Stop einzubauen würde genau die qualitativ besten
    Setups zerstören, die kurz nach dem Breakout konsolidieren, bevor sie
    explodieren. Der ATR-Trail schützt das Kapital ausreichend.

Hardcoded:
    INITIAL_CAPITAL = 10.000 €
    ORDER_FEE       = 20 €  (Round-Trip = 40 €  ≈ 0.8% auf 5.000 €)
    MAX_POSITIONS   = 2

Output:
    - Metriken, Hit-Rate, Netto-EV/Trade, Jahresaufteilung
    - Plot 1: Equity-Kurve vs. SPY
    - Plot 2: Gantt-Chart (Cash-Phasen sichtbar)
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import warnings
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

_here = Path(__file__).parent
sys.path.insert(0, str(_here))

from backtest_v6 import _load_tickers, _atr

# ── Hardcoded Strategie-Parameter ───────────────────────────────────────────
INITIAL_CAPITAL   = 10_000.0
ORDER_FEE         = 20.0
MAX_POSITIONS     = 2
ATR_INIT          = 2.0        # Phase 1: Tight Stop
ATR_TRAIL         = 3.5        # Phase 2: Earned Trail
BB_PERIOD         = 20         # Bollinger Band Periode
BB_STD            = 2.0        # Bollinger Band Std-Dev
BB_SQUEEZE_THRESH = 0.10       # BB_width / Close < 10% = Squeeze
VOL_MULTIPLIER    = 1.5        # Volumen-Mindest-Ratio
ROTATION_FACTOR   = 1.5        # Kandidat muss X-mal stärker sein
PYRAMID_THRESHOLD = 0.20       # +20% unrealized → aufstocken
MAX_PYRAMIDS      = 1
DEFAULT_YEARS     = 7.0
_RAW_DIR          = _here / "data" / "raw"
MIN_SHARES        = 5          # Mindest-Stückzahl (verhindert Micro-Positionen)


# ==============================================================================
# 1. DATEN LADEN
# ==============================================================================

def _load_ohlcv(tickers: list[str], years: float) -> dict[str, pd.DataFrame]:
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=int(years * 365.25))
    tset   = set(tickers)
    data: dict[str, pd.DataFrame] = {}
    for fpath in sorted(_RAW_DIR.glob("*_1d.parquet")):
        ticker = fpath.stem.replace("_1d", "")
        if ticker not in tset and ticker != "SPY":
            continue
        try:
            df = pd.read_parquet(fpath)
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.columns = [c.lower() for c in df.columns]
            df = df[df.index >= cutoff]
            if len(df) < 260 and ticker != "SPY":
                continue
            keep = [c for c in ["open","high","low","close","volume"]
                    if c in df.columns]
            if not {"open","high","low","close"}.issubset(df.columns):
                continue
            data[ticker] = df[keep]
        except Exception:
            pass
    return data


# ==============================================================================
# 2. SPY BÄRENMARKT-FILTER
# ==============================================================================

def build_spy_filter(spy_df: pd.DataFrame) -> pd.Series:
    """True = Gesamtmarkt ist bull (SPY > SMA50 AND SPY > SMA200)."""
    c      = spy_df["close"]
    sma50  = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()
    return (c > sma50) & (c > sma200)


# ==============================================================================
# 3. INDIKATOREN + PIVOT-TABELLEN
# ==============================================================================

def build_panels(
    data: dict[str, pd.DataFrame],
    spy_filter: pd.Series | None,
) -> dict[str, pd.DataFrame]:
    """Baut alle Date×Ticker Pivot-Tabellen auf."""
    cols: dict[str, dict] = {
        k: {} for k in [
            "open", "close", "high", "atr14", "trend_str", "entry_sig",
        ]
    }

    for ticker, df in data.items():
        if ticker == "SPY":
            continue
        c   = df["close"]
        h   = df["high"]
        vol = df.get("volume")

        sma200     = c.rolling(200).mean()
        sma20_vol  = vol.rolling(20).mean() if vol is not None else None
        atr14      = _atr(df, 14)

        # Bollinger Bänder
        sma_bb     = c.rolling(BB_PERIOD).mean()
        std_bb     = c.rolling(BB_PERIOD).std()
        bb_upper   = sma_bb + BB_STD * std_bb
        bb_lower   = sma_bb - BB_STD * std_bb
        bb_width   = (bb_upper - bb_lower) / c.replace(0, np.nan)

        valid = sma200.notna() & atr14.notna()
        idx   = c[valid].index

        cols["open"][ticker]      = df["open"][valid]
        cols["close"][ticker]     = c[valid]
        cols["high"][ticker]      = h[valid]
        cols["atr14"][ticker]     = atr14[valid]
        cols["trend_str"][ticker] = ((c - sma200) / sma200)[valid]

        # ── VCP Entry-Signal ─────────────────────────────────────────────────
        # ① Breakout_50: nur der erste True-Tag (Transition)
        high50_prev = h.shift(1).rolling(50).max()
        b50_raw     = c > high50_prev
        trig_b50    = b50_raw & ~b50_raw.shift(1).fillna(False)

        # ② BB-Squeeze GESTERN (kein Look-Ahead): bb_width[t-1] < thresh
        squeeze_prev = bb_width.shift(1) < BB_SQUEEZE_THRESH

        # ③ Volumen-Spike HEUTE
        vol_ok = (vol > sma20_vol * VOL_MULTIPLIER
                  if (vol is not None and sma20_vol is not None)
                  else pd.Series(False, index=c.index))

        # ④ Trend-Filter
        trend_ok = c > sma200

        sig = (trig_b50 & squeeze_prev & vol_ok & trend_ok)

        # ⑥ SPY Bärenmarkt-Filter anwenden (falls vorhanden)
        if spy_filter is not None:
            spy_aligned = spy_filter.reindex(c.index, method="ffill").fillna(False)
            sig = sig & spy_aligned

        cols["entry_sig"][ticker] = sig.reindex(idx).fillna(False).astype(bool)

    return {k: pd.DataFrame(v) for k, v in cols.items()}


# ==============================================================================
# 4. STOP-MANAGEMENT
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


# ==============================================================================
# 5. SIMULATION
# ==============================================================================

def run_backtest(
    pivots:  dict[str, pd.DataFrame],
    verbose: bool = False,
) -> tuple[pd.Series, list[dict], list[dict], int]:

    piv     = pivots
    dates   = piv["open"].index
    tickers = list(piv["open"].columns)

    cash            = INITIAL_CAPITAL
    portfolio:      dict[str, dict] = {}
    free_slots      = {1, 2}
    ticker_to_slot: dict[str, int] = {}

    completed:  list[dict] = []
    gantt_segs: list[dict] = []
    equity_log: dict       = {}
    invest_days = 0

    def _safe(df: pd.DataFrame, date, ticker) -> float:
        try:
            v = df.at[date, ticker]
            return float(v) if pd.notna(v) else math.nan
        except Exception:
            return math.nan

    def _open_pos(ticker: str, buy_date, buy_px: float,
                  atr_e: float, slot: int, reason: str = "ENTRY") -> bool:
        nonlocal cash
        # Ziel: current_equity / MAX_POSITIONS (dynamisch mit Gewinn/Verlust)
        mkt_approx  = sum(p["shares"] * p["avg_entry_price"]
                          for p in portfolio.values())
        current_eq  = cash + mkt_approx
        target      = current_eq / MAX_POSITIONS
        target      = min(target, cash - ORDER_FEE)   # nicht mehr als Cash vorhanden
        shares      = int((target - ORDER_FEE) / buy_px) if buy_px > 0 else 0
        if shares < MIN_SHARES:
            return False
        # Mindest-Positionsgröße: Gebühren < 5% des Trades
        if shares * buy_px < ORDER_FEE * 20:
            return False
        cost = shares * buy_px + ORDER_FEE
        if cost > cash:
            return False
        cash      -= cost
        portfolio[ticker] = {
            "slot":          slot,
            "entry_date":    buy_date,
            "entry_price":   buy_px,
            "shares":        shares,
            "cost":          cost,
            "atr_at_entry":  atr_e,
            "trailing_stop": buy_px - ATR_INIT * atr_e,
            "max_high":      buy_px,
            "earned_mode":   False,
            "pyramid_count": 0,
            "avg_entry_price": buy_px,
            "days_held":     0,
        }
        ticker_to_slot[ticker] = slot
        free_slots.discard(slot)
        if verbose:
            ts = _safe(piv["trend_str"], buy_date, ticker)
            print(f"  [{buy_date.date()}] {reason:<15} {ticker:<6} "
                  f"{shares}×@{buy_px:.2f}  "
                  f"Stop={buy_px - ATR_INIT * atr_e:.2f}  "
                  f"Trend={ts:.3f}  Slot={slot}")
        return True

    def _close_pos(ticker: str, sell_date, sell_px: float, reason: str) -> None:
        nonlocal cash
        pos      = portfolio[ticker]
        slot     = pos["slot"]
        proceeds = pos["shares"] * sell_px - ORDER_FEE
        pnl_net  = proceeds - pos["cost"]
        ret_pct  = pnl_net / pos["cost"] * 100
        cash    += proceeds

        gantt_segs.append({
            "slot":        slot,
            "ticker":      ticker,
            "start":       pos["entry_date"],
            "end":         sell_date,
            "ret_pct":     ret_pct,
            "is_rotation": reason == "ROTATION_OUT",
            "reason":      reason,
        })
        completed.append({
            "Ticker":    ticker,
            "Kauf":      pos["entry_date"].date(),
            "KaufPreis": round(pos["entry_price"], 2),
            "Verkauf":   sell_date.date(),
            "VKPreis":   round(sell_px, 2),
            "Shares":    pos["shares"],
            "PnL_€":     round(pnl_net, 2),
            "Return_%":  round(ret_pct, 2),
            "Hold_d":    pos["days_held"],
            "Earned":    pos["earned_mode"],
            "Pyr":       pos["pyramid_count"],
            "Exit":      reason,
        })
        free_slots.add(slot)
        del portfolio[ticker]
        ticker_to_slot.pop(ticker, None)
        if verbose:
            sign = "+" if pnl_net >= 0 else ""
            print(f"  [{sell_date.date()}] {reason:<15} {ticker:<6} "
                  f"@{sell_px:.2f}  {sign}{pnl_net:,.0f}€ ({sign}{ret_pct:.1f}%)")

    # ── Hauptschleife ────────────────────────────────────────────────────────
    for day_i in range(len(dates) - 1):
        today    = dates[day_i]
        tomorrow = dates[day_i + 1]

        # ── A. Positionen updaten, Exit prüfen ──────────────────────────────
        exits: list[tuple[str, str]] = []
        for ticker, pos in list(portfolio.items()):
            tc  = _safe(piv["close"], today, ticker)
            th  = _safe(piv["high"],  today, ticker)
            ta  = _safe(piv["atr14"], today, ticker)
            if math.isnan(tc):
                continue
            pos["days_held"] += 1
            _update_stop(pos, th, ta)
            # Nur ATR-Stop — kein Stall-Stop
            if tc < pos["trailing_stop"]:
                exits.append((ticker, "ATR_STOP"))

        for ticker, reason in exits:
            sell_px = _safe(piv["open"], tomorrow, ticker)
            if not math.isnan(sell_px):
                _close_pos(ticker, tomorrow, sell_px, reason)

        # ── B. Kandidaten für morgen finden ─────────────────────────────────
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
                    _open_pos(cand, tomorrow, buy_px, atr_e, slot)

                elif len(portfolio) >= MAX_POSITIONS:
                    # Predator-Rotation
                    weakest_t   = min(portfolio.keys(),
                        key=lambda t: _safe(piv["trend_str"], today, t))
                    weakest_str = _safe(piv["trend_str"], today, weakest_t)
                    cand_str    = _safe(piv["trend_str"], today, cand)

                    if (not math.isnan(cand_str) and not math.isnan(weakest_str)
                            and cand_str > ROTATION_FACTOR * weakest_str):
                        rot_sell = _safe(piv["open"], tomorrow, weakest_t)
                        if not math.isnan(rot_sell):
                            if verbose:
                                print(f"  [{tomorrow.date()}] ROTATION        "
                                      f"{weakest_t}→{cand}  "
                                      f"Trendstärke {weakest_str:.3f}→{cand_str:.3f} "
                                      f"({cand_str/weakest_str:.2f}×)")
                            freed = portfolio[weakest_t]["slot"]
                            _close_pos(weakest_t, tomorrow, rot_sell, "ROTATION_OUT")
                            _open_pos(cand, tomorrow, buy_px, atr_e,
                                      freed, "ROTATION_IN")
                            break

        # ── C. Pyramidisieren ────────────────────────────────────────────────
        if len(portfolio) == 1 and free_slots and not candidates:
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
                    if add_sh < MIN_SHARES:
                        continue
                    cost_add = add_sh * buy_px + ORDER_FEE
                    if cost_add > cash:
                        continue
                    old_v    = pos["shares"] * pos["avg_entry_price"]
                    new_avg  = (old_v + add_sh * buy_px) / (pos["shares"] + add_sh)
                    cash    -= cost_add
                    pos["cost"]          += cost_add
                    pos["shares"]        += add_sh
                    pos["avg_entry_price"] = new_avg
                    pos["pyramid_count"]   += 1
                    old_stop = pos["trailing_stop"]
                    pos["trailing_stop"] = max(old_stop, new_avg)
                    if verbose:
                        print(f"  [{tomorrow.date()}] PYRAMIDE        {ticker}  "
                              f"+{add_sh}×@{buy_px:.2f}  AvgEntry={new_avg:.2f}  "
                              f"Free-Ride-Stop={pos['trailing_stop']:.2f}")

        # ── D. Equity tracken ─────────────────────────────────────────────────
        equity = cash
        n_held = 0
        for ticker, pos in portfolio.items():
            cp     = _safe(piv["close"], today, ticker)
            equity += pos["shares"] * (cp if not math.isnan(cp) else pos["avg_entry_price"])
            n_held += 1
        equity_log[today] = equity
        if n_held > 0:
            invest_days += 1

    # Offene Positionen schliessen
    for ticker in list(portfolio.keys()):
        lp = _safe(piv["close"], dates[-1], ticker)
        if not math.isnan(lp):
            _close_pos(ticker, dates[-1], lp, "END_OF_BACKTEST")

    return pd.Series(equity_log).sort_index(), completed, gantt_segs, invest_days


# ==============================================================================
# 6. METRIKEN
# ==============================================================================

def compute_metrics(
    equity: pd.Series,
    trades: list[dict],
    invest_days: int,
) -> dict:
    if equity.empty or not trades:
        return {}

    total_days = len(equity)
    years      = max((equity.index[-1] - equity.index[0]).days / 365.25, 0.1)
    eq         = equity.ffill().bfill()

    total_ret = (eq.iloc[-1] / INITIAL_CAPITAL - 1) * 100
    cagr      = ((eq.iloc[-1] / INITIAL_CAPITAL) ** (1 / years) - 1) * 100
    peak      = eq.cummax()
    dd        = (eq - peak) / peak * 100
    max_dd    = dd.min()
    dr        = eq.pct_change().dropna()
    sharpe    = dr.mean() / dr.std() * (252 ** 0.5) if dr.std() > 0 else 0

    rets  = [t["Return_%"] for t in trades]
    wins  = [r for r in rets if r > 0]
    losses= [r for r in rets if r < 0]

    pf       = sum(wins) / abs(sum(losses)) if losses else float("inf")
    avg_win  = float(np.mean(wins))  if wins   else 0.0
    avg_loss = float(np.mean(losses))if losses else 0.0
    hit      = len(wins) / len(rets) * 100

    # Netto-EV/Trade: (hit% × avg_win + (1-hit%) × avg_loss)
    ev_per_trade = hit/100 * avg_win + (1 - hit/100) * avg_loss

    return {
        "total_ret_%":    round(total_ret, 2),
        "cagr_%":         round(cagr, 2),
        "max_dd_%":       round(max_dd, 2),
        "sharpe":         round(sharpe, 2),
        "n_trades":       len(trades),
        "hit_%":          round(hit, 1),
        "payoff":         round(abs(avg_win / avg_loss), 2) if avg_loss else 0,
        "profit_factor":  round(pf, 2),
        "ev_per_trade_%": round(ev_per_trade, 2),
        "avg_win_%":      round(avg_win, 2),
        "avg_loss_%":     round(avg_loss, 2),
        "avg_hold_d":     round(np.mean([t["Hold_d"] for t in trades]), 1),
        "max_win_%":      round(max(rets), 2),
        "total_fees":     len(trades) * ORDER_FEE * 2,
        "end_equity":     round(eq.iloc[-1], 2),
        "invest_pct":     round(invest_days / total_days * 100, 1),
        "n_rotations":    sum(1 for t in trades if t["Exit"] == "ROTATION_OUT"),
        "n_atr_exits":    sum(1 for t in trades if t["Exit"] == "ATR_STOP"),
        "n_pyramids":     sum(t["Pyr"] for t in trades),
    }


def print_summary(m: dict, trades: list[dict], years: float) -> None:
    fee_pct = ORDER_FEE * 2 / (INITIAL_CAPITAL / MAX_POSITIONS) * 100

    # EV-Bewertung
    ev = m['ev_per_trade_%']
    ev_verdict = ("✓ POSITIV (Strategie überlebt Gebühren)"  if ev > 0
                  else "✗ NEGATIV (Gebühren fressen Alpha)")

    print(f"""
{'=' * 72}
  BACKTEST v8 SMART MONEY VCP  |  2-Slot  |  {years:.0f} Jahre
{'=' * 72}

  Signal:   VCP-Breakout (Squeeze + Breakout_50 + Vol×{VOL_MULTIPLIER} + SMA200)
  Filter:   SPY > SMA50 & SMA200  |  BB_width[t-1] < {BB_SQUEEZE_THRESH*100:.0f}%
  Exit:     ATR-Trail {ATR_INIT}×→{ATR_TRAIL}×  |  Kein Stall-Stop
  Gebühren: {ORDER_FEE:.0f}€/Order → {ORDER_FEE*2:.0f}€/RT ({fee_pct:.1f}% auf {INITIAL_CAPITAL/MAX_POSITIONS:,.0f}€)

  ── PERFORMANCE ──────────────────────────────────────────────────────
  Gesamtrendite:        {m['total_ret_%']:>+8.2f}%
  CAGR:                 {m['cagr_%']:>+8.2f}%
  Max Drawdown:         {m['max_dd_%']:>+8.1f}%
  Sharpe Ratio:         {m['sharpe']:>8.2f}
  Endkapital:           {m['end_equity']:>10,.0f} €

  ── SIGNAL-QUALITÄT ──────────────────────────────────────────────────
  Trades gesamt:        {m['n_trades']:>5}
  Hit-Rate:             {m['hit_%']:>5.1f}%   (Ziel: >35%)
  Payoff Ratio:         {m['payoff']:>5.2f}   (Avg Win / Avg Loss)
  Profit Factor:        {m['profit_factor']:>5.2f}   (Brutto Win/Loss)
  Avg Gewinn:          {m['avg_win_%']:>+6.2f}%
  Avg Verlust:         {m['avg_loss_%']:>+6.2f}%

  ── NETTO-ERWARTUNGSWERT ─────────────────────────────────────────────
  EV/Trade (netto):    {m['ev_per_trade_%']:>+6.2f}%   → {ev_verdict}
  Avg Haltedauer:       {m['avg_hold_d']:>5.1f} Tage
  Max Einzelgewinn:    {m['max_win_%']:>+6.1f}%

  ── PORTFOLIO ────────────────────────────────────────────────────────
  Investitionsquote:    {m['invest_pct']:>5.1f}%  (Tage mit ≥1 Position)
  Gezahlte Gebühren:    {m['total_fees']:>8,.0f} €
  Rotationen:           {m['n_rotations']:>5}
  ATR-Stop-Exits:       {m['n_atr_exits']:>5}
  Pyramidisierungen:    {m['n_pyramids']:>5}
{'=' * 72}
""")

    # Jahresaufteilung
    if trades:
        df_tr = pd.DataFrame(trades)
        df_tr["Jahr"] = pd.to_datetime(df_tr["Kauf"]).dt.year
        yearly = df_tr.groupby("Jahr").agg(
            Trades     = ("Return_%", "count"),
            Hit_Rate   = ("Return_%", lambda x: (x > 0).mean() * 100),
            Avg_Return = ("Return_%", "mean"),
            Netto_PnL  = ("PnL_€",   "sum"),
        ).round(1)
        print("  JAHRESAUFTEILUNG:")
        print(f"  {'─' * 60}")
        for yr, row in yearly.iterrows():
            bar = "█" * max(0, int(row["Netto_PnL"] / 50))
            neg = "▓" * max(0, int(-row["Netto_PnL"] / 50))
            print(f"  {yr}:  {int(row['Trades']):>3} Trades  "
                  f"Hit: {row['Hit_Rate']:>4.1f}%  "
                  f"AvgRet: {row['Avg_Return']:>+5.1f}%  "
                  f"PnL: {row['Netto_PnL']:>+7,.0f}€  "
                  f"{'█' if row['Netto_PnL'] >= 0 else '▓'}"
                  f"{bar if row['Netto_PnL'] >= 0 else neg}")
        print()

    if trades:
        df_t = pd.DataFrame(trades)
        pd.set_option("display.width", 220)
        print("  ALLE TRADES:")
        print(f"  {'─' * 110}")
        print(df_t.to_string(index=False))
        print(f"  {'─' * 110}")


# ==============================================================================
# 7. VISUALISIERUNGEN
# ==============================================================================

def _load_spy_series(years: float) -> pd.Series | None:
    fpath = _RAW_DIR / "SPY_1d.parquet"
    if not fpath.exists():
        return None
    try:
        df = pd.read_parquet(fpath)
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.columns = [c.lower() for c in df.columns]
        cutoff = pd.Timestamp.today() - pd.Timedelta(days=int(years * 365.25))
        return df.loc[df.index >= cutoff, "close"]
    except Exception:
        return None


_DARK_BG  = "#0D1117"
_DARK_AX  = "#111827"
_GRID_COL = "#1F2937"

def _ax_dark(ax) -> None:
    ax.set_facecolor(_DARK_AX)
    for s in ax.spines.values():
        s.set_color("#374151")
    ax.tick_params(colors="#9CA3AF")
    ax.yaxis.label.set_color("#9CA3AF")
    ax.xaxis.label.set_color("#9CA3AF")
    ax.grid(True, color=_GRID_COL, linewidth=0.5)


def plot_equity(equity: pd.Series, years: float, out_path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 9),
        gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    fig.patch.set_facecolor(_DARK_BG)
    fig.suptitle(
        f"Backtest v8 Smart Money VCP  |  "
        f"VCP-Breakout + SPY-Filter  |  {years:.0f} Jahre",
        color="white", fontsize=13, fontweight="bold"
    )
    _ax_dark(ax1); _ax_dark(ax2)

    # Portfolio
    eq_norm = equity / equity.iloc[0] * INITIAL_CAPITAL
    ax1.plot(eq_norm.index, eq_norm.values,
             color="#22D3EE", linewidth=2, label="VCP v8", zorder=3)
    ax1.fill_between(eq_norm.index, INITIAL_CAPITAL, eq_norm.values,
                     where=(eq_norm.values >= INITIAL_CAPITAL),
                     color="#22D3EE", alpha=0.07)
    ax1.axhline(INITIAL_CAPITAL, color="#6B7280", linewidth=0.8, linestyle="--")

    # SPY Benchmark
    spy = _load_spy_series(years)
    if spy is not None:
        spy_a = spy.reindex(equity.index, method="ffill").dropna()
        if len(spy_a) > 0:
            spy_n = spy_a / spy_a.iloc[0] * INITIAL_CAPITAL
            ax1.plot(spy_n.index, spy_n.values,
                     color="#F97316", linewidth=1.3, alpha=0.75, label="SPY")

    ax1.set_ylabel("Kapital (€)", fontsize=10)
    ax1.legend(facecolor="#1F2937", edgecolor="#374151",
               labelcolor="white", fontsize=9)
    ax1.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}€"))

    # Drawdown
    peak = eq_norm.cummax()
    dd   = (eq_norm - peak) / peak * 100
    ax2.fill_between(dd.index, dd.values, 0, color="#EF4444", alpha=0.65)
    ax2.set_ylabel("Drawdown %", fontsize=9)
    ax2.set_ylim(min(dd.min() * 1.2, -5), 2)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


def plot_gantt(gantt_segs: list[dict], equity: pd.Series, out_path: str) -> None:
    if not gantt_segs:
        return

    fig, ax = plt.subplots(figsize=(17, 5))
    fig.patch.set_facecolor(_DARK_BG)
    ax.set_facecolor("#0F172A")
    for s in ax.spines.values():
        s.set_color("#374151")
    ax.tick_params(colors="#CCC")

    all_tickers  = sorted({s["ticker"] for s in gantt_segs})
    palette      = plt.cm.get_cmap("tab20", max(len(all_tickers), 1))
    ticker_color = {t: palette(i) for i, t in enumerate(all_tickers)}

    y_pos  = {1: 1.0, 2: 0.0}
    bar_h  = 0.72
    tspan  = (equity.index[-1] - equity.index[0]).days

    for seg in gantt_segs:
        slot   = seg["slot"]
        y      = y_pos[slot]
        start  = seg["start"]
        end    = seg["end"]
        ticker = seg["ticker"]
        ret    = seg["ret_pct"]
        width  = max((end - start).days, 1)
        color  = ticker_color[ticker]

        # Balken
        ax.barh(y, width, left=mdates.date2num(start),
                height=bar_h, color=color, alpha=0.88,
                edgecolor="#1F2937", linewidth=0.6)

        # Label
        if width > tspan * 0.012:
            mid = start + (end - start) / 2
            sign = "+" if ret >= 0 else ""
            ax.text(mdates.date2num(mid), y,
                    f"{ticker}\n{sign}{ret:.1f}%",
                    ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold")

        # Rotation marker
        if seg.get("is_rotation"):
            ax.plot(mdates.date2num(end), y, "D",
                    color="#F87171", markersize=9, zorder=5)

    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(["  Slot 2", "  Slot 1"], fontsize=11, color="white")
    ax.set_xlim(mdates.date2num(equity.index[0]), mdates.date2num(equity.index[-1]))
    ax.set_ylim(-0.65, 1.95)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.xticks(rotation=28, ha="right", fontsize=8, color="#CCC")

    legend_patches = [
        mpatches.Patch(color=ticker_color[t], label=t) for t in all_tickers
    ]
    legend_patches.append(plt.Line2D(
        [], [], marker="D", color="#F87171", linestyle="",
        markersize=8, label="Rotation-Exit"
    ))
    ax.legend(handles=legend_patches, loc="upper left",
              facecolor="#1E293B", edgecolor="#374151",
              labelcolor="white", fontsize=7.5,
              ncol=min(len(legend_patches), 8))

    n_rot = sum(1 for s in gantt_segs if s.get("is_rotation"))
    ax.set_title(
        f"Predator VCP v8  |  Slot-Zeitstrahl  |  "
        f"{len(gantt_segs)} Positionen  |  {n_rot} Rotationen  |  "
        f"Grau = Cash (auf der Seitenlinie)",
        fontsize=11, color="white", fontweight="bold", pad=12
    )
    ax.grid(True, axis="x", color="#1F2937", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


# ==============================================================================
# 8. MAIN
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest v8 Smart Money VCP")
    parser.add_argument("--years",   type=float, default=DEFAULT_YEARS)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-spy-filter", action="store_true",
                        help="SPY-Bärenmarkt-Filter deaktivieren")
    args = parser.parse_args()

    print("=" * 72)
    print("  BACKTEST v8 SMART MONEY VCP  |  Stille vor dem Sturm")
    print("=" * 72)
    print(f"""
  Signal:   Breakout_50 + BB-Squeeze[t-1]<{BB_SQUEEZE_THRESH*100:.0f}% + Vol×{VOL_MULTIPLIER} + SMA200
  Filter:   {'SPY > SMA50 & SMA200 (AKTIV)' if not args.no_spy_filter else 'SPY-Filter DEAKTIVIERT'}
  Exit:     ATR {ATR_INIT}× Tight → {ATR_TRAIL}× Earned  |  KEIN Stall-Stop
  Kapital:  {INITIAL_CAPITAL:,.0f}€  |  {MAX_POSITIONS} Slots  |  ~{INITIAL_CAPITAL/MAX_POSITIONS:,.0f}€/Trade
""")

    # 1. Daten laden
    t0      = time.time()
    tickers = _load_tickers()
    print(f"[1/4] Lade {len(tickers)} Ticker + SPY ({args.years:.0f} Jahre)...")
    data    = _load_ohlcv(tickers, args.years)
    has_vol = any("volume" in df.columns for df in data.values()
                  if not isinstance(df, type(None)))
    print(f"  {len(data)} Ticker in {time.time()-t0:.1f}s  "
          f"| Volume: {'✓' if has_vol else '✗ FEHLT'}"
          f"| SPY: {'✓' if 'SPY' in data else '✗ nicht gefunden'}")

    # 2. SPY-Filter
    spy_filter: pd.Series | None = None
    if not args.no_spy_filter:
        if "SPY" in data:
            spy_filter = build_spy_filter(data["SPY"])
            spy_bull   = spy_filter.sum()
            spy_total  = len(spy_filter)
            print(f"  SPY-Filter: {spy_bull}/{spy_total} Tage Bullenmarkt "
                  f"({spy_bull/spy_total*100:.0f}%)")
        else:
            print("  [WARN] SPY nicht gefunden – Filter deaktiviert")

    # 3. Pivots aufbauen
    print(f"\n[2/4] Indikator-Pivots aufbauen (VCP-Signal)...")
    t0     = time.time()
    pivots = build_panels(data, spy_filter)
    dates  = pivots["open"].index
    n_sig  = int(pivots["entry_sig"].fillna(False).values.sum())
    print(f"  Zeitraum: {dates[0].date()} → {dates[-1].date()}  "
          f"({len(dates):,} Tage, {len(pivots['open'].columns)} Ticker)")
    print(f"  VCP-Entry-Signale gesamt: {n_sig:,}  "
          f"({n_sig/len(pivots['open'].columns)/(args.years):.2f}/Ticker/Jahr)")

    if n_sig < 10:
        print("  [WARN] Sehr wenige Signale – prüfe BB-Squeeze-Threshold oder SPY-Filter.")

    # 4. Simulation
    print(f"\n[3/4] Simulation läuft...")
    t0 = time.time()
    equity, trades, gantt_segs, invest_days = run_backtest(pivots, args.verbose)
    print(f"  Fertig in {time.time()-t0:.1f}s")

    if not trades:
        print("\n  Keine Trades ausgeführt – Signal zu restriktiv?")
        print("  Tipp: --no-spy-filter oder BB-Squeeze-Threshold erhöhen")
        return

    # 5. Metriken & Ausgabe
    m = compute_metrics(equity, trades, invest_days)
    print_summary(m, trades, args.years)

    # 6. Plots
    print(f"[4/4] Visualisierungen speichern...")
    eq_path    = str(_here / "backtest_v8_equity.png")
    gantt_path = str(_here / "backtest_v8_gantt.png")
    plot_equity(equity, args.years, eq_path)
    plot_gantt(gantt_segs, equity, gantt_path)
    print(f"  Equity-Kurve: {eq_path}")
    print(f"  Gantt-Chart:  {gantt_path}\n")


if __name__ == "__main__":
    main()
