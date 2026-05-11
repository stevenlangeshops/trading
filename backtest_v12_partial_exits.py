"""
backtest_v12_partial_exits.py
====================================================================================
Partial Exits (Fractional Trading) v12.0

Architektonische Lösung für das Frankenstein-Problem:
  Statt Vollausstieg bei Überhitzung → Teilverkauf 50% + Runner

  Setup A  ─  Baseline (VCP v8.3)
    Reiner ATR-Stop, keine Teilverkäufe

  Setup B  ─  Partial Exit Machine
    Gleicher VCP-Einstieg. Drei Exit-Layer:

    Layer 1 – Smart Exhaustion Take-Profit:
      WENN: Earned Mode  AND  RSI>72  AND  dist_SMA50>7%  AND  ΔRSI<0
            AND has_partial_exit==False
      → TEILVERKAUF: Sofort 50% der Shares via nächstes Open.
        Verbuche PnL (Partial-Trade). Setze has_partial_exit=True.

    Layer 2 – Runner (die restlichen 50%):
      Exklusiv über 3.5× ATR-Stop (bereits Earned Mode).
      Exhaustion-Check deaktiviert.

    Layer 3 – Standard:
      2.0× ATR-Stop für nicht-earned Positionen (Frischkäufe).
      Diamond Hands Rotation (nur nicht-earned).

Trade-Aufzeichnung:
  exit_type = "Full"       Standard-Vollausstieg
  exit_type = "Partial_50" Teilverkauf (50%) durch Layer 1
  exit_type = "Runner"     Restlicher Verkauf nach Partial_50

Neu in Metriken:
  n_partial, n_runner, avg_partial_ret, avg_runner_ret,
  Kombinierte Rendite pro Original-Position (partial+runner Paare)

Portfolio (unveränderlich):
  INITIAL_CAPITAL = 10.000€  |  ORDER_FEE = 20€  |  MAX_POSITIONS = 2
  ROTATION_FACTOR = 1.5×     |  Diamond Hands (Earned nie rotierbar)

Verwendung:
  python backtest_v12_partial_exits.py
  python backtest_v12_partial_exits.py --years 7
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import matplotlib.ticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_here = Path(__file__).parent
sys.path.insert(0, str(_here))

from backtest_v6 import _load_tickers, _atr

# ── Konstanten ────────────────────────────────────────────────────────────────
INITIAL_CAPITAL   = 10_000.0
ORDER_FEE         = 20.0
MAX_POSITIONS     = 2
ATR_INIT          = 2.0
ATR_TRAIL         = 3.5
ROTATION_FACTOR   = 1.5
MIN_SHARES        = 5
PYRAMID_THRESHOLD = 0.20
MAX_PYRAMIDS      = 1
DEFAULT_YEARS     = 7.0

BB_PERIOD  = 20
BB_STD     = 2.0
BB_SQUEEZE = 0.10
VOL_MULT   = 1.5

EXHS_RSI      = 72.0
EXHS_DIST50   = 0.07
PARTIAL_FRAC  = 0.50     # Anteil des Teilverkaufs

_RAW_DIR = _here / "data" / "raw"
_OUT_PNG  = _here / "partial_exits_comparison.png"
_OUT_CSV  = _here / "partial_exits_trades.csv"


# ==============================================================================
# 1. DATEN
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
            keep = [c for c in ["open", "high", "low", "close", "volume"]
                    if c in df.columns]
            data[ticker] = df[keep].copy()
        except Exception:
            pass
    return data


# ==============================================================================
# 2. INDIKATOREN
# ==============================================================================

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=period - 1, adjust=False).mean()
    avg_l = loss.ewm(com=period - 1, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


# ==============================================================================
# 3. PANELS
# ==============================================================================

def build_panels(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    cols: dict[str, dict] = {
        k: {} for k in [
            "open", "close", "high", "low",
            "atr14", "trend_str", "entry_sig",
            "rsi14", "rsi14_prev", "dist_sma50",
        ]
    }
    n_sig_total = 0
    for ticker, df in data.items():
        c   = df["close"]
        lo  = df["low"]
        h   = df["high"]
        vol = df.get("volume")

        sma50  = c.rolling(50).mean()
        sma200 = c.rolling(200).mean()
        sma20  = c.rolling(BB_PERIOD).mean()
        std20  = c.rolling(BB_PERIOD).std()
        bb_w   = (sma20 + BB_STD * std20
                  - (sma20 - BB_STD * std20)) / c.replace(0, np.nan)
        atr14  = _atr(df, 14)
        rsi14  = _rsi(c, 14)
        sma20v = vol.rolling(20).mean() if vol is not None else None
        dist50 = (c - sma50) / sma50.replace(0, np.nan)

        high50_prev = h.shift(1).rolling(50).max()
        b50_raw     = c > high50_prev
        trig_b50    = b50_raw & ~b50_raw.shift(1).fillna(False)
        squeeze     = bb_w.shift(1) < BB_SQUEEZE
        vol_ok      = (vol > sma20v * VOL_MULT
                       if sma20v is not None
                       else pd.Series(False, index=c.index))
        trend_ok    = c > sma200
        sig         = trig_b50 & squeeze & vol_ok & trend_ok

        valid = sma200.notna() & atr14.notna() & rsi14.notna()
        idx   = c[valid].index
        n_sig_total += int(sig[valid].fillna(False).sum())

        cols["open"][ticker]       = df["open"][valid]
        cols["close"][ticker]      = c[valid]
        cols["high"][ticker]       = h[valid]
        cols["low"][ticker]        = lo[valid]
        cols["atr14"][ticker]      = atr14[valid]
        cols["trend_str"][ticker]  = ((c - sma200) / sma200)[valid]
        cols["entry_sig"][ticker]  = sig.reindex(idx).fillna(False).astype(bool)
        cols["rsi14"][ticker]      = rsi14[valid]
        cols["rsi14_prev"][ticker] = rsi14.shift(1)[valid]
        cols["dist_sma50"][ticker] = dist50[valid]

    panels = {k: pd.DataFrame(v) for k, v in cols.items()}
    panels["_n_sig"] = n_sig_total
    return panels


# ==============================================================================
# 4. SIMULATIONS-ENGINE
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
    pivots:       dict[str, pd.DataFrame],
    partial_exit: bool = False,
) -> tuple[pd.Series, list[dict], int]:
    """
    partial_exit=False → Setup A: reiner ATR-Stop
    partial_exit=True  → Setup B: Layer-1 Partial (50%) + Layer-2 Runner ATR
    """
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
            "slot":             slot,
            "entry_date":       buy_date,
            "entry_price":      buy_px,
            "shares":           shares,
            "initial_shares":   shares,        # Festgehalten für Runner-Berechnung
            "cost":             cost,
            "cost_per_share":   cost / shares,  # Konstant für prop. Kostenberechnung
            "atr_at_entry":     atr_e,
            "trailing_stop":    buy_px - ATR_INIT * atr_e,
            "max_high":         buy_px,
            "earned_mode":      False,
            "earned_date":      None,
            "pyramid_count":    0,
            "avg_entry_price":  buy_px,
            "days_held":        0,
            "max_unreal_pct":   0.0,
            # Partial-Exit-Tracking
            "has_partial_exit":    False,
            "partial_exit_date":   None,
            "partial_exit_px":     None,
            "partial_exit_ret":    None,
        }
        ticker_to_slot[ticker] = slot
        free_slots.discard(slot)
        return True

    def _close_partial(ticker: str, sell_date, sell_px: float) -> None:
        """Layer 1: Verkauft PARTIAL_FRAC der Shares. Position bleibt offen (Runner)."""
        nonlocal cash
        pos           = portfolio[ticker]
        sold_shares   = int(pos["shares"] * PARTIAL_FRAC)
        if sold_shares < 1:
            return
        cost_sold     = pos["cost_per_share"] * sold_shares
        proceeds      = sold_shares * sell_px - ORDER_FEE
        pnl           = proceeds - cost_sold
        ret_pct       = pnl / cost_sold * 100
        cash         += proceeds

        # Position aktualisieren (Runner)
        pos["shares"]           -= sold_shares
        pos["cost"]             -= cost_sold
        pos["has_partial_exit"]  = True
        pos["partial_exit_date"] = sell_date
        pos["partial_exit_px"]   = sell_px
        pos["partial_exit_ret"]  = round(ret_pct, 2)

        completed.append({
            "ticker":         ticker,
            "slot":           pos["slot"],
            "entry_date":     pos["entry_date"],
            "exit_date":      sell_date,
            "entry_price":    round(pos["entry_price"], 2),
            "exit_price":     round(sell_px, 2),
            "shares":         sold_shares,
            "pnl_€":          round(pnl, 2),
            "ret_%":          round(ret_pct, 2),
            "hold_d":         pos["days_held"],
            "earned_mode":    pos["earned_mode"],
            "earned_date":    pos["earned_date"],
            "pyramid_count":  pos["pyramid_count"],
            "max_unreal_%":   round(pos["max_unreal_pct"], 2),
            "exit_reason":    "Partial_50",
            "exit_type":      "Partial_50",
            "is_rotation":    False,
        })

    def _close_full(ticker: str, sell_date, sell_px: float,
                    exit_reason: str) -> None:
        """Full exit: verkauft alle verbleibenden Shares (könnte Runner sein)."""
        nonlocal cash
        pos      = portfolio[ticker]
        proceeds = pos["shares"] * sell_px - ORDER_FEE
        pnl      = proceeds - pos["cost"]
        ret_pct  = pnl / pos["cost"] * 100
        cash    += proceeds
        etype    = "Runner" if pos["has_partial_exit"] else "Full"
        completed.append({
            "ticker":         ticker,
            "slot":           pos["slot"],
            "entry_date":     pos["entry_date"],
            "exit_date":      sell_date,
            "entry_price":    round(pos["entry_price"], 2),
            "exit_price":     round(sell_px, 2),
            "shares":         pos["shares"],
            "pnl_€":          round(pnl, 2),
            "ret_%":          round(ret_pct, 2),
            "hold_d":         pos["days_held"],
            "earned_mode":    pos["earned_mode"],
            "earned_date":    pos["earned_date"],
            "pyramid_count":  pos["pyramid_count"],
            "max_unreal_%":   round(pos["max_unreal_pct"], 2),
            "exit_reason":    exit_reason,
            "exit_type":      etype,
            "is_rotation":    exit_reason == "Rotation",
            # Partial-Kontext (falls Runner)
            "partial_exit_date": pos["partial_exit_date"],
            "partial_exit_px":   pos["partial_exit_px"],
            "partial_exit_ret":  pos["partial_exit_ret"],
        })
        free_slots.add(pos["slot"])
        del portfolio[ticker]
        ticker_to_slot.pop(ticker, None)

    # ── Hauptschleife ─────────────────────────────────────────────────────────
    for day_i in range(len(dates) - 1):
        today    = dates[day_i]
        tomorrow = dates[day_i + 1]

        partial_triggers: list[str]           = []
        full_exits:       list[tuple[str,str]] = []

        for ticker, pos in list(portfolio.items()):
            tc  = _safe(piv["close"], today, ticker)
            th  = _safe(piv["high"],  today, ticker)
            ta  = _safe(piv["atr14"], today, ticker)
            if math.isnan(tc):
                continue

            pos["days_held"] += 1
            _update_stop(pos, th, ta)

            if pos["earned_mode"] and pos["earned_date"] is None:
                pos["earned_date"] = today

            unreal = (tc - pos["avg_entry_price"]) / pos["avg_entry_price"] * 100
            if unreal > pos["max_unreal_pct"]:
                pos["max_unreal_pct"] = unreal

            # ── Layer 1: Smart Exhaustion → Partial Trigger ──────────────────
            if (partial_exit
                    and pos["earned_mode"]
                    and not pos["has_partial_exit"]):
                rsi_t    = _safe(piv["rsi14"],      today, ticker)
                rsi_prev = _safe(piv["rsi14_prev"], today, ticker)
                dist50   = _safe(piv["dist_sma50"], today, ticker)
                overheat = (not (math.isnan(rsi_t) or math.isnan(dist50))
                            and rsi_t > EXHS_RSI and dist50 > EXHS_DIST50)
                rollover = (not (math.isnan(rsi_t) or math.isnan(rsi_prev))
                            and rsi_t < rsi_prev)
                if overheat and rollover:
                    partial_triggers.append(ticker)
                    continue   # kein Full-Exit-Check an diesem Tag

            # ── Layer 2/3: Stop prüfen ────────────────────────────────────────
            if tc < pos["trailing_stop"]:
                reason = "ATR"
                full_exits.append((ticker, reason))

        # Partial Exits ausführen (morgen open)
        for ticker in partial_triggers:
            sell_px = _safe(piv["open"], tomorrow, ticker)
            if not math.isnan(sell_px) and ticker in portfolio:
                _close_partial(ticker, tomorrow, sell_px)

        # Full Exits ausführen (morgen open)
        for ticker, reason in full_exits:
            if ticker not in portfolio:
                continue
            sell_px = _safe(piv["open"], tomorrow, ticker)
            if not math.isnan(sell_px):
                _close_full(ticker, tomorrow, sell_px, exit_reason=reason)

        # ── Neue Kandidaten + Diamond Hands Rotation ─────────────────────────
        candidates = [
            t for t in tickers
            if t not in portfolio
            and _safe(piv["entry_sig"], today, t) == 1.0
        ]
        if candidates:
            candidates.sort(
                key=lambda t: _safe(piv["trend_str"], today, t),
                reverse=True,
            )
            for cand in candidates:
                buy_px = _safe(piv["open"],  tomorrow, cand)
                atr_e  = _safe(piv["atr14"], today,    cand)
                if math.isnan(buy_px) or math.isnan(atr_e) or atr_e <= 0:
                    continue

                if len(portfolio) < MAX_POSITIONS and free_slots:
                    _open(cand, tomorrow, buy_px, atr_e, min(free_slots))

                elif len(portfolio) >= MAX_POSITIONS:
                    # Diamond Hands: nur nicht-earned rotierbar
                    rotatable = {t: p for t, p in portfolio.items()
                                 if not p["earned_mode"]}
                    if not rotatable:
                        continue
                    weakest_t = min(rotatable,
                                    key=lambda t: _safe(piv["trend_str"], today, t))
                    w_str     = _safe(piv["trend_str"], today, weakest_t)
                    c_str     = _safe(piv["trend_str"], today, cand)
                    if (not math.isnan(c_str) and not math.isnan(w_str)
                            and c_str > ROTATION_FACTOR * w_str):
                        rot_px = _safe(piv["open"], tomorrow, weakest_t)
                        if not math.isnan(rot_px):
                            freed = portfolio[weakest_t]["slot"]
                            _close_full(weakest_t, tomorrow, rot_px, "Rotation")
                            _open(cand, tomorrow, buy_px, atr_e, freed)
                            break

        # ── Pyramidisieren ────────────────────────────────────────────────────
        if len(portfolio) == 1 and free_slots:
            for ticker, pos in list(portfolio.items()):
                if pos["pyramid_count"] >= MAX_PYRAMIDS or pos["has_partial_exit"]:
                    continue   # keine Pyramide nach Partial Exit
                tc = _safe(piv["close"], today, ticker)
                if math.isnan(tc):
                    continue
                if (tc - pos["avg_entry_price"]) / pos["avg_entry_price"] >= PYRAMID_THRESHOLD:
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
                    pos["cost_per_share"]  = pos["cost"] / pos["shares"]  # aktualisieren
                    pos["avg_entry_price"] = new_avg
                    pos["pyramid_count"]  += 1
                    pos["trailing_stop"]   = max(pos["trailing_stop"], new_avg)

        # ── Equity loggen ─────────────────────────────────────────────────────
        equity = cash
        n_held = 0
        for ticker, pos in portfolio.items():
            cp     = _safe(piv["close"], today, ticker)
            equity += pos["shares"] * (cp if not math.isnan(cp)
                                       else pos["avg_entry_price"])
            n_held += 1
        equity_log[today] = equity
        if n_held > 0:
            invest_days += 1

    # Offene Positionen schliessen
    for ticker in list(portfolio.keys()):
        lp = _safe(piv["close"], dates[-1], ticker)
        if not math.isnan(lp):
            _close_full(ticker, dates[-1], lp, "End")

    return pd.Series(equity_log).sort_index(), completed, invest_days


# ==============================================================================
# 5. METRIKEN
# ==============================================================================

def compute_metrics(
    equity:      pd.Series,
    trades:      list[dict],
    invest_days: int,
) -> dict:
    eq    = equity.ffill().bfill()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.1)
    ret   = (eq.iloc[-1] / INITIAL_CAPITAL - 1) * 100
    cagr  = ((eq.iloc[-1] / INITIAL_CAPITAL) ** (1 / years) - 1) * 100
    peak  = eq.cummax()
    dd    = (eq - peak) / peak * 100
    dr    = eq.pct_change().dropna()
    sharpe = (dr.mean() / dr.std() * 252 ** 0.5) if dr.std() > 0 else 0.0

    # Nur Full + Runner für Standard-Metriken (kein Doppelzählen)
    main_trades = [t for t in trades if t["exit_type"] in ("Full", "Runner", "End")]
    partial_trd = [t for t in trades if t["exit_type"] == "Partial_50"]
    runner_trd  = [t for t in trades if t["exit_type"] == "Runner"]

    wins   = [t for t in main_trades if t["ret_%"] > 0]
    losses = [t for t in main_trades if t["ret_%"] <= 0]
    hit    = len(wins) / len(main_trades) * 100 if main_trades else 0
    avg_w  = float(np.mean([t["ret_%"] for t in wins]))   if wins   else 0.0
    avg_l  = float(np.mean([t["ret_%"] for t in losses])) if losses else 0.0
    payoff = abs(avg_w / avg_l) if avg_l != 0 else float("inf")
    pf     = (sum(t["ret_%"] for t in wins) /
              abs(sum(t["ret_%"] for t in losses))) if losses else float("inf")
    ev     = hit / 100 * avg_w + (1 - hit / 100) * avg_l

    # Partial-Stats
    p_ret  = float(np.mean([t["ret_%"] for t in partial_trd])) if partial_trd else 0.0
    p_wins = sum(1 for t in partial_trd if t["ret_%"] > 0)
    r_ret  = float(np.mean([t["ret_%"] for t in runner_trd]))  if runner_trd  else 0.0
    r_wins = sum(1 for t in runner_trd  if t["ret_%"] > 0)
    r_hold = float(np.mean([t["hold_d"] for t in runner_trd])) if runner_trd  else 0.0

    # Kombinierte Rendite pro Original-Position (Partial + Runner Paare)
    combo_pairs = []
    for rt in runner_trd:
        pt_match = next(
            (p for p in partial_trd
             if p["ticker"] == rt["ticker"]
             and p["entry_date"] == rt["entry_date"]),
            None,
        )
        if pt_match:
            # Gewichtete Kombinationsrendite (je 50% Gewicht)
            combo_ret = pt_match["ret_%"] * PARTIAL_FRAC + rt["ret_%"] * (1 - PARTIAL_FRAC)
            combo_pairs.append({
                "ticker":      rt["ticker"],
                "partial_ret": pt_match["ret_%"],
                "runner_ret":  rt["ret_%"],
                "combo_ret":   round(combo_ret, 2),
                "hold_d":      rt["hold_d"],
            })

    combo_avg = float(np.mean([c["combo_ret"] for c in combo_pairs])) if combo_pairs else 0.0

    n_atr = sum(1 for t in main_trades if t["exit_reason"] == "ATR")
    n_rot = sum(1 for t in trades if t.get("is_rotation"))
    n_end = sum(1 for t in trades if t["exit_reason"] == "End")
    rot_t = [t for t in trades if t.get("is_rotation")]
    rot_pnl = float(np.mean([t["ret_%"] for t in rot_t])) if rot_t else 0.0

    hold_w = float(np.mean([t["hold_d"] for t in wins]))   if wins   else 0.0
    hold_l = float(np.mean([t["hold_d"] for t in losses])) if losses else 0.0

    earned_t    = [t for t in main_trades if t["earned_mode"]]
    earned_rate = len(earned_t) / len(main_trades) * 100 if main_trades else 0.0
    earned_ret  = float(np.mean([t["ret_%"] for t in earned_t])) if earned_t else 0.0

    max_win_ret = max((t["ret_%"] for t in main_trades), default=0.0)
    max_win     = next((t for t in main_trades if t["ret_%"] == max_win_ret), None)

    annual = {}
    for year, grp in eq.groupby(eq.index.year):
        annual[year] = round((grp.iloc[-1] / grp.iloc[0] - 1) * 100, 1)

    dd_min_idx  = dd.idxmin()
    peak_before = eq[:dd_min_idx].idxmax()

    # Unique slot-count (Positionen, nicht Trades)
    n_positions = len(main_trades)

    return {
        "ret":          round(ret,    2),
        "cagr":         round(cagr,   2),
        "maxdd":        round(dd.min(), 1),
        "maxdd_date":   dd_min_idx,
        "maxdd_peak":   peak_before,
        "sharpe":       round(sharpe, 2),
        "n_positions":  n_positions,
        "n_total_rec":  len(trades),
        "n_atr":        n_atr,
        "n_partial":    len(partial_trd),
        "n_runner":     len(runner_trd),
        "n_rot":        n_rot,
        "n_end":        n_end,
        "fees_total":   len(trades) * ORDER_FEE * 2,
        "invest_pct":   round(invest_days / len(equity) * 100, 1),
        "hit":          round(hit,    1),
        "avg_win":      round(avg_w,  2),
        "avg_loss":     round(avg_l,  2),
        "payoff":       round(payoff, 2),
        "pf":           round(pf,     2),
        "ev":           round(ev,     2),
        "hold_w":       round(hold_w, 1),
        "hold_l":       round(hold_l, 1),
        "rot_avg_pnl":  round(rot_pnl, 2),
        "earned_rate":  round(earned_rate, 1),
        "earned_ret":   round(earned_ret,  2),
        # Partial Stats
        "p_n":          len(partial_trd),
        "p_wins":       p_wins,
        "p_ret":        round(p_ret,  2),
        "r_n":          len(runner_trd),
        "r_wins":       r_wins,
        "r_ret":        round(r_ret,  2),
        "r_hold":       round(r_hold, 1),
        "combo_n":      len(combo_pairs),
        "combo_avg":    round(combo_avg, 2),
        "combo_pairs":  combo_pairs,
        # Chart-Daten
        "max_win_ret":  round(max_win_ret, 2),
        "max_win":      max_win,
        "annual":       annual,
        "years":        round(years, 1),
        "end_cap":      round(eq.iloc[-1], 0),
        "_dd":          dd,
        "_eq":          eq,
        "_peak":        peak,
    }


# ==============================================================================
# 6. AUSGABE
# ==============================================================================

def print_comparison(ma: dict, mb: dict) -> None:
    sep  = "=" * 90
    line = "─" * 90

    def row(label: str, a, b, fmt: str = "{}", hi: str = "high") -> None:
        va = fmt.format(a) if not isinstance(a, str) else a
        vb = fmt.format(b) if not isinstance(b, str) else b
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            better_a = (a > b) if hi == "high" else (a < b)
            mark_a, mark_b = (" ◄", "  ") if better_a else ("  ", " ◄")
        else:
            mark_a = mark_b = "  "
        print(f"  {label:<42}  {va:>14}{mark_a}  {vb:>14}{mark_b}")

    print(f"\n{sep}")
    print(f"  PARTIAL EXITS v12.0  |  Setup A (Baseline) vs. Setup B (Partial Machine)")
    print(sep)
    print(f"  {'Metrik':<42}  {'Setup A (Baseline)':>16}  {'Setup B (Partial)':>16}")
    print(f"  {line}")

    print(f"\n  ─ RENDITE {'─'*79}")
    row("Gesamtrendite",       ma["ret"],    mb["ret"],    "{:>+.2f}%")
    row("CAGR (p.a.)",         ma["cagr"],   mb["cagr"],   "{:>+.2f}%")
    row("End-Kapital (€)",     ma["end_cap"],mb["end_cap"],"{:>,.0f}€")

    print(f"\n  ─ RISIKO {'─'*80}")
    row("Max Drawdown",        ma["maxdd"],  mb["maxdd"],  "{:>.1f}%", hi="high")
    row("Sharpe Ratio",        ma["sharpe"], mb["sharpe"], "{:.2f}")
    row("Investitionsquote",   ma["invest_pct"],mb["invest_pct"],"{:.1f}%")

    print(f"\n  ─ TRADE-QUALITÄT (Positionen ohne Doppelzählung) {'─'*39}")
    row("Anzahl Positionen",   ma["n_positions"],mb["n_positions"],"{:>}")
    row("Anzahl Trade-Records",ma["n_total_rec"],mb["n_total_rec"],"{:>}")
    row("Hit-Rate (Vollausg.)",ma["hit"],    mb["hit"],    "{:.1f}%")
    row("Ø Gewinner",          ma["avg_win"],mb["avg_win"],"{:>+.2f}%")
    row("Ø Verlierer",         ma["avg_loss"],mb["avg_loss"],"{:>+.2f}%", hi="low")
    row("Payoff-Ratio",        ma["payoff"], mb["payoff"], "{:.2f}")
    row("Profit Factor",       ma["pf"],     mb["pf"],     "{:.2f}")
    row("EV / Trade",          ma["ev"],     mb["ev"],     "{:>+.2f}%")

    print(f"\n  ─ EXIT-ANALYSE {'─'*74}")
    row("ATR-Exits",           ma["n_atr"],  mb["n_atr"],  "{:>}")
    row("Partial_50-Exits",    ma["n_partial"],mb["n_partial"],"{:>}")
    row("Runner-Exits",        ma["n_runner"],mb["n_runner"],"{:>}")
    row("Rotations-Exits",     ma["n_rot"],  mb["n_rot"],  "{:>}", hi="low")
    row("Ø PnL Rotation",      ma["rot_avg_pnl"],mb["rot_avg_pnl"],"{:>+.2f}%")
    row("Gezahlte Gebühren",   ma["fees_total"],mb["fees_total"],"{:>,.0f}€", hi="low")

    print(f"\n  ─ PARTIAL EXITS DETAIL (nur Setup B) {'─'*51}")
    print(f"  {'Partial_50-Exits':42}  {'—':>16}   {mb['p_n']:>14}")
    print(f"  {'  davon Winners':42}  {'—':>16}   {mb['p_wins']:>14}")
    print(f"  {'  Hit-Rate Partial':42}  {'—':>16}   "
          f"{mb['p_wins']/max(mb['p_n'],1)*100:>13.1f}%")
    print(f"  {'  Ø Rendite Partial_50':42}  {'—':>16}   {mb['p_ret']:>+13.2f}%")
    print(f"  {'Runner-Exits (nach Partial)':42}  {'—':>16}   {mb['r_n']:>14}")
    print(f"  {'  davon Winners':42}  {'—':>16}   {mb['r_wins']:>14}")
    print(f"  {'  Hit-Rate Runner':42}  {'—':>16}   "
          f"{mb['r_wins']/max(mb['r_n'],1)*100:>13.1f}%")
    print(f"  {'  Ø Rendite Runner':42}  {'—':>16}   {mb['r_ret']:>+13.2f}%")
    print(f"  {'  Ø Haltezeit Runner (ab Entry)':42}  {'—':>16}   {mb['r_hold']:>13.1f}d")
    print(f"  {'Kombinierte Paare (Partial+Runner)':42}  {'—':>16}   {mb['combo_n']:>14}")
    print(f"  {'  Ø Kombinierte Rendite':42}  {'—':>16}   {mb['combo_avg']:>+13.2f}%")

    print(f"\n  ─ TRADE LIFECYCLE {'─'*71}")
    row("Ø Haltezeit Winners", ma["hold_w"],    mb["hold_w"],    "{:.1f}d")
    row("Ø Haltezeit Losers",  ma["hold_l"],    mb["hold_l"],    "{:.1f}d", hi="low")
    row("Earned-Mode Rate",    ma["earned_rate"],mb["earned_rate"],"{:.1f}%")
    row("Ø Return Earned",     ma["earned_ret"],mb["earned_ret"], "{:>+.2f}%")

    print(f"\n  ─ MEGA-TRADE CHECK {'─'*70}")
    mwa = ma.get("max_win")
    mwb = mb.get("max_win")
    if mwa:
        et = mwa.get("exit_type", "Full")
        print(f"  {'Größter Winner Setup A':42}  "
              f"{mwa['ticker']} {mwa['ret_%']:>+.1f}% ({int(mwa['hold_d'])}d) [{et}]")
    if mwb:
        et = mwb.get("exit_type", "Full")
        print(f"  {'Größter Winner Setup B':42}  "
              f"{mwb['ticker']} {mwb['ret_%']:>+.1f}% ({int(mwb['hold_d'])}d) [{et}]")

    print(f"\n  ─ JÄHRLICHE RENDITEN {'─'*68}")
    all_years = sorted(set(ma["annual"]) | set(mb["annual"]))
    for yr in all_years:
        row(f"  {yr}", ma["annual"].get(yr, 0.0), mb["annual"].get(yr, 0.0), "{:>+.1f}%")

    print(f"\n{sep}")

    dt  = mb["n_total_rec"] - ma["n_total_rec"]
    ddd = mb["maxdd"] - ma["maxdd"]
    dr  = mb["ret"]   - ma["ret"]
    df  = mb["fees_total"] - ma["fees_total"]
    print(f"""
  BEWERTUNG der Kernfragen:
  {'─' * 65}
  Trade-Records:  {mb['n_total_rec']} vs. {ma['n_total_rec']}  (Δ{dt:>+d})
  Positionen:     {mb['n_positions']} vs. {ma['n_positions']}
  → Churning gelöst?  {"Ja ✓" if mb['n_positions'] <= ma['n_positions'] + 5 else f"Teilweise (+{mb['n_positions']-ma['n_positions']} Positionen)"}

  Max Drawdown:   {mb['maxdd']:.1f}% vs. {ma['maxdd']:.1f}%  (Δ{ddd:>+.1f}%)
  → Drawdown reduziert?  {"Ja ✓" if ddd > 0 else "Nein ✗"}

  Gesamtrendite:  {mb['ret']:>+.2f}% vs. {ma['ret']:>+.2f}%  (Δ{dr:>+.2f}%)
  → Rendite bewahrt?  {"Ja ✓" if dr >= -2 else f"Nein ✗  (Δ{dr:>+.2f}%)"}

  Gebühren:       {mb['fees_total']:>,.0f}€ vs. {ma['fees_total']:>,.0f}€  (Δ{df:>+,.0f}€)

  Partial-Exit Qualität:
    Ø Rendite Partial:  {mb['p_ret']:>+.2f}%  (Win-Rate {mb['p_wins']/max(mb['p_n'],1)*100:.0f}%)
    Ø Rendite Runner:   {mb['r_ret']:>+.2f}%  (Win-Rate {mb['r_wins']/max(mb['r_n'],1)*100:.0f}%)
    Ø Kombi-Rendite:    {mb['combo_avg']:>+.2f}%  (bei {mb['combo_n']} Paaren)
""")


def print_combo_pairs(mb: dict) -> None:
    pairs = mb.get("combo_pairs", [])
    if not pairs:
        return
    pairs_sorted = sorted(pairs, key=lambda x: x["combo_ret"], reverse=True)
    sep = "─" * 72
    print(f"  PARTIAL + RUNNER KOMBINATIONEN  (n={len(pairs)})")
    print(f"  {sep}")
    print(f"  {'Ticker':<8}  {'Partial %':>10}  {'Runner %':>10}  {'Kombi %':>10}  "
          f"{'Hold (d)':>9}  Note")
    print(f"  {sep}")
    for p in pairs_sorted:
        note = "✓ Gewinn" if p["combo_ret"] > 0 else "✗ Verlust"
        saved = " ★ Runner gerettet!" if p["runner_ret"] > p["partial_ret"] else ""
        print(f"  {p['ticker']:<8}  {p['partial_ret']:>+9.2f}%  "
              f"{p['runner_ret']:>+9.2f}%  {p['combo_ret']:>+9.2f}%  "
              f"{p['hold_d']:>8}d  {note}{saved}")
    print(f"  {sep}")
    print(f"  Ø Kombinierte Rendite: {mb['combo_avg']:>+.2f}%\n")


# ==============================================================================
# 7. CHART
# ==============================================================================

def plot_comparison(
    ma: dict, mb: dict,
    trades_a: list[dict], trades_b: list[dict],
    out_png: Path,
) -> None:
    eq_a = ma["_eq"]; pk_a = ma["_peak"]; dd_a = ma["_dd"]
    eq_b = mb["_eq"]; pk_b = mb["_peak"]; dd_b = mb["_dd"]

    C_A     = "#1565c0"
    C_B     = "#2e7d32"
    C_PART  = "#e65100"
    C_RUN   = "#f9a825"
    C_ROT   = "#9c27b0"
    C_WIN   = "#388e3c"
    C_LOS   = "#c62828"
    C_BG    = "#f8f9fa"
    C_GRD   = "#dee2e6"

    fig = plt.figure(figsize=(22, 18), dpi=150, facecolor=C_BG)
    fig.suptitle(
        "Partial Exits v12.0  |  Setup A (Baseline) vs. Setup B (VCP + Partial Take-Profit)",
        fontsize=13, fontweight="bold", y=0.99, color="#212121",
    )
    gs = fig.add_gridspec(
        4, 1, height_ratios=[3.5, 1.8, 1.8, 1.2],
        hspace=0.08, left=0.06, right=0.97, top=0.97, bottom=0.05,
    )

    # ── Panel 1: Equity ───────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(C_BG)
    ax1.fill_between(eq_a.index, eq_a, pk_a, where=(eq_a < pk_a), color=C_A, alpha=0.10)
    ax1.fill_between(eq_b.index, eq_b, pk_b, where=(eq_b < pk_b), color=C_B, alpha=0.10)
    ax1.plot(eq_a.index, eq_a, color=C_A, lw=2.1,
             label=(f"Setup A – Baseline  "
                    f"({ma['ret']:>+.1f}%  CAGR {ma['cagr']:>+.1f}%  "
                    f"DD {ma['maxdd']:.1f}%  Sharpe {ma['sharpe']:.2f})"))
    ax1.plot(eq_b.index, eq_b, color=C_B, lw=2.1,
             label=(f"Setup B – Partial  "
                    f"({mb['ret']:>+.1f}%  CAGR {mb['cagr']:>+.1f}%  "
                    f"DD {mb['maxdd']:.1f}%  Sharpe {mb['sharpe']:.2f})"))
    ax1.axhline(INITIAL_CAPITAL, color="#9e9e9e", lw=0.8, ls=":", alpha=0.9,
                label=f"Start ({INITIAL_CAPITAL:,.0f}€)")

    # Partial-Exit-Ereignisse als Dreiecke auf Equity-Kurve B
    for t in trades_b:
        if t["exit_type"] != "Partial_50":
            continue
        xd = t["exit_date"]
        if xd in eq_b.index:
            yp = float(eq_b.at[xd])
            ax1.plot(xd, yp, "v", color=C_PART, ms=5, alpha=0.7, zorder=5)

    for eq, dd, col, m in [(eq_a, dd_a, C_A, ma), (eq_b, dd_b, C_B, mb)]:
        di = dd.idxmin(); dy = float(eq.at[di])
        ax1.annotate(f"DD {m['maxdd']:.1f}%",
                     xy=(di, dy), xytext=(di, dy * 0.85),
                     arrowprops=dict(arrowstyle="->", color=col, lw=1.1),
                     fontsize=7.5, color=col, fontweight="bold", ha="center")

    for eq, col, m in [(eq_a, C_A, ma), (eq_b, C_B, mb)]:
        ax1.annotate(f"  {m['end_cap']:,.0f}€",
                     xy=(eq.index[-1], float(eq.iloc[-1])),
                     fontsize=9, color=col, fontweight="bold", va="center")

    part_handle = plt.Line2D([0], [0], marker="v", color=C_PART, lw=0, ms=7,
                              label=f"Partial-Exit (n={mb['p_n']})")
    ax1.set_ylabel("Kapital (€)", fontsize=9)
    ax1.tick_params(axis="x", labelbottom=False)
    ax1.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}€"))
    ax1.grid(True, color=C_GRD, lw=0.5)
    h, l = ax1.get_legend_handles_labels()
    ax1.legend(h + [part_handle], l + [part_handle.get_label()],
               loc="upper left", fontsize=8.5, framealpha=0.85)

    txt = (f"{'':28} {'Baseline':>9}  {'Partial':>9}\n"
           f"{'Positionen':28} {ma['n_positions']:>9}  {mb['n_positions']:>9}\n"
           f"{'Hit-Rate':28} {ma['hit']:>8.1f}%  {mb['hit']:>8.1f}%\n"
           f"{'Payoff':28} {ma['payoff']:>9.2f}  {mb['payoff']:>9.2f}\n"
           f"{'EV/Trade':28} {ma['ev']:>+8.2f}%  {mb['ev']:>+8.2f}%\n"
           f"{'Fees':28} {ma['fees_total']:>8,.0f}€  {mb['fees_total']:>8,.0f}€\n"
           f"{'Partial Exits':28} {'—':>9}   {mb['p_n']:>8}\n"
           f"{'Ø Partial Rendite':28} {'—':>9}   {mb['p_ret']:>+8.2f}%\n"
           f"{'Ø Runner Rendite':28} {'—':>9}   {mb['r_ret']:>+8.2f}%")
    ax1.text(0.985, 0.97, txt, transform=ax1.transAxes,
             fontsize=7.5, va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                       edgecolor="#cccccc", alpha=0.9),
             family="monospace")

    # ── Panel 2/3: Gantt ──────────────────────────────────────────────────────
    def _gantt(ax, trades, title, c_win):
        ax.set_facecolor(C_BG)
        ax.set_ylim(-0.6, 2.6)
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(["Slot 2", "Slot 1"], fontsize=8)
        ax.set_ylabel(title, fontsize=8.5)

        # Nur Full/Runner/End für den Hauptbalken (Laufzeit der gesamten Position)
        for t in trades:
            etype = t.get("exit_type", "Full")
            if etype == "Partial_50":
                continue
            y_bot = t["slot"] - 1
            x0 = mdates.date2num(t["entry_date"])
            x1 = mdates.date2num(t["exit_date"])
            w  = max(x1 - x0, 0.5)
            r  = t["exit_reason"]
            ret = t["ret_%"]
            if r == "Rotation":
                col, ec = C_ROT, "#4a148c"
            elif etype == "Runner":
                col, ec = C_RUN, "#e65100"
            elif r == "End":
                col, ec = "#607d8b", "#37474f"
            elif ret > 0:
                col, ec = c_win, "#1b5e20"
            else:
                col, ec = C_LOS, "#b71c1c"
            ax.add_patch(mpatches.FancyBboxPatch(
                (x0, y_bot + 0.08), w, 0.84,
                boxstyle="round,pad=0", facecolor=col, edgecolor=ec,
                lw=0.4, alpha=0.85))
            if w > 3:
                sign = "+" if ret >= 0 else ""
                ax.text((x0 + x1) / 2, y_bot + 0.5,
                        f"{t['ticker']} {sign}{ret:.0f}%",
                        ha="center", va="center",
                        fontsize=5, color="white", fontweight="bold", clip_on=True)

            # Partial-Exit-Marker (senkrechte Linie beim Teilverkauf)
            if etype == "Runner" and t.get("partial_exit_date"):
                xp = mdates.date2num(t["partial_exit_date"])
                ax.axvline(xp, ymin=(y_bot + 0.08) / 3,
                           ymax=(y_bot + 0.92) / 3,
                           color=C_PART, lw=1.5, alpha=0.85)

        ax.set_xlim(mdates.date2num(eq_a.index[0]) - 5,
                    mdates.date2num(eq_a.index[-1]) + 5)
        ax.grid(True, color=C_GRD, lw=0.4, axis="x")
        ax.tick_params(axis="x", labelbottom=False)
        patches = [mpatches.Patch(color=c_win, label="Winner (Full)"),
                   mpatches.Patch(color=C_LOS,  label="Loser"),
                   mpatches.Patch(color=C_RUN,  label="Runner (nach Partial)"),
                   mpatches.Patch(color=C_ROT,  label="Rotation")]
        ax.legend(handles=patches, loc="upper left",
                  fontsize=6.5, framealpha=0.7, ncol=4)

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    _gantt(ax2, trades_a, "Setup A\n(Baseline)", C_WIN)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    _gantt(ax3, trades_b, "Setup B\n(Partial)", C_B)

    # ── Panel 4: Jahresrenditen ───────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[3])
    ax4.set_facecolor(C_BG)
    all_years = sorted(set(ma["annual"]) | set(mb["annual"]))
    x = np.arange(len(all_years)); w = 0.38
    ba_v = [ma["annual"].get(yr, 0.0) for yr in all_years]
    bb_v = [mb["annual"].get(yr, 0.0) for yr in all_years]
    ba = ax4.bar(x - w/2, ba_v, w,
                 color=[C_WIN if v>=0 else C_LOS for v in ba_v],
                 edgecolor="#424242", lw=0.5, alpha=0.85, label="A Baseline")
    bb = ax4.bar(x + w/2, bb_v, w,
                 color=[C_B   if v>=0 else "#bf360c" for v in bb_v],
                 edgecolor="#424242", lw=0.5, alpha=0.85, label="B Partial")
    ax4.axhline(0, color="#424242", lw=0.8)
    ax4.set_xticks(x); ax4.set_xticklabels(all_years, fontsize=8)
    ax4.set_ylabel("Jahresrendite (%)", fontsize=9)
    ax4.grid(True, color=C_GRD, lw=0.4, axis="y")
    ax4.legend(fontsize=8, loc="upper left", framealpha=0.8)
    for bars, vals in [(ba, ba_v), (bb, bb_v)]:
        for bar, val in zip(bars, vals):
            sign = "+" if val >= 0 else ""
            va = "bottom" if val >= 0 else "top"
            ax4.text(bar.get_x() + bar.get_width()/2, val + (0.5 if val>=0 else -0.5),
                     f"{sign}{val:.0f}%", ha="center", va=va,
                     fontsize=6.5, color="#212121")

    plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"  Chart gespeichert: {out_png}")


# ==============================================================================
# 8. MAIN
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Partial Exits v12.0  |  VCP + 50% Partial Take-Profit")
    parser.add_argument("--years", type=float, default=DEFAULT_YEARS)
    args = parser.parse_args()

    sep = "=" * 72
    print(sep)
    print("  PARTIAL EXITS v12.0  |  VCP + Smart Exhaustion 50% Scale-Out")
    print(sep)
    print(f"""
  Layer 1 – Partial Take-Profit (50%):
    Earned Mode  AND  RSI>{EXHS_RSI:.0f}  AND  dSMA50>{EXHS_DIST50*100:.0f}%  AND  ΔRSI<0
    → Verkaufe 50% sofort. Verbuche Gewinn. Runner läuft weiter.

  Layer 2 – Runner (50%):
    Nur noch 3.5× ATR-Stop. Kein weiterer Exhaustion-Check.

  Layer 3 – Standard:
    2.0× ATR für Fresh-Trades. Diamond Hands Rotation.

  Portfolio:  {INITIAL_CAPITAL:,.0f}€  |  Fee {ORDER_FEE:.0f}€  |  {MAX_POSITIONS} Slots  |  {args.years:.0f} Jahre
""")

    print("[1/4] Lade Daten...")
    t0      = time.time()
    tickers = _load_tickers()
    data    = _load_ohlcv(tickers, args.years)
    print(f"  {len(data)} Ticker in {time.time()-t0:.1f}s")

    print("\n[2/4] Panels aufbauen...")
    t0     = time.time()
    panels = build_panels(data)
    dates  = panels["open"].index
    n_sig  = panels.pop("_n_sig")
    print(f"  Zeitraum:  {dates[0].date()} → {dates[-1].date()}")
    print(f"  Signale:   {n_sig:,}  |  Aufbau: {time.time()-t0:.1f}s")

    print("\n[3/4] Simulation Setup A (Baseline)...")
    t0 = time.time()
    eq_a, trades_a, inv_a = run_backtest(panels, partial_exit=False)
    print(f"  {time.time()-t0:.1f}s  |  {len(trades_a)} Trade-Records  "
          f"({sum(1 for t in trades_a if t['exit_type']=='Full')} Positionen)")

    print("\n[4/4] Simulation Setup B (Partial Exit Machine)...")
    t0 = time.time()
    eq_b, trades_b, inv_b = run_backtest(panels, partial_exit=True)
    n_part = sum(1 for t in trades_b if t["exit_type"] == "Partial_50")
    n_full = sum(1 for t in trades_b if t["exit_type"] in ("Full", "Runner", "End"))
    print(f"  {time.time()-t0:.1f}s  |  {len(trades_b)} Trade-Records  "
          f"({n_full} Positionen, davon {n_part} Partial_50)")

    ma = compute_metrics(eq_a, trades_a, inv_a)
    mb = compute_metrics(eq_b, trades_b, inv_b)

    print_comparison(ma, mb)
    print_combo_pairs(mb)

    # Top 5 Winners pro Setup
    for label, trades in [("Setup A", trades_a), ("Setup B", trades_b)]:
        main_t = [t for t in trades
                  if t.get("exit_type") in ("Full", "Runner", "End")]
        df_t   = pd.DataFrame(main_t).sort_values("ret_%", ascending=False)
        print(f"  TOP 5 WINNER [{label}]:")
        for _, r in df_t.head(5).iterrows():
            etype = r.get("exit_type", "Full")
            earned = " ★Earned" if r.get("earned_mode") else ""
            print(f"    {r['ticker']:<6}  "
                  f"{str(r['entry_date'])[:10]} → {str(r['exit_date'])[:10]}  "
                  f"({int(r['hold_d'])}d)  {r['ret_%']:>+.1f}%{earned}  [{etype}]")
        print()

    pd.DataFrame(trades_a).assign(setup="A").pipe(
        lambda df: pd.concat(
            [df, pd.DataFrame(trades_b).assign(setup="B")],
            ignore_index=True,
        )
    ).to_csv(_OUT_CSV, index=False)
    print(f"  Trades: {_OUT_CSV}")

    plot_comparison(ma, mb, trades_a, trades_b, _OUT_PNG)
    print("  FERTIG.\n")


if __name__ == "__main__":
    main()
