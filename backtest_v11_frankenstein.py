"""
backtest_v11_frankenstein.py
====================================================================================
Frankenstein v11.0  |  VCP-Entry + ML-Exit  |  Setup A vs. Setup B

Kernfrage: Schützt der ML-Exhaustion-Stop unser Kapital besser vor VCP-Umkehrformationen?

Beide Setups teilen sich denselben VCP-Einstieg:
    Breakout_50 + BB-Squeeze(<10%) + Vol>1.5× + Close>SMA200

Setup A  ─  Baseline Champion (v8.3)
    Ausstieg:  Asymmetrischer ATR-Stop  2.0× initial  →  3.5× (Earned Mode)

Setup B  ─  Frankenstein (VCP + ML-Exhaustion-Stop)
    Ausstieg:  Wie Setup A, ZUSÄTZLICH Exhaustion-Check jeden Tag:
               WENN RSI_14 > 72 UND (Close−SMA50)/SMA50 > 0.07
               → "Tight Trailing Stop" aktivieren:
                  Stop = max(aktueller_Stop, min(Low_heute, Low_gestern))
               Dieser "Erschöpfungs-Stop" überschreibt den ATR-Stop,
               wenn er näher am Kurs liegt (enger = protektiver).

Portfolio (unveränderlich):
    INITIAL_CAPITAL = 10.000€  |  ORDER_FEE = 20€  |  MAX_POSITIONS = 2
    ROTATION_FACTOR = 1.5×     |  Diamond Hands (Earned-Mode nie rotierbar)

Verwendung:
    python backtest_v11_frankenstein.py
    python backtest_v11_frankenstein.py --years 7
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

# ── Portfolio-Konstanten ──────────────────────────────────────────────────────
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

# VCP-Entry-Parameter
BB_PERIOD         = 20
BB_STD            = 2.0
BB_SQUEEZE        = 0.10
VOL_MULT          = 1.5

# ML-Exhaustion-Parameter (v9.6)
EXHS_RSI          = 72.0
EXHS_DIST50       = 0.07

_RAW_DIR = _here / "data" / "raw"
_OUT_PNG  = _here / "frankenstein_equity.png"
_OUT_CSV  = _here / "frankenstein_trades.csv"


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
            keep = [c for c in ["open", "high", "low", "close", "volume"]
                    if c in df.columns]
            data[ticker] = df[keep].copy()
        except Exception:
            pass
    return data


# ==============================================================================
# 2. TECHNISCHE INDIKATOREN
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
# 3. PANELS AUFBAUEN (Panels geteilt für beide Setups)
# ==============================================================================

def build_panels(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Baut alle Pivot-Panels auf. VCP-Entry-Signal und alle Exit-Indikatoren
    für beide Setups in einem einzigen Durchlauf.
    """
    cols: dict[str, dict] = {
        k: {} for k in [
            "open", "close", "high", "low",
            "atr14", "trend_str", "entry_sig",
            "rsi14", "dist_sma50", "low_2d_min",
        ]
    }
    n_sig_total = 0
    for ticker, df in data.items():
        c   = df["close"]
        lo  = df["low"]
        h   = df["high"]
        vol = df.get("volume")

        sma50    = c.rolling(50).mean()
        sma200   = c.rolling(200).mean()
        sma20    = c.rolling(BB_PERIOD).mean()
        std20    = c.rolling(BB_PERIOD).std()
        bb_w     = (sma20 + BB_STD * std20
                    - (sma20 - BB_STD * std20)) / c.replace(0, np.nan)
        atr14    = _atr(df, 14)
        rsi14    = _rsi(c, 14)
        sma20v   = vol.rolling(20).mean() if vol is not None else None
        dist50   = (c - sma50) / sma50.replace(0, np.nan)

        # Tight Trailing Stop: min(Low_heute, Low_gestern)
        low_2d   = pd.concat([lo, lo.shift(1)], axis=1).min(axis=1)

        # VCP-Entry-Signal (identisch für A und B)
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
        cols["dist_sma50"][ticker] = dist50[valid]
        cols["low_2d_min"][ticker] = low_2d[valid]

    panels = {k: pd.DataFrame(v) for k, v in cols.items()}
    panels["_n_sig"] = n_sig_total          # Hilfswert für Ausgabe
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
    pivots:      dict[str, pd.DataFrame],
    ml_exit:     bool = False,
) -> tuple[pd.Series, list[dict], int]:
    """
    ml_exit=False  →  Setup A: reiner ATR-Stop
    ml_exit=True   →  Setup B: ATR-Stop + ML-Exhaustion Tight Trailing Stop
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
            "cost":             cost,
            "atr_at_entry":     atr_e,
            "trailing_stop":    buy_px - ATR_INIT * atr_e,
            "max_high":         buy_px,
            "earned_mode":      False,
            "earned_date":      None,
            "pyramid_count":    0,
            "avg_entry_price":  buy_px,
            "days_held":        0,
            "max_unreal_pct":   0.0,
            # ML-Exhaustion-Tracking
            "exhausted":        False,
            "exhaust_date":     None,
            "exhaust_stop_days": 0,   # Tage, an denen Tight-Stop aktiv war
        }
        ticker_to_slot[ticker] = slot
        free_slots.discard(slot)
        return True

    def _close(ticker: str, sell_date, sell_px: float,
               exit_reason: str) -> None:
        nonlocal cash
        pos      = portfolio[ticker]
        proceeds = pos["shares"] * sell_px - ORDER_FEE
        pnl      = proceeds - pos["cost"]
        ret_pct  = pnl / pos["cost"] * 100
        cash    += proceeds
        completed.append({
            "ticker":            ticker,
            "slot":              pos["slot"],
            "entry_date":        pos["entry_date"],
            "exit_date":         sell_date,
            "entry_price":       round(pos["entry_price"], 2),
            "exit_price":        round(sell_px, 2),
            "shares":            pos["shares"],
            "pnl_€":             round(pnl, 2),
            "ret_%":             round(ret_pct, 2),
            "hold_d":            pos["days_held"],
            "earned_mode":       pos["earned_mode"],
            "earned_date":       pos["earned_date"],
            "pyramid_count":     pos["pyramid_count"],
            "max_unreal_%":      round(pos["max_unreal_pct"], 2),
            "exit_reason":       exit_reason,
            "is_rotation":       exit_reason == "Rotation",
            "exhausted":         pos["exhausted"],
            "exhaust_date":      pos["exhaust_date"],
            "exhaust_stop_days": pos["exhaust_stop_days"],
        })
        free_slots.add(pos["slot"])
        del portfolio[ticker]
        ticker_to_slot.pop(ticker, None)

    # ── Hauptschleife ─────────────────────────────────────────────────────────
    for day_i in range(len(dates) - 1):
        today    = dates[day_i]
        tomorrow = dates[day_i + 1]

        # A. Stops prüfen + Positionen aktualisieren
        exits: list[tuple[str, str]] = []
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

            # ── ML-Exhaustion-Check (nur Setup B) ────────────────────────────
            if ml_exit:
                rsi_t  = _safe(piv["rsi14"],     today, ticker)
                dist50 = _safe(piv["dist_sma50"], today, ticker)
                low2d  = _safe(piv["low_2d_min"], today, ticker)

                if (not math.isnan(rsi_t) and not math.isnan(dist50)
                        and rsi_t > EXHS_RSI and dist50 > EXHS_DIST50):
                    if not pos["exhausted"]:
                        pos["exhausted"]   = True
                        pos["exhaust_date"] = today

                if pos["exhausted"] and not math.isnan(low2d):
                    # Tight Trailing Stop = min(Low_heute, Low_gestern)
                    # überschreibt ATR-Stop wenn enger (höher)
                    if low2d > pos["trailing_stop"]:
                        pos["trailing_stop"] = low2d
                    pos["exhaust_stop_days"] += 1

            if tc < pos["trailing_stop"]:
                reason = ("Exhaust"
                          if (ml_exit and pos["exhausted"])
                          else "ATR")
                exits.append((ticker, reason))

        for ticker, reason in exits:
            sell_px = _safe(piv["open"], tomorrow, ticker)
            if not math.isnan(sell_px):
                _close(ticker, tomorrow, sell_px, exit_reason=reason)

        # B. Neue Kandidaten + Diamond Hands Rotation
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
                            _close(weakest_t, tomorrow, rot_px, "Rotation")
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
                    pos["avg_entry_price"] = new_avg
                    pos["pyramid_count"]  += 1
                    pos["trailing_stop"]   = max(pos["trailing_stop"], new_avg)

        # D. Equity loggen
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

    # Offene Positionen zum letzten Kurs schliessen
    for ticker in list(portfolio.keys()):
        lp = _safe(piv["close"], dates[-1], ticker)
        if not math.isnan(lp):
            _close(ticker, dates[-1], lp, "End")

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

    wins   = [t for t in trades if t["ret_%"] > 0]
    losses = [t for t in trades if t["ret_%"] <= 0]
    hit    = len(wins) / len(trades) * 100 if trades else 0
    avg_w  = float(np.mean([t["ret_%"] for t in wins]))   if wins   else 0.0
    avg_l  = float(np.mean([t["ret_%"] for t in losses])) if losses else 0.0
    payoff = abs(avg_w / avg_l) if avg_l != 0 else float("inf")
    pf     = (sum(t["ret_%"] for t in wins) /
              abs(sum(t["ret_%"] for t in losses))) if losses else float("inf")
    ev     = hit / 100 * avg_w + (1 - hit / 100) * avg_l

    n_atr = sum(1 for t in trades if t["exit_reason"] == "ATR")
    n_exh = sum(1 for t in trades if t["exit_reason"] == "Exhaust")
    n_rot = sum(1 for t in trades if t["exit_reason"] == "Rotation")
    n_end = sum(1 for t in trades if t["exit_reason"] == "End")

    exh_trades  = [t for t in trades if t["exit_reason"] == "Exhaust"]
    exh_wins    = [t for t in exh_trades if t["ret_%"] > 0]
    exh_avg_ret = float(np.mean([t["ret_%"] for t in exh_trades])) if exh_trades else 0.0
    exh_avg_dd  = float(np.mean([t["exhaust_stop_days"] for t in exh_trades])) if exh_trades else 0.0

    # Exhaustion-aktivierte Trades (Exhaustion wurde getriggert aber über ATR/Rotation beendet)
    exh_triggered = [t for t in trades if t.get("exhausted")]
    exh_trig_ret  = float(np.mean([t["ret_%"] for t in exh_triggered])) if exh_triggered else 0.0

    rot_exits = [t for t in trades if t["is_rotation"]]
    rot_pnl   = float(np.mean([t["ret_%"] for t in rot_exits])) if rot_exits else 0.0

    hold_w = float(np.mean([t["hold_d"] for t in wins]))   if wins   else 0.0
    hold_l = float(np.mean([t["hold_d"] for t in losses])) if losses else 0.0

    earned_t    = [t for t in trades if t["earned_mode"]]
    earned_rate = len(earned_t) / len(trades) * 100 if trades else 0.0
    earned_ret  = float(np.mean([t["ret_%"] for t in earned_t])) if earned_t else 0.0

    annual = {}
    for year, grp in eq.groupby(eq.index.year):
        annual[year] = round((grp.iloc[-1] / grp.iloc[0] - 1) * 100, 1)

    dd_min_idx    = dd.idxmin()
    peak_before   = eq[:dd_min_idx].idxmax()

    return {
        "ret":          round(ret,    2),
        "cagr":         round(cagr,   2),
        "maxdd":        round(dd.min(), 1),
        "maxdd_date":   dd_min_idx,
        "maxdd_peak":   peak_before,
        "sharpe":       round(sharpe, 2),
        "n_trades":     len(trades),
        "n_atr":        n_atr,
        "n_exh":        n_exh,
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
        # Exhaustion-Detail
        "exh_n":        n_exh,
        "exh_wins":     len(exh_wins),
        "exh_avg_ret":  round(exh_avg_ret, 2),
        "exh_avg_days": round(exh_avg_dd,  1),
        "exh_triggered": len(exh_triggered),
        "exh_trig_ret": round(exh_trig_ret, 2),
        # Charts
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
    sep  = "=" * 88
    line = "─" * 88

    def row(label: str, a, b, fmt: str = "{}", hi: str = "high") -> None:
        va = fmt.format(a) if not isinstance(a, str) else a
        vb = fmt.format(b) if not isinstance(b, str) else b
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            better_a = (a > b) if hi == "high" else (a < b)
            mark_a, mark_b = (" ◄", "  ") if better_a else ("  ", " ◄")
        else:
            mark_a = mark_b = "  "
        print(f"  {label:<38}  {va:>14}{mark_a}  {vb:>14}{mark_b}")

    print(f"\n{sep}")
    print(f"  FRANKENSTEIN v11.0  |  Setup A (Baseline) vs. Setup B (+ML-Exit)  |  VERGLEICH")
    print(sep)
    print(f"  {'Metrik':<38}  {'Setup A (Baseline)':>16}  {'Setup B (Frankenstein)':>16}")
    print(f"  {line}")

    print(f"\n  ─ RENDITE {'─'*77}")
    row("Gesamtrendite",      ma["ret"],     mb["ret"],     "{:>+.2f}%")
    row("CAGR (p.a.)",        ma["cagr"],    mb["cagr"],    "{:>+.2f}%")
    row("End-Kapital (€)",    ma["end_cap"], mb["end_cap"], "{:>,.0f}€")

    print(f"\n  ─ RISIKO {'─'*78}")
    row("Max Drawdown",       ma["maxdd"],   mb["maxdd"],   "{:>.1f}%", hi="high")
    row("Sharpe Ratio",       ma["sharpe"],  mb["sharpe"],  "{:.2f}")
    row("Investitionsquote",  ma["invest_pct"], mb["invest_pct"], "{:.1f}%")

    print(f"\n  ─ TRADE-QUALITÄT {'─'*70}")
    row("Anzahl Trades",      ma["n_trades"],mb["n_trades"],"{:>}")
    row("Hit-Rate",           ma["hit"],     mb["hit"],     "{:.1f}%")
    row("Ø Gewinner",         ma["avg_win"], mb["avg_win"], "{:>+.2f}%")
    row("Ø Verlierer",        ma["avg_loss"],mb["avg_loss"],"{:>+.2f}%", hi="low")
    row("Payoff-Ratio",       ma["payoff"],  mb["payoff"],  "{:.2f}")
    row("Profit Factor",      ma["pf"],      mb["pf"],      "{:.2f}")
    row("EV / Trade",         ma["ev"],      mb["ev"],      "{:>+.2f}%")

    print(f"\n  ─ EXIT-ANALYSE {'─'*72}")
    row("ATR-Stop-Exits",     ma["n_atr"],   mb["n_atr"],   "{:>}")
    row("Rotations-Exits",    ma["n_rot"],   mb["n_rot"],   "{:>}", hi="low")
    row("Ø PnL bei Rotation", ma["rot_avg_pnl"], mb["rot_avg_pnl"], "{:>+.2f}%")
    row("Offene Positionen",  ma["n_end"],   mb["n_end"],   "{:>}")
    row("Gezahlte Gebühren",  ma["fees_total"], mb["fees_total"], "{:>,.0f}€", hi="low")

    print(f"\n  ─ ML-EXHAUSTION-STOP (nur Setup B) {'─'*51}")
    print(f"  {'ML-Exhaustion Exits':38}  {'—':>16}   {mb['exh_n']:>14}")
    print(f"  {'  davon Winners':38}  {'—':>16}   {mb['exh_wins']:>14}")
    print(f"  {'  Ø Rendite Exhaustion-Exits':38}  {'—':>16}   {mb['exh_avg_ret']:>+13.2f}%")
    print(f"  {'  Ø Tage mit Tight-Stop aktiv':38}  {'—':>16}   {mb['exh_avg_days']:>13.1f}d")
    print(f"  {'Trades mit Exhaustion-Trigger':38}  {'—':>16}   {mb['exh_triggered']:>14}")
    print(f"  {'  Ø Rendite (exh. triggered)':38}  {'—':>16}   {mb['exh_trig_ret']:>+13.2f}%")

    print(f"\n  ─ TRADE LIFECYCLE {'─'*68}")
    row("Ø Haltezeit Winners", ma["hold_w"],  mb["hold_w"],  "{:.1f}d")
    row("Ø Haltezeit Losers",  ma["hold_l"],  mb["hold_l"],  "{:.1f}d", hi="low")
    row("Earned-Mode Rate",    ma["earned_rate"], mb["earned_rate"], "{:.1f}%")
    row("Ø Rendite Earned-Mode",ma["earned_ret"],mb["earned_ret"],"{:>+.2f}%")

    print(f"\n  ─ JÄHRLICHE RENDITEN {'─'*66}")
    all_years = sorted(set(ma["annual"]) | set(mb["annual"]))
    for yr in all_years:
        ra = ma["annual"].get(yr, 0.0)
        rb = mb["annual"].get(yr, 0.0)
        row(f"  {yr}", ra, rb, "{:>+.1f}%")

    print(f"\n{sep}")

    # Kernfrage beantworten
    dd_diff  = mb["maxdd"]  - ma["maxdd"]
    ret_diff = mb["ret"]    - ma["ret"]
    print(f"\n  KERNFRAGE: Schützt der ML-Exit vor VCP-Umkehrformationen am Top?")
    print(f"  {'─' * 65}")
    dd_answer  = "Ja ✓" if dd_diff > 0 else "Nein ✗"
    ret_answer = "Ja ✓" if ret_diff > 0 else "Nein ✗"
    print(f"  Drawdown reduziert:  {dd_diff:>+.1f}%  → {dd_answer}")
    print(f"  Rendite verbessert:  {ret_diff:>+.2f}%  → {ret_answer}")
    print(f"  Exhaustion-Exits:    {mb['exh_n']} Exits  |  "
          f"{mb['exh_wins']} Winners ({mb['exh_wins']/max(mb['exh_n'],1)*100:.0f}%)  |  "
          f"Ø {mb['exh_avg_ret']:>+.2f}%")
    print()


def print_exhaustion_trades(trades_b: list[dict]) -> None:
    """Druckt alle durch ML-Exhaustion beendeten Trades."""
    exh = [t for t in trades_b if t["exit_reason"] == "Exhaust"]
    if not exh:
        print("  Keine Exhaustion-Exits gefunden.\n")
        return
    sep  = "─" * 90
    print(f"\n  ML-EXHAUSTION-EXITS  |  {len(exh)} Trades  "
          f"(Tight-Stop = min(Low_heute, Low_gestern))")
    print(f"  {sep}")
    print(f"  {'Ticker':<7}  {'Entry':>12}  {'Exit':>12}  {'Hold':>5}  "
          f"{'Return':>8}  {'Exh.Date':>12}  {'TStop-Tage':>10}  Note")
    print(f"  {sep}")
    exh_sorted = sorted(exh, key=lambda t: t["ret_%"], reverse=True)
    for t in exh_sorted:
        ed    = str(t["entry_date"])[:10]
        xd    = str(t["exit_date"])[:10]
        exhd  = str(t["exhaust_date"])[:10] if t["exhaust_date"] else "—"
        note  = "Winner ★" if t["ret_%"] > 0 else "Loser  ✗"
        earned_mark = " [Earned]" if t["earned_mode"] else ""
        print(f"  {t['ticker']:<7}  {ed:>12}  {xd:>12}  "
              f"{int(t['hold_d']):>4}d  "
              f"{t['ret_%']:>+7.2f}%  {exhd:>12}  "
              f"{int(t['exhaust_stop_days']):>10}d  "
              f"{note}{earned_mark}")
    print(f"  {sep}")
    print(f"  Ø Rendite: {float(np.mean([t['ret_%'] for t in exh])):>+.2f}%  |  "
          f"Winners: {sum(1 for t in exh if t['ret_%']>0)}/{len(exh)}  |  "
          f"Ø Halt: {float(np.mean([t['hold_d'] for t in exh])):.1f}d\n")


# ==============================================================================
# 7. CHART
# ==============================================================================

def plot_frankenstein(
    ma: dict, mb: dict,
    trades_a: list[dict], trades_b: list[dict],
    out_png: Path,
) -> None:
    eq_a = ma["_eq"]; dd_a = ma["_dd"]; pk_a = ma["_peak"]
    eq_b = mb["_eq"]; dd_b = mb["_dd"]; pk_b = mb["_peak"]

    C_A    = "#1565c0"    # Baseline: Dunkelblau
    C_B    = "#2e7d32"    # Frankenstein: Dunkelgrün
    C_EXH  = "#e65100"    # Exhaustion-Exits: Orange
    C_ROT  = "#9c27b0"    # Rotation: Lila
    C_WIN  = "#388e3c"
    C_LOSS = "#c62828"
    C_BG   = "#f8f9fa"
    C_GRID = "#dee2e6"

    fig = plt.figure(figsize=(22, 18), dpi=150, facecolor=C_BG)
    fig.suptitle(
        "Frankenstein v11.0  |  Setup A (VCP Baseline) vs. Setup B (VCP + ML-Exhaustion-Stop)",
        fontsize=13, fontweight="bold", y=0.99, color="#212121",
    )
    gs = fig.add_gridspec(
        4, 1, height_ratios=[3.5, 1.8, 1.8, 1.2],
        hspace=0.08, left=0.06, right=0.97, top=0.97, bottom=0.05,
    )

    # ── Panel 1: Dual Equity Curve ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(C_BG)

    ax1.fill_between(eq_a.index, eq_a.values, pk_a.values,
                     where=(eq_a < pk_a), color=C_A, alpha=0.12)
    ax1.fill_between(eq_b.index, eq_b.values, pk_b.values,
                     where=(eq_b < pk_b), color=C_B, alpha=0.12)

    ax1.plot(eq_a.index, eq_a.values, color=C_A, linewidth=2.1,
             label=(f"Setup A – Baseline  "
                    f"({ma['ret']:>+.1f}%  |  CAGR {ma['cagr']:>+.1f}%  |  "
                    f"DD {ma['maxdd']:.1f}%  |  Sharpe {ma['sharpe']:.2f})"))
    ax1.plot(eq_b.index, eq_b.values, color=C_B, linewidth=2.1,
             label=(f"Setup B – Frankenstein  "
                    f"({mb['ret']:>+.1f}%  |  CAGR {mb['cagr']:>+.1f}%  |  "
                    f"DD {mb['maxdd']:.1f}%  |  Sharpe {mb['sharpe']:.2f})"))
    ax1.axhline(INITIAL_CAPITAL, color="#9e9e9e", lw=0.8, ls=":",
                alpha=0.9, label=f"Startkapital ({INITIAL_CAPITAL:,.0f}€)")

    # Max-DD Annotierung
    for eq, dd, col, m in [(eq_a, dd_a, C_A, ma), (eq_b, dd_b, C_B, mb)]:
        di = dd.idxmin()
        dy = float(eq.at[di])
        ax1.annotate(
            f"DD {m['maxdd']:.1f}%",
            xy=(di, dy), xytext=(di, dy * 0.85),
            arrowprops=dict(arrowstyle="->", color=col, lw=1.1),
            fontsize=7.5, color=col, fontweight="bold", ha="center",
        )

    # End-Kapital
    for eq, col, m in [(eq_a, C_A, ma), (eq_b, C_B, mb)]:
        ax1.annotate(f"  {m['end_cap']:,.0f}€",
                     xy=(eq.index[-1], float(eq.iloc[-1])),
                     fontsize=9, color=col, fontweight="bold", va="center")

    # Exhaustion-Exit-Ereignisse als vertikale Linien (nur Setup B)
    exh_dates = sorted({t["exit_date"] for t in trades_b
                        if t["exit_reason"] == "Exhaust"})
    for dt in exh_dates:
        if dt in eq_b.index:
            ax1.axvline(dt, color=C_EXH, linewidth=0.6, alpha=0.55, linestyle="--")

    # Dummy-Handle für Exhaustion-Legende
    exh_line = mpatches.Patch(color=C_EXH, alpha=0.6,
                               label=f"ML-Exhaustion-Exit (n={mb['exh_n']})")
    ax1.set_ylabel("Kapital (€)", fontsize=9)
    ax1.tick_params(axis="x", labelbottom=False)
    ax1.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}€"))
    ax1.grid(True, color=C_GRID, lw=0.5)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles + [exh_line], labels + [exh_line.get_label()],
               loc="upper left", fontsize=8.5, framealpha=0.85)

    # Kennzahlen-Textbox
    txt = (f"{'':24} {'Baseline':>10}  {'Franken.':>10}\n"
           f"{'Hit-Rate':24} {ma['hit']:>9.1f}%  {mb['hit']:>9.1f}%\n"
           f"{'Payoff':24} {ma['payoff']:>10.2f}  {mb['payoff']:>10.2f}\n"
           f"{'Profit Factor':24} {ma['pf']:>10.2f}  {mb['pf']:>10.2f}\n"
           f"{'EV/Trade':24} {ma['ev']:>+9.2f}%  {mb['ev']:>+9.2f}%\n"
           f"{'Trades':24} {ma['n_trades']:>10}  {mb['n_trades']:>10}\n"
           f"{'Fees':24} {ma['fees_total']:>9,.0f}€  {mb['fees_total']:>9,.0f}€\n"
           f"{'Exh.-Exits':24} {'—':>10}   {mb['exh_n']:>9}")
    ax1.text(0.985, 0.97, txt, transform=ax1.transAxes,
             fontsize=7.5, va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                       edgecolor="#cccccc", alpha=0.9),
             family="monospace")

    # ── Panel 2/3: Gantt-Charts ───────────────────────────────────────────────
    def _gantt(ax, trades, title, c_col):
        ax.set_facecolor(C_BG)
        ax.set_ylim(-0.6, 2.6)
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(["Slot 2", "Slot 1"], fontsize=8)
        ax.set_ylabel(title, fontsize=8.5)
        for t in trades:
            y_bot = t["slot"] - 1
            x0    = mdates.date2num(t["entry_date"])
            x1    = mdates.date2num(t["exit_date"])
            w     = max(x1 - x0, 0.5)
            reason = t["exit_reason"]
            ret    = t["ret_%"]
            if reason == "Rotation":
                col, ec = C_ROT, "#4a148c"
            elif reason == "Exhaust":
                col, ec = C_EXH, "#bf360c"
            elif reason == "End":
                col, ec = "#607d8b", "#37474f"
            elif ret > 0:
                col, ec = c_col, "#1b5e20"
            else:
                col, ec = C_LOSS, "#b71c1c"
            rect = mpatches.FancyBboxPatch(
                (x0, y_bot + 0.08), w, 0.84,
                boxstyle="round,pad=0", facecolor=col, edgecolor=ec,
                linewidth=0.4, alpha=0.85,
            )
            ax.add_patch(rect)
            if w > 3:
                sign = "+" if ret >= 0 else ""
                ax.text((x0 + x1) / 2, y_bot + 0.5,
                        f"{t['ticker']} {sign}{ret:.0f}%",
                        ha="center", va="center",
                        fontsize=5, color="white", fontweight="bold",
                        clip_on=True)
        ax.set_xlim(
            mdates.date2num(eq_a.index[0]) - 5,
            mdates.date2num(eq_a.index[-1]) + 5,
        )
        ax.grid(True, color=C_GRID, lw=0.4, axis="x")
        ax.tick_params(axis="x", labelbottom=False)
        patches = [
            mpatches.Patch(color=c_col,  label="Winner"),
            mpatches.Patch(color=C_LOSS, label="Loser"),
            mpatches.Patch(color=C_ROT,  label="Rotation"),
        ]
        if any(t["exit_reason"] == "Exhaust" for t in trades):
            patches.append(mpatches.Patch(color=C_EXH, label="Exhaustion"))
        ax.legend(handles=patches, loc="upper left",
                  fontsize=6.5, framealpha=0.7, ncol=4)

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    _gantt(ax2, trades_a, "Setup A\n(Baseline)", C_WIN)

    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    _gantt(ax3, trades_b, "Setup B\n(Frankenstein)", C_B)

    # ── Panel 4: Jahresrenditen ───────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[3])
    ax4.set_facecolor(C_BG)
    all_years = sorted(set(ma["annual"]) | set(mb["annual"]))
    x_pos     = np.arange(len(all_years))
    w_bar     = 0.38
    bars_a    = [ma["annual"].get(yr, 0.0) for yr in all_years]
    bars_b    = [mb["annual"].get(yr, 0.0) for yr in all_years]
    col_a_yr  = [C_WIN if v >= 0 else C_LOSS for v in bars_a]
    col_b_yr  = [C_B   if v >= 0 else "#bf360c" for v in bars_b]
    ba = ax4.bar(x_pos - w_bar / 2, bars_a, w_bar, color=col_a_yr,
                 edgecolor="#424242", lw=0.5, alpha=0.85, label="Setup A")
    bb = ax4.bar(x_pos + w_bar / 2, bars_b, w_bar, color=col_b_yr,
                 edgecolor="#424242", lw=0.5, alpha=0.85, label="Setup B")
    ax4.axhline(0, color="#424242", lw=0.8)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(all_years, fontsize=8)
    ax4.set_ylabel("Jahresrendite (%)", fontsize=9)
    ax4.grid(True, color=C_GRID, lw=0.4, axis="y")
    ax4.legend(fontsize=8, loc="upper left", framealpha=0.8)
    for bars, vals in [(ba, bars_a), (bb, bars_b)]:
        for bar, val in zip(bars, vals):
            sign = "+" if val >= 0 else ""
            va   = "bottom" if val >= 0 else "top"
            ax4.text(bar.get_x() + bar.get_width() / 2,
                     val + (0.5 if val >= 0 else -0.5),
                     f"{sign}{val:.0f}%",
                     ha="center", va=va, fontsize=6.5, color="#212121")

    plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"  Chart gespeichert: {out_png}")


# ==============================================================================
# 8. MAIN
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frankenstein v11.0  |  VCP + ML-Exhaustion-Stop")
    parser.add_argument("--years", type=float, default=DEFAULT_YEARS)
    args = parser.parse_args()

    sep = "=" * 72
    print(sep)
    print("  FRANKENSTEIN v11.0  |  VCP-Entry + ML-Exhaustion-Exit")
    print(sep)
    print(f"""
  Setup A (Baseline):  VCP-Entry + reiner ATR-Stop (2.0× → 3.5×)
  Setup B (Frankenstein):  VCP-Entry + ATR-Stop + ML-Exhaustion Tight-Stop

  ML-Exhaustion-Trigger:
    RSI_14 > {EXHS_RSI:.0f}  UND  (Close − SMA50) / SMA50 > {EXHS_DIST50:.0f}%
    → Tight Trailing Stop = min(Low_heute, Low_gestern)

  Portfolio:  {INITIAL_CAPITAL:,.0f}€  |  Fee {ORDER_FEE:.0f}€  |  {MAX_POSITIONS} Slots  |  {args.years:.0f} Jahre
  Rotation:   {ROTATION_FACTOR}× + Diamond Hands (Earned-Mode geschützt)
""")

    print("[1/4] Lade Daten...")
    t0      = time.time()
    tickers = _load_tickers()
    data    = _load_ohlcv(tickers, args.years)
    print(f"  {len(data)} Ticker in {time.time()-t0:.1f}s")

    print("\n[2/4] Panels aufbauen (VCP-Entry + Exit-Indikatoren)...")
    t0     = time.time()
    panels = build_panels(data)
    dates  = panels["open"].index
    n_sig  = panels.pop("_n_sig")
    print(f"  Zeitraum:  {dates[0].date()} → {dates[-1].date()}")
    print(f"  Ticker:    {len(panels['open'].columns)}")
    print(f"  VCP-Signale (gesamt):  {n_sig:,}")
    print(f"  Aufbau:    {time.time()-t0:.1f}s")

    print("\n[3/4] Simulation Setup A (Baseline ATR)...")
    t0 = time.time()
    eq_a, trades_a, inv_a = run_backtest(panels, ml_exit=False)
    print(f"  Fertig in {time.time()-t0:.1f}s  |  {len(trades_a)} Trades")

    print("\n[4/4] Simulation Setup B (Frankenstein ML-Exit)...")
    t0 = time.time()
    eq_b, trades_b, inv_b = run_backtest(panels, ml_exit=True)
    print(f"  Fertig in {time.time()-t0:.1f}s  |  {len(trades_b)} Trades")

    print("\n[5/5] Auswertung...")
    ma = compute_metrics(eq_a, trades_a, inv_a)
    mb = compute_metrics(eq_b, trades_b, inv_b)

    print_comparison(ma, mb)
    print_exhaustion_trades(trades_b)

    # Top 5 Winner/Loser
    for label, trades in [("Setup A", trades_a), ("Setup B", trades_b)]:
        df_t = pd.DataFrame(trades).sort_values("ret_%", ascending=False)
        print(f"  TOP 5 WINNER [{label}]:")
        for _, r in df_t.head(5).iterrows():
            exh = " [Exh]" if r.get("exit_reason") == "Exhaust" else ""
            print(f"    {r['ticker']:<6}  {str(r['entry_date'])[:10]} → "
                  f"{str(r['exit_date'])[:10]}  ({int(r['hold_d'])}d)  "
                  f"{r['ret_%']:>+.1f}%"
                  + (" ★Earned" if r.get("earned_mode") else "") + exh)
        print()

    # Export
    df_a = pd.DataFrame(trades_a).assign(setup="A")
    df_b = pd.DataFrame(trades_b).assign(setup="B")
    pd.concat([df_a, df_b], ignore_index=True).to_csv(_OUT_CSV, index=False)
    print(f"  Trades exportiert: {_OUT_CSV}")

    plot_frankenstein(ma, mb, trades_a, trades_b, _OUT_PNG)
    print(f"  FERTIG.\n")


if __name__ == "__main__":
    main()
