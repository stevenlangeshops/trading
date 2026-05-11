"""
backtest_v10_ml_champion.py
====================================================================================
Portfolio-Fusion v10.0  |  ML Champion  |  Setup A vs. Setup B

Direkter Vergleich zweier kompletter Handelslogiken auf identischem Kapital und
identischen Universum / Zeitraum:

  Setup A  ─  VCP Baseline (v8.3)
    Einstieg:  Breakout_50 + BB-Squeeze(<10%) + Vol>1.5× + Close>SMA200  (T+0)
    Ausstieg:  Asymmetrischer ATR-Stop  2.0× → 3.5× (Earned Mode)

  Setup B  ─  ML Machine (Silent Climb + Exhaustion)
    Einstieg (v9.5 Rule):  T+5 nach lokalem Tief (10-Tage Rolling-Min),
                           WENN am Decision Day gilt:
                             dist_sma200  > 0.0         (Langzeit-Aufwärtstrend)
                             0.02 < atr_pct < 0.05      (Normalvolatilität)
                             Δdist_sma200 (5d) ≤ 0.02   (Kurs noch nahe SMA200)
                             Δbb_width    (5d) ≤ −0.02  (Squeeze läuft → contraction)
    Ausstieg (v9.6 Hybrid):
                           ► Weiterhin 2.0× → 3.5× ATR-Stop
                           ► Exhaustion-Check: WENN RSI>72 UND dist_sma50>0.07
                             → Trailing-Exhaustion-Stop = Tief der letzten 2 Tage
                             (ersetzt ATR-Stop falls enger)

Portfolio-Konstanten (unveränderlich):
    INITIAL_CAPITAL = 10.000 €  |  ORDER_FEE = 20 €  |  MAX_POSITIONS = 2
    ROTATION_FACTOR = 1.5×      |  Diamond Hands (Earned-Positionen unrotierbar)

Verwendung:
    python backtest_v10_ml_champion.py
    python backtest_v10_ml_champion.py --years 7
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

# ── Portfolio-Konstanten (unveränderlich für beide Setups) ────────────────────
INITIAL_CAPITAL   = 10_000.0
ORDER_FEE         = 20.0
MAX_POSITIONS     = 2
ATR_INIT          = 2.0      # ATR-Stop initial
ATR_TRAIL         = 3.5      # ATR-Stop nach Earned Mode
ROTATION_FACTOR   = 1.5      # Diamond Hands aktiv
MIN_SHARES        = 5
PYRAMID_THRESHOLD = 0.20
MAX_PYRAMIDS      = 1
DEFAULT_YEARS     = 7.0

# Setup-A-spezifisch
BB_PERIOD         = 20
BB_STD            = 2.0
BB_SQUEEZE        = 0.10
VOL_MULT          = 1.5

# Setup-B-spezifisch (ML-Filter)
ML_TROUGH_WINDOW  = 10      # Rolling-Min zur Trough-Erkennung
ML_WAIT_DAYS      = 5       # Tage Wartezeit nach Trough
ML_SMA200_MIN     = 0.0     # dist_sma200 > 0
ML_ATR_MIN        = 0.02    # atr_pct muss > 0.02 sein
ML_ATR_MAX        = 0.05    # atr_pct muss < 0.05 sein
ML_DELTA_SMA200   = 0.02    # delta_dist_sma200 (5d) <= +0.02
ML_DELTA_BB       = -0.02   # delta_bb_width (5d) <= -0.02

# Exhaustion-Exit (v9.6)
EXHS_RSI_THRESH   = 72.0    # RSI-Schwelle
EXHS_DIST50_THRESH= 0.07    # dist_sma50-Schwelle

_RAW_DIR = _here / "data" / "raw"
_OUT_PNG  = _here / "champion_ml_comparison.png"
_OUT_CSV  = _here / "ml_champion_trades.csv"


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
# 2. TECHNISCHE INDIKATOREN (geteilt)
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
# 3. PIVOT-PANEL-BUILDER
# ==============================================================================

def build_panels_a(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Setup A: VCP v8.3 – Breakout_50 + BB-Squeeze + Vol + Trend."""
    cols: dict[str, dict] = {
        k: {} for k in ["open", "close", "high", "low",
                        "atr14", "trend_str", "entry_sig"]
    }
    for ticker, df in data.items():
        c   = df["close"]
        h   = df["high"]
        vol = df.get("volume")

        sma200    = c.rolling(200).mean()
        atr14     = _atr(df, 14)
        sma_bb    = c.rolling(BB_PERIOD).mean()
        std_bb    = c.rolling(BB_PERIOD).std()
        bb_w      = (sma_bb + BB_STD * std_bb
                     - (sma_bb - BB_STD * std_bb)) / c.replace(0, np.nan)
        sma20v    = vol.rolling(20).mean() if vol is not None else None

        valid = sma200.notna() & atr14.notna()
        idx   = c[valid].index

        # Entry: Breakout_50-Transition + BB-Squeeze + Vol + Trend
        high50_prev = h.shift(1).rolling(50).max()
        b50_raw     = c > high50_prev
        trig_b50    = b50_raw & ~b50_raw.shift(1).fillna(False)
        squeeze     = bb_w.shift(1) < BB_SQUEEZE
        vol_ok      = (vol > sma20v * VOL_MULT
                       if sma20v is not None
                       else pd.Series(False, index=c.index))
        trend_ok    = c > sma200
        sig         = trig_b50 & squeeze & vol_ok & trend_ok

        cols["open"][ticker]      = df["open"][valid]
        cols["close"][ticker]     = c[valid]
        cols["high"][ticker]      = h[valid]
        cols["low"][ticker]       = df["low"][valid]
        cols["atr14"][ticker]     = atr14[valid]
        cols["trend_str"][ticker] = ((c - sma200) / sma200)[valid]
        cols["entry_sig"][ticker] = sig.reindex(idx).fillna(False).astype(bool)

    return {k: pd.DataFrame(v) for k, v in cols.items()}


def build_panels_b(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Setup B: ML Machine v9.5/v9.6.
    Entry:    T+5 nach 10-Tage-Rolling-Min + ML-Conditions
    Exhaustion-Exit: RSI14, dist_sma50 → für Simulation übergeben
    """
    cols: dict[str, dict] = {
        k: {} for k in ["open", "close", "high", "low",
                        "atr14", "trend_str", "entry_sig",
                        "rsi14", "dist_sma50", "low_2d"]
    }
    for ticker, df in data.items():
        c   = df["close"]
        lo  = df["low"]
        vol = df.get("volume")

        sma20    = c.rolling(20).mean()
        sma50    = c.rolling(50).mean()
        sma200   = c.rolling(200).mean()
        std20    = c.rolling(20).std()
        bb_up    = sma20 + BB_STD * std20
        bb_lo    = sma20 - BB_STD * std20
        bb_width = (bb_up - bb_lo) / c.replace(0, np.nan)
        atr14    = _atr(df, 14)
        rsi14    = _rsi(c, 14)

        dist_sma200 = (c - sma200) / sma200.replace(0, np.nan)
        dist_sma50  = (c - sma50)  / sma50.replace(0, np.nan)
        atr_pct     = atr14 / c.replace(0, np.nan)

        # 5-Tage-Deltas
        delta_dist_sma200 = dist_sma200 - dist_sma200.shift(ML_WAIT_DAYS)
        delta_bb_width    = bb_width    - bb_width.shift(ML_WAIT_DAYS)

        # Trough-Proxy: close ist rolling 10-Tage-Minimum (rückwärts)
        rolling_min  = c.rolling(ML_TROUGH_WINDOW).min()
        is_trough    = (c == rolling_min)

        # Entry-Signal: is_trough war vor ML_WAIT_DAYS Tagen True, und heute
        # erfüllen wir alle ML-Bedingungen
        trough_shifted = is_trough.shift(ML_WAIT_DAYS).fillna(False)
        cond_sma200    = dist_sma200 > ML_SMA200_MIN
        cond_atr       = (atr_pct > ML_ATR_MIN) & (atr_pct < ML_ATR_MAX)
        cond_delta_200 = delta_dist_sma200 <= ML_DELTA_SMA200
        cond_delta_bb  = delta_bb_width    <= ML_DELTA_BB
        valid_base     = sma200.notna() & atr14.notna() & rsi14.notna()

        sig = (trough_shifted
               & cond_sma200 & cond_atr
               & cond_delta_200 & cond_delta_bb
               & valid_base)

        valid = valid_base
        idx   = c[valid].index

        # Low der letzten 2 Tage (für Exhaustion-Stop)
        low_2d = lo.rolling(2).min().shift(1)   # gestern und vorgestern

        cols["open"][ticker]      = df["open"][valid]
        cols["close"][ticker]     = c[valid]
        cols["high"][ticker]      = df["high"][valid]
        cols["low"][ticker]       = lo[valid]
        cols["low_2d"][ticker]    = low_2d[valid]
        cols["atr14"][ticker]     = atr14[valid]
        cols["trend_str"][ticker] = dist_sma200[valid]
        cols["entry_sig"][ticker] = sig.reindex(idx).fillna(False).astype(bool)
        cols["rsi14"][ticker]     = rsi14[valid]
        cols["dist_sma50"][ticker]= dist_sma50[valid]

    return {k: pd.DataFrame(v) for k, v in cols.items()}


# ==============================================================================
# 4. SIMULATIONS-ENGINE (parametrisiert für Setup A / B)
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
    pivots:   dict[str, pd.DataFrame],
    setup_b:  bool = False,
) -> tuple[pd.Series, list[dict], int]:
    """
    Einheitliche Engine für Setup A und B.
    setup_b=True aktiviert:
      - Exhaustion-Stop (RSI + Dist.SMA50)
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
            "exhausted":        False,   # v9.6 Exhaustion-Flag
            "exhaust_date":     None,
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
            "is_rotation":    exit_reason == "Rotation",
            "exhausted":      pos["exhausted"],
            "exhaust_date":   pos["exhaust_date"],
        })
        free_slots.add(pos["slot"])
        del portfolio[ticker]
        ticker_to_slot.pop(ticker, None)

    # ── Hauptschleife ─────────────────────────────────────────────────────────
    for day_i in range(len(dates) - 1):
        today    = dates[day_i]
        tomorrow = dates[day_i + 1]

        # A. Positionen aktualisieren + Stops prüfen
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

            # ── Exhaustion-Check (nur Setup B) ────────────────────────────────
            if setup_b and not pos["exhausted"]:
                rsi_t = _safe(piv["rsi14"],    today, ticker)
                d50_t = _safe(piv["dist_sma50"], today, ticker)
                if (not math.isnan(rsi_t) and not math.isnan(d50_t)
                        and rsi_t > EXHS_RSI_THRESH
                        and d50_t > EXHS_DIST50_THRESH):
                    pos["exhausted"]    = True
                    pos["exhaust_date"] = today

            # Wenn exhausted → enger Exhaustion-Stop (Low der letzten 2 Tage)
            if setup_b and pos["exhausted"]:
                low2d = _safe(piv["low_2d"], today, ticker)
                if not math.isnan(low2d) and low2d > pos["trailing_stop"]:
                    pos["trailing_stop"] = low2d

            if tc < pos["trailing_stop"]:
                reason = "Exhaust" if (setup_b and pos["exhausted"]) else "ATR"
                exits.append((ticker, reason))

        for ticker, reason in exits:
            sell_px = _safe(piv["open"], tomorrow, ticker)
            if not math.isnan(sell_px):
                _close(ticker, tomorrow, sell_px, exit_reason=reason)

        # B. Neue Kandidaten (Diamond Hands Rotation)
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
                    # Diamond Hands: nur nicht-earned Positionen rotierbar
                    rotatable = {t: p for t, p in portfolio.items()
                                 if not p["earned_mode"]}
                    if not rotatable:
                        continue
                    weakest_t   = min(rotatable,
                                      key=lambda t: _safe(piv["trend_str"], today, t))
                    weakest_str = _safe(piv["trend_str"], today, weakest_t)
                    cand_str    = _safe(piv["trend_str"], today, cand)
                    if (not math.isnan(cand_str) and not math.isnan(weakest_str)
                            and cand_str > ROTATION_FACTOR * weakest_str):
                        rot_px = _safe(piv["open"], tomorrow, weakest_t)
                        if not math.isnan(rot_px):
                            freed = portfolio[weakest_t]["slot"]
                            _close(weakest_t, tomorrow, rot_px, "Rotation")
                            _open(cand, tomorrow, buy_px, atr_e, freed)
                            break

        # C. Pyramidisieren (für beide Setups gleich)
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
    label:       str = "",
) -> dict:
    eq    = equity.ffill().bfill()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.1)
    ret   = (eq.iloc[-1] / INITIAL_CAPITAL - 1) * 100
    cagr  = ((eq.iloc[-1] / INITIAL_CAPITAL) ** (1 / years) - 1) * 100
    peak  = eq.cummax()
    dd    = (eq - peak) / peak * 100
    dr    = eq.pct_change().dropna()
    sharpe = (dr.mean() / dr.std() * 252 ** 0.5) if dr.std() > 0 else 0

    wins   = [t for t in trades if t["ret_%"] > 0]
    losses = [t for t in trades if t["ret_%"] <= 0]
    hit    = len(wins) / len(trades) * 100 if trades else 0
    avg_w  = float(np.mean([t["ret_%"] for t in wins]))   if wins   else 0.0
    avg_l  = float(np.mean([t["ret_%"] for t in losses])) if losses else 0.0
    payoff = abs(avg_w / avg_l)  if avg_l != 0 else float("inf")
    pf     = (sum(t["ret_%"] for t in wins) /
              abs(sum(t["ret_%"] for t in losses))) if losses else float("inf")
    ev     = hit / 100 * avg_w + (1 - hit / 100) * avg_l

    # Exit-Typen
    n_atr    = sum(1 for t in trades if t["exit_reason"] == "ATR")
    n_exh    = sum(1 for t in trades if t["exit_reason"] == "Exhaust")
    n_rot    = sum(1 for t in trades if t["exit_reason"] == "Rotation")
    n_end    = sum(1 for t in trades if t["exit_reason"] == "End")
    rot_pnl  = (float(np.mean([t["ret_%"] for t in trades if t["is_rotation"]])) if n_rot else 0.0)
    exh_wins = sum(1 for t in trades if t["exit_reason"] == "Exhaust" and t["ret_%"] > 0)

    # Hold-Zeiten
    hold_w = float(np.mean([t["hold_d"] for t in wins]))   if wins   else 0.0
    hold_l = float(np.mean([t["hold_d"] for t in losses])) if losses else 0.0

    # Jährliche Renditen
    annual = {}
    for year, grp in eq.groupby(eq.index.year):
        annual[year] = round((grp.iloc[-1] / grp.iloc[0] - 1) * 100, 1)

    dd_min_idx    = dd.idxmin()
    peak_before_dd = eq[:dd_min_idx].idxmax()

    return {
        "label":        label,
        "ret":          round(ret,   2),
        "cagr":         round(cagr,  2),
        "maxdd":        round(dd.min(), 1),
        "maxdd_date":   dd_min_idx,
        "maxdd_peak":   peak_before_dd,
        "sharpe":       round(sharpe, 2),
        "n_trades":     len(trades),
        "n_atr":        n_atr,
        "n_exh":        n_exh,
        "n_rot":        n_rot,
        "n_end":        n_end,
        "fees_total":   len(trades) * ORDER_FEE * 2,
        "invest_pct":   round(invest_days / len(equity) * 100, 1),
        "hit":          round(hit,   1),
        "avg_win":      round(avg_w, 2),
        "avg_loss":     round(avg_l, 2),
        "payoff":       round(payoff, 2),
        "pf":           round(pf,    2),
        "ev":           round(ev,    2),
        "hold_w":       round(hold_w, 1),
        "hold_l":       round(hold_l, 1),
        "rot_avg_pnl":  round(rot_pnl, 2),
        "exh_wins":     exh_wins,
        "annual":       annual,
        "years":        round(years, 1),
        "end_cap":      round(eq.iloc[-1], 0),
        "_dd":          dd,
        "_eq":          eq,
        "_peak":        peak,
    }


# ==============================================================================
# 6. KONSOLEN-AUSGABE
# ==============================================================================

def print_comparison(ma: dict, mb: dict) -> None:
    sep  = "=" * 86
    line = "─" * 86

    def _row(label: str, a, b, fmt: str = "{}", hi: str = "high") -> None:
        va = fmt.format(a) if not isinstance(a, str) else a
        vb = fmt.format(b) if not isinstance(b, str) else b
        # Hervorheben welcher Wert besser ist
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            better_a = (a > b) if hi == "high" else (a < b)
            mark_a   = " ◄" if better_a  else "  "
            mark_b   = " ◄" if not better_a else "  "
        else:
            mark_a = mark_b = "  "
        print(f"  {label:<32}  {va:>14}{mark_a}  {vb:>14}{mark_b}")

    print(f"\n{sep}")
    print(f"  PORTFOLIO-FUSION v10.0  |  Setup A vs. Setup B  |  DIREKTVERGLEICH")
    print(sep)
    print(f"  {'Metrik':<32}  {'Setup A (VCP)':>16}  {'Setup B (ML)':>16}")
    print(f"  {line}")

    print(f"\n  ── RENDITE ──────────────────────────────────────────────────────────")
    _row("Gesamtrendite (%)",       ma["ret"],    mb["ret"],    "{:>+.2f}%")
    _row("CAGR (%/Jahr)",           ma["cagr"],   mb["cagr"],   "{:>+.2f}%")
    _row("End-Kapital (€)",         ma["end_cap"],mb["end_cap"],"{:>,.0f}€")

    print(f"\n  ── RISIKO ───────────────────────────────────────────────────────────")
    _row("Max Drawdown (%)",        ma["maxdd"],  mb["maxdd"],  "{:>.1f}%", hi="high")
    _row("Sharpe Ratio",            ma["sharpe"], mb["sharpe"], "{:.2f}")
    _row("Investitionsquote (%)",   ma["invest_pct"], mb["invest_pct"], "{:.1f}%")

    print(f"\n  ── TRADE-QUALITÄT ───────────────────────────────────────────────────")
    _row("Anzahl Trades",           ma["n_trades"],mb["n_trades"],"{:>}")
    _row("Hit-Rate (%)",            ma["hit"],    mb["hit"],    "{:.1f}%")
    _row("Ø Gewinner (%)",          ma["avg_win"],mb["avg_win"],"{:>+.2f}%")
    _row("Ø Verlierer (%)",         ma["avg_loss"],mb["avg_loss"],"{:>+.2f}%", hi="low")
    _row("Payoff-Ratio",            ma["payoff"], mb["payoff"], "{:.2f}")
    _row("Profit Factor",           ma["pf"],     mb["pf"],     "{:.2f}")
    _row("EV / Trade (%)",          ma["ev"],     mb["ev"],     "{:>+.2f}%")

    print(f"\n  ── EXIT-ANALYSE ─────────────────────────────────────────────────────")
    _row("ATR-Stop-Exits",          ma["n_atr"],  mb["n_atr"],  "{:>}")
    if mb["n_exh"] > 0:
        _row("Exhaustion-Exits (B)",    "—",      mb["n_exh"],  "{:>}")
        _row("  davon Winners",         "—",      mb["exh_wins"],"{:>}")
    _row("Rotations-Exits",         ma["n_rot"],  mb["n_rot"],  "{:>}", hi="low")
    _row("Ø PnL bei Rotation (%)",  ma["rot_avg_pnl"],mb["rot_avg_pnl"],"{:>+.2f}%")
    _row("Gezahlte Gebühren (€)",   ma["fees_total"],mb["fees_total"],"{:>,.0f}€", hi="low")

    print(f"\n  ── TRADE LIFECYCLE ──────────────────────────────────────────────────")
    _row("Ø Haltezeit Winners (d)", ma["hold_w"], mb["hold_w"], "{:.1f}d")
    _row("Ø Haltezeit Losers (d)",  ma["hold_l"], mb["hold_l"], "{:.1f}d", hi="low")

    print(f"\n  ── JÄHRLICHE RENDITEN ───────────────────────────────────────────────")
    all_years = sorted(set(ma["annual"]) | set(mb["annual"]))
    for yr in all_years:
        ra = ma["annual"].get(yr, 0.0)
        rb = mb["annual"].get(yr, 0.0)
        _row(f"  {yr}", ra, rb, "{:>+.1f}%")

    print(f"\n{sep}\n")


# ==============================================================================
# 7. DUAL-CHART
# ==============================================================================

def plot_comparison(
    ma: dict, mb: dict,
    trades_a: list[dict], trades_b: list[dict],
    out_png: Path,
) -> None:
    eq_a  = ma["_eq"];  dd_a = ma["_dd"];  pk_a = ma["_peak"]
    eq_b  = mb["_eq"];  dd_b = mb["_dd"];  pk_b = mb["_peak"]

    C_A    = "#1a6fc4"    # Setup A: blau
    C_B    = "#e65100"    # Setup B: orange
    C_DD_A = "#90caf9"    # Drawdown A: hell-blau
    C_DD_B = "#ffcc80"    # Drawdown B: hell-orange
    C_BG   = "#f9f9f9"
    C_GRID = "#e0e0e0"
    C_WIN  = "#2e7d32"
    C_LOSS = "#c62828"
    C_ROT  = "#9c27b0"
    C_EXH  = "#ff6f00"

    fig = plt.figure(figsize=(22, 18), dpi=150, facecolor=C_BG)
    fig.suptitle(
        "Portfolio-Fusion v10.0  |  Setup A (VCP v8.3) vs. Setup B (ML Silent Climb + Exhaustion)",
        fontsize=13, fontweight="bold", y=0.99, color="#212121",
    )
    gs = fig.add_gridspec(
        4, 1,
        height_ratios=[3.5, 1.8, 1.8, 1.2],
        hspace=0.10,
        left=0.06, right=0.97, top=0.97, bottom=0.05,
    )

    # ── Panel 1: Equity-Kurven übereinander ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(C_BG)

    # Drawdown-Flächen (transparent, hinter den Kurven)
    ax1.fill_between(eq_a.index, eq_a.values, pk_a.values,
                     where=(eq_a < pk_a), color=C_DD_A, alpha=0.35, label="_fill_a")
    ax1.fill_between(eq_b.index, eq_b.values, pk_b.values,
                     where=(eq_b < pk_b), color=C_DD_B, alpha=0.35, label="_fill_b")

    ax1.plot(eq_a.index, eq_a.values, color=C_A, linewidth=2.0,
             label=f"Setup A – VCP  ({ma['ret']:>+.1f}%  |  CAGR {ma['cagr']:>+.1f}%"
                   f"  |  Sharpe {ma['sharpe']:.2f}  |  DD {ma['maxdd']:.1f}%)")
    ax1.plot(eq_b.index, eq_b.values, color=C_B, linewidth=2.0,
             label=f"Setup B – ML   ({mb['ret']:>+.1f}%  |  CAGR {mb['cagr']:>+.1f}%"
                   f"  |  Sharpe {mb['sharpe']:.2f}  |  DD {mb['maxdd']:.1f}%)")
    ax1.axhline(INITIAL_CAPITAL, color="#9e9e9e", linewidth=0.8,
                linestyle=":", alpha=0.9, label=f"Startkapital ({INITIAL_CAPITAL:,.0f}€)")

    # End-Kapital annotieren
    for eq, col, m in [(eq_a, C_A, ma), (eq_b, C_B, mb)]:
        ax1.annotate(
            f"  {m['end_cap']:,.0f}€",
            xy=(eq.index[-1], float(eq.iloc[-1])),
            fontsize=9, color=col, fontweight="bold", va="center",
        )

    # Max-DD annotieren
    for eq, dd, col in [(eq_a, dd_a, C_A), (eq_b, dd_b, C_B)]:
        di  = dd.idxmin()
        dy  = float(eq.at[di])
        dv  = round(float(dd.min()), 1)
        ax1.annotate(
            f"DD {dv}%",
            xy=(di, dy), xytext=(di, dy * 0.87),
            arrowprops=dict(arrowstyle="->", color=col, lw=1.1),
            fontsize=7.5, color=col, fontweight="bold", ha="center",
        )

    ax1.set_ylabel("Kapital (€)", fontsize=9)
    ax1.tick_params(axis="x", labelbottom=False)
    ax1.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}€"))
    ax1.grid(True, color=C_GRID, linewidth=0.5)
    ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.85)

    # Kennzahlen-Textbox
    txt = (f"{'':26} {'VCP':>8}  {'ML':>8}\n"
           f"{'Hit-Rate':26} {ma['hit']:>7.1f}%  {mb['hit']:>7.1f}%\n"
           f"{'Payoff':26} {ma['payoff']:>8.2f}  {mb['payoff']:>8.2f}\n"
           f"{'EV/Trade':26} {ma['ev']:>+7.2f}%  {mb['ev']:>+7.2f}%\n"
           f"{'Trades':26} {ma['n_trades']:>8}  {mb['n_trades']:>8}\n"
           f"{'Fees (€)':26} {ma['fees_total']:>8,.0f}  {mb['fees_total']:>8,.0f}")
    ax1.text(
        0.985, 0.97, txt,
        transform=ax1.transAxes, fontsize=7.5, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="#cccccc", alpha=0.88),
        family="monospace",
    )

    # ── Panel 2: Gantt – Setup A ──────────────────────────────────────────────
    def _gantt(ax, trades, title, c_win, c_loss, c_rot, c_exh, eq):
        ax.set_facecolor(C_BG)
        ax.set_ylim(-0.6, 2.6)
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(["Slot 2", "Slot 1"], fontsize=8)
        ax.set_ylabel(title, fontsize=8.5)
        for t in trades:
            slot  = t["slot"]
            y_bot = slot - 1
            x0    = mdates.date2num(t["entry_date"])
            x1    = mdates.date2num(t["exit_date"])
            w     = max(x1 - x0, 0.5)
            reason = t["exit_reason"]
            ret    = t["ret_%"]
            if reason == "Rotation":
                col, ec = c_rot, "#6a1b9a"
            elif reason in ("Exhaust",):
                col, ec = c_exh, "#bf360c"
            elif reason == "End":
                col, ec = "#607d8b", "#37474f"
            elif ret > 0:
                col, ec = c_win, "#1b5e20"
            else:
                col, ec = c_loss, "#b71c1c"
            rect = mpatches.FancyBboxPatch(
                (x0, y_bot + 0.08), w, 0.84,
                boxstyle="round,pad=0", facecolor=col, edgecolor=ec,
                linewidth=0.4, alpha=0.85,
            )
            ax.add_patch(rect)
            if w > 3:
                sign  = "+" if ret >= 0 else ""
                ax.text((x0 + x1) / 2, y_bot + 0.5,
                        f"{t['ticker']} {sign}{ret:.0f}%",
                        ha="center", va="center",
                        fontsize=5, color="white", fontweight="bold", clip_on=True)
        ax.set_xlim(
            mdates.date2num(eq.index[0]) - 5,
            mdates.date2num(eq.index[-1]) + 5,
        )
        ax.grid(True, color=C_GRID, linewidth=0.4, axis="x")
        ax.tick_params(axis="x", labelbottom=False)
        patches = [
            mpatches.Patch(color=c_win,  label="Winner"),
            mpatches.Patch(color=c_loss, label="Loser"),
            mpatches.Patch(color=c_rot,  label="Rotation"),
        ]
        if any(t["exit_reason"] == "Exhaust" for t in trades):
            patches.append(mpatches.Patch(color=c_exh, label="Exhaustion"))
        ax.legend(handles=patches, loc="upper left",
                  fontsize=6.5, framealpha=0.7, ncol=4)

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    _gantt(ax2, trades_a, "Setup A\n(VCP)", C_WIN, C_LOSS, C_ROT, C_EXH, eq_a)

    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    _gantt(ax3, trades_b, "Setup B\n(ML)", C_WIN, C_LOSS, C_ROT, C_EXH, eq_b)

    # ── Panel 4: Jahresrenditen nebeneinander ─────────────────────────────────
    ax4 = fig.add_subplot(gs[3])
    ax4.set_facecolor(C_BG)
    all_years = sorted(set(ma["annual"]) | set(mb["annual"]))
    x_pos     = np.arange(len(all_years))
    w_bar     = 0.38
    bars_a    = [ma["annual"].get(yr, 0.0) for yr in all_years]
    bars_b    = [mb["annual"].get(yr, 0.0) for yr in all_years]
    col_a     = [C_WIN if v >= 0 else C_LOSS for v in bars_a]
    col_b     = [C_B   if v >= 0 else "#bf360c" for v in bars_b]
    ba = ax4.bar(x_pos - w_bar/2, bars_a, w_bar, color=col_a,
                 edgecolor="#424242", linewidth=0.5, alpha=0.85, label="Setup A")
    bb = ax4.bar(x_pos + w_bar/2, bars_b, w_bar, color=col_b,
                 edgecolor="#424242", linewidth=0.5, alpha=0.85, label="Setup B")
    ax4.axhline(0, color="#424242", linewidth=0.8)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(all_years, fontsize=8)
    ax4.set_ylabel("Jahres-\nrendite (%)", fontsize=9)
    ax4.grid(True, color=C_GRID, linewidth=0.4, axis="y")
    ax4.legend(fontsize=8, loc="upper left", framealpha=0.8)
    for bars, vals in [(ba, bars_a), (bb, bars_b)]:
        for bar, val in zip(bars, vals):
            sign = "+" if val >= 0 else ""
            va   = "bottom" if val >= 0 else "top"
            off  = 0.4 if val >= 0 else -0.4
            ax4.text(bar.get_x() + bar.get_width() / 2, val + off,
                     f"{sign}{val:.0f}%",
                     ha="center", va=va, fontsize=6, color="#212121")

    plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"  Chart gespeichert: {out_png}")


# ==============================================================================
# 8. MAIN
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Portfolio-Fusion v10.0  |  ML Champion  |  Setup A vs. B")
    parser.add_argument("--years", type=float, default=DEFAULT_YEARS)
    args = parser.parse_args()

    sep = "=" * 72
    print(sep)
    print("  PORTFOLIO-FUSION v10.0  |  ML CHAMPION  |  Setup A vs. Setup B")
    print(sep)
    print(f"""
  Setup A (VCP v8.3):
    Entry:  Breakout_50 + BB-Squeeze<{BB_SQUEEZE*100:.0f}% + Vol>{VOL_MULT}× + Close>SMA200
    Exit:   ATR {ATR_INIT}× → {ATR_TRAIL}× (Earned Mode) + Diamond Hands Rotation

  Setup B (ML v9.5 / v9.6):
    Entry:  T+5 nach {ML_TROUGH_WINDOW}d-Rolling-Min + dist_SMA200>{ML_SMA200_MIN} +
            {ML_ATR_MIN}<ATR%<{ML_ATR_MAX} + Δdist_SMA200≤{ML_DELTA_SMA200} + Δbb_width≤{ML_DELTA_BB}
    Exit:   ATR {ATR_INIT}× → {ATR_TRAIL}× + Exhaustion-Stop (RSI>{EXHS_RSI_THRESH:.0f} & dSMA50>{EXHS_DIST50_THRESH})

  Portfolio:  {INITIAL_CAPITAL:,.0f}€  |  Fee {ORDER_FEE:.0f}€/Order  |  {MAX_POSITIONS} Slots  |  {args.years:.0f} Jahre
""")

    print("[1/5] Lade Daten...")
    t0      = time.time()
    tickers = _load_tickers()
    data    = _load_ohlcv(tickers, args.years)
    print(f"  {len(data)} Ticker geladen in {time.time()-t0:.1f}s")

    print("\n[2/5] Pivot-Panels aufbauen...")
    t0      = time.time()
    piv_a   = build_panels_a(data)
    piv_b   = build_panels_b(data)
    n_sig_a = int(piv_a["entry_sig"].fillna(False).values.sum())
    n_sig_b = int(piv_b["entry_sig"].fillna(False).values.sum())
    dates   = piv_a["open"].index
    print(f"  Zeitraum:          {dates[0].date()} → {dates[-1].date()}")
    print(f"  Signale Setup A:   {n_sig_a:,}")
    print(f"  Signale Setup B:   {n_sig_b:,}")
    print(f"  Aufbau:            {time.time()-t0:.1f}s")

    print("\n[3/5] Simulation Setup A (VCP)...")
    t0 = time.time()
    eq_a, trades_a, inv_a = run_backtest(piv_a, setup_b=False)
    print(f"  Fertig in {time.time()-t0:.1f}s  |  {len(trades_a)} Trades")

    print("\n[4/5] Simulation Setup B (ML)...")
    t0 = time.time()
    eq_b, trades_b, inv_b = run_backtest(piv_b, setup_b=True)
    print(f"  Fertig in {time.time()-t0:.1f}s  |  {len(trades_b)} Trades")

    print("\n[5/5] Auswertung & Charts...")
    ma = compute_metrics(eq_a, trades_a, inv_a, label="Setup A (VCP)")
    mb = compute_metrics(eq_b, trades_b, inv_b, label="Setup B (ML)")

    print_comparison(ma, mb)

    # CSV exportieren
    df_a = pd.DataFrame(trades_a).assign(setup="A")
    df_b = pd.DataFrame(trades_b).assign(setup="B")
    pd.concat([df_a, df_b], ignore_index=True).to_csv(_OUT_CSV, index=False)
    print(f"  Trades exportiert: {_OUT_CSV}")

    # Chart
    plot_comparison(ma, mb, trades_a, trades_b, _OUT_PNG)

    # Top 5 Winners / Losers pro Setup
    for label, trades in [("Setup A", trades_a), ("Setup B", trades_b)]:
        df_t = pd.DataFrame(trades).sort_values("ret_%", ascending=False)
        print(f"\n  TOP 5 WINNER  [{label}]:")
        for _, row in df_t.head(5).iterrows():
            print(f"    {row['ticker']:<6}  {str(row['entry_date'])[:10]} → "
                  f"{str(row['exit_date'])[:10]}  ({int(row['hold_d'])}d)  "
                  f"  {row['ret_%']:>+.1f}%"
                  + (" ★Earned" if row.get("earned_mode") else ""))
        print(f"\n  TOP 5 LOSER   [{label}]:")
        for _, row in df_t.tail(5).iterrows():
            print(f"    {row['ticker']:<6}  {str(row['entry_date'])[:10]} → "
                  f"{str(row['exit_date'])[:10]}  ({int(row['hold_d'])}d)  "
                  f"  {row['ret_%']:>+.1f}%")

    print(f"\n  FERTIG.\n")


if __name__ == "__main__":
    main()
