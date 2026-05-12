"""
backtest_v13_regime_filter.py
====================================================================================
Macro Regime Filter & Crash Defense v13.0

Testet die These: Ein Marktbreiten-Filter (% Aktien > SMA50) kann die großen
Crash-Drawdowns (-45% Corona, -29% 2022) deutlich reduzieren, ohne die
Bullenmarkt-Renditen zu zerstören.

Setup A  ─  Naive Baseline (VCP Champion v8.3)
    BREADTH_THRESHOLD = 0.0  (kauft immer, ignoriert den Markt)
    Exit: ATR 2.0x -> 3.5x

Setup B  -  Regime-Verteidiger (Macro Filter)
    Marktbreite = Anteil der Aktien in unserem Universum mit Close > SMA50

    GRUEN  breadth >= 0.40  -> Normalbetrieb (Kauefe erlaubt)
    ROT    breadth <  0.40  -> Krisenmodus

    Regel 1 - Kaufstopp:  Neue VCP-Positionen nur bei GRUEN
    Regel 2 - Fluchtreflex:  Wenn Ampel auf ROT springt,
              werden ALLE offenen Positionen sofort auf einen
              engen Krisen-Stop gesetzt:
              Stop = max(aktueller_Stop, max_high - 1.5x ATR)
              (enger als 2.0x -> fuehrt schnell zu Cash)

Visualisierung:
    Roter/grauer Hintergrund im Chart an allen ROT-Tagen,
    sichtbar wann und wie lange das System im Bunker sass.

Verwendung:
    python backtest_v13_regime_filter.py
    python backtest_v13_regime_filter.py --years 7 --breadth 0.40
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import warnings

# Windows-Terminal UTF-8 erzwingen
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
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
INITIAL_CAPITAL     = 10_000.0
ORDER_FEE           = 20.0
MAX_POSITIONS       = 2
ATR_INIT            = 2.0       # Stop initial
ATR_TRAIL           = 3.5       # Stop earned mode
ATR_CRISIS          = 1.5       # Krisen-Stop (enger, zieht schnell zu Cash)
ROTATION_FACTOR     = 1.5
MIN_SHARES          = 5
PYRAMID_THRESHOLD   = 0.20
MAX_PYRAMIDS        = 1
DEFAULT_YEARS       = 7.0
DEFAULT_BREADTH     = 0.40      # Schwelle GRÜN/ROT

BB_PERIOD   = 20
BB_STD      = 2.0
BB_SQUEEZE  = 0.10
VOL_MULT    = 1.5

_RAW_DIR = _here / "data" / "raw"
_OUT_PNG  = _here / "regime_comparison.png"
_OUT_CSV  = _here / "regime_trades.csv"


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
# 2. MARKTBREITE + PANELS
# ==============================================================================

def build_panels(
    data: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame | pd.Series]:
    """
    Baut VCP-Entry-Panel + Marktbreiten-Zeitreihe.
    Marktbreite = täglicher Anteil Aktien mit Close > SMA50.
    """
    cols: dict[str, dict] = {
        k: {} for k in [
            "open", "close", "high", "low",
            "atr14", "trend_str", "entry_sig",
        ]
    }
    # Für Marktbreite: Close > SMA50 je Ticker
    above_sma50: dict[str, pd.Series] = {}
    n_sig_total = 0

    for ticker, df in data.items():
        c   = df["close"]
        h   = df["high"]
        vol = df.get("volume")

        sma50  = c.rolling(50).mean()
        sma200 = c.rolling(200).mean()
        sma20  = c.rolling(BB_PERIOD).mean()
        std20  = c.rolling(BB_PERIOD).std()
        bb_w   = (sma20 + BB_STD * std20
                  - (sma20 - BB_STD * std20)) / c.replace(0, np.nan)
        atr14  = _atr(df, 14)
        sma20v = vol.rolling(20).mean() if vol is not None else None

        valid = sma200.notna() & atr14.notna()
        idx   = c[valid].index

        # Marktbreite: Close > SMA50 (nur wo SMA50 verfügbar)
        above_sma50[ticker] = (c > sma50).where(sma50.notna()).astype(float)

        # VCP Entry
        high50_prev = h.shift(1).rolling(50).max()
        b50_raw     = c > high50_prev
        trig_b50    = b50_raw & ~b50_raw.shift(1).fillna(False)
        squeeze     = bb_w.shift(1) < BB_SQUEEZE
        vol_ok      = (vol > sma20v * VOL_MULT
                       if sma20v is not None
                       else pd.Series(False, index=c.index))
        trend_ok    = c > sma200
        sig         = trig_b50 & squeeze & vol_ok & trend_ok

        n_sig_total += int(sig[valid].fillna(False).sum())

        cols["open"][ticker]      = df["open"][valid]
        cols["close"][ticker]     = c[valid]
        cols["high"][ticker]      = h[valid]
        cols["low"][ticker]       = df["low"][valid]
        cols["atr14"][ticker]     = atr14[valid]
        cols["trend_str"][ticker] = ((c - sma200) / sma200)[valid]
        cols["entry_sig"][ticker] = sig.reindex(idx).fillna(False).astype(bool)

    panels = {k: pd.DataFrame(v) for k, v in cols.items()}

    # Marktbreite: Mittelwert über alle Ticker pro Tag (NaN = fehlende Daten)
    breadth_df    = pd.DataFrame(above_sma50)
    breadth_daily = breadth_df.mean(axis=1, skipna=True)

    # Auf den gemeinsamen Index alignen
    common_idx     = panels["open"].index
    breadth_aligned = breadth_daily.reindex(common_idx).ffill()

    panels["breadth"]   = breadth_aligned      # pd.Series
    panels["_n_sig"]    = n_sig_total
    return panels


# ==============================================================================
# 3. SIMULATIONS-ENGINE
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
    pivots:          dict[str, pd.DataFrame | pd.Series],
    use_regime:      bool  = False,
    breadth_thresh:  float = DEFAULT_BREADTH,
) -> tuple[pd.Series, list[dict], int, pd.Series]:
    """
    use_regime=False -> Setup A: Baseline ohne Filter
    use_regime=True  -> Setup B: Gruen/Rot Ampel mit Kaufstopp + Fluchtreflex
    """
    piv      = {k: v for k, v in pivots.items() if k not in ("breadth", "_n_sig")}
    breadth  = pivots.get("breadth", pd.Series(dtype=float))
    dates    = piv["open"].index
    tickers  = list(piv["open"].columns)

    cash            = INITIAL_CAPITAL
    portfolio:      dict[str, dict] = {}
    free_slots      = {1, 2}
    ticker_to_slot: dict[str, int]  = {}
    completed:      list[dict]      = []
    equity_log:     dict            = {}
    regime_log:     dict            = {}   # für Plot: 0=GRÜN, 1=ROT
    invest_days     = 0

    def _safe(df, date, ticker) -> float:
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
            "max_unreal_pct":  0.0,
            "crisis_mode":     False,   # Wurde Fluchtreflex ausgelöst?
            "crisis_date":     None,
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
            "ticker":        ticker,
            "slot":          pos["slot"],
            "entry_date":    pos["entry_date"],
            "exit_date":     sell_date,
            "entry_price":   round(pos["entry_price"], 2),
            "exit_price":    round(sell_px, 2),
            "shares":        pos["shares"],
            "pnl_€":         round(pnl, 2),
            "ret_%":         round(ret_pct, 2),
            "hold_d":        pos["days_held"],
            "earned_mode":   pos["earned_mode"],
            "earned_date":   pos["earned_date"],
            "pyramid_count": pos["pyramid_count"],
            "max_unreal_%":  round(pos["max_unreal_pct"], 2),
            "exit_reason":   exit_reason,
            "is_rotation":   exit_reason == "Rotation",
            "crisis_exit":   exit_reason == "Crisis",
            "crisis_date":   pos["crisis_date"],
        })
        free_slots.add(pos["slot"])
        del portfolio[ticker]
        ticker_to_slot.pop(ticker, None)

    # ── Hauptschleife ─────────────────────────────────────────────────────────
    for day_i in range(len(dates) - 1):
        today    = dates[day_i]
        tomorrow = dates[day_i + 1]

        # Regime ermitteln
        br = float(breadth.at[today]) if today in breadth.index else float("nan")
        if math.isnan(br):
            br = breadth_thresh  # Fallback: neutral
        regime_green = (br >= breadth_thresh) if use_regime else True
        regime_log[today] = 0 if regime_green else 1

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

            # Fluchtreflex (Regel 2): ROT-Ampel -> Krisen-Stop
            if use_regime and not regime_green and not math.isnan(ta) and ta > 0:
                crisis_stop = pos["max_high"] - ATR_CRISIS * ta
                if crisis_stop > pos["trailing_stop"]:
                    pos["trailing_stop"] = crisis_stop
                if not pos["crisis_mode"]:
                    pos["crisis_mode"] = True
                    pos["crisis_date"] = today

            # Stop prüfen
            if tc < pos["trailing_stop"]:
                reason = "Crisis" if pos["crisis_mode"] else "ATR"
                exits.append((ticker, reason))

        for ticker, reason in exits:
            sell_px = _safe(piv["open"], tomorrow, ticker)
            if not math.isnan(sell_px):
                _close(ticker, tomorrow, sell_px, exit_reason=reason)

        # ── Neue Kandidaten (nur bei GRÜN) ────────────────────────────────────
        if regime_green or not use_regime:
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
                        w_str = _safe(piv["trend_str"], today, weakest_t)
                        c_str = _safe(piv["trend_str"], today, cand)
                        if (not math.isnan(c_str) and not math.isnan(w_str)
                                and c_str > ROTATION_FACTOR * w_str):
                            rot_px = _safe(piv["open"], tomorrow, weakest_t)
                            if not math.isnan(rot_px):
                                freed = portfolio[weakest_t]["slot"]
                                _close(weakest_t, tomorrow, rot_px, "Rotation")
                                _open(cand, tomorrow, buy_px, atr_e, freed)
                                break

        # ── Pyramidisieren (nur bei GRÜN) ─────────────────────────────────────
        if (regime_green or not use_regime) and len(portfolio) == 1 and free_slots:
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
            _close(ticker, dates[-1], lp, "End")

    eq_series  = pd.Series(equity_log).sort_index()
    reg_series = pd.Series(regime_log).sort_index()
    return eq_series, completed, invest_days, reg_series


# ==============================================================================
# 4. METRIKEN
# ==============================================================================

def compute_metrics(
    equity:      pd.Series,
    trades:      list[dict],
    invest_days: int,
    regime:      pd.Series,
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

    crisis_t = [t for t in trades if t.get("crisis_exit")]
    n_crisis  = len(crisis_t)
    crisis_ret = float(np.mean([t["ret_%"] for t in crisis_t])) if crisis_t else 0.0
    crisis_w   = sum(1 for t in crisis_t if t["ret_%"] > 0)

    n_atr = sum(1 for t in trades if t["exit_reason"] == "ATR")
    n_rot = sum(1 for t in trades if t["exit_reason"] == "Rotation")
    n_end = sum(1 for t in trades if t["exit_reason"] == "End")
    rot_t = [t for t in trades if t.get("is_rotation")]
    rot_pnl = float(np.mean([t["ret_%"] for t in rot_t])) if rot_t else 0.0

    hold_w = float(np.mean([t["hold_d"] for t in wins]))   if wins   else 0.0
    hold_l = float(np.mean([t["hold_d"] for t in losses])) if losses else 0.0

    earned_t    = [t for t in trades if t["earned_mode"]]
    earned_rate = len(earned_t) / len(trades) * 100 if trades else 0.0
    earned_ret  = float(np.mean([t["ret_%"] for t in earned_t])) if earned_t else 0.0

    # Regime-Statistiken
    red_days   = int((regime == 1).sum()) if len(regime) > 0 else 0
    total_days = len(regime)
    red_pct    = red_days / total_days * 100 if total_days > 0 else 0.0

    # Max Drawdown während roter Phase vs. grüner Phase
    red_dates  = regime[regime == 1].index
    green_dates = regime[regime == 0].index
    dd_red   = float(dd[dd.index.isin(red_dates)].min()) if len(red_dates)>0 else 0.0
    dd_green = float(dd[dd.index.isin(green_dates)].min()) if len(green_dates)>0 else 0.0

    # Jährliche Renditen
    annual = {}
    for year, grp in eq.groupby(eq.index.year):
        annual[year] = round((grp.iloc[-1] / grp.iloc[0] - 1) * 100, 1)

    dd_min_idx  = dd.idxmin()
    peak_before = eq[:dd_min_idx].idxmax()

    return {
        "ret":         round(ret,    2),
        "cagr":        round(cagr,   2),
        "maxdd":       round(dd.min(), 1),
        "maxdd_date":  dd_min_idx,
        "maxdd_peak":  peak_before,
        "sharpe":      round(sharpe, 2),
        "n_trades":    len(trades),
        "n_atr":       n_atr,
        "n_crisis":    n_crisis,
        "n_rot":       n_rot,
        "n_end":       n_end,
        "fees_total":  len(trades) * ORDER_FEE * 2,
        "invest_pct":  round(invest_days / len(equity) * 100, 1),
        "hit":         round(hit,    1),
        "avg_win":     round(avg_w,  2),
        "avg_loss":    round(avg_l,  2),
        "payoff":      round(payoff, 2),
        "pf":          round(pf,     2),
        "ev":          round(ev,     2),
        "hold_w":      round(hold_w, 1),
        "hold_l":      round(hold_l, 1),
        "rot_avg_pnl": round(rot_pnl, 2),
        "earned_rate": round(earned_rate, 1),
        "earned_ret":  round(earned_ret,  2),
        # Krisen-Stats
        "crisis_n":    n_crisis,
        "crisis_w":    crisis_w,
        "crisis_ret":  round(crisis_ret, 2),
        # Regime-Stats
        "red_days":    red_days,
        "red_pct":     round(red_pct, 1),
        "dd_red":      round(dd_red,  1),
        "dd_green":    round(dd_green, 1),
        "annual":      annual,
        "years":       round(years, 1),
        "end_cap":     round(eq.iloc[-1], 0),
        "_dd":         dd,
        "_eq":         eq,
        "_peak":       peak,
        "_regime":     regime,
    }


# ==============================================================================
# 5. AUSGABE
# ==============================================================================

def print_comparison(ma: dict, mb: dict, breadth_thresh: float) -> None:
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
    print(f"  REGIME FILTER v13.0  |  Setup A (Baseline) vs. Setup B (Regime-Verteidiger)")
    print(f"  Breadth-Schwelle: {breadth_thresh*100:.0f}% (< Schwelle = ROT, >= Schwelle = GRUEN)")
    print(sep)
    print(f"  {'Metrik':<42}  {'Setup A (Baseline)':>16}  {'Setup B (Regime)':>16}")
    print(f"  {line}")

    print(f"\n  ─ RENDITE {'─'*79}")
    row("Gesamtrendite",       ma["ret"],    mb["ret"],    "{:>+.2f}%")
    row("CAGR (p.a.)",         ma["cagr"],   mb["cagr"],   "{:>+.2f}%")
    row("End-Kapital (€)",     ma["end_cap"],mb["end_cap"],"{:>,.0f}€")

    print(f"\n  ─ RISIKO {'─'*80}")
    row("Max Drawdown",        ma["maxdd"],  mb["maxdd"],  "{:>.1f}%", hi="high")
    row("Sharpe Ratio",        ma["sharpe"], mb["sharpe"],  "{:.2f}")
    row("Investitionsquote",   ma["invest_pct"],mb["invest_pct"],"{:.1f}%")

    print(f"\n  ─ TRADE-QUALITÄT {'─'*72}")
    row("Anzahl Trades",       ma["n_trades"],mb["n_trades"],"{:>}")
    row("Hit-Rate",            ma["hit"],    mb["hit"],    "{:.1f}%")
    row("Ø Gewinner",          ma["avg_win"],mb["avg_win"],"{:>+.2f}%")
    row("Ø Verlierer",         ma["avg_loss"],mb["avg_loss"],"{:>+.2f}%", hi="low")
    row("Payoff-Ratio",        ma["payoff"], mb["payoff"], "{:.2f}")
    row("Profit Factor",       ma["pf"],     mb["pf"],     "{:.2f}")
    row("EV / Trade",          ma["ev"],     mb["ev"],     "{:>+.2f}%")

    print(f"\n  ─ EXIT-ANALYSE {'─'*74}")
    row("ATR-Exits",           ma["n_atr"],  mb["n_atr"],  "{:>}")
    row("Krisen-Exits (Flucht)",ma["n_crisis"],mb["n_crisis"],"{:>}")
    row("Rotations-Exits",     ma["n_rot"],  mb["n_rot"],  "{:>}", hi="low")
    row("Ø PnL Rotation",      ma["rot_avg_pnl"],mb["rot_avg_pnl"],"{:>+.2f}%")
    row("Gezahlte Gebühren",   ma["fees_total"],mb["fees_total"],"{:>,.0f}€", hi="low")

    print(f"\n  ─ KRISEN-EXIT DETAIL (nur Setup B) {'─'*53}")
    print(f"  {'Krisen-Exits (Fluchtreflex)':42}  {'—':>16}   {mb['crisis_n']:>14}")
    print(f"  {'  davon Winners':42}  {'—':>16}   {mb['crisis_w']:>14}")
    print(f"  {'  Hit-Rate':42}  {'—':>16}   "
          f"{mb['crisis_w']/max(mb['crisis_n'],1)*100:>13.1f}%")
    print(f"  {'  Ø Rendite Krisen-Exits':42}  {'—':>16}   {mb['crisis_ret']:>+13.2f}%")

    print(f"\n  ─ REGIME-STATISTIK (Setup B) {'─'*60}")
    print(f"  {'Rote Tage (ROT-Phase gesamt)':42}  {'—':>16}   {mb['red_days']:>13}d")
    print(f"  {'Anteil ROT-Tage':42}  {'—':>16}   {mb['red_pct']:>13.1f}%")
    print(f"  {'Max DD während ROT-Phasen':42}  {'—':>16}   {mb['dd_red']:>13.1f}%")
    print(f"  {'Max DD während GRÜN-Phasen':42}  {'—':>16}   {mb['dd_green']:>13.1f}%")

    print(f"\n  ─ TRADE LIFECYCLE {'─'*71}")
    row("Ø Haltezeit Winners", ma["hold_w"],    mb["hold_w"],    "{:.1f}d")
    row("Ø Haltezeit Losers",  ma["hold_l"],    mb["hold_l"],    "{:.1f}d", hi="low")
    row("Earned-Mode Rate",    ma["earned_rate"],mb["earned_rate"],"{:.1f}%")

    print(f"\n  ─ JÄHRLICHE RENDITEN {'─'*68}")
    all_years = sorted(set(ma["annual"]) | set(mb["annual"]))
    for yr in all_years:
        row(f"  {yr}", ma["annual"].get(yr,0.0), mb["annual"].get(yr,0.0), "{:>+.1f}%")

    print(f"\n{sep}")

    # Kernfrage
    ddd = mb["maxdd"] - ma["maxdd"]
    dr  = mb["ret"]   - ma["ret"]
    print(f"""
  KERNFRAGE: Max-DD unter -25%? Rendite erhalten?
  {'─' * 65}
  Max Drawdown:  {mb['maxdd']:.1f}%  {'✓ unter -25%!' if mb['maxdd'] > -25 else '✗ noch über -25%'}
  Δ Drawdown:    {ddd:>+.1f}%  {'◄ verbessert' if ddd > 0 else ''}
  Gesamtrendite: {mb['ret']:>+.2f}%  (Baseline: {ma['ret']:>+.2f}%,  Δ {dr:>+.2f}%)
  Rote Tage:     {mb['red_days']}d ({mb['red_pct']:.1f}%)  -> System sass in Cash
  Krisen-Exits:  {mb['crisis_n']} ({mb['crisis_ret']:>+.2f}% Ø)  |  Hit-Rate {mb['crisis_w']/max(mb['crisis_n'],1)*100:.0f}%
""")


# ==============================================================================
# 6. CHART
# ==============================================================================

def plot_comparison(
    ma: dict, mb: dict,
    trades_a: list[dict], trades_b: list[dict],
    breadth:  pd.Series,
    breadth_thresh: float,
    out_png:  Path,
) -> None:
    eq_a = ma["_eq"]; dd_a = ma["_dd"]; pk_a = ma["_peak"]
    eq_b = mb["_eq"]; dd_b = mb["_dd"]; pk_b = mb["_peak"]
    regime_b = mb["_regime"]

    C_A    = "#1565c0"
    C_B    = "#2e7d32"
    C_ROT  = "#e53935"
    C_CRI  = "#ff8f00"
    C_WIN  = "#388e3c"
    C_LOS  = "#c62828"
    C_ROT2 = "#9c27b0"
    C_BG   = "#f8f9fa"
    C_GRD  = "#dee2e6"

    fig = plt.figure(figsize=(22, 20), dpi=150, facecolor=C_BG)
    fig.suptitle(
        f"Regime Filter v13.0  |  Setup A (Baseline) vs. Setup B  "
            f"(VCP + Marktbreiten-Filter >= {breadth_thresh*100:.0f}%)",
        fontsize=13, fontweight="bold", y=0.99, color="#212121",
    )
    gs = fig.add_gridspec(
        5, 1, height_ratios=[3.5, 1.2, 1.8, 1.8, 1.2],
        hspace=0.08, left=0.06, right=0.97, top=0.97, bottom=0.04,
    )

    # ── Panel 1: Equity-Kurven + ROT-Hintergrund ──────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(C_BG)

    # Rote Hintergrund-Zonen (ROT-Phasen)
    red_mask = (regime_b == 1)
    if red_mask.any():
        in_red = False
        start_red = None
        for dt, is_red in red_mask.items():
            if is_red and not in_red:
                start_red = dt; in_red = True
            elif not is_red and in_red:
                ax1.axvspan(start_red, dt, color=C_ROT, alpha=0.08, lw=0)
                in_red = False
        if in_red and start_red is not None:
            ax1.axvspan(start_red, red_mask.index[-1], color=C_ROT, alpha=0.08, lw=0)

    ax1.fill_between(eq_a.index, eq_a, pk_a, where=(eq_a < pk_a), color=C_A, alpha=0.10)
    ax1.fill_between(eq_b.index, eq_b, pk_b, where=(eq_b < pk_b), color=C_B, alpha=0.10)
    ax1.plot(eq_a.index, eq_a, color=C_A, lw=2.1,
             label=(f"Setup A – Baseline  "
                    f"({ma['ret']:>+.1f}%  DD {ma['maxdd']:.1f}%  "
                    f"Sharpe {ma['sharpe']:.2f})"))
    ax1.plot(eq_b.index, eq_b, color=C_B, lw=2.1,
             label=(f"Setup B – Regime  "
                    f"({mb['ret']:>+.1f}%  DD {mb['maxdd']:.1f}%  "
                    f"Sharpe {mb['sharpe']:.2f})"))
    ax1.axhline(INITIAL_CAPITAL, color="#9e9e9e", lw=0.8, ls=":", alpha=0.9,
                label=f"Start ({INITIAL_CAPITAL:,.0f}€)")

    for eq, dd, col, m in [(eq_a, dd_a, C_A, ma), (eq_b, dd_b, C_B, mb)]:
        di = dd.idxmin(); dy = float(eq.at[di])
        ax1.annotate(f"DD {m['maxdd']:.1f}%",
                     xy=(di, dy), xytext=(di, dy * 0.83),
                     arrowprops=dict(arrowstyle="->", color=col, lw=1.1),
                     fontsize=8, color=col, fontweight="bold", ha="center")

    for eq, col, m in [(eq_a, C_A, ma), (eq_b, C_B, mb)]:
        ax1.annotate(f"  {m['end_cap']:,.0f}€",
                     xy=(eq.index[-1], float(eq.iloc[-1])),
                     fontsize=9, color=col, fontweight="bold", va="center")

    red_patch = mpatches.Patch(color=C_ROT, alpha=0.3, label=f"ROT-Phase (Breadth < {breadth_thresh*100:.0f}%)")
    ax1.set_ylabel("Kapital (€)", fontsize=9)
    ax1.tick_params(axis="x", labelbottom=False)
    ax1.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}€"))
    ax1.grid(True, color=C_GRD, lw=0.5)
    h, l = ax1.get_legend_handles_labels()
    ax1.legend(h + [red_patch], l + [red_patch.get_label()],
               loc="upper left", fontsize=8.5, framealpha=0.85)

    txt = (f"{'':26} {'Baseline':>9}  {'Regime':>9}\n"
           f"{'Trades':26} {ma['n_trades']:>9}  {mb['n_trades']:>9}\n"
           f"{'Hit-Rate':26} {ma['hit']:>8.1f}%  {mb['hit']:>8.1f}%\n"
           f"{'Payoff':26} {ma['payoff']:>9.2f}  {mb['payoff']:>9.2f}\n"
           f"{'PF':26} {ma['pf']:>9.2f}  {mb['pf']:>9.2f}\n"
           f"{'EV/Trade':26} {ma['ev']:>+8.2f}%  {mb['ev']:>+8.2f}%\n"
           f"{'Fees':26} {ma['fees_total']:>8,.0f}€  {mb['fees_total']:>8,.0f}€\n"
           f"{'Rote Tage':26} {'—':>9}   {mb['red_days']:>8}d\n"
           f"{'Krisen-Exits':26} {'—':>9}   {mb['crisis_n']:>9}")
    ax1.text(0.985, 0.97, txt, transform=ax1.transAxes,
             fontsize=7.5, va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                       edgecolor="#cccccc", alpha=0.9),
             family="monospace")

    # ── Panel 2: Marktbreite ──────────────────────────────────────────────────
    ax_br = fig.add_subplot(gs[1], sharex=ax1)
    ax_br.set_facecolor(C_BG)
    br_aligned = breadth.reindex(eq_a.index).ffill()
    ax_br.fill_between(br_aligned.index, br_aligned * 100, breadth_thresh * 100,
                        where=(br_aligned < breadth_thresh),
                        color=C_ROT, alpha=0.45, label="ROT")
    ax_br.fill_between(br_aligned.index, br_aligned * 100, breadth_thresh * 100,
                        where=(br_aligned >= breadth_thresh),
                        color=C_B,   alpha=0.25, label="GRÜN")
    ax_br.plot(br_aligned.index, br_aligned * 100, color="#424242", lw=0.9, alpha=0.8)
    ax_br.axhline(breadth_thresh * 100, color=C_ROT, lw=1.2, ls="--",
                   label=f"Schwelle {breadth_thresh*100:.0f}%")
    ax_br.set_ylabel("Markt-\nbreite (%)", fontsize=8.5)
    ax_br.set_ylim(0, 100)
    ax_br.tick_params(axis="x", labelbottom=False)
    ax_br.grid(True, color=C_GRD, lw=0.4)
    ax_br.legend(loc="upper left", fontsize=7, framealpha=0.8, ncol=3)

    # ── Panel 3/4: Gantt ──────────────────────────────────────────────────────
    def _gantt(ax, trades, title, c_win):
        ax.set_facecolor(C_BG)
        ax.set_ylim(-0.6, 2.6)
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(["Slot 2", "Slot 1"], fontsize=8)
        ax.set_ylabel(title, fontsize=8.5)
        # ROT-Hintergrund auch im Gantt
        if red_mask.any():
            in_red = False; start_red = None
            for dt, is_red in red_mask.items():
                if is_red and not in_red:
                    start_red = dt; in_red = True
                elif not is_red and in_red:
                    ax.axvspan(start_red, dt, color=C_ROT, alpha=0.06, lw=0)
                    in_red = False
            if in_red and start_red is not None:
                ax.axvspan(start_red, red_mask.index[-1], color=C_ROT, alpha=0.06, lw=0)
        for t in trades:
            y_bot = t["slot"] - 1
            x0 = mdates.date2num(t["entry_date"])
            x1 = mdates.date2num(t["exit_date"])
            w  = max(x1 - x0, 0.5)
            r  = t["exit_reason"]
            ret = t["ret_%"]
            if r == "Rotation":
                col, ec = C_ROT2, "#4a148c"
            elif r == "Crisis":
                col, ec = C_CRI, "#e65100"
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
        ax.set_xlim(mdates.date2num(eq_a.index[0]) - 5,
                    mdates.date2num(eq_a.index[-1]) + 5)
        ax.grid(True, color=C_GRD, lw=0.4, axis="x")
        ax.tick_params(axis="x", labelbottom=False)
        patches = [mpatches.Patch(color=c_win,  label="Winner"),
                   mpatches.Patch(color=C_LOS,  label="Loser"),
                   mpatches.Patch(color=C_CRI,  label="Krisen-Exit"),
                   mpatches.Patch(color=C_ROT2, label="Rotation")]
        ax.legend(handles=patches, loc="upper left",
                  fontsize=6.5, framealpha=0.7, ncol=4)

    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    _gantt(ax3, trades_a, "Setup A\n(Baseline)", C_WIN)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    _gantt(ax4, trades_b, "Setup B\n(Regime)", C_B)

    # ── Panel 5: Jahresrenditen ───────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[4])
    ax5.set_facecolor(C_BG)
    all_years = sorted(set(ma["annual"]) | set(mb["annual"]))
    x = np.arange(len(all_years)); w_bar = 0.38
    ba_v = [ma["annual"].get(yr, 0.0) for yr in all_years]
    bb_v = [mb["annual"].get(yr, 0.0) for yr in all_years]
    ba = ax5.bar(x - w_bar/2, ba_v, w_bar,
                 color=[C_WIN if v>=0 else C_LOS for v in ba_v],
                 edgecolor="#424242", lw=0.5, alpha=0.85, label="A Baseline")
    bb = ax5.bar(x + w_bar/2, bb_v, w_bar,
                 color=[C_B   if v>=0 else "#bf360c" for v in bb_v],
                 edgecolor="#424242", lw=0.5, alpha=0.85, label="B Regime")
    ax5.axhline(0, color="#424242", lw=0.8)
    ax5.set_xticks(x); ax5.set_xticklabels(all_years, fontsize=8)
    ax5.set_ylabel("Jahresrendite (%)", fontsize=9)
    ax5.grid(True, color=C_GRD, lw=0.4, axis="y")
    ax5.legend(fontsize=8, loc="upper left", framealpha=0.8)
    for bars, vals in [(ba, ba_v), (bb, bb_v)]:
        for bar, val in zip(bars, vals):
            sign = "+" if val >= 0 else ""
            va = "bottom" if val >= 0 else "top"
            ax5.text(bar.get_x() + bar.get_width()/2, val + (0.5 if val>=0 else -0.5),
                     f"{sign}{val:.0f}%", ha="center", va=va,
                     fontsize=6.5, color="#212121")

    plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"  Chart gespeichert: {out_png}")


# ==============================================================================
# 7. MAIN
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regime Filter v13.0  |  Macro Breadth + Crash Defense")
    parser.add_argument("--years",    type=float, default=DEFAULT_YEARS)
    parser.add_argument("--breadth",  type=float, default=DEFAULT_BREADTH,
                        help=f"Breadth-Schwelle GRÜN (default {DEFAULT_BREADTH})")
    args = parser.parse_args()

    sep = "=" * 72
    print(sep)
    print("  REGIME FILTER v13.0  |  Macro Breadth + Crash Defense")
    print(sep)
    print(f"""
  Marktbreite = Anteil Aktien mit Close > SMA50 im 260er-Universum

  GRUEN  breadth >= {args.breadth:.0%}  -> Normalbetrieb
  ROT    breadth <  {args.breadth:.0%}  -> Kaufstopp + Fluchtreflex (Stop -> 1.5x ATR)

  Portfolio:  {INITIAL_CAPITAL:,.0f}€  |  Fee {ORDER_FEE:.0f}€  |  {MAX_POSITIONS} Slots  |  {args.years:.0f} Jahre
""")

    print("[1/4] Lade Daten...")
    t0      = time.time()
    tickers = _load_tickers()
    data    = _load_ohlcv(tickers, args.years)
    print(f"  {len(data)} Ticker in {time.time()-t0:.1f}s")

    print("\n[2/4] Panels + Marktbreite berechnen...")
    t0     = time.time()
    panels = build_panels(data)
    dates  = panels["open"].index
    n_sig  = panels.pop("_n_sig")
    breadth = panels["breadth"]
    red_total = int((breadth < args.breadth).sum())
    print(f"  Zeitraum:    {dates[0].date()} -> {dates[-1].date()}")
    print(f"  Signale:     {n_sig:,}")
    print(f"  ROT-Tage:    {red_total:,} von {len(dates):,} "
          f"({red_total/len(dates)*100:.1f}%)")
    print(f"  Aufbau:      {time.time()-t0:.1f}s")

    # Kurze Breadth-Statistik
    print(f"\n  Marktbreite Statistik:")
    print(f"    Ø Breite:  {breadth.mean()*100:.1f}%")
    print(f"    Min:       {breadth.min()*100:.1f}%  ({breadth.idxmin().date()})")
    print(f"    Max:       {breadth.max()*100:.1f}%  ({breadth.idxmax().date()})")

    print("\n[3/4] Simulation Setup A (Baseline)...")
    t0 = time.time()
    eq_a, trades_a, inv_a, reg_a = run_backtest(panels, use_regime=False,
                                                  breadth_thresh=args.breadth)
    print(f"  {time.time()-t0:.1f}s  |  {len(trades_a)} Trades")

    print("\n[4/4] Simulation Setup B (Regime-Verteidiger)...")
    t0 = time.time()
    eq_b, trades_b, inv_b, reg_b = run_backtest(panels, use_regime=True,
                                                  breadth_thresh=args.breadth)
    n_crisis = sum(1 for t in trades_b if t.get("crisis_exit"))
    print(f"  {time.time()-t0:.1f}s  |  {len(trades_b)} Trades  "
          f"({n_crisis} Krisen-Exits)")

    ma = compute_metrics(eq_a, trades_a, inv_a, reg_a)
    mb = compute_metrics(eq_b, trades_b, inv_b, reg_b)

    print_comparison(ma, mb, args.breadth)

    # Top 5 Winners / Losers
    for label, trades in [("Setup A", trades_a), ("Setup B", trades_b)]:
        df_t = pd.DataFrame(trades).sort_values("ret_%", ascending=False)
        print(f"  TOP 5 WINNER [{label}]:")
        for _, r in df_t.head(5).iterrows():
            crisis_mark = " [Krise]" if r.get("crisis_exit") else ""
            print(f"    {r['ticker']:<6}  "
                  f"{str(r['entry_date'])[:10]} -> {str(r['exit_date'])[:10]}  "
                  f"({int(r['hold_d'])}d)  {r['ret_%']:>+.1f}%"
                  + (" [Earned]" if r.get("earned_mode") else "") + crisis_mark)
        print()

    # CSV
    pd.concat([
        pd.DataFrame(trades_a).assign(setup="A"),
        pd.DataFrame(trades_b).assign(setup="B"),
    ], ignore_index=True).to_csv(_OUT_CSV, index=False)
    print(f"  Trades: {_OUT_CSV}")

    plot_comparison(ma, mb, trades_a, trades_b, breadth, args.breadth, _OUT_PNG)
    print("  FERTIG.\n")


if __name__ == "__main__":
    main()
