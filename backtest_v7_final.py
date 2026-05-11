"""
backtest_v7_final.py
====================================================================================
Portfolio-Backtester v7  |  "Predator" – 2-Slot Konzentrations-Strategie

Konzept:
    Anstatt 5 kleine Positionen (2.000 € / Trade → 2% Gebührenbelastung)
    halten wir maximal 2 große Positionen (5.000 € / Trade → 0.8% Gebühren).
    Durch aggressive Rotation leiten wir Kapitalflüsse gezielt in die stärksten
    Megawellen und verlassen tote Wellen über den Stall-Stop sofort.

Entry-Signal (Sweetspot aus find_sweetspot_v7.py):
    ① Breakout_50:  Close > High_50d_prev         (Ausbruch aus 50-Tage-Range)
    ② Amplitude:    (Close - Open) / Open  > 5%   (Starke Ausbruchskerze)
    ③ Volumen:      Volume > SMA_Vol_20 × 1.5      (Volumenbestätigung)
    ④ Trend:        Close > SMA_200                (Nur im Aufwärtstrend)
    ⑤ Transition:   Trigger_B50 nur am ersten Tag  (Kein Rauschen)

Exit-Management:
    A – Stall-Stop:    ≥5 Tage im Trade + Unrealized PnL < 0 → Exit sofort
    B – Earned Trail:  2.0× ATR14 (Tight) → 3.5× ATR14 (Earned) nach Profit-Nachweis
    C – Pyramidisieren: Bei freiem Slot + ≥+20% unrealized → aufstocken + Free-Ride Stop

Rotation ("Predator"):
    Portfolio voll + neuer Kandidat ist > 1.5× stärker (Trendstärke) als
    die schwächste gehaltene Position → sofort rotieren

Hardcoded:
    INITIAL_CAPITAL = 10.000 €
    ORDER_FEE       = 20 € / Order  (Round-Trip = 40 €, ca. 0.8% auf 5.000 €)
    MAX_POSITIONS   = 2

Output:
    - Metriken & Trade-Tabelle in Konsole
    - Plot 1: Equity-Kurve vs. SPY
    - Plot 2: Gantt-Chart (Slot 1 & Slot 2 Zeitstrahl, Rotations-Events markiert)
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

_here   = Path(__file__).parent
sys.path.insert(0, str(_here))

from backtest_v6 import _load_tickers, _atr

# ── Hardcoded Strategie-Parameter ───────────────────────────────────────────
INITIAL_CAPITAL     = 10_000.0
ORDER_FEE           = 20.0
MAX_POSITIONS       = 2
ATR_INIT            = 2.0       # Tight-Phase (muss Profit beweisen)
ATR_TRAIL           = 3.5       # Earned-Phase (großzügiger Trail)
AMP_THRESHOLD       = 0.05      # min. Tagesrendite am Ausbruchstag
VOL_MULTIPLIER      = 1.5       # min. Volumen-Ratio
ROTATION_FACTOR     = 1.5       # Kandidat muss X-mal stärker sein
PYRAMID_THRESHOLD   = 0.20      # +20% unrealized → aufstocken
MAX_PYRAMIDS        = 1         # max. 1× aufstocken (da nur 2 Slots)
STALL_DAYS          = 5         # Tage bis Stall-Stop greift
DEFAULT_YEARS       = 7.0
_RAW_DIR            = _here / "data" / "raw"


# ==============================================================================
# 1. DATEN LADEN
# ==============================================================================

def _load_ohlcv(tickers: list[str], years: float) -> dict[str, pd.DataFrame]:
    """Lädt OHLCV inkl. Volume aus Parquet-Dateien."""
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=int(years * 365.25))
    tset   = set(tickers)
    data: dict[str, pd.DataFrame] = {}
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
            keep = [c for c in ["open","high","low","close","volume"]
                    if c in df.columns]
            if not {"open","high","low","close"}.issubset(df.columns):
                continue
            data[ticker] = df[keep]
        except Exception:
            pass
    return data


# ==============================================================================
# 2. INDIKATOREN + PIVOT-TABELLEN AUFBAUEN
# ==============================================================================

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_panels(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Baut alle Date×Ticker Pivot-Tabellen einmalig auf."""
    cols: dict[str, dict] = {
        k: {} for k in [
            "open", "close", "high", "atr14",
            "trend_str", "entry_sig", "vol_ratio", "day_amp",
        ]
    }
    for ticker, df in data.items():
        c   = df["close"]
        o   = df["open"]
        h   = df["high"]
        vol = df.get("volume")

        sma200    = c.rolling(200).mean()
        sma20_vol = vol.rolling(20).mean() if vol is not None else None
        atr14     = _atr(df, 14)
        valid     = sma200.notna() & atr14.notna()
        idx       = c[valid].index

        # Basis-Daten
        cols["open"][ticker]      = o[valid]
        cols["close"][ticker]     = c[valid]
        cols["high"][ticker]      = h[valid]
        cols["atr14"][ticker]     = atr14[valid]
        cols["trend_str"][ticker] = ((c - sma200) / sma200)[valid]

        # Entry-Signal: Breakout_50 Transition + Amplitude + Volume + SMA200
        high50_prev = h.shift(1).rolling(50).max()
        b50_raw     = (c > high50_prev)
        trig_b50    = b50_raw & ~b50_raw.shift(1).fillna(False)  # Transition

        amp_ok  = ((c - o) / o.replace(0, np.nan)) > AMP_THRESHOLD
        vol_ok  = (vol > sma20_vol * VOL_MULTIPLIER
                   if (vol is not None and sma20_vol is not None)
                   else pd.Series(False, index=c.index))
        trend_ok = c > sma200

        sig = (trig_b50 & amp_ok & vol_ok & trend_ok)
        cols["entry_sig"][ticker] = sig.reindex(idx).fillna(False).astype(bool)

        # Hilfs-Metriken
        vol_ratio = (vol / sma20_vol.replace(0, np.nan)
                     if (vol is not None and sma20_vol is not None)
                     else pd.Series(np.nan, index=c.index))
        cols["vol_ratio"][ticker] = vol_ratio.reindex(idx)
        cols["day_amp"][ticker]   = ((c - o) / o.replace(0, np.nan)).reindex(idx)

    return {k: pd.DataFrame(v) for k, v in cols.items()}


# ==============================================================================
# 3. HILFSFUNKTIONEN: STOP-MANAGEMENT
# ==============================================================================

def _update_stop(pos: dict, today_high: float, today_atr: float) -> None:
    """Aktualisiert den Trailing-Stop einer Position (niemals senken)."""
    if math.isnan(today_high) or math.isnan(today_atr) or today_atr <= 0:
        return
    if today_high > pos["max_high"]:
        pos["max_high"] = today_high

    if not pos["earned_mode"]:
        earned_trigger = pos["entry_price"] + ATR_INIT * pos["atr_at_entry"]
        if pos["max_high"] >= earned_trigger:
            pos["earned_mode"] = True

    new_stop = (pos["max_high"] - ATR_TRAIL * today_atr
                if pos["earned_mode"]
                else pos["entry_price"] - ATR_INIT * pos["atr_at_entry"])
    pos["trailing_stop"] = max(pos["trailing_stop"], new_stop)


# ==============================================================================
# 4. SIMULATION
# ==============================================================================

def run_backtest(
    pivots:  dict[str, pd.DataFrame],
    years:   float,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    """Hauptschleife des 2-Slot Predator-Backtesters."""

    piv     = pivots
    dates   = piv["open"].index
    tickers = list(piv["open"].columns)

    # Portfolio-Zustand
    cash       = INITIAL_CAPITAL
    portfolio: dict[str, dict] = {}   # ticker → position
    free_slots = {1, 2}
    ticker_to_slot: dict[str, int] = {}

    # Sammelstrukturen
    completed:    list[dict] = []
    gantt_segs:   list[dict] = []   # für Gantt-Chart
    equity_log:   dict       = {}
    invest_days   = 0

    def _safe(df: pd.DataFrame, date, ticker) -> float:
        try:
            v = df.at[date, ticker]
            return float(v) if pd.notna(v) else math.nan
        except Exception:
            return math.nan

    def _open_position(ticker: str, buy_date, buy_price: float,
                       atr_e: float, slot: int,
                       reason: str = "ENTRY") -> None:
        nonlocal cash
        target = cash / len(free_slots) if free_slots else cash
        shares = int((target - ORDER_FEE) / buy_price) if buy_price > 0 else 0
        if shares <= 0:
            return
        # Mindestgröße: Gebühren dürfen max 5% der Position betragen
        if shares * buy_price < ORDER_FEE * 20:
            return
        cost = shares * buy_price + ORDER_FEE
        cash -= cost
        portfolio[ticker] = {
            "slot":            slot,
            "entry_date":      buy_date,
            "entry_price":     buy_price,
            "shares":          shares,
            "cost":            cost,
            "atr_at_entry":    atr_e,
            "trailing_stop":   buy_price - ATR_INIT * atr_e,
            "max_high":        buy_price,
            "earned_mode":     False,
            "pyramid_count":   0,
            "days_since_entry":0,
            "avg_entry_price": buy_price,
        }
        ticker_to_slot[ticker] = slot
        free_slots.discard(slot)
        if verbose:
            print(f"  [{buy_date.date()}] {reason}: KAUF {ticker}  "
                  f"{shares}×@{buy_price:.2f}  Stop={buy_price-ATR_INIT*atr_e:.2f}  "
                  f"Slot={slot}")

    def _close_position(ticker: str, sell_date, sell_price: float,
                        reason: str) -> None:
        nonlocal cash
        pos    = portfolio[ticker]
        slot   = pos["slot"]
        proceeds = pos["shares"] * sell_price - ORDER_FEE
        pnl_net  = proceeds - pos["cost"]
        ret_pct  = pnl_net / pos["cost"] * 100
        cash    += proceeds

        gantt_segs.append({
            "slot":       slot,
            "ticker":     ticker,
            "start":      pos["entry_date"],
            "end":        sell_date,
            "ret_pct":    ret_pct,
            "is_rotation":reason == "ROTATION_OUT",
            "reason":     reason,
        })
        completed.append({
            "Ticker":    ticker,
            "Kauf":      pos["entry_date"].date(),
            "KaufPreis": round(pos["entry_price"], 2),
            "Verkauf":   sell_date.date(),
            "VKPreis":   round(sell_price, 2),
            "Shares":    pos["shares"],
            "PnL_€":     round(pnl_net, 2),
            "Return_%":  round(ret_pct, 2),
            "Haltedauer":  (sell_date - pos["entry_date"]).days,
            "Earned":    pos["earned_mode"],
            "Pyramide":  pos["pyramid_count"],
            "Exit":      reason,
        })
        free_slots.add(slot)
        del portfolio[ticker]
        if ticker in ticker_to_slot:
            del ticker_to_slot[ticker]
        if verbose:
            sign = "+" if pnl_net >= 0 else ""
            print(f"  [{sell_date.date()}] {reason}: VERKAUF {ticker}  "
                  f"@{sell_price:.2f}  PnL={sign}{pnl_net:,.0f}€  "
                  f"({sign}{ret_pct:.1f}%)")

    # ── Hauptschleife ────────────────────────────────────────────────────────
    for day_i in range(len(dates) - 1):
        today     = dates[day_i]
        tomorrow  = dates[day_i + 1]
        exits_today: list[tuple[str, str]] = []   # (ticker, reason)
        rotation_out: str | None = None

        # ── A. Bestehende Positionen aktualisieren & Exit prüfen ─────────────
        for ticker, pos in list(portfolio.items()):
            tc  = _safe(piv["close"],  today, ticker)
            th  = _safe(piv["high"],   today, ticker)
            ta  = _safe(piv["atr14"],  today, ticker)

            if math.isnan(tc):
                continue

            pos["days_since_entry"] += 1
            _update_stop(pos, th, ta)

            # Stall-Stop
            if pos["days_since_entry"] >= STALL_DAYS and tc < pos["avg_entry_price"]:
                exits_today.append((ticker, "STALL_STOP"))
                continue

            # ATR-Stop
            if tc < pos["trailing_stop"]:
                exits_today.append((ticker, "ATR_STOP"))

        # ── B. Exits ausführen (zu morgen's Open) ────────────────────────────
        for ticker, reason in exits_today:
            sell_px = _safe(piv["open"], tomorrow, ticker)
            if math.isnan(sell_px):
                continue
            _close_position(ticker, tomorrow, sell_px, reason)

        # ── C. Kandidaten-Signale für heute finden ───────────────────────────
        candidates = [
            t for t in tickers
            if t not in portfolio
            and _safe(piv["entry_sig"], today, t) == 1.0
        ]

        if candidates:
            # Nach Trendstärke sortieren
            candidates.sort(
                key=lambda t: _safe(piv["trend_str"], today, t),
                reverse=True
            )

            for cand in candidates:
                buy_px  = _safe(piv["open"],  tomorrow, cand)
                atr_e   = _safe(piv["atr14"], today,    cand)
                if math.isnan(buy_px) or math.isnan(atr_e) or atr_e <= 0:
                    continue

                if len(portfolio) < MAX_POSITIONS and free_slots:
                    slot = min(free_slots)
                    _open_position(cand, tomorrow, buy_px, atr_e, slot)

                elif len(portfolio) >= MAX_POSITIONS:
                    # Rotation prüfen: schwächste Position heraussuchen
                    weakest_t = min(
                        portfolio.keys(),
                        key=lambda t: _safe(piv["trend_str"], today, t)
                    )
                    weakest_str = _safe(piv["trend_str"], today, weakest_t)
                    cand_str    = _safe(piv["trend_str"], today, cand)

                    if (not math.isnan(cand_str) and not math.isnan(weakest_str)
                            and cand_str > ROTATION_FACTOR * weakest_str):
                        # Rotation: schwächste raus, neue Welle rein
                        rot_sell_px = _safe(piv["open"], tomorrow, weakest_t)
                        if not math.isnan(rot_sell_px):
                            if verbose:
                                print(f"  [{tomorrow.date()}] "
                                      f"ROTATION: {weakest_t}→{cand}  "
                                      f"Stärke {weakest_str:.3f}→{cand_str:.3f}  "
                                      f"(Faktor {cand_str/weakest_str:.2f}×)")
                            freed_slot = portfolio[weakest_t]["slot"]
                            _close_position(weakest_t, tomorrow,
                                            rot_sell_px, "ROTATION_OUT")
                            _open_position(cand, tomorrow, buy_px, atr_e,
                                           freed_slot, "ROTATION_IN")
                            rotation_out = cand
                            break   # nur eine Rotation pro Tag

        # ── D. Pyramidisieren ─────────────────────────────────────────────────
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
                    add_shares = int((cash - ORDER_FEE) / buy_px)
                    if add_shares <= 0:
                        continue
                    # Positionen zusammenführen
                    old_val  = pos["shares"]          * pos["avg_entry_price"]
                    add_val  = add_shares             * buy_px
                    new_avg  = (old_val + add_val) / (pos["shares"] + add_shares)
                    cost_add = add_shares * buy_px + ORDER_FEE
                    cash              -= cost_add
                    pos["cost"]       += cost_add
                    pos["shares"]     += add_shares
                    pos["avg_entry_price"] = new_avg
                    pos["pyramid_count"]   += 1
                    # Free-Ride Stop
                    old_stop   = pos["trailing_stop"]
                    free_ride  = new_avg
                    if free_ride > old_stop:
                        pos["trailing_stop"] = free_ride
                        fr_msg = f"FREE-RIDE Stop→{free_ride:.2f}"
                    else:
                        fr_msg = "Stop unverändert"
                    if verbose:
                        print(f"  [{tomorrow.date()}] PYRAMIDE: {ticker}  "
                              f"+{add_shares}× @{buy_px:.2f}  "
                              f"AvgEntry={new_avg:.2f}  {fr_msg}")

        # ── E. Equity tracken ─────────────────────────────────────────────────
        equity = cash
        n_held = 0
        for ticker, pos in portfolio.items():
            cp = _safe(piv["close"], today, ticker)
            equity += pos["shares"] * (cp if not math.isnan(cp) else pos["avg_entry_price"])
            n_held += 1
        equity_log[today] = equity
        if n_held > 0:
            invest_days += 1

    # Offene Positionen am Ende schliessen
    last_date = dates[-1]
    for ticker in list(portfolio.keys()):
        lp = _safe(piv["close"], last_date, ticker)
        if not math.isnan(lp):
            _close_position(ticker, last_date, lp, "END_OF_BACKTEST")

    equity_series = pd.Series(equity_log).sort_index()
    return equity_series, completed, gantt_segs, invest_days


# ==============================================================================
# 5. METRIKEN
# ==============================================================================

def compute_metrics(
    equity: pd.Series,
    trades: list[dict],
    invest_days: int = 0,
) -> dict:
    if equity.empty or len(trades) == 0:
        return {}

    total_days = len(equity)
    years      = (equity.index[-1] - equity.index[0]).days / 365.25

    eq_clean   = equity.ffill().bfill()
    total_ret  = (eq_clean.iloc[-1] / INITIAL_CAPITAL - 1) * 100
    cagr       = ((eq_clean.iloc[-1] / INITIAL_CAPITAL) ** (1 / years) - 1) * 100 if years > 0 else 0
    peak       = eq_clean.cummax()
    dd         = (eq_clean - peak) / peak * 100
    max_dd     = dd.min()
    daily_ret  = eq_clean.pct_change().dropna()
    sharpe     = (daily_ret.mean() / daily_ret.std() * (252 ** 0.5)
                  if daily_ret.std() > 0 else 0)

    rets  = [t["Return_%"] for t in trades]
    wins  = [r for r in rets if r > 0]
    losses= [r for r in rets if r <= 0]
    pf    = sum(wins) / abs(sum(losses)) if losses else float("inf")

    return {
        "total_ret_%":  round(total_ret, 2),
        "cagr_%":       round(cagr, 2),
        "max_dd_%":     round(max_dd, 2),
        "sharpe":       round(sharpe, 2),
        "n_trades":     len(trades),
        "hit_%":        round(len(wins) / len(trades) * 100, 1) if trades else 0,
        "payoff":       round(abs(np.mean(wins) / np.mean(losses)), 2) if wins and losses else 0,
        "profit_factor":round(pf, 2),
        "avg_hold_d":   round(np.mean([t["Haltedauer"] for t in trades]), 1),
        "total_fees":   len(trades) * ORDER_FEE * 2,
        "end_equity":   round(eq_clean.iloc[-1], 2),
        "invest_pct":   round(invest_days / total_days * 100, 1),
        "n_rotations":  sum(1 for t in trades if t["Exit"] == "ROTATION_OUT"),
        "n_stall_exits":sum(1 for t in trades if t["Exit"] == "STALL_STOP"),
        "n_atr_exits":  sum(1 for t in trades if t["Exit"] == "ATR_STOP"),
        "n_pyramids":   sum(t["Pyramide"] for t in trades),
    }


def print_summary(m: dict, trades: list[dict]) -> None:
    print(f"""
{'=' * 70}
  BACKTEST v7 FINAL  |  Predator  |  2-Slot Konzentration
{'=' * 70}

  Portfolio:   {INITIAL_CAPITAL:,.0f}€ Start  |  {MAX_POSITIONS} Slots  |  ~{INITIAL_CAPITAL/MAX_POSITIONS:,.0f}€/Trade
  Gebühren:    {ORDER_FEE:.0f}€/Order → {ORDER_FEE*2:.0f}€/Roundtrip ({ORDER_FEE*2/(INITIAL_CAPITAL/MAX_POSITIONS)*100:.1f}% Belastung)
  Signal:      Breakout_50 + Amp≥5% + Vol≥1.5× + SMA_200

  ──────────────────────────────────────────────────────────────────
  Gesamtrendite:      {m['total_ret_%']:>+8.2f}%
  CAGR:               {m['cagr_%']:>+8.2f}%
  Max Drawdown:       {m['max_dd_%']:>+8.1f}%
  Sharpe Ratio:       {m['sharpe']:>8.2f}
  Endkapital:         {m['end_equity']:>10,.0f} €

  ──────────────────────────────────────────────────────────────────
  Trades gesamt:      {m['n_trades']:>5}
  Hit-Rate:           {m['hit_%']:>5.1f}%
  Payoff Ratio:       {m['payoff']:>5.2f}
  Profit Factor:      {m['profit_factor']:>5.2f}
  Avg Haltedauer:     {m['avg_hold_d']:>5.1f} Tage
  Gezahlte Gebühren:  {m['total_fees']:>8,.0f} €

  ──────────────────────────────────────────────────────────────────
  Investitionsquote:  {m['invest_pct']:>5.1f}%  (Tage mit ≥1 Position)
  Rotationen:         {m['n_rotations']:>5}
  Stall-Stop-Exits:   {m['n_stall_exits']:>5}
  ATR-Stop-Exits:     {m['n_atr_exits']:>5}
  Pyramidisierungen:  {m['n_pyramids']:>5}
{'=' * 70}
""")

    if trades:
        df = pd.DataFrame(trades)
        pd.set_option("display.width", 200)
        pd.set_option("display.float_format", "{:.2f}".format)
        print("  ALLE TRADES (neueste zuletzt):")
        print(f"  {'─' * 105}")
        print(df.to_string(index=False))
        print(f"  {'─' * 105}")


# ==============================================================================
# 6. VISUALISIERUNGEN
# ==============================================================================

def _load_spy(years: float) -> pd.Series | None:
    """Versucht SPY aus Parquet zu laden."""
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
        df = df[df.index >= cutoff]
        return df["close"]
    except Exception:
        return None


def plot_equity(equity: pd.Series, years: float, out_path: str) -> None:
    """Plot 1: Equity-Kurve vs. SPY."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True
    )
    fig.patch.set_facecolor("#0D1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0D1117")
        for spine in ax.spines.values():
            spine.set_color("#444")
        ax.tick_params(colors="#AAA")
        ax.yaxis.label.set_color("#AAA")
        ax.xaxis.label.set_color("#AAA")

    # Equity normalisiert
    eq_norm = equity / equity.iloc[0] * INITIAL_CAPITAL
    ax1.plot(eq_norm.index, eq_norm.values,
             color="#00E5FF", linewidth=1.8, label="Predator v7")
    ax1.axhline(INITIAL_CAPITAL, color="#555", linewidth=0.8, linestyle="--")

    # SPY Benchmark
    spy = _load_spy(years)
    if spy is not None:
        spy = spy.reindex(equity.index, method="ffill").dropna()
        if len(spy) > 0:
            spy_norm = spy / spy.iloc[0] * INITIAL_CAPITAL
            ax1.plot(spy_norm.index, spy_norm.values,
                     color="#FF9800", linewidth=1.2, alpha=0.7, label="SPY")

    ax1.set_ylabel("Kapital (€)", fontsize=10, color="#AAA")
    ax1.set_title("Predator v7  |  2-Slot Konzentration  |  Sweetspot-Einstieg",
                  fontsize=12, color="white", fontweight="bold")
    ax1.legend(facecolor="#1E2330", edgecolor="#444",
               labelcolor="white", fontsize=9)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f"{x:,.0f}€"))
    ax1.grid(True, color="#2A2A2A", linewidth=0.5)

    # Drawdown
    peak = eq_norm.cummax()
    dd   = (eq_norm - peak) / peak * 100
    ax2.fill_between(dd.index, dd.values, 0,
                     color="#FF4444", alpha=0.6, label="Drawdown")
    ax2.set_ylabel("Drawdown %", fontsize=9, color="#AAA")
    ax2.set_ylim(min(dd.min() * 1.2, -5), 2)
    ax2.grid(True, color="#2A2A2A", linewidth=0.5)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


def plot_gantt(gantt_segs: list[dict], equity: pd.Series,
               trades: list[dict], out_path: str) -> None:
    """Plot 2: Gantt-Chart (Zeitstrahl Slot 1 & Slot 2)."""
    if not gantt_segs:
        return

    fig, ax = plt.subplots(figsize=(16, 5))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#111827")
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.tick_params(colors="#CCC")

    # Farb-Palette pro Ticker
    all_tickers = sorted({s["ticker"] for s in gantt_segs})
    palette     = plt.cm.get_cmap("tab20", max(len(all_tickers), 1))
    ticker_color = {t: palette(i) for i, t in enumerate(all_tickers)}

    y_pos   = {1: 1.0, 2: 0.0}
    bar_h   = 0.7
    min_date = equity.index[0]
    max_date = equity.index[-1]
    total_span = (max_date - min_date).days

    for seg in gantt_segs:
        slot   = seg["slot"]
        y      = y_pos[slot]
        start  = seg["start"]
        end    = seg["end"]
        ticker = seg["ticker"]
        color  = ticker_color[ticker]
        ret    = seg["ret_pct"]

        width = max((end - start).days, 1)
        bar = ax.barh(
            y, width, left=mdates.date2num(start),
            height=bar_h, color=color, alpha=0.85,
            edgecolor="#222", linewidth=0.5
        )

        # Ticker-Label im Balken
        mid = start + (end - start) / 2
        if width > total_span * 0.015:
            sign = "+" if ret >= 0 else ""
            label = f"{ticker}\n{sign}{ret:.1f}%"
            ax.text(mdates.date2num(mid), y, label,
                    ha="center", va="center", fontsize=7.5,
                    color="white", fontweight="bold")

        # Rotations-Event markieren
        if seg.get("is_rotation"):
            ax.plot(mdates.date2num(end), y,
                    "D", color="#FF4444", markersize=8, zorder=5,
                    label="Rotation-Exit" if seg == gantt_segs[0] else "")

    # Achsen formatieren
    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(["  Slot 2", "  Slot 1"], fontsize=10, color="white")
    ax.set_xlim(mdates.date2num(min_date), mdates.date2num(max_date))
    ax.set_ylim(-0.6, 1.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=30, ha="right", fontsize=8, color="#CCC")

    # Legende
    legend_patches = [
        mpatches.Patch(color=ticker_color[t], label=t)
        for t in all_tickers
    ]
    rotation_marker = plt.Line2D(
        [], [], marker="D", color="#FF4444", linestyle="",
        markersize=8, label="Rotation-Exit"
    )
    legend_patches.append(rotation_marker)
    ax.legend(handles=legend_patches, loc="upper left",
              facecolor="#1E2330", edgecolor="#444",
              labelcolor="white", fontsize=7,
              ncol=min(len(legend_patches), 8))

    n_rot = sum(1 for s in gantt_segs if s.get("is_rotation"))
    invest_segs = len(gantt_segs)
    ax.set_title(
        f"Predator v7  |  Slot-Zeitstrahl  |  "
        f"{invest_segs} Positionen  |  {n_rot} Rotationen",
        fontsize=11, color="white", fontweight="bold", pad=12
    )
    ax.grid(True, axis="x", color="#2A2A2A", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


# ==============================================================================
# 7. MAIN
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest v7 Final – Predator 2-Slot Strategie")
    parser.add_argument("--years",   type=float, default=DEFAULT_YEARS)
    parser.add_argument("--verbose", action="store_true",
                        help="Alle Trade-Logs ausgeben")
    args = parser.parse_args()

    print("=" * 70)
    print("  PREDATOR BACKTEST v7 FINAL  |  2-Slot Sweetspot-Strategie")
    print("=" * 70)
    print(f"""
  Entry: Breakout_50 + Amp≥{AMP_THRESHOLD*100:.0f}% + Vol≥{VOL_MULTIPLIER:.1f}× + SMA_200
  Exit:  Stall-Stop ({STALL_DAYS}d) | ATR-Trail {ATR_INIT}×→{ATR_TRAIL}× | Pyramide+Free-Ride
  Rotation: Faktor {ROTATION_FACTOR:.1f}×  |  Startkapital: {INITIAL_CAPITAL:,.0f}€  |  {MAX_POSITIONS} Slots
""")

    # 1. Daten laden
    t0 = time.time()
    tickers = _load_tickers()
    print(f"[1/4] Lade {len(tickers)} Ticker ({args.years:.0f} Jahre)...")
    data = _load_ohlcv(tickers, args.years)
    has_vol = any("volume" in df.columns for df in data.values())
    print(f"  {len(data)} Ticker in {time.time()-t0:.1f}s  "
          f"| Volume: {'ja ✓' if has_vol else 'FEHLT ✗'}")
    if not has_vol:
        print("  [WARN] Keine Volume-Daten – Amp+Vol-Filter inaktiv!")

    # 2. Pivots aufbauen
    print(f"\n[2/4] Indikator-Pivots aufbauen...")
    t0 = time.time()
    pivots = build_panels(data)
    dates  = pivots["open"].index
    n_sig  = int(pivots["entry_sig"].fillna(False).values.sum())
    print(f"  Zeitraum: {dates[0].date()} → {dates[-1].date()}  "
          f"({len(dates):,} Tage, {len(pivots['open'].columns)} Ticker)")
    print(f"  Entry-Signale gesamt: {n_sig:,}  "
          f"({n_sig / len(pivots['open'].columns) / 7:.1f}/Ticker/Jahr)")

    # 3. Simulation
    print(f"\n[3/4] Simulation läuft...")
    t0 = time.time()
    equity, trades, gantt_segs, invest_days = run_backtest(pivots, args.years, args.verbose)
    m = compute_metrics(equity, trades, invest_days)
    print(f"  Fertig in {time.time()-t0:.1f}s")

    # 4. Ausgabe
    print_summary(m, trades)

    # 5. Plots
    print(f"[4/4] Visualisierungen speichern...")
    eq_path   = str(_here / "backtest_v7_equity.png")
    gantt_path= str(_here / "backtest_v7_gantt.png")
    plot_equity(equity, args.years, eq_path)
    plot_gantt(gantt_segs, equity, trades, gantt_path)
    print(f"  Equity-Kurve: {eq_path}")
    print(f"  Gantt-Chart:  {gantt_path}")
    print()


if __name__ == "__main__":
    main()
