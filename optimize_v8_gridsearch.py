"""
optimize_v8_gridsearch.py
====================================================================================
2D Grid-Search  |  VCP Backtester v8.2  |  260 US-Aktien

Optimiert zwei Parameter gleichzeitig:
    1. BREADTH_THRESHOLD  – Marktbreite-Filter (intern, ohne SPY)
       Anteil der Aktien mit Close > SMA_200 muss >= Schwelle sein.
       → Schützt vor Bärenmarkt-Environments ohne externen Index.

    2. ROTATION_FACTOR    – Predator-Rotations-Bremse
       Neuer Kandidat muss X-fach stärker sein als schwächste Position.
       999.0 = Rotation komplett deaktiviert.

Neue Regel – "Diamond Hands":
    Eine Position im "Earned Mode" (Stop bereits auf 3.5× ATR ausgeweitet,
    Aktie hat sich im Profit bewiesen) wird NIEMALS durch Rotation verkauft.
    Nur "frische", noch unbewiesene Positionen können rotiert werden.

Hardcoded Engine (identisch zu v8):
    INITIAL_CAPITAL = 10.000 €
    ORDER_FEE       = 20 €  (Round-Trip = 40 €)
    MAX_POSITIONS   = 2
    Entry: VCP-Breakout (BB-Squeeze<10% + B50 + Vol×1.5 + SMA200)
    Exit:  ATR 2.0× → 3.5× Earned  |  Kein Stall-Stop

Grid-Größe: 5 × 6 = 30 Kombinationen
Laufzeit:   ca. 1-3 Minuten

Verwendung:
    python optimize_v8_gridsearch.py
    python optimize_v8_gridsearch.py --years 7 --top 20
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_here = Path(__file__).parent
sys.path.insert(0, str(_here))

from backtest_v6 import _load_tickers, _atr

# ── Hardcoded Engine-Parameter ───────────────────────────────────────────────
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
DEFAULT_YEARS       = 7.0
_RAW_DIR            = _here / "data" / "raw"

# ── Grid-Parameter ───────────────────────────────────────────────────────────
BREADTH_THRESHOLDS = [0.0, 0.20, 0.30, 0.40, 0.50]
ROTATION_FACTORS   = [1.2, 1.5, 2.0, 2.5, 3.0, 999.0]


# ==============================================================================
# 1. DATEN LADEN
# ==============================================================================

def _load_ohlcv(tickers: list[str], years: float) -> dict[str, pd.DataFrame]:
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
# 2. PIVOTS + MARKTBREITE
# ==============================================================================

def build_panels(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Baut alle Date×Ticker Pivots auf. Entry-Signal ohne Breadth-Filter."""
    cols: dict[str, dict] = {
        k: {} for k in ["open","close","high","atr14","trend_str","entry_sig"]
    }
    for ticker, df in data.items():
        c   = df["close"]
        h   = df["high"]
        vol = df.get("volume")

        sma200    = c.rolling(200).mean()
        sma20_vol = vol.rolling(20).mean() if vol is not None else None
        atr14     = _atr(df, 14)

        sma_bb  = c.rolling(BB_PERIOD).mean()
        std_bb  = c.rolling(BB_PERIOD).std()
        bb_w    = (sma_bb + BB_STD*std_bb - (sma_bb - BB_STD*std_bb)) / c.replace(0, np.nan)

        valid = sma200.notna() & atr14.notna()
        idx   = c[valid].index

        cols["open"][ticker]      = df["open"][valid]
        cols["close"][ticker]     = c[valid]
        cols["high"][ticker]      = h[valid]
        cols["atr14"][ticker]     = atr14[valid]
        cols["trend_str"][ticker] = ((c - sma200) / sma200)[valid]

        # VCP Entry (ohne Breadth – wird dynamisch in der Schleife geprüft)
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


def compute_daily_breadth(close_piv: pd.DataFrame) -> pd.Series:
    """
    Interne Marktbreite: Anteil der Aktien mit Close > SMA_200.
    Vollständig vektorisiert (kein Loop).
    """
    sma200  = close_piv.rolling(200).mean()
    above   = (close_piv > sma200).astype(float)
    # Pro Tag: Anteil der Ticker mit gültigem Wert
    breadth = above.sum(axis=1) / above.notna().sum(axis=1)
    return breadth.fillna(0.0)


# ==============================================================================
# 3. SIMULATION ENGINE  (parameterisiert)
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
    pivots:            dict[str, pd.DataFrame],
    breadth:           pd.Series,
    breadth_threshold: float,
    rotation_factor:   float,
) -> tuple[pd.Series, list[dict], int]:
    """
    Simuliert das 2-Slot VCP Portfolio für eine Parameterkombination.
    Diamond-Hands-Regel ist immer aktiv.
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
               atr_e: float, slot: int, is_rotation: bool = False) -> bool:
        nonlocal cash
        mkt = sum(p["shares"] * p["avg_entry_price"] for p in portfolio.values())
        target = min((cash + mkt) / MAX_POSITIONS, cash - ORDER_FEE)
        shares = int((target - ORDER_FEE) / buy_px) if buy_px > 0 else 0
        if shares < MIN_SHARES or shares * buy_px < ORDER_FEE * 20:
            return False
        cost = shares * buy_px + ORDER_FEE
        if cost > cash:
            return False
        cash -= cost
        portfolio[ticker] = {
            "slot":           slot,
            "entry_date":     buy_date,
            "entry_price":    buy_px,
            "shares":         shares,
            "cost":           cost,
            "atr_at_entry":   atr_e,
            "trailing_stop":  buy_px - ATR_INIT * atr_e,
            "max_high":       buy_px,
            "earned_mode":    False,
            "pyramid_count":  0,
            "avg_entry_price": buy_px,
            "days_held":      0,
        }
        ticker_to_slot[ticker] = slot
        free_slots.discard(slot)
        return True

    def _close(ticker: str, sell_date, sell_px: float, is_rotation: bool) -> None:
        nonlocal cash
        pos      = portfolio[ticker]
        slot     = pos["slot"]
        proceeds = pos["shares"] * sell_px - ORDER_FEE
        pnl      = proceeds - pos["cost"]
        ret_pct  = pnl / pos["cost"] * 100
        cash    += proceeds
        completed.append({
            "ticker":      ticker,
            "entry_date":  pos["entry_date"],
            "exit_date":   sell_date,
            "pnl":         pnl,
            "ret_%":       ret_pct,
            "hold_d":      pos["days_held"],
            "earned":      pos["earned_mode"],
            "is_rotation": is_rotation,
            "pyr":         pos["pyramid_count"],
        })
        free_slots.add(slot)
        del portfolio[ticker]
        ticker_to_slot.pop(ticker, None)

    # ── Hauptschleife ────────────────────────────────────────────────────────
    for day_i in range(len(dates) - 1):
        today    = dates[day_i]
        tomorrow = dates[day_i + 1]

        # A. Stops prüfen
        exits: list[str] = []
        for ticker, pos in list(portfolio.items()):
            tc = _safe(piv["close"], today, ticker)
            th = _safe(piv["high"],  today, ticker)
            ta = _safe(piv["atr14"], today, ticker)
            if math.isnan(tc):
                continue
            pos["days_held"] += 1
            _update_stop(pos, th, ta)
            if tc < pos["trailing_stop"]:
                exits.append(ticker)

        for ticker in exits:
            sell_px = _safe(piv["open"], tomorrow, ticker)
            if not math.isnan(sell_px):
                _close(ticker, tomorrow, sell_px, is_rotation=False)

        # B. Marktbreite prüfen
        today_breadth = float(breadth.at[today]) if today in breadth.index else 0.0
        candidates_allowed = today_breadth >= breadth_threshold

        # C. Kandidaten
        if candidates_allowed:
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
                            continue   # alle Positionen im Earned Mode

                        weakest_t   = min(
                            rotatable.keys(),
                            key=lambda t: _safe(piv["trend_str"], today, t)
                        )
                        weakest_str = _safe(piv["trend_str"], today, weakest_t)
                        cand_str    = _safe(piv["trend_str"], today, cand)

                        if (not math.isnan(cand_str)
                                and not math.isnan(weakest_str)
                                and rotation_factor < 999.0
                                and cand_str > rotation_factor * weakest_str):
                            rot_px = _safe(piv["open"], tomorrow, weakest_t)
                            if not math.isnan(rot_px):
                                freed = portfolio[weakest_t]["slot"]
                                _close(weakest_t, tomorrow, rot_px,
                                       is_rotation=True)
                                _open(cand, tomorrow, buy_px, atr_e,
                                      freed, is_rotation=True)
                                break

        # D. Pyramidisieren
        if candidates_allowed and len(portfolio) == 1 and free_slots:
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
                    old_v    = pos["shares"] * pos["avg_entry_price"]
                    new_avg  = (old_v + add_sh * buy_px) / (pos["shares"] + add_sh)
                    cash    -= cost_add
                    pos["cost"]           += cost_add
                    pos["shares"]         += add_sh
                    pos["avg_entry_price"] = new_avg
                    pos["pyramid_count"]  += 1
                    pos["trailing_stop"]   = max(pos["trailing_stop"], new_avg)

        # E. Equity
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
            _close(ticker, dates[-1], lp, is_rotation=False)

    return pd.Series(equity_log).sort_index(), completed, invest_days


# ==============================================================================
# 4. METRIKEN
# ==============================================================================

def compute_metrics(
    equity:      pd.Series,
    trades:      list[dict],
    invest_days: int,
) -> dict:
    if equity.empty or not trades:
        return {k: np.nan for k in [
            "Rendite_%","CAGR_%","MaxDD_%","Sharpe",
            "N_Trades","N_Rotations","Hit_%","Payoff","ProfitFactor",
            "EV_%","AvgHold_d","Fees_€","InvestPct_%","EndKap_€",
        ]}

    years  = max((equity.index[-1] - equity.index[0]).days / 365.25, 0.1)
    eq     = equity.ffill().bfill()
    ret    = (eq.iloc[-1] / INITIAL_CAPITAL - 1) * 100
    cagr   = ((eq.iloc[-1] / INITIAL_CAPITAL) ** (1/years) - 1) * 100
    peak   = eq.cummax()
    dd     = (eq - peak) / peak * 100
    dr     = eq.pct_change().dropna()
    sharpe = (dr.mean() / dr.std() * 252**0.5) if dr.std() > 0 else 0

    rets   = [t["ret_%"] for t in trades]
    wins   = [r for r in rets if r > 0]
    losses = [r for r in rets if r < 0]
    hit    = len(wins) / len(rets) * 100 if rets else 0
    avg_w  = float(np.mean(wins))  if wins   else 0.0
    avg_l  = float(np.mean(losses)) if losses else 0.0
    payoff = abs(avg_w / avg_l) if avg_l else float("inf")
    pf     = sum(wins) / abs(sum(losses)) if losses else float("inf")
    ev     = hit/100 * avg_w + (1 - hit/100) * avg_l

    n_rot  = sum(1 for t in trades if t["is_rotation"])

    return {
        "Rendite_%":     round(ret,   2),
        "CAGR_%":        round(cagr,  2),
        "MaxDD_%":       round(dd.min(), 1),
        "Sharpe":        round(sharpe, 2),
        "N_Trades":      len(trades),
        "N_Rotations":   n_rot,
        "Hit_%":         round(hit,  1),
        "Payoff":        round(payoff, 2),
        "ProfitFactor":  round(pf,   2),
        "EV_%":          round(ev,   2),
        "AvgHold_d":     round(np.mean([t["hold_d"] for t in trades]), 1),
        "Fees_€":        len(trades) * ORDER_FEE * 2,
        "InvestPct_%":   round(invest_days / len(equity) * 100, 1),
        "EndKap_€":      round(eq.iloc[-1], 0),
    }


# ==============================================================================
# 5. AUSGABE
# ==============================================================================

def print_results(df: pd.DataFrame, top_n: int = 15) -> None:
    df_s = df.sort_values("Rendite_%", ascending=False).reset_index(drop=True)

    print(f"\n{'=' * 120}")
    print(f"  GRIDSEARCH ERGEBNISSE  |  VCP v8.2  |  Diamond Hands  |  "
          f"{len(df_s)} Kombinationen  |  "
          f"Sorted by Gesamtrendite")
    print(f"{'=' * 120}")
    print(f"  {'Rg':>3}  {'Breadth':>7}  {'RotFak':>7}  "
          f"{'Rendite%':>9}  {'CAGR%':>7}  {'MaxDD%':>7}  "
          f"{'Sharpe':>7}  {'Trades':>6}  {'Rot':>4}  "
          f"{'Hit%':>5}  {'Payoff':>7}  {'EV%':>6}  "
          f"{'Hold':>6}  {'Fees€':>6}  {'EndKap€':>9}")
    print(f"  {'─' * 115}")

    for rank, row in df_s.head(top_n).iterrows():
        rot_str = "∞" if row["RotFak"] >= 999 else f"{row['RotFak']:.1f}"
        brd_str = f"{row['Breadth']:.0%}" if row["Breadth"] > 0 else "aus"
        marker  = " ★" if rank == 0 else "  "
        ev_col  = f"{row['EV_%']:>+5.2f}%" 
        print(f"  {rank+1:>3}{marker}  {brd_str:>7}  {rot_str:>7}  "
              f"  {row['Rendite_%']:>+8.2f}%"
              f"  {row['CAGR_%']:>+6.2f}%"
              f"  {row['MaxDD_%']:>+6.1f}%"
              f"  {row['Sharpe']:>7.2f}"
              f"  {int(row['N_Trades']):>6}"
              f"  {int(row['N_Rotations']):>4}"
              f"  {row['Hit_%']:>5.1f}%"
              f"  {row['Payoff']:>7.2f}"
              f"  {ev_col}"
              f"  {row['AvgHold_d']:>5.1f}d"
              f"  {int(row['Fees_€']):>6}"
              f"  {row['EndKap_€']:>9,.0f}")

    print(f"  {'─' * 115}")

    # Mittelwerte nach Breadth
    print(f"\n  DURCHSCHNITT NACH MARKTBREITE-FILTER:")
    print(f"  {'─' * 75}")
    for bval, grp in df_s.groupby("Breadth"):
        pos    = (grp["Rendite_%"] > 0).sum()
        label  = f">={bval:.0%}" if bval > 0 else "aus"
        print(f"  Breadth {label:<7}  "
              f"AvgRendite: {grp['Rendite_%'].mean():>+7.2f}%  |  "
              f"AvgHit: {grp['Hit_%'].mean():>5.1f}%  |  "
              f"Positiv: {pos}/{len(grp)}")

    # Mittelwerte nach Rotation
    print(f"\n  DURCHSCHNITT NACH ROTATIONS-FAKTOR:")
    print(f"  {'─' * 75}")
    for rval, grp in df_s.groupby("RotFak"):
        pos   = (grp["Rendite_%"] > 0).sum()
        label = "AUS" if rval >= 999 else f"{rval:.1f}×"
        print(f"  Rotation {label:<6}  "
              f"AvgRendite: {grp['Rendite_%'].mean():>+7.2f}%  |  "
              f"AvgTrades: {grp['N_Trades'].mean():>5.0f}  |  "
              f"AvgRot: {grp['N_Rotations'].mean():>4.0f}  |  "
              f"Positiv: {pos}/{len(grp)}")

    # Best overall
    best = df_s.iloc[0]
    rot_b = "AUS" if best["RotFak"] >= 999 else f"{best['RotFak']:.1f}×"
    brd_b = f">={best['Breadth']:.0%}" if best["Breadth"] > 0 else "DEAKTIVIERT"
    print(f"""
  BESTE KOMBINATION:
  {'─' * 65}
  Breadth-Threshold:  {brd_b}
  Rotations-Faktor:   {rot_b}
  Rendite:            {best['Rendite_%']:>+.2f}%  (CAGR {best['CAGR_%']:>+.2f}%)
  Max Drawdown:       {best['MaxDD_%']:>+.1f}%  |  Sharpe: {best['Sharpe']:.2f}
  Trades:             {int(best['N_Trades'])}  |  Rotationen: {int(best['N_Rotations'])}
  Hit-Rate:           {best['Hit_%']:.1f}%  |  Payoff: {best['Payoff']:.2f}
  EV/Trade:          {best['EV_%']:>+.2f}%  |  EndKap: {best['EndKap_€']:,.0f}€
""")


# ==============================================================================
# 6. MAIN
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid-Search v8.2  |  VCP + Breadth + Rotation")
    parser.add_argument("--years", type=float, default=DEFAULT_YEARS)
    parser.add_argument("--top",   type=int,   default=15)
    args = parser.parse_args()

    n_combos = len(BREADTH_THRESHOLDS) * len(ROTATION_FACTORS)

    print("=" * 70)
    print("  VCP GRIDSEARCH v8.2  |  Breadth × Rotation  |  Diamond Hands")
    print("=" * 70)
    print(f"""
  Engine:   VCP (BB-Squeeze<{BB_SQUEEZE_THRESH*100:.0f}% + B50 + Vol×{VOL_MULTIPLIER} + SMA200)
  Exit:     ATR {ATR_INIT}× → {ATR_TRAIL}× Earned  |  Kein Stall-Stop
  Neu:      Diamond Hands (Earned-Mode Positionen rotation-geschützt)
  Grid:     {len(BREADTH_THRESHOLDS)} Breadth × {len(ROTATION_FACTORS)} Rotation = {n_combos} Kombinationen
  Kapital:  {INITIAL_CAPITAL:,.0f}€  |  {MAX_POSITIONS} Slots
  Zeitraum: {args.years:.0f} Jahre
""")

    # 1. Daten
    print("[1/4] Lade Daten...")
    t0 = time.time()
    tickers = _load_tickers()
    data    = _load_ohlcv(tickers, args.years)
    print(f"  {len(data)} Ticker in {time.time()-t0:.1f}s")

    # 2. Pivots (einmalig!)
    print(f"\n[2/4] Pivots aufbauen + Marktbreite berechnen...")
    t0     = time.time()
    pivots = build_panels(data)
    dates  = pivots["open"].index
    breadth = compute_daily_breadth(pivots["close"])
    n_sig   = int(pivots["entry_sig"].fillna(False).values.sum())
    print(f"  Zeitraum: {dates[0].date()} → {dates[-1].date()} "
          f"({len(dates):,} Tage, {len(pivots['open'].columns)} Ticker)")
    print(f"  VCP-Signale (roh, ohne Breadth): {n_sig:,}")
    print(f"  Marktbreite heute:               "
          f"{breadth.iloc[-1]:.1%}  (Max: {breadth.max():.1%}  "
          f"Min: {breadth.min():.1%})")

    # Breadth-Statistik
    print(f"\n  Signale nach Breadth-Filter:")
    for thr in BREADTH_THRESHOLDS:
        if thr == 0.0:
            allowed_days = breadth.index
        else:
            allowed_days = breadth[breadth >= thr].index
        # Signale an erlaubten Tagen
        allowed_sigs = int(
            pivots["entry_sig"]
            .loc[pivots["entry_sig"].index.isin(allowed_days)]
            .fillna(False).values.sum()
        )
        pct = allowed_sigs / n_sig * 100 if n_sig > 0 else 0
        label = f">={thr:.0%}" if thr > 0 else "aus "
        print(f"    Breadth {label}:  {allowed_sigs:>5} Signale  "
              f"({pct:>4.0f}% des Roh-Pools)")
    print(f"  Pivot-Aufbau: {time.time()-t0:.1f}s")

    # 3. Grid Search
    print(f"\n[3/4] Grid Search ({n_combos} Kombinationen)...")
    t0      = time.time()
    results = []
    done    = 0

    for breadth_thr, rot_factor in itertools.product(
            BREADTH_THRESHOLDS, ROTATION_FACTORS):
        equity, trades, invest_days = run_backtest(
            pivots, breadth, breadth_thr, rot_factor)
        m = compute_metrics(equity, trades, invest_days)
        results.append({
            "Breadth": breadth_thr,
            "RotFak":  rot_factor,
            **m,
        })
        done += 1
        if done % 5 == 0:
            elapsed   = time.time() - t0
            eta       = elapsed / done * (n_combos - done)
            print(f"  {done:>2}/{n_combos}  ({done/n_combos*100:.0f}%)  "
                  f"Zeit: {elapsed:.1f}s  ETA: {eta:.1f}s",
                  flush=True)

    elapsed_total = time.time() - t0
    print(f"  Fertig: {n_combos} Simulationen in {elapsed_total:.1f}s  "
          f"({elapsed_total/n_combos*1000:.0f}ms/Simulation)")

    # 4. Ergebnisse
    df = pd.DataFrame(results)
    print_results(df, args.top)

    # CSV
    csv_path = _here / "v82_gridsearch_results.csv"
    df.sort_values("Rendite_%", ascending=False).to_csv(csv_path, index=False)
    print(f"  Vollständige Ergebnisse: {csv_path}\n")


if __name__ == "__main__":
    main()
