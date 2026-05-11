"""
find_sweetspot_v7.py
====================================================================================
Alpha Research: Breakout Wave Quality Analysis  |  v7  |  260 US-Aktien

Fragestellung:
    Wie "intensiv" muss ein Breakout_50 sein, damit die Welle wirklich trägt?
    Welche Kombination aus Amplitude, Volumen und Trendstärke (ADX) liefert
    den höchsten Profit Factor – also die "saubersten" Wellen?

Methodik:
    Standalone-Simulator (kein Portfolio-Management):
    → Jede Aktie wird unabhängig betrachtet.
    → Keine Positions-Begrenzung, kein Sizing, kein Rebalancing.
    → Gibt die reine SIGNAL-QUALITÄT wieder, unabhängig von Kapitalstruktur.

    Entry:  Breakout_50 + optionaler Intensitäts-Filter
    Exit:   Asymmetrischer ATR-Trail (2.0× Tight → 3.5× Earned)
            + optionaler Stall-Stop (N Tage im Minus → Exit)

Test-Matrix (4 × 4 × 4 = 64 Kombinationen):
    Variante A – Amplitude:     kein / ≥1% / ≥3% / ≥5% Tagesrendite
    Variante B – Volumen-Kraft: kein / ≥1.5× / ≥2.0× / ≥3.0× SMA_Vol_20
    Variante C – ADX:           kein / ≥20 / ≥25 / ≥30

Key Metric: Profit Factor = Bruttogewinn / |Bruttoverlust|
            (signal-qualität unabhängig von Gebühren)

Ausgabe:
    - Vollständige Rangliste (sortiert nach Profit Factor)
    - 4-Subplot Matplotlib-Heatmap:
        [1] Profit Factor Heatmap  (Amp × Vol)
        [2] Net Return Heatmap     (Amp × Vol)
        [3] ADX-Impact Barplot
        [4] Stall-Stop-Vergleich

Verwendung:
    python find_sweetspot_v7.py
    python find_sweetspot_v7.py --no-stall     # ohne Stall-Stop
    python find_sweetspot_v7.py --stall-days 3 # engerer Time-Stop
    python find_sweetspot_v7.py --years 5
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
matplotlib.use("Agg")   # kein Fenster – speichert als Datei

# ── Konstanten ───────────────────────────────────────────────────────────────
_here          = Path(__file__).parent
_RAW_DIR       = _here / "data" / "raw"
DEFAULT_YEARS  = 7.0
ATR_INIT       = 2.0      # Phase 1: enger Stop (muss Profit beweisen)
ATR_TRAIL      = 3.5      # Phase 2: weiter Earned-Trail
STALL_DAYS_DEF = 5        # Tage ohne Profit → Exit
MAX_HOLD_DAYS  = 252      # maximale Haltedauer (1 Jahr Sicherheitsnetz)
POSITION_SIZE  = 2_000.0  # € pro Trade (10 000 / 5 Slots)
ORDER_FEE      = 20.0     # € pro Order (Round-Trip = 40 €)
FEE_IMPACT_PCT = ORDER_FEE * 2 / POSITION_SIZE * 100  # ~2 % round-trip

# Test-Achsen
AMP_THRESHOLDS = [None, 0.01, 0.03, 0.05]   # None = kein Filter
VOL_MULTIPLIERS= [None, 1.5,  2.0,  3.0]
ADX_THRESHOLDS = [None, 20,   25,   30]

AMP_LABELS = ["kein", "≥1%", "≥3%", "≥5%"]
VOL_LABELS = ["kein", "≥1.5×", "≥2.0×", "≥3.0×"]
ADX_LABELS = ["kein ADX", "ADX≥20", "ADX≥25", "ADX≥30"]


# ── Ticker-Liste + Lade-Logik aus backtest_v6 importieren ───────────────────
from backtest_v6 import _load_tickers


# ==============================================================================
# 1. DATEN LADEN  (OHLCV inkl. Volume wenn verfügbar)
# ==============================================================================

def load_data(years: float) -> dict[str, pd.DataFrame]:
    tickers = _load_tickers()
    if not tickers:
        raise RuntimeError("Keine Ticker aus sector_map.json geladen.")
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=int(years * 365.25))
    files  = sorted(_RAW_DIR.glob("*_1d.parquet"))
    tset   = set(tickers)
    data: dict[str, pd.DataFrame] = {}
    for fpath in files:
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
            needed = [c for c in ["open","high","low","close","volume"]
                      if c in df.columns]
            if not {"open","high","low","close"}.issubset(df.columns):
                continue
            data[ticker] = df[needed]
        except Exception:
            pass
    return data


# ==============================================================================
# 2. INDIKATOREN
# ==============================================================================

def _atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _adx_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index (ADX) – reine Pandas-Implementierung."""
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)

    up_move   = h.diff()
    down_move = -l.diff()

    plus_dm  = np.where((up_move > down_move) & (up_move > 0),  up_move,   0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr_s    = pd.Series(tr.values, index=h.index).ewm(
                    span=period, adjust=False).mean()
    plus_di  = 100 * pd.Series(plus_dm, index=h.index).ewm(
                    span=period, adjust=False).mean() / atr_s
    minus_di = 100 * pd.Series(minus_dm, index=h.index).ewm(
                    span=period, adjust=False).mean() / atr_s

    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx


# ==============================================================================
# 3. INDIKATOREN PRO TICKER VORBERECHNEN  (einmalig!)
# ==============================================================================

class TickerCache:
    """Hält alle vorberechneten Zeitreihen für einen Ticker im Speicher."""
    __slots__ = (
        "df", "close", "high", "low", "open_",
        "atr14", "adx14", "vol_ratio", "day_ret",
        "high50_prev",
    )

    def __init__(self, df: pd.DataFrame) -> None:
        c   = df["close"]
        h   = df["high"]
        vol = df.get("volume")

        self.df         = df
        self.close      = c
        self.high       = h
        self.low        = df["low"]
        self.open_      = df["open"]
        self.atr14      = _atr_series(df, 14)
        self.adx14      = _adx_series(df, 14)
        self.day_ret    = c.pct_change()
        self.high50_prev= h.shift(1).rolling(50).max()

        if vol is not None:
            self.vol_ratio = vol / vol.rolling(20).mean()
        else:
            self.vol_ratio = pd.Series(np.nan, index=c.index)


def build_cache(data: dict[str, pd.DataFrame]) -> dict[str, TickerCache]:
    return {ticker: TickerCache(df) for ticker, df in data.items()}


# ==============================================================================
# 4. ENTRY-SIGNAL BERECHNEN  (für eine Kombination)
# ==============================================================================

def compute_entry_signal(
    cache:     TickerCache,
    amp_thr:   float | None,
    vol_mult:  float | None,
    adx_thr:   float | None,
) -> pd.Series:
    """
    Gibt Boolean-Series zurück: True am Tag des (gefilterten) Breakout_50.
    Transition: nur erster True-Tag nach False-Phase.
    """
    c = cache.close
    # Basis-Trigger: Close kreuzt 50-Tage-High nach oben
    raw = c > cache.high50_prev

    # Amplitude-Filter (Tagesrendite auf Ausbruchstag)
    if amp_thr is not None:
        raw = raw & (cache.day_ret >= amp_thr)

    # Volumen-Filter
    if vol_mult is not None:
        raw = raw & (cache.vol_ratio >= vol_mult)

    # ADX-Filter
    if adx_thr is not None:
        raw = raw & (cache.adx14 >= adx_thr)

    # Transition: nur erste True-Tage (False → True)
    return raw & ~raw.shift(1).fillna(False)


# ==============================================================================
# 5. EINZELTRADE-SIMULATION
# ==============================================================================

def simulate_trade(
    cache:      TickerCache,
    sig_idx:    int,
    stall_days: int | None,
) -> dict | None:
    """Simuliert einen einzelnen Trade ab Signal-Index. Gibt Trade-Dict zurück."""
    dates = cache.close.index
    entry_idx = sig_idx + 1
    if entry_idx >= len(dates) - 1:
        return None

    entry_price = float(cache.open_.iloc[entry_idx])
    atr_e       = float(cache.atr14.iloc[entry_idx])

    if not np.isfinite(entry_price) or not np.isfinite(atr_e) or atr_e <= 0:
        return None

    trailing_stop = entry_price - ATR_INIT * atr_e
    max_high      = float(cache.high.iloc[entry_idx])
    earned_mode   = False
    exit_idx      = None
    exit_reason   = "MAX_HOLD"

    for i in range(entry_idx + 1, min(entry_idx + MAX_HOLD_DAYS + 1, len(dates))):
        today_c   = float(cache.close.iloc[i])
        today_h   = float(cache.high.iloc[i])
        today_atr = float(cache.atr14.iloc[i])
        days_held = i - entry_idx

        if not np.isfinite(today_c):
            continue

        # Max-High nachziehen
        if today_h > max_high:
            max_high = today_h

        # Earned-Mode prüfen
        if not earned_mode and max_high >= entry_price + ATR_INIT * atr_e:
            earned_mode = True

        # Trailing-Stop aktualisieren (niemals senken)
        if np.isfinite(today_atr) and today_atr > 0:
            new_stop = (max_high - ATR_TRAIL * today_atr if earned_mode
                        else entry_price - ATR_INIT * atr_e)
            trailing_stop = max(trailing_stop, new_stop)

        # ── Exits ────────────────────────────────────────────────────────────
        if today_c < trailing_stop:
            exit_idx    = i + 1
            exit_reason = "ATR_STOP"
            break

        if (stall_days is not None
                and days_held >= stall_days
                and today_c < entry_price):
            exit_idx    = i + 1
            exit_reason = "STALL_STOP"
            break

    if exit_idx is None:
        exit_idx = min(entry_idx + MAX_HOLD_DAYS, len(dates) - 1)

    if exit_idx >= len(dates):
        return None

    exit_price = float(cache.open_.iloc[exit_idx])
    if not np.isfinite(exit_price) or exit_price <= 0:
        return None

    days_held  = exit_idx - entry_idx
    ret_gross  = (exit_price - entry_price) / entry_price * 100
    ret_net    = ret_gross - FEE_IMPACT_PCT

    return {
        "entry_date":  dates[entry_idx],
        "exit_date":   dates[exit_idx],
        "entry_price": entry_price,
        "exit_price":  exit_price,
        "days_held":   days_held,
        "ret_gross_%": ret_gross,
        "ret_net_%":   ret_net,
        "earned_mode": earned_mode,
        "exit_reason": exit_reason,
    }


# ==============================================================================
# 6. EINE KOMBINATION VOLLSTÄNDIG SIMULIEREN
# ==============================================================================

def run_combination(
    cache_map:  dict[str, TickerCache],
    amp_thr:    float | None,
    vol_mult:   float | None,
    adx_thr:    float | None,
    stall_days: int | None,
) -> list[dict]:
    """Simuliert alle Trades für alle Ticker für eine Kombination."""
    all_trades: list[dict] = []

    for ticker, cache in cache_map.items():
        sig = compute_entry_signal(cache, amp_thr, vol_mult, adx_thr)
        sig_dates = sig[sig].index
        if len(sig_dates) == 0:
            continue

        dates      = cache.close.index
        in_trade   = False
        exit_date_ = pd.Timestamp.min

        for sd in sig_dates:
            if in_trade and sd <= exit_date_:
                continue   # Kein neuer Trade während offenem Trade

            sig_idx = dates.get_loc(sd)
            trade   = simulate_trade(cache, sig_idx, stall_days)
            if trade is None:
                continue

            trade["ticker"] = ticker
            all_trades.append(trade)
            in_trade   = True
            exit_date_ = trade["exit_date"]

    return all_trades


# ==============================================================================
# 7. METRIKEN AUS TRADE-LISTE
# ==============================================================================

def compute_metrics(trades: list[dict]) -> dict:
    if not trades:
        return {k: np.nan for k in [
            "n_trades","hit_%","payoff","profit_factor",
            "avg_win_%","avg_loss_%","avg_hold_d",
            "max_win_%","max_loss_%",
            "gross_ret_%","net_ret_%","earned_%",
        ]}

    rets_g  = np.array([t["ret_gross_%"] for t in trades])
    rets_n  = np.array([t["ret_net_%"]   for t in trades])
    wins_g  = rets_g[rets_g > 0]
    loss_g  = rets_g[rets_g < 0]
    holds   = [t["days_held"] for t in trades]
    earned  = [t["earned_mode"] for t in trades]

    gross_win  = wins_g.sum() if len(wins_g) else 0.0
    gross_loss = abs(loss_g.sum()) if len(loss_g) else 1e-9
    pf         = gross_win / gross_loss if gross_loss > 0 else np.inf

    avg_win  = wins_g.mean() if len(wins_g) else 0.0
    avg_loss = loss_g.mean() if len(loss_g) else 0.0
    payoff   = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf

    return {
        "n_trades":      len(trades),
        "hit_%":         (rets_g > 0).mean() * 100,
        "payoff":        payoff,
        "profit_factor": pf,
        "avg_win_%":     avg_win,
        "avg_loss_%":    avg_loss,
        "avg_hold_d":    np.mean(holds),
        "max_win_%":     rets_g.max(),
        "max_loss_%":    rets_g.min(),
        "gross_ret_%":   rets_g.sum(),          # kumuliert
        "net_ret_%":     rets_n.sum(),           # kumuliert
        "earned_%":      np.mean(earned) * 100,
    }


def combo_label(amp, vol, adx) -> str:
    parts = []
    if amp  is not None: parts.append(f"Amp≥{amp*100:.0f}%")
    if vol  is not None: parts.append(f"Vol≥{vol:.1f}×")
    if adx  is not None: parts.append(f"ADX≥{adx}")
    return " + ".join(parts) if parts else "Baseline"


# ==============================================================================
# 8. HEATMAP & PLOTS
# ==============================================================================

def make_heatmap_data(
    results: list[dict],
    metric:  str,
    adx_val: float | None,
) -> np.ndarray:
    """Extrahiert 4×4 Matrix (Amp × Vol) für einen ADX-Wert."""
    mat = np.full((4, 4), np.nan)
    for row in results:
        if row["adx"] != adx_val:
            continue
        ai = AMP_THRESHOLDS.index(row["amp"])
        vi = VOL_MULTIPLIERS.index(row["vol"])
        mat[vi, ai] = row[metric]
    return mat


def plot_results(results: list[dict], stall_days: int | None) -> str:
    """Erzeugt eine 2×2-Heatmap-Figure und gibt Pfad zurück."""
    stall_tag = f"stall{stall_days}d" if stall_days else "nostall"
    out_path  = str(_here / f"sweetspot_heatmap_{stall_tag}.png")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Breakout_50 Wave Quality  |  "
        f"ATR-Stop {ATR_INIT}×→{ATR_TRAIL}×  |  "
        f"{'Stall-Stop: ' + str(stall_days) + 'd' if stall_days else 'Ohne Stall-Stop'}",
        fontsize=14, fontweight="bold"
    )

    def _heatmap(ax, matrix, title, fmt, cmap, vmin=None, vmax=None):
        im = ax.imshow(matrix, cmap=cmap, aspect="auto",
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(range(4)); ax.set_xticklabels(AMP_LABELS, fontsize=9)
        ax.set_yticks(range(4)); ax.set_yticklabels(VOL_LABELS, fontsize=9)
        ax.set_xlabel("Amplitude-Filter (Tagesrendite am Ausbruchstag)", fontsize=9)
        ax.set_ylabel("Volumen-Filter (× SMA_Vol_20)", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.8)
        for vi in range(4):
            for ai in range(4):
                val = matrix[vi, ai]
                if np.isfinite(val):
                    color = "white" if abs(val) > (vmax or 1) * 0.6 else "black"
                    ax.text(ai, vi, fmt.format(val),
                            ha="center", va="center", fontsize=8,
                            color=color, fontweight="bold")

    # ── Plot 1: Profit Factor (kein ADX) ─────────────────────────────────────
    pf_mat = make_heatmap_data(results, "profit_factor", None)
    pf_mat_capped = np.clip(pf_mat, 0, 5)
    _heatmap(axes[0, 0], pf_mat_capped,
             "Profit Factor  (Brutto, kein ADX-Filter)",
             "{:.2f}", "RdYlGn", vmin=0.5, vmax=3.5)

    # ── Plot 2: Net Return (kein ADX) ────────────────────────────────────────
    nr_mat  = make_heatmap_data(results, "net_ret_%", None)
    vmax_nr = max(abs(np.nanmin(nr_mat)), abs(np.nanmax(nr_mat)))
    _heatmap(axes[0, 1], nr_mat,
             "Kumulierte Netto-Rendite %  (kein ADX-Filter)",
             "{:+.0f}%", "RdYlGn", vmin=-vmax_nr, vmax=vmax_nr)

    # ── Plot 3: ADX-Impact (Baseline = kein Amp/Vol Filter) ──────────────────
    ax3  = axes[1, 0]
    adx_pf   = []
    adx_ntrd = []
    for adx_val in ADX_THRESHOLDS:
        row = next((r for r in results
                    if r["amp"] is None and r["vol"] is None
                    and r["adx"] == adx_val), None)
        adx_pf.append(row["profit_factor"] if row else np.nan)
        adx_ntrd.append(row["n_trades"]    if row else 0)

    x    = range(4)
    bars = ax3.bar(x, adx_pf, color=["#2196F3","#4CAF50","#FF9800","#F44336"],
                   alpha=0.85, edgecolor="black")
    ax3.set_xticks(x); ax3.set_xticklabels(ADX_LABELS, fontsize=9)
    ax3.set_ylabel("Profit Factor", fontsize=9)
    ax3.set_title("ADX-Filter Impact  (kein Amp/Vol Filter)", fontsize=11,
                  fontweight="bold")
    ax3.axhline(1.0, color="red", linestyle="--", linewidth=1, label="Break-Even PF=1")
    ax3.legend(fontsize=8)
    for bar, pf_val, nt in zip(bars, adx_pf, adx_ntrd):
        if np.isfinite(pf_val):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                     f"PF={pf_val:.2f}\n({nt:,} Trades)",
                     ha="center", va="bottom", fontsize=8)

    # ── Plot 4: Stall-Stop Exit-Grund Breakdown ──────────────────────────────
    ax4   = axes[1, 1]
    # Zeige Top-8 Kombis nach PF mit ihrer Stall-Stop-Rate
    top_combos = sorted(
        [r for r in results if r["adx"] is None],
        key=lambda r: r["profit_factor"],
        reverse=True
    )[:8]
    labels_c  = [combo_label(r["amp"], r["vol"], r["adx"]) for r in top_combos]
    pf_vals   = [r["profit_factor"] for r in top_combos]
    ntrades   = [r["n_trades"]       for r in top_combos]
    stall_pct = [r.get("stall_exit_%", 0) for r in top_combos]

    x4 = range(len(top_combos))
    ax4.bar(x4, pf_vals, color="#42A5F5", alpha=0.85, edgecolor="black",
            label="Profit Factor")
    ax4.set_xticks(x4)
    ax4.set_xticklabels(labels_c, rotation=30, ha="right", fontsize=8)
    ax4.set_ylabel("Profit Factor", fontsize=9)
    ax4.set_title("Top-8 Kombis (kein ADX)  |  Profit Factor + Stall-Exit-%",
                  fontsize=11, fontweight="bold")
    ax4b = ax4.twinx()
    ax4b.plot(x4, stall_pct, "ro-", linewidth=1.5, markersize=6,
              label="Stall-Stop-Rate %")
    ax4b.set_ylabel("Stall-Stop-Exit in % aller Trades", fontsize=9, color="red")
    ax4b.tick_params(axis="y", colors="red")
    ax4.legend(loc="upper right", fontsize=8)
    ax4b.legend(loc="upper left", fontsize=8)

    for xi, (pf_v, nt) in enumerate(zip(pf_vals, ntrades)):
        ax4.text(xi, pf_v + 0.02, f"{nt:,}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    return out_path


# ==============================================================================
# 9. ERGEBNIS-TABELLEN DRUCKEN
# ==============================================================================

def print_results(all_rows: list[dict], top_n: int = 20) -> None:
    df = pd.DataFrame(all_rows)
    df = df.sort_values("profit_factor", ascending=False).reset_index(drop=True)

    print(f"\n{'=' * 120}")
    print(f"  SWEETSPOT ANALYSE  |  Breakout_50 + Intensitäts-Filter  |  "
          f"ATR-Stop {ATR_INIT}×→{ATR_TRAIL}×  |  {len(df)} Kombinationen")
    print(f"{'=' * 120}")
    print(f"  {'Rg':>3}  {'Kombination':<30}  {'PF':>6}  {'Hit%':>6}  "
          f"{'Payoff':>7}  {'Trades':>6}  {'AvgWin%':>8}  {'AvgLoss%':>9}  "
          f"{'Hold-d':>7}  {'MaxWin%':>8}  {'NetRet%':>8}  {'Earned%':>8}  "
          f"{'Stall%':>7}")
    sep = f"  {'─' * 114}"
    print(sep)

    for rank, row in df.head(top_n).iterrows():
        marker = " ★" if rank == 0 else "  "
        lbl    = combo_label(row["amp"], row["vol"], row["adx"])
        if len(lbl) > 30:
            lbl = lbl[:28] + ".."
        pf_str = f"{row['profit_factor']:.2f}" if np.isfinite(row['profit_factor']) else "∞"
        print(f"  {rank+1:>3}{marker}  {lbl:<30}  {pf_str:>6}  "
              f"{row['hit_%']:>5.1f}%  "
              f"{row['payoff']:>7.2f}  "
              f"{int(row['n_trades']):>6}  "
              f"{row['avg_win_%']:>+7.2f}%  "
              f"{row['avg_loss_%']:>+8.2f}%  "
              f"{row['avg_hold_d']:>6.1f}d  "
              f"{row['max_win_%']:>+7.1f}%  "
              f"{row['net_ret_%']:>+7.0f}%  "
              f"{row['earned_%']:>7.1f}%  "
              f"{row.get('stall_exit_%', 0):>6.1f}%")

    print(sep)

    # ── Zusammenfassung nach Filter-Typ ─────────────────────────────────────
    print(f"\n  PROFIT FACTOR NACH FILTER-TYP:")
    print(f"  {'─' * 90}")

    print(f"  {'Amplitude-Filter':}")
    for i, (thr, lbl) in enumerate(zip(AMP_THRESHOLDS, AMP_LABELS)):
        grp = df[df["amp"] == thr] if thr is not None else df[df["amp"].isna()]
        if len(grp) == 0:
            grp = df[df["amp"].apply(lambda x: x is None if thr is None else x == thr)]
        avg_pf = grp["profit_factor"].mean()
        best   = grp.loc[grp["profit_factor"].idxmax()]
        print(f"    {lbl:<8}  avg PF: {avg_pf:>5.2f}  "
              f"| Best PF: {best['profit_factor']:>5.2f}  "
              f"({combo_label(best['amp'], best['vol'], best['adx'])})"
              f"  | Trades: {grp['n_trades'].mean():.0f} avg")

    print(f"\n  {'Volumen-Filter':}")
    for i, (mult, lbl) in enumerate(zip(VOL_MULTIPLIERS, VOL_LABELS)):
        grp = df[df["vol"].apply(lambda x: x is None if mult is None else x == mult)]
        if len(grp) == 0:
            continue
        avg_pf = grp["profit_factor"].mean()
        best   = grp.loc[grp["profit_factor"].idxmax()]
        print(f"    {lbl:<8}  avg PF: {avg_pf:>5.2f}  "
              f"| Best PF: {best['profit_factor']:>5.2f}  "
              f"({combo_label(best['amp'], best['vol'], best['adx'])})"
              f"  | Trades: {grp['n_trades'].mean():.0f} avg")

    print(f"\n  {'ADX-Filter':}")
    for i, (thr, lbl) in enumerate(zip(ADX_THRESHOLDS, ADX_LABELS)):
        grp = df[df["adx"].apply(lambda x: x is None if thr is None else x == thr)]
        if len(grp) == 0:
            continue
        avg_pf = grp["profit_factor"].mean()
        best   = grp.loc[grp["profit_factor"].idxmax()]
        print(f"    {lbl:<10}  avg PF: {avg_pf:>5.2f}  "
              f"| Best PF: {best['profit_factor']:>5.2f}  "
              f"({combo_label(best['amp'], best['vol'], best['adx'])})"
              f"  | Trades: {grp['n_trades'].mean():.0f} avg")

    print()


# ==============================================================================
# 10. MAIN
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweetspot Alpha Research v7  |  Breakout_50 Intensitäts-Matrix")
    parser.add_argument("--years",      type=float, default=DEFAULT_YEARS)
    parser.add_argument("--stall-days", type=int,   default=STALL_DAYS_DEF,
                        help="Tage bis Stall-Stop greift (0 = deaktivieren)")
    parser.add_argument("--no-stall",   action="store_true",
                        help="Stall-Stop vollständig deaktivieren")
    parser.add_argument("--top",        type=int,   default=20)
    args = parser.parse_args()

    stall = None if (args.no_stall or args.stall_days == 0) else args.stall_days

    print("=" * 70)
    print("  SWEETSPOT ALPHA RESEARCH v7  |  Breakout_50 Intensitäts-Matrix")
    print("=" * 70)
    print(f"""
  Simulator:     Standalone (kein Portfolio-Management)
  Strategie:     Breakout_50  +  variable Intensitäts-Filter
  ATR-Stop:      {ATR_INIT}× Tight (Phase 1) → {ATR_TRAIL}× Earned (Phase 2)
  Stall-Stop:    {'DEAKTIVIERT' if stall is None else f'{stall} Tage im Minus → Exit'}
  Position:      {POSITION_SIZE:,.0f}€ (Gebühren: {ORDER_FEE:.0f}€ IN + {ORDER_FEE:.0f}€ OUT = {FEE_IMPACT_PCT:.1f}% Impact)
  Datenzeitraum: {args.years:.0f} Jahre

  Test-Matrix ({len(AMP_THRESHOLDS)}×{len(VOL_MULTIPLIERS)}×{len(ADX_THRESHOLDS)} = {len(AMP_THRESHOLDS)*len(VOL_MULTIPLIERS)*len(ADX_THRESHOLDS)} Kombinationen):
    Amplitude:  {AMP_LABELS}
    Volumen:    {VOL_LABELS}
    ADX:        {ADX_LABELS}
""")

    # 1. Daten laden
    print("[1/4] Lade Daten...")
    t0   = time.time()
    data = load_data(args.years)
    has_vol = any("volume" in df.columns for df in data.values())
    print(f"  {len(data)} Ticker in {time.time()-t0:.1f}s  "
          f"| Volume: {'vorhanden ✓' if has_vol else 'fehlt ✗'}")
    if not has_vol:
        print("  [WARN] Keine Volume-Daten – Vol-Filter immer False!")

    # 2. Indikatoren vorberechnen
    print("\n[2/4] Indikatoren vorberechnen (ATR14, ADX14, Vol-Ratio)...")
    t0      = time.time()
    cache_m = build_cache(data)
    years_a = (next(iter(data.values())).index[-1]
               - next(iter(data.values())).index[0]).days / 365.25
    print(f"  {len(cache_m)} Ticker-Caches in {time.time()-t0:.1f}s  "
          f"| Zeitraum: {years_a:.1f} Jahre")

    # 3. Alle Kombinationen simulieren
    n_combos = len(AMP_THRESHOLDS) * len(VOL_MULTIPLIERS) * len(ADX_THRESHOLDS)
    print(f"\n[3/4] {n_combos} Kombinationen simulieren...")
    t0     = time.time()
    rows   = []
    done   = 0

    for amp in AMP_THRESHOLDS:
        for vol in VOL_MULTIPLIERS:
            for adx in ADX_THRESHOLDS:
                trades = run_combination(cache_m, amp, vol, adx, stall)
                m      = compute_metrics(trades)

                # Stall-Stop-Quote berechnen
                n_stall = sum(1 for t in trades
                              if t.get("exit_reason") == "STALL_STOP")
                stall_pct = n_stall / len(trades) * 100 if trades else 0.0

                rows.append({
                    "amp":            amp,
                    "vol":            vol,
                    "adx":            adx,
                    "kombination":    combo_label(amp, vol, adx),
                    **m,
                    "stall_exit_%":   stall_pct,
                })
                done += 1
                if done % 16 == 0:
                    elapsed   = time.time() - t0
                    eta       = elapsed / done * (n_combos - done)
                    print(f"  {done:>3}/{n_combos}  "
                          f"({done/n_combos*100:>4.1f}%)  "
                          f"Zeit bisher: {elapsed:.1f}s  ETA: {eta:.1f}s",
                          flush=True)

    elapsed_total = time.time() - t0
    print(f"  {n_combos} Kombinationen in {elapsed_total:.1f}s  "
          f"({elapsed_total/n_combos*1000:.0f}ms / Kombi)")

    # 4. Ausgabe
    print("\n[4/4] Ergebnisse...")
    print_results(rows, args.top)

    # Heatmap
    heatmap_path = plot_results(rows, stall)
    print(f"  Heatmap gespeichert: {heatmap_path}")

    # CSV
    df_out = pd.DataFrame(rows).sort_values("profit_factor", ascending=False)
    csv_path = _here / f"sweetspot_results_{('stall' + str(stall)) if stall else 'nostall'}.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"  CSV gespeichert:     {csv_path}")

    # Bestes Setup
    best = df_out.iloc[0]
    print(f"""
  FAZIT (Stall-Stop: {'AN, ' + str(stall) + 'd' if stall else 'AUS'}):
  {'─' * 70}
  Bester Sweetspot:  '{best['kombination']}'
  Profit Factor:      {best['profit_factor']:.3f}
  Hit Rate:           {best['hit_%']:.1f}%  |  Payoff: {best['payoff']:.2f}
  Trades:             {int(best['n_trades'])}  |  Avg Hold: {best['avg_hold_d']:.1f}d
  Net Return (kum.):  {best['net_ret_%']:>+.1f}%
  Avg Win:           {best['avg_win_%']:>+.2f}%  |  Avg Loss: {best['avg_loss_%']:>+.2f}%
  Max Win:           {best['max_win_%']:>+.1f}%
  Earned-Mode:        {best['earned_%']:.1f}% der Trades erreichen Phase 2
  Stall-Stop-Exits:   {best['stall_exit_%']:.1f}%
""")


if __name__ == "__main__":
    main()
