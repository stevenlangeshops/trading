"""
run_sensitivity.py
══════════════════════════════════════════════════════════════════════════════
Systematische Sensitivitätsanalyse des Portfolio-Layers.

Prinzip (Zero Signal-Contamination):
  Phase 1 – Score-Cache (einmalig):
      Lädt die gespeicherten Fold-Checkpoints und berechnet für jeden
      Handelstag im Out-of-Sample-Zeitraum die LSTM-Scores einmalig.
      → score_cache: Dict[pd.Timestamp, pd.Series[asset, score]]

  Phase 2 – Grid Search (N_combinations × schnell):
      Führt die Portfolio-Simulation mit verschiedenen Parametern durch,
      ohne das Modell neu zu laden. Nur Execution-Logik variiert.

Grid (81 Kombinationen):
  n_max          [5, 7, 9]
  rotation_buffer [2, 3, 4]
  hard_stop_pct  [0.20, 0.25, 0.30]
  fees           [0.001, 0.0015, 0.002]

Verwendung:
  Lokal (nach Download des Archivs):
      python run_sensitivity.py \\
          --ckpt-dir  path/to/extracted/kaggle_artifacts \\
          --walk-json path/to/v2_7d_walk_forward.json \\
          --asset-map path/to/asset_map.json \\
          --data-dir  path/to/data/raw \\
          --output    sensitivity_results.csv

  In Kaggle (nach dem Training, in kaggle_full_run.py aufrufen):
      step_sensitivity(features, asset_map, all_train_results, price_cache)
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger


# ══════════════════════════════════════════════════════════════════════════════
# Datenstrukturen
# ══════════════════════════════════════════════════════════════════════════════

MEGA_CAP_7 = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA']


@dataclass
class PortfolioParams:
    """Alle variablen Portfolio-Parameter für eine Grid-Search-Zelle."""
    n_max:           int   = 7
    n_mid:           int   = 3       # fix: Bull/Bear-Skalierung relativ zu n_max
    n_min:           int   = 1       # fix
    rotation_buffer: int   = 3
    hard_stop_pct:   float = 0.20
    fees:            float = 0.001
    init_cash:       float = 10_000.0
    # Universum-Blacklist: diese Tickers werden beim Ranking übersprungen.
    # Existierende Positionen darin laufen bis Hard-Stop / Rotation aus.
    exclude_tickers: List[str] = field(default_factory=list)

    def label(self) -> str:
        base = (f"n{self.n_max}_rb{self.rotation_buffer}_"
                f"hs{int(self.hard_stop_pct*100)}_f{int(self.fees*10000)}")
        if self.exclude_tickers:
            base += f"_ex{len(self.exclude_tickers)}"
        return base


# Score-Cache-Typ: Datum → pd.Series (asset → score, absteigend)
ScoreCache = Dict[pd.Timestamp, pd.Series]
IC_WINDOWS = [5, 10, 15, 20, 30, 40, 50, 60]


# ══════════════════════════════════════════════════════════════════════════════
# Score-Cache persistieren / laden
# ══════════════════════════════════════════════════════════════════════════════

def save_score_cache(score_cache: ScoreCache, path: str) -> None:
    """
    Speichert den Score-Cache als Parquet-Datei.

    Format: DataFrame mit DatetimeIndex (Tage) und Asset-Spalten.
    NaN = kein Score für dieses Asset an diesem Tag.
    """
    df = pd.DataFrame(score_cache).T          # Transpose: index=Datum, cols=Assets
    df.index.name = 'date'
    df.sort_index(inplace=True)
    df.to_parquet(path, compression='zstd')
    size_mb = Path(path).stat().st_size / 1024 / 1024
    logger.success(f"Score-Cache gespeichert: {path}  ({size_mb:.1f} MB, "
                   f"{len(df)} Tage, {len(df.columns)} Assets)")


def load_score_cache(path: str) -> ScoreCache:
    """
    Lädt einen zuvor gespeicherten Score-Cache aus einer Parquet-Datei.
    Gibt ein dict[pd.Timestamp, pd.Series] zurück – identisch mit dem
    Rückgabewert von build_score_cache().
    """
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    # UTC-Timezone entfernen → alle date-Keys sind tz-naive (konsistent mit
    # build_score_cache() und allen Vergleichen im Backtest-Loop).
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    cache: ScoreCache = {}
    for ts, row in df.iterrows():
        valid = row.dropna()
        if not valid.empty:
            cache[ts] = valid
    logger.success(f"Score-Cache geladen: {path}  "
                   f"({len(cache)} Tage, {len(df.columns)} Assets)")
    return cache


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Score-Cache aus Checkpoints aufbauen
# ══════════════════════════════════════════════════════════════════════════════

def build_score_cache(
    features:     pd.DataFrame,
    fold_results: List[dict],
    asset_map:    Dict[str, int],
    seq_len:      int = 64,
    device:       Optional[str] = None,
) -> ScoreCache:
    """
    Lädt jeden Fold-Checkpoint einmalig und berechnet die Cross-Section-Scores
    für alle Out-of-Sample-Tage.

    Returns
    -------
    ScoreCache: {date: pd.Series[asset → score, absteigend sortiert]}
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Score-Cache aufbauen | Device={device} | {len(fold_results)} Folds")

    # Lazy import: vermeidet Zirkel beim Einbinden aus kaggle_full_run
    from models_v2_single_horizon import SingleHorizonRankModel
    from backtest_v2_single_horizon import load_fold_model, predict_cross_section

    cache: ScoreCache = {}
    all_dates = features.index.get_level_values('date').unique().sort_values()

    for fold in fold_results:
        ckpt_path = fold.get('ckpt_path', '')
        if not Path(ckpt_path).exists():
            logger.warning(f"  Checkpoint nicht gefunden: {ckpt_path} — Fold {fold.get('fold_id')} übersprungen")
            continue

        model, _ = load_fold_model(ckpt_path, device)

        val_start = pd.Timestamp(fold['val_start'])
        val_end   = pd.Timestamp(fold['val_end'])

        cmp = all_dates.tz_localize(None) if getattr(all_dates, 'tz', None) else all_dates
        vs  = val_start.tz_localize(None) if val_start.tzinfo else val_start
        ve  = val_end.tz_localize(None)   if val_end.tzinfo   else val_end
        fold_dates = all_dates[(cmp >= vs) & (cmp <= ve)]

        t0 = time.time()
        n_dates = 0
        for date in fold_dates:
            scores = predict_cross_section(model, features, asset_map, date, seq_len, device)
            if len(scores) >= 2:
                cache[date] = scores
                n_dates += 1

        logger.info(f"  Fold {fold['fold_id']:2d}: {vs.date()} → {ve.date()} "
                    f"| {n_dates} Tage | {time.time()-t0:.1f}s")

    # Modell aus GPU entladen
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    logger.success(f"Score-Cache fertig: {len(cache)} Tage mit Scores")
    return cache


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Portfolio-Simulation auf Score-Cache
# ══════════════════════════════════════════════════════════════════════════════

def _pos_val(positions: dict, price_cache: dict, date: pd.Timestamp) -> float:
    val = 0.0
    for asset, pos in positions.items():
        pr = _get_price_local(price_cache, asset, date)
        if pr is not None:
            val += pos['shares'] * pr
    return val


def _get_price_local(price_cache: dict, asset: str, date: pd.Timestamp) -> Optional[float]:
    """Preis für asset am/vor date aus dem price_cache."""
    series = price_cache.get(asset)
    if series is None:
        return None
    try:
        idx = series.index.searchsorted(date, side='right') - 1
        if idx < 0:
            return None
        return float(series.iloc[idx])
    except Exception:
        return None


def _get_regime(spy_prices: Optional[pd.Series], date: pd.Timestamp) -> str:
    """SMA50/SMA200-basiertes Marktregime auf SPY."""
    if spy_prices is None:
        return 'neutral'
    try:
        past = spy_prices[spy_prices.index <= date]
        if len(past) < 50:
            return 'neutral'
        sma50  = past.iloc[-50:].mean()
        sma200 = past.iloc[-200:].mean() if len(past) >= 200 else past.mean()
        price  = past.iloc[-1]
        if price > sma50 > sma200:
            return 'bull'
        if price > sma200:
            return 'neutral'
        return 'bear'
    except Exception:
        return 'neutral'


def _adaptive_n(regime: str, n_max: int, n_mid: int, n_min: int) -> int:
    return {'bull': n_max, 'neutral': n_mid, 'bear': n_min}.get(regime, n_mid)


# ══════════════════════════════════════════════════════════════════════════════
# Policy-Engine: IC- und SPY-basierte n_max-Reduktion
# ══════════════════════════════════════════════════════════════════════════════

POLICIES = ("IC20", "IC30", "IC40", "SPY200", "C_Budget")

def _strip_tz_index(idx: pd.Index) -> pd.Index:
    """Entfernt Timezone-Info aus einem DatetimeIndex (tz-aware → tz-naive)."""
    if hasattr(idx, 'tz') and idx.tz is not None:
        return idx.tz_localize(None)
    return idx


def _ic_df_lookup(ic_df: pd.DataFrame, date: pd.Timestamp, col: str) -> float:
    """Sicherer IC-Lookup mit Datum-Normalisierung (tz-strip, ffill)."""
    ts  = date.tz_localize(None) if (hasattr(date, 'tzinfo') and date.tzinfo) else date
    idx = _strip_tz_index(ic_df.index)
    if ts in idx:
        pos = idx.get_loc(ts)
        return float(ic_df.iloc[pos][col])
    # nächster verfügbarer Wert davor (forward-fill)
    before = idx[idx <= ts]
    if len(before) == 0:
        return float('nan')
    pos = idx.get_loc(before[-1])
    return float(ic_df.iloc[pos][col])


def get_effective_n_max(
    date:       pd.Timestamp,
    base_n_max: int,
    policy:     Optional[str],
    ic_df:      Optional[pd.DataFrame],
    spy_sma200: Optional[pd.Series],
    spy_prices: Optional[pd.Series],
    reduced_n:  int = 3,
) -> tuple[int, bool]:
    """
    Berechnet den effektiven n_max für einen Handelstag gemäß Policy.

    Parameters
    ----------
    date        : Handelstag
    base_n_max  : Standard n_max aus PortfolioParams (z.B. 7)
    policy      : None | "IC20" | "IC30" | "IC40" | "SPY200"
    ic_df       : DataFrame mit Spalten ic_roll_20, ic_roll_30, ic_roll_40
    spy_sma200  : pd.Series mit vorberechneter 200-Tage-SMA auf SPY
    spy_prices  : pd.Series mit SPY-Schlusskursen
    reduced_n   : n_max wenn Trigger aktiv (Default 3)

    Returns
    -------
    (n_max_eff, trigger_active)
    """
    if policy is None:
        return base_n_max, False

    if policy in ("IC20", "IC30", "IC40"):
        if ic_df is None:
            return base_n_max, False
        col_map = {"IC20": "ic_roll_20", "IC30": "ic_roll_30", "IC40": "ic_roll_40"}
        col = col_map[policy]
        if col not in ic_df.columns:
            return base_n_max, False
        val = _ic_df_lookup(ic_df, date, col)
        if np.isnan(val):
            return base_n_max, False
        triggered = val < 0
        return (reduced_n if triggered else base_n_max), triggered

    if policy == "SPY200":
        if spy_prices is None or spy_sma200 is None:
            return base_n_max, False
        ts = date.tz_localize(None) if (hasattr(date, 'tzinfo') and date.tzinfo) else date
        # spy_prices und spy_sma200 sind bereits tz-naive (normalisiert in run_portfolio)
        past_spy = spy_prices[spy_prices.index <= ts]
        if past_spy.empty:
            return base_n_max, False
        spy_close = float(past_spy.iloc[-1])
        past_sma  = spy_sma200[spy_sma200.index <= ts]
        if past_sma.empty or np.isnan(past_sma.iloc[-1]):
            return base_n_max, False
        sma_val   = float(past_sma.iloc[-1])
        triggered = spy_close < sma_val
        return (reduced_n if triggered else base_n_max), triggered

    return base_n_max, False


def get_budget_factor(
    date:   pd.Timestamp,
    policy: Optional[str],
    ic_df:  Optional[pd.DataFrame],
    step:   float = 0.30,
) -> tuple[float, bool]:
    """
    Gestaffelter Budget-Multiplikator für Policy "C_Budget".

    Für jedes negative Rolling-IC-Fenster (IC20, IC30, IC40) wird der
    investierbare Anteil um ``step`` (30 %) reduziert, kumulativ:

        Keine Stufe aktiv  → 1.00  (100 % investiert)
        IC20 < 0           → 0.70  ( 70 %)
        IC20 + IC30 < 0    → 0.49  ( 49 %)
        IC20+IC30+IC40 < 0 → 0.343 ( ~34 %)

    Nur aktiv wenn policy == "C_Budget", sonst stets (1.0, False).

    Returns
    -------
    (budget_factor, any_reduction_active)
    """
    if policy != "C_Budget" or ic_df is None:
        return 1.0, False

    factor    = 1.0
    triggered = False
    for col in ("ic_roll_20", "ic_roll_30", "ic_roll_40"):
        if col not in ic_df.columns:
            continue
        val = _ic_df_lookup(ic_df, date, col)
        if not np.isnan(val) and val < 0:
            factor   *= (1.0 - step)
            triggered = True

    return factor, triggered


def build_ic_df(daily_ic: pd.Series, rolling_map: dict) -> pd.DataFrame:
    """
    Baut ic_df aus compute_daily_ic() + rolling_ic_report() Ergebnis.

    Returns
    -------
    pd.DataFrame mit Index=date, Spalten: ic, ic_roll_5, ..., ic_roll_60
    """
    df = pd.DataFrame({'ic': daily_ic})
    for w, series in rolling_map.items():
        df[f'ic_roll_{w}'] = series
    df = df.sort_index()
    # Timezone entfernen damit Vergleiche mit tz-naive Score-Cache-Dates funktionieren
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def run_portfolio(
    score_cache: ScoreCache,
    price_cache: dict,
    params:      PortfolioParams,
    policy:      Optional[str]           = None,
    ic_df:       Optional[pd.DataFrame]  = None,
) -> dict:
    """
    Führt die Portfolio-Simulation auf dem Score-Cache durch.

    Keine Modell-Operationen – reine Execution-Logik.
    Identisch zur Run-G-Strategie: Long-Only, Rotation, Hard-Stop.

    Parameters
    ----------
    policy : None | "IC20" | "IC30" | "IC40" | "SPY200"
        Steuert ob und wie n_max täglich angepasst wird.
    ic_df  : DataFrame mit Spalten ic_roll_20/30/40 (für IC*-Policies)

    Returns
    -------
    dict mit Backtest-Metriken + equity/equity_dates + days_n_max_reduced.
    """
    spy_prices = price_cache.get('SPY')   # bereits tz-naive (aus build_price_cache_local)

    # SPY 200-Tage-SMA einmalig vorberechnen (für SPY200-Policy)
    spy_sma200: Optional[pd.Series] = None
    if policy == "SPY200" and spy_prices is not None:
        spy_clean  = spy_prices[~spy_prices.index.duplicated(keep='last')].sort_index()
        spy_sma200 = spy_clean.rolling(200, min_periods=100).mean()

    cash      = params.init_cash
    positions: dict = {}
    equity        = [params.init_cash]
    equity_dates  = []
    trade_log: list[dict] = []
    days_n_max_reduced = 0   # Tage, an denen Policy n_max oder Budget reduziert hat

    sorted_dates = sorted(score_cache.keys())

    def _close(asset, pos, date, reason, regime):
        nonlocal cash
        pr = _get_price_local(price_cache, asset, date)
        if pr is None:
            return
        proceeds = pos['shares'] * pr * (1 - params.fees)
        cash += proceeds
        pnl  = (pr - pos['entry']) / pos['entry']
        trade_log.append({
            'date':        str(date.date()),
            'asset':       asset,
            'pnl_pct':     round(pnl * 100, 3),
            'exit_reason': reason,
            'regime':      regime,
            'hold_days':   pos.get('hold_days', 0),
        })

    # Blacklist einmalig als Set vorberechnen (O(1) Lookup pro Tag)
    _blacklist: set = set(params.exclude_tickers) if params.exclude_tickers else set()

    for date in sorted_dates:
        ranking = score_cache[date]

        # Universum-Filter: Blacklisted Ticker aus dem Ranking entfernen.
        # Das Modell hat sie bereits gescort – wir ignorieren sie nur bei der
        # Portfolio-Konstruktion. Bestehende Positionen darin laufen normal aus.
        if _blacklist:
            ranking = ranking[~ranking.index.isin(_blacklist)]

        # Hold-Days inkrementieren
        for pos in positions.values():
            pos['hold_days'] = pos.get('hold_days', 0) + 1

        regime = _get_regime(spy_prices, date)

        # Policy: n_max ggf. reduzieren (A-Policies, SPY)  oder  Budget skalieren (C_Budget)
        budget_factor = 1.0
        if policy == "C_Budget":
            budget_factor, triggered = get_budget_factor(
                date=date, policy=policy, ic_df=ic_df)
            n_max_eff = params.n_max
        else:
            n_max_eff, triggered = get_effective_n_max(
                date=date, base_n_max=params.n_max, policy=policy,
                ic_df=ic_df, spy_sma200=spy_sma200, spy_prices=spy_prices,
            )
        if triggered:
            days_n_max_reduced += 1
        # n_mid / n_min proportional skalieren wenn n_max-Policy aktiv
        if triggered and policy != "C_Budget" and n_max_eff < params.n_max:
            scale  = n_max_eff / params.n_max
            n_mid_ = max(1, round(params.n_mid * scale))
            n_min_ = max(1, round(params.n_min * scale))
        else:
            n_mid_ = params.n_mid
            n_min_ = params.n_min

        n_long = _adaptive_n(regime, n_max_eff, n_mid_, n_min_)

        # Hard-Stop
        to_close = []
        for asset, pos in positions.items():
            pr = _get_price_local(price_cache, asset, date)
            if pr is None:
                continue
            if (pr - pos['entry']) / pos['entry'] <= -params.hard_stop_pct:
                to_close.append((asset, 'hard_stop'))
        for asset, reason in to_close:
            _close(asset, positions[asset], date, reason, regime)
            del positions[asset]

        # Rotation
        to_close = []
        for asset in list(positions.keys()):
            if asset in ranking.index:
                rank_pos = list(ranking.index).index(asset)
                if rank_pos >= n_long + params.rotation_buffer:
                    to_close.append((asset, 'rotation'))
            else:
                to_close.append((asset, 'rotation'))
        for asset, reason in to_close:
            _close(asset, positions[asset], date, reason, regime)
            del positions[asset]

        # Neue Positionen
        top_n      = list(ranking.index[:n_long])
        free_slots = n_long - len(positions)
        total_val  = cash + _pos_val(positions, price_cache, date)
        if free_slots > 0:
            for cand in top_n:
                if free_slots <= 0:
                    break
                if cand in positions:
                    continue
                pr = _get_price_local(price_cache, cand, date)
                if pr is None or pr <= 0:
                    continue
                alloc  = total_val * budget_factor / n_long
                shares = alloc * (1 - params.fees) / pr
                if shares * pr < 100:
                    continue
                cash -= shares * pr * (1 + params.fees)
                positions[cand] = {
                    'shares': shares, 'entry': pr, 'hold_days': 0,
                }
                free_slots -= 1

        eq = cash + _pos_val(positions, price_cache, date)
        equity.append(eq)
        equity_dates.append(date)

    # ── Metriken ────────────────────────────────────────────────────────────
    eq_arr = np.array(equity[1:]) if len(equity) > 1 else np.array([params.init_cash])
    total_return = (eq_arr[-1] / eq_arr[0] - 1) * 100 if eq_arr[0] > 0 else 0.0
    peaks  = np.maximum.accumulate(eq_arr)
    dd     = (eq_arr - peaks) / peaks
    max_dd = float(dd.min()) * 100 if len(dd) else 0.0

    rets   = np.diff(equity) / np.array(equity[:-1])
    rets   = rets[1:]
    sharpe = (float(np.mean(rets)) / float(np.std(rets)) * np.sqrt(252)
              if len(rets) > 1 and np.std(rets) > 0 else 0.0)

    n_trades  = len(trade_log)
    wins      = [t for t in trade_log if t['pnl_pct'] > 0]
    win_rate  = len(wins) / n_trades * 100 if n_trades else 0.0
    avg_hold  = float(np.mean([t['hold_days'] for t in trade_log])) if trade_log else 0.0

    stops     = [t for t in trade_log if t['exit_reason'] == 'hard_stop']

    return {
        'total_return':       round(total_return, 2),
        'max_drawdown':       round(max_dd, 2),
        'sharpe':             round(sharpe, 3),
        'n_trades':           n_trades,
        'win_rate':           round(win_rate, 1),
        'avg_hold_days':      round(avg_hold, 1),
        'n_hard_stops':       len(stops),
        'days_n_max_reduced': days_n_max_reduced,
        'equity':             equity,
        'equity_dates':       equity_dates,
        'trade_log':          trade_log,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Grid Search
# ══════════════════════════════════════════════════════════════════════════════

PARAM_GRID = {
    'n_max':           [5, 7, 9],
    'rotation_buffer': [2, 3, 4],
    'hard_stop_pct':   [0.20, 0.25, 0.30],
    'fees':            [0.001, 0.0015, 0.002],
}


def grid_search(
    score_cache: ScoreCache,
    price_cache: dict,
    param_grid:  Optional[dict] = None,
    base_params: Optional[PortfolioParams] = None,
    verbose:     bool = True,
) -> pd.DataFrame:
    """
    Führt Grid Search über Portfolio-Parameter durch.

    Parameters
    ----------
    score_cache : vorberechnete Scores (Phase 1)
    price_cache : Preise aller Assets
    param_grid  : Dict[param_name → list of values]; Default = PARAM_GRID
    base_params : Basiswerte für nicht-variierte Parameter
    verbose     : Fortschritts-Log

    Returns
    -------
    pd.DataFrame mit allen Kombinationen + Metriken, sortiert nach Sharpe desc.
    """
    if param_grid is None:
        param_grid = PARAM_GRID
    if base_params is None:
        base_params = PortfolioParams()

    keys   = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    n      = len(combos)

    logger.info(f"Grid Search: {n} Kombinationen ({' × '.join(str(len(param_grid[k])) for k in keys)})")
    logger.info(f"  Parameter: {keys}")

    rows = []
    t0   = time.time()

    for i, combo in enumerate(combos, 1):
        kw = dict(zip(keys, combo))

        # Basiswerte übernehmen, dann Grid-Werte überschreiben
        p = PortfolioParams(
            n_max=kw.get('n_max', base_params.n_max),
            n_mid=base_params.n_mid,
            n_min=base_params.n_min,
            rotation_buffer=kw.get('rotation_buffer', base_params.rotation_buffer),
            hard_stop_pct=kw.get('hard_stop_pct', base_params.hard_stop_pct),
            fees=kw.get('fees', base_params.fees),
            init_cash=base_params.init_cash,
        )

        result = run_portfolio(score_cache, price_cache, p)

        row = {
            **kw,
            'total_return_%': result['total_return'],
            'max_drawdown_%': result['max_drawdown'],
            'sharpe':         result['sharpe'],
            'n_trades':       result['n_trades'],
            'win_rate_%':     result['win_rate'],
            'avg_hold_days':  result['avg_hold_days'],
            'n_hard_stops':   result['n_hard_stops'],
        }
        rows.append(row)

        if verbose and i % 10 == 0:
            elapsed = time.time() - t0
            eta     = elapsed / i * (n - i)
            logger.info(f"  [{i:3d}/{n}] {p.label()} | "
                        f"Sharpe={result['sharpe']:.3f}  "
                        f"Return={result['total_return']:+.1f}%  "
                        f"MaxDD={result['max_drawdown']:.1f}%  "
                        f"ETA {eta:.0f}s")

    df = pd.DataFrame(rows).sort_values('sharpe', ascending=False).reset_index(drop=True)
    df.index += 1   # Rang 1-basiert

    elapsed = time.time() - t0
    logger.success(f"Grid Search abgeschlossen in {elapsed:.1f}s ({elapsed/n*1000:.0f}ms pro Lauf)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Hilfsfunktionen: Artefakte laden
# ══════════════════════════════════════════════════════════════════════════════

def load_walk_forward_json(path: str | Path) -> List[dict]:
    """Lädt fold_results aus v2_7d_walk_forward.json."""
    data = json.loads(Path(path).read_text())
    folds = data.get('fold_summary', data.get('fold_results', []))
    if not folds:
        raise ValueError(f"Keine Fold-Daten in {path}")
    logger.info(f"Walk-Forward geladen: {len(folds)} Folds aus {path}")
    return folds


def load_asset_map(path: str | Path) -> Dict[str, int]:
    """Lädt asset_map.json → {ticker: id}."""
    am = json.loads(Path(path).read_text())
    logger.info(f"Asset-Map geladen: {len(am)} Assets")
    return am


def build_features_from_parquet(
    data_dir: str | Path,
    sector_neutral: bool = False,
) -> pd.DataFrame:
    """
    Baut das Feature-Panel aus dem Parquet-Verzeichnis auf.
    Identisch zu step_build_panel() in kaggle_full_run.py.

    build_panel() liest RAW_DIR als Modul-Konstante, daher wird sie
    vor dem Import-/Aufruf-Zeitpunkt überschrieben.

    Parameters
    ----------
    sector_neutral : bool
        False = klassischer Cross-Sectional z-Score (Standard)
        True  = Sektor-neutraler z-Score (pro Tag und GICS-Sektor)
    """
    import features.engineer as eng
    eng.RAW_DIR = Path(data_dir)
    features, _ = eng.build_panel(
        timeframe="1d", horizon=11, min_rows=300,
        sector_neutral=sector_neutral,
    )
    return features


def build_price_cache_local(asset_map: Dict[str, int], data_dir: str | Path) -> dict:
    """
    Baut price_cache aus Parquet-Dateien.

    Normalisiert alle Series-Indices auf tz-naive (UTC → ohne Timezone),
    damit Vergleiche mit den tz-naiven Score-Cache-Keys überall funktionieren.
    """
    from strategy.backtest import build_price_cache
    assets = list(asset_map.keys())
    if 'SPY' not in assets:
        assets.append('SPY')
    raw = build_price_cache(assets, raw_dir=Path(data_dir))
    normalized: dict = {}
    for asset, ps in raw.items():
        if ps is not None and hasattr(ps.index, 'tz') and ps.index.tz is not None:
            ps = pd.Series(ps.values, index=ps.index.tz_localize(None), name=ps.name)
        normalized[asset] = ps
    return normalized


# ══════════════════════════════════════════════════════════════════════════════
# Tearsheet – Subperioden-Analyse + Rolling Rank-IC
# ══════════════════════════════════════════════════════════════════════════════

# Marktphasen mit ihrer Bedeutung
DEFAULT_PERIODS = [
    ("2020       (COVID-Crash + Erholung)", "2020-01-01", "2020-12-31"),
    ("2021       (Post-COVID Rallye)",       "2021-01-01", "2021-12-31"),
    ("2022       (Bärenmarkt / Zinsanstieg)","2022-01-01", "2022-12-31"),
    ("2023-2024  (KI-Rallye)",               "2023-01-01", "2024-12-31"),
    ("2025+      (Neuland)",                 "2025-01-01", "2099-12-31"),
]


def _period_metrics(equity: list, dates: list, start: str, end: str) -> Optional[dict]:
    """Berechnet Return / MaxDD / Sharpe für eine Zeitscheibe der Equity-Kurve."""
    ts  = pd.Timestamp(start)
    te  = pd.Timestamp(end)
    # Timestamps tz-normalisieren: falls dates tz-aware sind, tz entfernen
    def _strip_tz(t):
        return t.tz_localize(None) if (isinstance(t, pd.Timestamp) and t.tzinfo is not None) else t
    idx = [(i, d) for i, d in enumerate(dates) if ts <= _strip_tz(d) <= te]
    if len(idx) < 5:
        return None
    positions = [i for i, _ in idx]
    # Equity-Array: equity ist um 1 versetzt gegenüber equity_dates
    eq = np.array([equity[p + 1] for p in positions])
    daily_rets = np.diff(eq) / eq[:-1]
    ret    = (eq[-1] / eq[0] - 1) * 100
    peaks  = np.maximum.accumulate(eq)
    max_dd = float(((eq - peaks) / peaks).min()) * 100
    sharpe = (float(np.mean(daily_rets) / np.std(daily_rets)) * np.sqrt(252)
              if len(daily_rets) > 1 and np.std(daily_rets) > 0 else 0.0)
    return {'return': ret, 'max_dd': max_dd, 'sharpe': sharpe, 'n_days': len(idx)}


def subperiod_report(
    equity:       list,
    equity_dates: list,
    periods:      Optional[list] = None,
    label:        str = "",
) -> None:
    """
    Druckt Return / Max-DD / Sharpe für jede Marktphase isoliert.

    Parameters
    ----------
    equity       : Equity-Liste aus run_portfolio()
    equity_dates : Datumsliste aus run_portfolio()
    periods      : [(name, start, end), ...] – Default = DEFAULT_PERIODS
    label        : optionale Kopfzeile (z.B. Konfigurations-Label)
    """
    if periods is None:
        periods = DEFAULT_PERIODS

    hdr = f"SUBPERIODEN-ANALYSE{f'  [{label}]' if label else ''}"
    logger.info("─" * 72)
    logger.info(hdr)
    logger.info(f"  {'Phase':<42} {'Return':>8}  {'MaxDD':>7}  {'Sharpe':>7}  {'Tage':>5}")
    logger.info("  " + "─" * 65)

    for name, start, end in periods:
        m = _period_metrics(equity, equity_dates, start, end)
        if m is None:
            logger.info(f"  {name:<42} {'–':>8}  {'–':>7}  {'–':>7}  {'–':>5}")
        else:
            logger.info(f"  {name:<42} "
                        f"{m['return']:>+7.1f}%  "
                        f"{m['max_dd']:>6.1f}%  "
                        f"{m['sharpe']:>7.3f}  "
                        f"{m['n_days']:>5d}")

    logger.info("─" * 72)


def compute_daily_ic(
    score_cache:  ScoreCache,
    price_cache:  dict,
    horizon:      int = 7,
) -> pd.Series:
    """
    Berechnet den täglichen Rank-IC (Spearman) – vollständig vektorisiert.

    Strategie:
      1. Einmalig für alle Assets eine Preismatrix (date × asset) aufbauen.
      2. Forward-Return-Matrix per shift(-horizon) berechnen.
      3. Pro Tag: Spearman zwischen Modell-Scores und Forward-Returns.

    Deutlich schneller als der Asset-für-Asset Loop (~10–30s statt Minuten).
    """
    from scipy.stats import spearmanr

    logger.info("compute_daily_ic: Preismatrix aufbauen ...")

    # ── Preismatrix aufbauen ──────────────────────────────────────────────────
    all_dates  = sorted(score_cache.keys())
    all_assets = sorted({a for s in score_cache.values() for a in s.index})

    price_df = pd.DataFrame(index=all_dates, columns=all_assets, dtype=float)

    for asset in all_assets:
        ps = price_cache.get(asset)
        if ps is None:
            continue
        ps_clean = ps[~ps.index.duplicated(keep='last')].sort_index()
        # reindex: für jeden Score-Datum den letzten verfügbaren Preis
        aligned = ps_clean.reindex(all_dates, method='ffill')
        price_df[asset] = aligned.values

    logger.info(f"  Preismatrix: {price_df.shape[0]} Tage × {price_df.shape[1]} Assets")

    # ── Forward-Return-Matrix ─────────────────────────────────────────────────
    # Für jeden Eintrag [t, asset]: Preis in horizon Handelstagen / Preis heute - 1
    # Wir brauchen die vollständige Preisserie (auch Tage ohne Score) für die
    # korrekte Vorwärtsverschiebung.

    # Vereinige alle Preisdaten auf einem gemeinsamen täglichen Index
    all_price_dates = sorted({
        d for ps in price_cache.values() if ps is not None for d in ps.index
    })
    full_price_df = pd.DataFrame(index=all_price_dates, columns=all_assets, dtype=float)
    for asset in all_assets:
        ps = price_cache.get(asset)
        if ps is None:
            continue
        ps_clean = ps[~ps.index.duplicated(keep='last')].sort_index()
        full_price_df[asset] = ps_clean.reindex(all_price_dates, method='ffill')

    # Forward-Return: Preis in `horizon` Handelstagen relativ zu heute
    fwd_price_df = full_price_df.shift(-horizon)
    fwd_ret_df   = (fwd_price_df / full_price_df - 1).reindex(all_dates)

    logger.info("  Forward-Return-Matrix berechnet. Spearman pro Tag ...")

    # ── Täglicher Spearman-IC ─────────────────────────────────────────────────
    ic_values: dict = {}
    for date in all_dates:
        scores   = score_cache[date]
        fwd_row  = fwd_ret_df.loc[date]

        # Schnittmenge: Assets mit Score UND gültigem Forward-Return
        common   = scores.index.intersection(fwd_row.dropna().index)
        if len(common) < 10:
            continue

        sc_vals  = scores[common].values
        ret_vals = fwd_row[common].values
        corr, _  = spearmanr(sc_vals, ret_vals)
        if not np.isnan(corr):
            ic_values[date] = float(corr)

    result = pd.Series(ic_values).sort_index()
    logger.success(f"  Täglicher IC berechnet: {len(result)} Tage "
                   f"| Ø IC={result.mean():+.4f} | Median={result.median():+.4f}")
    return result


def plot_rolling_ic(
    daily_ic:    pd.Series,
    rolling_map: dict[int, pd.Series] | None = None,
    windows:     list[int] = IC_WINDOWS,
    save_path:   str = "rolling_ic.png",
    label:       str = "",
    # Rückwärtskompatibilität
    window:      int = 60,
) -> None:
    """
    Zeichnet einen 3-Panel-Chart:
      1. Täglicher IC (Balken) + Rolling-IC-Linien für alle Fenster
      2. Monatlicher Ø-IC (Balken) mit Jahres-Durchschnitt
      3. Kumulativer IC (Langzeit-Trend)
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.gridspec import GridSpec
        import matplotlib.cm as cm

        if daily_ic.empty:
            logger.warning("plot_rolling_ic: keine IC-Daten.")
            return

        # Rolling-Map erzeugen falls nicht übergeben
        if rolling_map is None:
            rolling_map = {w: daily_ic.rolling(w, min_periods=max(1, w // 4)).mean()
                           for w in windows}

        cumic = daily_ic.cumsum()

        fig = plt.figure(figsize=(16, 12))
        gs  = GridSpec(3, 1, figure=fig, hspace=0.38,
                       height_ratios=[3.0, 2.5, 1.5])
        title_sfx = f" — {label}" if label else ""

        # ── Panel 1: Täglicher IC + alle Rolling-Linien ───────────────────────
        ax1 = fig.add_subplot(gs[0])
        colors_bar = ['#2E7D32' if v >= 0 else '#C62828' for v in daily_ic]
        ax1.bar(daily_ic.index, daily_ic.values,
                color=colors_bar, alpha=0.35, width=1.5, label='Täglicher IC')

        # Farbpalette für Rolling-Linien (kurze=warm, lange=kalt)
        cmap   = cm.get_cmap('RdYlBu_r', len(windows))
        lws    = [0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.2]   # Fenster aufsteigend
        for i, w in enumerate(windows):
            r = rolling_map.get(w)
            if r is None:
                continue
            ax1.plot(r.index, r.values,
                     color=cmap(i), linewidth=lws[i],
                     label=f'Roll-{w}d', alpha=0.85)

        ax1.axhline(0, color='black', linewidth=0.9)
        ax1.axhline(daily_ic.median(), color='navy', linewidth=0.8,
                    linestyle=':', alpha=0.5,
                    label=f'Gesamt-Median: {daily_ic.median():+.4f}')

        # Negative Phasen rot hinterlegen
        neg = daily_ic < 0
        in_neg, neg_start = False, None
        for dt, is_neg in neg.items():
            if is_neg and not in_neg:
                neg_start, in_neg = dt, True
            elif not is_neg and in_neg:
                ax1.axvspan(neg_start, dt, alpha=0.06, color='red')
                in_neg = False
        if in_neg:
            ax1.axvspan(neg_start, daily_ic.index[-1], alpha=0.06, color='red')

        for yr in range(daily_ic.index[0].year + 1, daily_ic.index[-1].year + 1):
            ax1.axvline(pd.Timestamp(f'{yr}-01-01'), color='gray',
                        linewidth=0.5, linestyle='--', alpha=0.4)

        ax1.set_title(f"Täglicher & Rolling Rank-IC (Spearman){title_sfx}  "
                      f"[Fenster: {windows}]",
                      fontsize=12, fontweight='bold')
        ax1.set_ylabel("Rank IC")
        ax1.legend(loc='upper left', fontsize=7, ncol=3)
        ax1.grid(True, alpha=0.2)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax1.get_xticklabels(), visible=False)

        # ── Panel 2: Monatlicher Ø-IC ─────────────────────────────────────────
        ax2 = fig.add_subplot(gs[1])
        monthly = daily_ic.resample('ME').mean()
        ax2.bar(monthly.index, monthly.values,
                color=['#2E7D32' if v >= 0 else '#C62828' for v in monthly],
                alpha=0.75, width=20, label='Monatlicher Ø-IC')
        ax2.axhline(0, color='black', linewidth=0.8)
        for yr in range(daily_ic.index[0].year + 1, daily_ic.index[-1].year + 1):
            ax2.axvline(pd.Timestamp(f'{yr}-01-01'), color='gray',
                        linewidth=0.6, linestyle='--', alpha=0.4)
        for yr, grp in daily_ic.groupby(daily_ic.index.year):
            mid = pd.Timestamp(f'{yr}-07-01')
            ax2.text(mid, 0.02, f'{grp.mean():+.3f}',
                     ha='center', fontsize=7.5, color='#333', fontweight='bold',
                     transform=ax2.get_xaxis_transform())
        ax2.set_title("Monatlicher Ø-IC (mit Jahres-Durchschnitt)", fontsize=11)
        ax2.set_ylabel("Ø IC")
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.2)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax2.get_xticklabels(), visible=False)

        # ── Panel 3: Kumulativer IC ───────────────────────────────────────────
        ax3 = fig.add_subplot(gs[2])
        ax3.plot(cumic.index, cumic.values,
                 color='#6A1B9A', linewidth=1.8, label='Kumulativer IC')
        ax3.fill_between(cumic.index, cumic.values, 0,
                         where=(cumic >= 0), alpha=0.15, color='#2E7D32')
        ax3.fill_between(cumic.index, cumic.values, 0,
                         where=(cumic < 0), alpha=0.15, color='#C62828')
        ax3.axhline(0, color='black', linewidth=0.8)
        for yr in range(daily_ic.index[0].year + 1, daily_ic.index[-1].year + 1):
            ax3.axvline(pd.Timestamp(f'{yr}-01-01'), color='gray',
                        linewidth=0.5, linestyle='--', alpha=0.4)
        ax3.set_title("Kumulativer IC (Langzeit-Trend)", fontsize=11)
        ax3.set_ylabel("Kum. IC")
        ax3.set_xlabel("Datum")
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.2)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.success(f"Rolling-IC-Chart gespeichert: {save_path}")

    except Exception as e:
        import traceback
        logger.warning(f"plot_rolling_ic Fehler: {e}\n{traceback.format_exc()}")


def rolling_ic_report(
    daily_ic: pd.Series,
    windows:  list[int] = IC_WINDOWS,
    label:    str = "",
) -> dict[int, pd.Series]:
    """
    Berechnet und druckt Rolling-IC-Statistiken für alle Fenster in `windows`.

    Returns
    -------
    dict {window: pd.Series(rolling_mean_ic)}
    """
    if daily_ic.empty:
        print("\n  [WARN] Keine IC-Daten verfügbar.")
        return {}

    pct_negative = (daily_ic < 0).mean() * 100

    # Längste zusammenhängende negative Streak
    neg_flags = (daily_ic < 0).astype(int).values
    max_streak = cur_streak = 0
    for v in neg_flags:
        if v:
            cur_streak += 1
            max_streak  = max(max_streak, cur_streak)
        else:
            cur_streak  = 0

    hdr = f"  ROLLING RANK-IC{f'  [{label}]' if label else ''}"
    print("\n" + "─" * 80)
    print(hdr)
    print("─" * 80)
    print(f"  Tage mit IC-Daten             : {len(daily_ic):>6d}")
    print(f"  Ø täglicher IC                : {daily_ic.mean():>+.4f}")
    print(f"  Median täglicher IC           : {daily_ic.median():>+.4f}")
    print(f"  % Tage mit IC < 0             : {pct_negative:>6.1f}%")
    print(f"  Längste negative Streak       : {max_streak:>6d} Tage")
    print(f"  IC-Stabilitäts-Score          : {(100 - pct_negative):.1f}% positive Tage")

    # Rolling-IC für alle Fenster
    rolling_map: dict[int, pd.Series] = {}
    print(f"\n  Rolling-IC Mediane pro Fenster:")
    for w in windows:
        r = daily_ic.rolling(window=w, min_periods=max(1, w // 4)).mean()
        rolling_map[w] = r
        pct_neg_r = (r < 0).mean() * 100
        print(f"    Fenster {w:>3d}d: Median={r.dropna().median():>+.4f}  "
              f"Ø={r.dropna().mean():>+.4f}  Tage<0={pct_neg_r:.1f}%")

    # Jahres-Aufschlüsselung des IC
    print(f"\n  Jährlicher Ø IC:")
    for yr, grp in daily_ic.groupby(daily_ic.index.year):
        bar_len = max(0, int((grp.mean() + 0.15) / 0.30 * 20))
        bar     = "█" * bar_len
        print(f"    {yr}: {grp.mean():>+.4f}  {bar}")
    print("─" * 80)
    return rolling_map


def save_ic_json(
    daily_ic:    pd.Series,
    rolling_map: dict[int, pd.Series],
    save_path:   str,
    windows:     list[int] = IC_WINDOWS,
) -> None:
    """
    Speichert IC-Zeitreihe + alle Rolling-IC-Fenster als JSON und CSV.

    Format (JSON):
    [{"date": "2020-02-10", "ic": 0.12,
      "ic_roll_5": ..., "ic_roll_10": ..., ..., "ic_roll_60": ...}, ...]
    """
    records = []
    for ts in daily_ic.index:
        rec = {'date': str(ts.date()), 'ic': round(float(daily_ic[ts]), 6)}
        for w in windows:
            key = f'ic_roll_{w}'
            val = rolling_map.get(w)
            rec[key] = round(float(val[ts]), 6) if val is not None and ts in val.index else None
        records.append(rec)

    # JSON
    json_path = Path(save_path)
    json_path.write_text(json.dumps(records, indent=2))

    # CSV (gleicher Pfad, andere Endung)
    csv_path = json_path.with_suffix('.csv')
    pd.DataFrame(records).to_csv(csv_path, index=False)

    logger.success(f"IC-Artefakt gespeichert: {json_path.name}  ({len(records)} Tage, "
                   f"{len(windows)} Rolling-Fenster)")


def full_tearsheet(
    score_cache:  ScoreCache,
    price_cache:  dict,
    df:           pd.DataFrame,
    horizon:      int = 7,
    ic_window:    int = 60,
) -> None:
    """
    Vollständiges Tearsheet für die Referenz-Konfiguration (n7/rb3/hs25%/f0.1%):
      1. Subperioden-Analyse (Jahres-/Marktphasen-Slices)
      2. Rolling Rank-IC (60-Tage-Fenster)

    Berechnet die Equity-Kurve der Referenz-Konfiguration neu (schnell, da
    score_cache bereits existiert) und zeigt beide Analysen.
    """
    # ── Referenz-Equity ───────────────────────────────────────────────────────
    ref_row = df[(df['n_max'] == 7) & (df['rotation_buffer'] == 3) &
                 (df['hard_stop_pct'] == 0.20) & (df['fees'] == 0.001)]
    ref_params = PortfolioParams()   # defaults = Referenz
    if not ref_row.empty:
        r = ref_row.iloc[0]
        print(f"\n  Tearsheet für Referenz-Konfiguration "
              f"(Rang {ref_row.index[0]} von {len(df)}): "
              f"Sharpe={r['sharpe']:.3f}  Return={r['total_return_%']:+.1f}%")
    else:
        print("\n  Tearsheet für Default-Konfiguration (n7/rb3/hs25%/f0.1%)")

    ref_result = run_portfolio(score_cache, price_cache, ref_params)

    # ── 1. Subperioden-Analyse ────────────────────────────────────────────────
    subperiod_report(
        ref_result['equity'],
        ref_result['equity_dates'],
        label="n7/rb3/hs25%/f0.1%",
    )

    # ── 2. Rolling Rank-IC ────────────────────────────────────────────────────
    logger.info("Rolling Rank-IC berechnen ...")
    daily_ic = compute_daily_ic(score_cache, price_cache, horizon=horizon)
    rolling_map = rolling_ic_report(daily_ic, windows=IC_WINDOWS,
                                    label=f"v2_{horizon}d")

    # ── 3. IC-Chart ───────────────────────────────────────────────────────────
    ic_plot = str(Path("rolling_ic.png"))   # wird in main() überschrieben
    plot_rolling_ic(daily_ic, rolling_map=rolling_map, windows=IC_WINDOWS,
                    save_path=ic_plot, label=f"v2_{horizon}d")
    return daily_ic, rolling_map


# ══════════════════════════════════════════════════════════════════════════════
# Policy-Vergleich: Baseline vs. A1/A2/A3/B
# ══════════════════════════════════════════════════════════════════════════════

def _subperiod_dd(equity: list, dates: list, start: str, end: str) -> float:
    """Maximaler Drawdown in einem Subzeitraum (für Policy-Report)."""
    m = _period_metrics(equity, dates, start, end)
    return m['max_dd'] if m else float('nan')


def _subperiod_ret(equity: list, dates: list, start: str, end: str) -> float:
    """Total Return in einem Subzeitraum."""
    m = _period_metrics(equity, dates, start, end)
    return m['return'] if m else float('nan')


def policy_comparison(
    score_cache:  ScoreCache,
    price_cache:  dict,
    daily_ic:     pd.Series,
    rolling_map:  dict,
    base_params:  Optional[PortfolioParams] = None,
    save_path:    Optional[str] = None,
) -> pd.DataFrame:
    """
    Führt 5 Backtests durch (Baseline + A1 + A2 + A3 + B) und vergleicht.

    Policies
    --------
    Baseline  : Kein IC/SPY-Filter
    A1 IC20   : n_max → 3 wenn ic_roll_20 < 0
    A2 IC30   : n_max → 3 wenn ic_roll_30 < 0
    A3 IC40   : n_max → 3 wenn ic_roll_40 < 0
    B  SPY200 : n_max → 3 wenn SPY-Close < SMA200
    C  Budget : Budget -30 % je negativem IC-Fenster (IC20/30/40 kumulativ)

    Returns
    -------
    pd.DataFrame mit allen Metriken inkl. Subperioden 2022/2025
    """
    if base_params is None:
        base_params = PortfolioParams()

    ic_df = build_ic_df(daily_ic, rolling_map)

    runs = [
        ("Baseline",  None),
        ("A1_IC20",   "IC20"),
        ("A2_IC30",   "IC30"),
        ("A3_IC40",   "IC40"),
        ("B_SPY200",  "SPY200"),
        ("C_Budget",  "C_Budget"),
    ]

    total_days = len(sorted(score_cache.keys()))
    rows = []
    equity_map: dict = {}   # {run_name: (equity_list, equity_dates_list)}

    logger.info("═" * 70)
    logger.info("  Policy-Vergleich: Baseline vs. A1 / A2 / A3 / B / C_Budget")
    logger.info(f"  base_params: n_max={base_params.n_max}  rb={base_params.rotation_buffer}"
                f"  hs={base_params.hard_stop_pct:.0%}  fees={base_params.fees:.3%}")
    logger.info("═" * 70)

    for run_name, policy in runs:
        logger.info(f"  Run {run_name:12s} (policy={policy}) ...")
        res = run_portfolio(score_cache, price_cache, base_params,
                            policy=policy, ic_df=ic_df)

        eq  = res['equity']
        eqd = res['equity_dates']
        equity_map[run_name] = (eq, eqd)   # für Chart merken – kein zweiter Backtest nötig
        pct_reduced = res['days_n_max_reduced'] / total_days * 100 if total_days else 0

        row = {
            'run':               run_name,
            'policy':            policy or 'None',
            'sharpe':            res['sharpe'],
            'total_return_%':    res['total_return'],
            'max_drawdown_%':    res['max_drawdown'],
            'n_trades':          res['n_trades'],
            'win_rate_%':        res['win_rate'],
            'avg_hold_days':     res['avg_hold_days'],
            'n_hard_stops':      res['n_hard_stops'],
            'days_n_max_reduced': res['days_n_max_reduced'],
            'pct_days_reduced':  round(pct_reduced, 1),
            # Subperioden
            'ret_2022_%':        round(_subperiod_ret(eq, eqd, '2022-01-01', '2022-12-31'), 1),
            'dd_2022_%':         round(_subperiod_dd(eq, eqd, '2022-01-01', '2022-12-31'), 1),
            'ret_2023_%':        round(_subperiod_ret(eq, eqd, '2023-01-01', '2023-12-31'), 1),
            'ret_2024_%':        round(_subperiod_ret(eq, eqd, '2024-01-01', '2024-12-31'), 1),
            'ret_2025_%':        round(_subperiod_ret(eq, eqd, '2025-01-01', '2025-12-31'), 1),
            'dd_2025_%':         round(_subperiod_dd(eq, eqd, '2025-01-01', '2025-12-31'), 1),
        }
        rows.append(row)
        trigger_label = ("budget_red" if policy == "C_Budget" else "n_max_red")
        logger.info(f"    Sharpe={row['sharpe']:.3f}  Return={row['total_return_%']:+.1f}%  "
                    f"MaxDD={row['max_drawdown_%']:.1f}%  "
                    f"DD-2022={row['dd_2022_%']:.1f}%  DD-2025={row['dd_2025_%']:.1f}%  "
                    f"{trigger_label}={row['pct_days_reduced']:.1f}%")

    df = pd.DataFrame(rows).set_index('run')

    # ── Konsolen-Report (kurz – Details in CSV) ──────────────────────────────
    base_row = df.loc['Baseline']
    logger.info("─── DELTA vs. Baseline ──────────────────────────────────────────────")
    logger.info(f"  {'Run':12s}  {'dSharpe':>8s}  {'dReturn':>10s}  {'dMaxDD':>8s}  "
                f"{'dDD-2022':>10s}  {'dDD-2025':>10s}")
    for run_name, row in df.iterrows():
        if run_name == 'Baseline':
            logger.info(f"  {'Baseline':12s}  {'---':>8s}  "
                        f"{'Ref':>10s}  {'---':>8s}  {'---':>10s}  {'---':>10s}")
            continue
        logger.info(f"  {run_name:12s}  "
                    f"{row['sharpe'] - base_row['sharpe']:>+8.3f}  "
                    f"{row['total_return_%'] - base_row['total_return_%']:>+10.1f}%  "
                    f"{row['max_drawdown_%'] - base_row['max_drawdown_%']:>+8.1f}%  "
                    f"{row['dd_2022_%'] - base_row['dd_2022_%']:>+10.1f}%  "
                    f"{row['dd_2025_%'] - base_row['dd_2025_%']:>+10.1f}%")
    logger.info("─────────────────────────────────────────────────────────────────────")

    if save_path:
        df.reset_index().to_csv(save_path, index=False)
        logger.success(f"Policy-Report gespeichert: {save_path}")

    logger.info("policy_comparison abgeschlossen.")
    return df, equity_map


def plot_policy_equity(
    policy_df:    pd.DataFrame,
    equity_map:   dict,
    all_dates:    list,
    save_path:    str = "policy_equity.png",
) -> None:
    """
    Equity-Kurven aller 5 Policy-Runs in einem Chart.

    Nutzt vorberechnete equity_map aus policy_comparison() –
    kein zweiter Backtest-Lauf nötig.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        logger.info("plot_policy_equity: Chart wird erstellt ...")

        run_order = ["Baseline", "A1_IC20", "A2_IC30", "A3_IC40", "B_SPY200", "C_Budget"]
        colors  = ['#212121', '#1565C0', '#0288D1', '#00796B', '#E65100', '#7B1FA2']
        lws     = [2.5, 1.6, 1.6, 1.6, 1.8, 2.0]
        lstyles = ['-', '--', '--', '--', '-.', (0, (3, 1, 1, 1))]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                        gridspec_kw={'height_ratios': [3, 1]})
        fig.suptitle("Policy-Vergleich: Baseline vs. IC20 / IC30 / IC40 / SPY200 / C_Budget",
                     fontsize=12, fontweight='bold')

        for run_name, color, lw, ls in zip(run_order, colors, lws, lstyles):
            if run_name not in equity_map:
                continue
            eq_list, eqd = equity_map[run_name]
            eq  = np.array(eq_list[1:])
            ret = (eq / eq[0] - 1) * 100
            dd  = (eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq) * 100
            sharpe = policy_df.loc[run_name, 'sharpe'] if run_name in policy_df.index else 0
            tot    = policy_df.loc[run_name, 'total_return_%'] if run_name in policy_df.index else 0
            label  = f"{run_name}  S={sharpe:.3f}  {tot:+.0f}%"
            ax1.plot(eqd, ret, color=color, linewidth=lw, linestyle=ls, label=label)
            # Drawdown nur als Linie (kein fill_between – zu langsam bei 1500 Punkten)
            ax2.plot(eqd, dd, color=color, linewidth=lw * 0.6, linestyle=ls, alpha=0.85)

        # Jahrstrennlinien (nur als axvline – schnell)
        if all_dates:
            for yr in range(all_dates[0].year + 1, all_dates[-1].year + 2):
                for ax in (ax1, ax2):
                    ax.axvline(pd.Timestamp(f'{yr}-01-01'), color='gray',
                               linewidth=0.5, linestyle=':', alpha=0.5)

        ax1.axhline(0, color='black', linewidth=0.6)
        ax1.set_ylabel("Kumulativer Return (%)")
        ax1.legend(loc='upper left', fontsize=8, framealpha=0.7)
        ax1.grid(True, alpha=0.15)
        plt.setp(ax1.get_xticklabels(), visible=False)

        ax2.axhline(0, color='black', linewidth=0.6)
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Datum")
        ax2.grid(True, alpha=0.15)

        logger.info("plot_policy_equity: savefig ...")
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        logger.success(f"Policy-Equity-Chart gespeichert: {save_path}")
    except Exception as e:
        import traceback
        logger.warning(f"plot_policy_equity Fehler: {e}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════════════════
# Ausgabe & Visualisierung
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(
    df:           pd.DataFrame,
    top_n:        int             = 15,
    score_cache:  Optional[ScoreCache] = None,
    price_cache:  Optional[dict]       = None,
    horizon:      int             = 7,
    ic_window:    int             = 60,
) -> None:
    """
    Gibt die Top-N Ergebnisse als formatierte Tabelle aus.

    Wenn score_cache und price_cache übergeben werden, wird zusätzlich
    das vollständige Tearsheet berechnet (Subperioden + Rolling IC).
    """
    logger.info("═" * 80)
    logger.info("  SENSITIVITAETSANALYSE - TOP-Ergebnisse nach Sharpe")
    logger.info("═" * 80)
    display_cols = ['n_max', 'rotation_buffer', 'hard_stop_pct', 'fees',
                    'sharpe', 'total_return_%', 'max_drawdown_%',
                    'n_trades', 'win_rate_%', 'avg_hold_days', 'n_hard_stops']
    top = df[display_cols].head(top_n)
    for line in top.to_string(
        float_format=lambda x: f"{x:+.2f}" if isinstance(x, float) else str(x),
    ).splitlines():
        logger.info(line)
    logger.info("═" * 80)

    # Referenz-Ergebnis (n_max=7, rb=3, hs=0.20, fees=0.001)
    ref = df[(df['n_max'] == 7) & (df['rotation_buffer'] == 3) &
             (df['hard_stop_pct'] == 0.20) & (df['fees'] == 0.001)]
    if not ref.empty:
        r = ref.iloc[0]
        logger.info(f"  Referenz (n7/rb3/hs20%/f0.1%): "
                    f"Rang {ref.index[0]}  |  "
                    f"Sharpe={r['sharpe']:.3f}  "
                    f"Return={r['total_return_%']:+.1f}%  "
                    f"MaxDD={r['max_drawdown_%']:.1f}%")

    logger.info("  Parameter-Sensitivitaet (Sharpe je Wert):")
    for col in ['n_max', 'rotation_buffer', 'hard_stop_pct', 'fees']:
        logger.info(f"  {col:18s}: " +
                    "  ".join(f"{v}->{df[df[col]==v]['sharpe'].mean():.3f}"
                               for v in sorted(df[col].unique())))

    # ── Erweitertes Tearsheet (optional) ─────────────────────────────────────
    if score_cache is not None and price_cache is not None:
        full_tearsheet(score_cache, price_cache, df,
                       horizon=horizon, ic_window=ic_window)


def plot_top_equity_curves(
    score_cache:  ScoreCache,
    price_cache:  dict,
    df:           pd.DataFrame,
    top_n:        int = 5,
    save_path:    str = "sensitivity_top_equity.png",
) -> None:
    """Zeichnet Equity-Kurven der Top-N Konfigurationen."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9),
                                        gridspec_kw={'height_ratios': [3, 1]})

        colors = plt.cm.tab10.colors

        for i, (_, row) in enumerate(df.head(top_n).iterrows()):
            p = PortfolioParams(
                n_max=int(row['n_max']),
                rotation_buffer=int(row['rotation_buffer']),
                hard_stop_pct=float(row['hard_stop_pct']),
                fees=float(row['fees']),
            )
            res     = run_portfolio(score_cache, price_cache, p)
            eq      = np.array(res['equity'][1:])
            dates   = res['equity_dates']
            ret_pct = (eq / eq[0] - 1) * 100
            peaks   = np.maximum.accumulate(eq)
            dd_pct  = (eq - peaks) / peaks * 100
            lbl = (f"n{p.n_max}/rb{p.rotation_buffer}/"
                   f"hs{int(p.hard_stop_pct*100)}%/f{p.fees*100:.2f}%  "
                   f"Sharpe={row['sharpe']:.3f}  "
                   f"{row['total_return_%']:+.0f}%")
            ax1.plot(dates, ret_pct, color=colors[i], linewidth=1.6, label=lbl)
            ax2.fill_between(dates, dd_pct, 0, alpha=0.2, color=colors[i])

        ax1.set_title("Sensitivitätsanalyse – Top-Konfigurationen (Equity)", fontsize=13)
        ax1.set_ylabel("Kumulativer Return (%)")
        ax1.legend(fontsize=8, loc='upper left')
        ax1.grid(True, alpha=0.25)

        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Datum")
        ax2.grid(True, alpha=0.25)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.success(f"Equity-Plot gespeichert: {save_path}")
    except Exception as e:
        logger.warning(f"Plot fehlgeschlagen: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Kaggle-Integration: als Schritt in kaggle_full_run.py aufrufbar
# ══════════════════════════════════════════════════════════════════════════════

def run_sensitivity_step(
    features:         pd.DataFrame,
    asset_map:        Dict[str, int],
    all_train_results: Dict[int, dict],
    price_cache:      dict,
    working_dir:      str | Path = "/kaggle/working",
    param_grid:       Optional[dict] = None,
) -> pd.DataFrame:
    """
    Wrapper für die Integration in kaggle_full_run.py.

    Verwendung in main():
        from run_sensitivity import run_sensitivity_step
        sensitivity_df = run_sensitivity_step(features, asset_map,
                                               all_train_results, price_cache)
    """
    working_dir = Path(working_dir)

    # Fold-Results des 7d-Modells extrahieren (oder des ersten verfügbaren)
    horizon = 7 if 7 in all_train_results else next(iter(all_train_results))
    fold_results = all_train_results[horizon]['fold_results']
    seq_len      = all_train_results[horizon].get('seq_len', 64)

    logger.info(f"Sensitivitätsanalyse für Horizont {horizon}d | {len(fold_results)} Folds")

    score_cache = build_score_cache(features, fold_results, asset_map, seq_len)
    df          = grid_search(score_cache, price_cache, param_grid)

    # Ergebnisse speichern
    csv_path = working_dir / "sensitivity_results.csv"
    df.to_csv(csv_path)
    logger.success(f"Ergebnisse gespeichert: {csv_path}")

    # Equity-Chart der Top-5
    plot_top_equity_curves(
        score_cache, price_cache, df,
        save_path=str(working_dir / "sensitivity_top_equity.png"),
    )

    print_summary(df)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# CLI-Einstieg (lokal oder Kaggle standalone)
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Portfolio-Sensitivitätsanalyse (Grid Search ohne Neutraining)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--ckpt-dir',   required=True,
                   help='Verzeichnis mit fold_X_best.pt Dateien')
    p.add_argument('--walk-json',  required=True,
                   help='Pfad zu v2_7d_walk_forward.json')
    p.add_argument('--asset-map',  required=True,
                   help='Pfad zu asset_map.json')
    p.add_argument('--data-dir',   required=True,
                   help='Verzeichnis mit .parquet Dateien (data/raw)')
    p.add_argument('--output',     default='sensitivity_results.csv',
                   help='Ausgabe-CSV')
    p.add_argument('--plot',       default='sensitivity_top_equity.png',
                   help='Ausgabe-PNG (Top-5 Equity-Kurven)')
    p.add_argument('--ic-plot',     default='rolling_ic.png',
                   help='Ausgabe-PNG (Rolling Rank-IC Chart)')
    p.add_argument('--score-cache', default=None,
                   help='Pfad zur Score-Cache .parquet Datei. '
                        'Falls vorhanden: laden statt neu berechnen. '
                        'Falls nicht vorhanden: berechnen und speichern.')
    p.add_argument('--horizon',        type=int, default=7,
                   help='Vorhersage-Horizont in Handelstagen')
    p.add_argument('--top-n',          type=int, default=15,
                   help='Anzahl Top-Ergebnisse in der Ausgabe')
    p.add_argument('--skip-grid-search', action='store_true', default=False,
                   help='Grid Search überspringen (auch wenn CSV noch nicht existiert)')
    p.add_argument('--policy-compare', action='store_true', default=False,
                   help='Policy-Vergleich Baseline/A1/A2/A3/B ausführen')
    p.add_argument('--policy-csv',     default='policy_comparison.csv',
                   help='Ausgabe-CSV für Policy-Vergleich')
    p.add_argument('--policy-plot',    default='policy_equity.png',
                   help='Ausgabe-PNG für Policy-Equity-Chart')
    # Feature-Engineering
    p.add_argument('--sector-neutral', action='store_true', default=False,
                   help='Sektor-neutrale Z-Score Normalisierung beim Feature-Panel-Aufbau.')
    # Normalisierungs-Vergleich
    p.add_argument('--cs-score-cache', default=None,
                   help='Score-Cache des global cross-sectional Modells (zum Vergleich).')
    p.add_argument('--norm-compare', action='store_true', default=False,
                   help='Phase 6: 1:1-Vergleich CS-Normalisierung vs. Sektor-neutral.')
    p.add_argument('--norm-compare-csv', default='normalization_comparison.csv',
                   help='Ausgabe-CSV für Normalisierungs-Vergleich.')
    p.add_argument('--norm-compare-plot', default='normalization_comparison.png',
                   help='Ausgabe-PNG für Normalisierungs-Vergleich Equity-Chart.')
    # Universum-Robustheit
    p.add_argument('--exclude-tickers', nargs='+', default=[],
                   metavar='TICKER',
                   help='Ticker aus dem Ranking ausschliessen (z.B. AAPL MSFT NVDA)')
    p.add_argument('--no-mega-cap', action='store_true', default=False,
                   help=f'Shortcut: schliesst Mag-7 aus ({", ".join(MEGA_CAP_7)})')
    p.add_argument('--device',     default=None,
                   help='cuda oder cpu (auto-detect wenn leer)')
    p.add_argument('--repo-dir',   default='.',
                   help='Pfad zum Repo (für Imports)')
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Phase 6: Normalisierungs-Vergleich (CS vs. Sektor-Neutral)
# ══════════════════════════════════════════════════════════════════════════════

def normalization_compare(
    cs_score_cache:  ScoreCache,
    sn_score_cache:  ScoreCache,
    price_cache:     dict,
    horizon:         int   = 7,
    save_csv:        str   = 'normalization_comparison.csv',
    save_plot:       str   = 'normalization_comparison.png',
) -> pd.DataFrame:
    """
    1:1-Vergleich: global cross-sectional Z-Score vs. sektor-neutral Z-Score.

    Feste Portfolio-Parameter: n_max=5, rotation_buffer=4, hard_stop=20 %
    Vier Runs: CS-Baseline, CS+A3, SN-Baseline, SN+A3
    (A3 = IC_roll_40 < 0 → n_max auf 3 reduzieren)

    Subperioden: 2020 (COVID), 2022 (Bärenmarkt), 2025 (Neuland)

    Returns
    -------
    DataFrame mit einer Zeile pro Run + Kennzahlen.
    """
    BASE_PARAMS = PortfolioParams(n_max=5, rotation_buffer=4, hard_stop_pct=0.20)

    RUNS = [
        ('CS_Baseline', cs_score_cache, None),
        ('CS_A3',       cs_score_cache, 'IC40'),
        ('CS_Budget',   cs_score_cache, 'C_Budget'),
        ('SN_Baseline', sn_score_cache, None),
        ('SN_A3',       sn_score_cache, 'IC40'),
        ('SN_Budget',   sn_score_cache, 'C_Budget'),
    ]

    SUBPERIODS = [
        ('2020', '2020-01-01', '2020-12-31'),
        ('2022', '2022-01-01', '2022-12-31'),
        ('2025', '2025-01-01', '2026-12-31'),
    ]

    logger.info("═" * 68)
    logger.info("  Phase 6: Normalisierungs-Vergleich  CS vs. Sektor-Neutral")
    logger.info(f"  Portfolio: n_max=5  rb=4  hs=20%  Policy: Baseline / A3 (IC40) / C_Budget")
    logger.info("═" * 68)

    rows       = []
    equity_map = {}   # label → (equity_dates, equity_values)

    for label, cache, policy in RUNS:
        logger.info(f"  Run {label} (policy={policy}) ...")

        # IC-Daten für IC-Policies berechnen (aus dem jeweiligen Score-Cache)
        ic_df_run: Optional[pd.DataFrame] = None
        if policy is not None:
            try:
                daily_ic_run = compute_daily_ic(cache, price_cache, horizon=horizon)
                if not daily_ic_run.empty:
                    rolling_map_run = rolling_ic_report(
                        daily_ic_run, windows=IC_WINDOWS, label=label)
                    ic_df_run = build_ic_df(daily_ic_run, rolling_map_run)
            except Exception as ic_exc:
                logger.warning(f"  IC-Berechnung für {label} fehlgeschlagen: {ic_exc}")

        res = run_portfolio(
            score_cache=cache,
            price_cache=price_cache,
            params=BASE_PARAMS,
            policy=policy,
            ic_df=ic_df_run,
        )

        eq    = res['equity']
        eqd   = res['equity_dates']
        equity_map[label] = (eqd, eq)

        # Subperioden-Kennzahlen
        sub_dd  = {sp: _subperiod_dd(eq, eqd, s, e) for sp, s, e in SUBPERIODS}
        sub_ret = {sp: _subperiod_ret(eq, eqd, s, e) for sp, s, e in SUBPERIODS}

        row = {
            'run':             label,
            'normalization':   'CrossSectional' if label.startswith('CS') else 'SectorNeutral',
            'policy':          policy or 'None',
            'sharpe':          round(res['sharpe'], 3),
            'total_return_%':  round(res['total_return'], 1),
            'max_drawdown_%':  round(res['max_drawdown'], 1),
            'n_trades':        res['n_trades'],
            'win_rate_%':      round(res['win_rate'], 1),
            'avg_hold_days':   round(res['avg_hold_days'], 1),
            'pct_days_reduced': round(
                res['days_n_max_reduced'] / max(len(eqd), 1) * 100, 1),
        }
        for sp, _, _ in SUBPERIODS:
            row[f'ret_{sp}_%']  = round(sub_ret[sp], 1)
            row[f'dd_{sp}_%']   = round(sub_dd[sp],  1)
        rows.append(row)

        logger.info(
            f"  → Sharpe={res['sharpe']:+.3f}  Return={res['total_return']:+.1f}%  "
            f"MaxDD={res['max_drawdown']:+.1f}%  "
            f"DD-2022={sub_dd['2022']:+.1f}%  DD-2025={sub_dd['2025']:+.1f}%"
        )

    df = pd.DataFrame(rows)
    df.to_csv(save_csv, index=False)
    logger.success(f"  Normalisierungs-Vergleich CSV: {save_csv}")

    # ── Equity-Chart ─────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(13, 6), dpi=90)
        colors  = {'CS_Baseline': '#888888', 'CS_A3': '#d62728',  'CS_Budget': '#e377c2',
                   'SN_Baseline': '#1f77b4', 'SN_A3': '#2ca02c', 'SN_Budget': '#7B1FA2'}
        styles  = {'CS_Baseline': '--', 'CS_A3': '-.', 'CS_Budget': ':',
                   'SN_Baseline': '--', 'SN_A3': '-',  'SN_Budget': (0, (3, 1, 1, 1))}

        all_dates_sorted = sorted(
            {d for (eqd, _) in equity_map.values() for d in eqd})
        for label, (eqd, eq) in equity_map.items():
            eq_s = pd.Series(eq[1:], index=eqd)
            eq_pct = (eq_s / eq_s.iloc[0] - 1) * 100
            ax.plot(eq_s.index, eq_pct.values,
                    label=label, color=colors[label],
                    linestyle=styles[label], linewidth=1.8)

        ax.set_title('Normalisierungs-Vergleich: CS vs. Sektor-Neutral\n'
                     'n_max=5  rb=4  hard_stop=20%  (Baseline / A3=IC40 / C_Budget)',
                     fontsize=11)
        ax.set_ylabel('Kumulierter Return (%)')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_plot, dpi=90)
        plt.close(fig)
        logger.success(f"  Normalisierungs-Vergleich Chart: {save_plot}")
    except Exception as plot_exc:
        logger.warning(f"  Chart fehlgeschlagen: {plot_exc}")

    return df


def main() -> None:
    # Unbuffered stdout: jede print()-Zeile flusht sofort → kein Pipe-Buffer-Blockade
    # beim Subprocess-Exit (Kaggle-Pipe-Buffer ist klein).
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    args = parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    # --no-mega-cap ist ein Shortcut für --exclude-tickers mit Mag-7
    exclude_tickers: List[str] = list(args.exclude_tickers)
    if args.no_mega_cap:
        for t in MEGA_CAP_7:
            if t not in exclude_tickers:
                exclude_tickers.append(t)

    logger.info("═" * 60)
    logger.info("  Sensitivitätsanalyse – Portfolio-Layer Grid Search")
    logger.info("═" * 60)
    logger.info(f"  --policy-compare : {args.policy_compare}")
    logger.info(f"  --score-cache    : {args.score_cache}")
    logger.info(f"  --skip-grid-search: {args.skip_grid_search}")
    logger.info(f"  --horizon        : {args.horizon}")
    if exclude_tickers:
        logger.info(f"  --exclude-tickers: {exclude_tickers}")

    # ── Artefakte laden ───────────────────────────────────────────────────────
    fold_results = load_walk_forward_json(args.walk_json)

    ckpt_dir = Path(args.ckpt_dir)
    for fold in fold_results:
        fname = Path(fold['ckpt_path']).name
        fold['ckpt_path'] = str(ckpt_dir / fname)

    asset_map   = load_asset_map(args.asset_map)
    cache_path  = args.score_cache
    score_exists = cache_path and Path(cache_path).exists()

    # ── Features nur laden wenn Score-Cache NICHT vorhanden ──────────────────
    # Features werden ausschließlich für build_score_cache() gebraucht.
    # Wenn der Cache schon existiert, können wir uns die ~15s sparen.
    if score_exists:
        logger.info(f"Score-Cache vorhanden → Feature-Panel wird übersprungen.")
        features = None
    else:
        mode = "sektor-neutral" if args.sector_neutral else "cross-sectional"
        logger.info(f"Feature-Panel aufbauen ({mode}, dauert ~30s) ...")
        features = build_features_from_parquet(args.data_dir,
                                               sector_neutral=args.sector_neutral)

    logger.info("Price-Cache aufbauen ...")
    price_cache = build_price_cache_local(asset_map, args.data_dir)

    # ── Phase 1: Score-Cache (einmalig oder aus Datei laden) ─────────────────
    if score_exists:
        logger.info(f"Score-Cache aus Datei laden: {cache_path}")
        score_cache = load_score_cache(cache_path)
    else:
        score_cache = build_score_cache(features, fold_results, asset_map,
                                        device=args.device)
        if cache_path:
            save_score_cache(score_cache, cache_path)

    # ── Phase 2: Grid Search + Plots ─────────────────────────────────────────
    # Grid Search überspringen wenn Output-CSV bereits vorhanden und Score-Cache unverändert
    grid_csv = Path(args.output)
    if (grid_csv.exists() and score_exists) or args.skip_grid_search:
        logger.info(f"Grid-Search-CSV vorhanden → Grid Search wird übersprungen: {grid_csv}")
        df = pd.read_csv(grid_csv, index_col=0)
    else:
        df = grid_search(score_cache, price_cache)
        df.to_csv(args.output)
        logger.success(f"CSV gespeichert: {args.output}")

        plot_top_equity_curves(score_cache, price_cache, df,
                               save_path=args.plot)

    print_summary(df, top_n=args.top_n)

    # Subperioden-Analyse für Referenz-Konfiguration
    ref_result = run_portfolio(score_cache, price_cache, PortfolioParams())
    subperiod_report(ref_result['equity'], ref_result['equity_dates'],
                     label='n7/rb3/hs20%/f0.1%')

    # ── Phase 3: Rolling Rank-IC ──────────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("  Phase 3: Rolling Rank-IC")
    logger.info("═" * 60)
    daily_ic    = pd.Series(dtype=float)
    rolling_map = {}
    try:
        daily_ic = compute_daily_ic(score_cache, price_cache, horizon=args.horizon)
        if daily_ic.empty:
            logger.warning("IC-Berechnung ergab 0 Tage – Chart wird übersprungen.")
        else:
            rolling_map = rolling_ic_report(daily_ic, windows=IC_WINDOWS,
                                            label=f'v2_{args.horizon}d')
            plot_rolling_ic(daily_ic, rolling_map=rolling_map, windows=IC_WINDOWS,
                            save_path=args.ic_plot, label=f'v2_{args.horizon}d')
            logger.success(f"IC-Chart gespeichert: {args.ic_plot}")

            ic_json_path = str(Path(args.ic_plot).with_name(
                f"rolling_ic_v2_{args.horizon}d.json"))
            save_ic_json(daily_ic, rolling_map,
                         save_path=ic_json_path, windows=IC_WINDOWS)
    except Exception as exc:
        logger.error(f"Phase 3 fehlgeschlagen: {exc}")
        import traceback
        traceback.print_exc()

    # ── Phase 4: Policy-Vergleich (optional) ─────────────────────────────────
    if args.policy_compare:
        logger.info("═" * 60)
        logger.info("  Phase 4: Policy-Vergleich (Baseline / A1 / A2 / A3 / B)")
        logger.info("═" * 60)
        try:
            if daily_ic.empty:
                logger.error("Policy-Vergleich benötigt IC-Daten (Phase 3 fehlgeschlagen).")
            else:
                policy_df, equity_map = policy_comparison(
                    score_cache=score_cache,
                    price_cache=price_cache,
                    daily_ic=daily_ic,
                    rolling_map=rolling_map,
                    base_params=PortfolioParams(),
                    save_path=args.policy_csv,
                )
                # Chart nutzt vorberechnete Equity-Kurven – kein zweiter Backtest-Lauf
                plot_policy_equity(
                    policy_df=policy_df,
                    equity_map=equity_map,
                    all_dates=sorted(score_cache.keys()),
                    save_path=args.policy_plot,
                )
        except Exception as exc:
            logger.error(f"Phase 4 fehlgeschlagen: {exc}")
            import traceback
            traceback.print_exc()

    # ── Phase 5: Universum-Robustheit (optional) ─────────────────────────────
    if exclude_tickers:
        logger.info("═" * 60)
        logger.info("  Phase 5: Universum-Robustheit")
        logger.info(f"  Blacklist ({len(exclude_tickers)} Ticker): {exclude_tickers}")
        logger.info("═" * 60)
        try:
            base_params = PortfolioParams()
            ex_params   = PortfolioParams(exclude_tickers=exclude_tickers)

            logger.info("  Run Full-Universe ...")
            res_full = run_portfolio(score_cache, price_cache, base_params)

            logger.info(f"  Run Ex-{len(exclude_tickers)}-Tickers ...")
            res_ex   = run_portfolio(score_cache, price_cache, ex_params)

            def _fmt(res: dict, label: str) -> str:
                return (
                    f"  {label:<22} "
                    f"Sharpe={res['sharpe']:+.3f}  "
                    f"Return={res['total_return']:+.1f}%  "
                    f"MaxDD={res['max_drawdown']:+.1f}%  "
                    f"Trades={res['n_trades']}"
                )

            logger.info("─" * 60)
            logger.info(_fmt(res_full, "Full-Universe"))
            logger.info(_fmt(res_ex,   f"Ex-{len(exclude_tickers)}-Tickers"))
            delta_sharpe = res_ex['sharpe']       - res_full['sharpe']
            delta_ret    = res_ex['total_return'] - res_full['total_return']
            delta_dd     = res_ex['max_drawdown'] - res_full['max_drawdown']
            logger.info(
                f"  {'Delta':<22} "
                f"Sharpe={delta_sharpe:+.3f}  "
                f"Return={delta_ret:+.1f}%  "
                f"MaxDD={delta_dd:+.1f}%"
            )
            logger.info("─" * 60)

            # Subperioden-Vergleich
            for label, year_start, year_end in [
                ("2022 (Bärenmarkt)", "2022-01-01", "2022-12-31"),
                ("2025+",            "2025-01-01", "2026-12-31"),
            ]:
                dd_full = _subperiod_dd(res_full['equity'], res_full['equity_dates'],
                                        year_start, year_end)
                dd_ex   = _subperiod_dd(res_ex['equity'],   res_ex['equity_dates'],
                                        year_start, year_end)
                logger.info(
                    f"  MaxDD {label:<18} "
                    f"Full={dd_full:+.1f}%  "
                    f"Ex={dd_ex:+.1f}%  "
                    f"Delta={dd_ex-dd_full:+.1f}%"
                )

            # CSV-Export
            robustness_csv = str(Path(args.policy_csv).with_name("universe_robustness.csv"))
            pd.DataFrame([
                {'run': 'Full-Universe', 'exclude': '',
                 'sharpe': res_full['sharpe'], 'total_return_%': res_full['total_return'],
                 'max_drawdown_%': res_full['max_drawdown'], 'n_trades': res_full['n_trades'],
                 'win_rate_%': res_full['win_rate'], 'avg_hold_days': res_full['avg_hold_days']},
                {'run': f'Ex-{len(exclude_tickers)}-Tickers', 'exclude': ','.join(exclude_tickers),
                 'sharpe': res_ex['sharpe'], 'total_return_%': res_ex['total_return'],
                 'max_drawdown_%': res_ex['max_drawdown'], 'n_trades': res_ex['n_trades'],
                 'win_rate_%': res_ex['win_rate'], 'avg_hold_days': res_ex['avg_hold_days']},
            ]).to_csv(robustness_csv, index=False)
            logger.success(f"  Robustness-CSV gespeichert: {robustness_csv}")

        except Exception as exc:
            logger.error(f"Phase 5 fehlgeschlagen: {exc}")
            import traceback
            traceback.print_exc()

    # ── Phase 6: Normalisierungs-Vergleich (optional) ────────────────────────
    if args.norm_compare:
        logger.info("═" * 60)
        logger.info("  Phase 6: Normalisierungs-Vergleich (CS vs. Sektor-Neutral)")
        logger.info("═" * 60)
        try:
            cs_cache_path = args.cs_score_cache
            if not cs_cache_path or not Path(cs_cache_path).exists():
                logger.error(
                    "Phase 6 benötigt --cs-score-cache <Pfad zum CS-Score-Cache>. "
                    "Datei nicht gefunden oder nicht angegeben."
                )
            else:
                logger.info(f"  Lade CS-Score-Cache: {cs_cache_path}")
                cs_score_cache = load_score_cache(cs_cache_path)
                logger.success(
                    f"  CS-Score-Cache geladen: {len(cs_score_cache)} Tage")

                normalization_compare(
                    cs_score_cache=cs_score_cache,
                    sn_score_cache=score_cache,
                    price_cache=price_cache,
                    horizon=args.horizon,
                    save_csv=args.norm_compare_csv,
                    save_plot=args.norm_compare_plot,
                )
        except Exception as exc:
            logger.error(f"Phase 6 fehlgeschlagen: {exc}")
            import traceback
            traceback.print_exc()

    # Expliziter Flush vor dem Exit – verhindert Blockade beim Pipe-Close
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    logger.info("run_sensitivity.py beendet.")


if __name__ == '__main__':
    main()
