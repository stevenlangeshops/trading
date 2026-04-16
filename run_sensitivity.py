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

@dataclass
class PortfolioParams:
    """Alle variablen Portfolio-Parameter für eine Grid-Search-Zelle."""
    n_max:           int   = 7
    n_mid:           int   = 3       # fix: Bull/Bear-Skalierung relativ zu n_max
    n_min:           int   = 1       # fix
    rotation_buffer: int   = 3
    hard_stop_pct:   float = 0.25
    fees:            float = 0.001
    init_cash:       float = 10_000.0

    def label(self) -> str:
        return (f"n{self.n_max}_rb{self.rotation_buffer}_"
                f"hs{int(self.hard_stop_pct*100)}_f{int(self.fees*10000)}")


# Score-Cache-Typ: Datum → pd.Series (asset → score, absteigend)
ScoreCache = Dict[pd.Timestamp, pd.Series]


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


def run_portfolio(
    score_cache: ScoreCache,
    price_cache: dict,
    params:      PortfolioParams,
) -> dict:
    """
    Führt die Portfolio-Simulation auf dem Score-Cache durch.

    Keine Modell-Operationen – reine Execution-Logik.
    Identisch zur Run-G-Strategie: Long-Only, Rotation, Hard-Stop.

    Returns
    -------
    dict mit Backtest-Metriken + equity/equity_dates für Plots.
    """
    spy_prices = price_cache.get('SPY')

    cash      = params.init_cash
    positions: dict = {}
    equity        = [params.init_cash]
    equity_dates  = []
    trade_log: list[dict] = []

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

    for date in sorted_dates:
        ranking = score_cache[date]

        # Hold-Days inkrementieren
        for pos in positions.values():
            pos['hold_days'] = pos.get('hold_days', 0) + 1

        regime = _get_regime(spy_prices, date)
        n_long = _adaptive_n(regime, params.n_max, params.n_mid, params.n_min)

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
                alloc  = total_val / n_long
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
        'total_return': round(total_return, 2),
        'max_drawdown':  round(max_dd, 2),
        'sharpe':        round(sharpe, 3),
        'n_trades':      n_trades,
        'win_rate':      round(win_rate, 1),
        'avg_hold_days': round(avg_hold, 1),
        'n_hard_stops':  len(stops),
        'equity':        equity,
        'equity_dates':  equity_dates,
        'trade_log':     trade_log,
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


def build_features_from_parquet(data_dir: str | Path) -> pd.DataFrame:
    """
    Baut das Feature-Panel aus dem Parquet-Verzeichnis auf.
    Identisch zu step_build_panel() in kaggle_full_run.py.

    build_panel() liest RAW_DIR als Modul-Konstante, daher wird sie
    vor dem Import-/Aufruf-Zeitpunkt überschrieben.
    """
    import features.engineer as eng
    eng.RAW_DIR = Path(data_dir)
    features, _ = eng.build_panel(timeframe="1d", horizon=11, min_rows=300)
    return features


def build_price_cache_local(asset_map: Dict[str, int], data_dir: str | Path) -> dict:
    """Baut price_cache aus Parquet-Dateien."""
    from strategy.backtest import build_price_cache
    assets = list(asset_map.keys())
    if 'SPY' not in assets:
        assets.append('SPY')
    return build_price_cache(assets, raw_dir=Path(data_dir))


# ══════════════════════════════════════════════════════════════════════════════
# Ausgabe & Visualisierung
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(df: pd.DataFrame, top_n: int = 15) -> None:
    """Gibt die Top-N Ergebnisse als formatierte Tabelle aus."""
    print("\n" + "═" * 100)
    print("  SENSITIVITÄTSANALYSE — TOP-Ergebnisse nach Sharpe")
    print("═" * 100)
    display_cols = ['n_max', 'rotation_buffer', 'hard_stop_pct', 'fees',
                    'sharpe', 'total_return_%', 'max_drawdown_%',
                    'n_trades', 'win_rate_%', 'avg_hold_days', 'n_hard_stops']
    top = df[display_cols].head(top_n)
    print(top.to_string(
        float_format=lambda x: f"{x:+.2f}" if isinstance(x, float) else str(x),
    ))
    print("═" * 100)

    # Referenz-Ergebnis (n_max=7, rb=3, hs=0.25, fees=0.001)
    ref = df[(df['n_max'] == 7) & (df['rotation_buffer'] == 3) &
             (df['hard_stop_pct'] == 0.25) & (df['fees'] == 0.001)]
    if not ref.empty:
        r = ref.iloc[0]
        print(f"\n  Referenz (n7/rb3/hs25%/f0.1%): "
              f"Rang {ref.index[0]}  |  "
              f"Sharpe={r['sharpe']:.3f}  "
              f"Return={r['total_return_%']:+.1f}%  "
              f"MaxDD={r['max_drawdown_%']:.1f}%")

    print("\n  Parameter-Sensitivität (Ø Sharpe je Wert):")
    for col in ['n_max', 'rotation_buffer', 'hard_stop_pct', 'fees']:
        print(f"  {col:18s}: " +
              "  ".join(f"{v}→{df[df[col]==v]['sharpe'].mean():.3f}" for v in sorted(df[col].unique())))


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
    p.add_argument('--top-n',      type=int, default=15,
                   help='Anzahl Top-Ergebnisse in der Ausgabe')
    p.add_argument('--device',     default=None,
                   help='cuda oder cpu (auto-detect wenn leer)')
    p.add_argument('--repo-dir',   default='.',
                   help='Pfad zum Repo (für Imports)')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Repo-Verzeichnis in sys.path aufnehmen
    repo_dir = Path(args.repo_dir).resolve()
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    logger.info("═" * 60)
    logger.info("  Sensitivitätsanalyse – Portfolio-Layer Grid Search")
    logger.info("═" * 60)

    # ── Artefakte laden ───────────────────────────────────────────────────────
    fold_results = load_walk_forward_json(args.walk_json)

    # Checkpoint-Pfade auf --ckpt-dir umbiegen (falls aus Archiv extrahiert)
    ckpt_dir = Path(args.ckpt_dir)
    for fold in fold_results:
        fname = Path(fold['ckpt_path']).name
        fold['ckpt_path'] = str(ckpt_dir / fname)

    asset_map = load_asset_map(args.asset_map)

    logger.info("Feature-Panel aufbauen (dauert ~30s) ...")
    features = build_features_from_parquet(args.data_dir)

    logger.info("Price-Cache aufbauen ...")
    price_cache = build_price_cache_local(asset_map, args.data_dir)

    # ── Phase 1: Score-Cache (einmalig) ──────────────────────────────────────
    score_cache = build_score_cache(features, fold_results, asset_map,
                                     device=args.device)

    # ── Phase 2: Grid Search ──────────────────────────────────────────────────
    df = grid_search(score_cache, price_cache)

    # ── Ausgabe ───────────────────────────────────────────────────────────────
    df.to_csv(args.output)
    logger.success(f"CSV gespeichert: {args.output}")

    plot_top_equity_curves(score_cache, price_cache, df,
                           save_path=args.plot)

    print_summary(df, top_n=args.top_n)


if __name__ == '__main__':
    main()
