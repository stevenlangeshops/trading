"""
backtest_v2_single_horizon.py
──────────────────────────────
Backtest + Benchmark-Report fuer den Single-Horizon-Vergleich.

Pro Horizont (4/7/11/15d):
  - Identische Strategie (Long-Only, Rotation, Hard-Stop 20%)
  - Ranking nach SingleHorizonRankModel-Score
  - Rolling Rank-IC Monitor (Fenster 5/10/15/20/30/40/50/60 Tage)

Am Ende: Vergleichstabelle v1_rank vs. alle v2-Horizonte + Equity-Plot.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger

from config_v2_single_horizon import SingleHorizonConfig, HORIZONS, get_config
from models_v2_single_horizon import SingleHorizonRankModel, rank_ic
from strategy.backtest import (
    build_price_cache,
    get_market_regime,
    adaptive_n,
    _get_price,
)

# ── Rolling-IC-Fenster ────────────────────────────────────────────────────────
IC_WINDOWS = [5, 10, 15, 20, 30, 40, 50, 60]


# ── Rank-IC Hilfsfunktionen ───────────────────────────────────────────────────

def compute_daily_rank_ic(
    scores: pd.Series,
    future_returns: pd.Series,
    min_pairs: int = 10,
) -> float:
    """
    Berechnet den täglichen Rank-IC (Spearman-Korrelation) zwischen
    Modell-Scores und realisierten Forward-Returns für einen Tag t.

    Parameters
    ----------
    scores         : pd.Series  [asset → score]   Modell-Vorhersage am Tag t
    future_returns : pd.Series  [asset → return]  7-Tage-Forward-Return ab t
    min_pairs      : int        Mindestanzahl gültiger Paare (sonst NaN)

    Returns
    -------
    float  Spearman-IC ∈ [-1, 1]  oder  NaN bei zu wenig Daten
    """
    from scipy.stats import spearmanr

    common = scores.index.intersection(future_returns.dropna().index)
    if len(common) < min_pairs:
        return float('nan')
    corr, _ = spearmanr(scores[common].values, future_returns[common].values)
    return float(corr) if not np.isnan(corr) else float('nan')


def _build_ic_series(
    score_log:   list[tuple],   # [(date, pd.Series[asset→score])]
    price_cache: dict,
    horizon:     int,
    windows:     list[int] = IC_WINDOWS,
) -> dict:
    """
    Berechnet IC-Zeitreihe + Rolling-ICs für alle OOS-Tage.

    Strategie:
      1. Preismatrix (date × asset) aufbauen – vektorisiert.
      2. Forward-Return-Matrix per shift(-horizon).
      3. Pro Tag: Spearman zwischen Score und Forward-Return.
      4. Rolling-ICs für alle Fenster in `windows`.

    Returns
    -------
    dict mit:
      'ic'        : pd.Series(index=date, values=ic)
      'rolling'   : {w: pd.Series}
      'records'   : list[dict]  (für JSON-Export)
    """
    if not score_log:
        return {'ic': pd.Series(dtype=float), 'rolling': {}, 'records': []}

    from scipy.stats import spearmanr

    all_dates  = [d for d, _ in score_log]
    all_assets = sorted({a for _, s in score_log for a in s.index})

    # ── Preismatrix ───────────────────────────────────────────────────────────
    # Vollständige Handelsdaten für alle Assets sammeln
    all_price_dates = sorted({
        d for ps in price_cache.values() if ps is not None
        for d in ps.index
    })
    full_price_df = pd.DataFrame(index=all_price_dates, columns=all_assets, dtype=float)
    for asset in all_assets:
        ps = price_cache.get(asset)
        if ps is None:
            continue
        ps_clean = ps[~ps.index.duplicated(keep='last')].sort_index()
        full_price_df[asset] = ps_clean.reindex(all_price_dates, method='ffill')

    # Forward-Return-Matrix: Preis in `horizon` Handelstagen / Preis heute - 1
    fwd_ret_df = (full_price_df.shift(-horizon) / full_price_df - 1)

    # ── Täglicher IC ─────────────────────────────────────────────────────────
    ic_values: dict[pd.Timestamp, float] = {}
    for date, scores in score_log:
        # Datum normalisieren (tz-strip)
        ts = date.tz_localize(None) if (hasattr(date, 'tzinfo') and date.tzinfo) else date
        if ts not in fwd_ret_df.index:
            # nächsten verfügbaren Handelstag suchen
            later = [d for d in fwd_ret_df.index if d >= ts]
            if not later:
                continue
            ts = later[0]
        fwd_row = fwd_ret_df.loc[ts]
        common  = scores.index.intersection(fwd_row.dropna().index)
        if len(common) < 10:
            continue
        corr, _ = spearmanr(scores[common].values, fwd_row[common].values)
        if not np.isnan(corr):
            ic_values[ts] = float(corr)

    ic = pd.Series(ic_values).sort_index()

    # ── Rolling-ICs ──────────────────────────────────────────────────────────
    rolling: dict[int, pd.Series] = {}
    for w in windows:
        rolling[w] = ic.rolling(window=w, min_periods=1).mean()

    # ── Records für JSON/CSV ──────────────────────────────────────────────────
    records = []
    for ts in ic.index:
        rec = {'date': str(ts.date()), 'ic': round(float(ic[ts]), 6)}
        for w in windows:
            rec[f'ic_roll_{w}'] = round(float(rolling[w][ts]), 6)
        records.append(rec)

    return {'ic': ic, 'rolling': rolling, 'records': records}


def _log_ic_summary(ic: pd.Series, rolling: dict[int, pd.Series], tag: str) -> None:
    """Gibt IC-Statistiken als Log-Zusammenfassung aus."""
    if ic.empty:
        logger.warning(f"[{tag}] IC-Zeitreihe leer – keine IC-Statistiken verfügbar.")
        return

    n_neg   = (ic < 0).sum()
    pct_neg = n_neg / len(ic) * 100

    logger.info("─" * 60)
    logger.info(f"[{tag}] Rolling Rank-IC Monitor  (HARD_STOP_PCT=0.20)")
    logger.info("─" * 60)
    logger.info(f"[{tag}]   Tage gesamt      : {len(ic)}")
    logger.info(f"[{tag}]   IC Median        : {ic.median():+.4f}")
    logger.info(f"[{tag}]   IC Mean          : {ic.mean():+.4f}")
    logger.info(f"[{tag}]   IC Std           : {ic.std():.4f}")
    logger.info(f"[{tag}]   Tage IC < 0      : {n_neg} ({pct_neg:.1f}%)")
    for w in [20, 60]:
        if w in rolling and not rolling[w].empty:
            r = rolling[w]
            pct_neg_r = (r < 0).sum() / len(r) * 100
            logger.info(f"[{tag}]   Roll-{w:2d} Median  : {r.median():+.4f}  "
                        f"(Tage < 0: {pct_neg_r:.1f}%)")
    # Jahres-ICs
    logger.info(f"[{tag}]   Jahres-IC:")
    for year, grp in ic.groupby(ic.index.year):
        logger.info(f"[{tag}]     {year}: Ø={grp.mean():+.4f}  Median={grp.median():+.4f}  "
                    f"n={len(grp)}")
    logger.info("─" * 60)


def save_ic_artifacts(
    ic_data:  dict,
    horizon:  int,
    out_dir:  str | Path,
) -> tuple[Path, Path]:
    """
    Speichert IC-Zeitreihe als JSON und CSV.

    Returns
    -------
    (json_path, csv_path)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem     = f"rolling_ic_v2_{horizon}d"
    json_path = out_dir / f"{stem}.json"
    csv_path  = out_dir / f"{stem}.csv"

    json_path.write_text(json.dumps(ic_data['records'], indent=2))

    if ic_data['records']:
        df = pd.DataFrame(ic_data['records'])
        df.to_csv(csv_path, index=False)

    logger.success(f"IC-Artefakte gespeichert: {json_path.name}, {csv_path.name}")
    return json_path, csv_path


# ── Modell laden ──────────────────────────────────────────────────────────────

def load_fold_model(ckpt_path: str, device: str) -> tuple[SingleHorizonRankModel, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    c = ckpt["config"]
    model = SingleHorizonRankModel(
        n_features=c["n_features"], n_assets=c["n_assets"],
        embed_dim=c["embed_dim"], hidden_dim=c["hidden_dim"],
        num_layers=c["num_layers"], dropout=0.0, seq_len=c["seq_len"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


# ── Cross-Section Prediction ──────────────────────────────────────────────────

@torch.no_grad()
def predict_cross_section(
    model:     SingleHorizonRankModel,
    features:  pd.DataFrame,
    asset_map: dict[str, int],
    date:      pd.Timestamp,
    seq_len:   int,
    device:    str,
) -> pd.Series:
    """Score pro Asset fuer einen Tag — absteigend sortiert."""
    try:
        assets_today = features.xs(date, level='date').index.tolist()
    except KeyError:
        return pd.Series(dtype=float)

    scores = {}
    for asset in assets_today:
        asset_id = asset_map.get(asset, 0)
        try:
            asset_feat = features.xs(asset, level='asset').sort_index()
        except KeyError:
            continue
        past = asset_feat[asset_feat.index <= date].iloc[-seq_len:]
        if len(past) < seq_len:
            continue
        x = torch.from_numpy(past.values.astype(np.float32)).unsqueeze(0).to(device)
        a = torch.tensor([asset_id], dtype=torch.long).to(device)
        score = model(x, a).item()
        scores[asset] = score

    return pd.Series(scores).sort_values(ascending=False)


# ── Backtest Engine ───────────────────────────────────────────────────────────

def run_backtest_single_horizon(
    features:     pd.DataFrame,
    fold_results: list[dict],
    asset_map:    dict[str, int],
    cfg:          SingleHorizonConfig,
    price_cache:  Optional[dict] = None,
) -> dict:
    """Run-G-identischer Backtest fuer einen einzelnen Horizont."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = cfg.tag

    all_assets = list(asset_map.keys())
    if "SPY" not in all_assets:
        all_assets.append("SPY")
    if price_cache is None:
        price_cache = build_price_cache(all_assets)

    spy_prices = price_cache.get("SPY")
    use_regime = spy_prices is not None

    all_dates = features.index.get_level_values('date').unique().sort_values()

    cash      = cfg.init_cash
    equity    = [cfg.init_cash]
    positions = {}
    trade_log = []
    equity_dates  = []
    daily_signals: list[dict] = []
    score_log:    list[tuple] = []   # [(date, ranking_series)] für IC-Berechnung

    logger.info(f"[{tag} Backtest] Long-Only  n_max={cfg.n_max}  "
                f"rotation_buffer={cfg.rotation_buffer}  "
                f"hard_stop={cfg.hard_stop_pct*100:.0f}%  "
                f"[HARD_STOP_PCT={cfg.hard_stop_pct:.2f}]")

    def _pos_val(positions, pc, dt):
        val = 0.0
        for a, p in positions.items():
            pr = _get_price(pc, a, dt)
            if pr is not None:
                val += p['shares'] * pr * p['direction']
        return val

    def _close(asset, pos, date, reason, regime):
        nonlocal cash
        price = _get_price(price_cache, asset, date)
        if price is None:
            return
        proceeds = pos['shares'] * price * (1 - cfg.fees) * pos['direction']
        cash += proceeds
        pnl = (price - pos['entry']) / pos['entry'] * pos['direction']
        trade_log.append({
            'date': str(date.date()), 'asset': asset,
            'direction': 'long', 'pnl_pct': round(pnl * 100, 3),
            'regime': regime, 'exit_reason': reason,
            'hold_days': pos.get('hold_days', 0),
        })

    for fold in fold_results:
        ckpt_path = fold['ckpt_path']
        if not Path(ckpt_path).exists():
            logger.warning(f"[{tag}] Checkpoint nicht gefunden: {ckpt_path}")
            continue

        model, _ = load_fold_model(ckpt_path, device)
        val_start = pd.Timestamp(fold['val_start'])
        val_end   = pd.Timestamp(fold['val_end'])

        cmp_dates = all_dates.tz_localize(None) if getattr(all_dates, 'tz', None) else all_dates
        vs = val_start.tz_localize(None) if val_start.tzinfo else val_start
        ve = val_end.tz_localize(None)   if val_end.tzinfo   else val_end
        fold_dates = all_dates[(cmp_dates >= vs) & (cmp_dates <= ve)]

        logger.info(f"[{tag}]   Fold {fold['fold_id']}: [{vs.date()} -> {ve.date()}]")

        for date in fold_dates:
            for pos in positions.values():
                pos['hold_days'] = pos.get('hold_days', 0) + 1

            regime = 'neutral'
            if use_regime:
                regime = get_market_regime(spy_prices, date)
            n_long = adaptive_n(regime, cfg.n_max, cfg.n_mid, cfg.n_min)

            ranking = predict_cross_section(model, features, asset_map, date, cfg.seq_len, device)
            if len(ranking) >= 2:
                score_log.append((date, ranking.copy()))   # OOS-Scores für IC sammeln
            if len(ranking) < 2:
                equity_dates.append(date)
                equity.append(cash + _pos_val(positions, price_cache, date))
                continue

            # Hard-Stop
            to_close = []
            for asset, pos in positions.items():
                price = _get_price(price_cache, asset, date)
                if price is None:
                    continue
                pnl = (price - pos['entry']) / pos['entry'] * pos['direction']
                if pnl <= -cfg.hard_stop_pct:
                    to_close.append((asset, 'hard_stop'))
            for asset, reason in to_close:
                _close(asset, positions[asset], date, reason, regime)
                del positions[asset]

            # Rotation
            top_n = list(ranking.index[:n_long])
            to_close = []
            for asset in list(positions.keys()):
                if asset in ranking.index:
                    rank_pos = list(ranking.index).index(asset)
                    if rank_pos >= n_long + cfg.rotation_buffer:
                        to_close.append((asset, 'rotation'))
                else:
                    to_close.append((asset, 'rotation'))
            for asset, reason in to_close:
                _close(asset, positions[asset], date, reason, regime)
                del positions[asset]

            # Neue Positionen
            free_slots = n_long - len(positions)
            if free_slots > 0:
                for cand in top_n:
                    if free_slots <= 0:
                        break
                    if cand in positions:
                        continue
                    price = _get_price(price_cache, cand, date)
                    if price is None or price <= 0:
                        continue
                    alloc = (cash + _pos_val(positions, price_cache, date)) / n_long
                    shares = alloc * (1 - cfg.fees) / price
                    if shares * price < 100:
                        continue
                    cash -= shares * price * (1 + cfg.fees)
                    positions[cand] = {
                        'shares': shares, 'entry': price, 'direction': 1, 'hold_days': 0,
                    }
                    free_slots -= 1

            eq = cash + _pos_val(positions, price_cache, date)
            equity.append(eq)
            equity_dates.append(date)

            daily_signals.append({
                'date': str(date.date()), 'regime': regime,
                'n_long': n_long, 'n_positions': len(positions),
                'score_top1': round(float(ranking.iloc[0]) * 100, 4) if len(ranking) > 0 else 0,
                'equity': round(eq, 2),
            })

    # ── Statistiken ───────────────────────────────────────────────────────────
    equity_arr = np.array(equity[1:])
    returns = np.diff(equity) / equity[:-1]
    returns = returns[1:]

    total_return = (equity[-1] / equity[0] - 1) * 100 if equity[0] > 0 else 0
    peaks = np.maximum.accumulate(equity_arr)
    dd = (equity_arr - peaks) / peaks
    max_dd = float(dd.min()) * 100 if len(dd) > 0 else 0

    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252))
    else:
        sharpe = 0.0

    n_trades = len(trade_log)
    wins = [t for t in trade_log if t['pnl_pct'] > 0]
    win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0
    avg_hold = np.mean([t['hold_days'] for t in trade_log]) if trade_log else 0

    exit_stats = {}
    for reason in ['rotation', 'hard_stop']:
        trades_r = [t for t in trade_log if t['exit_reason'] == reason]
        n_r = len(trades_r)
        exit_stats[reason] = {
            'n':        n_r,
            'pnl_sum':  round(sum(t['pnl_pct'] for t in trades_r), 1),
            'pnl_avg':  round(np.mean([t['pnl_pct'] for t in trades_r]), 2) if trades_r else 0,
            'hold_avg': round(np.mean([t['hold_days'] for t in trades_r]), 1) if trades_r else 0,
            'win_pct':  round(len([t for t in trades_r if t['pnl_pct'] > 0]) / n_r * 100, 1) if n_r > 0 else 0,
        }

    logger.success("═" * 60)
    logger.success(f"[{tag}] BACKTEST: Long-Only")
    logger.success("═" * 60)
    logger.success(f"[{tag}]   HARD_STOP_PCT  : {cfg.hard_stop_pct:.2f}  ({cfg.hard_stop_pct*100:.0f}%)")
    logger.success(f"[{tag}]   Total Return   : {total_return:+.2f}%")
    logger.success(f"[{tag}]   Max Drawdown   : {max_dd:.2f}%")
    logger.success(f"[{tag}]   Sharpe Ratio   : {sharpe:.3f}")
    logger.success(f"[{tag}]   Trades         : {n_trades}")
    logger.success(f"[{tag}]   Win Rate       : {win_rate:.1f}%")
    logger.success(f"[{tag}]   Avg Hold Days  : {avg_hold:.1f}")
    for reason, st in exit_stats.items():
        logger.success(f"[{tag}]   {reason:15s}: n={st['n']:4d}  pnl={st['pnl_sum']:+.0f}%  "
                       f"avg={st['pnl_avg']:+.1f}%  hold={st['hold_avg']:.1f}d  win={st['win_pct']:.0f}%")
    logger.success("═" * 60)

    # ── Rolling Rank-IC Monitor ────────────────────────────────────────────────
    logger.info(f"[{tag}] Rolling Rank-IC berechnen ({len(score_log)} OOS-Tage) ...")
    try:
        ic_data = _build_ic_series(score_log, price_cache, horizon=cfg.horizon)
        _log_ic_summary(ic_data['ic'], ic_data['rolling'], tag)
    except Exception as exc:
        logger.warning(f"[{tag}] IC-Berechnung fehlgeschlagen: {exc}")
        ic_data = {'ic': pd.Series(dtype=float), 'rolling': {}, 'records': []}

    return {
        'strategy':      tag,
        'horizon':       cfg.horizon,
        'total_return':  round(total_return, 2),
        'max_drawdown':  round(max_dd, 2),
        'sharpe':        round(sharpe, 3),
        'n_trades':      n_trades,
        'win_rate':      round(win_rate, 1),
        'avg_hold_days': round(avg_hold, 1),
        'exit_stats':    exit_stats,
        'equity':        equity,
        'equity_dates':  equity_dates,
        'trade_log':     trade_log,
        'daily_signals': daily_signals,
        'ic_data':       ic_data,        # Rolling-IC-Zeitreihe
    }


# ── Alle Horizonte backtesten ─────────────────────────────────────────────────

def backtest_all_horizons(
    features:      pd.DataFrame,
    all_train_results: dict[int, dict],
    asset_map:     dict[str, int],
    price_cache:   Optional[dict] = None,
) -> dict[int, dict]:
    """Backtest fuer alle Horizonte, gibt {horizon: result} zurueck."""
    all_bt = {}
    for h, train_res in all_train_results.items():
        cfg = get_config(h)
        bt = run_backtest_single_horizon(
            features=features,
            fold_results=train_res["fold_results"],
            asset_map=asset_map,
            cfg=cfg,
            price_cache=price_cache,
        )
        all_bt[h] = bt
    return all_bt


# ── Benchmark-Report ──────────────────────────────────────────────────────────

V1_REFERENCE = {
    'strategy':     'v1_rank',
    'horizon':      11,
    'total_return':  403.93,
    'max_drawdown': -55.48,
    'sharpe':        0.784,
    'n_trades':      471,
    'win_rate':      52.4,
    'avg_hold_days': 14.8,
    'mean_ic':       0.035,
}


def build_horizon_benchmark_report(
    all_bt_results:     dict[int, dict],
    all_train_results:  dict[int, dict],
    v1_result:          Optional[dict] = None,
    save_path:          Optional[str]  = None,
) -> dict:
    """
    Erstellt Vergleichstabelle: v1_rank vs. v2_4d/7d/11d/15d.
    """
    v1 = v1_result if v1_result else V1_REFERENCE
    v1_ic = v1.get('mean_ic', V1_REFERENCE['mean_ic'])

    rows = []
    rows.append({
        'model': 'v1_rank', 'horizon': '11d',
        'total_return': v1.get('total_return', V1_REFERENCE['total_return']),
        'max_drawdown': v1.get('max_drawdown', V1_REFERENCE['max_drawdown']),
        'sharpe':       v1.get('sharpe',       V1_REFERENCE['sharpe']),
        'n_trades':     v1.get('n_trades',     V1_REFERENCE['n_trades']),
        'win_rate':     v1.get('win_rate',     V1_REFERENCE['win_rate']),
        'avg_hold':     v1.get('avg_hold_days', V1_REFERENCE['avg_hold_days']),
        'rank_ic':      v1_ic,
    })

    for h in sorted(all_bt_results.keys()):
        bt = all_bt_results[h]
        tr = all_train_results.get(h, {})
        rows.append({
            'model':        bt.get('strategy', f'v2_{h}d'),
            'horizon':      f'{h}d',
            'total_return':  bt.get('total_return', 0),
            'max_drawdown':  bt.get('max_drawdown', 0),
            'sharpe':        bt.get('sharpe', 0),
            'n_trades':      bt.get('n_trades', 0),
            'win_rate':      bt.get('win_rate', 0),
            'avg_hold':      bt.get('avg_hold_days', 0),
            'rank_ic':       tr.get('mean_ic', 0),
        })

    # Tabelle loggen
    logger.info("\n" + "═" * 95)
    logger.info("HORIZONT-VERGLEICH: v1_rank vs. v2 Single-Horizon Modelle")
    logger.info("═" * 95)
    header = (f"{'Modell':<12} {'Horizon':>7} {'TotRet':>9} {'MaxDD':>8} "
              f"{'Sharpe':>7} {'Trades':>7} {'Win%':>6} {'AvgHold':>8} {'RankIC':>8}")
    logger.info(header)
    logger.info("─" * 95)
    for r in rows:
        logger.info(
            f"{r['model']:<12} {r['horizon']:>7} {r['total_return']:>+9.1f}% "
            f"{r['max_drawdown']:>8.1f}% {r['sharpe']:>7.3f} {r['n_trades']:>7} "
            f"{r['win_rate']:>5.1f}% {r['avg_hold']:>7.1f}d {r['rank_ic']:>8.4f}"
        )
    logger.info("═" * 95)

    # Bestes v2-Modell identifizieren
    v2_rows = [r for r in rows if r['model'] != 'v1_rank']
    if v2_rows:
        best = max(v2_rows, key=lambda r: r['sharpe'])
        logger.success(f"\nBestes v2-Modell nach Sharpe: {best['model']}  "
                       f"(Sharpe={best['sharpe']:.3f}  Return={best['total_return']:+.1f}%)")

    report = {"rows": rows}
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.success(f"Benchmark-Report: {save_path}")

    return report


# ── Equity-Vergleichs-Plot ────────────────────────────────────────────────────

def plot_horizon_comparison(
    all_bt_results: dict[int, dict],
    v1_result:      Optional[dict]  = None,
    benchmarks:     Optional[dict]  = None,
    save_path:      str = "horizon_comparison_equity.png",
):
    """Equity-Kurven: v1_rank vs. v2_4d/7d/11d/15d."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        colors = {
            'v1_rank': '#1565C0',
            4:  '#E65100',
            7:  '#2E7D32',
            11: '#6A1B9A',
            15: '#C62828',
        }

        fig, axes = plt.subplots(2, 1, figsize=(16, 11), gridspec_kw={'height_ratios': [3, 1]})

        ax = axes[0]

        # v1
        if v1_result and v1_result.get('equity_dates') and v1_result.get('equity'):
            dates = v1_result['equity_dates']
            eq = v1_result['equity']
            eq_plot = eq[1:] if len(eq) > len(dates) else eq
            ret = [(e / eq_plot[0] - 1) * 100 for e in eq_plot]
            ax.plot(dates, ret, color=colors['v1_rank'], linewidth=2.0,
                    label=f"v1_rank (11d) +{v1_result.get('total_return',0):.1f}%")

        # v2 Horizonte
        for h in sorted(all_bt_results.keys()):
            bt = all_bt_results[h]
            dates = bt.get('equity_dates', [])
            eq = bt.get('equity', [])
            if not dates or not eq:
                continue
            eq_plot = eq[1:] if len(eq) > len(dates) else eq
            ret = [(e / eq_plot[0] - 1) * 100 for e in eq_plot]
            ax.plot(dates, ret, color=colors.get(h, '#333'), linewidth=1.5,
                    label=f"v2_{h}d +{bt.get('total_return',0):.1f}%")

        if benchmarks:
            for key, color, ls in [('spy', '#FFA000', '--'), ('ew_bh', '#7B1FA2', ':')]:
                bm = benchmarks.get(key, {})
                if bm.get('equity') and bm.get('dates'):
                    eq = bm['equity']
                    ret = [(e / eq[0] - 1) * 100 for e in eq]
                    ax.plot(bm['dates'][:len(ret)], ret, color=color, linestyle=ls,
                            label=f"{bm.get('label',key)} +{bm.get('total_return',0):.1f}%")

        ax.set_title("Horizont-Vergleich: v1_rank vs. v2 Single-Horizon", fontsize=14, fontweight='bold')
        ax.set_ylabel("Kumulativer Return (%)")
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Drawdown
        ax2 = axes[1]
        items = []
        if v1_result and v1_result.get('equity_dates'):
            items.append(('v1_rank', v1_result, colors['v1_rank']))
        for h in sorted(all_bt_results.keys()):
            items.append((f'v2_{h}d', all_bt_results[h], colors.get(h, '#333')))

        for label, bt, color in items:
            dates = bt.get('equity_dates', [])
            eq = bt.get('equity', [])
            if not dates or not eq:
                continue
            eq_arr = np.array(eq[1:] if len(eq) > len(dates) else eq)
            peaks = np.maximum.accumulate(eq_arr)
            dd = (eq_arr - peaks) / peaks * 100
            ax2.fill_between(dates, dd, 0, alpha=0.2, color=color,
                             label=f"{label} (MaxDD: {dd.min():.1f}%)")

        ax2.set_ylabel("Drawdown (%)")
        ax2.legend(loc='lower left', fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.success(f"Plot: {save_path}")

    except Exception as e:
        logger.warning(f"Plot-Fehler: {e}")


# ── Equity-Chart: einzelner Horizont mit Benchmarks ──────────────────────────

def plot_equity_single(
    bt_result:  dict,
    benchmarks: Optional[dict] = None,
    run_id:     str            = "",
    save_path:  str            = "v2_7d_equity.png",
):
    """Detaillierter Equity-Chart für einen einzelnen Horizont.

    Zeigt oben die kumulativen Returns (Portfolio vs. SPY vs. EW-Universe)
    und unten den Drawdown.  Metriken werden als Text-Box eingeblendet.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        from matplotlib.gridspec import GridSpec

        h           = bt_result.get('horizon', '?')
        tag         = bt_result.get('strategy', f'v2_{h}d')
        dates       = bt_result.get('equity_dates', [])
        eq          = bt_result.get('equity', [])
        trade_log   = bt_result.get('trade_log', [])

        if not dates or not eq:
            logger.warning(f"plot_equity_single: keine Equity-Daten für {tag}")
            return

        eq_plot = eq[1:] if len(eq) > len(dates) else eq
        ret_pct = [(e / eq_plot[0] - 1) * 100 for e in eq_plot]

        eq_arr  = np.array(eq_plot)
        peaks   = np.maximum.accumulate(eq_arr)
        dd_pct  = (eq_arr - peaks) / peaks * 100

        title_tag = f"{tag}  |  {run_id}" if run_id else tag

        fig = plt.figure(figsize=(16, 10))
        gs  = GridSpec(2, 1, figure=fig, height_ratios=[3, 1], hspace=0.08)

        ax_eq = fig.add_subplot(gs[0])
        ax_dd = fig.add_subplot(gs[1], sharex=ax_eq)

        # ── Portfolio ─────────────────────────────────────────────────────────
        ax_eq.plot(dates, ret_pct, color='#1565C0', linewidth=2.2,
                   label=f"{tag}  {bt_result.get('total_return', 0):+.1f}%")

        # ── Benchmarks ────────────────────────────────────────────────────────
        bm_styles = {
            'spy':          ('#FFA000', '--', 'SPY Buy & Hold'),
            'ew_bh':        ('#7B1FA2', ':',  'EW-Universe Buy & Hold'),
            'ew_rebalanced':('#388E3C', '-.', 'EW-Universe rebalanciert'),
        }
        if benchmarks:
            for key, (color, ls, lbl) in bm_styles.items():
                bm = benchmarks.get(key, {})
                bm_eq   = bm.get('equity', [])
                bm_dts  = bm.get('dates',  [])
                if not bm_eq or not bm_dts:
                    continue
                bm_ret = [(e / bm_eq[0] - 1) * 100 for e in bm_eq]
                n = min(len(bm_dts), len(bm_ret))
                ax_eq.plot(bm_dts[:n], bm_ret[:n],
                           color=color, linestyle=ls, linewidth=1.4,
                           label=f"{lbl}  {bm.get('total_return', 0):+.1f}%")

        # ── Metriken-Box ──────────────────────────────────────────────────────
        wins = [t for t in trade_log if t.get('pnl_pct', 0) > 0]
        n_tr = len(trade_log)
        wr   = len(wins) / n_tr * 100 if n_tr else 0
        ah   = np.mean([t['hold_days'] for t in trade_log]) if trade_log else 0

        info = (
            f"Horizont   : {h}d\n"
            f"Return     : {bt_result.get('total_return', 0):+.2f}%\n"
            f"Max DD     : {bt_result.get('max_drawdown', 0):.2f}%\n"
            f"Sharpe     : {bt_result.get('sharpe', 0):.3f}\n"
            f"Trades     : {n_tr}\n"
            f"Win Rate   : {wr:.1f}%\n"
            f"Avg Hold   : {ah:.1f}d"
        )
        ax_eq.text(
            0.01, 0.98, info,
            transform=ax_eq.transAxes, va='top', fontsize=9,
            fontfamily='monospace',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='#ccc', boxstyle='round,pad=0.4'),
        )

        ax_eq.set_title(f"Equity-Kurve: {title_tag}", fontsize=13, fontweight='bold')
        ax_eq.set_ylabel("Kumulativer Return (%)")
        ax_eq.legend(loc='upper left', fontsize=9)
        ax_eq.grid(True, alpha=0.25)
        ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.0f}%"))
        plt.setp(ax_eq.get_xticklabels(), visible=False)

        # ── Drawdown ──────────────────────────────────────────────────────────
        ax_dd.fill_between(dates, dd_pct, 0, alpha=0.35, color='#C62828',
                           label=f"MaxDD {dd_pct.min():.1f}%")
        ax_dd.plot(dates, dd_pct, color='#C62828', linewidth=0.8)
        ax_dd.set_ylabel("Drawdown (%)")
        ax_dd.set_xlabel("Datum")
        ax_dd.legend(loc='lower left', fontsize=8)
        ax_dd.grid(True, alpha=0.25)
        ax_dd.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.success(f"Equity-Chart gespeichert: {save_path}")

    except Exception as e:
        logger.warning(f"plot_equity_single Fehler: {e}")
