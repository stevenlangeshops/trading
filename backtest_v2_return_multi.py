"""
backtest_v2_return_multi.py
────────────────────────────
Backtest für das Multi-Horizon Return-Modell v2.

Identische Strategie-Logik wie Run G (v1_rank):
  - Regime-Filter (SMA50/SMA200)
  - Rotation + Hard-Stop (25%)
  - n_max=7, rotation_buffer=3

Einziger Unterschied: statt v1-Score wird pred[portfolio_horizon] oder
ein combo_score als Ranking-Signal verwendet.

Output: dict mit gleicher Struktur wie run_backtest() in strategy/backtest.py,
damit v1 und v2 direkt verglichen werden können.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger

from config_v2_return_multi import V2Config
from models_v2_return_multi import LSTMReturnMultiV2
from strategy.backtest import (
    build_price_cache,
    get_market_regime,
    adaptive_n,
    _get_price,
    compute_benchmarks,
    plot_equity,
)


# ── v2 Modell laden ───────────────────────────────────────────────────────────

def load_v2_fold_model(ckpt_path: str, device: str) -> tuple[LSTMReturnMultiV2, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    c = ckpt["config"]
    model = LSTMReturnMultiV2(
        n_features=c["n_features"], n_assets=c["n_assets"],
        n_horizons=c["n_horizons"], embed_dim=c["embed_dim"],
        hidden_dim=c["hidden_dim"], num_layers=c["num_layers"],
        dropout=0.0, seq_len=c["seq_len"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


# ── Cross-Section Prediction ──────────────────────────────────────────────────

@torch.no_grad()
def predict_cross_section_v2(
    model:     LSTMReturnMultiV2,
    features:  pd.DataFrame,
    asset_map: dict[str, int],
    date:      pd.Timestamp,
    seq_len:   int,
    device:    str,
    cfg:       V2Config,
) -> pd.Series:
    """
    Generiert für einen Tag t Vorhersagen für alle Assets.
    Gibt eine pd.Series asset → ranking_score (sortiert absteigend) zurück.
    Zusätzlich speichert es die vollen Predictions als Attribut.
    """
    try:
        assets_today = features.xs(date, level='date').index.tolist()
    except KeyError:
        return pd.Series(dtype=float)

    scores = {}
    all_preds_dict = {}

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
        pred = model(x, a).squeeze(0).cpu().numpy()  # (n_horizons,)

        all_preds_dict[asset] = pred

        # Ranking-Score berechnen
        if cfg.combo_weights:
            score = sum(pred[idx] * w for idx, w in cfg.combo_weights.items())
        else:
            score = float(pred[cfg.portfolio_horizon_idx])

        scores[asset] = score

    result = pd.Series(scores).sort_values(ascending=False)
    result.attrs['full_preds'] = all_preds_dict
    return result


# ── Backtest Engine ───────────────────────────────────────────────────────────

def run_backtest_v2(
    features:      pd.DataFrame,
    targets_multi: pd.DataFrame,
    fold_results:  list[dict],
    asset_map:     dict[str, int],
    cfg:           V2Config = V2Config(),
    price_cache:   Optional[dict] = None,
) -> dict:
    """
    Run-G-identischer Backtest mit v2_return_multi Modell.

    Logik:
      - Tägliches Cross-Sectional Ranking nach portfolio_horizon (oder combo)
      - Regime-Filter (SMA50/SMA200) → adaptives N
      - Rotation mit rotation_buffer
      - Hard-Stop 25%
      - Kein ATR/DD/Crash/Signal-Filter
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_assets = list(asset_map.keys())
    spy_ticker = "SPY"
    if spy_ticker not in all_assets:
        all_assets.append(spy_ticker)

    if price_cache is None:
        price_cache = build_price_cache(all_assets)

    spy_prices = price_cache.get(spy_ticker)
    use_regime = spy_prices is not None

    all_dates = features.index.get_level_values('date').unique().sort_values()

    cash      = cfg.init_cash
    equity    = [cfg.init_cash]
    positions = {}
    trade_log = []
    equity_dates = []
    daily_signals: list[dict] = []

    h_label = f"{cfg.portfolio_horizon}d"
    if cfg.combo_weights:
        combo_str = "+".join(f"{w:.1f}*{cfg.horizons[i]}d" for i, w in cfg.combo_weights.items())
        h_label = f"combo({combo_str})"

    logger.info(f"[v2 Backtest] Long-Only  horizon={h_label}  n_max={cfg.n_max}  "
                f"rotation_buffer={cfg.rotation_buffer}  hard_stop={cfg.hard_stop_pct*100:.0f}%")

    def _position_value(positions, pc, dt):
        val = 0.0
        for a, p in positions.items():
            pr = _get_price(pc, a, dt)
            if pr is not None:
                val += p['shares'] * pr * p['direction']
        return val

    def _close_position(asset, pos, date, reason, regime):
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
            logger.warning(f"[v2] Checkpoint nicht gefunden: {ckpt_path}")
            continue

        model, _ = load_v2_fold_model(ckpt_path, device)
        val_start = pd.Timestamp(fold['val_start'])
        val_end   = pd.Timestamp(fold['val_end'])

        cmp_dates = all_dates.tz_localize(None) if getattr(all_dates, 'tz', None) else all_dates
        vs = val_start.tz_localize(None) if val_start.tzinfo else val_start
        ve = val_end.tz_localize(None)   if val_end.tzinfo   else val_end
        fold_dates = all_dates[(cmp_dates >= vs) & (cmp_dates <= ve)]

        logger.info(f"[v2]   Fold {fold['fold_id']}: [{vs.date()} → {ve.date()}]")

        for date in fold_dates:
            # Haltetage erhöhen
            for pos in positions.values():
                pos['hold_days'] = pos.get('hold_days', 0) + 1

            # Regime
            regime = 'neutral'
            if use_regime:
                regime = get_market_regime(spy_prices, date)
            n_long = adaptive_n(regime, cfg.n_max, cfg.n_mid, cfg.n_min)

            # Predictions
            preds = predict_cross_section_v2(model, features, asset_map, date, cfg.seq_len, device, cfg)
            if len(preds) < 2:
                equity_dates.append(date)
                equity.append(cash + _position_value(positions, price_cache, date))
                continue

            full_preds = preds.attrs.get('full_preds', {})
            score_top1 = float(preds.iloc[0]) if len(preds) > 0 else 0.0

            # ── Hard-Stop prüfen ──────────────────────────────────────────
            to_close = []
            for asset, pos in positions.items():
                price = _get_price(price_cache, asset, date)
                if price is None:
                    continue
                pnl = (price - pos['entry']) / pos['entry'] * pos['direction']
                if pnl <= -cfg.hard_stop_pct:
                    to_close.append((asset, 'hard_stop'))

            for asset, reason in to_close:
                _close_position(asset, positions[asset], date, reason, regime)
                del positions[asset]

            # ── Rotation ──────────────────────────────────────────────────
            top_n_assets = list(preds.index[:n_long])
            to_close = []
            for asset in list(positions.keys()):
                if asset in preds.index:
                    rank = list(preds.index).index(asset)
                    if rank >= n_long + cfg.rotation_buffer:
                        to_close.append((asset, 'rotation'))
                else:
                    to_close.append((asset, 'rotation'))

            for asset, reason in to_close:
                _close_position(asset, positions[asset], date, reason, regime)
                del positions[asset]

            # ── Neue Positionen eröffnen ──────────────────────────────────
            free_slots = n_long - len(positions)
            if free_slots > 0:
                for candidate in top_n_assets:
                    if free_slots <= 0:
                        break
                    if candidate in positions:
                        continue

                    price = _get_price(price_cache, candidate, date)
                    if price is None or price <= 0:
                        continue

                    alloc = (cash + _position_value(positions, price_cache, date)) / n_long
                    shares = alloc * (1 - cfg.fees) / price

                    if shares * price < 100:
                        continue

                    cash -= shares * price * (1 + cfg.fees)
                    positions[candidate] = {
                        'shares': shares, 'entry': price, 'direction': 1,
                        'hold_days': 0,
                    }
                    free_slots -= 1

            # Equity tracken
            eq = cash + _position_value(positions, price_cache, date)
            equity.append(eq)
            equity_dates.append(date)

            # Tägliche Signal-Daten
            pred_h_vals = {}
            if full_preds and len(preds) > 0:
                top1_asset = preds.index[0]
                top1_full  = full_preds.get(top1_asset, [])
                for i, h in enumerate(cfg.horizons):
                    if i < len(top1_full):
                        pred_h_vals[f"pred_{h}d_top1"] = round(float(top1_full[i]) * 100, 4)

            daily_signals.append({
                'date': str(date.date()),
                'regime': regime,
                'n_long': n_long,
                'n_positions': len(positions),
                'score_top1': round(score_top1 * 100, 4),
                'equity': round(eq, 2),
                **pred_h_vals,
            })

    # ── Ergebnis-Statistiken ──────────────────────────────────────────────────
    equity_arr = np.array(equity[1:])
    returns = np.diff(equity) / equity[:-1]
    returns = returns[1:]  # erstes Element weglassen

    total_return = (equity[-1] / equity[0] - 1) * 100 if equity[0] > 0 else 0

    # Max Drawdown
    peaks = np.maximum.accumulate(equity_arr)
    dd = (equity_arr - peaks) / peaks
    max_dd = float(dd.min()) * 100 if len(dd) > 0 else 0

    # Sharpe
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252))
    else:
        sharpe = 0.0

    n_trades = len(trade_log)
    wins = [t for t in trade_log if t['pnl_pct'] > 0]
    win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0
    avg_hold = np.mean([t['hold_days'] for t in trade_log]) if trade_log else 0

    # Exit-Stats
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
    logger.success(f"[v2] BACKTEST: Long-Only (horizon={h_label})")
    logger.success("═" * 60)
    logger.success(f"[v2]   Total Return : {total_return:+.2f}%")
    logger.success(f"[v2]   Max Drawdown : {max_dd:.2f}%")
    logger.success(f"[v2]   Sharpe Ratio : {sharpe:.3f}")
    logger.success(f"[v2]   Trades       : {n_trades}")
    logger.success(f"[v2]   Win Rate     : {win_rate:.1f}%")
    logger.success(f"[v2]   Avg Hold Days: {avg_hold:.1f}")
    for reason, st in exit_stats.items():
        logger.success(f"[v2]   {reason:15s}: n={st['n']:4d}  pnl={st['pnl_sum']:+.0f}%  "
                       f"avg={st['pnl_avg']:+.1f}%  hold={st['hold_avg']:.1f}d  win={st['win_pct']:.0f}%")
    logger.success("═" * 60)

    return {
        'strategy':      f"v2_return_multi_h{cfg.portfolio_horizon}",
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
        'horizons':      cfg.horizons,
        'portfolio_horizon': cfg.portfolio_horizon,
    }


# ── Benchmark-Vergleich ──────────────────────────────────────────────────────

def build_v1_vs_v2_report(
    v1_result: dict,
    v2_result: dict,
    save_path: Optional[str] = None,
) -> dict:
    """Erstellt einen JSON-Report mit v1_rank vs. v2_return_multi Kennzahlen."""

    def _extract(r: dict) -> dict:
        return {k: r.get(k) for k in [
            'total_return', 'max_drawdown', 'sharpe', 'n_trades', 'win_rate', 'avg_hold_days'
        ]}

    report = {
        "v1_rank_run_g":  _extract(v1_result),
        "v2_return_multi": _extract(v2_result),
    }

    # Delta
    report["delta"] = {}
    for k in ['total_return', 'max_drawdown', 'sharpe', 'n_trades', 'win_rate', 'avg_hold_days']:
        v1 = v1_result.get(k, 0) or 0
        v2 = v2_result.get(k, 0) or 0
        report["delta"][k] = round(v2 - v1, 2)

    logger.info("═" * 60)
    logger.info("VERGLEICH: v1_rank (Run G) vs. v2_return_multi")
    logger.info("═" * 60)
    logger.info(f"{'Kennzahl':<20s}  {'v1_rank':>12s}  {'v2_multi':>12s}  {'Delta':>10s}")
    logger.info("─" * 60)
    for k in ['total_return', 'max_drawdown', 'sharpe', 'n_trades', 'win_rate', 'avg_hold_days']:
        v1 = v1_result.get(k, 0) or 0
        v2 = v2_result.get(k, 0) or 0
        d  = report["delta"][k]
        logger.info(f"{k:<20s}  {v1:>12.2f}  {v2:>12.2f}  {d:>+10.2f}")
    logger.info("═" * 60)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.success(f"[v2] Benchmark-Report gespeichert: {save_path}")

    return report


# ── Vergleichs-Plot ───────────────────────────────────────────────────────────

def plot_v1_vs_v2(
    v1_result:  dict,
    v2_result:  dict,
    benchmarks: dict = None,
    save_path:  str  = "v1_vs_v2_equity.png",
):
    """Equity-Kurve: v1_rank vs. v2_return_multi vs. Benchmarks."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

        # Equity-Kurven
        ax = axes[0]
        for result, label, color in [
            (v1_result, f"v1_rank (Run G) +{v1_result.get('total_return',0):.1f}%", '#1565C0'),
            (v2_result, f"v2_multi h{v2_result.get('portfolio_horizon','?')} +{v2_result.get('total_return',0):.1f}%", '#E65100'),
        ]:
            dates = result.get('equity_dates', [])
            eq = result.get('equity', [])
            if dates and eq:
                eq_plot = eq[1:] if len(eq) > len(dates) else eq
                ret_pct = [(e / eq_plot[0] - 1) * 100 for e in eq_plot]
                ax.plot(dates, ret_pct, label=label, color=color, linewidth=1.5)

        if benchmarks:
            for key, color, ls in [('spy', '#FFA000', '--'), ('ew_bh', '#7B1FA2', ':')]:
                bm = benchmarks.get(key, {})
                if bm.get('equity') and bm.get('dates'):
                    eq = bm['equity']
                    ret_pct = [(e / eq[0] - 1) * 100 for e in eq]
                    ax.plot(bm['dates'][:len(ret_pct)], ret_pct, color=color,
                            linestyle=ls, label=f"{bm.get('label',key)} +{bm.get('total_return',0):.1f}%")

        ax.set_title("v1_rank (Run G) vs. v2_return_multi", fontsize=14, fontweight='bold')
        ax.set_ylabel("Kumulativer Return (%)")
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # Drawdown
        ax = axes[1]
        for result, label, color in [
            (v1_result, "v1_rank", '#1565C0'),
            (v2_result, "v2_multi", '#E65100'),
        ]:
            dates = result.get('equity_dates', [])
            eq = result.get('equity', [])
            if dates and eq:
                eq_arr = np.array(eq[1:] if len(eq) > len(dates) else eq)
                peaks = np.maximum.accumulate(eq_arr)
                dd = (eq_arr - peaks) / peaks * 100
                ax.fill_between(dates, dd, 0, alpha=0.3, color=color, label=f"{label} (MaxDD: {dd.min():.1f}%)")

        ax.set_ylabel("Drawdown (%)")
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.success(f"[v2] Plot gespeichert: {save_path}")

    except Exception as e:
        logger.warning(f"[v2] Plot-Fehler: {e}")
