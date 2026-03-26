"""
strategy/calibration.py
────────────────────────
Offline-Kalibrierung: Model-Score → Expected 11d Return.

Das LSTM gibt einen unkalibrierte Score aus (trainiert mit MSE + RankLoss).
Diese Kalibrierung lernt eine Abbildung score → E[return_11d], damit wir
einen Expected-Return-Filter mit sinnvoller Skala nutzen können.

Zwei Varianten:
  A) Lineare Regression   (Baseline, interpretierbar)
  B) Isotonic Regression   (monoton, nichtlinear, kein Overfitting-Risiko)

WICHTIG: Die Kalibrierung wird nur auf Out-of-Sample-Daten gefittet
(Walk-Forward Val-Perioden), sodass kein Lookahead-Bias entsteht.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger


# ── Daten sammeln ─────────────────────────────────────────────────────────────

def collect_score_return_pairs(
    features:     pd.DataFrame,
    targets:      pd.Series,
    fold_results: list[dict],
    asset_map:    dict[str, int],
    seq_len:      int = 64,
) -> pd.DataFrame:
    """
    Sammelt (date, asset, score, true_return_11d) über alle Walk-Forward
    Val-Perioden. Nutzt ausschließlich Out-of-Sample-Predictions.

    Returns:
        DataFrame mit Spalten: date, asset, score, true_return_11d
    """
    from strategy.backtest import load_fold_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_dates = features.index.get_level_values('date').unique().sort_values()
    rows: list[dict] = []

    for fold in fold_results:
        ckpt_path = fold['ckpt_path']
        if not Path(ckpt_path).exists():
            logger.warning(f"Kalibrierung: Checkpoint nicht gefunden: {ckpt_path}")
            continue

        model, _ = load_fold_model(ckpt_path, device)
        val_start = pd.Timestamp(fold['val_start'])
        val_end   = pd.Timestamp(fold['val_end'])

        cmp = all_dates.tz_localize(None) if getattr(all_dates, 'tz', None) else all_dates
        vs  = val_start.tz_localize(None) if val_start.tzinfo else val_start
        ve  = val_end.tz_localize(None) if val_end.tzinfo else val_end
        fold_dates = all_dates[(cmp >= vs) & (cmp <= ve)]

        logger.info(
            f"  Kalibrierung Fold {fold['fold_id']}: "
            f"{len(fold_dates)} Tage [{vs.date()} → {ve.date()}]"
        )

        for date in fold_dates:
            try:
                assets_today = features.xs(date, level='date').index.tolist()
            except KeyError:
                continue

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
                with torch.no_grad():
                    score = model(x, a).item()

                try:
                    true_ret = float(targets.loc[(date, asset)])
                except (KeyError, TypeError):
                    continue

                if np.isnan(true_ret):
                    continue

                rows.append({
                    'date':             date,
                    'asset':            asset,
                    'score':            score,
                    'true_return_11d':  true_ret,
                    'fold_id':          fold['fold_id'],
                })

    df = pd.DataFrame(rows)
    logger.info(f"  Kalibrierung: {len(df)} Score-Return-Paare gesammelt "
                f"({df['fold_id'].nunique()} Folds)")
    return df


# ── Kalibrierung fitten ──────────────────────────────────────────────────────

def fit_score_to_return_calibration(
    scores:  np.ndarray,
    returns: np.ndarray,
    method:  str = "isotonic",
) -> dict:
    """
    Fittet eine Kalibrierung Score → Expected Return.

    Args:
        method: "linear" oder "isotonic"

    Returns:
        dict mit 'model', 'method', 'coefficients' (für linear)
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.isotonic import IsotonicRegression

    mask = np.isfinite(scores) & np.isfinite(returns)
    s, r = scores[mask], returns[mask]

    if method == "linear":
        model = LinearRegression()
        model.fit(s.reshape(-1, 1), r)
        logger.info(
            f"  Linear Calib: a={model.intercept_:.6f}  b={model.coef_[0]:.6f}  "
            f"(score=0 → E[ret]={model.intercept_*100:.3f}%)"
        )
        return {'model': model, 'method': 'linear',
                'a': float(model.intercept_), 'b': float(model.coef_[0])}

    elif method == "isotonic":
        model = IsotonicRegression(out_of_bounds='clip', increasing=True)
        model.fit(s, r)
        logger.info(f"  Isotonic Calib: {len(model.X_thresholds_)} Stützstellen")
        return {'model': model, 'method': 'isotonic'}

    else:
        raise ValueError(f"Unbekannte Methode: {method}")


def predict_expected_return(
    calib: dict,
    scores: np.ndarray,
) -> np.ndarray:
    """Wendet die Kalibrierung an: score → expected_11d_return."""
    model = calib['model']
    if calib['method'] == 'linear':
        return model.predict(np.asarray(scores).reshape(-1, 1))
    else:
        return model.predict(np.asarray(scores))


# ── Qualitätsanalyse ─────────────────────────────────────────────────────────

def evaluate_calibration(
    df:       pd.DataFrame,
    calib:    dict,
    save_dir: Optional[str] = None,
) -> dict:
    """
    Evaluiert die Kalibrierung auf dem gesamten Datensatz.

    Returns:
        dict mit Korrelationen, Statistiken, Plot-Pfaden
    """
    from scipy import stats

    scores = df['score'].values
    true_ret = df['true_return_11d'].values
    expected_ret = predict_expected_return(calib, scores)

    # Korrelationen
    corr_score_true = float(np.corrcoef(scores, true_ret)[0, 1])
    corr_expected_true = float(np.corrcoef(expected_ret, true_ret)[0, 1])
    spearman_score = float(stats.spearmanr(scores, true_ret).statistic)
    spearman_expected = float(stats.spearmanr(expected_ret, true_ret).statistic)

    logger.info(f"  Korrelation(score, true_ret):          Pearson={corr_score_true:.4f}  Spearman={spearman_score:.4f}")
    logger.info(f"  Korrelation(expected_ret, true_ret):   Pearson={corr_expected_true:.4f}  Spearman={spearman_expected:.4f}")

    # Binning-Analyse: Score in 10 Quantile, avg true return pro Bin
    df_eval = df.copy()
    df_eval['expected_ret'] = expected_ret
    df_eval['score_decile'] = pd.qcut(df_eval['score'], 10, labels=False, duplicates='drop')
    bin_stats = df_eval.groupby('score_decile').agg(
        avg_score=('score', 'mean'),
        avg_true_ret=('true_return_11d', 'mean'),
        avg_expected_ret=('expected_ret', 'mean'),
        count=('score', 'size'),
    ).round(6)

    logger.info("  Binning (Score-Dezile):")
    logger.info(f"  {'Dezil':>6s}  {'Avg Score':>10s}  {'Avg True Ret':>12s}  {'Avg Exp Ret':>12s}  {'N':>6s}")
    for idx, row in bin_stats.iterrows():
        logger.info(
            f"  {idx:>6d}  {row['avg_score']:>10.5f}  {row['avg_true_ret']*100:>+11.3f}%  "
            f"{row['avg_expected_ret']*100:>+11.3f}%  {int(row['count']):>6d}"
        )

    # Statistiken zum erwarteten Return
    exp_stats = {
        'mean':   float(np.mean(expected_ret)),
        'median': float(np.median(expected_ret)),
        'std':    float(np.std(expected_ret)),
        'min':    float(np.min(expected_ret)),
        'max':    float(np.max(expected_ret)),
        'pct_negative': float((expected_ret < 0).mean() * 100),
    }
    logger.info(f"  Expected Return Stats: mean={exp_stats['mean']*100:.3f}%  "
                f"std={exp_stats['std']*100:.3f}%  "
                f"pct_negative={exp_stats['pct_negative']:.1f}%")

    result = {
        'corr_score_true_pearson':      round(corr_score_true, 4),
        'corr_score_true_spearman':     round(spearman_score, 4),
        'corr_expected_true_pearson':   round(corr_expected_true, 4),
        'corr_expected_true_spearman':  round(spearman_expected, 4),
        'method':                       calib['method'],
        'n_samples':                    len(df),
        'expected_return_stats':        exp_stats,
        'bin_stats':                    bin_stats.to_dict('index'),
    }

    # Plots
    if save_dir:
        _plot_calibration_diagnostics(df_eval, calib, save_dir)

    return result


def _plot_calibration_diagnostics(
    df:       pd.DataFrame,
    calib:    dict,
    save_dir: str,
):
    """Erzeugt 3 Diagnose-Plots für die Kalibrierung."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        scores = df['score'].values
        true_ret = df['true_return_11d'].values * 100
        exp_ret = df['expected_ret'].values * 100

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        # 1. Score vs True Return (Streudiagramm mit Hex-Binning)
        ax = axes[0]
        ax.hexbin(scores, true_ret, gridsize=50, cmap='Blues', mincnt=1)
        sx = np.linspace(scores.min(), scores.max(), 100)
        sy = predict_expected_return(calib, sx) * 100
        ax.plot(sx, sy, 'r-', linewidth=2, label=f'Kalibrierung ({calib["method"]})')
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.set_xlabel('Model Score')
        ax.set_ylabel('True 11d Return (%)')
        ax.set_title('Score vs. True Return')
        ax.legend()

        # 2. Expected Return vs True Return
        ax = axes[1]
        ax.hexbin(exp_ret, true_ret, gridsize=50, cmap='Greens', mincnt=1)
        lims = [min(exp_ret.min(), true_ret.min()), max(exp_ret.max(), true_ret.max())]
        ax.plot(lims, lims, 'r--', linewidth=1, label='Perfekte Kalibrierung')
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5)
        ax.set_xlabel('Expected 11d Return (%)')
        ax.set_ylabel('True 11d Return (%)')
        ax.set_title('Expected vs. True Return')
        ax.legend()

        # 3. Binning-Plot: Avg True Return pro Score-Dezil
        ax = axes[2]
        bin_data = df.groupby('score_decile').agg(
            avg_true=('true_return_11d', 'mean'),
            avg_exp=('expected_ret', 'mean'),
        )
        x = range(len(bin_data))
        ax.bar(x, bin_data['avg_true'] * 100, alpha=0.6, label='Avg True Return', color='#1565C0')
        ax.plot(x, bin_data['avg_exp'] * 100, 'ro-', linewidth=2, label='Avg Expected Return')
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.set_xlabel('Score-Dezil (0=niedrigste)')
        ax.set_ylabel('Avg 11d Return (%)')
        ax.set_title('Return pro Score-Dezil')
        ax.legend()

        plt.tight_layout()
        path = save_dir / "calibration_diagnostics.png"
        plt.savefig(str(path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.success(f"  Kalibrierungs-Plot gespeichert: {path}")

    except Exception as e:
        logger.warning(f"  Kalibrierungs-Plot Fehler: {e}")
