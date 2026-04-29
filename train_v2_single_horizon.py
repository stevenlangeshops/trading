"""
train_v2_single_horizon.py
─────────────────────────────
Walk-Forward Training fuer einen einzelnen Horizont (4/7/11/15d).

Strukturell identisch zu v1 models/trainer.py,
nur parametrisiert mit dem Horizont fuer Target-Berechnung und Checkpoint-Pfad.

Nutzung:
  from train_v2_single_horizon import train_single_horizon
  result = train_single_horizon(features, targets_h, asset_map, cfg)

  Oder:
  result = train_all_horizons(features, asset_map, raw_dir)
  → trainiert 4d, 7d, 11d, 15d sequentiell und gibt alle Ergebnisse zurueck.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

# PyTorch 2.10+ / Kaggle: AdamW kann torch._dynamo → sympy triggern; bei kaputtem sympy
# (AttributeError: module 'sympy' has no attribute 'core') vor import torch setzen.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

from config_v2_single_horizon import SingleHorizonConfig, HORIZONS, get_config
from models_v2_single_horizon import (
    SingleHorizonRankModel,
    CombinedRankLoss,
    rank_ic,
)
from models.dataset import (
    WalkForwardFold,
    create_walk_forward_folds,
    CrossSectionalDataset,
)


# ── Determinismus ─────────────────────────────────────────────────────────────

def seed_everything(seed: int = 42) -> None:
    """Setzt alle relevanten Random-Seeds für vollständig reproduzierbares Training.

    Deckt Python, NumPy, PyTorch (CPU + GPU) sowie cuDNN ab.  Muss **vor**
    jeder Modell-Initialisierung und DataLoader-Erstellung aufgerufen werden.

    Hinweis: ``cudnn.deterministic = True`` deaktiviert nicht-deterministische
    cuDNN-Kernel und kann das Training auf GPU leicht verlangsamen (~5–15 %).
    Für Produktionsläufe ist dieser Trade-off akzeptabel.

    Args:
        seed: Integer-Seed.  Default 42.  Pro Fold wird ``seed + fold_id``
              übergeben, damit jeder Fold unterschiedlich, aber reproduzierbar
              initialisiert wird.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)          # wirkt auch bei Single-GPU
    torch.backends.cudnn.deterministic = True  # keine nicht-det. cuDNN-Kernel
    torch.backends.cudnn.benchmark     = False  # kein Auto-Tuning (würde Seed brechen)


# ── Target-Berechnung ────────────────────────────────────────────────────────

def build_single_horizon_targets(
    raw_dir:    Path,
    horizon:    int,
    asset_list: Optional[list[str]] = None,
    timeframe:  str = "1d",
    min_rows:   int = 300,
) -> pd.Series:
    """
    Berechnet Forward-Return-Targets fuer einen einzelnen Horizont.
    Returns: pd.Series MultiIndex (date, asset) → float
    Identisch zum Target-Format von v1 (features.engineer.build_panel).
    """
    raw_files = sorted(raw_dir.glob(f"*_{timeframe}.parquet"))
    all_targets = {}

    for fpath in raw_files:
        ticker = fpath.stem.replace(f"_{timeframe}", "")
        if asset_list and ticker not in asset_list:
            continue
        df = pd.read_parquet(fpath)
        df.index = pd.to_datetime(df.index)
        df.columns = [c.lower() for c in df.columns]
        if len(df) < min_rows:
            continue

        close = df["close"]
        fwd_ret = close.pct_change(horizon).shift(-horizon)
        valid = fwd_ret.notna()
        fwd_ret = fwd_ret[valid]
        if len(fwd_ret) < 200:
            continue
        all_targets[ticker] = fwd_ret

    panel = pd.concat(all_targets, names=["asset", "date"])
    panel = panel.swaplevel().sort_index()
    panel.name = f"ret_{horizon}d"
    logger.info(f"[{horizon}d] Targets: {len(panel)} Zeilen, "
                f"{panel.index.get_level_values('asset').nunique()} Assets")
    return panel


# ── DataLoader Factory ────────────────────────────────────────────────────────

def make_dataloaders_sh(
    features:   pd.DataFrame,
    targets:    pd.Series,
    fold:       WalkForwardFold,
    asset_map:  dict[str, int],
    seq_len:    int = 64,
    batch_size: int = 512,
    seed:       int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Wrapper um CrossSectionalDataset (v1) mit Index-Alignment.

    Args:
        features:   Feature-Panel (MultiIndex date × asset).
        targets:    Target-Series (MultiIndex date × asset).
        fold:       Walk-Forward-Fold mit Train-/Val-Grenzen.
        asset_map:  {ticker → modell_id}-Mapping.
        seq_len:    LSTM-Lookback-Fenster.
        batch_size: Mini-Batch-Größe für den Training-DataLoader.
        seed:       Seed für den DataLoader-Generator – stellt deterministisches
                    Batch-Shuffling sicher (wichtig für Reproduzierbarkeit).
    """
    common_idx   = features.index.intersection(targets.index)
    feat_aligned = features.loc[common_idx]
    tgt_aligned  = targets.loc[common_idx]

    train_ds = CrossSectionalDataset(
        feat_aligned, tgt_aligned, asset_map, seq_len,
        start_date=fold.train_start, end_date=fold.train_end,
    )
    val_ds = CrossSectionalDataset(
        feat_aligned, tgt_aligned, asset_map, seq_len,
        start_date=fold.val_start, end_date=fold.val_end,
    )

    # Expliziter Generator: deterministisches Shuffling unabhängig vom globalen
    # RNG-Zustand, der zwischen den Epochen weiterwandert.
    g = torch.Generator()
    g.manual_seed(seed)

    train_ld = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        pin_memory=torch.cuda.is_available(), generator=g,
    )
    val_ld = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    return train_ld, val_ld


# ── Training / Eval Loops ─────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    for X, y, aid in loader:
        X, y, aid = X.to(device), y.to(device), aid.to(device)
        optimizer.zero_grad()
        preds = model(X, aid)
        loss  = criterion(preds, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * len(X)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for X, y, aid in loader:
        X, y, aid = X.to(device), y.to(device), aid.to(device)
        preds = model(X, aid)
        total_loss += criterion(preds, y).item() * len(X)
        all_preds.append(preds)
        all_targets.append(y)
    p = torch.cat(all_preds)
    t = torch.cat(all_targets)
    loss = total_loss / len(loader.dataset)
    ic   = rank_ic(p, t)
    mae  = (p - t).abs().mean().item()
    return loss, ic, mae


# ── Walk-Forward fuer einen Horizont ──────────────────────────────────────────

def train_single_horizon(
    features:  pd.DataFrame,
    targets:   pd.Series,
    asset_map: dict[str, int],
    cfg:       SingleHorizonConfig,
) -> dict:
    """Walk-Forward Training für einen einzelnen Horizont.

    Für jeden Fold wird ``seed_everything(cfg.seed + fold_id)`` aufgerufen,
    sodass alle 12 Folds unterschiedliche, aber deterministisch reproduzierbare
    Gewichts-Initialisierungen erhalten.

    Args:
        features:  Feature-Panel (MultiIndex date × asset).
        targets:   Forward-Return-Targets (MultiIndex date × asset).
        asset_map: {ticker → modell_id}-Mapping.
        cfg:       Vollständige Trainings-Konfiguration inkl. ``cfg.seed``.

    Returns:
        Dict mit ``fold_results``, ``mean_ic``, ``mean_loss``, ``mean_mae``.
    """
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    n_features = len(features.columns)
    n_assets   = max(asset_map.values()) + 1
    tag        = cfg.tag

    logger.info(f"[{tag}] Walk-Forward Training: Device={device}")
    logger.info(f"[{tag}]   Assets={n_assets}  Features={n_features}  "
                f"Horizon={cfg.horizon}d  seq_len={cfg.seq_len}")
    logger.info(f"[{tag}]   Loss: MSE + {cfg.rank_weight} * RankLoss  Seed={cfg.seed}")

    all_dates = features.index.get_level_values("date").unique()
    folds = create_walk_forward_folds(all_dates, cfg.train_years, cfg.val_months, cfg.step_months)
    logger.info(f"[{tag}]   {len(folds)} Walk-Forward Folds")

    ckpt_dir = cfg.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    fold_results = []

    for fold in folds:
        logger.info("─" * 60)
        logger.info(f"[{tag}] FOLD {fold.fold_id}")

        # Deterministischer Fold-Seed: Basis + fold_id → jeder Fold einzigartig,
        # aber bei gleichem cfg.seed immer identisch reproduzierbar.
        fold_seed = cfg.seed + fold.fold_id
        seed_everything(fold_seed)
        logger.info(f"[{tag}]   Seed={fold_seed}  (basis={cfg.seed} + fold={fold.fold_id})")

        train_ld, val_ld = make_dataloaders_sh(
            features, targets, fold, asset_map, cfg.seq_len, cfg.batch_size,
            seed=fold_seed,
        )
        if len(train_ld.dataset) < 100:
            logger.warning(f"[{tag}] Fold {fold.fold_id}: Zu wenig Daten")
            continue
        logger.info(f"[{tag}]   Train: {len(train_ld.dataset):,}  Val: {len(val_ld.dataset):,}")

        model = SingleHorizonRankModel(
            n_features=n_features, n_assets=n_assets,
            embed_dim=cfg.embed_dim, hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers, dropout=cfg.dropout,
            seq_len=cfg.seq_len,
        ).to(device)

        criterion = CombinedRankLoss(rank_weight=cfg.rank_weight, margin=cfg.rank_margin)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr / 100)

        best_val_loss  = float("inf")
        best_ic        = 0.0
        best_mae       = 0.0
        patience_count = 0
        ckpt_path      = ckpt_dir / f"fold_{fold.fold_id}_best.pt"

        logger.info(f"[{tag}]   {'Ep':>3}  {'TrLoss':>8}  {'VaLoss':>8}  "
                    f"{'IC':>7}  {'MAE':>7}  {'LR':>10}")

        for epoch in range(1, cfg.epochs + 1):
            tr_loss = train_epoch(model, train_ld, optimizer, criterion, device, cfg.grad_clip)
            va_loss, va_ic, va_mae = eval_epoch(model, val_ld, criterion, device)
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]

            logger.info(f"[{tag}]   {epoch:3d}  {tr_loss:8.5f}  {va_loss:8.5f}  "
                        f"{va_ic:7.4f}  {va_mae:7.5f}  {lr:10.7f}")

            if va_loss < best_val_loss:
                best_val_loss  = va_loss
                best_ic        = va_ic
                best_mae       = va_mae
                patience_count = 0
                torch.save({
                    "fold":        fold.fold_id,
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "val_loss":    va_loss,
                    "val_ic":      va_ic,
                    "horizon":     cfg.horizon,
                    "seed":        fold_seed,   # für spätere Reproduzierbarkeit
                    "config": {
                        "n_features": n_features, "n_assets": n_assets,
                        "embed_dim": cfg.embed_dim, "hidden_dim": cfg.hidden_dim,
                        "num_layers": cfg.num_layers, "seq_len": cfg.seq_len,
                        "horizon": cfg.horizon,
                    },
                }, ckpt_path)
            else:
                patience_count += 1
                if patience_count >= cfg.patience:
                    logger.warning(f"[{tag}] Early Stopping nach Epoche {epoch}")
                    break

        fold_results.append({
            "fold_id":       fold.fold_id,
            "train_start":   str(fold.train_start.date()),
            "val_start":     str(fold.val_start.date()),
            "val_end":       str(fold.val_end.date()),
            "best_val_loss": best_val_loss,
            "best_val_ic":   best_ic,
            "best_val_mae":  best_mae,
            "ckpt_path":     str(ckpt_path),
        })
        logger.success(f"[{tag}] Fold {fold.fold_id}: Loss={best_val_loss:.5f}  "
                       f"IC={best_ic:.4f}  MAE={best_mae:.5f}")

    mean_ic   = np.mean([r["best_val_ic"]   for r in fold_results]) if fold_results else 0.0
    mean_loss = np.mean([r["best_val_loss"] for r in fold_results]) if fold_results else 0.0
    mean_mae  = np.mean([r["best_val_mae"]  for r in fold_results]) if fold_results else 0.0

    if fold_results:
        logger.success("═" * 60)
        logger.success(f"[{tag}] Walk-Forward: {len(fold_results)} Folds")
        logger.success(f"[{tag}]   Ø Val-Loss : {mean_loss:.5f}")
        logger.success(f"[{tag}]   Ø Rank IC  : {mean_ic:.4f}  (v1 Ref: ~0.035)")
        logger.success(f"[{tag}]   Ø MAE      : {mean_mae:.5f}")
        logger.success("═" * 60)

    return {
        "horizon":      cfg.horizon,
        "tag":          tag,
        "fold_results": fold_results,
        "mean_ic":      float(mean_ic),
        "mean_loss":    float(mean_loss),
        "mean_mae":     float(mean_mae),
    }


# ── Alle Horizonte sequentiell trainieren ─────────────────────────────────────

def train_all_horizons(
    features:  pd.DataFrame,
    asset_map: dict[str, int],
    raw_dir:   Path,
    horizons:  list[int] = None,
) -> dict[int, dict]:
    """
    Trainiert alle Horizonte (4/7/11/15d) sequentiell.
    Returns: {horizon: result_dict}
    """
    if horizons is None:
        horizons = HORIZONS

    all_results = {}
    for h in horizons:
        logger.info(f"\n{'#'*60}")
        logger.info(f"# HORIZONT {h}d")
        logger.info(f"{'#'*60}")

        cfg = get_config(h)
        targets_h = build_single_horizon_targets(
            raw_dir=raw_dir, horizon=h,
            asset_list=list(asset_map.keys()),
        )
        result = train_single_horizon(features, targets_h, asset_map, cfg)
        all_results[h] = result

    # Zusammenfassung
    logger.success("\n" + "═" * 70)
    logger.success("HORIZONT-VERGLEICH (Training)")
    logger.success("═" * 70)
    logger.success(f"{'Horizont':>10}  {'Ø IC':>8}  {'Ø Loss':>8}  {'Ø MAE':>8}  {'Folds':>6}")
    logger.success("─" * 70)
    for h in horizons:
        r = all_results[h]
        logger.success(f"{r['tag']:>10}  {r['mean_ic']:8.4f}  "
                       f"{r['mean_loss']:8.5f}  {r['mean_mae']:8.5f}  "
                       f"{len(r['fold_results']):6d}")
    logger.success("═" * 70)

    return all_results
