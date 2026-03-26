"""
train_v2_return_multi.py
─────────────────────────
Walk-Forward Training für das Multi-Horizon Return-Modell v2.

Ablauf:
  1. build_panel() → Features (identisch zu v1)
  2. build_multi_horizon_targets() → Multi-Target DataFrame (4d, 7d, 11d, 15d)
  3. Walk-Forward Training mit LSTMReturnMultiV2 + CombinedMultiHorizonLoss
  4. Pro Fold: Checkpoint, OOS-Metriken (MSE/MAE/RankIC pro Horizont)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import Dataset, DataLoader

from config_v2_return_multi import V2Config
from models_v2_return_multi import (
    LSTMReturnMultiV2,
    CombinedMultiHorizonLoss,
    rank_ic_multi,
)
from models.dataset import WalkForwardFold, create_walk_forward_folds


# ── Multi-Horizon Targets ─────────────────────────────────────────────────────

def build_multi_horizon_targets(
    raw_dir:   Path,
    horizons:  list[int],
    asset_list: Optional[list[str]] = None,
    timeframe: str = "1d",
    min_rows:  int = 300,
) -> pd.DataFrame:
    """
    Berechnet Forward-Returns für mehrere Horizonte.

    Returns:
        DataFrame mit MultiIndex (date, asset), Spalten = [f"ret_{h}d" for h in horizons]
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
        target_df = pd.DataFrame(index=df.index)
        for h in horizons:
            target_df[f"ret_{h}d"] = close.pct_change(h).shift(-h)

        valid = target_df.notna().all(axis=1)
        target_df = target_df[valid]

        if len(target_df) < 200:
            continue

        all_targets[ticker] = target_df

    panel = pd.concat(all_targets, names=["asset", "date"])
    panel = panel.swaplevel().sort_index()

    logger.info(f"Multi-Horizon Targets: {len(panel)} Zeilen, "
                f"{panel.index.get_level_values('asset').nunique()} Assets, "
                f"Horizonte={horizons}")
    return panel


# ── Multi-Horizon Dataset ─────────────────────────────────────────────────────

class MultiHorizonDataset(Dataset):
    """
    Pro Sample:
      X        : (seq_len, n_features)
      y        : (n_horizons,)  — Forward Returns für 4/7/11/15 Tage
      asset_id : int
    """

    def __init__(
        self,
        features:     pd.DataFrame,
        targets_multi: pd.DataFrame,   # MultiIndex (date, asset), cols=ret_4d,ret_7d,...
        asset_map:    dict[str, int],
        seq_len:      int = 64,
        start_date:   Optional[pd.Timestamp] = None,
        end_date:     Optional[pd.Timestamp] = None,
    ):
        self.seq_len = seq_len
        self.n_features = len(features.columns)
        self.n_horizons = len(targets_multi.columns)
        self.samples: list[tuple] = []

        dates = features.index.get_level_values("date")
        mask = pd.Series(True, index=features.index)
        if start_date is not None:
            mask &= dates >= start_date
        if end_date is not None:
            mask &= dates <= end_date

        features = features[mask.values]

        common_idx = features.index.intersection(targets_multi.index)
        features      = features.loc[common_idx]
        targets_multi = targets_multi.loc[common_idx]

        assets = features.index.get_level_values("asset").unique()
        max_valid_id = max(asset_map.values()) if asset_map else 0

        for asset in assets:
            asset_id = asset_map.get(asset, 0)
            if asset_id > max_valid_id:
                asset_id = 0

            try:
                af = features.xs(asset, level="asset").sort_index()
                at = targets_multi.xs(asset, level="asset").sort_index()
            except KeyError:
                continue

            common = af.index.intersection(at.index)
            af = af.loc[common]
            at = at.loc[common]

            feat_arr   = af.values.astype(np.float32)
            target_arr = at.values.astype(np.float32)  # (T, n_horizons)

            for i in range(seq_len - 1, len(feat_arr)):
                x = feat_arr[i - seq_len + 1 : i + 1]
                y = target_arr[i]

                if np.isnan(x).any() or np.isnan(y).any() or np.isinf(y).any():
                    continue
                self.samples.append((x, y, asset_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, aid = self.samples[idx]
        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.tensor(aid, dtype=torch.long),
        )


def make_multi_dataloaders(
    features:      pd.DataFrame,
    targets_multi: pd.DataFrame,
    fold:          WalkForwardFold,
    asset_map:     dict[str, int],
    seq_len:       int = 64,
    batch_size:    int = 512,
) -> tuple[DataLoader, DataLoader]:
    train_ds = MultiHorizonDataset(
        features, targets_multi, asset_map, seq_len,
        start_date=fold.train_start, end_date=fold.train_end,
    )
    val_ds = MultiHorizonDataset(
        features, targets_multi, asset_map, seq_len,
        start_date=fold.val_start, end_date=fold.val_end,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


# ── Training Loop ─────────────────────────────────────────────────────────────

def train_epoch_v2(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    for X, y, aid in loader:
        X, y, aid = X.to(device), y.to(device), aid.to(device)
        optimizer.zero_grad()
        preds = model(X, aid)
        loss = criterion(preds, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * len(X)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch_v2(model, loader, criterion, device, cfg: V2Config):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for X, y, aid in loader:
        X, y, aid = X.to(device), y.to(device), aid.to(device)
        preds = model(X, aid)
        total_loss += criterion(preds, y).item() * len(X)
        all_preds.append(preds)
        all_targets.append(y)

    preds_all   = torch.cat(all_preds)
    targets_all = torch.cat(all_targets)

    loss = total_loss / len(loader.dataset)
    ics  = rank_ic_multi(preds_all, targets_all)

    components = criterion.components(preds_all, targets_all)

    # MAE pro Horizont
    maes = []
    for h in range(preds_all.shape[1]):
        mae = (preds_all[:, h] - targets_all[:, h]).abs().mean().item()
        maes.append(mae)

    return loss, ics, components, maes


# ── Walk-Forward Training ─────────────────────────────────────────────────────

def train_walk_forward_v2(
    features:      pd.DataFrame,
    targets_multi: pd.DataFrame,
    asset_map:     dict[str, int],
    cfg:           V2Config = V2Config(),
) -> dict:
    """Walk-Forward Training für v2_return_multi."""

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    n_features = len(features.columns)
    n_assets   = max(asset_map.values()) + 1
    n_horizons = len(cfg.horizons)

    logger.info(f"[v2] Walk-Forward Training: Device={device}")
    logger.info(f"[v2]   Assets={n_assets}  Features={n_features}  "
                f"Horizonte={cfg.horizons}  seq_len={cfg.seq_len}")

    all_dates = features.index.get_level_values("date").unique()
    folds = create_walk_forward_folds(
        all_dates, cfg.train_years, cfg.val_months, cfg.step_months,
    )
    logger.info(f"[v2]   {len(folds)} Walk-Forward Folds")
    for f in folds:
        logger.info(f"[v2]   {f}")

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fold_results = []

    for fold in folds:
        logger.info("─" * 60)
        logger.info(f"[v2] FOLD {fold.fold_id}")

        train_loader, val_loader = make_multi_dataloaders(
            features, targets_multi, fold, asset_map, cfg.seq_len, cfg.batch_size,
        )

        if len(train_loader.dataset) < 100:
            logger.warning(f"[v2] Fold {fold.fold_id}: Zu wenig Daten — übersprungen")
            continue

        logger.info(f"[v2]   Train: {len(train_loader.dataset):,}  "
                    f"Val: {len(val_loader.dataset):,}")

        model = LSTMReturnMultiV2(
            n_features=n_features, n_assets=n_assets, n_horizons=n_horizons,
            embed_dim=cfg.embed_dim, hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers, dropout=cfg.dropout, seq_len=cfg.seq_len,
        ).to(device)

        criterion = CombinedMultiHorizonLoss(cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr / 100)

        best_val_loss  = float("inf")
        patience_count = 0
        ckpt_path = cfg.checkpoint_dir / f"fold_{fold.fold_id}_best.pt"

        h_labels = [f"{h}d" for h in cfg.horizons]
        logger.info(f"[v2]   {'Ep':>3}  {'TrLoss':>8}  {'VaLoss':>8}  "
                    + "  ".join(f"IC_{l:>3}" for l in h_labels)
                    + f"  {'LR':>10}")

        for epoch in range(1, cfg.epochs + 1):
            train_loss = train_epoch_v2(model, train_loader, optimizer, criterion, device, cfg.grad_clip)
            val_loss, val_ics, val_comp, val_maes = eval_epoch_v2(model, val_loader, criterion, device, cfg)
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            ic_str = "  ".join(f"{ic:6.4f}" for ic in val_ics)
            logger.info(f"[v2]   {epoch:3d}  {train_loss:8.5f}  {val_loss:8.5f}  "
                        f"{ic_str}  {current_lr:10.7f}")

            if val_loss < best_val_loss:
                best_val_loss  = val_loss
                patience_count = 0
                torch.save({
                    "fold":        fold.fold_id,
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "val_loss":    val_loss,
                    "val_ics":     val_ics,
                    "val_maes":    val_maes,
                    "val_components": val_comp,
                    "config": {
                        "n_features":  n_features,
                        "n_assets":    n_assets,
                        "n_horizons":  n_horizons,
                        "embed_dim":   cfg.embed_dim,
                        "hidden_dim":  cfg.hidden_dim,
                        "num_layers":  cfg.num_layers,
                        "seq_len":     cfg.seq_len,
                        "horizons":    cfg.horizons,
                    },
                }, ckpt_path)
            else:
                patience_count += 1
                if patience_count >= cfg.patience:
                    logger.warning(f"[v2] Early Stopping nach Epoche {epoch}")
                    break

        fold_results.append({
            "fold_id":        fold.fold_id,
            "train_start":    str(fold.train_start.date()),
            "val_start":      str(fold.val_start.date()),
            "val_end":        str(fold.val_end.date()),
            "best_val_loss":  best_val_loss,
            "best_val_ics":   val_ics,
            "best_val_maes":  val_maes,
            "best_val_comp":  val_comp,
            "ckpt_path":      str(ckpt_path),
        })

        ic_rank = val_ics[cfg.rank_horizon_idx]
        logger.success(f"[v2] Fold {fold.fold_id}: Loss={best_val_loss:.5f}  "
                       f"IC_{cfg.rank_horizon}d={ic_rank:.4f}")

    # Zusammenfassung
    if fold_results:
        mean_ics = [
            np.mean([r["best_val_ics"][h] for r in fold_results])
            for h in range(n_horizons)
        ]
        mean_loss = np.mean([r["best_val_loss"] for r in fold_results])
        logger.success("═" * 60)
        logger.success(f"[v2] Walk-Forward abgeschlossen: {len(fold_results)} Folds")
        logger.success(f"[v2]   Ø Val-Loss: {mean_loss:.5f}")
        for i, h in enumerate(cfg.horizons):
            logger.success(f"[v2]   Ø Rank IC {h}d: {mean_ics[i]:.4f}")
        logger.success("═" * 60)

    return {
        "fold_results": fold_results,
        "mean_ics":     mean_ics if fold_results else [0.0] * n_horizons,
        "mean_loss":    float(mean_loss) if fold_results else 0.0,
        "horizons":     cfg.horizons,
    }
