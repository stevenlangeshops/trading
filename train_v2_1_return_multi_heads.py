"""
train_v2_1_return_multi_heads.py
─────────────────────────────────
Walk-Forward Training für v2.1: Shared Trunk + 3 Heads.

Targets: y7, y11, y15 — dabei y11 primär für Ranking, y7/y15 für Return-Regression.
Logging: pro Fold und Epoche alle Loss-Komponenten + Rank-IC (11d, 7d, 15d).
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

from config_v2_1_return_multi_heads import V21Config
from models_v2_1_return_multi_heads import (
    MultiHeadModelV2_1,
    MultiHeadLossV2_1,
    rank_ic,
)
from models.dataset import WalkForwardFold, create_walk_forward_folds


# ── Multi-Horizon Targets (7d, 11d, 15d) ─────────────────────────────────────

def build_multi_targets_v21(
    raw_dir:    Path,
    horizons:   list[int],
    asset_list: Optional[list[str]] = None,
    timeframe:  str = "1d",
    min_rows:   int = 300,
) -> pd.DataFrame:
    """Returns DataFrame MultiIndex (date, asset), cols = [ret_7d, ret_11d, ret_15d]."""
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
        tgt = pd.DataFrame(index=df.index)
        for h in horizons:
            tgt[f"ret_{h}d"] = close.pct_change(h).shift(-h)

        valid = tgt.notna().all(axis=1)
        tgt = tgt[valid]
        if len(tgt) < 200:
            continue
        all_targets[ticker] = tgt

    panel = pd.concat(all_targets, names=["asset", "date"])
    panel = panel.swaplevel().sort_index()
    logger.info(f"[v2.1] Targets: {len(panel)} Zeilen, "
                f"{panel.index.get_level_values('asset').nunique()} Assets, "
                f"Horizonte={horizons}")
    return panel


# ── Dataset ───────────────────────────────────────────────────────────────────

class MultiHeadDatasetV21(Dataset):
    """
    Pro Sample:
      X        : (seq_len, n_features)
      y7, y11, y15 : float
      asset_id : int
    """

    def __init__(
        self,
        features:      pd.DataFrame,
        targets_multi: pd.DataFrame,
        asset_map:     dict[str, int],
        horizons:      list[int],
        seq_len:       int = 64,
        start_date:    Optional[pd.Timestamp] = None,
        end_date:      Optional[pd.Timestamp] = None,
    ):
        self.seq_len    = seq_len
        self.n_horizons = len(horizons)
        self.col_names  = [f"ret_{h}d" for h in horizons]
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

        max_valid_id = max(asset_map.values()) if asset_map else 0
        for asset in features.index.get_level_values("asset").unique():
            asset_id = asset_map.get(asset, 0)
            if asset_id > max_valid_id:
                asset_id = 0
            try:
                af = features.xs(asset, level="asset").sort_index()
                at = targets_multi.xs(asset, level="asset").sort_index()
            except KeyError:
                continue

            common = af.index.intersection(at.index)
            af, at = af.loc[common], at.loc[common]
            feat_arr = af.values.astype(np.float32)
            tgt_arr  = at[self.col_names].values.astype(np.float32)

            for i in range(seq_len - 1, len(feat_arr)):
                x = feat_arr[i - seq_len + 1 : i + 1]
                y = tgt_arr[i]
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


def make_dataloaders_v21(
    features, targets_multi, fold, asset_map, horizons, seq_len=64, batch_size=512,
):
    train_ds = MultiHeadDatasetV21(
        features, targets_multi, asset_map, horizons, seq_len,
        start_date=fold.train_start, end_date=fold.train_end,
    )
    val_ds = MultiHeadDatasetV21(
        features, targets_multi, asset_map, horizons, seq_len,
        start_date=fold.val_start, end_date=fold.val_end,
    )
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                          pin_memory=torch.cuda.is_available())
    val_ld   = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                          pin_memory=torch.cuda.is_available())
    return train_ld, val_ld


# ── Training / Eval Loops ─────────────────────────────────────────────────────

def _unpack_targets(y: torch.Tensor):
    """y shape: (batch, 3) → y7, y11, y15 (matching col order [7d,11d,15d])."""
    return y[:, 0], y[:, 1], y[:, 2]


def train_epoch_v21(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    for X, y, aid in loader:
        X, y, aid = X.to(device), y.to(device), aid.to(device)
        y7, y11, y15 = _unpack_targets(y)
        optimizer.zero_grad()
        pred_7d, pred_15d, score_11d = model(X, aid)
        loss = criterion(pred_7d, y7, pred_15d, y15, score_11d, y11)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * len(X)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch_v21(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_p7, all_p15, all_s11 = [], [], []
    all_y7, all_y11, all_y15 = [], [], []

    for X, y, aid in loader:
        X, y, aid = X.to(device), y.to(device), aid.to(device)
        y7, y11, y15 = _unpack_targets(y)
        pred_7d, pred_15d, score_11d = model(X, aid)
        total_loss += criterion(pred_7d, y7, pred_15d, y15, score_11d, y11).item() * len(X)
        all_p7.append(pred_7d);  all_p15.append(pred_15d);  all_s11.append(score_11d)
        all_y7.append(y7);       all_y11.append(y11);       all_y15.append(y15)

    p7  = torch.cat(all_p7);  p15 = torch.cat(all_p15);  s11 = torch.cat(all_s11)
    t7  = torch.cat(all_y7);  t11 = torch.cat(all_y11);  t15 = torch.cat(all_y15)

    loss = total_loss / len(loader.dataset)
    ic_11d = rank_ic(s11, t11)
    ic_7d  = rank_ic(p7, t7)
    ic_15d = rank_ic(p15, t15)
    comps  = criterion.components(p7, t7, p15, t15, s11, t11)

    mae_7d  = (p7  - t7).abs().mean().item()
    mae_15d = (p15 - t15).abs().mean().item()
    mae_11d = (s11 - t11).abs().mean().item()

    return loss, ic_11d, ic_7d, ic_15d, comps, mae_7d, mae_15d, mae_11d


# ── Walk-Forward Training ─────────────────────────────────────────────────────

def train_walk_forward_v21(
    features:      pd.DataFrame,
    targets_multi: pd.DataFrame,
    asset_map:     dict[str, int],
    cfg:           V21Config = V21Config(),
) -> dict:
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    n_features = len(features.columns)
    n_assets   = max(asset_map.values()) + 1
    horizons   = cfg.all_horizons  # [7, 11, 15]

    logger.info(f"[v2.1] Walk-Forward Training: Device={device}")
    logger.info(f"[v2.1]   Assets={n_assets}  Features={n_features}  "
                f"Horizonte={horizons}  seq_len={cfg.seq_len}")
    logger.info(f"[v2.1]   Loss: w_7d={cfg.w_ret_7d}  w_15d={cfg.w_ret_15d}  "
                f"lambda_rank={cfg.lambda_rank}  w_rank_reg={cfg.w_rank_reg}")

    all_dates = features.index.get_level_values("date").unique()
    folds = create_walk_forward_folds(all_dates, cfg.train_years, cfg.val_months, cfg.step_months)
    logger.info(f"[v2.1]   {len(folds)} Walk-Forward Folds")
    for f in folds:
        logger.info(f"[v2.1]   {f}")

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fold_results = []

    for fold in folds:
        logger.info("─" * 60)
        logger.info(f"[v2.1] FOLD {fold.fold_id}")

        train_ld, val_ld = make_dataloaders_v21(
            features, targets_multi, fold, asset_map, horizons, cfg.seq_len, cfg.batch_size,
        )
        if len(train_ld.dataset) < 100:
            logger.warning(f"[v2.1] Fold {fold.fold_id}: Zu wenig Daten — uebersprungen")
            continue
        logger.info(f"[v2.1]   Train: {len(train_ld.dataset):,}  Val: {len(val_ld.dataset):,}")

        model = MultiHeadModelV2_1(
            n_features=n_features, n_assets=n_assets,
            embed_dim=cfg.embed_dim, hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers, dropout=cfg.dropout,
        ).to(device)

        criterion = MultiHeadLossV2_1(cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr / 100)

        best_val_loss  = float("inf")
        patience_count = 0
        ckpt_path = cfg.checkpoint_dir / f"fold_{fold.fold_id}_best.pt"

        logger.info(f"[v2.1]   {'Ep':>3}  {'TrLoss':>8}  {'VaLoss':>8}  "
                    f"{'IC_11d':>7}  {'IC_7d':>7}  {'IC_15d':>7}  {'LR':>10}")

        best_ics = [0.0, 0.0, 0.0]
        best_maes = [0.0, 0.0, 0.0]
        best_comps = {}

        for epoch in range(1, cfg.epochs + 1):
            train_loss = train_epoch_v21(model, train_ld, optimizer, criterion, device, cfg.grad_clip)
            val_loss, ic_11, ic_7, ic_15, comps, mae7, mae15, mae11 = \
                eval_epoch_v21(model, val_ld, criterion, device)
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]

            logger.info(f"[v2.1]   {epoch:3d}  {train_loss:8.5f}  {val_loss:8.5f}  "
                        f"{ic_11:7.4f}  {ic_7:7.4f}  {ic_15:7.4f}  {lr:10.7f}")

            if val_loss < best_val_loss:
                best_val_loss  = val_loss
                patience_count = 0
                best_ics  = [ic_7, ic_11, ic_15]
                best_maes = [mae7, mae11, mae15]
                best_comps = comps
                torch.save({
                    "fold":        fold.fold_id,
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "val_loss":    val_loss,
                    "val_ics":     {"ic_7d": ic_7, "ic_11d": ic_11, "ic_15d": ic_15},
                    "config": {
                        "n_features": n_features, "n_assets": n_assets,
                        "embed_dim": cfg.embed_dim, "hidden_dim": cfg.hidden_dim,
                        "num_layers": cfg.num_layers, "seq_len": cfg.seq_len,
                        "horizons": horizons,
                    },
                }, ckpt_path)
            else:
                patience_count += 1
                if patience_count >= cfg.patience:
                    logger.warning(f"[v2.1] Early Stopping nach Epoche {epoch}")
                    break

        fold_results.append({
            "fold_id":       fold.fold_id,
            "train_start":   str(fold.train_start.date()),
            "val_start":     str(fold.val_start.date()),
            "val_end":       str(fold.val_end.date()),
            "best_val_loss": best_val_loss,
            "best_val_ics":  {"ic_7d": best_ics[0], "ic_11d": best_ics[1], "ic_15d": best_ics[2]},
            "best_val_maes": {"mae_7d": best_maes[0], "mae_11d": best_maes[1], "mae_15d": best_maes[2]},
            "best_val_comps": best_comps,
            "ckpt_path":     str(ckpt_path),
        })
        logger.success(f"[v2.1] Fold {fold.fold_id}: Loss={best_val_loss:.5f}  "
                       f"IC_11d={best_ics[1]:.4f}  IC_7d={best_ics[0]:.4f}  IC_15d={best_ics[2]:.4f}")

    if fold_results:
        mean_ic_11 = np.mean([r["best_val_ics"]["ic_11d"] for r in fold_results])
        mean_ic_7  = np.mean([r["best_val_ics"]["ic_7d"]  for r in fold_results])
        mean_ic_15 = np.mean([r["best_val_ics"]["ic_15d"] for r in fold_results])
        mean_loss  = np.mean([r["best_val_loss"] for r in fold_results])
        logger.success("═" * 60)
        logger.success(f"[v2.1] Walk-Forward: {len(fold_results)} Folds")
        logger.success(f"[v2.1]   Ø Val-Loss : {mean_loss:.5f}")
        logger.success(f"[v2.1]   Ø Rank IC 11d: {mean_ic_11:.4f}  (v1 Ref: ~0.035)")
        logger.success(f"[v2.1]   Ø Rank IC  7d: {mean_ic_7:.4f}")
        logger.success(f"[v2.1]   Ø Rank IC 15d: {mean_ic_15:.4f}")
        logger.success("═" * 60)
    else:
        mean_ic_11 = mean_ic_7 = mean_ic_15 = mean_loss = 0.0

    return {
        "fold_results": fold_results,
        "mean_ic_11d":  float(mean_ic_11),
        "mean_ic_7d":   float(mean_ic_7),
        "mean_ic_15d":  float(mean_ic_15),
        "mean_loss":    float(mean_loss),
        "horizons":     horizons,
    }
