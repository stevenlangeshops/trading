"""
models/trainer.py
──────────────────
Walk-Forward Training Loop für Cross-Sectional LSTM.

Verlustfunktion:
  Wir nutzen eine Kombination aus:
  1. MSE Loss      — minimiert Vorhersagefehler auf absoluten Returns
  2. Rank Loss     — bestraft falsche Ranking-Reihenfolge zwischen Assets
                     (wichtiger als absolute Genauigkeit für Long-Short-Strategie)

  Total Loss = MSE + alpha * Rank Loss

Rank Loss (ListMLE / Pairwise):
  Für je zwei Assets i, j am gleichen Tag:
  Wenn return[i] > return[j], soll auch pred[i] > pred[j]
  → Optimiert direkt das was wir wollen: Ranking
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

from models.lstm_model import CrossSectionalLSTM
from models.dataset import WalkForwardFold, create_walk_forward_folds, make_dataloaders

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ── Verlustfunktionen ─────────────────────────────────────────────────────────

class RankLoss(nn.Module):
    """
    Pairwise Rank Loss: bestraft wenn ein Asset mit höherem true return
    einen niedrigeren predicted return bekommt als ein Asset mit niederem true return.

    Wichtig für Cross-Sectional Ranking — MSE allein optimiert nicht das Ranking.
    """

    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.margin = margin

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Alle Paare (i, j) wo target[i] > target[j]
        # Wir wollen pred[i] > pred[j] + margin
        n = len(preds)
        if n < 2:
            return torch.tensor(0.0, device=preds.device)

        # Effiziente Berechnung über Broadcasting
        pred_diff   = preds.unsqueeze(0)   - preds.unsqueeze(1)   # (n, n)
        target_diff = targets.unsqueeze(0) - targets.unsqueeze(1) # (n, n)

        # Nur Paare wo target[j] > target[i] (j hat höheren return)
        mask = target_diff > 0.001  # kleiner Threshold gegen Rauschen

        if not mask.any():
            return torch.tensor(0.0, device=preds.device)

        # Loss: max(0, margin - (pred[j] - pred[i])) für Paare wo target[j] > target[i]
        losses = torch.clamp(self.margin - pred_diff[mask], min=0)
        return losses.mean()


class CombinedLoss(nn.Module):
    """MSE + gewichteter Rank Loss."""

    def __init__(self, rank_weight: float = 0.5):
        super().__init__()
        self.mse       = nn.MSELoss()
        self.rank_loss = RankLoss(margin=0.001)
        self.rank_weight = rank_weight

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mse  = self.mse(preds, targets)
        rank = self.rank_loss(preds, targets)
        return mse + self.rank_weight * rank


# ── Metriken ──────────────────────────────────────────────────────────────────

def rank_ic(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Rank Information Coefficient (Spearman Korrelation zwischen Pred und Target).
    Wichtigste Metrik für Cross-Sectional Strategien.
    Wert nahe 0 = kein Skill, > 0.05 = praktisch nützlich.
    """
    from scipy.stats import spearmanr
    p = preds.cpu().numpy()
    t = targets.cpu().numpy()
    if len(p) < 5:
        return 0.0
    corr, _ = spearmanr(p, t)
    return float(corr) if not (corr != corr) else 0.0  # NaN check


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(
    model:      CrossSectionalLSTM,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    criterion:  nn.Module,
    device:     str,
    grad_clip:  float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0

    for X, y, asset_id in loader:
        X, y, asset_id = X.to(device), y.to(device), asset_id.to(device)
        optimizer.zero_grad()
        preds = model(X, asset_id)
        loss  = criterion(preds, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * len(X)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(
    model:     CrossSectionalLSTM,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    str,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for X, y, asset_id in loader:
        X, y, asset_id = X.to(device), y.to(device), asset_id.to(device)
        preds = model(X, asset_id)
        total_loss += criterion(preds, y).item() * len(X)
        all_preds.append(preds)
        all_targets.append(y)

    preds_all   = torch.cat(all_preds)
    targets_all = torch.cat(all_targets)
    ic          = rank_ic(preds_all, targets_all)

    return total_loss / len(loader.dataset), ic


# ── Walk-Forward Training ─────────────────────────────────────────────────────

def train_walk_forward(
    features,          # pd.DataFrame MultiIndex (date, asset) × features
    targets,           # pd.Series   MultiIndex (date, asset) → float
    asset_map,         # dict asset_name → int
    # Walk-Forward Parameter
    train_years:  float = 3.0,
    val_months:   float = 6.0,
    step_months:  float = 6.0,
    # Modell Parameter
    hidden_dim:   int   = 128,
    num_layers:   int   = 2,
    embed_dim:    int   = 16,
    dropout:      float = 0.3,
    seq_len:      int   = 64,
    # Training Parameter
    lr:           float = 5e-4,
    weight_decay: float = 1e-3,
    epochs:       int   = 50,
    patience:     int   = 7,
    batch_size:   int   = 512,
    rank_weight:  float = 0.5,
) -> dict:
    """
    Vollständiger Walk-Forward Training Loop.

    Gibt zurück:
        dict mit fold_results, best_model_path, mean_ic
    """
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    n_features = len(features.columns)
    # +1 weil IDs bei 1 starten (0 = padding), max ID = len(asset_map)
    # Embedding-Größe muss > max(asset_id) sein
    n_assets   = max(asset_map.values()) + 1

    logger.info(f"Walk-Forward Training: Device={device}")
    logger.info(f"  Assets={n_assets}  Features={n_features}  seq_len={seq_len}")

    # Folds erstellen
    all_dates = features.index.get_level_values("date").unique()
    folds     = create_walk_forward_folds(
        all_dates, train_years, val_months, step_months
    )
    logger.info(f"  {len(folds)} Walk-Forward Folds")
    for f in folds:
        logger.info(f"  {f}")

    fold_results = []

    for fold in folds:
        logger.info("─" * 60)
        logger.info(f"FOLD {fold.fold_id}")

        # DataLoader erstellen
        train_loader, val_loader = make_dataloaders(
            features, targets, fold, asset_map, seq_len, batch_size
        )

        if len(train_loader.dataset) < 100:
            logger.warning(f"  Fold {fold.fold_id}: Zu wenig Daten — übersprungen")
            continue

        logger.info(f"  Train: {len(train_loader.dataset):,} Samples  "
                    f"Val: {len(val_loader.dataset):,} Samples")

        # Modell (pro Fold neu initialisieren für saubere Evaluation)
        model = CrossSectionalLSTM(
            n_features = n_features,
            n_assets   = n_assets,
            embed_dim  = embed_dim,
            hidden_dim = hidden_dim,
            num_layers = num_layers,
            dropout    = dropout,
            seq_len    = seq_len,
        ).to(device)

        criterion = CombinedLoss(rank_weight=rank_weight)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr / 100
        )

        best_val_loss  = float("inf")
        patience_count = 0
        ckpt_path      = CHECKPOINT_DIR / f"fold_{fold.fold_id}_best.pt"

        logger.info(f"  {'Ep':>4}  {'TrainLoss':>10}  {'ValLoss':>10}  {'RankIC':>8}  {'LR':>10}")

        for epoch in range(1, epochs + 1):
            train_loss         = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_ic   = eval_epoch(model, val_loader, criterion, device)
            scheduler.step()
            current_lr         = optimizer.param_groups[0]["lr"]

            logger.info(f"  {epoch:4d}  {train_loss:10.5f}  {val_loss:10.5f}  "
                        f"{val_ic:8.4f}  {current_lr:10.7f}")

            if val_loss < best_val_loss:
                best_val_loss  = val_loss
                patience_count = 0
                torch.save({
                    "fold":        fold.fold_id,
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "val_loss":    val_loss,
                    "val_ic":      val_ic,
                    "config": {
                        "n_features": n_features,
                        "n_assets":   n_assets,
                        "embed_dim":  embed_dim,
                        "hidden_dim": hidden_dim,
                        "num_layers": num_layers,
                        "dropout":    dropout,
                        "seq_len":    seq_len,
                    },
                }, ckpt_path)
            else:
                patience_count += 1
                if patience_count >= patience:
                    logger.warning(f"  Early Stopping nach Epoche {epoch}")
                    break

        fold_results.append({
            "fold_id":      fold.fold_id,
            "train_start":  str(fold.train_start.date()),
            "val_start":    str(fold.val_start.date()),
            "val_end":      str(fold.val_end.date()),
            "best_val_loss": best_val_loss,
            "best_val_ic":  val_ic,
            "ckpt_path":    str(ckpt_path),
        })
        logger.success(f"  Fold {fold.fold_id}: Val-Loss={best_val_loss:.5f}  IC={val_ic:.4f}")

    # Zusammenfassung
    if fold_results:
        mean_ic   = sum(r["best_val_ic"]   for r in fold_results) / len(fold_results)
        mean_loss = sum(r["best_val_loss"] for r in fold_results) / len(fold_results)
        logger.success("═" * 60)
        logger.success(f"Walk-Forward abgeschlossen: {len(fold_results)} Folds")
        logger.success(f"  Ø Val-Loss : {mean_loss:.5f}")
        logger.success(f"  Ø Rank IC  : {mean_ic:.4f}  (> 0.05 = praktisch nützlich)")
        logger.success("═" * 60)

    return {
        "fold_results": fold_results,
        "mean_ic":      mean_ic if fold_results else 0.0,
    }
