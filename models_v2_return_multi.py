"""
models_v2_return_multi.py
──────────────────────────
LSTM mit Multi-Horizon Return-Prediction (v2).

Unterschiede zu v1_rank (CrossSectionalLSTM):
  - Output-Dimension = 4 statt 1 (pred_4d, pred_7d, pred_11d, pred_15d)
  - Loss = gewichteter Huber/MSE über 4 Horizonte + optionaler Pairwise-Ranking-Term
  - Gleiche LSTM + Attention + Embedding Architektur
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr

from config_v2_return_multi import V2Config


# ── Modell ────────────────────────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        scores  = self.attn(lstm_out).squeeze(-1)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        context = (lstm_out * weights).sum(dim=1)
        return context


class LSTMReturnMultiV2(nn.Module):
    """
    Multi-Horizon Return Prediction LSTM.

    Output: (batch, n_horizons) — ein erwarteter Return pro Horizont.
    """

    def __init__(
        self,
        n_features:  int,
        n_assets:    int,
        n_horizons:  int   = 4,
        embed_dim:   int   = 16,
        hidden_dim:  int   = 128,
        num_layers:  int   = 2,
        dropout:     float = 0.3,
        seq_len:     int   = 64,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_horizons = n_horizons
        self.hidden_dim = hidden_dim

        self.asset_embedding = nn.Embedding(n_assets, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size  = n_features + embed_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        self.attention = TemporalAttention(hidden_dim)
        self.norm      = nn.LayerNorm(hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(32, n_horizons),
        )
        self._init_weights()

    def forward(self, x: torch.Tensor, asset_id: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        emb = self.asset_embedding(asset_id)
        emb = emb.unsqueeze(1).expand(-1, seq_len, -1)
        x_in = torch.cat([x, emb], dim=-1)
        lstm_out, _ = self.lstm(x_in)
        context = self.attention(lstm_out)
        context = self.norm(context)
        return self.head(context)  # (batch, n_horizons)

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                if "lstm" in name:
                    n = param.size(0)
                    param.data[n // 4 : n // 2].fill_(1)
            elif "weight" in name and param.dim() == 2:
                nn.init.xavier_uniform_(param.data)


# ── Loss-Funktionen ──────────────────────────────────────────────────────────

class MultiHorizonRegressionLoss(nn.Module):
    """Gewichteter Regression-Loss über mehrere Horizonte."""

    def __init__(self, cfg: V2Config):
        super().__init__()
        self.weights = torch.tensor(cfg.horizon_weights, dtype=torch.float32)
        if cfg.reg_loss_type == "huber":
            self.loss_fn = nn.HuberLoss(delta=cfg.huber_delta, reduction='mean')
        else:
            self.loss_fn = nn.MSELoss(reduction='mean')

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        preds:   (batch, n_horizons)
        targets: (batch, n_horizons)
        """
        w = self.weights.to(preds.device)
        losses = torch.stack([
            self.loss_fn(preds[:, i], targets[:, i])
            for i in range(preds.shape[1])
        ])
        return (losses * w).sum()


class PairwiseRankLoss(nn.Module):
    """Pairwise Ranking-Loss für einen bestimmten Horizont-Index."""

    def __init__(self, horizon_idx: int, margin: float = 0.001):
        super().__init__()
        self.horizon_idx = horizon_idx
        self.margin = margin

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = preds[:, self.horizon_idx]
        t = targets[:, self.horizon_idx]
        n = len(p)
        if n < 2:
            return torch.tensor(0.0, device=preds.device)

        pred_diff   = p.unsqueeze(0) - p.unsqueeze(1)
        target_diff = t.unsqueeze(0) - t.unsqueeze(1)
        mask = target_diff > 0.001

        if not mask.any():
            return torch.tensor(0.0, device=preds.device)

        losses = torch.clamp(self.margin - pred_diff[mask], min=0)
        return losses.mean()


class CombinedMultiHorizonLoss(nn.Module):
    """
    Total Loss = Regression (multi-horizon) + lambda_rank * Ranking (single horizon).
    """

    def __init__(self, cfg: V2Config):
        super().__init__()
        self.reg_loss  = MultiHorizonRegressionLoss(cfg)
        self.rank_loss = PairwiseRankLoss(cfg.rank_horizon_idx, cfg.rank_margin)
        self.lambda_rank = cfg.lambda_rank

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        l_reg  = self.reg_loss(preds, targets)
        l_rank = self.rank_loss(preds, targets)
        return l_reg + self.lambda_rank * l_rank

    def components(self, preds: torch.Tensor, targets: torch.Tensor) -> dict:
        """Gibt die einzelnen Loss-Komponenten zurück (für Logging)."""
        with torch.no_grad():
            w = self.reg_loss.weights.to(preds.device)
            per_h = []
            for i in range(preds.shape[1]):
                per_h.append(self.reg_loss.loss_fn(preds[:, i], targets[:, i]).item())
            return {
                'reg_total': sum(per_h[i] * w[i].item() for i in range(len(per_h))),
                'reg_4':     per_h[0] if len(per_h) > 0 else 0,
                'reg_7':     per_h[1] if len(per_h) > 1 else 0,
                'reg_11':    per_h[2] if len(per_h) > 2 else 0,
                'reg_15':    per_h[3] if len(per_h) > 3 else 0,
                'rank':      self.rank_loss(preds, targets).item(),
            }


# ── Metriken ──────────────────────────────────────────────────────────────────

def rank_ic_multi(
    preds:   torch.Tensor,  # (N, n_horizons)
    targets: torch.Tensor,  # (N, n_horizons)
) -> list[float]:
    """Rank-IC (Spearman) pro Horizont."""
    p = preds.cpu().numpy()
    t = targets.cpu().numpy()
    ics = []
    for h in range(p.shape[1]):
        if len(p) < 5:
            ics.append(0.0)
            continue
        corr, _ = spearmanr(p[:, h], t[:, h])
        ics.append(float(corr) if not (corr != corr) else 0.0)
    return ics
