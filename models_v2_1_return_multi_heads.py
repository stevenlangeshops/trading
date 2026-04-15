"""
models_v2_1_return_multi_heads.py
──────────────────────────────────
Shared-Trunk LSTM mit separaten Heads für v2.1:

  Trunk: LSTM + Attention + LayerNorm (identisch zu v1 CrossSectionalLSTM)
    │
    ├── Head Return 7d:  FC → pred_7d   (trainiert mit Huber, kalibrierter Return)
    ├── Head Return 15d: FC → pred_15d  (trainiert mit Huber, kalibrierter Return)
    └── Head Ranking 11d: FC → score_11d (trainiert mit starkem RankLoss wie v1)

Warum separate Heads statt einem 3-Output-Head (wie v2)?
  v2 zeigte: ein gemeinsamer Head für 4 Horizonte halbierte den Rank-IC.
  v2.1 gibt dem Ranking-Head einen eigenen FC-Stack mit starkem RankLoss (λ=0.5),
  damit er sich auf reine Ordnungsoptimierung konzentrieren kann.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr

from config_v2_1_return_multi_heads import V21Config


# ── Trunk ─────────────────────────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        scores  = self.attn(lstm_out).squeeze(-1)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        return (lstm_out * weights).sum(dim=1)


class LSTMTrunkV2_1(nn.Module):
    """Shared LSTM backbone — identische Architektur wie v1 CrossSectionalLSTM."""

    def __init__(
        self,
        n_features: int,
        n_assets:   int,
        embed_dim:  int   = 16,
        hidden_dim: int   = 128,
        num_layers: int   = 2,
        dropout:    float = 0.3,
    ):
        super().__init__()
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

    def forward(self, x: torch.Tensor, asset_id: torch.Tensor) -> torch.Tensor:
        """Returns: shared representation (batch, hidden_dim)"""
        batch_size, seq_len, _ = x.shape
        emb = self.asset_embedding(asset_id).unsqueeze(1).expand(-1, seq_len, -1)
        x_in = torch.cat([x, emb], dim=-1)
        lstm_out, _ = self.lstm(x_in)
        context = self.attention(lstm_out)
        return self.norm(context)


# ── Separate Heads ────────────────────────────────────────────────────────────

def _make_fc_head(hidden_dim: int, dropout: float) -> nn.Sequential:
    """Standard FC-Head: 128→64→32→1 (gleiche Tiefe wie v1)."""
    return nn.Sequential(
        nn.Linear(hidden_dim, 64),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(64, 32),
        nn.GELU(),
        nn.Dropout(dropout / 2),
        nn.Linear(32, 1),
    )


class MultiHeadModelV2_1(nn.Module):
    """
    Shared Trunk + 3 separate Heads.

    forward() returns: (pred_7d, pred_15d, score_11d)  — each (batch,)
    """

    def __init__(
        self,
        n_features: int,
        n_assets:   int,
        embed_dim:  int   = 16,
        hidden_dim: int   = 128,
        num_layers: int   = 2,
        dropout:    float = 0.3,
    ):
        super().__init__()
        self.trunk     = LSTMTrunkV2_1(n_features, n_assets, embed_dim, hidden_dim, num_layers, dropout)
        self.head_7d   = _make_fc_head(hidden_dim, dropout)
        self.head_15d  = _make_fc_head(hidden_dim, dropout)
        self.head_rank = _make_fc_head(hidden_dim, dropout)
        self._init_weights()

    def forward(
        self, x: torch.Tensor, asset_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(x, asset_id)
        pred_7d   = self.head_7d(h).squeeze(-1)
        pred_15d  = self.head_15d(h).squeeze(-1)
        score_11d = self.head_rank(h).squeeze(-1)
        return pred_7d, pred_15d, score_11d

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


# ── Loss ──────────────────────────────────────────────────────────────────────

class PairwiseRankLoss(nn.Module):
    def __init__(self, margin: float = 0.001):
        super().__init__()
        self.margin = margin

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n = len(preds)
        if n < 2:
            return torch.tensor(0.0, device=preds.device)
        pred_diff   = preds.unsqueeze(0)   - preds.unsqueeze(1)
        target_diff = targets.unsqueeze(0) - targets.unsqueeze(1)
        mask = target_diff > 0.001
        if not mask.any():
            return torch.tensor(0.0, device=preds.device)
        return torch.clamp(self.margin - pred_diff[mask], min=0).mean()


class MultiHeadLossV2_1(nn.Module):
    """
    Total = w_7d * Huber(pred_7d, y7) + w_15d * Huber(pred_15d, y15)
          + lambda_rank * PairwiseRankLoss(score_11d, y11)
          + w_rank_reg * Huber(score_11d, y11)
    """

    def __init__(self, cfg: V21Config):
        super().__init__()
        self.w_7d       = cfg.w_ret_7d
        self.w_15d      = cfg.w_ret_15d
        self.lambda_rank = cfg.lambda_rank
        self.w_rank_reg  = cfg.w_rank_reg
        self.rank_loss   = PairwiseRankLoss(cfg.rank_margin)

        if cfg.reg_loss_type == "huber":
            self.reg_fn = nn.HuberLoss(delta=cfg.huber_delta, reduction='mean')
        else:
            self.reg_fn = nn.MSELoss(reduction='mean')

    def forward(
        self,
        pred_7d:   torch.Tensor, y7:  torch.Tensor,
        pred_15d:  torch.Tensor, y15: torch.Tensor,
        score_11d: torch.Tensor, y11: torch.Tensor,
    ) -> torch.Tensor:
        l_reg_7  = self.reg_fn(pred_7d,  y7)
        l_reg_15 = self.reg_fn(pred_15d, y15)
        l_rank   = self.rank_loss(score_11d, y11)
        l_reg_11 = self.reg_fn(score_11d, y11)

        return (self.w_7d * l_reg_7
                + self.w_15d * l_reg_15
                + self.lambda_rank * l_rank
                + self.w_rank_reg * l_reg_11)

    @torch.no_grad()
    def components(
        self,
        pred_7d, y7, pred_15d, y15, score_11d, y11,
    ) -> dict:
        return {
            'reg_7d':   self.reg_fn(pred_7d, y7).item(),
            'reg_15d':  self.reg_fn(pred_15d, y15).item(),
            'rank_11d': self.rank_loss(score_11d, y11).item(),
            'reg_11d':  self.reg_fn(score_11d, y11).item(),
        }


# ── Metriken ──────────────────────────────────────────────────────────────────

def rank_ic(preds: torch.Tensor, targets: torch.Tensor) -> float:
    p = preds.cpu().numpy()
    t = targets.cpu().numpy()
    if len(p) < 5:
        return 0.0
    corr, _ = spearmanr(p, t)
    return float(corr) if not (corr != corr) else 0.0
