"""
models_v2_single_horizon.py
──────────────────────────────
Generisches Single-Horizon-Modell fuer den Horizont-Vergleich.

Architektur 1:1 identisch zu v1 CrossSectionalLSTM:
  LSTM + TemporalAttention + LayerNorm + FC-Head → Score (1D)
Loss 1:1 identisch zu v1 CombinedLoss:
  MSE + lambda_rank * PairwiseRankLoss

Einziger Unterschied: der Target-Horizont (4/7/11/15 Tage)
wird extern bestimmt, nicht hier.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr


# ── Modell ────────────────────────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        scores  = self.attn(lstm_out).squeeze(-1)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        return (lstm_out * weights).sum(dim=1)


class SingleHorizonRankModel(nn.Module):
    """
    Identisch zu CrossSectionalLSTM (v1_rank).
    Einziger Unterschied: kein hardcodierter Horizon — der wird
    ueber das Target definiert, nicht im Modell selbst.
    """

    def __init__(
        self,
        n_features: int,
        n_assets:   int,
        embed_dim:  int   = 16,
        hidden_dim: int   = 128,
        num_layers: int   = 2,
        dropout:    float = 0.3,
        seq_len:    int   = 64,
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.embed_dim  = embed_dim

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
            nn.Linear(32, 1),
        )
        self._init_weights()

    def forward(self, x: torch.Tensor, asset_id: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        emb  = self.asset_embedding(asset_id).unsqueeze(1).expand(-1, seq_len, -1)
        x_in = torch.cat([x, emb], dim=-1)
        lstm_out, _ = self.lstm(x_in)
        context = self.attention(lstm_out)
        context = self.norm(context)
        return self.head(context).squeeze(-1)

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


# ── Loss (identisch zu v1 CombinedLoss) ──────────────────────────────────────

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


class CombinedRankLoss(nn.Module):
    """MSE + lambda_rank * PairwiseRankLoss — identisch zu v1."""

    def __init__(self, rank_weight: float = 0.5, margin: float = 0.001):
        super().__init__()
        self.mse       = nn.MSELoss()
        self.rank_loss = PairwiseRankLoss(margin=margin)
        self.rank_weight = rank_weight

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.mse(preds, targets) + self.rank_weight * self.rank_loss(preds, targets)


# ── Metriken ──────────────────────────────────────────────────────────────────

def rank_ic(preds: torch.Tensor, targets: torch.Tensor) -> float:
    p = preds.cpu().numpy()
    t = targets.cpu().numpy()
    if len(p) < 5:
        return 0.0
    corr, _ = spearmanr(p, t)
    return float(corr) if not (corr != corr) else 0.0
