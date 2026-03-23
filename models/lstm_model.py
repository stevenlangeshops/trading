"""
models/lstm_model.py
─────────────────────
Cross-Sectional LSTM mit Asset-Embeddings für Multi-Asset Return Prediction.

Architektur:
  Asset-Embedding  →  ┐
                       ├→ LSTM → Attention → LayerNorm → FC → Return (float)
  Feature-Sequenz  →  ┘

Asset-Embeddings:
  Jedes Asset bekommt einen erlernten Embedding-Vektor (dim=16).
  Dieser wird an jede Zeitschritt-Eingabe konkateniert.
  → Das Modell lernt asset-spezifische Eigenschaften
    (z.B. "Tech-Aktien haben andere Momentum-Dynamik als Anleihen-ETFs")

Warum Regression statt Klassifikation:
  - Target = Forward Return (kontinuierlich)
  - Erlaubt Cross-Sectional Ranking: sortiere Assets nach predicted return
  - Kein arbiträrer Threshold mehr notwendig
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """Gewichteter Durchschnitt über Zeitachse."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        # lstm_out: (batch, seq_len, hidden_dim)
        scores  = self.attn(lstm_out).squeeze(-1)          # (batch, seq_len)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)   # (batch, seq_len, 1)
        context = (lstm_out * weights).sum(dim=1)          # (batch, hidden_dim)
        return context


class CrossSectionalLSTM(nn.Module):
    """
    LSTM für Cross-Sectional Return Prediction.

    Args:
        n_features    : Anzahl technischer Features (z.B. 18)
        n_assets      : Anzahl Assets für Embedding
        embed_dim     : Dimension des Asset-Embeddings (default 16)
        hidden_dim    : LSTM Hidden Size
        num_layers    : LSTM Tiefe
        dropout       : Dropout Rate
        seq_len       : Sequenzlänge (nur für Dokumentation)
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

        # Asset Embedding: jedes Asset → Vektor der Länge embed_dim
        self.asset_embedding = nn.Embedding(n_assets, embed_dim, padding_idx=0)

        # LSTM Input = Features + Embedding (an jeden Zeitschritt konkateniert)
        lstm_input_dim = n_features + embed_dim

        self.lstm = nn.LSTM(
            input_size  = lstm_input_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        self.attention = TemporalAttention(hidden_dim)

        # LayerNorm statt BatchNorm (funktioniert bei beliebiger Batch-Größe)
        self.norm = nn.LayerNorm(hidden_dim)

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

    def forward(
        self,
        x:        torch.Tensor,  # (batch, seq_len, n_features)
        asset_id: torch.Tensor,  # (batch,)
    ) -> torch.Tensor:           # (batch,)

        batch_size, seq_len, _ = x.shape

        # Asset-Embedding: (batch, embed_dim) → an jeden Zeitschritt anhängen
        emb = self.asset_embedding(asset_id)              # (batch, embed_dim)
        emb = emb.unsqueeze(1).expand(-1, seq_len, -1)   # (batch, seq_len, embed_dim)

        # Input = Features + Embedding
        x_in = torch.cat([x, emb], dim=-1)               # (batch, seq_len, n_features+embed_dim)

        # LSTM + Attention
        lstm_out, _ = self.lstm(x_in)                    # (batch, seq_len, hidden_dim)
        context     = self.attention(lstm_out)            # (batch, hidden_dim)
        context     = self.norm(context)

        # Regression Head → Forward Return Vorhersage
        out = self.head(context).squeeze(-1)              # (batch,)
        return out

    def _init_weights(self):
        """Orthogonale Initialisierung für LSTM, Xavier für Linear."""
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                # Forget-Gate-Bias auf 1 (besseres Langzeitgedächtnis)
                if "lstm" in name:
                    n = param.size(0)
                    param.data[n // 4 : n // 2].fill_(1)
            elif "weight" in name and param.dim() == 2:
                nn.init.xavier_uniform_(param.data)
