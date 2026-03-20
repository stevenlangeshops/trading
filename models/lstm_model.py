"""
models/lstm_model.py
─────────────────────
PyTorch LSTM-Modell für Zeitreihen-Trading-Signale.

Unterstützt:
  • Klassifikation  (binary cross-entropy)
  • Regression      (MSE / Huber)

Architektur:
  Input  → LSTM (mehrere Layer, optional bidirektional)
         → Dropout
         → Attention (optional)
         → Fully Connected Head
  Output → Sigmoid (cls) | Linear (reg)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Attention ─────────────────────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    """
    Einfache Scaled-Dot-Product Attention über die Zeitachse.
    Gibt einen gewichteten Durchschnitt der LSTM-Ausgaben zurück.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        # lstm_out: (batch, seq_len, hidden_dim)
        scores = self.attn(lstm_out).squeeze(-1)           # (batch, seq_len)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)   # (batch, seq_len, 1)
        context = (lstm_out * weights).sum(dim=1)          # (batch, hidden_dim)
        return context


# ── Haupt-Modell ──────────────────────────────────────────────────────────────

class TradingLSTM(nn.Module):
    """
    LSTM-basiertes Modell für Trading-Signale.

    Args:
        n_features    : Anzahl der Eingangs-Features
        seq_len       : Sequenzlänge (wird nur für Dokumentation verwendet)
        hidden_dim    : Anzahl der LSTM-Neuronen pro Layer
        num_layers    : Anzahl der LSTM-Layer (Stacking)
        dropout       : Dropout-Rate (zwischen LSTM-Layern)
        bidirectional : Bidirektionales LSTM
        use_attention : Temporal Attention statt last-hidden-state
        mode          : "cls" = Klassifikation, "reg" = Regression
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int = 48,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
        use_attention: bool = True,
        mode: str = "cls",
    ) -> None:
        super().__init__()

        self.mode          = mode
        self.hidden_dim    = hidden_dim
        self.num_layers    = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.directions    = 2 if bidirectional else 1

        # LSTM
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_dim * self.directions

        # Attention
        if use_attention:
            self.attention = TemporalAttention(lstm_out_dim)

        # LayerNorm statt BatchNorm: funktioniert bei beliebiger Batch-Größe
        # (BatchNorm kollabiert bei einzelnen Assets mit abweichender Verteilung)
        self.bn = nn.LayerNorm(lstm_out_dim)
        self.head = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            out: (batch,) — Sigmoid für cls, linear für reg
        """
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*dir)

        if self.use_attention:
            context = self.attention(lstm_out)         # (batch, hidden*dir)
        else:
            context = lstm_out[:, -1, :]               # letzter Zeitschritt

        context = self.bn(context)
        out = self.head(context).squeeze(-1)           # (batch,)

        if self.mode == "cls":
            return torch.sigmoid(out)
        return out                                      # Regression: raw output

    def init_weights(self) -> None:
        """Orthogonale Initialisierung für LSTM-Gewichte."""
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                # Forget-Gate-Bias auf 1 setzen (besseres Gedächtnis)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1)
