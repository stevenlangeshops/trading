"""
config_v2_return_multi.py
──────────────────────────
Zentrale Konfiguration für das Multi-Horizon Return-Modell v2.

v1_rank  = bestehendes LSTM mit Score-Output (1D), trainiert mit MSE+RankLoss
v2_return_multi = neues LSTM mit 4 Return-Outputs (4d, 7d, 11d, 15d)
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class V2Config:
    # ── Horizonte ──────────────────────────────────────────────────────────
    horizons: list[int] = field(default_factory=lambda: [4, 7, 11, 15])

    # ── Loss-Gewichte pro Horizont (regression) ────────────────────────────
    # Indizes korrespondieren zu `horizons`: [w_4d, w_7d, w_11d, w_15d]
    horizon_weights: list[float] = field(default_factory=lambda: [0.25, 0.25, 0.25, 0.25])

    # ── Ranking-Loss ───────────────────────────────────────────────────────
    # Welcher Horizont-Index (0-basiert in `horizons`) für Pairwise-Ranking?
    rank_horizon_idx: int    = 2       # Index 2 → 11d (default)
    lambda_rank:      float  = 0.1     # Gewicht des Ranking-Loss im Total-Loss
    rank_margin:      float  = 0.001   # Margin für MarginRankingLoss

    # ── Regression-Loss-Typ ────────────────────────────────────────────────
    reg_loss_type: str = "huber"       # "huber" oder "mse"
    huber_delta:   float = 0.02        # Delta für HuberLoss (2% Return)

    # ── Modell-Architektur ─────────────────────────────────────────────────
    hidden_dim:  int   = 128
    num_layers:  int   = 2
    embed_dim:   int   = 16
    dropout:     float = 0.3
    seq_len:     int   = 64

    # ── Training ───────────────────────────────────────────────────────────
    lr:           float = 5e-4
    weight_decay: float = 1e-3
    epochs:       int   = 50
    patience:     int   = 7
    batch_size:   int   = 512
    grad_clip:    float = 1.0

    # ── Walk-Forward ───────────────────────────────────────────────────────
    train_years:  float = 3.0
    val_months:   float = 6.0
    step_months:  float = 6.0

    # ── Backtest ───────────────────────────────────────────────────────────
    portfolio_horizon_idx: int    = 2       # Welcher Horizont für Portfolio-Ranking (11d)
    n_max:                 int    = 7
    n_mid:                 int    = 3
    n_min:                 int    = 1
    hard_stop_pct:         float  = 0.25
    rotation_buffer:       int    = 3
    fees:                  float  = 0.001
    init_cash:             float  = 10_000.0

    # ── Combo-Score (optional) ─────────────────────────────────────────────
    # Wenn nicht leer: statt eines einzelnen Horizonts eine gewichtete Kombination
    # Beispiel: {1: 0.5, 2: 0.5} → 0.5*pred_7d + 0.5*pred_11d
    combo_weights: dict[int, float] = field(default_factory=dict)

    # ── Pfade ──────────────────────────────────────────────────────────────
    checkpoint_dir: Path = Path("checkpoints/v2_return_multi")

    @property
    def rank_horizon(self) -> int:
        return self.horizons[self.rank_horizon_idx]

    @property
    def portfolio_horizon(self) -> int:
        return self.horizons[self.portfolio_horizon_idx]

    @property
    def max_horizon(self) -> int:
        return max(self.horizons)
