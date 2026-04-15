"""
config_v2_1_return_multi_heads.py
──────────────────────────────────
Konfiguration für v2.1: Shared Trunk + Separate Heads (Return 7d/15d + Ranking 11d).

Design-Entscheidung:
  v2 hatte 4 Horizonte auf einem gemeinsamen Head → Ranking-IC halbiert.
  v2.1 trennt Return-Regression (7d, 15d) vom dedizierten Ranking (11d),
  damit der Ranking-Head mit starkem RankLoss trainiert werden kann (wie v1),
  während die Return-Heads kalibrierte Vorhersagen liefern.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class V21Config:
    # ── Horizonte ──────────────────────────────────────────────────────────
    return_horizons: list[int] = field(default_factory=lambda: [7, 15])
    rank_horizon:    int       = 11

    # ── Loss-Gewichte ──────────────────────────────────────────────────────
    # Return-Heads: Huber/MSE Regression
    w_ret_7d:    float = 0.5
    w_ret_15d:   float = 0.5
    # Ranking-Head: PairwiseRankLoss (wie v1 CombinedLoss)
    lambda_rank: float = 0.5       # starkes Ranking wie v1 (v2 hatte 0.1)
    rank_margin: float = 0.001
    # Optional: kleiner Level-Term für den Ranking-Head
    w_rank_reg:  float = 0.1       # MSE/Huber auf score_11d vs y_11d

    # ── Regression-Loss-Typ ────────────────────────────────────────────────
    reg_loss_type: str   = "huber"
    huber_delta:   float = 0.02

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
    n_max:           int   = 7
    n_mid:           int   = 3
    n_min:           int   = 1
    hard_stop_pct:   float = 0.25
    rotation_buffer: int   = 3
    fees:            float = 0.001
    init_cash:       float = 10_000.0

    # ── Pfade ──────────────────────────────────────────────────────────────
    checkpoint_dir: Path = Path("checkpoints/v2_1_return_multi_heads")

    @property
    def max_horizon(self) -> int:
        return max(*self.return_horizons, self.rank_horizon)

    @property
    def all_horizons(self) -> list[int]:
        return sorted(set(self.return_horizons + [self.rank_horizon]))
