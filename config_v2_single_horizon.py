"""
config_v2_single_horizon.py
─────────────────────────────
Konfiguration fuer den Single-Horizon-Vergleich:
  4 separate Modelle (v2_4d, v2_7d, v2_11d, v2_15d),
  strukturell identisch zu v1_rank, aber je mit eigenem Horizont.

Ziel: empirisch den optimalen Einzelhorizont finden.
"""

from dataclasses import dataclass
from pathlib import Path

HORIZONS = [4, 7, 11, 15]


@dataclass
class SingleHorizonConfig:
    horizon: int = 11

    # ── Loss ───────────────────────────────────────────────────────────
    rank_weight: float = 0.5    # lambda_rank — identisch zu v1 Run G
    rank_margin: float = 0.001

    # ── Modell (identisch zu v1) ───────────────────────────────────────
    hidden_dim:  int   = 128
    num_layers:  int   = 2
    embed_dim:   int   = 16
    dropout:     float = 0.3
    seq_len:     int   = 64

    # ── Training ───────────────────────────────────────────────────────
    lr:           float = 5e-4
    weight_decay: float = 1e-3
    epochs:       int   = 50
    patience:     int   = 7
    batch_size:   int   = 512
    grad_clip:    float = 1.0

    # ── Walk-Forward ───────────────────────────────────────────────────
    train_years:  float = 3.0
    val_months:   float = 6.0
    step_months:  float = 6.0

    # ── Backtest (Run G identisch) ─────────────────────────────────────
    n_max:           int   = 7
    n_mid:           int   = 3
    n_min:           int   = 1
    hard_stop_pct:   float = 0.25
    rotation_buffer: int   = 3
    fees:            float = 0.001
    init_cash:       float = 10_000.0

    @property
    def tag(self) -> str:
        return f"v2_{self.horizon}d"

    @property
    def checkpoint_dir(self) -> Path:
        return Path(f"checkpoints/v2_{self.horizon}d")


def get_config(horizon: int) -> SingleHorizonConfig:
    """Gibt die Basis-Config fuer einen bestimmten Horizont zurueck."""
    return SingleHorizonConfig(horizon=horizon)
