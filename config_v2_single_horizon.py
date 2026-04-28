"""
config_v2_single_horizon.py
────────────────────────────
Modell- und Portfolio-Konfiguration für den v2 Single-Horizon Trading Bot.

Architektur-Entscheidungen (Forschungsphase abgeschlossen Apr 2025):
  - Vorhersage-Horizont : 7 Handelstage
  - Normalisierung       : Sektor-neutral (pro Tag UND pro GICS-Sektor)
  - Risikomanagement     : A3-Policy (IC_roll_40 < 0 → n_max reduzieren)
  - Gewinnende Portfolio-Parameter: n_max=5, rotation_buffer=2, hard_stop=20 %

Smoke-Test-Modus (KAGGLE_SMOKE_TEST=1):
  Reduziert Epochen und Folds drastisch für schnelle Pipeline-Validierung (~10 Min).
"""

import os
from dataclasses import dataclass
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# Produktions-Konfiguration (Ergebnis der Backtesting-Phase)
# Alle Portfolio-Komponenten lesen ihre Defaults aus diesem Block.
# ══════════════════════════════════════════════════════════════════════════════

PROD_HORIZON:          int   = 7       # Vorhersage-Horizont in Handelstagen
PROD_N_MAX:            int   = 5       # Max. gleichzeitige Positionen (Grid-Search-Sieger)
PROD_N_MID:            int   = 2       # Positionen im Neutral-Regime (SMA50 ≈ SMA200)
PROD_N_MIN:            int   = 1       # Positionen im Bear-Regime (SPY < SMA200)
PROD_ROTATION_BUFFER:  int   = 2       # Position erst schließen wenn Rank > n_max + buffer
PROD_HARD_STOP_PCT:    float = 0.20    # Automatischer Verlust-Stop bei −20 %
PROD_FEES:             float = 0.001   # Handelsgebühren pro Trade (0.1 %)
PROD_SECTOR_NEUTRAL:   bool  = True    # Sektor-neutrale Z-Score Normalisierung (Sieger)
PROD_POLICY:           str   = "IC40"  # A3-Policy: n_max→3 wenn IC_roll_40 < 0
PROD_POLICY_REDUCED_N: int   = 3       # n_max während aktiver A3-Policy

HORIZONS = [4, 7, 11, 15]


@dataclass
class SingleHorizonConfig:
    """Vollständige Konfiguration für ein Single-Horizon-Modell.

    Deckt Modell-Architektur, Training, Walk-Forward-Split, Feature-Engineering
    und Backtest-Parameter ab. Alle Felder haben sinnvolle Produktions-Defaults.

    Args:
        horizon: Vorhersage-Horizont in Handelstagen.

    Example:
        >>> cfg = get_config(7)
        >>> cfg.sector_neutral
        True
    """

    horizon: int = PROD_HORIZON

    # ── Loss ───────────────────────────────────────────────────────────────────
    rank_weight: float = 0.5    # λ_rank – Gewicht des ListNet Rank-Loss
    rank_margin: float = 0.001  # Margin im Pairwise-Vergleich

    # ── Modell-Architektur ─────────────────────────────────────────────────────
    hidden_dim:  int   = 128
    num_layers:  int   = 2
    embed_dim:   int   = 16    # Asset-Embedding Dimension
    dropout:     float = 0.3
    seq_len:     int   = 64    # Lookback-Fenster in Handelstagen

    # ── Training ───────────────────────────────────────────────────────────────
    lr:           float = 5e-4
    weight_decay: float = 1e-3
    epochs:       int   = 50
    patience:     int   = 7    # Early-Stopping nach n Epochen ohne Verbesserung
    batch_size:   int   = 512
    grad_clip:    float = 1.0
    # Reproduzierbarer Seed: pro Fold wird seed + fold_id gesetzt.
    # Verschiedene Seeds per CLI (KAGGLE_SEED=42) testen, besten Lauf behalten.
    seed:         int   = 42

    # ── Walk-Forward Splits ────────────────────────────────────────────────────
    train_years:  float = 3.0  # Trainingsfenster (expandierend)
    val_months:   float = 6.0  # Validierungsfenster je Fold
    step_months:  float = 6.0  # Schrittweite zwischen Folds

    # ── Feature Engineering ────────────────────────────────────────────────────
    # Sektor-neutrale Normalisierung: Z-Score pro Tag UND pro GICS-Sektor.
    # Erfordert features/sector_map.json. Produktions-Default: True.
    sector_neutral: bool = PROD_SECTOR_NEUTRAL

    # ── Portfolio / Backtest ───────────────────────────────────────────────────
    n_max:           int   = PROD_N_MAX
    n_mid:           int   = PROD_N_MID
    n_min:           int   = PROD_N_MIN
    hard_stop_pct:   float = PROD_HARD_STOP_PCT
    rotation_buffer: int   = PROD_ROTATION_BUFFER
    fees:            float = PROD_FEES
    init_cash:       float = 10_000.0

    @property
    def tag(self) -> str:
        """Eindeutiger Bezeichner für diesen Horizont, z.B. 'v2_7d'."""
        return f"v2_{self.horizon}d"

    @property
    def checkpoint_dir(self) -> Path:
        """Verzeichnis für gespeicherte Modell-Checkpoints."""
        return Path(f"checkpoints/v2_{self.horizon}d")


def get_config(horizon: int) -> SingleHorizonConfig:
    """Gibt die Produktionskonfiguration für einen Horizont zurück.

    Wenn die Umgebungsvariable ``KAGGLE_SMOKE_TEST=1`` gesetzt ist, werden
    Epochen und Folds drastisch reduziert, sodass die gesamte Pipeline in
    ~10 Minuten durchläuft (vollständiger Integrations-Test ohne GPU-Stunden).

    Args:
        horizon: Vorhersage-Horizont in Handelstagen (z.B. 7).

    Returns:
        SingleHorizonConfig mit Produktions-Defaults und ggf. Smoke-Test-Overrides.

    Example:
        >>> cfg = get_config(7)
        >>> assert cfg.sector_neutral is True
        >>> assert cfg.n_max == 5
    """
    cfg = SingleHorizonConfig(horizon=horizon)

    if os.environ.get("KAGGLE_SMOKE_TEST", "").strip() == "1":
        cfg.epochs      = 3
        cfg.patience    = 99   # kein Early-Stopping im Smoke-Test
        cfg.train_years = 8.0  # bei ~9 Jahren Daten → ca. 2 Folds
        cfg.batch_size  = 512

    # Seed per Umgebungsvariable überschreibbar: KAGGLE_SEED=123
    seed_env = os.environ.get("KAGGLE_SEED", "").strip()
    if seed_env.isdigit():
        cfg.seed = int(seed_env)

    return cfg
