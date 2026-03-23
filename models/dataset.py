"""
models/dataset.py
──────────────────
Walk-Forward Dataset und DataLoader für Cross-Sectional LSTM.

Walk-Forward Schema (Expanding Window):
  Fold 0: Train [t0 .. t0+3J]   Val [t0+3J .. t0+3J+6M]
  Fold 1: Train [t0 .. t0+3J+6M] Val [t0+3J+6M .. t0+3J+12M]
  Fold 2: Train [t0 .. t0+3J+12M] Val [t0+3J+12M .. t0+3J+18M]
  ...

Pro Sample im Dataset:
  X : (seq_len, n_features)  — letzten seq_len Tage Features dieses Assets
  y : float                  — Forward Return dieses Assets
  asset_id : int             — Asset-Index für Embedding
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ── Walk-Forward Fold Definition ──────────────────────────────────────────────

@dataclass
class WalkForwardFold:
    fold_id:    int
    train_start: pd.Timestamp
    train_end:   pd.Timestamp
    val_start:   pd.Timestamp
    val_end:     pd.Timestamp

    def __repr__(self):
        return (f"Fold {self.fold_id}: "
                f"Train [{self.train_start.date()} → {self.train_end.date()}]  "
                f"Val [{self.val_start.date()} → {self.val_end.date()}]")


def create_walk_forward_folds(
    dates:          pd.DatetimeIndex,
    train_years:    float = 3.0,
    val_months:     float = 6.0,
    step_months:    float = 6.0,
    min_train_rows: int   = 500,
    embargo_months: float = 1.0,   # Pufferzone zwischen train_end und val_start
) -> list[WalkForwardFold]:
    """
    Erzeugt Walk-Forward Folds (Expanding Window) mit Embargo-Phase.

    Args:
        dates:          Alle verfügbaren Handelstage
        train_years:    Mindest-Trainingsperiode in Jahren
        val_months:     Validierungsperiode in Monaten
        step_months:    Schrittweite in Monaten
        min_train_rows: Mindest-Anzahl Trainingstage
        embargo_months: Pufferzone zwischen train_end und val_start (verhindert
                        Data Leakage durch überlappende Forward-Return-Fenster).
                        Muss >= horizon / 21 Handelstage sein.
                        Default 1 Monat (~21 Handelstage) für horizon <= 21.

    Warum Embargo?
        Target[t] = Return über [t, t+horizon]. Das letzte Training-Sample
        bei train_end hat ein Target das bis train_end+horizon reicht.
        Ohne Embargo überschneiden sich Train-Targets und Val-Periode →
        Lookahead-Bias / Data Leakage.

        Beispiel horizon=11:
          train_end  = 31. Dez
          Ohne Embargo: val_start = 1. Jan  ← letztes Train-Target reicht bis 12. Jan!
          Mit Embargo:  val_start = 1. Feb  ← sauber getrennt

    Returns:
        Liste von WalkForwardFold Objekten
    """
    dates  = pd.DatetimeIndex(sorted(set(dates)))
    t0     = dates[0]
    t_end  = dates[-1]

    train_delta   = pd.DateOffset(years=int(train_years),
                                  months=int((train_years % 1) * 12))
    val_delta     = pd.DateOffset(months=int(val_months))
    step_delta    = pd.DateOffset(months=int(step_months))
    embargo_delta = pd.DateOffset(months=int(embargo_months))

    folds   = []
    fold_id = 0
    # val_start = früheste mögliche Val-Start-Zeit (nach Training + Embargo)
    val_start = t0 + train_delta + embargo_delta

    while val_start + val_delta <= t_end:
        val_end = val_start + val_delta

        # Nächstliegender echter Handelstag
        ts_idx = dates.searchsorted(val_start)
        te_idx = dates.searchsorted(val_end)

        if ts_idx >= len(dates) or te_idx >= len(dates):
            break

        val_start_real = dates[min(ts_idx, len(dates) - 1)]
        val_end_real   = dates[min(te_idx, len(dates) - 1)]

        # train_end = letzter Tag VOR dem Embargo-Puffer
        # d.h. horizon Handelstage vor val_start_real
        embargo_start = val_start_real - embargo_delta
        train_end_idx = dates.searchsorted(embargo_start)
        if train_end_idx == 0 or train_end_idx < min_train_rows:
            val_start += step_delta
            continue

        fold = WalkForwardFold(
            fold_id      = fold_id,
            train_start  = t0,
            train_end    = dates[train_end_idx - 1],   # letzter Tag vor Embargo
            val_start    = val_start_real,
            val_end      = val_end_real,
        )
        folds.append(fold)
        fold_id   += 1
        val_start += step_delta

    return folds


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class CrossSectionalDataset(Dataset):
    """
    Dataset für Cross-Sectional Multi-Asset LSTM.

    Pro Sample:
      X        : (seq_len, n_features) — historische Features
      y        : float                 — Forward Return (Target)
      asset_id : int                   — Asset-Index für Embedding

    Aufbau:
      Für jeden Handelstag t und jedes Asset a:
        - Nehme die letzten seq_len Tage Features von Asset a
        - Target = Forward Return von Asset a an Tag t
    """

    def __init__(
        self,
        features:  pd.DataFrame,   # MultiIndex (date, asset) × features
        targets:   pd.Series,      # MultiIndex (date, asset) → float
        asset_map: dict[str, int], # asset_name → integer ID
        seq_len:   int = 64,
        start_date: Optional[pd.Timestamp] = None,
        end_date:   Optional[pd.Timestamp] = None,
    ):
        self.seq_len   = seq_len
        self.asset_map = asset_map
        self.n_features = len(features.columns)

        # Zeitraum filtern
        dates = features.index.get_level_values("date")
        mask  = pd.Series(True, index=features.index)
        if start_date is not None:
            mask &= dates >= start_date
        if end_date is not None:
            mask &= dates <= end_date

        features = features[mask.values]
        targets  = targets[mask.values]

        # Assets und Daten vorbereiten
        self.samples = []
        assets = features.index.get_level_values("asset").unique()

        for asset in assets:
            asset_id = asset_map.get(asset, 0)

            # Features und Targets für dieses Asset
            try:
                asset_feat   = features.xs(asset, level="asset").sort_index()
                asset_target = targets.xs(asset, level="asset").sort_index()
            except KeyError:
                continue

            # Gemeinsamen Zeitindex
            common_idx = asset_feat.index.intersection(asset_target.index)
            asset_feat   = asset_feat.loc[common_idx]
            asset_target = asset_target.loc[common_idx]

            feat_arr   = asset_feat.values.astype(np.float32)
            target_arr = asset_target.values.astype(np.float32)

            # Samples aufbauen:
            #   x = feat_arr[i-seq_len+1 : i+1]  ← letztes Element = Tag i (Entscheidungstag)
            #   y = target_arr[i]                 ← Forward Return ab Tag i
            #
            # Warum i-seq_len+1 statt i-seq_len?
            #   feat_arr[i-seq_len : i] würde Tag i *ausschließen* (Python-Slicing exklusiv).
            #   Das Modell sähe nur Features bis Tag i-1, müsste aber den Return
            #   ab Tag i vorhersagen → Off-by-One / fehlendes Signal des Entscheidungstages.
            #   Korrekt: Sequenz endet inklusive Tag i.
            #
            # range-Grenzen:
            #   Start: seq_len-1  → i=seq_len-1: x = feat_arr[0:seq_len] ✓
            #   Ende:  len-1      → i+1 <= len garantiert keinen IndexError
            for i in range(seq_len - 1, len(feat_arr)):
                x = feat_arr[i - seq_len + 1 : i + 1]   # (seq_len, n_features), inkl. Tag i
                y = target_arr[i]                        # Forward Return ab Tag i

                # NaN/Inf überspringen
                if np.isnan(x).any() or np.isnan(y) or np.isinf(y):
                    continue

                self.samples.append((x, y, asset_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y, asset_id = self.samples[idx]
        return (
            torch.from_numpy(x),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(asset_id, dtype=torch.long),
        )


# Import fix
from typing import Optional


# ── DataLoader Factory ────────────────────────────────────────────────────────

def make_dataloaders(
    features:    pd.DataFrame,
    targets:     pd.Series,
    fold:        WalkForwardFold,
    asset_map:   dict[str, int],
    seq_len:     int = 64,
    batch_size:  int = 512,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """
    Erstellt Train- und Val-DataLoader für einen Walk-Forward Fold.
    """
    train_ds = CrossSectionalDataset(
        features, targets, asset_map, seq_len,
        start_date = fold.train_start,
        end_date   = fold.train_end,
    )
    val_ds = CrossSectionalDataset(
        features, targets, asset_map, seq_len,
        start_date = fold.val_start,
        end_date   = fold.val_end,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,
        drop_last   = True,
        num_workers = num_workers,
        pin_memory  = torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size * 2,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = torch.cuda.is_available(),
    )

    return train_loader, val_loader
