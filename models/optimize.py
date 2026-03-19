"""
models/optimize.py
───────────────────
Hyperparameter-Optimierung mit Optuna.

Optimiert folgende Parameter automatisch:
  • threshold      – Mindest-Anstieg für Label=1  (0.10% – 0.50%)
  • horizon        – Vorausschauhorizont           (3 – 24 Perioden)
  • seq_len        – Rückblick-Fenster             (24 – 96 Perioden)
  • entry_threshold– Ab wann kaufen (Backtest)     (0.50 – 0.65)
  • hidden_dim     – LSTM-Größe                    (64 – 256)
  • num_layers     – LSTM-Tiefe                    (1 – 3)
  • dropout        – Regularisierung               (0.1 – 0.5)
  • lr             – Lernrate                      (1e-4 – 1e-2)

Zielgröße: Val-Return (Backtest auf Validierungsdaten)
Alternativ: Val-Loss minimieren

Aufruf:
    python models/optimize.py --ticker AAPL --timeframe 1h --trials 50
    python models/optimize.py --multi --timeframe 1h --trials 30
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from features.engineer import (
    FEATURE_COLS,
    add_indicators,
    add_labels,
    make_sequences,
    scale_features,
    temporal_split,
    _load_asset_list,
    SPLIT,
)
from models.lstm_model import TradingLSTM

# Pfade
RAW_DIR        = Path("data/raw")
FEATURE_DIR    = Path("features/processed")
RESULTS_DIR    = Path("logs")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Optuna: keine INFO-Meldungen von internen Trials
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


# ── Daten für einen Trial laden ───────────────────────────────────────────────

def load_data_for_trial(
    ticker:     str,
    timeframe:  str,
    horizon:    int,
    threshold:  float,
    seq_len:    int,
    label_mode: str = "cls",
    multi:      bool = False,
) -> tuple[TensorDataset, TensorDataset] | None:
    """
    Berechnet Features + Labels für die gegebenen Hyperparameter on-the-fly.
    Gibt (train_ds, val_ds) zurück oder None bei Fehler.
    """
    import pandas as pd

    if multi:
        tickers = _load_asset_list()
    else:
        tickers = [ticker]

    all_train_X, all_train_y = [], []
    all_val_X,   all_val_y   = [], []

    for t in tickers:
        fname = RAW_DIR / f"{t.replace('.','_')}_{timeframe}.parquet"
        if not fname.exists():
            continue
        df = pd.read_parquet(fname)
        if len(df) < seq_len + 50:
            continue

        try:
            df = add_indicators(df)
            df = add_labels(df, horizon, threshold)
            df.dropna(inplace=True)
        except Exception:
            continue

        if len(df) < seq_len + 20:
            continue

        label_col = "label_cls" if label_mode == "cls" else "label_reg"
        n_train   = int(len(df) * SPLIT[0])

        try:
            df_train_s, scaler = scale_features(df.iloc[:n_train], fit_scaler=True)
            df_rest_s,  _      = scale_features(df.iloc[n_train:],  fit_scaler=False, scaler=scaler)
        except Exception:
            continue

        df_s   = pd.concat([df_train_s, df_rest_s])
        X, y   = make_sequences(df_s, seq_len, label_col)
        splits = temporal_split(X, y)

        all_train_X.append(splits["train"][0])
        all_train_y.append(splits["train"][1])
        all_val_X.append(splits["val"][0])
        all_val_y.append(splits["val"][1])

    if not all_train_X:
        return None

    train_X = np.concatenate(all_train_X)
    train_y = np.concatenate(all_train_y)
    val_X   = np.concatenate(all_val_X)
    val_y   = np.concatenate(all_val_y)

    # Train shufflen
    idx     = np.random.permutation(len(train_X))
    train_X = train_X[idx]
    train_y = train_y[idx]

    train_ds = TensorDataset(
        torch.from_numpy(train_X), torch.from_numpy(train_y)
    )
    val_ds = TensorDataset(
        torch.from_numpy(val_X), torch.from_numpy(val_y)
    )
    return train_ds, val_ds


# ── Schnelles Training für einen Trial ───────────────────────────────────────

def quick_train(
    train_ds:   TensorDataset,
    val_ds:     TensorDataset,
    hidden_dim: int,
    num_layers: int,
    dropout:    float,
    lr:         float,
    device:     str,
    epochs:     int = 15,
    patience:   int = 5,
) -> tuple[TradingLSTM, float]:
    """Schnelles Training (weniger Epochen) für Hyperparameter-Suche."""
    n_features = train_ds[0][0].shape[-1]
    seq_len    = train_ds[0][0].shape[0]

    model = TradingLSTM(
        n_features=n_features, seq_len=seq_len,
        hidden_dim=hidden_dim, num_layers=num_layers,
        dropout=dropout, mode="cls",
    ).to(device)
    model.init_weights()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False)

    best_val_loss = float("inf")
    patience_cnt  = 0

    for epoch in range(epochs):
        # Train
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                val_loss += criterion(model(X_b.to(device)), y_b.to(device)).item() * len(X_b)
        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt  = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    return model, best_val_loss


# ── Val-Return berechnen (Backtest auf Val-Daten) ─────────────────────────────

@torch.no_grad()
def val_return(
    model:           TradingLSTM,
    val_ds:          TensorDataset,
    entry_threshold: float,
    exit_threshold:  float,
    device:          str,
    fee:             float = 0.001,
) -> float:
    """Berechnet den Return auf den Validierungsdaten."""
    model.eval()
    X_all = val_ds.tensors[0].to(device)
    preds = model(X_all).cpu().numpy()

    # Preise rekonstruieren: wir nutzen den normierten Close als Proxy
    # (echter Preis nicht verfügbar hier — nutzen normalisierten Close)
    # Stattdessen: direkt Return über Label-basierte Simulation
    y_all  = val_ds.tensors[1].numpy()
    n      = len(preds)
    cash   = 1.0
    pos    = 0.0
    in_tr  = False
    ep     = 0.0

    for i in range(n):
        pred = preds[i]
        # Simulierter Return: wenn Label=1, nehmen wir +threshold als Proxy
        sim_ret = float(y_all[i]) * 0.002  # vereinfacht

        if pred >= entry_threshold and not in_tr:
            pos  = cash * (1 - fee)
            cash = 0.0
            in_tr = True
            ep    = sim_ret
        elif in_tr and (pred <= exit_threshold or i == n - 1):
            cash  = pos * (1 + ep) * (1 - fee)
            pos   = 0.0
            in_tr = False

    total = cash + pos - 1.0
    return float(total)


# ── Optuna Objective ──────────────────────────────────────────────────────────

def make_objective(
    ticker:    str,
    timeframe: str,
    device:    str,
    multi:     bool,
):
    def objective(trial: optuna.Trial) -> float:
        import time
        t_start = time.time()

        # ── Hyperparameter vorschlagen ─────────────────────────────────────
        threshold       = trial.suggest_float("threshold",       0.001, 0.005, step=0.0005)
        horizon         = trial.suggest_int  ("horizon",         3,     24)
        seq_len         = trial.suggest_int  ("seq_len",         24,    96,    step=8)
        entry_threshold = trial.suggest_float("entry_threshold", 0.50,  0.65,  step=0.01)
        hidden_dim      = trial.suggest_categorical("hidden_dim",  [64, 128, 256])
        num_layers      = trial.suggest_int  ("num_layers",      1,     3)
        dropout         = trial.suggest_float("dropout",         0.1,   0.5,   step=0.05)
        lr              = trial.suggest_float("lr",              1e-4,  1e-2,  log=True)

        logger.info(
            f"Trial {trial.number + 1:>3}  "
            f"hidden={hidden_dim} layers={num_layers} lr={lr:.5f} "
            f"horizon={horizon} seq={seq_len} thr={threshold:.4f}"
        )

        # ── Daten laden ────────────────────────────────────────────────────
        result = load_data_for_trial(
            ticker, timeframe, horizon, threshold, seq_len, "cls", multi
        )
        if result is None:
            logger.warning(f"  Trial {trial.number + 1}: Keine Daten — übersprungen")
            raise optuna.exceptions.TrialPruned()

        train_ds, val_ds = result
        if len(train_ds) < 64:
            logger.warning(f"  Trial {trial.number + 1}: Zu wenig Daten ({len(train_ds)}) — übersprungen")
            raise optuna.exceptions.TrialPruned()

        logger.debug(f"  Daten geladen: {len(train_ds)} Train / {len(val_ds)} Val Sequenzen")

        # ── Training ───────────────────────────────────────────────────────
        try:
            model, val_loss = quick_train(
                train_ds, val_ds,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                lr=lr,
                device=device,
                epochs=15,
                patience=5,
            )
        except Exception as e:
            logger.warning(f"  Trial {trial.number + 1} Fehler: {e}")
            raise optuna.exceptions.TrialPruned()

        elapsed = time.time() - t_start

        # Bestes bisher?
        completed = [t for t in trial.study.trials if t.value is not None]
        best_so_far = min((t.value for t in completed), default=float("inf"))
        is_best = val_loss <= best_so_far
        marker = " <-- BEST" if is_best else ""

        logger.info(
            f"  => Val-Loss={val_loss:.5f}  ({elapsed:.0f}s){marker}"
        )

        # ── Zielgröße: Val-Loss (minimieren) ───────────────────────────────
        return val_loss

    return objective


# ── Haupt-Optimierung ─────────────────────────────────────────────────────────

def _ensure_data(ticker: str, timeframe: str, multi: bool, years: int = 5) -> None:
    """Stellt sicher dass Daten vorhanden sind; lädt sie bei Bedarf herunter."""
    from data.download_stocks import fetch_ticker, save_ticker, load_asset_list

    tickers = load_asset_list() if multi else [ticker]
    missing = []
    for t in tickers:
        fname = RAW_DIR / f"{t.replace('.', '_')}_{timeframe}.parquet"
        if not fname.exists():
            missing.append(t)

    if not missing:
        return

    logger.warning(f"{len(missing)} Datei(en) fehlen — starte automatischen Download...")
    for t in missing:
        logger.info(f"  Lade {t} ({timeframe}, {years} Jahre)...")
        df = fetch_ticker(t, timeframe, years)
        if df is not None and len(df) > 50:
            save_ticker(df, t, timeframe)
            logger.success(f"  ✓ {t}: {len(df)} Kerzen gespeichert")
        else:
            logger.error(f"  ✗ {t}: Keine Daten verfügbar (übersprungen)")


def run_optimization(
    ticker:    str   = "AAPL",
    timeframe: str   = "1h",
    trials:    int   = 50,
    multi:     bool  = False,
    jobs:      int   = 1,
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Optuna Optimierung startet: {trials} Trials  Device={device}")
    logger.info(f"Asset: {'COMBINED (Multi-Asset)' if multi else ticker}  [{timeframe}]")
    logger.info("─" * 60)

    # Daten sicherstellen (auto-download wenn fehlend)
    _ensure_data(ticker, timeframe, multi)

    # Vorab prüfen ob genug Daten vorhanden
    tickers_check = _load_asset_list() if multi else [ticker]  # uses engineer._load_asset_list
    available = [t for t in tickers_check
                 if (RAW_DIR / f"{t.replace('.', '_')}_{timeframe}.parquet").exists()]
    if not available:
        raise RuntimeError(
            f"Keine Datendateien in data/raw/ für {ticker} ({timeframe}).\n"
            f"Führe zuerst aus: python main.py stocks --ticker {ticker} --timeframe {timeframe} --years 5"
        )
    logger.info(f"Gefundene Assets: {', '.join(available)}")

    study = optuna.create_study(
        direction  = "minimize",     # Val-Loss minimieren
        pruner     = optuna.pruners.MedianPruner(n_startup_trials=5),
        sampler    = optuna.samplers.TPESampler(seed=42),
        study_name = f"trading_{ticker}_{timeframe}",
    )

    study.optimize(
        make_objective(ticker, timeframe, device, multi),
        n_trials   = trials,
        n_jobs     = jobs,
        show_progress_bar = False,
    )

    completed = [t for t in study.trials if t.value is not None]
    if not completed:
        raise RuntimeError(
            "Alle Trials wurden verworfen (Pruned). Mögliche Ursachen:\n"
            "  • Zu wenig Daten in data/raw/ (mind. 200 Kerzen benötigt)\n"
            "  • Timeframe passt nicht zur heruntergeladenen Datei\n"
            f"  Tipp: Prüfe ob data/raw/{ticker.replace('.','_')}_{timeframe}.parquet existiert."
        )

    best = study.best_trial
    logger.success("═" * 60)
    logger.success("BESTE PARAMETER GEFUNDEN")
    logger.success("═" * 60)
    for k, v in best.params.items():
        logger.success(f"  {k:<25}: {v}")
    logger.success(f"  {'Val-Loss':<25}: {best.value:.5f}")
    logger.success("═" * 60)

    # Ergebnisse speichern
    result = {
        "ticker":    ticker,
        "timeframe": timeframe,
        "multi":     multi,
        "trials":    trials,
        "best_params": best.params,
        "best_val_loss": best.value,
        "all_trials": [
            {"number": t.number, "val_loss": t.value, "params": t.params}
            for t in study.trials
            if t.value is not None
        ],
    }

    out_file = RESULTS_DIR / f"optuna_{ticker}_{timeframe}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    logger.success(f"Ergebnisse gespeichert: {out_file}")

    # Top-5 Trials ausgeben
    top5 = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value
    )[:5]
    logger.info("\nTop 5 Trials:")
    for t in top5:
        logger.info(f"  Trial {t.number:3d}  Val-Loss={t.value:.5f}  "
                    f"threshold={t.params.get('threshold','-'):.4f}  "
                    f"horizon={t.params.get('horizon','-')}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker",    default="AAPL")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--trials",    type=int,  default=50)
    parser.add_argument("--multi",     action="store_true",
                        help="Multi-Asset Training (alle Assets aus asset_list.txt)")
    parser.add_argument("--jobs",      type=int,  default=1,
                        help="Parallele Trials (nur bei mehreren CPU-Kernen sinnvoll)")
    args = parser.parse_args()

    run_optimization(
        ticker    = args.ticker,
        timeframe = args.timeframe,
        trials    = args.trials,
        multi     = args.multi,
        jobs      = args.jobs,
    )
