"""
models/trainer.py
──────────────────
Trainings-Loop mit Early Stopping, LR-Scheduling, Gradient Clipping.
Unterstützt einzelne Assets UND kombinierten Multi-Asset-Datensatz.

Aufruf:
    python models/trainer.py --ticker AAPL --timeframe 1h
    python models/trainer.py --ticker combined --timeframe 1h
    python models/trainer.py --symbol BTC/USDT --timeframe 1h
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from models.lstm_model import TradingLSTM

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_DIR    = Path("features/processed")

DEFAULTS = {
    "hidden_dim":    128,
    "num_layers":    2,
    "dropout":       0.3,
    "bidirectional": False,
    "use_attention": True,
    "lr":            1e-3,
    "weight_decay":  1e-3,   # erhöht: stärkere L2-Regularisierung gegen Overfitting
    "batch_size":    256,
    "epochs":        100,
    "patience":      10,
    "grad_clip":     1.0,
    "device":        "cuda" if torch.cuda.is_available() else "cpu",
}


def resolve_ticker(symbol: str | None, ticker: str | None) -> str:
    """Gibt den normalisierten Ticker-Key zurück."""
    if ticker:
        return ticker.replace("/", "_")
    if symbol:
        return symbol.replace("/", "_")
    return "BTC_USDT"


def load_split(ticker_key: str, timeframe: str, mode: str, split: str, device: str) -> TensorDataset:
    """Lädt einen Split aus features/processed/.
    Unterstützt beide Pfadformate:
      - Neu: features/processed/<ticker>_<timeframe>/<ticker>_<timeframe>_<mode>_<split>.pt
      - Alt: features/processed/<ticker>/<ticker>_<timeframe>_<mode>_<split>.pt
    """
    prefix       = f"{ticker_key}_{timeframe}_{mode}_"
    out_dir_new  = FEATURE_DIR / f"{ticker_key}_{timeframe}"
    out_dir_old  = FEATURE_DIR / ticker_key
    if (out_dir_new / f"{prefix}{split}.pt").exists():
        out_dir = out_dir_new
    elif (out_dir_old / f"{prefix}{split}.pt").exists():
        out_dir = out_dir_old
    else:
        out_dir = out_dir_new  # für klare Fehlermeldung
    path = out_dir / f"{prefix}{split}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Datei nicht gefunden: {path}\n"
            f"Bitte zuerst ausführen:\n"
            f"  python main.py features --symbol {ticker_key} --timeframe {timeframe}"
        )
    data = torch.load(path, map_location=device)
    return TensorDataset(data["X"], data["y"])


def binary_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return ((preds >= 0.5).float() == targets).float().mean().item()


def train_epoch(model, loader, optimizer, criterion, device, grad_clip):
    model.train()
    total = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item() * len(X)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, mode):
    model.eval()
    total, all_p, all_t = 0.0, [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        p    = model(X)
        total += criterion(p, y).item() * len(X)
        all_p.append(p.cpu()); all_t.append(y.cpu())
    preds   = torch.cat(all_p)
    targets = torch.cat(all_t)
    metric  = binary_accuracy(preds, targets) if mode == "cls" else 0.0
    return total / len(loader.dataset), metric


def train(
    symbol:     str | None = "BTC/USDT",
    timeframe:  str        = "1h",
    mode:       str        = "cls",
    ticker:     str | None = None,
    **kwargs,
) -> TradingLSTM:

    cfg    = {**DEFAULTS, **kwargs}
    device = cfg["device"]
    tk     = resolve_ticker(symbol, ticker)

    logger.info(f"Training: {tk} [{timeframe}]  Device={device}")

    train_ds = load_split(tk, timeframe, mode, "train", device)
    val_ds   = load_split(tk, timeframe, mode, "val",   device)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False)

    n_features = train_ds[0][0].shape[-1]
    seq_len    = train_ds[0][0].shape[0]
    logger.info(f"n_features={n_features}, seq_len={seq_len}, "
                f"train={len(train_ds)}, val={len(val_ds)}")

    model = TradingLSTM(
        n_features=n_features, seq_len=seq_len,
        hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"],
        dropout=cfg["dropout"], bidirectional=cfg["bidirectional"],
        use_attention=cfg["use_attention"], mode=mode,
    ).to(device)
    model.init_weights()

    criterion = nn.BCELoss() if mode == "cls" else nn.HuberLoss(delta=0.01)
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    # CosineAnnealingLR: LR sinkt sanft von lr bis lr/100 über alle Epochen
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=cfg["lr"] / 100
    )

    best_val_loss  = float("inf")
    patience_count = 0
    ckpt_path      = CHECKPOINT_DIR / f"{tk}_{timeframe}_{mode}_best.pt"

    logger.info("─" * 60)
    logger.info(f"{'Ep':>4} {'TrainLoss':>10} {'ValLoss':>10} {'ValAcc':>8} {'LR':>10}")
    logger.info("─" * 60)

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss         = train_epoch(model, train_loader, optimizer, criterion, device, cfg["grad_clip"])
        val_loss, val_acc  = eval_epoch(model, val_loader,   criterion, device, mode)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Overfitting-Warnung: Val-Loss > 1.5x Train-Loss
        overfit_ratio = val_loss / train_loss if train_loss > 0 else 1.0
        overfit_warn  = "  ⚠ overfit" if overfit_ratio > 1.5 else ""

        logger.info(f"{epoch:4d} {train_loss:10.5f} {val_loss:10.5f} "
                    f"{val_acc:8.4f} {current_lr:10.7f}{overfit_warn}")

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    val_loss,
                "val_metric":  val_acc,
                "config":      cfg,
            }, ckpt_path)
            logger.success(f"  ✓ Checkpoint gespeichert (val_loss={val_loss:.5f})")
        else:
            patience_count += 1
            if patience_count >= cfg["patience"]:
                logger.warning(f"Early Stopping nach Epoche {epoch}.")
                break

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    logger.success(f"Training abgeschlossen. Bestes Modell: Epoche {ckpt['epoch']}, "
                   f"Val-Loss={ckpt['val_loss']:.5f}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",    default="BTC/USDT")
    parser.add_argument("--ticker",    default=None)
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--mode",      choices=["cls","reg"], default="cls")
    parser.add_argument("--epochs",    type=int,   default=DEFAULTS["epochs"])
    parser.add_argument("--patience",  type=int,   default=DEFAULTS["patience"])
    parser.add_argument("--hidden_dim",type=int,   default=DEFAULTS["hidden_dim"])
    parser.add_argument("--batch_size",type=int,   default=DEFAULTS["batch_size"])
    parser.add_argument("--lr",        type=float, default=DEFAULTS["lr"])
    args = parser.parse_args()
    train(symbol=args.symbol, ticker=args.ticker, timeframe=args.timeframe,
          mode=args.mode, epochs=args.epochs, patience=args.patience,
          hidden_dim=args.hidden_dim, batch_size=args.batch_size, lr=args.lr)
