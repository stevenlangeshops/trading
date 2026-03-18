"""
strategy/backtest.py
─────────────────────
Backtesting der Modell-Signale.
Unterstützt einzelne Assets und kombinierte Modelle.

Aufruf:
    python strategy/backtest.py --ticker AAPL --timeframe 1h
    python strategy/backtest.py --ticker combined --timeframe 1h --test_on AAPL
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

from models.lstm_model import TradingLSTM

CHECKPOINT_DIR  = Path("checkpoints")
FEATURE_DIR     = Path("features/processed")
RAW_DIR         = Path("data/raw")
REPORT_DIR      = Path("logs")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

ENTRY_THRESHOLD = 0.55   # Default – via --entry_threshold überschreibbar
EXIT_THRESHOLD  = 0.45   # Default – via --exit_threshold überschreibbar


@torch.no_grad()
def run_inference(model, X_test, device, batch_size=512):
    model.eval()
    out = []
    for i in range(0, len(X_test), batch_size):
        out.append(model(X_test[i:i+batch_size].to(device)).cpu().numpy())
    return np.concatenate(out)


def manual_backtest(
    prices,
    preds,
    entry_thresh,
    exit_thresh,
    init_cash   = 10_000.0,
    fee         = 0.001,
    hold_days   = 24,     # Maximale Haltedauer (entspricht horizon aus Training)
    stop_loss   = 0.05,   # 5% Stop-Loss
):
    """
    Backtesting mit drei Exit-Bedingungen:
      1. Konfidenz fällt unter exit_thresh
      2. Position wurde hold_days Tage gehalten (horizon-basiert)
      3. Stop-Loss bei -stop_loss% Verlust
    """
    cash, position, entry_price = init_cash, 0.0, 0.0
    in_trade   = False
    hold_count = 0
    equity, trades = [], []

    for i, (price, pred) in enumerate(zip(prices, preds)):
        last = (i == len(prices) - 1)

        # ── Exit-Prüfung ────────────────────────────────────────────────
        if in_trade:
            hold_count += 1
            pnl_pct     = (price - entry_price) / entry_price

            should_exit = (
                pred  <= exit_thresh          # Signal schwach
                or hold_count >= hold_days    # Maximale Haltedauer erreicht
                or pnl_pct    <= -stop_loss   # Stop-Loss
                or last                       # Letzter Datenpunkt
            )

            if should_exit:
                proceeds = position * price * (1 - fee)
                pnl      = proceeds - entry_price * position
                reason   = (
                    "stop_loss"   if pnl_pct <= -stop_loss else
                    "horizon"     if hold_count >= hold_days else
                    "signal_exit" if pred <= exit_thresh else
                    "end"
                )
                trades.append({
                    "entry":      entry_price,
                    "exit":       price,
                    "pnl":        pnl,
                    "ret":        pnl / (entry_price * position),
                    "hold_days":  hold_count,
                    "reason":     reason,
                })
                cash       = proceeds
                position   = 0.0
                in_trade   = False
                hold_count = 0

        # ── Entry-Prüfung (nur wenn nicht in Position) ─────────────────
        if not in_trade and pred >= entry_thresh and not last:
            position    = (cash * (1 - fee)) / price
            entry_price = price
            cash        = 0.0
            in_trade    = True
            hold_count  = 0

        equity.append(cash + position * price)

    equity = np.array(equity)
    total  = (equity[-1] / init_cash - 1) * 100 if len(equity) else 0
    peak   = np.maximum.accumulate(equity)
    max_dd = ((equity - peak) / peak * 100).min() if len(equity) > 1 else 0
    # Tagesbasierter Sharpe (252 Handelstage/Jahr)
    rets   = np.diff(equity) / (equity[:-1] + 1e-9)
    sharpe = (rets.mean() / (rets.std() + 1e-9)) * np.sqrt(252) if len(rets) > 1 else 0
    n      = len(trades)
    win_r  = len([t for t in trades if t["pnl"] > 0]) / n * 100 if n else 0
    avg_r  = np.mean([t["ret"] for t in trades]) * 100 if n else 0

    # Exit-Gründe ausgeben
    if trades:
        from collections import Counter
        reasons = Counter(t["reason"] for t in trades)
        logger.info(f"Exit-Gründe: " + "  ".join(f"{k}={v}" for k, v in reasons.items()))
        avg_hold = np.mean([t["hold_days"] for t in trades])
        logger.info(f"Ø Haltedauer: {avg_hold:.1f} Tage")

    return {
        "Startkapital [USDT]":  init_cash,
        "Endkapital [USDT]":    round(equity[-1], 2) if len(equity) else init_cash,
        "Total Return [%]":     round(total, 2),
        "Max Drawdown [%]":     round(max_dd, 2),
        "Sharpe Ratio":         round(sharpe, 3),
        "Anzahl Trades":        n,
        "Win Rate [%]":         round(win_r, 2),
        "Ø Return/Trade [%]":   round(avg_r, 2),
    }, trades


def load_model(ticker_key: str, timeframe: str, mode: str, device: str,
               n_features: int, seq_len: int) -> tuple[TradingLSTM, dict]:
    ckpt_path = CHECKPOINT_DIR / f"{ticker_key}_{timeframe}_{mode}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Kein Checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt["config"]
    model = TradingLSTM(
        n_features=n_features, seq_len=seq_len,
        hidden_dim=cfg.get("hidden_dim", 128),
        num_layers=cfg.get("num_layers", 2),
        dropout=0.0,
        bidirectional=cfg.get("bidirectional", False),
        use_attention=cfg.get("use_attention", True),
        mode=mode,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    logger.success(f"Modell geladen: {ckpt_path.name}  "
                   f"(Epoche {ckpt['epoch']}, Val-Loss={ckpt['val_loss']:.5f})")
    return model, ckpt


def run_backtest(
    ticker:          str   = "BTC_USDT",
    timeframe:       str   = "1h",
    mode:            str   = "cls",
    init_cash:       float = 10_000.0,
    fees:            float = 0.001,
    test_on:         str | None = None,
    symbol:          str | None = None,
    entry_threshold: float = ENTRY_THRESHOLD,
    exit_threshold:  float = EXIT_THRESHOLD,
    hold_days:       int   = 24,
    stop_loss:       float = 0.05,
) -> dict:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Ticker normalisieren
    if symbol and not ticker:
        ticker = symbol.replace("/", "_")
    ticker = ticker.replace("/", "_")

    # Bei combined Modell: test_on muss angegeben sein, sonst default auf AAPL
    if ticker == "combined" and not test_on:
        test_on = "AAPL"
        logger.info("Combined Modell erkannt — teste auf AAPL (Standard). "
                    "Anderes Asset: --test_on MSFT")

    # Daten-Ticker: wenn test_on angegeben, auf diesem testen
    data_ticker = (test_on or ticker).replace("/", "_").replace(".", "_")

    # Test-Daten laden
    # Engineer speichert unter: features/processed/<ticker>_<timeframe>/<ticker>_<timeframe>_<mode>_<split>.pt
    # Fallback: features/processed/<ticker>/<ticker>_<timeframe>_<mode>_<split>.pt (altes Format)
    prefix   = f"{data_ticker}_{timeframe}_{mode}_"
    out_dir_new = FEATURE_DIR / f"{data_ticker}_{timeframe}"
    out_dir_old = FEATURE_DIR / data_ticker
    if (out_dir_new / f"{prefix}test.pt").exists():
        out_dir = out_dir_new
    elif (out_dir_old / f"{prefix}test.pt").exists():
        out_dir = out_dir_old
    else:
        out_dir = out_dir_new   # für klare Fehlermeldung
    pt_path  = out_dir / f"{prefix}test.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"Test-Daten nicht gefunden: {pt_path}")

    test_data  = torch.load(pt_path, map_location="cpu")
    X_test     = test_data["X"]
    n_features = X_test.shape[-1]
    seq_len    = X_test.shape[1]

    # Modell laden (ggf. kombiniertes Modell)
    model, ckpt = load_model(ticker, timeframe, mode, device, n_features, seq_len)

    # Inference
    preds = run_inference(model, X_test, device)
    logger.info(f"Vorhersagen: min={preds.min():.4f}  max={preds.max():.4f}  "
                f"mean={preds.mean():.4f}  std={preds.std():.4f}")

    # Adaptiver Threshold: wenn entry_threshold > max(preds), automatisch anpassen
    # Setze entry auf 75. Perzentil, exit auf 40. Perzentil der Vorhersagen
    p75 = float(np.percentile(preds, 75))
    p40 = float(np.percentile(preds, 40))
    if entry_threshold > preds.max():
        logger.warning(
            f"entry_threshold={entry_threshold:.3f} übersteigt max(preds)={preds.max():.4f}. "
            f"Setze automatisch auf 75. Perzentil: {p75:.4f}"
        )
        entry_threshold = p75
    if exit_threshold < preds.min() or exit_threshold >= entry_threshold:
        adj_exit = max(p40, preds.min() + 0.01)
        logger.warning(
            f"exit_threshold={exit_threshold:.3f} ungültig. "
            f"Setze automatisch auf 40. Perzentil: {adj_exit:.4f}"
        )
        exit_threshold = adj_exit

    logger.info(f"Aktive Thresholds — Entry: {entry_threshold:.4f}  Exit: {exit_threshold:.4f}")
    logger.info(f"Signale >= {entry_threshold:.4f}: {(preds >= entry_threshold).sum()} / {len(preds)}")

    # Rohpreise laden
    raw_candidates = [
        RAW_DIR / f"{data_ticker}_{timeframe}.parquet",
        RAW_DIR / f"{data_ticker.replace('_','.')}_{ timeframe}.parquet",
    ]
    price_file = next((p for p in raw_candidates if p.exists()), None)
    if price_file is None:
        raise FileNotFoundError(f"Rohpreise nicht gefunden für {data_ticker}")

    price_df   = pd.read_parquet(price_file)["close"]
    n_total    = len(price_df)
    test_start = int(n_total * 0.70) + int(n_total * 0.15) + seq_len
    price_test = price_df.iloc[test_start : test_start + len(preds)].values
    min_len    = min(len(price_test), len(preds))
    price_test = price_test[:min_len]
    preds      = preds[:min_len]

    bh = (price_test[-1] / price_test[0] - 1) * 100 if len(price_test) > 1 else 0

    results, trades = manual_backtest(
        price_test, preds, entry_threshold, exit_threshold,
        init_cash, fees, hold_days, stop_loss
    )

    logger.info("\n" + "=" * 55)
    logger.info(f"BACKTEST: {data_ticker} [{timeframe}]  "
                f"{'(Modell: ' + ticker + ')' if test_on else ''}")
    logger.info("=" * 55)
    for k, v in results.items():
        logger.info(f"  {k:<35}: {v}")
    logger.info(f"  {'Buy & Hold Return [%]':<35}: {round(bh, 2)}")
    logger.info("=" * 55)

    results["Buy & Hold Return [%]"] = round(bh, 2)
    csv_path = REPORT_DIR / f"backtest_{data_ticker}_{timeframe}_{mode}.csv"
    pd.DataFrame({k: [v] for k, v in results.items()}).to_csv(csv_path, index=False)
    logger.success(f"Report: {csv_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker",    default="BTC_USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--mode",      choices=["cls","reg"], default="cls")
    parser.add_argument("--cash",      type=float, default=10_000.0)
    parser.add_argument("--fees",      type=float, default=0.001)
    parser.add_argument("--test_on",   default=None,
                        help="Anderen Asset zum Testen (Out-of-Sample)")
    args = parser.parse_args()
    run_backtest(args.ticker, args.timeframe, args.mode, args.cash, args.fees, args.test_on)
