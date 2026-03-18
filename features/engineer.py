"""
features/engineer.py
─────────────────────
Feature-Engineering-Pipeline für einzelne Assets UND Multi-Asset-Training.

Aufruf (einzeln):
    python features/engineer.py --ticker AAPL --timeframe 1h

Aufruf (alle Assets aus asset_list.txt):
    python features/engineer.py --all --timeframe 1h --threshold 0.002
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import ta
import torch
from loguru import logger
from sklearn.preprocessing import RobustScaler

FEATURE_DIR = Path(__file__).parent / "processed"
RAW_DIR     = Path(__file__).parent.parent / "data" / "raw"
ASSET_LIST  = Path(__file__).parent.parent / "data" / "asset_list.txt"

# Defaults
HORIZON   : int   = 6
THRESHOLD : float = 0.002
SEQ_LEN   : int   = 48
SPLIT               = (0.70, 0.15, 0.15)


# ── Indikatoren ───────────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df    = df.copy()
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]

    df["rsi_14"]      = ta.momentum.RSIIndicator(close, window=14).rsi()
    macd              = ta.trend.MACD(close, window_fast=12, window_slow=26, window_sign=9)
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"]   = macd.macd_diff()
    bb                = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_width"]    = bb.bollinger_wband()
    df["bb_pct"]      = bb.bollinger_pband()
    for p in [9, 21, 50, 200]:
        df[f"ema_{p}"] = ta.trend.EMAIndicator(close, window=p).ema_indicator()
    df["atr_14"]       = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    df["volume_ema_20"]= ta.trend.EMAIndicator(vol, window=20).ema_indicator()
    df["log_return"]   = np.log(close / close.shift(1))
    df["body"]         = (df["close"] - df["open"]) / df["open"]
    df["upper_wick"]   = (df["high"] - df[["open","close"]].max(axis=1)) / df["open"]
    df["lower_wick"]   = (df[["open","close"]].min(axis=1) - df["low"]) / df["open"]
    return df


# ── Labels ────────────────────────────────────────────────────────────────────

def add_labels(df: pd.DataFrame, horizon: int, threshold: float) -> pd.DataFrame:
    df = df.copy()
    future          = df["close"].shift(-horizon)
    df["label_reg"] = (future - df["close"]) / df["close"]
    df["label_cls"] = (df["label_reg"] >= threshold).astype(int)
    return df


FEATURE_COLS = [
    "open","high","low","close","volume",
    "rsi_14","macd","macd_signal","macd_hist",
    "bb_width","bb_pct",
    "ema_9","ema_21","ema_50","ema_200",
    "atr_14","volume_ema_20",
    "log_return","body","upper_wick","lower_wick",
]


# ── Skalierung ────────────────────────────────────────────────────────────────

def scale_features(df, fit_scaler=True, scaler=None):
    if scaler is None:
        scaler = RobustScaler()
    data   = df[FEATURE_COLS].values
    scaled = scaler.fit_transform(data) if fit_scaler else scaler.transform(data)
    df_s   = pd.DataFrame(scaled, index=df.index, columns=FEATURE_COLS)
    df_s["label_cls"] = df["label_cls"].values
    df_s["label_reg"] = df["label_reg"].values
    return df_s, scaler


# ── Sequenzen ─────────────────────────────────────────────────────────────────

def make_sequences(df, seq_len, label_col="label_cls"):
    feat, label = df[FEATURE_COLS].values, df[label_col].values
    X, y = [], []
    for i in range(seq_len, len(df)):
        X.append(feat[i - seq_len : i])
        y.append(label[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ── Split ─────────────────────────────────────────────────────────────────────

def temporal_split(X, y, ratios=SPLIT):
    n  = len(X)
    i1 = int(n * ratios[0])
    i2 = int(n * (ratios[0] + ratios[1]))
    return {
        "train": (X[:i1],   y[:i1]),
        "val":   (X[i1:i2], y[i1:i2]),
        "test":  (X[i2:],   y[i2:]),
    }


# ── Einzelner Asset ───────────────────────────────────────────────────────────

def process_single(
    ticker:     str,
    timeframe:  str   = "1h",
    horizon:    int   = HORIZON,
    threshold:  float = THRESHOLD,
    seq_len:    int   = SEQ_LEN,
    label_mode: str   = "cls",
    save:       bool  = True,
) -> dict | None:
    """Verarbeitet einen einzelnen Asset. Gibt splits-Dict zurück."""
    fname = RAW_DIR / f"{ticker.replace('.','_')}_{timeframe}.parquet"
    if not fname.exists():
        logger.warning(f"  {ticker}: Datei nicht gefunden ({fname})")
        return None

    df = pd.read_parquet(fname)
    if len(df) < seq_len + 50:
        logger.warning(f"  {ticker}: Zu wenig Daten ({len(df)} Zeilen)")
        return None

    df = add_indicators(df)
    df = add_labels(df, horizon, threshold)
    df.dropna(inplace=True)

    if len(df) < seq_len + 20:
        logger.warning(f"  {ticker}: Nach dropna zu wenig Daten")
        return None

    label_col = "label_cls" if label_mode == "cls" else "label_reg"

    # Scaler nur auf Train-Daten fitten
    n_train = int(len(df) * SPLIT[0])
    df_train_s, scaler = scale_features(df.iloc[:n_train], fit_scaler=True)
    df_rest_s,  _      = scale_features(df.iloc[n_train:], fit_scaler=False, scaler=scaler)
    df_s = pd.concat([df_train_s, df_rest_s])

    X, y    = make_sequences(df_s, seq_len, label_col)
    splits  = temporal_split(X, y)

    if save:
        out_dir = FEATURE_DIR / f"{ticker.replace('.','_')}_{timeframe}"
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix  = f"{ticker.replace('.','_')}_{timeframe}_{label_mode}_"
        for name, (Xs, ys) in splits.items():
            torch.save(
                {"X": torch.from_numpy(Xs), "y": torch.from_numpy(ys)},
                out_dir / f"{prefix}{name}.pt",
            )
        import joblib
        joblib.dump(scaler, out_dir / f"{prefix}scaler.pkl")
        logger.success(f"  {ticker:<12} Train={splits['train'][0].shape[0]:5d}  "
                       f"Val={splits['val'][0].shape[0]:4d}  "
                       f"Test={splits['test'][0].shape[0]:4d}")

    return splits


# ── Multi-Asset: kombinierter Datensatz ───────────────────────────────────────

def build_combined_dataset(
    timeframe:  str   = "1h",
    horizon:    int   = HORIZON,
    threshold:  float = THRESHOLD,
    seq_len:    int   = SEQ_LEN,
    label_mode: str   = "cls",
    tickers:    list[str] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Verarbeitet alle Assets und kombiniert sie zu einem gemeinsamen Datensatz.
    Jeder Asset trägt seinen eigenen Train/Val/Test-Split bei.
    Der kombinierte Split wird als combined_<timeframe>_<mode>_*.pt gespeichert.
    """
    if tickers is None:
        tickers = _load_asset_list()

    all_splits: dict[str, list] = {"train": [], "val": [], "test": []}
    ok, fail = 0, 0

    logger.info(f"Multi-Asset Feature-Engineering: {len(tickers)} Assets")
    logger.info("─" * 55)

    for ticker in tickers:
        result = process_single(
            ticker, timeframe, horizon, threshold, seq_len, label_mode, save=True
        )
        if result is None:
            fail += 1
            continue
        for split_name in ["train", "val", "test"]:
            all_splits[split_name].append(result[split_name][0])   # X
            # y auch sammeln (separate Liste)
        ok += 1

    if ok == 0:
        raise RuntimeError("Kein einziger Asset konnte verarbeitet werden.")

    # Neu aufbauen mit X und y getrennt
    combined: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split_name in ["train", "val", "test"]:
        Xs, ys = [], []
        for ticker in tickers:
            fname = RAW_DIR / f"{ticker.replace('.','_')}_{timeframe}.parquet"
            if not fname.exists():
                continue
            out_dir = FEATURE_DIR / f"{ticker.replace('.','_')}_{timeframe}"
            prefix  = f"{ticker.replace('.','_')}_{timeframe}_{label_mode}_"
            pt_path = out_dir / f"{prefix}{split_name}.pt"
            if not pt_path.exists():
                continue
            data = torch.load(pt_path, map_location="cpu")
            Xs.append(data["X"].numpy())
            ys.append(data["y"].numpy())

        if not Xs:
            continue
        X_combined = np.concatenate(Xs, axis=0)
        y_combined = np.concatenate(ys, axis=0)

        # Shufflen (nur Train) damit Modell nicht Asset-Reihenfolge lernt
        if split_name == "train":
            idx = np.random.permutation(len(X_combined))
            X_combined = X_combined[idx]
            y_combined = y_combined[idx]

        combined[split_name] = (X_combined, y_combined)

        # Speichern
        out_dir = FEATURE_DIR / "combined"
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix  = f"combined_{timeframe}_{label_mode}_"
        torch.save(
            {"X": torch.from_numpy(X_combined), "y": torch.from_numpy(y_combined)},
            out_dir / f"{prefix}{split_name}.pt",
        )
        logger.info(f"  combined/{split_name}: X={X_combined.shape}  "
                    f"pos={y_combined.mean()*100:.1f}%")

    logger.success(f"Kombinierter Datensatz: {ok} Assets, {fail} fehlgeschlagen")
    return combined


def _load_asset_list() -> list[str]:
    tickers = []
    with open(ASSET_LIST) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                tickers.append(line)
    return tickers


# ── Pipeline-Einstieg ─────────────────────────────────────────────────────────

def run_pipeline(
    symbol:     str   = "BTC/USDT",
    timeframe:  str   = "1h",
    horizon:    int   = HORIZON,
    threshold:  float = THRESHOLD,
    seq_len:    int   = SEQ_LEN,
    label_mode: str   = "cls",
    multi:      bool  = False,
) -> dict:
    if multi:
        combined = build_combined_dataset(timeframe, horizon, threshold, seq_len, label_mode)
        # Für Kompatibilität mit trainer.py: combined als splits zurückgeben
        return {
            "splits":     combined,
            "n_features": len(FEATURE_COLS),
            "seq_len":    seq_len,
            "ticker":     "combined",
        }
    else:
        ticker = symbol.replace("/", "_")
        result = process_single(ticker, timeframe, horizon, threshold, seq_len, label_mode, save=True)
        if result is None:
            raise FileNotFoundError(f"Keine Daten für {ticker}")
        return {
            "splits":     result,
            "n_features": len(FEATURE_COLS),
            "seq_len":    seq_len,
            "ticker":     ticker,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",     default="BTC/USDT")
    parser.add_argument("--ticker",     default=None,       help="Direkter Ticker z.B. AAPL")
    parser.add_argument("--all",        action="store_true",help="Alle Assets aus asset_list.txt")
    parser.add_argument("--timeframe",  default="1h")
    parser.add_argument("--horizon",    type=int,   default=HORIZON)
    parser.add_argument("--threshold",  type=float, default=THRESHOLD)
    parser.add_argument("--seq_len",    type=int,   default=SEQ_LEN)
    parser.add_argument("--label_mode", choices=["cls","reg"], default="cls")
    args = parser.parse_args()

    if args.all:
        build_combined_dataset(args.timeframe, args.horizon, args.threshold,
                               args.seq_len, args.label_mode)
    elif args.ticker:
        process_single(args.ticker, args.timeframe, args.horizon,
                       args.threshold, args.seq_len, args.label_mode, save=True)
    else:
        run_pipeline(args.symbol, args.timeframe, args.horizon,
                     args.threshold, args.seq_len, args.label_mode)
    logger.success("Feature-Pipeline abgeschlossen.")
