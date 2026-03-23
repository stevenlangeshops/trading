"""
features/engineer.py
─────────────────────
Cross-Sectional Feature Engineering für Multi-Asset LSTM.

Kernidee:
  1. Pro Asset: technische Indikatoren berechnen (zeitreihen-intern)
  2. Pro Tag: alle Features über alle Assets hinweg z-Score normalisieren
             (Cross-Sectional Normalization)
  3. Target: 11-Tage Forward Return (Regression, nicht binäre Klassifikation)

Warum Cross-Sectional Normalization?
  - Ein SMA-Ratio von 1.05 bedeutet bei Apple etwas anderes als bei Nestlé
  - Nach CS-Norm bedeutet +1.0 "dieses Asset hat den höchsten Wert heute"
  - Das Modell lernt relative Stärke, nicht absolute Werte
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import ta

warnings.filterwarnings("ignore")

RAW_DIR     = Path("data/raw")
FEATURE_DIR = Path("features/processed")


# ── Technische Indikatoren (pro Asset, zeitreihen-intern) ─────────────────────

FEATURE_COLS = [
    # Trend
    "sma_ratio_20",    # Close / SMA20
    "sma_ratio_50",    # Close / SMA50
    "sma_ratio_200",   # Close / SMA200
    "ema_ratio_12",    # Close / EMA12
    "macd_diff",       # MACD Histogramm
    # Momentum
    "rsi_14",          # RSI 14
    "roc_5",           # Rate of Change 5T
    "roc_21",          # Rate of Change 21T
    "stoch_k",         # Stochastic %K
    # Volatilität
    "atr_ratio",       # ATR14 / Close (normiert)
    "bb_width",        # Bollinger Band Width
    "bb_pos",          # Position innerhalb Bollinger Bands
    # Volumen
    "volume_ratio_20", # Volume / SMA-Volume-20
    "obv_diff",        # OBV tägliche Änderung (normiert)
    # Preis-Struktur
    "high_low_ratio",  # (High-Low) / Close
    "ret_1d",          # 1-Tage Return
    "ret_5d",          # 5-Tage Return
    "ret_21d",         # 21-Tage Return
]


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet technische Indikatoren für ein einzelnes Asset.
    Input:  OHLCV DataFrame (index=Date)
    Output: DataFrame mit FEATURE_COLS Spalten
    """
    c = df["close"]
    h = df["high"]
    lo = df["low"]
    v  = df["volume"]

    out = pd.DataFrame(index=df.index)

    # ── Trend ────────────────────────────────────────────────────────────────
    out["sma_ratio_20"]  = c / c.rolling(20).mean()
    out["sma_ratio_50"]  = c / c.rolling(50).mean()
    out["sma_ratio_200"] = c / c.rolling(200).mean()
    out["ema_ratio_12"]  = c / ta.trend.EMAIndicator(c, window=12).ema_indicator()

    macd = ta.trend.MACD(c)
    out["macd_diff"] = macd.macd_diff()

    # ── Momentum ─────────────────────────────────────────────────────────────
    out["rsi_14"]  = ta.momentum.RSIIndicator(c, window=14).rsi() / 100.0
    out["roc_5"]   = c.pct_change(5)
    out["roc_21"]  = c.pct_change(21)
    out["stoch_k"] = ta.momentum.StochasticOscillator(
        h, lo, c, window=14).stoch() / 100.0

    # ── Volatilität ───────────────────────────────────────────────────────────
    atr = ta.volatility.AverageTrueRange(h, lo, c, window=14).average_true_range()
    out["atr_ratio"] = atr / c

    bb = ta.volatility.BollingerBands(c, window=20)
    bb_w = bb.bollinger_hband() - bb.bollinger_lband()
    out["bb_width"] = bb_w / c
    out["bb_pos"]   = (c - bb.bollinger_lband()) / (bb_w + 1e-9)

    # ── Volumen ───────────────────────────────────────────────────────────────
    vol_sma = v.rolling(20).mean()
    out["volume_ratio_20"] = v / (vol_sma + 1e-9)

    obv = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
    out["obv_diff"] = obv.pct_change().clip(-1, 1)

    # ── Preis-Struktur ────────────────────────────────────────────────────────
    out["high_low_ratio"] = (h - lo) / (c + 1e-9)
    out["ret_1d"]  = c.pct_change(1)
    out["ret_5d"]  = c.pct_change(5)
    out["ret_21d"] = c.pct_change(21)

    return out[FEATURE_COLS]


def compute_forward_return(df: pd.DataFrame, horizon: int = 11) -> pd.Series:
    """
    Berechnet den Forward Return für jede Zeile.
    ret[t] = (close[t+horizon] / close[t]) - 1
    """
    return df["close"].pct_change(horizon).shift(-horizon)


# ── Cross-Sectional Normalization ─────────────────────────────────────────────

def cross_sectional_zscore(
    panel: pd.DataFrame,
    min_assets: int = 5,
) -> pd.DataFrame:
    """
    Normalisiert Features täglich über alle Assets (Cross-Sectional z-Score).

    Input:  MultiIndex DataFrame  (Date, Asset) × Features
    Output: Gleiche Struktur, aber pro Tag z-Score normalisiert

    Warum:
      - SMA-Ratio 1.05 bedeutet bei AAPL etwas anderes als bei DBK.DE
      - Nach CS-Norm bedeutet +1.5 = "heute 1.5 Std-Abw. über Durchschnitt aller Assets"
      - Modell lernt relative Stärke statt absolute Niveaus
    """
    result = panel.copy()

    for date, group in panel.groupby(level="date"):
        if len(group) < min_assets:
            continue
        mu    = group.mean()
        sigma = group.std().replace(0, 1)  # Division durch 0 verhindern
        result.loc[date] = (group - mu) / sigma

    # Extreme Ausreißer cappen (±4 Std-Abw.)
    result = result.clip(-4, 4)
    return result


# ── Haupt-Pipeline ────────────────────────────────────────────────────────────

def build_panel(
    timeframe:  str   = "1d",
    horizon:    int   = 11,
    min_rows:   int   = 300,
    asset_list: Optional[list] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Baut den kompletten Panel-Datensatz:
      1. Lädt alle Parquet-Dateien
      2. Berechnet Indikatoren pro Asset
      3. Cross-Sectional z-Score Normalisierung
      4. Forward Returns als Target

    Returns:
        features : MultiIndex DataFrame (date, asset) × FEATURE_COLS
        targets  : MultiIndex Series    (date, asset) → forward_return
    """
    from loguru import logger

    raw_files = sorted(RAW_DIR.glob(f"*_{timeframe}.parquet"))
    if not raw_files:
        raise FileNotFoundError(f"Keine Parquet-Dateien in {RAW_DIR}")

    all_features = {}
    all_targets  = {}
    skipped      = []

    tickers = asset_list or [f.stem.replace(f"_{timeframe}", "") for f in raw_files]

    for fpath in raw_files:
        ticker = fpath.stem.replace(f"_{timeframe}", "")
        if asset_list and ticker not in asset_list:
            continue

        df = pd.read_parquet(fpath)
        df.index = pd.to_datetime(df.index)
        df.columns = [c.lower() for c in df.columns]

        if len(df) < min_rows:
            skipped.append(ticker)
            continue

        try:
            feats  = compute_indicators(df)
            target = compute_forward_return(df, horizon)

            # Gemeinsamen Index (beide ohne NaN)
            valid = feats.notna().all(axis=1) & target.notna()
            feats  = feats[valid]
            target = target[valid]

            if len(feats) < 200:
                skipped.append(ticker)
                continue

            all_features[ticker] = feats
            all_targets[ticker]  = target

        except Exception as e:
            logger.warning(f"  {ticker}: Fehler — {e}")
            skipped.append(ticker)
            continue

    if skipped:
        logger.warning(f"Übersprungen ({len(skipped)}): {', '.join(skipped[:10])}")

    logger.info(f"Assets geladen: {len(all_features)}")

    # ── MultiIndex Panel aufbauen ─────────────────────────────────────────────
    features_panel = pd.concat(all_features, names=["asset", "date"])
    features_panel = features_panel.swaplevel().sort_index()

    targets_panel  = pd.concat(all_targets,  names=["asset", "date"])
    targets_panel  = targets_panel.swaplevel().sort_index()
    targets_panel.name = "forward_return"

    # ── Cross-Sectional z-Score ───────────────────────────────────────────────
    logger.info("Cross-Sectional z-Score Normalisierung...")
    features_panel = cross_sectional_zscore(features_panel)

    logger.info(f"Panel: {len(features_panel)} Zeilen  "
                f"{features_panel.index.get_level_values('date').nunique()} Tage  "
                f"{features_panel.index.get_level_values('asset').nunique()} Assets")

    return features_panel, targets_panel
