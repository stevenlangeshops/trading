"""
features/engineer.py
─────────────────────
Cross-Sectional Feature Engineering für Multi-Asset LSTM.

Normalisierungs-Modi:
  cross_sectional (Standard):
    Z-Score pro Tag über ALLE Assets.
    +1.0 = "höchster Wert im gesamten Universum heute".
    Nachteil: Mega-Caps (NVDA, AAPL) dominieren – Modell lernt hauptsächlich
    Momentum innerhalb der größten Titel.

  sector_neutral (neu):
    Z-Score pro Tag UND pro GICS-Sektor.
    +1.0 = "höchster Wert innerhalb des eigenen Sektors heute".
    Vorteil: Äpfel mit Äpfeln vergleichen – ein Energy-RSI wird gegen andere
    Energy-Titel gemessen, nicht gegen Tech.  Das Modell ist gezwungen, echtes
    Cross-Sectional Alpha zu lernen, nicht nur Sektor-Momentum.

Target (Forward Return) bleibt in beiden Modi unverändert als Cross-Sectional
Rank-Loss über das gesamte Universum – nur die Inputs werden normalisiert.

Sektor-Metadaten: features/sector_map.json (GICS, manuell kuratiert).
Update-Script:    features/build_sector_map.py  (via yfinance).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import ta

warnings.filterwarnings("ignore")

RAW_DIR     = Path("data/raw")
FEATURE_DIR = Path("features/processed")

# Sektor-Map liegt neben dieser Datei
_SECTOR_MAP_PATH = Path(__file__).parent / "sector_map.json"


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


# ── Sektor-Map laden ─────────────────────────────────────────────────────────

def load_sector_map(path: Optional[Path] = None) -> dict[str, str]:
    """
    Lädt die GICS-Sektor-Zuordnung aus sector_map.json.

    Gibt leeres Dict zurück wenn die Datei fehlt (Fallback auf CS-Normalisierung).
    Meta-Keys (mit "_"-Präfix) werden gefiltert.
    """
    p = Path(path) if path else _SECTOR_MAP_PATH
    if not p.exists():
        return {}
    with open(p) as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


# ── Cross-Sectional Normalization ─────────────────────────────────────────────

def cross_sectional_zscore(
    panel: pd.DataFrame,
    min_assets: int = 5,
) -> pd.DataFrame:
    """
    Normalisiert Features täglich über ALLE Assets (klassischer CS z-Score).

    Input:  MultiIndex DataFrame  (date, asset) × Features
    Output: Gleiche Struktur, aber pro Tag z-Score normalisiert.

    Nachteil: Mega-Caps mit extremen Momentum-Werten dominieren den z-Score –
    das Modell lernt hauptsächlich, wer am stärksten im GESAMTEN Markt ist.
    """
    result = panel.copy()

    for date, group in panel.groupby(level="date"):
        if len(group) < min_assets:
            continue
        mu    = group.mean()
        sigma = group.std().replace(0, 1)   # Division durch 0 verhindern
        result.loc[date] = (group - mu) / sigma

    result = result.clip(-4, 4)
    return result


# ── Sector-Neutral Normalization (neu) ────────────────────────────────────────

def sector_neutral_zscore(
    panel:               pd.DataFrame,
    sector_map:          dict[str, str],
    min_per_sector:      int = 3,
    fallback_min_assets: int = 5,
) -> pd.DataFrame:
    """
    Sektor-neutrale Z-Score Normalisierung: pro Tag UND pro GICS-Sektor.

    Idee:
      +1.5 beim RSI bedeutet nicht mehr "höchster RSI im gesamten Markt",
      sondern "höchster RSI innerhalb des Tech-Sektors".  Damit werden
      Äpfel (Energy-Stocks) nicht mehr mit Birnen (IT-Wachstumstitel) verglichen.

    Fallback:
      Wenn ein Sektor < min_per_sector Assets hat (z.B. kleiner Nischen-Sektor
      oder unbekannter Ticker → 'Unknown'), wird der globale Tages-z-Score
      als Fallback verwendet.  Das verhindert instabile Einticker-Normalisierungen.

    Target (Forward Return) bleibt UNVERÄNDERT – nur die Inputs werden normalisiert.

    Parameters
    ----------
    panel          : MultiIndex DataFrame (date, asset) × FEATURE_COLS
    sector_map     : dict {ticker → GICS-Sektor}  (aus load_sector_map())
    min_per_sector : Mindest-Assets pro Sektor für Sektor-Normalisierung
    fallback_min_assets : Mindest-Assets pro Tag für globalen Fallback

    Returns
    -------
    MultiIndex DataFrame (date, asset) × FEATURE_COLS, sektor-neutral z-Score,
    gecappt auf ±4 Std-Abw.
    """
    feat_cols = list(panel.columns)

    # ── Flatten: MultiIndex → flaches DataFrame mit integer Index ────────────
    flat = panel.reset_index()           # Spalten: date, asset, feat1, feat2, ...
    flat["_sector"] = flat["asset"].map(sector_map).fillna("Unknown")

    # ── Per-(date, sector) Stats via vectorized groupby.transform ─────────────
    grp_sec  = flat.groupby(["date", "_sector"])
    sec_mean = grp_sec[feat_cols].transform("mean")
    sec_std  = grp_sec[feat_cols].transform("std").fillna(0)
    sec_std  = sec_std.where(sec_std > 0, 1.0)       # std=0 → 1 (kein Divide-by-0)
    sec_cnt  = grp_sec[feat_cols[0]].transform("count")

    # ── Per-date Stats als globaler Fallback ──────────────────────────────────
    grp_day   = flat.groupby("date")
    glob_mean = grp_day[feat_cols].transform("mean")
    glob_std  = grp_day[feat_cols].transform("std").fillna(0)
    glob_std  = glob_std.where(glob_std > 0, 1.0)
    day_cnt   = grp_day[feat_cols[0]].transform("count")

    # ── Auswahl: Sektor wenn >= min_per_sector, sonst global ─────────────────
    #    use_sec: (N,1) boolean → broadcast über alle Features
    use_sec = ((sec_cnt >= min_per_sector) & (day_cnt >= fallback_min_assets)).values

    mu  = np.where(use_sec[:, None], sec_mean.values,  glob_mean.values)
    std = np.where(use_sec[:, None], sec_std.values,   glob_std.values)

    z_values = (flat[feat_cols].values - mu) / std

    # ── MultiIndex wiederherstellen ───────────────────────────────────────────
    result = pd.DataFrame(z_values, columns=feat_cols, index=panel.index)
    return result.clip(-4, 4)


# ── Haupt-Pipeline ────────────────────────────────────────────────────────────

def build_panel(
    timeframe:       str   = "1d",
    horizon:         int   = 11,
    min_rows:        int   = 300,
    asset_list:      Optional[list] = None,
    sector_neutral:  bool  = False,
    sector_map_path: Optional[str]  = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Baut den kompletten Panel-Datensatz:
      1. Lädt alle Parquet-Dateien
      2. Berechnet technische Indikatoren pro Asset (zeitreihen-intern)
      3. Z-Score Normalisierung (cross-sectional ODER sektor-neutral)
      4. Forward Returns als Target (immer cross-sectional Rank-Loss)

    Parameters
    ----------
    sector_neutral : bool (default False)
        False → klassischer Cross-Sectional z-Score über alle Assets pro Tag.
        True  → Sektor-neutraler z-Score (pro Tag AND pro GICS-Sektor).
                Erfordert features/sector_map.json.
    sector_map_path : str, optional
        Pfad zur sector_map.json.  Standard: features/sector_map.json.

    Returns
    -------
    features : MultiIndex DataFrame (date, asset) × FEATURE_COLS
    targets  : MultiIndex Series    (date, asset) → forward_return
    """
    from loguru import logger

    raw_files = sorted(RAW_DIR.glob(f"*_{timeframe}.parquet"))
    if not raw_files:
        raise FileNotFoundError(f"Keine Parquet-Dateien in {RAW_DIR}")

    all_features: dict = {}
    all_targets:  dict = {}
    skipped:      list = []

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

            valid  = feats.notna().all(axis=1) & target.notna()
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

    # ── Z-Score Normalisierung ────────────────────────────────────────────────
    if sector_neutral:
        sector_map = load_sector_map(sector_map_path)
        if not sector_map:
            logger.warning(
                "sector_map.json nicht gefunden – Fallback auf Cross-Sectional z-Score."
            )
            logger.info("Cross-Sectional z-Score Normalisierung...")
            features_panel = cross_sectional_zscore(features_panel)
        else:
            n_mapped   = sum(1 for t in all_features if t in sector_map)
            n_unmapped = len(all_features) - n_mapped
            logger.info(
                f"Sektor-Neutrale z-Score Normalisierung "
                f"({n_mapped} Assets gemappt, {n_unmapped} → 'Unknown')..."
            )
            features_panel = sector_neutral_zscore(features_panel, sector_map)
    else:
        logger.info("Cross-Sectional z-Score Normalisierung...")
        features_panel = cross_sectional_zscore(features_panel)

    logger.info(f"Panel: {len(features_panel)} Zeilen  "
                f"{features_panel.index.get_level_values('date').nunique()} Tage  "
                f"{features_panel.index.get_level_values('asset').nunique()} Assets")

    return features_panel, targets_panel
