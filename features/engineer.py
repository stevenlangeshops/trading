"""
features/engineer.py
──────────────────────
Feature Engineering Pipeline für den v2 Multi-Asset LSTM Trading Bot.

Normalisierungs-Modi:
    cross_sectional (Referenz):
        Z-Score pro Tag über ALLE Assets.  Ein Score von +1.0 bedeutet
        "höchster Wert im gesamten Universum heute".  Nachteil: Mega-Caps
        (NVDA, AAPL) dominieren – das Modell lernt hauptsächlich Sektor-Momentum
        statt echten relativen Stärke-Unterschieden.

    sector_neutral (Produktion):
        Z-Score pro Tag UND pro GICS-Sektor.  Ein Score von +1.0 bedeutet
        "höchster Wert *innerhalb des eigenen Sektors* heute".  Energy-Titel
        werden gegen andere Energy-Titel gemessen, Tech gegen Tech.  Das Modell
        ist gezwungen, echtes Cross-Sectional Alpha zu lernen.

Target (Forward Return) bleibt in beiden Modi unverändert als Cross-Sectional
Rank-Loss über das gesamte Universum – nur die Input-Features werden normalisiert.

Sektor-Metadaten:  features/sector_map.json  (GICS, manuell kuratiert)
Update-Script:     features/build_sector_map.py  (via yfinance)
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

# Sektor-Map liegt im selben Verzeichnis wie diese Datei
_SECTOR_MAP_PATH = Path(__file__).parent / "sector_map.json"

# ── Feature-Spalten (Reihenfolge ist Teil des Modell-Interfaces) ──────────────
FEATURE_COLS = [
    # Trend
    "sma_ratio_20",    # Close / SMA(20)
    "sma_ratio_50",    # Close / SMA(50)
    "sma_ratio_200",   # Close / SMA(200)
    "ema_ratio_12",    # Close / EMA(12)
    "macd_diff",       # MACD-Histogramm
    # Momentum
    "rsi_14",          # RSI(14), skaliert auf [0, 1]
    "roc_5",           # Rate of Change 5 Tage
    "roc_21",          # Rate of Change 21 Tage
    "stoch_k",         # Stochastik %K, skaliert auf [0, 1]
    # Volatilität
    "atr_ratio",       # ATR(14) / Close
    "bb_width",        # Bollinger-Band-Breite / Close
    "bb_pos",          # Position innerhalb Bollinger Bands [0, 1]
    # Volumen
    "volume_ratio_20", # Volume / SMA-Volume(20)
    "obv_diff",        # OBV Tagesveränderung (normiert, gecappt)
    # Preis-Struktur
    "high_low_ratio",  # (High − Low) / Close
    "ret_1d",          # 1-Tage-Return
    "ret_5d",          # 5-Tage-Return
    "ret_21d",         # 21-Tage-Return
]


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Berechnet technische Indikatoren für ein einzelnes Asset.

    Alle Indikatoren sind zeitreihen-intern (kein Look-Ahead über Assets).
    Die Z-Score-Normalisierung erfolgt separat in ``build_panel()``.

    Args:
        df: OHLCV-DataFrame mit Spalten ``open``, ``high``, ``low``,
            ``close``, ``volume`` und DatetimeIndex.

    Returns:
        DataFrame mit Spalten gemäß ``FEATURE_COLS``, gleicher Index wie ``df``.
        Zeilen mit fehlenden Werten (Warm-up-Phase) enthalten NaN.
    """
    c  = df["close"]
    h  = df["high"]
    lo = df["low"]
    v  = df["volume"]

    out = pd.DataFrame(index=df.index)

    # Trend
    out["sma_ratio_20"]  = c / c.rolling(20).mean()
    out["sma_ratio_50"]  = c / c.rolling(50).mean()
    out["sma_ratio_200"] = c / c.rolling(200).mean()
    out["ema_ratio_12"]  = c / ta.trend.EMAIndicator(c, window=12).ema_indicator()
    out["macd_diff"]     = ta.trend.MACD(c).macd_diff()

    # Momentum
    out["rsi_14"]  = ta.momentum.RSIIndicator(c, window=14).rsi() / 100.0
    out["roc_5"]   = c.pct_change(5)
    out["roc_21"]  = c.pct_change(21)
    out["stoch_k"] = ta.momentum.StochasticOscillator(
        h, lo, c, window=14).stoch() / 100.0

    # Volatilität
    atr = ta.volatility.AverageTrueRange(h, lo, c, window=14).average_true_range()
    out["atr_ratio"] = atr / c
    bb   = ta.volatility.BollingerBands(c, window=20)
    bb_w = bb.bollinger_hband() - bb.bollinger_lband()
    out["bb_width"] = bb_w / c
    out["bb_pos"]   = (c - bb.bollinger_lband()) / (bb_w + 1e-9)

    # Volumen
    out["volume_ratio_20"] = v / (v.rolling(20).mean() + 1e-9)
    obv = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
    out["obv_diff"] = obv.pct_change().clip(-1, 1)

    # Preis-Struktur
    out["high_low_ratio"] = (h - lo) / (c + 1e-9)
    out["ret_1d"]  = c.pct_change(1)
    out["ret_5d"]  = c.pct_change(5)
    out["ret_21d"] = c.pct_change(21)

    return out[FEATURE_COLS]


def compute_forward_return(df: pd.DataFrame, horizon: int = 11) -> pd.Series:
    """Berechnet den Forward Return als Modell-Target.

    ``ret[t] = (close[t + horizon] / close[t]) − 1``

    Der Wert bei Datum ``t`` spiegelt die Rendite der nächsten ``horizon``
    Handelstage wider.  Die letzten ``horizon`` Zeilen enthalten NaN.

    Args:
        df:      OHLCV-DataFrame mit ``close``-Spalte.
        horizon: Vorhersage-Horizont in Handelstagen.

    Returns:
        Series mit gleichem Index wie ``df``.
    """
    return df["close"].pct_change(horizon).shift(-horizon)


# ── Sektor-Map ────────────────────────────────────────────────────────────────

def load_sector_map(path: Optional[Path] = None) -> dict[str, str]:
    """Lädt die GICS-Sektor-Zuordnung aus ``sector_map.json``.

    Die Datei ordnet jeden Ticker einem GICS-Sektor zu (z.B.
    ``"AAPL": "Information Technology"``).  Meta-Einträge mit
    ``_``-Präfix werden ignoriert.

    Args:
        path: Optionaler Pfad zur JSON-Datei.  Standard: ``features/sector_map.json``
              neben dieser Datei.

    Returns:
        Dict ``{ticker → sektor}``.  Leer wenn Datei nicht gefunden –
        ``build_panel()`` fällt dann automatisch auf Cross-Sectional zurück.
    """
    p = Path(path) if path else _SECTOR_MAP_PATH
    if not p.exists():
        return {}
    with open(p) as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


# ── Normalisierung ────────────────────────────────────────────────────────────

def cross_sectional_zscore(
    panel: pd.DataFrame,
    min_assets: int = 5,
) -> pd.DataFrame:
    """Klassische Cross-Sectional Z-Score Normalisierung (Referenz-Modus).

    Normalisiert jeden Feature-Wert täglich über **alle** Assets im Universum.
    Dient als Vergleichsbasis zur sektor-neutralen Variante.

    Args:
        panel:      MultiIndex-DataFrame ``(date, asset) × FEATURE_COLS``.
        min_assets: Mindest-Anzahl Assets pro Tag (Tage mit weniger werden
                    übersprungen und behalten ihre Rohwerte).

    Returns:
        MultiIndex-DataFrame gleicher Struktur, z-Score normalisiert und
        auf ±4 Standardabweichungen gecappt.
    """
    result = panel.copy()
    for date, group in panel.groupby(level="date"):
        if len(group) < min_assets:
            continue
        mu    = group.mean()
        sigma = group.std().replace(0, 1)
        result.loc[date] = (group - mu) / sigma
    return result.clip(-4, 4)


def sector_neutral_zscore(
    panel:               pd.DataFrame,
    sector_map:          dict[str, str],
    min_per_sector:      int = 3,
    fallback_min_assets: int = 5,
) -> pd.DataFrame:
    """Sektor-neutrale Z-Score Normalisierung (Produktions-Modus).

    Normalisiert jeden Feature-Wert täglich **innerhalb des eigenen GICS-Sektors**.
    Ein RSI-Score von +1.5 bedeutet damit "höchster RSI im Tech-Sektor" statt
    "höchster RSI im gesamten Markt".  Das Modell lernt relative Stärke zwischen
    vergleichbaren Unternehmen, nicht Sektor-Momentum.

    Fallback-Logik:
        Wenn ein Sektor weniger als ``min_per_sector`` Assets aufweist
        (z.B. sehr kleine Sektoren oder unbekannte Ticker → Kategorie ``Unknown``),
        wird der globale Tages-z-Score als Fallback verwendet, um instabile
        Normalisierungen bei kleinen Stichproben zu vermeiden.

    Args:
        panel:               MultiIndex-DataFrame ``(date, asset) × FEATURE_COLS``.
        sector_map:          Dict ``{ticker → GICS-Sektor}`` aus ``load_sector_map()``.
        min_per_sector:      Mindest-Assets pro (Datum, Sektor) für Sektor-Normalisierung.
        fallback_min_assets: Mindest-Assets pro Tag für globalen Fallback.

    Returns:
        MultiIndex-DataFrame gleicher Struktur, sektor-neutral z-Score normalisiert
        und auf ±4 Standardabweichungen gecappt.

    Note:
        Der Forward Return (Target) wird in diesem Schritt **nicht** verändert –
        das Modell lernt weiterhin über den Rank-Loss gegen das gesamte Universum.
    """
    feat_cols = list(panel.columns)

    # MultiIndex auflösen → flaches DataFrame für vektorisierten groupby
    flat = panel.reset_index()
    flat["_sector"] = flat["asset"].map(sector_map).fillna("Unknown")

    # Sektor-Level: Mittelwert und Std pro (Datum, Sektor)
    grp_sec  = flat.groupby(["date", "_sector"])
    sec_mean = grp_sec[feat_cols].transform("mean")
    sec_std  = grp_sec[feat_cols].transform("std").fillna(0)
    sec_std  = sec_std.where(sec_std > 0, 1.0)
    sec_cnt  = grp_sec[feat_cols[0]].transform("count")

    # Globales Tages-Level als Fallback
    grp_day   = flat.groupby("date")
    glob_mean = grp_day[feat_cols].transform("mean")
    glob_std  = grp_day[feat_cols].transform("std").fillna(0)
    glob_std  = glob_std.where(glob_std > 0, 1.0)
    day_cnt   = grp_day[feat_cols[0]].transform("count")

    # Sektor-Normalisierung wenn Sektor groß genug, sonst global
    use_sec = ((sec_cnt >= min_per_sector) & (day_cnt >= fallback_min_assets)).values
    mu  = np.where(use_sec[:, None], sec_mean.values,  glob_mean.values)
    std = np.where(use_sec[:, None], sec_std.values,   glob_std.values)

    z_values = (flat[feat_cols].values - mu) / std

    result = pd.DataFrame(z_values, columns=feat_cols, index=panel.index)
    return result.clip(-4, 4)


# ── Haupt-Pipeline ────────────────────────────────────────────────────────────

def build_panel(
    timeframe:       str            = "1d",
    horizon:         int            = 11,
    min_rows:        int            = 300,
    asset_list:      Optional[list] = None,
    sector_neutral:  bool           = True,
    sector_map_path: Optional[str]  = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Baut den vollständigen Feature-Panel-Datensatz für das Training.

    Pipeline:
        1. Lädt alle Parquet-Dateien aus ``RAW_DIR``.
        2. Berechnet technische Indikatoren pro Asset (zeitreihen-intern,
           kein Look-Ahead zwischen Assets).
        3. Z-Score Normalisierung: sektor-neutral (Standard) oder cross-sectional.
        4. Forward Return als Target (immer cross-sectional Rank-Loss).

    Args:
        timeframe:       Parquet-Datei-Suffix, z.B. ``"1d"`` für Tagesdaten.
        horizon:         Vorhersage-Horizont in Handelstagen für den Forward Return.
        min_rows:        Mindest-Datenpunkte pro Asset (kürzere werden übersprungen).
        asset_list:      Optionale Whitelist von Tickern.  ``None`` = alle.
        sector_neutral:  ``True`` (Standard) = sektor-neutrale Z-Score Normalisierung.
                         ``False`` = klassischer Cross-Sectional z-Score (Referenz).
        sector_map_path: Optionaler Pfad zu ``sector_map.json``.  Wenn nicht angegeben,
                         wird ``features/sector_map.json`` neben dieser Datei verwendet.

    Returns:
        Tuple ``(features, targets)``:
            - ``features``: MultiIndex-DataFrame ``(date, asset) × FEATURE_COLS``
            - ``targets``:  MultiIndex-Series ``(date, asset) → forward_return``

    Raises:
        FileNotFoundError: Wenn keine Parquet-Dateien in ``RAW_DIR`` gefunden werden.
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
        df.index   = pd.to_datetime(df.index)
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
            logger.warning(f"  {ticker}: übersprungen – {e}")
            skipped.append(ticker)

    if skipped:
        logger.warning(f"Übersprungen ({len(skipped)}): {', '.join(skipped[:10])}")

    logger.info(f"Assets geladen: {len(all_features)}")

    # MultiIndex Panel aufbauen: (date, asset) × Features
    features_panel = pd.concat(all_features, names=["asset", "date"]).swaplevel().sort_index()
    targets_panel  = pd.concat(all_targets,  names=["asset", "date"]).swaplevel().sort_index()
    targets_panel.name = "forward_return"

    # Z-Score Normalisierung
    if sector_neutral:
        sector_map = load_sector_map(sector_map_path)
        if not sector_map:
            logger.warning(
                "sector_map.json nicht gefunden – Fallback auf Cross-Sectional z-Score."
            )
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

    logger.info(
        f"Panel: {len(features_panel)} Zeilen  "
        f"{features_panel.index.get_level_values('date').nunique()} Tage  "
        f"{features_panel.index.get_level_values('asset').nunique()} Assets"
    )
    return features_panel, targets_panel
