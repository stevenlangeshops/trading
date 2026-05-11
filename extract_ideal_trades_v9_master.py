"""
extract_ideal_trades_v9_master.py
====================================================================================
Master Refactoring  |  Target Quality + Lead-Lag Event Study  |  v9.2

Drei Verbesserungen gegenüber v9.0:

1. TARGET QUALITY statt Velocity:
   TQ = Return_pct / max(IntraWave_MaxDD_pct, 0.5)
   → Misst nicht nur Renditegeschwindigkeit, sondern auch Glattheit der Welle.
     Ein +30%-Trade mit 1% IntraDD ist hochwertiger als einer mit 4.9% IntraDD.
     Das Minimum von 0.5% verhindert Division-by-Zero bei perfekten Wellen.

2. TIME-WINDOW FEATURE ENGINEERING (T-5 bis T+5):
   Anstatt Features nur an Tag 0 zu messen, werden sie an 11 Tagen um den
   Wellentiefpunkt (T=0) herum erfasst. Namensschema: feat_rsi14_tm5 ... tp5
   → Liefert die Lead-Lag-Struktur: Welche Indikatoren BEVOR der Tiefpunkt
     eintritt, sind prädiktiv? Ab wann (nach T0) verbessern sich die Signale?
   ► T-5 bis T-1: rückwärtig (kein Look-Ahead, für echte Trading-Signale nutzbar)
   ► T0:          exakter Tiefpunkt
   ► T+1 bis T+5: vorwärtig (rein akademisch, für Prozessverständnis)

3. SPEARMAN 2D-MATRIX:
   Zeilen = Basis-Features, Spalten = T-5...T+5
   → Gibt an einem Blick, welche Features & welcher Zeitversatz am prädiktivsten sind.
   → Filter: Features mit max|r| < 0.05 über alle Zeitschritte werden ausgeblendet.

Outputs:
  ideal_trades_v9_master.csv  – Vollständiger Trainings-Datensatz
  Console                     – 2D Lead-Lag Matrix + Bucket-Analyse

Verwendung:
  python extract_ideal_trades_v9_master.py
  python extract_ideal_trades_v9_master.py --years 7 --min-profit 0.15 --max-pullback 0.05
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_here = Path(__file__).parent
sys.path.insert(0, str(_here))

from backtest_v6 import _load_tickers, _atr

# ── Konstanten ────────────────────────────────────────────────────────────────
DEFAULT_YEARS       = 7.0
DEFAULT_MIN_PROFIT  = 0.15
DEFAULT_MAX_PB      = 0.05
MIN_WAVE_DAYS       = 5
MIN_DATA_ROWS       = 260
MIN_WAVES_CORR      = 30       # Mindest-n für Korrelationsberechnung
MATRIX_FILTER_R     = 0.05     # Features unter diesem |r| werden ausgeblendet
_RAW_DIR   = _here / "data" / "raw"
_OUT_CSV   = _here / "ideal_trades_v9_master.csv"

# Basis-Features (8) – werden an jedem Zeitschritt berechnet
BASE_FEATURES = [
    "atr_pct",
    "bb_width",
    "dist_sma200",
    "rsi14",
    "vol_spike",
    "dist_sma50",
    "dist_sma20",
    "macd_hist_norm",
]

# Zeitfenster: T-5 bis T+5 (11 Schritte, in Handelstagen)
TIME_STEPS = list(range(-5, 6))


def _col(feat: str, t: int) -> str:
    """Kanonischer Spaltenname: feat_rsi14_tm3, feat_rsi14_t0, feat_rsi14_tp2"""
    if t == 0:
        return f"feat_{feat}_t0"
    sign = "p" if t > 0 else "m"
    return f"feat_{feat}_t{sign}{abs(t)}"


def _t_label(t: int) -> str:
    """Lesbares Label für Tabellenköpfe: t-5, t-4, ..., t0, t+1, ..."""
    if t == 0:
        return "t0"
    return f"t{'+' if t > 0 else ''}{t}"


# ==============================================================================
# 1. DATEN LADEN
# ==============================================================================

def _load_ohlcv(tickers: list[str], years: float) -> dict[str, pd.DataFrame]:
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=int(years * 365.25))
    tset   = set(tickers)
    data:  dict[str, pd.DataFrame] = {}
    for fpath in sorted(_RAW_DIR.glob("*_1d.parquet")):
        ticker = fpath.stem.replace("_1d", "")
        if ticker not in tset:
            continue
        try:
            df = pd.read_parquet(fpath)
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.columns = [c.lower() for c in df.columns]
            df = df[df.index >= cutoff]
            if len(df) < MIN_DATA_ROWS:
                continue
            if not {"open", "high", "low", "close"}.issubset(df.columns):
                continue
            keep = [c for c in ["open","high","low","close","volume"]
                    if c in df.columns]
            data[ticker] = df[keep].copy()
        except Exception:
            pass
    return data


# ==============================================================================
# 2. WAVE FINDER mit IntraWave-MaxDD + Target_Quality
# ==============================================================================

def find_ideal_waves(
    ticker:       str,
    df:           pd.DataFrame,
    min_profit:   float = DEFAULT_MIN_PROFIT,
    max_pullback: float = DEFAULT_MAX_PB,
) -> list[dict]:
    """
    Findet retrospektiv qualifizierende Aufwärtswellen und berechnet:
      - IntraWave_MaxDD: Größter Pullback vom laufenden Hoch INNERHALB der Welle
        (immer < max_pullback, da die Welle sonst schon geendet wäre)
      - Target_Quality: Return_pct / max(IntraWave_MaxDD, 0.5)
        Glatte, starke Wellen → hohe TQ | Volatile, zögerliche Wellen → niedrige TQ

    Look-Ahead-Bias ist im Wave-Finder explizit erlaubt (Labeling-Schritt).
    Die Feature-Werte werden SEPARAT aus reinen Vergangenheitsdaten berechnet.
    """
    closes = df["close"].values
    dates  = df.index
    n      = len(closes)
    waves  = []

    i = 0
    while i < n - MIN_WAVE_DAYS:

        # Phase 1: Trough identifizieren
        trough_px = closes[i]
        trough_i  = i
        j         = i + 1
        while j < n:
            px = closes[j]
            if px < trough_px:
                trough_px = px
                trough_i  = j
                j += 1
                continue
            if (px - trough_px) / trough_px >= max_pullback:
                break    # Trough gesichert (Preis hat sich um max_pullback% erholt)
            j += 1

        if j >= n:
            break

        # Phase 2: Welle von trough_i aus tracken + IntraWave-DD messen
        peak_px      = closes[trough_i]
        peak_i       = trough_i
        max_dd_pct   = 0.0   # Größter Pullback vom laufenden Hoch innerhalb der Welle
        wave_ended   = False

        for k in range(trough_i + 1, n):
            px = closes[k]
            if px > peak_px:
                peak_px = px
                peak_i  = k
            current_dd = (peak_px - px) / peak_px * 100
            if current_dd > max_dd_pct:
                max_dd_pct = current_dd
            if current_dd / 100 > max_pullback:
                wave_ended = True
                break

        if not wave_ended:
            # Restlauf bis Datenende
            rem_max_i = trough_i + int(np.argmax(closes[trough_i:]))
            if closes[rem_max_i] > peak_px:
                peak_px = closes[rem_max_i]
                peak_i  = rem_max_i

        total_ret   = (peak_px - trough_px) / trough_px
        duration_d  = max((dates[peak_i] - dates[trough_i]).days, 1)
        duration_td = peak_i - trough_i

        if total_ret >= min_profit and duration_td >= MIN_WAVE_DAYS:
            ret_pct        = total_ret * 100
            target_quality = ret_pct / max(max_dd_pct, 0.5)
            waves.append({
                "ticker":         ticker,
                "start_date":     dates[trough_i],
                "end_date":       dates[peak_i],
                "start_price":    round(float(trough_px), 4),
                "peak_price":     round(float(peak_px),   4),
                "duration_d":     duration_d,
                "duration_td":    duration_td,
                "return_pct":     round(ret_pct,        2),
                "intrawave_dd":   round(max_dd_pct,     2),
                "target_quality": round(target_quality, 4),
            })
            i = peak_i + 1
        else:
            i = trough_i + 1

    return waves


# ==============================================================================
# 3. FEATURE ENGINEERING  (8 Basis-Features, voll vektorisiert)
# ==============================================================================

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=period - 1, adjust=False).mean()
    avg_l = loss.ewm(com=period - 1, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet alle 8 Basis-Features als vollständige Zeitreihe.
    Vollständig vektorisiert, kein Look-Ahead in den Features selbst.
    Spalten entsprechen BASE_FEATURES.
    """
    c   = df["close"]
    vol = df.get("volume")

    sma20  = c.rolling(20).mean()
    sma50  = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()

    # Relative Distanz zu gleitenden Durchschnitten
    dist_sma20  = (c - sma20)  / sma20.replace(0,  np.nan)
    dist_sma50  = (c - sma50)  / sma50.replace(0,  np.nan)
    dist_sma200 = (c - sma200) / sma200.replace(0, np.nan)

    # RSI
    rsi14 = _rsi(c, 14)

    # Bollinger Bänder
    std20    = c.rolling(20).std()
    bb_up    = sma20 + 2.0 * std20
    bb_lo    = sma20 - 2.0 * std20
    bb_range = (bb_up - bb_lo).replace(0, np.nan)
    bb_width = bb_range / c.replace(0, np.nan)

    # MACD Histogramm (normiert auf Preis)
    ema12      = c.ewm(span=12, adjust=False).mean()
    ema26      = c.ewm(span=26, adjust=False).mean()
    macd_line  = ema12 - ema26
    sig9       = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist_norm = (macd_line - sig9) / c.replace(0, np.nan)

    # ATR (relativ zum Preis) – stärkster Predictor aus v9.0
    atr14   = _atr(df, 14)
    atr_pct = atr14 / c.replace(0, np.nan)

    # Volumen-Spike
    if vol is not None:
        sma20_v   = vol.rolling(20).mean().replace(0, np.nan)
        vol_spike = vol / sma20_v
    else:
        vol_spike = pd.Series(np.nan, index=c.index)

    return pd.DataFrame({
        "atr_pct":       atr_pct,
        "bb_width":      bb_width,
        "dist_sma200":   dist_sma200,
        "rsi14":         rsi14,
        "vol_spike":     vol_spike,
        "dist_sma50":    dist_sma50,
        "dist_sma20":    dist_sma20,
        "macd_hist_norm": macd_hist_norm,
    }, index=c.index)


# ==============================================================================
# 4. BUILD DATASET: Waves + Time-Window Features mergen
# ==============================================================================

def build_dataset(
    data:         dict[str, pd.DataFrame],
    min_profit:   float,
    max_pullback: float,
) -> pd.DataFrame:
    """
    Für jede Welle: Features an T-5 bis T+5 um den Tiefpunkt herum snappen.
    Wellen werden verworfen, wenn das T±5-Fenster außerhalb der verfügbaren
    Daten liegt (kein Auffüllen mit NaN – saubere Datenbasis für ML).
    """
    all_rows: list[dict] = []
    tickers   = sorted(data.keys())
    window    = max(abs(t) for t in TIME_STEPS)   # = 5

    for idx, ticker in enumerate(tickers, 1):
        df   = data[ticker]
        feats = compute_features(df)
        waves = find_ideal_waves(ticker, df, min_profit, max_pullback)

        if not waves:
            continue

        # Schneller Index-Lookup: date → integer position in feats
        date_to_pos = {d: i for i, d in enumerate(feats.index)}

        for w in waves:
            t0_date = w["start_date"]
            if t0_date not in date_to_pos:
                continue
            t0_pos = date_to_pos[t0_date]

            # Fenster-Bounds prüfen → Welle verwerfen wenn außerhalb
            if t0_pos - window < 0 or t0_pos + window >= len(feats):
                continue

            row = {**w}

            # Feature-Snapshot für jeden Zeitschritt
            for t in TIME_STEPS:
                snap = feats.iloc[t0_pos + t]
                for feat in BASE_FEATURES:
                    if feat in snap.index and pd.notna(snap[feat]):
                        row[_col(feat, t)] = float(snap[feat])
                    else:
                        row[_col(feat, t)] = float("nan")

            all_rows.append(row)

        if idx % 50 == 0:
            print(f"  [{idx:>3}/{len(tickers)}]  "
                  f"{ticker:<6}  Wellen gesamt: {len(all_rows):>5}")

    return pd.DataFrame(all_rows)


# ==============================================================================
# 5. LEAD-LAG SPEARMAN-MATRIX
# ==============================================================================

def _spearman_r(x: pd.Series, y: pd.Series) -> float:
    """Spearman über Rang-Pearson – ohne externe Abhängigkeit."""
    combined = pd.concat([x, y], axis=1).dropna()
    if len(combined) < MIN_WAVES_CORR:
        return float("nan")
    rx = combined.iloc[:, 0].rank()
    ry = combined.iloc[:, 1].rank()
    return float(rx.corr(ry))


def compute_lead_lag_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    2D-Matrix: Zeilen = BASE_FEATURES, Spalten = TIME_STEPS.
    Wert = Spearman(Feature[t], Target_Quality).
    """
    target = df["target_quality"]
    t_labels = [_t_label(t) for t in TIME_STEPS]
    matrix   = pd.DataFrame(index=BASE_FEATURES, columns=t_labels, dtype=float)

    for feat in BASE_FEATURES:
        for t in TIME_STEPS:
            col = _col(feat, t)
            if col not in df.columns:
                matrix.loc[feat, _t_label(t)] = float("nan")
                continue
            r = _spearman_r(target, df[col])
            matrix.loc[feat, _t_label(t)] = round(r, 4)

    return matrix.astype(float)


# ==============================================================================
# 6. AUSGABE
# ==============================================================================

def print_matrix(matrix: pd.DataFrame, n: int) -> None:
    """Druckt die gefilterte Lead-Lag-Korrelationsmatrix."""
    t_labels = [_t_label(t) for t in TIME_STEPS]

    # Features filtern: max |r| über alle Zeitschritte muss > MATRIX_FILTER_R
    keep = matrix[matrix.abs().max(axis=1) > MATRIX_FILTER_R]
    # Sortierung: nach maximaler absoluter Korrelation über alle Zeitschritte
    keep = keep.loc[keep.abs().max(axis=1).sort_values(ascending=False).index]

    print(f"\n  ┌─ SPEARMAN LEAD-LAG MATRIX  (Feature × Zeitschritt → Target_Quality)")
    print(f"  │  n = {n:,} Wellen  |  Filter: max|r| > {MATRIX_FILTER_R}")
    print(f"  │  T-5..T-1: rein vergangenheitsbasiert (für Live-Signale nutzbar)")
    print(f"  │  T+1..T+5: vorwärtsgerichtet  (akademisch, Prozessverständnis)")
    print(f"  │")

    # Header
    hdr_feat  = f"  │  {'Basis-Feature':<20}"
    hdr_times = "".join(f"  {_t_label(t):>5}" for t in TIME_STEPS)
    print(hdr_feat + hdr_times)
    print(f"  │  {'─' * 20}" + "─" * (7 * len(TIME_STEPS)))

    for feat, row in keep.iterrows():
        line = f"  │  {feat:<20}"
        peak_abs = row.abs().max()
        for t in TIME_STEPS:
            lbl = _t_label(t)
            val = row[lbl]
            if math.isnan(val):
                line += f"  {'—':>5}"
            else:
                # Hervorheben: Maximalwert in der Zeile bekommt Stern
                star = "*" if abs(val) == peak_abs else " "
                line += f"  {val:>+5.3f}{star}"
        print(line)

    print(f"  └{'─' * (20 + 7 * len(TIME_STEPS) + 4)}")

    # Wichtigste Einzelwerte
    print(f"\n  PEAK-KORRELATIONEN (je Feature):")
    print(f"  {'─' * 65}")
    for feat, row in keep.iterrows():
        best_t_i = row.abs().idxmax()
        best_r   = row[best_t_i]
        dir_str  = ("→ hohe Werte prädiktiv" if best_r > 0
                    else "→ niedrige Werte prädiktiv")
        timing   = ("(vergangenheitsbasiert ✓)"
                    if TIME_STEPS[list(keep.columns).index(best_t_i)] <= 0
                    else "(vorwärtsgerichtet, nur akademisch)")
        print(f"  {feat:<22}  peak r={best_r:>+.4f} @ {best_t_i:<5}  "
              f"{dir_str}  {timing}")


def print_bucket(df: pd.DataFrame) -> None:
    """Top-25% vs. Bottom-25% Target_Quality – Feature-Profil an T0."""
    thr_hi = df["target_quality"].quantile(0.75)
    thr_lo = df["target_quality"].quantile(0.25)
    top    = df[df["target_quality"] >= thr_hi]
    bot    = df[df["target_quality"] <= thr_lo]

    print(f"\n  ┌─ BUCKET-ANALYSE @ T=0  (Top-25% vs. Bottom-25% Target_Quality)")
    print(f"  │  Top-25%:    TQ ≥ {thr_hi:.2f}  (n={len(top):,})")
    print(f"  │  Bottom-25%: TQ ≤ {thr_lo:.2f}  (n={len(bot):,})")
    print(f"  │")
    print(f"  │  {'Basis-Feature':<22}  {'Top Median':>12}  "
          f"{'Bot Median':>12}  {'Δ':>8}  Bedeutung")
    print(f"  │  {'─' * 70}")

    rows = []
    for feat in BASE_FEATURES:
        col = _col(feat, 0)
        if col not in df.columns:
            continue
        t_med = top[col].median()
        b_med = bot[col].median()
        delta = t_med - b_med
        rows.append((feat, t_med, b_med, delta))

    rows.sort(key=lambda x: abs(x[3]), reverse=True)
    for feat, t_med, b_med, delta in rows:
        arrow = ("↑ höher vor schnellen Wellen" if delta > 0
                 else "↓ niedriger vor schnellen Wellen")
        print(f"  │  {feat:<22}  {t_med:>12.4f}  {b_med:>12.4f}  "
              f"{delta:>+8.4f}  {arrow}")
    print(f"  └{'─' * 72}")


def print_target_stats(df: pd.DataFrame) -> None:
    tq = df["target_quality"]
    ret = df["return_pct"]
    dd  = df["intrawave_dd"]

    print(f"\n  ┌─ WELLEN & TARGET_QUALITY STATISTIKEN {'─' * 31}")
    print(f"  │  Wellen gesamt:      {len(df):,}")
    print(f"  │  Ticker mit Wellen:  {df['ticker'].nunique()}")
    print(f"  │")
    print(f"  │  Return_pct:     Ø {ret.mean():>6.1f}%  |  Median {ret.median():>5.1f}%  "
          f"|  Max {ret.max():>6.1f}%")
    print(f"  │  IntraWave_DD:   Ø {dd.mean():>6.2f}%  |  Median {dd.median():>5.2f}%  "
          f"|  Max {dd.max():>6.2f}%")
    print(f"  │  Target_Quality: Ø {tq.mean():>6.2f}   |  Median {tq.median():>5.2f}   "
          f"|  Max {tq.max():>6.2f}")
    print(f"  │")

    # Jahresverteilung
    by_year = df.groupby(df["start_date"].dt.year)["target_quality"].agg(
        ["count","mean","median"])
    print(f"  │  Jahr   Wellen   Ø TQ    Med TQ")
    print(f"  │  {'─' * 38}")
    for yr, row in by_year.iterrows():
        bar = "█" * int(row["mean"] / by_year["mean"].max() * 20)
        print(f"  │  {yr}   {int(row['count']):>5}    {row['mean']:>5.2f}    "
              f"{row['median']:>5.2f}  {bar}")
    print(f"  └{'─' * 57}")


# ==============================================================================
# 7. MAIN
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lead-Lag Event Study v9.2  |  Target Quality + T-5..T+5")
    parser.add_argument("--years",        type=float, default=DEFAULT_YEARS)
    parser.add_argument("--min-profit",   type=float, default=DEFAULT_MIN_PROFIT)
    parser.add_argument("--max-pullback", type=float, default=DEFAULT_MAX_PB)
    args = parser.parse_args()

    print("=" * 74)
    print("  MASTER EVENT STUDY v9.2  |  Target Quality + Lead-Lag Feature Mining")
    print("=" * 74)
    print(f"""
  Target:      Target_Quality = Return_pct / max(IntraWave_DD%, 0.5)
  Zeitfenster: T-5 bis T+5  ({len(TIME_STEPS)} Schritte × {len(BASE_FEATURES)} Features = {len(TIME_STEPS)*len(BASE_FEATURES)} Spalten)
  Wave-Filter: min_profit={args.min_profit*100:.0f}%  |  max_pullback={args.max_pullback*100:.0f}%
  Zeitraum:    {args.years:.0f} Jahre  |  260 US-Aktien
""")

    # 1. Daten
    print("[1/5] Lade Daten...")
    t0      = time.time()
    tickers = _load_tickers()
    data    = _load_ohlcv(tickers, args.years)
    print(f"  {len(data)} Ticker in {time.time()-t0:.1f}s")

    # 2+3. Waves + Time-Window Features
    print(f"\n[2/5] Wave-Finder + Time-Window Feature Engineering...")
    print(f"  (Wave-Finder: retrospektiv, Look-Ahead für Labels OK)")
    print(f"  (Features: ausschliesslich Vergangenheitsdaten bis T+t)")
    t0  = time.time()
    df  = build_dataset(data, args.min_profit, args.max_pullback)
    elapsed = time.time() - t0
    print(f"\n  {len(df):,} Wellen mit vollständigem T±5-Fenster  ({elapsed:.1f}s)")

    if df.empty:
        print("  !! Keine Wellen gefunden. Parameter überprüfen.")
        return

    # NaN-Audit
    all_feat_cols = [_col(f, t) for f in BASE_FEATURES for t in TIME_STEPS]
    nan_rate = df[all_feat_cols].isna().mean() * 100
    high_nan = nan_rate[nan_rate > 15]
    if not high_nan.empty:
        print(f"\n  Features mit >15% NaN:")
        for col, pct in high_nan.items():
            print(f"    {col}: {pct:.1f}%")

    df_clean = df.dropna(subset=["target_quality"] + all_feat_cols)
    print(f"  Vollständig (kein NaN): {len(df_clean):,}  ({len(df_clean)/len(df)*100:.1f}%)")

    # 4. Lead-Lag Matrix
    print(f"\n[3/5] Spearman Lead-Lag Matrix berechnen...")
    t0     = time.time()
    matrix = compute_lead_lag_matrix(df_clean)
    print(f"  {matrix.shape[0]} Features × {matrix.shape[1]} Zeitschritte  "
          f"({time.time()-t0:.1f}s)")

    # 5. Output
    print(f"\n[4/5] Ausgabe:")
    print(f"\n{'=' * 74}")
    print(f"  MASTER ERGEBNISSE  |  Lead-Lag Event Study  |  n={len(df_clean):,}")
    print("=" * 74)

    print_target_stats(df_clean)
    print_matrix(matrix, len(df_clean))
    print_bucket(df_clean)

    # Top 10 beste Wellen (höchste TQ)
    top10 = df_clean.nlargest(10, "target_quality")
    print(f"\n  ┌─ TOP 10 WELLEN NACH TARGET_QUALITY {'─' * 35}")
    print(f"  │  {'Ticker':<7}  {'Start':>10}  {'Ende':>10}  "
          f"{'Dur':>5}  {'Ret%':>7}  {'IntrDD':>7}  {'TQ':>7}")
    print(f"  │  {'─' * 60}")
    for _, row in top10.iterrows():
        print(f"  │  {row['ticker']:<7}  "
              f"{str(row['start_date'])[:10]:>10}  "
              f"{str(row['end_date'])[:10]:>10}  "
              f"{int(row['duration_d']):>4}d  "
              f"{row['return_pct']:>+6.1f}%  "
              f"{row['intrawave_dd']:>6.2f}%  "
              f"{row['target_quality']:>7.2f}")
    print(f"  └{'─' * 60}")

    # 6. CSV
    print(f"\n[5/5] Speichern...")
    meta_cols = ["ticker","start_date","end_date","start_price","peak_price",
                 "duration_d","duration_td","return_pct","intrawave_dd","target_quality"]
    feat_cols = [_col(f, t) for f in BASE_FEATURES for t in TIME_STEPS]
    out_cols  = meta_cols + feat_cols
    df_clean[out_cols].sort_values(["start_date","ticker"]).to_csv(
        _OUT_CSV, index=False)
    print(f"  Datei: {_OUT_CSV}")
    print(f"  Zeilen: {len(df_clean):,}  |  Spalten: {len(out_cols)}")
    print(f"  (davon Feature-Spalten: {len(feat_cols)} = "
          f"{len(BASE_FEATURES)} Features × {len(TIME_STEPS)} Zeitschritte)")

    # Matrix auch als CSV
    matrix_path = _here / "leadlag_matrix.csv"
    matrix.round(4).to_csv(matrix_path)
    print(f"  Matrix: {matrix_path}")

    print(f"\n  FERTIG.")
    print(f"  Nächster Schritt: 'ideal_trades_v9_master.csv' als Trainings-")
    print(f"  datensatz für einen Decision Tree / Random Forest nutzen.\n")


if __name__ == "__main__":
    main()
