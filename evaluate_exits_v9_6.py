"""
evaluate_exits_v9_6.py
====================================================================================
Peak & Exhaustion Detection  |  Exit Strategy  |  v9.6

Ziel: Einen Decision Tree trainieren, der zwischen einem gesunden Trend-Tag (Hold)
und dem finalen Hochpunkt kurz vor dem Absturz (Sell/Exhaustion) unterscheidet.

Pro Welle werden exakt ZWEI Samples generiert:
  ► y=0  "Mid-Trend"  →  Tag bei 50% der Wellendauer  (Kurs läuft noch)
  ► y=1  "Peak/Sell"  →  Exakter Hochpunkt der Welle  (Exhaustion)

Features (Fokus auf Übertreibung & Klimax):
  feat_dist_sma20        (Close - SMA20) / SMA20      ← kurzfristiger Gummiband-Effekt
  feat_dist_sma50        (Close - SMA50) / SMA50
  feat_rsi14             Klassischer RSI(14)
  feat_rsi_3d_delta      RSI(14) − RSI(14) vor 3 Tagen ← parabolische Beschleunigung
  feat_bb_width          (BB_upper − BB_lower) / Close
  feat_vol_spike         Volume / SMA_Volume_20        ← Klimax-Volumen am Top?
  feat_consecutive_ups   Aufeinanderfolgende Tage mit Close > Open

Modell: DecisionTreeClassifier(max_depth=3, min_samples_leaf=50)
Output: Accuracy + Precision, Feature Importances, Text-Regeln, exit_samples_v9_6.csv

Verwendung:
  python evaluate_exits_v9_6.py
  python evaluate_exits_v9_6.py --min-profit 0.20 --max-pullback 0.05
"""

from __future__ import annotations

import argparse
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

try:
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.metrics import (accuracy_score, precision_score,
                                 classification_report, confusion_matrix)
    from sklearn.model_selection import cross_val_score
except ImportError:
    print("FEHLER: scikit-learn nicht installiert.  pip install scikit-learn")
    sys.exit(1)

# ── Konstanten ────────────────────────────────────────────────────────────────
DEFAULT_YEARS     = 7.0
DEFAULT_MIN_PROFIT = 0.15
DEFAULT_MAX_PB    = 0.05
MIN_WAVE_DAYS     = 5       # Mindest-Wellenlänge für sinnvolle Samples
MIN_DATA_ROWS     = 260
DT_MAX_DEPTH      = 3
DT_MIN_LEAF       = 50
_RAW_DIR          = _here / "data" / "raw"

FEAT_COLS = [
    "feat_dist_sma20",
    "feat_dist_sma50",
    "feat_rsi14",
    "feat_rsi_3d_delta",
    "feat_bb_width",
    "feat_vol_spike",
    "feat_consecutive_ups",
]

FEAT_LABELS = {
    "feat_dist_sma20":      "Dist.SMA20 (Überdehnung)",
    "feat_dist_sma50":      "Dist.SMA50",
    "feat_rsi14":           "RSI-14",
    "feat_rsi_3d_delta":    "ΔRSI-3d (Parabolisch)",
    "feat_bb_width":        "BB-Breite (Volatilität)",
    "feat_vol_spike":       "Vol-Spike (Klimax-Vol)",
    "feat_consecutive_ups": "Konsek. Up-Days",
}


# ==============================================================================
# 1. DATEN
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
            keep = [c for c in ["open", "high", "low", "close", "volume"]
                    if c in df.columns]
            data[ticker] = df[keep].copy()
        except Exception:
            pass
    return data


# ==============================================================================
# 2. FEATURES
# ==============================================================================

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=period - 1, adjust=False).mean()
    avg_l = loss.ewm(com=period - 1, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _consecutive_up_days(df: pd.DataFrame) -> pd.Series:
    """Anzahl aufeinanderfolgender Tage mit Close > Open (laufend zählen)."""
    up = (df["close"] > df["open"]).astype(int)
    result = np.zeros(len(df), dtype=int)
    count = 0
    for i in range(len(df)):
        if up.iloc[i] == 1:
            count += 1
        else:
            count = 0
        result[i] = count
    return pd.Series(result, index=df.index)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Berechnet alle Features mit Fokus auf Übertreibung/Klimax."""
    c      = df["close"]
    op     = df.get("open", c)
    vol    = df.get("volume")
    sma20  = c.rolling(20).mean()
    sma50  = c.rolling(50).mean()
    std20  = c.rolling(20).std()
    bb_up  = sma20 + 2.0 * std20
    bb_lo  = sma20 - 2.0 * std20
    bb_r   = (bb_up - bb_lo).replace(0, np.nan)
    rsi14  = _rsi(c, 14)
    sma20v = (vol.rolling(20).mean().replace(0, np.nan)
              if vol is not None else None)
    consec = _consecutive_up_days(df)
    return pd.DataFrame({
        "feat_dist_sma20":      (c - sma20)  / sma20.replace(0, np.nan),
        "feat_dist_sma50":      (c - sma50)  / sma50.replace(0, np.nan),
        "feat_rsi14":           rsi14,
        "feat_rsi_3d_delta":    rsi14 - rsi14.shift(3),
        "feat_bb_width":        bb_r / c.replace(0, np.nan),
        "feat_vol_spike":       (vol / sma20v if sma20v is not None
                                 else pd.Series(np.nan, index=c.index)),
        "feat_consecutive_ups": consec.astype(float),
    }, index=c.index)


# ==============================================================================
# 3. WAVE FINDER
# ==============================================================================

def find_ideal_waves(
    ticker: str, df: pd.DataFrame,
    min_profit: float, max_pullback: float,
) -> list[dict]:
    closes = df["close"].values
    dates  = df.index
    n      = len(closes)
    waves  = []
    i      = 0
    while i < n - MIN_WAVE_DAYS:
        trough_px = closes[i]
        trough_i  = i
        j = i + 1
        while j < n:
            px = closes[j]
            if px < trough_px:
                trough_px = px; trough_i = j; j += 1; continue
            if (px - trough_px) / trough_px >= max_pullback:
                break
            j += 1
        if j >= n:
            break
        peak_px = closes[trough_i]; peak_i = trough_i; wave_ended = False
        for k in range(trough_i + 1, n):
            px = closes[k]
            if px > peak_px:
                peak_px = px; peak_i = k
            if (peak_px - px) / peak_px > max_pullback:
                wave_ended = True; break
        if not wave_ended:
            best = trough_i + int(np.argmax(closes[trough_i:]))
            if closes[best] > peak_px:
                peak_px = closes[best]; peak_i = best
        total_ret = (peak_px - trough_px) / trough_px
        duration  = peak_i - trough_i
        if total_ret >= min_profit and duration >= MIN_WAVE_DAYS:
            waves.append({
                "ticker":      ticker,
                "trough_i":    trough_i,
                "peak_i":      peak_i,
                "trough_date": dates[trough_i],
                "peak_date":   dates[peak_i],
                "total_ret":   round(total_ret * 100, 2),
                "duration":    duration,
            })
            i = peak_i + 1
        else:
            i = trough_i + 1
    return waves


# ==============================================================================
# 4. SAMPLE GENERATION
# ==============================================================================

def build_samples(
    data:       dict[str, pd.DataFrame],
    min_profit: float,
    max_pullback: float,
) -> pd.DataFrame:
    """
    Generiert pro Welle exakt 2 Samples:
      y=0  Mid-Trend:  Tag bei 50% der Wellendauer
      y=1  Peak/Sell:  Exakter Hochpunkt
    """
    rows = []
    tickers = sorted(data.keys())
    n_waves_total = 0

    for idx, ticker in enumerate(tickers, 1):
        df     = data[ticker]
        feats  = compute_features(df)
        waves  = find_ideal_waves(ticker, df, min_profit, max_pullback)
        if not waves:
            continue

        date_arr    = feats.index
        date_to_pos = {d: i for i, d in enumerate(date_arr)}

        for w in waves:
            t_i = w["trough_i"]
            p_i = w["peak_i"]
            dur = w["duration"]

            # ── Mid-Trend Sample (y=0) ──────────────────────────────────────
            mid_i = t_i + max(1, dur // 2)
            if mid_i >= len(feats) or mid_i >= p_i:
                continue

            trough_date = w["trough_date"]
            if trough_date not in date_to_pos:
                continue
            t_pos   = date_to_pos[trough_date]
            mid_pos = t_pos + max(1, dur // 2)
            peak_pos = t_pos + dur

            if mid_pos >= len(feats) or peak_pos >= len(feats):
                continue

            snap_mid  = feats.iloc[mid_pos]
            snap_peak = feats.iloc[peak_pos]

            if snap_mid.isna().any() or snap_peak.isna().any():
                continue

            n_waves_total += 1

            def _make_row(snap, label, sample_type):
                r = {
                    "ticker":      ticker,
                    "wave_start":  w["trough_date"],
                    "wave_end":    w["peak_date"],
                    "sample_date": date_arr[mid_pos if label == 0 else peak_pos],
                    "label":       label,
                    "sample_type": sample_type,
                    "total_ret_pct": w["total_ret"],
                    "duration_days": dur,
                }
                for col in FEAT_COLS:
                    r[col] = float(snap[col]) if col in snap.index else float("nan")
                return r

            rows.append(_make_row(snap_mid,  label=0, sample_type="mid_trend"))
            rows.append(_make_row(snap_peak, label=1, sample_type="peak"))

        if idx % 60 == 0:
            print(f"  [{idx:>3}/{len(tickers)}]  Wellen: {n_waves_total:>4}")

    df_out = pd.DataFrame(rows)
    print(f"\n  Wellen gesamt:   {n_waves_total:>5,}")
    print(f"  Samples gesamt:  {len(df_out):>5,}  (je 1x Mid-Trend + 1x Peak)")
    return df_out


# ==============================================================================
# 5. MODELL
# ==============================================================================

def train_and_evaluate(df: pd.DataFrame) -> tuple[DecisionTreeClassifier, list[str]]:
    df_c = df[FEAT_COLS + ["label"]].dropna()
    print(f"\n  Samples nach NaN-Drop: {len(df_c):,}  "
          f"(Mid-Trend: {(df_c['label']==0).sum():,}  |  "
          f"Peak: {(df_c['label']==1).sum():,})")

    X = df_c[FEAT_COLS]
    y = df_c["label"]

    clf = DecisionTreeClassifier(
        max_depth=DT_MAX_DEPTH, min_samples_leaf=DT_MIN_LEAF,
        criterion="gini", random_state=42,
    )
    clf.fit(X, y)

    y_pred = clf.predict(X)
    acc    = accuracy_score(y, y_pred)
    prec   = precision_score(y, y_pred, zero_division=0)
    rec    = (y[(y == 1) & (y_pred == 1)].sum() /
              max(y.sum(), 1))

    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="precision")

    sep  = "=" * 72
    line = "─" * 72
    print(f"\n{sep}")
    print(f"  MODELL-PERFORMANCE  |  DecisionTree(depth={DT_MAX_DEPTH}, min_leaf={DT_MIN_LEAF})")
    print(sep)
    print(f"  In-Sample Accuracy:        {acc*100:>6.1f}%")
    print(f"  In-Sample Precision (Peak):{prec*100:>6.1f}%  "
          "← Wie oft lag der Tree richtig, wenn er 'Sell' sagte?")
    print(f"  In-Sample Recall (Peak):   {rec*100:>6.1f}%  "
          "← Wie viele echte Peaks wurden erkannt?")
    print(f"  5-Fold CV Precision:       "
          f"{cv_scores.mean()*100:>6.1f}% ± {cv_scores.std()*100:.1f}%  "
          "(Overfit-Check)")
    print(f"\n  Confusion Matrix  (Zeile = tatsächlich, Spalte = vorhergesagt):")
    cm = confusion_matrix(y, y_pred)
    for i, row in enumerate(cm):
        lbl = "Mid-Trend (0)" if i == 0 else "Peak/Sell (1)"
        print(f"    {lbl:<15}  "
              + "  ".join(f"{v:>5}" for v in row)
              + f"    {'TN/FP' if i==0 else 'FN/TP'}")
    print(f"\n{line}")
    print(classification_report(y, y_pred,
                                 target_names=["Mid-Trend (Hold)", "Peak (Sell)"],
                                 digits=3))
    return clf, FEAT_COLS


# ==============================================================================
# 6. AUSGABE
# ==============================================================================

def print_feature_importances(clf: DecisionTreeClassifier) -> None:
    sep  = "=" * 72
    line = "─" * 72
    print(f"\n{sep}")
    print(f"  FEATURE IMPORTANCES  |  Wer warnt am besten vor dem Absturz?")
    print(sep)
    imp = pd.Series(clf.feature_importances_, index=FEAT_COLS).sort_values(ascending=False)
    for feat, val in imp.items():
        bar   = "█" * int(val * 50)
        label = FEAT_LABELS.get(feat, feat)
        print(f"  {label:<35}  {val:.4f}  {bar}")


def _readable(feat: str) -> str:
    return FEAT_LABELS.get(feat, feat)


def print_text_rules(clf: DecisionTreeClassifier) -> None:
    sep  = "=" * 72
    line = "─" * 72
    print(f"\n{sep}")
    print(f"  ENTSCHEIDUNGSBAUM-REGELN  |  Focus: class 1 = Peak/Sell")
    print(f"  class 0 = Mid-Trend (Hold)  |  class 1 = Peak/Sell (Exhaustion)")
    print(sep)
    readable = [_readable(f) for f in FEAT_COLS]
    for l in export_text(clf, feature_names=readable, show_weights=True).split("\n"):
        if l.strip():
            print(f"  {l}")

    # Wenn-Dann-Regeln für SELL-Äste (value[1] > value[0])
    print(f"\n{sep}")
    print(f"  WENN-DANN VERKAUFS-REGELN  (nur Äste mit Peak/Sell-Mehrheit)")
    print(sep)
    tree = clf.tree_
    node_paths: dict[int, list[str]] = {}
    _collect_paths(clf, 0, [], node_paths, readable)

    sell_rules_found = False
    for node_id, path_conds in node_paths.items():
        if tree.children_left[node_id] != -1:
            continue
        vals   = tree.value[node_id][0]
        n_hold = vals[0]
        n_sell = vals[1]
        total  = n_hold + n_sell
        prec   = n_sell / total if total > 0 else 0.0
        if prec <= 0.5:
            continue
        sell_rules_found = True
        print(f"\n  WENN:")
        for cond in path_conds:
            print(f"    {cond}")
        print(f"  DANN:  Peak/Sell  "
              f"(Precision {prec:.1%}  |  "
              f"Hold={int(n_hold)}, Sell={int(n_sell)})")

    if not sell_rules_found:
        print("\n  Kein einzelner Blatt-Knoten mit Sell-Mehrheit gefunden.")
        print("  → Modell klassifiziert alle Knoten als Hold.")
        print("  → Tipp: min_samples_leaf verkleinern oder Features anpassen.")


def _collect_paths(
    clf: DecisionTreeClassifier, node_id: int,
    current_path: list[str], result: dict, feat_names: list[str],
) -> None:
    result[node_id] = list(current_path)
    tree = clf.tree_
    if tree.children_left[node_id] == -1:
        return
    fi     = tree.feature[node_id]
    thresh = tree.threshold[node_id]
    name   = feat_names[fi]
    _collect_paths(clf, tree.children_left[node_id],
                   current_path + [f"{name} <= {thresh:.4f}"], result, feat_names)
    _collect_paths(clf, tree.children_right[node_id],
                   current_path + [f"{name} >  {thresh:.4f}"], result, feat_names)


def print_distribution_stats(df: pd.DataFrame) -> None:
    """Zeigt Feature-Verteilung für Mid-Trend vs. Peak, um Übertreibungs-Signale zu verdeutlichen."""
    sep  = "=" * 72
    line = "─" * 72
    print(f"\n{sep}")
    print(f"  FEATURE STATISTIK  |  Mid-Trend (y=0) vs. Peak/Sell (y=1)")
    print(f"  Zeigt, welche Features am stärksten zwischen Hold und Exit unterscheiden.")
    print(sep)
    mid  = df[df["label"] == 0][FEAT_COLS].dropna()
    peak = df[df["label"] == 1][FEAT_COLS].dropna()
    print(f"  {'Feature':<35}  {'Mid-Ø':>8}  {'Peak-Ø':>8}  {'Diff':>8}  Interpretation")
    print(f"  {line}")
    for feat in FEAT_COLS:
        m = mid[feat].mean()
        p = peak[feat].mean()
        d = p - m
        label = FEAT_LABELS.get(feat, feat)
        arrow = "▲" if d > 0 else "▼"
        note = ""
        if feat == "feat_rsi14":
            note = "überkauft?" if p > 70 else "noch Raum"
        elif feat == "feat_dist_sma20":
            note = "überdehnt" if abs(d) > 0.02 else "gering"
        elif feat == "feat_vol_spike":
            note = "Klimax-Vol" if p > 1.5 else "normal"
        elif feat == "feat_consecutive_ups":
            note = "langer Lauf" if p > 3 else "kurz"
        print(f"  {label:<35}  {m:>+8.3f}  {p:>+8.3f}  "
              f"{arrow} {abs(d):>5.3f}  {note}")


# ==============================================================================
# 7. MAIN
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Peak & Exhaustion Detection  |  Exit Strategy  |  v9.6")
    parser.add_argument("--years",        type=float, default=DEFAULT_YEARS)
    parser.add_argument("--min-profit",   type=float, default=DEFAULT_MIN_PROFIT)
    parser.add_argument("--max-pullback", type=float, default=DEFAULT_MAX_PB)
    args = parser.parse_args()

    sep  = "=" * 72
    print(sep)
    print(f"  PEAK & EXHAUSTION DETECTION  |  Exit Strategy  |  v9.6")
    print(sep)
    print(f"""
  Samples:   y=0 Mid-Trend (bei 50% Wellendauer)  |  y=1 Peak (Hochpunkt)
  Features:  Übertreibungs-Indikatoren (RSI, Dist.SMA, Vol-Spike, Konsek.Ups)
  Modell:    DecisionTree(depth={DT_MAX_DEPTH}, min_leaf={DT_MIN_LEAF})
  Universum: 260 US-Aktien  |  {args.years:.0f} Jahre
  Wellen:    min_profit={args.min_profit*100:.0f}%  |  max_pullback={args.max_pullback*100:.0f}%
""")

    # 1. Daten laden
    print("[1/4] Lade Daten...")
    t0      = time.time()
    tickers = _load_tickers()
    data    = _load_ohlcv(tickers, args.years)
    print(f"  {len(data)} Ticker in {time.time()-t0:.1f}s")

    # 2. Samples generieren
    print(f"\n[2/4] Wave-Finder & Sample-Generierung...")
    t0      = time.time()
    df_samp = build_samples(data, args.min_profit, args.max_pullback)
    print(f"  Generierung in {time.time()-t0:.1f}s")

    if len(df_samp) < DT_MIN_LEAF * 4:
        print("FEHLER: Zu wenig Samples für sinnvolles Training.")
        return

    # 3. Feature-Verteilung
    print_distribution_stats(df_samp)

    # 4. Modell trainieren
    print(f"\n[3/4] Modell-Training...")
    clf, feat_cols = train_and_evaluate(df_samp)

    # 5. Ausgabe
    print(f"\n[4/4] Feature Importances & Regel-Extraktion...")
    print_feature_importances(clf)
    print_text_rules(clf)

    # 6. Export
    csv_path = _here / "exit_samples_v9_6.csv"
    df_samp.to_csv(csv_path, index=False)
    print(f"\n{'=' * 72}")
    print(f"  Samples exportiert: {csv_path}")
    print(f"  Zeilen: {len(df_samp):,}  |  Spalten: {len(df_samp.columns)}")
    print(f"  FERTIG.\n")


if __name__ == "__main__":
    main()
