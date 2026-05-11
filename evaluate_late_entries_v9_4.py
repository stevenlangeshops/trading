"""
evaluate_late_entries_v9_4.py
====================================================================================
Late Entry & Confirmation Tradeoff  |  v9.4

Hypothese: Ein Einstieg N Tage NACH dem Wellentiefpunkt (T0) opfert anfängliche
Rendite, ermöglicht aber durch zusätzliche Bestätigungs-Datenpunkte präzisere
Machine-Learning-Regeln → bessere Precision bei der Top-Tier-Klassifikation.

Für jeden wait_day ∈ {0, 1, 2, 3, 4, 5} wird:
  ► Decision Day = T0 + wait_day  (in Handelstagen)
  ► Remaining Return = (Peak - Close@Decision) / Close@Decision
  ► Setup verworfen wenn Remaining Return < 10% (nicht mehr rentabel)
  ► Rest_Drawdown = größter Pullback vom laufenden Hoch zwischen Decision und Peak
  ► Remaining_TQ  = Remaining_Return_pct / max(Rest_Drawdown, 0.5)
  ► Features: Indikatoren @ Decision + Deltas (Decision - T0)
  ► Modell: DecisionTreeClassifier(max_depth=3, min_samples_leaf=50)
  ► Ziel:   Top-33% Remaining_TQ (y=1) vs. Rest (y=0)

Output: Tradeoff-Tabelle + Text-Regeln für den Tag mit höchster Precision

Verwendung:
  python evaluate_late_entries_v9_4.py
  python evaluate_late_entries_v9_4.py --min-profit 0.15 --min-remaining 0.10
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

try:
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.metrics import precision_score
except ImportError:
    print("FEHLER: scikit-learn nicht installiert.  pip install scikit-learn")
    sys.exit(1)

# ── Konstanten ────────────────────────────────────────────────────────────────
DEFAULT_YEARS        = 7.0
DEFAULT_MIN_PROFIT   = 0.15
DEFAULT_MAX_PB       = 0.05
DEFAULT_MIN_REMAIN   = 0.10    # Verbleibende Mindest-Rendite ab Decision Day
MIN_WAVE_DAYS        = 5
MIN_DATA_ROWS        = 260
TOP_PCT              = 33.0    # Top-X% als Top-Tier definieren
WAIT_DAYS            = [0, 1, 2, 3, 4, 5]
DT_MAX_DEPTH         = 3
DT_MIN_LEAF          = 50
_RAW_DIR             = _here / "data" / "raw"

BASE_FEATURES = [
    "atr_pct", "bb_width", "dist_sma200",
    "rsi14", "vol_spike", "dist_sma50",
    "dist_sma20", "macd_hist_norm",
]


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
            if not {"open","high","low","close"}.issubset(df.columns):
                continue
            keep = [c for c in ["open","high","low","close","volume"]
                    if c in df.columns]
            data[ticker] = df[keep].copy()
        except Exception:
            pass
    return data


# ==============================================================================
# 2. FEATURE ENGINEERING
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
    c   = df["close"]
    vol = df.get("volume")

    sma20  = c.rolling(20).mean()
    sma50  = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()

    dist_sma20  = (c - sma20)  / sma20.replace(0, np.nan)
    dist_sma50  = (c - sma50)  / sma50.replace(0, np.nan)
    dist_sma200 = (c - sma200) / sma200.replace(0, np.nan)
    rsi14       = _rsi(c, 14)
    std20       = c.rolling(20).std()
    bb_up       = sma20 + 2.0 * std20
    bb_lo       = sma20 - 2.0 * std20
    bb_range    = (bb_up - bb_lo).replace(0, np.nan)
    bb_width    = bb_range / c.replace(0, np.nan)
    ema12       = c.ewm(span=12, adjust=False).mean()
    ema26       = c.ewm(span=26, adjust=False).mean()
    macd_line   = ema12 - ema26
    sig9        = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist_norm = (macd_line - sig9) / c.replace(0, np.nan)
    atr14       = _atr(df, 14)
    atr_pct     = atr14 / c.replace(0, np.nan)
    if vol is not None:
        sma20_v   = vol.rolling(20).mean().replace(0, np.nan)
        vol_spike = vol / sma20_v
    else:
        vol_spike = pd.Series(np.nan, index=c.index)

    return pd.DataFrame({
        "atr_pct": atr_pct, "bb_width": bb_width,
        "dist_sma200": dist_sma200, "rsi14": rsi14,
        "vol_spike": vol_spike, "dist_sma50": dist_sma50,
        "dist_sma20": dist_sma20, "macd_hist_norm": macd_hist_norm,
    }, index=c.index)


# ==============================================================================
# 3. WAVE FINDER
# ==============================================================================

def find_ideal_waves(
    ticker:       str,
    df:           pd.DataFrame,
    min_profit:   float,
    max_pullback: float,
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
                trough_px = px
                trough_i  = j
                j += 1
                continue
            if (px - trough_px) / trough_px >= max_pullback:
                break
            j += 1
        if j >= n:
            break

        peak_px    = closes[trough_i]
        peak_i     = trough_i
        wave_ended = False

        for k in range(trough_i + 1, n):
            px = closes[k]
            if px > peak_px:
                peak_px = px
                peak_i  = k
            if (peak_px - px) / peak_px > max_pullback:
                wave_ended = True
                break

        if not wave_ended:
            rem = trough_i + int(np.argmax(closes[trough_i:]))
            if closes[rem] > peak_px:
                peak_px = closes[rem]
                peak_i  = rem

        total_ret   = (peak_px - trough_px) / trough_px
        duration_td = peak_i - trough_i

        if total_ret >= min_profit and duration_td >= MIN_WAVE_DAYS:
            waves.append({
                "ticker":      ticker,
                "trough_i":    trough_i,
                "peak_i":      peak_i,
                "trough_date": dates[trough_i],
                "peak_date":   dates[peak_i],
                "trough_px":   float(trough_px),
                "peak_px":     float(peak_px),
                "total_ret":   round(total_ret * 100, 2),
            })
            i = peak_i + 1
        else:
            i = trough_i + 1

    return waves


# ==============================================================================
# 4. REST-DRAWDOWN BERECHNEN (Decision Day → Peak)
# ==============================================================================

def _rest_drawdown(closes: np.ndarray, start_i: int, end_i: int) -> float:
    """Größter Pullback vom laufenden Hoch zwischen start_i und end_i (inkl.)."""
    if start_i >= end_i:
        return 0.0
    seg    = closes[start_i: end_i + 1]
    peak   = seg[0]
    max_dd = 0.0
    for px in seg[1:]:
        if px > peak:
            peak = px
        dd = (peak - px) / peak * 100
        if dd > max_dd:
            max_dd = dd
    return max_dd


# ==============================================================================
# 5. BUILD DATASETS  (ein DataFrame pro wait_day)
# ==============================================================================

def build_all_datasets(
    data:         dict[str, pd.DataFrame],
    min_profit:   float,
    max_pullback: float,
    min_remaining: float,
) -> dict[int, pd.DataFrame]:
    """
    Gibt {wait_day: DataFrame} zurück.
    Features: 8 Snapshot + 8 Delta (Decision Day - T0) = 16 pro Modell.
    """
    # Initialisiere leere Listen für jeden wait_day
    rows: dict[int, list[dict]] = {w: [] for w in WAIT_DAYS}

    tickers = sorted(data.keys())
    for idx, ticker in enumerate(tickers, 1):
        df     = data[ticker]
        closes = df["close"].values
        feats  = compute_features(df)
        waves  = find_ideal_waves(ticker, df, min_profit, max_pullback)

        if not waves:
            continue

        date_to_pos = {d: i for i, d in enumerate(feats.index)}

        for w in waves:
            t0_i    = w["trough_i"]
            peak_i  = w["peak_i"]
            peak_px = w["peak_px"]

            # Snapshot-Features an T0
            t0_date = w["trough_date"]
            if t0_date not in date_to_pos:
                continue
            t0_pos = date_to_pos[t0_date]
            if t0_pos < 1 or t0_i < 0:
                continue

            snap_t0 = feats.iloc[t0_pos]
            if snap_t0.isna().any():
                continue

            for wait in WAIT_DAYS:
                dec_i = t0_i + wait
                if dec_i >= peak_i:
                    continue   # Decision Day nach Peak → sinnlos
                dec_pos = t0_pos + wait
                if dec_pos < 0 or dec_pos >= len(feats):
                    continue

                close_dec = closes[dec_i]
                if close_dec <= 0:
                    continue

                # Verbleibende Rendite vom Decision Day bis Peak
                rem_ret = (peak_px - close_dec) / close_dec * 100
                if rem_ret < min_remaining * 100:
                    continue   # Zu wenig übrig → überspringen

                # Rest-Drawdown (Decision Day → Peak)
                rest_dd = _rest_drawdown(closes, dec_i, peak_i)
                rem_tq  = rem_ret / max(rest_dd, 0.5)

                # Snapshot-Features am Decision Day
                snap_dec = feats.iloc[dec_pos]
                if snap_dec.isna().any():
                    continue

                row = {
                    "ticker":       ticker,
                    "trough_date":  w["trough_date"],
                    "peak_date":    w["peak_date"],
                    "wait_days":    wait,
                    "decision_date": feats.index[dec_pos],
                    "rem_ret_pct":  round(rem_ret,  2),
                    "rest_dd_pct":  round(rest_dd,  2),
                    "rem_tq":       round(rem_tq,   4),
                    "total_ret":    w["total_ret"],
                }

                for feat in BASE_FEATURES:
                    # Snapshot am Decision Day
                    v_dec = float(snap_dec[feat]) if feat in snap_dec.index else float("nan")
                    v_t0  = float(snap_t0[feat])  if feat in snap_t0.index  else float("nan")
                    row[f"snap_{feat}"] = v_dec
                    # Delta: Decision Day − T0
                    row[f"delta_{feat}"] = (v_dec - v_t0
                                            if not (math.isnan(v_dec) or math.isnan(v_t0))
                                            else float("nan"))
                rows[wait].append(row)

        if idx % 60 == 0:
            print(f"  [{idx:>3}/{len(tickers)}]  "
                  + "  ".join(f"w{w}:{len(rows[w]):>4}" for w in WAIT_DAYS))

    return {w: pd.DataFrame(rows[w]) for w in WAIT_DAYS}


# ==============================================================================
# 6. MODELL-TRAINING PRO WAIT_DAY
# ==============================================================================

def train_one_day(
    df:       pd.DataFrame,
    wait:     int,
    top_pct:  float = TOP_PCT,
) -> dict | None:
    """
    Trainiert einen DecisionTree für einen wait_day und gibt die Metriken zurück.
    """
    feat_cols = [c for c in df.columns
                 if c.startswith("snap_") or c.startswith("delta_")]
    df_c = df[feat_cols + ["rem_tq", "rem_ret_pct"]].dropna()
    if len(df_c) < DT_MIN_LEAF * 2:
        return None

    threshold = df_c["rem_tq"].quantile(1 - top_pct / 100)
    y  = (df_c["rem_tq"] >= threshold).astype(int)
    X  = df_c[feat_cols]

    clf = DecisionTreeClassifier(
        max_depth        = DT_MAX_DEPTH,
        min_samples_leaf = DT_MIN_LEAF,
        criterion        = "gini",
        random_state     = 42,
    )
    clf.fit(X, y)
    y_pred = clf.predict(X)

    # Precision: unter den vorhergesagten Top-Tier, wie viele sind wirklich Top?
    prec = precision_score(y, y_pred, zero_division=0)

    # Durchschnittliche verbleibende Rendite der echten True-Positives
    tp_mask = (y == 1) & (y_pred == 1)
    avg_ret_tp = float(df_c.loc[tp_mask, "rem_ret_pct"].mean()) if tp_mask.sum() > 0 else 0.0
    avg_ret_all_top = float(df_c.loc[y == 1, "rem_ret_pct"].mean())

    # Wichtigstes Feature
    imp = pd.Series(clf.feature_importances_, index=feat_cols)
    best_feat = imp.idxmax() if imp.max() > 0 else "—"
    best_feat_readable = (best_feat
                          .replace("snap_", "")
                          .replace("delta_", "Δ")
                          .replace("dist_", "dist"))

    # Wie viele Predictions = Top-Tier?
    n_predicted_top = int(y_pred.sum())
    n_true_top      = int(y.sum())

    return {
        "wait":             wait,
        "n_waves":          len(df_c),
        "n_true_top":       n_true_top,
        "threshold_tq":     round(threshold, 3),
        "precision":        round(prec * 100, 1),
        "n_predicted_top":  n_predicted_top,
        "avg_ret_tp":       round(avg_ret_tp,    1),
        "avg_ret_all_top":  round(avg_ret_all_top, 1),
        "best_feat":        best_feat_readable,
        "clf":              clf,
        "feat_cols":        feat_cols,
        "df_clean":         df_c,
        "y":                y,
    }


# ==============================================================================
# 7. AUSGABE
# ==============================================================================

def print_tradeoff_table(results: list[dict]) -> None:
    sep  = "=" * 82
    line = "─" * 82

    print(f"\n{sep}")
    print(f"  TRADEOFF REPORT: Late Entry & Confirmation  |  v9.4")
    print(sep)
    print(f"""
  Hypothese: Späteinstieg (T0+N) opfert Rendite, gewinnt aber Präzision.
  Modell:    DecisionTreeClassifier(max_depth={DT_MAX_DEPTH}, min_leaf={DT_MIN_LEAF})
  Target:    Top-{TOP_PCT:.0f}% Remaining_Target_Quality  (y=1 vs. y=0)
""")

    print(f"  {'Wait':>5}  {'n Wellen':>9}  {'n Top-Tier':>10}  "
          f"{'Precision':>10}  {'n Pred.Top':>10}  "
          f"{'Ø Ret.(TP)':>11}  {'Ø Ret.(all)':>12}  Wichtigstes Feature")
    print(f"  {line}")

    best_prec = max(r["precision"] for r in results)

    for r in results:
        marker = " ★" if r["precision"] == best_prec else "  "
        print(f"  T+{r['wait']}{marker}  "
              f"{r['n_waves']:>9,}  "
              f"{r['n_true_top']:>10,}  "
              f"{r['precision']:>9.1f}%  "
              f"{r['n_predicted_top']:>10,}  "
              f"{r['avg_ret_tp']:>+10.1f}%  "
              f"{r['avg_ret_all_top']:>+11.1f}%  "
              f"{r['best_feat']}")

    print(f"  {line}")

    # Rendite-Verlust durch späten Einstieg
    if len(results) > 1:
        base_ret = results[0]["avg_ret_all_top"]
        print(f"\n  RENDITE-VERLUST DURCH SPÄTEN EINSTIEG (vs. T0):")
        print(f"  {'─' * 55}")
        for r in results[1:]:
            lost = r["avg_ret_all_top"] - base_ret
            prec_gain = r["precision"] - results[0]["precision"]
            print(f"  T+{r['wait']}:  Rendite {lost:>+5.1f}%  |  "
                  f"Precision {prec_gain:>+5.1f}%")


def print_best_rules(result: dict) -> None:
    feat_cols = result["feat_cols"]
    clf       = result["clf"]

    # Lesbarer Name: snap_atr_pct → atr_pct@Dec | delta_rsi14 → Δrsi14
    readable = []
    for f in feat_cols:
        if f.startswith("snap_"):
            readable.append(f.replace("snap_", "") + "@Dec")
        elif f.startswith("delta_"):
            readable.append("Δ" + f.replace("delta_", ""))
        else:
            readable.append(f)

    wait = result["wait"]
    print(f"\n  ┌─ BESTE REGELN  (T+{wait}  |  Precision {result['precision']:.1f}%) {'─' * 35}")
    print(f"  │  Features: @Dec = Wert am Decision Day | Δ = Veränderung seit T0")
    print(f"  │  Legende:  class 0 = Normaler Trade  |  class 1 = Top-Tier")
    print(f"  │")

    rules = export_text(clf, feature_names=readable, show_weights=True)
    for line in rules.split("\n"):
        if line.strip():
            print(f"  │  {line}")
    print(f"  └{'─' * 72}")

    # Konkrete Schwellen zusammenfassen
    tree = clf.tree_
    print(f"\n  WENN-DANN REGELN  (T+{wait}, nur vergangenheitsbasiert nutzbar):")
    print(f"  {'─' * 65}")
    seen = set()
    for node_id in range(tree.node_count):
        if tree.children_left[node_id] == -1:
            continue
        fi    = tree.feature[node_id]
        thresh = tree.threshold[node_id]
        key    = (fi, round(thresh, 6))
        if key in seen:
            continue
        seen.add(key)
        fname = feat_cols[fi]
        label = readable[fi]
        left_i  = tree.children_left[node_id]
        right_i = tree.children_right[node_id]
        l_top = tree.value[left_i][0][1]  / tree.n_node_samples[left_i]
        r_top = tree.value[right_i][0][1] / tree.n_node_samples[right_i]
        op    = "≤" if l_top > r_top else ">"
        prec  = max(l_top, r_top)
        n     = tree.n_node_samples[node_id]
        flag  = "✓" if fname.startswith("snap_") or fname.startswith("delta_") else "✗"
        print(f"  WENN {label} {op} {thresh:.4f}")
        print(f"       → Top-Tier-Anteil: {prec:.1%}  (Basis: {n} Wellen)  {flag}")
        print()


def print_feature_importances_per_day(results: list[dict]) -> None:
    print(f"\n  ┌─ FEATURE IMPORTANCES PRO WAIT-DAY (Top-3) {'─' * 35}")
    print(f"  │  Format: feature_name (importance)")
    print(f"  │")
    for r in results:
        imp = pd.Series(r["clf"].feature_importances_, index=r["feat_cols"])
        top3 = imp[imp > 0].nlargest(3)
        if top3.empty:
            print(f"  │  T+{r['wait']}:  kein Feature genutzt")
            continue
        parts = []
        for feat, val in top3.items():
            label = (feat.replace("snap_", "")
                        .replace("delta_", "Δ")
                        .replace("dist_sma", "dSMA")
                        .replace("macd_hist_norm", "MACD"))
            parts.append(f"{label} ({val:.3f})")
        print(f"  │  T+{r['wait']}:  " + "  |  ".join(parts))
    print(f"  └{'─' * 72}")


# ==============================================================================
# 8. MAIN
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Late Entry Tradeoff  |  v9.4")
    parser.add_argument("--years",         type=float, default=DEFAULT_YEARS)
    parser.add_argument("--min-profit",    type=float, default=DEFAULT_MIN_PROFIT)
    parser.add_argument("--max-pullback",  type=float, default=DEFAULT_MAX_PB)
    parser.add_argument("--min-remaining", type=float, default=DEFAULT_MIN_REMAIN,
                        help="Mindest-Rest-Rendite ab Decision Day (default 0.10)")
    args = parser.parse_args()

    print("=" * 72)
    print("  LATE ENTRY & CONFIRMATION TRADEOFF  |  v9.4")
    print("=" * 72)
    print(f"""
  Wait-Days:     {WAIT_DAYS}
  Min. Rendite:  {args.min_profit*100:.0f}%  |  Max. Pullback: {args.max_pullback*100:.0f}%
  Min. Restren.: {args.min_remaining*100:.0f}%  (ab Decision Day bis Peak)
  Top-Tier:      Top-{TOP_PCT:.0f}% nach Remaining_Target_Quality
  Modell:        DecisionTree(depth={DT_MAX_DEPTH}, min_leaf={DT_MIN_LEAF})
""")

    # 1. Daten
    print("[1/4] Lade Daten...")
    t0      = time.time()
    tickers = _load_tickers()
    data    = _load_ohlcv(tickers, args.years)
    print(f"  {len(data)} Ticker in {time.time()-t0:.1f}s")

    # 2. Datasets
    print(f"\n[2/4] Wave-Finder + Decision-Day-Datasets aufbauen...")
    t0 = time.time()
    datasets = build_all_datasets(
        data, args.min_profit, args.max_pullback, args.min_remaining)
    print(f"\n  Fertig in {time.time()-t0:.1f}s")
    for w, df in datasets.items():
        clean = df.dropna()
        print(f"  T+{w}: {len(df):>5,} Wellen  ({len(clean):,} ohne NaN)")

    # 3. Training
    print(f"\n[3/4] Modell-Training pro Wait-Day...")
    results = []
    for wait in WAIT_DAYS:
        df = datasets[wait]
        res = train_one_day(df, wait)
        if res is None:
            print(f"  T+{wait}: zu wenig Daten – übersprungen")
            continue
        results.append(res)
        print(f"  T+{wait}: n={res['n_waves']:>5,}  "
              f"Precision={res['precision']:.1f}%  "
              f"n_pred_top={res['n_predicted_top']}  "
              f"Ø Ret(TP)={res['avg_ret_tp']:>+.1f}%")

    if not results:
        print("  Keine Ergebnisse. Parameter prüfen.")
        return

    # 4. Ausgabe
    print(f"\n[4/4] Ausgabe...")
    print_tradeoff_table(results)
    print_feature_importances_per_day(results)

    # Bester Tag nach Precision
    best = max(results, key=lambda r: r["precision"])
    print(f"\n  BESTER EINSTIEGSTAG: T+{best['wait']}  "
          f"(Precision {best['precision']:.1f}%  |  "
          f"n={best['n_waves']:,}  |  "
          f"Ø Rest-Rendite(TP)={best['avg_ret_tp']:+.1f}%)")
    print_best_rules(best)

    # CSV Export
    csv_path = _here / "late_entry_datasets.csv"
    all_dfs  = []
    for wait, df in datasets.items():
        all_dfs.append(df.assign(wait_days=wait))
    pd.concat(all_dfs, ignore_index=True).to_csv(csv_path, index=False)
    print(f"\n  Datensatz gespeichert: {csv_path}")
    print(f"  FERTIG.\n")


if __name__ == "__main__":
    main()
