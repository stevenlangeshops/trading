"""
evaluate_trend_late_entries_v9_5.py
====================================================================================
Trend-Following Tradeoff  |  T+0 bis T+5  |  Anti-Crash-Filter  |  v9.5

Unterschied zu v9.4: Am Decision Day wird ein strikter Anti-Crash-Filter angewandt:
  ► dist_sma200 > 0.0   (Aktie im langfristigen Aufwärtstrend)
  ► atr_pct    < 0.05   (Normale Volatilität, keine Crash-Rebound-Anomalien)

Damit werden COVID-Rebounds und andere Extremereignisse herausgefiltert.
Das verbleibende Universum besteht ausschließlich aus echten Trendfolge-Setups.

Optimaler Tag: Höchste Precision, bei der Ø Rest-Rendite noch > 15% liegt.

Verwendung:
  python evaluate_trend_late_entries_v9_5.py
  python evaluate_trend_late_entries_v9_5.py --sma200-min 0.0 --atr-max 0.05
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
DEFAULT_YEARS      = 7.0
DEFAULT_MIN_PROFIT = 0.15
DEFAULT_MAX_PB     = 0.05
DEFAULT_MIN_REMAIN = 0.10
DEFAULT_SMA200_MIN = 0.0    # Anti-Crash: dist_sma200 > X
DEFAULT_ATR_MAX    = 0.05   # Anti-Crash: atr_pct < X
MIN_WAVE_DAYS      = 5
MIN_DATA_ROWS      = 260
TOP_PCT            = 33.0
WAIT_DAYS          = [0, 1, 2, 3, 4, 5]
DT_MAX_DEPTH       = 3
DT_MIN_LEAF        = 40     # Etwas permissiver als v9.4 (Filter reduziert Datenmenge)
MIN_RET_THRESHOLD  = 15.0   # Mindest-Rest-Rendite für "optimalen Tag"
_RAW_DIR           = _here / "data" / "raw"

BASE_FEATURES = [
    "atr_pct", "bb_width", "dist_sma200", "rsi14",
    "vol_spike", "dist_sma50", "dist_sma20", "macd_hist_norm",
]


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
            if not {"open","high","low","close"}.issubset(df.columns):
                continue
            keep = [c for c in ["open","high","low","close","volume"]
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


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    c      = df["close"]
    vol    = df.get("volume")
    sma20  = c.rolling(20).mean()
    sma50  = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()
    std20  = c.rolling(20).std()
    bb_up  = sma20 + 2.0 * std20
    bb_lo  = sma20 - 2.0 * std20
    bb_r   = (bb_up - bb_lo).replace(0, np.nan)
    ema12  = c.ewm(span=12, adjust=False).mean()
    ema26  = c.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    sig9   = macd.ewm(span=9, adjust=False).mean()
    atr14  = _atr(df, 14)
    sma20v = (vol.rolling(20).mean().replace(0, np.nan)
              if vol is not None else None)
    return pd.DataFrame({
        "atr_pct":       atr14  / c.replace(0, np.nan),
        "bb_width":      bb_r   / c.replace(0, np.nan),
        "dist_sma200":   (c - sma200) / sma200.replace(0, np.nan),
        "rsi14":         _rsi(c, 14),
        "vol_spike":     (vol / sma20v if sma20v is not None
                          else pd.Series(np.nan, index=c.index)),
        "dist_sma50":    (c - sma50)  / sma50.replace(0, np.nan),
        "dist_sma20":    (c - sma20)  / sma20.replace(0, np.nan),
        "macd_hist_norm": (macd - sig9) / c.replace(0, np.nan),
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
            rem = trough_i + int(np.argmax(closes[trough_i:]))
            if closes[rem] > peak_px:
                peak_px = closes[rem]; peak_i = rem
        total_ret = (peak_px - trough_px) / trough_px
        if total_ret >= min_profit and (peak_i - trough_i) >= MIN_WAVE_DAYS:
            waves.append({
                "ticker": ticker, "trough_i": trough_i, "peak_i": peak_i,
                "trough_date": dates[trough_i], "peak_px": float(peak_px),
                "trough_px": float(trough_px), "total_ret": round(total_ret * 100, 2),
            })
            i = peak_i + 1
        else:
            i = trough_i + 1
    return waves


def _rest_drawdown(closes: np.ndarray, start_i: int, end_i: int) -> float:
    if start_i >= end_i:
        return 0.0
    seg = closes[start_i: end_i + 1]
    peak = seg[0]; max_dd = 0.0
    for px in seg[1:]:
        if px > peak:
            peak = px
        dd = (peak - px) / peak * 100
        if dd > max_dd:
            max_dd = dd
    return max_dd


# ==============================================================================
# 4. DATASETS MIT ANTI-CRASH-FILTER
# ==============================================================================

def build_all_datasets(
    data:          dict[str, pd.DataFrame],
    min_profit:    float,
    max_pullback:  float,
    min_remaining: float,
    sma200_min:    float,
    atr_max:       float,
) -> dict[int, pd.DataFrame]:
    """
    Wie v9.4, aber mit Anti-Crash-Filter am Decision Day:
      dist_sma200 > sma200_min  UND  atr_pct < atr_max
    Crash-Rebounds werden herausgefiltert → reines Trendfolge-Universum.
    """
    rows: dict[int, list[dict]] = {w: [] for w in WAIT_DAYS}
    # Separat zählen: gefiltert vs. behalten
    n_filtered: dict[int, int] = {w: 0 for w in WAIT_DAYS}

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
            t0_date = w["trough_date"]
            if t0_date not in date_to_pos:
                continue
            t0_pos = date_to_pos[t0_date]
            if t0_pos < 1:
                continue
            snap_t0 = feats.iloc[t0_pos]
            if snap_t0.isna().any():
                continue

            for wait in WAIT_DAYS:
                dec_i   = t0_i + wait
                dec_pos = t0_pos + wait
                if dec_i >= peak_i or dec_pos >= len(feats):
                    continue

                snap_dec = feats.iloc[dec_pos]
                if snap_dec.isna().any():
                    continue

                # ── Anti-Crash-Filter ─────────────────────────────────────
                dist200 = float(snap_dec["dist_sma200"])
                atr_p   = float(snap_dec["atr_pct"])
                if math.isnan(dist200) or math.isnan(atr_p):
                    continue
                if dist200 <= sma200_min or atr_p >= atr_max:
                    n_filtered[wait] += 1
                    continue

                # Verbleibende Rendite
                close_dec = closes[dec_i]
                if close_dec <= 0:
                    continue
                rem_ret = (peak_px - close_dec) / close_dec * 100
                if rem_ret < min_remaining * 100:
                    continue

                rest_dd = _rest_drawdown(closes, dec_i, peak_i)
                rem_tq  = rem_ret / max(rest_dd, 0.5)

                row = {
                    "ticker": ticker,
                    "trough_date": w["trough_date"],
                    "peak_date": feats.index[min(peak_i, len(feats)-1)],
                    "wait_days": wait,
                    "decision_date": feats.index[dec_pos],
                    "rem_ret_pct": round(rem_ret, 2),
                    "rest_dd_pct": round(rest_dd, 2),
                    "rem_tq": round(rem_tq, 4),
                    "total_ret": w["total_ret"],
                    "dist_sma200_dec": round(dist200, 4),
                    "atr_pct_dec": round(atr_p, 4),
                }
                for feat in BASE_FEATURES:
                    v_dec = float(snap_dec[feat]) if feat in snap_dec.index else float("nan")
                    v_t0  = float(snap_t0[feat])  if feat in snap_t0.index  else float("nan")
                    row[f"snap_{feat}"]  = v_dec
                    row[f"delta_{feat}"] = (v_dec - v_t0
                                            if not (math.isnan(v_dec) or math.isnan(v_t0))
                                            else float("nan"))
                rows[wait].append(row)

        if idx % 60 == 0:
            print(f"  [{idx:>3}/{len(tickers)}]  "
                  + "  ".join(f"w{w}:{len(rows[w]):>4}" for w in WAIT_DAYS))

    print(f"\n  Anti-Crash-Filter (dist_SMA200>{sma200_min:.1f}, ATR<{atr_max:.2f}):")
    for w in WAIT_DAYS:
        total_cands = len(rows[w]) + n_filtered[w]
        pct = n_filtered[w] / total_cands * 100 if total_cands > 0 else 0
        print(f"    T+{w}: {n_filtered[w]:>5} herausgefiltert  "
              f"({pct:.1f}%)  |  {len(rows[w]):>5} behalten")

    return {w: pd.DataFrame(rows[w]) for w in WAIT_DAYS}


# ==============================================================================
# 5. TRAINING + AUSWERTUNG
# ==============================================================================

def train_one_day(df: pd.DataFrame, wait: int) -> dict | None:
    feat_cols = [c for c in df.columns
                 if c.startswith("snap_") or c.startswith("delta_")]
    df_c = df[feat_cols + ["rem_tq", "rem_ret_pct"]].dropna()
    if len(df_c) < DT_MIN_LEAF * 2:
        return None

    threshold = df_c["rem_tq"].quantile(1 - TOP_PCT / 100)
    y  = (df_c["rem_tq"] >= threshold).astype(int)
    X  = df_c[feat_cols]

    clf = DecisionTreeClassifier(
        max_depth=DT_MAX_DEPTH, min_samples_leaf=DT_MIN_LEAF,
        criterion="gini", random_state=42,
    )
    clf.fit(X, y)
    y_pred = clf.predict(X)

    prec        = precision_score(y, y_pred, zero_division=0)
    tp_mask     = (y == 1) & (y_pred == 1)
    avg_ret_tp  = float(df_c.loc[tp_mask, "rem_ret_pct"].mean()) if tp_mask.sum() > 0 else 0.0
    avg_ret_all = float(df_c.loc[y == 1, "rem_ret_pct"].mean())

    imp       = pd.Series(clf.feature_importances_, index=feat_cols)
    best_feat = imp.idxmax() if imp.max() > 0 else "—"
    best_readable = (best_feat
                     .replace("snap_", "")
                     .replace("delta_", "Δ")
                     .replace("dist_sma", "dSMA")
                     .replace("macd_hist_norm", "MACD"))
    return {
        "wait":            wait,
        "n_waves":         len(df_c),
        "n_true_top":      int(y.sum()),
        "threshold_tq":    round(threshold, 3),
        "precision":       round(prec * 100, 1),
        "n_predicted_top": int(y_pred.sum()),
        "avg_ret_tp":      round(avg_ret_tp,  1),
        "avg_ret_all_top": round(avg_ret_all, 1),
        "best_feat":       best_readable,
        "clf":             clf,
        "feat_cols":       feat_cols,
        "df_clean":        df_c,
        "y":               y,
    }


# ==============================================================================
# 6. AUSGABE
# ==============================================================================

def _readable(feat: str) -> str:
    if feat.startswith("snap_"):
        return feat.replace("snap_", "") + "@Dec"
    return "Δ" + feat.replace("delta_", "")


def print_tradeoff_table(
    results: list[dict], best: dict,
    sma200_min: float, atr_max: float,
) -> None:
    sep  = "=" * 88
    line = "─" * 88
    print(f"\n{sep}")
    print(f"  TREND-FOLLOWING TRADEOFF  |  Anti-Crash-Filter  |  v9.5")
    print(sep)
    print(f"""
  Anti-Crash-Filter:  dist_SMA200 > {sma200_min:.1f}  UND  ATR% < {atr_max*100:.0f}%
  → Filtert Crash-Rebounds heraus; nur echte Trendfolge-Setups verbleiben.
  Modell:  DecisionTree(depth={DT_MAX_DEPTH}, min_leaf={DT_MIN_LEAF})
  Optimum: Höchste Precision mit Ø Rest-Rendite ≥ {MIN_RET_THRESHOLD:.0f}%
""")
    print(f"   {'Wait':>5}  {'n Wellen':>9}  {'n Top-Tier':>10}  "
          f"{'Precision':>10}  {'n Pred.Top':>10}  "
          f"{'Ø Ret(TP)':>10}  {'Ø Ret(all)':>11}  Wichtigstes Feature")
    print(f"  {line}")

    for r in results:
        is_best   = r["wait"] == best["wait"]
        ret_ok    = r["avg_ret_all_top"] >= MIN_RET_THRESHOLD
        marker    = " ★" if is_best else ("  " if ret_ok else " ✗")
        ret_warn  = "" if ret_ok else " (<15%)"
        print(f"  T+{r['wait']}{marker}  "
              f"{r['n_waves']:>9,}  "
              f"{r['n_true_top']:>10,}  "
              f"{r['precision']:>9.1f}%  "
              f"{r['n_predicted_top']:>10,}  "
              f"{r['avg_ret_tp']:>+9.1f}%  "
              f"{r['avg_ret_all_top']:>+10.1f}%{ret_warn}  "
              f"{r['best_feat']}")

    print(f"  {line}")

    if len(results) > 1:
        base = results[0]
        print(f"\n  TRADEOFF vs. T+0  (Rendite-Verlust | Precision-Gewinn):")
        print(f"  {'─' * 58}")
        for r in results[1:]:
            ret_diff  = r["avg_ret_all_top"] - base["avg_ret_all_top"]
            prec_diff = r["precision"]        - base["precision"]
            flag = " ← OPTIMUM" if r["wait"] == best["wait"] else ""
            print(f"  T+{r['wait']}:  "
                  f"Rendite {ret_diff:>+5.1f}%  |  "
                  f"Precision {prec_diff:>+5.1f}%{flag}")


def print_best_rules(result: dict) -> None:
    feat_cols = result["feat_cols"]
    clf       = result["clf"]
    readable  = [_readable(f) for f in feat_cols]
    wait      = result["wait"]

    print(f"\n  ┌─ OPTIMALE REGELN  (T+{wait}  |  Precision {result['precision']:.1f}%  |  "
          f"Ø Rendite verbleibend {result['avg_ret_all_top']:+.1f}%) {'─' * 20}")
    print(f"  │  @Dec = Wert am Decision Day  |  Δ = Veränderung seit T0")
    print(f"  │  class 0 = Normaler Trade  |  class 1 = Top-Tier")
    print(f"  │")
    for line in export_text(clf, feature_names=readable, show_weights=True).split("\n"):
        if line.strip():
            print(f"  │  {line}")
    print(f"  └{'─' * 75}")

    # Konkrete Wenn-Dann-Regeln
    tree = clf.tree_
    seen = set()
    print(f"\n  WENN-DANN REGELN  (T+{wait}):")
    print(f"  {'─' * 65}")
    for node_id in range(tree.node_count):
        if tree.children_left[node_id] == -1:
            continue
        fi     = tree.feature[node_id]
        thresh = tree.threshold[node_id]
        key    = (fi, round(thresh, 8))
        if key in seen:
            continue
        seen.add(key)
        left_i  = tree.children_left[node_id]
        right_i = tree.children_right[node_id]
        l_top   = tree.value[left_i][0][1]  / tree.n_node_samples[left_i]
        r_top   = tree.value[right_i][0][1] / tree.n_node_samples[right_i]
        op      = "≤" if l_top > r_top else ">"
        prec    = max(l_top, r_top)
        n       = tree.n_node_samples[node_id]
        label   = readable[fi]
        print(f"  WENN {label} {op} {thresh:.4f}")
        print(f"       → Top-Tier-Anteil: {prec:.1%}  (Basis: {n} Trades)")
        print()


def print_feature_importances(results: list[dict]) -> None:
    print(f"\n  ┌─ TOP-3 FEATURES PRO WAIT-DAY {'─' * 42}")
    print(f"  │")
    for r in results:
        imp  = pd.Series(r["clf"].feature_importances_, index=r["feat_cols"])
        top3 = imp[imp > 0].nlargest(3)
        if top3.empty:
            print(f"  │  T+{r['wait']}:  —")
            continue
        parts = [f"{_readable(f).replace('dist_sma','dSMA').replace('macd_hist_norm','MACD')}"
                 f" ({v:.3f})" for f, v in top3.items()]
        print(f"  │  T+{r['wait']}:  " + "  |  ".join(parts))
    print(f"  └{'─' * 72}")


# ==============================================================================
# 7. MAIN
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trend-Following Tradeoff  |  Anti-Crash-Filter  |  v9.5")
    parser.add_argument("--years",         type=float, default=DEFAULT_YEARS)
    parser.add_argument("--min-profit",    type=float, default=DEFAULT_MIN_PROFIT)
    parser.add_argument("--max-pullback",  type=float, default=DEFAULT_MAX_PB)
    parser.add_argument("--min-remaining", type=float, default=DEFAULT_MIN_REMAIN)
    parser.add_argument("--sma200-min",    type=float, default=DEFAULT_SMA200_MIN,
                        help="Anti-Crash: dist_sma200 muss > X sein (default 0.0)")
    parser.add_argument("--atr-max",       type=float, default=DEFAULT_ATR_MAX,
                        help="Anti-Crash: atr_pct muss < X sein (default 0.05)")
    args = parser.parse_args()

    print("=" * 72)
    print("  TREND-FOLLOWING TRADEOFF  |  Anti-Crash-Filter  |  v9.5")
    print("=" * 72)
    print(f"""
  Anti-Crash-Filter:  dist_SMA200 > {args.sma200_min:.2f}  |  ATR% < {args.atr_max*100:.0f}%
  Wave-Filter:        min_profit={args.min_profit*100:.0f}%  |  max_pullback={args.max_pullback*100:.0f}%
  Min. Restrendite:   {args.min_remaining*100:.0f}%  |  Top-Tier: Top-{TOP_PCT:.0f}%
  Modell:             DecisionTree(depth={DT_MAX_DEPTH}, min_leaf={DT_MIN_LEAF})
  Universum:          260 US-Aktien  |  {args.years:.0f} Jahre
""")

    # 1. Daten
    print("[1/4] Lade Daten...")
    t0      = time.time()
    tickers = _load_tickers()
    data    = _load_ohlcv(tickers, args.years)
    print(f"  {len(data)} Ticker in {time.time()-t0:.1f}s")

    # 2. Datasets
    print(f"\n[2/4] Wave-Finder + Anti-Crash-Filter + Datasets...")
    t0 = time.time()
    datasets = build_all_datasets(
        data, args.min_profit, args.max_pullback,
        args.min_remaining, args.sma200_min, args.atr_max,
    )
    print(f"\n  Aufbau in {time.time()-t0:.1f}s")

    # 3. Training
    print(f"\n[3/4] Modell-Training pro Wait-Day...")
    results = []
    for wait in WAIT_DAYS:
        res = train_one_day(datasets[wait], wait)
        if res is None:
            print(f"  T+{wait}: zu wenig Daten – übersprungen")
            continue
        results.append(res)
        print(f"  T+{wait}: n={res['n_waves']:>5,}  "
              f"Prec={res['precision']:.1f}%  "
              f"Pred_Top={res['n_predicted_top']:>4}  "
              f"Ø Ret(TP)={res['avg_ret_tp']:>+.1f}%  "
              f"Ø Ret(all)={res['avg_ret_all_top']:>+.1f}%")

    if not results:
        print("  Keine Ergebnisse – Anti-Crash-Filter zu streng? Parameter lockern.")
        return

    # Optimalen Tag bestimmen: höchste Precision mit Ø Rest-Rendite >= MIN_RET_THRESHOLD
    valid = [r for r in results if r["avg_ret_all_top"] >= MIN_RET_THRESHOLD]
    best  = max(valid, key=lambda r: r["precision"]) if valid else max(results, key=lambda r: r["precision"])

    # 4. Ausgabe
    print(f"\n[4/4] Ausgabe...")
    print_tradeoff_table(results, best, args.sma200_min, args.atr_max)
    print_feature_importances(results)
    print(f"\n  OPTIMALER EINSTIEGSTAG: T+{best['wait']}  "
          f"(Precision {best['precision']:.1f}%  |  "
          f"Ø Restrendite {best['avg_ret_all_top']:+.1f}%  |  "
          f"n={best['n_waves']:,})")
    print_best_rules(best)

    # CSV
    csv_path = _here / "trend_late_entry_datasets.csv"
    all_dfs  = []
    for wait, df in datasets.items():
        all_dfs.append(df.assign(wait_days_key=wait))
    pd.concat(all_dfs, ignore_index=True).to_csv(csv_path, index=False)
    print(f"\n  Datensatz: {csv_path}")
    print(f"  FERTIG.\n")


if __name__ == "__main__":
    main()
