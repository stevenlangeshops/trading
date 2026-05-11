"""
extract_ideal_trades_v9.py
====================================================================================
Target Generation + Feature Engineering  |  Paradigmenwechsel v9.0

Statt Regeln zu raten (Backtesting) analysieren wir RETROSPEKTIV:
  1. Welche Aufwärtswellen wären in der Geschichte "perfekt" gewesen? → Targets
  2. Welche technischen Indikatoren lagen an Tag 0 dieser Wellen vor? → Features
  3. Welche Features korrelieren am stärksten mit der Rendite-Geschwindigkeit? → Edge

Kernkonzept:
  ► Wave-Finder:  Segmentiert jede Preisreihe in qualifizierende Aufwärtswellen.
                  Kein Look-Ahead-Bias im Feature-Teil – Wellen sind rückwärtige Labels.
  ► Feature-Snapshot: Alle Indikatoren AUSSCHLIESSLICH aus Daten bis Tag 0 der Welle.
  ► Velocity als Zielgröße: Netto_Rendite_% / Dauer_Tage → misst "Kraft" der Welle.
  ► Spearman-Korrelation: Robuster Rang-Zusammenhang (kein Normalverteilungs-Annahme).
  ► Bucket-Analyse: Top-25% vs. Bottom-25% der Velocity – Feature-Profile vergleichen.

Outputs:
  ideal_trades_features.csv  – Trainings-Datensatz für Decision Trees / ML
  Console-Report             – Korrelationstabelle, Bucket-Analyse, Top-Wellen

Verwendung:
  python extract_ideal_trades_v9.py
  python extract_ideal_trades_v9.py --years 7 --min-profit 0.15 --max-pullback 0.05
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
DEFAULT_MIN_PROFIT  = 0.15   # Mindest-Rendite einer Welle (15 %)
DEFAULT_MAX_PB      = 0.05   # Max. Pullback vom Peak bevor Welle endet (5 %)
MIN_WAVE_DAYS       = 5      # Kürzeste erlaubte Welle in Handelstagen
MIN_DATA_ROWS       = 260    # Mindest-Datenreihen pro Ticker
_RAW_DIR   = _here / "data" / "raw"
_OUT_CSV   = _here / "ideal_trades_features.csv"

FEATURE_COLS = [
    "feat_dist_sma20",
    "feat_dist_sma50",
    "feat_dist_sma200",
    "feat_rsi14",
    "feat_bb_width",
    "feat_bb_pos",
    "feat_macd_hist_norm",
    "feat_vol_spike",
    "feat_atr_pct",
    "feat_adx14",
    "feat_roc5",
    "feat_roc20",
    "feat_trend_align",
    "feat_sma20_slope",
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
            if not {"open", "high", "low", "close"}.issubset(df.columns):
                continue
            keep = [c for c in ["open","high","low","close","volume"]
                    if c in df.columns]
            data[ticker] = df[keep].copy()
        except Exception:
            pass
    return data


# ==============================================================================
# 2. WAVE FINDER  (Target Generation)
# ==============================================================================

def find_ideal_waves(
    ticker:       str,
    df:           pd.DataFrame,
    min_profit:   float = DEFAULT_MIN_PROFIT,
    max_pullback: float = DEFAULT_MAX_PB,
) -> list[dict]:
    """
    Findet retrospektiv alle qualifizierenden Aufwärtswellen in einer Preisreihe.

    Algorithmus (2-Phasen-Suche):
      Phase 1 – Trough-Identifikation:
        Ab Startpunkt vorwärts gehen. Solange der Kurs fällt, den Trough nach unten
        aktualisieren. Sobald der Kurs um max_pullback% vom Trough gestiegen ist,
        ist der Trough "gesichert" (keine weiteres Absinken möglich ohne neuen Trough).

      Phase 2 – Wellen-Tracking:
        Vom gesicherten Trough aus den Peak nachziehen. Die Welle endet, wenn der
        Kurs um mehr als max_pullback% vom laufenden Peak abfällt.

    Look-Ahead-Bias: Der Wave-Finder verwendet Zukunftsdaten zur LABEL-Generierung.
    Das ist korrekt – wir suchen retrospektive Targets. Die Features werden SEPARAT
    ausschließlich aus Vergangenheitsdaten bis Tag 0 berechnet.

    Rückgabe: Liste von Wellen-Dictionaries (für alle qualifizierenden Wellen).
    """
    closes = df["close"].values
    dates  = df.index
    n      = len(closes)
    waves  = []

    i = 0
    while i < n - MIN_WAVE_DAYS:

        # ── Phase 1: Trough finden ──────────────────────────────────────────
        trough_px = closes[i]
        trough_i  = i
        j         = i + 1

        while j < n:
            px = closes[j]
            if px < trough_px:
                # Trough noch tiefer
                trough_px = px
                trough_i  = j
                j += 1
                continue
            # Preisanstieg vom Trough aus
            if (px - trough_px) / trough_px >= max_pullback:
                break   # Trough ist gesichert
            j += 1

        if j >= n:
            break   # Ende der Daten

        # ── Phase 2: Welle vom Trough tracken ──────────────────────────────
        peak_px = closes[trough_i]
        peak_i  = trough_i
        wave_ended = False

        for k in range(trough_i + 1, n):
            px = closes[k]
            if px > peak_px:
                peak_px = px
                peak_i  = k
            drawdown = (peak_px - px) / peak_px
            if drawdown > max_pullback:
                wave_ended = True
                break   # Welle beendet – Pullback > max_pullback

        if not wave_ended:
            peak_i = np.argmax(closes[trough_i:]) + trough_i   # Restlauf bis Ende

        total_ret    = (peak_px - trough_px) / trough_px
        duration_cal = (dates[peak_i] - dates[trough_i]).days
        duration_td  = peak_i - trough_i   # Handelstage

        if (total_ret >= min_profit
                and duration_td >= MIN_WAVE_DAYS
                and duration_cal >= 1):
            velocity = (total_ret * 100) / duration_cal
            waves.append({
                "ticker":       ticker,
                "start_date":   dates[trough_i],
                "end_date":     dates[peak_i],
                "start_price":  round(float(trough_px), 4),
                "peak_price":   round(float(peak_px),   4),
                "duration_d":   duration_cal,
                "duration_td":  duration_td,
                "return_pct":   round(total_ret * 100, 2),
                "velocity":     round(velocity, 5),
            })
            i = peak_i + 1   # Nächste Suche startet nach dem Peak
        else:
            i = trough_i + 1  # Kein qualifizierender Trade – einen Schritt weiter


    return waves


# ==============================================================================
# 3. FEATURE ENGINEERING  (technische Indikatoren als relative Werte)
# ==============================================================================

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta  = series.diff()
    gain   = delta.clip(lower=0)
    loss   = (-delta).clip(lower=0)
    avg_g  = gain.ewm(com=period - 1, adjust=False).mean()
    avg_l  = loss.ewm(com=period - 1, adjust=False).mean()
    rs     = avg_g / avg_l.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1
    ).max(axis=1)
    up_m  = h.diff()
    dn_m  = -l.diff()
    pdm   = np.where((up_m > dn_m) & (up_m > 0),  up_m.values, 0.0)
    ndm   = np.where((dn_m > up_m) & (dn_m > 0),  dn_m.values, 0.0)
    atr_s = pd.Series(tr.values,  index=c.index).ewm(span=period, adjust=False).mean()
    pdi   = 100 * pd.Series(pdm, index=c.index).ewm(span=period, adjust=False).mean() / atr_s
    ndi   = 100 * pd.Series(ndm, index=c.index).ewm(span=period, adjust=False).mean() / atr_s
    dx    = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return dx.ewm(span=period, adjust=False).mean()


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet alle relativen technischen Features für einen Ticker.
    Alle Features sind dimensionslos (kein absoluter Preis-Level).
    Kein Look-Ahead-Bias: jede Zeile nutzt nur Daten bis zum jeweiligen Tag.
    """
    c   = df["close"]
    h   = df["high"]
    l   = df["low"]
    vol = df.get("volume")

    # ── Gleitende Durchschnitte ──────────────────────────────────────────────
    sma20  = c.rolling(20).mean()
    sma50  = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()

    # Relative Distanz zum Durchschnitt (= Momentum-Kraft)
    dist_sma20  = (c - sma20)  / sma20.replace(0, np.nan)
    dist_sma50  = (c - sma50)  / sma50.replace(0, np.nan)
    dist_sma200 = (c - sma200) / sma200.replace(0, np.nan)

    # SMA20-Steigung (5-Tage Δ, normiert) – misst ob Trend beschleunigt
    sma20_slope = sma20.pct_change(5)

    # ── Momentum / Oszillatoren ──────────────────────────────────────────────
    rsi14    = _rsi(c, 14)
    roc5     = c.pct_change(5)
    roc20    = c.pct_change(20)

    # MACD Histogramm normiert auf Preis (macht ihn skalenunabhängig)
    ema12    = c.ewm(span=12, adjust=False).mean()
    ema26    = c.ewm(span=26, adjust=False).mean()
    macd     = ema12 - ema26
    sig9     = macd.ewm(span=9, adjust=False).mean()
    macd_norm = (macd - sig9) / c.replace(0, np.nan)

    # ── Volatilität ──────────────────────────────────────────────────────────
    std20    = c.rolling(20).std()
    bb_up    = sma20 + 2.0 * std20
    bb_lo    = sma20 - 2.0 * std20
    bb_range = (bb_up - bb_lo).replace(0, np.nan)
    bb_width = bb_range / c.replace(0, np.nan)   # Klein = Squeeze = VCP
    bb_pos   = (c - bb_lo) / bb_range            # 0=unteres Band, 1=oberes Band

    atr14    = _atr(df, 14)
    atr_pct  = atr14 / c.replace(0, np.nan)      # Relative Schwankungsbreite

    adx14    = _adx(df, 14)

    # ── Volumen ──────────────────────────────────────────────────────────────
    if vol is not None:
        sma20_v   = vol.rolling(20).mean().replace(0, np.nan)
        vol_spike = vol / sma20_v
    else:
        vol_spike = pd.Series(np.nan, index=c.index)

    # ── Struktur-Feature ────────────────────────────────────────────────────
    # 1 = vollständige Trendausrichtung (SMA20 > SMA50 > SMA200)
    trend_align = ((sma20 > sma50) & (sma50 > sma200)).astype(float)

    return pd.DataFrame({
        "feat_dist_sma20":      dist_sma20,
        "feat_dist_sma50":      dist_sma50,
        "feat_dist_sma200":     dist_sma200,
        "feat_rsi14":           rsi14,
        "feat_bb_width":        bb_width,
        "feat_bb_pos":          bb_pos,
        "feat_macd_hist_norm":  macd_norm,
        "feat_vol_spike":       vol_spike,
        "feat_atr_pct":         atr_pct,
        "feat_adx14":           adx14,
        "feat_roc5":            roc5,
        "feat_roc20":           roc20,
        "feat_trend_align":     trend_align,
        "feat_sma20_slope":     sma20_slope,
    }, index=c.index)


# ==============================================================================
# 4. MERGE + SPEARMAN ANALYSE
# ==============================================================================

def build_dataset(
    data:        dict[str, pd.DataFrame],
    min_profit:  float,
    max_pullback: float,
) -> pd.DataFrame:
    """
    Hauptfunktion: Waves finden + Features an Tag 0 snappen + zusammenführen.
    """
    all_rows: list[dict] = []
    tickers  = sorted(data.keys())

    for idx, ticker in enumerate(tickers, 1):
        df  = data[ticker]

        # a) Features (volle Zeitreihe, kein Look-Ahead in den Features selbst)
        feats = compute_features(df)

        # b) Waves finden (verwendet Zukunftsdaten → labels/targets, OK)
        waves = find_ideal_waves(ticker, df, min_profit, max_pullback)

        if not waves:
            continue

        # c) Für jede Welle: Feature-Snapshot am Starttag
        for w in waves:
            sd = w["start_date"]
            if sd not in feats.index:
                continue
            snap = feats.loc[sd].to_dict()
            all_rows.append({**w, **snap})

        if idx % 50 == 0:
            print(f"  [{idx:>3}/{len(tickers)}]  "
                  f"{ticker:<6}  Waves bisher: {len(all_rows)}")

    return pd.DataFrame(all_rows)


def spearman_corr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet Spearman-Rang-Korrelation jedes Features mit 'velocity'.
    Spearman ist robust gegenüber Ausreißern und Nicht-Normalverteilung.
    """
    target = "velocity"
    rows   = []
    for col in FEATURE_COLS:
        if col not in df.columns:
            continue
        sub  = df[[target, col]].dropna()
        if len(sub) < 30:
            rows.append({"feature": col, "spearman_r": np.nan,
                         "n": len(sub), "p_val": np.nan})
            continue
        # Rangkorrelation = Pearson auf Rängen
        r_tgt = sub[target].rank()
        r_fea = sub[col].rank()
        corr  = r_tgt.corr(r_fea)
        # p-Wert Approximation via t-Verteilung
        n   = len(sub)
        t   = corr * math.sqrt((n - 2) / max(1 - corr**2, 1e-12))
        # Einfache Näherung (zweiseitig)
        from scipy.special import stdtr
        p = 2 * float(stdtr(n - 2, -abs(t)))
        rows.append({"feature": col, "spearman_r": round(corr, 4),
                     "n": n, "p_val": round(p, 5)})

    result = pd.DataFrame(rows).sort_values("spearman_r",
                                            key=abs, ascending=False)
    return result.reset_index(drop=True)


def bucket_analysis(df: pd.DataFrame, q: float = 0.25) -> pd.DataFrame:
    """
    Vergleicht Feature-Profile im Top-Quartil (schnellste Wellen)
    vs. Bottom-Quartil (langsamste Wellen).
    """
    thr_hi = df["velocity"].quantile(1 - q)
    thr_lo = df["velocity"].quantile(q)
    top    = df[df["velocity"] >= thr_hi]
    bot    = df[df["velocity"] <= thr_lo]

    rows = []
    for col in FEATURE_COLS:
        if col not in df.columns:
            continue
        t_med = top[col].median()
        b_med = bot[col].median()
        diff  = t_med - b_med
        rows.append({
            "feature":    col,
            "top25_med":  round(t_med, 4),
            "bot25_med":  round(b_med, 4),
            "delta":      round(diff,  4),
        })

    result = pd.DataFrame(rows).sort_values("delta", key=abs, ascending=False)
    return result.reset_index(drop=True)


# ==============================================================================
# 5. CONSOLE REPORT
# ==============================================================================

def print_report(
    df:        pd.DataFrame,
    corr_df:   pd.DataFrame,
    bucket_df: pd.DataFrame,
    min_profit: float,
    max_pullback: float,
) -> None:
    sep  = "=" * 74
    line = "─" * 74

    # ── Wave-Statistiken ─────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  IDEAL WAVE ANALYSE  |  v9.0  |  Target Generation + Feature Mining")
    print(sep)

    years_span = (df["start_date"].max() - df["start_date"].min()).days / 365.25
    tickers_w  = df["ticker"].nunique()
    print(f"""
  Wave-Finder Parameter:
    Min. Rendite:   {min_profit*100:.0f}%
    Max. Pullback:  {max_pullback*100:.0f}%  (vom Peak)

  Datensatz-Übersicht:
    Gefundene Wellen:   {len(df):,}
    Ticker mit Wellen:  {tickers_w}
    Zeitspanne:         {years_span:.1f} Jahre
    Features:           {len([c for c in FEATURE_COLS if c in df.columns])}
""")

    ret   = df["return_pct"]
    vel   = df["velocity"]
    dur   = df["duration_d"]

    print(f"  ┌─ WELLEN-VERTEILUNG {'─' * 50}")
    print(f"  │  Rendite:    Ø {ret.mean():>6.1f}%  |  Median {ret.median():>5.1f}%  "
          f"|  Max {ret.max():>6.1f}%  |  P75 {ret.quantile(0.75):>5.1f}%")
    print(f"  │  Dauer:      Ø {dur.mean():>5.0f}d  |  Median {dur.median():>5.0f}d  "
          f"|  Max {dur.max():>5.0f}d")
    print(f"  │  Velocity:   Ø {vel.mean():>6.3f}  |  Median {vel.median():>5.3f}  "
          f"|  Max {vel.max():>6.3f}")
    print(f"  │")

    # Jahresverteilung der Wellen-Starts
    by_year = df.groupby(df["start_date"].dt.year)["velocity"].agg(
        ["count", "mean", "median"])
    print(f"  │  Jahr    N-Wellen   Ø Velocity   Med. Velocity")
    print(f"  │  {'─' * 46}")
    for year, row in by_year.iterrows():
        print(f"  │  {year}    {int(row['count']):>6}     "
              f"{row['mean']:>8.3f}     {row['median']:>8.3f}")
    print(f"  └{'─' * 57}\n")

    # ── Spearman-Korrelation ──────────────────────────────────────────────────
    print(f"  ┌─ SPEARMAN-KORRELATION: Feature → Velocity {'─' * 28}")
    print(f"  │  (Positiv = Hoher Feature-Wert → schnelle Welle | n = {len(df):,})")
    print(f"  │")
    print(f"  │  {'#':>3}  {'Feature':<24}  {'r':>8}  {'n':>6}  {'p-Wert':>8}  Stärke")
    print(f"  │  {'─' * 60}")
    for rank, row in corr_df.iterrows():
        r       = row["spearman_r"]
        p       = row["p_val"]
        star    = "***" if p < 0.001 else ("** " if p < 0.01 else
                  ("*  " if p < 0.05 else "   "))
        bar_len = int(abs(r) * 20)
        bar     = ("▶" * bar_len) if r >= 0 else ("◀" * bar_len)
        p_str   = f"{p:.5f}" if not math.isnan(p) else "    —   "
        print(f"  │  {rank+1:>3}. {row['feature']:<24}  {r:>+8.4f}  "
              f"{int(row['n']):>6}  {p_str:>8}  {star} {bar}")
    print(f"  └{'─' * 57}\n")

    # ── Bucket-Analyse ────────────────────────────────────────────────────────
    thr_hi = df["velocity"].quantile(0.75)
    thr_lo = df["velocity"].quantile(0.25)
    n_top  = (df["velocity"] >= thr_hi).sum()
    n_bot  = (df["velocity"] <= thr_lo).sum()

    print(f"  ┌─ BUCKET-ANALYSE: Top-25% vs. Bottom-25% Velocity {'─' * 20}")
    print(f"  │  Top-25%:    Velocity ≥ {thr_hi:.3f}  (n={n_top})")
    print(f"  │  Bottom-25%: Velocity ≤ {thr_lo:.3f}  (n={n_bot})")
    print(f"  │")
    print(f"  │  {'Feature':<24}  {'Top25 Median':>13}  "
          f"{'Bot25 Median':>13}  {'Delta':>8}  Richtung")
    print(f"  │  {'─' * 68}")
    for _, row in bucket_df.head(14).iterrows():
        d     = row["delta"]
        arrow = "↑ höher in Top" if d > 0 else "↓ höher in Bot"
        print(f"  │  {row['feature']:<24}  {row['top25_med']:>13.4f}  "
              f"{row['bot25_med']:>13.4f}  {d:>+8.4f}  {arrow}")
    print(f"  └{'─' * 57}\n")

    # ── Top 10 schnellste Wellen ──────────────────────────────────────────────
    top10 = df.nlargest(10, "velocity")
    print(f"  ┌─ TOP 10 SCHNELLSTE WELLEN (nach Velocity) {'─' * 27}")
    print(f"  │  {'Ticker':<7}  {'Start':>10}  {'Ende':>10}  "
          f"{'Dauer':>6}  {'Rendite':>8}  {'Velocity':>9}")
    print(f"  │  {'─' * 60}")
    for _, row in top10.iterrows():
        print(f"  │  {row['ticker']:<7}  "
              f"{str(row['start_date'])[:10]:>10}  "
              f"{str(row['end_date'])[:10]:>10}  "
              f"{int(row['duration_d']):>5}d  "
              f"{row['return_pct']:>+7.1f}%  "
              f"{row['velocity']:>9.4f}")
    print(f"  └{'─' * 57}\n")

    # ── Key Insights ─────────────────────────────────────────────────────────
    top_feat = corr_df.iloc[0]
    bot_feat = corr_df.iloc[-1]
    print(f"  KEY INSIGHTS:")
    print(f"  {'─' * 57}")
    print(f"  Stärkste positive Korrelation: "
          f"{top_feat['feature']}  (r={top_feat['spearman_r']:+.4f})")
    print(f"  Stärkste negative Korrelation: "
          f"{bot_feat['feature']}  (r={bot_feat['spearman_r']:+.4f})")
    print(f"")
    pos_corr = corr_df[corr_df["spearman_r"] > 0]
    neg_corr = corr_df[corr_df["spearman_r"] < 0]
    print(f"  Positive Features ({len(pos_corr)}):  "
          + "  ".join(f["feature"].replace("feat_","") for _, f in pos_corr.iterrows()))
    print(f"  Negative Features ({len(neg_corr)}):  "
          + "  ".join(f["feature"].replace("feat_","") for _, f in neg_corr.iterrows()))
    print(f"")


# ==============================================================================
# 6. MAIN
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ideal Wave Extraction + Feature Mining  |  v9.0")
    parser.add_argument("--years",       type=float, default=DEFAULT_YEARS)
    parser.add_argument("--min-profit",  type=float, default=DEFAULT_MIN_PROFIT,
                        help="Mindest-Rendite einer Welle (default 0.15 = 15%%)")
    parser.add_argument("--max-pullback",type=float, default=DEFAULT_MAX_PB,
                        help="Max. Pullback vom Peak bis Wellen-Ende (default 0.05)")
    args = parser.parse_args()

    print("=" * 74)
    print("  EXTRACT IDEAL TRADES  |  v9.0  |  Target Generation + Feature Mining")
    print("=" * 74)
    print(f"""
  Wave-Parameter:   min_profit={args.min_profit*100:.0f}%  |  max_pullback={args.max_pullback*100:.0f}%
  Zeitraum:         {args.years:.0f} Jahre
  Universum:        260 US-Aktien
""")

    # 1. Daten laden
    print("[1/5] Lade Daten...")
    t0      = time.time()
    tickers = _load_tickers()
    data    = _load_ohlcv(tickers, args.years)
    print(f"  {len(data)} Ticker geladen in {time.time()-t0:.1f}s")

    # 2 + 3. Waves + Features + Merge
    print(f"\n[2/5] Wave-Finder + Feature-Engineering + Merge...")
    print(f"  (Wave-Finder nutzt Zukunftsdaten für Labels – kein Bias in Features)")
    t0 = time.time()
    df = build_dataset(data, args.min_profit, args.max_pullback)
    elapsed = time.time() - t0

    if df.empty:
        print("  !! Keine qualifizierenden Wellen gefunden. Parameter überprüfen.")
        return

    print(f"\n  {len(df):,} Wellen gefunden in {elapsed:.1f}s")
    print(f"  {df['ticker'].nunique()} Ticker mit mindestens einer Welle")

    # NaN-Audit
    nan_rate = df[FEATURE_COLS].isna().mean() * 100
    high_nan = nan_rate[nan_rate > 20]
    if not high_nan.empty:
        print(f"\n  WARNUNG: Features mit >20% NaN:")
        for col, pct in high_nan.items():
            print(f"    {col}: {pct:.1f}%")

    # Zeilen ohne NaN im Zielvariablen-Bereich
    df_clean = df.dropna(subset=["velocity"] + FEATURE_COLS)
    print(f"  Vollständige Zeilen (kein NaN): {len(df_clean):,}  "
          f"({len(df_clean)/len(df)*100:.1f}%)")

    # 4. Spearman-Korrelation
    print(f"\n[3/5] Spearman-Korrelationsanalyse...")
    try:
        corr_df = spearman_corr(df_clean)
    except ImportError:
        # Fallback ohne scipy (p-Wert nicht verfügbar)
        print("  scipy nicht verfügbar – p-Werte werden nicht berechnet")
        rows = []
        for col in FEATURE_COLS:
            if col not in df_clean.columns:
                continue
            sub  = df_clean[["velocity", col]].dropna()
            r    = sub["velocity"].rank().corr(sub[col].rank())
            rows.append({"feature": col, "spearman_r": round(r, 4),
                         "n": len(sub), "p_val": float("nan")})
        corr_df = pd.DataFrame(rows).sort_values(
            "spearman_r", key=abs, ascending=False).reset_index(drop=True)

    # 5. Bucket-Analyse
    print(f"\n[4/5] Bucket-Analyse (Top-25%% vs. Bottom-25%% Velocity)...")
    bucket_df = bucket_analysis(df_clean)

    # 6. Report
    print(f"\n[5/5] Ausgabe...")
    print_report(df_clean, corr_df, bucket_df, args.min_profit, args.max_pullback)

    # CSV speichern (alle Spalten: wave-info + features)
    out_cols = (
        ["ticker", "start_date", "end_date",
         "start_price", "peak_price",
         "duration_d", "duration_td", "return_pct", "velocity"]
        + [c for c in FEATURE_COLS if c in df_clean.columns]
    )
    df_clean[out_cols].sort_values(
        ["start_date", "ticker"]
    ).to_csv(_OUT_CSV, index=False)
    print(f"  Datensatz gespeichert: {_OUT_CSV}")
    print(f"  Zeilen: {len(df_clean):,}  |  Spalten: {len(out_cols)}")
    print(f"\n  FERTIG. Nächster Schritt: Decision Tree / Random Forest auf")
    print(f"  'ideal_trades_features.csv' trainieren, um optimale Entry-Regeln")
    print(f"  zu extrahieren.\n")


if __name__ == "__main__":
    main()
