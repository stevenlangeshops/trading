"""
research_signals.py
====================================================================================
Signal-Edge Research fuer das 260-Aktien-Universum  |  Trading v6 Vorstudie

Fragestellung:
    Welche regelbasierten Signale aus unseren 18 technischen Indikatoren
    haben auf unserem Universum einen messbaren statistischen Vorteil (Edge)?

Datenquelle:
    1. Parquet-Dateien in data/raw/ (falls vorhanden, z.B. vom letzten Kaggle-Run)
    2. Fallback: yfinance-Batch-Download

Indikatoren (via features/engineer.py - RAW, ohne Z-Score-Normalisierung):
    Trend:      sma_ratio_20/50/200, ema_ratio_12, macd_diff
    Momentum:   rsi_14, roc_5, roc_21, stoch_k
    Volatility: atr_ratio, bb_width, bb_pos
    Volume:     volume_ratio_20, obv_diff
    Preis:      high_low_ratio, ret_1d, ret_5d, ret_21d

Signal-Typen:
    Alle Signale sind TRANSITION-Signale:
    Signal[t] = Bedingung[t] == True UND Bedingung[t-1] == False
    Vorteil: Keine Autokorrelation aufeinanderfolgender Signal-Tage.
    Die letzten 3 Signale sind Kombinations-Signale (Transition + Level-Filter).

Forward Returns (kein Look-Ahead-Bias):
    fwd_Nd[t] = Close[t+N] / Close[t] - 1
    Interpretation: Rendite wenn man am Signal-Tag T zum Schlusskurs kauft
    und nach N Handelstagen wieder verkauft.

Verwendung:
    python research_signals.py
    python research_signals.py --years 7 --min-events 50
    python research_signals.py --sector SMA200_Bull --save-csv
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---- Pfade ------------------------------------------------------------------
_REPO_ROOT  = Path(__file__).parent.resolve()
_RAW_DIR    = _REPO_ROOT / "data" / "raw"
_SECTOR_MAP = _REPO_ROOT / "features" / "sector_map.json"

# ---- Konstanten -------------------------------------------------------------
HORIZONS     = [5, 20, 60]      # Forward-Return Horizonte (Handelstage)
MIN_ROWS     = 260               # Mindest-Handelstage pro Ticker fuer Warm-up
BB_PCT_WINDOW = 252              # Fenster fuer Bollinger-Band-Breitenpercentil


# ==============================================================================
# 1. TICKER-LISTE
# ==============================================================================

def _load_tickers() -> list[str]:
    raw = json.loads(_SECTOR_MAP.read_text())
    return sorted(t for t in raw if not t.startswith("_"))


# ==============================================================================
# 2. OHLCV-DATEN  (Parquet -> yfinance Fallback)
# ==============================================================================

def _load_from_parquet(tickers: list[str], years: float) -> dict[str, pd.DataFrame]:
    """Laedt OHLCV-Daten aus bestehenden Parquet-Dateien in data/raw/."""
    files = sorted(_RAW_DIR.glob("*_1d.parquet"))
    if not files:
        return {}

    cutoff = pd.Timestamp.today() - pd.Timedelta(days=int(years * 365.25))
    data: dict[str, pd.DataFrame] = {}

    for fpath in files:
        ticker = fpath.stem.replace("_1d", "")
        if ticker not in tickers:
            continue
        try:
            df = pd.read_parquet(fpath)
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.columns = [c.lower() for c in df.columns]
            df = df[df.index >= cutoff]
            if len(df) >= MIN_ROWS and {"open","high","low","close","volume"}.issubset(df.columns):
                data[ticker] = df[["open","high","low","close","volume"]]
        except Exception:
            pass

    return data


def _load_from_yfinance(tickers: list[str], years: float) -> dict[str, pd.DataFrame]:
    """Laedt OHLCV-Daten via yfinance-Batch-Download."""
    import yfinance as yf
    import logging
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    end_dt   = date.today()
    start_dt = end_dt - timedelta(days=int(years * 365.25) + 60)

    print(f"  yfinance Batch-Download: {start_dt} -> {end_dt}  ({len(tickers)} Ticker) ...")

    raw = yf.download(
        tickers, start=str(start_dt), end=str(end_dt),
        auto_adjust=True, progress=False, threads=True,
    )
    if raw.empty:
        raise RuntimeError("yfinance lieferte keine Daten.")

    data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            df = raw.xs(ticker, axis=1, level=1).copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
            df.columns = [c.lower() for c in df.columns]
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.dropna(subset=["close"])
            if len(df) >= MIN_ROWS:
                data[ticker] = df[["open","high","low","close","volume"]]
        except Exception:
            pass

    return data


def load_universe(tickers: list[str], years: float) -> dict[str, pd.DataFrame]:
    """Laedt Daten: zuerst Parquet, dann yfinance-Fallback."""
    data = _load_from_parquet(tickers, years)
    if data:
        print(f"  {len(data)} Ticker aus Parquet-Dateien geladen.")
        return data

    print("  Keine Parquet-Dateien gefunden - nutze yfinance ...")
    data = _load_from_yfinance(tickers, years)
    print(f"  {len(data)}/{len(tickers)} Ticker erfolgreich geladen.")
    return data


# ==============================================================================
# 3. INDIKATOREN  (direkt aus features/engineer.py)
# ==============================================================================

def _compute_raw_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Berechnet alle 18 Indikatoren via features/engineer.py.

    Gibt die RAW-Werte zurueck (KEIN Z-Score - wir brauchen die
    absoluten Niveaus fuer sinnvolle boolesche Regeln).
    """
    sys.path.insert(0, str(_REPO_ROOT))
    from features.engineer import compute_indicators
    return compute_indicators(df)


# ==============================================================================
# 4. SIGNAL-DEFINITIONEN  (22 boolesche Transition-Signale)
# ==============================================================================
#
# Jedes Signal ist ein TRANSITION-Signal:
#   signal[t] = cond[t] == True  AND  cond[t-1] == False
#
# Dadurch:
#   - Keine Autokorrelation aufeinanderfolgender Tage
#   - Sauber interpretierbare Event-Study
#   - Klar definierter Einstiegspunkt
#
# Die letzten 3 Signale sind Kombinations-Signale:
#   Transition (Signal A) + Level-Filter (Bedingung B muss gleichzeitig True)
#   Praxis-Logik: "MACD dreht bullisch WAEHREND RSI noch nicht ueberkauft"
#
# Namens-Konvention: <Kategorie>_<Beschreibung>
# ============================================================================

def _transition(cond: pd.Series) -> pd.Series:
    """Gibt True nur am ersten Tag der Bedingung (Uebergang False -> True)."""
    return cond & ~cond.shift(1).fillna(False)


def build_signals(ind: pd.DataFrame) -> pd.DataFrame:
    """Berechnet alle 22 booleschen Signale aus den Roh-Indikatoren.

    Args:
        ind: DataFrame mit Spalten gemaess FEATURE_COLS aus engineer.py
             (RAW, nicht z-score-normalisiert).

    Returns:
        DataFrame gleichen Index, Spalten = Signal-Namen (bool).
    """
    s = pd.DataFrame(index=ind.index)

    # =========================================================================
    # A) TREND
    # =========================================================================

    # Kurs kreuzt SMA200 nach oben (Beginn Aufwaertstrend)
    s["SMA200_Bull"] = _transition(ind["sma_ratio_200"] > 1.0)

    # Kurs faellt unter SMA200 (Beginn Abwaertstrend - wie verhalten sich Aktien danach?)
    s["SMA200_Bear"] = _transition(ind["sma_ratio_200"] < 1.0)

    # Kurs kreuzt SMA50 nach oben (mittelfristiger Trend)
    s["SMA50_Bull"]  = _transition(ind["sma_ratio_50"] > 1.0)

    # Golden Cross: SMA50 kreuzt SMA200 nach oben
    # Berechnung: SMA50 = Close/sma_ratio_50 > Close/sma_ratio_200 = SMA200
    # <=> 1/sma_ratio_50 > 1/sma_ratio_200 <=> sma_ratio_200 > sma_ratio_50
    golden = ind["sma_ratio_200"] > ind["sma_ratio_50"]
    s["Golden_Cross"]  = _transition(golden)

    # Death Cross: SMA50 kreuzt SMA200 nach unten
    s["Death_Cross"]   = _transition(~golden)

    # Kurs kreuzt EMA12 nach oben (kurzfristiger Trend-Flip)
    s["EMA12_Bull"]    = _transition(ind["ema_ratio_12"] > 1.0)

    # =========================================================================
    # B) MOMENTUM / RSI / STOCHASTIK
    # =========================================================================

    # RSI faellt erstmals unter 30 (Ueberverkauft, potentielle Mean-Reversion)
    s["RSI_Oversold"]   = _transition(ind["rsi_14"] < 0.30)

    # RSI steigt erstmals ueber 70 (Ueberkauft)
    s["RSI_Overbought"] = _transition(ind["rsi_14"] > 0.70)

    # RSI kreuzt 50 von unten (tritt in bullische Zone ein)
    s["RSI_Mid_Bull"]   = _transition(ind["rsi_14"] > 0.50)

    # Stochastik faellt unter 20 (ueberverkauft)
    s["Stoch_Oversold"]   = _transition(ind["stoch_k"] < 0.20)

    # Stochastik steigt ueber 80 (ueberkauft)
    s["Stoch_Overbought"] = _transition(ind["stoch_k"] > 0.80)

    # ROC-21 kreuzt +10% nach oben (starkes Momentum-Signal)
    s["ROC21_Breakout"]   = _transition(ind["roc_21"] > 0.10)

    # ROC-21 faellt unter -10% (Crash-Szenario - wie erholt sich die Aktie?)
    s["ROC21_Crash"]      = _transition(ind["roc_21"] < -0.10)

    # =========================================================================
    # C) MACD
    # =========================================================================

    # MACD-Histogramm dreht von negativ auf positiv (Kaufsignal)
    s["MACD_Bull"] = _transition(ind["macd_diff"] > 0.0)

    # MACD-Histogramm dreht von positiv auf negativ (Verkaufssignal)
    s["MACD_Bear"] = _transition(ind["macd_diff"] < 0.0)

    # =========================================================================
    # D) BOLLINGER BANDS
    # =========================================================================

    # Kurs bricht erstmals unter das untere Bollinger Band (Oversold-Zone)
    s["BB_Lower_Break"]  = _transition(ind["bb_pos"] < 0.0)

    # Kurs bricht erstmals ueber das obere Bollinger Band (Overbought-Zone)
    s["BB_Upper_Break"]  = _transition(ind["bb_pos"] > 1.0)

    # =========================================================================
    # E) VOLUMEN
    # =========================================================================

    # Bullischer Volumen-Spike: > 2x Durchschnitt bei steigendem Kurs
    vol_spike_bull = (ind["volume_ratio_20"] > 2.0) & (ind["ret_1d"] > 0.0)
    s["VolSpike_Bull"] = _transition(vol_spike_bull)

    # Baerischer Volumen-Spike: > 2x Durchschnitt bei fallendem Kurs
    vol_spike_bear = (ind["volume_ratio_20"] > 2.0) & (ind["ret_1d"] < 0.0)
    s["VolSpike_Bear"] = _transition(vol_spike_bear)

    # =========================================================================
    # F) KOMBINATIONS-SIGNALE  (Transition A + Level-Filter B)
    # =========================================================================

    # RSI dreht bullisch (< 30) UND Kurs unter unterem Bollinger Band
    # -> starkes Doppel-Oversold-Signal
    double_oversold = (ind["rsi_14"] < 0.30) & (ind["bb_pos"] < 0.0)
    s["Multi_DoubleOversold"]  = _transition(double_oversold)

    # MACD dreht bullisch UND RSI noch nicht ueberkauft (< 65)
    # -> frueher Einstieg ohne Overkauf-Risiko
    macd_bull_level = (ind["macd_diff"] > 0.0) & (ind["rsi_14"] < 0.65)
    s["Multi_MACD_RSI"]        = _transition(macd_bull_level)

    # Kurs kreuzt SMA200 nach oben UND Volumen > 1.5x Durchschnitt
    # -> bestaetiger Ausbruch (Preis + Volumen)
    sma200_vol = (ind["sma_ratio_200"] > 1.0) & (ind["volume_ratio_20"] > 1.5)
    s["Multi_SMA200_Volume"]   = _transition(sma200_vol)

    return s.fillna(False)


# ==============================================================================
# 5. FORWARD RETURNS  (kein Look-Ahead-Bias)
# ==============================================================================

def compute_forward_returns(close: pd.Series) -> pd.DataFrame:
    """fwd_Nd[t] = Close[t+N] / Close[t] - 1.

    Kauf am Signal-Tag T zum Schlusskurs, Verkauf nach N Tagen.
    Keine Lookahead-Bias: shift(-N) schaut nur in die Zukunft,
    nicht rueckwaerts in die Vergangenheit.
    """
    fwd = pd.DataFrame(index=close.index)
    for n in HORIZONS:
        fwd[f"fwd_{n}d"] = close.shift(-n) / close - 1.0
    return fwd


# ==============================================================================
# 6. PANEL AUFBAUEN
# ==============================================================================

def build_panel(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Verarbeitet alle Ticker und baut einen langen Panel-DataFrame.

    Schritte pro Ticker:
      1. compute_indicators()  -> 18 Roh-Indikatoren
      2. build_signals()       -> 22 boolesche Signale
      3. compute_forward_returns() -> 3 Forward-Return-Spalten
      4. Alles zusammenfuehren, Ticker-Spalte hinzufuegen

    Returns:
        Langer DataFrame: (Date) x (Signals + FwdReturns + Ticker-Spalte)
    """
    parts = []
    n_ok = n_fail = 0

    for ticker, ohlcv in data.items():
        try:
            ind = _compute_raw_indicators(ohlcv)

            # Nur Zeilen mit vollstaendigen Indikatoren (Warm-up-Phase entfernen)
            valid = ind.notna().all(axis=1)
            ind   = ind[valid]
            ohlcv_v = ohlcv[valid]

            if len(ind) < 100:
                n_fail += 1
                continue

            sigs = build_signals(ind)
            fwds = compute_forward_returns(ohlcv_v["close"])

            combined = pd.concat([sigs, fwds], axis=1)
            combined["ticker"] = ticker
            parts.append(combined)
            n_ok += 1

        except Exception as e:
            n_fail += 1

    print(f"  Panel aufgebaut: {n_ok} Ticker OK, {n_fail} fehlgeschlagen.")
    return pd.concat(parts).reset_index().rename(columns={"index": "date"})


# ==============================================================================
# 7. EVENT STUDY
# ==============================================================================

SIGNAL_NAMES = [
    # Trend
    "SMA200_Bull", "SMA200_Bear", "SMA50_Bull",
    "Golden_Cross", "Death_Cross", "EMA12_Bull",
    # Momentum
    "RSI_Oversold", "RSI_Overbought", "RSI_Mid_Bull",
    "Stoch_Oversold", "Stoch_Overbought",
    "ROC21_Breakout", "ROC21_Crash",
    # MACD
    "MACD_Bull", "MACD_Bear",
    # Bollinger
    "BB_Lower_Break", "BB_Upper_Break",
    # Volumen
    "VolSpike_Bull", "VolSpike_Bear",
    # Kombination
    "Multi_DoubleOversold", "Multi_MACD_RSI", "Multi_SMA200_Volume",
]


def _sig_stats(returns: pd.Series, signal: str, horizon: int) -> dict:
    """Berechnet Kennzahlen fuer eine (Signal, Horizont)-Gruppe."""
    n       = len(returns)
    mean_r  = returns.mean()
    med_r   = returns.median()
    std_r   = returns.std(ddof=1)
    hit_r   = (returns > 0).mean()
    t_stat  = mean_r / (std_r / np.sqrt(n)) if std_r > 0 else 0.0
    pct_5   = np.percentile(returns, 5)
    pct_95  = np.percentile(returns, 95)
    return {
        "Signal":    signal,
        "Horizont":  f"{horizon}d",
        "N":         n,
        "Hit_%":     hit_r * 100,
        "Mean_%":    mean_r * 100,
        "Median_%":  med_r * 100,
        "Std_%":     std_r * 100,
        "t-Stat":    t_stat,
        "P5_%":      pct_5 * 100,
        "P95_%":     pct_95 * 100,
    }


def run_event_study(panel: pd.DataFrame, min_events: int) -> pd.DataFrame:
    """Fuehrt die Event Study fuer alle Signale und Horizonte durch.

    Returns:
        Langer DataFrame (eine Zeile pro Signal x Horizont).
    """
    rows = []

    # Baseline (kein Signal - alle Beobachtungen)
    for n in HORIZONS:
        col = f"fwd_{n}d"
        ret = panel[col].dropna()
        rows.append(_sig_stats(ret, "** BASELINE **", n))

    # Signale
    for sig in SIGNAL_NAMES:
        if sig not in panel.columns:
            continue
        mask = panel[sig].fillna(False)
        for n in HORIZONS:
            col = f"fwd_{n}d"
            ret = panel.loc[mask, col].dropna()
            if len(ret) < min_events:
                continue
            rows.append(_sig_stats(ret, sig, n))

    return pd.DataFrame(rows)


# ==============================================================================
# 8. AUSGABE
# ==============================================================================

def _pivot_and_sort(df_long: pd.DataFrame) -> pd.DataFrame:
    """Baut breite Vergleichstabelle (eine Zeile pro Signal)."""
    parts = {}
    for n in HORIZONS:
        sub = df_long[df_long["Horizont"] == f"{n}d"].set_index("Signal")
        for col, label in [
            ("N",       f"N_{n}d"),
            ("Hit_%",   f"HR_{n}d"),
            ("Mean_%",  f"MR_{n}d"),
            ("Median_%",f"Med_{n}d"),
            ("t-Stat",  f"t_{n}d"),
        ]:
            parts[label] = sub[col]

    wide = pd.DataFrame(parts)
    # Sortiere nach 20d Mean Return
    if "MR_20d" in wide.columns:
        wide = wide.sort_values("MR_20d", ascending=False)
    return wide


def _format_wide(wide: pd.DataFrame) -> pd.DataFrame:
    """Formatiert Zahlen fuer die Konsolen-Ausgabe."""
    fmt = wide.copy()
    for n in HORIZONS:
        if f"HR_{n}d" in fmt: fmt[f"HR_{n}d"] = fmt[f"HR_{n}d"].map(lambda x: f"{x:.1f}%")
        if f"MR_{n}d" in fmt: fmt[f"MR_{n}d"] = fmt[f"MR_{n}d"].map(lambda x: f"{x:+.2f}%")
        if f"Med_{n}d" in fmt: fmt[f"Med_{n}d"] = fmt[f"Med_{n}d"].map(lambda x: f"{x:+.2f}%")
        if f"t_{n}d"  in fmt: fmt[f"t_{n}d"]  = fmt[f"t_{n}d"].map(lambda x: f"{x:+.2f}")
        if f"N_{n}d"  in fmt: fmt[f"N_{n}d"]  = fmt[f"N_{n}d"].map(lambda x: f"{int(x):,}")
    return fmt


def print_main_table(wide: pd.DataFrame) -> None:
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.width", 200)

    fmt = _format_wide(wide)
    line = "=" * 185

    print(f"\n{line}")
    print("  SIGNAL EDGE RESEARCH  |  Ergebnis-Uebersicht  |  Sortiert nach: Mean Return 20d")
    print(f"  Alle Signale: TRANSITION (Signal faeuert nur am ersten Tag der Bedingung)")
    print(line)
    print(fmt.to_string())
    print(line)
    print()
    print("  Legende:")
    print("  N     = Ereignis-Anzahl")
    print("  HR    = Hit-Rate (% mit positivem Forward Return)")
    print("  MR    = Mean Forward Return (Mittelwert)")
    print("  Med   = Median Forward Return (robuster gegen Ausreisser)")
    print("  t     = t-Statistik: |t| > 2 = statistisch signifikant (p < 0.05)")
    print("  Survivorship-Bias: Delisted Aktien sind nicht im Universum enthalten.")
    print()


def print_signal_counts(panel: pd.DataFrame) -> None:
    """Zeigt wie haeufig jedes Signal vorkommt."""
    total = len(panel)
    print("  SIGNAL-HAEUFIGKEITEN:")
    print(f"  {'Signal':<25}  {'Ereignisse':>10}  {'Anteil':>8}")
    print("  " + "-" * 50)
    for sig in SIGNAL_NAMES:
        if sig not in panel.columns:
            continue
        n   = panel[sig].fillna(False).sum()
        pct = n / total * 100
        print(f"  {sig:<25}  {int(n):>10,}  {pct:>7.2f}%")
    print()


def print_significance_summary(df_long: pd.DataFrame) -> None:
    """Druckt Schnell-Fazit: Signale mit statistisch signifikantem Edge."""
    base_20 = df_long.loc[
        (df_long["Signal"] == "** BASELINE **") & (df_long["Horizont"] == "20d"), "Mean_%"
    ]
    baseline = base_20.values[0] if len(base_20) else 0.0

    candidates = df_long[
        (df_long["Signal"] != "** BASELINE **") &
        (df_long["Horizont"] == "20d") &
        (df_long["t-Stat"].abs() > 2.0)
    ].copy()
    better = candidates[candidates["Mean_%"] > baseline].sort_values("Mean_%", ascending=False)
    worse  = candidates[candidates["Mean_%"] < baseline].sort_values("Mean_%")

    print("  SCHNELL-FAZIT: Signale mit |t| > 2 (statistisch signifikant)")
    print(f"  Baseline 20d Mean Return: {baseline:+.2f}%")
    print()

    if not better.empty:
        print("  + BESSER als Baseline (potenzielle Long-Kandidaten):")
        for _, r in better.iterrows():
            bar = "#" * min(20, max(1, int(abs(r["Mean_%"] - baseline) * 10)))
            print(f"    [{bar:<20}]  {r['Signal']:<25}  MR={r['Mean_%']:+.2f}%  "
                  f"HR={r['Hit_%']:.1f}%  t={r['t-Stat']:+.2f}  N={int(r['N']):,}")
    else:
        print("  (Kein Long-Signal signifikant besser als Baseline)")
    print()

    if not worse.empty:
        print("  - SCHLECHTER als Baseline (potenzielle Short-/Vorsicht-Signale):")
        for _, r in worse.iterrows():
            print(f"    {r['Signal']:<25}  MR={r['Mean_%']:+.2f}%  "
                  f"HR={r['Hit_%']:.1f}%  t={r['t-Stat']:+.2f}  N={int(r['N']):,}")
    print()


def print_percentile_dist(panel: pd.DataFrame, min_events: int) -> None:
    """Rendite-Verteilung (Perzentile) fuer den 20d-Horizont."""
    col = "fwd_20d"
    print("  RENDITE-VERTEILUNG (20d) - Perzentile:")
    hdr = f"  {'Signal':<25}  {'N':>6}  {'P5':>7}  {'P25':>7}  {'P50':>7}  {'P75':>7}  {'P95':>7}  {'Mean':>7}"
    print(hdr)
    print("  " + "-" * 80)

    def _row(label, ret):
        if len(ret) < min_events:
            return
        p = np.percentile(ret, [5, 25, 50, 75, 95])
        print(f"  {label:<25}  {len(ret):>6,}  "
              f"{p[0]*100:>+6.1f}%  {p[1]*100:>+6.1f}%  {p[2]*100:>+6.1f}%  "
              f"{p[3]*100:>+6.1f}%  {p[4]*100:>+6.1f}%  {ret.mean()*100:>+6.1f}%")

    _row("** BASELINE **", panel[col].dropna())
    for sig in SIGNAL_NAMES:
        if sig not in panel.columns:
            continue
        ret = panel.loc[panel[sig].fillna(False), col].dropna()
        _row(sig, ret)

    print("  " + "-" * 80 + "\n")


def print_sector_breakdown(panel: pd.DataFrame, data: dict,
                           signal: str, horizon: int = 20) -> None:
    """Zeigt ob ein Signal sektoren-uebergreifend funktioniert."""
    sector_map = json.loads(_SECTOR_MAP.read_text())
    panel_s = panel.copy()
    panel_s["sector"] = panel_s["ticker"].map(sector_map)

    col = f"fwd_{horizon}d"
    if signal not in panel_s.columns:
        print(f"  Signal '{signal}' nicht gefunden.")
        return

    mask = panel_s[signal].fillna(False)
    grp = panel_s.loc[mask, ["sector", col]].dropna()
    if grp.empty:
        print("  Keine Ereignisse.")
        return

    stats = (
        grp.groupby("sector")[col]
        .agg(N="count", Mean=lambda x: x.mean() * 100, HR=lambda x: (x > 0).mean() * 100)
        .sort_values("Mean", ascending=False)
    )
    print(f"\n  SEKTOR-BREAKDOWN: {signal}  |  Horizont {horizon}d")
    print("  " + "-" * 60)
    print(f"  {'Sektor':<30}  {'N':>5}  {'Mean':>8}  {'Hit-%':>7}")
    print("  " + "-" * 60)
    for sector, row in stats.iterrows():
        print(f"  {sector:<30}  {int(row['N']):>5,}  {row['Mean']:>+7.2f}%  {row['HR']:>6.1f}%")
    print("  " + "-" * 60 + "\n")


# ==============================================================================
# 9. MAIN
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Signal Edge Research | Trading v6")
    parser.add_argument("--years",      type=float, default=5.0,
                        help="Historische Zeitspanne in Jahren (Standard: 5)")
    parser.add_argument("--min-events", type=int,   default=30,
                        help="Minimale Ereignisanzahl fuer Report (Standard: 30)")
    parser.add_argument("--sector",     type=str,   default=None,
                        help="Sektor-Breakdown fuer dieses Signal (z.B. RSI_Oversold)")
    parser.add_argument("--no-dist",    action="store_true",
                        help="Rendite-Verteilungs-Tabelle ausblenden")
    parser.add_argument("--save-csv",   action="store_true",
                        help="Ergebnisse als CSV speichern")
    args = parser.parse_args()

    print("=" * 70)
    print("  SIGNAL EDGE RESEARCH  |  Trading v6 Vorstudie")
    print("=" * 70)

    # 1. Ticker
    tickers = _load_tickers()
    print(f"\n[1/4] Universum: {len(tickers)} Ticker aus sector_map.json")

    # 2. Daten
    print(f"\n[2/4] OHLCV-Daten laden ({args.years:.0f} Jahre) ...")
    data = load_universe(tickers, args.years)
    if not data:
        print("FEHLER: Keine Daten geladen.")
        return

    # 3. Panel
    print(f"\n[3/4] Indikatoren und Signale berechnen ...")
    panel = build_panel(data)
    total = len(panel)
    print(f"  Panel: {total:,} Beobachtungen ({panel['ticker'].nunique()} Assets)")
    print()
    print_signal_counts(panel)

    # 4. Event Study
    print(f"[4/4] Event Study ...")
    df_long = run_event_study(panel, args.min_events)
    wide    = _pivot_and_sort(df_long)

    # Ausgabe
    print_main_table(wide)
    print_significance_summary(df_long)

    if not args.no_dist:
        print_percentile_dist(panel, args.min_events)

    if args.sector:
        print_sector_breakdown(panel, data, args.sector)

    if args.save_csv:
        path = _REPO_ROOT / "research_signals_results.csv"
        wide.to_csv(path)
        print(f"  CSV gespeichert: {path}")


if __name__ == "__main__":
    main()
