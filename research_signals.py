"""
research_signals.py
==============================================================================
Signal-Edge Research für das 260-Aktien-Universum (Trading v6 Vorstudie).

Fragestellung:
    Welche klassischen technischen Signale haben auf unserem Universum einen
    messbaren statistischen Vorteil (Edge), und wie groß ist er?

Methodik:
    - Event Study: Signal tritt am Tag T auf → Forward Return von T bis T+N.
    - Alle Signale sind TRANSITION-Signale (feuern nur am ersten Tag der
      Bedingung, nicht an allen Folgetagen) → vermeidet Autokorrelation.
    - Kein Look-Ahead-Bias: Rolling-Windows nutzen ausschließlich Daten vor T.
    - Survival-Bias-Hinweis: Es wird das heutige 260er-Universum verwendet,
      d.h. delisted Aktien der Vergangenheit sind nicht enthalten.

Signale:
    Breakout_20       Close überschreitet erstmals 20-Tage-Hoch (Momentum)
    Breakout_50       Close überschreitet erstmals 50-Tage-Hoch (Momentum)
    SMA_Crossover     SMA50 kreuzt SMA200 nach oben (Trend-Bestätigung)
    RSI_Oversold      RSI14 fällt erstmals unter 30 (Mean-Reversion)
    MACD_Bullish      MACD-Histogramm wechselt von negativ zu positiv
    Volatility_Contr  Bollinger-Band-Breite fällt in unterstes 10%-Perzentil

Auswertung:
    Für jedes Signal und jeden Horizont (5 / 20 / 60 Tage):
        N             Anzahl Ereignisse
        Hit-Rate      Anteil positiver Forward Returns
        Mean Return   Durchschnittlicher Forward Return
        Median Return Medianer Forward Return (robust gegenüber Ausreißern)
        t-Statistik   Signifikanztest: Mean / (Std / √N)

Verwendung:
    python research_signals.py
    python research_signals.py --years 3
    python research_signals.py --years 5 --min-events 50

Ausgabe:
    Konsolen-Tabelle, sortiert nach bestem 20-Tage-Mean-Return.
    Optional: Speichert CSV → research_signals_results.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# -- Pfade ---------------------------------------------------------------------
_REPO_ROOT  = Path(__file__).parent.resolve()
_SECTOR_MAP = _REPO_ROOT / "features" / "sector_map.json"

# -- Konstanten ----------------------------------------------------------------
HORIZONS    = [5, 20, 60]           # Forward-Return Horizonte in Handelstagen
MIN_HISTORY = 252                   # Mindest-Handelstage pro Ticker (ca. 1 Jahr)
BBW_LOOKBACK = 252                  # Bollinger-Band-Breite Perzentil-Fenster


# ==============================================================================
# 1. Ticker-Liste laden
# ==============================================================================

def _load_tickers() -> list[str]:
    raw = json.loads(_SECTOR_MAP.read_text())
    return sorted(t for t in raw if not t.startswith("_"))


# ==============================================================================
# 2. OHLCV-Daten laden
# ==============================================================================

def _download_universe(tickers: list[str], years: int) -> dict[str, pd.DataFrame]:
    """Lädt alle Ticker in einem yfinance-Batch-Call.

    Returns:
        Dict {ticker → OHLCV-DataFrame mit tz-naivem DatetimeIndex}.
    """
    import yfinance as yf

    end_dt   = date.today()
    start_dt = end_dt - timedelta(days=int(years * 365.25) + 30)   # Puffer

    print(f"  Lade {len(tickers)} Ticker: {start_dt} → {end_dt} ({years} Jahre) …")

    raw = yf.download(
        tickers,
        start       = str(start_dt),
        end         = str(end_dt),
        auto_adjust = True,
        progress    = False,
        threads     = True,
    )

    if raw.empty:
        raise RuntimeError("yfinance lieferte keine Daten.")

    data: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                df = raw.xs(ticker, axis=1, level=1).copy()
            else:
                df = raw.copy()

            df.columns = [c.lower() for c in df.columns]
            required   = {"open", "high", "low", "close", "volume"}
            if not required.issubset(df.columns):
                continue

            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.dropna(subset=["close"])

            if len(df) < MIN_HISTORY:
                continue

            data[ticker] = df[["open", "high", "low", "close", "volume"]]

        except Exception:
            pass

    print(f"  {len(data)}/{len(tickers)} Ticker erfolgreich geladen.")
    return data


# ==============================================================================
# 3. Indikatoren (keine Look-Ahead-Bias)
# ==============================================================================

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder'scher RSI via exponentiell gewichtetem Moving Average."""
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd_histogram(close: pd.Series,
                    fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD-Histogramm = MACD-Linie minus Signal-Linie."""
    ema_fast    = close.ewm(span=fast,   adjust=False).mean()
    ema_slow    = close.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


def _bollinger_band_width(close: pd.Series, period: int = 20,
                          num_std: float = 2.0) -> pd.Series:
    """Normalisierte Bollinger-Band-Breite: (Upper − Lower) / Middle."""
    sma   = close.rolling(period).mean()
    std   = close.rolling(period).std(ddof=1)
    upper = sma + num_std * std
    lower = sma - num_std * std
    return (upper - lower) / sma.replace(0.0, np.nan)


# ==============================================================================
# 4. Signal-Berechnung (Transitions – kein Look-Ahead-Bias)
# ==============================================================================

_SIGNAL_NAMES = [
    "Breakout_20",
    "Breakout_50",
    "SMA_Crossover",
    "RSI_Oversold",
    "MACD_Bullish",
    "Volatility_Contr",
]


def _compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Berechnet alle booleschen Transition-Signale für einen Ticker.

    Transition-Signal: Feuert am Tag T, wenn die Bedingung an T erstmals
    erfüllt ist (an T−1 war sie noch nicht erfüllt).

    Args:
        df: OHLCV-DataFrame eines einzelnen Tickers.

    Returns:
        DataFrame mit Signal-Spalten (bool) + Forward-Return-Spalten (float).
    """
    close  = df["close"]
    high   = df["high"]
    result = pd.DataFrame(index=df.index)

    # -- Forward Returns (kein Look-Ahead: shift nach HINTEN in der Zeit) -----
    # fwd_Nd[T] = Close[T+N] / Close[T] − 1  (Kauf am Close T, Verkauf T+N)
    for n in HORIZONS:
        result[f"fwd_{n}d"] = close.shift(-n) / close - 1.0

    # -- Breakout_20: Schlusskurs bricht über 20-Tage-Hoch --------------------
    # Hoch der letzten 20 Tage (OHNE heute): shift(1) → Rolling Max
    high20 = close.shift(1).rolling(20).max()
    cond20 = close > high20
    result["Breakout_20"] = cond20 & ~cond20.shift(1).fillna(False)

    # -- Breakout_50: analog für 50 Tage --------------------------------------
    high50 = close.shift(1).rolling(50).max()
    cond50 = close > high50
    result["Breakout_50"] = cond50 & ~cond50.shift(1).fillna(False)

    # -- SMA_Crossover: SMA50 kreuzt SMA200 nach oben -------------------------
    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    above  = sma50 > sma200
    result["SMA_Crossover"] = above & ~above.shift(1).fillna(False)

    # -- RSI_Oversold: RSI14 fällt erstmals unter 30 --------------------------
    rsi      = _rsi(close, 14)
    under30  = rsi < 30.0
    result["RSI_Oversold"] = under30 & ~under30.shift(1).fillna(False)

    # -- MACD_Bullish: Histogramm wechselt von negativ zu positiv -------------
    hist    = _macd_histogram(close)
    pos_h   = hist > 0.0
    result["MACD_Bullish"] = pos_h & ~pos_h.shift(1).fillna(False)

    # -- Volatility_Contraction: BBW fällt in unterstes 10%-Perzentil ---------
    bbw       = _bollinger_band_width(close)
    # Rollendes 10%-Perzentil über BBW_LOOKBACK Tage (nur Vergangenheit)
    bbw_q10   = bbw.shift(1).rolling(BBW_LOOKBACK, min_periods=BBW_LOOKBACK // 2).quantile(0.10)
    in_contr  = bbw < bbw_q10
    result["Volatility_Contr"] = in_contr & ~in_contr.shift(1).fillna(False)

    return result


# ==============================================================================
# 5. Panel aufbauen
# ==============================================================================

def _build_panel(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Fasst alle Ticker zu einem langen Panel zusammen.

    Returns:
        DataFrame mit (Date, Ticker) als MultiIndex.
    """
    parts = []
    for ticker, df in data.items():
        sig_df         = _compute_signals(df)
        sig_df["ticker"] = ticker
        parts.append(sig_df)

    panel = pd.concat(parts).reset_index().rename(columns={"index": "date"})
    return panel


# ==============================================================================
# 6. Event Study
# ==============================================================================

def _event_study(panel: pd.DataFrame, min_events: int) -> pd.DataFrame:
    """Berechnet Signal-Statistiken für alle Horizonte.

    Args:
        panel:      Langer Panel-DataFrame (alle Ticker, alle Tage).
        min_events: Minimale Ereignisanzahl für Aufnahme in den Report.

    Returns:
        DataFrame mit einer Zeile pro (Signal, Horizont).
    """
    rows = []

    # -- Baseline: alle Beobachtungen (kein Signal) ----------------------------
    for n in HORIZONS:
        col = f"fwd_{n}d"
        ret = panel[col].dropna()
        rows.append(_stats_row("— Baseline (alle)", n, ret))

    # -- Signale ---------------------------------------------------------------
    for sig in _SIGNAL_NAMES:
        if sig not in panel.columns:
            continue
        mask = panel[sig].fillna(False)
        for n in HORIZONS:
            col = f"fwd_{n}d"
            ret = panel.loc[mask, col].dropna()
            if len(ret) < min_events:
                continue
            rows.append(_stats_row(sig, n, ret))

    return pd.DataFrame(rows)


def _stats_row(signal: str, horizon: int, returns: pd.Series) -> dict:
    """Berechnet Kennzahlen für eine (Signal, Horizont)-Gruppe."""
    n       = len(returns)
    mean_r  = returns.mean()
    med_r   = returns.median()
    std_r   = returns.std(ddof=1)
    hit     = (returns > 0).mean()
    t_stat  = (mean_r / (std_r / np.sqrt(n))) if std_r > 0 else 0.0
    ann_fac = 252.0 / horizon                  # annualisierungs-Faktor
    sharpe  = (mean_r * ann_fac) / (std_r * np.sqrt(ann_fac)) if std_r > 0 else 0.0

    return {
        "Signal":        signal,
        "Horizont":      f"{horizon}d",
        "N":             n,
        "Hit-Rate":      hit,
        "Mean Return":   mean_r,
        "Median Return": med_r,
        "Std":           std_r,
        "t-Stat":        t_stat,
        "Sharpe (ann.)": sharpe,
    }


# ==============================================================================
# 7. Pivot-Tabelle für übersichtliche Ausgabe
# ==============================================================================

def _pivot_results(df_long: pd.DataFrame) -> pd.DataFrame:
    """Baut die breite Vergleichstabelle (eine Zeile pro Signal)."""
    cols_order = []
    wide_parts = {}

    for n in HORIZONS:
        sub = df_long[df_long["Horizont"] == f"{n}d"].set_index("Signal")
        for col, label in [
            ("N",             f"N_{n}d"),
            ("Hit-Rate",      f"HR_{n}d"),
            ("Mean Return",   f"MR_{n}d"),
            ("Median Return", f"Med_{n}d"),
            ("t-Stat",        f"t_{n}d"),
        ]:
            wide_parts[label] = sub[col]
            cols_order.append(label)

    wide = pd.DataFrame(wide_parts)

    # Sortiere nach 20-Tage Mean Return (absteigend)
    sort_col = "MR_20d" if "MR_20d" in wide.columns else wide.columns[2]
    return wide.sort_values(sort_col, ascending=False)


# ==============================================================================
# 8. Formatierter Print
# ==============================================================================

def _print_table(wide: pd.DataFrame) -> None:
    """Gibt die Ergebnistabelle lesbar in die Konsole aus."""
    pd.set_option("display.max_columns",  50)
    pd.set_option("display.width",       160)
    pd.set_option("display.float_format", "{:.4f}".format)

    fmt = wide.copy()
    for n in HORIZONS:
        hr_col  = f"HR_{n}d"
        mr_col  = f"MR_{n}d"
        med_col = f"Med_{n}d"
        t_col   = f"t_{n}d"
        n_col   = f"N_{n}d"

        if hr_col  in fmt.columns: fmt[hr_col]  = fmt[hr_col].map(lambda x: f"{x*100:.1f}%")
        if mr_col  in fmt.columns: fmt[mr_col]  = fmt[mr_col].map(lambda x: f"{x*100:+.2f}%")
        if med_col in fmt.columns: fmt[med_col] = fmt[med_col].map(lambda x: f"{x*100:+.2f}%")
        if t_col   in fmt.columns: fmt[t_col]   = fmt[t_col].map(lambda x: f"{x:+.2f}")
        if n_col   in fmt.columns: fmt[n_col]   = fmt[n_col].map(lambda x: f"{int(x):,}")

    sep = "=" * 155
    print(f"\n{sep}")
    print("  SIGNAL EDGE RESEARCH — Ergebnisübersicht")
    print(f"  Universum: 260 US-Aktien  |  Sortiert nach: Mean Return 20d")
    print(sep)
    print(fmt.to_string())
    print(sep)

    # Legende
    print("\n  Legende:")
    print("    N       = Anzahl Ereignisse")
    print("    HR      = Hit-Rate (Anteil positiver Forward Returns)")
    print("    MR      = Mean Forward Return (Mittelwert)")
    print("    Med     = Median Forward Return (robuster gegenüber Ausreißern)")
    print("    t       = t-Statistik (|t| > 2 ≈ statistisch signifikant)")
    print("    Signale feuern als TRANSITIONS (nur am ersten Tag der Bedingung)")
    print("    Survival-Bias-Hinweis: Delisted Aktien nicht enthalten.\n")


# ==============================================================================
# 9. Zusätzliche Einzelanalyse: Rendite-Verteilung je Signal
# ==============================================================================

def _print_distribution(panel: pd.DataFrame, min_events: int) -> None:
    """Druckt Perzentil-Verteilung der 20-Tage-Returns je Signal."""
    print("\n  RENDITE-VERTEILUNG (20d Forward Return, Perzentile)")
    print("  " + "-" * 90)
    header = f"  {'Signal':<22}  {'N':>6}  {'5%':>7}  {'25%':>7}  {'50%':>7}  {'75%':>7}  {'95%':>7}  {'Mean':>7}"
    print(header)
    print("  " + "-" * 90)

    # Baseline
    ret = panel["fwd_20d"].dropna()
    _print_dist_row("— Baseline", ret)

    for sig in _SIGNAL_NAMES:
        if sig not in panel.columns:
            continue
        ret = panel.loc[panel[sig].fillna(False), "fwd_20d"].dropna()
        if len(ret) < min_events:
            continue
        _print_dist_row(sig, ret)

    print("  " + "-" * 90 + "\n")


def _print_dist_row(label: str, ret: pd.Series) -> None:
    p = np.percentile(ret, [5, 25, 50, 75, 95])
    print(
        f"  {label:<22}  {len(ret):>6,}  "
        f"{p[0]*100:>+6.1f}%  {p[1]*100:>+6.1f}%  {p[2]*100:>+6.1f}%  "
        f"{p[3]*100:>+6.1f}%  {p[4]*100:>+6.1f}%  {ret.mean()*100:>+6.1f}%"
    )


# ==============================================================================
# 10. Sektor-Breakdown
# ==============================================================================

def _sector_breakdown(panel: pd.DataFrame, data: dict[str, pd.DataFrame],
                      signal: str, horizon: int = 20) -> None:
    """Zeigt, ob ein Signal über alle Sektoren hinweg funktioniert."""
    sector_map = json.loads(_SECTOR_MAP.read_text())
    ticker_sector = {t: sector_map.get(t, "Unknown") for t in data}
    panel_s = panel.copy()
    panel_s["sector"] = panel_s["ticker"].map(ticker_sector)

    col = f"fwd_{horizon}d"
    if signal not in panel_s.columns:
        return

    mask = panel_s[signal].fillna(False)
    if mask.sum() == 0:
        return

    grp = panel_s.loc[mask, ["sector", col]].dropna()
    if grp.empty:
        return

    stats = (
        grp.groupby("sector")[col]
        .agg(N="count", Mean=lambda x: x.mean(), HR=lambda x: (x > 0).mean())
        .sort_values("Mean", ascending=False)
    )
    stats["Mean"] = stats["Mean"].map(lambda x: f"{x*100:+.2f}%")
    stats["HR"]   = stats["HR"].map(lambda x: f"{x*100:.1f}%")

    print(f"\n  SEKTOR-BREAKDOWN: {signal} | Horizont {horizon}d")
    print("  " + "-" * 55)
    print(stats.to_string())
    print("  " + "-" * 55)


# ==============================================================================
# 11. Hauptprogramm
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Signal Edge Research für das 260-Aktien-Universum."
    )
    parser.add_argument("--years",      type=float, default=5.0,
                        help="Historische Zeitspanne in Jahren (Standard: 5)")
    parser.add_argument("--min-events", type=int,   default=30,
                        help="Minimale Ereignisanzahl für Report (Standard: 30)")
    parser.add_argument("--save-csv",   action="store_true",
                        help="Ergebnisse als CSV speichern")
    parser.add_argument("--sector",     type=str,   default=None,
                        help="Sektor-Breakdown für dieses Signal anzeigen (z.B. Breakout_20)")
    parser.add_argument("--no-dist",    action="store_true",
                        help="Rendite-Verteilungs-Tabelle unterdrücken")
    args = parser.parse_args()

    print("=" * 60)
    print("  SIGNAL EDGE RESEARCH  |  Trading v6 Vorstudie")
    print("=" * 60)

    # -- 1. Ticker laden -------------------------------------------------------
    tickers = _load_tickers()
    print(f"\n[1/4] Ticker: {len(tickers)} Symbole aus sector_map.json")

    # -- 2. Daten laden --------------------------------------------------------
    print(f"\n[2/4] OHLCV-Download ({args.years:.0f} Jahre via yfinance) …")
    data = _download_universe(tickers, args.years)

    if not data:
        print("FEHLER: Keine Daten geladen. Abbruch.")
        return

    # -- 3. Signale + Panel ----------------------------------------------------
    print(f"\n[3/4] Signale berechnen und Panel aufbauen …")
    panel = _build_panel(data)

    total_obs    = len(panel)
    total_events = {sig: panel[sig].fillna(False).sum()
                    for sig in _SIGNAL_NAMES if sig in panel.columns}

    print(f"  Panel: {total_obs:,} Beobachtungen")
    print(f"  Ereignisse je Signal:")
    for sig, n in total_events.items():
        pct = n / total_obs * 100
        print(f"    {sig:<22} {n:>6,}  ({pct:.1f}%)")

    # -- 4. Event Study --------------------------------------------------------
    print(f"\n[4/4] Event Study …")
    df_long = _event_study(panel, args.min_events)
    wide    = _pivot_results(df_long)

    # -- Output ----------------------------------------------------------------
    _print_table(wide)

    if not args.no_dist:
        _print_distribution(panel, args.min_events)

    if args.sector:
        _sector_breakdown(panel, data, args.sector)

    if args.save_csv:
        out_path = _REPO_ROOT / "research_signals_results.csv"
        wide.to_csv(out_path)
        print(f"  CSV gespeichert: {out_path}")

    # -- Schnell-Fazit ---------------------------------------------------------
    print("  SCHNELL-FAZIT:")
    print("  -" * 30)
    if "MR_20d" in wide.index or "MR_20d" in wide.columns:
        baseline_mr = df_long.loc[
            (df_long["Signal"] == "— Baseline (alle)") & (df_long["Horizont"] == "20d"),
            "Mean Return"
        ]
        baseline_val = baseline_mr.values[0] * 100 if len(baseline_mr) else 0.0

        # Signale mit t-Stat > 2 und besser als Baseline
        sig_stats = df_long[
            (df_long["Signal"] != "— Baseline (alle)") &
            (df_long["Horizont"] == "20d")
        ].copy()
        sig_stats = sig_stats[sig_stats["t-Stat"].abs() > 2.0]
        sig_stats = sig_stats[sig_stats["Mean Return"] > baseline_val / 100]
        sig_stats = sig_stats.sort_values("Mean Return", ascending=False)

        if not sig_stats.empty:
            print(f"  Baseline 20d Mean Return: {baseline_val:+.2f}%")
            print(f"  Signale mit |t| > 2 UND besser als Baseline:")
            for _, row in sig_stats.iterrows():
                print(
                    f"    + {row['Signal']:<22}  "
                    f"MR={row['Mean Return']*100:+.2f}%  "
                    f"t={row['t-Stat']:+.2f}  "
                    f"N={int(row['N']):,}"
                )
        else:
            print("  Kein Signal ist statistisch signifikant besser als die Baseline.")
    print()


if __name__ == "__main__":
    main()
