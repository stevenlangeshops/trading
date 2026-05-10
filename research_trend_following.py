"""
research_trend_following.py
====================================================================================
Trend-Following Trade-Simulation  |  260 US-Aktien  |  Trading v6 Vorstudie

Strategie:
    Kein fixer Zeithorizont.  Wir "reiten" den Trend, bis er bricht.

    ENTRY  Close[T] bricht uber N-Tage-Hoch (Intraday-High) UND liegt uber SMA200
           Kauf am naechsten Handelstag T+1 zum Open-Preis.

    EXIT   Close[T] faellt unter SMA50 (Trailing Stop)
           Verkauf am naechsten Handelstag T+1 zum Open-Preis.

    OVERLAP-REGEL
           Solange eine Position offen ist, werden neue Entry-Signale fuer
           denselben Ticker ignoriert.  Pro Ticker max. 1 offene Position.

Entry-Signale:
    Breakout_50    Close > 50-Tage-Hoch (High) UND Close > SMA200
    Breakout_100   Close > 100-Tage-Hoch (High) UND Close > SMA200
    Breakout_50_nofilter  Close > 50-Tage-Hoch ohne SMA200-Filter (Vergleich)

Trade-Statistiken:
    N               Anzahl abgeschlossener Trades
    Hit-%           % der Trades mit Return > 0
    Avg-Win-%       Durchschn. Rendite aller Gewinner-Trades
    Avg-Loss-%      Durchschn. Rendite aller Verlierer-Trades
    Payoff-Ratio    abs(Avg-Win / Avg-Loss)  -- > 1.0 = gut
    Expectancy-%    Hit-Rate * Avg-Win + (1-Hit-Rate) * Avg-Loss
    Profit-Factor   Summe Gewinne / abs(Summe Verluste)  -- > 1.0 = profitabel
    Max-Win-%       Groesster Einzelgewinn (zeigt die "Fat Tails"!)
    Max-Loss-%      Groesster Einzelverlust
    Avg-Hold        Durchschn. Haltedauer (Handelstage)
    Median-Hold     Medianer Hold

Verwendung:
    python research_trend_following.py
    python research_trend_following.py --years 7
    python research_trend_following.py --top 20       # Top/Bottom Trades
    python research_trend_following.py --ticker NVDA  # Einzelner Ticker
    python research_trend_following.py --save-csv
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---- Pfade ------------------------------------------------------------------
_REPO_ROOT  = Path(__file__).parent.resolve()
_RAW_DIR    = _REPO_ROOT / "data" / "raw"
_SECTOR_MAP = _REPO_ROOT / "features" / "sector_map.json"

# ---- Konfiguration ----------------------------------------------------------
MIN_ROWS = 260        # Mindest-Handelstage pro Ticker (Warm-up)


# ==============================================================================
# 1. DATEN LADEN  (Parquet -> yfinance Fallback)
# ==============================================================================

def _load_tickers() -> list[str]:
    raw = json.loads(_SECTOR_MAP.read_text())
    return sorted(t for t in raw if not t.startswith("_"))


def _load_from_parquet(tickers: list[str], years: float) -> dict[str, pd.DataFrame]:
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
            required = {"open", "high", "low", "close"}
            if len(df) >= MIN_ROWS and required.issubset(df.columns):
                data[ticker] = df[list(required)]
        except Exception:
            pass
    return data


def _load_from_yfinance(tickers: list[str], years: float) -> dict[str, pd.DataFrame]:
    import yfinance as yf, logging
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    end_dt   = date.today()
    start_dt = end_dt - timedelta(days=int(years * 365.25) + 60)
    print(f"  yfinance Batch-Download: {start_dt} -> {end_dt} ({len(tickers)} Ticker)...")
    raw = yf.download(tickers, start=str(start_dt), end=str(end_dt),
                      auto_adjust=True, progress=False, threads=True)
    if raw.empty:
        raise RuntimeError("yfinance lieferte keine Daten.")
    data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            df = raw.xs(ticker, axis=1, level=1).copy() \
                if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
            df.columns = [c.lower() for c in df.columns]
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.dropna(subset=["close"])
            if len(df) >= MIN_ROWS:
                data[ticker] = df[["open", "high", "low", "close"]]
        except Exception:
            pass
    return data


def load_universe(tickers: list[str], years: float) -> dict[str, pd.DataFrame]:
    data = _load_from_parquet(tickers, years)
    if data:
        print(f"  {len(data)} Ticker aus Parquet-Dateien.")
        return data
    print("  Keine Parquets gefunden -- nutze yfinance...")
    data = _load_from_yfinance(tickers, years)
    print(f"  {len(data)}/{len(tickers)} Ticker geladen.")
    return data


# ==============================================================================
# 2. INDIKATOREN (vektorisiert, kein Look-Ahead)
# ==============================================================================

def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Berechnet alle benotigten Indikatoren fuer einen Ticker.

    Look-Ahead-Bias-Schutz:
        N-Tage-Hoch:  shift(1) -> rollendes Max ueber High[T-N .. T-1]
        SMA50/200:    close.rolling(N).mean() beinhaltet Close[T], das ist
                      korrekt, da wir es mit Close[T] vergleichen (gleicher Tag).
    """
    c = df["close"]
    h = df["high"]

    df = df.copy()
    df["sma50"]     = c.rolling(50).mean()
    df["sma200"]    = c.rolling(200).mean()
    df["high50"]    = h.shift(1).rolling(50).max()    # Hoch der letzten 50 Tage (OHNE heute)
    df["high100"]   = h.shift(1).rolling(100).max()   # Hoch der letzten 100 Tage

    # Entry-Bedingungen (True/False)
    df["sig_b50"]   = (c > df["high50"])  & (c > df["sma200"])
    df["sig_b100"]  = (c > df["high100"]) & (c > df["sma200"])
    df["sig_b50nf"] = (c > df["high50"])                         # kein SMA200-Filter

    # Exit-Bedingung
    df["exit_cond"] = c < df["sma50"]

    return df


# ==============================================================================
# 3. TRADE-SIMULATION (State Machine pro Ticker)
# ==============================================================================

SIGNAL_COLS = {
    "Breakout_50":        "sig_b50",
    "Breakout_100":       "sig_b100",
    "Breakout_50_noSMA":  "sig_b50nf",
}


def _simulate_ticker(df: pd.DataFrame, ticker: str) -> list[dict]:
    """State-Machine-Simulation fuer einen Ticker und alle Signale.

    Gibt eine Liste von Trade-Dicts zurueck.
    Offene Trades am Ende des Datensatzes werden NICHT gezaehlt.
    """
    arr_open  = df["open"].values
    arr_close = df["close"].values
    arr_sma50 = df["sma50"].values
    arr_exit  = df["exit_cond"].values
    idx       = df.index
    n         = len(df)

    all_trades = []

    for sig_name, sig_col in SIGNAL_COLS.items():
        arr_sig = df[sig_col].values

        in_trade      = False
        entry_idx     = -1
        entry_price   = np.nan

        for i in range(1, n):
            if np.isnan(arr_open[i]):
                continue

            if not in_trade:
                # Entry: Signal feuerte gestern -> Kauf heute zum Open
                if arr_sig[i - 1]:
                    entry_price = arr_open[i]
                    if np.isnan(entry_price) or entry_price <= 0:
                        continue
                    in_trade  = True
                    entry_idx = i

            else:
                # Exit: gesterns Close < SMA50 -> Verkauf heute zum Open
                if arr_exit[i - 1]:
                    exit_price = arr_open[i]
                    if np.isnan(exit_price) or exit_price <= 0:
                        in_trade = False
                        continue

                    ret        = exit_price / entry_price - 1.0
                    hold_days  = i - entry_idx          # Handelstage

                    all_trades.append({
                        "signal":       sig_name,
                        "ticker":       ticker,
                        "entry_date":   idx[entry_idx],
                        "exit_date":    idx[i],
                        "entry_price":  round(entry_price, 4),
                        "exit_price":   round(exit_price, 4),
                        "return":       ret,
                        "hold_days":    hold_days,
                    })
                    in_trade = False

    return all_trades


# ==============================================================================
# 4. ALLE TICKER SIMULIEREN
# ==============================================================================

def run_simulation(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Fuehrt die Trade-Simulation fuer alle Ticker durch.

    Returns:
        DataFrame mit allen abgeschlossenen Trades (eine Zeile pro Trade).
    """
    all_trades = []
    n_ok = n_fail = 0

    for ticker, raw_df in data.items():
        try:
            df = _add_indicators(raw_df)
            # Nur Zeilen mit vollstaendigen Indikatoren (nach Warm-up-Phase)
            df = df.dropna(subset=["sma200", "high100"])
            trades = _simulate_ticker(df, ticker)
            all_trades.extend(trades)
            n_ok += 1
        except Exception:
            n_fail += 1

    print(f"  Simulation: {n_ok} Ticker OK, {n_fail} fehlgeschlagen.")
    print(f"  Abgeschlossene Trades gesamt: {len(all_trades):,}")

    if not all_trades:
        return pd.DataFrame()

    trades_df = pd.DataFrame(all_trades)
    trades_df["win"] = trades_df["return"] > 0
    return trades_df


# ==============================================================================
# 5. STATISTIKEN
# ==============================================================================

def _stats_for_group(trades: pd.DataFrame, signal: str) -> dict:
    """Berechnet alle Kennzahlen fuer eine Gruppe von Trades."""
    n       = len(trades)
    returns = trades["return"]
    wins    = returns[returns > 0]
    losses  = returns[returns <= 0]

    avg_win  = wins.mean()   if len(wins)   > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0
    hit_rate = len(wins) / n

    payoff    = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
    exp       = hit_rate * avg_win + (1 - hit_rate) * avg_loss
    sum_wins  = wins.sum()
    sum_loss  = abs(losses.sum())
    pf        = sum_wins / sum_loss if sum_loss > 0 else np.inf

    hold      = trades["hold_days"]

    return {
        "Signal":           signal,
        "N":                n,
        "Hit-%":            hit_rate * 100,
        "Avg-Win-%":        avg_win * 100,
        "Avg-Loss-%":       avg_loss * 100,
        "Payoff-Ratio":     payoff,
        "Expectancy-%":     exp * 100,
        "Profit-Factor":    pf,
        "Max-Win-%":        returns.max() * 100,
        "Max-Loss-%":       returns.min() * 100,
        "Avg-Hold-Days":    hold.mean(),
        "Median-Hold-Days": hold.median(),
        "N-Winners":        len(wins),
        "N-Losers":         len(losses),
    }


def compute_summary(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Erstellt die Zusammenfassungstabelle pro Signal."""
    rows = []

    # Gesamt (alle Signale zusammen als Referenz)
    rows.append(_stats_for_group(trades_df, "** ALLE SIGNALE **"))

    # Pro Signal
    for sig in trades_df["signal"].unique():
        grp = trades_df[trades_df["signal"] == sig]
        rows.append(_stats_for_group(grp, sig))

    df = pd.DataFrame(rows).set_index("Signal")
    return df.sort_values("Expectancy-%", ascending=False)


# ==============================================================================
# 6. AUSGABE
# ==============================================================================

def print_summary(summary: pd.DataFrame) -> None:
    """Gibt die Haupt-Zusammenfassungstabelle aus."""
    fmt = summary.copy()
    pct_cols = ["Hit-%", "Avg-Win-%", "Avg-Loss-%", "Expectancy-%",
                "Max-Win-%", "Max-Loss-%"]
    for c in pct_cols:
        if c in fmt:
            fmt[c] = fmt[c].map(lambda x: f"{x:+.2f}%")
    for c in ["Payoff-Ratio", "Profit-Factor"]:
        if c in fmt:
            fmt[c] = fmt[c].map(lambda x: f"{x:.2f}" if np.isfinite(x) else "inf")
    for c in ["Avg-Hold-Days", "Median-Hold-Days"]:
        if c in fmt:
            fmt[c] = fmt[c].map(lambda x: f"{x:.1f}d")
    for c in ["N", "N-Winners", "N-Losers"]:
        if c in fmt:
            fmt[c] = fmt[c].map(lambda x: f"{int(x):,}")

    pd.set_option("display.max_columns", 30)
    pd.set_option("display.width", 200)

    line = "=" * 170
    print(f"\n{line}")
    print("  TREND-FOLLOWING SIMULATION  |  Ergebnis-Uebersicht")
    print(f"  Entry: Kauf am naechsten Open  |  Exit: SMA50-Bruch -> Verkauf am naechsten Open")
    print(line)
    print(fmt.drop(columns=["N-Winners", "N-Losers"]).to_string())
    print(line)
    print()
    print("  Legende:")
    print("  Payoff-Ratio   = abs(Avg-Win / Avg-Loss)  -- > 1.0 = jeder Gewinner > jeder Verlierer")
    print("  Expectancy-%   = Hit-Rate * Avg-Win + (1-Hit-Rate) * Avg-Loss  -- muss > 0 sein")
    print("  Profit-Factor  = Summe Gewinne / Summe Verluste  -- > 1.0 = profitabel gesamt")
    print("  Max-Win-%      = Groesster Einzeltrade (zeigt die 'Fat Tails'!)")
    print()


def print_return_distribution(trades_df: pd.DataFrame) -> None:
    """Druckt Rendite-Histogramm je Signal (ASCII)."""
    bins = [-1.0, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.50, 1.0, float("inf")]
    labels = [
        "< -20%", "-20%...-10%", "-10%...-5%", "-5%...0%",
        "0%...+5%", "+5%...+10%", "+10%...+20%", "+20%...+50%",
        "+50%...+100%", "> +100%"
    ]

    print("  RENDITE-VERTEILUNG (alle abgeschlossenen Trades):")
    print(f"  {'Rendite-Klasse':<18}  {'|':1}", end="")

    for sig in ["Breakout_50", "Breakout_100", "Breakout_50_noSMA"]:
        print(f"  {sig:<18}", end="")
    print()
    print("  " + "-" * 90)

    sig_trades = {
        sig: trades_df[trades_df["signal"] == sig]["return"].values
        for sig in ["Breakout_50", "Breakout_100", "Breakout_50_noSMA"]
        if sig in trades_df["signal"].values
    }

    for i, label in enumerate(labels):
        lo = bins[i]
        hi = bins[i + 1]
        print(f"  {label:<18}  |", end="")
        for sig, rets in sig_trades.items():
            mask = (rets > lo) & (rets <= hi) if hi != float("inf") else (rets > lo)
            pct  = mask.sum() / len(rets) * 100 if len(rets) > 0 else 0.0
            bar  = "#" * int(pct / 2)
            print(f"  {pct:>4.1f}% {bar:<25}", end="")
        print()
    print()


def print_top_trades(trades_df: pd.DataFrame, top_n: int = 15) -> None:
    """Druckt die groessten Gewinner- und Verlierer-Trades."""
    cols = ["signal", "ticker", "entry_date", "exit_date",
            "entry_price", "exit_price", "return", "hold_days"]

    winners = (trades_df.nlargest(top_n, "return")[cols].copy())
    losers  = (trades_df.nsmallest(top_n, "return")[cols].copy())

    for df_t, title in [(winners, f"TOP {top_n} GEWINNER-TRADES"),
                        (losers,  f"TOP {top_n} VERLIERER-TRADES")]:
        df_t = df_t.copy()
        df_t["return"] = df_t["return"].map(lambda x: f"{x*100:+.1f}%")
        df_t["entry_date"] = df_t["entry_date"].dt.strftime("%Y-%m-%d")
        df_t["exit_date"]  = df_t["exit_date"].dt.strftime("%Y-%m-%d")
        df_t["entry_price"] = df_t["entry_price"].map(lambda x: f"${x:.2f}")
        df_t["exit_price"]  = df_t["exit_price"].map(lambda x: f"${x:.2f}")
        df_t.columns = ["Signal", "Ticker", "Entry", "Exit",
                         "Entry-$", "Exit-$", "Return", "Hold-d"]
        print(f"  {title}:")
        print("  " + "-" * 100)
        print("  " + df_t.to_string(index=False))
        print()


def print_yearly_breakdown(trades_df: pd.DataFrame) -> None:
    """Rendite und Hit-Rate aufgeschlusselt nach Entry-Jahr."""
    trades_df = trades_df.copy()
    trades_df["year"] = pd.to_datetime(trades_df["entry_date"]).dt.year

    print("  JAEHRLICHE AUFSCHLUESSLUNG (nach Entry-Jahr, Signal: Breakout_50):")
    print(f"  {'Jahr':<6}  {'N':>5}  {'Hit-%':>7}  {'Avg-Ret':>9}  "
          f"{'Avg-Win':>9}  {'Avg-Loss':>9}  {'Payoff':>7}  {'Max-Win':>9}")
    print("  " + "-" * 75)

    sub = trades_df[trades_df["signal"] == "Breakout_50"]
    for year, grp in sub.groupby("year"):
        rets  = grp["return"]
        wins  = rets[rets > 0]
        losses= rets[rets <= 0]
        hit   = len(wins) / len(rets) * 100
        avg_w = wins.mean() * 100  if len(wins)   else 0.0
        avg_l = losses.mean() * 100 if len(losses) else 0.0
        payoff = abs(avg_w / avg_l) if avg_l != 0 else 99.9
        print(
            f"  {year:<6}  {len(rets):>5,}  {hit:>6.1f}%  "
            f"{rets.mean()*100:>+8.2f}%  {avg_w:>+8.2f}%  "
            f"{avg_l:>+8.2f}%  {payoff:>7.2f}  {rets.max()*100:>+8.1f}%"
        )
    print()


def print_sector_breakdown(trades_df: pd.DataFrame,
                           signal: str = "Breakout_50") -> None:
    """Zeigt ob Trend-Following sektoren-uebergreifend funktioniert."""
    sector_map = json.loads(_SECTOR_MAP.read_text())
    sub = trades_df[trades_df["signal"] == signal].copy()
    sub["sector"] = sub["ticker"].map(sector_map)

    print(f"  SEKTOR-BREAKDOWN (Signal: {signal}):")
    print(f"  {'Sektor':<30}  {'N':>5}  {'Hit-%':>7}  "
          f"{'Avg-Ret':>9}  {'Payoff':>7}  {'Max-Win':>9}")
    print("  " + "-" * 75)

    for sector, grp in sub.groupby("sector"):
        rets  = grp["return"]
        wins  = rets[rets > 0]
        losses= rets[rets <= 0]
        hit   = len(wins) / len(rets) * 100 if len(rets) > 0 else 0.0
        avg_w = wins.mean()   if len(wins)   > 0 else 0.0
        avg_l = losses.mean() if len(losses) > 0 else 0.0
        payoff = abs(avg_w / avg_l) if avg_l != 0 else 99.9
        print(
            f"  {str(sector):<30}  {len(rets):>5,}  {hit:>6.1f}%  "
            f"{rets.mean()*100:>+8.2f}%  {payoff:>7.2f}  {rets.max()*100:>+8.1f}%"
        )
    print()


def print_holding_distribution(trades_df: pd.DataFrame) -> None:
    """ASCII-Histogramm der Haltedauern."""
    bins   = [0, 5, 10, 20, 40, 60, 90, 120, 180, 365, 9999]
    labels = ["1-5d", "6-10d", "11-20d", "21-40d", "41-60d",
              "61-90d", "91-120d", "121-180d", "181-365d", ">365d"]

    print("  HALTEDAUER-VERTEILUNG (Handelstage):")
    print(f"  {'Klasse':<12}  {'|':1}", end="")
    for sig in ["Breakout_50", "Breakout_100"]:
        print(f"  {sig:<18}", end="")
    print()
    print("  " + "-" * 65)

    sig_hold = {
        sig: trades_df[trades_df["signal"] == sig]["hold_days"].values
        for sig in ["Breakout_50", "Breakout_100"]
        if sig in trades_df["signal"].values
    }

    for i, label in enumerate(labels):
        lo = bins[i];  hi = bins[i + 1]
        print(f"  {label:<12}  |", end="")
        for sig, holds in sig_hold.items():
            mask = (holds > lo) & (holds <= hi)
            pct  = mask.sum() / len(holds) * 100 if len(holds) > 0 else 0.0
            bar  = "#" * int(pct / 2)
            print(f"  {pct:>4.1f}% {bar:<25}", end="")
        print()
    print()


# ==============================================================================
# 7. MAIN
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trend-Following Trade-Simulation | Trading v6"
    )
    parser.add_argument("--years",    type=float, default=7.0,
                        help="Historische Zeitspanne in Jahren (Standard: 7)")
    parser.add_argument("--top",      type=int,   default=15,
                        help="Anzahl Top/Bottom Trades (Standard: 15)")
    parser.add_argument("--ticker",   type=str,   default=None,
                        help="Nur diesen Ticker analysieren (z.B. NVDA)")
    parser.add_argument("--no-top",   action="store_true",
                        help="Top-Trade-Tabellen ausblenden")
    parser.add_argument("--no-dist",  action="store_true",
                        help="Verteilungs-Tabellen ausblenden")
    parser.add_argument("--save-csv", action="store_true",
                        help="Alle Trades als CSV speichern")
    args = parser.parse_args()

    print("=" * 70)
    print("  TREND-FOLLOWING TRADE-SIMULATION  |  Trading v6 Vorstudie")
    print("=" * 70)

    # 1. Ticker laden
    all_tickers = _load_tickers()
    tickers = [args.ticker] if args.ticker else all_tickers
    print(f"\n[1/3] Universum: {len(tickers)} Ticker")

    # 2. Daten laden
    print(f"\n[2/3] OHLCV-Daten ({args.years:.0f} Jahre) ...")
    data = load_universe(tickers, args.years)
    if not data:
        print("FEHLER: Keine Daten geladen.")
        return

    # 3. Simulation
    print(f"\n[3/3] Trade-Simulation laeuft ...")
    trades_df = run_simulation(data)
    if trades_df.empty:
        print("Keine abgeschlossenen Trades gefunden.")
        return

    # Ausgabe
    summary = compute_summary(trades_df)
    print_summary(summary)

    if not args.no_top:
        print_top_trades(trades_df, args.top)

    print_yearly_breakdown(trades_df)
    print_sector_breakdown(trades_df, "Breakout_50")

    if not args.no_dist:
        print_return_distribution(trades_df)
        print_holding_distribution(trades_df)

    if args.save_csv:
        path = _REPO_ROOT / "trend_following_trades.csv"
        trades_df.to_csv(path, index=False)
        print(f"  Alle Trades gespeichert: {path}  ({len(trades_df):,} Trades)")

    # Schnell-Fazit
    print("  SCHNELL-FAZIT:")
    print("  " + "-" * 65)
    for sig in ["Breakout_50", "Breakout_100", "Breakout_50_noSMA"]:
        sub = trades_df[trades_df["signal"] == sig]
        if sub.empty:
            continue
        wins   = sub[sub["return"] > 0]["return"]
        losses = sub[sub["return"] <= 0]["return"]
        hit    = len(wins) / len(sub) * 100
        payoff = abs(wins.mean() / losses.mean()) if len(losses) and losses.mean() != 0 else 0
        exp    = (hit/100 * wins.mean() + (1 - hit/100) * losses.mean()) * 100
        pf     = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else 0
        icon   = "+" if exp > 0 else "-"
        print(f"  [{icon}] {sig:<22}  N={len(sub):>4,}  Hit={hit:.1f}%  "
              f"Payoff={payoff:.2f}  Exp={exp:+.2f}%  PF={pf:.2f}  "
              f"Max-Win={sub['return'].max()*100:+.0f}%")
    print()


if __name__ == "__main__":
    main()
