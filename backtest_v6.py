"""
backtest_v6.py
====================================================================================
Portfolio-Backtester v6  |  Systematic Trend Following  |  260 US-Aktien

Strategie:
    ENTRY   Close[T] > 50-Tage-Hoch (High der letzten 50 Tage, OHNE heute)
            UND Close[T] > SMA200[T]
            -> Kauf am naechsten Open[T+1]
            Ranking bei mehreren Kandidaten: (Close - SMA200) / SMA200 absteigend

    EXIT    Close[T] < SMA50[T]  (Trailing Stop)
            -> Verkauf am naechsten Open[T+1]

Portfolio-Regeln:
    Startkapital:        10.000 (EUR/USD, Preise in USD - Verhaeltnis ist irrelevant)
    Max. Positionen:     5 (Equal-Weight: Zielgroesse = Equity / 5)
    Gebühren:            20 fixe Einheiten pro Order (Buy = +20, Sell = +20, Round-Trip = 40)
    Ausfuehrung:         Signal am Close[T] -> Order am Open[T+1]
    Overlap-Sperre:      Kein Neukauf einer bereits gehaltenen Aktie

Ausgaben:
    1. Equity-Kurven-Chart (Portfolio vs. SPY Benchmark) -> backtest_v6_equity.png
    2. Trade-Tabelle (Kontoauszug aller abgeschlossenen Trades) in Konsole
    3. Performance-Kennzahlen (Total Return, Sharpe, Max-DD, Payoff, etc.)

Verwendung:
    python backtest_v6.py
    python backtest_v6.py --years 7 --capital 25000 --fee 10 --max-pos 8
    python backtest_v6.py --market-filter        # Nur kaufen wenn SPY > SMA200
    python backtest_v6.py --no-chart --save-csv  # Kein Chart, Trades als CSV
"""

from __future__ import annotations

import argparse
import json
import math
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

# ---- Default-Konfiguration --------------------------------------------------
DEFAULT_CAPITAL  = 10_000.0
DEFAULT_FEE      = 20.0
DEFAULT_MAX_POS  = 5
MIN_ROWS         = 260          # Warm-up Mindest-Handelstage


# ==============================================================================
# 1. DATEN LADEN
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
            if len(df) >= MIN_ROWS and {"open","high","low","close"}.issubset(df.columns):
                data[ticker] = df[["open","high","low","close"]]
        except Exception:
            pass
    return data


def _load_from_yfinance(tickers: list[str], years: float) -> dict[str, pd.DataFrame]:
    import yfinance as yf, logging
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    end_dt   = date.today()
    start_dt = end_dt - timedelta(days=int(years * 365.25) + 60)
    print(f"  yfinance Download: {start_dt} -> {end_dt} ({len(tickers)} Ticker)...")
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
                data[ticker] = df[["open","high","low","close"]]
        except Exception:
            pass
    return data


def load_universe(tickers: list[str], years: float) -> dict[str, pd.DataFrame]:
    data = _load_from_parquet(tickers, years)
    if data:
        print(f"  {len(data)} Ticker aus Parquet-Dateien.")
        return data
    print("  Keine Parquet-Dateien -- nutze yfinance ...")
    data = _load_from_yfinance(tickers, years)
    print(f"  {len(data)}/{len(tickers)} Ticker geladen.")
    return data


def _download_spy(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series | None:
    """Laedt SPY Schlusskurse als Benchmark."""
    import yfinance as yf, logging
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    try:
        df = yf.download("SPY", start=str(start.date()), end=str(end.date()),
                         auto_adjust=True, progress=False)
        if df.empty:
            return None
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close.index = pd.to_datetime(close.index).tz_localize(None)
        return close.dropna()
    except Exception:
        return None


# ==============================================================================
# 2. PANEL AUFBAUEN  (vektorisiert, Pivot-Tabellen)
# ==============================================================================

def build_pivot_panel(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Berechnet alle Indikatoren pro Ticker und baut Pivot-Tabellen.

    Pivot-Tabellen: DatetimeIndex x Ticker-Spalten.
    Erlaubt O(1)-Zugriff pro Tag fuer die Simulation.

    Returns:
        Dict mit Schluesseln: 'open', 'close', 'exit_sig',
        'entry_sig', 'trend_str'
    """
    opens       : dict[str, pd.Series] = {}
    closes      : dict[str, pd.Series] = {}
    exit_sigs   : dict[str, pd.Series] = {}
    entry_sigs  : dict[str, pd.Series] = {}
    trend_strs  : dict[str, pd.Series] = {}

    for ticker, df in data.items():
        c = df["close"]
        h = df["high"]
        o = df["open"]

        sma50       = c.rolling(50).mean()
        sma200      = c.rolling(200).mean()
        high50_prev = h.shift(1).rolling(50).max()   # Hoch letzter 50 Tage OHNE heute

        exit_sig  = c < sma50                            # Kurs bricht SMA50 nach unten
        entry_sig = (c > high50_prev) & (c > sma200)    # 50-Tage-Breakout + Trendfilter
        trend_str = (c - sma200) / sma200               # Trendstaerke fuer Ranking

        # Nur Zeilen nach abgeschlossener Warm-up-Phase behalten
        valid = sma200.notna() & high50_prev.notna()

        opens[ticker]      = o[valid]
        closes[ticker]     = c[valid]
        exit_sigs[ticker]  = exit_sig[valid]
        entry_sigs[ticker] = entry_sig[valid]
        trend_strs[ticker] = trend_str[valid]

    return {
        "open":      pd.DataFrame(opens),
        "close":     pd.DataFrame(closes),
        "exit_sig":  pd.DataFrame(exit_sigs),
        "entry_sig": pd.DataFrame(entry_sigs),
        "trend_str": pd.DataFrame(trend_strs),
    }


# ==============================================================================
# 3. PORTFOLIO-SIMULATION (tagesweise)
# ==============================================================================

def run_backtest(
    pivots:        dict[str, pd.DataFrame],
    initial_cap:   float = DEFAULT_CAPITAL,
    fee:           float = DEFAULT_FEE,
    max_pos:       int   = DEFAULT_MAX_POS,
    market_filter: pd.Series | None = None,   # SPY > SMA200 fuer jeden Tag
) -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    """Kernlogik der Portfolio-Simulation.

    Args:
        pivots:        Pivot-Tabellen aus build_pivot_panel().
        initial_cap:   Startkapital.
        fee:           Fixe Ordergebuehr (Buy UND Sell je einmal).
        max_pos:       Maximale Anzahl gleichzeitiger Positionen.
        market_filter: Optionale Boolean-Series (Index=Datum).
                       Wenn True fuer Datum T, darf am T+1 gekauft werden.

    Returns:
        equity_df:     DataFrame (Datum, Total-Equity, Cash, Positionen).
        completed:     Liste abgeschlossener Trade-Dicts.
        open_pos_df:   Offene Positionen am Ende.
    """
    df_open      = pivots["open"]
    df_exit      = pivots["exit_sig"]
    df_entry     = pivots["entry_sig"]
    df_trend     = pivots["trend_str"]

    all_dates  = df_open.index.tolist()
    n_dates    = len(all_dates)

    cash       = initial_cap
    portfolio  : dict[str, dict] = {}   # ticker -> {shares, entry_price, entry_date, reason}
    equity_log : list[dict]      = []
    completed  : list[dict]      = []

    for i, today in enumerate(all_dates):

        # ------------------------------------------------------------------ #
        # A) Tageswert berechnen (Anfang des Tages, BEVOR Orders ausgefuehrt)
        # ------------------------------------------------------------------ #
        pos_value = 0.0
        for t, pos in portfolio.items():
            price = df_open.at[today, t] if (today in df_open.index and t in df_open.columns) else np.nan
            if not np.isnan(price):
                pos_value += pos["shares"] * price

        total_equity = cash + pos_value

        equity_log.append({
            "date":       today,
            "equity":     total_equity,
            "cash":       cash,
            "n_pos":      len(portfolio),
        })

        if i == 0:
            continue   # Ersten Tag nur loggen, kein Trading (kein "gestern")

        prev_date = all_dates[i - 1]

        # ------------------------------------------------------------------ #
        # B) EXITS: gestern exit_sig -> heute zum Open verkaufen
        # ------------------------------------------------------------------ #
        if prev_date in df_exit.index:
            prev_exits = df_exit.loc[prev_date]
            for ticker in list(portfolio.keys()):
                if ticker not in prev_exits.index:
                    continue
                if not prev_exits[ticker]:
                    continue                        # kein Exit-Signal gestern

                if today not in df_open.index or ticker not in df_open.columns:
                    continue
                sell_price = df_open.at[today, ticker]
                if np.isnan(sell_price) or sell_price <= 0:
                    continue

                pos        = portfolio.pop(ticker)
                proceeds   = pos["shares"] * sell_price - fee
                cash      += proceeds
                pnl_gross  = pos["shares"] * (sell_price - pos["entry_price"])
                pnl_net    = pnl_gross - 2 * fee    # Kauf-Fee + Verkauf-Fee

                completed.append({
                    "ticker":       ticker,
                    "entry_date":   pos["entry_date"],
                    "entry_price":  pos["entry_price"],
                    "exit_date":    today,
                    "exit_price":   sell_price,
                    "shares":       pos["shares"],
                    "pnl_gross":    pnl_gross,
                    "pnl_net":      pnl_net,
                    "hold_days":    i - pos["entry_day_idx"],
                    "entry_reason": pos["reason"],
                    "exit_reason":  "Close < SMA50",
                    "return":       (sell_price / pos["entry_price"] - 1),
                })

        # ------------------------------------------------------------------ #
        # C) ENTRIES: gestern entry_sig -> heute zum Open kaufen
        # ------------------------------------------------------------------ #
        available_slots = max_pos - len(portfolio)
        if available_slots <= 0:
            continue

        # Optionaler Marktfilter: SPY muss gestern ueber SMA200 gewesen sein
        if market_filter is not None:
            if prev_date not in market_filter.index or not market_filter.get(prev_date, True):
                continue

        if prev_date not in df_entry.index:
            continue

        prev_entries = df_entry.loc[prev_date]
        prev_trends  = df_trend.loc[prev_date]

        # Kandidaten sammeln: entry_sig == True, nicht bereits im Portfolio
        candidates = []
        for ticker in prev_entries.index:
            if not prev_entries[ticker]:
                continue
            if ticker in portfolio:
                continue
            if today not in df_open.index or ticker not in df_open.columns:
                continue

            open_price = df_open.at[today, ticker]
            if np.isnan(open_price) or open_price <= 0:
                continue

            trend_val = prev_trends[ticker] if ticker in prev_trends.index else 0.0
            candidates.append({
                "ticker":    ticker,
                "open":      open_price,
                "trend_str": float(trend_val) if not np.isnan(trend_val) else 0.0,
            })

        # Ranking nach Trendstaerke (absteigend) -> beste zuerst
        candidates.sort(key=lambda x: x["trend_str"], reverse=True)

        # Kaufen (bis max Positionen voll oder Cash erschoepft)
        for cand in candidates[:available_slots]:
            open_price    = cand["open"]
            target_value  = total_equity / max_pos         # Equal-Weight Ziel
            max_spend     = min(target_value, cash - fee)  # darf nicht mehr als Cash

            if max_spend < open_price + fee:
                continue    # zu wenig Cash fuer mindestens 1 Aktie + Gebuehr

            shares = int(math.floor((max_spend - fee) / open_price))
            if shares < 1:
                continue

            cost   = shares * open_price + fee
            cash  -= cost

            trend_pct = cand["trend_str"] * 100
            reason    = f"Breakout_50 | Trend: {trend_pct:+.1f}%"

            portfolio[cand["ticker"]] = {
                "shares":        shares,
                "entry_price":   open_price,
                "entry_date":    today,
                "entry_day_idx": i,
                "cost":          cost,
                "reason":        reason,
            }

    # ---------------------------------------------------------------------- #
    # Offene Positionen am Ende
    # ---------------------------------------------------------------------- #
    open_rows = []
    last_date = all_dates[-1]
    for ticker, pos in portfolio.items():
        last_price = np.nan
        if last_date in df_open.index and ticker in df_open.columns:
            last_price = df_open.at[last_date, ticker]
        unreal_pnl = pos["shares"] * (last_price - pos["entry_price"]) if not np.isnan(last_price) else np.nan
        open_rows.append({
            "ticker":      ticker,
            "entry_date":  pos["entry_date"],
            "entry_price": pos["entry_price"],
            "last_price":  last_price,
            "shares":      pos["shares"],
            "unreal_pnl":  unreal_pnl,
            "return":      (last_price / pos["entry_price"] - 1) if not np.isnan(last_price) else np.nan,
        })

    equity_df    = pd.DataFrame(equity_log).set_index("date")
    open_pos_df  = pd.DataFrame(open_rows)
    return equity_df, completed, open_pos_df


# ==============================================================================
# 4. PERFORMANCE-KENNZAHLEN
# ==============================================================================

def compute_metrics(equity_df: pd.DataFrame, completed: list[dict],
                    initial_cap: float, fee: float) -> dict:
    """Berechnet alle wesentlichen Performance-Kennzahlen."""
    eq = equity_df["equity"]

    # Rendite
    total_ret  = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
    n_years    = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr       = ((eq.iloc[-1] / eq.iloc[0]) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0.0

    # Drawdown
    rolling_max = eq.cummax()
    dd          = (eq - rolling_max) / rolling_max
    max_dd      = dd.min() * 100

    # Sharpe (annualisiert, taeglich)
    daily_ret = eq.pct_change().dropna()
    sharpe    = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0.0

    # Trade-Statistiken
    if completed:
        rets   = [t["return"] for t in completed]
        pnls   = [t["pnl_net"] for t in completed]
        wins   = [r for r in rets if r > 0]
        losses = [r for r in rets if r <= 0]
        hit    = len(wins) / len(rets) * 100
        avg_w  = np.mean(wins) * 100   if wins   else 0.0
        avg_l  = np.mean(losses) * 100 if losses else 0.0
        payoff = abs(avg_w / avg_l)    if avg_l  else np.inf
        exp    = hit / 100 * avg_w + (1 - hit / 100) * avg_l
        total_fees = len(rets) * 2 * fee   # Round-Trip
    else:
        hit = avg_w = avg_l = payoff = exp = total_fees = 0.0

    return {
        "start_date":    eq.index[0].date(),
        "end_date":      eq.index[-1].date(),
        "n_years":       round(n_years, 1),
        "start_capital": initial_cap,
        "end_equity":    round(eq.iloc[-1], 2),
        "total_ret_%":   round(total_ret, 2),
        "cagr_%":        round(cagr, 2),
        "max_dd_%":      round(max_dd, 2),
        "sharpe":        round(sharpe, 2),
        "n_trades":      len(completed),
        "hit_%":         round(hit, 1),
        "avg_win_%":     round(avg_w, 2),
        "avg_loss_%":    round(avg_l, 2),
        "payoff_ratio":  round(payoff, 2),
        "expectancy_%":  round(exp, 2),
        "total_fees":    round(total_fees, 2),
    }


# ==============================================================================
# 5. CHART
# ==============================================================================

def plot_equity_curve(
    equity_df:    pd.DataFrame,
    spy_close:    pd.Series | None,
    metrics:      dict,
    initial_cap:  float,
    output_path:  Path,
    market_filter_active: bool = False,
) -> None:
    """Erstellt Equity-Kurven-Chart (Portfolio vs. SPY Benchmark)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9),
        facecolor="#0d1117",
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12},
    )

    eq   = equity_df["equity"]
    dates = eq.index

    # ---- Equity-Kurve ------------------------------------------------------- #
    ax1.set_facecolor("#161b22")
    ax1.plot(dates, eq, color="#58a6ff", linewidth=2.0,
             label=f"Portfolio v6  ({metrics['total_ret_%']:+.1f}%  |  CAGR {metrics['cagr_%']:+.1f}%)")
    ax1.fill_between(dates, initial_cap, eq,
                     where=(eq >= initial_cap), color="#238636", alpha=0.20)
    ax1.fill_between(dates, initial_cap, eq,
                     where=(eq < initial_cap),  color="#da3633", alpha=0.20)
    ax1.axhline(initial_cap, color="#30363d", linewidth=1, linestyle="--", label="Startkapital")

    # SPY Benchmark
    if spy_close is not None and len(spy_close) >= 2:
        spy_start = spy_close[spy_close.index >= dates[0]]
        if len(spy_start) >= 2:
            spy_norm = spy_start / spy_start.iloc[0] * initial_cap
            spy_ret  = (spy_norm.iloc[-1] / spy_norm.iloc[0] - 1) * 100
            ax1.plot(spy_norm.index, spy_norm, color="#f0883e",
                     linewidth=1.6, linestyle="--",
                     label=f"S&P 500 (SPY)  ({spy_ret:+.1f}%)")

    ax1.set_title(
        f"Backtest v6  |  Breakout_50 + SMA200-Filter + SMA50-Exit"
        + ("  |  Marktfilter (SPY > SMA200)" if market_filter_active else ""),
        color="#e6edf3", fontsize=13, pad=10,
    )
    ax1.set_ylabel("Portfolio-Wert", color="#8b949e", fontsize=10)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:,.0f}"
    ))
    ax1.tick_params(colors="#8b949e", labelsize=8)
    ax1.spines[:].set_color("#30363d")
    ax1.legend(fontsize=9, facecolor="#161b22", edgecolor="#30363d",
               labelcolor="#e6edf3", loc="upper left")
    ax1.grid(axis="y", color="#21262d", linewidth=0.7)
    ax1.set_xlim(dates[0], dates[-1])

    # ---- Drawdown ----------------------------------------------------------- #
    ax2.set_facecolor("#161b22")
    rolling_max = eq.cummax()
    dd = (eq - rolling_max) / rolling_max * 100
    ax2.fill_between(dates, 0, dd, color="#da3633", alpha=0.60)
    ax2.plot(dates, dd, color="#da3633", linewidth=0.8)
    ax2.axhline(0, color="#30363d", linewidth=0.8)

    ax2.set_ylabel("Drawdown %", color="#8b949e", fontsize=9)
    ax2.set_xlabel("Datum", color="#8b949e", fontsize=9)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax2.tick_params(colors="#8b949e", labelsize=8)
    ax2.spines[:].set_color("#30363d")
    ax2.grid(axis="y", color="#21262d", linewidth=0.7)
    ax2.set_xlim(dates[0], dates[-1])

    # Datumsformat
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # Annotation: Kennzahlen
    ann = (
        f"Sharpe: {metrics['sharpe']:.2f}   "
        f"MaxDD: {metrics['max_dd_%']:.1f}%   "
        f"Trades: {metrics['n_trades']}   "
        f"Hit: {metrics['hit_%']:.0f}%   "
        f"Payoff: {metrics['payoff_ratio']:.2f}"
    )
    ax1.text(0.01, 0.03, ann, transform=ax1.transAxes,
             color="#8b949e", fontsize=8.5, va="bottom")

    fig.patch.set_facecolor("#0d1117")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"  Chart gespeichert: {output_path}")


# ==============================================================================
# 6. AUSGABE TABELLEN
# ==============================================================================

def print_metrics(metrics: dict, open_pos_df: pd.DataFrame) -> None:
    """Gibt Performance-Kennzahlen strukturiert aus."""
    print(f"""
  PERFORMANCE-ZUSAMMENFASSUNG
  {"=" * 55}
  Zeitraum:          {metrics['start_date']}  ->  {metrics['end_date']}  ({metrics['n_years']} Jahre)
  Startkapital:          {metrics['start_capital']:>12,.2f}
  Endkapital:            {metrics['end_equity']:>12,.2f}
  Gesamtrendite:         {metrics['total_ret_%']:>+11.2f} %
  CAGR (p.a.):           {metrics['cagr_%']:>+11.2f} %
  Max. Drawdown:         {metrics['max_dd_%']:>+11.2f} %
  Sharpe Ratio:          {metrics['sharpe']:>12.2f}
  {"=" * 55}
  Abgeschlossene Trades: {metrics['n_trades']:>12,}
  Hit-Rate:              {metrics['hit_%']:>11.1f} %
  Avg. Gewinn:           {metrics['avg_win_%']:>+11.2f} %
  Avg. Verlust:          {metrics['avg_loss_%']:>+11.2f} %
  Payoff-Ratio:          {metrics['payoff_ratio']:>12.2f}
  Erwartungswert/Trade:  {metrics['expectancy_%']:>+11.2f} %
  Summe Gebühren:        {metrics['total_fees']:>12,.2f}
  {"=" * 55}
""")

    if not open_pos_df.empty:
        print("  OFFENE POSITIONEN (nicht im Endkapital realisiert):")
        print("  " + "-" * 70)
        for _, row in open_pos_df.iterrows():
            ret = row["return"] * 100 if pd.notna(row["return"]) else float("nan")
            unr = row["unreal_pnl"] if pd.notna(row["unreal_pnl"]) else float("nan")
            print(f"  {row['ticker']:<6}  "
                  f"Entry: {row['entry_date'].date()}  @${row['entry_price']:.2f}  "
                  f"Letzer Kurs: ${row['last_price']:.2f}  "
                  f"Unrealisiert: {ret:>+.1f}%  ({unr:>+.0f})")
        print()


def print_trade_table(completed: list[dict], top_n: int = 999) -> None:
    """Gibt den vollstaendigen Kontoauszug als DataFrame aus."""
    if not completed:
        print("  Keine abgeschlossenen Trades.")
        return

    df = pd.DataFrame(completed)
    df["entry_date"] = pd.to_datetime(df["entry_date"]).dt.strftime("%Y-%m-%d")
    df["exit_date"]  = pd.to_datetime(df["exit_date"]).dt.strftime("%Y-%m-%d")
    df["return_%"]   = (df["return"] * 100).round(2)
    df["pnl_net"]    = df["pnl_net"].round(2)

    display = df[[
        "ticker", "entry_date", "entry_price", "exit_date", "exit_price",
        "shares", "return_%", "pnl_net", "hold_days", "entry_reason", "exit_reason"
    ]].copy()

    display.columns = [
        "Ticker", "Kaufdatum", "Kaufkurs", "Verkaufdatum", "Verkaufkurs",
        "Stueck", "Return-%", "P&L netto", "Haltezeit-d", "Kaufgrund", "Verkaufgrund"
    ]

    display = display.sort_values("Kaufdatum")
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", top_n)
    pd.set_option("display.float_format", "{:.2f}".format)

    print(f"  KONTOAUSZUG  ({len(display)} abgeschlossene Trades):")
    print("  " + "=" * 185)
    print("  " + display.to_string(index=False))
    print("  " + "=" * 185)

    # Top Gewinner / Verlierer
    print(f"\n  TOP 10 GEWINNER:")
    top_w = display.nlargest(10, "Return-%")[["Ticker","Kaufdatum","Verkaufdatum","Return-%","P&L netto","Haltezeit-d"]]
    print("  " + top_w.to_string(index=False))

    print(f"\n  TOP 10 VERLIERER:")
    top_l = display.nsmallest(10, "Return-%")[["Ticker","Kaufdatum","Verkaufdatum","Return-%","P&L netto","Haltezeit-d"]]
    print("  " + top_l.to_string(index=False))


# ==============================================================================
# 7. MAIN
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Portfolio-Backtester v6 | Trend Following"
    )
    parser.add_argument("--years",         type=float, default=7.0,
                        help="Zeitraum in Jahren (Standard: 7)")
    parser.add_argument("--capital",       type=float, default=DEFAULT_CAPITAL,
                        help=f"Startkapital (Standard: {DEFAULT_CAPITAL:,.0f})")
    parser.add_argument("--fee",           type=float, default=DEFAULT_FEE,
                        help=f"Ordergebueehr pro Trade (Standard: {DEFAULT_FEE:.0f})")
    parser.add_argument("--max-pos",       type=int,   default=DEFAULT_MAX_POS,
                        help=f"Max. Positionen (Standard: {DEFAULT_MAX_POS})")
    parser.add_argument("--market-filter", action="store_true",
                        help="Nur kaufen wenn SPY > SMA200 (Markt im Aufwaertstrend)")
    parser.add_argument("--no-chart",      action="store_true",
                        help="Chart-Generierung ueberspringen")
    parser.add_argument("--save-csv",      action="store_true",
                        help="Trade-Tabelle als CSV speichern")
    parser.add_argument("--ticker",        type=str,   default=None,
                        help="Nur diesen Ticker simulieren (Test)")
    args = parser.parse_args()

    print("=" * 70)
    print("  BACKTEST v6  |  Systematic Trend Following  |  260 US-Aktien")
    print("=" * 70)
    print(f"\n  Startkapital:    {args.capital:>10,.0f}")
    print(f"  Max. Positionen: {args.max_pos:>10}")
    print(f"  Ordergebueehr:   {args.fee:>10.0f}  (x2 = {args.fee*2:.0f} pro Round-Trip)")
    print(f"  Zeitraum:        {args.years:.0f} Jahre")
    print(f"  Marktfilter:     {'aktiv (SPY > SMA200)' if args.market_filter else 'inaktiv'}")

    # ---- 1. Ticker --------------------------------------------------------- #
    all_tickers = _load_tickers()
    tickers = [args.ticker] if args.ticker else all_tickers
    print(f"\n[1/5] Universum: {len(tickers)} Ticker")

    # ---- 2. Daten ---------------------------------------------------------- #
    print(f"\n[2/5] OHLCV-Daten ({args.years:.0f} Jahre) ...")
    data = load_universe(tickers, args.years)
    if not data:
        print("FEHLER: Keine Daten. Abbruch.")
        return

    # ---- 3. Panel ---------------------------------------------------------- #
    print(f"\n[3/5] Indikatoren und Pivot-Tabellen aufbauen ...")
    pivots = build_pivot_panel(data)

    dates      = pivots["open"].index
    start_date = dates[0]
    end_date   = dates[-1]
    print(f"  Zeitraum: {start_date.date()} -> {end_date.date()}")
    print(f"  Handelstage: {len(dates):,}")

    # SPY laden (Benchmark + optionaler Marktfilter)
    print(f"\n[4/5] SPY Benchmark laden ...")
    spy_close = _download_spy(start_date, end_date)
    market_filter_series = None

    if spy_close is not None:
        print(f"  SPY: {len(spy_close)} Tage")
        if args.market_filter:
            spy_sma200 = spy_close.rolling(200).mean()
            market_filter_series = (spy_close > spy_sma200).reindex(dates, method="ffill")
            n_filter_days = market_filter_series.sum()
            print(f"  Marktfilter aktiv: {n_filter_days:,}/{len(dates):,} Tage Kaeufe erlaubt")
    else:
        print("  SPY konnte nicht geladen werden.")
        if args.market_filter:
            print("  Marktfilter deaktiviert (keine SPY-Daten).")

    # ---- 4. Simulation ----------------------------------------------------- #
    print(f"\n[5/5] Simulation laeuft ...")
    equity_df, completed, open_pos_df = run_backtest(
        pivots        = pivots,
        initial_cap   = args.capital,
        fee           = args.fee,
        max_pos       = args.max_pos,
        market_filter = market_filter_series,
    )

    print(f"  Abgeschlossene Trades: {len(completed):,}")
    print(f"  Offene Positionen am Ende: {len(open_pos_df)}")

    # ---- Metriken ---------------------------------------------------------- #
    metrics = compute_metrics(equity_df, completed, args.capital, args.fee)
    print_metrics(metrics, open_pos_df)

    # ---- Chart ------------------------------------------------------------- #
    if not args.no_chart:
        out_png = _REPO_ROOT / "backtest_v6_equity.png"
        plot_equity_curve(
            equity_df            = equity_df,
            spy_close            = spy_close,
            metrics              = metrics,
            initial_cap          = args.capital,
            output_path          = out_png,
            market_filter_active = args.market_filter,
        )

    # ---- Trade-Tabelle ----------------------------------------------------- #
    print_trade_table(completed)

    # ---- CSV --------------------------------------------------------------- #
    if args.save_csv and completed:
        csv_path = _REPO_ROOT / "backtest_v6_trades.csv"
        pd.DataFrame(completed).to_csv(csv_path, index=False)
        print(f"\n  Trades gespeichert: {csv_path}  ({len(completed)} Trades)")


if __name__ == "__main__":
    main()
