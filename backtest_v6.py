"""
backtest_v6.py
====================================================================================
Portfolio-Backtester v6.3  |  Systematic Trend Following  |  260 US-Aktien

STRATEGIE-UPGRADES v6.3  (Asymmetric Stops & Capital Protection):
  ──────────────────────────────────────────────────────────────────
  1. HARDCODED Finanzielle Realitäten
     INITIAL_CAPITAL = 10.000 €
     ORDER_FEE       = 20.00 € pro Order (Round-Trip = 40 €!)

  2. Asymmetrischer "Earned" Trailing Stop (zwei Phasen):
     Phase 1 – TIGHT (Position muss sich beweisen):
       Stop  = Kaufpreis  -  ATR_INIT × ATR14_at_Entry
       → Enger anfänglicher Schutz gegen Fake-Breakouts
     Phase 2 – EARNED (Position hat Profit bewiesen):
       Trigger: max_High_seit_Kauf > Kaufpreis + ATR_INIT × ATR14_at_Entry
       Stop  = MaxHigh_seit_Kauf  -  ATR_TRAIL × ATR14_heute
       → Großzügiger Trailing Stop auf dem "Hausberg" der Aktie
     Eiserne Regel: Stop darf NIEMALS sinken!

  3. Free-Ride Schutz beim Pyramidisieren:
     Sobald eine Position aufgestockt wird, wird der Stop SOFORT auf
     max(aktueller_Stop, neuer_Avg_Entry_Price) gesetzt.
     → Auf einer pyramidisierten Position kann KEIN Geld verloren werden!

  4. Bewährte Regeln (unverändert):
     Rotations-Bremse: Faktor 2.0×, NUR profitable Positionen
     Max. 5 Slots, gleichgewichtet

Kern-Strategie:
    ENTRY   Close[T] > 50-Tage-Hoch UND Close[T] > SMA200
            → Kauf am nächsten Open[T+1]
            Ranking: (Close – SMA200) / SMA200 absteigend
    EXIT    Asymmetrischer ATR Stop (Phasen-Wechsel, niemals sinkend)
            → Verkauf am nächsten Open[T+1]

Verwendung:
    python backtest_v6.py                         # 10.000 €, 20 € Fee
    python backtest_v6.py --capital 100000 --fee 5
    python backtest_v6.py --atr-init 2.5 --atr-trail 4.0
    python backtest_v6.py --market-filter --save-csv
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

# =============================================================================
# HARDCODED FINANZIELLE REALITÄTEN  (v6.3)
# =============================================================================
INITIAL_CAPITAL = 10_000.0      # 10.000 EUR – nicht verhandelbar!
ORDER_FEE       = 20.0          # 20 EUR fix pro Order → Round-Trip = 40 EUR

# ---- Strategie-Parameter (konfigurierbar per CLI) --------------------------
DEFAULT_MAX_POS           = 5
DEFAULT_ATR_PERIOD        = 14
DEFAULT_ATR_INIT          = 2.0   # Tight initial stop: Entry - N×ATR
DEFAULT_ATR_TRAIL         = 3.5   # Earned trailing stop: MaxHigh - N×ATR
DEFAULT_ROTATION_FACTOR   = 2.0   # Rotation: Kandidat muss 2× stärker sein
DEFAULT_PYRAMID_THRESHOLD = 0.20  # Pyramidisieren ab +20% Unrealized Profit
DEFAULT_MAX_PYRAMIDS      = 2     # Max. Aufstockungen pro Position
MIN_ROWS                  = 260


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
            needed = {"open", "high", "low", "close"}
            if len(df) >= MIN_ROWS and needed.issubset(df.columns):
                data[ticker] = df[sorted(needed)]
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
    print("  Keine Parquet-Dateien -- nutze yfinance...")
    data = _load_from_yfinance(tickers, years)
    print(f"  {len(data)}/{len(tickers)} Ticker geladen.")
    return data


def _download_spy(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series | None:
    import yfinance as yf, logging
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    try:
        df = yf.download("SPY", start=str(start.date()), end=str(end.date()),
                         auto_adjust=True, progress=False)
        if df.empty:
            return None
        s = df["Close"]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return s.dropna()
    except Exception:
        return None


# ==============================================================================
# 2. PANEL AUFBAUEN  (vektorisiert, Date × Ticker)
# ==============================================================================

def _atr(df: pd.DataFrame, period: int = DEFAULT_ATR_PERIOD) -> pd.Series:
    """Average True Range (Wilder-Methode)."""
    h, l, c = df["high"], df["low"], df["close"]
    prev_c   = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def build_pivot_panel(data: dict[str, pd.DataFrame],
                      atr_period: int = DEFAULT_ATR_PERIOD) -> dict[str, pd.DataFrame]:
    """Berechnet Indikatoren, baut Date×Ticker Pivot-Tabellen.

    Keys: open, close, high, atr14, entry_sig, trend_str
    """
    opens, closes, highs, atrs = {}, {}, {}, {}
    entry_sigs, trend_strs    = {}, {}

    for ticker, df in data.items():
        c           = df["close"]
        h           = df["high"]
        sma200      = c.rolling(200).mean()
        high50_prev = h.shift(1).rolling(50).max()
        atr14       = _atr(df, atr_period)
        trend_str   = (c - sma200) / sma200
        entry_sig   = (c > high50_prev) & (c > sma200)
        valid       = sma200.notna() & high50_prev.notna() & atr14.notna()

        opens[ticker]      = df["open"][valid]
        closes[ticker]     = c[valid]
        highs[ticker]      = h[valid]
        atrs[ticker]       = atr14[valid]
        entry_sigs[ticker] = entry_sig[valid]
        trend_strs[ticker] = trend_str[valid]

    return {
        "open":      pd.DataFrame(opens),
        "close":     pd.DataFrame(closes),
        "high":      pd.DataFrame(highs),
        "atr14":     pd.DataFrame(atrs),
        "entry_sig": pd.DataFrame(entry_sigs),
        "trend_str": pd.DataFrame(trend_strs),
    }


# ==============================================================================
# 3. ASYMMETRISCHER TRAILING STOP  (Kernlogik v6.3)
# ==============================================================================

def _compute_stop(pos: dict, today_high: float, today_atr: float,
                  atr_init: float, atr_trail: float,
                  ticker: str, today, verbose: bool,
                  mode_log: list) -> None:
    """Aktualisiert den Trailing Stop für eine Position (Two-Phase-Mechanik).

    Phase 1 – TIGHT:
        Stop = Kaufpreis – atr_init × ATR_at_Entry  (fest, kein Nachziehen)
        Trigger-Bedingung: max_high >= Kaufpreis + atr_init × ATR_at_Entry

    Phase 2 – EARNED (nach Trigger):
        Stop = max_high – atr_trail × ATR_heute  (trails nach oben)

    Eiserne Regel: trailing_stop darf NIEMALS sinken.
    """
    if math.isnan(today_high) or math.isnan(today_atr) or today_atr <= 0:
        return

    # Max High aktualisieren
    if today_high > pos["max_high"]:
        pos["max_high"] = today_high

    # Phase-Wechsel prüfen
    if not pos["earned_mode"]:
        earned_trigger = pos["entry_price"] + atr_init * pos["atr_at_entry"]
        if pos["max_high"] >= earned_trigger:
            pos["earned_mode"] = True
            msg = (f"  [EARNED MODE] {ticker}  "
                   f"MaxHigh ${pos['max_high']:.2f} > Trigger ${earned_trigger:.2f}  "
                   f"(Entry ${pos['entry_price']:.2f} + {atr_init}×ATR)  "
                   f"-> Wechsel auf {atr_trail}×ATR Trailing Stop")
            mode_log.append(msg)
            if verbose:
                print(msg)

    # Neuen Stop berechnen
    if pos["earned_mode"]:
        new_stop = pos["max_high"] - atr_trail * today_atr
    else:
        # Tight: Stop bleibt am Entry-ATR verankert (kein Nachziehen in Phase 1)
        new_stop = pos["entry_price"] - atr_init * pos["atr_at_entry"]

    # Stop niemals senken
    pos["trailing_stop"] = max(pos["trailing_stop"], new_stop)


# ==============================================================================
# 4. SIMULATION (tagesweise State Machine)
# ==============================================================================

def run_backtest(
    pivots:             dict[str, pd.DataFrame],
    initial_cap:        float = INITIAL_CAPITAL,
    fee:                float = ORDER_FEE,
    max_pos:            int   = DEFAULT_MAX_POS,
    atr_init:           float = DEFAULT_ATR_INIT,
    atr_trail:          float = DEFAULT_ATR_TRAIL,
    rotation_factor:    float = DEFAULT_ROTATION_FACTOR,
    pyramid_threshold:  float = DEFAULT_PYRAMID_THRESHOLD,
    max_pyramids:       int   = DEFAULT_MAX_PYRAMIDS,
    enable_pyramid:     bool  = True,
    market_filter:      pd.Series | None = None,
    verbose:            bool  = True,
) -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    """Portfolio-Simulation v6.3.

    TAGESABLAUF (heute = T+1, prev = T):
      1. Equity loggen
      2. EXITS  : close[T] < trailing_stop[T] → Verkauf bei open[T+1]
      3. PYRAMIDS: pyramid_flag → Aufstocken bei open[T+1] + FREE-RIDE Stop
      4. ROTATION: 2× Faktor, nur profitable Positionen
      5. ENTRIES : freie Slots mit Breakout-Kandidaten füllen
      6. STOP UPDATE: asymmetrischer Two-Phase Stop nachziehen
      7. PYRAMID FLAG setzen (wenn close > +threshold%)

    Position-State:
      shares, units, entry_price (gewichteter Avg), entry_date, entry_day_idx,
      reason, max_high, trailing_stop,
      earned_mode (bool: Phase 1 tight / Phase 2 trailing),
      atr_at_entry (für Phase-1-Berechnung + Earned-Trigger),
      pyramid_flag, pyramid_count
    """
    df_open  = pivots["open"]
    df_close = pivots["close"]
    df_high  = pivots["high"]
    df_atr   = pivots["atr14"]
    df_entry = pivots["entry_sig"]
    df_trend = pivots["trend_str"]

    all_dates    = df_open.index.tolist()
    slot_size    = initial_cap / max_pos
    cash         = initial_cap
    portfolio    : dict[str, dict]  = {}
    equity_log   : list[dict]       = []
    completed    : list[dict]       = []
    n_rotations  = 0
    n_pyramids   = 0
    pyramid_log  : list[str]        = []
    mode_log     : list[str]        = []   # Earned-Mode Übergänge

    # ---------- Hilfsfunktionen ------------------------------------------

    def _total_units() -> int:
        return sum(p.get("units", 1) for p in portfolio.values())

    def _safe_open(ticker: str) -> float:
        try:
            v = df_open.at[today, ticker]
            return float(v) if pd.notna(v) else math.nan
        except Exception:
            return math.nan

    def _safe_atr(date_, ticker: str) -> float:
        try:
            v = df_atr.at[date_, ticker]
            return float(v) if pd.notna(v) else math.nan
        except Exception:
            return math.nan

    def _sell(ticker: str, sell_price: float, sell_date, day_idx: int,
              reason: str) -> None:
        nonlocal cash
        pos      = portfolio.pop(ticker)
        proceeds = pos["shares"] * sell_price - fee
        cash    += proceeds
        # Round-Trip Gebühren: 1 Kauf + alle Pyramiden-Käufe + dieser Verkauf
        total_orders = 1 + pos.get("pyramid_count", 0) + 1
        pnl_net = pos["shares"] * (sell_price - pos["entry_price"]) \
                  - total_orders * fee
        completed.append({
            "ticker":          ticker,
            "entry_date":      pos["entry_date"],
            "entry_price":     pos["entry_price"],
            "exit_date":       sell_date,
            "exit_price":      sell_price,
            "shares":          pos["shares"],
            "units":           pos.get("units", 1),
            "pnl_net":         pnl_net,
            "hold_days":       day_idx - pos["entry_day_idx"],
            "entry_reason":    pos["reason"],
            "exit_reason":     reason,
            "return":          sell_price / pos["entry_price"] - 1,
            "max_high":        pos["max_high"],
            "trailing_stop_final": pos["trailing_stop"],
            "earned_mode":     pos["earned_mode"],
            "pyramid_count":   pos.get("pyramid_count", 0),
            "atr_at_entry":    pos["atr_at_entry"],
        })

    def _new_buy(ticker: str, buy_price: float, buy_date, day_idx: int,
                 trend_val: float, entry_atr: float,
                 reason_prefix: str) -> bool:
        """Kauft eine neue Position (1 Slot, Phase 1: Tight Stop)."""
        nonlocal cash
        max_spend = min(slot_size, cash - fee)
        if max_spend < buy_price + fee or max_spend < 2 * fee:
            return False
        shares = int(math.floor((max_spend - fee) / buy_price))
        if shares < 1:
            return False
        cash -= shares * buy_price + fee
        init_stop = buy_price - atr_init * entry_atr
        portfolio[ticker] = {
            "shares":         shares,
            "units":          1,
            "entry_price":    buy_price,
            "entry_date":     buy_date,
            "entry_day_idx":  day_idx,
            "reason":         f"{reason_prefix} | Trend: {trend_val*100:+.1f}%",
            "max_high":       buy_price,
            "trailing_stop":  init_stop,
            "earned_mode":    False,        # startet in Phase 1 (TIGHT)
            "atr_at_entry":   entry_atr,    # für Earned-Trigger und Phase-1-Stop
            "pyramid_flag":   False,
            "pyramid_count":  0,
        }
        return True

    def _pyramid_buy(ticker: str, buy_price: float, buy_date, day_idx: int) -> bool:
        """Stockt eine bestehende Gewinner-Position auf + FREE-RIDE Stop."""
        nonlocal cash, n_pyramids
        if ticker not in portfolio:
            return False
        pos = portfolio[ticker]
        if pos.get("pyramid_count", 0) >= max_pyramids:
            return False
        max_spend = min(slot_size, cash - fee)
        if max_spend < buy_price + fee or max_spend < 2 * fee:
            return False
        add_shares = int(math.floor((max_spend - fee) / buy_price))
        if add_shares < 1:
            return False

        cost = add_shares * buy_price + fee
        cash -= cost

        # Gewichteten Durchschnitt-Einstandspreis aktualisieren
        old_value          = pos["shares"] * pos["entry_price"]
        pos["shares"]     += add_shares
        pos["entry_price"] = (old_value + add_shares * buy_price) / pos["shares"]
        pos["units"]       = pos.get("units", 1) + 1
        pos["pyramid_count"] += 1
        pos["pyramid_flag"]  = False

        # ---- FREE-RIDE STOP: Stop auf min. Average Entry anheben -----------
        old_stop    = pos["trailing_stop"]
        free_ride_stop = pos["entry_price"]   # neuer gewichteter Avg
        if free_ride_stop > old_stop:
            pos["trailing_stop"] = free_ride_stop
            msg = (f"  [FREE-RIDE] {ticker}"
                   f"  Stop: ${old_stop:.2f} -> ${free_ride_stop:.2f}"
                   f"  (Avg-Entry nach Pyramide #{pos['pyramid_count']})"
                   f"  | +{add_shares} Stk @ ${buy_price:.2f}"
                   f"  | Gesamt: {pos['shares']} Stk")
        else:
            msg = (f"  [PYRAMIDISIEREN] {ticker}"
                   f"  +{add_shares} Stk @ ${buy_price:.2f}"
                   f"  | Pyramid #{pos['pyramid_count']}/{max_pyramids}"
                   f"  | Gesamt: {pos['shares']} Stk"
                   f"  | Avg-Entry: ${pos['entry_price']:.2f}"
                   f"  | Stop unver.: ${pos['trailing_stop']:.2f}")

        pyramid_log.append(msg)
        if verbose:
            print(msg)
        n_pyramids += 1
        return True

    # ---------- Haupt-Loop -----------------------------------------------

    for i, today in enumerate(all_dates):

        # 1. EQUITY LOGGEN ------------------------------------------------
        pos_value = 0.0
        for t, pos in portfolio.items():
            try:
                cp = df_close.at[today, t] if t in df_close.columns else math.nan
                pos_value += pos["shares"] * (float(cp) if pd.notna(cp)
                                              else pos["entry_price"])
            except Exception:
                pos_value += pos["shares"] * pos["entry_price"]

        equity_log.append({
            "date":        today,
            "equity":      cash + pos_value,
            "cash":        cash,
            "n_pos":       len(portfolio),
            "total_units": _total_units(),
        })

        if i == 0:
            continue
        prev_date = all_dates[i - 1]

        # ================================================================ #
        # 2. EXITS: close[T-1] < trailing_stop → Verkauf bei open[T]      #
        # ================================================================ #
        for ticker in list(portfolio.keys()):
            if ticker not in df_close.columns:
                continue
            try:
                prev_close = df_close.at[prev_date, ticker]
                stop       = portfolio[ticker]["trailing_stop"]
            except KeyError:
                continue
            if pd.notna(prev_close) and prev_close < stop:
                sell_price = _safe_open(ticker)
                if not math.isnan(sell_price):
                    _sell(ticker, sell_price, today, i, "ATR Stop")

        # ================================================================ #
        # 3. PYRAMIDISIEREN (Priorität vor neuen Entries)                  #
        # ================================================================ #
        if enable_pyramid:
            for ticker in list(portfolio.keys()):
                if not portfolio[ticker].get("pyramid_flag", False):
                    continue
                if _total_units() >= max_pos:
                    portfolio[ticker]["pyramid_flag"] = False
                    continue
                buy_price = _safe_open(ticker)
                if not math.isnan(buy_price):
                    _pyramid_buy(ticker, buy_price, today, i)

        # ================================================================ #
        # 4. KANDIDATEN von gestern sammeln                                 #
        # ================================================================ #
        if prev_date not in df_entry.index:
            _bulk_update(portfolio, today, df_high, df_atr,
                         df_close, atr_init, atr_trail,
                         pyramid_threshold, enable_pyramid, max_pyramids,
                         max_pos, _total_units, mode_log, verbose)
            continue

        prev_entries = df_entry.loc[prev_date]
        prev_trends  = df_trend.loc[prev_date]
        prev_closes  = df_close.loc[prev_date] \
            if prev_date in df_close.index else pd.Series(dtype=float)

        market_ok = True
        if market_filter is not None:
            market_ok = bool(market_filter.get(prev_date, True))

        candidates = []
        if market_ok:
            for ticker in prev_entries.index:
                if not prev_entries[ticker] or ticker in portfolio:
                    continue
                op  = _safe_open(ticker)
                atr = _safe_atr(prev_date, ticker)
                if math.isnan(op) or op <= 0 or math.isnan(atr) or atr <= 0:
                    continue
                ts = float(prev_trends[ticker]) \
                    if ticker in prev_trends.index else 0.0
                candidates.append({"ticker": ticker, "open": op,
                                   "trend_str": ts, "atr": atr})
            candidates.sort(key=lambda x: x["trend_str"], reverse=True)

        # ================================================================ #
        # 5. ROTATION (streng: 2× Faktor, NUR profitable Positionen)      #
        # ================================================================ #
        if market_ok and _total_units() >= max_pos and candidates:
            holding_strengths: dict[str, float] = {}
            for t in portfolio:
                ts_val = float(prev_trends[t]) \
                    if t in prev_trends.index else 0.0
                holding_strengths[t] = ts_val

            weakest_t  = min(holding_strengths, key=holding_strengths.get)
            weakest_ts = holding_strengths[weakest_t]

            if weakest_ts > 0:  # Rotation NUR wenn Schwächste noch über SMA200
                best_cand = candidates[0]
                threshold = weakest_ts * rotation_factor

                weak_close = float(prev_closes.get(weakest_t, math.nan)) \
                    if weakest_t in prev_closes.index else math.nan
                in_profit  = (not math.isnan(weak_close) and
                              weak_close > portfolio[weakest_t]["entry_price"])

                if in_profit and best_cand["trend_str"] > threshold:
                    sell_price = _safe_open(weakest_t)
                    if not math.isnan(sell_price) and sell_price > 0:
                        _sell(weakest_t, sell_price, today, i,
                              f"Rotation -> {best_cand['ticker']} "
                              f"({best_cand['trend_str']*100:+.1f}%"
                              f" vs {weakest_ts*100:+.1f}%)")
                        ok = _new_buy(best_cand["ticker"], best_cand["open"],
                                      today, i, best_cand["trend_str"],
                                      best_cand["atr"], "Rotation: Breakout_50")
                        if ok:
                            n_rotations += 1
                            candidates = [c for c in candidates
                                          if c["ticker"] != best_cand["ticker"]]

        # ================================================================ #
        # 6. STANDARD ENTRIES                                               #
        # ================================================================ #
        available = max_pos - _total_units()
        for cand in candidates[:available]:
            _new_buy(cand["ticker"], cand["open"], today, i,
                     cand["trend_str"], cand["atr"], "Breakout_50")

        # ================================================================ #
        # 7. STOP UPDATE + PYRAMID FLAG SETZEN                             #
        # ================================================================ #
        _bulk_update(portfolio, today, df_high, df_atr,
                     df_close, atr_init, atr_trail,
                     pyramid_threshold, enable_pyramid, max_pyramids,
                     max_pos, _total_units, mode_log, verbose)

    # -----------------------------------------------------------------------
    equity_df = pd.DataFrame(equity_log).set_index("date")
    equity_df.attrs["n_rotations"] = n_rotations
    equity_df.attrs["n_pyramids"]  = n_pyramids
    equity_df.attrs["pyramid_log"] = pyramid_log
    equity_df.attrs["mode_log"]    = mode_log
    return equity_df, completed, _open_positions_df(portfolio, all_dates, df_close)


def _bulk_update(
    portfolio: dict, today,
    df_high: pd.DataFrame, df_atr: pd.DataFrame, df_close: pd.DataFrame,
    atr_init: float, atr_trail: float,
    pyramid_threshold: float, enable_pyramid: bool,
    max_pyramids: int, max_pos: int, total_units_fn,
    mode_log: list, verbose: bool,
) -> None:
    """Trailing-Stop nachziehen + Pyramid-Flag setzen für alle Positionen."""
    today_close_row = df_close.loc[today] \
        if today in df_close.index else pd.Series(dtype=float)

    for ticker, pos in portfolio.items():
        # --- ATR-Daten holen ---------------------------------------------
        try:
            h   = float(df_high.at[today, ticker]) \
                if ticker in df_high.columns else math.nan
            atr = float(df_atr.at[today, ticker]) \
                if ticker in df_atr.columns else math.nan
        except Exception:
            h = atr = math.nan

        # --- Trailing Stop (asymmetrisch, zwei Phasen) -------------------
        _compute_stop(pos, h, atr, atr_init, atr_trail,
                      ticker, today, verbose, mode_log)

        # --- Pyramid-Flag setzen -----------------------------------------
        if (enable_pyramid
                and not pos.get("pyramid_flag", False)
                and pos.get("pyramid_count", 0) < max_pyramids
                and total_units_fn() < max_pos):
            try:
                cp = float(today_close_row.get(ticker, math.nan)) \
                    if ticker in today_close_row.index else math.nan
                if not math.isnan(cp):
                    if (cp / pos["entry_price"] - 1) >= pyramid_threshold:
                        pos["pyramid_flag"] = True
            except Exception:
                pass


def _open_positions_df(portfolio: dict, all_dates: list,
                       df_close: pd.DataFrame) -> pd.DataFrame:
    if not portfolio or not all_dates:
        return pd.DataFrame()
    last = all_dates[-1]
    rows = []
    for ticker, pos in portfolio.items():
        try:
            lp = float(df_close.at[last, ticker]) \
                if ticker in df_close.columns else math.nan
        except Exception:
            lp = math.nan
        ret = lp / pos["entry_price"] - 1 if not math.isnan(lp) else math.nan
        rows.append({
            "ticker":          ticker,
            "entry_date":      pos["entry_date"],
            "entry_price":     pos["entry_price"],
            "last_close":      lp,
            "trailing_stop":   pos["trailing_stop"],
            "max_high":        pos["max_high"],
            "earned_mode":     pos["earned_mode"],
            "shares":          pos["shares"],
            "units":           pos.get("units", 1),
            "pyramid_count":   pos.get("pyramid_count", 0),
            "unrealized_ret":  ret,
        })
    return pd.DataFrame(rows)


# ==============================================================================
# 5. PERFORMANCE-KENNZAHLEN
# ==============================================================================

def compute_metrics(equity_df: pd.DataFrame, completed: list[dict],
                    initial_cap: float, fee: float) -> dict:
    eq        = equity_df["equity"].ffill().bfill()
    total_ret = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
    n_years   = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr      = ((eq.iloc[-1] / eq.iloc[0]) ** (1 / n_years) - 1) * 100 \
                if n_years > 0 else 0.0
    rolling_max = eq.cummax()
    max_dd    = ((eq - rolling_max) / rolling_max).min() * 100
    daily_ret = eq.pct_change().dropna()
    sharpe    = daily_ret.mean() / daily_ret.std() * np.sqrt(252) \
                if daily_ret.std() > 0 else 0.0

    n_rots = equity_df.attrs.get("n_rotations", 0)
    n_pyrs = equity_df.attrs.get("n_pyramids",  0)

    rets   = [t["return"] for t in completed] if completed else []
    wins   = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    n      = len(rets)
    hit    = len(wins) / n * 100    if n      else 0.0
    avg_w  = np.mean(wins) * 100    if wins   else 0.0
    avg_l  = np.mean(losses) * 100  if losses else 0.0
    payoff = abs(avg_w / avg_l)     if avg_l  else float("inf")
    exp    = hit / 100 * avg_w + (1 - hit / 100) * avg_l
    max_w  = max(rets) * 100        if rets   else 0.0
    total_fees = sum((1 + t.get("pyramid_count", 0) + 1) * fee
                     for t in completed)
    avg_hold = np.mean([t["hold_days"] for t in completed]) if completed else 0.0
    earned_n = sum(1 for t in completed if t.get("earned_mode", False))

    return {
        "start_date":    eq.index[0].date(),
        "end_date":      eq.index[-1].date(),
        "n_years":       round(n_years, 1),
        "start_cap":     initial_cap,
        "end_equity":    round(eq.iloc[-1], 2),
        "total_ret_%":   round(total_ret, 2),
        "cagr_%":        round(cagr, 2),
        "max_dd_%":      round(max_dd, 2),
        "sharpe":        round(sharpe, 2),
        "n_trades":      n,
        "n_rotations":   n_rots,
        "n_pyramids":    n_pyrs,
        "n_earned":      earned_n,
        "hit_%":         round(hit, 1),
        "avg_win_%":     round(avg_w, 2),
        "avg_loss_%":    round(avg_l, 2),
        "max_win_%":     round(max_w, 2),
        "payoff":        round(payoff, 2),
        "expect_%":      round(exp, 2),
        "total_fees":    round(total_fees, 2),
        "avg_hold_d":    round(avg_hold, 1),
    }


# ==============================================================================
# 6. CHART
# ==============================================================================

def plot_equity_curve(equity_df: pd.DataFrame, spy_close: pd.Series | None,
                      metrics: dict, initial_cap: float, output_path: Path,
                      market_filter_active: bool = False,
                      atr_init: float = DEFAULT_ATR_INIT,
                      atr_trail: float = DEFAULT_ATR_TRAIL) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9), facecolor="#0d1117",
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12},
    )
    eq    = equity_df["equity"].ffill().bfill()
    dates = eq.index

    ax1.set_facecolor("#161b22")
    ax1.plot(dates, eq, color="#58a6ff", linewidth=2.0,
             label=(f"Portfolio v6.3  "
                    f"({metrics['total_ret_%']:+.1f}%  |  "
                    f"CAGR {metrics['cagr_%']:+.1f}%  |  "
                    f"Sharpe {metrics['sharpe']:.2f})"))
    ax1.fill_between(dates, initial_cap, eq,
                     where=(eq >= initial_cap), color="#238636", alpha=0.20)
    ax1.fill_between(dates, initial_cap, eq,
                     where=(eq < initial_cap),  color="#da3633", alpha=0.20)
    ax1.axhline(initial_cap, color="#30363d", linewidth=1, linestyle="--",
                label=f"Startkapital ({initial_cap:,.0f})")

    if spy_close is not None:
        spy_sub = spy_close[spy_close.index >= dates[0]]
        if len(spy_sub) >= 2:
            spy_norm = spy_sub / spy_sub.iloc[0] * initial_cap
            spy_ret  = (spy_norm.iloc[-1] / spy_norm.iloc[0] - 1) * 100
            ax1.plot(spy_norm.index, spy_norm, color="#f0883e",
                     linewidth=1.6, linestyle="--",
                     label=f"S&P 500 (SPY)  ({spy_ret:+.1f}%)")

    subtitle = (f"Tight {atr_init}×ATR → Earned {atr_trail}×ATR  |  "
                f"Rot. ×{metrics['n_rotations']}  |  Pyr. ×{metrics['n_pyramids']}")
    if market_filter_active:
        subtitle += "  |  Marktfilter"
    ax1.set_title(f"Backtest v6.3  |  {subtitle}", color="#e6edf3", fontsize=12, pad=10)
    ax1.set_ylabel("Portfolio-Wert", color="#8b949e", fontsize=10)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax1.tick_params(colors="#8b949e", labelsize=8)
    ax1.spines[:].set_color("#30363d")
    ax1.legend(fontsize=9, facecolor="#161b22", edgecolor="#30363d",
               labelcolor="#e6edf3", loc="upper left")
    ax1.grid(axis="y", color="#21262d", linewidth=0.7)
    ax1.set_xlim(dates[0], dates[-1])

    ann = (f"MaxDD: {metrics['max_dd_%']:.1f}%   "
           f"Trades: {metrics['n_trades']}  "
           f"(Earned: {metrics['n_earned']}, Rot: {metrics['n_rotations']}, "
           f"Pyr: {metrics['n_pyramids']})   "
           f"Hit: {metrics['hit_%']:.0f}%   "
           f"Payoff: {metrics['payoff']:.2f}   "
           f"AvgHold: {metrics['avg_hold_d']:.0f}d")
    ax1.text(0.01, 0.03, ann, transform=ax1.transAxes,
             color="#8b949e", fontsize=8.5, va="bottom")

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

    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    fig.patch.set_facecolor("#0d1117")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"\n  Chart gespeichert: {output_path}")


# ==============================================================================
# 7. AUSGABE
# ==============================================================================

def print_metrics(metrics: dict, open_pos_df: pd.DataFrame,
                  mode_log: list, pyramid_log: list) -> None:
    print(f"""
  PERFORMANCE-ZUSAMMENFASSUNG (v6.3)
  {"=" * 66}
  Zeitraum:              {metrics['start_date']}  ->  {metrics['end_date']}  ({metrics['n_years']} Jahre)
  Startkapital:                  {metrics['start_cap']:>12,.2f}
  Endkapital:                    {metrics['end_equity']:>12,.2f}
  Gesamtrendite:                 {metrics['total_ret_%']:>+11.2f} %
  CAGR (p.a.):                   {metrics['cagr_%']:>+11.2f} %
  Max. Drawdown:                 {metrics['max_dd_%']:>+11.2f} %
  Sharpe Ratio:                  {metrics['sharpe']:>12.2f}
  {"=" * 66}
  Abgeschlossene Trades:         {metrics['n_trades']:>12,}
    davon Phase 2 "Earned":      {metrics['n_earned']:>12,}
    davon Rotations-Exits:       {metrics['n_rotations']:>12,}
    davon Pyramidisierungen:     {metrics['n_pyramids']:>12,}
  Avg. Haltedauer:               {metrics['avg_hold_d']:>12.1f} Tage
  Hit-Rate:                      {metrics['hit_%']:>11.1f} %
  Avg. Gewinn:                   {metrics['avg_win_%']:>+11.2f} %
  Avg. Verlust:                  {metrics['avg_loss_%']:>+11.2f} %
  Max. Einzelgewinn:             {metrics['max_win_%']:>+11.2f} %
  Payoff-Ratio:                  {metrics['payoff']:>12.2f}
  Erwartungswert/Trade:          {metrics['expect_%']:>+11.2f} %
  Summe Gebühren:                {metrics['total_fees']:>12,.2f}
  {"=" * 66}
""")
    if not open_pos_df.empty:
        print("  OFFENE POSITIONEN (Backtest-Ende):")
        print(f"  {'Ticker':<7}  {'Entry':<12}  {'Entry-$':>8}  {'Letzt-$':>8}  "
              f"{'Stop':>9}  {'MaxHigh':>9}  {'Phase':>8}  {'Ret-%':>7}  {'Pyr':>3}")
        print("  " + "-" * 90)
        for _, r in open_pos_df.iterrows():
            ret   = r["unrealized_ret"] * 100 if pd.notna(r["unrealized_ret"]) else float("nan")
            phase = "EARNED" if r["earned_mode"] else "TIGHT"
            print(f"  {r['ticker']:<7}  {str(r['entry_date'].date()):<12}  "
                  f"${r['entry_price']:>7.2f}  ${r['last_close']:>7.2f}  "
                  f"${r['trailing_stop']:>8.2f}  ${r['max_high']:>8.2f}  "
                  f"{phase:>8}  {ret:>+6.1f}%  {int(r['pyramid_count']):>3}")
        print()

    if mode_log:
        print(f"\n  PHASE-WECHSEL LOG ({len(mode_log)} Übergänge TIGHT -> EARNED):")
        print("  " + "=" * 80)
        for m in mode_log:
            print(m)
        print()

    if pyramid_log:
        print(f"\n  PYRAMIDISIERUNGS- & FREE-RIDE LOG ({len(pyramid_log)} Einträge):")
        print("  " + "=" * 80)
        for m in pyramid_log:
            print(m)
        print()


def print_trade_table(completed: list[dict]) -> None:
    if not completed:
        print("  Keine abgeschlossenen Trades.")
        return

    df = pd.DataFrame(completed)
    df["entry_date"]  = pd.to_datetime(df["entry_date"]).dt.strftime("%Y-%m-%d")
    df["exit_date"]   = pd.to_datetime(df["exit_date"]).dt.strftime("%Y-%m-%d")
    df["ret_%"]       = (df["return"] * 100).round(2)
    df["pnl_net"]     = df["pnl_net"].round(2)
    df["phase"]       = df["earned_mode"].map({True: "EARNED", False: "TIGHT"})

    display = df[["ticker","entry_date","entry_price","exit_date","exit_price",
                  "shares","units","pyramid_count","phase",
                  "ret_%","pnl_net","hold_days","exit_reason"]].copy()
    display.columns = ["Ticker","Kaufdatum","Kauf-$","Verkaufdatum","Verk-$",
                       "Stk","Units","Pyr","Phase","Ret-%","P&L-netto",
                       "Hold-d","Exit-Grund"]
    display = display.sort_values("Kaufdatum")

    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 1000)
    pd.set_option("display.float_format", "{:.2f}".format)

    print(f"\n  KONTOAUSZUG  ({len(display)} Trades):")
    print("  " + "=" * 210)
    print("  " + display.to_string(index=False))
    print("  " + "=" * 210)

    print(f"\n  TOP 10 GEWINNER:")
    top_w = display.nlargest(10, "Ret-%")[
        ["Ticker","Kaufdatum","Verkaufdatum","Ret-%","P&L-netto","Hold-d","Phase","Pyr"]]
    print("  " + top_w.to_string(index=False))

    print(f"\n  TOP 10 VERLIERER:")
    top_l = display.nsmallest(10, "Ret-%")[
        ["Ticker","Kaufdatum","Verkaufdatum","Ret-%","P&L-netto","Hold-d","Phase"]]
    print("  " + top_l.to_string(index=False))

    # Exit-Statistik
    print(f"\n  EXIT-GRUND STATISTIK:")
    print("  " + "-" * 52)
    from collections import Counter
    reasons = Counter()
    for t in completed:
        key = "Rotation" if t["exit_reason"].startswith("Rotation") else t["exit_reason"]
        reasons[key] += 1
    for reason, count in reasons.most_common():
        pct = count / len(completed) * 100
        print(f"  {reason:<38}  {count:>4}  ({pct:.1f}%)")

    # Earned vs. Tight Vergleich
    earned = [t for t in completed if t.get("earned_mode", False)]
    tight  = [t for t in completed if not t.get("earned_mode", False)]
    print(f"\n  PHASE-VERGLEICH (Earned vs. Tight):")
    print("  " + "-" * 52)
    for label, grp in [("EARNED (Phase 2)", earned), ("TIGHT  (Phase 1)", tight)]:
        if grp:
            rets = [t["return"] for t in grp]
            wins = [r for r in rets if r > 0]
            print(f"  {label}: N={len(grp):>4}  Hit={len(wins)/len(rets)*100:.0f}%  "
                  f"AvgRet={np.mean(rets)*100:>+6.2f}%  "
                  f"MaxWin={max(rets)*100:>+6.1f}%  "
                  f"MaxLoss={min(rets)*100:>+6.1f}%  "
                  f"AvgHold={np.mean([t['hold_days'] for t in grp]):.0f}d")
    print()


# ==============================================================================
# 8. MAIN
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest v6.3 | Asymmetrischer ATR Stop + Rotation + Pyramidisieren")
    parser.add_argument("--years",             type=float, default=7.0)
    parser.add_argument("--capital",           type=float, default=INITIAL_CAPITAL,
                        help=f"Startkapital (HARDCODE: {INITIAL_CAPITAL:.0f})")
    parser.add_argument("--fee",               type=float, default=ORDER_FEE,
                        help=f"Ordergebühr pro Trade (HARDCODE: {ORDER_FEE:.0f})")
    parser.add_argument("--max-pos",           type=int,   default=DEFAULT_MAX_POS)
    parser.add_argument("--atr-init",          type=float, default=DEFAULT_ATR_INIT,
                        help=f"Multiplikator Phase 1 (Tight Stop, Standard: {DEFAULT_ATR_INIT})")
    parser.add_argument("--atr-trail",         type=float, default=DEFAULT_ATR_TRAIL,
                        help=f"Multiplikator Phase 2 (Earned Trail, Standard: {DEFAULT_ATR_TRAIL})")
    parser.add_argument("--rotation-factor",   type=float, default=DEFAULT_ROTATION_FACTOR)
    parser.add_argument("--pyramid-threshold", type=float, default=DEFAULT_PYRAMID_THRESHOLD)
    parser.add_argument("--max-pyramids",      type=int,   default=DEFAULT_MAX_PYRAMIDS)
    parser.add_argument("--no-pyramid",        action="store_true")
    parser.add_argument("--market-filter",     action="store_true")
    parser.add_argument("--no-chart",          action="store_true")
    parser.add_argument("--save-csv",          action="store_true")
    parser.add_argument("--quiet",             action="store_true",
                        help="Phase-Wechsel & Pyramid-Logs nicht inline ausgeben")
    args = parser.parse_args()

    print("=" * 70)
    print("  BACKTEST v6.3  |  Asymm. ATR Stop + Rotation (2×) + Free-Ride")
    print("=" * 70)
    print(f"""
  ╔══════════════════════════════════════════════════════╗
  ║  HARDCODED FINANZIELLE REALITÄTEN                    ║
  ║  Startkapital:   {args.capital:>10,.2f}  EUR               ║
  ║  Order-Fee:      {args.fee:>10.2f}  EUR (Round-Trip: {args.fee*2:.0f} EUR) ║
  ╚══════════════════════════════════════════════════════╝

  STOP-LOGIK:
    Phase 1 (TIGHT):   Stop  = Entry - {args.atr_init:.1f} × ATR14  (Fake-Breakout-Schutz)
    Trigger:           MaxHigh > Entry + {args.atr_init:.1f} × ATR14_at_Entry
    Phase 2 (EARNED):  Stop  = MaxHigh - {args.atr_trail:.1f} × ATR14  (Trend reiten)

  ROTATION:            {args.rotation_factor:.1f}× Faktor | NUR profitable Positionen
  PYRAMIDISIEREN:      {'DEAKTIVIERT' if args.no_pyramid else f'ab +{args.pyramid_threshold:.0%} | max {args.max_pyramids}× | FREE-RIDE Stop'}
  MARKTFILTER:         {'aktiv (SPY > SMA200)' if args.market_filter else 'inaktiv'}
""")

    # 1-2. Ticker & Daten
    tickers = _load_tickers()
    print(f"[1/5] Universum: {len(tickers)} Ticker")
    print(f"\n[2/5] OHLCV-Daten ({args.years:.0f} Jahre)...")
    data = load_universe(tickers, args.years)
    if not data:
        print("FEHLER: Keine Daten.")
        return

    # 3. Panel
    print(f"\n[3/5] Pivot-Tabellen aufbauen...")
    pivots = build_pivot_panel(data)
    dates  = pivots["open"].index
    print(f"  Zeitraum: {dates[0].date()} -> {dates[-1].date()}  ({len(dates):,} Handelstage)")

    # 4. SPY
    print(f"\n[4/5] SPY Benchmark...")
    spy_close = _download_spy(dates[0], dates[-1])
    market_filter_series = None
    if spy_close is not None:
        print(f"  SPY: {len(spy_close)} Tage")
        if args.market_filter:
            spy_sma200 = spy_close.rolling(200).mean()
            market_filter_series = (spy_close > spy_sma200).reindex(dates, method="ffill")
    else:
        print("  SPY nicht verfügbar.")

    # 5. Simulation
    print(f"\n[5/5] Simulation startet...")
    print("  " + "─" * 70)

    equity_df, completed, open_pos_df = run_backtest(
        pivots             = pivots,
        initial_cap        = args.capital,
        fee                = args.fee,
        max_pos            = args.max_pos,
        atr_init           = args.atr_init,
        atr_trail          = args.atr_trail,
        rotation_factor    = args.rotation_factor,
        pyramid_threshold  = args.pyramid_threshold,
        max_pyramids       = args.max_pyramids,
        enable_pyramid     = not args.no_pyramid,
        market_filter      = market_filter_series,
        verbose            = not args.quiet,
    )
    print("  " + "─" * 70)

    n_rot       = equity_df.attrs.get("n_rotations", 0)
    n_pyr       = equity_df.attrs.get("n_pyramids",  0)
    pyramid_log = equity_df.attrs.get("pyramid_log", [])
    mode_log    = equity_df.attrs.get("mode_log",    [])

    print(f"\n  Abgeschlossene Trades:       {len(completed):,}")
    print(f"  Earned-Mode Übergänge:       {len(mode_log):,}")
    print(f"  Rotations-Exits:             {n_rot:,}")
    print(f"  Pyramidisierungen:           {n_pyr:,}")
    print(f"  Offene Positionen am Ende:   {len(open_pos_df)}")

    metrics = compute_metrics(equity_df, completed, args.capital, args.fee)
    print_metrics(metrics, open_pos_df, mode_log, pyramid_log)

    if not args.no_chart:
        plot_equity_curve(equity_df, spy_close, metrics, args.capital,
                          _REPO_ROOT / "backtest_v6_equity.png",
                          args.market_filter, args.atr_init, args.atr_trail)

    print_trade_table(completed)

    if args.save_csv and completed:
        csv_path = _REPO_ROOT / "backtest_v6_trades.csv"
        pd.DataFrame(completed).to_csv(csv_path, index=False)
        print(f"\n  Trades gespeichert: {csv_path}")

    print("\n  SCHNELL-FAZIT:")
    print("  " + "─" * 66)
    print(f"  Startkapital:    {args.capital:>10,.0f} EUR  |  Fee: {args.fee:.0f} EUR/Order")
    print(f"  Endkapital:      {metrics['end_equity']:>10,.2f} EUR")
    print(f"  Gesamtrendite:   {metrics['total_ret_%']:>+10.2f}%")
    print(f"  CAGR:            {metrics['cagr_%']:>+10.2f}%  |  Sharpe: {metrics['sharpe']:.2f}")
    print(f"  Max-Drawdown:    {metrics['max_dd_%']:>+10.1f}%")
    print(f"  Trades:          {metrics['n_trades']:>10}  "
          f"(Earned: {metrics['n_earned']}, Rot: {n_rot}, Pyr: {n_pyr})")
    print(f"  Avg. Haltedauer: {metrics['avg_hold_d']:>10.1f} Tage")
    print(f"  Payoff-Ratio:    {metrics['payoff']:>10.2f}  |  "
          f"Expectancy: {metrics['expect_%']:>+.2f}%")
    print(f"  Summe Gebühren:  {metrics['total_fees']:>10,.2f} EUR")
    print()


if __name__ == "__main__":
    main()
