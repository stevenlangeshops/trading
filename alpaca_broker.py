"""
alpaca_broker.py
══════════════════════════════════════════════════════════════════════════════
Execution-Layer für Alpaca Paper/Live Trading.

Nutzt das offizielle ``alpaca-py`` SDK (kein Konflikt mit yfinance/websockets).

Lädt die Ziel-Ticker aus live_inference.py und rebalanciert das Portfolio:
  A) Verkauf: Positionen die nicht in der Ziel-Liste sind → Market-Sell
  B) Sizing:  Equal-Weight – Equity / Anzahl_Ticker pro Position
  C) Kauf/Anpassung: Delta zwischen Ziel-Stückzahl und aktueller Stückzahl
     Gap-Filter: Neukäufe werden blockiert wenn |Gap $| > 1.5 × ATR-14
  D) Stop-Loss: GTC Stop-Market-Orders @ Einstieg × (1 – stop_loss_pct)

Umgebungsvariablen (zwingend):
    APCA_API_KEY_ID     – Alpaca API Key ID
    APCA_API_SECRET_KEY – Alpaca API Secret Key

Optional:
    APCA_PAPER          – ``"true"`` (Default) → Paper-Trading-Endpoint
                          ``"false"``           → Live-Trading-Endpoint

Verwendung (standalone):
    from alpaca_broker import get_portfolio_state, execute_target_allocation
    state  = get_portfolio_state()
    result = execute_target_allocation(["AAPL", "MSFT", "NVDA"], dry_run=True)
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Ticker-Symbole die bei Alpaca anders heißen als bei yfinance/Yahoo.
# yfinance-Symbol → Alpaca-Symbol
_SYMBOL_MAP: dict[str, str] = {
    "BF-B": "BF.B",
    "BF-A": "BF.A",
}

def _to_alpaca(symbol: str) -> str:
    """Konvertiert ein yfinance-Symbol in das Alpaca-Format."""
    return _SYMBOL_MAP.get(symbol, symbol)

def _from_alpaca(symbol: str) -> str:
    """Konvertiert ein Alpaca-Symbol zurück in das yfinance-Format."""
    reverse = {v: k for k, v in _SYMBOL_MAP.items()}
    return reverse.get(symbol, symbol)


# ══════════════════════════════════════════════════════════════════════════════
# Interne Helfer
# ══════════════════════════════════════════════════════════════════════════════

def _get_clients() -> tuple:
    """Erstellt einen authentifizierten Alpaca Trading- und Data-Client.

    Liest Credentials aus den Umgebungsvariablen ``APCA_API_KEY_ID`` und
    ``APCA_API_SECRET_KEY``.

    Returns:
        Tupel ``(TradingClient, StockHistoricalDataClient)``.

    Raises:
        EnvironmentError: Wenn die Credentials-Variablen fehlen.
        ImportError:      Wenn ``alpaca-py`` nicht installiert ist.
    """
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.data.historical import StockHistoricalDataClient
    except ImportError:
        raise ImportError(
            "alpaca-py nicht installiert.\n"
            "Bitte ausfuehren: pip install alpaca-py"
        )

    key_id  = os.environ.get("APCA_API_KEY_ID",     "").strip()
    secret  = os.environ.get("APCA_API_SECRET_KEY", "").strip()
    paper   = os.environ.get("APCA_PAPER",          "true").strip().lower() != "false"

    if not key_id or not secret:
        raise EnvironmentError(
            "Umgebungsvariablen fehlen:\n"
            "  APCA_API_KEY_ID     – dein Alpaca Key ID\n"
            "  APCA_API_SECRET_KEY – dein Alpaca Secret Key\n"
            "Setze sie z.B. per: $env:APCA_API_KEY_ID='PK...'"
        )

    trading_client = TradingClient(
        api_key    = key_id,
        secret_key = secret,
        paper      = paper,
    )
    data_client = StockHistoricalDataClient(
        api_key    = key_id,
        secret_key = secret,
    )
    return trading_client, data_client


def _is_market_open(trading_client) -> bool:
    """Gibt True zurück wenn der US-Aktienmarkt aktuell geöffnet ist."""
    try:
        return trading_client.get_clock().is_open
    except Exception as exc:
        logger.warning(f"Markt-Status-Check fehlgeschlagen: {exc}")
        return False


def _compute_gap_filter_data(
    data_client,
    symbols:     list[str],
    atr_period:  int = 14,
    lookback_days: int = 35,
) -> dict[str, dict]:
    """Berechnet gestrigen Schlusskurs und ATR-14 für den Gap-Filter.

    Formel True Range (TR):
        TR = max(High − Low, |High − Prev_Close|, |Low − Prev_Close|)
    Formel ATR-14:
        SMA der letzten 14 TR-Werte.

    Args:
        data_client:   Alpaca StockHistoricalDataClient.
        symbols:       Alpaca-Symbole der Ziel-Ticker.
        atr_period:    Anzahl Tage für ATR-SMA (Standard: 14).
        lookback_days: Kalender-Tage Rückblick für den Bars-Download.

    Returns:
        Dict ``{symbol → {"prev_close": float|None, "atr14": float|None}}``.
        Werte sind ``None`` wenn Datenmangel vorliegt (→ Fallback-Filter greift).
    """
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from datetime import date as _date, timedelta as _timedelta

    start   = (_date.today() - _timedelta(days=lookback_days)).isoformat()
    result: dict[str, dict] = {}

    try:
        bars_req  = StockBarsRequest(
            symbol_or_symbols = symbols,
            timeframe         = TimeFrame.Day,
            start             = start,
        )
        bars_resp = data_client.get_stock_bars(bars_req)

        for sym, bar_list in bars_resp.items():
            if not bar_list or len(bar_list) < 2:
                result[sym] = {"prev_close": None, "atr14": None}
                continue

            prev_close = float(bar_list[-2].close)
            atr14: float | None = None

            # Mindestens atr_period + 1 Bars nötig (1 extra für prev_close der TR)
            if len(bar_list) >= atr_period + 1:
                bars_window = bar_list[-(atr_period + 1):]
                tr_values   = []
                for i in range(1, len(bars_window)):
                    high   = float(bars_window[i].high)
                    low    = float(bars_window[i].low)
                    prev_c = float(bars_window[i - 1].close)
                    tr     = max(high - low, abs(high - prev_c), abs(low - prev_c))
                    tr_values.append(tr)
                if tr_values:
                    atr14 = sum(tr_values) / len(tr_values)

            result[sym] = {"prev_close": prev_close, "atr14": atr14}

    except Exception as exc:
        logger.warning(f"Gap-Filter-Daten konnten nicht geladen werden: {exc}")

    # Fehlende Symbole mit None auffüllen
    for sym in symbols:
        result.setdefault(sym, {"prev_close": None, "atr14": None})

    return result


def _get_latest_prices(
    data_client,
    symbols: list[str],
) -> dict[str, float]:
    """Holt die aktuellsten Handelspreise für eine Liste von Symbolen.

    Args:
        data_client: Alpaca StockHistoricalDataClient.
        symbols:     Liste von Ticker-Symbolen.

    Returns:
        Dict ``{ticker → Preis}``.  Fehlende Ticker werden ausgelassen.
    """
    from alpaca.data.requests import StockLatestTradeRequest

    prices: dict[str, float] = {}
    try:
        req    = StockLatestTradeRequest(symbol_or_symbols=symbols)
        trades = data_client.get_stock_latest_trade(req)
        for sym, trade in trades.items():
            if trade and trade.price:
                prices[sym] = float(trade.price)
        return prices
    except Exception as exc:
        logger.warning(f"get_stock_latest_trade fehlgeschlagen: {exc}")

    # Fallback: einzeln abfragen
    for sym in symbols:
        try:
            req   = StockLatestTradeRequest(symbol_or_symbols=sym)
            trade = data_client.get_stock_latest_trade(req)
            prices[sym] = float(trade[sym].price)
        except Exception as exc2:
            logger.warning(f"  [WARN] Kein Preis fuer {sym}: {exc2}")

    return prices


def _submit_order(
    trading_client,
    symbol:  str,
    qty:     int,
    side:    str,
    dry_run: bool,
) -> Optional[object]:
    """Sendet eine Market-Order (Buy oder Sell).

    Args:
        trading_client: Alpaca TradingClient.
        symbol:         Ticker-Symbol.
        qty:            Stückzahl (positiv, ≥ 1).
        side:           ``"buy"`` oder ``"sell"``.
        dry_run:        Bei True nur loggen, nicht senden.

    Returns:
        Alpaca Order-Objekt oder ``None`` bei Fehler / Dry-Run.
    """
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    if qty < 1:
        logger.debug(f"  [SKIP] {symbol}: qty={qty} < 1 – keine Order")
        return None

    tag      = "[DRY-RUN] " if dry_run else ""
    alpaca_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
    logger.info(f"  {tag}ORDER  {side.upper():4s}  {symbol:<8}  qty={qty}")

    if dry_run:
        return None

    try:
        req   = MarketOrderRequest(
            symbol        = symbol,
            qty           = qty,
            side          = alpaca_side,
            time_in_force = TimeInForce.DAY,
        )
        order = trading_client.submit_order(req)
        logger.info(f"  Order-ID: {order.id}  Status: {order.status}")
        return order
    except Exception as exc:
        logger.error(f"  [FEHLER] Order {side} {symbol} qty={qty}: {exc}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Stop-Loss Helper
# ══════════════════════════════════════════════════════════════════════════════

def set_stop_losses(
    trading_client,
    stop_pct:       float          = 0.20,
    dry_run:        bool           = False,
    symbols_filter: set[str] | None = None,
) -> list[dict]:
    """Setzt/aktualisiert GTC Stop-Loss Orders für gehandelte Positionen.

    Ablauf:
      1. Offene Stop-Sell-Orders der betroffenen Positionen stornieren.
      2. Pro Position eine neue Stop-Market-Order platzieren:
         Stop-Preis = ``avg_entry_price × (1 − stop_pct)``

    Args:
        trading_client: Alpaca TradingClient.
        stop_pct:       Verlust-Schwelle als Dezimalzahl (0.20 = 20 %).
        dry_run:        Bei True nur loggen, keine echten Orders.
        symbols_filter: Wenn angegeben, werden nur Stops für diese Symbole
                        gesetzt. Nicht enthaltene Positionen bleiben unberührt.
                        ``None`` verarbeitet alle offenen Positionen.

    Returns:
        Liste von Dicts mit ``symbol``, ``stop_price``, ``qty``, ``order_id``.
    """
    from alpaca.trading.requests import GetOrdersRequest, StopOrderRequest
    from alpaca.trading.enums   import OrderSide, TimeInForce, QueryOrderStatus

    tag       = "[DRY-RUN] " if dry_run else ""
    positions = trading_client.get_all_positions()
    results   = []

    if not positions:
        print("  Keine offenen Positionen – keine Stop-Loss Orders gesetzt.")
        return results

    # Auf gefilterte Symbole einschränken wenn angegeben
    if symbols_filter is not None:
        positions = [p for p in positions if p.symbol in symbols_filter]
        if not positions:
            print("  Keine gehandelten Positionen – Stop-Loss Orders unveraendert.")
            return results

    # ── Alle offenen Stop-Sell-Orders laden (einmalig, dann filtern) ──────────
    try:
        open_orders = trading_client.get_orders(
            filter=GetOrdersRequest(status=QueryOrderStatus.OPEN)
        )
        existing_stops = {
            o.symbol: o
            for o in open_orders
            if str(o.order_type).lower() == "stop"
            and str(o.side).lower() == "sell"
        }
    except Exception as exc:
        logger.warning(f"  Offene Orders konnten nicht geladen werden: {exc}")
        existing_stops = {}

    for pos in positions:
        symbol    = pos.symbol
        qty       = int(float(pos.qty))
        avg_entry = float(pos.avg_entry_price)
        curr_price = float(pos.current_price)
        stop_price = round(avg_entry * (1.0 - stop_pct), 2)

        # Sicherheitscheck: Stop muss unter aktuellem Kurs liegen
        if stop_price >= curr_price:
            print(
                f"  [SKIP] {symbol}: Stop ${stop_price:.2f} >= Kurs ${curr_price:.2f} "
                f"– Position bereits im Hard-Stop-Bereich"
            )
            continue

        # Alte Stop-Order stornieren
        if symbol in existing_stops:
            old = existing_stops[symbol]
            old_stop = float(old.stop_price) if old.stop_price else 0
            print(f"  Storniere alte Stop-Order {symbol} @ ${old_stop:.2f}")
            if not dry_run:
                try:
                    trading_client.cancel_order_by_id(str(old.id))
                except Exception as exc:
                    logger.warning(f"  Cancel Stop {symbol}: {exc}")

        print(
            f"  {tag}{symbol:<8}  Entry=${avg_entry:.2f}  "
            f"Stop=${stop_price:.2f}  (-{stop_pct*100:.0f}%)  Qty={qty}"
        )

        if not dry_run:
            try:
                req = StopOrderRequest(
                    symbol        = symbol,
                    qty           = qty,
                    side          = OrderSide.SELL,
                    stop_price    = stop_price,
                    time_in_force = TimeInForce.GTC,
                )
                order = trading_client.submit_order(req)
                results.append({
                    "symbol":     symbol,
                    "stop_price": stop_price,
                    "qty":        qty,
                    "order_id":   str(order.id),
                })
                logger.info(f"  Stop-Loss gesetzt: {symbol} @ ${stop_price:.2f}")
            except Exception as exc:
                logger.error(f"  [FEHLER] Stop-Loss {symbol}: {exc}")
                print(f"  [FEHLER] {symbol}: {exc}")
        else:
            results.append({
                "symbol":     symbol,
                "stop_price": stop_price,
                "qty":        qty,
                "order_id":   "DRY-RUN",
            })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Öffentliche API
# ══════════════════════════════════════════════════════════════════════════════

def get_portfolio_state() -> dict:
    """Lädt den aktuellen Portfolio-Zustand von Alpaca.

    Returns:
        Dict mit:
          - ``equity``     (float): Gesamtwert des Portfolios in USD.
          - ``cash``       (float): Verfügbares Barvermögen in USD.
          - ``positions``  (dict):  ``{ticker → Stückzahl}`` aller offenen Positionen.
          - ``market_open`` (bool): Ob der Markt gerade geöffnet ist.

    Raises:
        EnvironmentError: Wenn Alpaca-Credentials fehlen.
    """
    trading_client, _ = _get_clients()
    account           = trading_client.get_account()
    raw_pos           = trading_client.get_all_positions()

    positions = {p.symbol: float(p.qty) for p in raw_pos}

    state = {
        "equity":      float(account.equity),
        "cash":        float(account.cash),
        "positions":   positions,
        "market_open": _is_market_open(trading_client),
    }

    logger.info(
        f"Portfolio: Equity=${state['equity']:,.2f}  "
        f"Cash=${state['cash']:,.2f}  "
        f"Positionen={len(positions)}  "
        f"Markt={'offen' if state['market_open'] else 'geschlossen'}"
    )
    return state


def execute_target_allocation(
    target_tickers:      list[str],
    dry_run:             bool  = False,
    sell_delay_s:        float = 1.0,
    stop_loss_pct:       float = 0.20,
    gap_filter_atr_mult: float = 1.5,
    gap_filter_pct:      float = 0.05,
) -> dict:
    """Rebalanciert das Portfolio nach der "Composition Change"-Logik.

    **Signal-Vergleich (Composition Check):**
    Vergleicht die aktuell gehaltenen Positionen als Set mit ``target_tickers``.

    **Szenario A – Keine Änderung (identische Sets):**
      Überspringt alle Trades. Gewichte driften frei. Kein Turnover-Drag.
      Stop-Loss-Orders werden dennoch aktualisiert.

    **Szenario B – Kompositions-Änderung:**

      **A – Verkauf:** Alle aktuell gehaltenen Positionen die NICHT in
        ``target_tickers`` stehen werden vollständig glattgestellt.

      **B – Sizing:** Equal-Weight-Allokation.
        ``ziel_wert_pro_ticker = equity / len(target_tickers)``

      **C – Kauf / Anpassung:** Für jeden Ziel-Ticker wird die Differenz
        zwischen Ziel-Stückzahl (``floor(ziel_wert / preis)``) und aktueller
        Stückzahl ermittelt und per Market-Order ausgeglichen.
        Neukäufe werden durch einen ATR-14-Gap-Filter geprüft:
        ``Gap = Kurs − Vortags-Schluss``; wenn ``|Gap| > gap_filter_atr_mult × ATR_14``
        wird der Kauf übersprungen. Fallback auf prozentualen Filter falls ATR
        nicht berechenbar.

    **D – Stop-Loss (immer):** GTC Stop-Market-Orders für alle gehaltenen
      Positionen. Stop-Preis = ``avg_entry_price × (1 − stop_loss_pct)``

    Args:
        target_tickers:      Geordnete Liste der Ziel-Ticker (Modell-Ranking).
        dry_run:             Bei True werden Orders nur geloggt, nicht gesendet.
        sell_delay_s:        Wartezeit in Sekunden nach Verkäufen, damit das
                             Portfolio-Update propagiert werden kann.
        stop_loss_pct:       Hard-Stop als Dezimalzahl (Standard: 0.20 = 20 %).
        gap_filter_atr_mult: Multiplikator für ATR-14-Limit (Standard: 1.5).
                             Neukäufe werden blockiert wenn |Gap $| > mult × ATR_14.
        gap_filter_pct:      Fallback-Filter (Standard: 0.05 = 5 %) falls ATR
                             für einen Ticker nicht berechenbar ist.

    Returns:
        Dict mit ``sells``, ``buys``, ``skipped``, ``errors``, ``equity_used``,
        ``stop_losses``.

    Raises:
        EnvironmentError: Wenn Alpaca-Credentials fehlen.
        RuntimeError:     Wenn der Markt geschlossen ist (und kein dry_run).
        ValueError:       Wenn ``target_tickers`` leer ist.
    """
    if not target_tickers:
        raise ValueError("target_tickers ist leer – keine Allokation möglich.")

    trading_client, data_client = _get_clients()
    tag    = "[DRY-RUN] " if dry_run else ""

    # Symbole in Alpaca-Format konvertieren (z.B. BF-B → BF.B)
    alpaca_tickers = [_to_alpaca(t) for t in target_tickers]

    result: dict = {
        "sells":       [],
        "buys":        [],
        "skipped":     [],
        "errors":      [],
        "equity_used": 0.0,
        "stop_losses": [],
    }

    print(f"\n{'='*60}")
    print(f"  {tag}ALPACA EXECUTION")
    print(f"  Ziel-Portfolio: {target_tickers}")
    print(f"{'='*60}")

    # ── Markt-Status prüfen ───────────────────────────────────────────────────
    market_open = _is_market_open(trading_client)
    if not market_open and not dry_run:
        raise RuntimeError(
            "US-Aktienmarkt ist geschlossen. Market-Orders werden abgelehnt.\n"
            "Verwende --dry-run fuer eine Simulation, oder starte nach\n"
            "Markteroefffnung (15:30 Uhr CEST / 09:30 ET)."
        )
    if not market_open:
        logger.warning("Markt ist geschlossen – Dry-Run wird trotzdem ausgefuehrt.")

    # ── Portfolio-Zustand laden ───────────────────────────────────────────────
    account     = trading_client.get_account()
    equity      = float(account.equity)
    raw_pos     = trading_client.get_all_positions()
    current_pos = {p.symbol: float(p.qty) for p in raw_pos}
    target_set  = set(target_tickers)

    alpaca_target_set = set(alpaca_tickers)

    print(f"\n  Equity:              ${equity:>12,.2f}")
    print(f"  Aktuelle Positionen: {list(current_pos.keys()) or '(keine)'}")
    print(f"  Ziel-Positionen:     {alpaca_tickers}")

    # ── Signal-Vergleich: Composition Change Check ────────────────────────────
    # Vergleich als Set: Reihenfolge irrelevant, nur Ticker-Menge zählt.
    composition_changed = set(current_pos.keys()) != alpaca_target_set
    result["equity_used"] = equity

    if not composition_changed:
        # ════════════════════════════════════════════════════════════════════════
        # Szenario A: Keine Änderung – Gewichte driften lassen
        # ════════════════════════════════════════════════════════════════════════
        print(f"\n{'─'*60}")
        print(
            f"  Top {len(target_tickers)} unveraendert. "
            f"Lasse Gewichte driften. Keine Trades ausgefuehrt."
        )
        print(f"{'─'*60}")
        result["skipped"] = list(alpaca_target_set)

    else:
        # ════════════════════════════════════════════════════════════════════════
        # Szenario B: Kompositions-Änderung – vollständiger Equal-Weight Reset
        # ════════════════════════════════════════════════════════════════════════
        added   = alpaca_target_set - set(current_pos.keys())
        removed = set(current_pos.keys()) - alpaca_target_set
        print(f"\n  Portfolio-Zusammensetzung geaendert. Fuehre vollstaendigen Equal-Weight Reset durch.")
        if added:
            print(f"  Neu hinzugekommen: {sorted(added)}")
        if removed:
            print(f"  Herausgefallen:    {sorted(removed)}")

        # ── A: Verkauf – Positionen die nicht im Ziel sind ───────────────────
        print(f"\n--- A) Verkauf (nicht im Ziel-Portfolio) ---")
        to_sell = [sym for sym in current_pos if sym not in alpaca_target_set]

        if not to_sell:
            print("  Keine Positionen zu verkaufen.")
        else:
            for sym in to_sell:
                qty = int(current_pos[sym])
                print(f"  {tag}Verkaufe alle {qty} Stueck {sym}")
                if not dry_run:
                    try:
                        trading_client.close_position(sym)
                        result["sells"].append(sym)
                        logger.info(f"  Position {sym} geschlossen.")
                    except Exception as exc:
                        msg = f"close_position({sym}) fehlgeschlagen: {exc}"
                        logger.error(f"  [FEHLER] {msg}")
                        result["errors"].append(msg)
                else:
                    result["sells"].append(sym)

            if not dry_run:
                print(f"  Warte {sell_delay_s:.0f}s auf Portfolio-Update ...")
                time.sleep(sell_delay_s)
                # Positionen nach Verkäufen neu laden
                raw_pos     = trading_client.get_all_positions()
                current_pos = {p.symbol: float(p.qty) for p in raw_pos}

        # ── B: Sizing – Equal-Weight ──────────────────────────────────────────
        print(f"\n--- B) Sizing (Equal-Weight) ---")
        n_targets    = len(target_tickers)
        target_value = equity / n_targets

        print(f"  Equity:              ${equity:>12,.2f}")
        print(f"  Anzahl Ziel-Ticker:  {n_targets}")
        print(f"  Ziel-Wert pro Pos.:  ${target_value:>12,.2f}")

        # ── C: Kauf / Rebalancing mit ATR-14-Gap-Filter ───────────────────────
        print(f"\n--- C) Kauf / Rebalancing (Gap-Filter: {gap_filter_atr_mult}× ATR-14) ---")
        gap_data      = _compute_gap_filter_data(data_client, alpaca_tickers)
        prices        = _get_latest_prices(data_client, alpaca_tickers)
        traded_symbols: set[str] = set()   # Symbole mit tatsächlich gesendeter Order

        for sym in alpaca_tickers:
            price = prices.get(sym)
            if price is None or price <= 0:
                msg = f"Kein Preis fuer {sym} – uebersprungen"
                logger.warning(f"  [WARN] {msg}")
                result["skipped"].append(sym)
                continue

            target_qty  = int(math.floor(target_value / price))
            current_qty = int(current_pos.get(sym, 0))
            delta       = target_qty - current_qty

            # ── ATR-basierter Gap-Filter (nur bei Neukäufen) ─────────────────
            gap_str = ""
            if delta > 0 and current_qty == 0:
                gd         = gap_data.get(sym, {})
                prev_close = gd.get("prev_close")
                atr14      = gd.get("atr14")

                if prev_close is not None:
                    gap_abs = price - prev_close

                    if atr14 is not None and atr14 > 0:
                        max_gap = gap_filter_atr_mult * atr14
                        gap_str = f"  Gap=${gap_abs:+.2f}  Limit={gap_filter_atr_mult}×ATR=${max_gap:.2f}"

                        if abs(gap_abs) > max_gap:
                            print(
                                f"  [GAP-SKIP] Kauf von {sym} uebersprungen: "
                                f"Gap (${abs(gap_abs):.2f}) ueberschreitet Limit "
                                f"({gap_filter_atr_mult}x ATR = ${max_gap:.2f})"
                            )
                            result["skipped"].append(
                                f"{sym} (Gap ${gap_abs:+.2f} > {gap_filter_atr_mult}×ATR ${max_gap:.2f})"
                            )
                            continue
                        else:
                            print(
                                f"  [GAP-PASS] {sym}: Gap (${abs(gap_abs):.2f}) "
                                f"ist im Limit ({gap_filter_atr_mult}x ATR = ${max_gap:.2f})"
                            )

                    else:
                        # Fallback: prozentualer Filter (ATR nicht verfügbar)
                        gap_pct = gap_abs / prev_close
                        gap_str = f"  Gap={gap_pct:+.1%}  [Fallback >{gap_filter_pct:.0%}]"

                        if abs(gap_pct) > gap_filter_pct:
                            print(
                                f"  [GAP-SKIP] Kauf von {sym} uebersprungen (ATR n/v – Fallback): "
                                f"Gap ({gap_pct:+.1%}) > {gap_filter_pct:.0%}"
                            )
                            result["skipped"].append(
                                f"{sym} (Gap {gap_pct:+.1%} > Fallback {gap_filter_pct:.0%})"
                            )
                            continue
                        else:
                            print(
                                f"  [GAP-PASS] {sym} (ATR n/v – Fallback): "
                                f"Gap ({gap_pct:+.1%}) <= {gap_filter_pct:.0%}"
                            )

            print(
                f"  {sym:<8}  Preis=${price:>8.2f}  "
                f"Ziel={target_qty:>4}  Aktuell={current_qty:>4}  Delta={delta:>+4}{gap_str}"
            )

            if delta == 0:
                print(f"           bereits korrekt gewichtet – keine Order")
                result["skipped"].append(sym)
                continue

            if delta > 0:
                order = _submit_order(trading_client, sym, delta, "buy", dry_run)
                if order is not None or dry_run:
                    result["buys"].append({"symbol": sym, "qty": delta, "price": price})
                    traded_symbols.add(sym)
            else:
                order = _submit_order(trading_client, sym, abs(delta), "sell", dry_run)
                if order is not None or dry_run:
                    result["sells"].append(f"{sym} (Rebalancing)")
                    traded_symbols.add(sym)

    # ── D: Stop-Loss Orders – nur für tatsächlich gehandelte Positionen ─────
    print(f"\n--- D) Hard-Stop-Loss Orders ({stop_loss_pct*100:.0f}% Verlust-Limit) ---")
    if not composition_changed:
        # Szenario A: kein Trade → avg_entry_price unveraendert → kein Update noetig
        print("  Keine Trades ausgefuehrt – Stop-Loss Orders bleiben unveraendert.")
    elif not traded_symbols:
        # Alle Positionen waren bereits korrekt gewichtet oder durch Gap-Filter blockiert
        print("  Keine Orders gesendet – Stop-Loss Orders bleiben unveraendert.")
    else:
        print(f"  Aktualisiere Stops fuer gehandelte Positionen: {sorted(traded_symbols)}")
        if not dry_run:
            # Kurze Pause damit Kauf-Orders vollstaendig verarbeitet sind
            time.sleep(sell_delay_s)
        result["stop_losses"] = set_stop_losses(
            trading_client = trading_client,
            stop_pct       = stop_loss_pct,
            dry_run        = dry_run,
            symbols_filter = traded_symbols,
        )

    # ── Zusammenfassung ───────────────────────────────────────────────────────
    stops_summary = [
        f"{s['symbol']} @ ${s['stop_price']:.2f}"
        for s in result["stop_losses"]
    ]
    szenario_label = "A (Drift)" if not composition_changed else "B (Reset)"
    print(f"\n{'='*60}")
    print(f"  {tag}EXECUTION ABGESCHLOSSEN  [Szenario {szenario_label}]")
    if composition_changed:
        print(f"  Verkauft:      {result['sells']}")
        print(f"  Gekauft:       {[b['symbol'] for b in result['buys']]}")
        print(f"  Uebersprungen: {result['skipped']}")
    else:
        print(f"  Keine Trades (Komposition unveraendert)")
    print(f"  Stop-Loss:     {stops_summary}")
    if result["errors"]:
        print(f"  FEHLER ({len(result['errors'])}):")
        for err in result["errors"]:
            print(f"    - {err}")
    print(f"{'='*60}\n")

    return result
