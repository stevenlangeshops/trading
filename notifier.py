"""
notifier.py
══════════════════════════════════════════════════════════════════════════════
Telegram-Benachrichtigungen für den Trading-Bot.

Funktionen:
    send_telegram_message()   – Einfache Text-Nachricht
    send_portfolio_report()   – Vollständiger Report mit Chart:
                                  - Portfoliowert & Gesamt-P&L
                                  - P&L je gehaltener / verkaufter Position
                                  - Performance-Vergleich vs. MSCI World
                                    (24h, 1W, 1M, 1Y)
                                  - Equity-Kurven-Chart als Bild

Umgebungsvariablen:
    TELEGRAM_TOKEN    – Bot-Token (von @BotFather)
    TELEGRAM_CHAT_ID  – Ziel-Chat-ID
"""

from __future__ import annotations

import io
import logging
import os
import warnings
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# MSCI World Proxy – iShares MSCI World ETF (NYSE)
_MSCI_TICKER = "URTH"


# ══════════════════════════════════════════════════════════════════════════════
# Interne Helfer
# ══════════════════════════════════════════════════════════════════════════════

def _credentials() -> tuple[str, str]:
    token   = os.getenv("TELEGRAM_TOKEN",  "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID","").strip()
    return token, chat_id


def _post_text(token: str, chat_id: str, text: str) -> None:
    url  = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    r    = requests.post(url, data=data, timeout=15)
    if not r.ok:
        print(f"[WARN] Telegram sendMessage fehlgeschlagen: {r.status_code} {r.text[:200]}")


def _post_photo(token: str, chat_id: str, img_bytes: bytes, caption: str) -> None:
    url   = f"https://api.telegram.org/bot{token}/sendPhoto"
    data  = {"chat_id": chat_id, "caption": caption, "parse_mode": "HTML"}
    files = {"photo": ("chart.png", img_bytes, "image/png")}
    r     = requests.post(url, data=data, files=files, timeout=30)
    if r.ok:
        print(f"  [Telegram] Bild gesendet ({len(img_bytes)//1024} KB)")
    else:
        print(f"  [WARN] Telegram sendPhoto fehlgeschlagen: {r.status_code} {r.text[:300]}")
        # Fallback: nur Text senden
        _post_text(token, chat_id, caption)


def _alpaca_client():
    """Gibt einen Alpaca TradingClient zurück (wenn Credentials vorhanden)."""
    key    = os.getenv("APCA_API_KEY_ID",     "").strip()
    secret = os.getenv("APCA_API_SECRET_KEY", "").strip()
    paper  = os.getenv("APCA_PAPER", "true").lower() != "false"
    if not key or not secret:
        return None
    from alpaca.trading.client import TradingClient
    return TradingClient(api_key=key, secret_key=secret, paper=paper)


def _get_portfolio_history(client, periods=("1A", "6M", "3M", "1M", "1W")) -> Optional[pd.DataFrame]:
    """Lädt die tägliche Equity-History aus Alpaca.

    Probiert Perioden von lang nach kurz; gibt DataFrame mit Index=Datum,
    Spalte 'equity' zurück, oder None bei Fehler.
    """
    from alpaca.trading.requests import GetPortfolioHistoryRequest
    for period in periods:
        try:
            hist = client.get_portfolio_history(
                filter=GetPortfolioHistoryRequest(period=period, timeframe="1D")
            )
            if not hist or not hist.equity:
                continue
            ts     = [datetime.fromtimestamp(t, tz=timezone.utc) for t in hist.timestamp]
            equity = [float(e) if e is not None else float("nan") for e in hist.equity]
            df     = pd.DataFrame({"equity": equity}, index=pd.DatetimeIndex(ts))
            df     = df.dropna()
            if len(df) >= 2:
                return df
        except Exception as exc:
            logger.debug(f"Portfolio-History {period} fehlgeschlagen: {exc}")
    return None


def _download_msci(start: datetime, end: datetime) -> Optional[pd.Series]:
    """Lädt MSCI-World-Proxy (URTH) von yfinance."""
    import yfinance as yf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            df = yf.download(
                _MSCI_TICKER,
                start=start.date(),
                end=(end + timedelta(days=1)).date(),
                auto_adjust=True,
                progress=False,
            )
            if df.empty:
                return None
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close.index = pd.to_datetime(close.index).tz_localize("UTC")
            return close.dropna()
        except Exception as exc:
            logger.warning(f"MSCI-World Download fehlgeschlagen: {exc}")
            return None


def _perf_pct(series: pd.Series, days: int) -> Optional[float]:
    """Berechnet prozentuale Veränderung über die letzten `days` Tage."""
    if series is None or len(series) < 2:
        return None
    cutoff = series.index[-1] - pd.Timedelta(days=days)
    sub    = series[series.index >= cutoff]
    if len(sub) < 2:
        return None
    return (sub.iloc[-1] / sub.iloc[0] - 1) * 100


def _make_chart(
    portfolio_hist:    Optional[pd.DataFrame],
    msci_series:       Optional[pd.Series],
    positions:         list,
    sold_symbols:      list[str],
    title_date:        str,
    portfolio_start:   Optional[datetime] = None,
) -> bytes:
    """Erstellt Matplotlib-Chart und gibt PNG-Bytes zurück."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker
    import numpy as np

    # ── Layout ────────────────────────────────────────────────────────────────
    has_equity  = portfolio_hist is not None and len(portfolio_hist) >= 2
    has_msci    = msci_series is not None and len(msci_series) >= 2
    show_curves = has_equity or has_msci          # oberes Panel immer wenn mind. MSCI vorhanden
    has_pos     = len(positions) > 0
    n_rows      = (1 if show_curves else 0) + (1 if has_pos else 0)
    if n_rows == 0:
        n_rows = 1

    fig_h = 5 * n_rows
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(11, fig_h),
        facecolor="#0d1117",
        gridspec_kw={"hspace": 0.45},
    )
    if n_rows == 1:
        axes = [axes]

    ax_idx = 0

    # ── 1. Equity-Kurven ─────────────────────────────────────────────────────
    if show_curves:
        ax = axes[ax_idx]; ax_idx += 1
        ax.set_facecolor("#161b22")

        # Startdatum: ältestes Datum aus verfügbaren Serien
        start_ts = None

        # Portfolio-Linie (nur wenn History vorhanden)
        if has_equity:
            eq      = portfolio_hist["equity"]
            eq_norm = eq / eq.iloc[0] * 100
            start_ts = eq.index[0]

            ax.plot(eq.index, eq_norm, color="#58a6ff", linewidth=2.2,
                    label="Mein Portfolio", zorder=3)
            ax.fill_between(eq.index, 100, eq_norm,
                            where=(eq_norm >= 100),
                            color="#238636", alpha=0.25)
            ax.fill_between(eq.index, 100, eq_norm,
                            where=(eq_norm < 100),
                            color="#da3633", alpha=0.25)
        else:
            # Hinweis: noch keine Portfolio-History
            ax.text(0.5, 0.65,
                    "Portfolio-History noch nicht verfügbar\n(Paper-Account zu neu)",
                    transform=ax.transAxes, ha="center", va="center",
                    color="#8b949e", fontsize=9, style="italic")

        # MSCI World Linie – immer zeigen, ab Depot-Startdatum
        if has_msci:
            # Priorität: 1. Portfolio-Startdatum (aus History)
            #             2. account.created_at (übergeben als portfolio_start)
            #             3. start_ts aus Portfolio-Equity
            msci_start = None
            if portfolio_start is not None:
                msci_start = pd.Timestamp(portfolio_start).tz_convert("UTC")
            elif start_ts is not None:
                msci_start = start_ts

            msci_sub = msci_series
            if msci_start is not None:
                msci_sub = msci_series[msci_series.index >= msci_start]
                if len(msci_sub) < 2:   # Fallback: nimm alle Daten
                    msci_sub = msci_series
            if len(msci_sub) >= 2:
                msci_norm = msci_sub / msci_sub.iloc[0] * 100
                ax.plot(msci_norm.index, msci_norm,
                        color="#f0883e", linewidth=1.8,
                        linestyle="--", label="MSCI World (URTH)", zorder=2)
                ax.axhline(msci_norm.iloc[0], color="#30363d",
                           linewidth=1, linestyle=":")

        ax.set_title(f"Portfolio vs. MSCI World | {title_date}",
                     color="#e6edf3", fontsize=13, pad=10)
        ax.set_ylabel("Normiert (Start = 100)", color="#8b949e", fontsize=9)
        ax.tick_params(colors="#8b949e", labelsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        ax.spines[:].set_color("#30363d")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
        ax.legend(fontsize=9, facecolor="#161b22", edgecolor="#30363d",
                  labelcolor="#e6edf3", loc="upper left")
        ax.grid(axis="y", color="#21262d", linewidth=0.8)

    # ── 2. Positions-P&L Balken ───────────────────────────────────────────────
    if has_pos:
        ax = axes[ax_idx]; ax_idx += 1
        ax.set_facecolor("#161b22")

        symbols = [p.symbol for p in positions]
        pnl_pct = [float(p.unrealized_plpc) * 100 for p in positions]
        pnl_abs = [float(p.unrealized_pl)        for p in positions]
        colors  = ["#238636" if v >= 0 else "#da3633" for v in pnl_pct]

        # Verkaufte Positionen (grau, gestrichelt)
        for sym in sold_symbols:
            symbols.append(f"{sym} ✗")
            pnl_pct.append(0)
            pnl_abs.append(0)
            colors.append("#484f58")

        y_pos = range(len(symbols))
        bars  = ax.barh(list(y_pos), pnl_pct, color=colors,
                        height=0.6, edgecolor="#21262d")

        for i, (bar, pct, abs_) in enumerate(zip(bars, pnl_pct, pnl_abs)):
            sign = "+" if pct >= 0 else ""
            label = f"  {sign}{pct:.1f}%  ({sign}${abs_:,.0f})"
            x_offset = max(pct, 0) + 0.05 if pct >= 0 else min(pct, 0) - 0.05
            ax.text(x_offset, bar.get_y() + bar.get_height() / 2,
                    label, va="center",
                    ha="left" if pct >= 0 else "right",
                    color="#e6edf3", fontsize=8.5)

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(symbols, color="#e6edf3", fontsize=9)
        ax.axvline(0, color="#8b949e", linewidth=1)
        ax.set_title("Unrealisierter P&L je Position (aktuell gehalten)",
                     color="#e6edf3", fontsize=11, pad=8)
        ax.set_xlabel("Unrealisierter Gewinn / Verlust (%)",
                      color="#8b949e", fontsize=9)
        ax.tick_params(colors="#8b949e", labelsize=8)
        ax.spines[:].set_color("#30363d")
        ax.grid(axis="x", color="#21262d", linewidth=0.8)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

    fig.patch.set_facecolor("#0d1117")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="#0d1117")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _fmt_perf(pf: Optional[float], msci: Optional[float]) -> str:
    """Formatiert eine Zeile der Performance-Tabelle."""
    def _fmt(v):
        if v is None:
            return "  n/a  "
        sign = "+" if v >= 0 else ""
        return f"{sign}{v:.2f}%"
    return f"Portfolio {_fmt(pf)}  |  MSCI World {_fmt(msci)}"


# ══════════════════════════════════════════════════════════════════════════════
# Öffentliche API
# ══════════════════════════════════════════════════════════════════════════════

def send_telegram_message(message: str) -> None:
    """Sendet eine einfache Markdown-Textnachricht via Telegram.

    Args:
        message: Nachrichtentext (Markdown erlaubt).
    """
    token, chat_id = _credentials()
    if not token or not chat_id:
        print("Telegram-Config fehlt. Nachricht konnte nicht gesendet werden.")
        return
    try:
        _post_text(token, chat_id, message)
    except Exception as exc:
        print(f"Fehler beim Senden der Nachricht: {exc}")


def send_portfolio_report(
    top_tickers:      list[str],
    scores:           "pd.Series",
    n_eff:            int,
    a3_active:        bool,
    ic_roll_40:       Optional[float],
    target_date:      "pd.Timestamp",
    execution_result: Optional[dict] = None,
    dry_run:          bool = False,
) -> None:
    """Sendet vollständigen Portfolio-Report als Chart-Bild mit Text-Caption.

    Enthält:
      - Portfoliowert & Gesamt-P&L
      - P&L je Position (gehalten & verkauft)
      - Performance vs. MSCI World (24h / 1W / 1M / 1Y)
      - Equity-Kurven-Chart als PNG

    Args:
        top_tickers:      Aktuell gekaufte Ticker (nach Inference).
        scores:           Score-Serie aller Assets.
        n_eff:            Effektive Anzahl Positionen.
        a3_active:        Ob A3-Policy aktiv ist.
        ic_roll_40:       Aktueller IC_roll_40-Wert.
        target_date:      Datum des Inference-Laufs.
        execution_result: Rückgabe von execute_target_allocation().
        dry_run:          True → Dry-Run-Hinweis in der Nachricht.
    """
    token, chat_id = _credentials()
    if not token or not chat_id:
        print("[WARN] Telegram-Config fehlt – kein Report gesendet.")
        return

    mode_tag = " 🔵 DRY-RUN" if dry_run else ""

    # ── Alpaca-Daten laden ────────────────────────────────────────────────────
    client       = _alpaca_client()
    account      = None
    positions    = []
    port_history = None

    portfolio_start = None
    if client:
        try:
            account         = client.get_account()
            positions       = client.get_all_positions()
            portfolio_start = getattr(account, "created_at", None)
        except Exception as exc:
            logger.warning(f"Alpaca account/positions fehlgeschlagen: {exc}")
        try:
            port_history = _get_portfolio_history(client)
        except Exception as exc:
            logger.warning(f"Portfolio-History fehlgeschlagen: {exc}")

    # ── Equity-Werte ──────────────────────────────────────────────────────────
    equity       = float(account.equity)       if account else 0.0
    last_equity  = float(account.last_equity)  if account else 0.0
    cash         = float(account.cash)         if account else 0.0
    total_pl_abs = equity - float(account.last_equity) if account else 0.0  # Tages-P&L
    total_pl_pct = (total_pl_abs / last_equity * 100) if last_equity else 0.0

    # Gesamt-unrealisierter P&L über alle Positionen
    unrealized_total = sum(float(p.unrealized_pl) for p in positions)
    cost_total       = sum(float(p.cost_basis)    for p in positions)
    unrealized_pct   = (unrealized_total / cost_total * 100) if cost_total else 0.0

    # ── Performance-Zeitreihen ────────────────────────────────────────────────
    now   = datetime.now(tz=timezone.utc)
    start = now - timedelta(days=400)

    msci_series  = _download_msci(start, now)
    port_series  = (port_history["equity"].rename("equity")
                    if port_history is not None else None)
    if port_series is not None:
        port_series.index = port_series.index.tz_convert("UTC")

    def perf(s, days):
        return _perf_pct(s, days)

    p24h  = perf(port_series,  1)
    p1w   = perf(port_series,  7)
    p1m   = perf(port_series, 30)
    p1y   = perf(port_series,365)

    m24h  = perf(msci_series,  1)
    m1w   = perf(msci_series,  7)
    m1m   = perf(msci_series, 30)
    m1y   = perf(msci_series,365)

    # ── Verkaufte Positionen aus execution_result ─────────────────────────────
    sold_raw  = []
    stop_info = []
    if execution_result:
        sold_raw  = [s for s in execution_result.get("sells", [])
                     if "Rebalancing" not in s]
        stop_info = execution_result.get("stop_losses", [])

    held_symbols = {p.symbol for p in positions}

    # ── Text-Caption aufbauen ─────────────────────────────────────────────────
    sign_u = "+" if unrealized_total >= 0 else ""
    sign_d = "+" if total_pl_abs    >= 0 else ""

    pos_lines = []
    for p in sorted(positions, key=lambda x: float(x.unrealized_plpc), reverse=True):
        pct = float(p.unrealized_plpc) * 100
        abs_ = float(p.unrealized_pl)
        s = "+" if pct >= 0 else ""
        pos_lines.append(f"  {p.symbol:<6} {s}{pct:.1f}%  ({s}${abs_:,.0f})")

    for sym in sold_raw:
        clean = sym.replace(" (Rebalancing)", "")
        pos_lines.append(f"  {clean:<6} verkauft")

    ic_str  = f"{ic_roll_40:+.4f}" if ic_roll_40 is not None else "n/a"
    pol_str = "AKTIV" if a3_active else "Inaktiv"
    top_str = "\n".join(
        f"  {i+1}. {t}  Score={scores.get(t, 0):+.4f}"
        for i, t in enumerate(top_tickers)
    )

    caption = (
        f"<b>Trading-Bot Report{mode_tag} | {target_date.date()}</b>\n"
        f"{'─'*34}\n"
        f"<b>Portfoliowert:</b>  ${equity:>12,.2f}\n"
        f"<b>Gesamt P&amp;L:</b>     {sign_u}${unrealized_total:,.0f}  ({sign_u}{unrealized_pct:.1f}%)\n"
        f"<b>Tages-P&amp;L:</b>      {sign_d}${total_pl_abs:,.0f}  ({sign_d}{total_pl_pct:.1f}%)\n"
        f"<b>Cash:</b>           ${cash:>12,.2f}\n"
        f"{'─'*34}\n"
        f"<b>Positionen P&amp;L:</b>\n"
        + ("\n".join(pos_lines) if pos_lines else "  (keine Positionen)") + "\n"
        + f"{'─'*34}\n"
        f"<b>Performance vs. MSCI World:</b>\n"
        f"  24h:  {_fmt_perf(p24h, m24h)}\n"
        f"  1W:   {_fmt_perf(p1w,  m1w)}\n"
        f"  1M:   {_fmt_perf(p1m,  m1m)}\n"
        f"  1Y:   {_fmt_perf(p1y,  m1y)}\n"
        f"{'─'*34}\n"
        f"<b>Ziel-Allokation ({n_eff} Positionen):</b>\n"
        f"{top_str}\n"
        f"IC_roll_40: {ic_str}  |  A3-Policy: {pol_str}\n"
        + (
            f"{'─'*34}\n"
            f"<b>Stop-Loss Orders (20%):</b>\n"
            + "\n".join(
                f"  {s['symbol']:<6} Stop @ ${s['stop_price']:.2f}"
                for s in stop_info
            )
            if stop_info else ""
        )
    )

    # Caption auf 1024 Zeichen kürzen (Telegram-Limit für Foto-Captions)
    if len(caption) > 1024:
        caption = caption[:1020] + "\n..."

    # ── Chart generieren ──────────────────────────────────────────────────────
    try:
        img_bytes = _make_chart(
            portfolio_hist  = port_history,
            msci_series     = msci_series,
            positions       = list(positions),
            sold_symbols    = sold_raw,
            title_date      = str(target_date.date()),
            portfolio_start = portfolio_start,
        )
        _post_photo(token, chat_id, img_bytes, caption)
    except Exception as exc:
        logger.warning(f"Chart-Generierung fehlgeschlagen ({exc}) – sende nur Text.")
        _post_text(token, chat_id, caption)
