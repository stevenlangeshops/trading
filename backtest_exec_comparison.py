"""
backtest_exec_comparison.py
────────────────────────────────────────────────────────────────────────────────
Vergleich der Execution-Logiken im Walk-Forward-Backtest (Horizont 7d).

Drei Varianten:
  CLASSIC        – bisherige Logik: täglicher Rotation-Buffer-Check,
                   neue Slots werden zu Equal-Weight eröffnet, bestehende
                   Positionen werden nie aktiv rebalanciert.

  COMP_CHANGE    – neue Logik: Trade nur wenn sich die Portfolio-Komposition
                   (Top-N-Set) ändert. Bei Änderung: vollständiger Equal-Weight-
                   Reset aller Positionen. Bei unveränderter Komposition: Drift.

  COMP_CHANGE_ATR – wie COMP_CHANGE, zusätzlich ATR-14-Gap-Filter für Neukäufe:
                   Kauf wird übersprungen wenn |Preis − Vortag| > 1.5 × ATR-14.

Verwendung (Kaggle, nach step_backtest_single_horizons):
    from backtest_exec_comparison import compare_exec_modes
    comparison = compare_exec_modes(features, fold_results, asset_map, cfg, price_cache)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger

from config_v2_single_horizon import SingleHorizonConfig, get_config
from backtest_v2_single_horizon import (
    load_fold_model,
    predict_cross_section,
    _build_ic_series,
    _log_ic_summary,
)
from strategy.backtest import build_price_cache, get_market_regime, adaptive_n, _get_price


# ── ATR-Hilfsfunktionen ───────────────────────────────────────────────────────

def _estimate_atr14(
    price_series: pd.Series,
    date:         pd.Timestamp,
    period:       int = 14,
) -> Optional[float]:
    """ATR-Proxy aus Schlusskursen: SMA der letzten `period` |Close-Differenzen|.

    Entspricht der True-Range-Formel wenn Intraday-Gaps dominieren
    (typisch für die meisten Aktien). Ausreichend für Backtest-Zwecke.

    Returns:
        ATR-Schätzwert in Dollar, oder ``None`` wenn zu wenig Daten.
    """
    series = price_series[price_series.index <= date].tail(period + 1)
    if len(series) < period + 1:
        return None
    diffs = series.diff().abs().dropna().tail(period)
    if len(diffs) < period:
        return None
    return float(diffs.mean())


# ── Backtest: Neue Composition-Change-Logik ───────────────────────────────────

def run_backtest_comp_change(
    features:     pd.DataFrame,
    fold_results: list[dict],
    asset_map:    dict[str, int],
    cfg:          SingleHorizonConfig,
    price_cache:  dict,
    atr_gap_mult: float = 0.0,
    atr_period:   int   = 14,
) -> dict:
    """Backtest mit Composition-Change-Rebalancing-Logik.

    Unterschied zur klassischen Logik:
      - **Szenario A (set identisch):** Keine Trades. Gewichte driften frei.
      - **Szenario B (set geändert):** Vollständiger Equal-Weight-Reset.
        Bestehende Positionen werden auf target_value rebalanciert (buy/sell delta).
        Neue Positionen werden zum target_value gekauft (optional Gap-Filter).

    Args:
        features:     MultiIndex-Panel (date × asset) × FEATURE_COLS.
        fold_results: Walk-Forward-Fold-Definitionen mit ckpt_path.
        asset_map:    {ticker → asset_id}.
        cfg:          SingleHorizonConfig (n_max, fees, hard_stop_pct, ...).
        price_cache:  {ticker → pd.Series(close_prices)}.
        atr_gap_mult: Multiplikator für ATR-14-Gap-Filter (0 = deaktiviert).
        atr_period:   Periode für ATR-Berechnung (Standard: 14).

    Returns:
        Dict mit Sharpe, Return, MaxDD, trade_log, equity_dates, etc.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    atr_active = atr_gap_mult > 0
    tag = cfg.tag + "_cc" + (f"_atr{atr_gap_mult:.1f}" if atr_active else "")

    all_assets = list(asset_map.keys())
    if "SPY" not in all_assets:
        all_assets.append("SPY")

    spy_prices = price_cache.get("SPY")
    use_regime = spy_prices is not None

    all_dates = features.index.get_level_values("date").unique().sort_values()

    cash      = cfg.init_cash
    equity    = [cfg.init_cash]
    positions: dict = {}       # {symbol: {shares, entry, direction, hold_days}}
    trade_log: list = []
    equity_dates: list = []
    daily_signals: list = []
    score_log: list = []

    gap_skips = 0              # Anzahl durch Gap-Filter blockierter Käufe

    logger.info(f"[{tag}] Composition-Change Backtest  "
                f"n_max={cfg.n_max}  hard_stop={cfg.hard_stop_pct*100:.0f}%  "
                f"ATR-Filter={'%.1f×ATR-%d' % (atr_gap_mult, atr_period) if atr_active else 'AUS'}")

    def _pos_val(pos_dict, pc, dt):
        return sum(
            p["shares"] * (pr := _get_price(pc, a, dt)) * p["direction"]
            for a, p in pos_dict.items()
            if (pr := _get_price(pc, a, dt)) is not None
        )

    def _close(asset, pos, date, reason, regime):
        nonlocal cash
        price = _get_price(price_cache, asset, date)
        if price is None:
            return
        cash += pos["shares"] * price * (1 - cfg.fees) * pos["direction"]
        pnl = (price - pos["entry"]) / pos["entry"] * pos["direction"]
        trade_log.append({
            "date":      str(date.date()),
            "asset":     asset,
            "direction": "long",
            "pnl_pct":   round(pnl * 100, 3),
            "regime":    regime,
            "exit_reason": reason,
            "hold_days": pos.get("hold_days", 0),
        })

    # ── Fold-Iteration ────────────────────────────────────────────────────────
    for fold in fold_results:
        ckpt_path = fold["ckpt_path"]
        if not Path(ckpt_path).exists():
            logger.warning(f"[{tag}] Checkpoint nicht gefunden: {ckpt_path}")
            continue

        model, _ = load_fold_model(ckpt_path, device)
        val_start = pd.Timestamp(fold["val_start"])
        val_end   = pd.Timestamp(fold["val_end"])

        cmp_dates = all_dates.tz_localize(None) if getattr(all_dates, "tz", None) else all_dates
        vs = val_start.tz_localize(None) if val_start.tzinfo else val_start
        ve = val_end.tz_localize(None)   if val_end.tzinfo   else val_end
        fold_dates = all_dates[(cmp_dates >= vs) & (cmp_dates <= ve)]

        logger.info(f"[{tag}]   Fold {fold['fold_id']}: [{vs.date()} → {ve.date()}]")

        for date in fold_dates:
            for pos in positions.values():
                pos["hold_days"] = pos.get("hold_days", 0) + 1

            regime = "neutral"
            if use_regime:
                regime = get_market_regime(spy_prices, date)
            n_long = adaptive_n(regime, cfg.n_max, cfg.n_mid, cfg.n_min)

            ranking = predict_cross_section(
                model, features, asset_map, date, cfg.seq_len, device
            )
            if len(ranking) >= 2:
                score_log.append((date, ranking.copy()))
            if len(ranking) < 2:
                equity_dates.append(date)
                equity.append(cash + _pos_val(positions, price_cache, date))
                continue

            # ── Hard Stop ─────────────────────────────────────────────────────
            to_close = []
            for asset, pos in positions.items():
                price = _get_price(price_cache, asset, date)
                if price is None:
                    continue
                pnl = (price - pos["entry"]) / pos["entry"] * pos["direction"]
                if pnl <= -cfg.hard_stop_pct:
                    to_close.append((asset, "hard_stop"))
            for asset, reason in to_close:
                _close(asset, positions[asset], date, reason, regime)
                del positions[asset]

            # ── Composition Check ─────────────────────────────────────────────
            top_n_set   = set(ranking.index[:n_long])
            current_set = set(positions.keys())

            if top_n_set == current_set:
                # ── Szenario A: Drift ─────────────────────────────────────────
                eq = cash + _pos_val(positions, price_cache, date)
                equity.append(eq)
                equity_dates.append(date)
                daily_signals.append({
                    "date":        str(date.date()),
                    "regime":      regime,
                    "n_long":      n_long,
                    "n_positions": len(positions),
                    "rebalanced":  False,
                    "equity":      round(eq, 2),
                })
                continue

            # ── Szenario B: Composition geändert → Equal-Weight Reset ─────────
            # Schritt 1: Positionen schließen die nicht mehr im Target sind
            for asset in list(positions.keys()):
                if asset not in top_n_set:
                    _close(asset, positions[asset], date, "comp_change", regime)
                    del positions[asset]

            # Schritt 2: Equity nach Verkäufen neu berechnen → target_value
            total_equity = cash + _pos_val(positions, price_cache, date)
            target_value = total_equity / n_long

            # Vortages-Datum für Gap-Filter
            date_idx = cmp_dates.searchsorted(date)
            prev_dt = all_dates[date_idx - 1] if date_idx > 0 else None

            # Schritt 3: Alle Top-N-Positionen auf Equal-Weight bringen
            for asset in ranking.index[:n_long]:
                price = _get_price(price_cache, asset, date)
                if price is None or price <= 0:
                    continue

                if asset in positions:
                    # Bestehende Position: auf target_value rebalancieren
                    target_shares = (target_value * (1 - cfg.fees)) / price
                    delta = target_shares - positions[asset]["shares"]
                    if delta > 0.01:
                        cost = delta * price * (1 + cfg.fees)
                        if cash >= cost:
                            cash -= cost
                            positions[asset]["shares"] += delta
                    elif delta < -0.01:
                        cash += abs(delta) * price * (1 - cfg.fees)
                        positions[asset]["shares"] += delta
                else:
                    # Neue Position: optional Gap-Filter prüfen
                    if atr_active and prev_dt is not None:
                        ps = price_cache.get(asset)
                        if ps is not None:
                            atr = _estimate_atr14(ps, date, period=atr_period)
                            prev_price = _get_price(price_cache, asset, prev_dt)
                            if (
                                atr is not None
                                and atr > 0
                                and prev_price is not None
                            ):
                                gap = abs(price - prev_price)
                                if gap > atr_gap_mult * atr:
                                    gap_skips += 1
                                    logger.debug(
                                        f"  [GAP-SKIP] {asset}: "
                                        f"Gap ${gap:.2f} > {atr_gap_mult}×ATR ${atr:.2f}"
                                    )
                                    continue

                    target_shares = (target_value * (1 - cfg.fees)) / price
                    cost = target_shares * price * (1 + cfg.fees)
                    if cash >= cost:
                        cash -= cost
                        positions[asset] = {
                            "shares":    target_shares,
                            "entry":     price,
                            "direction": 1,
                            "hold_days": 0,
                        }

            eq = cash + _pos_val(positions, price_cache, date)
            equity.append(eq)
            equity_dates.append(date)
            daily_signals.append({
                "date":        str(date.date()),
                "regime":      regime,
                "n_long":      n_long,
                "n_positions": len(positions),
                "rebalanced":  True,
                "equity":      round(eq, 2),
            })

    # ── Statistiken ───────────────────────────────────────────────────────────
    equity_arr   = np.array(equity[1:])
    returns      = np.diff(equity) / equity[:-1]
    returns      = returns[1:]
    total_return = (equity[-1] / equity[0] - 1) * 100 if equity[0] > 0 else 0
    peaks        = np.maximum.accumulate(equity_arr)
    dd           = (equity_arr - peaks) / peaks
    max_dd       = float(dd.min()) * 100 if len(dd) > 0 else 0
    sharpe       = (
        float(np.mean(returns) / np.std(returns) * np.sqrt(252))
        if len(returns) > 1 and np.std(returns) > 0 else 0.0
    )

    n_trades = len(trade_log)
    wins     = [t for t in trade_log if t["pnl_pct"] > 0]
    win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0
    avg_hold = np.mean([t["hold_days"] for t in trade_log]) if trade_log else 0

    rebal_days = sum(1 for s in daily_signals if s.get("rebalanced"))
    drift_days = len(daily_signals) - rebal_days

    exit_stats = {}
    for reason in ("comp_change", "hard_stop"):
        trades_r = [t for t in trade_log if t["exit_reason"] == reason]
        n_r      = len(trades_r)
        exit_stats[reason] = {
            "n":        n_r,
            "pnl_sum":  round(sum(t["pnl_pct"] for t in trades_r), 1),
            "pnl_avg":  round(np.mean([t["pnl_pct"] for t in trades_r]), 2) if trades_r else 0,
            "hold_avg": round(np.mean([t["hold_days"] for t in trades_r]), 1) if trades_r else 0,
            "win_pct":  round(
                len([t for t in trades_r if t["pnl_pct"] > 0]) / n_r * 100, 1
            ) if n_r > 0 else 0,
        }

    logger.success("═" * 60)
    logger.success(f"[{tag}] COMPOSITION-CHANGE Backtest")
    logger.success("═" * 60)
    logger.success(f"[{tag}]   Total Return    : {total_return:+.2f}%")
    logger.success(f"[{tag}]   Max Drawdown    : {max_dd:.2f}%")
    logger.success(f"[{tag}]   Sharpe Ratio    : {sharpe:.3f}")
    logger.success(f"[{tag}]   Trades          : {n_trades}")
    logger.success(f"[{tag}]   Win Rate        : {win_rate:.1f}%")
    logger.success(f"[{tag}]   Avg Hold Days   : {avg_hold:.1f}")
    logger.success(f"[{tag}]   Rebalancing-Tage: {rebal_days}")
    logger.success(f"[{tag}]   Drift-Tage      : {drift_days}  "
                   f"({100*drift_days/(rebal_days+drift_days+1e-9):.0f}%)")
    if atr_active:
        logger.success(f"[{tag}]   ATR-Gap-Skips   : {gap_skips}")
    for reason, st in exit_stats.items():
        logger.success(
            f"[{tag}]   {reason:15s}: n={st['n']:4d}  pnl={st['pnl_sum']:+.0f}%  "
            f"avg={st['pnl_avg']:+.1f}%  hold={st['hold_avg']:.1f}d  win={st['win_pct']:.0f}%"
        )
    logger.success("═" * 60)

    # Rolling-IC berechnen
    try:
        ic_data = _build_ic_series(score_log, price_cache, horizon=cfg.horizon)
        _log_ic_summary(ic_data["ic"], ic_data["rolling"], tag)
    except Exception as exc:
        logger.warning(f"[{tag}] IC-Berechnung fehlgeschlagen: {exc}")
        ic_data = {"ic": pd.Series(dtype=float), "rolling": {}, "records": []}

    return {
        "strategy":       tag,
        "horizon":        cfg.horizon,
        "mode":           "comp_change" + ("_atr" if atr_active else ""),
        "total_return":   round(total_return, 2),
        "max_drawdown":   round(max_dd, 2),
        "sharpe":         round(sharpe, 3),
        "n_trades":       n_trades,
        "win_rate":       round(win_rate, 1),
        "avg_hold_days":  round(avg_hold, 1),
        "rebal_days":     rebal_days,
        "drift_days":     drift_days,
        "drift_pct":      round(100 * drift_days / (rebal_days + drift_days + 1e-9), 1),
        "gap_skips":      gap_skips,
        "exit_stats":     exit_stats,
        "equity":         equity,
        "equity_dates":   equity_dates,
        "trade_log":      trade_log,
        "daily_signals":  daily_signals,
        "ic_data":        ic_data,
    }


# ── 3-fach-Vergleich ──────────────────────────────────────────────────────────

def compare_exec_modes(
    features:     pd.DataFrame,
    fold_results: list[dict],
    asset_map:    dict[str, int],
    cfg:          SingleHorizonConfig,
    price_cache:  dict,
    atr_gap_mult: float = 1.5,
) -> dict:
    """Führt Backtest in 3 Execution-Modi durch und gibt Vergleichstabelle aus.

    Modi:
      1. CLASSIC        – Rotation-Buffer (bisherige Logik)
      2. COMP_CHANGE    – Composition-Change ohne Gap-Filter
      3. COMP_CHANGE_ATR – Composition-Change + ATR-14-Gap-Filter

    Args:
        features:     Feature-Panel (date × asset).
        fold_results: Walk-Forward-Fold-Liste.
        asset_map:    Ticker → Asset-ID.
        cfg:          Produktions-Konfiguration (Horizont 7d).
        price_cache:  Close-Preis-Cache.
        atr_gap_mult: Multiplikator für ATR-14-Grenze (Standard: 1.5).

    Returns:
        Dict ``{"classic": ..., "comp_change": ..., "comp_change_atr": ...}``
        mit vollständigen Backtest-Ergebnissen pro Modus.
    """
    from backtest_v2_single_horizon import run_backtest_single_horizon

    logger.info("\n" + "═" * 70)
    logger.info("EXECUTION-MODE-VERGLEICH  (Horizont 7d)")
    logger.info("═" * 70)

    logger.info("[1/3] CLASSIC: Rotation-Buffer-Rebalancing ...")
    classic = run_backtest_single_horizon(
        features, fold_results, asset_map, cfg, price_cache
    )
    classic["mode"] = "CLASSIC"

    logger.info("[2/3] COMP_CHANGE: Composition-Only-Rebalancing ...")
    comp = run_backtest_comp_change(
        features, fold_results, asset_map, cfg, price_cache, atr_gap_mult=0.0
    )
    comp["mode"] = "COMP_CHANGE"

    logger.info(f"[3/3] COMP_CHANGE_ATR: + ATR-14-Gap-Filter ({atr_gap_mult}×ATR) ...")
    comp_atr = run_backtest_comp_change(
        features, fold_results, asset_map, cfg, price_cache, atr_gap_mult=atr_gap_mult
    )
    comp_atr["mode"] = "COMP_CHANGE_ATR"

    _print_comparison([classic, comp, comp_atr])

    return {
        "classic":         classic,
        "comp_change":     comp,
        "comp_change_atr": comp_atr,
    }


def _print_comparison(results: list[dict]) -> None:
    """Gibt die Vergleichstabelle als formatiertes Log aus."""
    logger.info("\n" + "═" * 100)
    logger.info("VERGLEICH: Classic vs. Composition-Change Rebalancing")
    logger.info("═" * 100)
    hdr = (
        f"{'Modus':<20} {'TotRet':>9} {'MaxDD':>8} {'Sharpe':>8} "
        f"{'Trades':>8} {'Win%':>7} {'AvgHold':>9} "
        f"{'Rebal-T':>9} {'Drift%':>8} {'GapSkip':>9}"
    )
    logger.info(hdr)
    logger.info("─" * 100)
    for r in results:
        mode   = r.get("mode", r.get("strategy", "?"))
        drift  = r.get("drift_pct", "—")
        drift_s = f"{drift:.0f}%" if isinstance(drift, float) else "—"
        skips  = r.get("gap_skips", "—")
        skips_s = str(skips) if isinstance(skips, int) else "—"
        rebal  = r.get("rebal_days", "—")
        rebal_s = str(rebal) if isinstance(rebal, int) else "—"
        logger.info(
            f"{mode:<20} {r.get('total_return', 0):>+9.1f}% "
            f"{r.get('max_drawdown', 0):>8.1f}% "
            f"{r.get('sharpe', 0):>8.3f} "
            f"{r.get('n_trades', 0):>8} "
            f"{r.get('win_rate', 0):>6.1f}% "
            f"{r.get('avg_hold_days', 0):>8.1f}d "
            f"{rebal_s:>9} "
            f"{drift_s:>8} "
            f"{skips_s:>9}"
        )
    logger.info("═" * 100)
    logger.info(
        "Hinweis: Drift% = Anteil Handelstage OHNE Trade (Szenario A). "
        "GapSkip = durch ATR-Filter blockierte Neukäufe."
    )


def plot_exec_comparison(
    results: dict,
    save_path: str = "exec_comparison_equity.png",
) -> None:
    """Equity-Kurven-Vergleich der drei Execution-Modi."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker

        fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                                 gridspec_kw={"height_ratios": [3, 1]})
        ax = axes[0]
        ax2 = axes[1]

        colors = {
            "classic":         "#1565C0",
            "comp_change":     "#2E7D32",
            "comp_change_atr": "#E65100",
        }
        labels = {
            "classic":         "CLASSIC (Rotation-Buffer)",
            "comp_change":     "COMP_CHANGE (Drift erlaubt)",
            "comp_change_atr": "COMP_CHANGE + ATR-Gap-Filter",
        }

        for key, bt in results.items():
            dates  = bt.get("equity_dates", [])
            eq     = bt.get("equity", [])
            if not dates or not eq:
                continue
            eq_plot = eq[1:] if len(eq) > len(dates) else eq
            ret     = [(e / eq_plot[0] - 1) * 100 for e in eq_plot]
            ax.plot(
                dates, ret,
                color=colors.get(key, "#333"),
                linewidth=2.0,
                label=f"{labels.get(key, key)}  {bt.get('total_return', 0):+.1f}%  "
                      f"Sharpe={bt.get('sharpe', 0):.3f}",
            )
            # Drawdown
            eq_arr = np.array(eq_plot)
            peaks  = np.maximum.accumulate(eq_arr)
            dd     = (eq_arr - peaks) / peaks * 100
            ax2.fill_between(dates, dd, 0, alpha=0.18, color=colors.get(key, "#333"))
            ax2.plot(dates, dd, color=colors.get(key, "#333"), linewidth=0.8,
                     label=f"{labels.get(key, key)}  MaxDD={dd.min():.1f}%")

        ax.set_title(
            "Execution-Mode-Vergleich: Classic vs. Composition-Change Rebalancing",
            fontsize=13, fontweight="bold",
        )
        ax.set_ylabel("Kumulativer Return (%)")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:+.0f}%")
        )

        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Datum")
        ax2.legend(loc="lower left", fontsize=8)
        ax2.grid(True, alpha=0.25)
        ax2.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:.0f}%")
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.success(f"Exec-Comparison-Chart: {save_path}")
    except Exception as exc:
        logger.warning(f"plot_exec_comparison Fehler: {exc}")
