"""
strategy/backtest.py
─────────────────────
Cross-Sectional Backtest für Walk-Forward LSTM.

Zwei Varianten:
  A) Long-Only:     Kaufe täglich die Top-N Assets nach predicted return
  B) Long-Short:    Kaufe Top-N, leerverkaufe Bottom-N

Adaptives N:
  N ist nicht fest sondern hängt vom Marktregime ab:
  - Trending (SPY > SMA50): N = n_max (z.B. 5)
  - Seitwärts (SPY < SMA50): N = n_min (z.B. 2)
  - Bär (SPY < SMA200):  N = 0 (kein Trade / nur Shorts)

Täglich wird das Portfolio neu gewichtet (equal weight).
Transaktionskosten werden bei jedem Kauf/Verkauf abgezogen.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger

from models.lstm_model import CrossSectionalLSTM

CHECKPOINT_DIR = Path("checkpoints")
RAW_DIR        = Path("data/raw")


# ── Price-Cache ───────────────────────────────────────────────────────────────

def build_price_cache(
    assets:  list[str],
    raw_dir: Path = RAW_DIR,
) -> dict[str, pd.Series]:
    """
    Lädt alle Close-Preis-Serien einmalig in einen Dict.

    Statt bei jedem Trade die Parquet-Datei von Disk zu lesen, wird dieser
    Cache einmalig beim Backtest-Start aufgebaut und danach nur noch im
    Speicher nachgeschlagen.

    Returns:
        Dict: asset-Name → pd.Series(index=DatetimeIndex, values=close)
    """
    cache: dict[str, pd.Series] = {}
    for asset in assets:
        fpath = raw_dir / f"{asset.replace('.', '_')}_1d.parquet"
        if not fpath.exists():
            logger.warning(f"price_cache: {asset} nicht gefunden, wird übersprungen")
            continue
        try:
            df = pd.read_parquet(fpath, columns=['close'])
            df.index = pd.to_datetime(df.index)
            cache[asset] = df['close']
        except Exception as exc:
            logger.warning(f"price_cache: {asset} Ladefehler ({exc})")
    logger.info(f"price_cache: {len(cache)}/{len(assets)} Assets geladen")
    return cache


# ── Modell laden ──────────────────────────────────────────────────────────────

def load_fold_model(ckpt_path: str, device: str) -> tuple[CrossSectionalLSTM, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt["config"]
    model = CrossSectionalLSTM(
        n_features = cfg["n_features"],
        n_assets   = cfg["n_assets"],
        embed_dim  = cfg["embed_dim"],
        hidden_dim = cfg["hidden_dim"],
        num_layers = cfg["num_layers"],
        dropout    = 0.0,   # kein Dropout bei Inference
        seq_len    = cfg["seq_len"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


# ── Marktregime bestimmen ─────────────────────────────────────────────────────

def get_market_regime(
    spy_prices: pd.Series,
    date:       pd.Timestamp,
    window_fast: int = 50,
    window_slow: int = 200,
) -> str:
    """
    Bestimmt das Marktregime anhand von SPY SMAs.

    Returns:
        'bull'    — SPY > SMA50 > SMA200  → volle Position
        'neutral' — SPY > SMA200 aber < SMA50 → reduzierte Position
        'bear'    — SPY < SMA200  → Long-Only: kein Trade, Short bleibt
    """
    past = spy_prices[spy_prices.index <= date]
    if len(past) < window_slow:
        return 'neutral'

    price    = float(past.iloc[-1])
    sma_fast = float(past.iloc[-window_fast:].mean())
    sma_slow = float(past.iloc[-window_slow:].mean())

    if price > sma_fast and sma_fast > sma_slow:
        return 'bull'
    elif price > sma_slow:
        return 'neutral'
    else:
        return 'bear'


def adaptive_n(
    regime:  str,
    n_max:   int = 5,
    n_mid:   int = 3,
    n_min:   int = 1,
) -> int:
    """Gibt N in Abhängigkeit vom Marktregime zurück."""
    return {'bull': n_max, 'neutral': n_mid, 'bear': n_min}[regime]


# ── Tages-Signale generieren ──────────────────────────────────────────────────

@torch.no_grad()
def predict_cross_section(
    model:     CrossSectionalLSTM,
    features:  pd.DataFrame,   # MultiIndex (date, asset)
    asset_map: dict[str, int],
    date:      pd.Timestamp,
    seq_len:   int,
    device:    str,
) -> pd.Series:
    """
    Generiert für einen Tag t Vorhersagen für alle verfügbaren Assets.
    Returns: pd.Series  asset → predicted_return (sortiert absteigend)
    """
    # Alle Assets die an diesem Tag verfügbar sind
    try:
        assets_today = features.xs(date, level='date').index.tolist()
    except KeyError:
        return pd.Series(dtype=float)

    preds = {}
    all_dates = features.index.get_level_values('date').unique().sort_values()
    date_idx  = all_dates.searchsorted(date)

    for asset in assets_today:
        asset_id = asset_map.get(asset, 0)
        try:
            asset_feat = features.xs(asset, level='asset').sort_index()
        except KeyError:
            continue

        # Letzte seq_len Tage bis inkl. date
        past = asset_feat[asset_feat.index <= date].iloc[-seq_len:]
        if len(past) < seq_len:
            continue

        x = torch.from_numpy(past.values.astype(np.float32)).unsqueeze(0).to(device)
        a = torch.tensor([asset_id], dtype=torch.long).to(device)
        pred = model(x, a).item()
        preds[asset] = pred

    return pd.Series(preds).sort_values(ascending=False)


# ── Backtest Engine ───────────────────────────────────────────────────────────

def run_backtest(
    features:         pd.DataFrame,
    targets:          pd.Series,
    fold_results:     list[dict],
    asset_map:        dict[str, int],
    # Strategie-Parameter
    n_max:            int   = 5,      # Max Positionen bei Bull-Markt
    n_mid:            int   = 3,      # Positionen bei neutralem Markt
    n_min:            int   = 1,      # Min Positionen bei Bär-Markt
    long_short:       bool  = False,  # True = Long-Short, False = Long-Only
    fees:             float = 0.001,  # 0.1% Transaktionskosten pro Seite
    init_cash:        float = 10_000.0,
    seq_len:          int   = 64,
    # Dynamisches Positions-Management
    stop_loss_pct:    float = 0.05,   # 5% Stop-Loss vom Einstiegskurs
    rotation_buffer:  int   = 2,      # Rotation nur wenn neuer Kandidat >= buffer Ränge besser
    # Regime-Filter
    use_regime:       bool  = True,   # Adaptives N aktivieren
    spy_ticker:       str   = "SPY",  # Referenz-Asset für Regime
    # Optionaler Pre-built Price-Cache
    price_cache:      Optional[dict] = None,
) -> dict:
    """
    Führt den Cross-Sectional Backtest über alle Folds durch.

    Täglich:
      1. Regime bestimmen (adaptives N)
      2. Frische Modell-Vorhersagen für alle Assets
      3. Jede gehaltene Position prüfen:
         a. Stop-Loss: exit wenn Verlust > stop_loss_pct
         b. Rotation: exit wenn Rang schlechter als n_long + rotation_buffer
            UND ein freier besserer Kandidat existiert
      4. Freie Slots mit den besten ungehaltenen Top-N Kandidaten befüllen
      5. Positionen können beliebig lang gehalten werden (kein festes Zeitlimit)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Price-Cache einmalig aufbauen (falls nicht von außen übergeben)
    if price_cache is None:
        all_assets = list(asset_map.keys())
        if use_regime and spy_ticker not in all_assets:
            all_assets.append(spy_ticker)
        price_cache = build_price_cache(all_assets)

    # SPY Preisreihe für Regime-Filter aus Cache
    spy_prices = None
    if use_regime:
        spy_prices = price_cache.get(spy_ticker)
        if spy_prices is None:
            logger.warning(f"{spy_ticker} nicht im price_cache, Regime-Filter deaktiviert")
            use_regime = False

    all_dates = features.index.get_level_values('date').unique().sort_values()

    # Equity-Kurve und Trade-Log
    equity    = [init_cash]
    cash      = init_cash
    positions = {}   # asset → {'shares': float, 'entry': float, 'direction': +1/-1}
    trade_log = []
    equity_dates = []

    strategy_name = "Long-Short" if long_short else "Long-Only"
    logger.info(
        f"Backtest: {strategy_name}  n_max={n_max}  fees={fees:.3f}  "
        f"stop_loss={stop_loss_pct*100:.1f}%  rotation_buffer={rotation_buffer}"
    )

    def _close_position(asset: str, pos: dict, date, reason: str, regime: str) -> None:
        """Schließt eine Position, bucht Erlös und schreibt Trade-Log."""
        nonlocal cash
        price = _get_price(price_cache, asset, date)
        if price is None:
            return
        proceeds = pos['shares'] * price * (1 - fees) * pos['direction']
        cash    += proceeds
        pnl      = (price - pos['entry']) / pos['entry'] * pos['direction']
        trade_log.append({
            'date':       str(date.date()),
            'asset':      asset,
            'direction':  'long' if pos['direction'] == 1 else 'short',
            'pnl_pct':    round(pnl * 100, 3),
            'regime':     regime,
            'exit_reason': reason,
            'hold_days':  pos.get('hold_days', 0),
        })

    for fold in fold_results:
        ckpt_path = fold['ckpt_path']
        if not Path(ckpt_path).exists():
            logger.warning(f"Checkpoint nicht gefunden: {ckpt_path}")
            continue

        model, ckpt_meta = load_fold_model(ckpt_path, device)
        val_start = pd.Timestamp(fold['val_start'])
        val_end   = pd.Timestamp(fold['val_end'])

        logger.info(f"  Fold {fold['fold_id']}: [{val_start.date()} → {val_end.date()}]")

        cmp_dates = all_dates.tz_localize(None) if getattr(all_dates, 'tz', None) is not None else all_dates
        vs = val_start.tz_localize(None) if val_start.tzinfo is not None else val_start
        ve = val_end.tz_localize(None)   if val_end.tzinfo   is not None else val_end
        fold_dates = all_dates[(cmp_dates >= vs) & (cmp_dates <= ve)]

        for date in fold_dates:
            # ── Regime bestimmen ──────────────────────────────────────────
            if use_regime and spy_prices is not None:
                regime = get_market_regime(spy_prices, date)
                n_long = adaptive_n(regime, n_max, n_mid, n_min)
                if regime == 'bear' and long_short:
                    n_long = 0
            else:
                regime = 'neutral'
                n_long = n_mid

            # ── Vorhersagen generieren ────────────────────────────────────
            preds = predict_cross_section(
                model, features, asset_map, date, seq_len, device
            )
            if len(preds) < 2:
                # Kein Signal → Positionen halten, Equity tracken
                for pos in positions.values():
                    pos['hold_days'] = pos.get('hold_days', 0) + 1
                equity.append(cash + _position_value(positions, price_cache, date))
                equity_dates.append(date)
                continue

            # Rang-Lookup: asset → Rang (0 = bester)
            rank_of = {asset: i for i, asset in enumerate(preds.index)}

            # ── Stop-Loss & Rotation prüfen ───────────────────────────────
            to_close: list[tuple[str, str]] = []  # (asset, reason)

            for asset, pos in positions.items():
                current_price = _get_price(price_cache, asset, date)
                if current_price is None:
                    continue

                # a) Stop-Loss
                gross_loss = (current_price - pos['entry']) / pos['entry'] * pos['direction']
                if gross_loss < -stop_loss_pct:
                    to_close.append((asset, 'stop_loss'))
                    continue

                # b) Rotation: nur wenn Rang deutlich außerhalb Top-N
                asset_rank = rank_of.get(asset, len(preds))
                if asset_rank >= n_long + rotation_buffer:
                    # Prüfen ob ein besserer Kandidat existiert der noch nicht gehalten wird
                    held_after_close = set(positions) - {a for a, _ in to_close}
                    candidates = [a for a in preds.index[:n_long]
                                  if a not in held_after_close]
                    if candidates:
                        to_close.append((asset, 'rotation'))

            # ── Geschlossene Positionen abwickeln ────────────────────────
            for asset, reason in to_close:
                _close_position(asset, positions[asset], date, reason, regime)
                del positions[asset]

            # ── Freie Slots mit besten Kandidaten füllen ─────────────────
            free_slots = n_long - len(positions)
            if free_slots > 0:
                held = set(positions.keys())
                new_longs = [a for a in preds.index if a not in held][:free_slots]

                # Kapital gleichmäßig auf freie Slots verteilen
                # (bestehende Positionen bleiben in ihrer bisherigen Größe)
                if new_longs and cash > 0:
                    per_position = cash / len(new_longs)
                    for asset in new_longs:
                        price = _get_price(price_cache, asset, date)
                        if price is None or price <= 0:
                            continue
                        shares = (per_position * (1 - fees)) / price
                        positions[asset] = {
                            'shares':    shares,
                            'entry':     price,
                            'direction': 1,
                            'hold_days': 0,
                        }
                        cash -= per_position

            # Short-Seite (Variante B): Bottom-N leer verkaufen
            if long_short and n_long > 0:
                held = set(positions.keys())
                n_short      = min(n_long, len(preds) - n_long)
                short_assets = [a for a in preds.index[-n_short:] if a not in held]
                # Shorts schließen die nicht mehr Bottom-N sind
                for asset, pos in list(positions.items()):
                    if pos['direction'] == -1:
                        asset_rank = rank_of.get(asset, 0)
                        if asset_rank < len(preds) - n_short - rotation_buffer:
                            _close_position(asset, pos, date, 'rotation_short', regime)
                            del positions[asset]
                # Neue Shorts eröffnen
                held = set(positions.keys())
                short_free = n_short - sum(1 for p in positions.values() if p['direction'] == -1)
                if short_free > 0 and cash > 0:
                    new_shorts = [a for a in preds.index[-n_short:] if a not in held][:short_free]
                    per_position = cash / max(len(new_shorts), 1)
                    for asset in new_shorts:
                        price = _get_price(price_cache, asset, date)
                        if price is None or price <= 0:
                            continue
                        shares = (per_position * (1 - fees)) / price
                        positions[asset] = {
                            'shares':    shares,
                            'entry':     price,
                            'direction': -1,
                            'hold_days': 0,
                        }
                        cash += per_position * (1 - fees)

            # Hold-Days aktualisieren
            for pos in positions.values():
                pos['hold_days'] = pos.get('hold_days', 0) + 1

            # ── Equity berechnen ──────────────────────────────────────────
            portfolio_value = cash + _position_value(positions, price_cache, date)
            equity.append(portfolio_value)
            equity_dates.append(date)

    # Letzte Positionen auflösen (Ende des Backtests)
    if positions and equity_dates:
        last_date = equity_dates[-1]
        for asset, pos in list(positions.items()):
            _close_position(asset, pos, last_date, 'end_of_backtest', 'n/a')
        positions = {}

    # ── Metriken berechnen ────────────────────────────────────────────────────
    equity_arr = np.array(equity[1:], dtype=float)
    if len(equity_arr) == 0:
        return {}

    total_return = (equity_arr[-1] / init_cash - 1) * 100
    peak         = np.maximum.accumulate(equity_arr)
    max_dd       = ((equity_arr - peak) / (peak + 1e-9) * 100).min()
    daily_rets   = np.diff(equity_arr) / (equity_arr[:-1] + 1e-9)
    sharpe       = (daily_rets.mean() / (daily_rets.std() + 1e-9)) * np.sqrt(252)

    wins      = [t for t in trade_log if t['pnl_pct'] > 0]
    win_r     = len(wins) / len(trade_log) * 100 if trade_log else 0
    avg_hold  = sum(t.get('hold_days', 0) for t in trade_log) / len(trade_log) if trade_log else 0
    stop_hits = sum(1 for t in trade_log if t.get('exit_reason') == 'stop_loss')
    rotations = sum(1 for t in trade_log if 'rotation' in t.get('exit_reason', ''))

    logger.success("═" * 55)
    logger.success(f"BACKTEST: {strategy_name}")
    logger.success("═" * 55)
    logger.success(f"  Total Return  : {total_return:+.2f}%")
    logger.success(f"  Max Drawdown  : {max_dd:.2f}%")
    logger.success(f"  Sharpe Ratio  : {sharpe:.3f}")
    logger.success(f"  Trades        : {len(trade_log)}")
    logger.success(f"  Win Rate      : {win_r:.1f}%")
    logger.success(f"  Avg Hold Days : {avg_hold:.1f}")
    logger.success(f"  Stop-Loss Hits: {stop_hits}  |  Rotations: {rotations}")
    logger.success("═" * 55)

    return {
        'strategy':      strategy_name,
        'total_return':  round(total_return, 2),
        'max_drawdown':  round(max_dd, 2),
        'sharpe':        round(sharpe, 3),
        'n_trades':      len(trade_log),
        'win_rate':      round(win_r, 1),
        'avg_hold_days': round(avg_hold, 1),
        'stop_loss_hits': stop_hits,
        'rotations':     rotations,
        'equity':        equity_arr.tolist(),
        'equity_dates':  [str(d.date()) for d in equity_dates],
        'trade_log':     trade_log,
    }


# ── Benchmark-Berechnung ──────────────────────────────────────────────────────

def compute_benchmarks(
    price_cache:  dict[str, pd.Series],
    equity_dates: list,
    asset_map:    dict[str, int],
    spy_ticker:   str = "SPY",
    init_cash:    float = 10_000.0,
) -> dict:
    """
    Berechnet drei Benchmarks über denselben Zeitraum wie der Backtest:

    1. SPY Buy & Hold          — Standard-Marktbenchmark (S&P 500)
    2. EW Universe B&H         — Equal-weighted Buy & Hold aller 79 Assets
                                 (passives "alles halten" im eigenen Universum)
    3. EW Universe Rebalanced  — Monatlich auf equal-weight zurückgewichtet
                                 (fängt den "Rebalancing Premium" ein)

    Für einen Cross-Sectional-Strategievergleich ist EW Universe der fairste
    Benchmark: er zeigt ob unsere Aktienauswahl dem "einfach alles halten"
    überlegen ist, ohne Marktlevel-Effekte herauszurechnen.
    """
    if not equity_dates:
        return {}

    dates = pd.DatetimeIndex([pd.Timestamp(d) for d in equity_dates])
    start_date = dates[0]
    end_date   = dates[-1]

    # ── 1. SPY Buy & Hold ─────────────────────────────────────────────────
    spy_series  = price_cache.get(spy_ticker)
    spy_equity  = []
    spy_ret     = None
    if spy_series is not None:
        p0 = float(spy_series[spy_series.index <= start_date].iloc[-1]) if len(spy_series[spy_series.index <= start_date]) > 0 else None
        if p0:
            for d in dates:
                p = _get_price(price_cache, spy_ticker, d)
                spy_equity.append(init_cash * (p / p0) if p else (spy_equity[-1] if spy_equity else init_cash))
            spy_ret = (spy_equity[-1] / init_cash - 1) * 100

    # ── 2. EW Universe Buy & Hold (kein Rebalancing) ─────────────────────
    assets = list(asset_map.keys())
    # Startpreise aller Assets
    p0_map: dict[str, float] = {}
    for a in assets:
        s = price_cache.get(a)
        if s is not None:
            past = s[s.index <= start_date]
            if len(past) > 0:
                p0_map[a] = float(past.iloc[-1])

    ew_assets     = [a for a in assets if a in p0_map]
    ew_equity     = []
    ew_ret        = None
    if ew_assets:
        n = len(ew_assets)
        per_asset_cash = init_cash / n
        # Anteile am Starttag kaufen (keine Fees für Benchmark)
        shares_map = {a: per_asset_cash / p0_map[a] for a in ew_assets}
        for d in dates:
            total = sum(
                shares_map[a] * (_get_price(price_cache, a, d) or p0_map[a])
                for a in ew_assets
            )
            ew_equity.append(total)
        ew_ret = (ew_equity[-1] / init_cash - 1) * 100

    # ── 3. EW Universe monatlich rebalanciert ─────────────────────────────
    ewr_equity    = [init_cash]
    last_rebal    = start_date
    rebal_shares  = {a: (init_cash / len(ew_assets)) / p0_map[a] for a in ew_assets} if ew_assets else {}
    ewr_ret       = None

    if ew_assets:
        for d in dates:
            # Monatliches Rebalancing: Portfoliowert gleichmäßig neu aufteilen
            if (d.year * 12 + d.month) > (last_rebal.year * 12 + last_rebal.month):
                port_value = sum(
                    rebal_shares[a] * (_get_price(price_cache, a, d) or p0_map[a])
                    for a in ew_assets
                )
                per_asset = port_value / len(ew_assets)
                rebal_shares = {
                    a: per_asset / max(_get_price(price_cache, a, d) or p0_map[a], 1e-9)
                    for a in ew_assets
                }
                last_rebal = d
            total = sum(
                rebal_shares[a] * (_get_price(price_cache, a, d) or p0_map[a])
                for a in ew_assets
            )
            ewr_equity.append(total)
        ewr_equity = ewr_equity[1:]  # ersten Startwert entfernen
        ewr_ret = (ewr_equity[-1] / init_cash - 1) * 100

    # Sharpe-Berechnung für Benchmarks
    def _sharpe(eq_list):
        arr  = np.array(eq_list)
        rets = np.diff(arr) / (arr[:-1] + 1e-9)
        return float((rets.mean() / (rets.std() + 1e-9)) * np.sqrt(252))

    def _maxdd(eq_list):
        arr  = np.array(eq_list)
        peak = np.maximum.accumulate(arr)
        return float(((arr - peak) / (peak + 1e-9) * 100).min())

    result = {
        'dates':          [str(d.date()) for d in dates],
        'spy':            {'label': 'SPY Buy & Hold',           'equity': spy_equity,  'total_return': round(spy_ret, 2) if spy_ret is not None else None, 'sharpe': _sharpe(spy_equity) if spy_equity else None, 'max_drawdown': _maxdd(spy_equity) if spy_equity else None},
        'ew_bh':          {'label': 'EW Universe Buy & Hold',   'equity': ew_equity,   'total_return': round(ew_ret, 2) if ew_ret is not None else None,  'sharpe': _sharpe(ew_equity) if ew_equity else None,  'max_drawdown': _maxdd(ew_equity) if ew_equity else None},
        'ew_rebalanced':  {'label': 'EW Universe Rebalanciert', 'equity': ewr_equity,  'total_return': round(ewr_ret, 2) if ewr_ret is not None else None, 'sharpe': _sharpe(ewr_equity) if ewr_equity else None, 'max_drawdown': _maxdd(ewr_equity) if ewr_equity else None},
    }

    logger.info("─" * 55)
    logger.info("BENCHMARKS (gleicher Zeitraum wie Backtest)")
    for k, bm in result.items():
        if k == 'dates' or bm.get('total_return') is None:
            continue
        logger.info(f"  {bm['label']:30s}  Return: {bm['total_return']:+7.1f}%  Sharpe: {bm.get('sharpe', 0):.3f}  MaxDD: {bm.get('max_drawdown', 0):.1f}%")
    logger.info("─" * 55)

    return result


# ── Equity-Kurve plotten ──────────────────────────────────────────────────────

def plot_equity(
    result_a:   dict,
    result_b:   dict,
    benchmarks: Optional[dict] = None,
    save_path:  str = '/kaggle/working/equity_curve.png',
):
    """Vergleichs-Plot: Long-Only vs Long-Short vs Benchmarks."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

        # ── Equity Kurven ─────────────────────────────────────────────────
        dates_a = pd.to_datetime(result_a['equity_dates'])
        dates_b = pd.to_datetime(result_b['equity_dates'])
        eq_a    = np.array(result_a['equity'])
        eq_b    = np.array(result_b['equity'])

        ax1.plot(dates_a, eq_a / eq_a[0] * 100 - 100,
                 label=f"Long-Only  ({result_a['total_return']:+.1f}%)",
                 color='#2196F3', linewidth=2.5, zorder=5)
        ax1.plot(dates_b, eq_b / eq_b[0] * 100 - 100,
                 label=f"Long-Short ({result_b['total_return']:+.1f}%)",
                 color='#4CAF50', linewidth=2.5, zorder=5)

        # Benchmarks einzeichnen
        bm_colors = {'spy': '#FF9800', 'ew_bh': '#9C27B0', 'ew_rebalanced': '#F44336'}
        if benchmarks:
            bm_dates = pd.to_datetime(benchmarks.get('dates', []))
            for key, color in bm_colors.items():
                bm = benchmarks.get(key, {})
                if bm.get('equity') and bm.get('total_return') is not None:
                    eq_bm = np.array(bm['equity'])
                    ax1.plot(bm_dates[:len(eq_bm)], eq_bm / eq_bm[0] * 100 - 100,
                             label=f"{bm['label']} ({bm['total_return']:+.1f}%)",
                             color=color, linewidth=1.5, linestyle='--', alpha=0.85)

        ax1.axhline(0, color='gray', linewidth=0.8, alpha=0.5)
        ax1.set_title('Equity-Kurve vs Benchmarks', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Kumulativer Return (%)')
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        # ── Drawdown ──────────────────────────────────────────────────────
        def drawdown(eq):
            arr  = np.array(eq)
            peak = np.maximum.accumulate(arr)
            return (arr - peak) / (peak + 1e-9) * 100

        ax2.fill_between(dates_a, drawdown(eq_a), 0, alpha=0.35, color='#2196F3',
                         label=f"Long-Only  (MaxDD: {result_a['max_drawdown']:.1f}%)")
        ax2.fill_between(dates_b, drawdown(eq_b), 0, alpha=0.35, color='#4CAF50',
                         label=f"Long-Short (MaxDD: {result_b['max_drawdown']:.1f}%)")
        # SPY Drawdown als Referenz
        if benchmarks and benchmarks.get('spy', {}).get('equity'):
            eq_spy = np.array(benchmarks['spy']['equity'])
            ax2.plot(bm_dates[:len(eq_spy)], drawdown(eq_spy),
                     color='#FF9800', linewidth=1.2, linestyle='--', alpha=0.8,
                     label=f"SPY (MaxDD: {benchmarks['spy'].get('max_drawdown', 0):.1f}%)")
        ax2.set_title('Drawdown-Vergleich', fontsize=12)
        ax2.set_ylabel('Drawdown (%)')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        logger.success(f"Plot gespeichert: {save_path}")

    except Exception as e:
        logger.warning(f"Plot-Fehler: {e}")


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────

def _get_price(
    cache: dict[str, pd.Series],
    asset: str,
    date:  pd.Timestamp,
) -> Optional[float]:
    """
    Gibt den Close-Preis eines Assets an einem Tag aus dem Price-Cache zurück.

    Nutzt binäre Suche (searchsorted) — O(log n) statt O(n).
    Gibt None zurück wenn das Asset nicht im Cache ist.
    """
    series = cache.get(asset)
    if series is None or len(series) == 0:
        return None
    idx = series.index.searchsorted(date)
    idx = min(idx, len(series) - 1)
    return float(series.iloc[idx])


def _position_value(
    positions:   dict,
    price_cache: dict[str, pd.Series],
    date:        pd.Timestamp,
) -> float:
    """Berechnet den aktuellen Marktwert aller offenen Positionen aus dem Cache."""
    total = 0.0
    for asset, pos in positions.items():
        price = _get_price(price_cache, asset, date)
        if price:
            total += pos['shares'] * price * pos['direction']
    return total
