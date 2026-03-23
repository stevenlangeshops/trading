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
    features:    pd.DataFrame,
    targets:     pd.Series,
    fold_results: list[dict],
    asset_map:   dict[str, int],
    # Strategie-Parameter
    n_max:       int   = 5,     # Max Positionen bei Bull-Markt
    n_mid:       int   = 3,     # Positionen bei neutralem Markt
    n_min:       int   = 1,     # Min Positionen bei Bär-Markt
    long_short:  bool  = False, # True = Long-Short, False = Long-Only
    fees:        float = 0.001, # 0.1% Transaktionskosten
    init_cash:   float = 10_000.0,
    seq_len:     int   = 64,
    # Regime-Filter
    use_regime:  bool  = True,  # Adaptives N aktivieren
    spy_ticker:  str   = "SPY", # Referenz-Asset für Regime
) -> dict:
    """
    Führt den Cross-Sectional Backtest über alle Folds durch.

    Pro Handelstag:
      1. Regime bestimmen (adaptives N)
      2. Modell-Vorhersagen für alle Assets
      3. Top-N kaufen (Long), Bottom-N leerverkaufen (Short, nur Variante B)
      4. Nach horizon Tagen glattstellen
      5. Performance berechnen
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # SPY Preisreihe für Regime-Filter laden
    spy_prices = None
    if use_regime:
        spy_file = RAW_DIR / f"{spy_ticker}_1d.parquet"
        if spy_file.exists():
            spy_df    = pd.read_parquet(spy_file)
            spy_df.index = pd.to_datetime(spy_df.index)
            spy_prices = spy_df["close"]
        else:
            logger.warning(f"SPY nicht gefunden, Regime-Filter deaktiviert")
            use_regime = False

    all_dates = features.index.get_level_values('date').unique().sort_values()

    # Equity-Kurve und Trade-Log
    equity    = [init_cash]
    cash      = init_cash
    positions = {}   # asset → {'shares': float, 'entry': float, 'direction': +1/-1}
    trade_log = []
    equity_dates = []

    strategy_name = "Long-Short" if long_short else "Long-Only"
    logger.info(f"Backtest: {strategy_name}  n_max={n_max}  fees={fees:.3f}")

    for fold in fold_results:
        ckpt_path = fold['ckpt_path']
        if not Path(ckpt_path).exists():
            logger.warning(f"Checkpoint nicht gefunden: {ckpt_path}")
            continue

        model, ckpt_meta = load_fold_model(ckpt_path, device)
        val_start = pd.Timestamp(fold['val_start'])
        val_end   = pd.Timestamp(fold['val_end'])

        logger.info(f"  Fold {fold['fold_id']}: [{val_start.date()} → {val_end.date()}]")

        # Handelstage in diesem Val-Zeitraum
        fold_dates = all_dates[(all_dates >= val_start) & (all_dates <= val_end)]

        for date in fold_dates:
            # ── Regime bestimmen ──────────────────────────────────────────
            if use_regime and spy_prices is not None:
                regime = get_market_regime(spy_prices, date)
                n_long = adaptive_n(regime, n_max, n_mid, n_min)
                if regime == 'bear' and long_short:
                    n_long = 0   # Im Bärenmarkt: nur Shorts
            else:
                regime = 'neutral'
                n_long = n_mid

            # ── Vorhersagen generieren ────────────────────────────────────
            preds = predict_cross_section(
                model, features, asset_map, date, seq_len, device
            )
            if len(preds) < 2:
                equity.append(cash + _position_value(positions, features, date))
                equity_dates.append(date)
                continue

            # ── Bestehende Positionen auflösen ────────────────────────────
            for asset, pos in list(positions.items()):
                price = _get_price(features, asset, date)
                if price is None:
                    continue
                proceeds = pos['shares'] * price * (1 - fees) * pos['direction']
                cash    += proceeds
                pnl      = (price - pos['entry']) / pos['entry'] * pos['direction']
                trade_log.append({
                    'date':      str(date.date()),
                    'asset':     asset,
                    'direction': 'long' if pos['direction'] == 1 else 'short',
                    'pnl_pct':   round(pnl * 100, 3),
                    'regime':    regime,
                })
            positions = {}

            # ── Neue Positionen aufbauen ──────────────────────────────────
            n_actual   = min(n_long, len(preds))
            top_assets = preds.index[:n_actual].tolist()

            # Short-Seite: Bottom-N (nur Variante B)
            short_assets = []
            if long_short and n_actual > 0:
                n_short      = min(n_actual, len(preds) - n_actual)
                short_assets = preds.index[-n_short:].tolist()

            total_positions = len(top_assets) + len(short_assets)
            if total_positions == 0:
                equity.append(cash)
                equity_dates.append(date)
                continue

            per_position = cash / total_positions

            for asset in top_assets:
                price = _get_price(features, asset, date)
                if price is None or price <= 0:
                    continue
                shares = (per_position * (1 - fees)) / price
                positions[asset] = {'shares': shares, 'entry': price, 'direction': 1}
                cash -= per_position

            for asset in short_assets:
                price = _get_price(features, asset, date)
                if price is None or price <= 0:
                    continue
                # Short: wir erhalten den Erlös sofort
                shares = (per_position * (1 - fees)) / price
                positions[asset] = {'shares': shares, 'entry': price, 'direction': -1}
                cash += per_position * (1 - fees)  # Short-Erlös

            # ── Equity berechnen ──────────────────────────────────────────
            portfolio_value = cash + _position_value(positions, features, date)
            equity.append(portfolio_value)
            equity_dates.append(date)

    # Letzte Positionen auflösen
    if positions and equity_dates:
        last_date = equity_dates[-1]
        for asset, pos in positions.items():
            price = _get_price(features, asset, last_date)
            if price:
                cash += pos['shares'] * price * (1 - fees) * pos['direction']
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

    wins  = [t for t in trade_log if t['pnl_pct'] > 0]
    win_r = len(wins) / len(trade_log) * 100 if trade_log else 0

    logger.success("═" * 55)
    logger.success(f"BACKTEST: {strategy_name}")
    logger.success("═" * 55)
    logger.success(f"  Total Return  : {total_return:+.2f}%")
    logger.success(f"  Max Drawdown  : {max_dd:.2f}%")
    logger.success(f"  Sharpe Ratio  : {sharpe:.3f}")
    logger.success(f"  Trades        : {len(trade_log)}")
    logger.success(f"  Win Rate      : {win_r:.1f}%")
    logger.success("═" * 55)

    return {
        'strategy':     strategy_name,
        'total_return': round(total_return, 2),
        'max_drawdown': round(max_dd, 2),
        'sharpe':       round(sharpe, 3),
        'n_trades':     len(trade_log),
        'win_rate':     round(win_r, 1),
        'equity':       equity_arr.tolist(),
        'equity_dates': [str(d.date()) for d in equity_dates],
        'trade_log':    trade_log,
    }


# ── Equity-Kurve plotten ──────────────────────────────────────────────────────

def plot_equity(
    result_a: dict,
    result_b: dict,
    bh_return: Optional[float] = None,
    save_path: str = '/kaggle/working/equity_curve.png',
):
    """Vergleichs-Plot: Long-Only vs Long-Short vs Buy & Hold."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # ── Equity Kurven ─────────────────────────────────────────────────
        dates_a = pd.to_datetime(result_a['equity_dates'])
        dates_b = pd.to_datetime(result_b['equity_dates'])
        eq_a    = np.array(result_a['equity'])
        eq_b    = np.array(result_b['equity'])

        ax1.plot(dates_a, eq_a / eq_a[0] * 100 - 100,
                 label=f"Long-Only  (Return: {result_a['total_return']:+.1f}%)",
                 color='#2196F3', linewidth=2)
        ax1.plot(dates_b, eq_b / eq_b[0] * 100 - 100,
                 label=f"Long-Short (Return: {result_b['total_return']:+.1f}%)",
                 color='#4CAF50', linewidth=2)
        if bh_return is not None:
            ax1.axhline(bh_return, color='#FF9800', linestyle='--',
                        linewidth=1.5, label=f"SPY Buy & Hold ({bh_return:+.1f}%)")
        ax1.axhline(0, color='gray', linewidth=0.8, alpha=0.5)
        ax1.set_title('Equity-Kurve: Long-Only vs Long-Short', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Kumulativer Return (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        # ── Drawdown ──────────────────────────────────────────────────────
        def drawdown(eq):
            peak = np.maximum.accumulate(eq)
            return (eq - peak) / (peak + 1e-9) * 100

        ax2.fill_between(dates_a, drawdown(eq_a), 0, alpha=0.4, color='#2196F3',
                         label=f"Long-Only  (MaxDD: {result_a['max_drawdown']:.1f}%)")
        ax2.fill_between(dates_b, drawdown(eq_b), 0, alpha=0.4, color='#4CAF50',
                         label=f"Long-Short (MaxDD: {result_b['max_drawdown']:.1f}%)")
        ax2.set_title('Drawdown', fontsize=12)
        ax2.set_ylabel('Drawdown (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        logger.success(f"Plot gespeichert: {save_path}")

    except Exception as e:
        logger.warning(f"Plot-Fehler: {e}")


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────

def _get_price(features: pd.DataFrame, asset: str, date: pd.Timestamp) -> Optional[float]:
    """Gibt den Close-Preis eines Assets an einem Tag zurück (aus raw parquet)."""
    # Wir nutzen ret_1d aus Features um Preis zu rekonstruieren — nicht ideal.
    # Besser: direkt aus raw parquet lesen.
    try:
        fpath = RAW_DIR / f"{asset.replace('.', '_')}_1d.parquet"
        if not fpath.exists():
            return None
        df    = pd.read_parquet(fpath, columns=['close'])
        df.index = pd.to_datetime(df.index)
        idx   = df.index.searchsorted(date)
        idx   = min(idx, len(df) - 1)
        return float(df.iloc[idx]['close'])
    except Exception:
        return None


def _position_value(
    positions: dict,
    features:  pd.DataFrame,
    date:      pd.Timestamp,
) -> float:
    """Berechnet den aktuellen Marktwert aller offenen Positionen."""
    total = 0.0
    for asset, pos in positions.items():
        price = _get_price(features, asset, date)
        if price:
            total += pos['shares'] * price * pos['direction']
    return total
