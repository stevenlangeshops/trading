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


# ── ATR-Cache (Average True Range) ────────────────────────────────────────────

def compute_atr(
    high:   pd.Series,
    low:    pd.Series,
    close:  pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Berechnet Wilder's Average True Range (ATR) per pandas.

    True Range (TR) kombiniert drei Volatilitätskomponenten:
      1. Tages-Range:         H - L
      2. Overnight-Gap oben:  |H - prev_Close|
      3. Overnight-Gap unten: |L - prev_Close|

    Wilder's Glättung entspricht einem EWM mit alpha = 1/period,
    was äquivalent zu com = period - 1 ist.

    Robust gegen Lücken/Gaps: Gaps fließen als Overnight-Komponenten in
    die TR ein und erhöhen den ATR automatisch, ohne Sonderbehandlung.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).rename('hl'),
            (high - prev_close).abs().rename('hpc'),
            (low  - prev_close).abs().rename('lpc'),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def build_atr_cache(
    assets:  list[str],
    raw_dir: Path = RAW_DIR,
    period:  int  = 14,
) -> dict[str, pd.Series]:
    """
    Lädt OHLCV-Daten und berechnet den ATR einmalig für alle Assets.

    Wird einmalig vor dem Backtest aufgebaut und dann pro Tag/Asset per
    _get_atr() nachgeschlagen — kein wiederholtes Einlesen von Disk.

    Returns:
        Dict: asset-Name → pd.Series(index=DatetimeIndex, values=ATR)
              NaN-Werte in den ersten `period` Zeilen sind erwartet.
    """
    cache: dict[str, pd.Series] = {}
    missing_cols: list[str]     = []

    for asset in assets:
        fpath = raw_dir / f"{asset.replace('.', '_')}_1d.parquet"
        if not fpath.exists():
            continue
        try:
            df = pd.read_parquet(fpath, columns=['high', 'low', 'close'])
            df.index = pd.to_datetime(df.index)
            cache[asset] = compute_atr(df['high'], df['low'], df['close'], period)
        except Exception as exc:
            # Fehlende high/low-Spalten (z.B. bei ETF-Proxys) still überspringen
            missing_cols.append(asset)
            logger.debug(f"atr_cache: {asset} übersprungen ({exc})")

    if missing_cols:
        logger.warning(f"atr_cache: {len(missing_cols)} Assets ohne OHLCV-Spalten übersprungen")
    logger.info(f"atr_cache: {len(cache)}/{len(assets)} Assets geladen (period={period})")
    return cache


# ── RUN H1: SPY ATR Crash-Signal ─────────────────────────────────────────────

def compute_spy_crash_series(
    raw_dir:            Path  = RAW_DIR,
    spy_ticker:         str   = "SPY",
    atr_window:         int   = 14,
    lookback_high_days: int   = 10,
    crash_atr_mult:     float = 2.0,
) -> pd.Series:
    """
    # RUN H1: SPY ATR-basiertes Crash-Erkennungs-Signal.

    Crash-Bedingung (täglich, Zero-Lookahead):
        spy_crash_t = True  wenn:
            close_t  <=  max(close_{t-lookback_high_days} … close_{t-1})
                         − crash_atr_mult × ATR14_t

    Interpretation: SPY ist in den letzten ~10 Tagen mehr als 2× seine
    typische Tages-Schwankung (ATR14) unter das lokale Hoch gefallen.

    Zero-Lookahead-Garantie:
    - rolling_high nutzt .shift(1) → basiert auf Vortagen, nicht heute.
    - ATR14 ist kausal (True-Range-Mittel der letzten 14 Handelstage).

    Returns:
        pd.Series(bool, index=DatetimeIndex):  True = Crash-Signal aktiv
    """
    fpath = raw_dir / f"{spy_ticker.replace('.', '_')}_1d.parquet"
    if not fpath.exists():
        logger.warning(f"compute_spy_crash_series: {fpath} nicht gefunden — Crash-Schutz deaktiviert")
        return pd.Series(dtype=bool)

    try:
        df = pd.read_parquet(fpath).sort_index()
        df.columns = [c.lower() if isinstance(c, str) else str(c).lower() for c in df.columns]

        # True Range → ATR (benötigt high/low; Fallback via close-Diff)
        if 'high' in df.columns and 'low' in df.columns:
            close_prev = df['close'].shift(1)
            tr = pd.concat([
                df['high'] - df['low'],
                (df['high'] - close_prev).abs(),
                (df['low']  - close_prev).abs(),
            ], axis=1).max(axis=1)
        else:
            logger.warning("compute_spy_crash_series: high/low fehlen — nutze Pseudo-ATR via |Δclose|")
            tr = df['close'].diff().abs()

        atr = tr.rolling(atr_window, min_periods=atr_window).mean()

        # Lokales Hoch der letzten lookback_high_days Tage (EXKL. heute → shift(1))
        rolling_high = df['close'].rolling(lookback_high_days, min_periods=1).max().shift(1)

        # Crash-Signal
        crash = (df['close'] <= rolling_high - crash_atr_mult * atr).fillna(False)

        n_crash = int(crash.sum())
        logger.info(
            f"  SPY Crash-Signal: {n_crash} Crash-Tage von {len(crash)} "
            f"(ATR{atr_window}, lookback={lookback_high_days}d, mult={crash_atr_mult}×)"
        )
        return crash

    except Exception as exc:
        logger.warning(f"compute_spy_crash_series: Fehler '{exc}' — Crash-Schutz deaktiviert")
        return pd.Series(dtype=bool)


# ── RUN H2: 3-Tage Crash-Signal ──────────────────────────────────────────────

def compute_spy_crash_3d(
    spy_close:          pd.Series,
    crash_3d_return_thresh: float = 0.05,
    crash_3d_vol_mult:      float = 1.5,
    crash_3d_lookback_vol:  int   = 20,
) -> pd.Series:
    """
    # RUN H2: Schnelles 3-Tage-Crash-Signal auf SPY-Close.

    Triggert wenn BEIDE Bedingungen gleichzeitig gelten:
      1. SPY ist in 3 Tagen um >= crash_3d_return_thresh (z.B. 8%) gefallen:
            r3 = spy_close[t] / spy_close[t-3] - 1  <=  -crash_3d_return_thresh

      2. Der absolute 3-Tage-Return ist >= crash_3d_vol_mult × "normaler" 3-Tage-Move:
            |r3| >= crash_3d_vol_mult × (avg_daily_move_20 × 3)

         avg_daily_move_20 = rolling mean(|ln(close[t]/close[t-1])|, 20)

    Zero-Lookahead:
      - r3 nutzt close[t] und close[t-3] — alles bekannt am Handelsende.
      - avg_daily_move nutzt .shift(1) → basiert auf Vortagen.

    Returns:
        pd.Series(bool, index=DatetimeIndex):  True = 3-Tage-Crash aktiv
    """
    close = spy_close.sort_index().dropna()
    if len(close) < crash_3d_lookback_vol + 5:
        logger.warning("compute_spy_crash_3d: zu wenig Daten — leere Serie")
        return pd.Series(False, index=close.index)

    # 3-Tage-Return
    r3 = close / close.shift(3) - 1.0

    # Avg daily move (log-returns, shifted um Lookahead zu vermeiden)
    log_ret = np.log(close / close.shift(1)).abs()
    avg_daily_move = log_ret.shift(1).rolling(crash_3d_lookback_vol, min_periods=10).mean()

    # Normal expected 3-day move
    normal_3d_move = avg_daily_move * 3.0

    # Crash-Bedingung
    cond_return = r3 <= -crash_3d_return_thresh
    cond_vol    = r3.abs() >= crash_3d_vol_mult * normal_3d_move

    crash = (cond_return & cond_vol).fillna(False)

    n_crash = int(crash.sum())
    logger.info(
        f"  SPY Crash-3d: {n_crash} Tage von {len(crash)} "
        f"(thresh={crash_3d_return_thresh*100:.0f}%, mult={crash_3d_vol_mult}×, "
        f"vol_lookback={crash_3d_lookback_vol}d)"
    )
    return crash


def analyze_crash_3d_phases(
    crash_3d:   pd.Series,
    spy_close:  pd.Series,
) -> list[dict]:
    """
    # RUN H2: Gruppiert aufeinanderfolgende Crash-3d-Tage zu Phasen.

    Aufeinanderfolgende True-Tage (mit max. 2 Lücken-Tagen) werden zu einer
    Phase zusammengefasst. Für jede Phase: Start, Ende, Dauer, SPY-Drawdown.
    """
    if crash_3d.sum() == 0:
        return []

    # Timezone normalisieren (alle tz-naive machen)
    crash_norm = crash_3d.copy()
    if crash_norm.index.tz is not None:
        crash_norm.index = crash_norm.index.tz_localize(None)
    crash_dates = crash_norm[crash_norm].index.sort_values()

    spy = spy_close.sort_index().copy()
    if spy.index.tz is not None:
        spy.index = spy.index.tz_localize(None)

    phases: list[dict] = []
    phase_start = crash_dates[0]
    phase_end   = crash_dates[0]

    for d in crash_dates[1:]:
        gap = (d - phase_end).days
        if gap <= 5:  # max 5 Kalendertage Lücke → selbe Phase
            phase_end = d
        else:
            # Phase abschließen
            mask = (spy.index >= phase_start) & (spy.index <= phase_end)
            spy_phase = spy[mask]
            if len(spy_phase) > 1:
                peak = spy_phase.cummax()
                dd = ((spy_phase - peak) / (peak + 1e-9)).min() * 100
            else:
                dd = 0.0
            phases.append({
                'start':     str(phase_start.date()),
                'end':       str(phase_end.date()),
                'n_days':    int(((phase_end - phase_start).days) + 1),
                'crash_days': int(crash_norm.loc[phase_start:phase_end].sum()),
                'spy_dd_pct': round(float(dd), 2),
            })
            phase_start = d
            phase_end   = d

    # Letzte Phase
    mask = (spy.index >= phase_start) & (spy.index <= phase_end)
    spy_phase = spy[mask]
    if len(spy_phase) > 1:
        peak = spy_phase.cummax()
        dd = ((spy_phase - peak) / (peak + 1e-9)).min() * 100
    else:
        dd = 0.0
    phases.append({
        'start':     str(phase_start.date()),
        'end':       str(phase_end.date()),
        'n_days':    int(((phase_end - phase_start).days) + 1),
        'crash_days': int(crash_norm.loc[phase_start:phase_end].sum()),
        'spy_dd_pct': round(float(dd), 2),
    })

    return phases


def _get_spy_crash(crash_series: "pd.Series | None", date: pd.Timestamp) -> bool:
    """# RUN H1: Gibt das SPY Crash-Flag für einen bestimmten Handelstag zurück."""
    if crash_series is None or len(crash_series) == 0:
        return False
    try:
        aligned = _align_date_tz(date, crash_series.index)
        if aligned in crash_series.index:
            return bool(crash_series.loc[aligned])
    except Exception:
        pass
    return False


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
    spy_prices:  pd.Series,
    date:        pd.Timestamp,
    window_fast: int = 50,
    window_slow: int = 200,
) -> str:
    """
    Bestimmt das Marktregime anhand von SPY SMAs.

        'bull'    — SPY > SMA50 > SMA200  → volle Position (n_max)
        'neutral' — SPY > SMA200 aber < SMA50 → reduzierte Position (n_mid)
        'bear'    — SPY < SMA200  → Long-Only: keine neuen Longs (n_min)

    Bewusst NUR SMA-basiert: ein 5–10% Rücksetzer vom kurzfristigen Hoch
    ist normales Marktgeräusch (passiert 4–6× pro Jahr) und darf nicht
    die Positionsanzahl reduzieren — das würde bull-market Alpha vernichten.
    """
    # Timezone normalisieren (price_cache ist UTC-aware, date ggf. tz-naiv)
    spy_norm = spy_prices.copy()
    spy_norm.index = spy_norm.index.tz_localize(None) if spy_norm.index.tz is not None else spy_norm.index
    date_norm = date.tz_localize(None) if getattr(date, 'tzinfo', None) is not None else date

    past = spy_norm[spy_norm.index <= date_norm]
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
    n_max:            int   = 7,      # Max Positionen bei Bull-Markt (Run F: 7 statt 5 → nat. Diversifikation)
    n_mid:            int   = 3,      # Positionen bei neutralem Markt
    n_min:            int   = 1,      # Min Positionen bei Bär-Markt
    long_short:       bool  = False,  # True = Long-Short, False = Long-Only
    fees:             float = 0.001,  # 0.1% Transaktionskosten pro Seite
    init_cash:        float = 10_000.0,
    seq_len:          int   = 64,
    # ATR-basierter Trailing Stop
    # Run G: DEAKTIVIERT — empirische Analyse über alle Runs zeigt:
    #   ATR-Stop zerstört konsequent Wert (Run E: -521%, Run F: -704%).
    #   Das Rotations-Signal des LSTM ist überlegen (Run F: +956%, win=69%).
    #   ATR-Stops feuern auf Positionen, die der Ranker noch als Top-N hält —
    #   Rotation exitiert früher und profitabler (+5.5% avg vs -9.9% ATR).
    #   Portfolioschutz übernimmt DD-Control + Regime-Filter + Hard-Stop.
    use_atr_trailing:  bool  = False,  # Run G: deaktiviert (empirisch belegt)
    atr_period:        int   = 14,     # ATR-Periode (nur relevant wenn True)
    atr_k:             float = 3.5,    # Multiplikator (nur relevant wenn True)
    atr_min_hold_days: int   = 3,      # Min. Haltetage vor Stop-Aktivierung
    # Hard Stop: maximaler Verlust vom Einstiegskurs (nur Gap-Down-Schutz)
    # Bewusst großzügig (25%): soll nur echte Crash-Gaps abfangen, nicht
    # normale Rücksetzer, die der ATR-Trailing-Stop besser handhabt.
    # Analyse zeigt: 15% Hard-Stop verursachte -545% PnL-Drain durch zu
    # frühe Exits von Positionen, die via ATR/Rotation besser geendet hätten.
    hard_stop_pct:     float = 0.25,   # 25% Hard-Stop — nur Gap-Schutz
    rotation_buffer:  int   = 3,      # Run F: 3 statt 2 — weniger unnötige Rotationen
    # Regime-Filter
    use_regime:       bool  = True,   # Adaptives N aktivieren
    spy_ticker:       str   = "SPY",  # Referenz-Asset für Regime
    # Korrelations-Cap (Run E): 1.0 = deaktiviert (Run F Standard)
    # Nur aktivieren wenn < 1.0 — dann teuer (O(n²) rolling-corr pro Tag).
    corr_cap:         float = 1.0,    # Run F: deaktiviert (0.80 → 1.0)
    corr_window:      int   = 60,     # Lookback-Fenster für Korrelationsberechnung (Handelstage)
    # Volatilitäts-basiertes Sizing (Risk Parity): Run F → deaktiviert, Equal-Weight
    use_vol_sizing:   bool  = False,  # Run F: False — Risk-Parity übersteuerte Modellsignal
    risk_per_trade:   float = 0.01,   # 1% Portfoliowert Risiko pro Position (bei Stop-Auslösung)
    # ── Portfolio-Drawdown-Control (Run F, neu) ────────────────────────────────
    # Zweistufiger DD-Schutz: reduziert n_max progressiv wenn das Portfolio
    # unter Wasser gerät, ohne Positionen zwangsweise zu schließen.
    #
    # Stufe 1 (dd_threshold_1): n_max / dd_reduction_factor  → z.B. 7 → 3
    # Stufe 2 (dd_threshold_2): n_max = n_min (1), n_mid = 0 → Defensivmodus
    # Erholung: DD muss dd_recovery_margin über dem Triggerlevel liegen bevor
    #           Normalmodus zurückkehrt (verhindert ständiges Hin- und Her).
    # ── Portfolio-Drawdown-Control ────────────────────────────────────────────
    # Run G: use_dd_control=False — Baseline-Messung der reinen Rotation+Hard-Stop-Edge.
    # DD wird weiterhin getrackt (für Reporting), beeinflusst aber KEINE Positionsanzahl.
    # Aktiviere use_dd_control=True für zukünftige Runs um Portfolioschutz zuzuschalten.
    use_dd_control:      bool  = False,  # Run G: OFF (Baseline, keine N-Anpassung)
    dd_threshold_1:      float = 0.25,   # 25% DD → Stufe 1 (nur aktiv wenn use_dd_control=True)
    dd_threshold_2:      float = 0.40,   # 40% DD → Stufe 2 (nur aktiv wenn use_dd_control=True)
    dd_reduction_factor: int   = 2,      # n_max // dd_reduction_factor bei Stufe 1
    dd_recovery_margin:  float = 0.05,   # DD muss 5% besser sein vor Erholung
    # ── RUN H1: SPY-ATR Crash-Schutz ─────────────────────────────────────────
    # Einfacher kombinierter Crash-Schutz: schaltet in "Halbgas-Modus" wenn
    # GLEICHZEITIG gilt:
    #   (a) SPY-ATR-Signal zeigt Crash  (spy_crash == True)
    #   (b) Portfolio-Drawdown >= dd_crash_threshold
    # Im Halbgas-Modus: n_max und n_mid werden halbiert (weniger Positionen,
    # mehr Cash-Quote). Ausstieg erst wenn BEIDE Bedingungen nicht mehr gelten
    # + dd_crash_recovery Puffer.
    # Rotation, Ranking, Modell und Hard-Stop bleiben vollständig unverändert.
    use_crash_protection:   bool  = True,    # Run H1: kombinierter Crash-Schutz
    spy_atr_window:         int   = 14,      # ATR-Periode für SPY
    spy_lookback_high_days: int   = 10,      # Lookback für lokales SPY-Hoch
    spy_crash_atr_mult:     float = 2.0,     # Multiplikator: Crash wenn < high - mult*ATR
    dd_crash_threshold:     float = 0.25,    # Portfolio-DD-Schwelle (25%) für Crash-Modus
    dd_crash_recovery:      float = 0.10,    # Erholungs-Puffer (10%) für Modus-Rückkehr
    spy_crash_series:       Optional[pd.Series] = None,  # Pre-computed (None = intern)
    # ── Mindest-Expected-Return Filter ─────────────────────────────────────
    # Geht nur in Aktien, deren Modell-Output (pred_return) hoch genug ist.
    # In Phasen, in denen selbst die Top-gerankte Aktie einen negativen oder
    # sehr niedrigen Return erwartet, werden KEINE neuen Positionen eröffnet.
    # Bestehende Positionen werden weiter nach Rotation/Stop gemanagt.
    use_min_expected_return_filter: bool  = True,
    min_expected_return_top:       float = 0.0,   # Schwelle für Top-1 pred_return
    # Optional: stärkere Bedingung auf den Durchschnitt der Top-N
    use_avg_topN_filter:           bool  = False,
    min_avg_expected_return_topN:  float = 0.0,   # nur wenn use_avg_topN_filter=True
    # Bestehende Positionen mit stark negativem pred_return als Rotations-
    # Kandidat behandeln (verkaufen), auch wenn sie noch im Top-N-Rang sind.
    existing_pos_exit_margin:      float = 0.02,  # Marge unter threshold → Exit-Kandidat
    # ── Signalstärke-Filter ────────────────────────────────────────────────
    # Misst, wie klar das Modell heute zwischen Gewinnern und Verlierern
    # unterscheidet. Wenn alle Scores ähnlich → Signal ist "weak" → weniger
    # oder gar nicht handeln. Unabhängig vom absoluten Score-Niveau.
    use_signal_strength_filter: bool  = True,
    # Variante A: Spread zwischen Top-1 Score und Median aller Scores
    use_score_spread_filter:   bool  = True,
    min_score_spread_top1_med: float = 0.002,  # ~0.2 Pp — anpassbar per Backtest
    # Variante B: Standardabweichung der Scores im gesamten Universum
    use_score_std_filter:      bool  = False,
    min_score_std_universe:    float = 0.003,  # ~0.3 Pp
    # Aktion wenn Signal schwach:
    #   "no_new"   : keine neuen Positionen, bestehende normal managen
    #   "reduce_n" : n_long um signal_filter_n_factor reduzieren
    signal_filter_action:      str   = "no_new",
    signal_filter_n_factor:    float = 0.5,    # N * 0.5 bei "reduce_n"
    # Optionale Pre-built Caches (werden intern aufgebaut wenn None)
    price_cache:      Optional[dict] = None,
    atr_cache:        Optional[dict] = None,
) -> dict:
    """
    Führt den Cross-Sectional Backtest über alle Folds durch.

    Täglich:
      1. Regime bestimmen (adaptives N via SMA50/SMA200)
      2. Frische Modell-Vorhersagen für alle Assets
      3. ATR-Trailing-Stop aller Long-Positionen nach oben nachziehen
      4. Jede gehaltene Position prüfen:
         a. Hard-Stop (25%, nur Gap-Schutz) → ATR-Trailing → Fixed-Stop
         b. Rotation: exit wenn Rang > n_long + rotation_buffer
      5. Freie Slots mit Top-Kandidaten füllen — dabei:
         - Run G Baseline: Pure Model-Ranking, Equal-Weight, kein ATR, kein Korr.-Filter
         - Optional: corr_cap < 1.0 aktiviert Rolling-Korrelations-Filter
      6. DD-Tracking (immer) für Reporting-Zwecke.
         DD-Control (use_dd_control=True): passt N dynamisch an — in Run G deaktiviert.
         Ziel: isolierte Messung der reinen Rotation + Hard-Stop Edge.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Caches einmalig aufbauen falls nicht von außen übergeben
    all_assets_for_cache = list(asset_map.keys())
    if use_regime and spy_ticker not in all_assets_for_cache:
        all_assets_for_cache.append(spy_ticker)

    if price_cache is None:
        price_cache = build_price_cache(all_assets_for_cache)

    if atr_cache is None and use_atr_trailing:
        atr_cache = build_atr_cache(
            list(asset_map.keys()),  # SPY braucht keinen ATR-Cache
            period=atr_period,
        )

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

    # ── DD-Control Tracking ───────────────────────────────────────────────────
    max_equity_so_far = init_cash   # laufendes Maximum für DD-Berechnung
    dd_mode           = 0           # 0=normal, 1=reduziert, 2=defensiv
    dd_mode_days      = 0           # Tage im aktuellen Modus > 0
    dd_events: list[dict] = []      # Log: {date, mode, dd_pct, action}

    # ── RUN H1: Crash-Schutz Tracking ────────────────────────────────────────
    # spy_crash_series intern berechnen wenn nicht übergeben
    _crash_series: Optional[pd.Series] = spy_crash_series
    if use_crash_protection and _crash_series is None:
        _crash_series = compute_spy_crash_series(
            raw_dir            = RAW_DIR,
            spy_ticker         = spy_ticker,
            atr_window         = spy_atr_window,
            lookback_high_days = spy_lookback_high_days,
            crash_atr_mult     = spy_crash_atr_mult,
        )
        if len(_crash_series) == 0:
            logger.warning("  Crash-Schutz deaktiviert (keine SPY-Daten)")
            use_crash_protection = False

    equity_peak_h1      = init_cash   # laufendes Equity-Peak für H1-DD
    crash_mode_active   = False
    crash_mode_days     = 0
    crash_phases:  list[dict] = []    # abgeschlossene Crash-Phasen
    _phase_start:  Optional[pd.Timestamp] = None
    _phase_max_dd: float = 0.0

    # ── Expected-Return-Filter Tracking ──────────────────────────────────
    filter_active_days   = 0           # Tage an denen allow_new_entries=False
    filter_active_dates: list[str] = []
    top1_preds:  list[float] = []      # best pred_return pro Tag
    topN_preds:  list[float] = []      # avg top-N pred_return pro Tag
    # Equity-Tracking getrennt nach Filter-Zustand (für spätere Auswertung)
    equity_filter_on:  list[float] = []   # daily returns an Filter-aktiv Tagen
    equity_filter_off: list[float] = []   # daily returns an normalen Tagen

    # ── Signalstärke-Filter Tracking ──────────────────────────────────────
    sig_weak_days          = 0
    sig_strong_days        = 0
    sig_spreads:    list[float] = []
    sig_stds:       list[float] = []
    sig_weak_spreads: list[float] = []
    sig_strong_spreads: list[float] = []
    equity_sig_weak:  list[float] = []
    equity_sig_strong: list[float] = []
    daily_signals: list[dict] = []     # Rohdaten pro Handelstag

    strategy_name = "Long-Short" if long_short else "Long-Only"
    stop_desc = (
        f"ATR-Trailing(period={atr_period}, k={atr_k})"
        if use_atr_trailing
        else f"Hard-Stop-only({hard_stop_pct*100:.0f}%)"
    )
    dd_desc = (
        f"dd_ctrl=ON(th1={dd_threshold_1*100:.0f}%/th2={dd_threshold_2*100:.0f}%)"
        if use_dd_control else "dd_ctrl=OFF"
    )
    logger.info(
        f"Backtest: {strategy_name}  n_max={n_max}(mid={n_mid},min={n_min})  "
        f"fees={fees:.3f}  rotation_buffer={rotation_buffer}"
    )
    logger.info(
        f"  Exits: atr={'ON' if use_atr_trailing else 'OFF'}  "
        f"hard_stop={hard_stop_pct*100:.0f}%  fix_stop=OFF  "
        f"[Run G Baseline]"
    )
    filter_desc = (
        f"min_ret_filter={'ON' if use_min_expected_return_filter else 'OFF'}"
        f"(top1>={min_expected_return_top*100:.1f}%)"
        + (f" avg_topN>={min_avg_expected_return_topN*100:.1f}%" if use_avg_topN_filter else "")
    )
    logger.info(
        f"  Portfolio: corr_cap={'OFF' if corr_cap >= 1.0 else f'{corr_cap:.2f}'}  "
        f"vol_sizing={'OFF' if not use_vol_sizing else f'ON(risk={risk_per_trade:.2%})'}  "
        f"{dd_desc}"
    )
    sig_desc = "OFF"
    if use_signal_strength_filter:
        parts = []
        if use_score_spread_filter:
            parts.append(f"spread>={min_score_spread_top1_med}")
        if use_score_std_filter:
            parts.append(f"std>={min_score_std_universe}")
        sig_desc = f"ON({'+'.join(parts)}) action={signal_filter_action}"
    logger.info(f"  Filter: {filter_desc}")
    logger.info(f"  Signal: {sig_desc}")

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
            # ── Portfolio-DD immer tracken (für Reporting) ────────────────
            # Berechnung läuft unabhängig von use_dd_control, damit das
            # Reporting immer zeigt wie tief der DD tatsächlich war.
            current_pv = cash + _position_value(positions, price_cache, date)
            max_equity_so_far = max(max_equity_so_far, current_pv)
            current_dd = (current_pv - max_equity_so_far) / (max_equity_so_far + 1e-9)

            if use_dd_control:
                # ── DD-Control: N dynamisch anpassen ──────────────────────
                prev_dd_mode = dd_mode
                if current_dd < -dd_threshold_2:
                    dd_mode = 2
                elif current_dd < -dd_threshold_1:
                    dd_mode = 1
                else:
                    # Erholung: erst zurück wenn DD über (threshold - recovery_margin)
                    if dd_mode == 2 and current_dd > -(dd_threshold_2 - dd_recovery_margin):
                        dd_mode = 1
                    if dd_mode == 1 and current_dd > -(dd_threshold_1 - dd_recovery_margin):
                        dd_mode = 0

                if dd_mode != prev_dd_mode:
                    action = {0: 'NORMAL', 1: 'REDUZIERT', 2: 'DEFENSIV'}[dd_mode]
                    logger.warning(
                        f"  DD-Control [{date.date()}]: Modus → {action}  "
                        f"DD={current_dd*100:.1f}%  equity={current_pv:,.0f}"
                    )
                    dd_events.append({
                        'date': str(date.date()),
                        'mode': dd_mode,
                        'dd_pct': round(current_dd * 100, 2),
                        'action': action,
                    })

                if dd_mode > 0:
                    dd_mode_days += 1

                # Effektives N für diesen Tag (basierend auf DD-Modus)
                if dd_mode == 2:
                    eff_n_max = n_min
                    eff_n_mid = 0
                    eff_n_min = 0
                elif dd_mode == 1:
                    eff_n_max = max(1, n_max // dd_reduction_factor)
                    eff_n_mid = max(1, n_mid // dd_reduction_factor)
                    eff_n_min = n_min
                else:
                    eff_n_max, eff_n_mid, eff_n_min = n_max, n_mid, n_min
            else:
                # DD-Control deaktiviert (Run G/H1 Baseline):
                # N bleibt immer bei den konfigurierten Maximalwerten.
                eff_n_max, eff_n_mid, eff_n_min = n_max, n_mid, n_min

            # ── RUN H1: Crash-Schutz State-Machine ───────────────────────
            # Läuft NACH dem DD-Control, kann dessen eff_n weiter reduzieren.
            # Bedingung Eintritt:  spy_crash=True  UND  portfolio_dd >= threshold
            # Bedingung Austritt:  spy_crash=False UND  portfolio_dd < (threshold - recovery)
            if use_crash_protection:
                equity_peak_h1 = max(equity_peak_h1, current_pv)
                equity_dd_h1 = (current_pv - equity_peak_h1) / (equity_peak_h1 + 1e-9)
                spy_crash_today = _get_spy_crash(_crash_series, date)

                if not crash_mode_active:
                    # Eintritt: BEIDE Bedingungen müssen gleichzeitig gelten
                    if spy_crash_today and equity_dd_h1 <= -dd_crash_threshold:
                        crash_mode_active = True
                        _phase_start      = date
                        _phase_max_dd     = equity_dd_h1
                        logger.warning(
                            f"  HALBGAS-START [{date.date()}]: "
                            f"SPY-Crash=True  DD={equity_dd_h1*100:.1f}%  "
                            f"equity={current_pv:,.0f}"
                        )
                else:
                    # Im Crash-Modus: laufenden Worst-DD tracken
                    _phase_max_dd = min(_phase_max_dd, equity_dd_h1)
                    # Austritt: BEIDE müssen nicht mehr gelten
                    recovery_lvl = -(dd_crash_threshold - dd_crash_recovery)
                    if not spy_crash_today and equity_dd_h1 > recovery_lvl:
                        crash_mode_active = False
                        phase_days = crash_mode_days - sum(p['days'] for p in crash_phases)
                        crash_phases.append({
                            'start':   str(_phase_start.date()),
                            'end':     str(date.date()),
                            'days':    phase_days,
                            'max_dd':  round(_phase_max_dd * 100, 2),
                        })
                        logger.warning(
                            f"  HALBGAS-ENDE  [{date.date()}]: "
                            f"SPY-Crash={spy_crash_today}  DD={equity_dd_h1*100:.1f}%  "
                            f"Phase-Max-DD={_phase_max_dd*100:.1f}%"
                        )

                if crash_mode_active:
                    crash_mode_days += 1
                    # Halbgas: n_max und n_mid halbieren (floor), n_min unverändert
                    eff_n_max = max(1, eff_n_max // 2)
                    eff_n_mid = max(1, eff_n_mid // 2)
                    # eff_n_min bleibt

            # ── Regime bestimmen ──────────────────────────────────────────
            if use_regime and spy_prices is not None:
                regime = get_market_regime(spy_prices, date)
                n_long = adaptive_n(regime, eff_n_max, eff_n_mid, eff_n_min)
                if regime == 'bear' and long_short:
                    n_long = 0
            else:
                regime = 'neutral'
                n_long = eff_n_mid

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

            # ── Expected-Return-Filter ─────────────────────────────────────
            best_pred = float(preds.iloc[0]) if len(preds) > 0 else 0.0
            avg_topN  = float(preds.iloc[:max(n_long, 1)].mean()) if len(preds) > 0 else 0.0
            top1_preds.append(best_pred)
            topN_preds.append(avg_topN)

            allow_new_entries = True
            if use_min_expected_return_filter:
                if best_pred < min_expected_return_top:
                    allow_new_entries = False

            if allow_new_entries and use_avg_topN_filter:
                if avg_topN < min_avg_expected_return_topN:
                    allow_new_entries = False

            if not allow_new_entries:
                filter_active_days += 1
                filter_active_dates.append(str(date.date()))

            # ── Signalstärke-Filter ────────────────────────────────────────
            signal_weak_today = False
            scores_arr = preds.values
            score_top1 = float(scores_arr[0])  # preds ist absteigend sortiert
            score_med  = float(np.median(scores_arr))
            score_spread = score_top1 - score_med
            score_std    = float(np.std(scores_arr, ddof=0))

            sig_spreads.append(score_spread)
            sig_stds.append(score_std)

            if use_signal_strength_filter:
                if use_score_spread_filter and score_spread < min_score_spread_top1_med:
                    signal_weak_today = True
                if use_score_std_filter and score_std < min_score_std_universe:
                    signal_weak_today = True

            if signal_weak_today:
                sig_weak_days += 1
                sig_weak_spreads.append(score_spread)
                if signal_filter_action == "no_new":
                    allow_new_entries = False
                elif signal_filter_action == "reduce_n":
                    n_long = max(1, int(n_long * signal_filter_n_factor))
            else:
                sig_strong_days += 1
                sig_strong_spreads.append(score_spread)

            # Rohdaten für diesen Tag
            daily_signals.append({
                'date':           str(date.date()),
                'regime':         regime if use_regime else 'n/a',
                'n_long':         n_long,
                'n_assets':       len(preds),
                'n_positions':    len(positions),
                'score_top1':     round(score_top1, 6),
                'score_med':      round(score_med, 6),
                'score_spread':   round(score_spread, 6),
                'score_std':      round(score_std, 6),
                'score_top5_avg': round(float(preds.iloc[:5].mean()), 6) if len(preds) >= 5 else None,
                'score_bot5_avg': round(float(preds.iloc[-5:].mean()), 6) if len(preds) >= 5 else None,
                'signal_weak':    signal_weak_today,
                'allow_new':      allow_new_entries,
                'best_pred':      round(best_pred, 6),
                'equity':         round(cash + _position_value(positions, price_cache, date), 2),
            })

            # ── ATR-Trailing-Stop täglich nachziehen (nur Long, nur nach oben) ──
            # Formel: stop_candidate = close - k * ATR
            #         new_stop       = max(old_stop, stop_candidate)
            # So wird der Stop bei steigenden Kursen mitgezogen, nie gesenkt.
            if use_atr_trailing and atr_cache:
                for asset, pos in positions.items():
                    if pos['direction'] != 1:
                        continue
                    current_price = _get_price(price_cache, asset, date)
                    atr_val       = _get_atr(atr_cache, asset, date)
                    if current_price is None or atr_val is None:
                        continue
                    stop_candidate = current_price - atr_k * atr_val
                    # Nur nach oben nachziehen — niemals den Stop senken
                    pos['trailing_stop'] = max(
                        pos.get('trailing_stop', stop_candidate),
                        stop_candidate,
                    )

            # ── Stop & Rotation prüfen ────────────────────────────────────
            to_close: list[tuple[str, str]] = []  # (asset, reason)

            for asset, pos in positions.items():
                current_price = _get_price(price_cache, asset, date)
                if current_price is None:
                    continue

                # Stop-Loss — zwei Ebenen (Run F: kein fester 5%-Stop mehr):
                #
                #    1. Hard-Stop (25%): Nur für extreme Gap-Downs (z.B. Insolvenz über Nacht).
                #       Greift immer, übersteuert ATR. Bewusst großzügig (25%) damit normale
                #       Rücksetzer den ATR-Trailing-Stop durchlaufen.
                #
                #    2. ATR-Trailing: volatilitätsangepasster nachgezogener Stop.
                #       Aktiv nach atr_min_hold_days. Zieht nur nach oben nach.
                #       Kein fester 5%-Stop mehr — das hat in Run E -660% PnL-Drain verursacht.
                gross_loss = (current_price - pos['entry']) / pos['entry'] * pos['direction']
                if gross_loss < -hard_stop_pct:
                    to_close.append((asset, 'hard_stop'))
                    continue

                hold_days_so_far = pos.get('hold_days', 0)
                atr_active = hold_days_so_far >= atr_min_hold_days
                if use_atr_trailing and pos.get('trailing_stop') is not None and atr_active:
                    if current_price < pos['trailing_stop']:
                        to_close.append((asset, 'atr_trailing_stop'))
                        continue

                # b) Expected-Return-Filter: bestehende Position mit stark negativem
                #    pred_return als Exit-Kandidat markieren (schrittweise Exposure-Reduktion)
                if use_min_expected_return_filter:
                    asset_pred = float(preds.get(asset, 0.0))
                    exit_thresh = min_expected_return_top - existing_pos_exit_margin
                    if asset_pred < exit_thresh:
                        to_close.append((asset, 'low_pred_return'))
                        continue

                # c) Rotation: nur wenn Rang deutlich außerhalb Top-N
                asset_rank = rank_of.get(asset, len(preds))
                if asset_rank >= n_long + rotation_buffer:
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
            if free_slots > 0 and allow_new_entries:
                held = set(positions.keys())

                # Run F: Pure Model-Ranking ohne Korrelations-Filter (corr_cap=1.0 Standard).
                # Run E zeigte: Korrelations-Filter zwang niedrig-gerankte Kandidaten in
                # das Portfolio → 750 von 994 Rotationen nach ≤3 Tagen → mehr Gebühren.
                #
                # Optional (corr_cap < 1.0): Aktiviert O(n²) Rolling-Korr-Check.
                if corr_cap < 1.0:
                    raw_candidates = [a for a in preds.index if a not in held][: free_slots * 5]
                    accepted: list[str] = []
                    for asset in raw_candidates:
                        if len(accepted) >= free_slots:
                            break
                        too_correlated = any(
                            _rolling_return_corr(
                                price_cache, asset, other, date, corr_window
                            ) > corr_cap
                            for other in list(held) + accepted
                        )
                        if not too_correlated:
                            accepted.append(asset)
                    new_longs = accepted
                else:
                    # Standard (Run F): Top-gerankte Assets direkt
                    new_longs = [a for a in preds.index if a not in held][:free_slots]

                if new_longs and cash > 0:
                    # Run E: Volatilitäts-basiertes Sizing (Risk Parity)
                    # ALT: per_position = cash / len(new_longs)  → Equal-Weight
                    # NEU: Positionsgröße so wählen dass Stop-Auslösung genau
                    #      risk_per_trade * Portfolio-Value kostet.
                    #
                    #   Stop-Distanz (in Preis-Einheiten) = atr_k * ATR
                    #   → stop_dist_pct = (atr_k * ATR) / price
                    #   → position_value = (risk_per_trade * equity) / stop_dist_pct
                    #
                    # Proportionale Skalierung wenn Summe > verfügbares Cash.
                    # Fallback auf Equal-Weight wenn ATR nicht verfügbar oder
                    # use_vol_sizing=False.
                    if use_vol_sizing:
                        total_equity = cash + _position_value(positions, price_cache, date)
                        raw_sizes: dict[str, float] = {}

                        for asset in new_longs:
                            price   = _get_price(price_cache, asset, date)
                            atr_val = _get_atr(atr_cache, asset, date) if (use_atr_trailing and atr_cache) else None
                            if price is None or price <= 0:
                                continue
                            if atr_val is not None and atr_val > 0:
                                # Risiko-basierte Positionsgröße
                                stop_dist_pct = (atr_k * atr_val) / price
                                raw_sizes[asset] = (risk_per_trade * total_equity) / stop_dist_pct
                            else:
                                # Fallback Equal-Weight für Assets ohne ATR
                                raw_sizes[asset] = cash / len(new_longs)

                        # Proportionale Skalierung: Summe darf verfügbares Cash nicht überschreiten
                        total_raw = sum(raw_sizes.values())
                        scale = min(1.0, cash / total_raw) if total_raw > 0 else 1.0

                        for asset, pos_value in raw_sizes.items():
                            price   = _get_price(price_cache, asset, date)
                            atr_val = _get_atr(atr_cache, asset, date) if (use_atr_trailing and atr_cache) else None
                            if price is None or price <= 0:
                                continue
                            scaled_value = pos_value * scale
                            shares = (scaled_value * (1 - fees)) / price
                            initial_stop = (
                                price - atr_k * atr_val
                                if atr_val is not None
                                else price * (1 - hard_stop_pct)  # Run F: hard_stop als Fallback
                            )
                            positions[asset] = {
                                'shares':        shares,
                                'entry':         price,
                                'direction':     1,
                                'hold_days':     0,
                                'trailing_stop': initial_stop,
                            }
                            cash -= scaled_value

                    else:
                        # Equal-Weight (Standard in Run F)
                        per_position = cash / len(new_longs)
                        for asset in new_longs:
                            price   = _get_price(price_cache, asset, date)
                            atr_val = _get_atr(atr_cache, asset, date) if (use_atr_trailing and atr_cache) else None
                            if price is None or price <= 0:
                                continue
                            shares = (per_position * (1 - fees)) / price
                            initial_stop = (
                                price - atr_k * atr_val
                                if atr_val is not None
                                else price * (1 - hard_stop_pct)  # Run F: hard_stop als Fallback
                            )
                            positions[asset] = {
                                'shares':        shares,
                                'entry':         price,
                                'direction':     1,
                                'hold_days':     0,
                                'trailing_stop': initial_stop,
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

            # Daily-Return nach Filter-Zustand tracken
            if len(equity) >= 3:
                day_ret = (equity[-1] - equity[-2]) / (equity[-2] + 1e-9)
                if not allow_new_entries:
                    equity_filter_on.append(day_ret)
                else:
                    equity_filter_off.append(day_ret)
                if signal_weak_today:
                    equity_sig_weak.append(day_ret)
                else:
                    equity_sig_strong.append(day_ret)

    # Letzte Positionen auflösen (Ende des Backtests)
    if positions and equity_dates:
        last_date = equity_dates[-1]
        for asset, pos in list(positions.items()):
            _close_position(asset, pos, last_date, 'end_of_backtest', 'n/a')
        positions = {}

    # RUN H1: Noch offene Crash-Phase abschließen
    if use_crash_protection and crash_mode_active and _phase_start is not None and equity_dates:
        phase_days = crash_mode_days - sum(p['days'] for p in crash_phases)
        crash_phases.append({
            'start':   str(_phase_start.date()),
            'end':     str(equity_dates[-1].date()) + ' (Backtest-Ende)',
            'days':    phase_days,
            'max_dd':  round(_phase_max_dd * 100, 2),
        })

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

    # ── PnL-Beitrag pro Exit-Typ ──────────────────────────────────────────────
    def _exit_stats(reason_filter: str) -> dict:
        subset = [t for t in trade_log if reason_filter in t.get('exit_reason', '')]
        if not subset:
            return {'n': 0, 'pnl_sum': 0.0, 'pnl_avg': 0.0, 'hold_avg': 0.0, 'win_pct': 0.0}
        pnls  = [t['pnl_pct'] for t in subset]
        holds = [t.get('hold_days', 0) for t in subset]
        return {
            'n':        len(subset),
            'pnl_sum':  round(sum(pnls), 1),
            'pnl_avg':  round(sum(pnls) / len(pnls), 2),
            'hold_avg': round(sum(holds) / len(holds), 1),
            'win_pct':  round(sum(1 for p in pnls if p > 0) / len(pnls) * 100, 1),
        }

    stats_hard    = _exit_stats('hard_stop')
    stats_atr     = _exit_stats('atr_trailing_stop')
    stats_fix     = _exit_stats('stop_loss')
    stats_rot     = _exit_stats('rotation')
    stats_lowpred = _exit_stats('low_pred_return')

    stop_hits = stats_hard['n'] + stats_atr['n'] + stats_fix['n']

    logger.success("═" * 63)
    logger.success(f"BACKTEST: {strategy_name}")
    logger.success("═" * 63)
    logger.success(f"  Total Return  : {total_return:+.2f}%")
    logger.success(f"  Max Drawdown  : {max_dd:.2f}%")
    logger.success(f"  Sharpe Ratio  : {sharpe:.3f}")
    logger.success(f"  Trades        : {len(trade_log)}")
    logger.success(f"  Win Rate      : {win_r:.1f}%")
    logger.success(f"  Avg Hold Days : {avg_hold:.1f}")
    logger.success("  ── PnL pro Exit-Typ ──────────────────────────────")
    logger.success(
        f"  Rotation     : n={stats_rot['n']:4d}  "
        f"sum={stats_rot['pnl_sum']:+8.0f}%  avg={stats_rot['pnl_avg']:+5.1f}%  "
        f"hold={stats_rot['hold_avg']:.1f}d  win={stats_rot['win_pct']:.0f}%"
    )
    logger.success(
        f"  ATR-Stop     : n={stats_atr['n']:4d}  "
        f"sum={stats_atr['pnl_sum']:+8.0f}%  avg={stats_atr['pnl_avg']:+5.1f}%  "
        f"hold={stats_atr['hold_avg']:.1f}d  win={stats_atr['win_pct']:.0f}%"
    )
    logger.success(
        f"  Hard-Stop    : n={stats_hard['n']:4d}  "
        f"sum={stats_hard['pnl_sum']:+8.0f}%  avg={stats_hard['pnl_avg']:+5.1f}%  "
        f"hold={stats_hard['hold_avg']:.1f}d  win={stats_hard['win_pct']:.0f}%"
    )
    if stats_lowpred['n'] > 0:
        logger.success(
            f"  Low-Pred-Exit: n={stats_lowpred['n']:4d}  "
            f"sum={stats_lowpred['pnl_sum']:+8.0f}%  avg={stats_lowpred['pnl_avg']:+5.1f}%  "
            f"hold={stats_lowpred['hold_avg']:.1f}d  win={stats_lowpred['win_pct']:.0f}%"
        )
    if stats_fix['n'] > 0:
        logger.warning(
            f"  Fix-Stop(5%) : n={stats_fix['n']:4d}  "
            f"sum={stats_fix['pnl_sum']:+8.0f}%  avg={stats_fix['pnl_avg']:+5.1f}%  "
            f"(sollte 0 sein in Run H1!)"
        )

    # ── Expected-Return-Filter Reporting ────────────────────────────────────
    if use_min_expected_return_filter:
        total_days = len(top1_preds)
        avg_top1 = np.mean(top1_preds) if top1_preds else 0.0
        avg_topN_all = np.mean(topN_preds) if topN_preds else 0.0
        logger.success(f"  ── Expected-Return-Filter ────────────────────────")
        logger.success(
            f"  Filter aktiv  : {filter_active_days}/{total_days} Tage "
            f"({filter_active_days/total_days*100:.1f}%)" if total_days > 0 else
            "  Filter aktiv  : 0/0 Tage"
        )
        logger.success(f"  Avg Top-1 Pred: {avg_top1*100:.3f}%")
        logger.success(f"  Avg Top-N Pred: {avg_topN_all*100:.3f}%")
        logger.success(f"  Low-Pred Exits : {stats_lowpred['n']}")

        if equity_filter_on and equity_filter_off:
            avg_ret_on  = np.mean(equity_filter_on) * 100
            avg_ret_off = np.mean(equity_filter_off) * 100
            std_on      = np.std(equity_filter_on) * 100
            std_off     = np.std(equity_filter_off) * 100
            logger.success(
                f"  Avg Daily-Ret (Filter ON) : {avg_ret_on:+.4f}%  "
                f"std={std_on:.4f}%  ({len(equity_filter_on)} Tage)"
            )
            logger.success(
                f"  Avg Daily-Ret (Filter OFF): {avg_ret_off:+.4f}%  "
                f"std={std_off:.4f}%  ({len(equity_filter_off)} Tage)"
            )

    # ── Signalstärke-Filter Reporting ──────────────────────────────────────
    if use_signal_strength_filter:
        total_sig_days = sig_weak_days + sig_strong_days
        logger.success(f"  ── Signalstärke-Filter ───────────────────────────")
        logger.success(
            f"  Weak-Signal   : {sig_weak_days}/{total_sig_days} Tage "
            f"({sig_weak_days/total_sig_days*100:.1f}%)" if total_sig_days > 0 else
            "  Weak-Signal   : 0/0 Tage"
        )
        logger.success(
            f"  Action        : {signal_filter_action}  "
            f"(spread>={min_score_spread_top1_med}, std>={min_score_std_universe})"
        )
        if sig_spreads:
            logger.success(
                f"  Avg Spread    : {np.mean(sig_spreads)*100:.4f}%  "
                f"(weak={np.mean(sig_weak_spreads)*100:.4f}%  "
                f"strong={np.mean(sig_strong_spreads)*100:.4f}%)"
                if sig_weak_spreads and sig_strong_spreads else
                f"  Avg Spread    : {np.mean(sig_spreads)*100:.4f}%"
            )
        if sig_stds:
            logger.success(f"  Avg Std       : {np.mean(sig_stds)*100:.4f}%")
        if equity_sig_weak and equity_sig_strong:
            avg_w = np.mean(equity_sig_weak) * 100
            avg_s = np.mean(equity_sig_strong) * 100
            logger.success(
                f"  Avg DailyRet  : weak={avg_w:+.4f}%({len(equity_sig_weak)}d)  "
                f"strong={avg_s:+.4f}%({len(equity_sig_strong)}d)"
            )

    if use_dd_control and dd_mode_days > 0:
        logger.success(
            f"  DD-Control   : {dd_mode_days} Tage im Schutz-Modus  "
            f"({len(dd_events)} Modus-Wechsel)"
        )
    # RUN H1: Crash-Schutz Reporting
    if use_crash_protection:
        logger.success(f"  ── RUN H1: Crash-Schutz ──────────────────────────")
        logger.success(
            f"  Halbgas-Modus: {crash_mode_days} Tage  ({len(crash_phases)} Phasen)"
        )
        for ph in crash_phases:
            logger.success(
                f"    Phase {ph['start']} → {ph['end']}  "
                f"({ph['days']}d)  Max-DD={ph['max_dd']:.1f}%"
            )
        # Vergleich mit Run G Referenzwerten
        logger.success(f"  ── Vergleich Run G → Run H1 (Long-Only) ──────────")
        run_g = {'return': 403.93, 'dd': -55.48, 'sharpe': 0.784, 'trades': 471, 'hold': 14.8}
        logger.success(
            f"  {'Kennzahl':15s}  {'Run G':>10s}  {'Run H1':>10s}  {'Delta':>10s}"
        )
        pairs = [
            ('Total Return%', run_g['return'],  total_return),
            ('Max Drawdown%', run_g['dd'],       max_dd),
            ('Sharpe',        run_g['sharpe'],   sharpe),
            ('Trades',        run_g['trades'],   float(len(trade_log))),
            ('Avg Hold Days', run_g['hold'],     avg_hold),
        ]
        for label, g_val, h_val in pairs:
            delta = h_val - g_val
            logger.success(
                f"  {label:15s}  {g_val:>10.2f}  {h_val:>10.2f}  {delta:>+10.2f}"
            )
    logger.success("═" * 63)

    return {
        'strategy':       strategy_name,
        'total_return':   round(total_return, 2),
        'max_drawdown':   round(max_dd, 2),
        'sharpe':         round(sharpe, 3),
        'n_trades':       len(trade_log),
        'win_rate':       round(win_r, 1),
        'avg_hold_days':  round(avg_hold, 1),
        'stop_loss_hits': stop_hits,
        'rotations':      stats_rot['n'],
        'exit_stats': {
            'rotation':          stats_rot,
            'atr_trailing_stop': stats_atr,
            'hard_stop':         stats_hard,
            'fix_stop':          stats_fix,
            'low_pred_return':   stats_lowpred,
        },
        'expected_return_filter': {
            'enabled':             use_min_expected_return_filter,
            'min_top1_threshold':  min_expected_return_top,
            'avg_topN_enabled':    use_avg_topN_filter,
            'avg_topN_threshold':  min_avg_expected_return_topN,
            'filter_active_days':  filter_active_days,
            'total_days':          len(top1_preds),
            'pct_active':          round(filter_active_days / len(top1_preds) * 100, 1) if top1_preds else 0.0,
            'avg_top1_pred':       round(float(np.mean(top1_preds)) * 100, 4) if top1_preds else 0.0,
            'avg_topN_pred':       round(float(np.mean(topN_preds)) * 100, 4) if topN_preds else 0.0,
            'low_pred_exits':      stats_lowpred['n'],
            'avg_daily_ret_filter_on':  round(float(np.mean(equity_filter_on)) * 100, 4) if equity_filter_on else None,
            'avg_daily_ret_filter_off': round(float(np.mean(equity_filter_off)) * 100, 4) if equity_filter_off else None,
        },
        'signal_strength_filter': {
            'enabled':              use_signal_strength_filter,
            'spread_filter':        use_score_spread_filter,
            'std_filter':           use_score_std_filter,
            'min_spread':           min_score_spread_top1_med,
            'min_std':              min_score_std_universe,
            'action':               signal_filter_action,
            'weak_days':            sig_weak_days,
            'strong_days':          sig_strong_days,
            'total_days':           sig_weak_days + sig_strong_days,
            'pct_weak':             round(sig_weak_days / (sig_weak_days + sig_strong_days) * 100, 1) if (sig_weak_days + sig_strong_days) > 0 else 0.0,
            'avg_spread':           round(float(np.mean(sig_spreads)) * 100, 4) if sig_spreads else 0.0,
            'avg_std':              round(float(np.mean(sig_stds)) * 100, 4) if sig_stds else 0.0,
            'avg_spread_weak':      round(float(np.mean(sig_weak_spreads)) * 100, 4) if sig_weak_spreads else None,
            'avg_spread_strong':    round(float(np.mean(sig_strong_spreads)) * 100, 4) if sig_strong_spreads else None,
            'avg_daily_ret_weak':   round(float(np.mean(equity_sig_weak)) * 100, 4) if equity_sig_weak else None,
            'avg_daily_ret_strong': round(float(np.mean(equity_sig_strong)) * 100, 4) if equity_sig_strong else None,
        },
        # DD-Control Stats
        'dd_control': {
            'enabled':        use_dd_control,
            'days_in_mode':   dd_mode_days,
            'n_mode_switches': len(dd_events),
            'events':         dd_events,
        },
        # RUN H1: Crash-Schutz Stats
        'crash_protection': {
            'enabled':         use_crash_protection,
            'crash_mode_days': crash_mode_days,
            'n_phases':        len(crash_phases),
            'phases':          crash_phases,
            'config': {
                'spy_atr_window':         spy_atr_window,
                'spy_lookback_high_days': spy_lookback_high_days,
                'spy_crash_atr_mult':     spy_crash_atr_mult,
                'dd_crash_threshold':     dd_crash_threshold,
                'dd_crash_recovery':      dd_crash_recovery,
            },
        },
        'equity':        equity_arr.tolist(),
        'equity_dates':  [str(d.date()) for d in equity_dates],
        'trade_log':     trade_log,
        'daily_signals': daily_signals,
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

    # Timezone-Normalisierung: Cache-Index ist UTC-aware, dates sind tz-naiv.
    # Für Vergleiche (<=, >=) müssen beide auf tz-naiv normalisiert werden.
    def _norm(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
        return idx.tz_localize(None) if idx.tz is not None else idx

    def _norm_ts(ts: pd.Timestamp) -> pd.Timestamp:
        return ts.tz_localize(None) if ts.tzinfo is not None else ts

    start_cmp = _norm_ts(start_date)

    def _series_before(s: pd.Series, ts: pd.Timestamp) -> pd.Series:
        """Slice einer ggf. tz-aware Series bis inkl. ts (tz-naiv)."""
        idx_norm = _norm(s.index)
        return s[idx_norm <= _norm_ts(ts)]

    # ── 1. SPY Buy & Hold ─────────────────────────────────────────────────
    spy_series  = price_cache.get(spy_ticker)
    spy_equity  = []
    spy_ret     = None
    if spy_series is not None:
        past0 = _series_before(spy_series, start_date)
        p0    = float(past0.iloc[-1]) if len(past0) > 0 else None
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
            past = _series_before(s, start_date)
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
        has_a = result_a.get('equity_dates') and result_a.get('equity')
        has_b = result_b.get('equity_dates') and result_b.get('equity')

        if has_a:
            dates_a = pd.to_datetime(result_a['equity_dates'])
            eq_a    = np.array(result_a['equity'])
            ax1.plot(dates_a, eq_a / eq_a[0] * 100 - 100,
                     label=f"Long-Only  ({result_a['total_return']:+.1f}%)",
                     color='#2196F3', linewidth=2.5, zorder=5)

        if has_b:
            dates_b = pd.to_datetime(result_b['equity_dates'])
            eq_b    = np.array(result_b['equity'])
            ax1.plot(dates_b, eq_b / eq_b[0] * 100 - 100,
                     label=f"Long-Short ({result_b['total_return']:+.1f}%)",
                     color='#4CAF50', linewidth=2.5, zorder=5)

        if not has_a and not has_b:
            logger.warning("plot_equity: Keine Equity-Daten vorhanden")
            plt.close(fig)
            return

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

        if has_a:
            ax2.fill_between(dates_a, drawdown(eq_a), 0, alpha=0.35, color='#2196F3',
                             label=f"Long-Only  (MaxDD: {result_a['max_drawdown']:.1f}%)")
        if has_b:
            ax2.fill_between(dates_b, drawdown(eq_b), 0, alpha=0.35, color='#4CAF50',
                             label=f"Long-Short (MaxDD: {result_b['max_drawdown']:.1f}%)")
        # SPY Drawdown als Referenz
        if benchmarks and benchmarks.get('spy', {}).get('equity'):
            eq_spy = np.array(benchmarks['spy']['equity'])
            bm_d = pd.to_datetime(benchmarks.get('dates', []))
            if len(bm_d) == 0 and has_a:
                bm_d = dates_a
            ax2.plot(bm_d[:len(eq_spy)], drawdown(eq_spy),
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

def _align_date_tz(date: pd.Timestamp, index: pd.DatetimeIndex) -> pd.Timestamp:
    """
    Gleicht die Timezone von `date` an `index` an, damit searchsorted funktioniert.

    Equity-Dates werden als naive Strings gespeichert ("2020-01-28"),
    der Price-Cache-Index ist UTC-aware. Ohne Angleichung wirft searchsorted:
    'Cannot compare tz-naive and tz-aware datetime-like objects'.
    """
    date = pd.Timestamp(date)
    if index.tz is not None and date.tzinfo is None:
        return date.tz_localize("UTC")
    if index.tz is None and date.tzinfo is not None:
        return date.tz_localize(None)
    return date


# ── Run E: Korrelations-Hilfsfunktion ────────────────────────────────────────

def _rolling_return_corr(
    price_cache: dict[str, pd.Series],
    asset1:      str,
    asset2:      str,
    date:        pd.Timestamp,
    window:      int = 60,
) -> float:
    """
    Rolling-Return-Korrelation zweier Assets über die letzten `window`
    Handelstage bis inkl. `date`.

    Zero-Lookahead-sicher: nutzt ausschließlich Schlusskurse die zum
    Zeitpunkt der Transaktion (Close von `date`) bekannt sind.
    Gibt 0.0 zurück wenn Daten fehlen oder < 20 gemeinsame Tage vorhanden.
    """
    s1 = price_cache.get(asset1)
    s2 = price_cache.get(asset2)
    if s1 is None or s2 is None:
        return 0.0

    d1 = _align_date_tz(date, s1.index)
    d2 = _align_date_tz(date, s2.index)

    # Letztes `window`-Fenster bis inkl. date (Close bekannt)
    s1_w = s1[s1.index <= d1].iloc[-window:]
    s2_w = s2[s2.index <= d2].iloc[-window:]

    common = s1_w.index.intersection(s2_w.index)
    if len(common) < 20:
        return 0.0

    r1 = s1_w.loc[common].pct_change().dropna()
    r2 = s2_w.loc[common].pct_change().dropna()
    if len(r1) < 10 or r1.std() < 1e-9 or r2.std() < 1e-9:
        return 0.0

    return float(r1.corr(r2))


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
    date = _align_date_tz(date, series.index)
    idx = series.index.searchsorted(date)
    idx = min(idx, len(series) - 1)
    return float(series.iloc[idx])


def _get_atr(
    cache: dict[str, pd.Series],
    asset: str,
    date:  pd.Timestamp,
) -> Optional[float]:
    """
    Gibt den ATR-Wert eines Assets an einem Tag aus dem ATR-Cache zurück.

    Nutzt dieselbe searchsorted-Logik wie _get_price — O(log n).
    Gibt None zurück bei fehlendem Asset oder NaN-Wert (Warm-up-Phase).
    """
    series = cache.get(asset)
    if series is None or len(series) == 0:
        return None
    date = _align_date_tz(date, series.index)
    idx = series.index.searchsorted(date)
    idx = min(idx, len(series) - 1)
    val = float(series.iloc[idx])
    return None if np.isnan(val) else val


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
