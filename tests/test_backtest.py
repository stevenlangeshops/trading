"""
tests/test_backtest.py
──────────────────────
Unit-Tests für strategy/backtest.py.

Testet ausschließlich die Backtest-Logik (Price-Cache, Selektion, Equity) —
kein echtes LSTM-Modell, keine Parquet-Daten vom Disk notwendig.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from strategy.backtest import (
    _get_price,
    _position_value,
    build_price_cache,
    run_backtest,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

DATES_3 = pd.date_range("2024-01-02", periods=3, freq="B")
DATES_2 = pd.date_range("2024-01-02", periods=2, freq="B")


def _make_parquet(tmp_path: Path, ticker: str, prices: list[float]) -> None:
    """Schreibt eine minimale Parquet-Datei mit Close-Preisen."""
    dates = pd.date_range("2024-01-02", periods=len(prices), freq="B")
    df = pd.DataFrame({"close": prices}, index=dates)
    df.to_parquet(tmp_path / f"{ticker}_1d.parquet")


# ── Test 1: build_price_cache ─────────────────────────────────────────────────

def test_build_price_cache_loads_parquet(tmp_path):
    """build_price_cache lädt Close-Preise korrekt aus einer Parquet-Datei."""
    _make_parquet(tmp_path, "AAPL", [150.0, 152.0, 155.0])

    cache = build_price_cache(["AAPL"], raw_dir=tmp_path)

    assert "AAPL" in cache
    assert len(cache["AAPL"]) == 3
    assert cache["AAPL"].iloc[0] == pytest.approx(150.0)
    assert cache["AAPL"].iloc[2] == pytest.approx(155.0)


def test_build_price_cache_skips_missing(tmp_path):
    """Fehlende Parquet-Datei wird übersprungen — kein Crash."""
    cache = build_price_cache(["UNKNOWN_XYZ"], raw_dir=tmp_path)
    assert "UNKNOWN_XYZ" not in cache
    assert len(cache) == 0


# ── Test 2: _get_price ────────────────────────────────────────────────────────

def test_get_price_returns_correct_value():
    """_get_price gibt den richtigen Preis für ein bekanntes Datum zurück."""
    cache = {
        "AAPL": pd.Series([100.0, 105.0, 110.0], index=DATES_3),
    }
    assert _get_price(cache, "AAPL", DATES_3[0]) == pytest.approx(100.0)
    assert _get_price(cache, "AAPL", DATES_3[1]) == pytest.approx(105.0)
    assert _get_price(cache, "AAPL", DATES_3[2]) == pytest.approx(110.0)


def test_get_price_cache_miss_returns_none():
    """_get_price gibt None zurück wenn Asset nicht im Cache ist."""
    cache = {}
    result = _get_price(cache, "AAPL", DATES_3[0])
    assert result is None


# ── Test 3: _position_value ───────────────────────────────────────────────────

def test_position_value_long():
    """_position_value berechnet Long-Position korrekt."""
    cache = {
        "AAPL": pd.Series([100.0, 105.0], index=DATES_2),
    }
    positions = {
        "AAPL": {"shares": 10.0, "entry": 100.0, "direction": 1},
    }
    # Am zweiten Tag: 10 Shares × 105 = 1050
    val = _position_value(positions, cache, DATES_2[1])
    assert val == pytest.approx(1050.0)


# ── Test 4: Long-Only kauft Top-Asset, Equity wächst ─────────────────────────

def test_long_only_top_asset_bought_equity_grows(tmp_path):
    """
    Kerntest der Backtest-Logik:

    Setup:
      - 2 Assets: ASSET_A (pred=+0.10), ASSET_B (pred=+0.02)
      - ASSET_A steigt +5% (100 → 105), ASSET_B steigt +2% (100 → 102)
      - Long-Only mit n_max=1 → nur ASSET_A wird gekauft
      - fees=0 (Gebühren rausnehmen damit Equity klar messbar)

    Erwartung:
      - Equity am Ende > Startkapital (10_000)
      - ASSET_B erscheint nicht im trade_log
      - ASSET_A erscheint im trade_log
    """
    init_cash = 10_000.0
    dates = pd.date_range("2024-01-02", periods=2, freq="B")

    # Synthetischer Price-Cache
    price_cache = {
        "ASSET_A": pd.Series([100.0, 105.0], index=dates),
        "ASSET_B": pd.Series([100.0, 102.0], index=dates),
    }

    # Minimales synthetisches features-DataFrame (MultiIndex date × asset)
    # Inhalt spielt keine Rolle da predict_cross_section gemockt wird
    idx = pd.MultiIndex.from_tuples(
        [(dates[0], "ASSET_A"), (dates[0], "ASSET_B"),
         (dates[1], "ASSET_A"), (dates[1], "ASSET_B")],
        names=["date", "asset"],
    )
    features = pd.DataFrame(
        np.zeros((4, 18)),
        index=idx,
        columns=[f"feat_{i}" for i in range(18)],
    )
    targets = pd.Series(np.zeros(4), index=idx)

    asset_map = {"ASSET_A": 1, "ASSET_B": 2}

    # Mock: Modell laden (kein echtes Checkpoint nötig)
    mock_model = MagicMock()

    # Mock: predict_cross_section gibt kontrollierte Vorhersagen zurück.
    # Nur am ersten Tag pred zurückgeben (an Tag 2 kaufen wir bereits auf Basis von Tag 1).
    def fake_predict(model, features, asset_map, date, seq_len, device):
        return pd.Series({"ASSET_A": 0.10, "ASSET_B": 0.02}).sort_values(ascending=False)

    # Leere Temp-Datei als Dummy-Checkpoint — load_fold_model ist gemockt,
    # aber Path.exists() muss True zurückgeben damit der Fold nicht übersprungen wird.
    dummy_ckpt = tmp_path / "fold_0_best.pt"
    dummy_ckpt.write_bytes(b"")

    fold_results = [{
        "fold_id":   0,
        "ckpt_path": str(dummy_ckpt),
        "val_start": str(dates[0].date()),
        "val_end":   str(dates[1].date()),
    }]

    with (
        patch("strategy.backtest.load_fold_model", return_value=(mock_model, {})),
        patch("strategy.backtest.predict_cross_section", side_effect=fake_predict),
    ):
        result = run_backtest(
            features     = features,
            targets      = targets,
            fold_results = fold_results,
            asset_map    = asset_map,
            n_max        = 1,
            n_mid        = 1,
            n_min        = 1,
            long_short   = False,
            fees         = 0.0,         # Gebühren aus damit Equity klar messbar
            init_cash    = init_cash,
            seq_len      = 1,
            use_regime   = False,       # kein SPY-Lookup nötig
            price_cache  = price_cache,
        )

    assert result, "run_backtest gab leeres Ergebnis zurück"

    # Equity am Ende > Startkapital
    final_equity = result["equity"][-1]
    assert final_equity > init_cash, (
        f"Equity ({final_equity:.2f}) sollte über Startkapital ({init_cash:.2f}) liegen"
    )

    # Trade-Log: ASSET_A wurde gehandelt, ASSET_B nicht
    traded_assets = {t["asset"] for t in result["trade_log"]}
    assert "ASSET_A" in traded_assets, "ASSET_A sollte im trade_log sein"
    assert "ASSET_B" not in traded_assets, "ASSET_B sollte NICHT im trade_log sein"
