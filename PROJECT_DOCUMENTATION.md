# Trading Bot v5 — Vollständige Projektdokumentation

**Stand:** 16. April 2026
**Repo:** `https://github.com/stevenlangeshops/trading.git`
**Kaggle User:** `busersteven`
**Kaggle Dataset:** `busersteven/trading-raw-data` (260 Parquet-Dateien)
**Kaggle Results:** `busersteven/trading-results` (permanente Ergebnisspeicherung)

> **Aktuelles Referenzmodell (neue Basis):** `v2_7d` — Single-Horizon LSTM, 7-Tage-Vorhersagehorizont
> Total Return **+1.420,9 %** | Sharpe **1,151** | Max DD **-55,65 %** | Run-Datum 16. April 2026
> Git-Commit: `10f9a17` | Reproduzierbar via: `kaggle datasets download busersteven/trading-results --unzip`

---

## 1. Projektziel

Ein vollautomatischer Trading-Bot in Python/PyTorch, der ein großes Anlageuniversum (aktuell 260 S&P-500-Aktien + ETFs) verwaltet. Basierend auf historischen Tagesdaten und technischen Indikatoren wird ein **Cross-Sectional Ranking** durchgeführt, um die Top-Performer der nächsten 1–3 Wochen zu identifizieren. Das Modell allokiert Kapital in die am höchsten gerankten Aktien.

### Eiserne Quant-Regeln

1. **Zero Lookahead-Bias** — Absolute Strenge bei zeitlicher Trennung. Niemals zukünftige Daten für heutige Features.
2. **Walk-Forward Validation** — Expanding Window Folds. Kein statischer Train/Test-Split.
3. **Rank über Regression** — Optimierung auf relatives Ranking (Rank IC, Rank Loss), da absolute Finanzzeitreihen zu verrauscht.
4. **Kaggle-Ready** — Alle Outputs, Logs und Checkpoints für `/kaggle/working/` oder dynamisch erkannte Pfade.

---

## 2. Architektur-Überblick

```
┌──────────────────────────────────────────────────────────┐
│  Kaggle Notebook (kaggle_notebook.ipynb)                 │
│    └─ exec(kaggle_full_run.py)                           │
│                                                          │
│  Pipeline:                                               │
│  1. Git clone repo                                       │
│  2. CUDA Health-Check                                    │
│  3. pip install dependencies                             │
│  4. Parquet-Daten aus Kaggle Dataset laden               │
│  5. build_panel() → Features + Targets                   │
│  6. Asset-Map (Ticker → ID)                              │
│  7. Walk-Forward Training (v1 oder v2)                   │
│  8. Backtest (Long-Only)                                 │
│  9. Artefakte packen (tar.gz)                            │
│ 10. Ergebnisse in Kaggle Dataset persistieren            │
└──────────────────────────────────────────────────────────┘
```

---

## 3. Modellvarianten

### 3.1 v2_single_horizon — NEUES REFERENZMODELL (Stand April 2026)

> **Das ist unsere neue Arbeitsbasis.** Alle zukünftigen Experimente bauen auf dieser Architektur auf.

- **Modell:** `SingleHorizonRankModel` (models_v2_single_horizon.py)
- **Architektur:** LSTM + Asset-Embedding → 1 Score pro Asset/Tag
- **Output:** Ranking-Score, korreliert mit Forward-Return des jeweiligen Horizonts
- **Loss:** MSE + 0.5 × PairwiseRankLoss (identisch zu Run G)
- **Training:** train_v2_single_horizon.py → `train_all_horizons()`
- **Backtest:** backtest_v2_single_horizon.py → `run_backtest_single_horizon()`
- **Konfiguration:** config_v2_single_horizon.py → `SingleHorizonConfig`

**Bestes Ergebnis: v2_7d (16. April 2026)**

| Kennzahl | Wert |
|---|---|
| Total Return | **+1.420,9 %** |
| Max Drawdown | -55,65 % |
| Sharpe Ratio | **1,151** |
| Rank IC (Ø) | 0,0622 |
| Trades | 643 |
| Win Rate | 50,9 % |
| Avg Hold Days | 10,6 Tage |
| Walk-Forward Folds | 12 (2020–2026) |

**Reproduktion:**
```bash
kaggle datasets download busersteven/trading-results --unzip
# kaggle_artifacts.tar.gz enthält: 12 Checkpoints (fold_0..11_best.pt),
# v2_7d_equity.png, run_manifest.json, pipeline.log
```
Git-Commit: `10f9a17` | Kaggle-Notebook-Schalter: `SMOKE_TEST=False, HORIZONS=[7]`

---

### 3.2 v1_rank (Run G — historische Baseline)

- **Modell:** `CrossSectionalLSTM` (models/lstm_model.py)
- **Output:** 1 Score pro Asset/Tag (unkalibriert, korreliert mit 11d-Forward-Return)
- **Loss:** MSE + 0.5 × PairwiseRankLoss
- **Training:** models/trainer.py → `train_walk_forward()`
- **Backtest:** strategy/backtest.py → `run_backtest()`
- **Bestes Ergebnis (Run G):**
  - Total Return: **+403,9 %**
  - Max Drawdown: **-55,48 %**
  - Sharpe: **0,784**
  - Trades: 471, Win-Rate: 52,4 %, Avg Hold: 14,8 Tage
  - Rank IC: 0,035

### 3.3 v2_return_multi (Zwischenexperiment, nicht weiterverfolgt)

- **Modell:** `LSTMReturnMultiV2` (models_v2_return_multi.py)
- **Output:** 4 erwartete Returns pro Asset/Tag (4d, 7d, 11d, 15d)
- **Loss:** gewichteter Huber (4 Horizonte) + 0.1 × PairwiseRankLoss (auf 11d)
- **Status:** Testlauf durchgeführt; v2_single_horizon erwies sich als überlegener Ansatz

---

## 4. Datei-Referenz

### 4.1 Kernmodule

| Datei | Beschreibung |
|-------|-------------|
| `features/engineer.py` | 18 technische Indikatoren + Cross-Sectional Z-Score + Forward-Return Target |
| `models/lstm_model.py` | `CrossSectionalLSTM`: LSTM + TemporalAttention + Asset-Embedding → 1 Score |
| `models/dataset.py` | `WalkForwardFold`, `CrossSectionalDataset`, `create_walk_forward_folds()` |
| `models/trainer.py` | `RankLoss`, `CombinedLoss`, `train_walk_forward()` (v1) |
| `strategy/backtest.py` | Komplette Backtest-Engine: Regime-Filter, Rotation, Hard-Stop, alle Filter |
| `strategy/calibration.py` | Score→Return Kalibrierung (Linear + Isotonic Regression) |

### 4.2 v2 Single-Horizon Module (aktuelles Referenzmodell)

| Datei | Beschreibung |
|-------|-------------|
| `config_v2_single_horizon.py` | `SingleHorizonConfig` Dataclass; KAGGLE_SMOKE_TEST-Support |
| `models_v2_single_horizon.py` | `SingleHorizonRankModel`: LSTM + Asset-Embedding → 1 Score |
| `train_v2_single_horizon.py` | Walk-Forward Training, `train_all_horizons()` |
| `backtest_v2_single_horizon.py` | Backtest, `plot_equity_single()`, Benchmark-Report |

### 4.3 v2 Multi-Horizon Module (Zwischenexperiment)

| Datei | Beschreibung |
|-------|-------------|
| `config_v2_return_multi.py` | `V2Config` Dataclass: alle Parameter zentral |
| `models_v2_return_multi.py` | `LSTMReturnMultiV2` + `CombinedMultiHorizonLoss` |
| `train_v2_return_multi.py` | Multi-Horizon Targets + Dataset + Walk-Forward Training |
| `backtest_v2_return_multi.py` | Backtest + v1-vs-v2 Report + Vergleichs-Plot |

### 4.4 Kaggle-Integration

| Datei | Beschreibung |
|-------|-------------|
| `kaggle_notebook.ipynb` | Notebook mit `SMOKE_TEST`/`HORIZONS`-Schalter; lädt `kaggle_full_run.py` per wget + exec |
| `scripts/kaggle_full_run.py` | Komplette Pipeline (Schritte 1–9, 20–21); SMOKE_TEST, KAGGLE_SH_HORIZONS, Manifest |
| `scripts/kaggle_kernel_api.py` | CLI-Wrapper für Kaggle API (Push, Poll, Download) |
| `scripts/kaggle_watch.py` | Autonomer Job-Watcher (Poll, Diagnose, Auto-Resubmit) |
| `scripts/kaggle_status.py` | Terminal-Dashboard für Watcher-Status |

### 4.5 Daten

| Datei | Beschreibung |
|-------|-------------|
| `data/asset_list_sp500.txt` | 260 S&P-500-Aktien + ETFs (SPY, QQQ, IWM, GLD, TLT, etc.) |
| `data/download_stocks.py` | yfinance → Parquet (parallelisiert via ThreadPool) |
| `data/raw/dataset-metadata.json` | Kaggle Dataset Metadaten (`trading-raw-data`) |

### 4.6 Tests & Sonstiges

| Datei | Beschreibung |
|-------|-------------|
| `tests/test_backtest.py` | Unit-Tests: price_cache, position_value, Long-Only Backtest |
| `main.py` | Lokaler CLI-Einstieg (download, train, backtest, optimize) |
| `download_stocks_local.py` | Lokales Windows-Script für yfinance-Download |

---

## 5. Feature-Pipeline

### 5.1 Technische Indikatoren (18 Features)

```
Trend:       sma_ratio_20, sma_ratio_50, sma_ratio_200, ema_ratio_12, macd_diff
Momentum:    rsi_14, roc_5, roc_21, stoch_k
Volatilität: atr_ratio, bb_width, bb_pos
Volumen:     volume_ratio_20, obv_diff
Preis:       high_low_ratio, ret_1d, ret_5d, ret_21d
```

### 5.2 Normalisierung

Pro Handelstag werden alle Features über alle Assets hinweg **Cross-Sectional Z-Score** normalisiert:
- `z = (x - mean_all_assets) / std_all_assets`
- Capping bei ±4 Standardabweichungen
- Ergebnis: +1.5 bedeutet "dieses Asset hat den Feature-Wert 1.5 Std-Abw. über dem Tagesdurchschnitt"

### 5.3 Targets

- **v1:** `forward_return_11d = (close[t+11] / close[t]) - 1.0`
- **v2:** 4 Targets: `ret_4d, ret_7d, ret_11d, ret_15d`

---

## 6. Modell-Architektur

### Gemeinsame Basis (v1 und v2)

```
Asset-Embedding (dim=16) ──┐
                            ├─ Concat pro Zeitschritt
Feature-Sequenz (seq=64) ──┘
        │
        ▼
    LSTM (hidden=128, layers=2, dropout=0.3)
        │
        ▼
    Temporal Attention (gewichteter Durchschnitt über Zeitachse)
        │
        ▼
    LayerNorm
        │
        ▼
    FC Head: Linear(128→64) → GELU → Dropout → Linear(64→32) → GELU → Dropout → Linear(32→N)
```

| | v1_rank | v2_return_multi |
|--|---------|-----------------|
| N (Output) | 1 | 4 |
| Loss | MSE + 0.5 × RankLoss | Huber(4h, δ=0.02) + 0.1 × RankLoss(11d) |
| Klasse | `CrossSectionalLSTM` | `LSTMReturnMultiV2` |

### Walk-Forward Training

- **Expanding Window:** Trainingsstart immer bei t0, Trainingsende wächst
- **Embargo:** 1 Monat Puffer zwischen Training und Validierung (verhindert Lookahead über Forward-Return-Fenster)
- **Parameter:** train_years=3.0, val_months=6.0, step_months=6.0
- **~9–12 Folds** über den Gesamtzeitraum ~2016–2026
- **Early Stopping:** Patience 7 Epochen, max 50 Epochen
- **Optimizer:** AdamW (lr=5e-4, weight_decay=1e-3, CosineAnnealing)
- **Checkpoints:** `checkpoints/fold_X_best.pt` (v1), `checkpoints/v2_return_multi/fold_X_best.pt` (v2)

---

## 7. Backtest-Strategie (Run G Setup)

### Täglicher Loop

1. **Regime bestimmen** via SMA50/SMA200 auf SPY:
   - Bull (SPY > SMA50 > SMA200): n_max = 7 Positionen
   - Neutral (SPY > SMA200): n_mid = 3
   - Bear (SPY < SMA200): n_min = 1

2. **Cross-Section Prediction** für alle ~260 Assets → Ranking

3. **Exits prüfen:**
   - Hard-Stop: -25% vom Einstiegskurs → sofortiger Exit (Gap-Down-Schutz)
   - Rotation: Asset fällt unter Rang `n_long + rotation_buffer(3)` → Exit

4. **Neue Positionen:** Top-N Kandidaten, Equal-Weight-Allokation

### Deaktivierte Mechanismen (in Run G)

| Mechanismus | Status | Grund der Deaktivierung |
|-------------|--------|------------------------|
| ATR-Trailing-Stop | OFF | Zerstörte konsequent Wert (Run E: -521%, Run F: -704%) |
| Fixed Stop-Loss | OFF (entfernt) | 5% Stop verursachte massiven PnL-Drain |
| Portfolio DD-Control | OFF | Zu sensitiv, 71% der Zeit in Schutz, verpasste Rallies |
| Korrelations-Filter | OFF (cap=1.0) | Erzwang niedrig-gerankte Aktien, mehr Churn |
| Risk-Parity Sizing | OFF | Übersteuerte Modellsignal, verdünnte Momentum |
| SPY-ATR Crash-Schutz | OFF | Zu wenig, zu spät — reduzierte nur neue Entries |
| Signal-Spread-Filter | OFF | 86% aktiv, invertierter Effekt (weak=besser) |
| Expected-Return-Filter | OFF | Scores immer positiv, Filter nie ausgelöst |
| Kalibrierter Return-Filter | OFF | 0% negative Expected Returns nach Kalibrierung |

---

## 8. Run-Historie und Ergebnisse

### v1_rank Experimente (260 Assets, ~2020–2026, Horizont implizit 11d)

| Run | Config | Total Return | Max DD | Sharpe | Trades | Kern-Erkenntnis |
|-----|--------|-------------|--------|--------|--------|----------------|
| **D** | 260 Assets, GPU, Stop-Loss 15% | +175% | -60% | 0.582 | 520 | Hard-Stop zu eng, Short-Term-Filter kontraproduktiv |
| **E** | Corr-Filter, Risk-Parity, ATR k=2.5 | +29.7% | -60% | — | — | Alle neuen Mechanismen schadeten der Performance |
| **F** | ATR k=3.5, DD-Control 20/30%, n=7 | +17.9% | -68% | 0.298 | 248 | ATR-Stop (-704% PnL), DD-Control zu sensitiv |
| **G** | Pure Rotation + Hard-Stop 25% | **+403.9%** | -55.5% | **0.784** | 471 | Beste v1-Konfiguration. Einfachheit gewinnt. |
| **H1** | G + SPY-ATR Crash-Schutz | +335.3% | -54.5% | 0.724 | 406 | Marginale DD-Verbesserung, deutlicher Return-Verlust |
| **H2** | G + Signal-Spread-Filter | -18.3% | -42% | -0.028 | 108 | Katastrophal: Filter 86% aktiv, invertierter Effekt |
| **G_calib** | G + kalibrierter E[ret]-Filter ≥0% | +403.9% | -55.5% | 0.784 | 471 | Filter nie ausgelöst (E[ret] immer >0.5%) |

### v2_single_horizon Ergebnisse (260 Assets, 2020–2026) ← AKTUELLE BASIS

| Run | Horizont | Total Return | Max DD | Sharpe | Rank IC | Trades | Datum |
|-----|----------|-------------|--------|--------|---------|--------|-------|
| **v2_4d** | 4 Tage | +826,9% | -65,4% | 0,887 | 0,058 | ~900 | Apr 2026 |
| **v2_7d** ⭐ | **7 Tage** | **+1.420,9%** | **-55,7%** | **1,151** | **0,062** | **643** | **16. Apr 2026** |
| **v2_11d** | 11 Tage | +697,8% | -51,2% | 0,912 | 0,041 | ~530 | Apr 2026 |
| **v2_15d** | 15 Tage | +521,3% | -48,9% | 0,823 | 0,038 | ~450 | Apr 2026 |

> ⭐ **v2_7d ist das neue Referenzmodell.** Beste Kombination aus Return und Sharpe bei vergleichbarem Drawdown.

**v2_7d Exit-Statistik (Detail):**

| Ausstiegsgrund | Trades | Ø PnL | Ø Haltedauer | Win Rate |
|---|---|---|---|---|
| Rotation (planmäßig) | 629 (97,8 %) | +1,86 % | 10,4 Tage | 52,0 % |
| Hard-Stop (-25 %) | 14 (2,2 %) | -29,76 % | 19,4 Tage | 0,0 % |

**v2_7d Walk-Forward IC pro Fold:**

| Fold | Zeitraum | Rank IC | Bemerkung |
|---|---|---|---|
| 0 | Feb–Aug 2020 | -0,007 | COVID-Crash, negatives Signal |
| 1 | Aug 2020–Feb 2021 | +0,063 | Erholung |
| 2 | Feb–Aug 2021 | +0,068 | Stabil |
| 3 | Aug 2021–Feb 2022 | -0,008 | Zinswende-Unsicherheit |
| 4 | Feb–Aug 2022 | +0,150 | Sehr stark (Bärenmarkt) |
| 5 | Aug 2022–Feb 2023 | +0,001 | Nahezu null |
| 6 | Feb–Aug 2023 | +0,103 | Stark |
| 7 | Aug 2023–Feb 2024 | +0,153 | Stärkster Fold |
| 8 | Feb–Aug 2024 | +0,052 | Moderat |
| 9 | Aug 2024–Feb 2025 | +0,082 | Gut |
| 10 | Feb–Aug 2025 | +0,062 | Gut |
| 11 | Aug 2025–Feb 2026 | +0,027 | Moderat |

### Benchmarks (gleicher Zeitraum 2020–2026)

| Benchmark | Return | Sharpe | Max DD |
|-----------|--------|--------|--------|
| SPY Buy & Hold | +60,6% | 0,452 | -37,4% |
| EW Universe Buy & Hold | +192,8% | 0,940 | -34,6% |
| EW Universe Rebalanciert | +167,3% | 0,937 | -34,9% |
| **v2_7d (unser Modell)** | **+1.420,9%** | **1,151** | **-55,7%** |

### Kern-Erkenntnisse

1. **Kürzerer Horizont = besseres Signal:** 7d schlägt 11d deutlich (v1_rank lief auf 11d). Das Modell lernt kurzfristige Momentum-Muster effektiver.
2. **Rotation ist das Alpha:** 629 Rotation-Exits mit Ø +1,86 % PnL und 52 % Win-Rate. Der Hard-Stop (-25 %) feuert nur selten (14×) und kostet Ø -29,76 % — unvermeidlich als Tail-Risk-Schutz.
3. **Einfachheit gewinnt weiterhin:** Alle Zusatz-Mechanismen (ATR, DD-Control, Filter) schaden. Die reine LSTM-Rotation mit Hard-Stop bleibt ungeschlagen.

---

## 9. Kalibrierungs-Analyse

### Score→Return Kalibrierung (Isotonic Regression)

- **394.315 Score-Return-Paare** gesammelt (Out-of-Sample Walk-Forward Val-Perioden)
- **Train:** 315.452 Paare (2020–2024), **Val:** 78.863 Paare (2024–2026)
- **Korrelation Val-Set:** Pearson=0.0856, Spearman=0.0772

### Dezil-Analyse

| Dezil | Avg Score | Avg True 11d Ret | Avg Expected Ret |
|-------|-----------|-----------------|-----------------|
| 0 (niedrigste) | 0.00706 | **-0.55%** | +0.51% |
| 5 (Mitte) | 0.00913 | +0.57% | +1.35% |
| 9 (höchste) | 0.00951 | **+2.05%** | +1.82% |

**Befund:** Ranking funktioniert (Spread Dezil 0→9: ~2.6 Pp), aber absolute Scores sind in einem extrem engen Band (0.005–0.010) und nach Kalibrierung **immer positiv** (min +0.5%). Ein Filter ≥0% greift daher nie.

---

## 10. Offene Probleme und Designentscheidungen

### 10.1 Hard-Stop-Problem

28 Trades, alle mit -25% oder mehr, summieren sich zu -779% PnL. Das sind die verbleibenden Tail-Risk-Events. Bisherige Schutzversuche (ATR, DD-Control, Crash-Halbgas) haben entweder nichts gebracht oder Alpha vernichtet.

### 10.2 Score-Skala

Das LSTM produziert Scores in einem sehr engen positiven Band (~0.005–0.010). Die Differenzierung zwischen Gewinnern und Verlierern ist da, aber die absolute Skala ist nicht informativ genug für Schwellwert-Filter. Deshalb wurde v2_return_multi entwickelt.

### 10.3 Kaggle-Umgebung

- **GPU:** T4 x2 bevorzugt. P100 ist inkompatibel mit PyTorch 2.x/Python 3.12 (SM_60 < SM_70)
- **Module Caching:** `sys.modules` muss vor jedem Import geleert werden (Kaggle cached alte Versionen bei Notebook-Reruns)
- **GitHub Raw Caching:** wget-URLs brauchen Cache-Buster (`?cb={timestamp}`)
- **Artifact-Persistenz:** Interaktive Session-Dateien werden nach ~20 Min gelöscht → Upload in `busersteven/trading-results` Dataset

---

## 11. Verzeichnisstruktur

```
trading/
├── .gitignore
├── README.md
├── KAGGLE_KERNEL_RUNNER.md
├── PROJECT_DOCUMENTATION.md          ← diese Datei
├── requirements.txt
├── requirements_local.txt
├── main.py                            # Lokaler CLI-Einstieg
├── download_stocks_local.py           # Lokaler Windows yfinance Download
├── kaggle_notebook.ipynb              # Kaggle Notebook (2 Zellen)
│
├── config_v2_return_multi.py          # v2 Config (Dataclass)
├── models_v2_return_multi.py          # v2 LSTM (4 Outputs)
├── train_v2_return_multi.py           # v2 Walk-Forward Training
├── backtest_v2_return_multi.py        # v2 Backtest + Vergleich
│
├── data/
│   ├── asset_list_sp500.txt           # 260 Ticker
│   ├── download_stocks.py             # yfinance → Parquet
│   ├── download.py                    # Crypto via CCXT (legacy)
│   └── raw/
│       └── dataset-metadata.json
│
├── features/
│   └── engineer.py                    # 18 Features + CS-Z-Score + Targets
│
├── models/
│   ├── lstm_model.py                  # CrossSectionalLSTM (v1)
│   ├── dataset.py                     # Walk-Forward Folds + Dataset
│   ├── trainer.py                     # Walk-Forward Training (v1)
│   └── optimize.py                    # Optuna Hyperparameter-Suche
│
├── strategy/
│   ├── backtest.py                    # Komplette Backtest-Engine (v1)
│   └── calibration.py                 # Score→Return Kalibrierung
│
├── scripts/
│   ├── kaggle_full_run.py             # Haupt-Pipeline für Kaggle
│   ├── kaggle_kernel_api.py           # Kaggle API CLI-Wrapper
│   ├── kaggle_watch.py                # Autonomer Job-Watcher
│   ├── kaggle_status.py               # Terminal-Dashboard
│   ├── kaggle_backtest_run.py         # Nur-Backtest Kaggle-Entry
│   └── _backtest_kaggle_launcher.py   # Launcher für Backtest-Kernel
│
├── tests/
│   └── test_backtest.py               # Unit-Tests
│
└── checkpoints/                       # (gitignored)
    ├── fold_0_best.pt ... fold_N.pt   # v1 Checkpoints
    └── v2_return_multi/
        └── fold_0_best.pt ...         # v2 Checkpoints
```

---

## 12. Git-Historie (wichtige Commits)

```
── NEUE BASIS (April 2026) ──────────────────────────────────────────────
10f9a17 fix: Smoke-Test train_years 14→8 (Daten nur 2017-2026, ~9 Jahre)  ← AKTUELL
3e34283 refactor: Smoke-Test-Schalter direkt in kaggle_notebook.ipynb integriert
e4ce5e7 feat: Smoke-Test-Modus fuer schnellen Pipeline-Check (~10-15min)
bd4b730 feat: 7d-Fokus-Run mit vollstaendigem Artefakt-Archiv (plot_equity_single,
        step_create_run_manifest, v2_Xd_equity.png, run_manifest.json)

── SINGLE-HORIZON PHASE ─────────────────────────────────────────────────
0c6159e fix: sympy>=1.13 + TORCHDYNAMO_DISABLE + _purge_sympy (Kaggle PyTorch 2.10)
[...]    feat: v2 Single-Horizon Training/Backtest (4d/7d/11d/15d separate Modelle)
[...]    feat: KAGGLE_SH_HORIZONS env-var + Notebook-Schalter fuer Horizonte

── MULTI-HORIZON PHASE ──────────────────────────────────────────────────
ebcbea6 v2: volle Parameter (50 Ep, hidden=128) wie Run G
22e0d94 v2 Full Run: alle 260 Assets, V2_MAX_ASSETS=0
6fadc73 v2_return_multi: Multi-Horizon Return-Modell (4/7/11/15d)

── v1_rank EXPERIMENTE (Run A–H) ────────────────────────────────────────
6087674 Run G_calib: Score-to-Return Kalibrierung + Expected-Return-Filter
544a8d3 feat: Run G - use_dd_control=False (reine Rotation + Hard-Stop)
9e839ff feat: Run G - ATR-Trailing deaktiviert
a845f47 feat: Run H1 - SPY-ATR Crash-Schutz
ab9e61d feat: Run F - DD-Control
a71e0a9 feat: Run F - stop_loss_pct entfernt
b41e23a feat: Run E - Korrelations-Cap + Risk-Parity
54dddec feat: S&P500 Expansion auf 260 Assets
```

---

## 13. Kaggle-Workflow (Schritt für Schritt)

### Einmalige Einrichtung

1. Kaggle Account: `busersteven`
2. API Token als Kaggle Secret `KAGGLE_KEY` hinterlegt
3. Dataset `trading-raw-data` mit 260 Parquet-Dateien erstellt
4. Dataset `trading-results` für permanente Ergebnisspeicherung erstellt
5. Notebook `trading-bot-v5-fullrun` erstellt, T4 GPU konfiguriert

### Einen Run starten

1. Code-Änderungen committen und pushen (`git push`)
2. Kaggle Notebook öffnen
3. **Run All** klicken
4. Das Notebook lädt `kaggle_full_run.py` per wget von GitHub und führt es aus
5. Warten (~30 Min GPU, ~60–90 Min CPU)
6. Ergebnisse werden automatisch in `trading-results` Dataset gespeichert
7. Alternativ: `kaggle_artifacts.tar.gz` manuell herunterladen (innerhalb 20 Min)

### Modus-Schalter im Notebook (oben in der Python-Zelle)

```python
SMOKE_TEST = False  # True = Schnelltest ~10-15 Min | False = echter Run ~2-3h
HORIZONS   = [7]    # Horizonte in Tagen, z.B. [7] oder [4, 7, 11, 15]
```

`SMOKE_TEST = True` setzt automatisch: 15 Assets, 3 Epochen, ~2 Folds — alle Pipeline-Schritte laufen trotzdem durch.

### Bekannte Kaggle-Fallstricke

- **P100 GPU:** Inkompatibel mit PyTorch 2.x (SM_60). Immer T4 wählen.
- **Module Caching:** Bei "Run All" ohne Kernel-Restart cached Python alte Modulversionen → `sys.modules` Clearing in kaggle_full_run.py eingebaut.
- **GitHub Raw Cache:** raw.githubusercontent.com cached aggressiv → Cache-Buster Timestamp in wget-URL.
- **Artifact-Löschung:** Interaktive Session-Dateien nach ~20 Min weg → automatischer Upload in permanentes Dataset.

---

## 14. Lokale Entwicklung (Windows)

### Setup

```powershell
cd c:\steven\trading_v5\trading
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Daten herunterladen

```powershell
python download_stocks_local.py --years 10
# oder:
python data/download_stocks.py --asset-file data/asset_list_sp500.txt --timeframe 1d --years 10
```

### Tests

```powershell
python -m pytest tests/ -v
```

### Syntax-Check

```powershell
python -m py_compile strategy/backtest.py
python -m py_compile scripts/kaggle_full_run.py
```

---

## 15. Parametervergleich v1 vs v2

| Parameter | v1_rank (Run G) | v2_return_multi |
|-----------|----------------|-----------------|
| Output-Dimension | 1 Score | 4 Returns (4d,7d,11d,15d) |
| Loss (Regression) | MSE | Huber (δ=0.02), je 0.25 Gewicht |
| Loss (Ranking) | 0.5 × PairwiseRankLoss | 0.1 × PairwiseRankLoss (auf 11d) |
| Ranking-Horizont | implizit 11d | explizit 11d (konfigurierbar) |
| hidden_dim | 128 | 128 |
| num_layers | 2 | 2 |
| embed_dim | 16 | 16 |
| dropout | 0.3 | 0.3 |
| seq_len | 64 | 64 |
| lr | 5e-4 | 5e-4 |
| weight_decay | 1e-3 | 1e-3 |
| epochs | 50 | 50 |
| patience | 7 | 7 |
| batch_size | 512 | 512 |
| train_years | 3.0 | 3.0 |
| val_months | 6.0 | 6.0 |
| step_months | 6.0 | 6.0 |
| n_max / n_mid / n_min | 7 / 3 / 1 | 7 / 3 / 1 |
| hard_stop_pct | 0.25 | 0.25 |
| rotation_buffer | 3 | 3 |
| fees | 0.001 | 0.001 |
| ATR/DD/Crash/Filter | alle OFF | alle OFF |

---

## 16. Nächste Schritte (Stand April 2026)

### Kurzfristig (auf Basis v2_7d)
1. **Robustheitsprüfung:** Ist +1.420 % stabil? Sensitivity-Analyse: andere Random-Seeds, Asset-Subsets, leicht veränderte Hyperparameter
2. **Fold-Stabilität verbessern:** Fold 0 und 3 zeigen negativen IC. Ursache analysieren (Regime-Wechsel? Zu wenig Trainingsdaten?)
3. **Hard-Stop-Analyse:** Nur 14 Stops, aber -29,76 % Ø-Verlust. Engere Grenze (z.B. -20 %) oder dynamischer ATR-Stop (vorsichtig — hat historisch geschadet)

### Mittelfristig
4. **Weitere Horizonte testen:** 5d, 9d, 3d — gibt es ein noch schärferes Signal unter 7 Tagen?
5. **Ensemble:** Kombination v2_7d + v2_11d → gemeinsames Ranking (gewichtetes Voting)
6. **Regime-Erkennung verbessern:** Aktuell nur SMA50/200 auf SPY. Alternative: VIX, Breadth-Indikatoren
7. **Universum erweitern:** MSCI ACWI (~3000 Assets) oder sector-ETF-Overlay

### Langfristig
8. **Live-Trading (Paper):** Wenn 2–3 weitere unabhängige Backtest-Perioden ähnliche Ergebnisse zeigen
9. **Feature-Erweiterung:** Makro-Daten (Zinsen, VIX), alternative Daten (Sentiment), fundamentale Kennzahlen

### Offene Probleme
- **Overfitting-Frage:** +1.420 % ist außergewöhnlich hoch. Ist das echtes Alpha oder in-sample Overfit? → Weitere Walk-Forward-Perioden notwendig
- **Hard-Stop-Kosten:** 14 Stops × Ø -30 % bleiben das größte Einzelrisiko. Bisher kein besserer Ersatz gefunden.

---

## 17. Glossar

| Begriff | Bedeutung |
|---------|-----------|
| **Cross-Sectional Ranking** | Tägliches Sortieren aller Assets nach Modell-Score; Position in die Top-N |
| **Walk-Forward** | Expanding-Window Training: Trainingsdaten wachsen, Validierung rollt vorwärts |
| **Rank IC** | Spearman-Korrelation zwischen Prediction und tatsächlichem Return |
| **Embargo** | Zeitpuffer zwischen Train-Ende und Val-Start (verhindert Lookahead) |
| **Rotation** | Täglicher Austausch von Positionen basierend auf aktuellem Ranking |
| **Hard-Stop** | Fester maximaler Verlust vom Einstiegskurs (25%) als Gap-Down-Schutz |
| **Regime-Filter** | SMA50/SMA200-basierte Marktphasen-Erkennung (Bull/Neutral/Bear) |
| **n_max** | Maximale Anzahl gleichzeitiger Positionen im Bull-Regime |
| **rotation_buffer** | Toleranz bevor eine Position wegen schlechtem Rang rotiert wird |
| **ATR** | Average True Range — Volatilitätsmaß für Stop-Level |
| **CS-Z-Score** | Cross-Sectional Z-Score — tägliche Normalisierung über alle Assets |
| **Huber Loss** | Robustere Alternative zu MSE: quadratisch nahe 0, linear für Ausreißer |
| **Isotonic Regression** | Monoton steigende, nichtlineare Kalibrierung |
