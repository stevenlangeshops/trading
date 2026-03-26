# Trading Bot v5 — Vollständige Projektdokumentation

**Stand:** 26. März 2026
**Repo:** `https://github.com/stevenlangeshops/trading.git`
**Kaggle User:** `busersteven`
**Kaggle Dataset:** `busersteven/trading-raw-data` (260 Parquet-Dateien)
**Kaggle Results:** `busersteven/trading-results` (permanente Ergebnisspeicherung)

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

## 3. Zwei Modellvarianten

### 3.1 v1_rank (Run G — bisheriges Produktionsmodell)

- **Modell:** `CrossSectionalLSTM` (models/lstm_model.py)
- **Output:** 1 Score pro Asset/Tag (unkalibriert, korreliert mit 11d-Forward-Return)
- **Loss:** MSE + 0.5 × PairwiseRankLoss
- **Training:** models/trainer.py → `train_walk_forward()`
- **Backtest:** strategy/backtest.py → `run_backtest()`
- **Bestes Ergebnis (Run G):**
  - Total Return: **+403.93%**
  - Max Drawdown: **-55.48%**
  - Sharpe: **0.784**
  - Trades: 471, Win-Rate: 55.2%, Avg Hold: 14.8 Tage

### 3.2 v2_return_multi (neu, in Erprobung)

- **Modell:** `LSTMReturnMultiV2` (models_v2_return_multi.py)
- **Output:** 4 erwartete Returns pro Asset/Tag (4d, 7d, 11d, 15d)
- **Loss:** gewichteter Huber (4 Horizonte) + 0.1 × PairwiseRankLoss (auf 11d)
- **Training:** train_v2_return_multi.py → `train_walk_forward_v2()`
- **Backtest:** backtest_v2_return_multi.py → `run_backtest_v2()`
- **Ziel:** Kalibrierte Return-Vorhersagen ermöglichen einen Filter ("nicht handeln wenn E[ret] < 0")
- **Status:** Erster Testlauf mit 20 Assets erfolgreich, Full-Run mit 260 Assets läuft

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

### 4.2 v2 Multi-Horizon Module

| Datei | Beschreibung |
|-------|-------------|
| `config_v2_return_multi.py` | `V2Config` Dataclass: alle Parameter zentral |
| `models_v2_return_multi.py` | `LSTMReturnMultiV2` + `CombinedMultiHorizonLoss` |
| `train_v2_return_multi.py` | Multi-Horizon Targets + Dataset + Walk-Forward Training |
| `backtest_v2_return_multi.py` | Backtest + v1-vs-v2 Report + Vergleichs-Plot |

### 4.3 Kaggle-Integration

| Datei | Beschreibung |
|-------|-------------|
| `kaggle_notebook.ipynb` | 2-Zellen-Notebook: lädt `kaggle_full_run.py` per wget + exec |
| `scripts/kaggle_full_run.py` | Komplette Pipeline (Schritte 1–12), Modus-Schalter V2_ONLY/V2_MAX_ASSETS |
| `scripts/kaggle_kernel_api.py` | CLI-Wrapper für Kaggle API (Push, Poll, Download) |
| `scripts/kaggle_watch.py` | Autonomer Job-Watcher (Poll, Diagnose, Auto-Resubmit) |
| `scripts/kaggle_status.py` | Terminal-Dashboard für Watcher-Status |

### 4.4 Daten

| Datei | Beschreibung |
|-------|-------------|
| `data/asset_list_sp500.txt` | 260 S&P-500-Aktien + ETFs (SPY, QQQ, IWM, GLD, TLT, etc.) |
| `data/download_stocks.py` | yfinance → Parquet (parallelisiert via ThreadPool) |
| `data/raw/dataset-metadata.json` | Kaggle Dataset Metadaten (`trading-raw-data`) |

### 4.5 Tests & Sonstiges

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

### Hauptergebnisse (260 Assets, ~2020–2026)

| Run | Config | Total Return | Max DD | Sharpe | Trades | Kern-Erkenntnis |
|-----|--------|-------------|--------|--------|--------|----------------|
| **D** | 260 Assets, GPU, Stop-Loss 15% | +175% | -60% | 0.582 | 520 | Hard-Stop zu eng, Short-Term-Filter kontraproduktiv |
| **E** | Corr-Filter, Risk-Parity, ATR k=2.5 | +29.7% | -60% | — | — | Alle neuen Mechanismen schadeten der Performance |
| **F** | ATR k=3.5, DD-Control 20/30%, n=7 | +17.9% | -68% | 0.298 | 248 | ATR-Stop (-704% PnL), DD-Control zu sensitiv |
| **G** | Pure Rotation + Hard-Stop 25% | **+403.9%** | -55.5% | **0.784** | 471 | **Bestes Ergebnis.** Einfachheit gewinnt |
| **H1** | G + SPY-ATR Crash-Schutz | +335.3% | -54.5% | 0.724 | 406 | Marginale DD-Verbesserung, deutlicher Return-Verlust |
| **H2 Signal** | G + Signal-Spread-Filter | -18.3% | -42% | -0.028 | 108 | Katastrophal: Filter 86% aktiv, invertierter Effekt |
| **G_calib** | G + kalibrierter E[ret]-Filter ≥0% | +403.9% | -55.5% | 0.784 | 471 | Filter nie ausgelöst (E[ret] immer >0.5%) |

### Benchmarks (gleicher Zeitraum)

| Benchmark | Return | Sharpe | Max DD |
|-----------|--------|--------|--------|
| SPY Buy & Hold | +60.6% | 0.452 | -37.4% |
| EW Universe Buy & Hold | +192.8% | 0.940 | -34.6% |
| EW Universe Rebalanciert | +167.3% | 0.937 | -34.9% |

### Kern-Erkenntnis

Das reine Rotationssignal des LSTM ist stark (Rotation: +1273% PnL, 58% Win-Rate). Alle Versuche, zusätzliche Schutzschichten einzubauen, haben das Alpha reduziert statt erhöht. Der einzige verbleibende Verlustbringer ist der Hard-Stop (-779% PnL, 28 Trades, 0% Win-Rate), der als Gap-Down-Schutz dennoch sinnvoll ist.

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

## 12. Git-Historie (letzte 25 Commits)

```
ebcbea6 v2: volle Parameter (50 Ep, hidden=128) wie Run G
22e0d94 v2 Full Run: alle 260 Assets, V2_MAX_ASSETS=0
0fa61b3 v2 Test-Modus: nur v2, 20 Assets, CPU-optimierte Parameter
6fadc73 v2_return_multi: Multi-Horizon Return-Modell (4/7/11/15d) mit Training, Backtest und v1-Vergleich
6087674 Run G_calib: Score-to-Return Kalibrierung + Expected-Return-Filter
f6c78a3 fix: Cache-Buster fuer raw.githubusercontent + sys.modules clear im Notebook
35130cc fix: sys.modules Cache leeren vor Backtest-Import (Kaggle Re-Run)
f671f73 feat: Signal-Diagnose-Plot (4-Panel: Equity+Spread+Std+Positionen)
ba0e6c0 feat: daily_signals Rohdaten pro Handelstag speichern (JSON)
1b1d6ed feat: Signalstaerke-Filter (Spread Top1-Median) + plot_equity fix
ad78141 fix: plot_equity crash bei leerem result_b (Long-Short deaktiviert)
b3ae93a feat: Expected-Return-Filter + Low-Pred-Exit
d6ac40a feat: Run H2 - 3-Tage Crash-Signal Analyse (kein Backtest, nur SPY-Signal)
a845f47 feat: Run H1 - SPY-ATR Crash-Schutz (Halbgas-Modus) + nur Long-Only
544a8d3 feat: Run G - use_dd_control=False (Baseline), DD-Tracking getrennt von Control
9e839ff feat: Run G - ATR-Trailing deaktiviert, DD-Schwellen 25%/40%
ab9e61d feat: Run F - DD-Control + erweitertes Exit-Reporting
a71e0a9 feat: Run F - stop_loss_pct entfernt, corr_cap deaktiviert, n_max=7
b41e23a feat: Run E - Korrelations-Cap + Risk-Parity-Sizing
f0594e7 fix: Regime-Filter revert auf SMA-only, Hard-Stop auf 25pct
d68f8a2 fix: Hard-Stop 15pct + Regime-Kurzzeit-Drawdown-Filter
870aefc fix: Benchmark-Timezone und GPU-Flag nach Notebook-Rerun
54dddec feat: S&P500 Expansion 260 Assets und Checkpoint-Kompatibilitaetspruefung
a31e3b4 fix: Timezone-Fehler in compute_benchmarks + ATR-Log-Text
4a1f90d fix: CUDA-Check Hauptprozess-Fallback + Checkpoint-Pfade fuer notebook Add-Data
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

### Modus-Schalter in kaggle_full_run.py

```python
V2_ONLY       = True    # True = nur v2, False = v1 + v2
V2_MAX_ASSETS = 0       # 0 = alle Assets, >0 = Subset für Tests
```

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

## 16. Nächste Schritte (Stand März 2026)

1. **v2 Full-Run auswerten:** Ergebnisse des 260-Asset-CPU-Laufs analysieren und mit Run G vergleichen
2. **v2 mit GPU:** Wenn GPU-Tokens verfügbar, Full-Run mit vollen Parametern wiederholen
3. **Return-Filter testen:** Wenn v2-Kalibrierung besser ist als v1, Expected-Return-Filter mit Schwelle >0 testen
4. **Combo-Score testen:** `0.5 × pred_7d + 0.5 × pred_11d` als alternatives Ranking-Signal
5. **Universum erweitern:** Skalierung auf MSCI ACWI (~3000 Assets)
6. **Hard-Stop Alternative:** Offenes Problem — besseres Tail-Risk-Management finden
7. **Live-Trading Vorbereitung:** Wenn Backtest-Ergebnisse stabil, Übergang zu Paper-Trading

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
