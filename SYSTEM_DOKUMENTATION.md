# Trading-Bot v5 – Systemdokumentation

**Stand:** Mai 2026  
**Autor:** Quant-Engineering-Prozess (Kaggle → Hetzner)  
**Status:** Produktiv (Paper-Trading, Alpaca)

---

## Inhaltsverzeichnis

1. [Executive Summary](#1-executive-summary)
2. [Systemarchitektur](#2-systemarchitektur)
3. [Datenpipeline](#3-datenpipeline)
4. [Feature-Engineering](#4-feature-engineering)
5. [Modell-Architektur](#5-modell-architektur)
6. [Training & Walk-Forward-Validierung](#6-training--walk-forward-validierung)
7. [Produktions-Ensemble](#7-produktions-ensemble)
8. [Portfolio-Konstruktion & Backtesting](#8-portfolio-konstruktion--backtesting)
9. [A3-Policy (Risikoreduktion bei negativem IC)](#9-a3-policy-risikoreduktion-bei-negativem-ic)
10. [Live-Inference-Pipeline](#10-live-inference-pipeline)
11. [Execution-Layer (Alpaca)](#11-execution-layer-alpaca)
12. [Monitoring & Notifications](#12-monitoring--notifications)
13. [Erkenntnisse aus der Modellentwicklung](#13-erkenntnisse-aus-der-modellentwicklung)
14. [Vollständige Parameterübersicht](#14-vollständige-parameterübersicht)
15. [Infrastruktur & Deployment](#15-infrastruktur--deployment)

---

## 1. Executive Summary

Der Trading-Bot v5 ist ein vollautomatisches Aktien-Ranking-System das täglich die 5 attraktivsten Aktien aus einem kuratierten Universum von 260 US-Titeln (S&P 500-Ausschnitt) identifiziert und über Alpaca Paper/Live-Trading ausführt.

**Kernprinzipien:**

| Aspekt | Entscheidung |
|--------|-------------|
| Vorhersage-Ziel | 7-Tage-Forward-Return (Ranking, kein Kursziel) |
| Modell | LSTM + Temporal Attention (Sequenzmodell auf Tageskerzen) |
| Normalisierung | Sektor-neutrale Z-Scores (GICS-Sektoren) |
| Portfolio | Equal-Weight, 5 Positionen, 20 % Hard-Stop |
| Risikopolicy | A3: IC_roll_40 < 0 → Positionsreduktion auf 3 |
| Training | Walk-Forward (12 Folds), deterministisch (seed_everything) |
| Produktionsmodus | Ensemble aus 5 Modellen (Seeds 42–46), alle Daten |
| Ausführung | Alpaca Paper-Trading, täglich 15:45 CET (15 min nach US-Open) |
| Monitoring | Telegram-Report mit Chart, P&L, Performance vs. MSCI World |

---

## 2. Systemarchitektur

```
┌────────────────────────────────────────────────────────────┐
│                    DATEN (lokal / Kaggle)                   │
│  data/raw/*.parquet  ←  update_raw_data.py  ← yfinance     │
└────────────────────────┬───────────────────────────────────┘
                         │ 260 Ticker, täglich OHLCV
                         ▼
┌────────────────────────────────────────────────────────────┐
│                 TRAINING (Kaggle GPU)                       │
│  features/engineer.py  →  train_v2_single_horizon.py       │
│  Walk-Forward (12 Folds)  oder  Produktions-Ensemble (5×)  │
│  → checkpoints/production/prod_model_seed{42..46}.pt       │
└────────────────────────┬───────────────────────────────────┘
                         │ Modelle (.pt)
                         ▼
┌────────────────────────────────────────────────────────────┐
│              LIVE-INFERENCE (Hetzner VPS)                   │
│  live_inference.py                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Download │→ │ Features │→ │ Ensemble │→ │ A3-Check │  │
│  │ yfinance │  │ (sekt.-n)│  │ (5 Mdl.) │  │IC_roll_40│  │
│  └──────────┘  └──────────┘  └──────────┘  └────┬─────┘  │
└───────────────────────────────────────────────────┼────────┘
                                                    │ Top-5 Ticker
                                                    ▼
┌────────────────────────────────────────────────────────────┐
│              EXECUTION (Alpaca Paper-Trading)               │
│  alpaca_broker.py                                           │
│  A) Verkauf Off-Target  B) Equal-Weight Sizing              │
│  C) Kauf/Rebalancing    D) GTC Stop-Loss (−20%)            │
└────────────────────────┬───────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────┐
│              MONITORING (Telegram)                          │
│  notifier.py: Chart (Portfolio vs. MSCI World) + P&L       │
└────────────────────────────────────────────────────────────┘
```

---

## 3. Datenpipeline

### 3.1 Universum

- **260 Ticker** aus dem S&P 500 (kuratiert, liquide Large-/Mid-Caps)
- Ausschluss von Titeln die im Trainingszeitraum delistet wurden
- Bekannte Delistings/Übernahmen (Stand 2026): ANSS, DRE, HES, K, MMC, MRO, PXD

### 3.2 Rohdaten

```
Quelle:       yfinance (auto_adjust=True – Split- und Dividenden-bereinigt)
Format:       Parquet, ein File pro Ticker in data/raw/
Felder:       open, high, low, close, volume (Spalten lowercase)
Zeitraum:     ~2015–heute (≈10 Jahre Tageskerzen)
Update:       update_raw_data.py (inkrementell, min_rows=1)
```

### 3.3 Daten-Update-Prozess

```python
# update_raw_data.py
# Inkrementell: nur fehlende Tage seit letztem Parquet-Eintrag
run_update(min_rows=1)          # delta-Download
# oder: Vollständige Neuanlage
run_full_download()
```

**Bekannte Fixes:**
- Timezone-Strip vor Datumsvergleich (`tz_localize(None)`) um `TypeError: Cannot compare tz-naive and tz-aware` zu vermeiden

---

## 4. Feature-Engineering

### 4.1 Technische Indikatoren

Alle Features werden auf **täglichen Schlusskursen** berechnet (konsistent mit Training und Inference):

| Feature | Berechnung | Kategorie |
|---------|-----------|-----------|
| `sma_ratio_20` | Close / SMA(20) | Trend |
| `sma_ratio_50` | Close / SMA(50) | Trend |
| `sma_ratio_200` | Close / SMA(200) | Trend |
| `ema_ratio_12` | Close / EMA(12) | Trend |
| `macd_diff` | MACD-Histogramm | Momentum |
| `rsi_14` | RSI(14) / 100 | Momentum |
| `roc_5` | Return über 5 Tage | Momentum |
| `roc_21` | Return über 21 Tage | Momentum |
| `stoch_k` | Stochastic %K / 100 | Momentum |
| `atr_ratio` | ATR(14) / Close | Volatilität |
| `bb_width` | (BB_upper − BB_lower) / Close | Volatilität |
| `bb_pos` | Position in Bollinger-Bändern [0,1] | Volatilität |
| `volume_ratio_20` | Volume / SMA(Volume, 20) | Volumen |
| `obv_diff` | OBV pct_change, geclippt auf [−1, 1] | Volumen |
| `high_low_ratio` | (High − Low) / Close | Volatilität |
| `ret_1d` | 1-Tage-Return | Return |
| `ret_5d` | 5-Tage-Return | Return |
| `ret_21d` | 21-Tage-Return | Return |

**Gesamt: 18 Features + 1 Asset-Embedding = 34 Eingabe-Dimensionen (16 Embed-Dim)**

### 4.2 Normalisierung

**Gewählte Methode: Sektor-neutrale Z-Scores (Produktions-Default)**

```
Für jeden Tag t und jeden GICS-Sektor s:
  z_i = (feature_i − μ_{s,t}) / σ_{s,t}

Bedingungen:
  - Mindestens 3 Assets pro Sektor/Tag (sonst globaler Fallback)
  - Mindestens 5 Assets global für Fallback
  - Clip auf ±4 σ (Outlier-Unterdrückung)
```

**Alternative: Cross-Sectional Z-Score (verworfen für Produktion)**

```
z_i = (feature_i − μ_t) / σ_t  (über alle Assets, ohne Sektor-Gruppierung)
```

**Erkenntnis:** Sektor-neutrale Normalisierung reduziert Sektor-Bias und führt zu einem robusteren Signal (insbesondere in sektoralen Rotationsphasen).

### 4.3 Trainings-Target

```python
fwd_ret = close.pct_change(horizon=7).shift(-7)
# Forward-Return: Rendite der nächsten 7 Handelstage
# Kein Look-Ahead: shift(-horizon) nur in Training, nicht in Live-Inference
```

---

## 5. Modell-Architektur

### 5.1 `SingleHorizonRankModel`

```
Input: (batch, seq_len=64, n_features=18)
       + asset_id → nn.Embedding(n_assets, embed_dim=16)

Concatenation: (batch, seq_len, 18+16=34)
       ↓
nn.LSTM(input=34, hidden=128, layers=2, dropout=0.3)
       ↓ (batch, seq_len, 128)
TemporalAttention
  Linear(128→1) → Softmax(dim=seq_len) → gewichtete Summe
       ↓ (batch, 128)
nn.LayerNorm(128)
       ↓
MLP-Head:
  Linear(128→64) → GELU → Dropout(0.3)
  Linear(64→32)  → GELU → Dropout(0.15)
  Linear(32→1)           → Score (unbegrenzt)
       ↓ (batch, 1)
```

### 5.2 Gewichtsinit

| Gewicht | Initialisierung |
|---------|----------------|
| `weight_ih_*` (LSTM Input) | Xavier Uniform |
| `weight_hh_*` (LSTM Hidden) | Orthogonal |
| Forget-Gate Bias | 1.0 (Standard LSTM-Trick) |
| Lineare Layer | Xavier Uniform |
| Alle Biases | 0.0 |

### 5.3 Loss-Funktion

```
CombinedRankLoss = MSE(pred, target) + λ_rank × PairwiseRankLoss

PairwiseRankLoss: Für alle Paare (i,j) mit target_i - target_j > 0.001:
  loss += max(0, margin - (pred_i - pred_j))
  margin = 0.001
  λ_rank = 0.5
```

**Rationale:** Reines MSE würde die Ranking-Ordnung nicht direkt optimieren. Der Pairwise-Term stellt sicher dass das Modell die relative Reihenfolge der Assets lernt.

### 5.4 Evaluationsmetrik

```python
rank_ic = spearmanr(predictions, targets).correlation
# Spearman-Rang-Korrelation zwischen vorhergesagten Scores und realisierten Returns
# Zielwert: IC > 0.035 (v1-Referenz)
# Typisch in Produktion: IC_roll_40 ≈ +0.07
```

---

## 6. Training & Walk-Forward-Validierung

### 6.1 Walk-Forward-Schema

```
Datenzeitraum: 2015–2026 (ca. 11 Jahre, 252 Handelstage/Jahr)

Fold-Struktur (Expanding Window):
  Fold 0:  Train [2015–2018], Val [2018–H1 2019]
  Fold 1:  Train [2015–H2 2018], Val [H2 2019]
  ...
  Fold 11: Train [2015–H2 2024], Val [H1 2025]

Parameter:
  train_years  = 3.0  (Mindest-Trainingslänge; expandiert)
  val_months   = 6.0  (Validierungsfenster)
  step_months  = 6.0  (Schrittweite → 2 Folds pro Jahr)
  Gesamt:      12 Folds
```

### 6.2 Training-Setup

| Parameter | Wert |
|-----------|------|
| Optimizer | AdamW |
| Lernrate | 5 × 10⁻⁴ |
| Weight Decay | 1 × 10⁻³ |
| LR-Scheduler | CosineAnnealingLR (T_max=50, η_min=lr/100) |
| Batch Size | 512 |
| Gradient Clipping | 1.0 |
| Epochs | 50 |
| Early Stopping | Patience 7 (auf Val-Loss) |
| DataLoader Shuffle | Generator mit Fold-Seed |

### 6.3 Determinismus (`seed_everything`)

```python
def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
```

**Pro Fold:** `seed_everything(cfg.seed + fold_id)` – jeder Fold ist isoliert reproduzierbar.

---

## 7. Produktions-Ensemble

### 7.1 Motivation

Walk-Forward-Validierung liefert Out-of-Sample-Metriken – für den Live-Betrieb werden jedoch alle verfügbaren Daten genutzt. Das Ensemble aus 5 Modellen (verschiedene Seeds) reduziert Modellrauschen.

### 7.2 Training

```python
ensemble_seeds = [42, 43, 44, 45, 46]
# Pro Seed:
#   1. seed_everything(seed)
#   2. Modell auf ALLEN verfügbaren Daten trainieren (kein Val-Split)
#   3. Speichern: checkpoints/production/prod_model_seed{seed}.pt
```

**Kein traditioneller OOS-Backtest möglich** da alle Daten im Training genutzt wurden. Qualitätskontrolle über Train-Loss-Konvergenz.

### 7.3 Inference

```python
# Alle 5 Modelle laden, unabhängig scoren, Scores mitteln:
score_ensemble = mean([model_i.score(features) for model_i in ensemble])

# Mindest-Quorum: mind. 50 % der Modelle müssen einen Ticker scoren
# (robuster gegen Datenlücken einzelner Modelle)
```

---

## 8. Portfolio-Konstruktion & Backtesting

### 8.1 Portfolio-Regeln (Backtest & Live)

| Regel | Parameter | Begründung |
|-------|-----------|-----------|
| Max. Positionen | `n_max = 5` | Grid-Search-Sieger |
| Gewichtung | Equal-Weight (je 20%) | Einfach, robust |
| Rotation Buffer | 2 | Position hält bis Rank > n_max+2, reduziert Turnover |
| Hard Stop | −20% | Kapitalschutz bei Trending-Verlierern |
| Handelsgebühren | 0.1% pro Trade | Konservative Schätzung |
| Startkapital | $10,000 (Backtest) | Vergleichsbasis |

### 8.2 Backtest-Ergebnisse (Walk-Forward-Validierung)

Aus der Entwicklungsphase (April 2026, Fold 0–11):

| Konfiguration | Sharpe | Max DD | Return |
|---------------|--------|--------|--------|
| Seed 42 (kein seed_everything) | 0.553 | — | — |
| Seed 1 | — | — | — |
| Seed 7 | — | — | — |
| **seed_everything + Seed 42** | **~0.961** | — | — |
| Cross-Sectional Z-Score (Ref.) | 0.553 | — | — |
| **Sektor-neutral Z-Score** | **besser** | — | — |

**Beobachtungen:**
- Ohne `seed_everything` stark schwankende Ergebnisse (Seed 42 ohne: 0.553 vs. mit: ~0.961)
- Fold 11 (2. Halbjahr 2024 – H1 2025) zeigte gelegentlich negative IC-Werte → A3-Policy designt
- Sektor-neutrale Normalisierung konsistent besser als Cross-Sectional

### 8.3 IC-History (Live-Betrieb)

```
Datei: rolling_ic_v2_7d.csv
Update: update_ic.py (täglich, nach Inference)
IC_roll_40 = rolling mean der letzten 40 IC-Werte
Aktuell (15.04.2026): IC_roll_40 = +0.0698
```

---

## 9. A3-Policy (Risikoreduktion bei negativem IC)

### 9.1 Motivation

Wenn das Modell in einem Marktregime systematisch schlechte Vorhersagen liefert (negativer rollender IC), sollte die Positionsgröße reduziert werden.

### 9.2 Regel

```
Wenn IC_roll_40 < 0:
  → n_max reduzieren: 5 → 3 (PROD_POLICY_REDUCED_N)
  → Weniger Positionen = weniger Exposure in schwachem Regime

Wenn IC_roll_40 ≥ 0:
  → Normalbetrieb: n_max = 5
```

### 9.3 Früheres Budget-Policy-Konzept (verworfen)

In der Forschungsphase wurde eine mehrstufige Budget-Reduktion getestet:
- IC_roll_20 < 0 → 30% weniger investiert Kapital
- IC_roll_30 < 0 → weitere 30% Reduktion
- IC_roll_40 < 0 → weitere 30% Reduktion

**Entscheidung:** Zu komplex, zu viele Parameter. Vereinfacht zu binärer n_max-Reduktion (A3-Policy).

---

## 10. Live-Inference-Pipeline

### 10.1 Ablauf (täglich, 15:45 CET)

```
[1/5] Metadaten laden
      → sector_map.json (260 Ticker + GICS-Sektor)
      → asset_map.json  (Ticker → Model-ID, alphabetisch, 1-basiert)

[2/5] OHLCV-Download
      → yfinance, ~320 Handelstage Lookback
      → end_date = letzter Handelstag (nicht heute! Markt noch offen)
      → Bekannte Delistings werden automatisch ausgeschlossen

[3/5] Feature-Engineering
      → compute_indicators() – alle 18 technischen Features
      → sector_neutral_zscore() – pro Tag+Sektor normalisiert
      → Lookback: seq_len=64 Handelstage pro Asset

[4/5] Ensemble-Scoring
      → 5 Modelle laden aus checkpoints/production/
      → score_universe() pro Modell → ungewichteter Mittelwert
      → Mindest-Quorum: 50% der Modelle
      → Fallback: letzter verfügbarer Tag wenn target_date nicht im Panel

[5/5] A3-Policy + Output
      → IC_roll_40 aus rolling_ic_v2_7d.csv
      → n_eff = 3 wenn IC_roll_40 < 0, sonst 5
      → Top-n_eff Ticker nach Score → Ziel-Allokation
```

### 10.2 Wichtige Design-Entscheidungen

**Welcher Kursstand geht ins Scoring?**

> Immer der **gestrige Schlusskurs** (US-Schlusskurs ~22:00 CET).

yfinance liefert am laufenden Handelstag keine vollständigen Tageskerzen. Das ist korrekt und konsistent mit dem Training (Modell wurde auf Tagesschlusskursen trainiert). Das Gap-Risiko (Eröffnungskurs weicht stark vom Vortagsschluss ab) wird durch den Gap-Filter im Execution-Layer abgefangen.

**Timing: Warum 15:45 CET?**

| Zeitpunkt | Problem |
|-----------|---------|
| 15:30 CET (direkt bei Open) | Opening Auction: breite Spreads, hohe Volatilität, schlechte Ausführung |
| **15:45 CET (15 min nach Open)** | **Spreads enger, Preisfindung stabil, noch früh im Handelstag** |
| 16:10 CET (vorher) | Unnötig spät, kein Vorteil gegenüber 15:45 |

### 10.3 `LiveConfig` – Konfigurationsparameter

| Parameter | Wert | Bedeutung |
|-----------|------|-----------|
| `horizon` | 7 | Vorhersage-Horizont (Tage) |
| `seq_len` | 64 | LSTM-Lookback-Fenster |
| `download_days` | 320 | yfinance-Downloadfenster (≥ 200 + SMA200-Warmup) |
| `top_n` | 5 | Anzahl Ziel-Positionen (Standard) |
| `a3_policy_window` | 40 | Rolling-IC-Fenster für A3-Policy |
| `a3_reduced_n` | 3 | Positionen wenn A3-Policy aktiv |
| `prod_ckpt_dir` | `checkpoints/production/` | Ensemble-Modelle |
| `predictions_csv` | `live_predictions_history.csv` | Tägliche Scoring-History |
| `ic_history_csv` | `rolling_ic_v2_7d.csv` | IC-Rolling-History |

---

## 11. Execution-Layer (Alpaca)

### 11.1 Schritt A – Verkauf

```
Für jede aktuell gehaltene Position:
  Wenn Ticker NICHT in Ziel-Liste:
    → api.close_position(symbol)  [Market-äquivalent, sofortige Ausführung]
→ Danach: 1 Sekunde warten (Settlement-Update)
```

### 11.2 Schritt B – Sizing (Equal-Weight)

```
target_value = account.equity / len(target_tickers)
Beispiel: $100,000 / 5 = $20,000 pro Position
```

### 11.3 Schritt C – Kauf / Rebalancing

```
Für jeden Ziel-Ticker:
  price = api.get_snapshot(symbol).latest_trade.price
  target_qty = floor(target_value / price)    [nur ganze Aktien]
  delta = target_qty - current_qty

  ── Gap-Filter (nur bei Neukäufen) ──────────────────
  gap = (price - yesterday_close) / yesterday_close
  Wenn |gap| > 5%:
    → Kauf überspringen (zu hohes Gap-Risiko)
  ────────────────────────────────────────────────────

  delta > 0 → Market-Buy für delta Stück
  delta < 0 → Market-Sell für |delta| Stück (Rebalancing)
  delta = 0 → keine Order
```

**Symbol-Mapping:** Bestimmte Ticker haben bei Alpaca andere Bezeichnungen:
- `BF-B` (yfinance) → `BF.B` (Alpaca)
- `BF-A` (yfinance) → `BF.A` (Alpaca)

### 11.4 Schritt D – Hard-Stop-Loss

```
Für jede offene Position nach Rebalancing:
  1. Bestehende GTC-Stop-Orders für Symbol stornieren
  2. stop_price = avg_entry_price × 0.80  (= −20%)
  3. Sicherheitscheck: stop_price < current_price
     (sonst Position bereits im Hard-Stop-Bereich → skip)
  4. StopOrderRequest(symbol, qty, SELL, stop_price, GTC) → submit
```

**Eigenschaften der Stop-Loss-Orders:**
- **GTC (Good Till Cancelled):** Bleibt aktiv bis Ausführung oder manuelle Stornierung
- **Stop-Market:** Wird zu Market-Order wenn Triggerpreis erreicht
- **Tägliches Reset:** Bei jedem Rebalancing werden alte Stops storniert und neue gesetzt (basierend auf aktuellem Einstiegspreis)
- **Automatisch:** Alpaca führt die Order auch ohne laufenden Bot aus

### 11.5 Execution-Parameter

| Parameter | Wert | Bedeutung |
|-----------|------|-----------|
| `stop_loss_pct` | 0.20 | Hard-Stop bei −20% vom Einstieg |
| `gap_filter_pct` | 0.05 | Max. Gap ±5% für Neukäufe |
| `sell_delay_s` | 1.0 | Pause nach Verkäufen (Settlement) |
| Handelstyp | Market | Sofortige Ausführung |
| Orderbörse | Alpaca (Paper) | `https://paper-api.alpaca.markets` |

---

## 12. Monitoring & Notifications

### 12.1 Telegram-Bot

**Bot:** `@Stevens_trading_bot`

**Täglicher Report** (nach jedem Inference-Lauf):

```
Trading-Bot Report | YYYY-MM-DD
──────────────────────────────────
Portfoliowert:  $100,653.30
Gesamt P&L:     +$653  (+0.65%)
Tages-P&L:      +$23  (+0.02%)
Cash:           $0.00
──────────────────────────────────
Positionen P&L:
  EWY    +5.2%  (+$1,234)
  VFC    -1.3%  (-$234)
  ...
──────────────────────────────────
Performance vs. MSCI World:
  24h:  Portfolio +0.3% | MSCI +0.1%
  1W:   Portfolio +1.2% | MSCI +0.8%
  1M:   Portfolio +3.4% | MSCI +2.1%
  1Y:   Portfolio n/a   | MSCI n/a
──────────────────────────────────
Ziel-Allokation (5 Positionen):
  1. VFC   Score=+0.0453
  ...
Stop-Loss Orders (20%):
  EWY  Stop @ $124.39
  VFC  Stop @ $14.94
```

**Chart-Bild:**
- Oben: Portfolio-Equity vs. MSCI World (URTH), normiert auf 100 ab Depoteröffnung (29.04.2026)
- Unten: P&L-Balken je gehaltener Position

**Fehler-Benachrichtigung:** Bei Absturz des Bots kommt sofort eine Telegram-Fehlermeldung mit Stack-Trace.

### 12.2 Logfiles

```
/opt/trading/logs/inference_YYYYMMDD.log  – täglich pro Lauf
/opt/trading/logs/cron.log                – Cron-Ausgaben
Rotation: 30 Tage, täglich komprimiert
```

---

## 13. Erkenntnisse aus der Modellentwicklung

### 13.1 Feature-Engineering: Sektor-Neutral vs. Cross-Sectional

**Experiment:** Direkter A/B-Vergleich (identische Portfolio-Parameter: n_max=5, rb=2, hard_stop=20%, A3-Policy):

**Ergebnis:** Sektor-neutrale Z-Scores → bessere Sharpe und konsistentere IC-Werte. Begründung: Cross-Sectional-Normalisierung enthält Sektor-Bias (z.B. Energie vs. Tech in unterschiedlichen Marktphasen); sektor-neutrale Normalisierung macht das Signal „Welcher Titel ist gut **innerhalb seines Sektors**?" berechenbar.

**Produktionsentscheidung:** `sector_neutral = True` (permanent)

### 13.2 Portfolio-Parameter: Grid-Search-Ergebnisse

Aus dem Sensitivity-Modul (`run_sensitivity.py`), Multi-Parameter-Grid:

| n_max | Rotation-Buffer | Ergebnis |
|-------|----------------|---------|
| 3 | 2 | unterdiversifiziert, hohe Volatilität |
| 5 | 2 | **bestes Sharpe-Ratio** ✓ |
| 5 | 4 | höherer Turnover, schlechtere Netto-Performance |
| 7 | 2 | zu viele Positionen, Alphaverwässerung |

**Produktionsentscheidung:** `n_max=5`, `rotation_buffer=2`

### 13.3 Trainings-Determinismus & Seed-Evaluation

**Problem:** Ohne feste Seeds variierten Ergebnisse erheblich zwischen Trainingsdurchläufen:
- Seed 42 (ohne `seed_everything`): Sharpe ~0.553
- Dieselbe Konfiguration mit `seed_everything(42)`: Sharpe ~0.961

**Ursache:** Non-deterministische CUDA-Operationen (CuDNN), zufällige Batch-Reihenfolge ohne Generator, Python-Hash-Randomness.

**Fix:** `seed_everything()` deckt alle Zufallsquellen ab:
- Python `random`, `PYTHONHASHSEED`
- NumPy
- PyTorch CPU + CUDA
- `cudnn.deterministic = True`, `cudnn.benchmark = False`

**Seed-Evaluation (Walk-Forward, alle Folds):**

| Seed | Besonderheit | Empfehlung |
|------|-------------|-----------|
| 42 | Balanced, stabil | **Produktions-Seed** ✓ |
| 1 | Leicht variabel | Ensemble-Kandidat |
| 7 | Moderate Varianz | Ensemble-Kandidat |

**Produktionsentscheidung:** Ensemble mit Seeds [42, 43, 44, 45, 46]

### 13.4 IC-Analyse über Folds

- **Folds 0–10:** Konsistent positive IC-Werte (Spearman IC > 0)
- **Fold 11** (H2 2024 – H1 2025): Gelegentlich negative IC-Werte
  - Marktregime-Wechsel (starke Mega-Cap-Konzentration, breiterer Markt schwach)
  - → Designte A3-Policy als strukturelle Antwort

### 13.5 A3-Policy: Budget-Konzept vs. n_max-Reduktion

**Getestetes Konzept (verworfen):** Mehrstufige Budget-Reduktion:
```
IC_roll_20 < 0 → 70% investiert
IC_roll_30 < 0 → 49% investiert  
IC_roll_40 < 0 → 34% investiert
```
**Problem:** Identische Performance zu binärer n_max-Reduktion, aber deutlich komplexer und schwerer zu testen.

**Gewähltes Konzept:** Binäre Policy:
```
IC_roll_40 < 0 → n_max = 3 (statt 5)
IC_roll_40 ≥ 0 → n_max = 5
```

### 13.6 Produktions-Ensemble: Evaluationslogik

Da das Ensemble auf **allen verfügbaren Daten** trainiert wurde, existiert kein traditioneller Out-of-Sample-Backtest. Qualitätssicherung über:
1. **Train-Loss-Konvergenz:** Alle 5 Modelle müssen stabile Konvergenz zeigen
2. **Score-Streuung:** Ensemble-Mittelwert reduziert Modell-spezifisches Rauschen
3. **Live-IC-Tracking:** `update_ic.py` berechnet täglich realisierten IC gegen tatsächliche Returns → `rolling_ic_v2_7d.csv`

### 13.7 Gap-Risiko bei Marktöffnung

**Beobachtung:** Das Scoring basiert auf gestrigen Schlusskursen. Bei signifikanten Gap-Ups/Downs (z.B. nach Earnings) kauft das Modell ohne diese Information.

**Entscheidung:** Gap-Filter im Execution-Layer (±5% Schwelle) für Neukäufe. Bestehende Positionen werden durch den Stop-Loss (−20%) geschützt.

**Warum keine Intraday-Features?** Das LSTM wurde auf täglichen Schlusskursen trainiert. Intraday-Features würden eine Out-of-Distribution-Eingabe erzeugen und die Modellqualität verschlechtern. Die richtige Lösung wäre ein separates Intraday-Modell – außerhalb des aktuellen Scope.

---

## 14. Vollständige Parameterübersicht

### 14.1 Modell-Hyperparameter

| Parameter | Wert | Datei |
|-----------|------|-------|
| `hidden_dim` | 128 | `config_v2_single_horizon.py` |
| `num_layers` | 2 | `config_v2_single_horizon.py` |
| `embed_dim` | 16 | `config_v2_single_horizon.py` |
| `dropout` | 0.3 | `config_v2_single_horizon.py` |
| `seq_len` | 64 | `config_v2_single_horizon.py` |
| n_features | 18 | `features/engineer.py` |
| Input-Dim (inkl. Embed) | 34 | berechnet |

### 14.2 Training-Hyperparameter

| Parameter | Wert | Datei |
|-----------|------|-------|
| `lr` | 5 × 10⁻⁴ | `config_v2_single_horizon.py` |
| `weight_decay` | 1 × 10⁻³ | `config_v2_single_horizon.py` |
| `epochs` | 50 | `config_v2_single_horizon.py` |
| `patience` | 7 | `config_v2_single_horizon.py` |
| `batch_size` | 512 | `config_v2_single_horizon.py` |
| `grad_clip` | 1.0 | `config_v2_single_horizon.py` |
| `rank_weight` | 0.5 | `config_v2_single_horizon.py` |
| `rank_margin` | 0.001 | `config_v2_single_horizon.py` |
| `seed` | 42 | `config_v2_single_horizon.py` |
| Ensemble-Seeds | [42, 43, 44, 45, 46] | `config_v2_single_horizon.py` |

### 14.3 Walk-Forward-Parameter

| Parameter | Wert |
|-----------|------|
| `train_years` | 3.0 |
| `val_months` | 6.0 |
| `step_months` | 6.0 |
| Anzahl Folds | 12 |

### 14.4 Portfolio-Parameter (Produktion)

| Parameter | Wert |
|-----------|------|
| `n_max` | 5 |
| `n_mid` | 2 |
| `n_min` | 1 |
| `rotation_buffer` | 2 |
| `hard_stop_pct` | 0.20 (20%) |
| `fees` | 0.001 (0.1%) |
| `sector_neutral` | True |
| `policy` | IC40 |
| `policy_reduced_n` | 3 |

### 14.5 Execution-Parameter (Alpaca)

| Parameter | Wert |
|-----------|------|
| `stop_loss_pct` | 0.20 (20%) |
| `gap_filter_pct` | 0.05 (5%) |
| `sell_delay_s` | 1.0 s |
| Ordertyp | Market |
| Stop-Ordertyp | GTC Stop-Market |
| Aktienbruchteile | Nein (nur ganze Aktien) |

### 14.6 Live-Inference-Parameter

| Parameter | Wert |
|-----------|------|
| `horizon` | 7 |
| `download_days` | 320 |
| `a3_policy_window` | 40 |
| `a3_reduced_n` | 3 |
| `top_n` | 5 |
| Ausführungszeit | 15:45 CET (Mo–Fr) |
| MSCI-World-Proxy | URTH (iShares MSCI World ETF) |

---

## 15. Infrastruktur & Deployment

### 15.1 Trainings-Infrastruktur (Kaggle)

```
Plattform:    Kaggle Notebooks (GPU T4/P100)
Notebook:     kaggle_notebook.ipynb
Steuerung:    Umgebungsvariablen:
              KAGGLE_SEED=42
              KAGGLE_PROD_MODE=1  (Produktions-Ensemble)
              KAGGLE_SMOKE_TEST=1 (schneller Integrationstest)
Artefakte:    kaggle_artifacts.tar.gz
              → prod_model_seed{42..46}.pt
              → asset_map.json
              → v2_7d_walk_forward.json
```

### 15.2 Produktions-Infrastruktur (Hetzner VPS)

```
Server:       Ubuntu 24.04 LTS, 4 GB RAM
IP:           178.105.87.138
SSH-Key:      ~/.ssh/hetzner_trading (Ed25519)
Python:       3.12 (System-Python), venv unter /opt/trading/.venv
Projektpfad:  /opt/trading/
```

### 15.3 Cron-Job

```cron
# /etc/cron.d/trading-inference
45 13 * * 1-5  root  TZ=Europe/Berlin  /opt/trading/run_inference.sh
# = 15:45 CET (Sommerzeit) / 14:45 CET (Winterzeit)
# Montag–Freitag, 15 Minuten nach US-Marktöffnung
```

### 15.4 Deployment-Workflow

```bash
# 1. Lokal: Code + Modelle hochladen
SERVER_IP=178.105.87.138 ./deploy/upload_to_hetzner.sh

# 2. Server-Setup (einmalig)
ssh root@178.105.87.138
./deploy/setup_hetzner.sh

# 3. .env befüllen
nano /opt/trading/.env
# APCA_API_KEY_ID=...
# APCA_API_SECRET_KEY=...
# TELEGRAM_TOKEN=...
# TELEGRAM_CHAT_ID=...

# 4. Testlauf (ohne echte Orders)
/opt/trading/run_dryrun.sh

# 5. Ab jetzt: vollautomatisch via Cron
```

### 15.5 Datei-Übersicht

| Datei | Zweck |
|-------|-------|
| `live_inference.py` | Hauptskript: Download → Features → Scoring → Output |
| `alpaca_broker.py` | Execution: A/B/C/D, Stop-Loss-Management |
| `notifier.py` | Telegram: Chart-Report, Fehlerbenachrichtigung |
| `update_ic.py` | Täglicher IC-Update: Predictions vs. realisierte Returns |
| `update_raw_data.py` | Inkrementeller OHLCV-Download für Trainings-Daten |
| `config_v2_single_horizon.py` | Alle Hyperparameter und Produktionskonstanten |
| `models_v2_single_horizon.py` | LSTM + Attention + MLP-Head Modell-Definition |
| `train_v2_single_horizon.py` | Training: Walk-Forward + Produktions-Ensemble |
| `features/engineer.py` | Feature-Berechnung und Z-Score-Normalisierung |
| `backtest_v2_single_horizon.py` | OOS-Backtest und IC-Berechnung |
| `deploy/setup_hetzner.sh` | Server-Setup (Python, venv, Pakete, Cron) |
| `deploy/upload_to_hetzner.sh` | rsync/scp Deployment-Script |
| `deploy/requirements_server.txt` | Gepinnte Produktions-Abhängigkeiten |

---

*Dokumentation automatisch generiert aus Quellcode-Analyse, Mai 2026.*
