# trading_v2 – Neural Network Trading Bot

LSTM-basierter Krypto-Trading-Bot mit PyTorch, VectorBT und CCXT.

---

## Projektstruktur

```
trading_v2/
├── data/
│   ├── download.py        # OHLCV-Daten via CCXT laden
│   └── raw/               # Gespeicherte Parquet-Dateien
├── features/
│   ├── engineer.py        # Feature-Engineering + Labeling + Skalierung
│   └── processed/         # Tensor-Dateien (.pt) für PyTorch
├── models/
│   ├── lstm_model.py      # TradingLSTM (Architektur + Attention)
│   └── trainer.py         # Trainings-Loop mit Early Stopping
├── strategy/
│   └── backtest.py        # VectorBT Backtesting
├── checkpoints/           # Beste Modell-Weights (.pt)
├── logs/                  # Log-Dateien + HTML-Backtest-Reports
├── main.py                # Zentraler Einstiegspunkt
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## Lokale Installation

```bash
# 1. Virtuelle Umgebung
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

# 2. Abhängigkeiten
pip install -r requirements.txt

# 3. Umgebungsvariablen
copy .env.example .env          # Windows
# cp .env.example .env          # Linux/Mac
```

---

## Nutzung (lokal)

```bash
# Schritt 1: Daten laden (5000 Stunden-Kerzen BTC/USDT von Binance)
python main.py download --symbol BTC/USDT --timeframe 1h --limit 5000

# Schritt 2: Features berechnen + Labels erstellen
python main.py features --symbol BTC/USDT --timeframe 1h --mode cls

# Schritt 3: LSTM trainieren
python main.py train --symbol BTC/USDT --timeframe 1h --mode cls --epochs 100

# Schritt 4: Backtesting
python main.py backtest --symbol BTC/USDT --timeframe 1h --mode cls

# Oder alles auf einmal:
python main.py all --symbol BTC/USDT --timeframe 1h
```

### Modi

| Parameter     | Wert  | Bedeutung                          |
|---------------|-------|------------------------------------|
| `--mode`      | `cls` | Klassifikation: Preis steigt ≥ x % |
| `--mode`      | `reg` | Regression: zukünftiger Close-Preis|
| `--timeframe` | `1h`  | 1-Stunden-Kerzen                   |
| `--timeframe` | `4h`  | 4-Stunden-Kerzen                   |
| `--horizon`   | `6`   | Vorausschauhorizont (Perioden)     |
| `--threshold` | `0.005`| Aufwärts-Schwelle für Label=1    |

---

## Docker-Setup

### Voraussetzungen

| Software       | Mindestversion | Download                              |
|----------------|----------------|---------------------------------------|
| Docker Desktop | 24.x           | https://www.docker.com/products/docker-desktop |
| Docker Compose | 2.x            | (in Docker Desktop enthalten)         |

### Image bauen & Container starten

```bash
# Im Projektverzeichnis C:\steven\trading_v2

# Image bauen (einmalig oder nach Code-Änderungen)
docker compose build

# Vollständige Pipeline starten (Download → Features → Training → Backtest)
docker compose up

# Nur einzelne Schritte ausführen:
docker compose run --rm trading-bot download --symbol BTC/USDT --timeframe 1h
docker compose run --rm trading-bot features --symbol BTC/USDT --mode cls
docker compose run --rm trading-bot train    --symbol BTC/USDT --mode cls --epochs 50
docker compose run --rm trading-bot backtest --symbol BTC/USDT --mode cls
```

### Persistente Daten

Durch die `volumes`-Konfiguration in `docker-compose.yml` bleiben alle Daten
auf dem Host erhalten – auch nach Neustart oder Löschen des Containers:

```
./data/raw/          ← Rohdaten (Parquet)
./features/processed/← Vorberechnete Tensors
./checkpoints/       ← Beste Modell-Weights
./logs/              ← Logs + HTML-Reports
```

### GPU-Support (NVIDIA)

1. [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installieren.
2. In `docker-compose.yml` den `trading-bot-gpu`-Service einkommentieren.
3. In `requirements.txt` den PyTorch-Block auf die CUDA-Version anpassen:
   ```
   torch==2.3.1+cu121
   ```

### Nützliche Docker-Befehle

```bash
# Laufende Container anzeigen
docker ps

# Logs verfolgen
docker compose logs -f

# Container stoppen
docker compose down

# Image + Volumes komplett entfernen
docker compose down --volumes --rmi all
```

---

## Features & Modell-Architektur

### Technische Indikatoren

| Indikator          | Parameter    |
|--------------------|--------------|
| RSI                | 14 Perioden  |
| MACD               | 12 / 26 / 9  |
| Bollinger Bänder   | 20 / 2σ      |
| EMA                | 9, 21, 50, 200|
| ATR                | 14 Perioden  |
| Log-Return, Candle-Features | – |

### LSTM-Architektur

```
Input (batch, 48, 21)
  └─ LSTM (128 Units, 2 Layer, Dropout 0.3)
       └─ Temporal Attention
            └─ BatchNorm
                 └─ FC(128→64→32→1)
                      └─ Sigmoid (cls) / Linear (reg)
```

---

## Backtesting-Kennzahlen (VectorBT)

Der HTML-Report in `logs/backtest_*.html` enthält:
- Total Return, Sharpe Ratio, Max Drawdown
- Win Rate, Profit Factor
- Equity-Kurve vs. Buy & Hold

---

## Hinweis

Dieses Projekt dient ausschließlich **Bildungs- und Forschungszwecken**.
Es stellt keine Finanz- oder Anlageberatung dar.
Vergangene Performance ist kein Indikator für zukünftige Ergebnisse.
