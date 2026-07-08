# AGENTS.md

## Cursor Cloud specific instructions

Dieses Repo ist ein Python-Forschungs-/Backtesting-System (PyTorch-LSTM Aktien-Trading-Bot).
Es gibt **keinen langlaufenden Server und keine Datenbank** – alles läuft als Batch-/Skript-Jobs.

### Umgebung / Abhängigkeiten
- Python 3.12, CPU-only (keine GPU in der Cloud-VM). Der Code erkennt das Device automatisch
  (`"cuda" if torch.cuda.is_available() else "cpu"`), es ist kein manueller Eingriff nötig.
- Die Python-Pakete werden vom Update-Skript ins User-Site-Verzeichnis installiert
  (`python3 -m pip install --break-system-packages ...`). Es wird **bewusst kein venv** verwendet,
  weil das Basis-Image kein `python3.12-venv` (ensurepip) mitbringt.
- `torch` wird als CPU-Wheel installiert (`--index-url https://download.pytorch.org/whl/cpu`),
  um den großen CUDA-Download zu vermeiden. `requirements.txt` pinnt `torch==2.3.1`; das
  CPU-Wheel `2.3.1+cpu` erfüllt diesen Pin.
- `pytest` ist **nicht** in `requirements.txt` deklariert; das Update-Skript installiert es separat.
  Tests immer via `python3 -m pytest tests/ -v` starten (nicht das `pytest`-Konsolenskript, das
  in `~/.local/bin` liegt und ggf. nicht im PATH ist).

### Beispieldaten (wichtig, nicht offensichtlich)
- `data/raw/` ist gitignored und auf einem frischen Checkout **leer**. Vor jedem Feature-/Trainings-/
  Backtest-Lauf die mitgelieferten Beispieldaten entpacken:
  `mkdir -p data/raw && unzip -o -q data/raw.zip -d data/`  (79 Tages-Parquet-Dateien, inkl. `SPY`).
- Größere 260-Ticker-Datensätze liegen auf Kaggle (nur für den Kaggle-GPU-Workflow), nicht im Repo.

### Was ausführen / Code-Pfade
- **Live-Code-Pfad** (durch `tests/` abgedeckt): `features.engineer.build_panel` (Feature-Panel,
  sektor-neutraler Z-Score) → `train_v2_single_horizon.train_single_horizon` (Walk-Forward-LSTM) →
  `strategy.backtest.run_backtest` (Cross-Sectional-Backtest). Die Referenz-Pipeline orchestriert
  `scripts/kaggle_full_run.py` (für Kaggle-Pfade `/kaggle/...` geschrieben, lokal nur als Vorlage).
- `main.py` (`train`/`backtest`/`features`) bildet die **ältere v1-API** ab und ist teilweise veraltet
  (z.B. ruft `main.py backtest` `run_backtest` mit einer nicht mehr passenden Signatur auf). Für neue
  Arbeiten die v2-Single-Horizon-Skripte nutzen (`config/models/train/backtest_v2_single_horizon.py`).
- Schneller End-to-End-Nachweis: `KAGGLE_SMOKE_TEST=1` reduziert Epochen (→3) und Folds für einen
  vollständigen, aber kurzen Pipeline-Durchlauf.

### Ausgaben
- Ergebnisse/Artefakte landen in `checkpoints/`, `logs/`, `results/` bzw. `/kaggle/working/`
  (alle gitignored). Beim Trainieren wird `checkpoints/v2_<horizon>d/fold_*_best.pt` erzeugt.

### Optionale Integrationen (brauchen Secrets/Netz, für Kern-Backtest nicht nötig)
- yfinance (Daten-Download), Kaggle-API (GPU-Training/Artefakt-Upload), Alpaca (Live/Paper-Trading,
  `APCA_*`), Telegram (`TELEGRAM_TOKEN`/`TELEGRAM_CHAT_ID`, für `daily_scan_report.py`).
