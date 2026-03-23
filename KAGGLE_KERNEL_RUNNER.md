# Kaggle Kernel Runner

Mit diesem Wrapper kannst du deinen lokalen Code nach Kaggle (GPU) pushen, den Kernel bis zum Ende poll-en und anschließend Artefakte (Logs/Checkpoints/Plots) herunterladen.

## 1) Voraussetzungen

1. Kaggle Token:
   - Öffne Kaggle -> Account -> API -> "Create New Token"
   - Lade `kaggle.json` herunter und platziere sie unter `~/.kaggle/kaggle.json`
2. Stelle sicher, dass `kaggle` CLI auf deinem Rechner verfügbar ist.
   - Der Wrapper installiert sie bei Bedarf automatisch via `pip install kaggle`.

## 2) Smoke Test (GPU-Check)

Ersetze `DEIN_KAGGLE_USERNAME` und wähle einen Kernel-Slug, z.B. `trading-bot-v5-smoke`.

```powershell
python scripts/kaggle_kernel_api.py run `
  --kernel-id "DEIN_KAGGLE_USERNAME/trading-bot-v5-smoke" `
  --cmd "python -c \"import torch; print('cuda_available=', torch.cuda.is_available()); print('torch=', torch.__version__)\""
```

## 3) Eigene Trainings-/Backtest-Tasks auf Kaggle ausführen

Beispiel (Training):

```powershell
python scripts/kaggle_kernel_api.py run `
  --kernel-id "DEIN_KAGGLE_USERNAME/trading-bot-v5-train" `
  --cmd "python main.py train --ticker combined --timeframe 1h --mode reg --epochs 2 --patience 1" `
  --timeout-seconds 3600 `
  --accelerator gpu `
  --enable-internet `
  --install-deps
```

Wichtig: Der Wrapper kopiert nur Code (und schließt große lokale Datenartefakte wie `data/raw.zip` aus). Falls dein Task Daten/Features erst erzeugen muss, muss das innerhalb des Kaggle-Kernels passieren (wie im Beispiel).

## 4) Artefakte ansehen

Nach erfolgreichem Run findest du die heruntergeladenen Artefakte unter:

`kaggle_kernel_runs/<kernel-id>/<timestamp>/artifacts/`

Dort liegt u.a.:
- `kernel_summary.json` (Return Code, Dauer, Kommando)
- `kaggle_cmd_stdout_stderr.txt` (Logs vom Kernel-Run)
- ggf. `logs/*.log`, `checkpoints/*.pt`, `equity_curve.png`

