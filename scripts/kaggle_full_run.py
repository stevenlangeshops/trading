"""
scripts/kaggle_full_run.py
───────────────────────────
Vollstaendige Trading-Bot Pipeline fuer Kaggle GPU:
  1. Repo klonen
  2. Abhaengigkeiten installieren
  3. Parquet-Daten kopieren (aus Dataset trading-raw-data)
  4. build_panel() - Features + Targets
  5. train_walk_forward() - 12 Folds, ~2-3h GPU
  6. Checkpoints + Asset-Map speichern
  7. run_backtest() - Long-Only + Long-Short
  8. Ergebnisse als JSON + PNG speichern
  9. Alles in kaggle_artifacts.tar.gz packen
 10. Ergebnisse in permanentes Kaggle Dataset hochladen (busersteven/trading-results)

Erwartet:
  /kaggle/input/trading-raw-data/*.parquet  (260 S&P-500 Parquet-Dateien)

Output nach /kaggle/working/:
  best_model.pt, fold_*_best.pt, asset_map.json,
  walk_forward_results.json, backtest_results_*.json,
  equity_curve.png, kaggle_artifacts.tar.gz
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

WORKING  = Path("/kaggle/working")
REPO_DIR = WORKING / "repo"
# Kaggle-Dataset-Pfade — probiert alle bekannten Varianten.
# raw.zip enthaelt: raw/AAPL_1d.parquet, raw/SPY_1d.parquet, ...
# Daher: sowohl /kaggle/input/trading-raw-data/ als auch .../raw/ pruefen.
INPUT_DIRS = [
    Path("/kaggle/input/datasets/busersteven/trading-raw-data/raw"),  # Notebook-Modus (web UI)
    Path("/kaggle/input/trading-raw-data/raw"),   # API-Modus (kernels push)
    Path("/kaggle/input/trading-raw-data"),
    Path("/kaggle/input/trading-raw-data-v2/raw"),
    Path("/kaggle/input/trading-raw-data-v2"),
]
KAGGLE_INPUT = Path("/kaggle/input")
t0 = time.time()

# Zentrales Log-File das ALLES mitschreibt (stdout wird zusaetzlich gespiegelt)
LOG_FILE = WORKING / "pipeline.log"
_log_fh  = None   # wird in main() geoeffnet


def _init_log() -> None:
    global _log_fh
    _log_fh = open(LOG_FILE, "w", encoding="utf-8", buffering=1)


def log_write(msg: str) -> None:
    """Schreibt in pipeline.log UND auf stdout."""
    import builtins
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    builtins.print(line, flush=True)   # builtins.print direkt – kein Rekursions-Risiko
    if _log_fh:
        _log_fh.write(line + "\n")
        _log_fh.flush()


def run(cmd: list, cwd: Path = WORKING, check: bool = True) -> str:
    log_write(f"$ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(
        cmd, cwd=str(cwd),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    out = result.stdout or ""
    for line in out.splitlines():
        log_write(f"  {line}")
    if check and result.returncode != 0:
        raise RuntimeError(f"rc={result.returncode}: {cmd[0]}")
    return out


def elapsed() -> str:
    return f"{(time.time() - t0) / 60:.1f}min"


# ── Schritt 1: Repo klonen ────────────────────────────────────────────────────

def step_clone():
    log_write(f"\n{'='*60}\nSCHRITT 1: Repo klonen [{elapsed()}]\n{'='*60}")
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    run(["git", "clone", "--depth=1",
         "https://github.com/stevenlangeshops/trading.git",
         str(REPO_DIR)])
    log_write(f"Repo-Inhalt: {[f.name for f in REPO_DIR.iterdir()]}")


# ── Schritt 1b: CUDA Health-Check ────────────────────────────────────────────

def step_check_cuda():
    """
    Prueft ob CUDA nutzbar ist — via Subprocess, bevor torch im Hauptprozess
    importiert wird.

    Entscheidungsbaum:
      SM >= 7.0  und matmul OK  →  GPU-Modus (PyTorch 2.x unterstuetzt SM_70+)
      SM >= 7.0  aber defekt    →  cu124-Reinstall, danach nochmal testen
      SM <  7.0  (z.B. P100)   →  sofort CPU-Modus, kein Reinstall
                                   (PyTorch 2.x + Python 3.12 unterstuetzt SM_60
                                    grundsaetzlich nicht mehr)
      Kein CUDA                 →  CPU-Modus
    """
    log_write(f"\n{'='*60}\nSCHRITT 1b: CUDA Health-Check [{elapsed()}]\n{'='*60}")

    # Stale Flags aus eventuell vorherigem Notebook-Run immer zuerst löschen,
    # damit Env-Vars aus einem P100-Run nicht in einen T4-Run hinüberlaufen.
    os.environ.pop("KAGGLE_GPU_INCOMPATIBLE", None)
    os.environ.pop("CUDA_VISIBLE_DEVICES",    None)

    # Schritt 1: SM-Version und matmul-Test in Subprocess
    probe_code = (
        "import torch, sys, warnings; warnings.filterwarnings('ignore'); "
        "print(f'torch={torch.__version__}'); "
        "avail = torch.cuda.is_available(); "
        "print(f'cuda_avail={avail}'); "
        "cap = torch.cuda.get_device_capability(0) if avail else (0,0); "
        "dev = torch.cuda.get_device_name(0) if avail else 'N/A'; "
        "print(f'gpu={dev}'); "
        "print(f'sm={cap[0]}.{cap[1]}'); "
        "sys.exit(0) if not avail else None; "
        "t = torch.zeros(4,4,device='cuda') @ torch.ones(4,4,device='cuda'); "
        "print('matmul=OK'); sys.exit(0)"
    )

    r = subprocess.run(
        [sys.executable, "-c", probe_code],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    probe_out = r.stdout or ""
    for line in probe_out.splitlines():
        log_write(f"  {line}")

    # SM-Version aus Ausgabe lesen
    sm_major = 0
    for line in probe_out.splitlines():
        if line.startswith("sm="):
            try:
                sm_major = int(line.split("=")[1].split(".")[0])
            except Exception:
                pass

    cuda_ok = r.returncode == 0 and "matmul=OK" in probe_out

    if cuda_ok:
        log_write("  CUDA : OK -> GPU-Training aktiv (SM gefunden, matmul=OK)")
        os.environ["KAGGLE_GPU_OK"] = "1"
        return

    if sm_major > 0 and sm_major < 7:
        # P100 (SM_60): fundamental inkompatibel mit Python 3.12 + PyTorch 2.x.
        log_write(f"  SM_{sm_major}.x (P100) erkannt — fundamental inkompatibel, CPU-Modus.")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["KAGGLE_GPU_INCOMPATIBLE"] = "1"
        return

    if sm_major >= 7:
        # SM >= 7.0 kompatibel, aber matmul schlug fehl -> cu124 nachinstallieren
        log_write(f"  SM_{sm_major}.x kompatibel, aber matmul fehlgeschlagen -> cu124 installieren ...")
        run([
            sys.executable, "-m", "pip", "install", "-q",
            "torch", "torchvision",
            "--index-url", "https://download.pytorch.org/whl/cu124",
            "--upgrade",
        ])
        r2 = subprocess.run(
            [sys.executable, "-c", probe_code],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        for line in (r2.stdout or "").splitlines():
            log_write(f"  {line}")
        if r2.returncode == 0 and "matmul=OK" in (r2.stdout or ""):
            log_write("  CUDA nach Reinstall : OK")
            return

    # Subprocess fehlgeschlagen (z.B. MKL-Ladefehler im Subprocess aber torch im
    # Hauptprozess bereits geladen und funktionsfaehig) -> direkt im Hauptprozess pruefen.
    if r.returncode != 0 and sm_major == 0:
        log_write("  Subprocess-Fehler (MKL?) -> pruefe CUDA direkt im Hauptprozess ...")
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _cap = _torch.cuda.get_device_capability(0)
                _sm  = _cap[0]
                _dev = _torch.cuda.get_device_name(0)
                log_write(f"  Hauptprozess: gpu={_dev}  sm={_sm}.{_cap[1]}")
                if _sm >= 7:
                    # Matmul-Test direkt
                    try:
                        _a = _torch.zeros(4, 4, device="cuda") @ _torch.ones(4, 4, device="cuda")
                        log_write(f"  Hauptprozess CUDA matmul=OK -> GPU-Modus (SM={_sm})")
                        os.environ["KAGGLE_GPU_OK"] = "1"
                        return   # GPU funktioniert -> kein CPU-Flag setzen
                    except Exception as _e:
                        log_write(f"  Hauptprozess matmul fehlgeschlagen: {_e}")
                else:
                    log_write(f"  SM_{_sm}.x < 7.0 -> CPU-Modus")
            else:
                log_write("  Hauptprozess: kein CUDA verfuegbar")
        except Exception as _ex:
            log_write(f"  Hauptprozess-Check fehlgeschlagen: {_ex}")

    log_write("  CUDA nicht nutzbar -> CPU-Modus (CUDA_VISIBLE_DEVICES='')")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["KAGGLE_GPU_INCOMPATIBLE"] = "1"


# ── Schritt 2: Abhaengigkeiten ────────────────────────────────────────────────

def step_install():
    log_write(f"\n{'='*60}\nSCHRITT 2: Abhaengigkeiten [{elapsed()}]\n{'='*60}")
    # Kaggle hat PyTorch vorinstalliert, wir brauchen nur ta + loguru + scipy
    run([sys.executable, "-m", "pip", "install",
         "ta==0.11.0", "loguru==0.7.2", "scipy",
         "--quiet", "--no-warn-script-location"])


# ── Schritt 3: Parquet-Daten kopieren ────────────────────────────────────────

def step_copy_data():
    log_write(f"\n{'='*60}\nSCHRITT 3: Parquet-Daten kopieren [{elapsed()}]\n{'='*60}")
    raw_dst = REPO_DIR / "data" / "raw"
    raw_dst.mkdir(parents=True, exist_ok=True)

    # Zeige was ueberhaupt unter /kaggle/input/ vorhanden ist (Diagnostik)
    if KAGGLE_INPUT.exists():
        entries = sorted(p.name for p in KAGGLE_INPUT.iterdir())
        log_write(f"  /kaggle/input/ Inhalt: {entries}")
    else:
        log_write("  WARNUNG: /kaggle/input/ existiert nicht!")

    copied = 0

    # Schritt 1: Vordefinierte Pfade durchsuchen
    for src_dir in INPUT_DIRS:
        if not src_dir.exists():
            log_write(f"  [SKIP] {src_dir}")
            continue
        files = list(src_dir.glob("*.parquet"))
        if not files:
            log_write(f"  [LEER]  {src_dir} (kein .parquet)")
            continue
        log_write(f"  [FOUND] {src_dir}: {len(files)} Parquet-Dateien")
        for f in files:
            shutil.copy2(f, raw_dst / f.name)
            copied += 1
        break  # erste vorhandene Quelle reicht

    # Schritt 2: Falls nichts gefunden -> rekursiv ALLE .parquet unter /kaggle/input/ suchen
    if copied == 0 and KAGGLE_INPUT.exists():
        log_write("  Durchsuche /kaggle/input/ rekursiv nach .parquet Dateien ...")
        all_parquet = list(KAGGLE_INPUT.rglob("*.parquet"))
        if all_parquet:
            log_write(f"  Gefunden: {len(all_parquet)} Dateien unter {all_parquet[0].parent}")
            for f in all_parquet:
                shutil.copy2(f, raw_dst / f.name)
                copied += 1
        else:
            log_write("  Keine .parquet Dateien unter /kaggle/input/ gefunden.")

    log_write(f"  {copied} Dateien nach {raw_dst} kopiert")

    # Schritt 3: Fallback raw.zip
    if copied == 0:
        raw_zip = REPO_DIR / "data" / "raw.zip"
        if raw_zip.exists():
            import zipfile
            log_write(f"  Fallback: {raw_zip} entpacken")
            with zipfile.ZipFile(raw_zip) as zf:
                zf.extractall(raw_dst)
            copied = len(list(raw_dst.glob("*.parquet")))
            log_write(f"  {copied} Dateien aus raw.zip entpackt")

    if copied == 0:
        raise RuntimeError(
            "Keine Parquet-Daten gefunden!\n"
            "Dataset 'trading-raw-data' muss zum Notebook hinzugefuegt sein.\n"
            f"Aktueller /kaggle/input/ Inhalt: {[p.name for p in KAGGLE_INPUT.iterdir()] if KAGGLE_INPUT.exists() else 'n/a'}"
        )
    return copied


# ── Schritt 4: Features bauen ────────────────────────────────────────────────

def step_build_panel():
    log_write(f"\n{'='*60}\nSCHRITT 4: build_panel() [{elapsed()}]\n{'='*60}")
    sys.path.insert(0, str(REPO_DIR))
    os.chdir(str(REPO_DIR))

    from features.engineer import build_panel
    features, targets = build_panel(timeframe="1d", horizon=11, min_rows=300)
    log_write(f"  features: {features.shape}")
    log_write(f"  targets:  {targets.shape}")
    log_write(f"  Assets:   {features.index.get_level_values('asset').nunique()}")
    print(f"  Zeitraum: {features.index.get_level_values('date').min().date()} "
          f"bis {features.index.get_level_values('date').max().date()}", flush=True)
    return features, targets


# ── Schritt 5: Asset-Map ──────────────────────────────────────────────────────

def step_build_asset_map(features):
    log_write(f"\n{'='*60}\nSCHRITT 5: Asset-Map [{elapsed()}]\n{'='*60}")
    assets    = sorted(features.index.get_level_values("asset").unique().tolist())
    asset_map = {a: i + 1 for i, a in enumerate(assets)}  # IDs bei 1 starten
    log_write(f"  {len(asset_map)} Assets, IDs 1..{max(asset_map.values())}")

    asset_map_path = WORKING / "asset_map.json"
    with open(asset_map_path, "w") as f:
        json.dump(asset_map, f, indent=2)
    log_write(f"  Gespeichert: {asset_map_path}")
    return asset_map


# ── Schritt 6: Training ───────────────────────────────────────────────────────

def step_train(features, targets, asset_map):
    log_write(f"\n{'='*60}\nSCHRITT 6: train_walk_forward() [{elapsed()}]\n{'='*60}")

    # Modul-Cache leeren damit bei erneutem Run immer der frische Code geladen wird.
    # (Ohne Kernel-Restart bleibt trainer.py sonst aus dem vorherigen Lauf gecacht.)
    for _mod in list(sys.modules.keys()):
        if _mod.startswith(("models", "features", "strategy")):
            del sys.modules[_mod]

    # Checkpoints-Verzeichnis sicherstellen (Fallback falls trainer.py es nicht anlegt)
    (WORKING / "checkpoints").mkdir(parents=True, exist_ok=True)

    from models.trainer import train_walk_forward

    # KAGGLE_GPU_OK=1 wird von step_check_cuda explizit gesetzt wenn T4/A100/etc. OK.
    # KAGGLE_GPU_INCOMPATIBLE=1 wird gesetzt wenn P100 (SM<7) oder kein CUDA.
    # Beide Flags werden am Anfang von step_check_cuda gelöscht; so überlebt
    # kein stale-Flag aus einem früheren Notebook-Run ohne Kernel-Restart.
    gpu_ok = os.environ.get("KAGGLE_GPU_OK", "0") == "1"
    if gpu_ok:
        log_write("  Modus: GPU — volle Parameter (12 Folds, 50 Epochs, hidden=128)")
        _epochs, _patience, _hidden, _layers, _step_months = 50, 7, 128, 2, 6.0
    else:
        # CPU-Modus (P100 oder kein CUDA): reduzierte Parameter fuer Zeitlimit.
        log_write("  Modus: CPU — 9 Folds, 30 Ep, hidden=96")
        _epochs, _patience, _hidden, _layers, _step_months = 30, 6, 96, 2, 8.0

    results = train_walk_forward(
        features     = features,
        targets      = targets,
        asset_map    = asset_map,
        # Walk-Forward
        train_years  = 3.0,
        val_months   = 6.0,
        step_months  = _step_months,   # CPU: 12 Monate -> ~6 Folds statt 12
        # Modell
        hidden_dim   = _hidden,
        num_layers   = _layers,
        embed_dim    = 16,
        dropout      = 0.3,
        seq_len      = 64,
        # Training
        lr           = 5e-4,
        weight_decay = 1e-3,
        epochs       = _epochs,
        patience     = _patience,
        batch_size   = 512,
        rank_weight  = 0.5,
    )

    # trainer.py liefert fold_results bereits mit ckpt_path, val_start, val_end
    fold_results = results.get("fold_results", [])

    # Ergebnisse persistieren
    wf_path = WORKING / "walk_forward_results.json"
    with open(wf_path, "w") as f:
        json.dump(fold_results, f, indent=2, default=str)
    log_write(f"  Gespeichert: {wf_path}")

    mean_ic = sum(r.get("best_val_ic", 0) for r in fold_results) / max(len(fold_results), 1)
    log_write(f"  Mean IC ueber {len(fold_results)} Folds: {mean_ic:.4f}")
    return fold_results


# ── Schritt 7: Backtest ───────────────────────────────────────────────────────

def step_backtest(features, targets, asset_map, fold_results):
    log_write(f"\n{'='*60}\nSCHRITT 7: Backtest [{elapsed()}]\n{'='*60}")

    # Modul-Cache leeren — ohne dies nutzt Python bei erneutem "Run All"
    # die alte (gecachte) Version von strategy.backtest statt des frisch
    # geklonten Codes. Typischer Fehler: "unexpected keyword argument".
    for _mod in list(sys.modules.keys()):
        if _mod.startswith(("strategy", "models", "features")):
            del sys.modules[_mod]

    from strategy.backtest import (
        run_backtest, build_price_cache, build_atr_cache,
        plot_equity, compute_benchmarks
    )

    raw_dir    = REPO_DIR / "data" / "raw"
    all_assets = list(asset_map.keys())
    if "SPY" not in all_assets:
        all_assets.append("SPY")
    price_cache = build_price_cache(all_assets, raw_dir=raw_dir)

    atr_cache = None
    log_write("  ATR-Cache: DEAKTIVIERT (Rotation > ATR-Stop)")

    # ── Schritt 7a: Score→Return Kalibrierung ──────────────────────────────
    log_write(f"\n  ── Kalibrierung: Score → Expected Return [{elapsed()}] ──")
    calib_model = None
    calib_eval  = None
    try:
        from strategy.calibration import (
            collect_score_return_pairs,
            fit_score_to_return_calibration,
            predict_expected_return as calib_predict,
            evaluate_calibration,
        )
        import numpy as np

        log_write("  Sammle Score-Return-Paare aus Walk-Forward Val-Perioden...")
        calib_df = collect_score_return_pairs(
            features=features, targets=targets,
            fold_results=fold_results, asset_map=asset_map,
        )

        if len(calib_df) > 100:
            scores_arr  = calib_df['score'].values
            returns_arr = calib_df['true_return_11d'].values

            # 80/20-Split zeitlich (nach Datum sortiert, damit kein Lookahead)
            calib_df_sorted = calib_df.sort_values('date')
            n_train = int(len(calib_df_sorted) * 0.8)
            train_df = calib_df_sorted.iloc[:n_train]
            val_df   = calib_df_sorted.iloc[n_train:]

            log_write(f"  Train: {len(train_df)}  Val: {len(val_df)} Paare")
            log_write(f"  Train-Zeitraum: {train_df['date'].min().date()} → {train_df['date'].max().date()}")
            log_write(f"  Val-Zeitraum  : {val_df['date'].min().date()} → {val_df['date'].max().date()}")

            # Isotonic Regression (monoton, robust)
            calib_model = fit_score_to_return_calibration(
                train_df['score'].values,
                train_df['true_return_11d'].values,
                method='isotonic',
            )

            # Auch lineare Kalibrierung zum Vergleich loggen
            calib_linear = fit_score_to_return_calibration(
                train_df['score'].values,
                train_df['true_return_11d'].values,
                method='linear',
            )
            log_write(f"  Linear: E[ret] = {calib_linear['a']*100:.4f}% + {calib_linear['b']*100:.4f}% * score")

            # Evaluation auf Val-Set
            log_write("  Evaluation auf Val-Set:")
            calib_eval = evaluate_calibration(
                val_df, calib_model,
                save_dir=str(WORKING),
            )

            # Auch auf Gesamt-Set evaluieren
            log_write("  Evaluation auf Gesamt-Set:")
            calib_eval_full = evaluate_calibration(calib_df, calib_model)

            # Kalibrierungs-Stats als JSON speichern
            calib_report = {
                'n_pairs_total':    len(calib_df),
                'n_pairs_train':    len(train_df),
                'n_pairs_val':      len(val_df),
                'method':           calib_model['method'],
                'val_metrics':      calib_eval,
                'full_metrics':     calib_eval_full,
            }
            with open(WORKING / "calibration_report.json", "w") as f:
                json.dump(calib_report, f, indent=2, default=str)
            log_write(f"  calibration_report.json gespeichert")

            # Kalibrierte Scores als CSV speichern (für Offline-Analyse)
            calib_df['expected_return'] = calib_predict(calib_model, calib_df['score'].values)
            calib_df.to_csv(str(WORKING / "calibration_data.csv"), index=False)
            log_write(f"  calibration_data.csv gespeichert ({len(calib_df)} Zeilen)")
        else:
            log_write(f"  [WARN] Zu wenig Daten fuer Kalibrierung: {len(calib_df)}")
    except Exception as e:
        log_write(f"  [WARN] Kalibrierung fehlgeschlagen: {e}")
        import traceback
        log_write(traceback.format_exc())

    # ── Schritt 7b: Run G_calib Backtest ───────────────────────────────────
    # Run G Baseline-Setup + kalibrierter Expected-Return-Filter
    run_cfg = dict(
        use_atr_trailing      = False,
        use_dd_control        = False,
        hard_stop_pct         = 0.25,
        use_crash_protection  = False,
        use_min_expected_return_filter = False,
        use_avg_topN_filter           = False,
        existing_pos_exit_margin      = 0.02,
        use_signal_strength_filter    = False,
        # Kalibrierter Return-Filter
        calibration_model     = calib_model,
        min_calibrated_return = 0.0,
    )

    log_write(f"\n  ── Run G_calib: Backtest mit kalibriertem Filter [{elapsed()}] ──")
    if calib_model:
        log_write(f"  Filter: E[ret_top1] >= {run_cfg['min_calibrated_return']*100:.1f}%  "
                  f"(Methode: {calib_model['method']})")
    else:
        log_write("  Kein Kalibrier-Modell — Run G ohne Filter")

    result_a = run_backtest(
        features=features, targets=targets,
        fold_results=fold_results, asset_map=asset_map,
        long_short=False, price_cache=price_cache, atr_cache=atr_cache,
        **run_cfg,
    )
    result_b = {}

    # Benchmarks berechnen
    benchmarks = {}
    try:
        benchmarks = compute_benchmarks(
            price_cache=price_cache,
            equity_dates=result_a.get("equity_dates", []),
            asset_map=asset_map,
            init_cash=10_000.0,
        )
        for key in ("spy", "ew_bh", "ew_rebalanced"):
            bm = benchmarks.get(key, {})
            if bm.get("total_return") is not None:
                log_write(
                    f"  Benchmark {bm['label']:30s}  "
                    f"Return: {bm['total_return']:+7.1f}%  "
                    f"Sharpe: {bm.get('sharpe', 0):.3f}"
                )
        with open(WORKING / "benchmarks.json", "w") as f:
            slim_bm = {k: {kk: vv for kk, vv in v.items() if kk != "equity"}
                       for k, v in benchmarks.items() if k != "dates"}
            json.dump(slim_bm, f, indent=2)
    except Exception as e:
        log_write(f"  [WARN] compute_benchmarks: {e}")

    # JSON-Ergebnisse speichern
    def slim(r):
        return {k: v for k, v in r.items()
                if k not in ("equity", "trade_log", "equity_dates", "daily_signals")}

    with open(WORKING / "backtest_results_long_only.json", "w") as f:
        json.dump(slim(result_a), f, indent=2)
    with open(WORKING / "backtest_results_long_short.json", "w") as f:
        json.dump(slim(result_b), f, indent=2)

    with open(WORKING / "trade_log_long_only.json", "w") as f:
        json.dump(result_a.get("trade_log", []), f, indent=2)

    with open(WORKING / "daily_signals.json", "w") as f:
        json.dump(result_a.get("daily_signals", []), f, indent=1)
    log_write(f"  daily_signals.json: {len(result_a.get('daily_signals', []))} Tage")

    try:
        plot_equity(result_a, result_b,
                    benchmarks=benchmarks,
                    save_path=str(WORKING / "equity_curve.png"))
    except Exception as e:
        log_write(f"  [WARN] plot_equity: {e}")

    try:
        from strategy.backtest import plot_signals
        plot_signals(result_a, save_path=str(WORKING / "signal_diagnostics.png"))
    except Exception as e:
        log_write(f"  [WARN] plot_signals: {e}")

    return result_a, result_b


# ── Schritt 8: Tar-Archiv ─────────────────────────────────────────────────────

def step_pack_artifacts(result_a: dict, result_b: dict):
    log_write(f"\n{'='*60}\nSCHRITT 8: Artefakte packen [{elapsed()}]\n{'='*60}")

    collect = [
        WORKING / "asset_map.json",
        WORKING / "walk_forward_results.json",
        WORKING / "backtest_results_long_only.json",
        WORKING / "backtest_results_long_short.json",
        WORKING / "benchmarks.json",
        WORKING / "trade_log_long_only.json",
        WORKING / "trade_log_long_short.json",
        WORKING / "equity_curve.png",
        WORKING / "kernel_summary.json",
        WORKING / "crash_3d_analysis.json",
        WORKING / "daily_signals.json",
        WORKING / "signal_diagnostics.png",
        WORKING / "calibration_report.json",
        WORKING / "calibration_data.csv",
        WORKING / "calibration_diagnostics.png",
        # v2_return_multi Artefakte
        WORKING / "v2_walk_forward_results.json",
        WORKING / "v2_backtest_results.json",
        WORKING / "v2_trade_log.json",
        WORKING / "v2_daily_signals.json",
        WORKING / "benchmark_v1_vs_v2_multi.json",
        WORKING / "v1_vs_v2_equity.png",
    ]

    # Checkpoints (v1 + v2)
    ckpt_dir = REPO_DIR / "checkpoints"
    if ckpt_dir.is_dir():
        collect += list(ckpt_dir.glob("*.pt"))
        collect += list(ckpt_dir.glob("*.json"))
    ckpt_v2 = REPO_DIR / "checkpoints" / "v2_return_multi"
    if ckpt_v2.is_dir():
        collect += list(ckpt_v2.glob("*.pt"))

    tar_path = WORKING / "kaggle_artifacts.tar.gz"
    with tarfile.open(str(tar_path), "w:gz") as tf:
        seen: set[str] = set()
        for p in collect:
            if p.exists() and str(p) not in seen:
                seen.add(str(p))
                tf.add(str(p), arcname=p.name)

    sz = tar_path.stat().st_size // 1024
    log_write(f"  {tar_path.name}: {sz} KB, {len(seen)} Dateien")

    run_refs = {
        "run_g_long_only": {"total_return": 403.93, "max_drawdown": -55.48, "sharpe": 0.784,
                             "n_trades": 471,  "avg_hold_days": 14.8,
                             "note": "Baseline: reine Rotation + Hard-Stop, kein Filter"},
        "run_h1_long_only": {"total_return": 335.25, "max_drawdown": -54.49, "sharpe": 0.724,
                              "n_trades": 406, "avg_hold_days": 14.3,
                              "note": "SPY-ATR Halbgas-Modus"},
        "run_h2_signal_filter": {"total_return": -18.3, "max_drawdown": -41.99, "sharpe": -0.028,
                                  "n_trades": 108, "note": "Signal-Spread-Filter, 86% aktiv"},
        "benchmarks":       {"spy_bh": "+60.6%", "ew_universe_bh": "+192.8%",
                              "ew_rebalanced": "+167.3%"},
    }
    summary = {
        "return_code":  0,
        "duration_min": round((time.time() - t0) / 60, 1),
        "long_only":  {k: v for k, v in result_a.items()
                       if k not in ("equity", "trade_log", "equity_dates", "daily_signals")},
        "long_short": {k: v for k, v in result_b.items()
                       if k not in ("equity", "trade_log", "equity_dates", "daily_signals")},
        "run_references": run_refs,
    }
    (WORKING / "kernel_summary.json").write_text(json.dumps(summary, indent=2))
    (WORKING / "kaggle_cmd_exit_code.txt").write_text("0")
    (WORKING / "kaggle_cmd_stdout_stderr.txt").write_text(
        f"Pipeline erfolgreich in {summary['duration_min']} Minuten.\n"
        f"Long-Only  Total Return: {result_a.get('total_return', '?')}%\n"
    )
    log_write("\n" + json.dumps(summary, indent=2))


# ── Schritt 9: Ergebnisse in Kaggle Dataset persistieren ──────────────────────

def step_persist_results():
    """
    Lädt die wichtigsten Ergebnisse in ein permanentes Kaggle Dataset hoch,
    damit sie nach Ende der interaktiven Session nicht verloren gehen.

    Die Dateien in /kaggle/working/ sind in einer interaktiven Session nur
    temporär verfügbar (~20 Minuten nach Sessionende). Ein Kaggle Dataset
    hingegen ist dauerhaft und kann jederzeit per API heruntergeladen werden.

    Setup (einmalig):
      1. kaggle.com → Datasets → "New Dataset" → Name: "trading-results" → Create
      2. Notebook Settings → Add-ons → Secrets:
           Name:  KAGGLE_KEY
           Value: <dein API-Key, z.B. KGAT_b1c32e3f8e3fd192150c7bbfd78b8dec>

    Nach jedem Run erscheint eine neue Version unter:
      kaggle.com/busersteven/trading-results
    Download lokal:
      kaggle datasets download busersteven/trading-results
    """
    log_write(f"\n{'='*60}\nSCHRITT 9: Ergebnisse persistieren [{elapsed()}]\n{'='*60}")

    # ── API-Key ermitteln ──────────────────────────────────────────────────────
    # Bevorzuge Kaggle Notebook Secret (sicherste Methode), dann Env-Var-Fallback.
    kaggle_key = None
    try:
        from kaggle_secrets import UserSecretsClient  # nur in Kaggle-Notebooks verfügbar
        kaggle_key = UserSecretsClient().get_secret("KAGGLE_KEY")
    except Exception:
        pass
    if not kaggle_key:
        kaggle_key = os.environ.get("KAGGLE_KEY") or os.environ.get("KAGGLE_API_TOKEN")

    if not kaggle_key:
        log_write(
            "  [SKIP] Kein KAGGLE_KEY gefunden - Ergebnisse werden nicht persistiert.\n"
            "  Tipp: Notebook Settings → Add-ons → Secrets → KAGGLE_KEY hinzufuegen."
        )
        return

    # ── kaggle.json konfigurieren ──────────────────────────────────────────────
    kaggle_cfg = Path("/root/.kaggle/kaggle.json")
    kaggle_cfg.parent.mkdir(parents=True, exist_ok=True)
    kaggle_cfg.write_text(json.dumps({"username": "busersteven", "key": kaggle_key}))
    kaggle_cfg.chmod(0o600)

    # ── Upload-Verzeichnis befüllen ────────────────────────────────────────────
    ts         = time.strftime("%Y%m%d_%H%M%S")
    upload_dir = WORKING / "dataset_upload"
    upload_dir.mkdir(exist_ok=True)

    # Zu persistierende Dateien (bewusst kein .pt um Dataset-Größe gering zu halten;
    # das vollständige tar.gz enthält die Checkpoints)
    upload_files = [
        "kaggle_artifacts.tar.gz",           # alles inkl. Checkpoints
        "equity_curve.png",                  # Equity-Kurve mit Benchmarks
        "kernel_summary.json",               # kompakte Ergebnis-Übersicht
        "benchmarks.json",                   # SPY / EW-Vergleich
        "backtest_results_long_only.json",
        "backtest_results_long_short.json",
        "walk_forward_results.json",
        "pipeline.log",                      # vollständiges Prozess-Log
    ]
    copied = []
    for fname in upload_files:
        src = WORKING / fname
        if src.exists():
            shutil.copy(src, upload_dir / fname)
            copied.append(fname)

    if not copied:
        log_write("  [SKIP] Keine Dateien zum Hochladen gefunden.")
        return

    log_write(f"  Dateien fuer Upload: {copied}")

    # ── Dataset-Metadaten schreiben ────────────────────────────────────────────
    dataset_meta = {
        "title":    "trading-results",
        "id":       "busersteven/trading-results",
        "licenses": [{"name": "other"}],
    }
    (upload_dir / "dataset-metadata.json").write_text(json.dumps(dataset_meta, indent=2))

    # ── Version hochladen ──────────────────────────────────────────────────────
    cmd = [
        "kaggle", "datasets", "version",
        "-p", str(upload_dir),
        "-m", f"Pipeline Run {ts}",
        "--dir-mode", "zip",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                            errors="replace")

    if result.returncode == 0:
        log_write(
            f"  [OK] Dataset aktualisiert: https://kaggle.com/busersteven/trading-results\n"
            f"  Lokal herunterladen: kaggle datasets download busersteven/trading-results"
        )
    else:
        # Zweiter Versuch: Dataset existiert vielleicht noch nicht → erstmal anlegen
        log_write(f"  [WARN] version fehlgeschlagen ({result.stderr[:200]}), versuche create ...")
        cmd_create = [
            "kaggle", "datasets", "create",
            "-p", str(upload_dir),
        ]
        result2 = subprocess.run(cmd_create, capture_output=True, text=True,
                                 encoding="utf-8", errors="replace")
        if result2.returncode == 0:
            log_write("  [OK] Dataset erstellt: https://kaggle.com/busersteven/trading-results")
        else:
            log_write(
                f"  [FAIL] Auch create fehlgeschlagen: {result2.stderr[:400]}\n"
                f"  Bitte manuell 'Save Version' im Notebook-UI klicken."
            )


# ── Schritt 6b: Checkpoints aus Dataset laden (Backtest-Only-Modus) ───────────

def step_load_checkpoints(asset_map: dict) -> list[dict] | None:
    """
    Lädt vorhandene Fold-Checkpoints aus dem Kaggle Dataset 'trading-results'.

    Prüft zusätzlich, ob die gespeicherten Checkpoints mit der aktuellen
    Asset-Anzahl kompatibel sind (n_assets muss übereinstimmen).
    Bei Inkompatibilität (z.B. nach Asset-Expansion) → Training erzwungen.

    Gibt fold_results zurück wenn alle Checkpoints vorhanden und kompatibel sind,
    sonst None → Training wird normal ausgeführt.
    """
    import torch as _torch
    import json  as _json

    backtest_only = os.environ.get("BACKTEST_ONLY", "0") == "1"
    n_assets_current = len(asset_map) + 1   # +1 wegen 1-basiertem Embedding-Index

    # Mögliche Quellen für vorhandene Checkpoints.
    candidate_dirs = [
        WORKING / "checkpoints",
        Path("/kaggle/input/trading-results"),
        Path("/kaggle/input/trading-results/checkpoints"),
        Path("/kaggle/input/datasets/busersteven/trading-results"),
        Path("/kaggle/input/datasets/busersteven/trading-results/checkpoints"),
    ]

    wf_json  = None
    ckpt_src = None
    for d in candidate_dirs:
        wf_candidate = d / "walk_forward_results.json"
        pts = list(d.glob("fold_*_best.pt")) if d.exists() else []
        log_write(f"  Suche Checkpoints: {d}  -> wf={wf_candidate.exists()}  pts={len(pts)}")
        if wf_candidate.exists() and pts:
            wf_json  = wf_candidate
            ckpt_src = d
            break

    if wf_json is None:
        if backtest_only:
            log_write("  [WARN] BACKTEST_ONLY gesetzt aber keine Checkpoints gefunden.")
        else:
            log_write("  Keine vorhandenen Checkpoints -> Training wird ausgefuehrt.")
        return None

    # Kompatibilitaets-Check: erstes .pt laden und n_assets vergleichen
    first_pt = next(ckpt_src.glob("fold_*_best.pt"))
    try:
        ckpt_meta = _torch.load(str(first_pt), map_location="cpu", weights_only=False)
        n_assets_saved = ckpt_meta.get("config", {}).get("n_assets", None)
        if n_assets_saved is not None and n_assets_saved != n_assets_current:
            log_write(
                f"  [INFO] Checkpoint-Assets={n_assets_saved} != aktuell={n_assets_current}"
                f" (Asset-Expansion erkannt) -> Neutraining erforderlich."
            )
            return None
        log_write(f"  Checkpoint-Assets={n_assets_saved} kompatibel mit aktuell={n_assets_current}")
    except Exception as e:
        log_write(f"  [WARN] Checkpoint-Kompatibilitaets-Check fehlgeschlagen: {e}")

    fold_results = _json.loads(wf_json.read_text())

    # Checkpoint-Pfade auf aktuellen Speicherort zeigen lassen
    ckpt_target = WORKING / "checkpoints"
    ckpt_target.mkdir(parents=True, exist_ok=True)

    updated = []
    for fold in fold_results:
        src_pt = ckpt_src / f"fold_{fold['fold_id']}_best.pt"
        dst_pt = ckpt_target / f"fold_{fold['fold_id']}_best.pt"
        if src_pt.exists() and not dst_pt.exists():
            shutil.copy(src_pt, dst_pt)
        if dst_pt.exists():
            fold["ckpt_path"] = str(dst_pt)
            updated.append(fold)

    if len(updated) == len(fold_results):
        log_write(
            f"  [OK] {len(updated)} Checkpoints geladen aus {ckpt_src}\n"
            f"  Training wird uebersprungen (BACKTEST_ONLY oder vorhandene Checkpoints)."
        )
        return updated

    missing = len(fold_results) - len(updated)
    log_write(f"  [WARN] {missing} Checkpoints fehlen -> Training wird ausgefuehrt.")
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

# ── v2_return_multi Pipeline Steps ────────────────────────────────────────────

def step_build_multi_targets(features, asset_map):
    """Schritt 10: Multi-Horizon Targets für v2 berechnen."""
    log_write(f"\n{'='*60}\nSCHRITT 10: v2 Multi-Horizon Targets [{elapsed()}]\n{'='*60}")

    for _mod in list(sys.modules.keys()):
        if _mod.startswith(("config_v2", "models_v2", "train_v2", "backtest_v2")):
            del sys.modules[_mod]

    from config_v2_return_multi import V2Config
    from train_v2_return_multi import build_multi_horizon_targets

    cfg = V2Config()
    raw_dir = REPO_DIR / "data" / "raw"

    targets_multi = build_multi_horizon_targets(
        raw_dir=raw_dir, horizons=cfg.horizons,
        asset_list=list(asset_map.keys()),
    )
    log_write(f"  Multi-Targets: {len(targets_multi)} Zeilen, "
              f"Horizonte={cfg.horizons}")
    return targets_multi, cfg


def step_train_v2(features, targets_multi, asset_map, cfg):
    """Schritt 11: Walk-Forward Training für v2_return_multi."""
    log_write(f"\n{'='*60}\nSCHRITT 11: v2 Training [{elapsed()}]\n{'='*60}")

    from train_v2_return_multi import train_walk_forward_v2

    cfg.checkpoint_dir = REPO_DIR / "checkpoints" / "v2_return_multi"
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Volle Parameter wie v1 Run G (50 Epochen, hidden=128)
    log_write(f"  v2 Modus: {cfg.epochs} Ep, hidden={cfg.hidden_dim}, "
              f"patience={cfg.patience}, seq_len={cfg.seq_len}")

    wf_result = train_walk_forward_v2(features, targets_multi, asset_map, cfg)

    wf_path = WORKING / "v2_walk_forward_results.json"
    import json as _json
    with open(wf_path, "w") as f:
        safe = {k: v for k, v in wf_result.items() if k != "fold_results"}
        safe["fold_summary"] = [
            {k: v for k, v in fr.items() if k != "best_val_comp"}
            for fr in wf_result["fold_results"]
        ]
        _json.dump(safe, f, indent=2, default=str)
    log_write(f"  v2_walk_forward_results.json gespeichert")

    return wf_result


def step_backtest_v2(features, targets_multi, asset_map, v2_fold_results, cfg, v1_result_a):
    """Schritt 12: Backtest für v2 + Vergleich mit v1 (Run G)."""
    log_write(f"\n{'='*60}\nSCHRITT 12: v2 Backtest [{elapsed()}]\n{'='*60}")

    for _mod in list(sys.modules.keys()):
        if _mod.startswith(("config_v2", "models_v2", "train_v2", "backtest_v2")):
            del sys.modules[_mod]

    from backtest_v2_return_multi import (
        run_backtest_v2, build_v1_vs_v2_report, plot_v1_vs_v2,
    )
    from strategy.backtest import build_price_cache, compute_benchmarks

    raw_dir = REPO_DIR / "data" / "raw"
    all_assets = list(asset_map.keys())
    if "SPY" not in all_assets:
        all_assets.append("SPY")
    price_cache = build_price_cache(all_assets, raw_dir=raw_dir)

    v2_result = run_backtest_v2(
        features=features, targets_multi=targets_multi,
        fold_results=v2_fold_results, asset_map=asset_map,
        cfg=cfg, price_cache=price_cache,
    )

    # Ergebnisse speichern
    def slim(r):
        return {k: v for k, v in r.items()
                if k not in ("equity", "trade_log", "equity_dates", "daily_signals")}

    with open(WORKING / "v2_backtest_results.json", "w") as f:
        json.dump(slim(v2_result), f, indent=2)
    with open(WORKING / "v2_trade_log.json", "w") as f:
        json.dump(v2_result.get("trade_log", []), f, indent=2)
    with open(WORKING / "v2_daily_signals.json", "w") as f:
        json.dump(v2_result.get("daily_signals", []), f, indent=1)

    # v1 vs v2 Vergleich (nur wenn v1-Daten vorhanden)
    if v1_result_a and v1_result_a.get("total_return") is not None:
        try:
            build_v1_vs_v2_report(
                v1_result=v1_result_a, v2_result=v2_result,
                save_path=str(WORKING / "benchmark_v1_vs_v2_multi.json"),
            )
        except Exception as e:
            log_write(f"  [WARN] v1 vs v2 report: {e}")
    else:
        log_write("  v1-Ergebnis nicht vorhanden — Vergleich uebersprungen")
        # Run-G Referenzwerte stattdessen loggen
        log_write("  Run G Referenz: Total Return +403.93%  Sharpe 0.784  MaxDD -55.48%")

    # Benchmarks
    benchmarks = {}
    try:
        benchmarks = compute_benchmarks(
            price_cache=price_cache,
            equity_dates=v2_result.get("equity_dates", []),
            asset_map=asset_map, init_cash=10_000.0,
        )
        for key in ("spy", "ew_bh", "ew_rebalanced"):
            bm = benchmarks.get(key, {})
            if bm.get("total_return") is not None:
                log_write(f"  Benchmark {bm['label']:30s}  "
                          f"Return: {bm['total_return']:+7.1f}%  "
                          f"Sharpe: {bm.get('sharpe', 0):.3f}")
    except Exception as e:
        log_write(f"  [WARN] v2 benchmarks: {e}")

    # Vergleichs-Plot (v2 standalone oder v1 vs v2)
    try:
        plot_v1_vs_v2(
            v1_result=v1_result_a if v1_result_a else {},
            v2_result=v2_result,
            benchmarks=benchmarks,
            save_path=str(WORKING / "v1_vs_v2_equity.png"),
        )
    except Exception as e:
        log_write(f"  [WARN] v2 plot: {e}")

    return v2_result


def main() -> int:
    _init_log()
    log_write(f"Trading Bot Full Pipeline | {time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    log_write(f"Log-Datei: {LOG_FILE}")

    # ── Modus-Schalter ────────────────────────────────────────────────
    # V2_ONLY = True  → v1 überspringen, nur v2_return_multi
    # V2_MAX_ASSETS   → Asset-Limit für schnellen Testlauf (0 = alle)
    V2_ONLY       = True
    V2_MAX_ASSETS = 0    # 0 = alle Assets (260 S&P 500)
    # ──────────────────────────────────────────────────────────────────

    try:
        step_clone()
        step_check_cuda()
        step_install()
        step_copy_data()

        # Features + Targets mit optionalem Asset-Limit
        features, targets = step_build_panel()
        asset_map         = step_build_asset_map(features)

        if V2_MAX_ASSETS > 0 and len(asset_map) > V2_MAX_ASSETS:
            log_write(f"\n  [v2 TEST] Asset-Limit: {V2_MAX_ASSETS} von {len(asset_map)} Assets")
            import random
            all_assets_list = sorted(asset_map.keys())
            # SPY immer behalten (fuer Regime-Filter)
            keep = {"SPY"} if "SPY" in asset_map else set()
            remaining = [a for a in all_assets_list if a not in keep]
            random.seed(42)
            keep.update(random.sample(remaining, min(V2_MAX_ASSETS - len(keep), len(remaining))))
            keep = sorted(keep)
            # Features und Targets auf Subset filtern
            idx_mask = features.index.get_level_values("asset").isin(keep)
            features = features[idx_mask]
            targets  = targets[idx_mask]
            asset_map = {a: i + 1 for i, a in enumerate(keep)}
            log_write(f"  [v2 TEST] {len(asset_map)} Assets ausgewaehlt: {', '.join(keep[:10])}...")

        result_a = {}   # v1 Ergebnis (leer wenn V2_ONLY)
        result_b = {}

        if not V2_ONLY:
            # ── v1_rank (Run G) ───────────────────────────────────────
            fold_results = step_load_checkpoints(asset_map)
            if fold_results is None:
                fold_results = step_train(features, targets, asset_map)
            result_a, result_b = step_backtest(features, targets, asset_map, fold_results)

        # ── v2_return_multi ───────────────────────────────────────────
        try:
            targets_multi, cfg_v2 = step_build_multi_targets(features, asset_map)
            v2_wf = step_train_v2(features, targets_multi, asset_map, cfg_v2)
            v2_result = step_backtest_v2(
                features, targets_multi, asset_map,
                v2_wf["fold_results"], cfg_v2, result_a,
            )
        except Exception as v2_exc:
            import traceback
            log_write(f"\n[v2 ERROR]\n{traceback.format_exc()}")
            v2_result = {}

        # Pack: v2 Ergebnisse als result_a wenn V2_ONLY
        if V2_ONLY and v2_result:
            result_a = v2_result

        step_pack_artifacts(result_a, result_b)
        log_write(f"\n[DONE] Gesamtdauer: {elapsed()}")
        try:
            step_persist_results()
        except Exception as persist_exc:
            log_write(f"  [WARN] step_persist_results: {persist_exc}")
        return 0
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        log_write(f"\n[ERROR]\n{tb}")
        (WORKING / "kaggle_cmd_stdout_stderr.txt").write_text(tb)
        (WORKING / "kaggle_cmd_exit_code.txt").write_text("1")
        (WORKING / "kernel_summary.json").write_text(
            json.dumps({"return_code": 1, "error": str(exc)})
        )
        # Log-File schliessen bevor tar
        if _log_fh:
            _log_fh.flush()
        # Minimal-Tar damit Download-Wrapper immer etwas findet
        tar_path = WORKING / "kaggle_artifacts.tar.gz"
        with tarfile.open(str(tar_path), "w:gz") as tf:
            for fname in ("kaggle_cmd_stdout_stderr.txt", "kaggle_cmd_exit_code.txt",
                          "kernel_summary.json", "pipeline.log"):
                p = WORKING / fname
                if p.exists():
                    tf.add(str(p), arcname=fname)
        # Auch bei Fehler versuchen zu persistieren (z.B. das Log ist wertvoll)
        try:
            step_persist_results()
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
else:
    # Wird via exec() aus einem Jupyter-Notebook aufgerufen.
    # sys.exit() wuerde Jupyter abwuergen — stattdessen direkt main() aufrufen.
    main()
