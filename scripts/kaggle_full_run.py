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

Erwartet:
  /kaggle/input/trading-raw-data/*.parquet  (79 Parquet-Dateien)

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
    Path("/kaggle/input/trading-raw-data/raw"),   # aus ZIP: Unterordner raw/
    Path("/kaggle/input/trading-raw-data"),        # flach (falls direkt hochgeladen)
    Path("/kaggle/input/trading-raw-data-v2/raw"),
    Path("/kaggle/input/trading-raw-data-v2"),
]
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
        log_write("  CUDA : OK -> GPU-Training aktiv")
        return

    if sm_major > 0 and sm_major < 7:
        # P100 (SM_60): das vorinstallierte PyTorch+cu128 unterstuetzt SM_60 nicht.
        # Loesung: PyTorch+cu118 installieren — CUDA 11.8 unterstuetzt SM_60 explizit.
        # WICHTIG: --force-reinstall noetig, weil pip --upgrade cu118 nicht
        # installiert wenn die aktuelle Version (z.B. 2.10.0+cu128) > max-cu118-Version.
        log_write(f"  SM_{sm_major}.x (P100) erkannt.")
        log_write("  Vorinstalliertes torch+cu128 unterstuetzt SM_60 nicht.")
        log_write("  Installiere torch==2.5.1+cu118 (force-reinstall) ...")
        pip_r = subprocess.run(
            [
                sys.executable, "-m", "pip", "install",
                "torch==2.5.1+cu118", "torchvision==0.20.1+cu118",
                "--index-url", "https://download.pytorch.org/whl/cu118",
                "--force-reinstall", "--no-deps",
            ],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        for line in (pip_r.stdout or "").splitlines()[-10:]:
            log_write(f"  pip> {line}")
        if pip_r.returncode != 0:
            log_write("  pip install cu118 fehlgeschlagen -> CPU-Fallback.")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["KAGGLE_GPU_INCOMPATIBLE"] = "1"
            return
        r2 = subprocess.run(
            [sys.executable, "-c", probe_code],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        for line in (r2.stdout or "").splitlines():
            log_write(f"  {line}")
        if r2.returncode == 0 and "matmul=OK" in (r2.stdout or ""):
            log_write("  CUDA cu118 + SM_60 : OK -> GPU-Training aktiv!")
            return
        log_write("  cu118 fehlgeschlagen -> CPU-Fallback.")
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

    # Letzter Ausweg
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

    copied = 0
    for src_dir in INPUT_DIRS:
        if not src_dir.exists():
            log_write(f"  [SKIP] {src_dir}")
            continue
        files = list(src_dir.glob("*.parquet"))
        log_write(f"  [FOUND] {src_dir}: {len(files)} Parquet-Dateien")
        for f in files:
            shutil.copy2(f, raw_dst / f.name)
            copied += 1
        break  # erste vorhandene Quelle reicht

    log_write(f"  {copied} Dateien nach {raw_dst} kopiert")
    if copied == 0:
        # Fallback: raw.zip im Repo entpacken (falls vorhanden)
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
            "Bitte Dataset 'trading-raw-data' zum Kernel hinzufuegen:\n"
            "  Kaggle UI -> Kernel Settings -> Add Data -> trading-raw-data"
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

    from models.trainer import train_walk_forward

    gpu_ok = os.environ.get("KAGGLE_GPU_INCOMPATIBLE", "0") != "1"
    if gpu_ok:
        log_write("  Modus: GPU — volle Parameter (12 Folds, 50 Epochs, hidden=128)")
        _epochs, _patience, _hidden, _layers, _step_months = 50, 7, 128, 2, 6.0
    else:
        # Echter CPU-Fallback (sollte nicht mehr vorkommen da cu118 P100 unterstuetzt)
        log_write("  Modus: CPU — reduzierte Parameter (6 Folds, 15 Epochs, hidden=64)")
        _epochs, _patience, _hidden, _layers, _step_months = 15, 4,  64, 1, 12.0

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

    from strategy.backtest import run_backtest, build_price_cache, plot_equity

    raw_dir     = REPO_DIR / "data" / "raw"
    all_assets  = list(asset_map.keys())
    price_cache = build_price_cache(all_assets, raw_dir=raw_dir)

    result_a = run_backtest(
        features=features, targets=targets,
        fold_results=fold_results, asset_map=asset_map,
        long_short=False, price_cache=price_cache,
    )
    result_b = run_backtest(
        features=features, targets=targets,
        fold_results=fold_results, asset_map=asset_map,
        long_short=True, price_cache=price_cache,
    )

    # JSON ohne equity/trade_log (zu gross)
    def slim(r):
        return {k: v for k, v in r.items() if k not in ("equity", "trade_log", "equity_dates")}

    with open(WORKING / "backtest_results_long_only.json", "w") as f:
        json.dump(slim(result_a), f, indent=2)
    with open(WORKING / "backtest_results_long_short.json", "w") as f:
        json.dump(slim(result_b), f, indent=2)

    # Voller Trade-Log separat
    with open(WORKING / "trade_log_long_only.json", "w") as f:
        json.dump(result_a.get("trade_log", []), f, indent=2)
    with open(WORKING / "trade_log_long_short.json", "w") as f:
        json.dump(result_b.get("trade_log", []), f, indent=2)

    try:
        plot_equity(result_a, result_b, save_path=str(WORKING / "equity_curve.png"))
    except Exception as e:
        log_write(f"  [WARN] plot_equity: {e}")

    return result_a, result_b


# ── Schritt 8: Tar-Archiv ─────────────────────────────────────────────────────

def step_pack_artifacts(result_a: dict, result_b: dict):
    log_write(f"\n{'='*60}\nSCHRITT 8: Artefakte packen [{elapsed()}]\n{'='*60}")

    collect = [
        WORKING / "asset_map.json",
        WORKING / "walk_forward_results.json",
        WORKING / "backtest_results_long_only.json",
        WORKING / "backtest_results_long_short.json",
        WORKING / "trade_log_long_only.json",
        WORKING / "trade_log_long_short.json",
        WORKING / "equity_curve.png",
        WORKING / "kernel_summary.json",
    ]

    # Checkpoints
    ckpt_dir = REPO_DIR / "checkpoints"
    if ckpt_dir.is_dir():
        collect += list(ckpt_dir.glob("*.pt"))
        collect += list(ckpt_dir.glob("*.json"))

    tar_path = WORKING / "kaggle_artifacts.tar.gz"
    with tarfile.open(str(tar_path), "w:gz") as tf:
        seen: set[str] = set()
        for p in collect:
            if p.exists() and str(p) not in seen:
                seen.add(str(p))
                tf.add(str(p), arcname=p.name)

    sz = tar_path.stat().st_size // 1024
    log_write(f"  {tar_path.name}: {sz} KB, {len(seen)} Dateien")

    summary = {
        "return_code":  0,
        "duration_min": round((time.time() - t0) / 60, 1),
        "long_only":  {k: v for k, v in result_a.items()
                       if k not in ("equity", "trade_log", "equity_dates")},
        "long_short": {k: v for k, v in result_b.items()
                       if k not in ("equity", "trade_log", "equity_dates")},
    }
    (WORKING / "kernel_summary.json").write_text(json.dumps(summary, indent=2))
    (WORKING / "kaggle_cmd_exit_code.txt").write_text("0")
    (WORKING / "kaggle_cmd_stdout_stderr.txt").write_text(
        f"Pipeline erfolgreich in {summary['duration_min']} Minuten.\n"
        f"Long-Only  Total Return: {result_a.get('total_return', '?')}%\n"
        f"Long-Short Total Return: {result_b.get('total_return', '?')}%\n"
    )
    log_write("\n" + json.dumps(summary, indent=2))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    _init_log()
    log_write(f"Trading Bot Full Pipeline | {time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    log_write(f"Log-Datei: {LOG_FILE}")
    try:
        step_clone()
        step_check_cuda()
        step_install()
        step_copy_data()
        features, targets = step_build_panel()
        asset_map         = step_build_asset_map(features)
        fold_results      = step_train(features, targets, asset_map)
        result_a, result_b = step_backtest(features, targets, asset_map, fold_results)
        step_pack_artifacts(result_a, result_b)
        log_write(f"\n[DONE] Gesamtdauer: {elapsed()}")
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
        return 1


if __name__ == "__main__":
    sys.exit(main())
