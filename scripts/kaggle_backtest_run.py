"""
scripts/kaggle_backtest_run.py
──────────────────────────────
Kaggle-seitiger Entrypoint für den vollständigen Backtest-Run.

Wird von kaggle_kernel_api.py als kaggle_entrypoint.py hochgeladen.
Läuft in /kaggle/working/ mit Zugriff auf:
  /kaggle/input/trading-raw-data/  ← 79 Parquet-Dateien

Ablauf:
  1. Repo klonen (git clone)
  2. Pakete installieren
  3. Parquet-Daten nach data/raw/ kopieren
  4. build_panel() → Features + Targets
  5. run_backtest() Long-Only + Long-Short
  6. Ergebnisse als JSON + PNG speichern
"""

import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

WORKING    = Path("/kaggle/working")
REPO_DIR   = WORKING / "repo"
RAW_DATA   = Path("/kaggle/input/trading-raw-data")
INPUT_DIRS = [
    Path("/kaggle/input/trading-raw-data"),
    Path("/kaggle/input/trading-raw-data-v2"),
]

t0         = time.time()


def run(cmd: list[str], cwd: Path = WORKING, check: bool = True) -> str:
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    proc = subprocess.run(
        cmd, cwd=str(cwd),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    print(proc.stdout or "", flush=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"Kommando fehlgeschlagen (rc={proc.returncode}): {cmd}")
    return proc.stdout or ""


def step1_clone_repo():
    print("\n══ SCHRITT 1: Repo klonen ══", flush=True)
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    run(["git", "clone", "--depth=1",
         "https://github.com/stevenlangeshops/trading.git",
         str(REPO_DIR)])


def step2_install_deps():
    print("\n══ SCHRITT 2: Abhängigkeiten installieren ══", flush=True)
    req = REPO_DIR / "requirements.txt"
    run([sys.executable, "-m", "pip", "install",
         "ta==0.11.0", "loguru==0.7.2", "scipy", "--quiet"])


def step3_copy_data():
    print("\n══ SCHRITT 3: Parquet-Daten kopieren ══", flush=True)
    raw_dst = REPO_DIR / "data" / "raw"
    raw_dst.mkdir(parents=True, exist_ok=True)

    copied = 0
    for src_dir in INPUT_DIRS:
        if not src_dir.exists():
            print(f"  [SKIP] {src_dir} nicht vorhanden", flush=True)
            continue
        for f in src_dir.glob("*.parquet"):
            shutil.copy2(f, raw_dst / f.name)
            copied += 1

    print(f"  {copied} Parquet-Dateien kopiert nach {raw_dst}", flush=True)
    if copied == 0:
        raise RuntimeError("Keine Parquet-Dateien gefunden! Dataset korrekt eingebunden?")


def step4_build_panel():
    print("\n══ SCHRITT 4: build_panel() ══", flush=True)
    sys.path.insert(0, str(REPO_DIR))
    os.chdir(str(REPO_DIR))

    from features.engineer import build_panel
    features, targets = build_panel(timeframe="1d", horizon=11, min_rows=300)
    print(f"  features: {features.shape}  targets: {targets.shape}", flush=True)
    return features, targets


def step5_load_asset_map_and_folds():
    print("\n══ SCHRITT 5: Asset-Map + Fold-Ergebnisse laden ══", flush=True)
    ckpt_dir = REPO_DIR / "checkpoints"

    asset_map_path = WORKING / "asset_map.json"
    if not asset_map_path.exists():
        asset_map_path = ckpt_dir / "asset_map.json"

    if not asset_map_path.exists():
        raise FileNotFoundError(
            "asset_map.json nicht gefunden. "
            "Bitte zuerst einen Training-Run durchführen und asset_map.json im Output speichern."
        )

    with open(asset_map_path) as f:
        asset_map = json.load(f)
    print(f"  {len(asset_map)} Assets geladen", flush=True)

    # Walk-Forward Ergebnisse (fold_results mit ckpt_path)
    wf_path = WORKING / "walk_forward_results.json"
    if not wf_path.exists():
        wf_path = ckpt_dir / "walk_forward_results.json"

    if not wf_path.exists():
        raise FileNotFoundError(
            "walk_forward_results.json nicht gefunden. "
            "Training-Run muss zuerst laufen."
        )

    with open(wf_path) as f:
        fold_results = json.load(f)
    print(f"  {len(fold_results)} Folds geladen", flush=True)
    return asset_map, fold_results


def step6_run_backtest(features, targets, asset_map, fold_results):
    print("\n══ SCHRITT 6: Backtest Long-Only + Long-Short ══", flush=True)
    from strategy.backtest import run_backtest, build_price_cache, plot_equity

    all_assets = list(asset_map.keys())
    price_cache = build_price_cache(all_assets, raw_dir=REPO_DIR / "data" / "raw")

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

    # Ergebnisse speichern
    with open(WORKING / "backtest_results_long_only.json", "w") as f:
        json.dump({k: v for k, v in result_a.items() if k != "trade_log"}, f, indent=2)
    with open(WORKING / "backtest_results_long_short.json", "w") as f:
        json.dump({k: v for k, v in result_b.items() if k != "trade_log"}, f, indent=2)

    # Equity-Kurve plotten
    try:
        plot_equity(result_a, result_b, save_path=str(WORKING / "equity_curve.png"))
    except Exception as e:
        print(f"  [WARN] plot_equity: {e}", flush=True)

    return result_a, result_b


def collect_and_tar(result_a: dict, result_b: dict):
    print("\n══ Artefakte packen ══", flush=True)
    files = [
        WORKING / "backtest_results_long_only.json",
        WORKING / "backtest_results_long_short.json",
        WORKING / "equity_curve.png",
    ]
    # Checkpoints aus dem Repo (vom Training)
    ckpt_dir = REPO_DIR / "checkpoints"
    if ckpt_dir.is_dir():
        files += list(ckpt_dir.glob("*.pt")) + list(ckpt_dir.glob("*.json"))

    artifact_tar = WORKING / "kaggle_artifacts.tar.gz"
    with tarfile.open(str(artifact_tar), "w:gz") as tf:
        for p in files:
            if p.exists():
                tf.add(str(p), arcname=p.name)

    sz = artifact_tar.stat().st_size // 1024
    print(f"  kaggle_artifacts.tar.gz ({sz} KB)", flush=True)

    # Summary
    summary = {
        "return_code":   0,
        "duration_s":    round(time.time() - t0, 1),
        "long_only":     {k: v for k, v in result_a.items() if k not in ("equity", "trade_log", "equity_dates")},
        "long_short":    {k: v for k, v in result_b.items() if k not in ("equity", "trade_log", "equity_dates")},
    }
    (WORKING / "kernel_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    try:
        step1_clone_repo()
        step2_install_deps()
        step3_copy_data()
        features, targets = step4_build_panel()
        asset_map, fold_results = step5_load_asset_map_and_folds()
        result_a, result_b = step6_run_backtest(features, targets, asset_map, fold_results)
        collect_and_tar(result_a, result_b)
        print(f"\n[DONE] Gesamtdauer: {time.time() - t0:.0f}s", flush=True)
        return 0
    except Exception as exc:
        import traceback
        msg = traceback.format_exc()
        print(f"\n[ERROR] {msg}", flush=True)
        (WORKING / "kaggle_cmd_stdout_stderr.txt").write_text(msg)
        (WORKING / "kaggle_cmd_exit_code.txt").write_text("1")
        return 1


if __name__ == "__main__":
    sys.exit(main())
