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
# Kaggle-Dataset-Pfade (probiert alle Varianten)
INPUT_DIRS = [
    Path("/kaggle/input/trading-raw-data"),
    Path("/kaggle/input/trading-raw-data-v2"),
    Path("/kaggle/input/tradingrawdata"),
]
t0 = time.time()


def run(cmd: list, cwd: Path = WORKING, check: bool = True) -> str:
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    result = subprocess.run(
        cmd, cwd=str(cwd),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    out = result.stdout or ""
    print(out, flush=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"rc={result.returncode}: {cmd[0]}")
    return out


def elapsed() -> str:
    return f"{(time.time() - t0) / 60:.1f}min"


# ── Schritt 1: Repo klonen ────────────────────────────────────────────────────

def step_clone():
    print(f"\n{'='*60}\nSCHRITT 1: Repo klonen [{elapsed()}]\n{'='*60}", flush=True)
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    run(["git", "clone", "--depth=1",
         "https://github.com/stevenlangeshops/trading.git",
         str(REPO_DIR)])
    print(f"Repo-Inhalt: {[f.name for f in REPO_DIR.iterdir()]}", flush=True)


# ── Schritt 2: Abhaengigkeiten ────────────────────────────────────────────────

def step_install():
    print(f"\n{'='*60}\nSCHRITT 2: Abhaengigkeiten [{elapsed()}]\n{'='*60}", flush=True)
    # Kaggle hat PyTorch vorinstalliert, wir brauchen nur ta + loguru + scipy
    run([sys.executable, "-m", "pip", "install",
         "ta==0.11.0", "loguru==0.7.2", "scipy",
         "--quiet", "--no-warn-script-location"])


# ── Schritt 3: Parquet-Daten kopieren ────────────────────────────────────────

def step_copy_data():
    print(f"\n{'='*60}\nSCHRITT 3: Parquet-Daten kopieren [{elapsed()}]\n{'='*60}", flush=True)
    raw_dst = REPO_DIR / "data" / "raw"
    raw_dst.mkdir(parents=True, exist_ok=True)

    copied = 0
    for src_dir in INPUT_DIRS:
        if not src_dir.exists():
            print(f"  [SKIP] {src_dir}", flush=True)
            continue
        files = list(src_dir.glob("*.parquet"))
        print(f"  [FOUND] {src_dir}: {len(files)} Parquet-Dateien", flush=True)
        for f in files:
            shutil.copy2(f, raw_dst / f.name)
            copied += 1
        break  # erste vorhandene Quelle reicht

    print(f"  {copied} Dateien nach {raw_dst} kopiert", flush=True)
    if copied == 0:
        # Fallback: raw.zip im Repo entpacken (falls vorhanden)
        raw_zip = REPO_DIR / "data" / "raw.zip"
        if raw_zip.exists():
            import zipfile
            print(f"  Fallback: {raw_zip} entpacken", flush=True)
            with zipfile.ZipFile(raw_zip) as zf:
                zf.extractall(raw_dst)
            copied = len(list(raw_dst.glob("*.parquet")))
            print(f"  {copied} Dateien aus raw.zip entpackt", flush=True)

    if copied == 0:
        raise RuntimeError(
            "Keine Parquet-Daten gefunden!\n"
            "Bitte Dataset 'trading-raw-data' zum Kernel hinzufuegen:\n"
            "  Kaggle UI -> Kernel Settings -> Add Data -> trading-raw-data"
        )
    return copied


# ── Schritt 4: Features bauen ────────────────────────────────────────────────

def step_build_panel():
    print(f"\n{'='*60}\nSCHRITT 4: build_panel() [{elapsed()}]\n{'='*60}", flush=True)
    sys.path.insert(0, str(REPO_DIR))
    os.chdir(str(REPO_DIR))

    from features.engineer import build_panel
    features, targets = build_panel(timeframe="1d", horizon=11, min_rows=300)
    print(f"  features: {features.shape}", flush=True)
    print(f"  targets:  {targets.shape}", flush=True)
    print(f"  Assets:   {features.index.get_level_values('asset').nunique()}", flush=True)
    print(f"  Zeitraum: {features.index.get_level_values('date').min().date()} "
          f"bis {features.index.get_level_values('date').max().date()}", flush=True)
    return features, targets


# ── Schritt 5: Asset-Map ──────────────────────────────────────────────────────

def step_build_asset_map(features):
    print(f"\n{'='*60}\nSCHRITT 5: Asset-Map [{elapsed()}]\n{'='*60}", flush=True)
    assets    = sorted(features.index.get_level_values("asset").unique().tolist())
    asset_map = {a: i + 1 for i, a in enumerate(assets)}  # IDs bei 1 starten
    print(f"  {len(asset_map)} Assets, IDs 1..{max(asset_map.values())}", flush=True)

    asset_map_path = WORKING / "asset_map.json"
    with open(asset_map_path, "w") as f:
        json.dump(asset_map, f, indent=2)
    print(f"  Gespeichert: {asset_map_path}", flush=True)
    return asset_map


# ── Schritt 6: Training ───────────────────────────────────────────────────────

def step_train(features, targets, asset_map):
    print(f"\n{'='*60}\nSCHRITT 6: train_walk_forward() [{elapsed()}]\n{'='*60}", flush=True)

    from models.trainer import train_walk_forward

    results = train_walk_forward(
        features     = features,
        targets      = targets,
        asset_map    = asset_map,
        # Walk-Forward (Doku Abschnitt 8)
        train_years  = 3.0,
        val_months   = 6.0,
        step_months  = 6.0,
        # Modell
        hidden_dim   = 128,
        num_layers   = 2,
        embed_dim    = 16,
        dropout      = 0.3,
        seq_len      = 64,
        # Training
        lr           = 5e-4,
        weight_decay = 1e-3,
        epochs       = 50,
        patience     = 7,
        batch_size   = 512,
        rank_weight  = 0.5,
    )

    # trainer.py liefert fold_results bereits mit ckpt_path, val_start, val_end
    fold_results = results.get("fold_results", [])

    # Ergebnisse persistieren
    wf_path = WORKING / "walk_forward_results.json"
    with open(wf_path, "w") as f:
        json.dump(fold_results, f, indent=2, default=str)
    print(f"  Gespeichert: {wf_path}", flush=True)

    mean_ic = sum(r.get("best_val_ic", 0) for r in fold_results) / max(len(fold_results), 1)
    print(f"  Mean IC ueber {len(fold_results)} Folds: {mean_ic:.4f}", flush=True)
    return fold_results


# ── Schritt 7: Backtest ───────────────────────────────────────────────────────

def step_backtest(features, targets, asset_map, fold_results):
    print(f"\n{'='*60}\nSCHRITT 7: Backtest [{elapsed()}]\n{'='*60}", flush=True)

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
        print(f"  [WARN] plot_equity: {e}", flush=True)

    return result_a, result_b


# ── Schritt 8: Tar-Archiv ─────────────────────────────────────────────────────

def step_pack_artifacts(result_a: dict, result_b: dict):
    print(f"\n{'='*60}\nSCHRITT 8: Artefakte packen [{elapsed()}]\n{'='*60}", flush=True)

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
    print(f"  {tar_path.name}: {sz} KB, {len(seen)} Dateien", flush=True)

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
    print("\n" + json.dumps(summary, indent=2), flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    print(f"Trading Bot Full Pipeline | {time.strftime('%Y-%m-%d %H:%M:%S UTC')}", flush=True)
    try:
        step_clone()
        step_install()
        step_copy_data()
        features, targets = step_build_panel()
        asset_map         = step_build_asset_map(features)
        fold_results      = step_train(features, targets, asset_map)
        result_a, result_b = step_backtest(features, targets, asset_map, fold_results)
        step_pack_artifacts(result_a, result_b)
        print(f"\n[DONE] Gesamtdauer: {elapsed()}", flush=True)
        return 0
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        print(f"\n[ERROR]\n{tb}", flush=True)
        (WORKING / "kaggle_cmd_stdout_stderr.txt").write_text(tb)
        (WORKING / "kaggle_cmd_exit_code.txt").write_text("1")
        (WORKING / "kernel_summary.json").write_text(
            json.dumps({"return_code": 1, "error": str(exc)})
        )
        # Minimal-Tar damit Download-Wrapper immer etwas findet
        tar_path = WORKING / "kaggle_artifacts.tar.gz"
        with tarfile.open(str(tar_path), "w:gz") as tf:
            for fname in ("kaggle_cmd_stdout_stderr.txt", "kaggle_cmd_exit_code.txt", "kernel_summary.json"):
                p = WORKING / fname
                if p.exists():
                    tf.add(str(p), arcname=fname)
        return 1


if __name__ == "__main__":
    sys.exit(main())
