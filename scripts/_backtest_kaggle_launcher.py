"""Startet den Backtest-Kaggle-Run via kaggle_kernel_api."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
from scripts.kaggle_kernel_api import (
    SubmitConfig, cmd_run, build_kaggle_env,
    ensure_kaggle_cli, require_credentials_or_raise,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

args = argparse.Namespace(
    kernel_id="busersteven/trading-bot-v5-fullrun",
    kernel_title="trading-bot-v5-fullrun",
    cmd="placeholder",           # wird nicht genutzt (custom_entrypoint)
    timeout_seconds=12000,       # 3.3h max (Kaggle Limit = 12h)
    accelerator="gpu",
    poll_seconds=120,            # alle 2 Minuten Status abfragen
    staging_root=str(REPO_ROOT / ".kaggle_kernel_upload"),
    runs_root=str(REPO_ROOT / "kaggle_kernel_runs"),
    enable_internet=True,
    install_deps=False,
    custom_entrypoint=str(REPO_ROOT / "scripts" / "kaggle_full_run.py"),
    dataset_sources="busersteven/trading-raw-data",
)

sys.exit(cmd_run(args))
