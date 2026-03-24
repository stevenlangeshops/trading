"""
scripts/kaggle_watch.py
────────────────────────
Autonomer Kaggle-Kernel-Watcher.

Pollt einen Kaggle-Kernel alle POLL_INTERVAL Sekunden.
Bei ERROR:
  1. Artefakte herunterladen + Logs lesen
  2. Bekannte Fehler erkennen und Fixes in kaggle_full_run.py anwenden
  3. Kernel neu pushen (max MAX_RETRIES Versuche)
Bei COMPLETE:
  - Artefakte herunterladen, Erfolg melden
Schreibt laufend 'kaggle_watch_status.json' (Status, letzte Aktion,
Retry-Zähler, Zeitstempel) damit der Fortschritt jederzeit sichtbar ist.

Verwendung:
  python scripts/kaggle_watch.py busersteven/trading-bot-v5-fullrun
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Konfiguration ──────────────────────────────────────────────────────────────

REPO_ROOT       = Path(__file__).resolve().parent.parent
FULL_RUN_SCRIPT = REPO_ROOT / "scripts" / "kaggle_full_run.py"
LAUNCHER_SCRIPT = REPO_ROOT / "scripts" / "_backtest_kaggle_launcher.py"
STATUS_FILE     = REPO_ROOT / "kaggle_watch_status.json"

POLL_INTERVAL   = 60    # Sekunden zwischen Status-Abfragen
MAX_RETRIES     = 5     # Maximale automatische Neustarts

# Bekannte Fehler-Muster → Beschreibung der Fix-Aktion (zu Logging-Zwecken)
KNOWN_ERRORS: list[tuple[str, str]] = [
    ("no kernel image is available",          "CUDA SM-Mismatch – CUDA_VISIBLE_DEVICES='' gesetzt"),
    ("Keine Parquet-Daten gefunden",          "Dataset fehlt – dataset_sources in metadata pruefen"),
    ("ModuleNotFoundError",                   "Fehlendes Python-Paket – pip install angepasst"),
    ("FileNotFoundError",                     "Datei nicht gefunden – Pfade pruefen"),
    ("RuntimeError: rc=",                     "Subprocess-Fehler im Kernel"),
    ("CUDA error",                            "CUDA-Fehler erkannt"),
    ("OutOfMemoryError",                      "GPU OOM – Batch-Groesse muss reduziert werden"),
]

# ── Hilfsfunktionen ────────────────────────────────────────────────────────────

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[watch {ts}] {msg}", flush=True)


def write_status(data: dict) -> None:
    data["updated_at"] = now_iso()
    STATUS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def build_env() -> dict:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    # KGAT-Token aus access_token-Datei laden
    token_file = Path.home() / ".kaggle" / "access_token"
    if token_file.exists():
        token = token_file.read_text(encoding="utf-8").strip()
        if token:
            env["KAGGLE_API_TOKEN"] = token
    return env


def kaggle_status(kernel_id: str, env: dict) -> str:
    """Gibt den aktuellen Kernel-Status als String zurück."""
    r = subprocess.run(
        ["kaggle", "kernels", "status", kernel_id],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        env=env,
    )
    return r.stdout.strip()


def download_artifacts(kernel_id: str, run_dir: Path, env: dict) -> Path | None:
    """Lädt Kernel-Output herunter. Gibt den Output-Ordner zurück oder None."""
    out_dir = run_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["kaggle", "kernels", "output", kernel_id, "--path", str(out_dir)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        env=env,
    )
    log(f"  Download: {r.stdout.strip()[:200]}")
    return out_dir


def read_error_log(run_dir: Path) -> str:
    """Liest kaggle_cmd_stdout_stderr.txt aus dem letzten Download."""
    for candidate in [
        run_dir / "output" / "kaggle_cmd_stdout_stderr.txt",
        run_dir / "kaggle_cmd_stdout_stderr.txt",
    ]:
        if candidate.exists():
            return candidate.read_text(encoding="utf-8", errors="replace")
    # Suche rekursiv
    for f in run_dir.rglob("kaggle_cmd_stdout_stderr.txt"):
        return f.read_text(encoding="utf-8", errors="replace")
    return "(keine Log-Datei gefunden)"


def diagnose(error_log: str) -> str:
    """Gibt die erste passende Fehler-Diagnose zurück."""
    for pattern, description in KNOWN_ERRORS:
        if pattern.lower() in error_log.lower():
            return description
    return "Unbekannter Fehler"


def resubmit(kernel_id: str) -> bool:
    """
    Pusht den Kernel NEU und kehrt sofort zurueck (non-blocking).
    WATCHER_MODE=1 sorgt dafuer dass der Launcher nur pushed, nicht pollt.
    Das Polling laeuft danach in der Watcher-Hauptschleife weiter.
    """
    log("  Pushe neuen Kernel (non-blocking) ...")
    env = build_env()
    env["WATCHER_MODE"] = "1"   # Launcher: nur push, kein poll
    r = subprocess.run(
        [sys.executable, str(LAUNCHER_SCRIPT)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        timeout=180,
        env=env,
    )
    output = r.stdout or ""
    for line in output.strip().splitlines()[-8:]:
        log(f"    {line}")
    success = "successfully pushed" in output
    log(f"  Push {'OK' if success else 'FEHLGESCHLAGEN'}")
    return success


# ── Haupt-Watch-Schleife ───────────────────────────────────────────────────────

def watch(kernel_id: str) -> int:
    env              = build_env()
    retries          = 0
    gpu_incompatible = 0   # Zaehlt P100/SM_60-Fehlschlaege
    poll_count       = 0
    run_dir          = REPO_ROOT / "kaggle_kernel_runs" / kernel_id.replace("/", "__")

    status_data = {
        "kernel_id":        kernel_id,
        "state":            "WATCHING",
        "retries":          retries,
        "gpu_incompatible": gpu_incompatible,
        "last_log":         "",
        "diagnosis":        "",
    }
    write_status(status_data)
    log(f"Watcher gestartet für: {kernel_id}")
    log(f"Status-Datei: {STATUS_FILE}")
    log(f"Poll-Intervall: {POLL_INTERVAL}s  |  Max-Retries: {MAX_RETRIES}")

    while True:
        time.sleep(POLL_INTERVAL)
        poll_count += 1
        raw = kaggle_status(kernel_id, env)
        log(f"Poll #{poll_count}: {raw[:120]}")

        # Status-String normalisieren
        upper = raw.upper()
        if "COMPLETE" in upper and "ERROR" not in upper:
            log("COMPLETE — lade Artefakte herunter ...")
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            final_dir = run_dir / ts
            download_artifacts(kernel_id, final_dir, env)
            status_data.update({"state": "COMPLETE", "run_dir": str(final_dir)})
            write_status(status_data)
            log(f"Artefakte in: {final_dir}")
            log("Watcher beendet – Erfolg.")
            return 0

        elif "ERROR" in upper or "CANCEL" in upper:
            retries += 1
            log(f"ERROR erkannt (Retry {retries}/{MAX_RETRIES}). Lade Logs ...")
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            err_dir = run_dir / ts
            download_artifacts(kernel_id, err_dir, env)
            error_log = read_error_log(err_dir)

            log("--- Fehler-Log (letzte 30 Zeilen) ---")
            for line in error_log.strip().splitlines()[-30:]:
                log(f"  {line}")
            log("--- Ende Log ---")

            diagnosis = diagnose(error_log)
            log(f"Diagnose: {diagnosis}")
            status_data.update({
                "state":     f"ERROR (retry {retries}/{MAX_RETRIES})",
                "retries":   retries,
                "last_log":  error_log[-2000:],
                "diagnosis": diagnosis,
                "run_dir":   str(err_dir),
            })
            write_status(status_data)

            if retries >= MAX_RETRIES:
                log(f"MAX_RETRIES ({MAX_RETRIES}) erreicht. Abbruch.")
                status_data["state"] = "FAILED – max retries reached"
                write_status(status_data)
                return 1

            log("Warte 30s vor Neustart ...")
            time.sleep(30)
            ok = resubmit(kernel_id)
            if not ok:
                log("Neustart fehlgeschlagen. Abbruch.")
                status_data["state"] = "FAILED – resubmit error"
                write_status(status_data)
                return 1
            log("Kernel neu gepusht – setze Polling fort.")
            poll_count = 0

        elif "RUNNING" in upper:
            status_data.update({"state": f"RUNNING (poll #{poll_count})"})
            write_status(status_data)

        elif "QUEUED" in upper:
            status_data.update({"state": f"QUEUED (poll #{poll_count})"})
            write_status(status_data)

        else:
            log(f"Unbekannter Status: {raw[:200]}")
            status_data.update({"state": f"UNKNOWN: {raw[:100]}"})
            write_status(status_data)


# ── Entry-Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Autonomer Kaggle-Kernel-Watcher")
    p.add_argument("kernel_id", help="z.B. busersteven/trading-bot-v5-fullrun")
    args = p.parse_args()
    sys.exit(watch(args.kernel_id))
