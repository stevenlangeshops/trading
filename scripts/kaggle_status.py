"""
scripts/kaggle_status.py
─────────────────────────
Zeigt den aktuellen Status des Kaggle-Watchers im Terminal an.

Verwendung:
  python scripts/kaggle_status.py          # einmalige Anzeige
  python scripts/kaggle_status.py --watch  # alle 10s automatisch aktualisieren
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

REPO_ROOT   = Path(__file__).resolve().parent.parent
STATUS_FILE = REPO_ROOT / "kaggle_watch_status.json"

# Letzten N Zeilen aus dem Watcher-Terminal lesen
def _find_terminals_dir() -> Path:
    """Sucht das Cursor-Terminals-Verzeichnis für dieses Projekt."""
    home = Path.home()
    base = home / ".cursor" / "projects"
    # Direkt bekannter Pfad
    direct = base / "c-steven-trading-v5-trading" / "terminals"
    if direct.exists():
        return direct
    # Fallback: alle Unterordner mit "trading" im Namen durchsuchen
    if base.exists():
        for proj in base.iterdir():
            if "trading" in proj.name.lower():
                t = proj / "terminals"
                if t.exists():
                    return t
    return direct  # auch wenn nicht vorhanden – damit Fehlermeldung sinnvoll ist

TERMINALS_DIR = _find_terminals_dir()

TAIL_LINES = 20


def find_watcher_terminal() -> Path | None:
    """Sucht die Terminal-Datei des laufenden kaggle_watch.py Prozesses."""
    if not TERMINALS_DIR.exists():
        return None
    candidates = []
    for f in TERMINALS_DIR.glob("*.txt"):
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
            if "kaggle_watch.py" in text:
                candidates.append((f.stat().st_mtime, f))
        except Exception:
            pass
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def tail(path: Path, n: int) -> list[str]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-n:]


def read_status() -> dict:
    if not STATUS_FILE.exists():
        return {"state": "Status-Datei nicht gefunden", "updated_at": "—"}
    return json.loads(STATUS_FILE.read_text(encoding="utf-8"))


def color(text: str, code: str) -> str:
    """ANSI-Farbe falls Terminal sie unterstützt."""
    if sys.stdout.isatty():
        return f"\033[{code}m{text}\033[0m"
    return text


STATE_COLORS = {
    "COMPLETE":  "32",   # grün
    "RUNNING":   "36",   # cyan
    "QUEUED":    "33",   # gelb
    "ERROR":     "31",   # rot
    "FAILED":    "31",   # rot
    "WATCHING":  "34",   # blau
}


def state_color(state: str) -> str:
    for key, code in STATE_COLORS.items():
        if key in state.upper():
            return color(state, code)
    return state


LOG_TAIL = 40   # Zeilen aus pipeline.log anzeigen


def find_pipeline_log(n: int = LOG_TAIL) -> list[str] | None:
    """Sucht pipeline.log in den zuletzt heruntergeladenen Artefakten."""
    run_dir = REPO_ROOT / "kaggle_kernel_runs"
    if not run_dir.exists():
        return None
    logs = sorted(run_dir.rglob("pipeline.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        return None
    lines = logs[0].read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-n:]


def display() -> None:
    status  = read_status()
    term_f  = find_watcher_terminal()
    now_utc = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

    SEP  = "=" * 58
    LINE = "-" * 58
    print()
    print(color(SEP, "1"))
    print(color("  KAGGLE WATCHER STATUS", "1"))
    print(color(SEP, "1"))
    print(f"  Kernel   : {status.get('kernel_id', '-')}")
    print(f"  Zustand  : {state_color(status.get('state', '-'))}")
    print(f"  Retries  : {status.get('retries', 0)} / 5")
    print(f"  Zuletzt  : {status.get('updated_at', '-')}")
    print(f"  Jetzt    : {now_utc}")

    diagnosis = status.get("diagnosis", "")
    if diagnosis:
        print(f"  Diagnose : {color(diagnosis, '33')}")

    run_dir = status.get("run_dir", "")
    if run_dir:
        print(f"  Artefakte: {run_dir}")

    print(color(LINE, "2"))

    if term_f:
        print(f"  Watcher-Log ({term_f.name}, letzte {TAIL_LINES} Zeilen):")
        print()
        for line in tail(term_f, TAIL_LINES):
            if "ERROR" in line or "FAIL" in line or "fehler" in line.lower():
                print("  " + color(line, "31"))
            elif "COMPLETE" in line or "Erfolg" in line:
                print("  " + color(line, "32"))
            elif "RUNNING" in line or "QUEUED" in line:
                print("  " + color(line, "36"))
            elif line.startswith("[watch"):
                print("  " + color(line, "34"))
            else:
                print("  " + line)
    else:
        print("  (Watcher-Terminal nicht gefunden)")

    # pipeline.log aus letztem Artefakt-Download
    pipe_lines = find_pipeline_log()
    if pipe_lines:
        print(color(LINE, "2"))
        print(f"  pipeline.log (letzte {LOG_TAIL} Zeilen aus letztem Download):")
        print()
        for line in pipe_lines:
            if "[ERROR]" in line or "Traceback" in line or "Error" in line:
                print("  " + color(line, "31"))
            elif "[DONE]" in line or "erfolgreich" in line.lower():
                print("  " + color(line, "32"))
            elif "SCHRITT" in line:
                print("  " + color(line, "33"))
            else:
                print("  " + line)

    last_log = status.get("last_log", "")
    if last_log:
        print(color(LINE, "2"))
        print("  Letzter Fehler-Log (letzte 15 Zeilen):")
        print()
        for line in last_log.strip().splitlines()[-15:]:
            print("  " + color(line, "31"))

    # Hinweis auf Live-URL
    kernel_id = status.get("kernel_id", "")
    if kernel_id and "RUNNING" in status.get("state", "").upper():
        print(color(LINE, "2"))
        user, slug = (kernel_id.split("/") + [""])[:2]
        print(f"  Live-Output im Browser: https://www.kaggle.com/code/{user}/{slug}")
        print(f"  (Kaggle-API unterstuetzt kein Live-Log-Streaming)")

    print(color(SEP, "1"))
    print()


def main() -> None:
    p = argparse.ArgumentParser(description="Kaggle Watcher Status-Dashboard")
    p.add_argument("--watch", action="store_true",
                   help="Alle 10 Sekunden automatisch aktualisieren")
    p.add_argument("--interval", type=int, default=10,
                   help="Update-Intervall in Sekunden (default: 10)")
    args = p.parse_args()

    if args.watch:
        print(f"Live-Modus: aktualisiert alle {args.interval}s  (Ctrl+C zum Beenden)")
        try:
            while True:
                # Bildschirm löschen
                os.system("cls" if os.name == "nt" else "clear")
                display()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nBeendet.")
    else:
        display()


if __name__ == "__main__":
    main()
