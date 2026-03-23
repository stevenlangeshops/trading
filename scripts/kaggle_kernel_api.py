import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
import tarfile


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SubmitConfig:
    kernel_id: str  # format: <owner>/<kernel-slug>
    kernel_title: str
    cmd_tokens: list[str]
    timeout_seconds: int
    accelerator: str  # e.g. "gpu"
    poll_seconds: int
    staging_root: Path
    runs_root: Path
    enable_internet: bool
    install_deps: bool
    exclude_upload_files: tuple[str, ...]


def _run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def ensure_kaggle_cli() -> None:
    # Prefer installed `kaggle` command; fall back to `pip install kaggle`.
    if shutil.which("kaggle") is not None:
        return

    print("[kaggle] CLI nicht gefunden -> installiere 'kaggle' via pip...", file=sys.stderr)
    proc = _run([sys.executable, "-m", "pip", "install", "kaggle"])
    if proc.returncode != 0:
        raise RuntimeError(f"kaggle CLI Installation fehlgeschlagen:\n{proc.stdout}")

    if shutil.which("kaggle") is None:
        raise RuntimeError("kaggle CLI ist nach Installation immer noch nicht verfügbar.")


def load_kaggle_token() -> str | None:
    """
    Liest den KGAT-Token aus den möglichen Quellen (Priorität: Env-Var → access_token-Datei).
    Gibt None zurück, wenn kein Token gefunden wurde.
    """
    # Env-Var: KAGGLE_API_TOKEN (neues Format, CLI 2.x)
    token = os.environ.get("KAGGLE_API_TOKEN")
    if token:
        return token.strip()

    # Datei: ~/.kaggle/access_token (neues Format, CLI 2.x)
    access_token_file = Path.home() / ".kaggle" / "access_token"
    if access_token_file.exists():
        try:
            return access_token_file.read_text(encoding="utf-8").strip()
        except Exception:
            pass

    return None


def load_kaggle_credentials() -> tuple[str | None, str | None]:
    """
    Kompatibilitätsfunktion für Legacy-Format (kaggle.json / Env-Vars).
    Gibt (username, key) zurück oder (None, None).
    """
    env_user = os.environ.get("KAGGLE_USERNAME") or os.environ.get("KAGGLE_USER")
    env_key = os.environ.get("KAGGLE_API_KEY") or os.environ.get("KAGGLE_KEY")
    if env_user and env_key:
        return env_user, env_key

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        try:
            data = json.loads(kaggle_json.read_text(encoding="utf-8"))
            return data.get("username"), data.get("key")
        except Exception:
            pass

    return None, None


def require_credentials_or_raise() -> None:
    """Wirft RuntimeError wenn weder KGAT-Token noch Legacy-Credentials vorhanden sind."""
    token = load_kaggle_token()
    if token:
        return

    _user, key = load_kaggle_credentials()
    if key:
        return

    raise RuntimeError(
        "Kaggle-Zugangsdaten fehlen.\n"
        "  Neu (KGAT-Token): Speichere den Token in '~/.kaggle/access_token'\n"
        "                    oder setze Env-Var KAGGLE_API_TOKEN=KGAT_...\n"
        "  Legacy:           Lege '~/.kaggle/kaggle.json' an\n"
        "                    oder setze KAGGLE_USERNAME + KAGGLE_API_KEY."
    )


def build_kaggle_env() -> dict[str, str]:
    """
    Erstellt ein os.environ-Dict mit den korrekten Kaggle-Credentials,
    damit Subprocess-Calls (kaggle kernels push/status/output) authentifiziert sind.
    """
    env = os.environ.copy()

    # Neues Format: KGAT-Token hat Vorrang
    token = load_kaggle_token()
    if token:
        env["KAGGLE_API_TOKEN"] = token
        return env

    # Legacy-Format: Username + Key
    user, key = load_kaggle_credentials()
    if user and key:
        env["KAGGLE_USERNAME"] = user
        env["KAGGLE_KEY"] = key
        return env

    return env


def shlex_to_tokens(cmd: str) -> list[str]:
    # Robust: erlaubt Quotes/Spaces in Argumenten.
    return shlex.split(cmd, posix=True)


def safe_rmtree(path: Path) -> None:
    if not path.exists():
        return
    # Schutz: nur unter staging_root aufräumen.
    root = str(path.resolve())
    if "kaggle_kernel_upload" not in root:
        raise RuntimeError(f"Refuse to delete unexpected path: {root}")
    shutil.rmtree(path)


def copy_code_for_kernel(staging_dir: Path, exclude_files: tuple[str, ...]) -> None:
    # Copy only code + lightweight files. We intentionally skip large data artifacts.
    # Keep directory layout so `from data...` imports work.
    src_dirs = [
        REPO_ROOT / "data",
        REPO_ROOT / "features",
        REPO_ROOT / "models",
        REPO_ROOT / "strategy",
    ]
    root_files = [
        REPO_ROOT / "main.py",
        REPO_ROOT / "requirements.txt",
    ]

    for f in root_files:
        shutil.copy2(f, staging_dir / f.name)

    def ignore_by_name(directory: str, names: list[str]) -> set[str]:
        ignored = set()
        for n in names:
            if n in exclude_files:
                ignored.add(n)
            # Glob-like handling: we currently only need to skip zip artifacts.
            if any(p == "*.zip" for p in exclude_files) and n.lower().endswith(".zip"):
                ignored.add(n)
            if n == "__pycache__":
                ignored.add(n)
        return ignored

    for d in src_dirs:
        dst = staging_dir / d.name
        shutil.copytree(d, dst, dirs_exist_ok=True, ignore=lambda _dir, names, _d=d: ignore_by_name(str(_d), names))


def write_kernel_metadata(staging_dir: Path, cfg: SubmitConfig) -> None:
    _owner, kernel_slug = cfg.kernel_id.split("/", 1)
    title = cfg.kernel_title
    metadata = {
        "id": cfg.kernel_id,
        "title": title,
        "code_file": "kaggle_entrypoint.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "false",
        "enable_gpu": "true",
        "enable_internet": "true" if cfg.enable_internet else "false",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }

    (staging_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def write_task_and_entrypoint(staging_dir: Path, cfg: SubmitConfig) -> None:
    # Task direkt als Python-Dict in den Entrypoint einbetten – keine externe JSON-Datei nötig.
    task = {
        "cmd_tokens": cfg.cmd_tokens,
        "install_deps": cfg.install_deps,
    }

    # Task-Config wird direkt eingebettet – repr() erzeugt gültiges Python (True/False/None, nicht JSON).
    task_repr = repr(task)
    entrypoint = f'''import os
import subprocess
import sys
import tarfile
import time
import json
from pathlib import Path

# ── Eingebettete Task-Konfiguration ───────────────────────────────────────────
TASK = {task_repr}

# ── Kaggle Pfade ──────────────────────────────────────────────────────────────
# Code-Dateien (main.py, data/, ...) landen in /kaggle/working/
# wenn sie über kernels push hochgeladen wurden.
WORKING_DIR = Path("/kaggle/working")
WORKING_DIR.mkdir(parents=True, exist_ok=True)

# sys.path: /kaggle/working/ ergänzen damit `from data...` Imports greifen
if str(WORKING_DIR) not in sys.path:
    sys.path.insert(0, str(WORKING_DIR))


def _run(cmd: list[str], cwd: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, check=False,
    )


def _safe_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _collect_artifacts(working_dir: Path) -> list[Path]:
    found = []
    for subdir, suffixes in [
        ("logs",        {{".log", ".png", ".json"}}),
        ("checkpoints", {{".pt",  ".pth"}}),
        ("results",     {{".csv", ".json", ".png", ".html"}}),
    ]:
        d = working_dir / subdir
        if d.is_dir():
            for p in d.rglob("*"):
                if p.is_file() and p.suffix in suffixes:
                    found.append(p)
    for name in ("equity_curve.png",):
        p = working_dir / name
        if p.exists():
            found.append(p)
    return found


def _tar_files(output_tar: Path, files: list[Path]) -> None:
    with tarfile.open(str(output_tar), "w:gz") as tf:
        seen: set[str] = set()
        for p in files:
            key = str(p.resolve())
            if p.exists() and key not in seen:
                seen.add(key)
                tf.add(str(p), arcname=p.name)


def main() -> int:
    t0 = time.time()
    summary: dict = {{"start_ts": t0, "cmd": TASK.get("cmd_tokens"), "return_code": None, "duration_s": None}}
    rc = 1
    combined_out = ""

    stdout_path  = WORKING_DIR / "kaggle_cmd_stdout_stderr.txt"
    exit_path    = WORKING_DIR / "kaggle_cmd_exit_code.txt"
    summary_path = WORKING_DIR / "kernel_summary.json"
    artifact_tar = WORKING_DIR / "kaggle_artifacts.tar.gz"

    try:
        if TASK.get("install_deps", False):
            req = WORKING_DIR / "requirements.txt"
            dep_proc = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(req)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, check=False,
            )
            combined_out += dep_proc.stdout or ""
            if dep_proc.returncode != 0:
                rc = dep_proc.returncode
                return rc

        cmd_tokens = TASK["cmd_tokens"]
        proc = _run(cmd_tokens, cwd=str(WORKING_DIR))
        rc = proc.returncode
        combined_out += proc.stdout or ""
        return rc

    except Exception as exc:
        combined_out += f"\\n[entrypoint ERROR] {{type(exc).__name__}}: {{exc}}\\n"
        rc = 99
        return rc

    finally:
        summary["return_code"] = rc
        summary["duration_s"]  = time.time() - t0

        _safe_write(stdout_path,  combined_out)
        _safe_write(exit_path,    str(rc))
        _safe_write(summary_path, json.dumps(summary, indent=2))

        artifacts = _collect_artifacts(WORKING_DIR)
        artifacts += [stdout_path, exit_path, summary_path]

        if artifact_tar.exists():
            artifact_tar.unlink()
        _tar_files(artifact_tar, artifacts)

        sz = artifact_tar.stat().st_size // 1024 if artifact_tar.exists() else 0
        print(f"[kaggle_entrypoint] return_code={{rc}}")
        print(f"[kaggle_entrypoint] artifacts.tar.gz geschrieben ({{sz}} KB)")

    return rc


if __name__ == "__main__":
    sys.exit(main())
'''
    (staging_dir / "kaggle_entrypoint.py").write_text(entrypoint, encoding="utf-8")


def prepare_staging_dir(cfg: SubmitConfig) -> Path:
    staging_dir = cfg.staging_root / cfg.kernel_id.replace("/", "__")
    safe_rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    copy_code_for_kernel(staging_dir, exclude_files=cfg.exclude_upload_files)
    write_kernel_metadata(staging_dir, cfg)
    write_task_and_entrypoint(staging_dir, cfg)
    return staging_dir


def parse_status_complete_or_failed(status_text: str) -> tuple[bool, bool]:
    t = status_text.lower()
    complete = "complete" in t or "completed" in t
    failed = "failed" in t or "error" in t
    return complete, failed


def poll_kernel_until_done(
    kernel_id: str,
    poll_seconds: int,
    timeout_seconds: int,
    runs_dir: Path,
    kaggle_env: dict[str, str],
) -> None:
    t_deadline = time.time() + timeout_seconds
    status_log = runs_dir / "kernel_status_poll.log"
    status_log.parent.mkdir(parents=True, exist_ok=True)
    poll_count = 0

    while True:
        proc = _run(["kaggle", "kernels", "status", kernel_id], env=kaggle_env)
        out = proc.stdout or ""
        status_log.write_text(out, encoding="utf-8")
        poll_count += 1
        print(f"[kaggle] Status-Poll #{poll_count}: {out.strip()}", file=sys.stderr)

        complete, failed = parse_status_complete_or_failed(out)
        if complete or failed:
            return

        if time.time() > t_deadline:
            raise TimeoutError(f"Kaggle kernel did not finish within {timeout_seconds}s: {kernel_id}")

        time.sleep(poll_seconds)


def download_kernel_artifacts(
    kernel_id: str,
    download_dir: Path,
    kaggle_env: dict[str, str],
) -> Path | None:
    download_dir.mkdir(parents=True, exist_ok=True)

    # Regex: only grab our tarball.
    pattern = r"kaggle_artifacts\.tar\.gz$"
    proc = _run(
        [
            "kaggle",
            "kernels",
            "output",
            kernel_id,
            "-p",
            str(download_dir),
            "--file-pattern",
            pattern,
            "-q",
        ],
        env=kaggle_env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"kaggle kernels output fehlgeschlagen:\n{proc.stdout}")

    candidates = list(download_dir.glob("kaggle_artifacts*.tar.gz"))
    if not candidates:
        return None
    # Pick the newest one.
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


def extract_tar(tar_path: Path, extract_to: Path) -> None:
    extract_to.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(extract_to)


def cmd_run(args: argparse.Namespace) -> int:
    cmd_tokens = shlex_to_tokens(args.cmd)
    if not cmd_tokens:
        raise ValueError("--cmd darf nicht leer sein.")

    _owner, kernel_slug = args.kernel_id.split("/", 1)
    kernel_title = args.kernel_title or kernel_slug

    cfg = SubmitConfig(
        kernel_id=args.kernel_id,
        kernel_title=kernel_title,
        cmd_tokens=cmd_tokens,
        timeout_seconds=args.timeout_seconds,
        accelerator=args.accelerator,
        poll_seconds=args.poll_seconds,
        staging_root=Path(args.staging_root),
        runs_root=Path(args.runs_root),
        enable_internet=bool(args.enable_internet),
        install_deps=bool(args.install_deps),
        exclude_upload_files=(
            "raw.zip",  # large local data artifacts
            "*.zip",
        ),
    )

    ensure_kaggle_cli()
    require_credentials_or_raise()
    kaggle_env = build_kaggle_env()

    runs_dir = cfg.runs_root / cfg.kernel_id.replace("/", "__") / time.strftime("%Y%m%d-%H%M%S")
    runs_dir.mkdir(parents=True, exist_ok=True)

    staging_dir = prepare_staging_dir(cfg)
    (runs_dir / "staging_dir.txt").write_text(str(staging_dir), encoding="utf-8")

    push_cmd = ["kaggle", "kernels", "push", "-p", str(staging_dir), "-t", str(cfg.timeout_seconds)]
    if cfg.accelerator:
        push_cmd.extend(["--accelerator", cfg.accelerator])

    push_log = runs_dir / "kaggle_push.log"
    print(f"[kaggle] push: {cfg.kernel_id} (timeout={cfg.timeout_seconds}s, accelerator={cfg.accelerator})", file=sys.stderr)
    push_proc = _run(push_cmd, env=kaggle_env)
    push_log.write_text(push_proc.stdout or "", encoding="utf-8")
    print(push_proc.stdout or "", file=sys.stderr)
    if push_proc.returncode != 0:
        raise RuntimeError(f"kaggle kernels push fehlgeschlagen:\n{push_proc.stdout}")

    print(f"[kaggle] Kernel gestartet. Polling alle {cfg.poll_seconds}s ...", file=sys.stderr)
    poll_kernel_until_done(cfg.kernel_id, cfg.poll_seconds, cfg.timeout_seconds, runs_dir, kaggle_env)

    # Download and extract artifacts.
    download_dir = runs_dir / "download"
    tar_path = download_kernel_artifacts(cfg.kernel_id, download_dir, kaggle_env)
    if tar_path is not None:
        extract_to = runs_dir / "artifacts"
        extract_tar(tar_path, extract_to)
        print(f"[kaggle] Fertig. Artefakte ausgepackt nach:\n  {extract_to}", file=sys.stderr)
    else:
        print("[kaggle] Kein artifacts.tar.gz gefunden (Kernel evtl. noch nicht output-ready).", file=sys.stderr)

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Kaggle Kernel Runner (push/status/output + Artifact-Download).")
    sub = p.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Kernel pushen und bis completion poll-en.")
    p_run.add_argument("--kernel-id", required=True, help="Format: <owner>/<kernel-slug>")
    p_run.add_argument("--kernel-title", default=None, help="Kernel Titel (Slug muss passen; default=kernel-slug)")
    p_run.add_argument(
        "--cmd",
        required=True,
        help="Command, die im Kaggle Kernel ausgefuehrt wird, z.B. \"python main.py train --ticker combined --timeframe 1h --mode reg --epochs 2\"",
    )
    p_run.add_argument("--timeout-seconds", type=int, default=3600)
    p_run.add_argument("--accelerator", default="gpu", help="Kaggle accelerators: häufig 'gpu' oder 'cpu'.")
    p_run.add_argument("--poll-seconds", type=int, default=30)
    p_run.add_argument("--staging-root", default=str(REPO_ROOT / ".kaggle_kernel_upload"))
    p_run.add_argument("--runs-root", default=str(REPO_ROOT / "kaggle_kernel_runs"))
    p_run.add_argument("--enable-internet", action="store_true", help="Internet im Kernel erlauben (für pip install).")
    p_run.add_argument("--install-deps", action="store_true", help="requirements.txt im Kernel installieren.")
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        return cmd_run(args)
    raise AssertionError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

