"""Build a real git repository whose commits are the example snapshots (v1 -> v2 -> v3).

Used by ``scripts/make_demo_history.py``, the tests and the demo. Each snapshot directory is a
complete tree; files missing from a snapshot are deleted in that commit (so git can detect the
``text_utils.py -> preprocessing.py`` rename).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SNAPSHOTS = [
    (ROOT / "examples" / "sample_repo_history" / "v1", "v1: plain-text authentication, basic preprocessing"),
    (ROOT / "examples" / "sample_repo_history" / "v2", "v2: hashed passwords, retries, rename text_utils -> preprocessing"),
    (ROOT / "examples" / "sample_repo", "v3: authenticate -> validate_user, MAX_RETRIES 3 -> 5, token sessions"),
]


def _git(repo: Path, *args: str) -> str:
    env_args = ["-c", "user.name=CodeFusion Demo", "-c", "user.email=demo@codefusion.local", "-c", "commit.gpgsign=false",
                "-c", "core.autocrlf=false"]
    out = subprocess.run(["git", *env_args, "-C", str(repo), *args], capture_output=True, check=True)
    return out.stdout.decode().strip()


def make_demo_history(target: str | Path, snapshots: list[tuple[Path, str]] | None = None) -> list[str]:
    """Create ``target`` as a git repo with one commit per snapshot. Returns commit SHAs (oldest first)."""
    target = Path(target)
    if target.exists():
        shutil.rmtree(target, onerror=_force_remove)
    target.mkdir(parents=True)
    _git(target, "init", "-q")
    _git(target, "checkout", "-q", "-b", "main")
    shas = []
    for snap, message in snapshots or DEFAULT_SNAPSHOTS:
        for p in target.iterdir():
            if p.name == ".git":
                continue
            shutil.rmtree(p) if p.is_dir() else p.unlink()
        for f in sorted(Path(snap).rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts:
                dst = target / f.relative_to(snap)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(f.read_bytes().replace(b"\r\n", b"\n"))
        _git(target, "add", "-A")
        _git(target, "commit", "-q", "-m", message)
        shas.append(_git(target, "rev-parse", "HEAD"))
    return shas


def _force_remove(func, path, _exc):  # pragma: no cover - Windows read-only .git objects
    import os
    import stat

    os.chmod(path, stat.S_IWRITE)
    func(path)
