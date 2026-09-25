"""Git access via GitPython (preferred) or the git CLI.

Blob SHAs from ``git ls-tree`` identify file *contents*: a file whose blob SHA did not change
between commits is not re-read, re-parsed or re-embedded.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from codefusion.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class CommitInfo:
    sha: str
    message: str
    author: str
    date: str
    parent: str | None

    def to_dict(self) -> dict:
        return {"sha": self.sha, "message": self.message, "author": self.author, "date": self.date, "parent": self.parent}


@dataclass
class FileChange:
    status: str  # A, M, D, R
    path: str
    old_path: str | None = None
    similarity: int | None = None


class GitTracker:
    def __init__(self, repo_path: str | Path) -> None:
        self.path = Path(repo_path).resolve()
        self._repo = None
        try:
            import git  # GitPython

            self._repo = git.Repo(self.path)
        except Exception as exc:  # noqa: BLE001 - fall back to plain CLI
            log.debug("GitPython unavailable, using git CLI", extra={"data": {"error": repr(exc)}})
        if not (self.path / ".git").exists() and self._repo is None:
            self._run("rev-parse", "--git-dir")  # raises if not a repository

    def _run(self, *args: str) -> str:
        if self._repo is not None:
            return self._repo.git.execute(["git", *args], stdout_as_string=True, strip_newline_in_stdout=False)
        out = subprocess.run(["git", "-C", str(self.path), *args], capture_output=True, check=True)
        return out.stdout.decode("utf-8", errors="replace")

    @staticmethod
    def is_repo(path: str | Path) -> bool:
        try:
            subprocess.run(["git", "-C", str(path), "rev-parse", "--git-dir"], capture_output=True, check=True)
            return True
        except Exception:  # noqa: BLE001
            return False

    def resolve(self, rev: str) -> str:
        return self._run("rev-parse", rev).strip()

    def head(self) -> str:
        return self.resolve("HEAD")

    def commits(self, rev: str = "HEAD", max_count: int | None = None) -> list[CommitInfo]:
        """Commits reachable from ``rev`` (first-parent), oldest first."""
        args = ["log", "--first-parent", "--format=%H%x1f%P%x1f%an%x1f%aI%x1f%s%x1e", rev]
        if max_count:
            args.insert(1, f"--max-count={max_count}")
        out = []
        for rec in self._run(*args).split("\x1e"):
            rec = rec.strip()
            if not rec:
                continue
            sha, parents, author, date, msg = (rec.split("\x1f") + [""] * 5)[:5]
            out.append(CommitInfo(sha, msg, author, date, parents.split()[0] if parents.strip() else None))
        return list(reversed(out))

    def tree(self, commit: str) -> dict[str, str]:
        """``{path: blob_sha}`` for all blobs in ``commit``."""
        files: dict[str, str] = {}
        for line in self._run("ls-tree", "-r", "--full-tree", commit).splitlines():
            meta, _, path = line.partition("\t")
            parts = meta.split()
            if len(parts) == 3 and parts[1] == "blob":
                files[path] = parts[2]
        return files

    def read(self, commit: str, path: str) -> str:
        # GitPython reads blobs from the object database in-process (no subprocess per file).
        if self._repo is not None:
            try:
                blob = self._repo.commit(commit).tree / path
                return blob.data_stream.read().decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001 - fall back to the CLI
                pass
        return self._run("show", f"{commit}:{path}")

    def read_blob(self, blob_sha: str) -> str:
        return self._run("cat-file", "-p", blob_sha)

    def diff(self, a: str, b: str) -> list[FileChange]:
        out = []
        for line in self._run("diff", "--name-status", "-M", a, b).splitlines():
            parts = line.split("\t")
            if not parts or not parts[0]:
                continue
            st = parts[0]
            if st.startswith("R") and len(parts) >= 3:
                out.append(FileChange("R", parts[2], parts[1], int(st[1:] or 0)))
            elif len(parts) >= 2:
                out.append(FileChange(st[0], parts[1]))
        return out

    def renames(self, a: str, b: str) -> dict[str, str]:
        """``{new_path: old_path}`` for files git detects as renamed."""
        return {c.path: c.old_path for c in self.diff(a, b) if c.status == "R" and c.old_path}


class DirectorySource:
    """Non-git directory treated like a single working-tree 'commit'; blob id = sha1(content)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def tree(self, include_ext: list[str], exclude_dirs: list[str], max_bytes: int) -> dict[str, str]:
        files: dict[str, str] = {}
        excl = set(exclude_dirs)
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(d for d in dirnames if d not in excl and not d.startswith("."))
            for fn in sorted(filenames):
                if not any(fn.endswith(e) for e in include_ext):
                    continue
                p = Path(dirpath) / fn
                try:
                    data = p.read_bytes()
                except OSError:
                    continue
                if len(data) > max_bytes:
                    continue
                rel = p.relative_to(self.root).as_posix()
                files[rel] = hashlib.sha1(data).hexdigest()
        return files

    def read(self, path: str) -> str:
        return (self.root / path).read_text(encoding="utf-8", errors="replace")
