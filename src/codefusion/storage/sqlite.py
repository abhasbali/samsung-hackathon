"""SQLite persistence for snippet versions, commits, commit membership and lineage."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from codefusion.types import Snippet

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS snippets (
    idx INTEGER PRIMARY KEY,
    snippet_id TEXT NOT NULL,
    repository TEXT, file_path TEXT, symbol_key TEXT, lineage_id TEXT,
    type TEXT, name TEXT, content_hash TEXT, ast_hash TEXT,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_snip_id ON snippets(snippet_id);
CREATE INDEX IF NOT EXISTS ix_snip_symbol ON snippets(symbol_key);
CREATE TABLE IF NOT EXISTS commits (
    sha TEXT PRIMARY KEY, ord INTEGER, repository TEXT, message TEXT, author TEXT, date TEXT, parent TEXT
);
CREATE TABLE IF NOT EXISTS commit_members (sha TEXT, idx INTEGER, PRIMARY KEY (sha, idx));
CREATE TABLE IF NOT EXISTS lineage (old_idx INTEGER, new_idx INTEGER, relation TEXT, similarity REAL, PRIMARY KEY (old_idx, new_idx));
"""


class IndexDB:
    def __init__(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.executescript(SCHEMA)

    def set_meta(self, key: str, value: Any) -> None:
        self.conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, json.dumps(value)))

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def write_snippets(self, snippets: list[Snippet], start: int = 0) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO snippets VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [
                (start + i, s.snippet_id, s.repository, s.file_path, s.symbol_key, s.lineage_id, s.type, s.name,
                 s.content_hash, s.ast_hash, json.dumps(s.to_dict(), ensure_ascii=False))
                for i, s in enumerate(snippets)
            ],
        )

    def read_snippets(self) -> list[Snippet]:
        return [Snippet.from_dict(json.loads(d)) for (d,) in self.conn.execute("SELECT data FROM snippets ORDER BY idx")]

    def write_commits(self, commits: list[dict[str, Any]], members: dict[str, list[int]]) -> None:
        self.conn.execute("DELETE FROM commits")
        self.conn.execute("DELETE FROM commit_members")
        self.conn.executemany(
            "INSERT INTO commits VALUES (?,?,?,?,?,?,?)",
            [(c["sha"], i, c.get("repository"), c.get("message"), c.get("author"), c.get("date"), c.get("parent")) for i, c in enumerate(commits)],
        )
        self.conn.executemany("INSERT INTO commit_members VALUES (?,?)", [(sha, int(i)) for sha, idxs in members.items() for i in idxs])

    def read_commits(self) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
        commits = [
            {"sha": r[0], "repository": r[2], "message": r[3], "author": r[4], "date": r[5], "parent": r[6]}
            for r in self.conn.execute("SELECT * FROM commits ORDER BY ord")
        ]
        members: dict[str, list[int]] = {}
        for sha, idx in self.conn.execute("SELECT sha, idx FROM commit_members ORDER BY sha, idx"):
            members.setdefault(sha, []).append(idx)
        return commits, members

    def write_lineage(self, edges: list[tuple[int, int, str, float]]) -> None:
        self.conn.execute("DELETE FROM lineage")
        self.conn.executemany("INSERT OR REPLACE INTO lineage VALUES (?,?,?,?)", edges)

    def read_lineage(self) -> list[tuple[int, int, str, float]]:
        return [tuple(r) for r in self.conn.execute("SELECT old_idx, new_idx, relation, similarity FROM lineage")]  # type: ignore[misc]

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()
