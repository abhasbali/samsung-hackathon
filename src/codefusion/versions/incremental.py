"""Version-aware, incremental repository indexing (P1) with lineage tracking (bonus).

For commit B after commit A::

    git ls-tree B  ->  compare blob SHAs with A  ->  changed / added / deleted files
    parse + chunk changed files only
    snippets whose (symbol_key, content_hash) already exist are reused (no re-embedding)
    new snippet versions -> embeddings (cache-aware), symbol index, graph (append)
    BM25 rebuilt from cached tokens (seconds), commit membership registered
    lineage: old vs new snippets of changed files, using git rename detection

The same code path indexes a plain directory (no git): blob ids are content SHA-1s and the
"commit" is ``worktree``; re-running after edits only re-processes changed files.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from codefusion.logging_utils import get_logger
from codefusion.parsing.chunker import ChunkerConfig, chunk_source
from codefusion.parsing.tree_sitter_parser import detect_language
from codefusion.retrieval.pipeline import CodeFusionEngine
from codefusion.types import Snippet
from codefusion.versions.git_tracker import CommitInfo, DirectorySource, GitTracker
from codefusion.versions.lineage import LineageMatcher, new_lineage_id

log = get_logger(__name__)


@dataclass
class UpdateStats:
    repository: str
    commit: str
    base_commit: str | None
    files_total: int = 0
    files_changed: int = 0
    files_added: int = 0
    files_deleted: int = 0
    files_reused: int = 0
    snippets_live: int = 0
    snippets_new_versions: int = 0
    snippets_reused: int = 0
    embeddings_computed: int = 0
    embeddings_reused: int = 0
    lineage_edges: int = 0
    lineage_relations: dict[str, int] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _chunker_cfg(engine: CodeFusionEngine) -> ChunkerConfig:
    c = engine.cfg.chunking
    return ChunkerConfig(
        max_chunk_lines=c.max_chunk_lines,
        max_module_block_lines=c.max_module_block_lines,
        class_skeletons=c.class_skeletons,
        include_module_blocks=c.include_module_blocks,
        prefer_tree_sitter=c.prefer_tree_sitter,
    )


class IncrementalIndexer:
    def __init__(self, engine: CodeFusionEngine, repo_path: str | Path, repository: str | None = None) -> None:
        self.engine = engine
        self.path = Path(repo_path).resolve()
        self.repository = repository or self.path.name
        self.is_git = GitTracker.is_repo(self.path)
        self.git = GitTracker(self.path) if self.is_git else None
        self.dir_source = None if self.is_git else DirectorySource(self.path)
        self.matcher = LineageMatcher(engine.cfg.versions.lineage_similarity)

    # ------------------------------------------------------------------ state
    @property
    def state(self) -> dict[str, Any]:
        return self.engine.store.repo_state.setdefault(self.repository, {"commit": None, "files": {}, "indexed_commits": []})

    def _filter(self, tree: dict[str, str]) -> dict[str, str]:
        c = self.engine.cfg.chunking
        excl = set(c.exclude_dirs)
        out = {}
        for p, blob in tree.items():
            if not any(p.endswith(e) for e in c.include_extensions):
                continue
            if any(part in excl for part in p.split("/")[:-1]):
                continue
            out[p] = blob
        return out

    # ------------------------------------------------------------------ main entry points
    def index_commit(self, rev: str = "HEAD", make_latest: bool = True) -> UpdateStats:
        if self.git is None:
            raise ValueError(f"{self.path} is not a git repository; use index_worktree()")
        sha = self.git.resolve(rev)
        info = next((c for c in self.git.commits(sha, max_count=1)), CommitInfo(sha, "", "", "", None))
        tree = self._filter(self.git.tree(sha))
        renames = self.git.renames(self.state["commit"], sha) if self.state["commit"] else {}
        return self._index(sha, info.to_dict(), tree, lambda p: self.git.read(sha, p), renames, make_latest)  # type: ignore[union-attr]

    def index_history(self, rev: str = "HEAD", max_commits: int | None = None) -> list[UpdateStats]:
        """Index every first-parent commit up to ``rev`` (oldest first), incrementally."""
        if self.git is None:
            raise ValueError("history indexing needs a git repository")
        commits = self.git.commits(rev)
        if max_commits:
            commits = commits[-max_commits:]
        return [self.index_commit(c.sha) for c in commits]

    def index_worktree(self) -> UpdateStats:
        src = self.dir_source or DirectorySource(self.path)
        c = self.engine.cfg.chunking
        tree = src.tree(c.include_extensions, c.exclude_dirs, c.max_file_bytes)
        sha = "worktree"
        info = {"sha": sha, "message": "working tree", "author": "", "date": time.strftime("%Y-%m-%dT%H:%M:%S"), "parent": self.state["commit"]}
        return self._index(sha, info, tree, src.read, {}, True)

    # ------------------------------------------------------------------ core
    def _index(self, sha: str, info: dict[str, Any], tree: dict[str, str], read, renames: dict[str, str], make_latest: bool) -> UpdateStats:
        eng = self.engine
        st = self.state
        base = st["commit"]
        prev_files: dict[str, dict[str, Any]] = st["files"]
        stats = UpdateStats(self.repository, sha, base)
        stats.files_total = len(tree)
        emb0 = (eng.embedder.stats.computed, eng.embedder.stats.reused) if eng.embedder else (0, 0)
        t0 = time.perf_counter()

        changed = [p for p, b in tree.items() if p not in prev_files or prev_files[p]["blob"] != b]
        deleted = [p for p in prev_files if p not in tree]
        stats.files_added = sum(1 for p in changed if p not in prev_files)
        stats.files_changed = len(changed) - stats.files_added
        stats.files_deleted = len(deleted)
        stats.files_reused = len(tree) - len(changed)

        # Parse + chunk only changed files.
        t_parse = time.perf_counter()
        new_file_snips: dict[str, list[Snippet]] = {}
        for p in changed:
            try:
                text = read(p)
            except Exception as exc:  # noqa: BLE001
                log.warning("could not read file", extra={"data": {"path": p, "error": repr(exc)}})
                continue
            lang = detect_language(p) or "text"
            new_file_snips[p] = chunk_source(text, p, self.repository, sha, lang, _chunker_cfg(eng))
        stats.seconds["parse"] = round(time.perf_counter() - t_parse, 4)

        # Add snippet versions (existing versions are reused by snippet_id).
        t_add = time.perf_counter()
        all_new = [s for snips in new_file_snips.values() for s in snips]
        before = len(eng.store.snippets)
        idxs = eng.add_snippets(all_new)
        stats.snippets_new_versions = len(eng.store.snippets) - before
        stats.snippets_reused = len(all_new) - stats.snippets_new_versions
        stats.seconds["embed_and_index"] = round(time.perf_counter() - t_add, 4)

        # Lineage between the old and new versions of changed/deleted files.
        t_lin = time.perf_counter()
        pos = 0
        new_members: dict[str, list[int]] = {}
        for p, snips in new_file_snips.items():
            new_members[p] = idxs[pos : pos + len(snips)]
            pos += len(snips)
        old_pairs: list[tuple[int, Snippet]] = []
        for p in list(changed) + deleted:
            old_path = p if p in prev_files else None
            if old_path is None:
                continue
            old_pairs += [(i, eng.store.snippets[i]) for i in prev_files[old_path]["idxs"]]
        # (Old paths of git-renamed files are in `deleted`, so their snippets are already in old_pairs.)
        new_idx_set = {i for lst in new_members.values() for i in lst}
        old_idx_set = {i for i, _ in old_pairs}
        # Versions present in both (unchanged snippets inside a changed file) keep their identity.
        old_pairs = [(i, s) for i, s in old_pairs if i not in new_idx_set]
        new_pairs = [(i, eng.store.snippets[i]) for i in sorted(new_idx_set) if i not in old_idx_set and not eng.store.snippets[i].lineage_id]
        edges = self.matcher.match(old_pairs, new_pairs, renames) if (old_pairs and new_pairs) else []
        for e in edges:
            eng.store.snippets[e.new_idx].lineage_id = eng.store.snippets[e.old_idx].lineage_id or new_lineage_id(eng.store.snippets[e.old_idx])
            stats.lineage_relations[e.relation] = stats.lineage_relations.get(e.relation, 0) + 1
        eng.add_lineage([e.as_tuple() for e in edges])
        for i in new_idx_set:
            if not eng.store.snippets[i].lineage_id:
                eng.store.snippets[i].lineage_id = new_lineage_id(eng.store.snippets[i])
        stats.lineage_edges = len(edges)
        stats.seconds["lineage"] = round(time.perf_counter() - t_lin, 4)

        # Commit membership = reused files' snippets + new snippets of changed files.
        files_state: dict[str, dict[str, Any]] = {}
        for p, blob in tree.items():
            if p in new_members:
                files_state[p] = {"blob": blob, "idxs": new_members[p]}
            elif p in prev_files:
                files_state[p] = prev_files[p]
        members = [i for f in files_state.values() for i in f["idxs"]]
        eng.store.register_commit(self.repository, info, members, make_latest=make_latest)
        st["commit"] = sha
        st["files"] = files_state
        st.setdefault("indexed_commits", []).append(sha)
        stats.snippets_live = len(members)

        t_bm = time.perf_counter()
        eng.finalize()
        stats.seconds["bm25_rebuild"] = round(time.perf_counter() - t_bm, 4)
        if eng.embedder:
            stats.embeddings_computed = eng.embedder.stats.computed - emb0[0]
            stats.embeddings_reused = eng.embedder.stats.reused - emb0[1]
        stats.seconds["total"] = round(time.perf_counter() - t0, 4)
        log.info("indexed version", extra={"data": {k: v for k, v in stats.to_dict().items() if k not in ("seconds", "lineage_relations")}})
        return stats


def index_path(engine: CodeFusionEngine, path: str | Path, repository: str | None = None, history: bool = False,
               max_commits: int | None = None) -> list[UpdateStats]:
    """Index a git repository (HEAD, or full history) or a plain directory."""
    idx = IncrementalIndexer(engine, path, repository)
    if idx.is_git:
        return idx.index_history(max_commits=max_commits) if history else [idx.index_commit("HEAD")]
    return [idx.index_worktree()]
