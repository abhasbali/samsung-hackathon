"""FastAPI service exposing CodeFusion retrieval (retrieval results only — no answer generation).

Run::

    uvicorn codefusion.api.main:app --port 8000

Environment:
    CODEFUSION_CONFIG     YAML config (default configs/full.yaml)
    CODEFUSION_INDEX_DIR  index directory to load/save (default: config.index_dir)
    CODEFUSION_ALLOW_FALLBACK  "1" to fall back to hashing embeddings if the model is unavailable
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from codefusion import __version__
from codefusion.config import load_config
from codefusion.logging_utils import get_logger
from codefusion.retrieval.pipeline import CodeFusionEngine
from codefusion.versions.incremental import IncrementalIndexer, index_path

log = get_logger(__name__)
ROOT = Path(__file__).resolve().parents[3]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=20000)
    top_k: int = Field(10, ge=1, le=100)
    version: str | None = Field(None, description="commit sha/prefix, 'latest' (default) or 'all' for history")


class IndexRequest(BaseModel):
    path: str
    repository: str | None = None
    history: bool = False
    max_commits: int | None = None


class VersionIndexRequest(BaseModel):
    path: str
    repository: str | None = None
    rev: str | None = Field(None, description="commit to index incrementally; omit with history=true for full history")
    history: bool = False
    max_commits: int | None = None


class _State:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.engine: CodeFusionEngine | None = None
        self.config_path = os.environ.get("CODEFUSION_CONFIG", str(ROOT / "configs" / "full.yaml"))
        self.index_dir: str | None = os.environ.get("CODEFUSION_INDEX_DIR")
        self.allow_fallback = os.environ.get("CODEFUSION_ALLOW_FALLBACK", "0") == "1"
        self.load_error: str | None = None

    def get(self) -> CodeFusionEngine:
        with self.lock:
            if self.engine is None:
                cfg = load_config(self.config_path)
                idx = Path(self.index_dir or cfg.index_dir)
                if not idx.is_absolute():
                    idx = ROOT / idx
                self.index_dir = str(idx)
                if (idx / "index.db").exists():
                    self.engine = CodeFusionEngine.load(idx, cfg, allow_model_fallback=self.allow_fallback)
                else:
                    self.engine = CodeFusionEngine(cfg, allow_model_fallback=self.allow_fallback)
            return self.engine


state = _State()
app = FastAPI(title="CodeFusion", version=__version__, description="Code-intelligence retrieval engine")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health() -> dict[str, Any]:
    eng = state.engine
    return {
        "status": "ok",
        "version": __version__,
        "engine_loaded": eng is not None,
        "snippets": len(eng.store.snippets) if eng else 0,
        "config": state.config_path,
    }


@app.post("/index/repository")
def index_repository(req: IndexRequest) -> dict[str, Any]:
    p = Path(req.path)
    if not p.exists():
        raise HTTPException(404, f"path not found: {req.path}")
    with state.lock:
        eng = state.get()
        t0 = time.perf_counter()
        stats = index_path(eng, p, req.repository, history=req.history, max_commits=req.max_commits)
        eng.save(state.index_dir)
    return {"seconds": round(time.perf_counter() - t0, 3), "updates": [s.to_dict() for s in stats], "stats": eng.stats()}


def _search(req: SearchRequest, explain: bool) -> dict[str, Any]:
    eng = state.get()
    if not eng.store.snippets:
        raise HTTPException(409, "index is empty: POST /index/repository first")
    try:
        resp = eng.search(req.query, top_k=req.top_k, version=req.version, explain=explain)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    return resp.to_dict(explain=explain)


@app.post("/search")
def search(req: SearchRequest) -> dict[str, Any]:
    return _search(req, explain=False)


@app.post("/search/explain")
def search_explain(req: SearchRequest) -> dict[str, Any]:
    return _search(req, explain=True)


@app.get("/stats")
def stats() -> dict[str, Any]:
    return state.get().stats()


@app.post("/versions/index")
def versions_index(req: VersionIndexRequest) -> dict[str, Any]:
    p = Path(req.path)
    if not p.exists():
        raise HTTPException(404, f"path not found: {req.path}")
    with state.lock:
        eng = state.get()
        idx = IncrementalIndexer(eng, p, req.repository)
        if not idx.is_git:
            raise HTTPException(400, "versions/index requires a git repository")
        if req.history:
            stats = idx.index_history(max_commits=req.max_commits)
        else:
            stats = [idx.index_commit(req.rev or "HEAD")]
        eng.save(state.index_dir)
    return {"updates": [s.to_dict() for s in stats]}


@app.get("/versions/{repo}")
def versions(repo: str) -> dict[str, Any]:
    eng = state.get()
    commits = [c for c in eng.store.commits if c.get("repository") == repo]
    if not commits:
        raise HTTPException(404, f"no indexed versions for repository {repo!r}")
    rel_counts: dict[str, int] = {}
    for _, _, rel, _ in eng.store.lineage:
        rel_counts[rel] = rel_counts.get(rel, 0) + 1
    return {
        "repository": repo,
        "latest": eng.store.latest.get(repo),
        "commits": [{**c, "snippets": int(len(eng.store.members.get(c["sha"], [])))} for c in commits],
        "lineage_relations": rel_counts,
    }


@app.get("/snippets/{snippet_id}/lineage")
def lineage(snippet_id: str) -> dict[str, Any]:
    eng = state.get()
    idx = eng.store.id_to_idx.get(snippet_id)
    if idx is None or eng.graph is None:
        raise HTTPException(404, "unknown snippet")
    chain = eng.graph.lineage(idx)
    rels = {(o, n): r for o, n, r, _ in eng.store.lineage}
    out = []
    for i, j in enumerate(chain):
        sn = eng.store.snippets[j]
        out.append({"snippet_id": sn.snippet_id, "symbol": sn.qualified_name, "file": sn.file_path, "commit": sn.commit,
                    "commits": sn.commits, "relation_from_previous": rels.get((chain[i - 1], j)) if i else None,
                    "content": sn.content})
    return {"lineage": out}


@app.get("/graph/callchain")
def callchain(snippet_id: str, depth: int = 3) -> dict[str, Any]:
    eng = state.get()
    idx = eng.store.id_to_idx.get(snippet_id)
    if idx is None or eng.graph is None:
        raise HTTPException(404, "unknown snippet")
    mask = eng.store.mask(None, eng.cfg.versions.default_scope) if eng.cfg.versions.enabled else None
    return eng.graph.call_chain(idx, depth=max(1, min(depth, 5)), mask=mask)


@app.get("/experiments")
def experiments() -> dict[str, Any]:
    """Stored, *measured* experiment results only (nothing is synthesised here).

    Reads whatever the evaluation scripts wrote under ``artifacts/``; a section is ``[]`` when that
    experiment has not been run, which the UI shows as "Not evaluated".
    """
    art = ROOT / "artifacts"

    def load(pattern: str) -> list[dict[str, Any]]:
        out = []
        for f in sorted(art.glob(pattern)):
            try:
                out.append({"file": f.relative_to(ROOT).as_posix(), **json.loads(f.read_text(encoding="utf-8"))})
            except (OSError, ValueError):
                continue
        return out

    official = [r for r in load("eval/*_summary.json")]
    for r in official:
        r.pop("config_dump", None)
    return {
        "official_runs": official,
        "ablations": load("ablations/ablations_*.json"),
        "embedding_benchmark": load("benchmarks/embeddings_*.json"),
        "versioning_benchmark": load("benchmarks/versioning_*.json"),
    }


@app.get("/demo/queries")
def demo_queries() -> list[dict[str, Any]]:
    f = ROOT / "examples" / "demo_queries.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else []


_dist = ROOT / "frontend" / "dist"
if _dist.exists():
    app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")

    @app.get("/")
    def index_html() -> FileResponse:
        return FileResponse(_dist / "index.html")
