"""CodeFusion retrieval engine.

Pipeline (each stage can be disabled from config for ablations)::

    query -> preprocess + intent -> [dense | BM25 | symbol] -> graph expansion (seeded)
          -> weighted RRF (query-adaptive) -> reranker (top-N) -> structural boosts
          -> dedup / lineage collapse / MMR -> top-k with full provenance

The engine owns a *temporal snippet store*: every distinct snippet version is stored once
(append-only) with the list of commits it appears in, so one set of indexes serves
"latest", "at commit X" and "across all versions" retrieval.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from codefusion.config import CodeFusionConfig, config_to_dict
from codefusion.graph.builder import CodeGraph
from codefusion.graph.scoring import GraphRetriever, StructuralScorer
from codefusion.logging_utils import get_logger
from codefusion.query.expansion import VocabularyExpander, concept_expansion
from codefusion.query.preprocess import ProcessedQuery, QueryPreprocessor
from codefusion.representations import build_views, dense_text
from codefusion.retrieval.bm25 import BM25Retriever
from codefusion.retrieval.dense import DenseRetriever
from codefusion.retrieval.diversity import deduplicate, mmr
from codefusion.retrieval.embeddings import CachedEmbedder, EmbeddingCache, create_provider
from codefusion.retrieval.fusion import fuse, reciprocal_rank_fusion
from codefusion.retrieval.reranker import Reranker, combine_rerank, create_reranker
from codefusion.retrieval.symbols import SymbolIndex
from codefusion.types import Candidate, Hit, Snippet

log = get_logger(__name__)


# --------------------------------------------------------------------------- store
class SnippetStore:
    """Append-only store of snippet versions plus commit membership."""

    def __init__(self) -> None:
        self.snippets: list[Snippet] = []
        self.views: list[dict[str, str]] = []
        self.id_to_idx: dict[str, int] = {}
        self.commits: list[dict[str, Any]] = []
        self.members: dict[str, np.ndarray] = {}
        self.latest: dict[str, str] = {}  # repository -> latest indexed commit
        self.repo_state: dict[str, dict[str, Any]] = {}
        self.lineage: list[tuple[int, int, str, float]] = []

    def __len__(self) -> int:
        return len(self.snippets)

    def register_commit(self, repository: str, info: dict[str, Any], members: Sequence[int], make_latest: bool = True) -> None:
        sha = info["sha"]
        if sha not in self.members:
            self.commits.append({**info, "repository": repository})
        self.members[sha] = np.array(sorted(set(members)), dtype=np.int64)
        for i in self.members[sha]:
            sn = self.snippets[int(i)]
            if sha not in sn.commits:
                sn.commits.append(sha)
        if make_latest:
            self.latest[repository] = sha

    def resolve_version(self, version: str) -> str | None:
        if version in self.members:
            return version
        matches = [s for s in self.members if s.startswith(version)]
        return matches[0] if len(matches) == 1 else None

    def mask(self, version: str | None, default_scope: str = "latest") -> np.ndarray | None:
        """Boolean mask of live snippets. ``None`` means every stored version is eligible."""
        n = len(self.snippets)
        if not self.members or n == 0:
            return None
        scope = version or default_scope
        if scope in ("all", "*", "history"):
            return None
        m = np.zeros(n, dtype=bool)
        if scope == "latest":
            for sha in self.latest.values():
                m[self.members[sha]] = True
            # Snippets not tied to any commit (e.g. ad-hoc documents) stay searchable.
            tied = np.zeros(n, dtype=bool)
            for idxs in self.members.values():
                tied[idxs] = True
            m |= ~tied
            return m
        sha = self.resolve_version(scope)
        if sha is None:
            raise KeyError(f"unknown version {scope!r}")
        m[self.members[sha]] = True
        return m

    def commit_index(self, sha: str) -> int:
        for i, c in enumerate(self.commits):
            if c["sha"] == sha:
                return i
        return -1


# --------------------------------------------------------------------------- results
@dataclass
class SearchResult:
    rank: int
    idx: int
    score: float
    snippet: dict[str, Any]
    provenance: dict[str, Any]
    structural_evidence: list[str] = field(default_factory=list)
    graph_evidence: list[str] = field(default_factory=list)
    lineage: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "score": round(self.score, 6),
            "snippet": self.snippet,
            "provenance": self.provenance,
            "structural_evidence": self.structural_evidence,
            "graph_evidence": self.graph_evidence,
            "lineage": self.lineage,
        }


@dataclass
class SearchResponse:
    query: str
    intent: str
    intent_confidence: float
    results: list[SearchResult]
    timings_ms: dict[str, float]
    candidate_counts: dict[str, int]
    weights: dict[str, float]
    version_scope: str
    query_analysis: dict[str, Any]
    retrievers: dict[str, str]
    dropped: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self, explain: bool = True) -> dict[str, Any]:
        d = {
            "query": self.query,
            "intent": self.intent,
            "intent_confidence": self.intent_confidence,
            "version_scope": self.version_scope,
            "timings_ms": self.timings_ms,
            "candidate_counts": self.candidate_counts,
            "retrievers": self.retrievers,
            "results": [r.to_dict() for r in self.results],
        }
        if explain:
            d["weights"] = {k: round(v, 4) for k, v in self.weights.items()}
            d["query_analysis"] = self.query_analysis
            d["dropped"] = self.dropped
        return d


class _Timer:
    def __init__(self) -> None:
        self.t: dict[str, float] = {}

    def __call__(self, name: str):
        timer = self

        class _Ctx:
            def __enter__(self_inner):
                self_inner.t0 = time.perf_counter()

            def __exit__(self_inner, *exc):
                timer.t[name] = timer.t.get(name, 0.0) + (time.perf_counter() - self_inner.t0) * 1000.0

        return _Ctx()


# --------------------------------------------------------------------------- engine
class CodeFusionEngine:
    def __init__(self, cfg: CodeFusionConfig, embedder: CachedEmbedder | None = None, allow_model_fallback: bool = False) -> None:
        self.cfg = cfg
        np.random.seed(cfg.seed)
        self.store = SnippetStore()
        self.preprocessor = QueryPreprocessor(cfg.query)
        self.embedder = embedder
        self.dense: DenseRetriever | None = None
        self.retriever_status: dict[str, str] = {}
        if cfg.dense.enabled:
            if self.embedder is None:
                provider = create_provider(cfg.dense, allow_fallback=allow_model_fallback)
                cache = EmbeddingCache(Path(cfg.cache_dir) / "embeddings.sqlite") if cfg.dense.cache_embeddings else None
                self.embedder = CachedEmbedder(provider, cache, batch_size=cfg.dense.batch_size)
            self.dense = DenseRetriever(self.embedder, cfg.dense)
            self.retriever_status["dense"] = f"enabled ({self.embedder.provider.name})"
        else:
            self.retriever_status["dense"] = "disabled"
        self.bm25 = BM25Retriever(cfg.bm25) if cfg.bm25.enabled else None
        self.retriever_status["bm25"] = "enabled" if self.bm25 else "disabled"
        self.symbols = SymbolIndex(cfg.symbols) if cfg.symbols.enabled else None
        self.retriever_status["symbol"] = "enabled" if self.symbols else "disabled"
        needs_graph = cfg.graph.enabled or cfg.structural.enabled or cfg.versions.enabled
        self.graph = CodeGraph(self.store.snippets, cfg.graph.backend, cfg.graph.max_symbol_fanout) if needs_graph else None
        self.graph_retriever = GraphRetriever(self.graph, cfg.graph) if (cfg.graph.enabled and self.graph) else None
        self.retriever_status["graph"] = f"enabled ({self.graph.store.backend})" if self.graph_retriever else "disabled"
        self.structural = StructuralScorer(self.graph, cfg.structural, cfg.fusion.rrf_k) if cfg.structural.enabled else None
        self.reranker: Reranker | None = create_reranker(cfg.reranker)
        self.retriever_status["reranker"] = ("enabled (" + self.reranker.name + ")") if self.reranker else (
            "unavailable (fallback to fused ranking)" if cfg.reranker.enabled else "disabled")
        self.vocab = VocabularyExpander() if cfg.query.expansion else None
        self._bm25_dirty = False
        self.index_seconds: dict[str, float] = {"parse": 0.0, "dense": 0.0, "bm25": 0.0, "symbol": 0.0, "graph": 0.0}

    # ------------------------------------------------------------------ indexing
    def add_snippets(self, snippets: Sequence[Snippet]) -> list[int]:
        """Append snippet versions not yet stored. Returns the store index of every input snippet."""
        idxs: list[int] = []
        new: list[Snippet] = []
        for sn in snippets:
            i = self.store.id_to_idx.get(sn.snippet_id)
            if i is None:
                i = len(self.store.snippets) + len(new)
                self.store.id_to_idx[sn.snippet_id] = i
                if sn.commit and sn.commit not in sn.commits:
                    sn.commits = [sn.commit]
                new.append(sn)
            idxs.append(i)
        if not new:
            return idxs
        start = len(self.store.snippets)
        views = [build_views(s) for s in new]
        self.store.snippets.extend(new)
        self.store.views.extend(views)
        if self.dense is not None:
            t0 = time.perf_counter()
            self.dense.add_documents([dense_text(s, self.cfg.dense.document_view) for s in new])
            self.index_seconds["dense"] += time.perf_counter() - t0
        if self.symbols is not None:
            t0 = time.perf_counter()
            self.symbols.add(new, start)
            self.index_seconds["symbol"] += time.perf_counter() - t0
        if self.graph is not None:
            t0 = time.perf_counter()
            self.graph.add_snippets(start)
            self.index_seconds["graph"] += time.perf_counter() - t0
        if self.vocab is not None:
            self.vocab.add_identifiers([s.defines + s.calls + s.references[:50] for s in new])
        self._bm25_dirty = True
        return idxs

    def add_lineage(self, edges: Sequence[tuple[int, int, str, float]]) -> None:
        for old, new, rel, sim in edges:
            self.store.lineage.append((old, new, rel, sim))
            if self.graph is not None:
                self.graph.add_lineage_edge(old, new)

    def finalize(self) -> None:
        if self.bm25 is not None and self._bm25_dirty:
            t0 = time.perf_counter()
            self.bm25.build(self.store.views, [s.content_hash + s.symbol_key for s in self.store.snippets])
            self.index_seconds["bm25"] += time.perf_counter() - t0
        self._bm25_dirty = False

    # ------------------------------------------------------------------ search
    def process_query(self, query: str, version: str | None = None) -> ProcessedQuery:
        pq = self.preprocessor.process(query, version)
        if self.cfg.query.expansion:
            terms = concept_expansion(pq.text, self.cfg.query.expansion_max_terms)
            if self.vocab is not None:
                terms += [t for t in self.vocab.expand(pq.text, self.cfg.query.expansion_max_terms) if t not in terms]
            # Expansion terms feed BM25 only: injecting them as symbols proved noisy
            # ("request" -> get/post matched every dict.get call).
            pq.expansion_terms = terms[: self.cfg.query.expansion_max_terms * 2]
        if self.bm25 is not None:
            pq.bm25_tokens = self.bm25.query_tokens(pq.text, pq.expansion_terms)
        return pq

    def search(self, query: str | ProcessedQuery, top_k: int = 10, version: str | None = None, explain: bool = True) -> SearchResponse:
        cfg = self.cfg
        if self._bm25_dirty:
            self.finalize()
        tm = _Timer()
        t_start = time.perf_counter()
        with tm("preprocess"):
            pq = query if isinstance(query, ProcessedQuery) else self.process_query(query, version)
        evolution = pq.intent == "EVOLUTION"
        scope = version or ("all" if (evolution and cfg.versions.enabled) else cfg.versions.default_scope)
        mask = self.store.mask(scope, cfg.versions.default_scope) if cfg.versions.enabled else None
        # Several versions of one lineage are kept only when the query asks about history;
        # a plain search over "all" versions still collapses each lineage to its best version.
        keep_history = evolution

        ranked: dict[str, list[Hit]] = {}
        counts: dict[str, int] = {}
        if self.dense is not None and self.dense.size:
            with tm("dense"):
                ranked["dense"] = self.dense.search(pq, cfg.dense.top_k, mask)
        if self.bm25 is not None and self.bm25.size:
            with tm("bm25"):
                ranked["bm25"] = self.bm25.search(pq, cfg.bm25.top_k, mask)
        if self.symbols is not None and self.symbols.size:
            with tm("symbol"):
                ranked["symbol"] = self.symbols.search(pq, cfg.symbols.top_k, mask)
        if self.graph_retriever is not None:
            with tm("graph"):
                seeds = reciprocal_rank_fusion({k: v for k, v in ranked.items() if v}, None, cfg.fusion.rrf_k)
                seed_hits = [Hit(c.idx, c.fused_score, r + 1, "seed") for r, c in enumerate(seeds[: cfg.graph.seeds])]
                ranked["graph"] = self.graph_retriever.search(pq, seed_hits, mask)
        for k, v in ranked.items():
            counts[k] = len(v)

        with tm("fusion"):
            fused, weights = fuse(cfg.fusion, ranked, pq.intent)
            pool = fused[: cfg.fusion.candidate_pool]
        counts["fused"] = len(fused)

        if self.reranker is not None and pool:
            with tm("rerank"):
                n = min(cfg.reranker.candidates, len(pool))
                head = pool[:n]
                docs = [self.store.snippets[c.idx].content[:6000] for c in head]
                scores = self.reranker.score(pq.dense_text or pq.text, docs)
                combined = combine_rerank([c.idx for c in head], scores, cfg.reranker.mode, cfg.reranker.alpha, cfg.fusion.rrf_k)
                for c, s, comb in zip(head, scores, combined):
                    c.reranker_score = float(s)
                    c.final_score = float(comb)
                # Keep reranked head strictly above the tail.
                tail_top = max((c.final_score for c in pool[n:]), default=0.0)
                offset = max(0.0, tail_top - min(c.final_score for c in head) + 1e-9)
                for c in head:
                    c.final_score += offset
                pool = sorted(head, key=lambda c: (-c.final_score, c.idx)) + pool[n:]
            counts["reranked"] = n

        if self.structural is not None and pool:
            with tm("structural"):
                self.structural.apply(pq, pool[: max(50, top_k * 3)], self.store.snippets, mask)
                pool = sorted(pool, key=lambda c: (-c.final_score, c.fused_rank or 0, c.idx))

        dropped: list[Candidate] = []
        with tm("diversity"):
            if cfg.diversity.enabled:
                if cfg.diversity.mmr and len(pool) > top_k:
                    vecs = self.dense.vectors if self.dense is not None else None
                    pool = mmr(pool[: top_k * 4], vecs, self.store.snippets, cfg.diversity.mmr_lambda, top_k * 4) + pool[top_k * 4 :]
                final, dropped = deduplicate(pool, self.store.snippets, cfg.diversity, keep_history=keep_history, limit=top_k)
            else:
                final = pool[:top_k]
            if keep_history and evolution:
                final = self._order_history(final)
        counts["final"] = len(final)
        timings = {k: round(v, 3) for k, v in tm.t.items()}
        timings["total"] = round((time.perf_counter() - t_start) * 1000.0, 3)

        results = [self._result(rank, c, keep_history) for rank, c in enumerate(final, start=1)]
        return SearchResponse(
            query=pq.original,
            intent=pq.intent,
            intent_confidence=pq.intent_confidence,
            results=results,
            timings_ms=timings,
            candidate_counts=counts,
            weights=weights,
            version_scope=scope,
            query_analysis=pq.summary(),
            retrievers=self.retriever_status,
            dropped=[{"file": self.store.snippets[c.idx].file_path, "name": self.store.snippets[c.idx].qualified_name,
                      "reason": c.dropped_reason} for c in dropped[:10]] if explain else [],
        )

    def _order_history(self, final: list[Candidate]) -> list[Candidate]:
        """For evolution queries, keep lineage groups together, oldest version first within a group."""
        groups: dict[str, list[Candidate]] = {}
        order: list[str] = []
        for c in final:
            lid = self.store.snippets[c.idx].lineage_id or f"_{c.idx}"
            if lid not in groups:
                groups[lid] = []
                order.append(lid)
            groups[lid].append(c)
        out: list[Candidate] = []
        for lid in order:
            out.extend(sorted(groups[lid], key=lambda c: self._first_commit_ord(c.idx)))
        return out

    def _first_commit_ord(self, idx: int) -> int:
        sn = self.store.snippets[idx]
        ords = [self.store.commit_index(s) for s in sn.commits]
        ords = [o for o in ords if o >= 0]
        return min(ords) if ords else 10**9

    def _result(self, rank: int, c: Candidate, with_lineage: bool) -> SearchResult:
        sn = self.store.snippets[c.idx]
        snippet = {
            "snippet_id": sn.snippet_id,
            "file": sn.file_path,
            "lines": [sn.start_line, sn.end_line],
            "symbol": sn.qualified_name,
            "name": sn.name,
            "type": sn.type,
            "language": sn.language,
            "repository": sn.repository,
            "commit": sn.commit,
            "commits": sn.commits[-5:],
            "lineage_id": sn.lineage_id,
            "signature": sn.signature,
            "content": sn.content,
        }
        prov = c.provenance()
        prov["final_rank"] = rank
        prov["final_score"] = round(c.final_score, 6)
        lineage = None
        if with_lineage and self.graph is not None:
            chain = self.graph.lineage(c.idx)
            if len(chain) > 1:
                lineage = [{"idx": i, "symbol": self.store.snippets[i].qualified_name, "file": self.store.snippets[i].file_path,
                            "commit": self.store.snippets[i].commit, "is_this": i == c.idx} for i in chain]
        return SearchResult(rank, c.idx, c.final_score, snippet, prov, list(dict.fromkeys(c.structural_evidence)),
                            list(dict.fromkeys(c.graph_evidence)), lineage)

    # ------------------------------------------------------------------ batch (evaluation)
    def search_many(self, queries: Sequence[str], top_k: int = 100, version: str | None = None) -> list[list[tuple[int, float]]]:
        """Batch search used by evaluation: query embeddings are computed in one batched pass."""
        if self._bm25_dirty:
            self.finalize()
        pqs = [self.process_query(q, version) for q in queries]
        if self.dense is not None and self.dense.size:
            vecs = self.embedder.encode_queries([p.dense_text or p.text for p in pqs])  # type: ignore[union-attr]
            for p, v in zip(pqs, vecs):
                p.dense_vector = v
        out: list[list[tuple[int, float]]] = []
        for p in pqs:
            resp = self.search(p, top_k=top_k, version=version, explain=False)
            out.append([(r.idx, r.score) for r in resp.results])
        return out

    # ------------------------------------------------------------------ stats / persistence
    def stats(self) -> dict[str, Any]:
        return {
            "snippet_versions": len(self.store.snippets),
            "commits": len(self.store.commits),
            "repositories": sorted(self.store.latest),
            "latest": self.store.latest,
            "lineage_edges": len(self.store.lineage),
            "dense": {"vectors": self.dense.size, "dim": self.embedder.dim if self.embedder else None,
                      "model": self.embedder.provider.name if self.embedder else None,
                      "index_type": self.cfg.dense.index_type} if self.dense else None,
            "bm25_docs": self.bm25.size if self.bm25 else None,
            "symbols": len(self.symbols.postings) if self.symbols else None,
            "graph": self.graph.stats() if self.graph else None,
            "index_seconds": {k: round(v, 3) for k, v in self.index_seconds.items()},
            "embedding_stats": vars(self.embedder.stats) if self.embedder else None,
            "retrievers": self.retriever_status,
            "config": self.cfg.name,
        }

    def save(self, index_dir: str | Path | None = None) -> Path:
        from codefusion.storage.artifacts import export_snippets_parquet
        from codefusion.storage.sqlite import IndexDB

        d = Path(index_dir or self.cfg.index_dir)
        d.mkdir(parents=True, exist_ok=True)
        db_path = d / "index.db"
        if db_path.exists():
            db_path.unlink()
        db = IndexDB(db_path)
        db.write_snippets(self.store.snippets)
        db.write_commits(self.store.commits, {k: v.tolist() for k, v in self.store.members.items()})
        db.write_lineage(self.store.lineage)
        db.set_meta("latest", self.store.latest)
        db.set_meta("repo_state", self.store.repo_state)
        db.set_meta("config", config_to_dict(self.cfg))
        db.set_meta("embedding_fingerprint", self.embedder.provider.fingerprint if self.embedder else None)
        db.close()
        if self.dense is not None:
            np.save(d / "vectors.npy", self.dense.vectors)
        export_snippets_parquet(self.store.snippets, d / "snippets.parquet")
        log.info("index saved", extra={"data": {"dir": str(d), "snippets": len(self.store.snippets)}})
        return d

    @classmethod
    def load(cls, index_dir: str | Path, cfg: CodeFusionConfig | None = None, embedder: CachedEmbedder | None = None,
             allow_model_fallback: bool = False) -> CodeFusionEngine:
        from codefusion.storage.sqlite import IndexDB

        d = Path(index_dir)
        db = IndexDB(d / "index.db")
        if cfg is None:
            cfg = CodeFusionConfig.model_validate(db.get_meta("config"))
        eng = cls(cfg, embedder=embedder, allow_model_fallback=allow_model_fallback)
        snippets = db.read_snippets()
        commits, members = db.read_commits()
        lineage = db.read_lineage()
        stored_fp = db.get_meta("embedding_fingerprint")
        repo_state = db.get_meta("repo_state", {}) or {}
        latest = db.get_meta("latest", {}) or {}
        db.close()
        vec_path = d / "vectors.npy"
        reuse_vectors = (eng.dense is not None and vec_path.exists() and stored_fp == eng.embedder.provider.fingerprint)  # type: ignore[union-attr]
        if reuse_vectors:
            dense, eng.dense = eng.dense, None  # add without re-encoding, then attach stored vectors
            eng.add_snippets(snippets)
            eng.dense = dense
            eng.dense.add_vectors(np.load(vec_path))
        else:
            eng.add_snippets(snippets)
        for c in commits:
            eng.store.commits.append(c)
            eng.store.members[c["sha"]] = np.array(members.get(c["sha"], []), dtype=np.int64)
        eng.store.latest = latest
        eng.store.repo_state = repo_state
        eng.add_lineage(lineage)
        eng.finalize()
        log.info("index loaded", extra={"data": {"dir": str(d), "snippets": len(snippets), "reused_vectors": reuse_vectors}})
        return eng
