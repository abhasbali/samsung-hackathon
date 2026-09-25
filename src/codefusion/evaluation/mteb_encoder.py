"""MTEB integration for the official AppsRetrieval (CoIR) evaluation.

Two models, both evaluated with ``mteb.evaluate`` and both producing the official
``TaskResult`` JSON:

* :class:`PrePostPipelineEncoder` — an ``AbsEncoder`` (the organisers' template): query/document
  pre-processing + the configured code embedding model (with its documented prompts) +
  L2 normalisation. MTEB performs cosine retrieval.
* :class:`CodeFusionSearchModel` — additionally implements MTEB's ``SearchProtocol``
  (``index``/``search``), so the *full* CodeFusion pipeline (dense + BM25 + symbols + fusion +
  optional reranking/structural scoring) produces the rankings MTEB scores.

Both share the persistent embedding cache, so switching between them — or re-running
ablations — never re-encodes the corpus.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from codefusion.config import CodeFusionConfig
from codefusion.logging_utils import get_logger
from codefusion.parsing.chunker import analyze_document
from codefusion.query.preprocess import QueryPreprocessor
from codefusion.retrieval.embeddings import CachedEmbedder, EmbeddingCache, create_provider, resolve_model

log = get_logger(__name__)

try:
    from mteb.models.abs_encoder import AbsEncoder
    from mteb.models.model_meta import ModelMeta, ScoringFunction
    from mteb.types import PromptType

    _HAS_MTEB = True
except Exception:  # noqa: BLE001 - mteb is required only for official evaluation
    AbsEncoder = object  # type: ignore[assignment,misc]
    ModelMeta = None  # type: ignore[assignment]
    PromptType = None  # type: ignore[assignment]
    ScoringFunction = None  # type: ignore[assignment]
    _HAS_MTEB = False


def corpus_text(row: dict[str, Any]) -> str:
    """Same text MTEB's own dataloader builds for a corpus row (title + text)."""
    title = row.get("title") or ""
    text = row.get("text") or ""
    return f"{title} {text}".strip() if title else text.strip()


def _meta(name: str, cfg: CodeFusionConfig, dim: int | None, model_type: str) -> Any:
    spec = resolve_model(cfg.dense.model)
    return ModelMeta.create_empty(
        overwrites={
            "name": name,
            "revision": "codefusion-1.0.0",
            "embed_dim": dim,
            "similarity_fn_name": ScoringFunction.COSINE,
            "open_weights": True,
            "framework": ["PyTorch", "Sentence Transformers"],
            "adapted_from": spec.hf_id,
            "model_type": [model_type],
            "use_instructions": bool(spec.query_prompt),
        }
    )


class PrePostPipelineEncoder(AbsEncoder):  # type: ignore[misc,valid-type]
    """Dense encoder with CodeFusion pre/post-processing (MTEB ``AbsEncoder``)."""

    def __init__(self, cfg: CodeFusionConfig, embedder: CachedEmbedder | None = None, name: str | None = None) -> None:
        if not _HAS_MTEB:
            raise ImportError("mteb is required for official evaluation: pip install mteb")
        self.cfg = cfg
        if embedder is None:
            provider = create_provider(cfg.dense)
            cache = EmbeddingCache(f"{cfg.cache_dir}/embeddings.sqlite") if cfg.dense.cache_embeddings else None
            embedder = CachedEmbedder(provider, cache, batch_size=cfg.dense.batch_size)
        self.embedder = embedder
        self.preprocessor = QueryPreprocessor(cfg.query)
        self.model_prompts = None
        self.mteb_model_meta = _meta(name or f"codefusion/{cfg.name}", cfg, embedder.dim, "dense")
        self.encode_seconds = {"query": 0.0, "document": 0.0}

    def preprocess_query(self, text: str) -> str:
        return self.preprocessor.process(text).dense_text

    def encode(self, inputs, *, task_metadata, hf_split: str, hf_subset: str, prompt_type=None, **kwargs: Any) -> np.ndarray:  # type: ignore[override]
        texts = [t for batch in inputs for t in batch["text"]]
        t0 = time.perf_counter()
        if prompt_type == PromptType.query:
            emb = self.embedder.encode_queries([self.preprocess_query(t) for t in texts])
            self.encode_seconds["query"] += time.perf_counter() - t0
        else:
            emb = self.embedder.encode_documents([t.strip() for t in texts])
            self.encode_seconds["document"] += time.perf_counter() - t0
        return emb


class CodeFusionSearchModel(PrePostPipelineEncoder):
    """Full hybrid pipeline exposed through MTEB's ``SearchProtocol``."""

    def __init__(self, cfg: CodeFusionConfig, embedder: CachedEmbedder | None = None, name: str | None = None) -> None:
        from codefusion.retrieval.pipeline import CodeFusionEngine

        # Hybrid runs may disable dense; only build an embedder if needed.
        if cfg.dense.enabled:
            super().__init__(cfg, embedder, name)
        else:
            if not _HAS_MTEB:
                raise ImportError("mteb is required for official evaluation: pip install mteb")
            self.cfg = cfg
            self.embedder = None  # type: ignore[assignment]
            self.preprocessor = QueryPreprocessor(cfg.query)
            self.model_prompts = None
            self.mteb_model_meta = _meta(name or f"codefusion/{cfg.name}", cfg, None, "hybrid")
            self.encode_seconds = {"query": 0.0, "document": 0.0}
        self.mteb_model_meta = self.mteb_model_meta.model_copy(update={"model_type": ["hybrid"]})
        self.engine_cls = CodeFusionEngine
        self.engine = None
        self.doc_ids: list[str] = []
        self.timing: dict[str, float] = {}
        self.query_latencies_ms: list[float] = []

    def index(self, corpus, *, task_metadata, hf_split: str, hf_subset: str, encode_kwargs, num_proc=None) -> None:
        t0 = time.perf_counter()
        eng = self.engine_cls(self.cfg, embedder=self.embedder)
        self.doc_ids = list(corpus["id"])
        snippets = []
        t_parse = time.perf_counter()
        for row in corpus:
            text = corpus_text(row)
            sn = analyze_document(text, str(row["id"]), language=_lang(row), prefer_tree_sitter=self.cfg.chunking.prefer_tree_sitter)
            snippets.append(sn)
        self.timing["parse_s"] = time.perf_counter() - t_parse
        eng.add_snippets(snippets)
        eng.finalize()
        self.engine = eng
        self.timing["index_s"] = time.perf_counter() - t0
        self.timing.update({f"index_{k}_s": v for k, v in eng.index_seconds.items()})
        log.info("mteb corpus indexed", extra={"data": {"docs": len(self.doc_ids), **{k: round(v, 2) for k, v in self.timing.items()}}})

    def search(self, queries, *, task_metadata, hf_split: str, hf_subset: str, top_k: int, encode_kwargs,
               top_ranked=None, num_proc=None) -> dict[str, dict[str, float]]:
        eng = self.engine
        if eng is None:
            raise RuntimeError("index() must be called before search()")
        qids = list(queries["id"])
        texts = [_query_text(r) for r in queries]
        t0 = time.perf_counter()
        pqs = [eng.process_query(t) for t in texts]
        if eng.dense is not None:
            vecs = self.embedder.encode_queries([p.dense_text or p.text for p in pqs])
            for p, v in zip(pqs, vecs):
                p.dense_vector = v
        self.timing["query_encode_s"] = time.perf_counter() - t0
        k = max(top_k, 10)
        results: dict[str, dict[str, float]] = {}
        self.query_latencies_ms = []
        for qid, p in zip(qids, pqs):
            t1 = time.perf_counter()
            resp = eng.search(p, top_k=k, explain=False)
            self.query_latencies_ms.append((time.perf_counter() - t1) * 1000.0)
            # Scores must be strictly decreasing with rank for MTEB's sort to reproduce our order.
            n = len(resp.results)
            results[str(qid)] = {self.doc_ids[r.idx]: float(n - i) + float(r.score) * 1e-6 for i, r in enumerate(resp.results)}
        self.timing["search_s"] = time.perf_counter() - t0
        return results


def _lang(row: dict[str, Any]) -> str:
    lang = (row.get("language") or "python") if isinstance(row, dict) else "python"
    return str(lang).lower() if str(lang).lower() in ("python", "javascript", "java", "go", "typescript") else "python"


def _query_text(row: dict[str, Any]) -> str:
    text = row.get("text", "")
    if isinstance(text, list):  # conversational queries: join turns
        text = "\n".join(str(t) for t in text)
    instr = row.get("instruction")
    return f"{instr} {text}".strip() if instr else str(text)
