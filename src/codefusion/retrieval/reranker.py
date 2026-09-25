"""Second-stage reranking of the fused top-N candidates.

* :class:`CrossEncoderReranker` — any SentenceTransformers ``CrossEncoder`` (default
  ``Alibaba-NLP/gte-reranker-modernbert-base``, a 149M ModernBERT reranker trained with code
  retrieval data that loads without remote code).
* :class:`EmbeddingReranker` — re-scores candidates with a (typically larger) bi-encoder;
  cheap on CPU because only N candidates + 1 query are encoded per query.

If a reranker cannot be loaded, :func:`create_reranker` returns ``None`` and the pipeline
keeps the fused ranking (graceful fallback). Latency is recorded separately.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np

from codefusion.config.schema import DenseConfig, RerankerConfig
from codefusion.logging_utils import get_logger

log = get_logger(__name__)


class Reranker(ABC):
    name: str = "reranker"

    @abstractmethod
    def score(self, query: str, documents: list[str]) -> np.ndarray:
        """Return one relevance score per document (higher is better)."""


class CrossEncoderReranker(Reranker):
    def __init__(self, cfg: RerankerConfig) -> None:
        from sentence_transformers import CrossEncoder

        t0 = time.perf_counter()
        self.model = CrossEncoder(
            cfg.model,
            device=cfg.device,
            max_length=cfg.max_length,
            trust_remote_code=cfg.trust_remote_code,
        )
        self.batch_size = cfg.batch_size
        self.name = cfg.model
        log.info("reranker loaded", extra={"data": {"model": cfg.model, "seconds": round(time.perf_counter() - t0, 1)}})

    def score(self, query: str, documents: list[str]) -> np.ndarray:
        if not documents:
            return np.zeros(0, dtype=np.float32)
        scores = self.model.predict([(query, d) for d in documents], batch_size=self.batch_size, show_progress_bar=False)
        return np.asarray(scores, dtype=np.float32).reshape(-1)


class EmbeddingReranker(Reranker):
    def __init__(self, cfg: RerankerConfig) -> None:
        from codefusion.retrieval.embeddings import create_provider

        dcfg = DenseConfig(model=cfg.model, device=cfg.device, batch_size=cfg.batch_size, max_seq_length=cfg.max_length)
        self.provider = create_provider(dcfg)
        self.name = f"embed:{cfg.model}"

    def score(self, query: str, documents: list[str]) -> np.ndarray:
        if not documents:
            return np.zeros(0, dtype=np.float32)
        q = self.provider.encode_queries([query])[0]
        d = self.provider.encode_documents(documents)
        return (d @ q).astype(np.float32)


def create_reranker(cfg: RerankerConfig) -> Reranker | None:
    if not cfg.enabled:
        return None
    try:
        return CrossEncoderReranker(cfg) if cfg.type == "cross-encoder" else EmbeddingReranker(cfg)
    except Exception as exc:  # noqa: BLE001 - network/auth/dependency issues
        log.warning("reranker unavailable; using fused ranking", extra={"data": {"model": cfg.model, "error": repr(exc)[:300]}})
        return None


def combine_rerank(fused_order: list[int], rerank_scores: np.ndarray, mode: str, alpha: float, k: int = 60) -> np.ndarray:
    """Combine fused ranking with reranker scores in *rank space* (scale-free).

    ``replace``: reranker order only. ``interpolate``:
    ``alpha / (k + fused_rank) + (1 - alpha) / (k + rerank_rank)``.
    """
    n = len(fused_order)
    rr_rank = np.empty(n, dtype=np.int64)
    rr_rank[np.argsort(-rerank_scores, kind="stable")] = np.arange(1, n + 1)
    if mode == "replace":
        return 1.0 / (k + rr_rank)
    fused_rank = np.arange(1, n + 1)
    return alpha / (k + fused_rank) + (1 - alpha) / (k + rr_rank)
