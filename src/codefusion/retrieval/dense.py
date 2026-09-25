"""Dense retrieval: query embedding -> L2 normalisation -> FAISS inner product (= cosine) -> top-k.

The vector index is append-only (new snippet versions are added; old versions stay for
historical/evolutionary retrieval). Version-scoped searches pass a boolean ``mask`` and are
answered exactly with a NumPy matmul over the live subset.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np

from codefusion.config.schema import DenseConfig
from codefusion.logging_utils import get_logger
from codefusion.retrieval.base import Retriever, competition_ranks, top_k_from_scores
from codefusion.retrieval.embeddings import CachedEmbedder, l2_normalize
from codefusion.types import Hit

log = get_logger(__name__)

try:  # optional at import time; the numpy path is exact and always available
    import faiss  # type: ignore

    _HAS_FAISS = True
except Exception:  # noqa: BLE001
    faiss = None
    _HAS_FAISS = False


class VectorIndex:
    """FAISS wrapper (Flat IP by default) that also keeps the raw matrix for exact/masked search."""

    def __init__(self, dim: int, index_type: str = "flat", hnsw_m: int = 32, ivf_nlist: int = 256, ivf_nprobe: int = 16) -> None:
        self.dim = dim
        self.index_type = index_type
        self.hnsw_m = hnsw_m
        self.ivf_nlist = ivf_nlist
        self.ivf_nprobe = ivf_nprobe
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self._faiss = None
        self._faiss_count = 0

    def __len__(self) -> int:
        return int(self.vectors.shape[0])

    def _make_faiss(self, n: int):
        if not _HAS_FAISS:
            return None
        if self.index_type == "hnsw":
            idx = faiss.IndexHNSWFlat(self.dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            idx.hnsw.efSearch = max(64, self.hnsw_m * 2)
            return idx
        if self.index_type == "ivf" and n >= self.ivf_nlist * 39:
            quant = faiss.IndexFlatIP(self.dim)
            idx = faiss.IndexIVFFlat(quant, self.dim, self.ivf_nlist, faiss.METRIC_INNER_PRODUCT)
            idx.train(self.vectors)
            idx.nprobe = self.ivf_nprobe
            return idx
        return faiss.IndexFlatIP(self.dim)

    def add(self, vecs: np.ndarray) -> None:
        vecs = np.ascontiguousarray(vecs, dtype=np.float32)
        if vecs.size == 0:
            return
        self.vectors = np.vstack([self.vectors, vecs]) if len(self) else vecs
        if self._faiss is None:
            self._faiss = self._make_faiss(len(self))
            if self._faiss is not None:
                self._faiss.add(self.vectors)
                self._faiss_count = len(self)
        else:
            self._faiss.add(vecs)
            self._faiss_count += len(vecs)

    def search(self, queries: np.ndarray, k: int, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        queries = np.ascontiguousarray(np.atleast_2d(queries), dtype=np.float32)
        n = len(self)
        k = min(k, n)
        if k == 0:
            return np.zeros((len(queries), 0), np.float32), np.zeros((len(queries), 0), np.int64)
        if (mask is None or mask.all()) and self._faiss is not None:
            scores, idx = self._faiss.search(queries, k)
            return scores, idx
        sims = queries @ self.vectors.T
        if mask is not None:
            sims = np.where(mask[None, :], sims, -np.inf)
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        rows = np.arange(len(queries))[:, None]
        order = np.argsort(-sims[rows, part], axis=1, kind="stable")
        idx = part[rows, order]
        return sims[rows, idx].astype(np.float32), idx.astype(np.int64)


class DenseRetriever(Retriever):
    name = "dense"

    def __init__(self, embedder: CachedEmbedder, cfg: DenseConfig) -> None:
        self.embedder = embedder
        self.cfg = cfg
        self.index = VectorIndex(embedder.dim, cfg.index_type, cfg.hnsw_m, cfg.ivf_nlist, cfg.ivf_nprobe)
        self.encode_seconds = 0.0

    @property
    def size(self) -> int:
        return len(self.index)

    @property
    def vectors(self) -> np.ndarray:
        return self.index.vectors

    def add_documents(self, texts: Sequence[str]) -> np.ndarray:
        t0 = time.perf_counter()
        vecs = self.embedder.encode_documents(texts)
        if self.cfg.normalize:
            vecs = l2_normalize(vecs)
        self.index.add(vecs)
        self.encode_seconds += time.perf_counter() - t0
        return vecs

    def add_vectors(self, vecs: np.ndarray) -> None:
        self.index.add(l2_normalize(vecs) if self.cfg.normalize else vecs)

    def encode_query(self, text: str) -> np.ndarray:
        v = self.embedder.encode_queries([text])[0]
        return l2_normalize(v)[0] if self.cfg.normalize else v

    def search_vector(self, qvec: np.ndarray, top_k: int, mask: np.ndarray | None = None) -> list[Hit]:
        scores, idx = self.index.search(qvec[None, :], top_k, mask)
        hits: list[Hit] = []
        for s, i in zip(scores[0], idx[0]):
            if i < 0 or not np.isfinite(s):
                continue
            hits.append(Hit(idx=int(i), score=float(s), rank=len(hits) + 1, source=self.name))
        return competition_ranks(hits)

    def search(self, query, top_k: int, mask: np.ndarray | None = None) -> list[Hit]:  # type: ignore[override]
        if self.size == 0:
            return []
        if query.dense_vector is None:
            query.dense_vector = self.encode_query(query.dense_text or query.text)
        return self.search_vector(query.dense_vector, top_k, mask)

    def scores_for(self, qvec: np.ndarray, idxs: list[int]) -> np.ndarray:
        return self.index.vectors[idxs] @ qvec


__all__ = ["DenseRetriever", "VectorIndex", "competition_ranks", "top_k_from_scores"]
