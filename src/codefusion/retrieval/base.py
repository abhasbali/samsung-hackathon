"""Retriever abstraction shared by dense, lexical, symbol and graph retrievers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from codefusion.types import Hit

if TYPE_CHECKING:
    from codefusion.query.preprocess import ProcessedQuery


def top_k_from_scores(
    scores: np.ndarray,
    k: int,
    source: str,
    mask: np.ndarray | None = None,
    min_score: float | None = None,
) -> list[Hit]:
    """Deterministic top-k (ties broken by lower index) over a dense score array."""
    scores = np.asarray(scores, dtype=np.float64)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    n = scores.shape[0]
    if n == 0 or k <= 0:
        return []
    k = min(k, n)
    part = np.argpartition(-scores, k - 1)[:k] if k < n else np.arange(n)
    order = sorted(part.tolist(), key=lambda i: (-scores[i], i))
    hits: list[Hit] = []
    for i in order:
        s = float(scores[i])
        if not np.isfinite(s) or (min_score is not None and s <= min_score):
            continue
        hits.append(Hit(idx=int(i), score=s, rank=len(hits) + 1, source=source))
    return competition_ranks(hits)


def competition_ranks(hits: list[Hit]) -> list[Hit]:
    """Give exactly tied scores the same (best) rank: 1, 1, 3, ...

    A retriever that cannot tell documents apart (e.g. a symbol present in every snippet)
    must not inject an arbitrary index-order preference into rank fusion.
    """
    prev: float | None = None
    rank = 0
    for pos, h in enumerate(hits, start=1):
        if prev is None or h.score != prev:
            rank = pos
            prev = h.score
        h.rank = rank
    return hits


class Retriever(ABC):
    """A first-stage retriever over the snippet store (documents addressed by integer index)."""

    name: str = "base"

    @abstractmethod
    def search(self, query: ProcessedQuery, top_k: int, mask: np.ndarray | None = None) -> list[Hit]:
        """Return up to ``top_k`` hits, best first. ``mask[i]`` False excludes document ``i``."""

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of indexed documents."""
