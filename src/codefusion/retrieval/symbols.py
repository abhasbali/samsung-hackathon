"""Lightweight inverted symbol index (definitions, calls, imports, attributes, references).

Lookup is exact on a *normalised* key (``validate_user`` == ``validateUser``), so queries
mentioning a symbol in any casing convention hit its definition and usages. The index is
append-only, which makes incremental version updates O(changed snippets).
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from codefusion.config.schema import SymbolConfig
from codefusion.parsing.symbols import extract_symbols
from codefusion.retrieval.base import Retriever, top_k_from_scores
from codefusion.types import Hit, Snippet


class SymbolIndex(Retriever):
    name = "symbol"

    def __init__(self, cfg: SymbolConfig) -> None:
        self.cfg = cfg
        self.postings: dict[str, list[tuple[int, str]]] = defaultdict(list)
        self.display: dict[str, str] = {}
        self._n = 0

    @property
    def size(self) -> int:
        return self._n

    def add(self, snippets: list[Snippet], start_idx: int) -> None:
        for off, sn in enumerate(snippets):
            idx = start_idx + off
            for rec in extract_symbols(sn):
                self.postings[rec.normalized].append((idx, rec.role))
                self.display.setdefault(rec.normalized, rec.symbol)
        self._n = max(self._n, start_idx + len(snippets))

    def build(self, snippets: list[Snippet]) -> None:
        self.postings.clear()
        self.display.clear()
        self._n = 0
        self.add(snippets, 0)

    def has(self, normalized: str) -> bool:
        return normalized in self.postings

    def df(self, normalized: str) -> int:
        return len({i for i, _ in self.postings.get(normalized, [])})

    def lookup(self, normalized: str, roles: set[str] | None = None) -> list[tuple[int, str]]:
        return [(i, r) for i, r in self.postings.get(normalized, []) if roles is None or r in roles]

    def scores(self, symbol_candidates: list[tuple[str, float]], intent: str | None = None) -> tuple[np.ndarray, dict[int, list[str]]]:
        scores = np.zeros(self._n, dtype=np.float32)
        evidence: dict[int, list[str]] = defaultdict(list)
        n = max(self._n, 1)
        mult = self.cfg.intent_role_weights.get(intent or "", {})
        for sym, q_weight in symbol_candidates:
            if len(sym) < self.cfg.min_query_symbol_len:
                continue
            posting = self.postings.get(sym)
            if not posting:
                continue
            idf = math.log(1.0 + n / (1 + self.df(sym)))
            for idx, role in posting:
                w = self.cfg.role_weights.get(role, 0.5) * mult.get(role, 1.0)
                scores[idx] += q_weight * w * idf
                if len(evidence[idx]) < 6:
                    evidence[idx].append(f"{role}:{self.display.get(sym, sym)}")
        return scores, evidence

    def search(self, query, top_k: int, mask: np.ndarray | None = None) -> list[Hit]:  # type: ignore[override]
        if not query.symbol_candidates or self._n == 0:
            return []
        scores, evidence = self.scores(query.symbol_candidates, query.intent)
        hits = top_k_from_scores(scores, top_k, self.name, mask, min_score=0.0)
        for h in hits:
            h.evidence = evidence.get(h.idx, [])
        return hits
