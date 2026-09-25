"""Code-aware multi-field BM25 built on ``bm25s``.

Each view (raw code, identifiers, structural, context, docstring/comments) gets its own
BM25 index; a query's score is the weighted sum of per-field BM25 scores (BM25F-lite).
Tokenisation uses :class:`~codefusion.parsing.identifiers.CodeTokenizer`, so
``getUserToken`` matches "user token" and ``get_user_token`` alike.

Tokenised fields are cached per ``content_hash`` so an incremental re-index only
tokenises changed snippets; the (cheap) BM25 matrices are rebuilt from the cache.
"""

from __future__ import annotations

import time

import numpy as np

from codefusion.config.schema import BM25Config
from codefusion.logging_utils import get_logger
from codefusion.parsing.identifiers import CodeTokenizer
from codefusion.retrieval.base import Retriever, top_k_from_scores
from codefusion.types import Hit

log = get_logger(__name__)


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(self, cfg: BM25Config) -> None:
        import bm25s  # noqa: F401 - fail early with a clear error

        self.cfg = cfg
        self.tokenizer = CodeTokenizer(
            split_identifiers=cfg.identifier_splitting,
            stem_tokens=cfg.stemming,
            remove_stopwords=cfg.remove_stopwords,
        )
        self.fields = {f: w for f, w in cfg.fields.items() if w > 0}
        self._indexes: dict[str, object] = {}
        self._vocab: dict[str, dict[str, int]] = {}
        self._token_cache: dict[tuple[str, str], list[str]] = {}
        self._n = 0
        self.build_seconds = 0.0

    @property
    def size(self) -> int:
        return self._n

    def tokenize_views(self, views: list[dict[str, str]], keys: list[str]) -> dict[str, list[list[str]]]:
        """Tokenise each field, reusing cached tokens for unchanged ``keys`` (content hashes)."""
        out: dict[str, list[list[str]]] = {f: [] for f in self.fields}
        hits = 0
        for v, key in zip(views, keys):
            for f in self.fields:
                ck = (f, key)
                toks = self._token_cache.get(ck)
                if toks is None:
                    toks = self.tokenizer.tokenize(v.get(f, ""))
                    self._token_cache[ck] = toks
                else:
                    hits += 1
                out[f].append(toks)
        log.debug("bm25 tokenisation", extra={"data": {"docs": len(views), "cache_hits": hits}})
        return out

    def build(self, views: list[dict[str, str]], keys: list[str] | None = None) -> None:
        import bm25s

        t0 = time.perf_counter()
        keys = keys or [str(i) for i in range(len(views))]
        tokens = self.tokenize_views(views, keys)
        self._indexes.clear()
        self._vocab.clear()
        for f, toks in tokens.items():
            # Guarantee at least one token per doc so bm25s never sees an empty corpus row.
            toks = [t if t else ["<empty>"] for t in toks]
            idx = bm25s.BM25(k1=self.cfg.k1, b=self.cfg.b, method=self.cfg.method)
            idx.index(toks, show_progress=False)
            self._indexes[f] = idx
            self._vocab[f] = idx.vocab_dict
        self._n = len(views)
        self.build_seconds = time.perf_counter() - t0
        log.info("bm25 index built", extra={"data": {"docs": self._n, "fields": list(self.fields), "seconds": round(self.build_seconds, 3)}})

    def query_tokens(self, text: str, extra_terms: list[str] | None = None) -> list[str]:
        toks = self.tokenizer.tokenize(text, dedupe=True)
        if extra_terms:
            seen = set(toks)
            for t in self.tokenizer.tokenize(" ".join(extra_terms), dedupe=True):
                if t not in seen:
                    toks.append(t)
                    seen.add(t)
        return toks

    def scores(self, tokens: list[str]) -> np.ndarray:
        total = np.zeros(self._n, dtype=np.float32)
        for f, w in self.fields.items():
            idx = self._indexes.get(f)
            if idx is None:
                continue
            vocab = self._vocab[f]
            q = [t for t in tokens if t in vocab]
            if not q:
                continue
            total += w * np.asarray(idx.get_scores(q), dtype=np.float32)
        return total

    def search(self, query, top_k: int, mask: np.ndarray | None = None) -> list[Hit]:  # type: ignore[override]
        tokens = query.bm25_tokens if query.bm25_tokens else self.query_tokens(query.text, query.expansion_terms)
        if not tokens or self._n == 0:
            return []
        return top_k_from_scores(self.scores(tokens), top_k, self.name, mask, min_score=0.0)
