"""EXPERIMENTAL: late-interaction (ColBERT-style MaxSim) retrieval / reranking.

``score(q, d) = sum_{t in q} max_{u in d} cos(E(t), E(u))`` over per-token embeddings.

Token encoders:

* :class:`TransformerTokenEncoder` — per-token hidden states of any HF encoder (e.g.
  ``jinaai/jina-colbert-v2`` or the configured code model), L2-normalised, stored as float16.
* :class:`HashingTokenEncoder` — deterministic hashed token vectors; used for tests only.

:class:`LateInteractionRetriever` can search a small corpus exhaustively or rerank a candidate
list, so it plugs into the same evaluation harness as single-vector dense retrieval
(ablation ``M``). It is never required by the core system.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from codefusion.logging_utils import get_logger
from codefusion.parsing.identifiers import CodeTokenizer

log = get_logger(__name__)


class HashingTokenEncoder:
    def __init__(self, dim: int = 128) -> None:
        self.dim = dim
        self.tok = CodeTokenizer(stem_tokens=True)

    def _vec(self, token: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "little")
        v = np.random.default_rng(seed).normal(size=self.dim).astype(np.float32)
        return v / np.linalg.norm(v)

    def encode(self, texts: Sequence[str], is_query: bool = False) -> list[np.ndarray]:
        out = []
        for t in texts:
            toks = self.tok.tokenize(t) or ["<empty>"]
            out.append(np.stack([self._vec(x) for x in toks]).astype(np.float16))
        return out


class TransformerTokenEncoder:
    def __init__(self, model_name: str, device: str = "cpu", max_length: int = 512, query_prefix: str = "", doc_prefix: str = "") -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device).eval()
        self.device = device
        self.max_length = max_length
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix

    def encode(self, texts: Sequence[str], is_query: bool = False, batch_size: int = 8) -> list[np.ndarray]:
        prefix = self.query_prefix if is_query else self.doc_prefix
        out: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = [prefix + t for t in texts[i : i + batch_size]]
            enc = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                hid = self.model(**enc).last_hidden_state
            hid = self.torch.nn.functional.normalize(hid, dim=-1)
            for h, m in zip(hid, enc["attention_mask"]):
                out.append(h[m.bool()].float().cpu().numpy().astype(np.float16))
        return out


def maxsim(q: np.ndarray, d: np.ndarray) -> float:
    sims = q.astype(np.float32) @ d.astype(np.float32).T
    return float(sims.max(axis=1).sum())


class LateInteractionRetriever:
    name = "late_interaction"

    def __init__(self, encoder) -> None:
        self.encoder = encoder
        self.doc_tokens: list[np.ndarray] = []

    def add_documents(self, texts: Sequence[str]) -> None:
        self.doc_tokens.extend(self.encoder.encode(list(texts), is_query=False))

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        q = self.encoder.encode([query], is_query=True)[0]
        scores = [(i, maxsim(q, d)) for i, d in enumerate(self.doc_tokens)]
        return sorted(scores, key=lambda x: (-x[1], x[0]))[:top_k]

    def rerank(self, query: str, candidates: Sequence[int]) -> list[tuple[int, float]]:
        q = self.encoder.encode([query], is_query=True)[0]
        scores = [(i, maxsim(q, self.doc_tokens[i])) for i in candidates]
        return sorted(scores, key=lambda x: (-x[1], x[0]))

    def memory_mb(self) -> float:
        return sum(d.nbytes for d in self.doc_tokens) / 2**20
