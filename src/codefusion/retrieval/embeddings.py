"""Embedding providers.

``EmbeddingProvider`` is the single abstraction every dense component uses:

* :class:`SentenceTransformerProvider` — any SentenceTransformers-compatible model, with the
  *model-specific* query/document prompts from :data:`MODEL_REGISTRY` (each model's
  documented retrieval format is respected; models are **not** all encoded identically).
* :class:`HashingProvider` — a deterministic, dependency-light lexical embedding (signed feature
  hashing of code-aware tokens). Used for CI/tests and as a last-resort fallback. It is a real
  (weak) embedding, never a stand-in for reported benchmark numbers.

:class:`CachedEmbedder` adds a persistent SQLite cache keyed by ``sha1(model|prompt|text)`` so
unchanged snippets are never re-embedded (incremental indexing, resumable CPU evaluations).
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from codefusion.config.schema import DenseConfig
from codefusion.logging_utils import get_logger
from codefusion.parsing.identifiers import CodeTokenizer

log = get_logger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    query_prompt: str = ""
    document_prompt: str = ""
    trust_remote_code: bool = False
    padding_side: str | None = None
    max_seq_length: int = 1024
    params: str = ""
    notes: str = ""


# Prompts follow each model's documentation (model card / config_sentence_transformers.json).
MODEL_REGISTRY: dict[str, ModelSpec] = {
    "qodo-1.5b": ModelSpec(
        key="qodo-1.5b",
        hf_id="Qodo/Qodo-Embed-1-1.5B",
        # config_sentence_transformers.json ships an empty prompt map; MTEB meta: use_instructions=False.
        trust_remote_code=False,
        params="1.5B",
        notes="gte-Qwen2-1.5B based, last-token pooling, 1536-d; no instructions.",
    ),
    "jina-code-1.5b": ModelSpec(
        key="jina-code-1.5b",
        hf_id="jinaai/jina-code-embeddings-1.5b",
        query_prompt="Find the most relevant code snippet given the following query:\n",
        document_prompt="Candidate code snippet:\n",
        padding_side="left",
        params="1.5B",
        notes="Qwen2.5-Coder-1.5B based, last-token pooling, nl2code prompts, 1536-d.",
    ),
    "jina-code-0.5b": ModelSpec(
        key="jina-code-0.5b",
        hf_id="jinaai/jina-code-embeddings-0.5b",
        query_prompt="Find the most relevant code snippet given the following query:\n",
        document_prompt="Candidate code snippet:\n",
        padding_side="left",
        params="0.5B",
        notes="Qwen2.5-Coder-0.5B based, last-token pooling, nl2code prompts, 896-d.",
    ),
    "jina-v2-base-code": ModelSpec(
        key="jina-v2-base-code",
        hf_id="jinaai/jina-embeddings-v2-base-code",
        trust_remote_code=True,
        params="161M",
        notes="JinaBERT (ALiBi), mean pooling, 768-d; no prompts. Its remote code requires transformers<5.",
    ),
    "minilm": ModelSpec(
        key="minilm",
        hf_id="sentence-transformers/all-MiniLM-L6-v2",
        max_seq_length=256,
        params="22M",
        notes="General-domain tiny model; used only for fast smoke tests.",
    ),
    "hashing": ModelSpec(key="hashing", hf_id="hashing", params="0", notes="Deterministic feature hashing; tests/CI only."),
}


def resolve_model(name: str) -> ModelSpec:
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]
    for spec in MODEL_REGISTRY.values():
        if spec.hf_id == name:
            return spec
    return ModelSpec(key=name, hf_id=name, notes="unregistered model: no prompts applied")


class EmbeddingProvider(ABC):
    """Encodes queries and documents into L2-normalised float32 vectors."""

    name: str = "provider"
    query_prompt: str = ""
    document_prompt: str = ""
    # Identifies everything that changes the vectors (model, truncation, dtype, quantisation).
    fingerprint: str = "provider"

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def _encode(self, texts: Sequence[str], prompt: str, batch_size: int) -> np.ndarray: ...

    def encode_queries(self, texts: Sequence[str], batch_size: int = 16) -> np.ndarray:
        return self._encode(texts, self.query_prompt, batch_size)

    def encode_documents(self, texts: Sequence[str], batch_size: int = 16) -> np.ndarray:
        return self._encode(texts, self.document_prompt, batch_size)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


class HashingProvider(EmbeddingProvider):
    """Signed feature hashing over code-aware tokens with sublinear TF. No model download."""

    def __init__(self, dim: int = 768) -> None:
        self.name = "hashing"
        self.fingerprint = f"hashing|{dim}"
        self._dim = dim
        self.tok = CodeTokenizer(stem_tokens=True, remove_stopwords=True)

    @property
    def dim(self) -> int:
        return self._dim

    def _encode(self, texts: Sequence[str], prompt: str, batch_size: int) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            counts: dict[str, int] = {}
            for tok in self.tok.tokenize(t):
                counts[tok] = counts.get(tok, 0) + 1
            for tok, c in counts.items():
                h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
                out[i, h % self._dim] += (1.0 if (h >> 63) & 1 else -1.0) * (1.0 + np.log(c))
        return l2_normalize(out)


class SentenceTransformerProvider(EmbeddingProvider):
    """SentenceTransformers model with model-specific prompts and CPU-friendly options."""

    def __init__(self, spec: ModelSpec, cfg: DenseConfig) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        if cfg.torch_threads:
            torch.set_num_threads(cfg.torch_threads)
        torch.manual_seed(0)
        dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[cfg.dtype]
        tok_kwargs = {"padding_side": spec.padding_side} if spec.padding_side else None
        t0 = time.perf_counter()
        self.model = SentenceTransformer(
            spec.hf_id,
            device=cfg.device,
            trust_remote_code=spec.trust_remote_code,
            model_kwargs={"torch_dtype": dtype},
            tokenizer_kwargs=tok_kwargs,
        )
        self.model.max_seq_length = cfg.max_seq_length or spec.max_seq_length
        if cfg.quantize_int8 and cfg.device == "cpu":
            self.model = torch.ao.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
        self.model.eval()
        self.name = spec.key
        self.fingerprint = f"{spec.hf_id}|len{self.model.max_seq_length}|{cfg.dtype}{'|int8' if cfg.quantize_int8 else ''}"
        self.query_prompt = cfg.query_prompt if cfg.query_prompt is not None else spec.query_prompt
        self.document_prompt = cfg.document_prompt if cfg.document_prompt is not None else spec.document_prompt
        self.normalize = cfg.normalize
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self._dim = int(get_dim() or 0)
        self.load_seconds = time.perf_counter() - t0
        log.info(
            "embedding model loaded",
            extra={"data": {"model": spec.hf_id, "dim": self._dim, "max_seq_length": self.model.max_seq_length,
                            "int8": cfg.quantize_int8, "dtype": cfg.dtype, "seconds": round(self.load_seconds, 1)}},
        )

    @property
    def dim(self) -> int:
        return self._dim

    def _encode(self, texts: Sequence[str], prompt: str, batch_size: int) -> np.ndarray:
        import torch

        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        with torch.inference_mode():
            emb = self.model.encode(
                list(texts),
                prompt=prompt or None,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )
        emb = np.asarray(emb, dtype=np.float32)
        return l2_normalize(emb) if self.normalize else emb


class ProviderUnavailable(RuntimeError):
    pass


def create_provider(cfg: DenseConfig, allow_fallback: bool = False) -> EmbeddingProvider:
    """Instantiate the configured provider. Raises :class:`ProviderUnavailable` with a clear reason
    (auth/network/missing dependency) unless ``allow_fallback`` is set, in which case the hashing
    provider is returned and a warning is logged."""
    spec = resolve_model(cfg.model)
    if cfg.provider == "hashing" or spec.key == "hashing":
        return HashingProvider(cfg.hashing_dim)
    try:
        return SentenceTransformerProvider(spec, cfg)
    except Exception as exc:  # noqa: BLE001 - network/auth/dependency errors are all reported
        msg = f"embedding model {spec.hf_id!r} unavailable: {type(exc).__name__}: {str(exc)[:300]}"
        if allow_fallback:
            log.warning(msg + " -> falling back to hashing provider")
            return HashingProvider(cfg.hashing_dim)
        raise ProviderUnavailable(msg) from exc


class EmbeddingCache:
    """Persistent SQLite store of embeddings keyed by (namespace, sha1(prompt|text))."""

    def __init__(self, path: str | Path | None) -> None:
        self.path = str(path) if path else ":memory:"
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS emb (ns TEXT NOT NULL, key TEXT NOT NULL, dim INTEGER, vec BLOB, PRIMARY KEY (ns, key))"
        )
        self.conn.commit()

    @staticmethod
    def key(prompt: str, text: str) -> str:
        return hashlib.sha1(f"{prompt}\x00{text}".encode("utf-8", errors="replace")).hexdigest()

    def get_many(self, ns: str, keys: list[str]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        with self._lock:
            for i in range(0, len(keys), 900):
                chunk = keys[i : i + 900]
                q = f"SELECT key, dim, vec FROM emb WHERE ns=? AND key IN ({','.join('?' * len(chunk))})"
                for k, dim, blob in self.conn.execute(q, [ns, *chunk]):
                    out[k] = np.frombuffer(blob, dtype=np.float32, count=dim).copy()
        return out

    def put_many(self, ns: str, items: list[tuple[str, np.ndarray]]) -> None:
        with self._lock:
            self.conn.executemany(
                "INSERT OR REPLACE INTO emb (ns, key, dim, vec) VALUES (?, ?, ?, ?)",
                [(ns, k, int(v.shape[0]), np.asarray(v, dtype=np.float32).tobytes()) for k, v in items],
            )
            self.conn.commit()

    def count(self, ns: str | None = None) -> int:
        with self._lock:
            if ns is None:
                return int(self.conn.execute("SELECT COUNT(*) FROM emb").fetchone()[0])
            return int(self.conn.execute("SELECT COUNT(*) FROM emb WHERE ns=?", (ns,)).fetchone()[0])

    def close(self) -> None:
        self.conn.close()


@dataclass
class EmbedStats:
    computed: int = 0
    reused: int = 0
    seconds: float = 0.0


class CachedEmbedder:
    """Provider + cache. Encodes only texts whose (prompt, text) key is not cached.

    Work is committed to the cache every ``chunk_size`` texts so multi-hour CPU runs can be
    interrupted and resumed without losing progress.
    """

    def __init__(self, provider: EmbeddingProvider, cache: EmbeddingCache | None, batch_size: int = 16,
                 chunk_size: int = 256, max_chars: int = 12000) -> None:
        self.provider = provider
        self.cache = cache
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.max_chars = max_chars
        self.stats = EmbedStats()

    @property
    def dim(self) -> int:
        return self.provider.dim

    def _run(self, texts: Sequence[str], kind: str) -> np.ndarray:
        prompt = self.provider.query_prompt if kind == "query" else self.provider.document_prompt
        texts = [t[: self.max_chars] if t else " " for t in texts]
        ns = f"{self.provider.fingerprint}:{kind}"
        keys = [EmbeddingCache.key(prompt, t) for t in texts]
        found = self.cache.get_many(ns, list(dict.fromkeys(keys))) if self.cache else {}
        missing_order = [i for i, k in enumerate(keys) if k not in found]
        # Encode each distinct missing text once.
        uniq: dict[str, int] = {}
        for i in missing_order:
            uniq.setdefault(keys[i], i)
        todo = list(uniq.items())
        # Longest-first chunks keep padding waste low and give an early, pessimistic ETA.
        todo.sort(key=lambda kv: -len(texts[kv[1]]))
        t0 = time.perf_counter()
        for c in range(0, len(todo), self.chunk_size):
            chunk = todo[c : c + self.chunk_size]
            enc = (self.provider.encode_queries if kind == "query" else self.provider.encode_documents)(
                [texts[i] for _, i in chunk], batch_size=self.batch_size
            )
            items = [(k, enc[j]) for j, (k, _) in enumerate(chunk)]
            for k, v in items:
                found[k] = v
            if self.cache:
                self.cache.put_many(ns, items)
            if len(todo) > self.chunk_size:
                done = min(c + self.chunk_size, len(todo))
                el = time.perf_counter() - t0
                log.info("encoding progress", extra={"data": {"kind": kind, "done": done, "total": len(todo),
                                                               "elapsed_s": round(el, 1), "eta_s": round(el / done * (len(todo) - done), 1)}})
        self.stats.computed += len(todo)
        self.stats.reused += len(texts) - len(missing_order)
        self.stats.seconds += time.perf_counter() - t0
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([found[k] for k in keys]).astype(np.float32)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._run(texts, "query")

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._run(texts, "document")
