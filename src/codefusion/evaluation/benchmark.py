"""Embedding-model benchmark on AppsRetrieval (full split or a seeded subsample).

For every candidate model the benchmark reports, *or* the reason it could not run:

* NDCG@10 / MRR@10 / Recall@100 (dense-only retrieval, FAISS IndexFlatIP over L2-normalised vectors)
* indexing time and throughput (documents actually encoded in this run; cache hits are reported)
* online query latency P50/P95: one query at a time, embedding + vector search (what the API sees)
* process RAM (RSS growth after loading the model) and peak VRAM when running on CUDA

The biggest model is not assumed to win; the table is the evidence for the default choice.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from codefusion.config import load_config
from codefusion.evaluation.metrics import current_memory_mb, evaluate_rankings, percentiles
from codefusion.logging_utils import get_logger

log = get_logger(__name__)

CANDIDATES = ["qodo-1.5b", "jina-code-1.5b", "jina-code-0.5b", "jina-v2-base-code"]


def benchmark_model(model: str, task, base_config: str | Path, overrides: list[str] | None = None,
                    latency_queries: int = 50, use_cache: bool = True) -> dict[str, Any]:
    from codefusion.evaluation.mteb_encoder import corpus_text
    from codefusion.query.preprocess import QueryPreprocessor
    from codefusion.retrieval.dense import VectorIndex
    from codefusion.retrieval.embeddings import (
        CachedEmbedder,
        EmbeddingCache,
        ProviderUnavailable,
        create_provider,
        resolve_model,
    )

    cfg = load_config(base_config, [f"dense.model={model}", *(overrides or [])])
    spec = resolve_model(model)
    row: dict[str, Any] = {"model": model, "hf_id": spec.hf_id, "params": spec.params, "status": "ok",
                           "ndcg_at_10": None, "mrr_at_10": None, "recall_at_100": None,
                           "p50_query_latency_ms": None, "p95_query_latency_ms": None,
                           "indexing_s": None, "docs_encoded": None, "docs_per_s": None,
                           "ram_mb": None, "vram_mb": None, "max_seq_length": cfg.dense.max_seq_length}
    ram0 = current_memory_mb()
    try:
        provider = create_provider(cfg.dense)
    except ProviderUnavailable as exc:
        row["status"] = f"NOT RUN: {exc}"
        log.warning("model unavailable", extra={"data": {"model": model, "error": str(exc)[:200]}})
        return row
    ram1 = current_memory_mb()
    cuda = _cuda_reset(cfg.dense.device)
    cache = EmbeddingCache(Path(cfg.cache_dir) / "embeddings.sqlite") if (use_cache and cfg.dense.cache_embeddings) else None
    emb = CachedEmbedder(provider, cache, batch_size=cfg.dense.batch_size)
    split = task.dataset["default"]["test"]
    doc_ids = list(split["corpus"]["id"])
    docs = [corpus_text(r) for r in split["corpus"]]
    qids = [str(q) for q in split["queries"]["id"]]
    pre = QueryPreprocessor(cfg.query)
    qtexts = [pre.process(t).dense_text for t in split["queries"]["text"]]

    t0 = time.perf_counter()
    dvec = emb.encode_documents(docs)
    idx_s = time.perf_counter() - t0
    index = VectorIndex(dvec.shape[1], cfg.dense.index_type, cfg.dense.hnsw_m, cfg.dense.ivf_nlist, cfg.dense.ivf_nprobe)
    index.add(dvec)
    qvec = emb.encode_queries(qtexts)
    _, top = index.search(qvec, 100, None)
    run = {qid: [doc_ids[i] for i in row_ if i >= 0] for qid, row_ in zip(qids, top)}
    m = evaluate_rankings(run, split["relevant_docs"])

    # Online latency: uncached, one query at a time (embedding + search).
    lat: list[float] = []
    for t in qtexts[:latency_queries]:
        t1 = time.perf_counter()
        v = provider.encode_queries([t], batch_size=1)
        index.search(v, 10, None)
        lat.append((time.perf_counter() - t1) * 1000.0)
    p = percentiles(lat)
    computed = emb.stats.computed
    row.update({
        "ndcg_at_10": round(m["ndcg_at_10"], 5), "mrr_at_10": round(m["mrr_at_10"], 5),
        "recall_at_100": round(m["recall_at_100"], 5),
        "p50_query_latency_ms": round(p["p50"], 1) if p["p50"] is not None else None,
        "p95_query_latency_ms": round(p["p95"], 1) if p["p95"] is not None else None,
        "indexing_s": round(idx_s, 1), "docs_encoded": computed, "docs_reused_from_cache": emb.stats.reused,
        "docs_per_s": round(computed / emb.stats.seconds, 2) if computed and emb.stats.seconds else None,
        "ram_mb": round(ram1 - ram0, 1) if ram0 is not None and ram1 is not None else None,
        "vram_mb": _cuda_peak() if cuda else None,
        "dim": provider.dim, "load_s": round(getattr(provider, "load_seconds", 0.0), 1),
        "query_prompt": provider.query_prompt, "document_prompt": provider.document_prompt,
    })
    if emb.stats.reused and not computed:
        row["note"] = "all document embeddings came from the cache; indexing_s is not an encode time"
    del provider, emb
    return row


def _cuda_reset(device: str) -> bool:
    if not device.startswith("cuda"):
        return False
    try:
        import torch

        torch.cuda.reset_peak_memory_stats()
        return True
    except Exception:  # noqa: BLE001
        return False


def _cuda_peak() -> float | None:
    try:
        import torch

        return round(torch.cuda.max_memory_allocated() / 2**20, 1)
    except Exception:  # noqa: BLE001
        return None


def run_benchmark(models: list[str], base_config: str | Path = "configs/baseline_dense.yaml", max_queries: int | None = 200,
                  max_corpus: int | None = 2000, out_dir: str | Path = "artifacts/benchmarks", overrides: list[str] | None = None,
                  latency_queries: int = 50, use_cache: bool = True) -> list[dict[str, Any]]:
    from codefusion.evaluation.official import load_task

    cfg = load_config(base_config, overrides)
    task = load_task(max_queries, max_corpus, cfg.seed)
    split = task.dataset["default"]["test"]
    tag = f"subset_q{len(split['queries'])}_c{len(split['corpus'])}" if (max_queries or max_corpus) else "full"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for model in models:
        log.info("benchmarking model", extra={"data": {"model": model, "dataset": tag}})
        try:
            row = benchmark_model(model, task, base_config, overrides, latency_queries, use_cache)
        except Exception as exc:  # noqa: BLE001 - e.g. OOM on a large model: report it, continue
            row = {"model": model, "status": f"FAILED: {type(exc).__name__}: {str(exc)[:200]}"}
            log.exception("benchmark failed", extra={"data": {"model": model}})
        row["dataset"] = tag
        rows.append(row)
        _write(rows, out, tag)
        import gc

        gc.collect()
    return rows


def _write(rows: list[dict[str, Any]], out: Path, tag: str) -> None:
    (out / f"embeddings_{tag}.json").write_text(
        json.dumps({"task": "AppsRetrieval", "dataset": tag, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": rows},
                   indent=2, default=str), encoding="utf-8")
    (out / f"embeddings_{tag}.md").write_text(to_markdown(rows, tag), encoding="utf-8")


def to_markdown(rows: list[dict[str, Any]], tag: str) -> str:
    def f(v: Any, nd: int = 4) -> str:
        if v is None:
            return "n/a"
        return f"{v:.{nd}f}" if isinstance(v, float) else str(v)

    lines = [f"### Embedding models on AppsRetrieval ({tag})", "",
             "| Model | Params | NDCG@10 | MRR@10 | P50 query ms | P95 query ms | Index s | docs/s | RAM Δ MB | Peak VRAM MB | Status |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ram = r.get("ram_mb")
        ram = None if ram is not None and ram < 0 else ram  # negative = memory freed from the previous model
        lines.append(f"| {r['model']} | {r.get('params', '')} | {f(r.get('ndcg_at_10'))} | {f(r.get('mrr_at_10'))} | "
                     f"{f(r.get('p50_query_latency_ms'), 1)} | {f(r.get('p95_query_latency_ms'), 1)} | {f(r.get('indexing_s'), 1)} | "
                     f"{f(r.get('docs_per_s'), 2)} | {f(ram, 0)} | {f(r.get('vram_mb'), 0)} | {r.get('status')} |")
    lines += ["", "RAM Δ is the process RSS change while loading the model (n/a when negative: the previous model's memory "
              "was released). On GPU runs the weights live in VRAM, so Peak VRAM is the meaningful footprint. "
              "Index s / docs/s are 0 / n/a when every embedding came from the shared cache."]
    return "\n".join(lines) + "\n"
