"""Official AppsRetrieval evaluation through ``mteb.evaluate``.

``mode="dense"`` evaluates :class:`PrePostPipelineEncoder` (MTEB performs the retrieval);
``mode="hybrid"`` evaluates :class:`CodeFusionSearchModel` (CodeFusion performs retrieval via
MTEB's SearchProtocol). Both write MTEB's ``TaskResult.to_dict()`` JSON.

Subset runs (``max_queries``) are for smoke tests only; their output files are always labelled
``*_subset_*`` and never overwrite the official ``appsretrieval_results.json``.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

from codefusion.config import CodeFusionConfig, config_to_dict
from codefusion.evaluation.metrics import evaluate_rankings, peak_memory_mb, percentiles
from codefusion.logging_utils import get_logger

log = get_logger(__name__)

TASK_NAME = "AppsRetrieval"


def load_task(max_queries: int | None = None, max_corpus: int | None = None, seed: int = 42):
    """Load the MTEB task, optionally sub-sampled (queries + their relevant docs + distractors)."""
    import mteb

    task = mteb.get_task(TASK_NAME)
    task.load_data()
    if not max_queries and not max_corpus:
        return task
    split = task.dataset["default"]["test"]
    queries, corpus, qrels = split["queries"], split["corpus"], split["relevant_docs"]
    rng = random.Random(seed)
    q_idx = list(range(len(queries)))
    if max_queries and max_queries < len(q_idx):
        q_idx = sorted(rng.sample(q_idx, max_queries))
    queries = queries.select(q_idx)
    keep_q = set(queries["id"])
    qrels = {q: r for q, r in qrels.items() if q in keep_q}
    needed = {d for r in qrels.values() for d in r}
    ids = corpus["id"]
    rel_idx = [i for i, d in enumerate(ids) if d in needed]
    other = [i for i, d in enumerate(ids) if d not in needed]
    if max_corpus and max_corpus > len(rel_idx):
        extra = sorted(rng.sample(other, min(len(other), max_corpus - len(rel_idx))))
    elif max_corpus:
        extra = []
    else:
        extra = other
    corpus = corpus.select(sorted(rel_idx + extra))
    split.update({"queries": queries, "corpus": corpus, "relevant_docs": qrels})
    log.info("subsampled task", extra={"data": {"queries": len(queries), "corpus": len(corpus)}})
    return task


def _extract_scores(task_result_dict: dict[str, Any]) -> dict[str, Any]:
    scores = task_result_dict.get("scores", {}).get("test", [{}])
    s = scores[0] if scores else {}
    keys = ("main_score", "ndcg_at_1", "ndcg_at_5", "ndcg_at_10", "ndcg_at_100", "mrr_at_1", "mrr_at_5", "mrr_at_10",
            "mrr_at_100", "recall_at_10", "recall_at_100", "map_at_10")
    return {k: s.get(k) for k in keys if k in s}


def run_official(
    cfg: CodeFusionConfig,
    mode: str = "dense",
    output: str | Path = "appsretrieval_results.json",
    max_queries: int | None = None,
    max_corpus: int | None = None,
    batch_size: int = 64,
    predictions_dir: str | Path | None = None,
    embedder=None,
) -> dict[str, Any]:
    import mteb

    from codefusion.evaluation.mteb_encoder import CodeFusionSearchModel, PrePostPipelineEncoder

    t_all = time.perf_counter()
    task = load_task(max_queries, max_corpus, cfg.seed)
    split = task.dataset["default"]["test"]
    n_q, n_c = len(split["queries"]), len(split["corpus"])
    subset = bool(max_queries or max_corpus)
    output = Path(output)
    if subset and "subset" not in output.name:
        output = output.with_name(f"{output.stem}_subset_q{n_q}_c{n_c}{output.suffix}")

    t_model = time.perf_counter()
    model = (CodeFusionSearchModel if mode == "hybrid" else PrePostPipelineEncoder)(cfg, embedder=embedder)
    model_load_s = time.perf_counter() - t_model
    t_eval = time.perf_counter()
    result = mteb.evaluate(
        model,
        [task],
        encode_kwargs={"batch_size": batch_size},
        cache=None,
        overwrite_strategy="always",
        prediction_folder=str(predictions_dir) if predictions_dir else None,
        show_progress_bar=False,
    )
    eval_s = time.perf_counter() - t_eval
    task_result = list(result.task_results)[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    # MTEB's canonical serialiser (to_dict() carries a datetime that plain json.dump rejects).
    task_result.to_disk(output)
    d = json.loads(output.read_text(encoding="utf-8"))

    scores = _extract_scores(d)
    summary: dict[str, Any] = {
        "task": TASK_NAME,
        "mode": mode,
        "config": cfg.name,
        "model": cfg.dense.model if cfg.dense.enabled else None,
        "subset": subset,
        "n_queries": n_q,
        "n_corpus": n_c,
        "official_json": str(output),
        "ndcg_at_10": scores.get("ndcg_at_10"),
        "mrr_at_10": scores.get("mrr_at_10"),
        "scores": scores,
        "model_load_s": round(model_load_s, 2),
        "evaluation_s": round(eval_s, 2),
        "total_s": round(time.perf_counter() - t_all, 2),
        "peak_memory_mb": peak_memory_mb(),
        "enabled_components": cfg.enabled_components(),
        "embedding_stats": vars(model.embedder.stats) if getattr(model, "embedder", None) else None,
        "encode_seconds": getattr(model, "encode_seconds", None),
    }
    if mode == "hybrid":
        summary["hybrid_timing_s"] = {k: round(v, 3) for k, v in model.timing.items()}
        summary["query_latency_ms"] = percentiles(model.query_latencies_ms)
    summary_path = output.with_name(output.stem + "_summary.json")
    summary_path.write_text(json.dumps({**summary, "config_dump": config_to_dict(cfg)}, indent=2, default=str), encoding="utf-8")
    log.info("official evaluation finished", extra={"data": {k: summary[k] for k in ("mode", "ndcg_at_10", "mrr_at_10", "n_queries", "total_s")}})
    return summary


def evaluate_engine_rankings(engine, task, top_k: int = 100) -> tuple[dict[str, float], list[float]]:
    """Run an already-indexed engine over a loaded task and score it with our own metrics."""
    split = task.dataset["default"]["test"]
    doc_ids = list(split["corpus"]["id"])
    qids = list(split["queries"]["id"])
    texts = list(split["queries"]["text"])
    pqs = [engine.process_query(t) for t in texts]
    if engine.dense is not None:
        vecs = engine.embedder.encode_queries([p.dense_text or p.text for p in pqs])
        for p, v in zip(pqs, vecs):
            p.dense_vector = v
    run: dict[str, list[str]] = {}
    lat: list[float] = []
    for qid, p in zip(qids, pqs):
        t0 = time.perf_counter()
        resp = engine.search(p, top_k=top_k, explain=False)
        lat.append((time.perf_counter() - t0) * 1000.0)
        run[str(qid)] = [doc_ids[r.idx] for r in resp.results]
    return evaluate_rankings(run, split["relevant_docs"]), lat
