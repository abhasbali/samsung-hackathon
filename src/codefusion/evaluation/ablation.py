"""Automated ablation runner on AppsRetrieval (CoIR) through CodeFusion's own engine.

Every experiment is a set of dotted config overrides applied to one base config, so each row of
the output differs from its predecessor by exactly the component named in the table. All
experiments share one :class:`CachedEmbedder`: the corpus is encoded once (or not at all when the
official run already filled the cache) and every dense experiment reuses those vectors.

Metrics come from :func:`codefusion.evaluation.metrics.evaluate_rankings`, which mirrors the
pytrec_eval definitions MTEB uses (experiment A reproduces the official dense score, which the
runner checks when an official result file is available).

Experiments that cannot run in the current environment (missing model, Joern not installed, ...)
are written to the table with ``status`` set to the reason and **no** metric values.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codefusion.config import CodeFusionConfig, apply_overrides, config_to_dict, load_config
from codefusion.evaluation.metrics import peak_memory_mb, percentiles
from codefusion.logging_utils import get_logger

log = get_logger(__name__)

# Components switched off to obtain the "nothing but X" starting points.
_ALL_OFF = [
    "bm25.enabled=false",
    "symbols.enabled=false",
    "graph.enabled=false",
    "structural.enabled=false",
    "reranker.enabled=false",
    "diversity.enabled=false",
    "fusion.query_adaptive=false",
    "query.expansion=false",
    "versions.enabled=false",
]
_PLAIN_BM25 = ["bm25.identifier_splitting=false", "bm25.fields={raw: 1.0}"]
_CODE_BM25_FIELDS = "bm25.fields={raw: 1.0, identifiers: 0.0, structural: 0.3, context: 0.3, docstring: 0.5}"


@dataclass
class Experiment:
    key: str
    name: str
    overrides: list[str]
    description: str = ""
    requires: list[str] = field(default_factory=list)  # "reranker", "joern", "late_interaction"


def default_experiments() -> list[Experiment]:
    """A-N from the challenge brief, each adding one component to the previous row."""
    d = ["dense.enabled=true"]
    rrf = ["fusion.method=rrf", "fusion.rrf_k=60"]
    D = [*_ALL_OFF, *d, "bm25.enabled=true", *_PLAIN_BM25, *rrf]
    E = [*D, "bm25.identifier_splitting=true"]
    F = [*E, _CODE_BM25_FIELDS]
    G = [*F, "symbols.enabled=true"]
    H = [*G, "reranker.enabled=true"]
    I = [*G, "graph.enabled=true", "structural.enabled=true"]  # noqa: E741 - experiment letter
    J = [*I, "fusion.query_adaptive=true"]
    return [
        Experiment("A", "dense only", [*_ALL_OFF, *d, "fusion.method=dense_only"], "code embedding + FAISS IndexFlatIP"),
        Experiment("B", "BM25 only", [*_ALL_OFF, "dense.enabled=false", "bm25.enabled=true", *_PLAIN_BM25, *rrf],
                   "bm25s over raw text, plain tokenisation"),
        Experiment("C", "dense + BM25 (score fusion)", [*D, "fusion.method=score"], "min-max normalised score sum"),
        Experiment("D", "dense + BM25 + RRF", D, "reciprocal rank fusion, k=60, equal weights"),
        Experiment("D2", "dense + BM25 + RRF (dense-weighted)", [*D, "fusion.weights={dense: 1.0, bm25: 0.3, symbol: 0.3, graph: 0.3}"],
                   "RRF with BM25 down-weighted to 0.3"),
        Experiment("E", "+ identifier preprocessing", E, "camel/snake splitting + stemming in BM25"),
        Experiment("F", "+ structural representation", F, "BM25F over raw + structural + context + docstring views"),
        Experiment("G", "+ symbol retrieval", G, "normalised inverted symbol index"),
        Experiment("H", "+ reranker", H, "cross-encoder over the fused top-N", requires=["reranker"]),
        Experiment("I", "+ code graph", I, "1-hop graph expansion + structural scoring"),
        Experiment("J", "+ query-adaptive weighting", J, "intent-specific RRF weights"),
        Experiment("K", "full system", [*J, "diversity.enabled=true", "query.expansion=true", "versions.enabled=true"],
                   "configs/full.yaml components"),
        Experiment("L", "+ diversity/MMR", [*J, "diversity.enabled=true", "diversity.mmr=true"], "dedup + MMR (lambda 0.75)"),
        Experiment("M", "+ late interaction", J, "MaxSim rerank of the fused top-N", requires=["late_interaction"]),
        Experiment("N", "+ Joern / CPG", J, "CPG data-flow edges in the graph", requires=["joern"]),
    ]


def _availability(req: str, cfg: CodeFusionConfig) -> str | None:
    """Return None if a requirement is met, otherwise a human-readable NOT RUN reason."""
    if req == "joern":
        from codefusion.experimental.joern import joern_available

        return None if joern_available() else "NOT RUN: Joern CLI not installed (see docs/evaluation.md)"
    if req == "reranker":
        return None  # checked after engine construction (the engine owns the loaded model)
    if req == "late_interaction":
        return "NOT RUN: late interaction needs a per-token encoder pass over the corpus (see docs/evaluation.md)"
    return None


def run_ablations(
    base_config: str | Path,
    experiments: list[Experiment] | None = None,
    only: list[str] | None = None,
    max_queries: int | None = None,
    max_corpus: int | None = None,
    out_dir: str | Path = "artifacts/ablations",
    extra_overrides: list[str] | None = None,
    reference_official: str | Path | None = None,
) -> list[dict[str, Any]]:
    from codefusion.evaluation.mteb_encoder import _lang, corpus_text
    from codefusion.evaluation.official import evaluate_engine_rankings, load_task
    from codefusion.evaluation.tracking import log_run
    from codefusion.parsing.chunker import analyze_document
    from codefusion.retrieval.embeddings import CachedEmbedder, EmbeddingCache, create_provider
    from codefusion.retrieval.pipeline import CodeFusionEngine

    experiments = experiments or default_experiments()
    if only:
        wanted = {k.upper() for k in only}
        experiments = [e for e in experiments if e.key.upper() in wanted]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = load_config(base_config, extra_overrides)
    base_dict = config_to_dict(base)

    task = load_task(max_queries, max_corpus, base.seed)
    split = task.dataset["default"]["test"]
    n_q, n_c = len(split["queries"]), len(split["corpus"])
    subset = bool(max_queries or max_corpus)
    tag = f"subset_q{n_q}_c{n_c}" if subset else "full"
    corpus_rows = list(split["corpus"])

    # Parse the corpus once; every experiment gets fresh copies of the parsed snippets.
    t0 = time.perf_counter()
    parsed = [analyze_document(corpus_text(r), str(r["id"]), language=_lang(r), prefer_tree_sitter=base.chunking.prefer_tree_sitter)
              for r in corpus_rows]
    parse_s = time.perf_counter() - t0
    log.info("ablation corpus parsed", extra={"data": {"docs": n_c, "queries": n_q, "seconds": round(parse_s, 1)}})

    embedder: CachedEmbedder | None = None
    rows: list[dict[str, Any]] = []
    for exp in experiments:
        cfg = CodeFusionConfig.model_validate(apply_overrides(base_dict, exp.overrides))
        cfg.name = f"ablation-{exp.key}"
        row: dict[str, Any] = {
            "experiment": exp.key, "name": exp.name, "description": exp.description,
            "model": cfg.dense.model if cfg.dense.enabled else None,
            "enabled_components": cfg.enabled_components(), "dataset": tag,
            "n_queries": n_q, "n_corpus": n_c, "ndcg_at_10": None, "mrr_at_10": None,
            "p50_latency_ms": None, "p95_latency_ms": None, "indexing_s": None, "status": "ok",
        }
        reason = next((r for r in (_availability(q, cfg) for q in exp.requires) if r), None)
        if reason:
            row["status"] = reason
            rows.append(row)
            log.warning("ablation skipped", extra={"data": {"experiment": exp.key, "reason": reason}})
            continue
        try:
            if cfg.dense.enabled and embedder is None:
                provider = create_provider(cfg.dense)
                cache = EmbeddingCache(Path(cfg.cache_dir) / "embeddings.sqlite") if cfg.dense.cache_embeddings else None
                embedder = CachedEmbedder(provider, cache, batch_size=cfg.dense.batch_size)
            before = (embedder.stats.computed, embedder.stats.seconds) if embedder else (0, 0.0)
            t_idx = time.perf_counter()
            eng = CodeFusionEngine(cfg, embedder=embedder if cfg.dense.enabled else None)
            if cfg.reranker.enabled and eng.reranker is None:
                raise _NotRun(f"NOT RUN: reranker {cfg.reranker.model!r} could not be loaded")
            eng.add_snippets([_fresh(s) for s in parsed])
            eng.finalize()
            idx_s = time.perf_counter() - t_idx
            t_q = time.perf_counter()
            scores, lat = evaluate_engine_rankings(eng, task, top_k=100)
            eval_s = time.perf_counter() - t_q
            lat_p = percentiles(lat)
            row.update({
                "ndcg_at_10": round(scores["ndcg_at_10"], 5),
                "mrr_at_10": round(scores["mrr_at_10"], 5),
                "recall_at_100": round(scores["recall_at_100"], 5),
                "p50_latency_ms": round(lat_p["p50"], 2) if lat_p["p50"] is not None else None,
                "p95_latency_ms": round(lat_p["p95"], 2) if lat_p["p95"] is not None else None,
                # parse time is shared across experiments and reported separately
                "indexing_s": round(idx_s + parse_s, 2),
                "index_stage_s": {k: round(v, 2) for k, v in eng.index_seconds.items()} | {"parse": round(parse_s, 2)},
                "embeddings_computed": (embedder.stats.computed - before[0]) if embedder else 0,
                "embedding_compute_s": round(embedder.stats.seconds - before[1], 1) if embedder else 0.0,
                "eval_s": round(eval_s, 1),
                "peak_memory_mb": peak_memory_mb(),
            })
            log_run(cfg, f"ablation-{exp.key}", {"ndcg_at_10": row["ndcg_at_10"], "mrr_at_10": row["mrr_at_10"],
                                                  "p50_latency_ms": row["p50_latency_ms"] or 0.0,
                                                  "p95_latency_ms": row["p95_latency_ms"] or 0.0,
                                                  "indexing_s": row["indexing_s"]}, extra={"dataset": tag})
        except _NotRun as nr:
            row["status"] = str(nr)
        except Exception as exc:  # noqa: BLE001 - one broken experiment must not lose the others
            row["status"] = f"FAILED: {type(exc).__name__}: {str(exc)[:200]}"
            log.exception("ablation failed", extra={"data": {"experiment": exp.key}})
        rows.append(row)
        log.info("ablation done", extra={"data": {k: row[k] for k in ("experiment", "ndcg_at_10", "mrr_at_10", "status")}})
        _write(rows, out, tag, base_config, reference_official)  # persist after every experiment
    return rows


class _NotRun(RuntimeError):
    pass


def _fresh(sn):
    """Shallow copy so per-experiment mutation (commits, lineage ids) never leaks between runs."""
    import copy

    return copy.copy(sn)


def _write(rows: list[dict[str, Any]], out: Path, tag: str, base_config: str | Path, reference: str | Path | None) -> None:
    payload: dict[str, Any] = {"task": "AppsRetrieval", "dataset": tag, "base_config": str(base_config),
                               "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": rows}
    if reference and Path(reference).exists():
        ref = json.loads(Path(reference).read_text(encoding="utf-8"))
        s = ref.get("scores", {}).get("test", [{}])[0]
        payload["official_reference"] = {"file": str(reference), "ndcg_at_10": s.get("ndcg_at_10"), "mrr_at_10": s.get("mrr_at_10")}
    (out / f"ablations_{tag}.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    cols = ["experiment", "name", "ndcg_at_10", "mrr_at_10", "recall_at_100", "p50_latency_ms", "p95_latency_ms",
            "indexing_s", "model", "enabled_components", "status"]
    with (out / f"ablations_{tag}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([";".join(r[c]) if isinstance(r.get(c), list) else r.get(c) for c in cols])
    (out / f"ablations_{tag}.md").write_text(to_markdown(rows, tag), encoding="utf-8")


def _fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def to_markdown(rows: list[dict[str, Any]], tag: str) -> str:
    lines = [
        f"### AppsRetrieval ablations ({tag})",
        "",
        "| Exp | Configuration | NDCG@10 | MRR@10 | P50 ms | P95 ms | Index s | Status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['experiment']} | {r['name']} | {_fmt(r['ndcg_at_10'])} | {_fmt(r['mrr_at_10'])} | "
            f"{_fmt(r['p50_latency_ms'], 1)} | {_fmt(r['p95_latency_ms'], 1)} | {_fmt(r['indexing_s'], 1)} | {r['status']} |"
        )
    lines += ["", "Latency is per-query engine search time; batched query embedding is excluded and reported in the JSON.", ""]
    return "\n".join(lines)
