"""Retrieval metrics (NDCG@k, MRR@k, Recall@k) and resource helpers.

These mirror the pytrec_eval definitions MTEB uses; the evaluation scripts cross-check them
against MTEB's own numbers on the same rankings.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Mapping

import numpy as np


def ndcg_at_k(ranked: list[str], relevant: Mapping[str, int], k: int = 10) -> float:
    dcg = 0.0
    for i, d in enumerate(ranked[:k]):
        rel = relevant.get(d, 0)
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(i + 2)
    ideal = sorted((r for r in relevant.values() if r > 0), reverse=True)[:k]
    idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(ranked: list[str], relevant: Mapping[str, int], k: int = 10) -> float:
    for i, d in enumerate(ranked[:k]):
        if relevant.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked: list[str], relevant: Mapping[str, int], k: int = 100) -> float:
    rel = {d for d, r in relevant.items() if r > 0}
    if not rel:
        return 0.0
    return len(rel & set(ranked[:k])) / len(rel)


def evaluate_rankings(
    run: Mapping[str, list[str]],
    qrels: Mapping[str, Mapping[str, int]],
    ks: tuple[int, ...] = (1, 5, 10, 100),
) -> dict[str, float]:
    """Mean metrics over queries present in ``qrels`` (missing runs count as zero)."""
    out: dict[str, list[float]] = {}
    for qid, rels in qrels.items():
        ranked = run.get(qid, [])
        for k in ks:
            out.setdefault(f"ndcg_at_{k}", []).append(ndcg_at_k(ranked, rels, k))
            out.setdefault(f"mrr_at_{k}", []).append(mrr_at_k(ranked, rels, k))
            out.setdefault(f"recall_at_{k}", []).append(recall_at_k(ranked, rels, k))
    return {k: float(np.mean(v)) if v else 0.0 for k, v in out.items()}


def percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "mean": None}
    arr = np.asarray(values, dtype=np.float64)
    return {"p50": float(np.percentile(arr, 50)), "p95": float(np.percentile(arr, 95)), "mean": float(arr.mean())}


def peak_memory_mb() -> float | None:
    """Peak resident memory of this process (Windows: peak working set; POSIX: ru_maxrss)."""
    try:
        import psutil

        mi = psutil.Process(os.getpid()).memory_info()
        peak = getattr(mi, "peak_wset", None)
        if peak:
            return round(peak / 2**20, 1)
    except Exception:  # noqa: BLE001
        pass
    try:
        import resource

        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(r / 1024, 1)  # Linux reports KiB
    except Exception:  # noqa: BLE001
        return None


def current_memory_mb() -> float | None:
    try:
        import psutil

        return round(psutil.Process(os.getpid()).memory_info().rss / 2**20, 1)
    except Exception:  # noqa: BLE001
        return None


class Stopwatch:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0
