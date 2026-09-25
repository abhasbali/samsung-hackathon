"""Multi-retriever fusion.

* **RRF** (default): ``score(d) = sum_i w_i / (k + rank_i(d))`` — scale-free, robust to
  retrievers whose raw scores are incomparable (cosine vs. BM25 vs. symbol counts).
* **score**: per-retriever min-max normalisation then weighted sum (kept for ablations).
* **dense_only**: pass-through of the dense ranking.

Query-adaptive weighting multiplies the base weights by per-intent multipliers from
config. No weight set is claimed optimal — they are ablated.
"""

from __future__ import annotations

import numpy as np

from codefusion.config.schema import FusionConfig
from codefusion.types import Candidate, Hit


def resolve_weights(cfg: FusionConfig, intent: str | None) -> dict[str, float]:
    weights = dict(cfg.weights)
    if cfg.query_adaptive and intent:
        mult = cfg.intent_weights.get(intent, {})
        weights = {k: v * mult.get(k, 1.0) for k, v in weights.items()}
    return weights


def reciprocal_rank_fusion(
    ranked: dict[str, list[Hit]],
    weights: dict[str, float] | None = None,
    k: int = 60,
) -> list[Candidate]:
    weights = weights or {}
    cands: dict[int, Candidate] = {}
    for name, hits in ranked.items():
        w = float(weights.get(name, 1.0))
        for h in hits:
            c = cands.get(h.idx)
            if c is None:
                c = cands[h.idx] = Candidate(idx=h.idx)
            c.ranks[name] = h.rank
            c.raw_scores[name] = h.score
            contrib = w / (k + h.rank) if w > 0 else 0.0
            c.contributions[name] = contrib
            c.fused_score += contrib
            if name == "graph" and h.evidence:
                c.graph_evidence.extend(h.evidence)
            elif h.evidence:
                c.structural_evidence.extend(f"{name}:{e}" for e in h.evidence)
    return _sorted(cands)


def score_fusion(ranked: dict[str, list[Hit]], weights: dict[str, float] | None = None) -> list[Candidate]:
    weights = weights or {}
    cands: dict[int, Candidate] = {}
    for name, hits in ranked.items():
        if not hits:
            continue
        s = np.array([h.score for h in hits], dtype=np.float64)
        lo, hi = float(s.min()), float(s.max())
        w = float(weights.get(name, 1.0))
        for h in hits:
            norm = 1.0 if hi == lo else (h.score - lo) / (hi - lo)
            c = cands.setdefault(h.idx, Candidate(idx=h.idx))
            c.ranks[name] = h.rank
            c.raw_scores[name] = h.score
            c.contributions[name] = w * norm
            c.fused_score += w * norm
    return _sorted(cands)


def passthrough(hits: list[Hit], name: str = "dense") -> list[Candidate]:
    out = []
    for h in hits:
        c = Candidate(idx=h.idx, fused_score=h.score)
        c.ranks[name] = h.rank
        c.raw_scores[name] = h.score
        c.contributions[name] = h.score
        out.append(c)
    return _sorted({c.idx: c for c in out})


def _sorted(cands: dict[int, Candidate]) -> list[Candidate]:
    ordered = sorted(cands.values(), key=lambda c: (-c.fused_score, min(c.ranks.values(), default=10**9), c.idx))
    for r, c in enumerate(ordered, start=1):
        c.fused_rank = r
        c.final_score = c.fused_score
    return ordered


def fuse(cfg: FusionConfig, ranked: dict[str, list[Hit]], intent: str | None) -> tuple[list[Candidate], dict[str, float]]:
    weights = resolve_weights(cfg, intent)
    ranked = {k: v for k, v in ranked.items() if v}
    if cfg.method == "dense_only" or (len(ranked) == 1 and "dense" in ranked):
        return passthrough(ranked.get("dense", [])), weights
    if cfg.method == "score":
        return score_fusion(ranked, weights), weights
    return reciprocal_rank_fusion(ranked, weights, cfg.rrf_k), weights
