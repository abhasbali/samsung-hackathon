"""Deduplication and diversity for the final ranking.

Removes (or demotes) candidates that add no new information:

* identical ``content_hash`` (exact duplicates, e.g. the same function in several commits),
* identical ``ast_hash`` (formatting/comment-only differences),
* near-duplicates by Jaccard similarity over identifier shingles,
* several versions of the same lineage (unless the query asks about history).

Optional MMR re-orders by ``lambda * relevance - (1 - lambda) * max_sim_to_selected``.
"""

from __future__ import annotations

import re

import numpy as np

from codefusion.config.schema import DiversityConfig
from codefusion.types import Candidate, Snippet

_TOK = re.compile(r"[A-Za-z_]\w+")


def shingles(text: str, n: int = 3) -> frozenset[str]:
    toks = _TOK.findall(text)
    if len(toks) < n:
        return frozenset(toks)
    return frozenset(" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def deduplicate(
    cands: list[Candidate],
    snippets: list[Snippet],
    cfg: DiversityConfig,
    keep_history: bool = False,
    limit: int | None = None,
) -> tuple[list[Candidate], list[Candidate]]:
    """Return (kept, dropped). ``dropped`` candidates carry ``dropped_reason`` for explainability."""
    kept: list[Candidate] = []
    dropped: list[Candidate] = []
    seen_content: dict[str, int] = {}
    seen_ast: dict[str, int] = {}
    seen_lineage: dict[str, int] = {}
    kept_shingles: list[frozenset[str]] = []
    for c in cands:
        sn = snippets[c.idx]
        reason = None
        if cfg.dedup_content and sn.content_hash in seen_content:
            reason = f"duplicate content of rank {seen_content[sn.content_hash]}"
        elif cfg.dedup_ast and sn.ast_hash and sn.ast_hash in seen_ast:
            reason = f"same AST as rank {seen_ast[sn.ast_hash]}"
        elif cfg.collapse_lineage and not keep_history and sn.lineage_id and sn.lineage_id in seen_lineage:
            reason = f"older/other version of rank {seen_lineage[sn.lineage_id]}"
        elif cfg.near_duplicate_threshold < 1.0:
            sh = shingles(sn.content)
            for j, ks in enumerate(kept_shingles):
                if len(sh) > 5 and jaccard(sh, ks) >= cfg.near_duplicate_threshold:
                    if keep_history and snippets[kept[j].idx].lineage_id == sn.lineage_id:
                        continue  # history queries want the versions
                    reason = f"near-duplicate of rank {j + 1}"
                    break
        if reason:
            c.dropped_reason = reason
            dropped.append(c)
            continue
        rank = len(kept) + 1
        kept.append(c)
        kept_shingles.append(shingles(sn.content))
        seen_content.setdefault(sn.content_hash, rank)
        if sn.ast_hash:
            seen_ast.setdefault(sn.ast_hash, rank)
        if sn.lineage_id:
            seen_lineage.setdefault(sn.lineage_id, rank)
        if limit is not None and len(kept) >= limit:
            break
    return kept, dropped


def mmr(cands: list[Candidate], vectors: np.ndarray | None, snippets: list[Snippet], lam: float, k: int) -> list[Candidate]:
    """Maximal Marginal Relevance using embedding cosine (or shingle Jaccard if no vectors)."""
    if len(cands) <= 1:
        return cands
    rel = np.array([c.final_score for c in cands], dtype=np.float64)
    if rel.max() > rel.min():
        rel = (rel - rel.min()) / (rel.max() - rel.min())
    else:
        rel = np.ones_like(rel)
    if vectors is not None and len(vectors):
        v = vectors[[c.idx for c in cands]]
        sim = v @ v.T
    else:
        sh = [shingles(snippets[c.idx].content) for c in cands]
        sim = np.array([[jaccard(a, b) for b in sh] for a in sh])
    selected: list[int] = []
    remaining = list(range(len(cands)))
    while remaining and len(selected) < k:
        if not selected:
            best = remaining[int(np.argmax(rel[remaining]))]
        else:
            red = sim[np.ix_(remaining, selected)].max(axis=1)
            scores = lam * rel[remaining] - (1 - lam) * red
            best = remaining[int(np.argmax(scores))]
        selected.append(best)
        remaining.remove(best)
    return [cands[i] for i in selected] + [cands[i] for i in remaining]
