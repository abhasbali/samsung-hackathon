"""Graph retrieval and transparent structural scoring.

Graph retrieval never walks the whole graph: it starts from (a) intent-targeted symbols
("Who calls X" -> callers of X) and (b) the top seeds of first-stage retrieval, then expands
1–2 hops. Structural scoring adds small, explained boosts on top of the fused ranking.
"""

from __future__ import annotations

import re

import numpy as np

from codefusion.config.schema import GraphConfig, StructuralConfig
from codefusion.graph.builder import CodeGraph
from codefusion.parsing.identifiers import normalize_symbol
from codefusion.types import Candidate, Hit


def _resolve_targets(graph: CodeGraph, targets: list[str], mask: np.ndarray | None) -> list[str]:
    """Keep query targets that exist as symbols (defined or called) in the graph."""
    out = []
    for t in targets:
        last = re.split(r"[.:/]", t)[-1]
        if graph.definitions(last, mask) or graph.callers(last, mask):
            out.append(last)
    return out


class GraphRetriever:
    name = "graph"

    def __init__(self, graph: CodeGraph, cfg: GraphConfig) -> None:
        self.graph = graph
        self.cfg = cfg

    def search(self, query, seeds: list[Hit], mask: np.ndarray | None = None) -> list[Hit]:
        g = self.graph
        scores: dict[int, float] = {}
        evidence: dict[int, list[str]] = {}

        def add(i: int, s: float, why: str) -> None:
            if s <= 0:
                return
            scores[i] = scores.get(i, 0.0) + s
            ev = evidence.setdefault(i, [])
            if len(ev) < 5 and why not in ev:
                ev.append(why)

        targets = _resolve_targets(g, query.target_symbols or query.identifiers, mask)
        intent = query.intent
        for t in targets:
            if intent == "CALLER":
                for i in g.callers(t, mask):
                    add(i, 2.0, f"calls {t}")
            elif intent == "CALLEE":
                for i in g.callees(t, mask):
                    add(i, 2.0, f"called by {t}")
                for i in g.definitions(t, mask):
                    add(i, 1.0, f"defines {t}")
            elif intent in ("DEFINITION", "CONFIGURATION"):
                for i in g.definitions(t, mask):
                    add(i, 2.0, f"defines {t}")
            elif intent in ("USAGE", "DEPENDENCY"):
                for i in g.references(t, mask):
                    add(i, 1.5, f"uses {t}")
            elif intent == "DATA_FLOW":
                for d in g.definitions(t, mask):
                    add(d, 1.0, f"defines {t}")
                    for i in g.callers_of(d, mask):
                        add(i, 1.2, f"passes data into {t}")
                    for i, conf in g.callees_of(d, mask):
                        add(i, 1.0 * conf, f"receives data from {t}")
            else:
                for i in g.definitions(t, mask):
                    add(i, 1.0, f"defines {t}")
        # Seed expansion (1..depth hops) from first-stage results.
        for rank, h in enumerate(seeds[: self.cfg.seeds], start=1):
            seed_w = 1.0 / rank
            for j, (dist, rel) in g.neighbors(h.idx, self.cfg.depth, mask).items():
                add(j, seed_w * (self.cfg.decay ** dist), rel)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[: self.cfg.top_k]
        return [Hit(idx=i, score=s, rank=r, source=self.name, evidence=evidence.get(i, [])) for r, (i, s) in enumerate(ranked, start=1)]


class StructuralScorer:
    """Adds configurable boosts (in units of a rank-1 RRF contribution) with evidence strings."""

    def __init__(self, graph: CodeGraph | None, cfg: StructuralConfig, rrf_k: int = 60) -> None:
        self.graph = graph
        self.cfg = cfg
        self.unit = 1.0 / (rrf_k + 1)

    def apply(self, query, cands: list[Candidate], snippets, mask: np.ndarray | None = None) -> None:
        if not cands:
            return
        cfg = self.cfg
        qsyms = {normalize_symbol(re.split(r"[.:/]", t)[-1]) for t in (query.target_symbols + query.identifiers)}
        qsyms = {s for s in qsyms if len(s) >= 3}
        qwords = {w.lower() for w in re.findall(r"[A-Za-z_]\w+", query.text)}
        top = [c.idx for c in cands[: cfg.top_n_for_neighbours]]
        top_callees: set[int] = set()
        top_callers: set[int] = set()
        if self.graph is not None:
            for t in top:
                top_callees.update(i for i, _ in self.graph.callees_of(t, mask))
                top_callers.update(self.graph.callers_of(t, mask))
        # Snippets defining two different queried symbols, for call-path evidence.
        target_defs: list[int] = []
        if self.graph is not None and len(qsyms) >= 2:
            for s in list(qsyms)[:4]:
                d = self.graph.definitions(s, mask)
                if d:
                    target_defs.append(d[0])
        path_nodes: set[int] = set()
        if self.graph is not None and len(target_defs) >= 2:
            for a in target_defs:
                for b in target_defs:
                    if a != b:
                        path_nodes.update(self.graph.shortest_call_path(a, b, 4, mask)[1:-1])
        # Intent-specific relations to the *queried* symbols.
        intent = getattr(query, "intent", "GENERAL")
        target_callers: dict[int, str] = {}
        target_callees: dict[int, str] = {}
        if self.graph is not None and intent in ("CALLER", "CALLEE", "DATA_FLOW"):
            for t in (query.target_symbols or query.identifiers)[:4]:
                last = re.split(r"[.:/]", t)[-1]
                if intent in ("CALLER", "DATA_FLOW"):
                    for i in self.graph.callers(last, mask):
                        target_callers.setdefault(i, last)
                if intent in ("CALLEE", "DATA_FLOW"):
                    for i in self.graph.callees(last, mask):
                        target_callees.setdefault(i, last)
        flow_scale = 0.5 if intent == "DATA_FLOW" else 1.0
        def_w = cfg.defines_symbol * {"CALLER": 0.0, "CALLEE": 0.5, "DEFINITION": 1.5, "CONFIGURATION": 1.5}.get(intent, 1.0)
        # For CALLEE queries a snippet that *calls* the target is a caller, i.e. the wrong direction.
        call_w = cfg.calls_symbol * {"CALLER": 2.0, "CALLEE": 0.0, "DEFINITION": 0.3, "CONFIGURATION": 0.3}.get(intent, 1.0)
        for c in cands:
            sn = snippets[c.idx]
            boost = 0.0
            ev: list[str] = []
            defined = {normalize_symbol(x) for x in [sn.name, *sn.defines] if x}
            called = {normalize_symbol(re.split(r"[.:/]", x)[-1]) for x in sn.calls}
            refs = {normalize_symbol(x) for x in sn.references}
            hit_def = qsyms & defined
            if hit_def and def_w:
                boost += def_w
                ev.append(f"defines queried symbol {sorted(hit_def)[0]}")
            hit_call = (qsyms & called) - hit_def
            if hit_call:
                boost += call_w
                ev.append(f"calls queried symbol {sorted(hit_call)[0]}")
            if c.idx in target_callers:
                boost += cfg.caller_of_target * flow_scale
                ev.append(f"caller of {target_callers[c.idx]}")
            if c.idx in target_callees:
                boost += cfg.callee_of_target * flow_scale
                ev.append(f"callee of {target_callees[c.idx]}")
            elif (qsyms & refs) - hit_def:
                boost += cfg.references_symbol
                ev.append("references queried symbol")
            if c.idx in top_callees and c.idx not in top:
                boost += cfg.callee_of_top
                ev.append("callee of a top result")
            if c.idx in top_callers and c.idx not in top:
                boost += cfg.caller_of_top
                ev.append("caller of a top result")
            imp_heads = {i.replace("/", ".").split(".")[0].lower() for i in sn.imports}
            imp_hit = imp_heads & qwords
            if imp_hit:
                boost += cfg.import_match
                ev.append(f"imports {sorted(imp_hit)[0]}")
            if c.idx in path_nodes:
                boost += cfg.call_path
                ev.append("on call path between queried symbols")
            if boost:
                c.structural_boost = boost * self.unit
                c.structural_evidence.extend(ev)
                c.final_score += c.structural_boost
