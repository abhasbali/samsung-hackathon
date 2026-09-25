"""Builds and queries the code graph (a lightweight Code-Property-Graph-inspired structure).

Calls are linked *through symbol nodes* (``s:caller -CALLS-> y:name <-DEFINES- s:callee``), so
resolution stays correct when snippets are appended incrementally: a newly indexed definition
is automatically linked to every existing call site. Resolution confidence prefers the same
file, then files imported by the caller, then unique definitions.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np

from codefusion.graph.store import GraphStore
from codefusion.logging_utils import get_logger
from codefusion.parsing.identifiers import normalize_symbol
from codefusion.types import Snippet

log = get_logger(__name__)


def _last(name: str) -> str:
    for sep in ("::", ".", "/"):
        if sep in name:
            name = name.rsplit(sep, 1)[-1]
    return name


def _module_names(path: str) -> list[str]:
    """``src/app/auth.py`` -> [``src.app.auth``, ``app.auth``, ``auth``]; ``pkg/__init__.py`` -> ``pkg``."""
    p = path.replace("\\", "/")
    for ext in (".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".java", ".go"):
        if p.endswith(ext):
            p = p[: -len(ext)]
            break
    if p.endswith("/__init__") or p.endswith("/index"):
        p = p.rsplit("/", 1)[0]
    parts = [x for x in p.split("/") if x and x != "."]
    return [".".join(parts[i:]) for i in range(len(parts))]


class CodeGraph:
    def __init__(self, snippets: list[Snippet], backend: str = "rustworkx", max_fanout: int = 50) -> None:
        self.snippets = snippets  # shared, append-only list owned by the engine
        self.store = GraphStore(backend)
        # A name defined in more than ``max_fanout`` places cannot be resolved to one definition, so
        # call/caller expansion skips it (0 = no limit). Without this, corpora of independent programs
        # (every file defines main/solve) turn each hop into a scan of the whole corpus.
        self.max_fanout = max_fanout
        self.module_to_file: dict[str, str] = {}
        self.file_imports: dict[str, set[str]] = defaultdict(set)
        self._n = 0

    # ------------------------------------------------------------------ building
    def add_snippets(self, start_idx: int, end_idx: int | None = None) -> None:
        end_idx = len(self.snippets) if end_idx is None else end_idx
        st = self.store
        new_methods: list[int] = []
        for idx in range(start_idx, end_idx):
            sn = self.snippets[idx]
            skey = f"s:{idx}"
            st.add_node(skey, "snippet", type=sn.type, name=sn.name, file=sn.file_path, qualified_name=sn.qualified_name)
            if sn.repository:
                st.add_node(f"r:{sn.repository}", "repository", name=sn.repository)
            if sn.file_path:
                fkey = f"f:{sn.file_path}"
                st.add_node(fkey, "file", path=sn.file_path, language=sn.language)
                if sn.repository:
                    st.add_edge(f"r:{sn.repository}", fkey, "CONTAINS")
                st.add_edge(fkey, skey, "CONTAINS")
                for m in _module_names(sn.file_path):
                    self.module_to_file.setdefault(m, sn.file_path)
                for imp in sn.imports:
                    self.file_imports[sn.file_path].add(imp)
            if sn.commit:
                ckey = f"c:{sn.commit}"
                st.add_node(ckey, "commit", sha=sn.commit)
                st.add_edge(skey, ckey, "IN_COMMIT")
            defined = set(sn.defines)
            if sn.type in ("function", "method", "class") and sn.name:
                defined.add(sn.name)
            if sn.type in ("method", "class") and sn.qualified_name:
                defined.add(sn.qualified_name)
            for d in defined:
                self._link(skey, d, "DEFINES")
            called = {_last(c) for c in sn.calls}
            for c in called:
                self._link(skey, c, "CALLS")
            for r in set(sn.references) | {_last(a) for a in sn.attributes}:
                if len(r) >= 3 and r not in called and r not in defined:
                    self._link(skey, r, "REFERENCES")
            for b in sn.bases:
                self._link(skey, _last(b), "EXTENDS")
            if sn.type == "method" and sn.parent:
                new_methods.append(idx)
        # Class -> method containment and method overrides.
        for idx in new_methods:
            sn = self.snippets[idx]
            parent = normalize_symbol(_last(sn.parent or ""))
            same_file = [int(k[2:]) for k in self.store.successors(f"f:{sn.file_path}", ["CONTAINS"]) if k.startswith("s:")]
            for cidx in same_file:
                cls = self.snippets[cidx]
                if (cls.type == "class" and normalize_symbol(cls.name or "") == parent
                        and cls.start_line <= sn.start_line <= cls.end_line):
                    self.store.add_edge(f"s:{cidx}", f"s:{idx}", "CONTAINS")
                    for base in cls.bases:
                        for bidx in self._resolvable(self.definitions(_last(base))):
                            for midx in self._members(bidx):
                                if self.snippets[midx].name == sn.name:
                                    self.store.add_edge(f"s:{idx}", f"s:{midx}", "OVERRIDES")
        # File imports that resolve to indexed files.
        for f, imps in self.file_imports.items():
            for imp in imps:
                target = self._resolve_module(imp)
                if target and target != f:
                    self.store.add_edge(f"f:{f}", f"f:{target}", "IMPORTS")
        self._n = end_idx

    def add_lineage_edge(self, old_idx: int, new_idx: int) -> None:
        self.store.add_edge(f"s:{old_idx}", f"s:{new_idx}", "EVOLVED_TO")

    def _link(self, skey: str, symbol: str, etype: str) -> None:
        ns = normalize_symbol(symbol)
        if len(ns) < 2:
            return
        ykey = f"y:{ns}"
        if not self.store.has(ykey):
            self.store.add_node(ykey, "symbol", name=symbol)
        if etype == "DEFINES":
            self.store.add_edge(skey, ykey, "DEFINES")
        else:
            self.store.add_edge(skey, ykey, etype)

    def _resolve_module(self, imp: str) -> str | None:
        imp = imp.replace("/", ".").lstrip(".")
        parts = imp.split(".")
        for i in range(len(parts), 0, -1):
            f = self.module_to_file.get(".".join(parts[:i]))
            if f:
                return f
        return None

    def _members(self, class_idx: int) -> list[int]:
        return [int(k[2:]) for k in self.store.successors(f"s:{class_idx}", ["CONTAINS"]) if k.startswith("s:")]

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _alive(idx: int, mask: np.ndarray | None) -> bool:
        return mask is None or (idx < len(mask) and bool(mask[idx]))

    def _ambiguous(self, ykey: str) -> bool:
        return bool(self.max_fanout) and len(self.store.predecessors(ykey, ["DEFINES"])) > self.max_fanout

    def _resolvable(self, defs: list[int]) -> list[int]:
        return defs if not self.max_fanout or len(defs) <= self.max_fanout else []

    def _snips(self, keys: Iterable[str], mask: np.ndarray | None) -> list[int]:
        out = []
        for k in keys:
            if k.startswith("s:"):
                i = int(k[2:])
                if self._alive(i, mask):
                    out.append(i)
        return sorted(set(out))

    # ------------------------------------------------------------------ public API
    def definitions(self, symbol: str, mask: np.ndarray | None = None) -> list[int]:
        return self._snips(self.store.predecessors(f"y:{normalize_symbol(symbol)}", ["DEFINES"]), mask)

    def callers(self, symbol: str, mask: np.ndarray | None = None) -> list[int]:
        return self._snips(self.store.predecessors(f"y:{normalize_symbol(_last(symbol))}", ["CALLS"]), mask)

    def references(self, symbol: str, mask: np.ndarray | None = None) -> list[int]:
        return self._snips(self.store.predecessors(f"y:{normalize_symbol(_last(symbol))}", ["REFERENCES", "CALLS"]), mask)

    def callees_of(self, idx: int, mask: np.ndarray | None = None) -> list[tuple[int, float]]:
        """Definitions called by snippet ``idx`` with a resolution confidence in (0, 1]."""
        sn = self.snippets[idx]
        imported_files = {self._resolve_module(i) for i in self.file_imports.get(sn.file_path, ())} - {None}
        out: dict[int, float] = {}
        for ykey in self.store.successors(f"s:{idx}", ["CALLS"]):
            if self._ambiguous(ykey):
                continue
            defs = self._snips(self.store.predecessors(ykey, ["DEFINES"]), mask)
            defs = [d for d in defs if d != idx and self.snippets[d].type != "module"]
            for d in defs:
                f = self.snippets[d].file_path
                conf = 1.0 if f == sn.file_path else 0.9 if f in imported_files else (0.8 if len(defs) == 1 else 0.5 / len(defs))
                out[d] = max(out.get(d, 0.0), conf)
        return sorted(out.items(), key=lambda kv: (-kv[1], kv[0]))

    def callers_of(self, idx: int, mask: np.ndarray | None = None) -> list[int]:
        out: set[int] = set()
        for ykey in self.store.successors(f"s:{idx}", ["DEFINES"]):
            if self._ambiguous(ykey):
                continue
            out.update(self._snips(self.store.predecessors(ykey, ["CALLS"]), mask))
        out.discard(idx)
        return sorted(out)

    def callees(self, symbol: str, mask: np.ndarray | None = None) -> list[int]:
        out: set[int] = set()
        for d in self._resolvable(self.definitions(symbol, mask)):
            out.update(i for i, _ in self.callees_of(d, mask))
        return sorted(out)

    def imports(self, file_path: str) -> dict[str, list[str]]:
        return {
            "modules": sorted(self.file_imports.get(file_path, set())),
            "files": sorted(k[2:] for k in self.store.successors(f"f:{file_path}", ["IMPORTS"])),
        }

    def neighbors(self, idx: int, depth: int = 1, mask: np.ndarray | None = None) -> dict[int, tuple[int, str]]:
        """Snippets within ``depth`` call/containment/inheritance hops -> (distance, relation)."""
        result: dict[int, tuple[int, str]] = {}
        frontier = {idx: "seed"}
        for dist in range(1, depth + 1):
            nxt: dict[int, str] = {}
            for cur in frontier:
                for c, _ in self.callees_of(cur, mask):
                    nxt.setdefault(c, f"callee of {self.snippets[cur].name}")
                for c in self.callers_of(cur, mask):
                    nxt.setdefault(c, f"caller of {self.snippets[cur].name}")
                for k in self.store.successors(f"s:{cur}", ["CONTAINS", "OVERRIDES"]) + self.store.predecessors(f"s:{cur}", ["CONTAINS", "OVERRIDES"]):
                    if k.startswith("s:"):
                        j = int(k[2:])
                        if self._alive(j, mask):
                            nxt.setdefault(j, f"member/owner of {self.snippets[cur].name}")
            for j, rel in nxt.items():
                if j != idx and j not in result:
                    result[j] = (dist, rel)
            frontier = {j: r for j, r in nxt.items() if j not in frontier}
        return result

    def lineage(self, idx: int) -> list[int]:
        """Full version chain (oldest -> newest) through EVOLVED_TO edges."""
        back = [idx]
        cur = idx
        seen = {idx}
        while True:
            prev = [int(k[2:]) for k in self.store.predecessors(f"s:{cur}", ["EVOLVED_TO"]) if k.startswith("s:")]
            prev = [p for p in prev if p not in seen]
            if not prev:
                break
            cur = prev[0]
            seen.add(cur)
            back.append(cur)
        chain = list(reversed(back))
        cur = idx
        while True:
            nxt = [int(k[2:]) for k in self.store.successors(f"s:{cur}", ["EVOLVED_TO"]) if k.startswith("s:")]
            nxt = [n for n in nxt if n not in seen]
            if not nxt:
                break
            cur = nxt[0]
            seen.add(cur)
            chain.append(cur)
        return chain

    def call_chain(self, idx: int, depth: int = 3, mask: np.ndarray | None = None) -> dict:
        """Upstream callers and downstream callees around ``idx`` for visualisation."""

        def down(i: int, d: int, seen: set[int]) -> list[dict]:
            if d == 0:
                return []
            out = []
            for c, conf in self.callees_of(i, mask)[:8]:
                if c in seen:
                    continue
                out.append({"idx": c, "name": self.snippets[c].qualified_name, "file": self.snippets[c].file_path,
                            "confidence": round(conf, 2), "children": down(c, d - 1, seen | {c})})
            return out

        def up(i: int, d: int, seen: set[int]) -> list[dict]:
            if d == 0:
                return []
            out = []
            for c in self.callers_of(i, mask)[:8]:
                if c in seen:
                    continue
                out.append({"idx": c, "name": self.snippets[c].qualified_name, "file": self.snippets[c].file_path,
                            "parents": up(c, d - 1, seen | {c})})
            return out

        sn = self.snippets[idx]
        return {"idx": idx, "name": sn.qualified_name, "file": sn.file_path, "callers": up(idx, depth, {idx}), "callees": down(idx, depth, {idx})}

    def shortest_call_path(self, src: int, dst: int, max_depth: int = 4, mask: np.ndarray | None = None) -> list[int]:
        """BFS over resolved CALLS edges from ``src`` to ``dst`` (both snippet indices)."""
        from collections import deque

        prev: dict[int, int] = {src: -1}
        dq = deque([(src, 0)])
        while dq:
            cur, d = dq.popleft()
            if cur == dst:
                path = [cur]
                while prev[path[-1]] != -1:
                    path.append(prev[path[-1]])
                return list(reversed(path))
            if d >= max_depth:
                continue
            for c, _ in self.callees_of(cur, mask):
                if c not in prev:
                    prev[c] = cur
                    dq.append((c, d + 1))
        return []

    def stats(self) -> dict:
        s = self.store.stats()
        s["snippets"] = self._n
        return s
