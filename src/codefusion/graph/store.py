"""In-memory property graph with a rustworkx backend (NetworkX fallback).

Node keys (strings):

* ``r:<repo>`` repository · ``f:<path>`` file · ``s:<idx>`` snippet version (index into the
  snippet store) · ``y:<normalized symbol>`` symbol · ``c:<sha>`` commit

Edge types: CONTAINS, DEFINES, CALLS, REFERENCES, IMPORTS, EXTENDS, IMPLEMENTS, OVERRIDES,
EVOLVED_TO, IN_COMMIT.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Any

from codefusion.logging_utils import get_logger

log = get_logger(__name__)

EDGE_TYPES = ("CONTAINS", "DEFINES", "CALLS", "REFERENCES", "IMPORTS", "EXTENDS", "IMPLEMENTS", "OVERRIDES", "EVOLVED_TO", "IN_COMMIT")


class _RxBackend:
    name = "rustworkx"

    def __init__(self) -> None:
        import rustworkx as rx

        self.rx = rx
        self.g = rx.PyDiGraph(multigraph=False)
        self.key_to_id: dict[str, int] = {}

    def add_node(self, key: str, attrs: dict[str, Any]) -> int:
        nid = self.key_to_id.get(key)
        if nid is None:
            nid = self.g.add_node({"key": key, **attrs})
            self.key_to_id[key] = nid
        else:
            self.g[nid].update(attrs)
        return nid

    def add_edge(self, a: str, b: str, etype: str) -> None:
        ia, ib = self.key_to_id[a], self.key_to_id[b]
        existing = self.g.get_edge_data(ia, ib) if self.g.has_edge(ia, ib) else None
        if existing is None:
            self.g.add_edge(ia, ib, {etype})
        else:
            existing.add(etype)

    def has(self, key: str) -> bool:
        return key in self.key_to_id

    def attrs(self, key: str) -> dict[str, Any]:
        return self.g[self.key_to_id[key]]

    def out_edges(self, key: str) -> Iterable[tuple[str, set[str]]]:
        nid = self.key_to_id.get(key)
        if nid is None:
            return []
        return [(self.g[t]["key"], data) for _, t, data in self.g.out_edges(nid)]

    def in_edges(self, key: str) -> Iterable[tuple[str, set[str]]]:
        nid = self.key_to_id.get(key)
        if nid is None:
            return []
        return [(self.g[s]["key"], data) for s, _, data in self.g.in_edges(nid)]

    def remove_node(self, key: str) -> None:
        nid = self.key_to_id.pop(key, None)
        if nid is not None:
            self.g.remove_node(nid)

    def counts(self) -> tuple[int, int]:
        return self.g.num_nodes(), self.g.num_edges()


class _NxBackend:
    name = "networkx"

    def __init__(self) -> None:
        import networkx as nx

        self.g = nx.DiGraph()

    def add_node(self, key: str, attrs: dict[str, Any]) -> int:
        if key in self.g:
            self.g.nodes[key].update(attrs)
        else:
            self.g.add_node(key, key=key, **attrs)
        return 0

    def add_edge(self, a: str, b: str, etype: str) -> None:
        if self.g.has_edge(a, b):
            self.g[a][b]["types"].add(etype)
        else:
            self.g.add_edge(a, b, types={etype})

    def has(self, key: str) -> bool:
        return key in self.g

    def attrs(self, key: str) -> dict[str, Any]:
        return self.g.nodes[key]

    def out_edges(self, key: str) -> Iterable[tuple[str, set[str]]]:
        if key not in self.g:
            return []
        return [(t, d["types"]) for _, t, d in self.g.out_edges(key, data=True)]

    def in_edges(self, key: str) -> Iterable[tuple[str, set[str]]]:
        if key not in self.g:
            return []
        return [(s, d["types"]) for s, _, d in self.g.in_edges(key, data=True)]

    def remove_node(self, key: str) -> None:
        if key in self.g:
            self.g.remove_node(key)

    def counts(self) -> tuple[int, int]:
        return self.g.number_of_nodes(), self.g.number_of_edges()


class GraphStore:
    """Backend-neutral property graph used by retrieval, the API and the Neo4j exporter."""

    def __init__(self, backend: str = "rustworkx") -> None:
        if backend == "rustworkx":
            try:
                self._b: _RxBackend | _NxBackend = _RxBackend()
            except Exception as exc:  # noqa: BLE001
                log.warning("rustworkx unavailable, using networkx", extra={"data": {"error": repr(exc)}})
                self._b = _NxBackend()
        else:
            self._b = _NxBackend()

    @property
    def backend(self) -> str:
        return self._b.name

    def add_node(self, key: str, kind: str, **attrs: Any) -> None:
        self._b.add_node(key, {"kind": kind, **attrs})

    def add_edge(self, a: str, b: str, etype: str) -> None:
        if etype not in EDGE_TYPES:
            raise ValueError(f"unknown edge type {etype}")
        self._b.add_edge(a, b, etype)

    def has(self, key: str) -> bool:
        return self._b.has(key)

    def attrs(self, key: str) -> dict[str, Any]:
        return self._b.attrs(key)

    def successors(self, key: str, etypes: Iterable[str] | None = None) -> list[str]:
        want = set(etypes) if etypes else None
        return [k for k, ts in self._b.out_edges(key) if want is None or ts & want]

    def predecessors(self, key: str, etypes: Iterable[str] | None = None) -> list[str]:
        want = set(etypes) if etypes else None
        return [k for k, ts in self._b.in_edges(key) if want is None or ts & want]

    def edges_out(self, key: str) -> list[tuple[str, set[str]]]:
        return list(self._b.out_edges(key))

    def edges_in(self, key: str) -> list[tuple[str, set[str]]]:
        return list(self._b.in_edges(key))

    def remove_node(self, key: str) -> None:
        self._b.remove_node(key)

    def stats(self) -> dict[str, Any]:
        n, e = self._b.counts()
        return {"backend": self.backend, "nodes": n, "edges": e}

    def bfs(self, start: str, depth: int, etypes: Iterable[str] | None = None, both: bool = True) -> dict[str, int]:
        """Distances of nodes within ``depth`` hops of ``start``."""
        want = set(etypes) if etypes else None
        dist = {start: 0}
        dq = deque([start])
        while dq:
            cur = dq.popleft()
            if dist[cur] >= depth:
                continue
            nbrs = [k for k, ts in self._b.out_edges(cur) if want is None or ts & want]
            if both:
                nbrs += [k for k, ts in self._b.in_edges(cur) if want is None or ts & want]
            for nb in nbrs:
                if nb not in dist:
                    dist[nb] = dist[cur] + 1
                    dq.append(nb)
        return dist
