"""Optional Neo4j export for visualising code relationships and version lineage.

Never required for indexing, search or evaluation. Requires ``pip install neo4j`` and a
running server (see ``docker-compose.yml`` profile ``neo4j``). Credentials come from
``NEO4J_URI`` / ``NEO4J_USER`` / ``NEO4J_PASSWORD``.
"""

from __future__ import annotations

import os
from typing import Any

from codefusion.graph.builder import CodeGraph
from codefusion.logging_utils import get_logger

log = get_logger(__name__)


class Neo4jUnavailable(RuntimeError):
    pass


class Neo4jExporter:
    def __init__(self, uri: str | None = None, user: str | None = None, password: str | None = None) -> None:
        try:
            from neo4j import GraphDatabase  # type: ignore
        except ImportError as exc:
            raise Neo4jUnavailable("neo4j driver not installed: pip install neo4j") from exc
        uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        user = user or os.environ.get("NEO4J_USER", "neo4j")
        password = password or os.environ.get("NEO4J_PASSWORD")
        if not password:
            raise Neo4jUnavailable("set NEO4J_PASSWORD to export to Neo4j")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity()

    def export(self, graph: CodeGraph, batch: int = 500) -> dict[str, int]:
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        st = graph.store
        keys = [f"s:{i}" for i in range(len(graph.snippets))]
        seen: set[str] = set()
        for k in keys:
            for key in [k] + [n for n, _ in st.edges_out(k)]:
                if key in seen or not st.has(key):
                    continue
                seen.add(key)
                a = st.attrs(key)
                nodes.append({"key": key, "kind": a.get("kind", "node"), "name": str(a.get("name", a.get("path", key)))})
            for dst, types in st.edges_out(k):
                for t in types:
                    edges.append({"src": k, "dst": dst, "type": t})
        with self.driver.session() as s:
            s.run("CREATE CONSTRAINT cf_key IF NOT EXISTS FOR (n:CFNode) REQUIRE n.key IS UNIQUE")
            for i in range(0, len(nodes), batch):
                s.run("UNWIND $rows AS r MERGE (n:CFNode {key: r.key}) SET n.kind = r.kind, n.name = r.name", rows=nodes[i : i + batch])
            for i in range(0, len(edges), batch):
                s.run(
                    "UNWIND $rows AS r MATCH (a:CFNode {key: r.src}), (b:CFNode {key: r.dst}) "
                    "MERGE (a)-[e:REL {type: r.type}]->(b)",
                    rows=edges[i : i + batch],
                )
        log.info("exported graph to neo4j", extra={"data": {"nodes": len(nodes), "edges": len(edges)}})
        return {"nodes": len(nodes), "edges": len(edges)}

    def close(self) -> None:
        self.driver.close()
