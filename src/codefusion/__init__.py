"""CodeFusion: a code-intelligence retrieval engine.

CodeFusion ranks code snippets by relevance to a natural-language query by
combining AST-aware chunking, code-specific dense retrieval, identifier-aware
BM25, a symbol index, a lightweight code graph, query-intent-adaptive fusion,
optional reranking and version/lineage-aware indexing.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
