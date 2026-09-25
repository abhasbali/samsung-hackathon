"""Core data types shared across CodeFusion."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any


def sha1_hex(text: str, n: int = 16) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:n]


@dataclass
class Snippet:
    """A semantically meaningful unit of code (function, method, class, module block).

    ``snippet_id`` identifies this *version* of the unit. ``symbol_key`` is stable across
    versions (``file_path::qualified_name``) and ``lineage_id`` follows the unit across renames.
    """

    snippet_id: str
    content: str
    file_path: str = ""
    language: str = "python"
    type: str = "module"  # function | method | class | module | function_part | document
    name: str = ""
    qualified_name: str = ""
    parent: str | None = None
    repository: str = ""
    commit: str | None = None
    start_line: int = 1
    end_line: int = 1
    signature: str = ""
    docstring: str = ""
    parameters: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    attributes: list[str] = field(default_factory=list)  # attribute accesses: user.token
    returns: list[str] = field(default_factory=list)
    raises: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    bases: list[str] = field(default_factory=list)
    defines: list[str] = field(default_factory=list)  # symbols defined here (module constants etc.)
    comments: list[str] = field(default_factory=list)
    string_literals: list[str] = field(default_factory=list)
    content_hash: str = ""
    ast_hash: str = ""
    shape_hash: str = ""  # identifier-insensitive structure hash (rename detection)
    lineage_id: str = ""
    commits: list[str] = field(default_factory=list)  # all commits in which this version exists
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = sha1_hex(self.content, 20)
        if not self.qualified_name:
            self.qualified_name = f"{self.parent}.{self.name}" if self.parent else self.name

    @property
    def symbol_key(self) -> str:
        return f"{self.file_path}::{self.qualified_name or self.name or self.start_line}"

    @property
    def lines(self) -> str:
        return f"{self.start_line}-{self.end_line}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Snippet:
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Hit:
    """A single retriever's result for one document (index into the snippet store)."""

    idx: int
    score: float
    rank: int  # 1-based
    source: str = ""
    evidence: list[str] = field(default_factory=list)


@dataclass
class Candidate:
    """A fused candidate carrying full retrieval provenance for explainability."""

    idx: int
    fused_score: float = 0.0
    ranks: dict[str, int] = field(default_factory=dict)
    raw_scores: dict[str, float] = field(default_factory=dict)
    contributions: dict[str, float] = field(default_factory=dict)
    reranker_score: float | None = None
    structural_boost: float = 0.0
    structural_evidence: list[str] = field(default_factory=list)
    graph_evidence: list[str] = field(default_factory=list)
    final_score: float = 0.0
    fused_rank: int | None = None
    dropped_reason: str | None = None

    def provenance(self) -> dict[str, Any]:
        out: dict[str, Any] = {f"{k}_rank": v for k, v in self.ranks.items()}
        out.update({f"{k}_score": round(v, 6) for k, v in self.raw_scores.items()})
        out["rrf_score"] = round(self.fused_score, 6)
        out["fused_rank"] = self.fused_rank
        out["contributions"] = {k: round(v, 6) for k, v in self.contributions.items()}
        out["reranker_score"] = None if self.reranker_score is None else round(self.reranker_score, 6)
        out["structural_boost"] = round(self.structural_boost, 6)
        return out
