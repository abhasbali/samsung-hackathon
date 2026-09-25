"""Function/class lineage across versions.

Matching cascade between the snippets that disappeared (old) and appeared (new) in a change:

1. same ``symbol_key`` (file + qualified name)            -> ``modified``
2. same qualified name in a file git reports as renamed  -> ``file_renamed``
3. identical ``content_hash`` / ``ast_hash`` elsewhere   -> ``moved``
4. same kind + high content similarity                   -> ``renamed`` (name changed) or ``evolved``

Each new version inherits the ``lineage_id`` of its predecessor, so every version of a function
shares one id across commits and renames (``authenticate@A -> authenticate@B -> verify_user@C``).
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

from codefusion.retrieval.diversity import jaccard, shingles
from codefusion.types import Snippet, sha1_hex


@dataclass
class LineageEdge:
    old_idx: int
    new_idx: int
    relation: str
    similarity: float

    def as_tuple(self) -> tuple[int, int, str, float]:
        return (self.old_idx, self.new_idx, self.relation, round(self.similarity, 4))


def new_lineage_id(sn: Snippet) -> str:
    return "L" + sha1_hex(f"{sn.repository}|{sn.symbol_key}|{sn.commit}", 12)


def similarity(a: Snippet, b: Snippet) -> float:
    ja = jaccard(shingles(a.content, 2), shingles(b.content, 2))
    ra = difflib.SequenceMatcher(None, a.content[:4000], b.content[:4000], autojunk=False).quick_ratio()
    shape = 0.1 if a.shape_hash and a.shape_hash == b.shape_hash else 0.0
    return min(1.0, 0.5 * ja + 0.5 * ra + shape)


def _qualified_key(sn: Snippet, path: str) -> str:
    return f"{path}::{sn.qualified_name}"


class LineageMatcher:
    def __init__(self, min_similarity: float = 0.6) -> None:
        self.min_similarity = min_similarity

    def match(
        self,
        old: list[tuple[int, Snippet]],
        new: list[tuple[int, Snippet]],
        renames: dict[str, str] | None = None,
    ) -> list[LineageEdge]:
        renames = renames or {}
        edges: list[LineageEdge] = []
        old_left = {i: s for i, s in old}
        new_left = {i: s for i, s in new}

        def take(oi: int, ni: int, rel: str, sim: float) -> None:
            edges.append(LineageEdge(oi, ni, rel, sim))
            old_left.pop(oi, None)
            new_left.pop(ni, None)

        # 1-2: same symbol key (directly or through a git-detected file rename)
        by_key: dict[str, int] = {}
        for oi, s in old_left.items():
            by_key.setdefault(s.symbol_key, oi)
        for ni, s in list(new_left.items()):
            oi = by_key.get(s.symbol_key)
            if oi is not None and oi in old_left:
                take(oi, ni, "modified", similarity(old_left[oi], s))
                continue
            old_path = renames.get(s.file_path)
            if old_path:
                oi = by_key.get(_qualified_key(s, old_path))
                if oi is not None and oi in old_left:
                    take(oi, ni, "file_renamed", similarity(old_left[oi], s))
        # 3: identical content / AST moved elsewhere
        by_hash: dict[str, int] = {}
        for oi, s in old_left.items():
            by_hash.setdefault(s.content_hash, oi)
            if s.ast_hash:
                by_hash.setdefault("ast:" + s.ast_hash, oi)
        for ni, s in list(new_left.items()):
            oi = by_hash.get(s.content_hash)
            if oi is None and s.ast_hash:
                oi = by_hash.get("ast:" + s.ast_hash)
            if oi is not None and oi in old_left:
                take(oi, ni, "moved", 1.0)
        # 4: similarity among remaining definitions of the same kind (greedy best-first)
        pairs: list[tuple[float, int, int]] = []
        for ni, ns in new_left.items():
            if ns.type == "module":
                continue
            for oi, os_ in old_left.items():
                if os_.type != ns.type or os_.language != ns.language:
                    continue
                sim = similarity(os_, ns)
                if sim >= self.min_similarity:
                    pairs.append((sim, oi, ni))
        for sim, oi, ni in sorted(pairs, key=lambda t: (-t[0], t[1], t[2])):
            if oi in old_left and ni in new_left:
                rel = "renamed" if old_left[oi].name != new_left[ni].name else "evolved"
                take(oi, ni, rel, sim)
        return edges
