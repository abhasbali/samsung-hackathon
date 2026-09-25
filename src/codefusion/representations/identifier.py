"""IDENTIFIER view: decomposed identifiers of the snippet, in reading order.

``def authenticate(user): return jwt.verify(user.token)`` ->
``authenticate user jwt verify token``
"""

from __future__ import annotations

from codefusion.parsing.identifiers import extract_identifiers, split_identifier
from codefusion.types import Snippet


def identifier_view(sn: Snippet, max_terms: int = 400) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for ident in extract_identifiers(sn.content):
        parts = split_identifier(ident)
        for term in [ident, *parts] if len(parts) > 1 else [ident]:
            t = term.lower()
            if len(t) < 2 or t in seen:
                continue
            seen.add(t)
            out.append(t)
            if len(out) >= max_terms:
                return " ".join(out)
    return " ".join(out)
