"""CONTEXT view: where the snippet lives and how it connects.

::

    FILE auth/service.py
    CLASS AuthService
    FUNCTION authenticate(self, user)
    DOC Verify a user's JWT token.
    CALLS jwt.verify
"""

from __future__ import annotations

from codefusion.types import Snippet


def context_view(sn: Snippet, max_calls: int = 15) -> str:
    lines: list[str] = []
    if sn.file_path and sn.type != "document":
        lines.append(f"FILE {sn.file_path}")
    if sn.parent:
        lines.append(f"CLASS {sn.parent}")
    if sn.type in ("function", "method", "function_part"):
        lines.append(f"FUNCTION {sn.signature or sn.name}")
    elif sn.type == "class":
        lines.append(f"CLASS {sn.signature or sn.name}")
    elif sn.name and not sn.name.startswith("<"):
        lines.append(f"DEFINES {sn.name}")
    if sn.docstring:
        lines.append("DOC " + " ".join(sn.docstring.split())[:400])
    if sn.calls:
        lines.append("CALLS " + ", ".join(sn.calls[:max_calls]))
    if sn.comments:
        lines.append("COMMENTS " + " | ".join(sn.comments[:5])[:300])
    return "\n".join(lines)


def context_header(sn: Snippet) -> str:
    """Compact header prepended to raw code for dense 'context+raw' encoding."""
    bits = []
    if sn.file_path and sn.type != "document":
        bits.append(f"# file: {sn.file_path}")
    if sn.parent:
        bits.append(f"# class: {sn.parent}")
    return "\n".join(bits)
