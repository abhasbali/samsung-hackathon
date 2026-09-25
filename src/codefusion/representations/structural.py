"""STRUCTURAL view: a deterministic, AST-derived outline of what the code *does*.

::

    FUNCTION authenticate
    PARAM user
    CALL jwt.verify
    ACCESS user.token
    RETURN jwt.verify(user.token)
"""

from __future__ import annotations

from codefusion.types import Snippet

_KIND = {"function": "FUNCTION", "method": "METHOD", "class": "CLASS", "function_part": "FUNCTION",
         "module": "MODULE", "document": "PROGRAM"}


def structural_view(sn: Snippet, max_items: int = 40) -> str:
    lines: list[str] = []
    kind = _KIND.get(sn.type, sn.type.upper())
    if sn.name and not sn.name.startswith("<"):
        lines.append(f"{kind} {sn.name}")
    else:
        lines.append(kind)
    lines += [f"DECORATOR {d}" for d in sn.decorators[:5]]
    lines += [f"EXTENDS {b}" for b in sn.bases[:5]]
    lines += [f"PARAM {p}" for p in sn.parameters[:max_items]]
    lines += [f"IMPORT {i}" for i in sn.imports[:max_items]]
    lines += [f"DEFINES {d}" for d in sn.defines[:max_items] if d != sn.name]
    lines += [f"CALL {c}" for c in sn.calls[:max_items]]
    lines += [f"ACCESS {a}" for a in sn.attributes[:max_items]]
    lines += [f"RAISE {r}" for r in sn.raises[:10]]
    lines += [f"RETURN {r}" for r in sn.returns[:10]]
    return "\n".join(lines)
