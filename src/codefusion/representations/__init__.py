"""Multi-view representations of a snippet (all deterministic, no LLM)."""

from __future__ import annotations

from codefusion.representations.context import context_header, context_view
from codefusion.representations.identifier import identifier_view
from codefusion.representations.raw import normalize_code, raw_view
from codefusion.representations.structural import structural_view
from codefusion.types import Snippet

VIEW_NAMES = ("raw", "identifiers", "structural", "context", "docstring")


def build_views(sn: Snippet) -> dict[str, str]:
    """Return every searchable view for a snippet."""
    return {
        "raw": raw_view(sn),
        "identifiers": identifier_view(sn),
        "structural": structural_view(sn),
        "context": context_view(sn),
        "docstring": " ".join([sn.docstring, *sn.comments, *sn.string_literals[:10]]).strip(),
    }


def dense_text(sn: Snippet, view: str = "raw", max_chars: int = 12000) -> str:
    """Text fed to the document encoder for a given dense view setting."""
    if view == "raw" and sn.type == "document":
        # Whole benchmark documents are encoded verbatim (identical to what MTEB feeds an encoder),
        # which also lets the dense-only and hybrid evaluation paths share one embedding cache.
        text = sn.content
    elif view == "raw":
        text = raw_view(sn)
    elif view == "context+raw":
        header = context_header(sn)
        text = f"{header}\n{raw_view(sn)}" if header else raw_view(sn)
    elif view == "structural":
        text = structural_view(sn)
    elif view == "raw+structural":
        text = f"{raw_view(sn)}\n\n{structural_view(sn)}"
    else:
        raise ValueError(f"unknown dense view {view!r}")
    return text[:max_chars]


__all__ = [
    "VIEW_NAMES",
    "build_views",
    "context_header",
    "context_view",
    "dense_text",
    "identifier_view",
    "normalize_code",
    "raw_view",
    "structural_view",
]
