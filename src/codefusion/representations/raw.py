"""RAW view: the code itself, lightly normalised."""

from __future__ import annotations

import re

from codefusion.types import Snippet

_TRAILING_WS = re.compile(r"[ \t]+$", re.M)
_MANY_BLANKS = re.compile(r"\n{3,}")


def normalize_code(text: str, strip_comments: bool = False, language: str = "python") -> str:
    text = text.replace("\r\n", "\n").replace("\t", "    ")
    if strip_comments:
        if language == "python":
            text = re.sub(r"(?m)^\s*#.*$", "", text)
        else:
            text = re.sub(r"(?m)^\s*//.*$", "", text)
            text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = _TRAILING_WS.sub("", text)
    text = _MANY_BLANKS.sub("\n\n", text)
    return text.strip("\n")


def raw_view(sn: Snippet, strip_comments: bool = False, include_signature_for_parts: bool = True) -> str:
    body = normalize_code(sn.content, strip_comments, sn.language)
    if include_signature_for_parts and sn.type == "function_part" and sn.extra.get("part", 1) > 1 and sn.signature:
        body = f"{sn.signature}\n    # ... (part {sn.extra.get('part')}/{sn.extra.get('parts')})\n{body}"
    return body
