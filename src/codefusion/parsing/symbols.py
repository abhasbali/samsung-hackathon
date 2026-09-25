"""Symbol extraction: turns snippet facts into (symbol, role) records.

Roles and their default weights for symbol retrieval:

====== ======================================== ======
role   meaning                                  weight
====== ======================================== ======
def    the snippet defines the symbol           3.0
call   the snippet calls the symbol             1.5
import the snippet imports the symbol           1.2
attr   the snippet accesses ``obj.symbol``      1.0
ref    the identifier appears in the snippet    0.5
param  the symbol is a parameter               0.3
====== ======================================== ======

Optional **SCIP** enrichment: if a ``index.scip`` file exists and the ``scip`` CLI is on
``PATH``, precise cross-file definitions/references are merged in (see :func:`load_scip_occurrences`).
Everything works without it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from codefusion.logging_utils import get_logger
from codefusion.parsing.identifiers import normalize_symbol
from codefusion.types import Snippet

log = get_logger(__name__)

DEFAULT_ROLE_WEIGHTS: dict[str, float] = {"def": 3.0, "call": 1.5, "import": 1.2, "attr": 1.0, "ref": 0.5, "param": 0.3}


@dataclass(frozen=True)
class SymbolRecord:
    symbol: str  # as written (last segment for dotted names is also emitted separately)
    normalized: str
    role: str


def _segments(name: str) -> list[str]:
    """``jwt.verify`` -> [``jwt.verify``, ``verify``]; ``a/b/c`` -> [``a/b/c``, ``c``]."""
    out = [name]
    for sep in (".", "/", "::"):
        if sep in name:
            last = name.rsplit(sep, 1)[-1]
            if last:
                out.append(last)
    return out


def extract_symbols(sn: Snippet) -> list[SymbolRecord]:
    recs: list[SymbolRecord] = []

    def add(names: Iterable[str], role: str) -> None:
        for n in names:
            for s in _segments(n):
                ns = normalize_symbol(s)
                if len(ns) >= 2:
                    recs.append(SymbolRecord(s, ns, role))

    defined = list(sn.defines) or ([sn.name] if sn.name and not sn.name.startswith("<") else [])
    if sn.type in ("function", "method", "class", "function_part") and sn.name:
        defined = [sn.name, *defined]
    add(defined, "def")
    if sn.qualified_name and sn.type in ("method", "class"):
        add([sn.qualified_name], "def")
    add(sn.calls, "call")
    add(sn.imports, "import")
    add(sn.attributes, "attr")
    add(sn.references, "ref")
    add(sn.parameters, "param")
    add(sn.extra.get("members", []), "ref")  # a class *contains* its methods; it does not define them
    add(sn.extra.get("scip_definitions", []), "def")
    add(sn.extra.get("scip_references", []), "ref")
    # Keep the strongest role per normalized symbol.
    best: dict[str, SymbolRecord] = {}
    for r in recs:
        cur = best.get(r.normalized)
        if cur is None or DEFAULT_ROLE_WEIGHTS[r.role] > DEFAULT_ROLE_WEIGHTS[cur.role]:
            best[r.normalized] = r
    return list(best.values())


# ---------------------------------------------------------------------------- optional SCIP
def scip_available() -> bool:
    return shutil.which("scip") is not None


def load_scip_occurrences(index_path: str | Path) -> dict[str, list[dict]]:
    """Return ``{relative_path: [{"symbol", "line", "is_definition"}]}`` from a SCIP index.

    Uses ``scip print --json``. Returns ``{}`` (and logs) if SCIP tooling is absent.
    """
    index_path = Path(index_path)
    if not index_path.exists() or not scip_available():
        log.info("SCIP enrichment skipped", extra={"data": {"index": str(index_path), "cli": scip_available()}})
        return {}
    try:
        out = subprocess.run(["scip", "print", "--json", str(index_path)], capture_output=True, text=True, timeout=600, check=True)
        data = json.loads(out.stdout)
    except Exception as exc:  # noqa: BLE001
        log.warning("SCIP parsing failed", extra={"data": {"error": repr(exc)}})
        return {}
    result: dict[str, list[dict]] = {}
    for doc in data.get("documents", []):
        occs = []
        for occ in doc.get("occurrences", []):
            rng = occ.get("range") or [0]
            sym = occ.get("symbol", "")
            if not sym or sym.startswith("local "):
                continue
            occs.append({"symbol": sym, "line": int(rng[0]) + 1, "is_definition": bool(int(occ.get("symbol_roles", occ.get("symbolRoles", 0))) & 1)})
        result[doc.get("relative_path", doc.get("relativePath", ""))] = occs
    return result


def scip_display_name(symbol: str) -> str:
    """``scip-python python pkg 1.0 `mod.auth`/AuthService#login().`` -> ``login``."""
    tail = symbol.rstrip(".").split("/")[-1].split("#")[-1]
    return tail.replace("().", "").replace("()", "").strip("`.:")


def enrich_with_scip(snippets: list[Snippet], occurrences: dict[str, list[dict]]) -> int:
    """Attach SCIP definitions/references to snippets by file + line range. Returns #snippets enriched."""
    if not occurrences:
        return 0
    n = 0
    for sn in snippets:
        occs = occurrences.get(sn.file_path)
        if not occs:
            continue
        defs, refs = [], []
        for o in occs:
            if sn.start_line <= o["line"] <= sn.end_line:
                (defs if o["is_definition"] else refs).append(scip_display_name(o["symbol"]))
        if defs or refs:
            sn.extra["scip_definitions"] = sorted(set(defs))
            sn.extra["scip_references"] = sorted(set(refs))
            n += 1
    return n
