"""AST-aware semantic chunking.

Units are functions, methods, classes and module-level blocks — never arbitrary token
windows. Two size fallbacks keep semantic boundaries intact:

* **Large functions** are split between *top-level body statements* into ``function_part``
  snippets; every part keeps the parent signature in its metadata/context view.
* **Classes with methods** become a *skeleton* (header, docstring, class-level statements
  and method signatures) because every method is already its own snippet — this avoids
  indexing each method body twice.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from codefusion.logging_utils import get_logger
from codefusion.parsing.fallback import _regex_info, parse_python_ast, parse_regex
from codefusion.parsing.tree_sitter_parser import CodeUnit, ParsedFile, detect_language, get_extractor
from codefusion.types import Snippet, sha1_hex

log = get_logger(__name__)

_KEYWORDS = frozenset(
    "False None True and as assert async await break class continue def del elif else except finally for from "
    "global if import in is lambda nonlocal not or pass raise return try while with yield self cls this new var "
    "let const function func return public private static void int str float bool dict list set tuple len range "
    "print".split()
)


@dataclass
class ChunkerConfig:
    max_chunk_lines: int = 150
    max_module_block_lines: int = 80
    min_part_lines: int = 5
    class_skeletons: bool = True
    include_module_blocks: bool = True
    prefer_tree_sitter: bool = True


def parse_source(source: str, language: str, prefer_tree_sitter: bool = True) -> ParsedFile:
    """Parse with tree-sitter when available, else stdlib ``ast`` (Python), else regex."""
    if prefer_tree_sitter:
        ext = get_extractor(language)
        if ext is not None:
            try:
                return ext.parse(source)
            except Exception as exc:  # noqa: BLE001
                log.warning("tree-sitter parse failed; falling back", extra={"data": {"error": repr(exc)}})
    if language == "python":
        try:
            return parse_python_ast(source)
        except (SyntaxError, ValueError, RecursionError):
            pass
    return parse_regex(source, language)


def _clean_refs(refs: list[str], params: list[str], limit: int = 200) -> list[str]:
    pset = set(params)
    return [r for r in refs if r not in _KEYWORDS and r not in pset and len(r) > 1][:limit]


def _file_imports_used(file_imports: list[str], refs: set[str], calls: list[str]) -> list[str]:
    heads = {c.split(".")[0] for c in calls}
    out = []
    for imp in file_imports:
        parts = imp.replace("/", ".").split(".")
        if parts[0] in refs or parts[-1] in refs or parts[0] in heads or parts[-1] in heads:
            out.append(imp)
    return out


def _class_skeleton(unit: CodeUnit, source_lines: list[str], methods: list[CodeUnit]) -> str:
    out: list[str] = []
    ranges = sorted((m.start_line, m.end_line, m.signature) for m in methods)
    ln = unit.start_line
    while ln <= unit.end_line:
        hit = next((r for r in ranges if r[0] == ln), None)
        if hit is not None:
            indent = re.match(r"\s*", source_lines[ln - 1]).group(0) if ln - 1 < len(source_lines) else ""
            out.append(f"{indent}{hit[2]} ...")
            ln = hit[1] + 1
            continue
        if ln - 1 < len(source_lines):
            out.append(source_lines[ln - 1])
        ln += 1
    return "\n".join(out)


def _split_unit(unit: CodeUnit, source_lines: list[str], cfg: ChunkerConfig) -> list[tuple[int, int]]:
    """Group body statements into line ranges no larger than ``max_chunk_lines``."""
    stmts = unit.body_statements
    if not stmts:
        # No statement info: split on blank lines near the size limit.
        ranges, s = [], unit.start_line
        while s <= unit.end_line:
            e = min(s + cfg.max_chunk_lines - 1, unit.end_line)
            ranges.append((s, e))
            s = e + 1
        return ranges
    ranges: list[tuple[int, int]] = []
    cur_s = unit.start_line  # first part includes the signature
    cur_e = stmts[0][0] - 1 if stmts[0][0] > unit.start_line else unit.start_line
    for sl, el, _, _ in stmts:
        if el - cur_s + 1 > cfg.max_chunk_lines and cur_e - cur_s + 1 >= cfg.min_part_lines:
            ranges.append((cur_s, cur_e))
            cur_s = sl
        cur_e = el
    ranges.append((cur_s, max(cur_e, unit.end_line)))
    return ranges


def chunk_source(
    source: str,
    file_path: str,
    repository: str = "",
    commit: str | None = None,
    language: str | None = None,
    cfg: ChunkerConfig | None = None,
) -> list[Snippet]:
    """Split one file into semantic :class:`Snippet` s."""
    cfg = cfg or ChunkerConfig()
    language = language or detect_language(file_path) or "text"
    parsed = parse_source(source, language, cfg.prefer_tree_sitter)
    lines = source.splitlines()
    file_imports = parsed.file_info.imports
    snippets: list[Snippet] = []
    used_keys: dict[str, int] = {}

    def make(**kw: object) -> Snippet:
        qn = str(kw.get("qualified_name") or kw.get("name") or "")
        n = used_keys.get(qn, 0)
        used_keys[qn] = n + 1
        if n:
            kw["qualified_name"] = f"{qn}#{n + 1}"
        sn = Snippet(snippet_id="", repository=repository, commit=commit, file_path=file_path, language=language, **kw)  # type: ignore[arg-type]
        sn.snippet_id = sha1_hex(f"{repository}|{sn.symbol_key}|{sn.content_hash}", 16)
        sn.extra.setdefault("parser", parsed.parser)
        return sn

    units_by_parent: dict[str, list[CodeUnit]] = {}
    for u in parsed.units:
        if u.kind == "method" and u.parent:
            units_by_parent.setdefault(u.parent, []).append(u)

    for u in parsed.units:
        qual = f"{u.parent}.{u.name}" if u.parent else u.name
        refs = _clean_refs(u.info.references, u.parameters)
        imports = _dedupe_list(u.info.imports + _file_imports_used(file_imports, set(refs), u.info.calls))
        common = dict(
            name=u.name,
            parent=u.parent,
            signature=u.signature,
            docstring=u.docstring,
            parameters=u.parameters,
            decorators=u.decorators,
            bases=u.bases,
        )
        if u.kind == "class":
            methods = units_by_parent.get(qual, [])
            content = _class_skeleton(u, lines, methods) if (cfg.class_skeletons and methods) else u.text
            snippets.append(
                make(
                    content=content, type="class", qualified_name=qual, start_line=u.start_line, end_line=u.end_line,
                    calls=u.info.calls, imports=imports, references=refs, attributes=u.info.attributes,
                    returns=u.info.returns, raises=u.info.raises, defines=[u.name],
                    comments=u.info.comments, string_literals=u.info.strings, ast_hash=u.info.ast_hash,
                    shape_hash=u.info.shape_hash,
                    extra={"member_signatures": u.member_signatures, "members": [m.name for m in methods], "skeleton": bool(methods)},
                    **common,
                )
            )
            continue
        n_lines = u.end_line - u.start_line + 1
        if n_lines <= cfg.max_chunk_lines:
            snippets.append(
                make(
                    content=u.text, type=u.kind, qualified_name=qual, start_line=u.start_line, end_line=u.end_line,
                    calls=u.info.calls, imports=imports, references=refs, attributes=u.info.attributes,
                    returns=u.info.returns, raises=u.info.raises, defines=[u.name], comments=u.info.comments,
                    string_literals=u.info.strings, ast_hash=u.info.ast_hash, shape_hash=u.info.shape_hash, **common,
                )
            )
            continue
        parts = _split_unit(u, lines, cfg)
        for k, (s, e) in enumerate(parts, start=1):
            text = "\n".join(lines[s - 1 : e])
            info = _regex_info(text)
            snippets.append(
                make(
                    content=text, type="function_part", qualified_name=f"{qual}[part{k}]", start_line=s, end_line=e,
                    calls=info.calls, imports=imports, references=_clean_refs(info.references, u.parameters),
                    attributes=info.attributes, returns=info.returns, raises=info.raises,
                    defines=[u.name] if k == 1 else [], comments=info.comments, ast_hash=info.ast_hash,
                    shape_hash=info.shape_hash, extra={"part": k, "parts": len(parts), "parent_symbol": qual},
                    **common,
                )
            )

    if cfg.include_module_blocks:
        for b in parsed.module_blocks:
            text = b.text.strip("\n")
            if not text.strip():
                continue
            # Split very long module blocks at statement-free line boundaries.
            block_lines = text.splitlines()
            step = cfg.max_module_block_lines
            for off in range(0, len(block_lines), step):
                piece = "\n".join(block_lines[off : off + step])
                if not piece.strip():
                    continue
                info = b.info if len(block_lines) <= step else _regex_info(piece)
                assigned = [a for a in info.assigned if a not in _KEYWORDS]
                only_imports = all(
                    ln.strip().startswith(("import ", "from ", "#", "package ", "require", "use ")) or not ln.strip()
                    for ln in piece.splitlines()
                )
                if assigned:
                    name = ",".join(assigned[:3])
                elif only_imports:
                    name = "<imports>"
                else:
                    name = f"<module:{info.ast_hash[:6]}>"
                refs = _clean_refs(info.references, [])
                snippets.append(
                    make(
                        content=piece, type="module", name=name, qualified_name=name, parent=None,
                        start_line=b.start_line + off, end_line=b.start_line + off + len(piece.splitlines()) - 1,
                        calls=info.calls, imports=info.imports, references=refs, attributes=info.attributes,
                        returns=info.returns, raises=info.raises, defines=assigned, comments=info.comments,
                        string_literals=info.strings, ast_hash=info.ast_hash, shape_hash=info.shape_hash,
                    )
                )
    snippets.sort(key=lambda s: (s.start_line, s.type != "class"))
    return snippets


def analyze_document(
    text: str,
    doc_id: str,
    language: str = "python",
    prefer_tree_sitter: bool = True,
    repository: str = "",
) -> Snippet:
    """Treat a whole document (e.g. an MTEB corpus entry) as one snippet with file-level facts."""
    parsed = parse_source(text, language, prefer_tree_sitter)
    info = parsed.file_info
    defined = [u.name for u in parsed.units]
    params = [p for u in parsed.units for p in u.parameters]
    docstrings = [u.docstring for u in parsed.units if u.docstring]
    kind_counts: dict[str, int] = {}
    for u in parsed.units:
        kind_counts[u.kind] = kind_counts.get(u.kind, 0) + 1
    sn = Snippet(
        snippet_id=doc_id,
        content=text,
        file_path=doc_id,
        language=language,
        type="document",
        name=defined[0] if defined else "",
        qualified_name=doc_id,
        repository=repository,
        start_line=1,
        end_line=max(1, text.count("\n") + 1),
        docstring="\n".join(docstrings)[:2000],
        parameters=_dedupe_list(params),
        calls=info.calls,
        imports=info.imports,
        references=_clean_refs(info.references, []),
        attributes=info.attributes,
        returns=info.returns,
        raises=info.raises,
        defines=_dedupe_list(defined + [a for a in info.assigned if a not in _KEYWORDS]),
        comments=info.comments,
        string_literals=info.strings,
        ast_hash=info.ast_hash,
        shape_hash=info.shape_hash,
        extra={"parser": parsed.parser, "unit_kinds": kind_counts, "has_errors": parsed.has_errors},
    )
    return sn


def _dedupe_list(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    return [s for s in seq if s and not (s in seen or seen.add(s))]
