"""Parsers used when tree-sitter (or a grammar) is unavailable.

* Python: the standard-library :mod:`ast` module (exact, but fails on invalid syntax).
* Anything else / unparsable Python: a regex + blank-line-block heuristic that still
  extracts identifiers, calls and imports so lexical and symbol retrieval keep working.
"""

from __future__ import annotations

import ast
import hashlib
import re

from codefusion.parsing.tree_sitter_parser import CodeUnit, ModuleBlock, NodeInfo, ParsedFile, _clean_callee, _dedupe

_CALL_RE = re.compile(r"([A-Za-z_][\w.]*)\s*\(")
_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+([\w.]+)\s+import\s+([\w, *]+)|import\s+([\w., ]+)|#include\s*[<\"]([^>\"]+)|.*require\(['\"]([^'\"]+))",
    re.M,
)
_IDENT_RE = re.compile(r"[A-Za-z_]\w*")
_DEF_RE = re.compile(r"^\s*(?:export\s+)?(?:async\s+)?(?:def|function|func|class|fn)\s+([A-Za-z_]\w*)", re.M)
_NOT_CALLS = frozenset({"if", "for", "while", "return", "switch", "catch", "print", "elif", "and", "or", "not", "in", "with"})


def _regex_info(text: str) -> NodeInfo:
    info = NodeInfo()
    info.calls = _dedupe([_clean_callee(c) for c in _CALL_RE.findall(text) if c not in _NOT_CALLS])
    imps: list[str] = []
    for m in _IMPORT_RE.finditer(text):
        if m.group(1):
            imps.append(m.group(1))
            imps.extend(f"{m.group(1)}.{p.strip()}" for p in m.group(2).split(",") if p.strip() and p.strip() != "*")
        elif m.group(3):
            imps.extend(p.strip().split(" as ")[0] for p in m.group(3).split(","))
        elif m.group(4):
            imps.append(m.group(4))
        elif m.group(5):
            imps.append(m.group(5))
    info.imports = _dedupe(imps)
    info.references = _dedupe(_IDENT_RE.findall(text), limit=600)
    info.attributes = _dedupe(re.findall(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+", text))
    info.comments = _dedupe([c.strip() for c in re.findall(r"(?:#|//)\s*(.+)$", text, re.M)], limit=50)
    info.raises = _dedupe(re.findall(r"(?:raise|throw)\s+(?:new\s+)?([A-Za-z_][\w.]*)", text), limit=20)
    info.returns = _dedupe([r.strip()[:120] for r in re.findall(r"\breturn\b\s*([^\n;]+)", text)], limit=20)
    info.assigned = _dedupe(re.findall(r"^([A-Za-z_]\w*)\s*=(?!=)", text, re.M), limit=100)
    norm = re.sub(r"\s+", " ", re.sub(r"(#|//).*$", "", text, flags=re.M)).strip()
    info.ast_hash = hashlib.sha1(norm.encode()).hexdigest()[:20]
    info.shape_hash = hashlib.sha1(_IDENT_RE.sub("ID", norm).encode()).hexdigest()[:20]
    return info


def parse_regex(source: str, language: str, max_block_lines: int = 60) -> ParsedFile:
    """Blank-line-separated blocks; each block that starts with a definition keyword becomes a unit."""
    lines = source.splitlines()
    units: list[CodeUnit] = []
    blocks: list[ModuleBlock] = []
    start = 0
    offsets = [0]
    for ln in lines:
        offsets.append(offsets[-1] + len(ln) + 1)

    def emit(s: int, e: int) -> None:
        text = "\n".join(lines[s:e]).strip("\n")
        if not text.strip():
            return
        m = _DEF_RE.search(text)
        info = _regex_info(text)
        if m and m.start() < 200:
            kind = "class" if re.search(r"\bclass\s+" + re.escape(m.group(1)), text[: m.end()]) else "function"
            units.append(
                CodeUnit(
                    kind=kind, name=m.group(1), parent=None, start_line=s + 1, end_line=e,
                    start_byte=offsets[s], end_byte=offsets[e], text=text,
                    signature=text.splitlines()[0][:300] if text else "", docstring="", parameters=[],
                    decorators=[], bases=[], info=info,
                )
            )
        else:
            blocks.append(ModuleBlock(s + 1, e, offsets[s], offsets[e], text, info))

    i = 0
    while i <= len(lines):
        at_break = i == len(lines) or (not lines[i].strip() and i > start and (i + 1 >= len(lines) or not lines[i + 1].startswith((" ", "\t"))))
        if at_break or i - start >= max_block_lines:
            emit(start, i)
            start = i + 1 if at_break else i
        i += 1
    return ParsedFile(language=language, units=units, module_blocks=blocks, file_info=_regex_info(source), parser="regex")


class _PyInfoVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.info = NodeInfo()

    def visit_Call(self, node: ast.Call) -> None:
        try:
            self.info.calls.append(_clean_callee(ast.unparse(node.func)))
        except Exception:  # noqa: BLE001
            pass
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        self.info.imports.extend(a.name for a in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        self.info.imports.append(mod)
        self.info.imports.extend(f"{mod}.{a.name}" for a in node.names if a.name != "*")

    def visit_Name(self, node: ast.Name) -> None:
        self.info.references.append(node.id)
        if isinstance(node.ctx, ast.Store):
            self.info.assigned.append(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        try:
            a = _clean_callee(ast.unparse(node))
            if "." in a:
                self.info.attributes.append(a)
        except Exception:  # noqa: BLE001
            pass
        self.info.references.append(node.attr)
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is not None:
            self.info.returns.append(ast.unparse(node.value)[:120])
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        if node.exc is not None:
            m = re.match(r"([A-Za-z_][\w.]*)", ast.unparse(node.exc))
            if m:
                self.info.raises.append(m.group(1))
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and 2 < len(node.value) <= 200:
            self.info.strings.append(node.value)


def _py_info(node: ast.AST, source: str) -> NodeInfo:
    v = _PyInfoVisitor()
    v.visit(node)
    info = v.info
    for name in ("calls", "imports", "references", "attributes", "returns", "raises", "strings", "assigned"):
        setattr(info, name, _dedupe(getattr(info, name)))
    info.ast_hash = hashlib.sha1(ast.dump(node, annotate_fields=False).encode()).hexdigest()[:20]
    shape = re.sub(r"'[^']*'", "ID", ast.dump(node, annotate_fields=False))
    info.shape_hash = hashlib.sha1(shape.encode()).hexdigest()[:20]
    return info


def parse_python_ast(source: str) -> ParsedFile:
    """Parse Python with the stdlib. Raises ``SyntaxError`` on invalid code."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for ln in lines:
        offsets.append(offsets[-1] + len(ln))

    def seg(n: ast.AST) -> tuple[int, int, str]:
        start = min([n.lineno] + [d.lineno for d in getattr(n, "decorator_list", [])])  # type: ignore[attr-defined]
        end = n.end_lineno or n.lineno  # type: ignore[attr-defined]
        return start, end, "".join(lines[start - 1 : end])

    units: list[CodeUnit] = []

    def handle(n: ast.AST, parents: list[str], in_class: bool) -> None:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            s, e, text = seg(n)
            params = [a.arg for a in n.args.posonlyargs + n.args.args + n.args.kwonlyargs if a.arg not in ("self", "cls")]
            if n.args.vararg:
                params.append(n.args.vararg.arg)
            if n.args.kwarg:
                params.append(n.args.kwarg.arg)
            sig_end = n.body[0].lineno - 1 if n.body else n.lineno
            sig = " ".join(ln.strip() for ln in lines[n.lineno - 1 : max(sig_end, n.lineno)])
            units.append(
                CodeUnit(
                    kind="method" if in_class else "function", name=n.name, parent=".".join(parents) or None,
                    start_line=s, end_line=e, start_byte=offsets[s - 1], end_byte=offsets[e], text=text,
                    signature=sig[:300], docstring=ast.get_docstring(n) or "", parameters=params,
                    decorators=[ast.unparse(d).split("(")[0] for d in n.decorator_list], bases=[],
                    info=_py_info(n, source),
                    body_statements=[(b.lineno, b.end_lineno or b.lineno, offsets[b.lineno - 1], offsets[b.end_lineno or b.lineno]) for b in n.body],
                )
            )
        elif isinstance(n, ast.ClassDef):
            s, e, text = seg(n)
            idx = len(units)
            for b in n.body:
                handle(b, parents + [n.name], True)
            methods = units[idx:]
            units.insert(
                idx,
                CodeUnit(
                    kind="class", name=n.name, parent=".".join(parents) or None, start_line=s, end_line=e,
                    start_byte=offsets[s - 1], end_byte=offsets[e], text=text,
                    signature=lines[n.lineno - 1].strip()[:300], docstring=ast.get_docstring(n) or "",
                    parameters=[], decorators=[ast.unparse(d).split("(")[0] for d in n.decorator_list],
                    bases=[ast.unparse(b) for b in n.bases], info=_py_info(n, source),
                    member_signatures=[m.signature for m in methods], has_methods=bool(methods),
                ),
            )

    blocks: list[ModuleBlock] = []
    run: list[ast.stmt] = []

    def flush() -> None:
        if not run:
            return
        s, e = run[0].lineno, run[-1].end_lineno or run[-1].lineno
        mod = ast.Module(body=list(run), type_ignores=[])
        blocks.append(ModuleBlock(s, e, offsets[s - 1], offsets[e], "".join(lines[s - 1 : e]), _py_info(mod, source)))
        run.clear()

    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            flush()
            handle(stmt, [], False)
        else:
            run.append(stmt)
    flush()
    return ParsedFile(language="python", units=units, module_blocks=blocks, file_info=_py_info(tree, source), parser="python-ast")
