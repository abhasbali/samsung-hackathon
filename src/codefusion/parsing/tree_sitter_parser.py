"""Tree-sitter based, language-spec-driven code analysis.

One generic walker extracts definitions (functions, methods, classes), calls, imports,
parameters, attribute accesses, identifiers, returns, raised exceptions, decorators,
base classes, comments and docstrings. Per-language differences live entirely in
:class:`LanguageSpec` tables, so adding a language is a data change.

If ``tree_sitter`` or a grammar package is not installed, :func:`get_extractor` returns
``None`` and callers fall back to :mod:`codefusion.parsing.fallback`.
"""

from __future__ import annotations

import hashlib
import importlib
import re
from dataclasses import dataclass, field
from functools import cache
from typing import Any

from codefusion.logging_utils import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class LanguageSpec:
    name: str
    extensions: tuple[str, ...]
    module: str
    language_attr: str = "language"
    function_types: frozenset[str] = frozenset()
    class_types: frozenset[str] = frozenset()
    decorated_types: frozenset[str] = frozenset()
    var_function_types: frozenset[str] = frozenset()  # e.g. JS `const f = () => ...`
    call_types: frozenset[str] = frozenset()
    import_types: frozenset[str] = frozenset()
    comment_types: frozenset[str] = frozenset({"comment"})
    string_types: frozenset[str] = frozenset({"string"})
    identifier_types: frozenset[str] = frozenset({"identifier"})
    attribute_types: frozenset[str] = frozenset()
    return_types: frozenset[str] = frozenset({"return_statement"})
    raise_types: frozenset[str] = frozenset()
    assignment_types: frozenset[str] = frozenset()
    call_function_fields: tuple[str, ...] = ("function",)
    call_object_field: str | None = None  # Java method_invocation has object + name
    params_field: str = "parameters"
    body_field: str = "body"
    bases_fields: tuple[str, ...] = ()
    receiver_field: str | None = None  # Go methods
    docstring_in_body: bool = False


PY = LanguageSpec(
    name="python",
    extensions=(".py", ".pyw", ".pyi"),
    module="tree_sitter_python",
    function_types=frozenset({"function_definition"}),
    class_types=frozenset({"class_definition"}),
    decorated_types=frozenset({"decorated_definition"}),
    call_types=frozenset({"call"}),
    import_types=frozenset({"import_statement", "import_from_statement", "future_import_statement"}),
    attribute_types=frozenset({"attribute"}),
    raise_types=frozenset({"raise_statement"}),
    assignment_types=frozenset({"assignment", "augmented_assignment"}),
    bases_fields=("superclasses",),
    docstring_in_body=True,
)

JS = LanguageSpec(
    name="javascript",
    extensions=(".js", ".jsx", ".mjs", ".cjs"),
    module="tree_sitter_javascript",
    function_types=frozenset({"function_declaration", "method_definition", "generator_function_declaration"}),
    class_types=frozenset({"class_declaration", "class"}),
    var_function_types=frozenset({"variable_declarator"}),
    call_types=frozenset({"call_expression", "new_expression"}),
    import_types=frozenset({"import_statement"}),
    string_types=frozenset({"string", "template_string"}),
    identifier_types=frozenset({"identifier", "property_identifier", "shorthand_property_identifier"}),
    attribute_types=frozenset({"member_expression"}),
    raise_types=frozenset({"throw_statement"}),
    assignment_types=frozenset({"variable_declarator", "assignment_expression"}),
    call_function_fields=("function", "constructor"),
    bases_fields=(),
)

TS = LanguageSpec(
    name="typescript",
    extensions=(".ts",),
    module="tree_sitter_typescript",
    language_attr="language_typescript",
    function_types=JS.function_types | {"function_signature", "method_signature"},
    class_types=JS.class_types | {"interface_declaration", "abstract_class_declaration"},
    var_function_types=JS.var_function_types,
    call_types=JS.call_types,
    import_types=JS.import_types,
    string_types=JS.string_types,
    identifier_types=JS.identifier_types | {"type_identifier"},
    attribute_types=JS.attribute_types,
    raise_types=JS.raise_types,
    assignment_types=JS.assignment_types,
    call_function_fields=JS.call_function_fields,
)

TSX = LanguageSpec(**{**TS.__dict__, "name": "tsx", "extensions": (".tsx",), "language_attr": "language_tsx"})

JAVA = LanguageSpec(
    name="java",
    extensions=(".java",),
    module="tree_sitter_java",
    function_types=frozenset({"method_declaration", "constructor_declaration"}),
    class_types=frozenset({"class_declaration", "interface_declaration", "enum_declaration", "record_declaration"}),
    call_types=frozenset({"method_invocation", "object_creation_expression"}),
    import_types=frozenset({"import_declaration"}),
    comment_types=frozenset({"line_comment", "block_comment"}),
    string_types=frozenset({"string_literal"}),
    identifier_types=frozenset({"identifier", "type_identifier"}),
    attribute_types=frozenset({"field_access"}),
    raise_types=frozenset({"throw_statement"}),
    assignment_types=frozenset({"variable_declarator", "assignment_expression"}),
    call_function_fields=("name", "type"),
    call_object_field="object",
    bases_fields=("superclass", "interfaces"),
)

GO = LanguageSpec(
    name="go",
    extensions=(".go",),
    module="tree_sitter_go",
    function_types=frozenset({"function_declaration", "method_declaration"}),
    class_types=frozenset({"type_spec"}),
    call_types=frozenset({"call_expression"}),
    import_types=frozenset({"import_declaration"}),
    string_types=frozenset({"interpreted_string_literal", "raw_string_literal"}),
    identifier_types=frozenset({"identifier", "field_identifier", "type_identifier", "package_identifier"}),
    attribute_types=frozenset({"selector_expression"}),
    raise_types=frozenset(),
    assignment_types=frozenset({"short_var_declaration", "assignment_statement", "const_spec", "var_spec"}),
    receiver_field="receiver",
)

LANGUAGE_SPECS: dict[str, LanguageSpec] = {s.name: s for s in (PY, JS, TS, TSX, JAVA, GO)}
EXTENSION_TO_LANGUAGE: dict[str, str] = {ext: s.name for s in LANGUAGE_SPECS.values() for ext in s.extensions}

_SELF_PREFIX = re.compile(r"^(self|this|cls)\.")
_CALL_ARGS = re.compile(r"\([^()]*\)")
_WS = re.compile(r"\s+")


def detect_language(file_path: str) -> str | None:
    for ext, lang in EXTENSION_TO_LANGUAGE.items():
        if file_path.lower().endswith(ext):
            return lang
    return None


@dataclass
class NodeInfo:
    """Facts collected from a subtree."""

    calls: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    attributes: list[str] = field(default_factory=list)
    returns: list[str] = field(default_factory=list)
    raises: list[str] = field(default_factory=list)
    comments: list[str] = field(default_factory=list)
    strings: list[str] = field(default_factory=list)
    assigned: list[str] = field(default_factory=list)
    ast_hash: str = ""
    shape_hash: str = ""


@dataclass
class CodeUnit:
    """A definition extracted from a file (before becoming a :class:`Snippet`)."""

    kind: str  # function | method | class
    name: str
    parent: str | None
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    text: str
    signature: str
    docstring: str
    parameters: list[str]
    decorators: list[str]
    bases: list[str]
    info: NodeInfo
    body_statements: list[tuple[int, int, int, int]] = field(default_factory=list)  # (sl, el, sb, eb)
    member_signatures: list[str] = field(default_factory=list)
    has_methods: bool = False


@dataclass
class ModuleBlock:
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    text: str
    info: NodeInfo


@dataclass
class ParsedFile:
    language: str
    units: list[CodeUnit]
    module_blocks: list[ModuleBlock]
    file_info: NodeInfo
    has_errors: bool = False
    parser: str = "tree-sitter"


def _dedupe(seq: list[str], limit: int = 400) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in seq:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
            if len(out) >= limit:
                break
    return out


def _clean_callee(text: str) -> str:
    text = _WS.sub("", text)
    prev = None
    while prev != text:  # strip nested call argument lists: a(b).c(d) -> a.c
        prev = text
        text = _CALL_ARGS.sub("", text)
    text = _SELF_PREFIX.sub("", text)
    if len(text) > 80 or not text:
        return ""
    return text


class TreeSitterExtractor:
    """Parses one language with tree-sitter and extracts :class:`ParsedFile` facts."""

    def __init__(self, spec: LanguageSpec) -> None:
        from tree_sitter import Language, Parser  # optional dependency

        mod = importlib.import_module(spec.module)
        lang_ptr = getattr(mod, spec.language_attr)()
        try:
            language = Language(lang_ptr)
        except TypeError:  # very old bindings take (ptr, name)
            language = Language(lang_ptr, spec.name)  # type: ignore[call-arg]
        try:
            parser = Parser(language)
        except TypeError:
            parser = Parser()
            parser.set_language(language)  # type: ignore[attr-defined]
        self.spec = spec
        self.parser = parser

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _text(src: bytes, node: Any) -> str:
        return src[node.start_byte : node.end_byte].decode("utf-8", errors="replace")

    def _name_of(self, node: Any, src: bytes) -> str:
        n = node.child_by_field_name("name")
        if n is not None:
            return self._text(src, n)
        for c in node.named_children:
            if c.type in self.spec.identifier_types:
                return self._text(src, c)
        return ""

    def _params(self, node: Any, src: bytes) -> list[str]:
        p = node.child_by_field_name(self.spec.params_field)
        if p is None:
            return []
        out: list[str] = []
        for c in p.named_children:
            if c.type in self.spec.comment_types:
                continue
            if c.type in self.spec.identifier_types:
                name = self._text(src, c)
            else:
                nm = c.child_by_field_name("name") or c.child_by_field_name("pattern")
                if nm is None:
                    stack = list(c.named_children)
                    nm = None
                    while stack:
                        x = stack.pop(0)
                        if x.type in self.spec.identifier_types:
                            nm = x
                            break
                        stack.extend(x.named_children)
                name = self._text(src, nm) if nm is not None else ""
            name = name.lstrip("*&")
            if name and name not in ("self", "cls", "this"):
                out.append(name)
        return out

    def _bases(self, node: Any, src: bytes) -> list[str]:
        out: list[str] = []
        fields = list(self.spec.bases_fields)
        for f in fields:
            b = node.child_by_field_name(f)
            if b is None:
                continue
            for c in b.named_children if b.named_children else [b]:
                t = self._text(src, c).strip()
                if t and len(t) < 80 and "=" not in t:
                    out.append(t.split("(")[0].replace("extends", "").replace("implements", "").strip())
        # JS/TS: class_heritage child
        for c in node.named_children:
            if c.type == "class_heritage":
                out.extend(re.findall(r"[A-Za-z_][\w.]*", self._text(src, c).replace("extends", " ").replace("implements", " ")))
        return _dedupe([o for o in out if o])

    def _decorators(self, node: Any, src: bytes) -> list[str]:
        return [self._text(src, c).lstrip("@").split("(")[0].strip() for c in node.named_children if c.type == "decorator"]

    def _signature(self, node: Any, src: bytes) -> str:
        body = node.child_by_field_name(self.spec.body_field)
        end = body.start_byte if body is not None else node.end_byte
        sig = src[node.start_byte : end].decode("utf-8", errors="replace").strip()
        sig = _WS.sub(" ", sig)
        return sig[:300]

    def _docstring(self, node: Any, src: bytes) -> str:
        if self.spec.docstring_in_body:
            body = node.child_by_field_name(self.spec.body_field)
            if body is not None and body.named_children:
                first = body.named_children[0]
                if first.type == "expression_statement" and first.named_children and first.named_children[0].type in self.spec.string_types:
                    return self._text(src, first.named_children[0]).strip("\"' \n\r\trbuRBU")[:2000]
        # Preceding doc comments (JS/Java/Go style)
        comments: list[str] = []
        prev = node.prev_named_sibling
        expected_row = node.start_point[0]
        while prev is not None and prev.type in self.spec.comment_types and prev.end_point[0] >= expected_row - 1:
            comments.insert(0, self._text(src, prev))
            expected_row = prev.start_point[0]
            prev = prev.prev_named_sibling
        doc = "\n".join(comments)
        doc = re.sub(r"^\s*(/\*\*?|\*/|\*|//+|#)\s?", "", doc, flags=re.M)
        return doc.strip()[:2000]

    def collect_info(self, root: Any, src: bytes, skip: set[tuple[int, int]] | None = None) -> NodeInfo:
        """Walk ``root``'s subtree collecting facts. ``skip`` holds byte ranges of nested units to exclude."""
        spec = self.spec
        info = NodeInfo()
        h_ast = hashlib.sha1()
        h_shape = hashlib.sha1()
        stack = [root]
        while stack:
            node = stack.pop()
            if skip and node is not root and (node.start_byte, node.end_byte) in skip:
                continue
            t = node.type
            if t in spec.comment_types:
                info.comments.append(self._text(src, node).lstrip("#/* ").strip()[:300])
                continue
            if node.is_named:
                h_shape.update(t.encode())
            if node.child_count == 0:
                leaf = self._text(src, node)
                h_ast.update(t.encode())
                h_ast.update(leaf.encode("utf-8", errors="replace"))
                if t in spec.identifier_types:
                    info.references.append(leaf)
            if t in spec.call_types:
                callee = ""
                for f in spec.call_function_fields:
                    fn = node.child_by_field_name(f)
                    if fn is not None:
                        callee = self._text(src, fn)
                        break
                if spec.call_object_field:
                    obj = node.child_by_field_name(spec.call_object_field)
                    if obj is not None and callee:
                        callee = f"{self._text(src, obj)}.{callee}"
                callee = _clean_callee(callee)
                if callee:
                    info.calls.append(callee)
            elif t in spec.import_types:
                info.imports.extend(self._import_names(node, src))
            elif t in spec.attribute_types:
                a = _clean_callee(self._text(src, node))
                if a and "." in a:
                    info.attributes.append(a)
            elif t in spec.return_types:
                val = _WS.sub(" ", self._text(src, node)).strip()
                val = re.sub(r"^return\s*", "", val).rstrip(";")
                if val:
                    info.returns.append(val[:120])
            elif t in spec.raise_types:
                m = re.search(r"(?:raise|throw)\s+(?:new\s+)?([A-Za-z_][\w.]*)", self._text(src, node))
                if m:
                    info.raises.append(m.group(1))
            elif t in spec.string_types:
                s = self._text(src, node).strip("\"'`")
                if 2 < len(s) <= 200:
                    info.strings.append(s)
            if t in spec.assignment_types:
                left = node.child_by_field_name("left") or node.child_by_field_name("name")
                if left is not None and left.type in spec.identifier_types | {"pattern_list", "expression_list", "identifier_list"}:
                    for name in re.findall(r"[A-Za-z_]\w*", self._text(src, left)):
                        info.assigned.append(name)
            stack.extend(reversed(node.children))
        info.calls = _dedupe(info.calls)
        info.imports = _dedupe(info.imports)
        info.references = _dedupe(info.references, limit=600)
        info.attributes = _dedupe(info.attributes)
        info.returns = _dedupe(info.returns, limit=20)
        info.raises = _dedupe(info.raises, limit=20)
        info.comments = _dedupe(info.comments, limit=50)
        info.strings = _dedupe(info.strings, limit=50)
        info.assigned = _dedupe(info.assigned, limit=100)
        info.ast_hash = h_ast.hexdigest()[:20]
        info.shape_hash = h_shape.hexdigest()[:20]
        return info

    def _import_names(self, node: Any, src: bytes) -> list[str]:
        text = _WS.sub(" ", self._text(src, node))
        lang = self.spec.name
        names: list[str] = []
        if lang == "python":
            m = re.match(r"from\s+([\w.]+)\s+import\s+(.+)", text)
            if m:
                mod = m.group(1)
                names.append(mod)
                for part in m.group(2).strip("() ").split(","):
                    part = part.strip().split(" as ")[0].strip()
                    if part and part != "*":
                        names.append(f"{mod}.{part}")
            else:
                for part in re.sub(r"^import\s+", "", text).split(","):
                    part = part.strip().split(" as ")[0].strip()
                    if part:
                        names.append(part)
        elif lang in ("javascript", "typescript", "tsx"):
            m = re.search(r"from\s+['\"]([^'\"]+)['\"]", text) or re.search(r"import\s+['\"]([^'\"]+)['\"]", text)
            if m:
                names.append(m.group(1))
            names.extend(re.findall(r"\b([A-Za-z_]\w*)\b(?=\s*(?:,|\}|\s+from))", text))
        elif lang == "java":
            m = re.search(r"import\s+(?:static\s+)?([\w.*]+)", text)
            if m:
                names.append(m.group(1))
        elif lang == "go":
            names.extend(re.findall(r"\"([^\"]+)\"", text))
        return names

    # ------------------------------------------------------------------ main entry
    def parse(self, source: str) -> ParsedFile:
        src = source.encode("utf-8", errors="replace")
        tree = self.parser.parse(src)
        root = tree.root_node
        units: list[CodeUnit] = []
        unit_ranges: set[tuple[int, int]] = set()
        top_level_def_ranges: list[tuple[int, int]] = []

        def visit(node: Any, parents: list[str], in_function: bool, in_class: bool, depth: int) -> None:
            spec = self.spec
            outer = node
            decorators: list[str] = []
            if node.type in spec.decorated_types:
                decorators = self._decorators(node, src)
                inner = node.child_by_field_name("definition")
                if inner is None:
                    for c in node.named_children:
                        if c.type in spec.function_types | spec.class_types:
                            inner = c
                if inner is None:
                    return
                node = inner
            t = node.type
            is_var_fn = False
            if t in spec.var_function_types:
                val = node.child_by_field_name("value")
                is_var_fn = val is not None and val.type in {"arrow_function", "function_expression", "function"}
            if (t in spec.function_types or is_var_fn) and not in_function:
                fn_node = node.child_by_field_name("value") if is_var_fn else node
                name = self._name_of(node, src) or "<anonymous>"
                parent = ".".join(parents) or None
                kind = "method" if in_class else "function"
                if spec.receiver_field:
                    recv = node.child_by_field_name(spec.receiver_field)
                    if recv is not None:
                        m = re.findall(r"[A-Za-z_]\w*", self._text(src, recv))
                        if m:
                            parent = m[-1]
                            kind = "method"
                info = self.collect_info(outer, src)
                body = fn_node.child_by_field_name(spec.body_field)
                stmts = []
                if body is not None:
                    for c in body.named_children:
                        stmts.append((c.start_point[0] + 1, c.end_point[0] + 1, c.start_byte, c.end_byte))
                units.append(
                    CodeUnit(
                        kind=kind,
                        name=name,
                        parent=parent,
                        start_line=outer.start_point[0] + 1,
                        end_line=outer.end_point[0] + 1,
                        start_byte=outer.start_byte,
                        end_byte=outer.end_byte,
                        text=self._text(src, outer),
                        signature=self._signature(fn_node, src),
                        docstring=self._docstring(fn_node, src) or (self._docstring(outer, src) if outer is not fn_node else ""),
                        parameters=self._params(fn_node, src),
                        decorators=decorators,
                        bases=[],
                        info=info,
                        body_statements=stmts,
                    )
                )
                unit_ranges.add((outer.start_byte, outer.end_byte))
                if depth == 0:
                    top_level_def_ranges.append((outer.start_byte, outer.end_byte))
                # Nested closures stay inside the function snippet (semantic boundary kept).
                return
            if t in spec.class_types and not in_function:
                if t == "type_spec":  # Go: only structs/interfaces are "classes"
                    ty = node.child_by_field_name("type")
                    if ty is None or ty.type not in ("struct_type", "interface_type"):
                        for c in node.children:
                            visit(c, parents, in_function, in_class, depth + 1)
                        return
                name = self._name_of(node, src) or "<anonymous>"
                body = node.child_by_field_name(spec.body_field)
                n_before = len(units)
                if body is not None:
                    for c in body.children:
                        visit(c, parents + [name], False, True, depth + 1)
                methods = units[n_before:]
                member_sigs = [m.signature for m in methods if m.parent == ".".join(parents + [name])]
                skip = {(m.start_byte, m.end_byte) for m in methods}
                info = self.collect_info(outer, src, skip=skip)
                units.insert(
                    n_before,
                    CodeUnit(
                        kind="class",
                        name=name,
                        parent=".".join(parents) or None,
                        start_line=outer.start_point[0] + 1,
                        end_line=outer.end_point[0] + 1,
                        start_byte=outer.start_byte,
                        end_byte=outer.end_byte,
                        text=self._text(src, outer),
                        signature=self._signature(node, src),
                        docstring=self._docstring(node, src) or self._docstring(outer, src),
                        parameters=[],
                        decorators=decorators,
                        bases=self._bases(node, src),
                        info=info,
                        member_signatures=member_sigs,
                        has_methods=bool(methods),
                    ),
                )
                unit_ranges.add((outer.start_byte, outer.end_byte))
                if depth == 0:
                    top_level_def_ranges.append((outer.start_byte, outer.end_byte))
                return
            for c in node.children:
                # Top-level statements wrapping definitions (export, lexical_declaration) keep depth 0.
                visit(c, parents, in_function, in_class, depth if node.type in _TRANSPARENT else depth + 1)

        for child in root.children:
            visit(child, [], False, False, 0)

        # Module-level blocks: maximal runs of top-level statements between definitions.
        blocks: list[ModuleBlock] = []
        run: list[Any] = []

        def flush() -> None:
            if not run:
                return
            sb, eb = run[0].start_byte, run[-1].end_byte
            info = self._collect_range(run, src)
            blocks.append(
                ModuleBlock(
                    start_line=run[0].start_point[0] + 1,
                    end_line=run[-1].end_point[0] + 1,
                    start_byte=sb,
                    end_byte=eb,
                    text=src[sb:eb].decode("utf-8", errors="replace"),
                    info=info,
                )
            )
            run.clear()

        def contains_def(node: Any) -> bool:
            return any(s >= node.start_byte and e <= node.end_byte for s, e in top_level_def_ranges)

        for child in root.named_children:
            if contains_def(child):
                flush()
                continue
            run.append(child)
        flush()

        file_info = self.collect_info(root, src)
        return ParsedFile(
            language=self.spec.name,
            units=units,
            module_blocks=blocks,
            file_info=file_info,
            has_errors=bool(getattr(root, "has_error", False)),
        )

    def _collect_range(self, nodes: list[Any], src: bytes) -> NodeInfo:
        infos = [self.collect_info(n, src) for n in nodes]
        merged = NodeInfo()
        for i in infos:
            merged.calls += i.calls
            merged.imports += i.imports
            merged.references += i.references
            merged.attributes += i.attributes
            merged.returns += i.returns
            merged.raises += i.raises
            merged.comments += i.comments
            merged.strings += i.strings
            merged.assigned += i.assigned
        for name in ("calls", "imports", "references", "attributes", "returns", "raises", "comments", "strings", "assigned"):
            setattr(merged, name, _dedupe(getattr(merged, name)))
        merged.ast_hash = hashlib.sha1("".join(i.ast_hash for i in infos).encode()).hexdigest()[:20]
        merged.shape_hash = hashlib.sha1("".join(i.shape_hash for i in infos).encode()).hexdigest()[:20]
        return merged


_TRANSPARENT = frozenset({"export_statement", "lexical_declaration", "variable_declaration", "program", "module"})


@cache
def get_extractor(language: str) -> TreeSitterExtractor | None:
    """Return a cached extractor, or ``None`` when tree-sitter or the grammar is unavailable."""
    spec = LANGUAGE_SPECS.get(language)
    if spec is None:
        return None
    try:
        return TreeSitterExtractor(spec)
    except Exception as exc:  # noqa: BLE001 - optional grammar packages
        log.warning("tree-sitter unavailable for language", extra={"data": {"language": language, "error": repr(exc)}})
        return None
