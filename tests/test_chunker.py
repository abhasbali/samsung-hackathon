from codefusion.parsing.chunker import ChunkerConfig, analyze_document, chunk_source, parse_source
from codefusion.parsing.fallback import parse_python_ast, parse_regex
from codefusion.parsing.symbols import extract_symbols
from codefusion.representations import build_views

AUTH = '''
import jwt
from config import SECRET_KEY

MAX_RETRIES = 3


class AuthService(BaseService):
    """Authenticates users."""

    def __init__(self, repo):
        self.repo = repo

    def authenticate(self, user, password=None):
        """Verify the JWT."""
        if not user.token:
            raise AuthError("missing token")
        return jwt.verify(user.token, SECRET_KEY)


def helper(x):
    return x + 1
'''


def by_name(snips):
    return {s.qualified_name: s for s in snips}


def test_ast_units_and_metadata(sample_repo):
    snips = by_name(chunk_source(AUTH, "auth/service.py", "repo", "abc123"))
    m = snips["AuthService.authenticate"]
    assert m.type == "method" and m.parent == "AuthService" and m.name == "authenticate"
    assert m.parameters == ["user", "password"]
    assert "jwt.verify" in m.calls and "AuthError" in m.calls
    assert "AuthError" in m.raises
    assert "user.token" in m.attributes
    assert m.docstring.startswith("Verify")
    assert m.start_line < m.end_line and m.commit == "abc123" and m.repository == "repo"
    assert m.content_hash and m.ast_hash and m.snippet_id
    assert any(i.startswith("jwt") for i in m.imports)  # file import used by the method
    cls = snips["AuthService"]
    assert cls.type == "class" and "BaseService" in cls.bases
    assert "authenticate" in cls.extra["members"] and cls.defines == ["AuthService"]
    assert "def authenticate(self, user, password=None): ..." in cls.content  # skeleton, not the body
    assert "jwt.verify" not in cls.content
    assert snips["helper"].type == "function"
    module = [s for s in snips.values() if s.type == "module"]
    assert any("MAX_RETRIES" in s.defines for s in module)


def test_snippet_ids_stable_and_content_sensitive():
    a = by_name(chunk_source(AUTH, "a.py", "r", "c1"))
    b = by_name(chunk_source(AUTH, "a.py", "r", "c2"))
    assert a["helper"].snippet_id == b["helper"].snippet_id  # same version in two commits
    c = by_name(chunk_source(AUTH.replace("x + 1", "x + 2"), "a.py", "r", "c3"))
    assert c["helper"].snippet_id != a["helper"].snippet_id


def test_ast_hash_ignores_formatting_and_comments():
    s1 = chunk_source("def f(a):\n    return a+1\n", "x.py")[0]
    s2 = chunk_source("def f(a):\n    # comment\n    return a + 1\n", "x.py")[0]
    assert s1.content_hash != s2.content_hash
    assert s1.ast_hash == s2.ast_hash


def test_large_function_split_on_statement_boundaries():
    body = "\n".join(f"    v{i} = compute_{i}(v{i - 1} if {i} else 0)" for i in range(1, 120))
    src = f"def big(x):\n    v0 = x\n{body}\n    return v119\n"
    parts = chunk_source(src, "big.py", cfg=ChunkerConfig(max_chunk_lines=40))
    parts = [p for p in parts if p.type == "function_part"]
    assert len(parts) >= 3
    assert parts[0].content.startswith("def big(x):")
    for p in parts:
        assert p.end_line - p.start_line + 1 <= 40
        assert p.signature.startswith("def big")
    # consecutive, non-overlapping and covering the function
    for a, b in zip(parts, parts[1:]):
        assert b.start_line == a.end_line + 1


def test_javascript_and_java_units():
    js = "import x from 'lib';\nclass A extends B { run(a) { return helper(a); } }\nconst f = (y) => y * 2;\nfunction g() { f(1); }\n"
    names = {s.qualified_name: s for s in chunk_source(js, "a.js")}
    assert "A.run" in names and names["A.run"].type == "method"
    assert "f" in names and "g" in names and "f" in names["g"].calls
    java = "class Svc { int add(int a, int b) { return Math.max(a, b); } }"
    jn = {s.qualified_name: s for s in chunk_source(java, "Svc.java")}
    assert jn["Svc.add"].parameters == ["a", "b"] and "Math.max" in jn["Svc.add"].calls


def test_fallback_parsers_work_without_tree_sitter():
    pf = parse_python_ast(AUTH)
    assert {u.name for u in pf.units} >= {"AuthService", "authenticate", "helper"}
    rx = parse_regex("func Handle(w, r) {\n  serve(r)\n}\n", "go")
    assert rx.units and rx.units[0].name == "Handle"
    # invalid python still produces snippets (tree-sitter or regex path)
    snips = chunk_source("def broken(:\n  pass\nx = 1\n", "b.py")
    assert snips


def test_python2_syntax_is_tolerated():
    pf = parse_source("print 'hello'\nx = raw_input()\n", "python")
    assert pf.file_info.references


def test_document_analysis_for_benchmark_corpus():
    doc = "n = int(input())\narr = list(map(int, input().split()))\nprint(sum(sorted(arr)[:n]))\n"
    sn = analyze_document(doc, "d1")
    assert sn.type == "document" and sn.snippet_id == "d1"
    assert {"input", "sorted", "print"} <= set(sn.calls)
    assert "arr" in sn.defines


def test_symbol_records_roles():
    snips = by_name(chunk_source(AUTH, "a.py"))
    recs = {(r.normalized, r.role) for r in extract_symbols(snips["AuthService.authenticate"])}
    assert ("authenticate", "def") in recs
    assert ("verify", "call") in recs and ("jwtverify", "call") in recs
    assert ("user", "param") in recs or ("user", "ref") in recs


def test_multi_view_representations():
    m = by_name(chunk_source(AUTH, "auth/service.py"))["AuthService.authenticate"]
    views = build_views(m)
    assert "authenticate" in views["identifiers"] and "token" in views["identifiers"]
    assert "METHOD authenticate" in views["structural"] and "CALL jwt.verify" in views["structural"]
    assert "PARAM user" in views["structural"] and "ACCESS user.token" in views["structural"]
    assert "FILE auth/service.py" in views["context"] and "CLASS AuthService" in views["context"]
    assert "Verify the JWT" in views["docstring"]
