from codefusion.graph.store import GraphStore


def _idx(eng, qualified):
    for i, s in enumerate(eng.store.snippets):
        if s.qualified_name == qualified:
            return i
    raise KeyError(qualified)


def test_graph_edges_from_sample_repo(indexed_engine):
    g = indexed_engine.graph
    st = g.stats()
    assert st["nodes"] > 20 and st["edges"] > 40 and st["backend"] in ("rustworkx", "networkx")
    main = _idx(indexed_engine, "main")
    names = {indexed_engine.store.snippets[i].qualified_name for i in g.callers("validate_user")}
    assert "main" in names
    callee_names = {indexed_engine.store.snippets[i].qualified_name for i, _ in g.callees_of(_idx(indexed_engine, "handle_request"))}
    assert {"preprocess_input", "predict", "AuthService.current_user"} <= callee_names
    assert {indexed_engine.store.snippets[i].qualified_name for i in g.definitions("MAX_RETRIES")} == {
        next(s.qualified_name for s in indexed_engine.store.snippets if "MAX_RETRIES" in s.defines)
    }
    imp = g.imports("main.py")
    assert "auth.py" in imp["files"] and "preprocessing.py" in imp["files"]
    assert main in g.callers("handle_request")


def test_callee_resolution_prefers_imported_file(indexed_engine):
    g = indexed_engine.graph
    pre = _idx(indexed_engine, "preprocess_input")
    res = dict(g.callees_of(pre))
    assert res[_idx(indexed_engine, "normalize_input")] == 1.0  # same file


def test_graph_expansion_neighbors_depth(indexed_engine):
    g = indexed_engine.graph
    pre = _idx(indexed_engine, "preprocess_input")
    n1 = g.neighbors(pre, depth=1)
    n2 = g.neighbors(pre, depth=2)
    assert _idx(indexed_engine, "normalize_input") in n1
    assert _idx(indexed_engine, "handle_request") in n1
    assert set(n1) <= set(n2) and len(n2) > len(n1)
    assert _idx(indexed_engine, "main") in n2 and n2[_idx(indexed_engine, "main")][0] == 2


def test_call_chain_and_path(indexed_engine):
    g = indexed_engine.graph
    main = _idx(indexed_engine, "main")
    chain = g.call_chain(main, depth=3)
    names = {c["name"] for c in chain["callees"]}
    assert "handle_request" in names
    path = g.shortest_call_path(main, _idx(indexed_engine, "normalize_input"))
    assert [indexed_engine.store.snippets[i].qualified_name for i in path] == ["main", "handle_request", "preprocess_input", "normalize_input"]


def test_class_contains_methods(indexed_engine):
    g = indexed_engine.graph
    cls = _idx(indexed_engine, "AuthService")
    members = g.store.successors(f"s:{cls}", ["CONTAINS"])
    assert f"s:{_idx(indexed_engine, 'AuthService.validate_user')}" in members


def test_ambiguous_names_are_not_expanded():
    """Many independent programs defining ``solve`` must not make every hop scan the corpus."""
    from codefusion.graph.builder import CodeGraph
    from codefusion.parsing.chunker import chunk_source

    src = "class Solution:\n    def solve(self, x):\n        return helper(x)\n\ndef helper(x):\n    return x\n\ndef main():\n    Solution().solve(1)\n"
    snippets = [s for i in range(60) for s in chunk_source(src, f"p{i}.py")]
    uniq = chunk_source("def unique_fn():\n    return 1\n\ndef caller():\n    return unique_fn()\n", "u.py")
    snippets += uniq
    g = CodeGraph(snippets, max_fanout=50)
    g.add_snippets(0)
    main0 = next(i for i, s in enumerate(snippets) if s.name == "main" and s.file_path == "p0.py")
    assert g.callees_of(main0) == []                  # 'solve' is defined 60 times: ambiguous
    assert g.callees("solve") == []
    assert len(g.definitions("solve")) == 60          # direct lookups stay complete
    solve0 = next(i for i, s in enumerate(snippets) if s.name == "solve" and s.file_path == "p0.py")
    owner = [int(k[2:]) for k in g.store.predecessors(f"s:{solve0}", ["CONTAINS"]) if k.startswith("s:")]
    assert [snippets[o].file_path for o in owner if snippets[o].type == "class"] == ["p0.py"]
    caller = next(i for i, s in enumerate(snippets) if s.name == "caller")
    unique = next(i for i, s in enumerate(snippets) if s.name == "unique_fn")
    assert [d for d, _ in g.callees_of(caller)] == [unique]  # unique names still resolve
    assert CodeGraph(snippets, max_fanout=0).max_fanout == 0


def test_networkx_backend_equivalent():
    for backend in ("rustworkx", "networkx"):
        gs = GraphStore(backend)
        gs.add_node("a", "x")
        gs.add_node("b", "x")
        gs.add_node("c", "x")
        gs.add_edge("a", "b", "CALLS")
        gs.add_edge("b", "c", "CALLS")
        gs.add_edge("a", "b", "REFERENCES")
        assert gs.successors("a", ["CALLS"]) == ["b"]
        assert gs.predecessors("c") == ["b"]
        assert gs.bfs("a", 2) == {"a": 0, "b": 1, "c": 2}
