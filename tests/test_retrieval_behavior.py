"""End-to-end retrieval behaviour on the sample repository (hashing embeddings, no downloads)."""

import pytest


def top_symbols(resp, n=3):
    return [r.snippet["symbol"] for r in resp.results[:n]]


def test_results_are_real_snippets(indexed_engine, sample_repo):
    resp = indexed_engine.search("How is input normalized before prediction?", top_k=5)
    assert resp.results
    for r in resp.results:
        text = (sample_repo / r.snippet["file"]).read_text()
        assert r.snippet["content"].strip() in text or r.snippet["type"] == "class"


@pytest.mark.parametrize(
    "query,expected",
    [
        ("Who calls validate_user?", "main"),
        ("Where is MAX_RETRIES defined?", "MAX_RETRIES"),
        ("What functions does handle_request call?", "preprocess_input"),
        ("How is input normalized before prediction?", "normalize_input"),
        ("Where is authentication implemented?", "AuthService"),
        ("What functions are called before the user is saved?", "register_user"),
    ],
)
def test_demo_queries_find_expected_snippet(indexed_engine, query, expected):
    resp = indexed_engine.search(query, top_k=5)
    assert any(expected in s for s in top_symbols(resp, 5)), (query, top_symbols(resp, 5))


def test_explain_payload(indexed_engine):
    resp = indexed_engine.search("Who calls validate_user?", top_k=3).to_dict(explain=True)
    assert resp["intent"] == "CALLER"
    for stage in ("preprocess", "dense", "bm25", "symbol", "graph", "fusion", "total"):
        assert stage in resp["timings_ms"]
    assert resp["candidate_counts"]["final"] == 3
    r0 = resp["results"][0]
    assert {"dense_rank", "rrf_score", "final_rank", "contributions"} <= set(r0["provenance"])
    assert resp["weights"]["graph"] > resp["weights"]["dense"]  # CALLER intent upweights graph


def test_components_can_be_disabled():
    from codefusion.retrieval.pipeline import CodeFusionEngine
    from codefusion.versions.incremental import index_path
    from conftest import SAMPLE_REPO, tiny_config

    for off in ("bm25.enabled", "symbols.enabled", "graph.enabled", "structural.enabled", "diversity.enabled", "dense.enabled"):
        eng = CodeFusionEngine(tiny_config(**{off: "false"}))
        index_path(eng, SAMPLE_REPO, "s")
        resp = eng.search("Where is MAX_RETRIES defined?", top_k=3)
        assert resp.results, off
