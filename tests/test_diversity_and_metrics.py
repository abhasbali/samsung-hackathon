import numpy as np

from codefusion.config.schema import DiversityConfig
from codefusion.evaluation.metrics import evaluate_rankings, mrr_at_k, ndcg_at_k
from codefusion.retrieval.diversity import deduplicate, jaccard, mmr, shingles
from codefusion.types import Candidate, Snippet


def _sn(i, content, lineage="", ast=""):
    return Snippet(snippet_id=str(i), content=content, lineage_id=lineage, ast_hash=ast or f"a{i}")


def test_dedup_exact_ast_near_and_lineage():
    body = "def f(values):\n    total = 0\n    for v in values:\n        total += v * weight\n    return total / len(values)\n"
    snippets = [
        _sn(0, body, "L1"),
        _sn(1, body, "L9"),  # exact duplicate content
        _sn(2, "def g(x):\n    return x\n", "L2", ast="same"),
        _sn(3, "def g(x):  # formatted\n    return x\n", "L3", ast="same"),  # same AST
        _sn(4, body.replace("weight", "weights"), "L1"),  # other version, same lineage
        _sn(5, "def h():\n    return 1\n", "L5"),
    ]
    cands = [Candidate(idx=i, final_score=1.0 - i * 0.1) for i in range(6)]
    kept, dropped = deduplicate(cands, snippets, DiversityConfig())
    assert [c.idx for c in kept] == [0, 2, 5]
    reasons = {c.idx: c.dropped_reason for c in dropped}
    assert "duplicate content" in reasons[1] and "AST" in reasons[3] and "version" in reasons[4]
    kept_h, _ = deduplicate([Candidate(idx=i, final_score=1.0) for i in (0, 4)], snippets,
                            DiversityConfig(near_duplicate_threshold=0.5), keep_history=True)
    assert [c.idx for c in kept_h] == [0, 4]  # history queries keep versions


def test_shingles_jaccard():
    a = shingles("alpha beta gamma delta")
    assert jaccard(a, a) == 1.0
    assert jaccard(a, shingles("x y z w")) == 0.0


def test_mmr_prefers_diverse():
    vecs = np.array([[1, 0], [0.99, 0.01], [0, 1]], dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    cands = [Candidate(idx=i, final_score=s) for i, s in enumerate([1.0, 0.95, 0.6])]
    order = [c.idx for c in mmr(cands, vecs, [], lam=0.5, k=3)]
    assert order[:2] == [0, 2]


def test_metrics():
    rel = {"d1": 1}
    assert ndcg_at_k(["d1", "x"], rel) == 1.0
    assert abs(ndcg_at_k(["x", "d1"], rel) - 1 / np.log2(3)) < 1e-12
    assert mrr_at_k(["x", "y", "d1"], rel) == 1 / 3
    assert mrr_at_k(["x"] * 10 + ["d1"], rel, 10) == 0.0
    m = evaluate_rankings({"q1": ["d1"], "q2": ["a", "d2"]}, {"q1": {"d1": 1}, "q2": {"d2": 1}, "q3": {"d3": 1}})
    assert abs(m["mrr_at_10"] - (1 + 0.5 + 0) / 3) < 1e-12
