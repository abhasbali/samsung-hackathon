import numpy as np

from codefusion.config.schema import BM25Config, DenseConfig, FusionConfig, QueryConfig, SymbolConfig
from codefusion.parsing.chunker import chunk_source
from codefusion.query.preprocess import QueryPreprocessor
from codefusion.representations import build_views
from codefusion.retrieval.bm25 import BM25Retriever
from codefusion.retrieval.dense import DenseRetriever, VectorIndex
from codefusion.retrieval.embeddings import CachedEmbedder, EmbeddingCache, HashingProvider, l2_normalize
from codefusion.retrieval.fusion import fuse, reciprocal_rank_fusion, resolve_weights
from codefusion.retrieval.symbols import SymbolIndex
from codefusion.types import Hit

DOCS = [
    "def get_user_token(user):\n    return user.auth_token\n",
    "def parse_config(path):\n    return yaml.safe_load(open(path))\n",
    "class HTTPRequestHandler:\n    def handle(self, request):\n        return request.body\n",
    "def retry_with_backoff(fn, max_retries=3):\n    for i in range(max_retries):\n        fn()\n",
]


def _pq(text):
    return QueryPreprocessor(QueryConfig()).process(text)


def _views():
    snips = [chunk_source(d, f"f{i}.py")[0] for i, d in enumerate(DOCS)]
    return snips, [build_views(s) for s in snips]


def test_bm25_identifier_aware_match():
    snips, views = _views()
    bm = BM25Retriever(BM25Config())
    bm.build(views, [s.content_hash for s in snips])
    q = _pq("user token")
    q.bm25_tokens = bm.query_tokens(q.text)
    hits = bm.search(q, 3)
    assert hits[0].idx == 0
    q2 = _pq("HTTP request handling")
    q2.bm25_tokens = bm.query_tokens(q2.text)
    assert bm.search(q2, 1)[0].idx == 2


def test_bm25_mask_and_token_cache():
    snips, views = _views()
    bm = BM25Retriever(BM25Config())
    bm.build(views, [s.content_hash for s in snips])
    n_cached = len(bm._token_cache)
    bm.build(views, [s.content_hash for s in snips])  # rebuild reuses tokenisation
    assert len(bm._token_cache) == n_cached
    q = _pq("user token")
    q.bm25_tokens = bm.query_tokens(q.text)
    mask = np.array([False, True, True, True])
    assert all(h.idx != 0 for h in bm.search(q, 4, mask))


def test_faiss_flat_ip_matches_exact_cosine():
    rng = np.random.default_rng(0)
    vecs = l2_normalize(rng.normal(size=(200, 32)).astype(np.float32))
    q = l2_normalize(rng.normal(size=(1, 32)).astype(np.float32))
    idx = VectorIndex(32, "flat")
    idx.add(vecs[:120])
    idx.add(vecs[120:])  # incremental append
    s, i = idx.search(q, 10)
    exact = np.argsort(-(vecs @ q[0]))[:10]
    assert list(i[0]) == list(exact)
    np.testing.assert_allclose(s[0], (vecs @ q[0])[exact], rtol=1e-5)
    mask = np.ones(200, dtype=bool)
    mask[exact[0]] = False
    _, im = idx.search(q, 5, mask)
    assert exact[0] not in im[0]


def test_hnsw_index_is_configurable():
    rng = np.random.default_rng(1)
    vecs = l2_normalize(rng.normal(size=(300, 16)).astype(np.float32))
    idx = VectorIndex(16, "hnsw")
    idx.add(vecs)
    _, i = idx.search(vecs[5:6], 1)
    assert i[0][0] == 5


def test_dense_retriever_with_cache(tmp_path):
    cache = EmbeddingCache(tmp_path / "e.sqlite")
    emb = CachedEmbedder(HashingProvider(128), cache)
    dr = DenseRetriever(emb, DenseConfig(model="hashing", provider="hashing"))
    dr.add_documents(DOCS)
    assert emb.stats.computed == 4
    hits = dr.search(_pq("retry with backoff max retries"), 2)
    assert hits[0].idx == 3
    emb2 = CachedEmbedder(HashingProvider(128), cache)
    emb2.encode_documents(DOCS)
    assert emb2.stats.computed == 0 and emb2.stats.reused == 4


def test_symbol_index_exact_and_normalized():
    snips = [chunk_source(d, f"f{i}.py")[0] for i, d in enumerate(DOCS)]
    si = SymbolIndex(SymbolConfig())
    si.build(snips)
    q = _pq("where is getUserToken defined?")
    hits = si.search(q, 5)
    assert hits and hits[0].idx == 0
    assert any("def:" in e for e in hits[0].evidence)
    q2 = _pq("who calls retry with backoff")
    assert si.search(q2, 1)[0].idx == 3  # adjacent-word join -> retrywithbackoff


def test_rrf_math_and_provenance():
    ranked = {
        "dense": [Hit(1, 0.9, 1), Hit(2, 0.8, 2)],
        "bm25": [Hit(2, 12.0, 1), Hit(3, 5.0, 2)],
    }
    fused = reciprocal_rank_fusion(ranked, {"dense": 1.0, "bm25": 1.0}, k=60)
    assert fused[0].idx == 2
    assert abs(fused[0].fused_score - (1 / 62 + 1 / 61)) < 1e-12
    assert fused[0].ranks == {"dense": 2, "bm25": 1}
    assert {c.idx for c in fused} == {1, 2, 3}
    # doc 2 is in both lists, so only a (near-)zero bm25 weight lets dense's #1 win.
    w = reciprocal_rank_fusion(ranked, {"dense": 1.0, "bm25": 0.0}, k=60)
    assert w[0].idx == 1
    assert w[0].contributions["dense"] == 1 / 61 and fused[0].contributions["bm25"] == 1 / 61


def test_tied_scores_share_a_rank():
    from codefusion.retrieval.base import top_k_from_scores

    hits = top_k_from_scores(np.array([0.5, 0.9, 0.5, 0.1]), 4, "x")
    assert [(h.idx, h.rank) for h in hits] == [(1, 1), (0, 2), (2, 2), (3, 4)]
    # a retriever that ranks everything equally contributes no preference to RRF
    fused = reciprocal_rank_fusion({"symbol": top_k_from_scores(np.ones(3), 3, "symbol")}, None, 60)
    assert len({round(c.fused_score, 12) for c in fused}) == 1


def test_query_adaptive_weights():
    cfg = FusionConfig()
    wd = resolve_weights(cfg, "DEFINITION")
    wf = resolve_weights(cfg, "DATA_FLOW")
    assert wd["symbol"] > wd["dense"]
    assert wf["dense"] > wf["bm25"]
    cfg.query_adaptive = False
    assert resolve_weights(cfg, "DEFINITION") == cfg.weights


def test_fuse_dense_only_passthrough():
    cfg = FusionConfig(method="dense_only")
    fused, _ = fuse(cfg, {"dense": [Hit(5, 0.7, 1), Hit(2, 0.6, 2)], "bm25": [Hit(2, 9, 1)]}, None)
    assert [c.idx for c in fused] == [5, 2]
