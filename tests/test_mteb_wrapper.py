"""MTEB wrapper tests on an in-memory mini corpus (no dataset or model download)."""

import numpy as np
import pytest

pytest.importorskip("mteb")
datasets = pytest.importorskip("datasets")

from codefusion.evaluation.mteb_encoder import CodeFusionSearchModel, PrePostPipelineEncoder  # noqa: E402
from codefusion.retrieval.embeddings import CachedEmbedder, HashingProvider  # noqa: E402
from conftest import tiny_config  # noqa: E402

CORPUS = {
    "d1": "n = int(input())\nprint(sum(range(n + 1)))\n",
    "d2": "s = input()\nprint(s[::-1])\n",
    "d3": "a, b = map(int, input().split())\nprint(max(a, b))\n",
}
QUERIES = {"q1": "Print the sum of all integers from 0 to n.", "q2": "Reverse the given string s and print it.",
           "q3": "Print the maximum of two integers a and b."}


def _data():
    corpus = datasets.Dataset.from_dict({"id": list(CORPUS), "text": list(CORPUS.values()), "title": [""] * 3})
    queries = datasets.Dataset.from_dict({"id": list(QUERIES), "text": list(QUERIES.values())})
    return corpus, queries


def _embedder():
    return CachedEmbedder(HashingProvider(256), None)


def test_prepost_encoder_encode_contract():
    from mteb.types import PromptType

    cfg = tiny_config()
    enc = PrePostPipelineEncoder(cfg, embedder=_embedder())
    batches = [{"text": ["sum of numbers", "reverse string"]}, {"text": ["maximum"]}]
    out = enc.encode(batches, task_metadata=None, hf_split="test", hf_subset="default", prompt_type=PromptType.query)
    assert out.shape == (3, 256)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-5)
    assert enc.mteb_model_meta.name.startswith("codefusion/")


def test_search_model_protocol_ranks_relevant_docs():
    from mteb.models import SearchProtocol

    cfg = tiny_config(**{"diversity.enabled": "false"})
    m = CodeFusionSearchModel(cfg, embedder=_embedder())
    assert isinstance(m, SearchProtocol)
    corpus, queries = _data()
    m.index(corpus, task_metadata=None, hf_split="test", hf_subset="default", encode_kwargs={})
    res = m.search(queries, task_metadata=None, hf_split="test", hf_subset="default", top_k=3, encode_kwargs={})
    assert set(res) == set(QUERIES)
    for docs in res.values():
        ranked = sorted(docs, key=docs.get, reverse=True)
        assert len(ranked) == 3
    # q1 shares real tokens with d1 (sum, range/n); the hashing embedder has no signal for q2/q3.
    assert sorted(res["q1"], key=res["q1"].get, reverse=True)[0] == "d1"
    assert m.query_latencies_ms and m.timing["index_s"] > 0
