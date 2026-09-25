import importlib

import pytest

from conftest import ROOT, SAMPLE_REPO

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEFUSION_CONFIG", str(ROOT / "configs" / "test_tiny.yaml"))
    monkeypatch.setenv("CODEFUSION_INDEX_DIR", str(tmp_path / "idx"))
    import codefusion.api.main as api

    api = importlib.reload(api)
    return TestClient(api.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_index_and_search(client):
    r = client.post("/index/repository", json={"path": str(SAMPLE_REPO), "repository": "sample"})
    assert r.status_code == 200, r.text
    assert r.json()["stats"]["snippet_versions"] > 10
    r = client.post("/search", json={"query": "Who calls validate_user?", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["intent"] == "CALLER" and len(body["results"]) == 3
    res = body["results"][0]
    assert {"rank", "score", "snippet", "provenance", "structural_evidence", "graph_evidence"} <= set(res)
    assert {"file", "lines", "symbol", "language", "commit", "content"} <= set(res["snippet"])
    assert body["timings_ms"]["total"] > 0
    ex = client.post("/search/explain", json={"query": "Where is MAX_RETRIES defined?"}).json()
    assert "weights" in ex and "query_analysis" in ex
    stats = client.get("/stats").json()
    assert stats["snippet_versions"] > 10
    sid = res["snippet"]["snippet_id"]
    chain = client.get("/graph/callchain", params={"snippet_id": sid}).json()
    assert "callees" in chain


def test_search_on_empty_index_is_409(client):
    assert client.post("/search", json={"query": "x"}).status_code == 409


def test_versions_endpoints(client, history_repo):
    repo, shas = history_repo
    r = client.post("/versions/index", json={"path": str(repo), "repository": "demo", "history": True})
    assert r.status_code == 200, r.text
    v = client.get("/versions/demo").json()
    assert [c["sha"] for c in v["commits"]] == shas and v["latest"] == shas[-1]
    hist = client.post("/search", json={"query": "normalize input", "version": "all", "top_k": 10}).json()
    assert any(x["snippet"]["file"] == "text_utils.py" for x in hist["results"])
    old = client.post("/search", json={"query": "authenticate", "version": shas[0][:8]}).json()
    assert old["results"]
    assert client.post("/search", json={"query": "x", "version": "deadbeef"}).status_code == 404
