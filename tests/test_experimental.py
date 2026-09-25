import json

from codefusion.experimental.joern import apply_to_graph, joern_available, load_cpg_facts, run_joern
from codefusion.experimental.late_interaction import HashingTokenEncoder, LateInteractionRetriever, maxsim


def test_joern_absent_is_graceful(tmp_path):
    if not joern_available():
        assert run_joern(tmp_path) is None


def test_cpg_fact_ingestion(indexed_engine, tmp_path, sample_repo):
    """Ingestion of the JSON format emitted by JOERN_SCRIPT (fixture, not product data)."""
    facts = [
        {"name": "preprocess_input", "fullName": "preprocessing.py:<module>.preprocess_input", "file": "preprocessing.py",
         "lineStart": 36, "lineEnd": 39, "callees": ["preprocessing.py:<module>.normalize_input"],
         "paramFlowsInto": ["preprocessing.py:<module>.normalize_input"]},
        {"name": "normalize_input", "fullName": "preprocessing.py:<module>.normalize_input", "file": "preprocessing.py",
         "lineStart": 10, "lineEnd": 20, "callees": [], "paramFlowsInto": []},
    ]
    p = tmp_path / "facts.json"
    p.write_text(json.dumps(facts))
    methods = load_cpg_facts(p)
    counts = apply_to_graph(indexed_engine, methods)
    assert counts["mapped"] == 2 and counts["flows"] == 1


def test_late_interaction_maxsim():
    li = LateInteractionRetriever(HashingTokenEncoder(64))
    li.add_documents(["def parse_config(path): return yaml.load(path)", "def send_email(to, body): smtp.send(to, body)"])
    assert li.search("parse the yaml config", 1)[0][0] == 0
    assert li.rerank("send an email", [0, 1])[0][0] == 1
    q = HashingTokenEncoder(64).encode(["abc"])[0]
    assert abs(maxsim(q, q) - 1.0) < 1e-2
    assert li.memory_mb() > 0
