from pathlib import Path

import pytest

from codefusion.config import apply_overrides, load_config

CONFIGS = sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=[p.name for p in CONFIGS])
def test_all_configs_validate(path):
    cfg = load_config(path)
    assert cfg.name
    assert cfg.enabled_components()


def test_required_configs_exist():
    names = {p.name for p in CONFIGS}
    assert {"baseline_dense.yaml", "hybrid.yaml", "full.yaml", "cpu_fast.yaml", "experimental_cpg.yaml"} <= names


def test_overrides_and_extends():
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "cpu_fast.yaml", ["dense.top_k=7", "graph.enabled=false"])
    assert cfg.dense.top_k == 7 and not cfg.graph.enabled
    assert cfg.dense.model == "jina-v2-base-code" and cfg.structural.enabled  # inherited from full.yaml
    assert apply_overrides({}, ["a.b.c=1"]) == {"a": {"b": {"c": 1}}}


def test_device_env_var(monkeypatch):
    monkeypatch.setenv("CODEFUSION_DEVICE", "cuda")
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "baseline_dense.yaml")
    assert cfg.dense.device == "cuda" and cfg.reranker.device == "cuda"
    assert load_config(None, ["dense.device=cpu"]).dense.device == "cpu"  # explicit override wins


def test_unknown_keys_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        load_config(None, ["dense.not_a_field=1"])


def test_optional_modules_import_without_optional_deps():
    import codefusion.experimental.joern as joern
    import codefusion.experimental.late_interaction as li
    import codefusion.graph.neo4j_adapter as neo

    assert joern.joern_available() in (True, False)
    assert hasattr(li, "LateInteractionRetriever")
    assert hasattr(neo, "Neo4jExporter")
