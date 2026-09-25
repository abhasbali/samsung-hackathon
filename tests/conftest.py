from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codefusion.config import load_config  # noqa: E402
from codefusion.retrieval.pipeline import CodeFusionEngine  # noqa: E402
from codefusion.versions.demo_history import make_demo_history  # noqa: E402
from codefusion.versions.incremental import IncrementalIndexer, index_path  # noqa: E402

SAMPLE_REPO = ROOT / "examples" / "sample_repo"


def tiny_config(**overrides):
    """Hashing-embedding config: exercises every stage without downloading a model."""
    ov = [f"{k}={v}" for k, v in overrides.items()]
    return load_config(ROOT / "configs" / "test_tiny.yaml", ov)


@pytest.fixture(scope="session")
def sample_repo() -> Path:
    return SAMPLE_REPO


@pytest.fixture()
def engine():
    return CodeFusionEngine(tiny_config())


@pytest.fixture(scope="session")
def indexed_engine():
    eng = CodeFusionEngine(tiny_config())
    index_path(eng, SAMPLE_REPO, "sample")
    return eng


@pytest.fixture()
def history_repo(tmp_path) -> tuple[Path, list[str]]:
    repo = tmp_path / "demo_repo"
    shas = make_demo_history(repo)
    return repo, shas


@pytest.fixture()
def history_engine(history_repo):
    repo, shas = history_repo
    eng = CodeFusionEngine(tiny_config())
    idx = IncrementalIndexer(eng, repo, "demo")
    stats = idx.index_history()
    return eng, shas, stats
