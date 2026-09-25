from pathlib import Path

from codefusion.parsing.chunker import chunk_source
from codefusion.retrieval.pipeline import CodeFusionEngine
from codefusion.versions.git_tracker import GitTracker
from codefusion.versions.incremental import IncrementalIndexer
from codefusion.versions.lineage import LineageMatcher
from conftest import tiny_config


def _find(eng, qualified, commit=None):
    out = [i for i, s in enumerate(eng.store.snippets) if s.qualified_name == qualified and (commit is None or commit in s.commits)]
    assert out, qualified
    return out


def test_subdirectory_of_a_repo_is_not_treated_as_the_repo(history_repo):
    """Indexing a folder inside some enclosing git repo must index that folder, not the whole repo."""
    repo, _ = history_repo
    sub = repo / "nested"
    sub.mkdir()
    assert GitTracker.is_repo(repo)
    assert not GitTracker.is_repo(sub)


def test_git_diff_detects_rename(history_repo):
    repo, shas = history_repo
    g = GitTracker(repo)
    assert [c.sha for c in g.commits()] == shas
    changes = {(c.status, c.path, c.old_path) for c in g.diff(shas[0], shas[1])}
    assert ("R", "preprocessing.py", "text_utils.py") in changes
    assert g.renames(shas[0], shas[1]) == {"preprocessing.py": "text_utils.py"}
    assert "auth.py" in g.tree(shas[2])


def test_incremental_indexing_reuses_unchanged(history_engine):
    eng, shas, stats = history_engine
    s1, s2, s3 = stats
    assert s1.files_added == s1.files_total and s1.base_commit is None
    # model.py is unchanged in every commit -> reused, never re-parsed
    assert s2.files_reused >= 1 and s3.files_reused >= 1
    # Only new snippet versions are embedded.
    for s in stats:
        assert s.embeddings_computed <= s.snippets_new_versions
    # unchanged snippets inside changed files are reused, not re-added
    assert s3.snippets_reused > 0
    assert len(eng.store.snippets) == sum(s.snippets_new_versions for s in stats)


def test_reindexing_same_commit_is_a_noop(history_repo):
    repo, shas = history_repo
    eng = CodeFusionEngine(tiny_config())
    idx = IncrementalIndexer(eng, repo, "demo")
    idx.index_commit(shas[-1])
    again = idx.index_commit(shas[-1])
    assert again.files_changed == 0 and again.files_added == 0
    assert again.snippets_new_versions == 0 and again.embeddings_computed == 0


def test_unchanged_snippets_are_not_re_embedded_on_edit(tmp_path, sample_repo):
    work = tmp_path / "wt"
    work.mkdir()
    for f in Path(sample_repo).glob("*.py"):
        (work / f.name).write_text(f.read_text())
    eng = CodeFusionEngine(tiny_config())
    idx = IncrementalIndexer(eng, work, "wt")
    full = idx.index_worktree()
    computed_before = eng.embedder.stats.computed
    p = work / "model.py"
    p.write_text(p.read_text().replace("BIAS = -1.5", "BIAS = -2.0"))
    upd = idx.index_worktree()
    assert upd.files_changed == 1 and upd.files_reused == full.files_total - 1
    assert upd.snippets_new_versions == 1  # only the module block holding BIAS changed
    assert eng.embedder.stats.computed - computed_before == 1


def test_lineage_tracks_renames(history_engine):
    eng, shas, _ = history_engine
    old = _find(eng, "AuthService.authenticate")[0]
    new = _find(eng, "AuthService.validate_user")[0]
    assert eng.store.snippets[old].lineage_id == eng.store.snippets[new].lineage_id
    rels = {(o, n): r for o, n, r, _ in eng.store.lineage}
    assert rels[(old, new)] == "renamed"
    chain = eng.graph.lineage(new)
    assert chain.index(old) < chain.index(new)
    # file rename text_utils.py -> preprocessing.py keeps normalize_input's lineage
    v1 = [i for i in _find(eng, "normalize_input") if eng.store.snippets[i].file_path == "text_utils.py"][0]
    v2 = [i for i in _find(eng, "normalize_input") if eng.store.snippets[i].file_path == "preprocessing.py"][0]
    assert eng.store.snippets[v1].lineage_id == eng.store.snippets[v2].lineage_id
    assert rels.get((v1, v2)) in ("file_renamed", "moved")


def test_lineage_matcher_cascade():
    a = chunk_source("def authenticate(u, p):\n    return check(u, p) and log(u)\n", "a.py", "r", "c1")[0]
    b = chunk_source("def verify_user(u, p):\n    return check(u, p) and log(u)\n", "a.py", "r", "c2")[0]
    c = chunk_source("def unrelated():\n    return 42\n", "a.py", "r", "c2")[0]
    edges = LineageMatcher(0.6).match([(0, a)], [(1, b), (2, c)])
    assert [(e.old_idx, e.new_idx, e.relation) for e in edges] == [(0, 1, "renamed")]


def test_version_scoped_search(history_engine):
    eng, shas, _ = history_engine
    v1 = eng.search("authenticate user password", top_k=10, version=shas[0])
    files = {r.snippet["file"] for r in v1.results}
    assert "text_utils.py" in files or "auth.py" in files
    assert all(shas[0] in r.snippet["commits"] for r in v1.results)
    latest = eng.search("normalize input", top_k=10)
    assert all(r.snippet["file"] != "text_utils.py" for r in latest.results)
    hist = eng.search("normalize input", top_k=10, version="all")
    assert any(r.snippet["file"] == "text_utils.py" for r in hist.results)


def test_evolution_query_returns_versions_with_lineage(history_engine):
    eng, shas, _ = history_engine
    resp = eng.search("How did authentication change between versions?", top_k=8)
    assert resp.intent == "EVOLUTION" and resp.version_scope == "all"
    lineages = [r.snippet["lineage_id"] for r in resp.results]
    assert len(lineages) != len(set(lineages))  # several versions of one lineage are kept
    assert any(r.lineage and len(r.lineage) >= 2 for r in resp.results)


def test_latest_scope_collapses_history(history_engine):
    eng, _, _ = history_engine
    resp = eng.search("user authentication", top_k=10, version="all")
    # non-evolution query over all versions: lineage collapse keeps one version per lineage
    lineages = [r.snippet["lineage_id"] for r in resp.results]
    assert len(lineages) == len(set(lineages))


def test_save_and_load_roundtrip(history_repo, history_engine, tmp_path):
    repo, _ = history_repo
    eng, shas, _ = history_engine
    eng.save(tmp_path / "idx")
    eng2 = CodeFusionEngine.load(tmp_path / "idx", tiny_config())
    assert len(eng2.store.snippets) == len(eng.store.snippets)
    assert eng2.store.latest == eng.store.latest
    q = "Who calls validate_user?"
    assert [r.snippet["snippet_id"] for r in eng.search(q).results] == [r.snippet["snippet_id"] for r in eng2.search(q).results]
    # Incremental state survives a reload: re-indexing HEAD adds nothing.
    again = IncrementalIndexer(eng2, repo, "demo").index_commit(shas[-1])
    assert again.snippets_new_versions == 0 and again.files_changed == 0
