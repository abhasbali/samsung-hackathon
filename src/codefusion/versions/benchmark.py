"""P1 benchmark: full rebuild vs incremental update across consecutive commits of a git repository.

For each commit ``c_i`` (i >= 1) of the selected first-parent range:

* **incremental** — the engine that already holds ``c_{i-1}`` indexes ``c_i``
  (git tree diff -> changed files only -> new snippet versions only are embedded);
* **full rebuild** — a fresh engine indexes ``c_i`` from scratch.

Neither side uses the persistent embedding cache, so every reused embedding on the incremental side
is reuse by snippet identity, and the full-rebuild side really re-encodes the whole commit. Both
engines share one loaded model so model-load time is excluded from both.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from codefusion.config import load_config
from codefusion.logging_utils import get_logger

log = get_logger(__name__)


def benchmark_versions(repo: str | Path, config: str | Path = "configs/full.yaml", overrides: list[str] | None = None,
                       max_commits: int = 6, rev: str = "HEAD") -> dict[str, Any]:
    from codefusion.retrieval.embeddings import CachedEmbedder, create_provider
    from codefusion.retrieval.pipeline import CodeFusionEngine
    from codefusion.versions.git_tracker import GitTracker
    from codefusion.versions.incremental import IncrementalIndexer

    cfg = load_config(config, [*(overrides or []), "dense.cache_embeddings=false"])
    repo = Path(repo).resolve()
    if not GitTracker.is_repo(repo):
        raise ValueError(f"{repo} is not a git repository")
    commits = GitTracker(repo).commits(rev)[-max_commits:]
    if len(commits) < 2:
        raise ValueError("need at least two commits")

    provider = create_provider(cfg.dense) if cfg.dense.enabled else None

    def fresh() -> CodeFusionEngine:
        emb = CachedEmbedder(provider, None, batch_size=cfg.dense.batch_size) if provider else None
        return CodeFusionEngine(cfg, embedder=emb)

    inc_engine = fresh()
    inc = IncrementalIndexer(inc_engine, repo, repo.name)
    first = inc.index_commit(commits[0].sha)
    steps: list[dict[str, Any]] = []
    for c in commits[1:]:
        t0 = time.perf_counter()
        st = inc.index_commit(c.sha)
        inc_engine.finalize()
        inc_s = time.perf_counter() - t0

        full_engine = fresh()
        t1 = time.perf_counter()
        full_st = IncrementalIndexer(full_engine, repo, repo.name).index_commit(c.sha)
        full_engine.finalize()
        full_s = time.perf_counter() - t1

        steps.append({
            "commit": c.sha[:10], "message": (c.message or "")[:80],
            "files_total": st.files_total, "files_changed": st.files_changed, "files_added": st.files_added,
            "files_deleted": st.files_deleted, "snippets_live": st.snippets_live,
            "incremental_s": round(inc_s, 3), "full_rebuild_s": round(full_s, 3),
            "speedup": round(full_s / inc_s, 2) if inc_s > 0 else None,
            "incremental_embeddings_computed": st.embeddings_computed,
            "incremental_snippets_reused": st.snippets_reused,
            "full_embeddings_computed": full_st.embeddings_computed,
            "lineage_edges": st.lineage_edges, "lineage_relations": st.lineage_relations,
        })
        log.info("version step", extra={"data": {k: steps[-1][k] for k in ("commit", "incremental_s", "full_rebuild_s", "speedup")}})
        del full_engine

    tot_inc = sum(s["incremental_s"] for s in steps)
    tot_full = sum(s["full_rebuild_s"] for s in steps)
    return {
        "repository": repo.name, "config": cfg.name, "model": cfg.dense.model if cfg.dense.enabled else None,
        "commits": len(commits), "initial_index": {"commit": commits[0].sha[:10], "seconds": first.seconds,
                                                   "snippets": first.snippets_live, "embeddings": first.embeddings_computed},
        "steps": steps,
        "totals": {"incremental_s": round(tot_inc, 3), "full_rebuild_s": round(tot_full, 3),
                   "speedup": round(tot_full / tot_inc, 2) if tot_inc else None,
                   "embeddings_incremental": sum(s["incremental_embeddings_computed"] for s in steps),
                   "embeddings_full": sum(s["full_embeddings_computed"] for s in steps)},
        "store_snippet_versions": len(inc_engine.store.snippets),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def write_report(result: dict[str, Any], out_dir: str | Path = "artifacts/benchmarks") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"versioning_{result['repository']}_{result['model'] or 'nodense'}".replace("/", "_")
    (out / f"{stem}.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    lines = [f"### Incremental vs full rebuild: {result['repository']} ({result['model']})", "",
             "| Commit | Files changed/added/deleted | Live snippets | Incremental s | Full rebuild s | Speed-up | Embeddings (inc / full) |",
             "|---|---|---|---|---|---|---|"]
    for s in result["steps"]:
        lines.append(f"| {s['commit']} | {s['files_changed']}/{s['files_added']}/{s['files_deleted']} of {s['files_total']} | "
                     f"{s['snippets_live']} | {s['incremental_s']:.3f} | {s['full_rebuild_s']:.3f} | {s['speedup']}x | "
                     f"{s['incremental_embeddings_computed']} / {s['full_embeddings_computed']} |")
    t = result["totals"]
    lines += ["", f"Total: incremental {t['incremental_s']:.2f}s vs full {t['full_rebuild_s']:.2f}s "
              f"({t['speedup']}x); embeddings computed {t['embeddings_incremental']} vs {t['embeddings_full']}.", ""]
    (out / f"{stem}.md").write_text("\n".join(lines), encoding="utf-8")
    return out / f"{stem}.json"
