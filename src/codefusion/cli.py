"""Console entry points (``codefusion-index`` / ``codefusion-search``) wrapping the scripts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from codefusion.config import load_config
from codefusion.retrieval.pipeline import CodeFusionEngine
from codefusion.versions.incremental import index_path


def _engine(cfg_path: str, index_dir: str | None, overrides: list[str], allow_fallback: bool) -> tuple[CodeFusionEngine, Path]:
    cfg = load_config(cfg_path, overrides)
    idx = Path(index_dir or cfg.index_dir)
    if (idx / "index.db").exists():
        return CodeFusionEngine.load(idx, cfg, allow_model_fallback=allow_fallback), idx
    return CodeFusionEngine(cfg, allow_model_fallback=allow_fallback), idx


def index_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Index a repository (git: HEAD or full history; else the directory).")
    ap.add_argument("path")
    ap.add_argument("--config", default="configs/full.yaml")
    ap.add_argument("--index-dir", default=None)
    ap.add_argument("--repository", default=None)
    ap.add_argument("--history", action="store_true", help="index every commit (incrementally) for version retrieval")
    ap.add_argument("--max-commits", type=int, default=None)
    ap.add_argument("--fresh", action="store_true", help="ignore an existing index")
    ap.add_argument("--allow-fallback", action="store_true", help="use hashing embeddings if the model is unavailable")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    idx_dir = Path(a.index_dir or cfg.index_dir)
    if a.fresh or not (idx_dir / "index.db").exists():
        eng = CodeFusionEngine(cfg, allow_model_fallback=a.allow_fallback)
    else:
        eng = CodeFusionEngine.load(idx_dir, cfg, allow_model_fallback=a.allow_fallback)
    t0 = time.perf_counter()
    stats = index_path(eng, a.path, a.repository, history=a.history, max_commits=a.max_commits)
    eng.save(idx_dir)
    print(json.dumps({"seconds": round(time.perf_counter() - t0, 3), "updates": [s.to_dict() for s in stats],
                      "stats": eng.stats()}, indent=2, default=str))
    return 0


def _print_human(resp: dict, show_code: int) -> None:
    print(f"\nQuery : {resp['query']}")
    print(f"Intent: {resp['intent']} (confidence {resp['intent_confidence']})   scope: {resp['version_scope']}")
    t = resp["timings_ms"]
    print("Timing: " + "  ".join(f"{k}={v:.1f}ms" for k, v in t.items()))
    print("Candidates: " + "  ".join(f"{k}={v}" for k, v in resp["candidate_counts"].items()))
    for r in resp["results"]:
        s = r["snippet"]
        prov = r["provenance"]
        ranks = ", ".join(f"{k[:-5]}#{v}" for k, v in prov.items() if k.endswith("_rank") and k not in ("fused_rank", "final_rank") and v)
        print(f"\n#{r['rank']}  {s['file']}:{s['lines'][0]}-{s['lines'][1]}  {s['symbol']}  [{s['type']}]  "
              f"commit={str(s['commit'])[:8]}  score={r['score']:.5f}")
        print(f"    retrievers: {ranks or '-'}")
        ev = r["structural_evidence"][:3] + r["graph_evidence"][:2]
        if ev:
            print("    evidence  : " + "; ".join(ev))
        if r.get("lineage"):
            print("    lineage   : " + " -> ".join(f"{x['symbol']}@{str(x['commit'])[:7]}" for x in r["lineage"]))
        if show_code:
            for line in s["content"].splitlines()[:show_code]:
                print("      | " + line)


def search_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Search an index built with index_repo.py.")
    ap.add_argument("--query", "-q", required=True, action="append", help="may be given multiple times")
    ap.add_argument("--config", default="configs/full.yaml")
    ap.add_argument("--index-dir", default=None)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--version", default=None, help="commit sha/prefix, 'latest' or 'all'")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--show-code", type=int, default=8, help="lines of code to print per result")
    ap.add_argument("--allow-fallback", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args(argv)
    eng, idx = _engine(a.config, a.index_dir, a.set, a.allow_fallback)
    if not eng.store.snippets:
        print(f"index at {idx} is empty; run scripts/index_repo.py first", file=sys.stderr)
        return 2
    for q in a.query:
        resp = eng.search(q, top_k=a.top_k, version=a.version, explain=True).to_dict(explain=True)
        if a.json:
            print(json.dumps(resp, indent=2, default=str))
        else:
            _print_human(resp, a.show_code)
    return 0
