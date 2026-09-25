"""Reproduce every stored result, in priority order, as one resumable pipeline.

    python scripts/run_experiment_suite.py                  # everything
    python scripts/run_experiment_suite.py --wait-pid 1234  # start after another run finishes
    python scripts/run_experiment_suite.py --steps ablations p0

Steps (each is skipped when its output already exists, unless --force):

  official    evaluate.py, baseline dense (jina-code-0.5b)        -> artifacts/eval/appsretrieval_results.json
  demo        run_demo_queries.py with configs/full.yaml           -> artifacts/demo/demo_transcript.md
  ablations   run_ablations.py on the full test split (no H)       -> artifacts/ablations/ablations_full.*
  reranker    A/D/G/H on a seeded subset (cross-encoder is slow)   -> artifacts/ablations/ablations_subset_*.*
  p0          select_p0_config.py + evaluate.py with p0_apps.yaml  -> artifacts/eval/appsretrieval_p0.json
  embeddings  benchmark_embeddings.py on a seeded subset           -> artifacts/benchmarks/embeddings_*.*
  versioning  benchmark_versions.py on a real repository           -> artifacts/benchmarks/versioning_*.*

Dense steps share the SQLite embedding cache, so the corpus is encoded exactly once.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
LOG = ROOT / "artifacts" / "eval" / "logs" / "suite.log"


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(name: str, args: list[str]) -> bool:
    log(f"[{name}] start: {' '.join(args)}")
    t0 = time.time()
    with (LOG.parent / f"suite_{name}.log").open("a", encoding="utf-8") as out:
        rc = subprocess.run([PY, *args], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT).returncode
    log(f"[{name}] exit {rc} after {(time.time() - t0) / 60:.1f} min")
    return rc == 0


def pid_alive(pid: int) -> bool:
    try:
        import psutil

        return psutil.pid_exists(pid)
    except ImportError:
        return True


STEPS = ["official", "demo", "ablations", "reranker", "p0", "embeddings", "versioning"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="*", default=STEPS, choices=STEPS)
    ap.add_argument("--wait-pid", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--subset-queries", type=int, default=300)
    ap.add_argument("--subset-corpus", type=int, default=3000)
    ap.add_argument("--bench-queries", type=int, default=100)
    ap.add_argument("--bench-corpus", type=int, default=500)
    ap.add_argument("--version-repo", default="https://github.com/pallets/click")
    args = ap.parse_args()

    if args.wait_pid:
        log(f"waiting for pid {args.wait_pid}")
        while pid_alive(args.wait_pid):
            time.sleep(60)
        log(f"pid {args.wait_pid} finished")

    def done(p: str) -> bool:
        return (ROOT / p).exists() and not args.force

    ok = True
    for step in args.steps:
        if step == "official":
            if done("artifacts/eval/appsretrieval_results.json"):
                log("[official] already present, skipping")
                continue
            ok &= run("official", ["evaluate.py", "--config", "configs/baseline_dense.yaml", "--mode", "dense",
                                   "--output", "artifacts/eval/appsretrieval_results.json"])
        elif step == "demo":
            if done("artifacts/demo/demo_transcript.md"):
                log("[demo] already present, skipping")
                continue
            ok &= run("demo", ["scripts/run_demo_queries.py", "--config", "configs/full.yaml"])
        elif step == "ablations":
            if done("artifacts/ablations/ablations_full.json"):
                log("[ablations] already present, skipping")
                continue
            ok &= run("ablations", ["scripts/run_ablations.py", "--config", "configs/full.yaml",
                                    "--only", "A", "B", "C", "D", "D2", "E", "F", "G", "I", "J", "K", "L", "M", "N"])
        elif step == "reranker":
            tag = f"artifacts/ablations/ablations_subset_q{args.subset_queries}"
            if any((ROOT / "artifacts/ablations").glob(f"ablations_subset_q{args.subset_queries}_*.json")) and not args.force:
                log(f"[reranker] {tag}* already present, skipping")
                continue
            ok &= run("reranker", ["scripts/run_ablations.py", "--config", "configs/full.yaml", "--only", "A", "D", "G", "H",
                                   "--max-queries", str(args.subset_queries), "--max-corpus", str(args.subset_corpus)])
        elif step == "p0":
            if not (ROOT / "artifacts/ablations/ablations_full.json").exists():
                log("[p0] no full ablation results; skipping")
                continue
            if not run("p0-select", ["scripts/select_p0_config.py"]):
                ok = False
                continue
            if done("artifacts/eval/appsretrieval_p0.json"):
                log("[p0] official p0 result already present, skipping")
                continue
            ok &= run("p0", ["evaluate.py", "--config", "configs/p0_apps.yaml", "--output", "artifacts/eval/appsretrieval_p0.json"])
        elif step == "embeddings":
            if any((ROOT / "artifacts/benchmarks").glob(f"embeddings_subset_q{args.bench_queries}_*.json")) and not args.force:
                log("[embeddings] already present, skipping")
                continue
            ok &= run("embeddings", ["scripts/benchmark_embeddings.py", "--max-queries", str(args.bench_queries),
                                     "--max-corpus", str(args.bench_corpus),
                                     "--models", "jina-code-0.5b", "jina-v2-base-code", "jina-code-1.5b", "qodo-1.5b"])
        elif step == "versioning":
            repo = ROOT / "artifacts" / "repos" / Path(args.version_repo).name
            if not repo.exists():
                log(f"[versioning] cloning {args.version_repo}")
                r = subprocess.run(["git", "clone", "--quiet", args.version_repo, str(repo)], cwd=ROOT)
                if r.returncode != 0:
                    log("[versioning] clone failed; skipping")
                    ok = False
                    continue
            ok &= run("versioning", ["scripts/benchmark_versions.py", "--repo", str(repo), "--commits", "6",
                                     "--config", "configs/full.yaml", "--set", "dense.model=minilm", "dense.max_seq_length=256"])
    log(f"suite finished ok={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
