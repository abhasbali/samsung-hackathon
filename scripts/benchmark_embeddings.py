"""Benchmark the candidate code embedding models on AppsRetrieval.

    # default: seeded subset (200 queries, their positives + distractors up to 2000 docs)
    python scripts/benchmark_embeddings.py

    # specific models / full test split
    python scripts/benchmark_embeddings.py --models jina-code-0.5b jina-v2-base-code --full

Models: qodo-1.5b, jina-code-1.5b, jina-code-0.5b, jina-v2-base-code (or any HF id).
A model that cannot be downloaded/loaded is reported as NOT RUN and the others continue.
Writes artifacts/benchmarks/embeddings_<dataset>.{json,md}.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codefusion.evaluation.benchmark import CANDIDATES, run_benchmark, to_markdown  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=CANDIDATES)
    ap.add_argument("--config", default="configs/baseline_dense.yaml")
    ap.add_argument("--max-queries", type=int, default=200)
    ap.add_argument("--max-corpus", type=int, default=2000)
    ap.add_argument("--full", action="store_true", help="use the full AppsRetrieval test split")
    ap.add_argument("--latency-queries", type=int, default=50)
    ap.add_argument("--no-cache", action="store_true", help="always re-encode (true indexing time)")
    ap.add_argument("--out", default="artifacts/benchmarks")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. dense.max_seq_length=512")
    args = ap.parse_args()
    mq, mc = (None, None) if args.full else (args.max_queries, args.max_corpus)
    rows = run_benchmark(args.models, args.config, mq, mc, args.out, args.set, args.latency_queries, not args.no_cache)
    print(to_markdown(rows, rows[0]["dataset"] if rows else "none"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
