"""Run the A-N ablation matrix on AppsRetrieval.

    # full test split (dense experiments reuse the embedding cache filled by evaluate.py)
    python scripts/run_ablations.py --config configs/full.yaml

    # quick subset, only some experiments
    python scripts/run_ablations.py --max-queries 200 --max-corpus 2000 --only A B D G

    # CI-sized run with hashing embeddings (no downloads)
    python scripts/run_ablations.py --config configs/test_tiny.yaml --max-queries 20 --max-corpus 150

Writes artifacts/ablations/ablations_<dataset>.{json,csv,md}. Experiments that cannot run
(reranker/Joern/late interaction unavailable) are listed with a NOT RUN status and no metrics.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codefusion.evaluation.ablation import default_experiments, run_ablations, to_markdown  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/full.yaml", help="base config every experiment overrides")
    ap.add_argument("--only", nargs="*", default=None, help="experiment keys, e.g. A B D2 G")
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--max-corpus", type=int, default=None)
    ap.add_argument("--out", default="artifacts/ablations")
    ap.add_argument("--set", nargs="*", default=[], help="extra overrides applied to the base config")
    ap.add_argument("--reference", default="artifacts/eval/appsretrieval_results.json",
                    help="official MTEB result used to cross-check experiment A")
    ap.add_argument("--list", action="store_true", help="print the experiment matrix and exit")
    args = ap.parse_args()

    if args.list:
        for e in default_experiments():
            print(f"{e.key:>3}  {e.name:<40} {e.description}")
            print("     " + " ".join(e.overrides))
        return 0
    rows = run_ablations(args.config, only=args.only, max_queries=args.max_queries, max_corpus=args.max_corpus,
                         out_dir=args.out, extra_overrides=args.set, reference_official=args.reference)
    tag = rows[0]["dataset"] if rows else "none"
    print(to_markdown(rows, tag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
