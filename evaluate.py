"""Official AppsRetrieval (CoIR) evaluation entry point.

Examples::

    # Official submission file (dense encoder path, as in the organisers' template)
    python evaluate.py --config configs/full.yaml --mode dense --output appsretrieval_results.json

    # Full CodeFusion hybrid pipeline evaluated through MTEB's SearchProtocol
    python evaluate.py --config configs/full.yaml --mode hybrid --output artifacts/eval/appsretrieval_hybrid.json

    # Quick smoke test on a subsample (writes *_subset_* files, never the official file)
    python evaluate.py --config configs/cpu_fast.yaml --max-queries 50 --max-corpus 500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from codefusion.config import load_config  # noqa: E402
from codefusion.evaluation.official import run_official  # noqa: E402
from codefusion.evaluation.tracking import log_run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/full.yaml")
    ap.add_argument("--mode", choices=["dense", "hybrid"], default=None,
                    help="dense = AbsEncoder path; hybrid = full pipeline via SearchProtocol (default: from config name)")
    ap.add_argument("--output", default="appsretrieval_results.json")
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--max-corpus", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--predictions-dir", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. dense.model=jina-code-1.5b")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    mode = args.mode or ("dense" if cfg.fusion.method == "dense_only" else "hybrid")
    summary = run_official(cfg, mode=mode, output=args.output, max_queries=args.max_queries, max_corpus=args.max_corpus,
                           batch_size=args.batch_size, predictions_dir=args.predictions_dir)
    log_run(cfg, f"official-{mode}", {"ndcg_at_10": summary["ndcg_at_10"], "mrr_at_10": summary["mrr_at_10"],
                                        "total_s": summary["total_s"]}, extra={"subset": summary["subset"]})
    print(json.dumps({k: v for k, v in summary.items() if k != "scores"}, indent=2, default=str))


if __name__ == "__main__":
    main()
