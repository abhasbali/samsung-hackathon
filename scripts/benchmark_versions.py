"""P1 benchmark: full rebuild vs incremental update over consecutive commits.

    # the bundled 3-commit demo history (created on the fly)
    python scripts/benchmark_versions.py

    # a real repository, last 8 first-parent commits, with a real embedding model
    git clone https://github.com/pallets/click artifacts/repos/click
    python scripts/benchmark_versions.py --repo artifacts/repos/click --commits 8 --config configs/cpu_fast.yaml

Writes artifacts/benchmarks/versioning_<repo>_<model>.{json,md}.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codefusion.versions.benchmark import benchmark_versions, write_report  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=None, help="git repository (default: build the demo history)")
    ap.add_argument("--config", default="configs/test_tiny.yaml")
    ap.add_argument("--commits", type=int, default=6)
    ap.add_argument("--rev", default="HEAD")
    ap.add_argument("--out", default="artifacts/benchmarks")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    repo = args.repo
    if repo is None:
        from codefusion.versions.demo_history import make_demo_history

        repo = "artifacts/demo_history_repo"
        make_demo_history(repo)
    res = benchmark_versions(repo, args.config, args.set, args.commits, args.rev)
    path = write_report(res, args.out)
    print(path.with_suffix(".md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
