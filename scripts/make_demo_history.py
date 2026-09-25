"""Create a real git repository from the example snapshots (v1 -> v2 -> v3).

    python scripts/make_demo_history.py                   # -> artifacts/demo_history_repo
    python scripts/index_repo.py artifacts/demo_history_repo --history
    python scripts/search.py -q "How did authentication change between versions?"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codefusion.versions.demo_history import make_demo_history  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", nargs="?", default="artifacts/demo_history_repo")
    args = ap.parse_args()
    shas = make_demo_history(args.target)
    for i, s in enumerate(shas, start=1):
        print(f"v{i}  {s}")
    print(f"created {args.target} with {len(shas)} commits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
