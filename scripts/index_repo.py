"""Index a repository or directory.

    python scripts/index_repo.py examples/sample_repo
    python scripts/index_repo.py path/to/git/repo --history          # every commit, incrementally
    python scripts/index_repo.py examples/sample_repo --config configs/cpu_fast.yaml
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codefusion.cli import index_main

if __name__ == "__main__":
    raise SystemExit(index_main())
