"""Official AppsRetrieval evaluation (same as ``python evaluate.py``; kept for the documented layout).

    python scripts/evaluate_mteb.py --config configs/baseline_dense.yaml --mode dense --output appsretrieval_results.json
    python scripts/evaluate_mteb.py --config configs/p0_apps.yaml --mode hybrid --output artifacts/eval/appsretrieval_hybrid.json
"""

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    sys.argv[0] = str(root / "evaluate.py")
    runpy.run_path(str(root / "evaluate.py"), run_name="__main__")
