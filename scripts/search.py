"""Search an index.

    python scripts/search.py --query "How is input validated before processing?"
    python scripts/search.py -q "Who calls validate_user?" -q "Where is MAX_RETRIES defined?" --top-k 3
    python scripts/search.py -q "How did authentication change between versions?" --version all
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codefusion.cli import search_main

if __name__ == "__main__":
    raise SystemExit(search_main())
