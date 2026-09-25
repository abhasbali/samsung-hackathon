"""EXPERIMENTAL: enrich a saved index's code graph with Joern CPG facts (optional).

    # needs Joern on PATH (Java 17+):
    #   curl -L https://github.com/joernio/joern/releases/latest/download/joern-install.sh | sh
    python scripts/index_repo.py examples/sample_repo
    python scripts/joern_enrich.py examples/sample_repo --index-dir artifacts/index

Adds CALLS edges that Joern resolved but the AST pass missed, plus parameter -> call data-flow
evidence, then saves the index. Without Joern it exits cleanly with a message; nothing else in
CodeFusion depends on it.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codefusion.experimental.joern import apply_to_graph, joern_available, load_cpg_facts, run_joern  # noqa: E402
from codefusion.retrieval.pipeline import CodeFusionEngine  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo")
    ap.add_argument("--index-dir", default="artifacts/index")
    ap.add_argument("--facts", default=None, help="reuse an existing Joern facts JSON instead of running Joern")
    args = ap.parse_args()
    facts = Path(args.facts) if args.facts else None
    if facts is None:
        if not joern_available():
            print("Joern is not installed (joern not on PATH); the CPG experiment is skipped. See docs/evaluation.md.")
            return 0
        facts = run_joern(args.repo)
        if facts is None:
            print("Joern run failed; see logs. The index is unchanged.")
            return 1
    eng = CodeFusionEngine.load(args.index_dir, allow_model_fallback=True)
    counters = apply_to_graph(eng, load_cpg_facts(facts, args.repo))
    eng.save(args.index_dir)
    print(json.dumps({"facts": str(facts), **counters}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
