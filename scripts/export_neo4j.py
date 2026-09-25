"""Export a saved CodeFusion index's code graph (incl. EVOLVED_TO lineage) to Neo4j. Optional.

    docker compose --profile neo4j up -d neo4j
    pip install neo4j
    python scripts/export_neo4j.py --index-dir artifacts/index

Then open http://localhost:7474 and try:
    MATCH p=(:CFNode)-[:REL {type: 'EVOLVED_TO'}]->(:CFNode) RETURN p LIMIT 50
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codefusion.config import load_config  # noqa: E402
from codefusion.graph.neo4j_adapter import Neo4jExporter, Neo4jUnavailable  # noqa: E402
from codefusion.retrieval.pipeline import CodeFusionEngine  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index-dir", default="artifacts/index")
    ap.add_argument("--config", default=None, help="defaults to the config stored in the index")
    args = ap.parse_args()
    cfg = load_config(args.config, ["dense.enabled=false"]) if args.config else None
    eng = CodeFusionEngine.load(args.index_dir, cfg, allow_model_fallback=True)
    if eng.graph is None:
        print("index has no code graph (graph/structural/versions all disabled)")
        return 1
    try:
        exp = Neo4jExporter()
    except Neo4jUnavailable as exc:
        print(f"Neo4j unavailable: {exc}")
        return 2
    try:
        print(json.dumps(exp.export(eng.graph), indent=2))
    finally:
        exp.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
