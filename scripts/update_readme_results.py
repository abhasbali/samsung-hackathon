"""Regenerate the result tables in README.md from measured artifacts (never hand-typed numbers).

    python scripts/update_readme_results.py

Fills the ``<!-- RESULTS:<NAME>:START/END -->`` blocks from artifacts/eval/*_summary.json,
artifacts/ablations/*.md, artifacts/benchmarks/embeddings_*.md and versioning_*.md. Sections
without result files say "Not evaluated yet".
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
NOT_RUN = "*Not evaluated yet.*"


def fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "n/a"


def official() -> str:
    rows = []
    for f in sorted((ART / "eval").glob("*_summary.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        rows.append(d)
    full = [d for d in rows if not d.get("subset")]
    if not full:
        return NOT_RUN if not rows else NOT_RUN + " Only subset smoke tests exist so far."
    lines = ["| Result file | Mode | Configuration | Model | NDCG@10 | MRR@10 | Recall@100 | Queries / docs | Device | Wall time | Peak RAM |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in full:
        s = d.get("scores", {})
        lines.append(f"| `{Path(d['official_json']).as_posix()}` | {d['mode']} | {d['config']} | {d.get('model') or '-'} | "
                     f"**{fmt(d.get('ndcg_at_10'))}** | **{fmt(d.get('mrr_at_10'))}** | {fmt(s.get('recall_at_100'))} | "
                     f"{d['n_queries']} / {d['n_corpus']} | {d.get('config_dump', {}).get('dense', {}).get('device', 'n/a')} | "
                     f"{d['total_s'] / 60:.1f} min | {fmt(d.get('peak_memory_mb'), 0)} MB |")
    lat = [d for d in full if d.get("query_latency_ms")]
    for d in lat:
        q = d["query_latency_ms"]
        lines.append("")
        lines.append(f"Hybrid `{d['config']}` search latency per query (excluding the batched query embedding): "
                     f"P50 {fmt(q.get('p50'), 1)} ms, P95 {fmt(q.get('p95'), 1)} ms.")
    lines.append("")
    lines.append("Full AppsRetrieval test split. The baseline run resumed its embedding cache across devices (partial CPU run, completed on a GPU); later runs reuse that cache, hence their short wall times. MTEB's own JSON is the result file; the table is generated from its summary.")
    return "\n".join(lines)


def latest_md(pattern: str, prefer_full: bool = True) -> str:
    files = sorted((ART / pattern.split("/")[0]).glob(pattern.split("/", 1)[1]))
    if not files:
        return NOT_RUN
    if prefer_full:
        files = sorted(files, key=lambda p: ("full" not in p.name, p.name))
    return "\n\n".join(f.read_text(encoding="utf-8").strip() for f in files)


def ablations() -> str:
    body = latest_md("ablations/ablations_*.md")
    full = ART / "ablations" / "ablations_full.json"
    if body.startswith(NOT_RUN) or not full.exists():
        return body
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from codefusion.evaluation.ablation import default_experiments

    data = json.loads(full.read_text(encoding="utf-8"))
    rows = data.get("rows", data) if isinstance(data, dict) else data
    done = {r["experiment"] for r in rows}
    # H (reranker) runs on the subset by design; everything else missing from the full run is listed.
    missing = [e for e in default_experiments() if e.key not in done and e.key != "H"]
    if missing:
        body += "\n\n**Not evaluated on the full split:** " + "; ".join(f"{e.key} ({e.name})" for e in missing) + "."
    return body


def main() -> int:
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    blocks = {
        "OFFICIAL": official(),
        "ABLATIONS": ablations(),
        "EMBEDDINGS": latest_md("benchmarks/embeddings_*.md"),
        "VERSIONING": latest_md("benchmarks/versioning_*.md", prefer_full=False),
    }
    for name, body in blocks.items():
        pat = re.compile(rf"(<!-- RESULTS:{name}:START -->)(.*?)(<!-- RESULTS:{name}:END -->)", re.S)
        text = pat.sub(lambda m, b=body: f"{m.group(1)}\n{b}\n{m.group(3)}", text)
    readme.write_text(text, encoding="utf-8")
    print("README results updated:", {k: ("n/a" if v.startswith(NOT_RUN) else "filled") for k, v in blocks.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
