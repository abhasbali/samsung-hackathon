"""Render the README charts from measured artifacts (never hand-typed numbers).

    python scripts/make_charts.py

Writes docs/charts/<name>-light.svg and <name>-dark.svg; the README shows the right one per
GitHub theme through <picture>. Dependency-free: the SVG is written directly. A chart whose
source file is missing is skipped. Called by scripts/update_readme_results.py as well.
"""

from __future__ import annotations

import json
import math
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OUT = ROOT / "docs" / "charts"

# Validated (scripts/validate_palette.js, dataviz skill) against GitHub's page surfaces:
# light #ffffff, dark #0d1117 - all checks pass for the blue/orange pair in both modes.
THEMES = {
    "light": {"s1": "#2a78d6", "s2": "#eb6834", "muted_mark": "#898781", "surface": "#ffffff",
              "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#6e6d68", "grid": "#e1e0d9", "axis": "#c3c2b7"},
    "dark": {"s1": "#3987e5", "s2": "#d95926", "muted_mark": "#898781", "surface": "#0d1117",
             "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#9d9b94", "grid": "#2c2c2a", "axis": "#383835"},
}
FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
W = 760


def _load(rel: str):
    p = ART / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _svg(title: str, subtitle: str, body: list[str], height: int, desc: str, t: dict) -> str:
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" viewBox="0 0 {W} {height}" '
        f'role="img" aria-labelledby="t d" font-family="{FONT}">',
        f"<title id=\"t\">{escape(title)}</title><desc id=\"d\">{escape(desc)}</desc>",
        f'<text x="24" y="30" font-size="16" font-weight="600" fill="{t["ink"]}">{escape(title)}</text>',
        f'<text x="24" y="50" font-size="12.5" fill="{t["ink2"]}">{escape(subtitle)}</text>',
        *body, "</svg>"])


def _hbar(x0: float, y: float, length: float, h: float, color: str, r: float = 4) -> str:
    """Horizontal bar: square at the baseline (x0), 4px rounded data-end."""
    if length <= 0:
        return ""
    r = min(r, length, h / 2)
    x1 = x0 + length
    return (f'<path d="M{x0:.1f},{y:.1f} H{x1 - r:.1f} Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f} '
            f'V{y + h - r:.1f} Q{x1:.1f},{y + h:.1f} {x1 - r:.1f},{y + h:.1f} H{x0:.1f} Z" fill="{color}"/>')


def _text(x, y, s, t, size=12, fill="ink2", anchor="start", weight=400) -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{t[fill]}" text-anchor="{anchor}" '
            f'font-weight="{weight}" style="font-variant-numeric: tabular-nums">{escape(str(s))}</text>')


def _legend(x: float, y: float, items: list[tuple[str, str]], t: dict) -> list[str]:
    out = []
    for label, key in items:
        out.append(f'<rect x="{x:.1f}" y="{y - 9:.1f}" width="10" height="10" rx="2" fill="{t[key]}"/>')
        out.append(_text(x + 15, y, label, t, 12, "ink2"))
        x += 30 + 7 * len(label)
    return out


# --------------------------------------------------------------------------- charts
def ablations(t: dict):
    data = _load("ablations/ablations_full.json")
    if not data:
        return None
    rows = [r for r in data["rows"] if r.get("ndcg_at_10") is not None]
    best = max(rows, key=lambda r: r["ndcg_at_10"])
    left, right, top, bh, gap = 250, 90, 78, 22, 12
    plot_w = W - left - right
    height = top + len(rows) * (bh + gap) + 40
    body = []
    for v in (0, 0.2, 0.4, 0.6, 0.8, 1.0):  # hairline grid, recessive
        x = left + v * plot_w
        body.append(f'<line x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{height - 34}" stroke="{t["grid"]}" stroke-width="1"/>')
        body.append(_text(x, height - 18, f"{v:.1f}", t, 11, "muted", "middle"))
    for i, r in enumerate(rows):
        y = top + i * (bh + gap)
        color = t["s1"] if r is best else t["muted_mark"]
        body.append(_text(left - 12, y + bh / 2 + 4, f'{r["experiment"]}  {r["name"]}', t, 12.5,
                          "ink" if r is best else "ink2", "end", 600 if r is best else 400))
        body.append(_hbar(left, y, r["ndcg_at_10"] * plot_w, bh, color))
        body.append(_text(left + r["ndcg_at_10"] * plot_w + 8, y + bh / 2 + 4, f'{r["ndcg_at_10"]:.3f}', t, 12, "ink"))
    body.append(f'<line x1="{left}" y1="{top - 6}" x2="{left}" y2="{height - 34}" stroke="{t["axis"]}" stroke-width="1"/>')
    title = f'Ablations: "{best["name"]}" scores highest (NDCG@10 {best["ndcg_at_10"]:.3f})'
    sub = f"AppsRetrieval full test split, {rows[0]['n_queries']} queries / {rows[0]['n_corpus']} documents. Highlighted: best configuration."
    desc = "; ".join(f'{r["experiment"]} {r["name"]}: NDCG@10 {r["ndcg_at_10"]:.3f}' for r in rows)
    return _svg(title, sub, body, height, desc, t)


def rank_curve(t: dict):
    data = _load("eval/appsretrieval_results.json")
    if not data:
        return None
    s = data["scores"]["test"][0]
    ks = sorted(int(k.split("_at_")[1]) for k in s if k.startswith("ndcg_at_"))
    series = [("Recall@k", "s2", [s[f"recall_at_{k}"] for k in ks]), ("NDCG@k", "s1", [s[f"ndcg_at_{k}"] for k in ks])]
    left, right, top, bottom = 60, 110, 84, 50
    height = 360
    pw, ph = W - left - right, height - top - bottom
    lo = math.floor(min(min(v) for _, _, v in series) * 10) / 10
    def X(k): return left + (math.log10(k) / math.log10(ks[-1])) * pw
    def Y(v): return top + (1 - (v - lo) / (1 - lo)) * ph
    body = []
    step = 0.05
    v = lo
    while v <= 1.0001:
        body.append(f'<line x1="{left}" y1="{Y(v):.1f}" x2="{left + pw}" y2="{Y(v):.1f}" stroke="{t["grid"]}" stroke-width="1"/>')
        body.append(_text(left - 8, Y(v) + 4, f"{v:.2f}", t, 11, "muted", "end"))
        v = round(v + step, 2)
    for k in ks:
        body.append(_text(X(k), top + ph + 20, k, t, 11, "muted", "middle"))
    body.append(_text(left + pw / 2, top + ph + 40, "cutoff k (log scale)", t, 11.5, "muted", "middle"))
    for name, key, vals in series:
        pts = " ".join(f"{X(k):.1f},{Y(v):.1f}" for k, v in zip(ks, vals))
        body.append(f'<polyline points="{pts}" fill="none" stroke="{t[key]}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        for k, v in zip(ks, vals):  # >=8px markers with a 2px surface ring
            body.append(f'<circle cx="{X(k):.1f}" cy="{Y(v):.1f}" r="4.5" fill="{t[key]}" stroke="{t["surface"]}" stroke-width="2"/>')
        body.append(_text(X(ks[-1]) + 10, Y(vals[-1]) + 4, f"{name} {vals[-1]:.3f}", t, 12, "ink"))
    k10 = ks.index(10)
    for _name, _key, vals in series:  # label the headline cutoff only
        body.append(_text(X(10), Y(vals[k10]) - 12, f"{vals[k10]:.3f}", t, 12, "ink", "middle", 600))
    body += _legend(left, 70, [(n, k) for n, k, _ in reversed(series)], t)
    title = f"Official run: NDCG@10 {s['ndcg_at_10']:.3f}, Recall@10 {s['recall_at_10']:.3f}"
    sub = "Ranking quality by cutoff k - jina-code-embeddings-0.5b, dense retrieval, full AppsRetrieval test split"
    desc = "; ".join(f"k={k}: NDCG {n:.3f}, Recall {r:.3f}" for k, n, r in zip(ks, series[1][2], series[0][2]))
    return _svg(title, sub, body, height, desc, t)


def embeddings(t: dict):
    data = _load("benchmarks/embeddings_subset_q100_c500.json")
    if not data:
        return None
    ok = [r for r in data["rows"] if r.get("ndcg_at_10") is not None and r.get("p50_query_latency_ms") is not None]
    skipped = [r["model"] for r in data["rows"] if r not in ok]
    left, right, top, bottom = 64, 40, 84, 70
    height = 380
    pw, ph = W - left - right, height - top - bottom
    xmax = math.ceil(max(r["p50_query_latency_ms"] for r in ok) / 100) * 100 + 50
    ylo = math.floor(min(r["ndcg_at_10"] for r in ok) * 10) / 10
    def X(v): return left + v / xmax * pw
    def Y(v): return top + (1 - (v - ylo) / (1 - ylo)) * ph
    body = []
    v = ylo
    while v <= 1.0001:
        body.append(f'<line x1="{left}" y1="{Y(v):.1f}" x2="{left + pw}" y2="{Y(v):.1f}" stroke="{t["grid"]}" stroke-width="1"/>')
        body.append(_text(left - 8, Y(v) + 4, f"{v:.1f}", t, 11, "muted", "end"))
        v = round(v + 0.1, 1)
    for xv in range(0, int(xmax) + 1, 100):
        body.append(_text(X(xv), top + ph + 20, xv, t, 11, "muted", "middle"))
    body.append(_text(left + pw / 2, top + ph + 40, "median query latency, ms (lower is faster)", t, 11.5, "muted", "middle"))
    body.append(_text(left - 44, top - 12, "NDCG@10", t, 11.5, "muted"))
    for r in ok:
        cx, cy = X(r["p50_query_latency_ms"]), Y(r["ndcg_at_10"])
        body.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="{t["s1"]}" stroke="{t["surface"]}" stroke-width="2"/>')
        near_right = cx > left + pw * 0.7
        body.append(_text(cx + (-12 if near_right else 12), cy - 10, f'{r["model"]} ({r["params"]})', t, 12.5, "ink",
                          "end" if near_right else "start", 600))
        body.append(_text(cx + (-12 if near_right else 12), cy + 18, f'NDCG@10 {r["ndcg_at_10"]:.3f} · {r["p50_query_latency_ms"]:.0f} ms', t, 11.5,
                          "ink2", "end" if near_right else "start"))
    if skipped:
        body.append(_text(left, height - 10, "Not run: " + ", ".join(skipped) + " (incompatible with the installed transformers version)", t, 11, "muted"))
    fastest_good = min(ok, key=lambda r: (r["p50_query_latency_ms"] if r["ndcg_at_10"] >= max(o["ndcg_at_10"] for o in ok) - 0.01 else 1e9))
    title = f"Embedding models: {fastest_good['model']} matches the best accuracy at the lowest latency"
    sub = f"AppsRetrieval seeded subset ({data['dataset']}), GPU run; NDCG@10 vs median query latency"
    desc = "; ".join(f'{r["model"]} ({r["params"]}): NDCG@10 {r["ndcg_at_10"]:.3f}, p50 {r["p50_query_latency_ms"]:.0f} ms' for r in ok)
    return _svg(title, sub, body, height, desc, t)


def versioning(t: dict):
    data = _load("benchmarks/versioning_click_minilm.json")
    if not data:
        return None
    steps, tot = data["steps"], data["totals"]
    left, right, top, bh, inner, gap = 250, 80, 90, 14, 3, 16
    plot_w = W - left - right
    height = top + len(steps) * (2 * bh + inner + gap) + 36
    xmax = max(s["full_rebuild_s"] for s in steps)
    xmax = math.ceil(xmax)
    body = _legend(24, 72, [("incremental update", "s1"), ("full rebuild", "s2")], t)
    for xv in range(0, xmax + 1, 2):
        x = left + xv / xmax * plot_w
        body.append(f'<line x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{height - 30}" stroke="{t["grid"]}" stroke-width="1"/>')
        body.append(_text(x, height - 14, f"{xv} s", t, 11, "muted", "middle"))
    for i, s in enumerate(steps):
        y = top + i * (2 * bh + inner + gap)
        changed = s["files_changed"] + s["files_added"] + s["files_deleted"]
        body.append(_text(left - 12, y + bh + 1, f'{s["commit"][:7]}', t, 12.5, "ink", "end", 600))
        body.append(_text(left - 12, y + bh + 15, f'{changed} of {s["files_total"]} files changed', t, 11, "muted", "end"))
        inc_w = max(s["incremental_s"] / xmax * plot_w, 2.5)  # keep sub-pixel bars visible
        body.append(_hbar(left, y, inc_w, bh, t["s1"], 3))
        body.append(_text(left + inc_w + 6, y + bh - 3, f'{s["incremental_s"]:.2f} s', t, 11.5, "ink"))
        fw = s["full_rebuild_s"] / xmax * plot_w
        body.append(_hbar(left, y + bh + inner, fw, bh, t["s2"], 3))
        body.append(_text(left + fw + 6, y + 2 * bh + inner - 3, f'{s["full_rebuild_s"]:.1f} s', t, 11.5, "ink2"))
    body.append(f'<line x1="{left}" y1="{top - 6}" x2="{left}" y2="{height - 30}" stroke="{t["axis"]}" stroke-width="1"/>')
    repo = Path(str(data.get("repository", "repo"))).name
    title = f"Version updates: incremental indexing is {tot['speedup']:.1f}x faster than a full rebuild"
    sub = (f"{repo} repository, {len(steps)} consecutive commits: {tot['incremental_s']:.1f} s vs {tot['full_rebuild_s']:.1f} s total; "
           f"{tot['embeddings_incremental']} vs {tot['embeddings_full']} embeddings computed")
    desc = "; ".join(f'{s["commit"][:7]}: incremental {s["incremental_s"]:.2f} s, full {s["full_rebuild_s"]:.1f} s' for s in steps)
    return _svg(title, sub, body, height, desc, t)


CHARTS = {"ablations": ablations, "rank_curve": rank_curve, "embeddings": embeddings, "versioning": versioning}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    made = []
    for name, fn in CHARTS.items():
        for mode, theme in THEMES.items():
            svg = fn(theme)
            if svg is None:
                break
            (OUT / f"{name}-{mode}.svg").write_text(svg, encoding="utf-8")
        else:
            made.append(name)
    print("charts written:", made)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
