"""Artifact helpers: Parquet/CSV/JSON exports and result files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from codefusion.logging_utils import get_logger
from codefusion.types import Snippet

log = get_logger(__name__)


def write_json(path: str | Path, data: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return p


def read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def export_snippets_parquet(snippets: list[Snippet], path: str | Path) -> Path | None:
    """Write snippet metadata (without list-heavy columns flattened) to Parquet for analysis."""
    try:
        import pandas as pd
    except ImportError:
        log.warning("pandas not installed; skipping parquet export")
        return None
    rows = []
    for i, s in enumerate(snippets):
        d = s.to_dict()
        d["idx"] = i
        for k in ("parameters", "calls", "imports", "references", "attributes", "returns", "raises", "decorators",
                  "bases", "defines", "comments", "string_literals", "commits"):
            d[k] = json.dumps(d[k], ensure_ascii=False)
        d["extra"] = json.dumps(d["extra"], ensure_ascii=False, default=str)
        rows.append(d)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        pd.DataFrame(rows).to_parquet(p, index=False)
    except Exception as exc:  # noqa: BLE001 - pyarrow optional
        log.warning("parquet export failed", extra={"data": {"error": repr(exc)}})
        return None
    return p


def write_table(rows: list[dict[str, Any]], base: str | Path) -> dict[str, str]:
    """Write rows as CSV + JSON (+ Markdown table). Returns written paths."""
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    out = {"json": str(write_json(base.with_suffix(".json"), rows))}
    if rows:
        cols: list[str] = []
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        import csv

        with open(base.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})
        out["csv"] = str(base.with_suffix(".csv"))
        md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in rows:
            md.append("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |")
        base.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
        out["md"] = str(base.with_suffix(".md"))
    return out


def _fmt(v: Any) -> str:
    if v is None:
        return "NOT RUN"
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, (list, dict)):
        return json.dumps(v)[:80]
    return str(v)
