"""EXPERIMENTAL: Joern Code Property Graph integration.

Goal: test whether CPG control/data-flow facts improve retrieval beyond CodeFusion's
lightweight AST call graph. Nothing in the core system depends on this module; if Joern is
absent every function returns quickly with ``available=False``.

Enable (Linux/macOS, needs Java 17+)::

    curl -L https://github.com/joernio/joern/releases/latest/download/joern-install.sh | sh
    python scripts/joern_enrich.py examples/sample_repo --index-dir artifacts/index

What it does: runs ``joern --script`` with :data:`JOERN_SCRIPT`, which exports, per method,
its file/line span, resolved callees and parameter->call-argument data-flow reachability as
JSON. :func:`load_cpg_facts` maps those facts onto snippets (file + line overlap) and
:func:`apply_to_graph` adds ``CALLS`` edges Joern resolved that the AST pass missed plus
``DATA_FLOW`` evidence used by the graph retriever for DATA_FLOW queries.

No claim is made that this improves results until measured with ``scripts/run_ablations.py``
(experiment ``N``).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from codefusion.logging_utils import get_logger

log = get_logger(__name__)

JOERN_SCRIPT = r"""
import io.shiftleft.semanticcpg.language._
import io.joern.dataflowengineoss.language._
@main def exec(inputPath: String, outFile: String) = {
  importCode(inputPath)
  val methods = cpg.method.isExternal(false).l.map { m =>
    val callees = m.call.callee.isExternal(false).fullName.dedup.l
    val flows = m.parameter.l.flatMap { p =>
      m.call.argument.reachableBy(p).l.map(a => a.method.fullName).dedup
    }
    ujson.Obj(
      "name" -> m.name, "fullName" -> m.fullName, "file" -> m.filename,
      "lineStart" -> m.lineNumber.getOrElse(-1).toString.toInt,
      "lineEnd" -> m.lineNumberEnd.getOrElse(-1).toString.toInt,
      "callees" -> callees, "paramFlowsInto" -> flows
    )
  }
  os.write.over(os.Path(outFile), ujson.write(ujson.Arr(methods: _*)))
}
"""


@dataclass
class CPGMethod:
    name: str
    full_name: str
    file: str
    line_start: int
    line_end: int
    callees: list[str] = field(default_factory=list)
    param_flows_into: list[str] = field(default_factory=list)


def joern_available() -> bool:
    return shutil.which("joern") is not None


def run_joern(repo_path: str | Path, timeout: int = 1800) -> Path | None:
    """Run Joern over ``repo_path``; returns the JSON facts file or ``None`` if unavailable/failed."""
    if not joern_available():
        log.info("Joern not installed; CPG experiment skipped")
        return None
    tmp = Path(tempfile.mkdtemp(prefix="codefusion_joern_"))
    script = tmp / "export.sc"
    out = tmp / "cpg_facts.json"
    script.write_text(JOERN_SCRIPT, encoding="utf-8")
    try:
        subprocess.run(["joern", "--script", str(script), "--param", f"inputPath={Path(repo_path).resolve()}",
                        "--param", f"outFile={out}"], check=True, timeout=timeout, capture_output=True)
    except Exception as exc:  # noqa: BLE001
        log.warning("Joern run failed", extra={"data": {"error": repr(exc)[:300]}})
        return None
    return out if out.exists() else None


def load_cpg_facts(path: str | Path, repo_root: str | Path | None = None) -> list[CPGMethod]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    root = str(Path(repo_root).resolve()) if repo_root else None
    out = []
    for m in data:
        f = str(m.get("file", ""))
        if root and f.startswith(root):
            f = f[len(root):].lstrip("/\\")
        out.append(CPGMethod(m.get("name", ""), m.get("fullName", ""), f.replace("\\", "/"), int(m.get("lineStart", -1)),
                             int(m.get("lineEnd", -1)), list(m.get("callees", [])), list(m.get("paramFlowsInto", []))))
    return out


def apply_to_graph(engine, methods: list[CPGMethod]) -> dict[str, int]:
    """Map CPG facts onto snippets and add edges/evidence. Returns counters."""
    if engine.graph is None:
        return {"mapped": 0, "calls_added": 0, "flows": 0}
    by_file: dict[str, list[int]] = {}
    for i, s in enumerate(engine.store.snippets):
        by_file.setdefault(s.file_path, []).append(i)
    name_to_idx: dict[str, list[int]] = {}
    mapped: dict[str, int] = {}
    for m in methods:
        for i in by_file.get(m.file, []):
            s = engine.store.snippets[i]
            if s.type in ("function", "method") and s.start_line <= m.line_start <= s.end_line:
                mapped[m.full_name] = i
                name_to_idx.setdefault(m.full_name, []).append(i)
    calls_added = flows = 0
    for m in methods:
        src = mapped.get(m.full_name)
        if src is None:
            continue
        for callee in m.callees:
            dst = mapped.get(callee)
            if dst is not None and dst != src:
                engine.graph._link(f"s:{src}", engine.store.snippets[dst].name, "CALLS")
                calls_added += 1
        for tgt in m.param_flows_into:
            dst = mapped.get(tgt)
            if dst is not None:
                ev = engine.store.snippets[src].extra.setdefault("cpg_flows_into", [])
                if engine.store.snippets[dst].qualified_name not in ev:
                    ev.append(engine.store.snippets[dst].qualified_name)
                    flows += 1
    return {"mapped": len(mapped), "calls_added": calls_added, "flows": flows}
