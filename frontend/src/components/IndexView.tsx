import { FormEvent, useEffect, useState } from "react";
import { api } from "../api";
import type { Json } from "../types";

export default function IndexView({ onIndexed }: { onIndexed: () => void }) {
  const [path, setPath] = useState("examples/sample_repo");
  const [history, setHistory] = useState(false);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<Json | null>(null);
  const [stats, setStats] = useState<Json | null>(null);
  const [versions, setVersions] = useState<Record<string, Json>>({});
  const [err, setErr] = useState<string | null>(null);

  const loadStats = async () => {
    try {
      const s = await api.stats();
      setStats(s);
      const v: Record<string, Json> = {};
      for (const repo of (s.repositories as string[]) ?? []) {
        try {
          v[repo] = await api.versions(repo);
        } catch {
          /* repository indexed without commits */
        }
      }
      setVersions(v);
    } catch (e) {
      setErr((e as Error).message);
    }
  };

  useEffect(() => {
    loadStats();
  }, []);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    try {
      setResult(await api.indexRepository(path, history));
      await loadStats();
      onIndexed();
    } catch (e2) {
      setErr((e2 as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="index-view">
      <section className="panel">
        <h2>Index a repository</h2>
        <p className="muted small">
          A path on the API server. Git repositories are indexed by commit (tree diff → only changed files are re-parsed and only new snippet versions are embedded);
          plain directories are indexed as a working tree. For the evolution demo run <code>python scripts/make_demo_history.py</code> and index{" "}
          <code>artifacts/demo_history_repo</code> with full history.
        </p>
        <form onSubmit={submit} className="query-row">
          <input className="query-input" value={path} onChange={(e) => setPath(e.target.value)} />
          <label className="check">
            <input type="checkbox" checked={history} onChange={(e) => setHistory(e.target.checked)} /> full git history
          </label>
          <button className="primary" disabled={busy}>
            {busy ? "Indexing…" : "Index"}
          </button>
        </form>
        {err && <div className="error small">{err}</div>}
        {result && (
          <table>
            <thead>
              <tr>
                <th>Commit</th>
                <th>Files changed/added/deleted/reused</th>
                <th>New snippet versions</th>
                <th>Reused</th>
                <th>Embeddings computed / reused</th>
                <th>Lineage</th>
                <th>Seconds</th>
              </tr>
            </thead>
            <tbody>
              {(result.updates as Json[]).map((u) => (
                <tr key={u.commit}>
                  <td>
                    <code>{String(u.commit).slice(0, 10)}</code>
                  </td>
                  <td>
                    {u.files_changed}/{u.files_added}/{u.files_deleted}/{u.files_reused}
                  </td>
                  <td className="num">{u.snippets_new_versions}</td>
                  <td className="num">{u.snippets_reused}</td>
                  <td className="num">
                    {u.embeddings_computed} / {u.embeddings_reused}
                  </td>
                  <td>{Object.entries(u.lineage_relations ?? {}).map(([k, v]) => `${k}:${v}`).join(" ") || "—"}</td>
                  <td className="num">{u.seconds?.total?.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <section className="panel">
        <h2>Index statistics</h2>
        {stats ? (
          <div className="stat-grid">
            <Stat k="Snippet versions" v={stats.snippet_versions} />
            <Stat k="Commits" v={stats.commits} />
            <Stat k="Lineage edges" v={stats.lineage_edges} />
            <Stat k="Dense model" v={stats.dense?.model ?? "off"} />
            <Stat k="Vectors" v={stats.dense?.vectors ?? "—"} />
            <Stat k="Symbols" v={stats.symbols ?? "—"} />
            <Stat k="Graph nodes" v={stats.graph?.nodes ?? "—"} />
            <Stat k="Graph edges" v={stats.graph?.edges ?? "—"} />
          </div>
        ) : (
          <div className="muted">—</div>
        )}
      </section>

      {Object.entries(versions).map(([repo, v]) => (
        <section className="panel" key={repo}>
          <h2>
            Versions of <code>{repo}</code>
          </h2>
          <ol className="timeline">
            {(v.commits as Json[]).map((c) => (
              <li key={c.sha} className={c.sha === v.latest ? "latest" : ""}>
                <code>{String(c.sha).slice(0, 10)}</code> {c.message} <span className="muted small">{c.snippets} snippets</span>
                {c.sha === v.latest && <span className="tag ok">latest</span>}
              </li>
            ))}
          </ol>
          {v.lineage_relations && Object.keys(v.lineage_relations).length > 0 && (
            <div className="muted small">lineage relations: {Object.entries(v.lineage_relations).map(([k, n]) => `${k} ${n}`).join(" · ")}</div>
          )}
        </section>
      ))}
    </div>
  );
}

function Stat({ k, v }: { k: string; v: unknown }) {
  return (
    <div className="stat">
      <div className="stat-v">{typeof v === "number" ? v.toLocaleString() : String(v)}</div>
      <div className="stat-k">{k}</div>
    </div>
  );
}
