import { useEffect, useState } from "react";
import { api } from "../api";
import type { Experiments, Json } from "../types";

const NE = <span className="not-eval">Not evaluated</span>;

function num(v: unknown, nd = 4) {
  return typeof v === "number" && Number.isFinite(v) ? v.toFixed(nd) : NE;
}

function Bar({ v, max }: { v: unknown; max: number }) {
  if (typeof v !== "number" || !max) return null;
  return <span className="mini-bar" style={{ width: `${Math.max(2, (v / max) * 100)}%` }} />;
}

function Section({ title, sub, children, empty }: { title: string; sub?: string; children: React.ReactNode; empty: boolean }) {
  return (
    <section className="panel exp">
      <h2>{title}</h2>
      {sub && <p className="muted small">{sub}</p>}
      {empty ? <div className="not-eval block">Not evaluated — no stored result file for this experiment yet.</div> : children}
    </section>
  );
}

function Official({ runs }: { runs: Json[] }) {
  const sorted = [...runs].sort((a, b) => Number(a.subset) - Number(b.subset));
  return (
    <table>
      <thead>
        <tr>
          <th>Run</th>
          <th>Mode</th>
          <th>Model</th>
          <th>Data</th>
          <th>NDCG@10</th>
          <th>MRR@10</th>
          <th>Recall@100</th>
          <th>Time</th>
        </tr>
      </thead>
      <tbody>
        {sorted.map((r) => (
          <tr key={r.file} className={r.subset ? "subset" : ""}>
            <td title={r.file}>{r.config}</td>
            <td>{r.mode}</td>
            <td>{r.model ?? "—"}</td>
            <td>
              {r.subset ? <span className="tag warn">subset</span> : <span className="tag ok">full test</span>} {r.n_queries} q / {r.n_corpus} docs
            </td>
            <td className="num">{num(r.ndcg_at_10)}</td>
            <td className="num">{num(r.mrr_at_10)}</td>
            <td className="num">{num(r.scores?.recall_at_100)}</td>
            <td className="num">{typeof r.total_s === "number" ? `${(r.total_s / 60).toFixed(1)} min` : "—"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Ablations({ files }: { files: Json[] }) {
  return (
    <>
      {files.map((f) => {
        const rows: Json[] = f.rows ?? [];
        const max = Math.max(0, ...rows.map((r) => (typeof r.ndcg_at_10 === "number" ? r.ndcg_at_10 : 0)));
        return (
          <div key={f.file} className="exp-block">
            <div className="muted small">
              {f.dataset === "full" ? <span className="tag ok">full test split</span> : <span className="tag warn">{f.dataset}</span>} base config{" "}
              <code>{f.base_config}</code> · {f.generated_at}
              {f.official_reference?.ndcg_at_10 !== undefined && (
                <> · official MTEB dense reference NDCG@10 {num(f.official_reference.ndcg_at_10)}</>
              )}
            </div>
            <table>
              <thead>
                <tr>
                  <th>Exp</th>
                  <th>Configuration</th>
                  <th>NDCG@10</th>
                  <th>MRR@10</th>
                  <th>P50 ms</th>
                  <th>P95 ms</th>
                  <th>Index s</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.experiment} title={(r.enabled_components ?? []).join(", ")}>
                    <td>{r.experiment}</td>
                    <td>
                      {r.name}
                      {r.status !== "ok" && <div className="small muted">{r.status}</div>}
                    </td>
                    <td className="num bar-cell">
                      <Bar v={r.ndcg_at_10} max={max} />
                      {num(r.ndcg_at_10)}
                    </td>
                    <td className="num">{num(r.mrr_at_10)}</td>
                    <td className="num">{num(r.p50_latency_ms, 1)}</td>
                    <td className="num">{num(r.p95_latency_ms, 1)}</td>
                    <td className="num">{num(r.indexing_s, 1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
      })}
    </>
  );
}

function Embeddings({ files }: { files: Json[] }) {
  return (
    <>
      {files.map((f) => (
        <div key={f.file} className="exp-block">
          <div className="muted small">
            {f.dataset === "full" ? <span className="tag ok">full test split</span> : <span className="tag warn">{f.dataset}</span>} · {f.generated_at}
          </div>
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Params</th>
                <th>NDCG@10</th>
                <th>MRR@10</th>
                <th>P50 query ms</th>
                <th>P95 query ms</th>
                <th>docs/s</th>
                <th>RAM MB</th>
              </tr>
            </thead>
            <tbody>
              {(f.rows ?? []).map((r: Json) => (
                <tr key={r.model}>
                  <td>
                    <code>{r.model}</code>
                    {r.status !== "ok" && <div className="small muted">{String(r.status).slice(0, 140)}</div>}
                  </td>
                  <td>{r.params}</td>
                  <td className="num">{num(r.ndcg_at_10)}</td>
                  <td className="num">{num(r.mrr_at_10)}</td>
                  <td className="num">{num(r.p50_query_latency_ms, 0)}</td>
                  <td className="num">{num(r.p95_query_latency_ms, 0)}</td>
                  <td className="num">{num(r.docs_per_s, 2)}</td>
                  <td className="num">{num(r.ram_mb, 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </>
  );
}

function Versioning({ files }: { files: Json[] }) {
  return (
    <>
      {files.map((f) => (
        <div key={f.file} className="exp-block">
          <div className="muted small">
            repository <code>{f.repository}</code> · model {f.model ?? "none"} · {f.commits} commits · {f.generated_at}
          </div>
          <table>
            <thead>
              <tr>
                <th>Commit</th>
                <th>Changed / added / deleted files</th>
                <th>Incremental s</th>
                <th>Full rebuild s</th>
                <th>Speed-up</th>
                <th>Embeddings inc / full</th>
              </tr>
            </thead>
            <tbody>
              {(f.steps ?? []).map((s: Json) => (
                <tr key={s.commit}>
                  <td>
                    <code>{s.commit}</code>
                  </td>
                  <td>
                    {s.files_changed}/{s.files_added}/{s.files_deleted} of {s.files_total}
                  </td>
                  <td className="num">{num(s.incremental_s, 3)}</td>
                  <td className="num">{num(s.full_rebuild_s, 3)}</td>
                  <td className="num">{s.speedup}×</td>
                  <td className="num">
                    {s.incremental_embeddings_computed} / {s.full_embeddings_computed}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </>
  );
}

export default function ExperimentsView() {
  const [data, setData] = useState<Experiments | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    api.experiments().then(setData).catch((e: Error) => setErr(e.message));
  }, []);
  if (err) return <div className="panel error">{err}</div>;
  if (!data) return <div className="panel muted">Loading stored results…</div>;
  return (
    <div className="experiments">
      <p className="muted">
        Every number on this page is read from result files written by the evaluation scripts under <code>artifacts/</code>. Nothing is estimated; experiments that
        have not been run show <span className="not-eval">Not evaluated</span>.
      </p>
      <Section title="Official AppsRetrieval (MTEB)" sub="mteb.get_task('AppsRetrieval'), test split. Subset rows are smoke tests, not scores." empty={!data.official_runs.length}>
        <Official runs={data.official_runs} />
      </Section>
      <Section title="Ablations" sub="Each row adds one component to the previous configuration (scripts/run_ablations.py)." empty={!data.ablations.length}>
        <Ablations files={data.ablations} />
      </Section>
      <Section title="Embedding models" sub="scripts/benchmark_embeddings.py — dense-only retrieval, single-query latency on CPU." empty={!data.embedding_benchmark.length}>
        <Embeddings files={data.embedding_benchmark} />
      </Section>
      <Section title="Incremental indexing (P1)" sub="scripts/benchmark_versions.py — full rebuild vs incremental update per commit." empty={!data.versioning_benchmark.length}>
        <Versioning files={data.versioning_benchmark} />
      </Section>
    </div>
  );
}
