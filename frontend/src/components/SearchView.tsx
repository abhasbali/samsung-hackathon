import { FormEvent, useEffect, useState } from "react";
import { api } from "../api";
import type { DemoQuery, SearchResponse } from "../types";
import ResultCard from "./ResultCard";
import { Funnel, RetrieverStatus, TimingBar } from "./Pipeline";

export default function SearchView() {
  const [query, setQuery] = useState("How is input validated before prediction?");
  const [topK, setTopK] = useState(10);
  const [version, setVersion] = useState("latest");
  const [demo, setDemo] = useState<DemoQuery[]>([]);
  const [resp, setResp] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api.demoQueries().then(setDemo).catch(() => setDemo([]));
  }, []);

  const run = async (q: string, v = version) => {
    if (!q.trim()) return;
    setBusy(true);
    setError(null);
    try {
      setResp(await api.search(q, topK, v === "latest" ? null : v));
    } catch (e) {
      setError((e as Error).message);
      setResp(null);
    } finally {
      setBusy(false);
    }
  };

  const submit = (e: FormEvent) => {
    e.preventDefault();
    run(query);
  };

  const pickDemo = (d: DemoQuery) => {
    setQuery(d.query);
    const v = d.version ?? "latest";
    setVersion(v);
    run(d.query, v);
  };

  return (
    <div className="search-layout">
      <section className="panel query-panel">
        <form onSubmit={submit} className="query-form">
          <label htmlFor="q" className="label">
            Query
          </label>
          <div className="query-row">
            <input id="q" className="query-input" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Ask about the codebase…" autoFocus />
            <button className="primary" disabled={busy || !query.trim()}>
              {busy ? "Searching…" : "Search"}
            </button>
          </div>
          <div className="query-options">
            <label>
              Versions
              <select value={version} onChange={(e) => setVersion(e.target.value)}>
                <option value="latest">latest</option>
                <option value="all">all (history)</option>
              </select>
            </label>
            <label>
              Top-k
              <select value={topK} onChange={(e) => setTopK(Number(e.target.value))}>
                {[5, 10, 20].map((k) => (
                  <option key={k} value={k}>
                    {k}
                  </option>
                ))}
              </select>
            </label>
          </div>
        </form>
        {demo.length > 0 && (
          <div className="demo-list">
            <span className="label">Demo scenarios</span>
            <div className="chips">
              {demo.map((d) => (
                <button key={d.query} className="chip" title={d.note} onClick={() => pickDemo(d)}>
                  {d.query}
                </button>
              ))}
            </div>
          </div>
        )}
      </section>

      {error && (
        <div className="panel error" role="alert">
          {error}
          {error.startsWith("409") && <div className="hint">Index a repository first (tab “Index &amp; versions”) or run scripts/index_repo.py.</div>}
        </div>
      )}

      {resp && (
        <>
          <section className="panel summary">
            <div className="summary-grid">
              <div>
                <div className="label">Detected intent</div>
                <div className="intent">
                  <span className={`intent-badge intent-${resp.intent.toLowerCase()}`}>{resp.intent}</span>
                  <span className="muted">confidence {resp.intent_confidence.toFixed(2)}</span>
                </div>
                {resp.query_analysis?.target_symbols && resp.query_analysis.target_symbols.length > 0 && (
                  <div className="muted small">
                    target symbols: {resp.query_analysis.target_symbols.map((s) => <code key={s}>{s}</code>)}
                  </div>
                )}
                <div className="muted small">version scope: {resp.version_scope}</div>
              </div>
              <div>
                <div className="label">Retrieval timing</div>
                <div className="total-ms">
                  {resp.timings_ms.total?.toFixed(1)} <span>ms</span>
                </div>
                <TimingBar timings={resp.timings_ms} />
              </div>
              <div>
                <div className="label">Retriever status</div>
                <RetrieverStatus retrievers={resp.retrievers} counts={resp.candidate_counts} weights={resp.weights} />
              </div>
              <div>
                <div className="label">Candidates</div>
                <Funnel counts={resp.candidate_counts} topK={resp.results.length} />
              </div>
            </div>
          </section>
          <section className="results">
            {resp.results.length === 0 && <div className="panel muted">No results.</div>}
            {resp.results.map((r) => (
              <ResultCard key={r.snippet.snippet_id} result={r} />
            ))}
            {resp.dropped && resp.dropped.length > 0 && (
              <details className="panel dropped">
                <summary>{resp.dropped.length} near-duplicate / historical candidates removed by diversity</summary>
                <ul>
                  {resp.dropped.map((d, i) => (
                    <li key={i}>
                      <code>{d.name || d.file}</code> — {d.reason}
                    </li>
                  ))}
                </ul>
              </details>
            )}
          </section>
        </>
      )}
    </div>
  );
}
