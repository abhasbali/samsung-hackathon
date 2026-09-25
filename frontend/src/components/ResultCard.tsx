import { useState } from "react";
import { api } from "../api";
import type { CallChain, Result } from "../types";
import CallChainView from "./CallChainView";
import CodeBlock from "./CodeBlock";

const SOURCES: [string, string][] = [
  ["dense", "Dense"],
  ["bm25", "BM25"],
  ["symbol", "Symbol"],
  ["graph", "Graph"],
];

function short(sha: string | null | undefined) {
  return sha ? sha.slice(0, 8) : "";
}

export default function ResultCard({ result }: { result: Result }) {
  const { snippet: s, provenance: p } = result;
  const [chain, setChain] = useState<CallChain | null>(null);
  const [chainErr, setChainErr] = useState<string | null>(null);
  const [showChain, setShowChain] = useState(false);
  const [expanded, setExpanded] = useState(result.rank <= 3);

  const toggleChain = async () => {
    const next = !showChain;
    setShowChain(next);
    if (next && !chain && !chainErr) {
      try {
        setChain(await api.callChain(s.snippet_id));
      } catch (e) {
        setChainErr((e as Error).message);
      }
    }
  };

  const title = s.type === "document" ? s.file : s.symbol || s.name || s.file;
  return (
    <article className="panel result">
      <header className="result-head">
        <span className="rank">#{result.rank}</span>
        <div className="result-title">
          <div className="file">{s.file}</div>
          <div className="symbol">
            <code>{title}{s.type === "function" || s.type === "method" ? "()" : ""}</code>
            <span className="tag">{s.type}</span>
            <span className="tag">{s.language}</span>
            {s.commit && <span className="tag commit" title={s.commit}>@{short(s.commit)}</span>}
          </div>
        </div>
        <div className="result-meta">
          <div className="lines">
            lines {s.lines[0]}–{s.lines[1]}
          </div>
          <div className="score" title="final score">{p.final_score.toFixed(4)}</div>
        </div>
      </header>

      <div className="why">
        <span className="label">Why retrieved</span>
        <div className="why-grid">
          {SOURCES.map(([k, label]) => {
            const rank = (p as unknown as Record<string, number | undefined>)[`${k}_rank`];
            const contrib = p.contributions?.[k];
            return (
              <div key={k} className={rank ? "why-cell hit" : "why-cell"}>
                <span>{label}</span>
                <b>{rank ? `rank ${rank}` : "—"}</b>
                {contrib !== undefined && <small>+{contrib.toFixed(4)}</small>}
              </div>
            );
          })}
          <div className="why-cell">
            <span>RRF</span>
            <b>#{p.fused_rank}</b>
            <small>{p.rrf_score.toFixed(4)}</small>
          </div>
          {p.reranker_score !== null && (
            <div className="why-cell hit">
              <span>Reranker</span>
              <b>{p.reranker_score.toFixed(3)}</b>
            </div>
          )}
          {p.structural_boost !== 0 && (
            <div className="why-cell hit">
              <span>Structure</span>
              <b>{p.structural_boost > 0 ? "+" : ""}{p.structural_boost.toFixed(4)}</b>
            </div>
          )}
        </div>
        {(result.structural_evidence.length > 0 || result.graph_evidence.length > 0) && (
          <ul className="evidence">
            {result.structural_evidence.slice(0, 6).map((e) => (
              <li key={"s" + e}>
                <span className="ev-kind">match</span> {e}
              </li>
            ))}
            {result.graph_evidence.slice(0, 6).map((e) => (
              <li key={"g" + e}>
                <span className="ev-kind graph">graph</span> {e}
              </li>
            ))}
          </ul>
        )}
      </div>

      {result.lineage && result.lineage.length > 1 && (
        <div className="lineage">
          <span className="label">Lineage</span>
          <ol>
            {result.lineage.map((l) => (
              <li key={l.idx} className={l.is_this ? "this" : ""}>
                <code>{l.symbol}</code> <span className="muted">{l.file}</span> <span className="tag commit">@{short(l.commit)}</span>
              </li>
            ))}
          </ol>
        </div>
      )}

      <div className="code-wrap">
        {expanded ? (
          <CodeBlock code={s.content} startLine={s.lines[0]} />
        ) : (
          <button className="link" onClick={() => setExpanded(true)}>
            Show code ({s.lines[1] - s.lines[0] + 1} lines)
          </button>
        )}
      </div>
      {s.type !== "document" && (
        <div className="result-actions">
          <button className="link" onClick={toggleChain}>
            {showChain ? "Hide call chain" : "Show call chain"}
          </button>
        </div>
      )}
      {showChain && (chainErr ? <div className="error small">{chainErr}</div> : chain ? <CallChainView chain={chain} /> : <div className="muted small">loading…</div>)}
    </article>
  );
}
