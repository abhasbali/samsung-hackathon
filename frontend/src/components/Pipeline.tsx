const STAGES = ["preprocess", "dense", "bm25", "symbol", "graph", "fusion", "rerank", "structural", "diversity"] as const;

export function TimingBar({ timings }: { timings: Record<string, number> }) {
  const parts = STAGES.filter((s) => (timings[s] ?? 0) > 0).map((s) => ({ s, ms: timings[s] }));
  const sum = parts.reduce((a, p) => a + p.ms, 0) || 1;
  return (
    <div>
      <div className="timing-bar" aria-hidden>
        {parts.map((p) => (
          <span key={p.s} className={`seg seg-${p.s}`} style={{ width: `${(p.ms / sum) * 100}%` }} title={`${p.s}: ${p.ms.toFixed(2)} ms`} />
        ))}
      </div>
      <div className="timing-legend">
        {parts.map((p) => (
          <span key={p.s}>
            <i className={`sw seg-${p.s}`} />
            {p.s} {p.ms < 1 ? p.ms.toFixed(2) : p.ms.toFixed(1)}
          </span>
        ))}
      </div>
    </div>
  );
}

const RETRIEVERS: [string, string][] = [
  ["dense", "Dense"],
  ["bm25", "BM25"],
  ["symbol", "Symbols"],
  ["graph", "Graph"],
  ["reranker", "Reranker"],
];

export function RetrieverStatus({
  retrievers,
  counts,
  weights,
}: {
  retrievers: Record<string, string>;
  counts: Record<string, number>;
  weights?: Record<string, number>;
}) {
  return (
    <ul className="retrievers">
      {RETRIEVERS.map(([key, label]) => {
        const status = retrievers[key] ?? "disabled";
        const on = status.startsWith("enabled");
        const n = counts[key] ?? (key === "reranker" ? counts.reranked : undefined);
        return (
          <li key={key} className={on ? "on" : "off"} title={status}>
            <span className={on ? "dot ok" : "dot"} />
            <span className="r-name">{label}</span>
            <span className="r-meta">
              {on ? (n !== undefined ? `${n} hits` : "—") : status.startsWith("unavailable") ? "unavailable" : "off"}
              {on && weights && weights[key] !== undefined && <em> · w {weights[key].toFixed(2)}</em>}
            </span>
          </li>
        );
      })}
    </ul>
  );
}

export function Funnel({ counts, topK }: { counts: Record<string, number>; topK: number }) {
  const first = ["dense", "bm25", "symbol", "graph"].reduce((a, k) => a + (counts[k] ?? 0), 0);
  const steps: [string, number | undefined][] = [
    ["retrieved", first],
    ["fused (unique)", counts.fused],
    ["reranked", counts.reranked],
    ["final", topK],
  ];
  return (
    <div className="funnel">
      {steps
        .filter(([, v]) => v !== undefined)
        .map(([k, v], i, arr) => (
          <span key={k} className="funnel-step">
            <b>{v}</b>
            <small>{k}</small>
            {i < arr.length - 1 && <span className="arrow">→</span>}
          </span>
        ))}
    </div>
  );
}
