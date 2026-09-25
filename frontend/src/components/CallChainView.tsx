import type { CallChain, ChainNode } from "../types";

/** Linearises the call neighbourhood into main() → preprocess() → … columns. */
function firstPath(nodes: ChainNode[] | undefined, key: "children" | "parents"): ChainNode[] {
  const path: ChainNode[] = [];
  let cur = nodes?.[0];
  while (cur) {
    path.push(cur);
    cur = cur[key]?.[0];
  }
  return path;
}

function Node({ n, focus }: { n: { name: string; file: string }; focus?: boolean }) {
  const short = n.name.split("::").pop() ?? n.name;
  return (
    <div className={focus ? "cc-node focus" : "cc-node"} title={n.file}>
      <code>{short}()</code>
      <small>{n.file}</small>
    </div>
  );
}

export default function CallChainView({ chain }: { chain: CallChain }) {
  const up = firstPath(chain.callers, "parents").reverse();
  const down = firstPath(chain.callees, "children");
  const otherCallers = chain.callers.slice(1);
  const otherCallees = chain.callees.slice(1);
  const empty = up.length === 0 && down.length === 0;
  return (
    <div className="callchain">
      <span className="label">Call chain (resolved CALLS edges, current version)</span>
      {empty ? (
        <div className="muted small">No resolved callers or callees for this snippet.</div>
      ) : (
        <div className="cc-flow">
          {up.map((n) => (
            <div key={"u" + n.idx} className="cc-step">
              <Node n={n} />
              <span className="cc-arrow">↓</span>
            </div>
          ))}
          <div className="cc-step">
            <Node n={chain} focus />
          </div>
          {down.map((n) => (
            <div key={"d" + n.idx} className="cc-step">
              <span className="cc-arrow">↓</span>
              <Node n={n} />
            </div>
          ))}
        </div>
      )}
      {(otherCallers.length > 0 || otherCallees.length > 0) && (
        <div className="cc-more small muted">
          {otherCallers.length > 0 && <div>also called by: {otherCallers.map((c) => <code key={c.idx}>{c.name}</code>)}</div>}
          {otherCallees.length > 0 && <div>also calls: {otherCallees.map((c) => <code key={c.idx}>{c.name}</code>)}</div>}
        </div>
      )}
    </div>
  );
}
