import { Fragment, ReactNode } from "react";

// Tiny, dependency-free highlighter: keywords, strings, comments, numbers. Good enough for a demo.
const KEYWORDS = new Set(
  (
    "def class return if elif else for while in not and or import from as with try except finally raise " +
    "lambda yield pass break continue None True False self async await function const let var new this " +
    "public private static void int string interface type extends implements package func go struct"
  ).split(" "),
);
const TOKEN = /(#[^\n]*|\/\/[^\n]*|"""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\b\d+(?:\.\d+)?\b|\b[A-Za-z_][A-Za-z0-9_]*\b)/g;

function highlight(line: string): ReactNode[] {
  const out: ReactNode[] = [];
  let last = 0;
  for (const m of line.matchAll(TOKEN)) {
    const t = m[0];
    const i = m.index ?? 0;
    if (i > last) out.push(line.slice(last, i));
    let cls = "";
    if (t.startsWith("#") || t.startsWith("//")) cls = "tk-com";
    else if (t[0] === '"' || t[0] === "'") cls = "tk-str";
    else if (/^\d/.test(t)) cls = "tk-num";
    else if (KEYWORDS.has(t)) cls = "tk-kw";
    else if (line[i + t.length] === "(") cls = "tk-fn";
    out.push(cls ? (
      <span key={i} className={cls}>
        {t}
      </span>
    ) : (
      t
    ));
    last = i + t.length;
  }
  if (last < line.length) out.push(line.slice(last));
  return out;
}

export default function CodeBlock({ code, startLine = 1, maxLines = 60 }: { code: string; startLine?: number; maxLines?: number }) {
  const lines = code.replace(/\s+$/, "").split("\n");
  const shown = lines.slice(0, maxLines);
  return (
    <pre className="code">
      <code>
        {shown.map((l, i) => (
          <Fragment key={i}>
            <span className="ln">{startLine + i}</span>
            {highlight(l)}
            {"\n"}
          </Fragment>
        ))}
        {lines.length > maxLines && <span className="muted">… {lines.length - maxLines} more lines</span>}
      </code>
    </pre>
  );
}
