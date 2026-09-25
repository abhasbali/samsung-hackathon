import { useEffect, useState } from "react";
import { api } from "./api";
import type { Health } from "./types";
import SearchView from "./components/SearchView";
import ExperimentsView from "./components/ExperimentsView";
import IndexView from "./components/IndexView";

type Tab = "search" | "experiments" | "index";

const TABS: { id: Tab; label: string }[] = [
  { id: "search", label: "Search" },
  { id: "experiments", label: "Experiments" },
  { id: "index", label: "Index & versions" },
];

function initialTab(): Tab {
  const h = window.location.hash.replace("#", "");
  return (TABS.some((t) => t.id === h) ? h : "search") as Tab;
}

export default function App() {
  const [tab, setTab] = useState<Tab>(initialTab);
  const [health, setHealth] = useState<Health | null>(null);
  const [healthErr, setHealthErr] = useState<string | null>(null);

  const refreshHealth = () =>
    api
      .health()
      .then((h) => {
        setHealth(h);
        setHealthErr(null);
      })
      .catch((e: Error) => setHealthErr(e.message));

  useEffect(() => {
    refreshHealth();
  }, []);

  const go = (t: Tab) => {
    setTab(t);
    window.location.hash = t;
  };

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo" aria-hidden>
            &lt;/&gt;
          </span>
          <div>
            <div className="brand-name">CodeFusion</div>
            <div className="brand-sub">code-intelligence retrieval</div>
          </div>
        </div>
        <nav className="tabs" role="tablist">
          {TABS.map((t) => (
            <button key={t.id} role="tab" aria-selected={tab === t.id} className={tab === t.id ? "tab active" : "tab"} onClick={() => go(t.id)}>
              {t.label}
            </button>
          ))}
        </nav>
        <div className="health" title={health?.config}>
          {healthErr ? (
            <span className="dot bad" />
          ) : (
            <span className={health ? "dot ok" : "dot"} />
          )}
          {healthErr ? "API offline" : health ? `${health.snippets.toLocaleString()} snippets · v${health.version}` : "connecting…"}
        </div>
      </header>
      <main className="content">
        {tab === "search" && <SearchView />}
        {tab === "experiments" && <ExperimentsView />}
        {tab === "index" && <IndexView onIndexed={refreshHealth} />}
      </main>
    </div>
  );
}
