import type { CallChain, DemoQuery, Experiments, Health, Json, SearchResponse } from "./types";

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      /* non-JSON error body */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => call<Health>("/health"),
  stats: () => call<Json>("/stats"),
  search: (query: string, top_k: number, version: string | null) =>
    call<SearchResponse>("/search/explain", { method: "POST", body: JSON.stringify({ query, top_k, version }) }),
  demoQueries: () => call<DemoQuery[]>("/demo/queries"),
  callChain: (snippetId: string, depth = 3) =>
    call<CallChain>(`/graph/callchain?snippet_id=${encodeURIComponent(snippetId)}&depth=${depth}`),
  experiments: () => call<Experiments>("/experiments"),
  versions: (repo: string) => call<Json>(`/versions/${encodeURIComponent(repo)}`),
  indexRepository: (path: string, history: boolean) =>
    call<Json>("/index/repository", { method: "POST", body: JSON.stringify({ path, history }) }),
};
