export interface Snippet {
  snippet_id: string;
  file: string;
  lines: [number, number];
  symbol: string;
  name: string;
  type: string;
  language: string;
  repository: string;
  commit: string | null;
  commits: string[];
  lineage_id: string | null;
  signature: string;
  content: string;
}

export interface Provenance {
  dense_rank?: number;
  bm25_rank?: number;
  symbol_rank?: number;
  graph_rank?: number;
  dense_score?: number;
  bm25_score?: number;
  symbol_score?: number;
  graph_score?: number;
  rrf_score: number;
  fused_rank: number;
  contributions: Record<string, number>;
  reranker_score: number | null;
  structural_boost: number;
  final_rank: number;
  final_score: number;
}

export interface LineageEntry {
  idx: number;
  symbol: string;
  file: string;
  commit: string | null;
  is_this: boolean;
}

export interface Result {
  rank: number;
  score: number;
  snippet: Snippet;
  provenance: Provenance;
  structural_evidence: string[];
  graph_evidence: string[];
  lineage: LineageEntry[] | null;
}

export interface SearchResponse {
  query: string;
  intent: string;
  intent_confidence: number;
  version_scope: string;
  timings_ms: Record<string, number>;
  candidate_counts: Record<string, number>;
  retrievers: Record<string, string>;
  results: Result[];
  weights?: Record<string, number>;
  query_analysis?: {
    identifiers?: string[];
    target_symbols?: string[];
    symbol_candidates?: [string, number][];
    expansion_terms?: string[];
    intent_evidence?: string[];
    [k: string]: unknown;
  };
  dropped?: { file: string; name: string; reason: string }[];
}

export interface DemoQuery {
  query: string;
  note?: string;
  version?: string;
}

export interface ChainNode {
  idx: number;
  name: string;
  file: string;
  confidence?: number;
  children?: ChainNode[];
  parents?: ChainNode[];
}

export interface CallChain {
  idx: number;
  name: string;
  file: string;
  callers: ChainNode[];
  callees: ChainNode[];
}

export interface Health {
  status: string;
  version: string;
  engine_loaded: boolean;
  snippets: number;
  config: string;
}

// Experiment files are passed through verbatim from artifacts/; fields are optional on purpose.
export type Json = Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any

export interface Experiments {
  official_runs: Json[];
  ablations: Json[];
  embedding_benchmark: Json[];
  versioning_benchmark: Json[];
}
