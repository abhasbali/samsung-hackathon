# Retrieval

This page follows one query through the pipeline, in the order `CodeFusionEngine.search` executes it (`src/codefusion/retrieval/pipeline.py`).

## 1. Query understanding (`query/`)

**Cleaning.** Whitespace is normalized for short queries. Long problem statements, like AppsRetrieval's, keep their layout. Optionally `query.strip_examples` removes sample input/output sections before dense encoding.

**Intent.** A rule-based classifier (`query/classifier.py`) scores 11 intents with weighted regular expressions. The highest total wins; ties follow a fixed priority. It also extracts **target symbols**:

| Query | Intent | Targets |
|---|---|---|
| Where is MAX_RETRIES defined? | DEFINITION | `MAX_RETRIES` |
| Who calls preprocess? | CALLER | `preprocess` |
| What functions does preprocess call? | CALLEE | `preprocess` |
| How is input transformed before authentication? | DATA_FLOW | – |
| Where is Redis used? | DEPENDENCY | `Redis` |
| How did authentication change? | EVOLUTION | – |

The classifier has no LLM dependency: classification takes microseconds. `tests/test_classifier.py` pins the behaviour above.

**Symbol candidates.** The preprocessor turns the query into weighted normalized symbols:

* Targets get weight 1.5.
* Code-looking identifiers and backticked names get 1.0.
* Adjacent-word joins get 0.8: "validate user" becomes `validateuser`, which matches `validate_user`, `validateUser` and `ValidateUser`. Connector words may sit inside a join (`retry with backoff` → `retrywithbackoff`) but not at its ends.
* Remaining content words get 0.35.

**Expansion** (optional, ablated) draws on a curated concept map ("authentication" → auth, login, token, jwt, verify, …) and on corpus vocabulary (query word "retries" → `MAX_RETRIES`, `retry_with_backoff`). Expansion terms feed BM25 only. Injecting them as symbols proved noisy: "request" matched every `dict.get`.

## 2. First-stage retrievers

### Dense (`retrieval/embeddings.py`, `retrieval/dense.py`)

```
query → model-specific query prompt → encoder → L2 normalize → FAISS IndexFlatIP → top dense.top_k (100)
```

Prompts follow each model card and are not shared across models:

| Model | Query prompt | Document prompt |
|---|---|---|
| jina-code-embeddings-0.5b / 1.5b | `Find the most relevant code snippet given the following query:\n` | `Candidate code snippet:\n` |
| Qodo-Embed-1-1.5B | none (the model ships no prompts, MTEB `use_instructions=False`) | none |
| jina-embeddings-v2-base-code | none | none |

`dense.document_view` chooses what is embedded: `raw`, `context+raw`, `structural` or `raw+structural`. Benchmark documents are always embedded verbatim, exactly as MTEB feeds them to an encoder, so the dense-only and hybrid evaluations share cached vectors. `dense.index_type` switches to HNSW (`hnsw_m`) or IVF (`ivf_nlist`, `ivf_nprobe`) for large corpora.

### Code-aware BM25 (`retrieval/bm25.py`)

One bm25s index per view: `raw`, `identifiers`, `structural`, `context`, `docstring`. The score is `Σ_field w_field · BM25_field(q, d)`, with weights in `bm25.fields`. The tokenizer (`parsing/identifiers.py`) emits both the original identifier and its components:

```
getUserAuthenticationToken → getuserauthenticationtoken, get, user, authentication, token
HTTPRequest                → httprequest, http, request
OAuth2Token                → oauth2token, oauth, oauth2, token
MAX_RETRIES                → max_retries, max, retries
```

Stemming and stopword removal apply to components, never to the preserved original identifier.

### Symbol index (`retrieval/symbols.py`)

Postings map a normalized symbol to `(snippet, role)`. The score is `Σ q_weight · role_weight · intent_multiplier(role) · idf`. The intent multipliers encode what each question type is after:

| Intent | def | call | ref | import |
|---|---|---|---|---|
| CALLER ("who calls X") | ×0.2 | ×2.0 | | |
| CALLEE ("what does X call") | ×1.5 | ×0 | ×0 | |
| DEFINITION / CONFIGURATION | ×1.5 | ×0.4 | ×0.3–0.5 | ×0.5 |
| USAGE / DEPENDENCY | ×0.4 | ×1.3–1.5 | ×1.2–1.5 | ×1.3–2.0 |
| ERROR | ×1.2 | ×1.5 (raise sites) | | ×0.3 |

For CALLEE, a mention of X anywhere else is a *caller* of X, so only X's definition is a symbol hit; the callees come from the graph retriever. ERROR queries also propose exception-class names (`database` → `databaseerror`, `databaseexception`).

SCIP enrichment is optional: `symbols.scip_index: path/to/index.scip`, with the `scip` CLI on PATH, merges precise cross-file occurrences.

**Ties.** Every retriever gives exactly tied scores the same rank (competition ranking 1, 1, 3). A retriever that cannot tell documents apart, for example a symbol present in every snippet, therefore contributes no arbitrary preference to fusion.

## 3. Graph expansion (`graph/scoring.py`)

Graph retrieval is seeded, never exhaustive:

1. **Targets.** For CALLER, the snippets that call the target. For CALLEE, the snippets the target's definition calls. For DATA_FLOW, both.
2. **Seeds.** The top `graph.seeds` results of an unweighted RRF over the first-stage lists.
3. **Expansion.** 1–`graph.depth` hops over CALLS, REFERENCES, EXTENDS, OVERRIDES, CONTAINS and IMPORTS, with `score = seed_strength · edge_weight · decay^hop`.

The result is a normal ranked list with evidence strings such as `caller of validate_user` or `callee of main`.

## 4. Fusion (`retrieval/fusion.py`)

```
RRF(d) = Σ_i  w_i(intent) / (k + rank_i(d)),   k = fusion.rrf_k (60)
w_i(intent) = fusion.weights[i] × fusion.intent_weights[intent][i]   (if query_adaptive)
```

Raw scores from unrelated retrievers are never averaged. For ablation C, min-max normalized score fusion (`fusion.method: score`) is also available. The default intent multipliers follow the brief:

| Intent | dense | bm25 | symbol | graph |
|---|---|---|---|---|
| DEFINITION | 0.6 | 1.3 | 2.0 | 0.3 |
| DATA_FLOW | 1.3 | 0.7 | 1.0 | 1.6 |
| CALLER / CALLEE | 0.6 | 0.8 | 1.5 | 2.0 |
| EVOLUTION | 1.0 | 1.0 | 1.2 | 0.8 |
| GENERAL | 1.0 | 1.0 | 1.0 | 1.0 |

These weights are **not** claimed to be optimal. Ablation J measures whether they help on AppsRetrieval, and the P0 config keeps them only if they do.

## 5. Reranking (`retrieval/reranker.py`)

The top `reranker.candidates` (30) fused candidates are re-scored and combined **in rank space**:

```
final = alpha / (k + fused_rank) + (1 - alpha) / (k + reranker_rank)      (mode: interpolate, alpha 0.3)
```

The default model is `Alibaba-NLP/gte-reranker-modernbert-base`: a 149M ModernBERT cross-encoder trained with code retrieval data, native to transformers with no remote code. `reranker.type: embedding` re-scores with a (larger) bi-encoder instead. If a reranker cannot be loaded (network, auth or incompatible remote code), the engine logs it, reports `unavailable (fallback to fused ranking)` and continues. Reranker latency appears separately as `timings_ms.rerank`.

## 6. Structural scoring

These are small, explained, additive boosts on the fused score, configured in `structural.*`:

| Signal | Default |
|---|---|
| candidate defines a queried symbol | +0.6 × unit |
| candidate calls a queried symbol (intent-scaled; ×0.1 for CALLEE) | +0.3 × unit |
| candidate references a queried symbol | +0.1 × unit |
| caller or callee of a top-3 result | +0.15 × unit |
| calls the target (CALLER / DATA_FLOW) or is called by it (CALLEE / DATA_FLOW) | +0.8 × unit |
| lies on a short call path between two queried symbols | +0.25 × unit |
| imports a queried module or API | +0.2 × unit |

The unit is `1/(rrf_k+1)`, one first-rank RRF vote, so boosts are comparable with fusion scores. Each boost appends evidence (`defines queried symbol maxretries`, `caller of validate_user`, …).

## 7. Diversity

In order: exact duplicate content, then identical AST (formatting-only changes), then another version of an already-kept lineage (unless EVOLUTION), then shingle-Jaccard ≥ 0.92 near-duplicates. Dropped candidates are returned with their reason in explain mode. MMR is optional (`diversity.mmr`, λ 0.75) and uses dense vectors when available.

## 8. Explainability

Every result carries:

```json
{"dense_rank": 4, "bm25_rank": 1, "symbol_rank": 2, "graph_rank": null,
 "rrf_score": 0.0482, "fused_rank": 1, "contributions": {"dense": 0.0156, "bm25": 0.0164, "symbol": 0.0161},
 "reranker_score": null, "structural_boost": 0.0098, "final_rank": 1, "final_score": 0.058}
```

It also carries `structural_evidence`, `graph_evidence`, `lineage`, and per-stage `timings_ms` for the query.
