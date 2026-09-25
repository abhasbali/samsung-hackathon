# Version-aware and evolutionary retrieval

## P1: incremental indexing

`versions/incremental.py`, `IncrementalIndexer`:

```mermaid
flowchart LR
  A[commit A indexed\nrepo_state: path → blob SHA] --> T[git ls-tree -r B]
  T --> C{per file}
  C -->|same blob| R[reuse: register existing\nsnippet versions for B]
  C -->|changed / added| P[read blob · AST chunk]
  C -->|deleted| X[drop from B's membership]
  P --> K{snippet (symbol_key, content_hash)\nalready stored?}
  K -->|yes| R2[reuse version\nno re-embedding]
  K -->|no| N[new version: embed · FAISS append ·\nsymbol postings · graph nodes/edges]
  N --> BM[BM25 rebuilt from token cache\nnew snippets tokenised only]
  N --> L[lineage matching\nold vs new of changed files]
  R & R2 & N --> M[members[B] · latest[repo] = B]
```

What is reused, and at which level:

| Level | Key | Effect |
|---|---|---|
| File | git blob SHA (content SHA-1 for plain directories) | Unchanged files are neither read nor parsed |
| Snippet version | `(symbol_key, content_hash)` | Unchanged functions in changed files are not re-added or re-embedded |
| Embedding | `sha1(model fingerprint, prompt, text)` in `artifacts/cache/embeddings.sqlite` | Identical text is never encoded twice, including across processes and restarts |
| BM25 tokens | `content_hash + symbol_key` | Only new snippets are tokenized; the sparse matrices are rebuilt from cached tokens |

`UpdateStats` records, per commit: files changed, added, deleted and reused; new snippet versions; reused snippets; embeddings computed and reused; lineage edges; and seconds per phase. `tests/test_versions.py` asserts that:

* across the demo history no commit embeds more than its new snippet versions (`embeddings_computed <= snippets_new_versions`), and unchanged files are reused,
* editing one constant in one file of the sample repo re-parses that file only and computes **exactly one** new embedding (`test_unchanged_snippets_are_not_re_embedded_on_edit`),
* re-indexing the same commit is a no-op,
* a saved index can be loaded and indexing continues incrementally.

### Benchmark: full rebuild vs incremental

```bash
python scripts/benchmark_versions.py                                              # demo history
python scripts/benchmark_versions.py --repo artifacts/repos/click --commits 6 \
       --config configs/full.yaml --set dense.model=minilm
```

For every commit `c_i`, the **incremental** engine, already holding `c_{i-1}`, indexes `c_i`. A **fresh** engine indexes `c_i` from scratch for comparison. Neither side uses the persistent embedding cache, so incremental reuse is by snippet identity only. Results go to `artifacts/benchmarks/versioning_*.{json,md}`; the README shows the measured tables.

## Version-scoped search

The store is append-only and temporal: each snippet version lists the commits it belongs to. A search accepts `version`:

* `null` / `latest`: each repository's latest indexed commit.
* `<sha or prefix>`: exactly that commit's snippets. The results equal a per-commit index, because masked dense search is exact.
* `all`: every stored version, with lineage collapse keeping one per lineage unless the query is an EVOLUTION question.

```bash
python scripts/search.py -q "normalize input" --version 3f2a1c
```

## Bonus: lineage and evolutionary retrieval

`versions/lineage.py`, `LineageMatcher`. For each changed file pair it compares the snippets that disappeared (old) with those that appeared (new):

| Step | Rule | Relation |
|---|---|---|
| 1 | same `symbol_key` (file + qualified name) | `modified` |
| 2 | same qualified name, and git reports the file as renamed (`git diff -M`) | `file_renamed` |
| 3 | identical `content_hash` or `ast_hash` elsewhere | `moved` |
| 4 | same kind and similarity ≥ `versions.lineage_similarity` (0.6), where similarity = 0.5·Jaccard(token-bigram shingles) + 0.5·difflib ratio (+0.1 for an identical AST shape) | `renamed` if the name changed, else `evolved` |

Each new version inherits its predecessor's `lineage_id`, and an `EVOLVED_TO` edge enters the graph:

```
auth.py::authenticate@v1 ──modified──▶ auth.py::authenticate@v2 ──renamed──▶ auth.py::AuthService.validate_user@v3
text_utils.py::normalize@v1 ──file_renamed──▶ preprocessing.py::normalize@v2 ──modified──▶ ...
```

### Evolutionary queries

When the intent is EVOLUTION ("How did authentication change between versions?"):

1. The scope becomes `all`.
2. Diversity keeps several versions of each lineage (exact duplicates are still removed).
3. The results of each lineage are grouped, oldest first.
4. Each result carries its lineage chain (`GET /snippets/{id}/lineage` returns the full chain with code).

For every other query, lineage collapse keeps only the best version per lineage, so the ten results show ten different pieces of code, not ten commits of one function.

### Try it

```bash
python scripts/make_demo_history.py      # artifacts/demo_history_repo: v1 → v2 → v3 real commits
python scripts/index_repo.py artifacts/demo_history_repo --history --index-dir artifacts/index_history
python scripts/search.py --index-dir artifacts/index_history -q "How did authentication change between versions?"
python scripts/search.py --index-dir artifacts/index_history -q "Where is MAX_RETRIES configured?" --version all
```

The demo history is built from `examples/sample_repo_history/v1`, `v2` and `examples/sample_repo` (v3). It contains a function rename (`authenticate` → `validate_user`), a file rename (`text_utils.py` → `preprocessing.py`), a constant change (`MAX_RETRIES` 3 → 5), added and deleted files, and unchanged files.
