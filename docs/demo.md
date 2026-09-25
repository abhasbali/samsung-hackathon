# Demo guide

A 5-minute walkthrough showing *why* different retrievers matter. Every result comes from the real engine over the bundled sample repository; nothing is pre-recorded.

## Setup

```bash
pip install -r requirements.txt
python scripts/index_repo.py examples/sample_repo                  # current version
python scripts/make_demo_history.py                                # v1 → v2 → v3 git history
python scripts/index_repo.py artifacts/demo_history_repo --history --index-dir artifacts/index_history

cd frontend && npm install && npm run build && cd ..
uvicorn codefusion.api.main:app --port 8000                        # UI at http://localhost:8000
```

(Use `CODEFUSION_INDEX_DIR=artifacts/index_history` to serve the history index instead. Offline, add `--config configs/test_tiny.yaml` to the index commands and set `CODEFUSION_CONFIG=configs/test_tiny.yaml`.)

The sample repository is a small web-service-style app:

```
main.py          main() → handle_request() → preprocess_input() → model.predict()
                 register_user() → AuthService … → UserRepository.save_user()
preprocessing.py normalize_input(), validate_input(), preprocess_input()
auth.py          hash_password(), issue_token(), verify_token(), AuthService.validate_user(), current_user()
database.py      connect(), with_retries() [uses MAX_RETRIES], UserRepository.find_user() / save_user()
config.py        MAX_RETRIES, RETRY_BACKOFF_SECONDS, DATABASE_URL, TOKEN_TTL_SECONDS, load_config()
model.py         sigmoid(), predict()
errors.py        custom exceptions
```

## Scenarios

The demo query chips in the UI come from `examples/demo_queries.json`.

| # | Query | What to point out |
|---|---|---|
| 1 | Where is authentication implemented? | IMPLEMENTATION intent. `auth.py` dominates. Query expansion adds auth vocabulary (token, verify, password) to BM25. |
| 2 | **Who calls validate_user?** | CALLER intent, target `validate_user`. The **caller** `main` ranks #1 and the definition #2: symbol call-sites (`symbol:call:validate_user`) plus graph evidence (`caller of validate_user`). A pure embedding search ranks the definition first, because it looks most similar to the question. |
| 3 | What functions are called before the user is saved? | DATA_FLOW ("before" makes it an ordering question, not CALLEE). `save_user` and its caller `register_user` lead. **Show call chain** on `save_user` draws `register_user() ↓ save_user() ↓ with_retries()`. |
| 4 | Where is MAX_RETRIES configured? | CONFIGURATION. `config.py` ranks first on `symbol:def:MAX_RETRIES` + `defines queried symbol`. `with_retries` (which imports and uses it) follows with `references queried symbol`. |
| 5 | How is input normalized before prediction? | DATA_FLOW over `preprocess_input → normalize_input / validate_input → predict`. The call chain view shows the pipeline. |
| 6 | What functions does preprocess_input call? | CALLEE. The definition, then its callees `validate_input` and `normalize_input`; the caller `handle_request` comes after them. For this intent a mention of the target elsewhere is a caller, so only the definition counts as a symbol hit. |
| 7 | Which errors are raised when the database connection fails? | ERROR. The query proposes exception-class names (`database` → `DatabaseError`), so the raise site `with_retries` (`call:DatabaseError`) and the `DatabaseError` definition lead instead of the files that merely `import errors`. |
| 8 | How did authentication change between versions? | Serve `artifacts/index_history`. EVOLUTION intent, scope `all`: `authenticate@v1 → @v2 → validate_user@v3` appear together with their lineage chain. Re-run query 4 with version `all`: only one version per lineage survives, and the explain panel lists the dropped historical duplicates. |

## What to show in each panel

* **Detected intent**, with confidence and target symbols.
* **Retrieval timing**: total ms and a per-stage bar (preprocess, dense, bm25, symbol, graph, fusion, rerank, structural, diversity).
The exact output of every scenario with the default model is recorded in `artifacts/demo/demo_transcript.md` (`python scripts/run_demo_queries.py`).

* **Retriever status**: whether each retriever is on, its hit count and its intent-adapted RRF weight. Compare query 2 (graph and symbol weighted up) with query 1.
* **Candidates**: retrieved → fused (unique) → reranked → final.
* **Why retrieved**: each retriever's rank and RRF contribution, reranker score, structural boost, and the evidence strings.
* **Experiments tab**: the measured AppsRetrieval, ablation, embedding and versioning tables. Anything not run shows *Not evaluated*.
* **Index & versions tab**: index a path live and watch per-commit reuse (files reused, embeddings computed vs reused, lineage relations).

## CLI equivalents

```bash
python scripts/search.py -q "Who calls validate_user?" --top-k 3
python scripts/search.py -q "Who calls validate_user?" --json | python -m json.tool | less   # full provenance
python scripts/search.py --index-dir artifacts/index_history -q "How did authentication change between versions?"
```
