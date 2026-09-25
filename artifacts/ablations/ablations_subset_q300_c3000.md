### AppsRetrieval ablations (subset_q300_c3000)

| Exp | Configuration | NDCG@10 | MRR@10 | P50 ms | P95 ms | Index s | Status |
|---|---|---|---|---|---|---|---|
| A | dense only | 0.8982 | 0.8737 | 2.6 | 3.9 | 6.9 | ok |
| D | dense + BM25 + RRF | 0.4236 | 0.2871 | 2.6 | 3.4 | 7.8 | ok |
| G | + symbol retrieval | 0.1615 | 0.1395 | 3.8 | 6.3 | 8.0 | ok |
| H | + reranker | 0.3287 | 0.2463 | 2502.3 | 2970.8 | 15.5 | ok |

Latency is per-query engine search time; batched query embedding is excluded and reported in the JSON.
