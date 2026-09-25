### AppsRetrieval ablations (full)

| Exp | Configuration | NDCG@10 | MRR@10 | P50 ms | P95 ms | Index s | Status |
|---|---|---|---|---|---|---|---|
| A | dense only | 0.8380 | 0.8076 | 2.9 | 4.3 | 17.0 | ok |
| B | BM25 only | 0.0220 | 0.0169 | 1.5 | 2.0 | 20.0 | ok |
| C | dense + BM25 (score fusion) | 0.5794 | 0.4637 | 3.9 | 6.0 | 16.6 | ok |
| D | dense + BM25 + RRF | 0.4542 | 0.3209 | 3.8 | 5.9 | 16.6 | ok |
| D2 | dense + BM25 + RRF (dense-weighted) | 0.7606 | 0.7043 | 3.8 | 6.0 | 17.1 | ok |
| E | + identifier preprocessing | 0.4538 | 0.3238 | 3.9 | 6.0 | 18.8 | ok |
| F | + structural representation | 0.4567 | 0.3267 | 4.6 | 7.5 | 19.7 | ok |
| G | + symbol retrieval | 0.1746 | 0.1243 | 6.0 | 16.2 | 19.2 | ok |

Latency is per-query engine search time; batched query embedding is excluded and reported in the JSON.
