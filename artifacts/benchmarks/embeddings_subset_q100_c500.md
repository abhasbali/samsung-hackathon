### Embedding models on AppsRetrieval (subset_q100_c500)

| Model | Params | NDCG@10 | MRR@10 | P50 query ms | P95 query ms | Index s | docs/s | RAM Δ MB | Peak VRAM MB | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| jina-code-0.5b | 0.5B | 0.9602 | 0.9500 | 105.2 | 288.0 | 0.0 | n/a | 1074 | 2073 | ok |
| jina-v2-base-code | 161M | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | NOT RUN: embedding model 'jinaai/jina-embeddings-v2-base-code' unavailable: ImportError: cannot import name 'find_pruneable_heads_and_indices' from 'transformers.pytorch_utils' (/usr/local/lib/python3.13/dist-packages/transformers/pytorch_utils.py) |
| jina-code-1.5b | 1.5B | 0.9616 | 0.9517 | 335.1 | 769.6 | 92.9 | 4.28 | 637 | 8878 | ok |
| qodo-1.5b | 1.5B | 0.6980 | 0.6500 | 345.9 | 783.4 | 97.5 | 4.14 | n/a | 8875 | ok |

RAM Δ is the process RSS change while loading the model (n/a when negative: the previous model's memory was released). On GPU runs the weights live in VRAM, so Peak VRAM is the meaningful footprint. Index s / docs/s are 0 / n/a when every embedding came from the shared cache.
