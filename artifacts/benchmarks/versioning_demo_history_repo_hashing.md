### Incremental vs full rebuild: demo_history_repo (hashing)

| Commit | Files changed/added/deleted | Live snippets | Incremental s | Full rebuild s | Speed-up | Embeddings (inc / full) |
|---|---|---|---|---|---|---|
| 38945a955e | 4/2/1 of 7 | 28 | 0.488 | 0.646 | 1.32x | 23 / 28 |
| 000d70b375 | 6/0/0 of 7 | 34 | 0.410 | 0.591 | 1.44x | 15 / 34 |

Total: incremental 0.90s vs full 1.24s (1.38x); embeddings computed 38 vs 62.
