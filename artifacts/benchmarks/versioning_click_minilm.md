### Incremental vs full rebuild: click (minilm)

| Commit | Files changed/added/deleted | Live snippets | Incremental s | Full rebuild s | Speed-up | Embeddings (inc / full) |
|---|---|---|---|---|---|---|
| 68e7ea7228 | 0/0/0 of 79 | 1471 | 0.017 | 6.846 | 412.58x | 0 / 1452 |
| 36baa15ff8 | 0/0/0 of 79 | 1471 | 0.012 | 5.604 | 483.23x | 0 / 1452 |
| 6aabf099bf | 8/12/1 of 90 | 1512 | 0.926 | 6.466 | 6.98x | 66 / 1486 |
| 3cbaa76b60 | 7/1/0 of 91 | 1520 | 1.014 | 5.397 | 5.32x | 28 / 1494 |
| 06b2a67874 | 4/0/0 of 91 | 1562 | 0.806 | 7.058 | 8.76x | 52 / 1536 |

Total: incremental 2.77s vs full 31.37s (11.3x); embeddings computed 146 vs 7420.
