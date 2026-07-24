### Live performance of the upgraded engine (Flask stack, MySQL 9.7.1)

100 sequential requests per method (after warm-up) + 50-request burst on 4 threads. Isolated process per method.

| Method | Median (ms) | Mean (ms) | p95 (ms) | Throughput (req/s) | RSS (MB) | Empty results |
|---|---|---|---|---|---|---|
| content | 1 | 1 | 1 | 1120.6 | 157 | 0% |
| collaborative | 1 | 1 | 1 | 2919.0 | 137 | 2% |
| trend | 26 | 26 | 28 | 135.3 | 137 | 0% |
| hybrid | 34 | 34 | 40 | 107.2 | 156 | 0% |
