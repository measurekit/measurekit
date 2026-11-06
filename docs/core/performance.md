# Performance of core

What the numerics of `@measurekit/core` cost against base R and stats, and how much its entry point adds to a bundle. core reads no
untrusted input of its own: the readers of the libraries do, and their pages give the memory they need.

```mermaid
flowchart LR
  speed["Each routine against base R or stats<br/>pnpm bench"] --> time["Budget of the fastest sample<br/>packages/core/test/budgets.ts"]
  size["Each entry point bundled alone<br/>pnpm test"] --> gzip["Gzip budget and the modules it may hold<br/>scripts/size-budgets.ts"]
```

## Speed

Each benchmark times the numerics against base R and stats on the same data. How each side is timed, how the ratio is
read, how the budgets are set and how to refresh this page are in the [method](../../validation/README.md#benchmarks).

<!-- charts -->

### Where measurekit is slower

```mermaid
%%{init: {"xyChart": {"height": 252, "xAxis": {"labelFontSize": 11}}}}%%
xychart-beta horizontal
  title "Tasks that take more than 1.5 times as long as the reference"
  x-axis ["numerics.cholesky", "numerics.normalQuantile", "random.normal"]
  y-axis "Time of measurekit over the time of the reference" 0 --> 3
  bar [2.1, 2, 1.8]
  line [1, 1, 1]
```

A bar above the line takes longer than the reference.

### Every task

```mermaid
%%{init: {"xyChart": {"height": 320, "xAxis": {"labelFontSize": 11}}}}%%
xychart-beta horizontal
  title "numerics, random"
  x-axis ["normalQuantile", "logistic", "cholesky", "uniform", "normal"]
  y-axis "Time of measurekit over the time of the reference" 0 --> 3
  bar [2, 0.49, 2.1, 0.74, 1.8]
  line [1, 1, 1, 1, 1]
```

<!-- /charts -->

### Results

<!-- tables -->

Measured on Apple M5 with 10 cores, 16 GB of memory, Node.js 22.23.4 and R version 4.6.1 (2026-06-24).

| Task                         | Size                              | measurekit (ms) | Reference            | Reference (ms) |    Ratio |
| ---------------------------- | --------------------------------- | --------------: | -------------------- | -------------: | -------: |
| **Normal quantile function** | 1,000,000 probabilities           |            16.1 | stats 4.6.1 `qnorm`  |           8.00 |   **2×** |
| Logistic function            | 1,000,000 points                  |            3.82 | stats 4.6.1 `plogis` |           7.83 |    0.49× |
| **Cholesky factorization**   | A correlation matrix of order 200 |           0.332 | base 4.6.1 `chol`    |          0.160 | **2.1×** |
| Uniform draws                | 1,000,000 draws                   |            3.74 | stats 4.6.1 `runif`  |           5.06 |    0.74× |
| **Normal draws**             | 1,000,000 draws                   |            33.5 | stats 4.6.1 `rnorm`  |           18.3 | **1.8×** |

- Normal quantile function: on Node.js 22, the CI runner's and these tables', the loop takes about twice as long as on Node.js 24 and 26: 8.7 ms on Node.js 26.8.1, run in turns with Node.js 22 on the same machine.
- Cholesky factorization: R's LAPACK calls OpenBLAS, whose kernels add and multiply two doubles per instruction and fuse each pair; JavaScript has neither. measurekit's loops share their reads across a block of 4 rows and 3 columns, and the same loops in C compiled without vector instructions or fused multiply-add take 0.77 times as long as measurekit's and, compiled with both, 0.48 times as long.
- Normal draws: on Node.js 22, the CI runner's and these tables', the loop takes about twice as long as on Node.js 24 and 26: 18 ms on Node.js 26.8.1, run in turns with Node.js 22 on the same machine.

<!-- /tables -->

## Size

Each entry point bundled alone with esbuild ([method](../../validation/README.md#sizes-and-peak-memory)).

| Entry point        | Minified (bytes) | Gzip (bytes) | Modules it holds besides its own |
| ------------------ | ---------------: | -----------: | -------------------------------- |
| `@measurekit/core` |           16,671 |        6,924 | nothing                          |
