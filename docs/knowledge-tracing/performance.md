# Performance of knowledge-tracing

What the hot paths of `@measurekit/knowledge-tracing` cost against their references, how much each entry point adds to a
bundle, and the memory that each reader of untrusted input needs on its worst input. A budget holds each number and
fails a gate of CI: `pnpm bench` for the time, `pnpm test` for the size and the memory.

```mermaid
flowchart LR
  speed["Each hot path against its reference<br/>pnpm bench"] --> time["Budget of the fastest sample<br/>packages/knowledge-tracing/test/budgets.ts"]
  size["Each entry point bundled alone<br/>pnpm test"] --> gzip["Gzip budget and the modules it may hold<br/>scripts/size-budgets.ts"]
  memory["Peak memory of each reader<br/>of untrusted input, pnpm test"] --> heap["Heap and rise of each case<br/>scripts/peak-memory.ts"]
```

## Speed

Each benchmark times the hot path of each module against pyBKT, or the R package that its [validation](validation.md)
uses, on the same data, with one thread. The ratio is the time of measurekit over the time of the reference: above 1,
measurekit is slower, and above 1.5 the row is in bold. A benchmark fails `pnpm bench` when its fastest sample is above
its budget in `packages/knowledge-tracing/test/budgets.ts`: five times the fastest sample of the CI runner, rounded up
to one significant digit, as STANDARDS.md section 11 sets every budget. How each side is timed, how the budgets were
measured and how to refresh this page are in [validation](../../validation/README.md#benchmarks).

<!-- charts -->

### Where measurekit is slower

```mermaid
%%{init: {"xyChart": {"height": 184, "xAxis": {"labelFontSize": 11}}}}%%
xychart-beta horizontal
  title "Tasks that take more than 1.5 times as long as the reference"
  x-axis ["logistic.fitItems"]
  y-axis "Time of measurekit over the time of the reference" 0 --> 2
  bar [1.7]
  line [1]
```

A bar above the line takes longer than the reference.

### Every task

```mermaid
%%{init: {"xyChart": {"height": 524, "xAxis": {"labelFontSize": 11}}}}%%
xychart-beta horizontal
  title "bkt, elo, evaluate, irt"
  x-axis ["fit", "fitSkills", "fitLessons", "fitPairs", "replay", "replay", "auc", "metricsByOpportunity", "replay", "nominal", "drift"]
  y-axis "Time of measurekit over the time of the reference" 0 --> 2
  bar [0.0051, 0.026, 0.0064, 0.0062, 0.025, 1, 0.34, 0.35, 0.25, 0.28, 0.06]
  line [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

```mermaid
%%{init: {"xyChart": {"height": 626, "xAxis": {"labelFontSize": 11}}}}%%
xychart-beta horizontal
  title "kst, logistic"
  x-axis ["unionClosure", "structureBase", "clauseStructure", "blim", "iita", "mdml", "identifiability", "validity", "diffTest", "features", "fitItems", "fitManyItems", "searchRates", "afm"]
  y-axis "Time of measurekit over the time of the reference" 0 --> 2
  bar [0.011, 0.000047, 0.0015, 1.1, 0.000026, 0.46, 0.84, 0.015, 0.00047, 0.17, 1.7, 0.88, 0.52, 0.031]
  line [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

<!-- /charts -->

### Results

<!-- tables -->

Measured on Apple M5 with 10 cores, 16 GB of memory, Node.js 22.23.4 and R version 4.6.1 (2026-06-24).

| Task                                                                                                                                  | Size                                             | measurekit (ms) | Reference                                 | Reference (ms) |     Ratio |
| ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ | --------------: | ----------------------------------------- | -------------: | --------: |
| Fit of the skills of a log by EM from five starts                                                                                     | 20,000 attempts of 200 learners at 5 skills      |            95.3 | pyBKT 1.4.3 `Model.fit`                   |         18,522 |   0.0051× |
| Fit of the skills of a log by EM from five starts, each learner at 20 of many skills                                                  | 20,000 attempts of 500 learners at 400 skills    |             645 | pyBKT 1.4.3 `Model.fit`                   |         24,790 |    0.026× |
| Fit of the skills of a log by EM from five starts, a fifth of its attempts without an outcome                                         | 20,000 attempts of 200 learners at 5 skills      |             133 | pyBKT 1.4.3 `Model.fit`, correct = -1     |         20,634 |   0.0064× |
| Fit of the skills of a log by EM from five starts, learn and forget rates by pair of items                                            | 20,000 attempts of 200 learners at 5 skills      |             213 | pyBKT 1.4.3 `Model.fit(multipair = True)` |         34,545 |   0.0062× |
| Replay of the log through the fitted model, a prediction and an update for each attempt                                               | 20,000 attempts of 200 learners at 5 skills      |            17.8 | pyBKT 1.4.3 `Model.predict`               |            710 |    0.025× |
| Replay of a log through the learner-item Elo model                                                                                    | 100,000 attempts of 2,000 learners at 400 items  |            52.1 | base R, a loop over the attempts          |           50.0 |        1× |
| The area under the ROC curve                                                                                                          | 100,000 predictions                              |            10.8 | pROC 1.19.1 `auc(roc(direction = "<"))`   |           31.5 |     0.34× |
| The AUC, RMSE, log-loss and accuracy of the predictions                                                                               | 100,000 predictions                              |            16.0 | none                                      |                |           |
| The metrics of predictions by opportunity, over all skills and for each skill                                                         | 50,000 events of 2,000 learners at 5 skills      |            25.4 | pROC 1.19.1 `auc(roc(direction = "<"))`   |           73.0 |     0.35× |
| Replay of the responses of learners, the grid posterior of the ability updated after each                                             | 2,000 learners, 40 responses each                |             736 | mirt 1.48 `fscores(method = "EAP")`       |          2,940 |     0.25× |
| Replay of the responses of learners to nominal items, the grid posterior of the ability updated after each                            | 2,000 learners, 40 responses each, 4 categories  |           1,098 | mirt 1.48 `fscores(method = "EAP")`       |          3,943 |     0.28× |
| Replay of the responses of learners, the posterior advanced by drift and updated after each                                           | 200 learners, 40 responses each, 241 points      |             526 | base R, the forward pass in logs          |          8,753 |     0.06× |
| Closure under union of the 12 atoms of a space                                                                                        | 12 elements, 125 states                          |           0.205 | kst 0.5.6 `kspace`                        |           18.7 |    0.011× |
| Base of a space                                                                                                                       | 12 elements, 125 states                          |           0.129 | kst 0.5.6 `kbase`                         |          2,721 | 0.000047× |
| Space of clauses, by enumeration of the subsets in base R                                                                             | 12 elements, 4096 subsets                        |           0.869 | base R `expand.grid` and `apply`          |            566 |   0.0015× |
| Quasi-ordinal space of a cycle of prerequisites, each element requiring the next                                                      | 1,000 elements                                   |            12.7 | none                                      |                |           |
| Fit of the basic local independence model of a chain of 12 items by EM                                                                | 1,000 persons, 12 items, 13 states               |            39.2 | pks 0.8.0 `blim`                          |           37.0 |      1.1× |
| Inductive item tree analysis of the responses                                                                                         | 1,000 persons, 12 items                          |            3.69 | DAKS 2.1.3 `iita(v = 1)`                  |        140,096 | 0.000026× |
| Fit of the basic local independence model of a chain of 12 items by minimum discrepancy maximum likelihood                            | 1,000 persons, 12 items, 13 states               |            94.3 | pks 0.8.0 `blim(method = "MDML")`         |            203 |     0.46× |
| Rank of the Jacobian of the pattern probabilities of the model of a chain of 12 items                                                 | 12 items, 13 states, 36 parameters               |            16.5 | pks 0.8.0 `jacobian` and `qr`             |           19.7 |     0.84× |
| Validity of the chain against the responses: gamma, violational coefficient, solution rates, discrepancy index and distance agreement | 1,000 persons, 12 items, 13 states               |            3.91 | kst 0.5.6 `kvalidate`                     |            260 |    0.015× |
| Test of the fit of the chain of 6 items to the responses                                                                              | 1,000 persons, 6 items                           |            1.71 | DAKS 2.1.3 `z_test(v = 1)`                |          3,656 |  0.00047× |
| Test of one edge of the chain of 12 items, on responses with 3 in 10 missing                                                          | 1,000 persons, 12 items                          |            2.66 | none                                      |                |           |
| The 16 features of LKT's other names, before each event                                                                               | 1,941 events of 17 learners, 16 features         |            15.5 | LKT 1.7.0 `computefeatures`               |           90.0 |     0.17× |
| **Fit with an intercept for each skill and item, a proficiency for each learner and a ridge of 1**                                    | 20,000 events of 200 learners at 1,000 items     |            70.8 | LiblineaR 2.10.26 `LiblineaR(type = 0)`   |           40.5 |  **1.7×** |
| Fit with an intercept for each skill and item, a proficiency for each learner and a ridge of 1                                        | 150,000 events of 3,000 learners at 49,546 items |             571 | LiblineaR 2.10.26 `LiblineaR(type = 0)`   |            652 |     0.88× |
| Search for the decay of a proportion of successes and the exponent of a recency                                                       | 3,600 events of 60 learners at 4 skills, 2 rates |             340 | LKT 1.7.0 `LKT(seedpars = c(0.5, 0.5))`   |            655 |     0.52× |
| Fit of the additive factors model by maximum likelihood                                                                               | 20,000 attempts of 200 learners at 5 skills      |            23.8 | base R `glm`, binomial family             |            759 |    0.031× |
| The candidates ranked by their distance to the band of success of a learner                                                           | 500 candidate items, an Elo model                |           0.180 | none                                      |                |           |
| A log built from and checked for events                                                                                               | 100,000 events                                   |            43.3 | none                                      |                |           |

- Fit of the skills of a log by EM from five starts, each learner at 20 of many skills: pyBKT stops a start when the log-likelihood changes by at most 0.005, or after 100 iterations, and measurekit at 1e-6 or 500, so measurekit runs more iterations: a median of 38 for a skill's best start, and 4 of the 400 skills reach 500.
- Replay of the responses of learners, the grid posterior of the ability updated after each: mirt scores each of the 40 prefixes of the responses from the first; measurekit updates the posterior by the one response and predicts it. A scoring of the final patterns alone takes a fortieth of the time.
- Replay of the responses of learners to nominal items, the grid posterior of the ability updated after each: as for the 2PL items, mirt scores each of the 40 prefixes of the responses from the first.
- Space of clauses, by enumeration of the subsets in base R: no package takes clauses.
- Fit of the basic local independence model of a chain of 12 items by minimum discrepancy maximum likelihood: pks and measurekit both stop within 1e-9 of the maximum: pks when no parameter changes by more than 1e-8, after 330 iterations, and measurekit when the log-likelihood changes by at most 1e-8, after 331. At pks's tolerance of 1e-4 it stops after 41 iterations, 2.5e-3 from the maximum.
- The 16 features of LKT's other names, before each event: measurekit replays the log once with every feature in one model.
- Fit with an intercept for each skill and item, a proficiency for each learner and a ridge of 1: the time of the reference includes the features and the design from the events, as measurekit's does; both stop at the gradient that LiblineaR reaches at epsilon 0.0001 (0.0074 at 1,000 items, 0.22 at 50,000), where measurekit takes longer at 1,000 items: its solver is JavaScript, each Hessian-vector product taking 3.5 times as long as the same loop in C, and its features and design take 0.7 and 0.4 times as long as the reference's; at its default gradient of 1e-8 it takes 1.7 times as long as at these.
- Search for the decay of a proportion of successes and the exponent of a recency: each fit of the search is one of LiblineaR in LKT and one of Newton's method in measurekit.
- Fit of the additive factors model by maximum likelihood: the proficiency of a learner is a fixed effect, as measurekit's default is none.

<!-- /tables -->

## Size

Each entry point bundled alone, as a host's bundler makes it: minified with esbuild and code splitting on, `export *` of
the entry, core from its sources. The bytes and the modules are those of the entry chunk and the chunks it imports
statically, which load with it; a module that a dynamic `import()` loads later is a chunk of its own and is not counted.
The modules each holds are those the import rules of [AGENTS.md](../../AGENTS.md#dependencies) allow it. Every package
declares `sideEffects: false`, so a named import holds only what that name needs.

| Entry point                              | Minified (bytes) | Gzip (bytes) | Modules it holds besides its own |
| ---------------------------------------- | ---------------: | -----------: | -------------------------------- |
| `@measurekit/knowledge-tracing`          |            2,572 |        1,355 | core                             |
| `@measurekit/knowledge-tracing/bkt`      |           26,192 |        9,325 | core, data, math                 |
| `@measurekit/knowledge-tracing/data`     |            6,343 |        2,465 | core                             |
| `@measurekit/knowledge-tracing/elo`      |            7,269 |        2,541 | core, data, math                 |
| `@measurekit/knowledge-tracing/evaluate` |           18,103 |        6,611 | core, data, math, model          |
| `@measurekit/knowledge-tracing/irt`      |           15,658 |        5,986 | core, data, math                 |
| `@measurekit/knowledge-tracing/kst`      |           48,903 |       17,672 | core, data, math                 |
| `@measurekit/knowledge-tracing/logistic` |           50,319 |       16,571 | core, data, math, model          |
| `@measurekit/knowledge-tracing/practice` |            6,712 |        2,482 | core, data, model                |
| `@measurekit/knowledge-tracing/sim`      |            5,046 |        1,879 | core, data, model                |

`scripts/test/entry-sizes.test.ts`, which `pnpm test --project scripts` runs without a build, bundles each entry point
and fails when it takes more than its budget after gzip, or holds a module that its budget in `scripts/size-budgets.ts`
does not list. A budget is 1.25 times the size measured after gzip, rounded up to the next 500 bytes. `node
scripts/entry-sizes.ts` prints the sizes; the table gives them on Node.js 22, the version of the runner, and the gzip of
Node.js 26 is up to 1.2 percent less.

## Memory

What each reader of untrusted input adds to the peak resident memory of a Node.js process, on its worst inputs within
the limits of [SECURITY.md](../../SECURITY.md): the peak of a process that parses the input from JSON and reads it, less
the peak of the same process that parses it and does not, the median of three pairs run one after the other. Each reader
grows in proportion to its input, but for the analyses of the relations between the elements of a knowledge structure,
whose limits bound them.

| Reader                       | Input                                                                | JSON (MB) | Peak added (MB) |
| ---------------------------- | -------------------------------------------------------------------- | --------- | --------------- |
| `buildEventLog`              | 1,000,000 events of 1,000 learners and 1,000 items                   | 67        | 76              |
| `buildEventLog`              | 1,000,000 events, each of its own learner, item and skill            | 82        | 12              |
| `bkt` with stored parameters | 100,000 skills                                                       | 6.6       | 22              |
| `kst` with a structure       | 10,000 states of 64 items                                            | 1.8       | 1               |
| `iita`                       | 1,000,000 events, each of its own learner and item, which it refuses | 73        | 100             |
| `iita`                       | 1,000 items, 276 learners: 24 candidates of 23.9 million cells       | 18        | 237             |
| `diffTest`                   | 1,000 items, 200 learners, a chain, from their log                   | 13        | 26              |
| `structureValidity`          | 1,000 items, 200 learners, a chain, from their log and structure     | 13        | 4               |
| `surmiseStructure`           | 1,000 elements, 12 free and a class of 988: 8,192 states             | 0.02      | 101             |
| `kst`, its masks             | 4,096 sets of 16 skills, 65,536 states                               | 0.33      | 240             |

A log takes at most about 1.2 times the bytes of its JSON, and stored BKT parameters about 3 times. The analyses of
knowledge structures take up to about 240 MB within their limits: IITA's candidates from 18 MB of responses, and the
masks of a model from a few hundred kilobytes of sets of skills.

`scripts/test/peak-memory.test.ts`, which `pnpm test --project scripts` runs without a build, measures the readers of
`scripts/peak-memory.ts`, every row of the table above among them, from the sources, each in a pair of child
processes whose heap is bounded, as STANDARDS.md section 7 asks: twice the least heap in which the reading ended on
Node.js 22, 24 and 26, rounded up to 16 MiB, so that a reading that keeps more alive at once fails, whenever the
collector runs and however busy the machine is. The test fails too when a reading adds more than its budget to the peak:
twice the largest rise measured on Node.js 22 and 24, on macOS and on Linux, rounded up to 10 MB, which bounds what the
heap does not hold, the buffers of typed arrays. `node scripts/peak-memory.ts` prints the rises, the median of three
pairs, with a heap of 8 GiB, as a host's process collects later than a bounded one, and `node scripts/peak-memory.ts
--heap` the least heap of each reading.
