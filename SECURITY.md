# Security

Report vulnerabilities privately through the repository's security advisory form rather than a public issue. Name the
package.

Points to be aware of when deploying `@measurekit/psychometrics`, `@measurekit/knowledge-tracing` or `@measurekit/core`:

- **No I/O.** Neither package reads files, makes network requests, reads environment variables or runs code it is given.
  None has a third-party runtime dependency: `@measurekit/psychometrics` and `@measurekit/knowledge-tracing` depend only
  on `@measurekit/core`.
- **Untrusted input.** Every public function validates its input and throws `DataError` on malformed data, so data from
  users or files can be passed in directly. The learner models of `@measurekit/knowledge-tracing` also check the states
  and parameters they are given, so states from a database or browser storage can be passed in directly. A learner can
  edit a state kept in browser storage; keep states on a server where estimates matter.
- **Resource use.** Time and memory grow with the size of the input. In `@measurekit/psychometrics`, item statistics
  take time proportional to persons × items, and `bootstrapEquating` time proportional to the persons its counts hold
  times the replications. In `@measurekit/knowledge-tracing`, a log is checked and replayed in time proportional to its
  events, and cross-validation fits the model once per fold. Limit the size of data accepted from untrusted sources
  before analyzing it on a server: a response matrix or a log takes at most about 50 times the bytes of its JSON, a
  Q-matrix or a diagnostic pool of a few hundred bytes up to 2 GB within the limits below
  ([peak memory](docs/performance.md#peak-memory)). The time of inductive item tree analysis grows faster than its
  limits bound: 1,000 learners take 10 ms at 12 items and 11 to 21 s at 400, where DAKS takes 2 minutes at 12 items
  ([kst](docs/knowledge-tracing/kst.md#inductive-item-tree-analysis)). So does the time of a statistic of every pair of
  items, which grows with the pairs times the persons, of Mokken's conditional association, with the cube of the items
  times the persons, 14 s at 1,000 items of 8 persons ([mokken](docs/psychometrics/mokken.md)). Run these on data from
  users in a worker with a deadline, and end the worker when it passes.
  An exploratory calibration of several dimensions starts from an eigendecomposition of the items' correlations, whose
  time grows as the cube of the items: about 3 minutes at 2,000 items
  ([multidimensional](docs/psychometrics/multidimensional.md#time)).
- **Limits.** Tables are sized by the values that occur in the data, not by the highest value, so that a small input
  with one large response costs no more than another of its size. What would grow faster than the input has a limit,
  and passing it throws `DataError`:

  | Input                                                                                              | Limit                                      | Guide                                                                                                         |
  | -------------------------------------------------------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
  | A response, or the score of a faceted score                                                        | An integer from 0 to 2³¹ - 1               | [data](docs/psychometrics/data.md)                                                                            |
  | Attributes of a Q-matrix; of a diagnostic adaptive test                                            | 20: a fit holds 2^K profiles, 1.0 GB       | [cdm](docs/psychometrics/cdm.md#calibration)                                                                  |
  | Items × 2^K of a diagnosis model; of a diagnostic adaptive test                                    | 2^26: 2.0 GB at 64 items of 20 attributes  | [cdm](docs/psychometrics/cdm.md#calibration), [cat](docs/psychometrics/cat.md#diagnostic-adaptive-tests)      |
  | Latent groups of a diagnosis model; of a diagnostic adaptive test                                  | 2^20 over all items, 2^k for k attributes  | [cdm](docs/psychometrics/cdm.md#calibration), [cat](docs/psychometrics/cat.md#diagnostic-adaptive-tests)      |
  | Success probabilities with incomplete standard errors                                              | 3,162 over all items: 80 MB of information | [cdm](docs/psychometrics/cdm.md#calibration)                                                                  |
  | Items of a limited-information statistic, of IRT or diagnosis; its moments, of C2 with categories  | 60 and 1,830: a covariance matrix of 27 MB | [fit](docs/psychometrics/fit.md#limited-information-fit), [cdm](docs/psychometrics/cdm.md#model-and-item-fit) |
  | Categories of a Mokken scale                                                                       | 10                                         | [mokken](docs/psychometrics/mokken.md)                                                                        |
  | Observed categories of an item in a polychoric correlation                                         | 8, as psych's `polychoric`                 | [dimensionality](docs/psychometrics/dimensionality.md#correlations)                                           |
  | Items of an analysis of their pairs: correlations, factors, omega                                  | 1,000: an analysis adds up to about 150 MB | [dimensionality](docs/psychometrics/dimensionality.md#correlations)                                           |
  | Items of a statistic of every pair: local dependence, Yen's Q3, a Mokken scale analysis            | 1,000: up to about 210 MB                  | [fit](docs/psychometrics/fit.md#local-dependence), [mokken](docs/psychometrics/mokken.md)                     |
  | Item steps of Molenaar-Sijtsma reliability, which holds a matrix of their pairs                    | 1,000: 9 MB                                | [mokken](docs/psychometrics/mokken.md#reliability)                                                            |
  | Cells of the tables of local dependence: each pair of categories of each pair of items             | 2^24: about 1 s, 260 MB                    | [fit](docs/psychometrics/fit.md#local-dependence)                                                             |
  | Highest total score of the items of S-X², whose time grows with the items times its square         | 300: 2 to 3 s                              | [fit](docs/psychometrics/fit.md)                                                                              |
  | Counts of a form assembly: forms × (forms + pool items)                                            | 2^26: 256 MB                               | [cat](docs/psychometrics/cat.md#form-assembly)                                                                |
  | Facets of a G study, with the object                                                               | 6, as GENOVA and urGENOVA                  | [generalizability](docs/psychometrics/generalizability.md)                                                    |
  | Reportable scores of the scale of a predicted theta score                                          | 10,000                                     | [scaling](docs/psychometrics/scaling.md)                                                                      |
  | Degree of a log-linear presmoothing, of each score of a table                                      | 20: a table's fit holds 441 terms per cell | [equating](docs/psychometrics/equating.md)                                                                    |
  | Nodes of a grid of several dimensions, in calibration and EAP                                      | 10⁶                                        | [multidimensional](docs/psychometrics/multidimensional.md#calibration)                                        |
  | Values at the nodes of an EM calibration of several dimensions                                     | 2²⁴: 300 MB, 700 MB with standard errors   | [multidimensional](docs/psychometrics/multidimensional.md#calibration)                                        |
  | Coefficients of a logistic fit without a ridge, or random θ                                        | 5,000: a dense matrix of 200 MB            | [logistic](docs/knowledge-tracing/logistic.md)                                                                |
  | States of a structure built from prerequisites or clauses                                          | `maxStates`, 10,000                        | [kst](docs/knowledge-tracing/kst.md#limits-and-errors)                                                        |
  | Learning paths; cells of the Jacobian of identifiability                                           | `maxPaths`, 10,000; 1e7                    | [kst](docs/knowledge-tracing/kst.md#limits-and-errors)                                                        |
  | Items of IITA, the difference test and the validity of a structure; elements of a surmise relation | 1,000: tables of a cell for each pair      | [kst](docs/knowledge-tracing/kst.md#limits-and-errors)                                                        |
  | Cells of IITA's candidates: a prerequisite each, and 8 for each item                               | 24 million: about 260 MB                   | [kst](docs/knowledge-tracing/kst.md#limits-and-errors)                                                        |
  | Cells of a KST model's masks: a byte for each state and set of skills                              | 2^28: 256 MB                               | [kst](docs/knowledge-tracing/kst.md#limits-and-errors)                                                        |

- **Randomness.** `createRandom` is a seeded generator for reproducible simulation and splits. It is not
  cryptographically secure; do not use it for secrets, tokens or test form assignment that must be unpredictable.
