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
- **Resource use.** Time and memory grow with the size of the input. In `@measurekit/psychometrics`, item
  statistics take time proportional to persons × items. In `@measurekit/knowledge-tracing`, a log is checked
  and replayed in time proportional to its events, and cross-validation fits the model once per fold. Limit the size
  of data accepted from untrusted sources before analyzing it on a server.
- **Limits.** Tables are sized by the values that occur in the data, not by the highest value, so that a small input
  with one large response costs no more than another of its size. What would grow faster than the input has a limit,
  and passing it throws `DataError`:

  | Input                                                       | Limit                                      | Guide                                                                         |
  | ----------------------------------------------------------- | ------------------------------------------ | ----------------------------------------------------------------------------- |
  | A response, or the score of a faceted score                 | An integer from 0 to 2³¹ - 1               | [data](docs/psychometrics/data.md)                                            |
  | Attributes of a Q-matrix; of a diagnostic adaptive test     | 20: a fit holds 2^K profiles, about 1 GB   | [cdm](docs/psychometrics/cdm.md#calibration)                                  |
  | Success probabilities with incomplete standard errors       | 3,162 over all items: 80 MB of information | [cdm](docs/psychometrics/cdm.md#calibration)                                  |
  | Categories of a Mokken scale                                | 10                                         | [mokken](docs/psychometrics/mokken.md)                                        |
  | Facets of an unbalanced G study, with the object            | 6                                          | [generalizability](docs/psychometrics/generalizability.md#unbalanced-designs) |
  | Coefficients of a logistic fit without a ridge, or random θ | 5,000: a dense matrix of 200 MB            | [logistic](docs/knowledge-tracing/logistic.md)                                |
  | States of a structure built from prerequisites or clauses   | `maxStates`, 10,000                        | [kst](docs/knowledge-tracing/kst.md#limits-and-errors)                        |
  | Learning paths; cells of the Jacobian of identifiability    | `maxPaths`, 10,000; 1e7                    | [kst](docs/knowledge-tracing/kst.md#limits-and-errors)                        |

- **Randomness.** `createRandom` is a seeded generator for reproducible simulation and splits. It is not
  cryptographically secure; do not use it for secrets, tokens or test form assignment that must be unpredictable.
