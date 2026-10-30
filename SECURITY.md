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
- **Resource use.** Time and memory grow with the size of the input. In `@measurekit/psychometrics`, item statistics take
  time proportional to persons × items. In `@measurekit/knowledge-tracing`, a log is checked and replayed in time
  proportional to its events, and cross-validation fits the model once per fold. Limit the size of data accepted from
  untrusted sources before analyzing it on a server.
- **Randomness.** `createRandom` is a seeded generator for reproducible simulation and splits. It is not
  cryptographically secure; do not use it for secrets, tokens or test form assignment that must be unpredictable.
