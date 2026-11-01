---
"@measurekit/knowledge-tracing": major
---

Knowledge tracing for JavaScript and TypeScript, validated against pyBKT and R: Bayesian knowledge tracing, logistic
models, Elo, online IRT and knowledge space theory, with serializable learner state, practice recommendations and
evaluation by cross-validation and learning curves.

An event may carry no outcome: `outcome: null` is a learning opportunity, such as a lesson or a video, which every model
accepts. BKT applies the learn and forget rates of the event's `resource` and estimates a lesson's rates from the
outcomes that follow it, and logistic models count such events as exposures with the features `exposures`,
`decayedExposures` and `recencyExposure`. Fits, cross-validation, splits, mastery criteria and simulation leave these
events out of what they predict and count: a fit gives them a `null` prediction, and `observedPairs` aligns the others for
the metrics. Events also carry `hints`, `duration` and `attemptNumber`, and `firstAttempts` keeps each learner's first
try at an item.

Online IRT follows a learner who changes with `drift: { variance, per }`: θ is a random walk between events, carried
across each gap by the grid's chain, computed by uniformization with a lower bound for the points that underflow.
`predict` answers at the attempt's time, `ability` and `posterior` at any time with `at`, and the fit's log-likelihood
is the forward likelihood, by which a host chooses the variance. `points` is required with `drift`, and a model with
drift reads the states of a model without it.

BKT holds rates at given values with `fixed`, keeps the rates it estimates within `bounds`, and takes the learn and forget
rates by the pair of the resource before an attempt and its own with `pairLearning`, as pyBKT's `fixed` and `multipair`.
