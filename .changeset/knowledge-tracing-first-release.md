---
"@measurekit/knowledge-tracing": major
---

Knowledge tracing for JavaScript and TypeScript, validated against pyBKT and R: Bayesian knowledge tracing, logistic
models, Elo, online IRT and knowledge space theory, including structures from alternative prerequisites, closure under union, bases,
atoms, traces, learning paths and neighbourhoods, with serializable learner state, practice recommendations and
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

Logistic models take the features of LKT's other names (`proportion`, `netSuccesses`, `powerOpportunities`, `logit`,
`decayedProportionFailed`, `recencySuccess`, `recencyFailure`, `base`, `basePractice`, `dash`, `ppe` and their success
and failure forms), and the time-window counts of DAS3H (`windowOpportunities`, `windowSuccesses`, `windowFailures`). A
skill's history gains a field only when a feature reads it, and `basePractice` reads the `duration` of every event.

Mastery criteria over a log are the probability of mastery (`thresholdMastery`), a streak (`streakMastery`), a moving
average (`movingAverageMastery`), a window of outcomes (`windowMastery`) and a predicted success (`predictedMastery`),
which serves logistic models and Elo.

Evaluation reports R² and Cohen's kappa beside AUC, RMSE, log-loss and accuracy. `crossValidate` and `compareModels`
take `unseenSkills: "skip"`, which leaves out the test events of skills that the training log has no outcome of, as a
split by time of a curriculum gives, so that BKT and logistic models can be evaluated on it. `compareByLearner` tests
whether two models predict differently well, with a paired t test of their losses averaged per learner.
`metricsByOpportunity` gives the metrics by opportunity, `learningCurve` takes `predictions` and `minOpportunities` in
an options object, and `powerLawFit` and `categorizeCurve` fit a curve's power law and give DataShop's category of it.

The basic local independence model of `kst` holds rates at given values with `fixed`, gives items one rate with `equal`, and
is fitted by maximum likelihood, minimum discrepancy or both together (`method: "mdml"`). A fit reports G² against the
saturated model with pks's degrees of freedom and its mean discrepancy, `compareFits` tests a restricted fit against a
general one, and `model.identifiability()` gives the rank of the Jacobian of the pattern probabilities. `structureValidity`
measures a structure against responses (gamma, the violational coefficient, the discrepancy index and the distance
agreement), `forwardGraded` and `backwardGraded` find where a structure is graded, and `diffTest` tests the fit of one
quasi order, or the difference between two, with a delta-method variance.
