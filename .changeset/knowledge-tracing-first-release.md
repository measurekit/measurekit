---
"@measurekit/knowledge-tracing": major
---

Knowledge tracing for JavaScript and TypeScript, validated against pyBKT and R: Bayesian knowledge tracing, logistic
models, Elo, online IRT and knowledge space theory, including structures from alternative prerequisites, closure under union, bases,
atoms, traces, learning paths and neighbourhoods, with serializable learner state, practice recommendations and
evaluation by cross-validation and learning curves.

A function takes what it cannot work without as arguments, `irt(parameters, options)`, `kst(structure, options)` and
`movingAverageMastery(log, decay, threshold)`, and its options hold only what has a default, which a strict reader refuses
to misspell.

Each module is imported from its subpath, such as `@measurekit/knowledge-tracing/bkt`, and each name from one entry. The
package's name gives only the learner model interface, `LearnerModel` and `ModelFit`, and what the modules share and core
owns: `createRandom`, `DataError`, `MeasurekitError` and the types of core that the modules take and return, `Random`,
`QMatrix`, `ThetaDistribution`, `ThresholdItem` and `NominalParameters`, so a user never imports `@measurekit/core`.

An event may carry no outcome: `outcome: null` is a learning opportunity, such as a lesson or a video, which every model
accepts. BKT applies the learn and forget rates of the event's `resource` and estimates a lesson's rates from the
outcomes that follow it, and logistic models count such events as exposures with the features `exposures`,
`decayedExposures` and `recencyExposure`. Fits, cross-validation, splits, mastery criteria and simulation leave these
events out of what they predict and count: a fit gives them a `null` prediction, and `alignPredictions` aligns the
others for the metrics. Events also carry `hints`, `duration` and `attemptNumber`, and `selectFirstAttempts` keeps each
learner's first try at an item.

Online IRT follows a learner who changes with `drift: { variance, per }`: θ is a random walk between events, carried
across each gap by the grid's chain, computed by uniformization with a lower bound for the points that underflow.
`predict` answers at the attempt's time, `ability` and `posterior` at any time with `at`, and the fit's log-likelihood
is the forward likelihood, by which a host chooses the variance. `points` is required with `drift`, and a model with
drift reads the states of a model without it. Online IRT takes nominal response items, `{ model: "nominal", a, c }`, as
`nominalResponse` of `@measurekit/psychometrics` calibrates them.

BKT holds rates at given values with `fixed`, keeps the rates it estimates within `bounds`, and takes the learn and forget
rates by the pair of the resource before an attempt and its own with `pairLearning`, as pyBKT's `fixed` and `multipair`.
A BKT fit reports each skill's own fit in `skills`: whether its EM converged, its iterations, the events with an outcome and
its log-likelihood.

Logistic models take the features of LKT's other names (`proportion`, `netSuccesses`, `powerOpportunities`, `logit`,
`decayedProportionFailed`, `recencySuccess`, `recencyFailure`, `base`, `basePractice`, `dash`, `ppe` and their success
and failure forms), and the time-window counts of DAS3H (`windowOpportunities`, `windowSuccesses`, `windowFailures`). A
skill's history gains a field only when a feature reads it, and `basePractice` reads the `duration` of every event.

Logistic models take intercepts for items beside those of skills (`intercepts`), terms over the skills, the item or the
learner with a coefficient for each skill or item or one for all (`{ feature, over, coefficient }`), covariates, and
standard errors (`standardErrors`), which expresses Best-LR of Gervet et al. and LKT's components and `numer`. Fits with
a ridge solve by Newton-CG on Hessian-vector products, so that 50,000 items are no obstacle; fits without one, and with a
random intercept, solve with dense matrices up to 5,000 coefficients. `searchRates` finds the decays and exponents of the
features by maximum likelihood, as LKT's `seedpars` does, and fits the central differences of its gradient only at the
lengths its line search accepts.

Mastery criteria over a log are the probability of mastery (`thresholdMastery`), a streak (`streakMastery`), a moving
average (`movingAverageMastery`), a window of outcomes (`windowMastery`) and a predicted success (`predictedMastery`),
which serves logistic models and Elo.

Evaluation reports R² and Cohen's kappa beside AUC, RMSE, log-loss and accuracy. `crossValidate` and `compareModels`
take `unseenSkills: "skip"`, which leaves out the test events of skills that the training log has no outcome of, as a
split by time of a curriculum gives, so that BKT and logistic models can be evaluated on it. `compareByLearner` tests
whether two models predict differently well, with a paired t test of their losses averaged per learner.
`metricsByOpportunity` gives the metrics by opportunity, `learningCurve` takes `predictions` and `minOpportunities` in
an options object, and `fitPowerLaw` and `categorizeCurve` fit a curve's power law and give DataShop's category of it.
Log-loss, in `logLoss`, `metrics` and `compareByLearner`, clips each prediction to [ε, 1 - ε] at the machine epsilon, as
scikit-learn's `log_loss` does, so a certain prediction that is wrong, such as a logistic model's past a linear predictor
of about 36.7, costs about 36.04 instead of making the loss infinite.

The basic local independence model of `kst` holds rates at given values with `fixed`, gives items one rate with `equal`, and
is fitted by maximum likelihood, minimum discrepancy or both together (`method: "mdml"`). A fit reports G² against the
saturated model with pks's degrees of freedom and its mean discrepancy, `compareFits` tests a restricted fit against a
general one, and `model.identifiability()` gives the rank of the Jacobian of the pattern probabilities. `structureValidity`
measures a structure against responses (gamma, the violational coefficient, the discrepancy index and the distance
agreement), `forwardGraded` and `backwardGraded` find where a structure is graded, and `diffTest` tests the fit of one
quasi order, or the difference between two, with a delta-method variance.

Ids may hold any character: the keys that join several, such as a learner and a skill or a pair of resources of BKT,
keep them apart. `iita`, `diffTest` and `structureValidity` refuse an incomplete table of responses at the cost of its
events, `surmiseStructure` closes prerequisites in time proportional to the elements times their prerequisites, and a
logistic feature whose value overflows throws `DataError`.
These analyses and `surmiseStructure` take at most 1,000 elements, IITA's candidates hold at most 24 million cells, and
the masks of a KST model at most 2^28, each refused with `DataError` past its limit. A KST fit gives each learner's most
likely state in `mostLikelyStates`, and their posteriors in `states` only with `posteriors: true`, as they take learners
× states × 8 bytes; `ModelFit.states` is optional for that, and every other model gives it.

Every array that a function returns, such as the folds of `splitByTime` and the evaluations of `compareModels`, is
`readonly`, as those inside its results are: a caller who sorts one in place copies it first.

`buildEventLog` builds a log from events, `selectFirstAttempts` keeps each learner's first attempts, `alignPredictions`
pairs the predictions with the outcomes of a log, `calibrationCurve` tabulates calibration in bins and `fitPowerLaw`
fits the power law of practice.

Every function documents each parameter and its result.

The main function of every module has an example in its documentation, which a test runs against the values its
comments state.

Every model and function refuses an option it does not take, naming it, and so do the records of settings inside the
options: BKT's `bounds` and `fixed` rates, KST's `fixed` and `equal`, IRT's `prior` and `drift`, Elo's `uncertainty` and
a logistic term.

`diffTest` takes incomplete response patterns, as an adaptive test, a placement or a booklet design leaves them: each
pair of items counts the learners who answered both, each item's solution rate those who answered it, a pair that no
learner answered both of leaves the fit measure, and the standard error is the delta method of the same estimate over
each learner's answers. Complete patterns give the estimate and variance as before. One edge of a quasi order is tested
by the order without it as the alternative, with `tail: "greater"`.
