---
"@measurekit/psychometrics": major
---

Psychometrics for JavaScript and TypeScript, validated against R and RSMTool: classical test theory with the value of subscores,
item response theory, including conditional maximum likelihood of the Rasch family and multidimensional models with exploratory and confirmatory calibration by EM and MH-RM,
rotation and scoring, cognitive diagnosis (G-DINA, DINA, DINO, A-CDM, LLM and R-RUM), adaptive, diagnostic adaptive, multidimensional
adaptive and multistage testing with testlets, classification by a confidence interval or the sequential probability ratio test and the derivation of routing cuts, item and person fit, with Snijders' lz* and the normed G, U3 and Ht of persons, the RMSD of items, Chen and Thissen's local-dependence chi-square, the Andersen, Martin-Löf and Wald tests of the Rasch family and kernel-smoothed item curves, differential item functioning for dichotomous and
polytomous items, with SIBTEST of items and bundles, IRT likelihood-ratio tests, Raju's areas, DFIT, the Liu-Agresti odds ratio and the logistic discriminant function, linking and equating, including equating through common items, IRT observed-score equating and
standard errors by bootstrap and the delta method, score scaling and prediction with norms and their standard errors, dimensionality by polychoric
correlations, parallel analysis, exploratory factors by minres or maximum likelihood with their fit, and omega hierarchical, explanatory item response models, including the linear logistic test model of
partial credit and rating scale items and the linear logistic model of change, generalizability theory, Mokken scale analysis with two-way imputation, Guttman errors and conditional association,
simulation, standard setting, rater agreement and rater effects, including the many-facet Rasch model and the evaluation of automated scores, with the standardized mean difference of a subgroup on the population's standard deviations, response
times with the joint model of speed and accuracy, and change between two occasions: gains, the reliability of a
difference and reliable change.

A function takes what it needs as arguments, `calibrate(data, model, options)`, `createAdaptiveTest(pool, stopping, options)` and
`assembleForms(pool, length, options)`, and its options hold only what has a default, which a strict reader refuses to
misspell.

Each module is imported from its subpath, such as `@measurekit/psychometrics/irt`, and each name from one entry. The
package's name gives only what every module uses and core owns: `createRandom`, `Random`, `DataError` and
`MeasurekitError`.

The EM calibrations of the multidimensional models, the many-facet Rasch model and the linear logistic test model
accelerate EM with SQUAREM, and the posterior means of a data set tabulate the log probabilities of each item once. At the
default tolerance the estimates stop closer to the maximum than plain EM would, by up to about 1e-3, and the iterations
fall, for the exploratory two-dimensional 2PL from 442 to 100.

`limitedInformationFit` tests a calibration on the margins of its test, by M2, M2* or C2 with RMSEA and its interval,
SRMSR, TLI and CFI, as mirt's `M2`. It reads Δ from the calibration, with one column for a parameter that items share and
those of an estimated normal distribution, and the estimated parameters from the `estimated` of each item. Where the
Jacobian's rank is below its columns, the statistic uses the whole complement of its span, with as many degrees of
freedom as the moments less its rank.

`kernelEquating` equates by the kernel method in the equivalent-groups, single-group, counterbalanced, chained and
post-stratification designs, as kequate: log-linear presmoothing, Gaussian continuization with bandwidths that minimize
the penalty, the percent relative errors of the moments, and standard errors by the delta method;
`kernelEquatingDifference` gives the standard error of the difference of two equatings, as kequate's `genseed`.

`linkCovariance` gives the standard errors and covariance of a link between separately calibrated forms, or of a chain
of them, from each form's covariance, counting a form between two links once, as `equateIRT::chainec`;
`linkedEquatingErrors` gives the standard errors of IRT true-score or observed-score equating through such a link.

`contrastingGroupsCutScores` sets a cut score at each boundary between adjacent categories by a logistic regression on
the score, which gives the contrasting-groups and Body of Work methods, and `beukCutScore` the Beuk compromise, which
crosses the observed failure rate as `hofsteeCutScore` does.

`borderlineRegressionCutScores` sets the cut of each station of a performance test, such as an OSCE, by regressing its
checklist marks on its global ratings and reading the line at the borderline grade, with the cut's standard error, R²
and the inter-grade discrimination; the test's cut is the sum of the stations' and its standard error counts the
covariances of their cuts. `borderlineGroupCutScores` takes the mean, or the median, of the marks of the persons at one
or several borderline grades.

Adaptive tests select by catR's other rules: `closestDifficulty` (bOpt), `expectedInformation` (MEI),
`likelihoodWeightedInformation` and `posteriorWeightedInformation` (MLWI, MPWI), `posteriorKullbackLeibler` (KLP),
`discriminationIndex` and `posteriorDiscriminationIndex` (GDI, GDIP), `expectedPosteriorVariance` (MEPV), and the
progressive and proportional methods of exposure control, and `startingItems` administers fixed items before any rule.

`livingstonLewisClassification` gives Livingston and Lewis's accuracy and consistency of classifications by cuts on a
total score, from a four-parameter beta distribution of true scores and a binomial error at the effective test length,
as `betafunctions::LL.CA` and `LL.CA.MC`.

Lord's chi-square can be purified: with `purify: true` the items it flags leave the anchors of the linking, which is
repeated on the two calibrations until two rounds flag the same items.

Generalizability studies accept designs with empty cells and unequal numbers of nested elements with
`unbalanced: true`, estimating the variance components by Henderson's Method 1, as urGENOVA does. A study reports whether
it is `balanced`, and a decision study of one that is not takes its sizes from the caller.

Links compose into chains: `chainTransformations` moves the scale of the first form of a chain to that of the last, as
`equateIRT::chainec` does. `scaleGrowth` reports the mean change, the ratio of standard deviations and the effect size
between the neighboring groups of a vertical scale, whether they come from a calibration with `groups` or from chained
links.

`drawJointThetas` draws thetas jointly from separate posteriors, one for each person or section, coupled by a Gaussian copula
with a correlation matrix you give: every posterior keeps its distribution, a correlation of 1 gives one quantile in all of
them, and 0 gives independent draws. It forecasts a composite of several scores, or bounds one when the correlation is unknown.

`calibrate` holds single parameters of estimated items at given values with `fixedParameters`, such as the lower
asymptote of a 3PL item at 1/4 while its slope and difficulty are estimated, as mirt's `est = FALSE`: a held parameter
has no standard error, prior or place in the covariance, and the AIC and BIC do not count it. `calibrateMultidimensional`
holds them the same way, by EM and MH-RM, so an M3PL item that two dimensions share keeps its lower asymptote at 1/4. A
held threshold of a graded item moves the starting values of the others to fit it (`ordered` of `ModelParameter`). Each
calibrated item records the `estimated` names, and `limitedInformationFit` and `summedScoreFit` read them, so that Δ and
the degrees of freedom count only the estimated parameters, as mirt's.

Lord's chi-square, Raju's area and the IRT likelihood-ratio test take `fixedParameters` too, held in both groups, as
difR's `c` holds the lower asymptotes of the 3PL in `difLord` and `difRaju`, and mirt's `est = FALSE` in a multiple-group
model: a held parameter is not compared or counted in the degrees of freedom. Lord's and Raju's hold the asymptotes
alone, as the link moves a and b, and Raju's area takes a 3PL or 4PL item whose asymptotes are held, whose areas are
d - c times the 2PL's.

`calibrate` takes `personWeights`, each person's survey weight, rescaled to sum to the persons with a response, as mirt's
`survey.weights` asks: every sum over persons counts the person times the weight, the regression of theta on covariates
included, so that the estimates are the population's and the standard errors keep the sample's size. Without the
option, every result is the same to the last digit and as fast. Lord's chi-square, Raju's area and the IRT
likelihood-ratio test take no weights.

The testlet response model of `calibrate` works with `groups`: groups of persons, each with its own distribution of theta
and the scale set by a standard normal group or fixed items, share one SD for each testlet, so that one concurrent
calibration covers passages given to several grades or forms.

`calibrate` regresses theta on persons' covariates with `covariates`: θ ~ N(xᵀβ, σ²), the coefficients estimated with the
items. With items that estimate their slopes, σ is 1 and the coefficients are in units of the residual SD; with `fixed`
items, which set the scale, σ is estimated. The result has a `regression` with the coefficients, the residual SD and, with
standard errors from the Hessian, the standard errors of the coefficients.

Augmented subscores borrow strength from the other sections of a test, with the reliability of each, and Mokken scale
analysis reports the reliability of the total score by the Molenaar-Sijtsma method.

The many-facet Rasch model estimates the bias of each pair of elements of two facets, such as a rater on an item, with its
standard error and Wald statistic, from the same marginal likelihood as the measures.

Cognitive diagnosis compares each item's G-DINA estimates with the DINA, DINO, A-CDM, LLM and R-RUM by Wald tests, and
selects a model for each item; the calibration gives the covariance of the success probabilities for them.
`cdmModelFit` and `cdmItemFit` give M2 with RMSEA and SRMSR, and each item's z statistics, as GDINA's `modelfit` and
`itemfit`, from exact sums over the profiles; `qMatrixIdentifiability` checks the conditions under which a Q-matrix
identifies the DINA and DINO. A design holds at most 15 attributes, 2^22 cells of items by profiles and 2^18 latent
groups over all items, which keeps a fit within about 300 MB; so does the pool of a diagnostic adaptive test.

`calibrateMultidimensional` calibrates several groups of persons on one scale, with the option `groups` and, for anchor
items, `fixed`, up to three dimensions by EM. Items have the same parameters in every group, and the result reports each
group's means and covariance in `groups`, as `mirt::multipleGroup` does.

Bifactor and two-tier models are calibrated by `calibrateBifactor`, which integrates the specific dimensions one group of
items at a time, as mirt's `bfactor` does, and runs 8 to 17 times as fast on 2,000 persons.

`exploratoryFactorAnalysis` fits several factors to the items by the minimum residual method, from Pearson or polychoric
correlations, with varimax or oblimin rotation, as `psych::fa(fm = "minres")` does. `minimumAveragePartial` is Velicer's
test for the number of factors, and `smoothCorrelations` makes a correlation matrix that is not positive definite so, as
`psych::cor.smooth` does.

`calibrateLongitudinal` calibrates the same persons answering the same items on two or three occasions. The trait of each
occasion is a dimension of a normal distribution whose means, standard deviations and correlations are estimated, with
the first occasion N(0, 1); items have one set of parameters across the occasions, and the result reports the moments of
each occasion and of the change between consecutive ones, as `mirt` with constrained items does. Its standard errors are
named as its estimates are: each item's by parameter, each occasion's mean and standard deviation, and the correlations by rows.

Adaptive tests keep to a blueprint. An item is in several content areas (`contentAreas`), `contentBounds` holds a minimum
and a maximum number of items in each area for any selection rule, `maximumPriorityIndex` orders the items by how near their
areas are to their bounds, and `contentMinimumsMet` keeps a stop on precision from coming before the minimums. A test ends
when its selection rule scores every eligible item `-Infinity`.

Fixed forms are assembled in parallel: `assembleForms` takes several forms from one pool in rotation, with enemy sets,
testlets taken whole, items in several content areas and a most of `overlap` items that two forms may share.

The options that restrict the items of an adaptive test, `contentTargets`, `enemies` and `withinTestlet`, are one type, `TestControlOptions`, which the options of the adaptive, diagnostic and multidimensional tests extend. Every function documents each parameter and its result.

`simulateAdaptiveTest`, `simulateDiagnosticTest` and `simulateMultidimensionalAdaptiveTest` take `testRandom`, the
generator of the test's random choices at each step of each person, so that a simulation gives the items of a host that
seeds each call of `start` and `answer`; without it, the generator of the responses makes them.

`simulateResponses` and `simulateMultidimensionalResponses` name the columns of their matrix by the items' ids when the
items carry them, as a test's or a calibration's do, and `item1` and on when they do not.

`generalizedGradedUnfolding` is the generalized graded unfolding model for agree-disagree items, whose persons agree most
near an item's location. `calibrateUnfolding` calibrates it as the GGUM package does: from Roberts and Laughlin's starts,
through the graded unfolding model and the model with a = 1, to every parameter, leaving out the persons who disagree
with every item and reporting how many it kept.

The guide of each module opens with a quick start and ends with its errors, the methods it leaves out, one row each with
its reason, and its differences from the references; `check:standards` fails a module guide without these sections. The guides show as recipes what a few lines over the functions
compute, each with a test against its reference: plausible values and Rubin's rules, responses missing not at random,
restricted exposure, the nested-model chi-square, bootstrap standard errors and the intervals of alpha and omega.

What would cost more than its input has a limit: a G study has at most 6 facets with the object, as GENOVA and urGENOVA;
a polychoric correlation takes items of at most 8 observed categories, as psych; a log-linear presmoothing has a degree
of at most 20; a predicted theta score reports at most 10,000 scores; and contrasting groups need persons in every
category up to the highest. The errors of links and of IRT equating take one Newton step from each link's minimum and
each score's theta for a moved parameter, and a scale score of a decimal step is the double nearest the decimal.

An EM calibration of several dimensions keeps at most 2^24 values of its items at the nodes of its grids, about 300 MB,
and a bifactor model's Louis standard errors count the tables of pairs of items toward it. The posterior mode of more
than five dimensions starts from a scan along the axes, so that it and MH-RM's log-likelihood take time in proportion
to the dimensions. A Gauss-Hermite rule holds for any number of points, and `readMirtMultidimensional` takes the
coefficients first, as `readMirtIrt` does.

Every eigendecomposition, of the analyses of correlations, the starts of exploratory and unfolding calibrations, the
smoothing of correlations, pseudo-inverses and rotations, is by Householder's reduction and the implicit QL algorithm,
which agree with base R's `eigen` within 1e-12 and take about 3 s at 1,000 items.

The analyses of the pairs of items, the correlations, factors, parallel analysis, the minimum average partial test and
the omegas, take at most 1,000 items, as each holds a matrix of every pair; so do local dependence, Yen's Q3 and every
Mokken scale analysis but the imputation and the person fit indices, and Molenaar-Sijtsma reliability takes at most
1,000 item steps. The tables of local dependence hold at most 2^24 cells, a cell for each pair of categories of each pair
of items, C2 takes at most the 1,830 moments of 60 dichotomous items, S-X² items of a highest total score of at most
300, `multistageCuts` a range at most 1,000 wide, and form assembly 2^26 counts of forms × (forms + pool items).

Multidimensional scoring has maximum likelihood and Warm's weighted likelihood estimates,
`multidimensionalMaximumLikelihoodEstimate` and `multidimensionalWeightedLikelihoodEstimate`, within a `range` in every
dimension, and an item's MDISC and MDIFF, `multidimensionalDiscrimination` and `multidimensionalDifficulty`. A
calibration that estimates the correlations of its dimensions, of a group, of all persons or of a two-tier model's
primary dimensions, needs more persons than those dimensions. An exploratory start correlates the items of sparse data
over each pair's common respondents. The posterior mean keeps the 8 grids asked for last, within 64 MiB of nodes.
Calibration, scoring, `multidimensionalInformation` and multidimensional adaptive tests take at most 100
dimensions, one limit, so that every calibration can be scored.

The many-facet Rasch model and the Rasch comparison of an LLTM take at most 1,000 parameters, a rating scale of 255
categories and 2^24 expected counts. The Mantel test and ordinal logistic regression of DIF take items of at most 255
categories, and the Mantel test's counts grow with the persons alone. `calibrateResponseTimes` gives no standard errors
above 2,000 items, and `calibrateSpeedAccuracy` refuses `standardErrors: "hessian"` above 2,000 estimated parameters.
`validateQMatrix` takes time that grows as 3^K for K attributes, and a profile of proportion 0 leaves the PVAF defined.
`linearChange` takes at most 1,000 effects, and finds an effect without a unique estimate in the cube of the effects.

The readers of data refuse a hole among the rows, ids, records, times or option labels, and check that a first row or
column is an array before its length sizes anything. The `range` of theta of the estimators and of conversion tables is at
most 10,000 wide. Groups of calibration are checked in time that grows with the persons, and a
group named as a property of every object, such as `constructor`, takes no distribution it was not given. A facet named so needs its size in a decision study and is averaged over in an expected rating as any other, and an item's standard errors, priors and held parameters refuse the three reserved keys, which are parameter names of none. The residual
gain refuses pre scores that are all equal however their mean rounds, the normalized gain of a group comes from the sums
over its persons, and the reliable change index stays finite where the squares of the standard errors would underflow.
A testlet that no person answered is refused, as the SD of its effect would be 0 / 0, and an unfolding calibration takes
at most 500 items, as its starts decompose the correlations of every pair.
A quadrature rule has at most 1,000 points, and the rules kept for reuse are the 8 asked for last.
Standard errors take at most 2,000 estimated parameters, in calibrations of one or more dimensions and of response times
alike, as their information holds every pair of them.
Conditional maximum likelihood checks the categories that occur before an item's highest response sizes its parameters.
Conditional maximum likelihood, and the Rasch tests that run it, take at most 2^29 products of the functions with two
items left out in a Newton step, which grow with the patterns of answered items times about the fourth power of their
items.
`abilityEstimates`, `multidimensionalEstimates`, `manyFacetEstimates` and `effortModeratedEstimates` take their
estimator as the option `estimator`, `simulateThetas` its distribution as `distribution`, `maximumDeterminant` its prior
as `prior`, `classifyDifficulty` and `classifyDiscrimination` their thresholds as `thresholds`,
`standardErrorOfMeasurement` its reliability as `reliability`, and `expectedRating` its elements as `levels`.
Every array and record that a function returns is `readonly`, as those inside its results are: a caller who sorts one in
place copies it first.

`buildResponseMatrix`, `buildResponseMatrixFromColumns` and `buildResponseRecords` build the data of a test, and
`createAdaptiveTest`, `createDiagnosticAdaptiveTest`, `createMultidimensionalAdaptiveTest` and `createMultistageTest`
create its adaptive tests. `readMirtIrt`, `readMirtSlopeIntercept` and `readMirtMultidimensional` read the coefficients
of mirt into items, and `writeMirtIrt`, `writeMirtSlopeIntercept` and `writeMirtMultidimensional` write them.

`attributeProfiles` returns each person's posterior over the profiles only with `posterior: true`, as it holds 2^K
numbers for each person; by default it returns their mastery, MAP and EAP profiles.

The main function of every module has an example in its documentation, which a test runs against the values its
comments state.

Every function refuses an option it does not take, naming it, and so do the records of settings inside the options and in
their place (priors, quadrature, thresholds, populations, bandwidths, content constraints, information targets, modules,
designs, scales, Hofstee bounds and Beuk judgments): a misspelled name left its default in place unseen, as a
calibration asked to hold `c` at 0.25 under a misspelled option estimated it at 0.68. `scaleConversion` takes the
`lowest`, `highest` and `step` of a scale, not the scale, and marginal reliability refuses `testlets`.

`calibrate` takes `fixedCovariance`, the covariance of the fixed items' estimates from the calibration that made them,
and carries their sampling error into the standard errors of the items and distributions it estimates: V + J Σ Jᵀ, the
pseudo maximum likelihood variance of restricted recalibration, validated against base R and by the coverage of 95%
intervals in simulations. The errors of a covariance that the IRT equating errors and `fixedCovariance` refuse name "the
matrix of" it.

`percentileRanks` and `normStatistics` take `personWeights`, a survey weight for each score of a norm sample: the
percentile rank is the weighted share below plus half the weighted share at the score, as Hmisc's `wtd.Ecdf` gives it,
and the norms' standard errors are the delta method over the weights, the survey package's linearization of `svymean`
and `svyvar` without its factor n / (n - 1), which reduces to `check.norms`' multinomial delta method for equal weights.

The evaluation of automated scores against human scores is a set of functions of the rater module, where the guide had
a recipe. `automatedScoreAgreement` gives RSMTool's machine-human and human-human statistics, their degradation, the
disattenuated correlation and each subgroup's statistics with its SMD on the population's standard deviations, after
rounding and trimming the machine scores to a `scale`. `validateAutomatedScores` judges them against
`ACCEPTANCE_THRESHOLDS`, Williamson, Xi and Breyer's, or a program's own, and lists each rule broken. `automatedScoreDrift`
gives the agreement by window of time, `routingCurve` the coverage and agreement at each confidence of the scorer,
`scoreMapping` and `mappedScore` a line to the human mean and SD or an isotonic regression from machine to human scores,
and `replayShift` the shift of responses scored again from their locked scores in SDs, with its standard error and whether
it stays within `maxShiftSd`.

`summedScoreTable`, `conversionTable` and `conditionalScaleError` take `testlets`, `{ items, sd }` as a calibration
gives them: the recursion of Lord and Wingersky runs over each testlet's items at θ + σ z for each node of the rule and
integrates the testlet's effect out (Cai's algorithm 2.0), so a form with a passage has its table and the conditional
errors of its scale scores under the testlet model. Validated against base R's enumeration of every response pattern and
mirt's `fscores(method = "EAPsum_2.0")`, within 1e-15. `"true-score"` tables refuse testlets.

`yenQ3CriticalValue` returns the critical value of Yen's Q3 by the parametric bootstrap of Christensen, Makransky and
Horton: replications drawn from a calibration's items with the data's missing responses, each calibrated with the
items' models and the same options, give their largest pair excess, whose quantile is the critical excess; the observed
pairs above it are flagged. A draw whose estimated item lacks a category is drawn again up to `maxRedraws`, and the pairs of an item
whose slope is above `slopeLimit`, near-deterministic, are left out and the item listed. The replications equal those
of `simulateThetas`, `simulateResponses`, `calibrate`, `abilityEstimates` and `yenQ3` composed on the same seed, and the
largest excess agrees with mirt's bootstrap in mean and SD within Monte Carlo error.

The module `similarity` tests whether two persons gave the same responses more often than two persons who answer alone,
as evidence of answer copying and collusion: `omegaIndex` (Wollack's ω), `generalizedBinomialTest` (van der Linden and
Sotaridona), `m4Index` (Maynes' M4) and `kIndex` (Holland's K with Sotaridona and Meijer's K1, K2, S1 and S2), of chosen
options or of scores, validated against CopyDetect 1.4. `selectPairs` forms the pairs of the persons who share a value
of a context, such as a room, a school, a network or a start minute, each pair once, leaving out a value shared by more
than `maxGroup` persons (200) and refusing more than `maxPairs` pairs (1,000,000).
