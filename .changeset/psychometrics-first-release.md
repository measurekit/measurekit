---
"@measurekit/psychometrics": major
---

Psychometrics for JavaScript and TypeScript, validated against R: classical test theory with the value of subscores,
item response theory, including conditional maximum likelihood of the Rasch family and multidimensional models with exploratory and confirmatory calibration by EM and MH-RM,
rotation and scoring, cognitive diagnosis (G-DINA, DINA, DINO, A-CDM, LLM and R-RUM), adaptive, diagnostic adaptive, multidimensional
adaptive and multistage testing with testlets and the derivation of routing cuts, item and person fit, differential item functioning for dichotomous and
polytomous items, with SIBTEST of items and bundles, IRT likelihood-ratio tests, Raju's areas, DFIT, the Liu-Agresti odds ratio and the logistic discriminant function, linking and equating, including equating through common items, IRT observed-score equating and
standard errors by bootstrap and the delta method, score scaling and prediction, dimensionality by polychoric
correlations and parallel analysis, explanatory item response models, including the linear logistic test model of
partial credit and rating scale items and the linear logistic model of change, generalizability theory, Mokken scale analysis,
simulation, standard setting, rater agreement and rater effects, including the many-facet Rasch model, response
times with the joint model of speed and accuracy, and change between two occasions: gains, the reliability of a
difference and reliable change.

The EM calibrations of the multidimensional models, the many-facet Rasch model and the linear logistic test model
accelerate EM with SQUAREM, and the posterior means of a data set tabulate the log probabilities of each item once. At the
default tolerance the estimates stop closer to the maximum than plain EM would, by up to about 1e-3, and the iterations
fall, for the exploratory two-dimensional 2PL from 442 to 100.

`limitedInformationFit` tests a calibration on the margins of its test, by M2, M2* or C2 with RMSEA and its interval,
SRMSR, TLI and CFI, as mirt's `M2`. It reads Δ from the calibration, with one column for a parameter that items share and
those of an estimated normal distribution, and checks the parameters against the calibration's AIC. Where the
Jacobian's rank is below its columns, the statistic uses the whole complement of its span, with as many degrees of
freedom as the moments less its rank.

`kernelEquating` equates by the kernel method in the equivalent-groups, single-group, counterbalanced, chained and
post-stratification designs, as kequate: log-linear presmoothing, Gaussian continuization with bandwidths that minimize
the penalty, the percent relative errors of the moments, and standard errors by the delta method;
`kernelEquatingDifference` gives the standard error of the difference of two equatings, as kequate's `genseed`.

`linkCovariance` gives the standard errors and covariance of a link between separately calibrated forms, or of a chain
of them, from each form's covariance, counting a form between two links once, as `equateIRT::chainec`;
`linkedEquatingErrors` gives the standard errors of IRT true-score or observed-score equating through such a link.

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
identifies the DINA and DINO.

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
each occasion and of the change between consecutive ones, as `mirt` with constrained items does.

Adaptive tests keep to a blueprint. An item is in several content areas (`contentAreas`), `contentBounds` holds a minimum
and a maximum number of items in each area for any selection rule, `maximumPriorityIndex` orders the items by how near their
areas are to their bounds, and `contentMinimumsMet` keeps a stop on precision from coming before the minimums. A test ends
when its selection rule scores every eligible item `-Infinity`.

Fixed forms are assembled in parallel: `assembleForms` takes several forms from one pool in rotation, with enemy sets,
testlets taken whole, items in several content areas and a most of `overlap` items that two forms may share.

The options that restrict the items of an adaptive test, `contentTargets`, `enemies` and `withinTestlet`, are one type, `TestControlOptions`, which the options of the adaptive, diagnostic and multidimensional tests extend. Every function documents each parameter and its result.
