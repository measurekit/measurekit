---
"@measurekit/psychometrics": major
---

Psychometrics for JavaScript and TypeScript, validated against R: classical test theory with the value of subscores,
item response theory, including multidimensional models with exploratory and confirmatory calibration by EM and MH-RM,
rotation and scoring, cognitive diagnosis (G-DINA, DINA, DINO, A-CDM, LLM and R-RUM), adaptive, diagnostic adaptive, multidimensional
adaptive and multistage testing with testlets and the derivation of routing cuts, item and person fit, differential item functioning for dichotomous and
polytomous items, linking and equating, including equating through common items, IRT observed-score equating and
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

Lord's chi-square can be purified: with `purify: true` the items it flags leave the anchors of the linking, which is
repeated on the two calibrations until two rounds flag the same items.

Generalizability studies accept designs with empty cells and unequal numbers of nested elements with
`unbalanced: true`, estimating the variance components by Henderson's Method 1, as urGENOVA does. A study reports whether
it is `balanced`, and a decision study of one that is not takes its sizes from the caller.

Links compose into chains: `chainTransformations` moves the scale of the first form of a chain to that of the last, as
`equateIRT::chainec` does. `scaleGrowth` reports the mean change, the ratio of standard deviations and the effect size
between the neighboring groups of a vertical scale, whether they come from a calibration with `groups` or from chained
links.

`calibrate` regresses theta on persons' covariates with `covariates`: θ ~ N(xᵀβ, σ²), the coefficients estimated with the
items. With items that estimate their slopes, σ is 1 and the coefficients are in units of the residual SD; with `fixed`
items, which set the scale, σ is estimated. The result has a `regression` with the coefficients, the residual SD and, with
standard errors from the Hessian, the standard errors of the coefficients.

Augmented subscores borrow strength from the other sections of a test, with the reliability of each, and Mokken scale
analysis reports the reliability of the total score by the Molenaar-Sijtsma method.

The many-facet Rasch model estimates the bias of each pair of elements of two facets, such as a rater on an item, with its
standard error and Wald statistic, from the same marginal likelihood as the measures.
