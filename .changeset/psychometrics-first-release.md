---
"@measurekit/psychometrics": major
---

Psychometrics for JavaScript and TypeScript, validated against R: classical test theory with the value of subscores,
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

`contrastingGroupsCutScores` sets a cut score at each boundary between adjacent categories by a logistic regression on
the score, which gives the contrasting-groups and Body of Work methods, and `beukCutScore` the Beuk compromise, which
crosses the observed failure rate as `hofsteeCutScore` does.

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
each occasion and of the change between consecutive ones, as `mirt` with constrained items does.

Adaptive tests keep to a blueprint. An item is in several content areas (`contentAreas`), `contentBounds` holds a minimum
and a maximum number of items in each area for any selection rule, `maximumPriorityIndex` orders the items by how near their
areas are to their bounds, and `contentMinimumsMet` keeps a stop on precision from coming before the minimums. A test ends
when its selection rule scores every eligible item `-Infinity`.

Fixed forms are assembled in parallel: `assembleForms` takes several forms from one pool in rotation, with enemy sets,
testlets taken whole, items in several content areas and a most of `overlap` items that two forms may share.

The options that restrict the items of an adaptive test, `contentTargets`, `enemies` and `withinTestlet`, are one type, `TestControlOptions`, which the options of the adaptive, diagnostic and multidimensional tests extend. Every function documents each parameter and its result.

`generalizedGradedUnfolding` is the generalized graded unfolding model for agree-disagree items, whose persons agree most
near an item's location. `calibrateUnfolding` calibrates it as the GGUM package does: from Roberts and Laughlin's starts,
through the graded unfolding model and the model with a = 1, to every parameter, leaving out the persons who disagree
with every item and reporting how many it kept.

The guide of each module lists under Not included the methods it leaves out, one bullet each with its reason, and
`check:standards` fails a module guide without the section. The guides show as recipes what a few lines over the functions
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
to the dimensions. A Gauss-Hermite rule holds for any number of points, and `fromMirtMultidimensional` takes the
coefficients first, as `fromMirtIrt` does.

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
Scoring, `multidimensionalInformation` and multidimensional adaptive tests take at most 100 dimensions.

The many-facet Rasch model and the Rasch comparison of an LLTM take at most 1,000 parameters, a rating scale of 255
categories and 2^24 expected counts. The Mantel test and ordinal logistic regression of DIF take items of at most 255
categories, and the Mantel test's counts grow with the persons alone. `calibrateResponseTimes` gives no standard errors
above 2,000 items, and `calibrateSpeedAccuracy` refuses `standardErrors: "hessian"` above 2,000 estimated parameters.
`validateQMatrix` takes time that grows as 3^K for K attributes, and a profile of proportion 0 leaves the PVAF defined.
`linearChange` takes at most 1,000 effects, and finds an effect without a unique estimate in the cube of the effects.
