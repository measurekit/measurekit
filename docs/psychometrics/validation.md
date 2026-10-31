# Validation

Every statistic of `@measurekit/psychometrics` is compared with an R reference; [simulation](#simulation) is checked
by its properties instead. R scripts in [`validation/`](../../validation/README.md) load public datasets or
simulate data with a fixed seed, run the reference, and write the inputs and expected results as JSON fixtures
to `packages/psychometrics/test/fixtures/`. The tests compare the results with the fixtures. CI runs only the
TypeScript tests; R runs when fixtures change.

## References

Fixtures were generated with R 4.6.1, psych 2.6.9, mirt 1.48, ltm 1.2-0, catR 3.17, TAM 4.3-25, eRm 1.0.10, difR 6.1.0,
lordif 0.4-2, MASS 7.3-65, rms 8.2.0, equate 2.0.9, equateIRT 2.5.2, irr 0.85, cacIRT 1.4, sirt 4.2-133, nlme 3.1-169,
LNIRT 0.5.1, numDeriv 2016.8-1.1, lme4 2.1.0, metafor 5.2.1, GDINA 2.13.2, alabama 2025.1.0, NPCD 1.0.11, CDM 8.3.14, cdcatR 1.0.7 and jsonlite
2.0.0.

| Statistic                                                                                                             | Reference                                                                                                                       |
| --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Alpha, `itemTotal` (`raw.r`), `itemRest` (`r.drop`), alpha if deleted                                                 | `psych::alpha` on the complete cases                                                                                            |
| Guttman split-half coefficient                                                                                        | `psych::splitHalf(raw = TRUE, brute = TRUE, covar = TRUE)`, the split's entry                                                   |
| Item count, mean and SD; biserial; D; KR-20; KR-21; split-half correlation and Spearman-Brown; SEM; option statistics | Base R, written out in `validation/psychometrics/ctt.R`                                                                         |
| McDonald's omega and the minres loadings                                                                              | Base R: Σ_(i<j) (r_ij - λ_i λ_j)² minimized by BFGS, then Newton; `psych::omega(nfactors = 1)`'s `omega.tot` agrees within 1e-4 |
| Scoring against a key                                                                                                 | `mirt::key2binary`                                                                                                              |
| Normal quantile, density and distribution function, logistic function, chi-square upper tail                          | `qnorm`, `dnorm`, `pnorm`, `plogis`, `pchisq(lower.tail = FALSE)`                                                               |
| Gauss-Hermite nodes and weights                                                                                       | Golub-Welsch: `eigen` of the Jacobi matrix, a different algorithm                                                               |
| Eigenvalues and eigenvectors of a symmetric matrix                                                                    | Base R `eigen(symmetric = TRUE)` (LAPACK), a different algorithm from the package's Jacobi method                               |

| IRT quantity                                                           | Reference                                                                                                                                                                                                                 |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1PL, 2PL and anchored 2PL estimates, SEs, log-likelihood, AIC, BIC     | `mirt(SE.type = "Richardson")`, `TOL = 1e-10`, 61 rectangular points; the 1PL as a 2PL with equal slopes                                                                                                                  |
| 4PL estimates with priors, SEs, log-likelihood, AIC, BIC               | `mirt(SE.type = "Oakes")` with `norm` priors on logit g and logit u                                                                                                                                                       |
| 2PL with Gauss-Hermite quadrature                                      | `ltm(GHk = 21)`, `summary()` for the SEs                                                                                                                                                                                  |
| 3PL with a beta prior on c                                             | Base R: direct maximization of the log-posterior (BFGS, then Newton) and `optimHess` for the SEs                                                                                                                          |
| GRM, GPCM and NRM estimates, log-likelihood, AIC, BIC                  | `mirt(itemtype = "graded", "gpcm", "nominal", SE.type = "Richardson")`, `TOL = 1e-10`, 61 points                                                                                                                          |
| PCM                                                                    | mirt's `gpcm` with equal slopes                                                                                                                                                                                           |
| RSM                                                                    | mirt's `rsm` with the slopes freed and equal and the latent variance fixed at 1                                                                                                                                           |
| Polytomous parameters and SEs                                          | Base R: mirt's estimates in this library's parameters; the delta method on `vcov()`, Jacobian by central differences                                                                                                      |
| Mixed-format 3PL and GPCM with anchors and a prior                     | `mirt(SE.type = "Oakes")` with a `norm` prior on logit g                                                                                                                                                                  |
| EAP and MAP theta, ML theta                                            | `fscores(method = "EAP", "MAP", "ML")`, 61 points                                                                                                                                                                         |
| ML, WLE and MAP standard errors                                        | Base R from `iteminfo`: 1 / √I(θ) and 1 / √(I(θ) + 1)                                                                                                                                                                     |
| WLE theta                                                              | Base R: the root of Warm's equation by `uniroot`, the largest weighted likelihood by `integrate`                                                                                                                          |
| Item and test information                                              | `iteminfo`, `testinfo`                                                                                                                                                                                                    |
| JML thresholds, thetas and log-likelihood                              | `TAM::tam.jml`, iterated by base R from its estimates to a change below 1e-12                                                                                                                                             |
| JML standard errors                                                    | Base R: the inverse threshold information matrix of each item, 1 / √I(θ) for persons                                                                                                                                      |
| mirt's IRT and slope-intercept coefficients                            | `coef(IRTpars = TRUE)`, `coef(IRTpars = FALSE)` and `probtrace` of the same items, every item type                                                                                                                        |
| Normal distributions of theta: anchored, in groups, online calibration | Base R: Newton's method on the marginal likelihood, its gradient by Fisher's identity, the Hessian by central differences of it, which gives the SEs; mirt with `optimizer = "NR"` and `multipleGroup` cross-check        |
| Empirical histogram on fixed items                                     | Base R: EM with the same update and stopping rule                                                                                                                                                                         |
| Testlet model: estimates, SEs, log-likelihood                          | Base R: the bifactor likelihood by dimension reduction, maximized by BFGS and Newton on central-difference gradients; `optimHess` for the SEs; mirt's `bfactor` with equal slopes and free testlet variances cross-checks |
| EAP with testlet effects integrated out                                | Base R: the double sum over the nodes written out                                                                                                                                                                         |
| Cross-product standard errors                                          | `mirt(SE.type = "crossprod")`                                                                                                                                                                                             |
| Covariance of the item parameters                                      | The delta method on mirt's `vcov()`, Jacobian by central differences; with priors by item, base R: `optimHess` inverted                                                                                                   |
| Priors by item                                                         | Base R: direct maximization of the log-posterior, the log-likelihood plus the log priors (BFGS, then Newton)                                                                                                              |
| Posterior of theta on 61 points                                        | Base R: the prior's rectangular weights times the likelihood from mirt's `probtrace`; its mean is mirt's EAP                                                                                                              |
| EAP summed-score tables                                                | `fscores(method = "EAPsum")`; base R's Lord-Wingersky recursion on `probtrace` for another prior                                                                                                                          |
| Marginal and empirical reliability                                     | `marginal_rxx`, `empirical_rxx`; base R's `integrate` for another prior                                                                                                                                                   |

| CAT quantity                                                 | Reference                                                                                               |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| Next item by maximum information, Kullback-Leibler and thOpt | `nextItem(criterion = "MFI", "KL", "thOpt")` at catR's estimate, after each response of a run           |
| Information and KL scores of every eligible item             | `Ii`; `KL()` divided by the largest likelihood on its grid, computed from `Pi`                          |
| The theta of an item's largest information                   | catR's closed form for the 4PL, written out from `nextItem`'s `thOpt`                                   |
| a-stratification of 2PL items                                | `aStratified` strata, and `nextItem(criterion = "bOpt")` with the other strata excluded                 |
| Theta and standard error after each response                 | `thetaEst`, `semTheta`: `BM`, `ML` and `WL` with `range = c(-6, 6)`; `EAP` with `parInt = c(-6, 6, 61)` |
| Content balancing                                            | `nextItem` with `cbControl`                                                                             |
| Stopping decisions                                           | `checkStopRule` with `length`, `precision` and `minInfo` on the remaining items, alone and together     |
| Randomesque, Sympson-Hetter, multistage routing, simulation  | Properties over seeded simulations and hand-derived routes (below)                                      |
| Enemy sets, exposure calibration, form assembly              | Properties (below); catR has no enemy sets or form assembly                                             |

| Diagnostic CAT quantity                                    | Reference                                                                                                    |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Posterior, mastery and MAP profile after each response     | Base R: the prior times the probability of each response at every profile, normalized                        |
| KL, PWKL, MPWKL, GDI and SHE scores of every eligible item | Base R, written out over the profiles from the definition of each rule; PWKL also against `cdcatR:::PWKL.M`  |
| Item sequences                                             | The replay in base R, selecting the highest score, the first on ties; `cdcat()` for GDI, MPWKL and JSD (SHE) |
| Fixed-precision test length                                | `cdcat(FIXED.LENGTH = FALSE, precision.cut = 0.8)` for GDI                                                   |
| Posterior and mastery thresholds                           | Base R: the largest posterior at least p; every mastery at least p or at most 1 - p                          |

| Fit statistic                                              | Reference                                                                 |
| ---------------------------------------------------------- | ------------------------------------------------------------------------- |
| Item infit and outfit mean squares and standardized values | eRm's `itemfit` on its `RM` and `PCM` items and `person.parameter` thetas |
| Person infit and outfit                                    | eRm's `personfit`, the same inputs                                        |
| lz                                                         | mirt's `personfit(Theta = )` `Zh`, at mirt's EAP thetas                   |
| Q3                                                         | mirt's `residuals(type = "Q3", Theta = )`, the same thetas                |
| S-X², degrees of freedom, significance                     | mirt's `itemfit(fit_stats = "S_X2")`                                      |

| DIF statistic                                                                | Reference                                                                                                                                                |
| ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Mantel-Haenszel chi-square, with and without the correction                  | Base R's `mantelhaen.test`; difMH's `MH` where they agree                                                                                                |
| Common odds ratio, MH D-DIF and its standard error                           | difR's `difMH`: `alphaMH`, `varLambda`                                                                                                                   |
| ETS categories                                                               | Base R: ETS's rule written out from the values above                                                                                                     |
| Logistic regression chi-squares and R² changes                               | difR's `difLogistic(type = "udif", "nudif", "both")`                                                                                                     |
| Lord's chi-square and the mean-sigma transformation                          | difR's `difLord(irtParam = )` on ltm calibrations run to convergence, checked against `LordChi2` and `itemRescale`                                       |
| Purification, complete data                                                  | Base R's statistics on difR's anchors; `difMH(purify = TRUE)` and `difLogistic(purify = TRUE)` agree in odds ratios, chi-squares, rounds and convergence |
| Sparse data, purified and not                                                | Base R: the matching score written out, `mantelhaen.test` and `glm` on each item's persons                                                               |
| Mantel chi-square, standardized mean difference, effect size, ETS categories | Base R written out from the rows of each score group; the chi-square also against `mantelhaen.test(correct = FALSE)` on dichotomized items               |
| Generalized Mantel-Haenszel chi-square and degrees of freedom                | Base R's `mantelhaen.test` on the 2 × K × S table of the categories used                                                                                 |
| Ordinal logistic regression deviances, R² changes, slope change              | `rms::lrm`, lordif's engine, run to `eps = 1e-14`; `MASS::polr` agrees within 1e-8 in the deviance; lordif's `rundif` on the same matching scores        |
| Polytomous sparse data and purification                                      | Base R: the matching score written out, `mantelhaen.test` and `lrm` on each item's persons; `rundif` agrees with the final criterion of each item        |

| Linking and equating quantity                              | Reference                                                                                                      |
| ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Mean-mean, mean-sigma, Haebara, Stocking-Lord: 2PL and 3PL | equateIRT's `direc(nq = 30)`                                                                                   |
| The same, mixed-format items                               | Base R: the moments and the criteria written out, minimized by `nlminb`                                        |
| True-score equating                                        | Base R: `uniroot` and Kolen and Brennan's line; equateIRT's `score(method = "TSE")` agrees                     |
| Linear equating                                            | equate's `equate(type = "linear")`                                                                             |
| Equipercentile equating                                    | Base R: Kolen and Brennan's definition; equate's `equate(type = "equipercentile")` within 1e-6                 |
| Log-linear presmoothing                                    | Base R's Poisson `glm` to a deviance change of 1e-15; equate's `presmoothing` within 1e-6                      |
| Tucker, Levine, chained linear equating                    | equate's `equate(type = "linear", method = "tucker", "levine", "chained")`                                     |
| Chained equipercentile, frequency estimation               | Base R: the definitions written out; equate's `equate(method = "chained", "frequency estimation")` within 1e-6 |
| Bivariate log-linear presmoothing                          | Base R's Poisson `glm` to a deviance change of 1e-15; equate's `presmoothing` within 1e-6                      |
| Bootstrap standard errors                                  | equate's `bootstrap(reps = 2000)`, within the Monte Carlo error of both                                        |
| Robust z of common items                                   | Base R: `median` and `IQR`                                                                                     |
| IRT observed-score equating                                | Base R: Lord and Wingersky's recursion; equateIRT's `score(method = "OSE")` within 1e-8                        |
| Delta-method standard errors of IRT equating               | Base R: numDeriv's Jacobian times the covariance of a mirt `multipleGroup` calibration                         |

| Scaling quantity                                                  | Reference                                                                                                                   |
| ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| EAP of the summed score: theta and standard error                 | mirt's `fscores(method = "EAPsum", full.scores = FALSE)` under N(0, 1); base R's recursion on `probtrace` for another prior |
| True-score theta and standard error                               | Base R: `uniroot` on mirt's `expected.test`; 1 / √`testinfo`                                                                |
| Conditional standard error of scale scores, predicted form scores | Base R: Lord and Wingersky's recursion on mirt's `probtrace`, written out                                                   |
| Predicted theta scores                                            | Base R: `pnorm` of the estimates that round to each multiple of the step, summed over the scores that report the same score |
| Scale scores of equated scores                                    | Base R: `approx` on the unrounded table, then rounding written out                                                          |
| Percentile ranks                                                  | Base R: the counts below and at a score; equate's `px` on `ACTmath`                                                         |

| Explanatory quantity                                                               | Reference                                                                                                                                                                                                             |
| ---------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Effects, standard errors, covariance, residual variance, restricted log-likelihood | Base R: the restricted likelihood maximized by `optimize` and the root of its score by `uniroot`, then generalized least squares; metafor's `rma(method = "REML")` at the same τ² and, where it converges, at its own |
| Predictions of new items                                                           | Base R: xᵀη̂ and √(τ² + xᵀ Cov x)                                                                                                                                                                                      |

| Standard-setting quantity                              | Reference                                                                         |
| ------------------------------------------------------ | --------------------------------------------------------------------------------- |
| Angoff cut score, the panelists' cut scores, SD and SE | Base R: `rowSums`, `mean`, `sd`                                                   |
| Bookmark page locations                                | Base R: `uniroot` on the cumulative category probabilities of mirt's `probtrace`  |
| Bookmark raw cut scores                                | mirt's `expected.test` at the page locations                                      |
| Hofstee cut score and failure rate                     | Base R: `uniroot` on the failure rates interpolated by `approxfun`, less the line |
| Classification accuracy and consistency                | cacIRT's `class.Rud`; each category's probability by base R's `pnorm`             |

| Rater statistic                                                             | Reference                                                                                                                                                                                                            |
| --------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Cohen's kappa, unweighted, linear, quadratic                                | irr's `kappa2(weight = "unweighted", "equal", "squared")`; psych's `cohen.kappa`                                                                                                                                     |
| Fleiss' kappa                                                               | irr's `kappam.fleiss`                                                                                                                                                                                                |
| The six intraclass correlations                                             | irr's `icc`; psych's `ICC(lmer = FALSE)` agrees within 1e-12                                                                                                                                                         |
| Many-facet Rasch measures, thresholds, mean and SD of theta, log-likelihood | Base R: the marginal likelihood written out on the package's 61 rectangular points, maximized by BFGS on its exact gradient and polished by Newton's method; TAM's `tam.mml.mfr(constraint = "items")` approaches it |
| Many-facet standard errors                                                  | Base R: `numDeriv::hessian` of the likelihood, and the delta method for the last element of each constraint                                                                                                          |
| Separation, reliability, homogeneity                                        | Base R written out from the measures and standard errors                                                                                                                                                             |
| Person EAP estimates, fair averages, expected ratings                       | Base R, from the same parameters                                                                                                                                                                                     |
| Element infit, outfit and standardized forms                                | Base R written out, at the same person estimates                                                                                                                                                                     |

| Generalizability quantity                         | Reference                                                                                                                                  |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Degrees of freedom, sums of squares, mean squares | Base R: `aov` on the balanced design                                                                                                       |
| Variance components                               | Base R: the expected mean squares of each design written out and solved with `solve`; lme4's REML estimates where no component is negative |
| Universe score and error variances, coefficients  | Base R: the textbook formulas of each design (Brennan, 2001), components below 0 taken as 0                                                |

| Response time quantity                                             | Reference                                                                                                                                                                                                                           |
| ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Lognormal model: intensities, discriminations, σ_τ, log-likelihood | nlme's `lme(log(t) ~ 0 + item, random = ~ 1 \| person, weights = varIdent(form = ~ 1 \| item), method = "ML")`, taken to the maximum of the written-out likelihood by Newton steps with numDeriv; LNIRT's `LNRT` by MCMC within 0.1 |
| Intensity standard errors, posterior speeds                        | Base R: (Σ X' V⁻¹ X)⁻¹ and the normal posterior written out; lme's `vcov` and BLUPs at its own estimates                                                                                                                            |
| Residuals and the person statistic                                 | Base R; LNIRT's `personfitLN`                                                                                                                                                                                                       |
| Normative thresholds, response time effort                         | Base R                                                                                                                                                                                                                              |
| Effort-moderated estimates                                         | mirt's `fscores(method = "EAP")` with rapid guesses as `NA`                                                                                                                                                                         |

| Cognitive diagnosis quantity                                               | Reference                                                                                                                              |
| -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| G-DINA, DINA and DINO probabilities, proportions, log-likelihood, AIC, BIC | GDINA's `GDINA` with `conv.crit = 1e-9`; for the DINA on 8 attributes, the proportions summed over profiles with equal ideal responses |
| Mastery, posteriors, MAP and EAP profiles                                  | GDINA's `personparm`; the MAP where GDINA's has a single mode                                                                          |
| Classification accuracy and consistency                                    | GDINA's `CA` by MAP; by EAP, `CA`'s computation in base R                                                                              |
| Q-matrix validation                                                        | GDINA's `Qval(method = "PVAF")`                                                                                                        |
| Nonparametric classification                                               | NPCD's `AlphaNP(method = "Hamming")`: distances and ties; the profile where NPCD has no tie                                            |
| Additive-model probabilities, coefficients, log-likelihood, AIC, BIC       | GDINA's `GDINA(model = "ACDM", "LLM", "RRUM")` at `conv.crit = 1e-8`, also with a model per item                                       |
| One additive M-step on fixed expected counts                               | Base R: Newton's method on the bounds the maximum meets, certified by the KKT conditions                                               |
| Simulation of cognitive diagnosis responses                                | Properties: frequencies, reproduction, recovery by calibration                                                                         |

The scripts also check the references against each other: KR-20 equals alpha for 0/1 items, the base R item-total
correlations equal psych's, and the base R Guttman coefficient equals psych's for every split psych enumerates.
psych enumerates splits only up to 15,000 of them, so for SAT12's 32 items the Guttman coefficient is base R only.

`validation/psychometrics/irt.R` checks its IRT references the same way:

- The base R marginal likelihood and gradient reproduce mirt's 2PL, and mirt's Oakes and Richardson SEs agree.
- The 1PL and mirt's Rasch model, and the PCM and RSM and mirt's `Rasch` and `rsm` with a free latent variance, have
  the same likelihood up to quadrature.
- mirt's GRM, GPCM, PCM and NRM estimates in this library's parameters equal `coef(IRTpars = TRUE)` within 1e-12.
- mirt's WLE equals the base R WLE for the 1PL, 2PL, PCM, GPCM, RSM and NRM, and catR's `thetaEst(method = "WL")`
  agrees wherever Warm's equation has a single root.
- The diagonal of the covariance matrix of the item parameters reproduces the standard errors above within 1e-6
  relative, the written-out 2PL likelihood reproduces mirt's log-likelihood at mirt's estimates within 1e-6, and the
  posterior mean under N(0, 1) is mirt's EAP within 1e-8.
- TAM's JML estimates are within 4e-9 of the base R solution of its steps, and its log-likelihood within 1e-6. For
  dichotomous items, the base R standard error equals TAM's `errorP` within 1e-6.

`validation/psychometrics/irt.R` simulates adaptive tests of 15 items by maximum information at the EAP estimate,
each with 2 of 6 pretest items, for the online calibration. The tests add a recovery check: pretest items seeded
into adaptive tests simulated by `simulateAdaptiveTest`, calibrated with the operational items fixed and a
normal distribution or a histogram, are within four standard errors of their true values, and the distribution's
mean and SD within 0.1.

`validation/psychometrics/explanatory.R` checks its references against each other: the analytic score of τ² equals the
central-difference derivative of the restricted likelihood within 1e-6, and metafor's `rma` with τ² given reproduces the
base R likelihood, coefficients and covariance within 1e-8. Its own search for τ² agrees with the base R maximum within
1e-6 for every design with an intercept; for the design without one it stops at τ² = 1.03, 2.3 below the maximum in
log-likelihood, so base R is the reference there. The tests add a simulation: 200 banks of 40 items, with sampling
errors of 0.08 to 0.18 and τ = 0.3, whose 90% intervals for 100 held-out items each cover between 85% and 95% of their
true difficulties (88.5% found), and a calibration of 3,000 simulated persons that recovers the effects within four
standard errors and τ within 0.1. Properties cover the order of the items, shifts of the parameter, and the symmetry
of the covariance.

`validation/psychometrics/scaling.R` checks its references against each other: the base R EAP of the summed score equals
mirt's EAPsum within 1e-10, and the base R percentile ranks equal equate's `px` of `ACTmath` within 1e-10. The tests add
a simulation: scores drawn from a posterior and an estimate's normal error, with a scale whose bounds are not multiples
of its step, fall at each reportable score within four binomial standard errors of the predicted probability over
100,000 seeded draws. Properties cover the rest: probabilities sum to 1, a posterior at one theta predicts the form's own
distribution there, and the error of a table equals the one by enumerating the response patterns.

`validation/psychometrics/cat.R` runs each adaptive test in catR step by step: before each item, catR's estimate
from the responses so far, by one method until the responses are mixed and another after, and catR's next item at
that estimate; then a response drawn once from a true theta with a fixed seed. The tests replay the responses
through `adaptiveTest`, which must select catR's item at every step. Where catR picks a content area at random, the
test accepts any area catR could pick, and checks that the item is the most informative of its area. Properties
cover the stochastic parts: randomesque selects only among the top items and reaches each of them; an item every
test selects first is administered at the rate of its exposure parameter, within four binomial standard errors
over 4,000 seeded persons; the last eligible item is never withheld; seeded simulations reproduce exactly.
Multistage routing is checked against routes derived by hand, at and beside the cut points. Enemy sets: over random
sets and responses, no test administers two items of a set. Exposure calibration: every rate of the last simulation
is within its tolerance of r, the result reproduces with its seed, and a test too long for r does not converge.
Form assembly is a heuristic without a reference solution: an assembled form of 30 from 300 items meets every
content constraint and comes within 5% of the information target, a target above any form's information selects the
most informative items, and the counts and information it reports equal those of its items.

`validation/psychometrics/cat.R` also replays diagnostic adaptive tests on the items of GDINA calibrations: `sim30GDINA`
with the G-DINA model (30 items, 5 attributes), and ECPE (`CDM::data.ecpe`, 28 items, 3 attributes) with the G-DINA
model, the DINA model and the G-DINA model under a linear hierarchy of the attributes, which gives three of the eight
profiles prior 0. The first four persons of each dataset take a test of 10 items with each rule, and the prior is the
calibration's profile proportions in multiples of 1/4096, so that its sum is exactly 1 as cdcatR checks. The base R
replay stores, before each item, the posterior, the mastery, the MAP profile and every item's score; the test replays
the persons' responses through `diagnosticAdaptiveTest` and compares each estimate, then scores the items from the
stored posterior, so a difference in a score is the rule's alone. cdcatR's `cdcat()` runs the same tests for GDI, MPWKL
and JSD, the mutual information of the response and the profile in bits: the Shannon entropy rule's score plus the
entropy of the posterior, the same for every item, so the rules rank items alike; the script stops if its item sequence
differs from the replay's. It starts with the item of the highest score (`startRule = "max"`) and takes the prior as
`att.prior` and `initial.distr`. The sequences of the fixed-precision test (`precision.cut = 0.8`) end where the tests
ended, 1 to 10 items. The stopping rules are compared on every state along the GDI replays, at 0.7 and 0.9. Properties:
posteriors sum to 1 and a profile of prior 0 keeps posterior 0; the posterior does not depend on the order of the
responses; no test administers an item twice or two enemies; seeds reproduce simulations.

cdcatR 1.0.7 is installed from CRAN source without its import cdmTools, which only its nonparametric rules use and whose
dependency clarabel needs a Rust compiler; `cat.R` describes the installation. cdcatR's PWKL takes its point estimate by
`sample(which(posterior == max(posterior)), 1)`, which for a single mode m draws a number from 1:m instead of returning
m, so its PWKL item sequences are not reproducible and are not a reference. The PWKL scores are compared with its
`PWKL.M` at the MAP profile instead.

`validation/psychometrics/fit.R`, `dif.R` and `equating.R` store the inputs each reference used: the items, the
thetas, the calibrations. The tests compute the statistics from those inputs, so a difference is the statistic's alone.

Several results of the reference packages are not references:

- **mirt's M-step for a normal distribution of theta.** mirt maximizes Σ n_q ln φ(θ_q) over the mean and the variance,
  without normalizing the weights over the nodes, so its fixed point is not the maximum of the log-likelihood it
  reports; with fixed items the missing information amplifies the gap. On the anchored fixture, N(0.63, 1.34²), its
  estimates are within 4.6e-3 of the maximum, whose log-likelihood is 7.6e-4 higher; where the distribution lies well
  inside the nodes, within 1.4e-9. The M-step of `calibrate` matches the weights' mean and variance to the
  counts', which maximizes the reported likelihood. mirt's default BFGS M-step also stops short;
  `optimizer = "NR"` converges.
- **mirt's Richardson standard errors with a free variance.** `SE.type = "Richardson"` leaves `COV_11` shifted by
  -0.0025 in the fitted object; `"Oakes"` gives the same standard errors and keeps the estimate.
- **mirt's `bfactor` with free testlet variances.** On the simulated testlets, mirt with `optimizer = "NR"` stops
  short, 1.5e-3 from the maximum and 1.5e-4 below its log-likelihood. On sirt's `data.read` it reaches a degenerate
  solution, testlet variances near 58 and a negative slope, 47.7 below the maximum's log-likelihood. The base R
  maximum is the reference; the log-likelihoods of `calibrate` and base R agree within 4e-10.
- **lme's ML fit.** nlme's optimizer stops 9e-8 below the maximum of the log-likelihood, 2e-5 from it in the
  discriminations; Newton steps on the written-out likelihood reach it. LNIRT's MCMC posterior means are within its
  Monte Carlo error, 0.08 on the simulation and 0.03 on CredentialForm1.
- **GDINA on the fraction subtraction data.** Under the DINA, profiles with equal ideal responses share any split of
  their proportion; GDINA's split depends on its starts, so the fixture sums proportions over such profiles, and
  persons' posteriors among them are not compared. Under the G-DINA with 8 attributes, GDINA's estimates have a
  log-likelihood 110.8 below those of `calibrateCdm`: the likelihood has several maxima, and the case is left
  out. `coef` rounds to 4 digits unless given `digits`; GDINA breaks MAP ties at random. GDINA's
  `CA(what = "EAP")` classifies every person in the first profile, as its row matching `whichrow_AinB` returns 1
  for every row; the EAP reference is `CA`'s computation in base R with the profiles matched by their patterns.
- **GDINA's additive models.** GDINA maximizes their M-steps with `constrOptim`'s barrier method, which stops near
  1e-6, so its EM never reaches a change below about 5e-9 and misses `conv.crit = 1e-10`; the fixtures use `1e-8`.
  `calibrateCdm` maximizes each M-step exactly and runs to `tolerance: 1e-10` for the comparison, where its
  log-likelihood is at least GDINA's. A MAP between posteriors within 1e-4 of the best counts as a tie in these
  fixtures, as GDINA's precision moves it. Persons and accuracy are compared at GDINA's own estimates, to which a
  posterior is sensitive. The G-DINA, DINA and DINO coefficients are compared in the per-item mix.
- **The additive M-step on fixed counts.** No solver reaches 1e-8 on boundary solutions: `constrOptim` fails on them,
  and the candidates of `slsqp` and the augmented Lagrangian stop short. The reference is Newton's method on the
  face of the bounds that a candidate meets, kept only when the KKT conditions hold: the objective is convex, so a
  point that meets them is the maximum. Counts for which they fail are not in the fixture.
- **Histograms.** The EM weights of an empirical histogram creep toward a spiky maximum without reaching it, in mirt
  as in `calibrate`, so a histogram is where EM stops; mirt stops by its items' change alone. The fixture checks the
  histogram on fixed items against base R's EM with the stopping rule of `calibrate`.
- **mirt's `expbeta` prior.** mirt 1.48 differentiates the prior with respect to g instead of logit g, so its EM stops
  short of the posterior mode it reports: on LSAT7 with Beta(5, 17) on g, its log-posterior is -2651.687, the base R
  mode's -2651.614. The 3PL is therefore validated against base R, and the 4PL and the mixed format against mirt with
  `norm` priors on logit g and logit u, which the test implements as a custom `ParameterPrior`.
- **mirt's WLE for the 3PL, 4PL and GRM.** mirt maximizes ln L + ½ ln I, which equals Warm's estimator only for
  exponential-family models: the 1PL, 2PL, PCM, GPCM, RSM and NRM. `weightedLikelihoodEstimate` solves Warm's
  equation, as catR does.
- **mirt's polytomous IRT standard errors.** mirt 1.48's `coef(IRTpars = TRUE, printSE = TRUE)` gives Science's
  Comfort item a GPCM b1 standard error of 1.111; the delta method on mirt's own covariance matrix gives 0.651, as
  `calibrate` does. Polytomous standard errors therefore come from the delta method.
- **TAM's JML standard errors.** TAM's `errorP` for a polytomous item uses only the diagonal of its threshold
  information matrix, and TAM does not apply the bias correction to it.
- **eRm's CML.** eRm estimates the Rasch model and the PCM by conditional maximum likelihood, a different estimator. On
  the complete cases of the JML data, its thresholds and TAM's corrected JML thresholds, both centered, differ by up to
  0.10. With missing responses its polytomous estimates are unreliable.
- **difR's continuity correction.** `difMH` subtracts 0.5 from \|Σ (A - E(A))\| even when it is below 0.5, so
  identical groups get a positive chi-square. `mantelHaenszel` and `mantelhaen.test` subtract it only from 0.5 up;
  on the fixtures the two differ for one simulated item.
- **difR's ETS codes.** `print` on a `difMH` result codes items by \|MH D-DIF\| alone. ETS's rule also requires
  significance: on verbal, 14 of the 24 items differ. On the simulation, both flag the three DIF items C.
- **lordif's rounding and missing responses.** `rundif` rounds its p-values, R² changes and slope change to 4 decimals
  and counts a missing response in n for Nagelkerke's R², so the comparison with it is within 5e-5 and on complete data
  only. It matches on the scores it is given: the fixtures pass it the rule's total scores, and theta from a graded
  response calibration (mirt's EAP), which are the `matching` option's.
- **polr's BFGS.** `MASS::polr` stops where BFGS does: against `rms::lrm` run to `eps = 1e-14` its deviances differ by
  up to 4e-9 and its coefficients by up to 1.1e-6, which moves the slope change by up to 1e-7. The references are
  lrm's, which `ordinalLogisticRegressionDif` agrees with within 3e-14 in the slope change.
- **difLord's ltm calibrations.** With ltm's default control, 40 EM cycles and then BFGS at `optim`'s relative
  tolerance, the estimates stop up to 1.5e-3 from the maximum on the fixtures' groups, which moves Lord's chi-square by
  up to 2%. The fixture passes difLord calibrations run to convergence.
- **TAM's EM and nodes.** `tam.mml.mfr` with `conv = 1e-10` runs its 3000 iterations without converging. On data.ratings1
  it stops 0.21 and 0.29 log-likelihood below the maximum, with the SD of theta 0.02 short, because TAM's fixed nodes
  from -6 to 6 cut the distribution at 3 SD there (SD 1.97) where the package places its nodes on the mean ± 6 SD. On
  data.sim.mfr and the simulation, where the SD is about 1, it stops within 3e-5 of the log-likelihood, 1e-4 in the
  measures and 5e-5 in the thresholds. The base R maximum is therefore the reference. TAM 4.3-25 cannot build the
  design for thresholds by a facet other than the items when that facet has more than three levels, which is
  `rater:step` in the simulation.
- **equate's percentile ranks** are rounded to 10 decimals, which moves equipercentile equivalents by up to 1e-7, and
  equate returns an X score above Y's scale with a percentile rank of 1 unchanged. The unrounded base R computation is
  the reference.
- **equateIRT's Rasch linking** fixes A = 1 for every method, and its `direc` takes dichotomous items only.
- **equate's frequency estimation** multiplies the groups' numbers of persons into its weights, so that its default
  `ws` weights `from` by n₁² / (n₁² + n₂²), and gives the persons at an anchor score that only one group has their own
  group's weight alone, so that the synthetic distributions lose mass. The definition's weight w₁ is equate's `ws = w₁
  n₂ / (w₁ n₂ + (1 - w₁) n₁)`, which the script passes; the fixtures have anchor scores in both groups.
- **equate's bootstrap** draws with R's generator, whose streams differ from `createRandom`'s, so the standard errors
  agree within their Monte Carlo error: about 1.6% of each standard error for 2000 replications and 3.2% for 500, as
  the standard deviation of B values has a relative standard error of 1 / √(2 (B - 1)). The test compares standard
  errors above 0.05 within 10%, about three standard deviations of the difference. Frequency estimation resamples are
  smoothed, which takes seconds, so the test runs 500 replications of it.
- **mirt's covariance of a concurrent calibration** is of the slopes and intercepts (a1, d); the script converts it to
  that of a and b = -d / a1 with the Jacobian of the transformation, and takes the means and variances of the groups
  from the same calibration as known populations. The standard errors of equating then come from numDeriv's
  Richardson Jacobian, which agrees with the central differences of the package within 1e-8.
- **lme4's REML estimates** equal the analysis of variance estimates of a balanced design when no component is negative,
  but lme4's optimizer stops where the REML criterion is flat to its tolerance. The script restarts bobyqa from its own
  estimate three times, after which the estimates agree within 3.3e-7; the test allows 1e-5. gtheory, which wraps lme4
  for G and D studies, was removed from CRAN in March 2025, so lme4 is the reference. REML keeps estimates at 0 or
  above, so the case with a negative component has the base R computation alone.

No package among the references implements the modified Angoff, Bookmark or Hofstee method, so their references are
base R written out from each method's definition. The tests add values derived by hand: the Bookmark locations
b + ln((RP - c) / (d - RP)) / a of logistic items, b + ln 2 at RP 2/3 for a Rasch item (Huynh, 2006), and the peak of
P(θ) I(θ) of a 3PL item at RP (2 + c) / 3 (Huynh, 1998).

## Simulation

R's generator differs from `createRandom`, so no R function reproduces a simulation. The tests of `sim` use fixed
seeds and bounds of four standard errors, which a correct simulation exceeds with a probability below 1e-4 per check:

- **Moments.** The mean and standard deviation of 100,000 draws of each distribution and of `simulateThetas`, and of
  20,000 items with each model's default distributions.
- **Categories.** The frequency of each category of 20,000 responses at one theta, for 3PL, GRM, GPCM, RSM and NRM
  items, against P(k | θ).
- **Missingness.** The share of missing responses, with the other responses those of the same seed without them.
- **Reproduction.** Equal seeds give equal simulations, for every model.
- **Recovery.** 20 simulated 2PL items, calibrated from 2,000 simulated persons: every estimate within four standard
  errors of the true value, and root mean square errors below 0.15 for a and 0.2 for b (0.10 and 0.16 found).

## Datasets

| Fixture                                | Data                                                                                                                                                                                                                                             | Covers                                                                                                                                 |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| `ctt-lsat7.json`                       | `mirt::LSAT7`, expanded: 1000 persons, 5 dichotomous items                                                                                                                                                                                       | Complete data, an odd number of items                                                                                                  |
| `ctt-simulated-2pl.json`               | 2PL simulation, `set.seed(20261029)`: 300 persons, 16 dichotomous items                                                                                                                                                                          | An even number of items                                                                                                                |
| `ctt-bfi-neuroticism.json`             | `psych::bfi` items N1-N5, scored 0-5: 2800 persons, 106 incomplete                                                                                                                                                                               | Polytomous items, missing responses                                                                                                    |
| `ctt-sat12.json`                       | `mirt::SAT12` with its key, 8 as missing: 600 persons, 32 items, 28 incomplete                                                                                                                                                                   | Scoring, distractor analysis, omitted responses                                                                                        |
| `math-functions.json`                  | 18 probabilities from 1e-300 to 1 - 1e-10, 13 points from -30 to 30; chi-squares from 0 to 400, 1 to 100 df                                                                                                                                      | Tails of the normal quantile and of the chi-square                                                                                     |
| `math-gauss-hermite.json`              | Rules of 1, 2, 5, 21 and 61 points                                                                                                                                                                                                               | Small and large rules                                                                                                                  |
| `math-eigen.json`                      | Base R `eigen`: a random symmetric 5 × 5 matrix, a repeated eigenvalue, LSAT7's and `psych::bfi`'s (25 items) correlations, small matrices                                                                                                       | Negative and repeated eigenvalues, 25 rows                                                                                             |
| `irt-lsat7.json`                       | `mirt::LSAT7`, expanded: 2PL, 2PL with two anchors, 3PL with a prior                                                                                                                                                                             | All 32 patterns and 4 with missing responses                                                                                           |
| `irt-simulated-1pl.json`               | 1PL simulation, `set.seed(20261029)`: 800 persons, 12 items, a = 1.3, 10% missing                                                                                                                                                                | Missing responses, a shared parameter                                                                                                  |
| `irt-simulated-4pl.json`               | 4PL simulation, `set.seed(20261029)`: 2000 persons, 12 items                                                                                                                                                                                     | Four parameters, priors on both asymptotes                                                                                             |
| `irt-science.json`                     | `mirt::Science`, recoded 0-3: 392 persons, 4 items with 4 categories                                                                                                                                                                             | GRM, GPCM, PCM, RSM, NRM; all 256 patterns and 4 with missing responses                                                                |
| `irt-simulated-mixed.json`             | Mixed-format simulation, `set.seed(20261029)`: 3000 persons, 9 3PL items, 3 GPCM items with 3 or 4 categories                                                                                                                                    | Mixed models, an anchor of each, a prior on c                                                                                          |
| `irt-simulated-pcm.json`               | Rasch and PCM simulation, `set.seed(20261029)`: 500 persons, 10 dichotomous and 4 polytomous items, 5% missing                                                                                                                                   | JML; 15 extreme total scores, missing responses                                                                                        |
| `cat-3pl.json`                         | catR's `genDichoMatrix(items = 100, model = "3PL", seed = 20261029)`: 4 runs of 15 items                                                                                                                                                         | MFI, KL, thOpt; MAP, EAP then ML, EAP then WLE; 15 stopping decisions                                                                  |
| `cat-2pl.json`                         | `genDichoMatrix(items = 100, model = "2PL")`: 4 strata of 25, 3 items from each                                                                                                                                                                  | a-stratification                                                                                                                       |
| `cat-gpcm.json`                        | `genPolyMatrix(items = 40, nrCat = 4, model = "GPCM")`: 2 to 4 categories, 2 runs of 15 items                                                                                                                                                    | Polytomous MFI and KL; MAP, EAP then ML                                                                                                |
| `cat-content.json`                     | `genDichoMatrix(items = 60, model = "3PL", cbControl)`: areas with targets 0.5, 0.3 and 0.2                                                                                                                                                      | Content balancing                                                                                                                      |
| `cat-cdm.json`                         | GDINA's `sim30GDINA` and `CDM::data.ecpe` calibrated by GDINA: G-DINA; ECPE also DINA and G-DINA under a linear hierarchy; 4 persons, 10 items, 5 rules each                                                                                     | The five diagnostic selection rules, posterior thresholds, a prior with zeros, DINA items, fixed precision                             |
| `interop-mirt.json`                    | 25 items set in mirt: 2PL, 3PL, 4PL, `graded` (one negative slope), `gpcm`, `Rasch`, `rsm`, `nominal`                                                                                                                                            | Both coefficient forms, probabilities                                                                                                  |
| `fit-rasch.json`                       | difR's `verbal`, 24 items under `RM`; `mirt::Science` recoded 0-3 under `PCM`                                                                                                                                                                    | Extreme scores, polytomous mean squares                                                                                                |
| `fit-summed-score.json`                | `mirt::LSAT7` and `SAT12` scored (omissions wrong) under the 2PL; Science under the GPCM and GRM                                                                                                                                                 | Dichotomous and polytomous collapsing                                                                                                  |
| `fit-residuals.json`                   | SAT12 scored with omissions missing under the 2PL; Science under the GPCM                                                                                                                                                                        | Missing responses, polytomous lz and Q3                                                                                                |
| `dif.json`                             | difR's `verbal`, Gender 1 focal; a 2PL simulation, `set.seed(20261029)`: 2 × 1000 persons, 20 items; both also with 2% of the responses missing                                                                                                  | Known uniform DIF in items 1, 2 and nonuniform in item 3; purification; 39% and 33% of the persons incomplete                          |
| `dif-polytomous.json`                  | `psych::bfi` N1-N5 (2694 complete cases, women focal); a graded response simulation, `set.seed(20261029)`: 2 × 1000 persons, 15 items of 4 categories; both also with 2% missing                                                                 | Known uniform DIF in items 1, 2 and nonuniform in item 3; graded EAP theta as the matching score; purification                         |
| `equating-linking.json`                | equateIRT's `est2pl`, `est3pl`, forms 1 to 2 over I1-I10; 10 mixed-format items, `set.seed(20261029)`                                                                                                                                            | 3PL, GRM, GPCM; a known transformation with noise                                                                                      |
| `equating-true-score.json`             | est3pl's form 2 and form 1 linked to it; the mixed-format items                                                                                                                                                                                  | Scores below Σ c, polytomous items                                                                                                     |
| `equating-observed-score.json`         | equate's `ACTmath` (40 items); the total scores of `KBneat` (36 items)                                                                                                                                                                           | Scores without persons                                                                                                                 |
| `standard-setting.json`                | Angoff ratings, `set.seed(20261029)`: 8 panelists, 30 items; 9 items set in mirt; SAT12, omissions wrong                                                                                                                                         | Every Bookmark page at RP 0.67 and 0.5; mean and whole Hofstee bounds; classification by one and three cuts                            |
| `irt-latent.json`                      | 2PL simulations, `set.seed(20261029)`: 4 anchors of 12 items, θ ~ N(0.5, 1.2²); 2 × 1000 persons; 2,000 adaptive tests of 37 operational and 6 pretest items                                                                                     | A free normal distribution, two groups, online calibration, a histogram                                                                |
| `irt-scoring.json`                     | LSAT7 under the 2PL and Science under the GPCM, the calibrations of `irt-lsat7` and `irt-science`                                                                                                                                                | Summed-score tables and marginal reliability under two priors; 1000 EAP estimates                                                      |
| `irt-cross-product.json`               | LSAT7, the simulated 1PL with missing responses, Science under the GPCM                                                                                                                                                                          | Cross-product standard errors, a shared slope                                                                                          |
| `irt-covariance.json`                  | The calibrations of `irt-lsat7` and `irt-science`; the LSAT7 2PL with a log-normal prior on a and priors by item                                                                                                                                 | Covariances between items; a prior replaced for one item                                                                               |
| `irt-posterior.json`                   | The calibrations of `irt-lsat7` and `irt-science`: 36 patterns of LSAT7, 8 of Science; under N(0, 1) and N(0.4, 1.3²)                                                                                                                            | Posteriors on 61 nodes, a prior other than N(0, 1), missing responses                                                                  |
| `scaling-conversion.json`              | LSAT7 under the 2PL and Science under the GPCM, as mirt calibrates them; forms 1 and 2 of equateIRT's `est3pl` (20 3PL items each) set in mirt                                                                                                   | Tables of both methods on four scales, a prior other than N(0, 1), polytomous forms, scores with no theta                              |
| `scaling-prediction.json`              | Posteriors of theta given responses to five 2PL practice items under N(0, 1) and N(0.3, 1.2²), predicted on the forms of the conversion fixture                                                                                                  | Form scores with pooled scale scores, theta scores on scales with bounds that are not multiples of the step, errors of 0, 0.25 and 0.6 |
| `scaling-norms.json`                   | `equate::ACTmath` form X (4329 persons); a simulated normal sample of scale scores (`set.seed(20261029)`, 2000 persons); one score                                                                                                               | Ties, scores outside the sample, scores between scores                                                                                 |
| `explanatory-regression.json`          | 2PL simulations (`set.seed(20261029)`) calibrated by mirt: 1000 persons and 20 items with three features and τ = 0.3; 1500 persons and 40 items with two features and τ = 0; 800 persons and 12 items with one feature and τ = 0.25              | b and ln a on features, with and without an intercept, τ² at its boundary, a single feature                                            |
| `equating-presmoothing.json`           | ACTmath's X and KBneat's X totals                                                                                                                                                                                                                | Degrees 2, 3, 4 and 6                                                                                                                  |
| `equating-neat.json`                   | equate's `KBneat`, internal anchor; a 2PL simulation, `set.seed(20261029)`, external anchor: 20-item forms, 10 anchor items, 700 and 500 persons                                                                                                 | Both anchors; the groups' shares and a weight of 0.3; Levine's true-score method                                                       |
| `equating-bivariate-presmoothing.json` | The X group of each case of `equating-neat.json`                                                                                                                                                                                                 | Degrees (4, 3), (3, 3), (2, 2), (4, 2); cross degrees 0 to 2                                                                           |
| `equating-bootstrap.json`              | equate's `bootstrap`, 2000 replications, `set.seed(20261029)`: `KBneat`, `ACTmath`, a single group of 800 persons under a 2PL                                                                                                                    | Five NEAT methods, random groups, a single group; smoothed resamples                                                                   |
| `equating-robust-z.json`               | A Rasch simulation, `set.seed(20261029)`: 15 items, 2 drifting; equateIRT's `est2pl`, forms 1 to 2, I1-I10; the mixed-format linking items                                                                                                       | Flagged items; polytomous locations                                                                                                    |
| `equating-irt-observed-score.json`     | Form 2 of equateIRT's `est3pl` and form 1 moved to its scale by Haebara; the mixed-format items of `equating-linking.json`                                                                                                                       | Weights 0.3 to 0.7; N(0, 1) and N(B, A²) populations; polytomous items                                                                 |
| `equating-irt-standard-errors.json`    | A concurrent mirt `multipleGroup` calibration, `set.seed(20261029)`: two 20-item 2PL forms sharing 8 items, 1500 persons per group; the mixed-format items with a made-up covariance                                                             | Common items; true-score and observed-score methods; 3PL, GRM and GPCM parameters                                                      |
| `irt-testlet.json`                     | A 2PL testlet simulation, `set.seed(20261029)`: 1000 persons, 3 testlets of 4 items, SDs 0.5, 0.8 and 1; sirt's `data.read`, 328 persons, 3 passages of 4 items                                                                                  | 21 nodes, as mirt's `bfactor`; EAP of 10 patterns                                                                                      |
| `rater.json`                           | irr's `anxiety` (20 subjects, 3 raters), also with two ratings missing; irr's `diagnoses` (30 subjects, 6 raters)                                                                                                                                | Weighted kappa, nominal categories, missing ratings                                                                                    |
| `rater-many-facet.json`                | sirt's `data.ratings1` (135 students, 5 criteria, 7 raters); TAM's `data.sim.mfr` (100 persons, 5 items, 5 raters, 3 topics); a simulation, `set.seed(20261029)`: 300 persons, 4 items, 6 raters, 2 raters per person, rater-specific thresholds | Shared and per-item or per-rater thresholds; a third facet; nested and incomplete designs                                              |
| `generalizability.json`                | Simulations, `set.seed(20261029)`: p x i (40 x 12), p x r x i (40 x 4 x 6), (r:p) x i (25 x 3 x 4), i:p, r:pi, i:r:p, (i:p) x (r:p); p x i (8 x 4) with a negative item component                                                                | Crossed and nested designs; labels repeated within parents; a negative component; fixed facets and nesting in decision studies         |
| `response-time.json`                   | A lognormal simulation, `set.seed(20261029)`: 400 persons, 12 items, 5% of times missing; LNIRT's `CredentialForm1`, items 1 to 10, 1636 persons, times of 0 as missing                                                                          | Calibration, speeds, residuals; NT10 and NT15 with a 30 s cap                                                                          |
| `cdm.json`                             | GDINA's `realdata_Tatsuoka1990`, fraction subtraction: 536 persons, 20 items, 8 attributes, under the DINA; a G-DINA simulation, `set.seed(20261029)`: 1500 persons, 15 items, 4 attributes, under the G-DINA and DINO                           | 256 profiles, indistinguishable profiles; persons, accuracy, PVAF, nonparametric classification of 30 persons                          |
| `cdm-additive.json`                    | GDINA's `realdata_ECPE` (2922 persons, 28 items, 3 attributes) and `sim30GDINA` (1000 persons, 30 items, 5 attributes)                                                                                                                           | A-CDM, LLM, R-RUM and a mix of all six models per item                                                                                 |
| `cdm-maximum.json`                     | 59 sets of expected counts of one item, `set.seed(20261029)`: 3 links, 2 to 4 attributes                                                                                                                                                         | DINA-like, DINO-like, non-monotone and extreme; 16 meet a bound                                                                        |

The upper and lower groups use 27% of the persons. Edge cases (constant items, persons and items without responses,
one item, two persons) are tested against values derived by hand in `test/ctt.test.ts`.

## Tolerances

| Quantity                                                                                         | Tolerance                                          | Largest difference found |
| ------------------------------------------------------------------------------------------------ | -------------------------------------------------- | ------------------------ |
| CTT statistics                                                                                   | 1e-10 absolute                                     | 4e-14                    |
| Normal quantile and density, logistic function                                                   | 1e-14 relative                                     | 3e-16                    |
| Gauss-Hermite nodes and weights                                                                  | 1e-12 absolute                                     | 3e-14                    |
| Eigenvalues and eigenvectors, up to sign                                                         | 1e-12 absolute                                     | 2e-14                    |
| IRT item parameters                                                                              | 1e-5 absolute                                      | 6e-7                     |
| IRT standard errors                                                                              | 1e-4 relative                                      | 9e-6                     |
| IRT log-likelihood, AIC, BIC at the estimates                                                    | 1e-6 absolute                                      | 3e-10                    |
| 4PL, polytomous and mixed-format parameters                                                      | 2e-4 absolute                                      | 1.1e-4                   |
| 4PL, polytomous and mixed-format standard errors                                                 | 1e-3 relative                                      | 8e-5                     |
| 4PL, polytomous and mixed-format log-likelihood, AIC, BIC                                        | 1e-4 absolute                                      | 7e-5                     |
| Log-likelihood at the reference's estimates                                                      | 1e-8 absolute                                      | 2e-11                    |
| EAP and WLE theta and standard error                                                             | 1e-10 absolute                                     | 6e-14                    |
| ML and MAP theta and standard error                                                              | 1e-5 absolute                                      | 7e-6                     |
| Item and test information                                                                        | 1e-12 absolute                                     | 3e-14                    |
| JML thresholds, standard errors, thetas, log-likelihood                                          | 1e-10 absolute                                     | 1.4e-12                  |
| mirt coefficient forms                                                                           | 1e-15 relative                                     | 4e-16                    |
| CAT ML, MAP and WLE theta and standard error against catR                                        | 1e-4 absolute                                      | 3e-5                     |
| CAT EAP theta and standard error against catR                                                    | 1e-6 absolute                                      | 6e-8                     |
| CAT information and KL scores                                                                    | 1e-12 relative                                     | 4e-15                    |
| CD-CAT scores of every rule against the written-out replay                                       | 1e-12 relative                                     | 9e-16                    |
| CD-CAT posterior and attribute mastery                                                           | 1e-12 absolute                                     | 6e-16                    |
| Theta of an item's largest information                                                           | 1e-10 absolute                                     | 4e-15                    |
| Category probabilities against `probtrace`                                                       | 1e-15 absolute                                     | 4e-16                    |
| Chi-square upper tail                                                                            | 1e-12 relative                                     | 2e-14                    |
| Infit, outfit, lz, Q3, S-X² and its significance                                                 | 1e-12 absolute                                     | 3e-14                    |
| Mantel-Haenszel statistics                                                                       | 1e-12 absolute                                     | 1e-15                    |
| Logistic regression chi-squares and R² changes                                                   | 1e-9 absolute                                      | 7e-11                    |
| Lord's chi-square                                                                                | 1e-5 relative                                      | 3e-7                     |
| Lord's mean-sigma transformation                                                                 | 1e-8 absolute                                      | 9e-10                    |
| Mean-mean and mean-sigma transformations                                                         | 1e-12 absolute                                     | 2e-16                    |
| Haebara and Stocking-Lord transformations                                                        | 1e-7 absolute                                      | 2.5e-8                   |
| Linear, equipercentile and true-score equivalents                                                | 1e-12 absolute                                     | 2e-13                    |
| Equipercentile equivalents against equate                                                        | 1e-6 absolute                                      | 1e-7                     |
| Tucker, Levine and chained linear equivalents against equate                                     | 1e-12 absolute                                     | 1.4e-14                  |
| Chained equipercentile and frequency estimation equivalents                                      | 1e-12 absolute                                     | 2.7e-13                  |
| The same against equate                                                                          | 1e-6 absolute                                      | 4.1e-8                   |
| Bivariate presmoothed counts, relative to max(1, count)                                          | 1e-11                                              | 5.5e-14                  |
| Bootstrap standard errors above 0.05 against equate's                                            | 10% relative                                       | 4.4%                     |
| Robust z                                                                                         | 1e-12 absolute                                     | 1.5e-15                  |
| IRT observed-score equivalents                                                                   | 1e-12 absolute                                     | 2.5e-14                  |
| Delta-method standard errors of IRT equating, relative                                           | 1e-6                                               | 8.5e-9                   |
| Cut scores, Bookmark locations                                                                   | 1e-12 absolute                                     | 4e-15                    |
| Normal distribution function                                                                     | 1e-14 relative                                     | 1.5e-16                  |
| Distributions of theta: parameters, mean, SD                                                     | 1e-7 absolute                                      | 3e-9                     |
| Distributions of theta: standard errors                                                          | 1e-6 relative                                      | 1.6e-9                   |
| Histogram weights on fixed items                                                                 | 1e-12 absolute                                     | 6e-15                    |
| Cross-product standard errors                                                                    | as the Hessian's: 1e-4, flat models 1e-3, relative | 4.4e-7; GPCM 2e-5        |
| Covariance of item parameters: 2PL, base R maximum                                               | 1e-4 of √(Σ_ii Σ_jj)                               | 1.6e-5                   |
| Covariance of GPCM parameters                                                                    | 1e-3 of √(Σ_ii Σ_jj)                               | 6.4e-5                   |
| Summed-score tables                                                                              | 1e-12 absolute                                     | 3e-15                    |
| Posterior of theta                                                                               | 1e-12 absolute                                     | 3e-15                    |
| EAP of the summed score: theta and standard error                                                | 1e-12 absolute                                     | 1.3e-15                  |
| True-score theta and standard error                                                              | 1e-10 absolute                                     | 7e-15                    |
| Scale scores and conditional errors of tables                                                    | 1e-10 absolute                                     | 4.5e-13                  |
| Conditional errors of scale scores at the reference's thetas                                     | 1e-12 absolute                                     | 6e-14                    |
| Probabilities of predicted scores, interpolated scale scores, percentile ranks                   | 1e-12 absolute                                     | 8e-17                    |
| Means of predicted scores                                                                        | 1e-11 absolute                                     | 2.3e-13                  |
| Item feature regression: effects, standard errors, covariance, residual variance, log-likelihood | 1e-10 absolute                                     | 9e-12                    |
| Predictions of new items                                                                         | 1e-10 absolute                                     | 2e-11                    |
| Marginal reliability, 61 nodes against `integrate`                                               | 1e-8 absolute                                      | 7e-10                    |
| Empirical reliability, omega and its loadings                                                    | 1e-10 absolute                                     | 2e-14                    |
| Purified and sparse Mantel-Haenszel statistics                                                   | 1e-12 absolute                                     | 4e-16                    |
| Purified and sparse logistic regression                                                          | 1e-9 absolute                                      | 9e-11                    |
| Mantel and generalized Mantel-Haenszel chi-squares, SMD, effect size                             | 1e-12 absolute                                     | 3.4e-13                  |
| Ordinal logistic regression chi-squares, significances, R², slope change                         | 1e-8 absolute                                      | 2e-10                    |
| Ordinal logistic regression against lordif's rounded statistics                                  | 5e-5 + 1e-9 absolute                               | 4.96e-5                  |
| Presmoothed counts                                                                               | 1e-11 relative                                     | 6e-13                    |
| Classification probabilities, accuracy, consistency                                              | 1e-14 absolute                                     | 1e-16                    |
| Kappa and intraclass correlations                                                                | 1e-12 absolute                                     | 2e-15                    |
| Many-facet measures, thresholds, mean and SD of theta                                            | 1e-8 absolute                                      | 1.3e-10                  |
| Many-facet log-likelihood                                                                        | 1e-8 absolute                                      | 4.5e-12                  |
| Many-facet standard errors, separation, reliability, homogeneity                                 | 1e-6 relative                                      | 6.2e-8                   |
| Many-facet EAP estimates, fit, fair and expected ratings, summaries from the same inputs         | 1e-10 absolute                                     | 2.4e-14                  |
| Generalizability sums of squares, mean squares, components, relative to max(1, value)            | 1e-10                                              | 6e-15                    |
| Components against lme4's REML estimates                                                         | 1e-5 absolute                                      | 3.3e-7                   |
| Decision study variances and coefficients, relative                                              | 1e-10                                              | 5e-15                    |
| Testlet parameters and SDs against the base R maximum                                            | 2e-5 absolute                                      | 9e-6                     |
| Testlet standard errors                                                                          | 1e-4 relative                                      | 9e-6                     |
| Testlet log-likelihood                                                                           | 1e-8 absolute                                      | 4e-10                    |
| Lognormal calibration against the maximum                                                        | 1e-8 absolute                                      | 2e-9                     |
| Speeds, residuals, statistics, thresholds, effort                                                | 1e-10 absolute                                     | 7e-15                    |
| Effort-moderated EAP estimates                                                                   | 1e-10 absolute                                     | 3e-15                    |
| Cognitive diagnosis probabilities, proportions, persons, accuracy, PVAF                          | 1e-5 absolute                                      | 1.1e-6                   |
| Cognitive diagnosis log-likelihood, AIC, BIC                                                     | 1e-8 absolute                                      | 5e-11                    |
| Additive-model probabilities, coefficients, proportions                                          | 1e-5 absolute                                      | 2.1e-6                   |
| Additive-model log-likelihood                                                                    | at least GDINA's less 1e-8                         | never below              |
| Persons and accuracy at GDINA's additive estimates                                               | 1e-6 absolute                                      | 6.7e-8                   |
| One additive M-step on fixed counts                                                              | 1e-8 absolute                                      | 1.0e-12                  |

Entries of a covariance matrix are compared relative to √(Σ_ii Σ_jj) of the reference, so that an off-diagonal entry is
on the correlation scale.

Calibrations run with `tolerance: 1e-10`, mirt's `TOL`. The parameter and SE tolerances cover where EM stops in
mirt: ltm and the base R mode, which converge further, agree within 3e-9. mirt's EM stops short on the flat
likelihoods of the 4PL, the polytomous models and the mixed format, whatever its `TOL`: the log-likelihood or
log-posterior of `calibrate` at its own estimates is higher, by 2e-9 for the 4PL, up to 3e-9 on Science and 2e-7 for
the mixed format, and the log-likelihood at mirt's estimates agrees within 2e-11. SEs come from numerical Hessians
on both sides. mirt's ML and MAP use `nlm` with numerical derivatives, good to about 7e-6; the base R WLE is exact. mirt
computes 1 - P by subtraction, so information is compared in absolute terms: in the tails mirt keeps few significant
digits. JML runs with `tolerance: 1e-12` against a base R solution iterated to 1e-12.

catR's `thetaEst` finds ML, MAP and WLE roots with `uniroot`'s default tolerance, about 1.2e-4; this library's are
good to 1e-12. catR's EAP uses the trapezoid rule on the 61 points of `posteriorMeanEstimate`, which halves the two
end weights; the difference grows when the posterior reaches the ends, after responses all in the lowest or
highest categories. Item choices match exactly at every step.

The fit statistics, the Mantel-Haenszel and Mantel statistics and equating are closed forms of the same inputs. glm
stops its iterations when the deviance changes by less than 1e-8 relative, the Newton steps of `logisticRegressionDif`
when no coefficient changes by more than 1e-10, so the logistic deviances differ near 1e-10. The ordinal logistic
regressions take the same Newton steps and are compared with lrm run to `eps = 1e-14`. Lord's chi-square compares
two calibrations, each with the parameter and SE differences above. equateIRT and the base R reference minimize
the Haebara and Stocking-Lord criteria with `nlminb`, which stops near 1e-8 from the minimum; the BFGS of the
linking functions stops at a gradient of 1e-10, or where rounding of the criterion leaves no step that makes
progress, within about 1e-8 of the minimum.

Fixtures store 17 significant digits, so they reproduce R's doubles exactly.

## Regenerating fixtures

[validation/README.md](../../validation/README.md) lists the scripts and the R packages they need. Update the versions
above when they change. A new statistic needs a reference in an R script, a fixture, a test that loads it, and a row in
the tables above.
