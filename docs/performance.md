# Performance

## Ratios

<!-- charts -->

### Where measurekit is slower

```mermaid
%%{init: {"xyChart": {"height": 388, "xAxis": {"labelFontSize": 11}}}}%%
xychart-beta horizontal
  title "Tasks that take more than 1.5 times as long as the reference"
  x-axis ["rater.manyFacetRasch", "md.estimateMean", "cdm.attributeProfiles", "md.calibrateExploratory", "numerics.cholesky", "cdm.calibrateGdina", "cdm.calibrateDino"]
  y-axis "Time of measurekit over the time of the reference" 0 --> 13
  bar [13, 13, 9.3, 5.3, 4.6, 2, 1.5]
  line [1, 1, 1, 1, 1, 1, 1]
```

A bar above the line takes longer than the reference, and md stands for multidimensional.

### Every task

```mermaid
%%{init: {"xyChart": {"height": 320, "xAxis": {"labelFontSize": 11}}}}%%
xychart-beta horizontal
  title "Core: numerics, random"
  x-axis ["normalQuantile", "logistic", "cholesky", "uniform", "normal"]
  y-axis "Time of measurekit over the time of the reference" 0 --> 5
  bar [1, 0.44, 4.6, 1.1, 0.87]
  line [1, 1, 1, 1, 1]
```

```mermaid
%%{init: {"xyChart": {"height": 524, "xAxis": {"labelFontSize": 11}}}}%%
xychart-beta horizontal
  title "Psychometrics: ctt, data, dif, dimensionality"
  x-axis ["itemStatistics", "cronbachAlpha", "mcdonaldOmega", "subscoreValue", "scoreResponses", "mantelHaenszel", "logisticRegression", "lordChiSquare", "ordinalLogistic", "polychoricCorrelations", "parallelAnalysis"]
  y-axis "Time of measurekit over the time of the reference" 0 --> 2
  bar [0.2, 0.033, 0.56, 0.016, 1.3, 0.052, 0.43, 0.55, 0.26, 0.0071, 0.0033]
  line [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

```mermaid
%%{init: {"xyChart": {"height": 728, "xAxis": {"labelFontSize": 11}}}}%%
xychart-beta horizontal
  title "Psychometrics: equating, explanatory, fit"
  x-axis ["linear", "equipercentile", "presmoothing", "bootstrapLinear", "tucker", "frequencyEstimation", "chainedEquipercentile", "stockingLordLinking", "haebaraLinking", "trueScore", "irtObservedScore", "itemFeatureRegression", "lltm", "itemFit", "personFit", "summedScoreFit", "yenQ3"]
  y-axis "Time of measurekit over the time of the reference" 0 --> 2
  bar [0.00081, 0.008, 0.029, 0.098, 0.0045, 0.0039, 0.015, 0.038, 0.029, 0.016, 0.12, 0.016, 0.66, 0.076, 0.21, 0.12, 0.042]
  line [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

```mermaid
%%{init: {"xyChart": {"height": 320, "xAxis": {"labelFontSize": 11}}}}%%
xychart-beta horizontal
  title "Psychometrics: generalizability, mokken"
  x-axis ["generalizabilityStudy", "scalability", "automatedItemSelection", "monotonicity", "invariantItemOrdering"]
  y-axis "Time of measurekit over the time of the reference" 0 --> 2
  bar [0.0027, 0.0091, 0.024, 0.16, 0.16]
  line [1, 1, 1, 1, 1]
```

```mermaid
%%{init: {"xyChart": {"height": 694, "xAxis": {"labelFontSize": 11}}}}%%
xychart-beta horizontal
  title "Psychometrics: responseTime, sim, irt"
  x-axis ["calibrateResponseTimes", "simulateResponses", "calibrate1pl", "calibrate2pl", "calibrate3pl", "calibrate2plCrossProduct", "calibrate2plHessian", "calibrateGraded", "calibrateGpcm", "calibrateTestlets", "abilityPosteriorMean", "abilityPosteriorMode", "abilityMaximumLikelihood", "abilityWeightedLikelihood", "jointMaximumLikelihood", "testInformation"]
  y-axis "Time of measurekit over the time of the reference" 0 --> 2
  bar [0.036, 0.35, 0.82, 0.63, 0.6, 0.33, 0.013, 0.27, 0.15, 0.0031, 0.7, 0.078, 0.11, 0.017, 0.47, 0.4]
  line [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

```mermaid
%%{init: {"xyChart": {"height": 592, "xAxis": {"labelFontSize": 11}}}}%%
xychart-beta horizontal
  title "Psychometrics: cat, cdm"
  x-axis ["simulateAdaptive", "simulateDiagnostic", "simulateMultidimensional", "calibrateGdina", "calibrateDina", "calibrateDino", "calibrateAcdm", "calibrateLlm", "calibrateRrum", "attributeProfiles", "accuracy", "validateQMatrix", "nonparametric"]
  y-axis "Time of measurekit over the time of the reference" 0 --> 10
  bar [0.021, 0.035, 0.013, 2, 1.2, 1.5, 0.39, 0.36, 0.62, 9.3, 1.2, 0.86, 0.06]
  line [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

```mermaid
%%{init: {"xyChart": {"height": 660, "xAxis": {"labelFontSize": 11}}}}%%
xychart-beta horizontal
  title "Psychometrics: rater, multidimensional"
  x-axis ["cohenKappa", "fleissKappa", "exactAgreement", "adjacentAgreement", "meanDifference", "intraclassCorrelation", "manyFacetRasch", "calibrateConfirmatory", "calibrateGraded", "calibrateExploratory", "calibrateMhrm", "estimateMode", "estimateMean", "rotateOblimin", "rotateVarimax"]
  y-axis "Time of measurekit over the time of the reference" 0 --> 13
  bar [0.19, 0.0079, 0.057, 0.062, 1.2, 0.0026, 13, 1.2, 0.48, 5.3, 0.93, 0.051, 13, 0.14, 0.22]
  line [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

```mermaid
%%{init: {"xyChart": {"height": 388, "xAxis": {"labelFontSize": 11}}}}%%
xychart-beta horizontal
  title "Knowledge tracing: bkt, elo, evaluate, irt, kst"
  x-axis ["fit", "replay", "replay", "auc", "replay", "blim", "iita"]
  y-axis "Time of measurekit over the time of the reference" 0 --> 2
  bar [0.012, 0.03, 0.52, 1.3, 0.22, 0.92, 0.000014]
  line [1, 1, 1, 1, 1, 1, 1]
```

```mermaid
%%{init: {"xyChart": {"height": 184, "xAxis": {"labelFontSize": 11}}}}%%
xychart-beta horizontal
  title "Knowledge tracing: logistic"
  x-axis ["afm"]
  y-axis "Time of measurekit over the time of the reference" 0 --> 2
  bar [0.036]
  line [1]
```

<!-- /charts -->

## Results

<!-- tables -->

Measured on Apple M5 with 10 cores, 16 GB of memory, Node.js 26.8.1 and R version 4.6.1 (2026-06-24).

### Core

| Task                       | Size                              | measurekit (ms) | Reference            | Reference (ms) |    Ratio |
| -------------------------- | --------------------------------- | --------------: | -------------------- | -------------: | -------: |
| Normal quantile function   | 1,000,000 probabilities           |            6.57 | stats 4.6.1 `qnorm`  |           6.50 |       1× |
| Logistic function          | 1,000,000 points                  |            2.74 | stats 4.6.1 `plogis` |           6.22 |    0.44× |
| **Cholesky factorization** | A correlation matrix of order 200 |           0.546 | base 4.6.1 `chol`    |          0.118 | **4.6×** |
| Uniform draws              | 1,000,000 draws                   |            3.66 | stats 4.6.1 `runif`  |           3.38 |     1.1× |
| Normal draws               | 1,000,000 draws                   |            11.5 | stats 4.6.1 `rnorm`  |           13.2 |    0.87× |

- Cholesky factorization: R's BLAS is OpenBLAS, whose kernels fuse multiplication and addition in vector registers; JavaScript has neither, and measurekit's loops are at the speed of scalar code.

### Psychometrics

| Task                                                                                   | Size                                                                          | measurekit (ms) | Reference                                                               | Reference (ms) |    Ratio |
| -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | --------------: | ----------------------------------------------------------------------- | -------------: | -------: |
| Item difficulty, discrimination and the statistics of the items                        | 3,000 persons, 50 items                                                       |            10.3 | psych 2.6.9 `alpha`                                                     |           52.0 |     0.2× |
| Cronbach's alpha                                                                       | 3,000 persons, 50 items                                                       |            1.82 | psych 2.6.9 `alpha`                                                     |           56.0 |   0.033× |
| McDonald's omega total of one factor                                                   | 3,000 persons, 20 items                                                       |            5.27 | psych 2.6.9 `omega(nfactors = 1)`                                       |           9.33 |    0.56× |
| The value of five subscores of ten items over the total score                          | 3,000 persons, 50 items                                                       |            2.15 | subscore 3.3 `CTTsub`                                                   |            135 |   0.016× |
| Scoring of the choices of persons against a key                                        | 2,000 persons, 50 items of 4 options                                          |            1.13 | base 4.6.1 `t(t(choices) == key)`                                       |          0.843 |     1.3× |
| The Mantel-Haenszel test of every item                                                 | 2,000 persons in two groups, 30 items                                         |            6.56 | difR 6.1.0 `difMH`                                                      |            127 |   0.052× |
| Logistic regression for uniform and nonuniform DIF of every item                       | 2,000 persons in two groups, 30 items                                         |            43.0 | difR 6.1.0 `difLogistic`                                                |            101 |    0.43× |
| Lord's chi-square of every item, with both groups calibrated by the 2PL                | 2,000 persons in two groups, 30 items                                         |             766 | difR 6.1.0 `difLord`                                                    |          1,392 |    0.55× |
| Ordinal logistic regression of every item                                              | 1,000 persons in two groups, 10 items of 4 categories                         |            33.8 | rms 8.2.0 `lrm, three models for an item`                               |            131 |    0.26× |
| Polychoric correlations of every pair of items                                         | 2,800 persons, 25 items of 6 categories                                       |            11.3 | psych 2.6.9 `polychoric`                                                |          1,599 |  0.0071× |
| Parallel analysis of components and factors from polychoric correlations, 20 data sets | 2,800 persons, 25 items of 6 categories                                       |             113 | psych 2.6.9 `fa.parallel(cor = "poly")`                                 |         34,406 |  0.0033× |
| Linear equating, random groups                                                         | 4,329 and 4,152 persons, 40 items                                             |           0.000 | equate 2.0.9 `equate(type = "linear")`                                  |          0.462 | 0.00081× |
| Equipercentile equating, random groups                                                 | 4,329 and 4,152 persons, 40 items                                             |           0.002 | equate 2.0.9 `equate(type = "equipercentile")`                          |          0.308 |   0.008× |
| Log-linear presmoothing of a score distribution, degree 3                              | 41 scores                                                                     |           0.022 | equate 2.0.9 `presmoothing(smoothmethod = "loglinear")`                 |          0.750 |   0.029× |
| Standard errors of the linear equating by 100 bootstrap replications                   | 4,329 and 4,152 persons, 40 items                                             |            25.6 | equate 2.0.9 `bootstrap`                                                |            262 |   0.098× |
| Tucker linear equating, nonequivalent groups                                           | 1,655 and 1,638 persons, 36 items, an anchor of 12                            |           0.006 | equate 2.0.9 `equate(method = "tucker")`                                |           1.40 |  0.0045× |
| Frequency estimation equipercentile equating                                           | 1,655 and 1,638 persons, 36 items, an anchor of 12                            |           0.009 | equate 2.0.9 `equate(method = "frequency estimation")`                  |           2.18 |  0.0039× |
| Chained equipercentile equating                                                        | 1,655 and 1,638 persons, 36 items, an anchor of 12                            |           0.011 | equate 2.0.9 `equate(method = "chained")`                               |          0.760 |   0.015× |
| Stocking-Lord linking of the common items                                              | Two forms of 3PL items that share 10                                          |           0.064 | equateIRT 2.5.2 `direc(method = "Stocking-Lord")`                       |           1.69 |   0.038× |
| Haebara linking of the common items                                                    | Two forms of 3PL items that share 10                                          |           0.058 | equateIRT 2.5.2 `direc(method = "Haebara")`                             |           2.00 |   0.029× |
| IRT true score equating of the linked forms                                            | Two forms of 3PL items that share 10                                          |           0.050 | equateIRT 2.5.2 `score(method = "TSE")`                                 |           3.18 |   0.016× |
| IRT observed score equating of the linked forms                                        | Two forms of 3PL items that share 10                                          |           0.295 | equateIRT 2.5.2 `score(method = "OSE")`                                 |           2.41 |    0.12× |
| Regression of item difficulties on three features by REML                              | 300 items, 3 features                                                         |           0.342 | metafor 5.2.1 `rma(method = "REML")`                                    |           22.0 |   0.016× |
| The linear logistic test model: the effects of three features on difficulty, by EM     | 1,000 persons, 20 items, 3 features                                           |             215 | mirt 1.48 `mirt(itemtype = "Rasch", itemdesign), Oakes standard errors` |            326 |    0.66× |
| Infit and outfit of the items                                                          | 2,000 persons, 40 items                                                       |            1.83 | mirt 1.48 `itemfit(fit_stats = "infit")`                                |           24.0 |   0.076× |
| Infit, outfit and the standardized log-likelihood of the persons                       | 2,000 persons, 40 items                                                       |            4.42 | mirt 1.48 `personfit`                                                   |           20.7 |    0.21× |
| The summed-score statistic S-X2 of the items                                           | 2,000 persons, 40 items                                                       |            5.55 | mirt 1.48 `itemfit(fit_stats = "S_X2")`                                 |           45.0 |    0.12× |
| Yen's Q3 of every pair of items                                                        | 2,000 persons, 40 items                                                       |            5.69 | mirt 1.48 `residuals(type = "Q3")`                                      |            134 |   0.042× |
| Variance components of a fully crossed design of three facets                          | 200 persons, 10 items, 4 raters: 8,000 scores                                 |            2.26 | lme4 2.1.0 `lmer, six random effects, REML`                             |            852 |  0.0027× |
| Scalability coefficients with standard errors                                          | 433 persons, 20 items of 5 categories                                         |            2.37 | mokken 3.1.2 `coefH(se = TRUE)`                                         |            261 |  0.0091× |
| Automated item selection into Mokken scales                                            | 433 persons, 20 items of 5 categories                                         |            1.01 | mokken 3.1.2 `aisp`                                                     |           42.0 |   0.024× |
| The check of monotonicity of every item                                                | 433 persons, 20 items of 5 categories                                         |            2.25 | mokken 3.1.2 `check.monotonicity`                                       |           14.0 |    0.16× |
| The check of invariant item ordering, with selection of items                          | 433 persons, 20 items of 5 categories                                         |            14.0 | mokken 3.1.2 `check.iio(method = "MIIO")`                               |           90.0 |    0.16× |
| The lognormal model of the response times                                              | 1,000 persons, 20 items                                                       |            6.88 | nlme 3.1.169 `lme(logtime ~ 0 + item, random = ~ 1                      |       person)` |      192 |
| The joint model of speed and accuracy, with the 2PL                                    | 1,000 persons, 20 items                                                       |             600 | none                                                                    |                |          |
| Speed of each person given the times of the items                                      | 1,000 persons, 20 items                                                       |            1.10 | none                                                                    |                |          |
| A conversion table of the posterior mean of theta given each summed score              | 40 items                                                                      |           0.344 | none                                                                    |                |          |
| A conversion table of the theta of each summed true score                              | 40 items                                                                      |           0.581 | none                                                                    |                |          |
| The cut score of a bookmark in an ordered item booklet                                 | 40 items                                                                      |           0.032 | none                                                                    |                |          |
| Accuracy and consistency of the classification by two cuts                             | 2,000 persons                                                                 |           0.359 | none                                                                    |                |          |
| Conversion of 2PL items to mirt's slope-intercept form                                 | 10,000 items                                                                  |            1.25 | none                                                                    |                |          |
| Conversion of 2PL items to mirt's IRT form                                             | 10,000 items                                                                  |            1.30 | none                                                                    |                |          |
| Simulation of the abilities of persons and their responses to 2PL items                | 2,000 persons, 40 items                                                       |            2.64 | mirt 1.48 `simdata(itemtype = "dich")`                                  |           7.50 |    0.35× |
| Calibration of the one-parameter model by EM                                           | 2,000 persons, 40 dichotomous items                                           |            86.4 | mirt 1.48 `mirt(itemtype = "Rasch")`                                    |            106 |    0.82× |
| Calibration of the two-parameter model by EM                                           | 2,000 persons, 40 dichotomous items                                           |             120 | mirt 1.48 `mirt(itemtype = "2PL")`                                      |            190 |    0.63× |
| Calibration of the three-parameter model by EM                                         | 2,000 persons, 40 dichotomous items                                           |             705 | mirt 1.48 `mirt(itemtype = "3PL")`                                      |          1,168 |     0.6× |
| The two-parameter model with cross-product standard errors                             | 2,000 persons, 40 dichotomous items                                           |             131 | mirt 1.48 `mirt(SE.type = "crossprod")`                                 |            395 |    0.33× |
| The two-parameter model with standard errors from the Hessian                          | 500 persons, 20 items                                                         |             102 | mirt 1.48 `mirt(SE.type = "Richardson")`                                |          7,932 |   0.013× |
| Calibration of the graded response model by EM                                         | 1,500 persons, 20 items of 5 categories                                       |            53.9 | mirt 1.48 `mirt(itemtype = "graded")`                                   |            199 |    0.27× |
| Calibration of the generalized partial credit model by EM                              | 1,500 persons, 20 items of 5 categories                                       |            87.4 | mirt 1.48 `mirt(itemtype = "gpcm")`                                     |            580 |    0.15× |
| Calibration of the testlet model by EM                                                 | 1,000 persons, 6 testlets of 5 items                                          |            67.7 | mirt 1.48 `bfactor`                                                     |         21,712 |  0.0031× |
| Ability by the posterior mean                                                          | 2,000 persons, the 40 items of the calibration                                |            24.4 | mirt 1.48 `fscores(method = "EAP")`                                     |           35.0 |     0.7× |
| Ability by the posterior mode                                                          | 2,000 persons, the 40 items of the calibration                                |            79.9 | mirt 1.48 `fscores(method = "MAP")`                                     |          1,025 |   0.078× |
| Ability by maximum likelihood                                                          | 2,000 persons, the 40 items of the calibration                                |            78.5 | mirt 1.48 `fscores(method = "ML")`                                      |            685 |    0.11× |
| Ability by weighted likelihood                                                         | 2,000 persons, the 40 items of the calibration                                |             155 | mirt 1.48 `fscores(method = "WLE")`                                     |          9,318 |   0.017× |
| Joint maximum likelihood calibration of the Rasch model                                | 2,000 persons, 40 items                                                       |            11.3 | TAM 4.3.25 `tam.jml`                                                    |           24.0 |    0.47× |
| The posterior mean of the ability given each summed score                              | 40 items                                                                      |           0.201 | none                                                                    |                |          |
| Information of a test at 121 values of theta                                           | 40 items                                                                      |           0.659 | mirt 1.48 `testinfo`                                                    |           1.64 |     0.4× |
| Simulation of adaptive tests by maximum information and the posterior mean             | 200 persons, a pool of 500 3PL items, 30 items each                           |             223 | catR 3.17 `randomCAT`                                                   |         10,498 |   0.021× |
| Simulation of adaptive tests that select testlets as units, by their mean information  | 200 persons, a pool of 500 3PL items with 400 in testlets of 4, 30 items each |             279 | none                                                                    |                |          |
| Simulation of diagnostic adaptive tests by the G-DINA discrimination index             | 200 persons, a pool of 20 items over 4 attributes, 10 items each              |            13.9 | cdcatR 1.0.7 `cdcat`                                                    |            394 |   0.035× |
| Simulation of multidimensional adaptive tests by the determinant of the information    | 100 persons, a pool of 200 two-dimensional items, 20 items each               |            75.2 | mirtCAT 1.14 `mirtCAT(criteria = "Drule")`                              |          5,769 |   0.013× |
| **Calibration of the G-DINA model by EM**                                              | 1,000 persons, 20 items, 4 attributes                                         |             222 | GDINA 2.13.2 `GDINA(model = "GDINA")`                                   |            111 |   **2×** |
| Calibration of the DINA model by EM                                                    | 1,000 persons, 20 items, 4 attributes                                         |            51.9 | GDINA 2.13.2 `GDINA(model = "DINA")`                                    |           43.0 |     1.2× |
| **Calibration of the DINO model by EM**                                                | 1,000 persons, 20 items, 4 attributes                                         |             221 | GDINA 2.13.2 `GDINA(model = "DINO")`                                    |            146 | **1.5×** |
| Calibration of the A-CDM model by EM                                                   | 1,000 persons, 20 items, 4 attributes                                         |            52.9 | GDINA 2.13.2 `GDINA(model = "ACDM")`                                    |            134 |    0.39× |
| Calibration of the LLM model by EM                                                     | 1,000 persons, 20 items, 4 attributes                                         |            67.1 | GDINA 2.13.2 `GDINA(model = "LLM")`                                     |            188 |    0.36× |
| Calibration of the R-RUM model by EM                                                   | 1,000 persons, 20 items, 4 attributes                                         |            68.8 | GDINA 2.13.2 `GDINA(model = "RRUM")`                                    |            111 |    0.62× |
| **Posterior profiles of the persons of a calibration**                                 | 1,000 persons, 20 items, 4 attributes                                         |            1.48 | GDINA 2.13.2 `personparm(what = "EAP")`                                 |          0.160 | **9.3×** |
| Accuracy and consistency of the classification                                         | 1,000 persons, 20 items, 4 attributes                                         |            1.58 | GDINA 2.13.2 `CA`                                                       |           1.33 |     1.2× |
| Validation of the Q-matrix by the proportion of variance accounted for                 | 1,000 persons, 20 items, 4 attributes                                         |            3.04 | GDINA 2.13.2 `Qval(method = "PVAF")`                                    |           3.56 |    0.86× |
| Nonparametric classification by the Hamming distance                                   | 1,000 persons, 20 items, 4 attributes                                         |            1.44 | NPCD 1.0.11 `AlphaNP`                                                   |           24.0 |    0.06× |
| Cohen's kappa of two raters                                                            | 2,000 subjects, 5 categories                                                  |           0.054 | irr 0.85 `kappa2`                                                       |          0.280 |    0.19× |
| Fleiss' kappa of five raters                                                           | 500 subjects, 5 raters, 5 categories                                          |           0.071 | irr 0.85 `kappam.fleiss`                                                |           9.00 |  0.0079× |
| The share of subjects on whom five raters agree exactly                                | 500 subjects, 5 raters, 5 categories                                          |           0.023 | irr 0.85 `agree(tolerance = 0)`                                         |          0.412 |   0.057× |
| The share of subjects whose ratings are within one category                            | 500 subjects, 5 raters, 5 categories                                          |           0.025 | irr 0.85 `agree(tolerance = 1)`                                         |          0.400 |   0.062× |
| The standardized mean difference of two raters                                         | 2,000 subjects, 2 raters                                                      |           0.216 | effsize 0.8.1 `cohen.d`                                                 |          0.180 |     1.2× |
| The six intraclass correlations of four raters                                         | 200 subjects, 4 raters                                                        |           0.017 | psych 2.6.9 `ICC`                                                       |           6.60 |  0.0026× |
| **The many-facet Rasch model of items and raters, with standard errors**               | 200 persons, 10 items, 4 raters: 8,000 scores                                 |             733 | TAM 4.3.25 `tam.mml.mfr(formulaA = ~ item + rater + step)`              |           56.0 |  **13×** |
| Calibration of a confirmatory two-dimensional 2PL by EM                                | 1,000 persons, 24 items, 2 correlated dimensions                              |           1,087 | mirt 1.48 `mirt(model, quadpts = 31)`                                   |            927 |     1.2× |
| Calibration of a confirmatory two-dimensional graded model by EM                       | 1,000 persons, 20 items of 4 categories, 2 correlated dimensions              |           1,115 | mirt 1.48 `mirt(itemtype = "graded", quadpts = 31)`                     |          2,311 |    0.48× |
| **Calibration of an exploratory two-dimensional 2PL by EM**                            | 1,000 persons, 20 items, 2 dimensions                                         |          17,392 | mirt 1.48 `mirt(data, 2, quadpts = 31)`                                 |          3,303 | **5.3×** |
| Calibration of a five-dimensional graded model by Metropolis-Hastings Robbins-Monro    | 800 persons, 25 items of 6 categories, 5 correlated dimensions                |          15,115 | mirt 1.48 `mirt(method = "MHRM")`                                       |         16,308 |    0.93× |
| Abilities by the posterior mode, in two dimensions                                     | 1,000 persons, 24 items                                                       |            45.4 | mirt 1.48 `fscores(method = "MAP")`                                     |            893 |   0.051× |
| **Abilities by the posterior mean, in two dimensions**                                 | 1,000 persons, 24 items                                                       |             694 | mirt 1.48 `fscores(method = "EAP")`                                     |           54.0 |  **13×** |
| Oblimin rotation of the loadings of a calibration                                      | 20 items, 2 dimensions                                                        |           0.033 | GPArotation 2026.8.2 `oblimin`                                          |          0.240 |    0.14× |
| Varimax rotation of the loadings of a calibration                                      | 20 items, 2 dimensions                                                        |           0.025 | GPArotation 2026.8.2 `Varimax`                                          |          0.115 |    0.22× |

- The lognormal model of the response times: measurekit also estimates the item discriminations.
- Simulation of adaptive tests that select testlets as units, by their mean information: catR has no testlets, so no reference does this task.
- Calibration of an exploratory two-dimensional 2PL by EM: mirt stops at its limit of 500 cycles when it has not converged.
- Calibration of a five-dimensional graded model by Metropolis-Hastings Robbins-Monro: both draw the abilities, so the time depends on the seed; mirt runs with one.

### Knowledge tracing

| Task                                                                                      | Size                                            | measurekit (ms) | Reference                               | Reference (ms) |     Ratio |
| ----------------------------------------------------------------------------------------- | ----------------------------------------------- | --------------: | --------------------------------------- | -------------: | --------: |
| Fit of the skills of a log by EM from five starts                                         | 20,000 attempts of 200 learners at 5 skills     |             165 | pyBKT 1.4.3 `Model.fit`                 |         13,502 |    0.012× |
| Replay of the log through the fitted model, a prediction and an update for each attempt   | 20,000 attempts of 200 learners at 5 skills     |            11.0 | pyBKT 1.4.3 `Model.predict`             |            363 |     0.03× |
| Replay of a log through the learner-item Elo model                                        | 100,000 attempts of 2,000 learners at 400 items |            18.4 | base R, a loop over the attempts        |           35.5 |     0.52× |
| The area under the ROC curve                                                              | 100,000 predictions                             |            23.8 | pROC 1.19.1 `auc(roc(direction = "<"))` |           19.0 |      1.3× |
| The AUC, RMSE, log-loss and accuracy of the predictions                                   | 100,000 predictions                             |            30.8 | none                                    |                |           |
| Replay of the responses of learners, the grid posterior of the ability updated after each | 2,000 learners, 40 responses each               |             451 | mirt 1.48 `fscores(method = "EAP")`     |          2,072 |     0.22× |
| Fit of the basic local independence model of a chain of 12 items by EM                    | 1,000 persons, 12 items, 13 states              |            26.3 | pks 0.8.0 `blim`                        |           28.5 |     0.92× |
| Inductive item tree analysis of the responses                                             | 1,000 persons, 12 items                         |            1.36 | DAKS 2.1.3 `iita(v = 1)`                |         97,144 | 0.000014× |
| Fit of the additive factors model by maximum likelihood                                   | 20,000 attempts of 200 learners at 5 skills     |            25.1 | base R `glm`, binomial family           |            689 |    0.036× |
| The candidates ranked by their distance to the band of success of a learner               | 500 candidate items, an Elo model               |           0.053 | none                                    |                |           |
| A log built from and checked for events                                                   | 100,000 events                                  |            17.0 | none                                    |                |           |

- Replay of the responses of learners, the grid posterior of the ability updated after each: mirt scores each of the 40 prefixes of the responses from the first; measurekit updates the posterior by the one response and predicts it. A scoring of the final patterns alone takes a fortieth of the time.
- Fit of the additive factors model by maximum likelihood: the proficiency of a learner is a fixed effect, as measurekit's default is none.

<!-- /tables -->

## Iterations

The iterations that each EM calibration of the tables needed to reach its tolerance, on the data of the benchmark. A ratio
of times is the product of a ratio of iterations and a ratio of the time of one. The one-parameter to the generalized partial
credit calibrations and the multidimensional ones of measurekit extrapolate EM with SQUAREM, as mirt does with Ramsay's
method; the others run plain EM. Every EM iteration counts, those that follow an extrapolation included.

| Calibration                         | measurekit | Reference |
| ----------------------------------- | ---------: | --------: |
| One-parameter model                 |         10 |         9 |
| Two-parameter model                 |         13 |        19 |
| Three-parameter model               |         81 |       183 |
| Graded response model               |         12 |        11 |
| Generalized partial credit model    |         19 |        37 |
| Confirmatory two-dimensional 2PL    |         13 |        24 |
| Confirmatory two-dimensional graded |         12 |        43 |
| Exploratory two-dimensional 2PL     |         93 |       135 |
| G-DINA                              |        187 |       184 |
| DINA                                |         44 |        45 |
| DINO                                |        195 |       231 |
| A-CDM                               |         40 |        33 |
| LLM                                 |         52 |        61 |
| R-RUM                               |         53 |        27 |

## Method

```mermaid
flowchart LR
  S["validation/benchmarks.R: simulate the inputs"] --> F[("bench-module.json fixtures")]
  F --> M["measurekit: test/module.bench.ts, Vitest"]
  F --> R["References: R packages, one thread"]
  F --> P["pyBKT: validation/benchmarks.py"]
  M --> T["Tables and charts of this page"]
  R --> T
  P --> T
  M --> B["Budgets: the fastest sample, in CI"]
```

Each benchmark times the hot path of a module, and the same task on the same data in the reference implementation that the
[validation](psychometrics/validation.md) uses: an R package, base R where no package does the task, or pyBKT. The ratio is
the time of measurekit over the time of the reference, so a ratio above 1 means that measurekit is slower. A ratio above
1.5 is in bold in the tables.

| Part       | How                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Data       | `validation/benchmarks.R` simulates the inputs with a fixed seed, or takes a public dataset, and writes them to `packages/<package>/test/fixtures/bench-<module>.json`. measurekit and its reference read the same data.                                                                                                                                                                                                                                                                                       |
| measurekit | `packages/<package>/test/<module>.bench.ts`, run by Vitest on Node.js with one thread, against the build of the package, which `pnpm bench` makes first: Vitest's module runner turns each import of a source file into a getter that the compiler cannot inline, so that it times a call of a small function several times as slow as it is for a user. A benchmark takes at least 10 samples, or 5 when it takes a second or more, after a warm-up. The tables give the median of the medians of three runs. |
| References | `validation/benchmarks/<package>/<module>.R`, run with one thread: `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` set R's BLAS. A task takes 15 samples under a second, 7 under 15 seconds, 3 under a minute and 2 above, after one call to warm up.                                                                                                                                                                                                                                                         |
| pyBKT      | `validation/benchmarks.py`, on the log of `bench-bkt.json`, with its processes off. It times what R does not.                                                                                                                                                                                                                                                                                                                                                                                                  |
| Settings   | A reference runs with the settings that make its work that of measurekit: the same quadrature, convergence tolerance and starts. The notes under a table name where they differ.                                                                                                                                                                                                                                                                                                                               |
| Ratio      | The median of measurekit over the median of the reference.                                                                                                                                                                                                                                                                                                                                                                                                                                                     |

An EM calibration stops when its estimates stop changing, so two programs that agree on the criterion can need different
numbers of iterations: the ratio is that of the time to the same answer, not of one iteration. The ability estimators and the
replays compare the work for each person, and the notes say where the reference does less.

## Running

```bash
pnpm install
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Rscript validation/benchmarks.R               # the inputs and the references
validation/knowledge-tracing/.venv/bin/python validation/benchmarks.py                  # pyBKT
pnpm bench                                                                              # measurekit, against the budgets
```

`Rscript validation/benchmarks.R irt mokken` runs the modules named, and `knowledge-tracing/irt` one module of a package.
To refresh the charts and tables, run each package with the JSON reporter three times, with the run number in the name of the
file, then `Rscript validation/benchmarks.R table`:

```bash
pnpm --filter @measurekit/core bench --reporter json --outputFile validation/benchmarks/results/measurekit-core-1.json
pnpm --filter @measurekit/psychometrics bench --reporter json --outputFile validation/benchmarks/results/measurekit-psychometrics-1.json
pnpm --filter @measurekit/knowledge-tracing bench --reporter json --outputFile validation/benchmarks/results/measurekit-knowledge-tracing-1.json
Rscript validation/benchmarks.R table
```

## Budgets

A benchmark fails when its fastest sample is above its budget, which is in `packages/<package>/test/budgets.ts`. The fastest
sample is the least affected by the other work of a shared machine. The budget of a time is five times the median of the
fastest samples of three runs, rounded up to one significant digit. The factor allows for CI hardware two to three times
slower than the laptop, and for noise; it is far below the cost of the changes that the budgets exist to catch. The `bench`
job of the CI workflow runs the benchmarks on every change. A budget changes only with the reason in the commit that
changes it.
