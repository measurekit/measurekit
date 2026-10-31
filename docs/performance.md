# Performance

## Ratios

<!-- charts -->

### Where measurekit is slower

```mermaid
%%{init: {"xyChart": {"height": 252, "xAxis": {"labelFontSize": 11}}}}%%
xychart-beta horizontal
  title "Tasks that take more than 1.5 times as long as the reference"
  x-axis ["cdm.attributeProfiles", "numerics.cholesky", "data.scoreResponses"]
  y-axis "Time of measurekit over the time of the reference" 0 --> 4
  bar [4, 4, 1.5]
  line [1, 1, 1]
```

A bar above the line takes longer than the reference, and md stands for multidimensional.

### Every task

```mermaid
%%{init: {"xyChart": {"height": 320, "xAxis": {"labelFontSize": 11}}}}%%
xychart-beta horizontal
  title "Core: numerics, random"
  x-axis ["normalQuantile", "logistic", "cholesky", "uniform", "normal"]
  y-axis "Time of measurekit over the time of the reference" 0 --> 4
  bar [0.83, 0.52, 4, 1.1, 0.94]
  line [1, 1, 1, 1, 1]
```

```mermaid
%%{init: {"xyChart": {"height": 762, "xAxis": {"labelFontSize": 11}}}}%%
xychart-beta horizontal
  title "Psychometrics: cat, cdm, ctt, data"
  x-axis ["simulateAdaptive", "simulateDiagnostic", "simulateMultidimensional", "calibrateGdina", "calibrateDina", "calibrateDino", "calibrateAcdm", "calibrateLlm", "calibrateRrum", "attributeProfiles", "accuracy", "validateQMatrix", "nonparametric", "itemStatistics", "cronbachAlpha", "mcdonaldOmega", "subscoreValue", "scoreResponses"]
  y-axis "Time of measurekit over the time of the reference" 0 --> 4
  bar [0.021, 0.037, 0.013, 0.85, 0.5, 0.62, 0.19, 0.17, 0.31, 4, 0.66, 1, 0.064, 0.2, 0.037, 0.62, 0.02, 1.5]
  line [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

```mermaid
%%{init: {"xyChart": {"height": 728, "xAxis": {"labelFontSize": 11}}}}%%
xychart-beta horizontal
  title "Psychometrics: dif, dimensionality, equating"
  x-axis ["mantelHaenszel", "logisticRegression", "lordChiSquare", "ordinalLogistic", "polychoricCorrelations", "parallelAnalysis", "linear", "equipercentile", "presmoothing", "bootstrapLinear", "tucker", "frequencyEstimation", "chainedEquipercentile", "stockingLordLinking", "haebaraLinking", "trueScore", "irtObservedScore"]
  y-axis "Time of measurekit over the time of the reference" 0 --> 2
  bar [0.076, 0.51, 0.76, 0.37, 0.0065, 0.0031, 0.00081, 0.0093, 0.035, 0.1, 0.0047, 0.0041, 0.015, 0.041, 0.037, 0.019, 0.14]
  line [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

```mermaid
%%{init: {"xyChart": {"height": 388, "xAxis": {"labelFontSize": 11}}}}%%
xychart-beta horizontal
  title "Psychometrics: explanatory, fit, generalizability"
  x-axis ["itemFeatureRegression", "lltm", "itemFit", "personFit", "summedScoreFit", "yenQ3", "generalizabilityStudy"]
  y-axis "Time of measurekit over the time of the reference" 0 --> 2
  bar [0.019, 0.29, 0.087, 0.23, 0.14, 0.047, 0.0028]
  line [1, 1, 1, 1, 1, 1, 1]
```

```mermaid
%%{init: {"xyChart": {"height": 592, "xAxis": {"labelFontSize": 11}}}}%%
xychart-beta horizontal
  title "Psychometrics: mokken, rater, responseTime, sim"
  x-axis ["scalability", "automatedItemSelection", "monotonicity", "invariantItemOrdering", "cohenKappa", "fleissKappa", "exactAgreement", "adjacentAgreement", "meanDifference", "intraclassCorrelation", "manyFacetRasch", "calibrateResponseTimes", "simulateResponses"]
  y-axis "Time of measurekit over the time of the reference" 0 --> 2
  bar [0.011, 0.032, 0.18, 0.19, 0.2, 0.008, 0.061, 0.061, 1.2, 0.0028, 1.2, 0.04, 0.38]
  line [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

```mermaid
%%{init: {"xyChart": {"height": 626, "xAxis": {"labelFontSize": 11}}}}%%
xychart-beta horizontal
  title "Psychometrics: irt"
  x-axis ["calibrate1pl", "calibrate2pl", "calibrate3pl", "calibrate2plCrossProduct", "calibrate2plHessian", "calibrateGraded", "calibrateGpcm", "calibrateTestlets", "abilityPosteriorMean", "abilityPosteriorMode", "abilityMaximumLikelihood", "abilityWeightedLikelihood", "jointMaximumLikelihood", "testInformation"]
  y-axis "Time of measurekit over the time of the reference" 0 --> 2
  bar [0.87, 0.59, 0.58, 0.33, 0.016, 0.31, 0.25, 0.0036, 0.71, 0.074, 0.12, 0.016, 0.41, 0.46]
  line [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

```mermaid
%%{init: {"xyChart": {"height": 422, "xAxis": {"labelFontSize": 11}}}}%%
xychart-beta horizontal
  title "Psychometrics: multidimensional"
  x-axis ["calibrateConfirmatory", "calibrateGraded", "calibrateExploratory", "calibrateMhrm", "estimateMode", "estimateMean", "rotateOblimin", "rotateVarimax"]
  y-axis "Time of measurekit over the time of the reference" 0 --> 2
  bar [0.34, 0.17, 0.64, 0.73, 0.047, 0.76, 0.14, 0.24]
  line [1, 1, 1, 1, 1, 1, 1, 1]
```

```mermaid
%%{init: {"xyChart": {"height": 388, "xAxis": {"labelFontSize": 11}}}}%%
xychart-beta horizontal
  title "Knowledge tracing: bkt, elo, evaluate, irt, kst"
  x-axis ["fit", "replay", "replay", "auc", "replay", "blim", "iita"]
  y-axis "Time of measurekit over the time of the reference" 0 --> 2
  bar [0.012, 0.034, 0.58, 1.3, 0.19, 0.97, 0.000015]
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

| Task                       | Size                              | measurekit (ms) | Reference            | Reference (ms) |  Ratio |
| -------------------------- | --------------------------------- | --------------: | -------------------- | -------------: | -----: |
| Normal quantile function   | 1,000,000 probabilities           |            4.25 | stats 4.6.1 `qnorm`  |           5.12 |  0.83× |
| Logistic function          | 1,000,000 points                  |            2.52 | stats 4.6.1 `plogis` |           4.80 |  0.52× |
| **Cholesky factorization** | A correlation matrix of order 200 |           0.465 | base 4.6.1 `chol`    |          0.118 | **4×** |
| Uniform draws              | 1,000,000 draws                   |            3.18 | stats 4.6.1 `runif`  |           3.00 |   1.1× |
| Normal draws               | 1,000,000 draws                   |            11.1 | stats 4.6.1 `rnorm`  |           11.8 |  0.94× |

- Cholesky factorization: R's BLAS is OpenBLAS, whose kernels fuse multiplication and addition in vector registers; JavaScript has neither, and measurekit's loops are at the speed of scalar code.

### Psychometrics

| Task                                                                                   | Size                                                                          | measurekit (ms) | Reference                                                               | Reference (ms) |    Ratio |
| -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | --------------: | ----------------------------------------------------------------------- | -------------: | -------: |
| Simulation of adaptive tests by maximum information and the posterior mean             | 200 persons, a pool of 500 3PL items, 30 items each                           |             196 | catR 3.17 `randomCAT`                                                   |          9,499 |   0.021× |
| Simulation of adaptive tests that select testlets as units, by their mean information  | 200 persons, a pool of 500 3PL items with 400 in testlets of 4, 30 items each |             202 | none                                                                    |                |          |
| Simulation of diagnostic adaptive tests by the G-DINA discrimination index             | 200 persons, a pool of 20 items over 4 attributes, 10 items each              |            12.2 | cdcatR 1.0.7 `cdcat`                                                    |            332 |   0.037× |
| Simulation of multidimensional adaptive tests by the determinant of the information    | 100 persons, a pool of 200 two-dimensional items, 20 items each               |            57.6 | mirtCAT 1.14 `mirtCAT(criteria = "Drule")`                              |          4,271 |   0.013× |
| Calibration of the G-DINA model by EM                                                  | 1,000 persons, 20 items, 4 attributes                                         |            74.4 | GDINA 2.13.2 `GDINA(model = "GDINA")`                                   |           87.0 |    0.85× |
| Calibration of the DINA model by EM                                                    | 1,000 persons, 20 items, 4 attributes                                         |            17.9 | GDINA 2.13.2 `GDINA(model = "DINA")`                                    |           35.5 |     0.5× |
| Calibration of the DINO model by EM                                                    | 1,000 persons, 20 items, 4 attributes                                         |            77.4 | GDINA 2.13.2 `GDINA(model = "DINO")`                                    |            124 |    0.62× |
| Calibration of the A-CDM model by EM                                                   | 1,000 persons, 20 items, 4 attributes                                         |            22.3 | GDINA 2.13.2 `GDINA(model = "ACDM")`                                    |            116 |    0.19× |
| Calibration of the LLM model by EM                                                     | 1,000 persons, 20 items, 4 attributes                                         |            28.4 | GDINA 2.13.2 `GDINA(model = "LLM")`                                     |            163 |    0.17× |
| Calibration of the R-RUM model by EM                                                   | 1,000 persons, 20 items, 4 attributes                                         |            29.6 | GDINA 2.13.2 `GDINA(model = "RRUM")`                                    |           97.0 |    0.31× |
| **Posterior profiles of the persons of a calibration**                                 | 1,000 persons, 20 items, 4 attributes                                         |           0.562 | GDINA 2.13.2 `personparm(what = "EAP")`                                 |          0.140 |   **4×** |
| Accuracy and consistency of the classification                                         | 1,000 persons, 20 items, 4 attributes                                         |           0.695 | GDINA 2.13.2 `CA`                                                       |           1.06 |    0.66× |
| Validation of the Q-matrix by the proportion of variance accounted for                 | 1,000 persons, 20 items, 4 attributes                                         |            2.79 | GDINA 2.13.2 `Qval(method = "PVAF")`                                    |           2.76 |       1× |
| Nonparametric classification by the Hamming distance                                   | 1,000 persons, 20 items, 4 attributes                                         |            1.30 | NPCD 1.0.11 `AlphaNP`                                                   |           20.3 |   0.064× |
| Item difficulty, discrimination and the statistics of the items                        | 3,000 persons, 50 items                                                       |            9.57 | psych 2.6.9 `alpha`                                                     |           47.0 |     0.2× |
| Cronbach's alpha                                                                       | 3,000 persons, 50 items                                                       |            1.76 | psych 2.6.9 `alpha`                                                     |           47.0 |   0.037× |
| McDonald's omega total of one factor                                                   | 3,000 persons, 20 items                                                       |            4.93 | psych 2.6.9 `omega(nfactors = 1)`                                       |           8.00 |    0.62× |
| The value of five subscores of ten items over the total score                          | 3,000 persons, 50 items                                                       |            2.07 | subscore 3.3 `CTTsub`                                                   |            106 |    0.02× |
| **Scoring of the choices of persons against a key**                                    | 2,000 persons, 50 items of 4 options                                          |            1.09 | base 4.6.1 `t(t(choices) == key)`                                       |          0.708 | **1.5×** |
| The Mantel-Haenszel test of every item                                                 | 2,000 persons in two groups, 30 items                                         |            6.34 | difR 6.1.0 `difMH`                                                      |           84.0 |   0.076× |
| Logistic regression for uniform and nonuniform DIF of every item                       | 2,000 persons in two groups, 30 items                                         |            42.7 | difR 6.1.0 `difLogistic`                                                |           84.0 |    0.51× |
| Lord's chi-square of every item, with both groups calibrated by the 2PL                | 2,000 persons in two groups, 30 items                                         |             772 | difR 6.1.0 `difLord`                                                    |          1,015 |    0.76× |
| Ordinal logistic regression of every item                                              | 1,000 persons in two groups, 10 items of 4 categories                         |            34.7 | rms 8.2.0 `lrm, three models for an item`                               |           93.0 |    0.37× |
| Polychoric correlations of every pair of items                                         | 2,800 persons, 25 items of 6 categories                                       |            9.71 | psych 2.6.9 `polychoric`                                                |          1,496 |  0.0065× |
| Parallel analysis of components and factors from polychoric correlations, 20 data sets | 2,800 persons, 25 items of 6 categories                                       |             101 | psych 2.6.9 `fa.parallel(cor = "poly")`                                 |         32,593 |  0.0031× |
| Linear equating, random groups                                                         | 4,329 and 4,152 persons, 40 items                                             |           0.000 | equate 2.0.9 `equate(type = "linear")`                                  |          0.412 | 0.00081× |
| Equipercentile equating, random groups                                                 | 4,329 and 4,152 persons, 40 items                                             |           0.002 | equate 2.0.9 `equate(type = "equipercentile")`                          |          0.260 |  0.0093× |
| Log-linear presmoothing of a score distribution, degree 3                              | 41 scores                                                                     |           0.022 | equate 2.0.9 `presmoothing(smoothmethod = "loglinear")`                 |          0.640 |   0.035× |
| Standard errors of the linear equating by 100 bootstrap replications                   | 4,329 and 4,152 persons, 40 items                                             |            24.6 | equate 2.0.9 `bootstrap`                                                |            242 |     0.1× |
| Tucker linear equating, nonequivalent groups                                           | 1,655 and 1,638 persons, 36 items, an anchor of 12                            |           0.006 | equate 2.0.9 `equate(method = "tucker")`                                |           1.28 |  0.0047× |
| Frequency estimation equipercentile equating                                           | 1,655 and 1,638 persons, 36 items, an anchor of 12                            |           0.008 | equate 2.0.9 `equate(method = "frequency estimation")`                  |           2.00 |  0.0041× |
| Chained equipercentile equating                                                        | 1,655 and 1,638 persons, 36 items, an anchor of 12                            |           0.011 | equate 2.0.9 `equate(method = "chained")`                               |          0.710 |   0.015× |
| Stocking-Lord linking of the common items                                              | Two forms of 3PL items that share 10                                          |           0.062 | equateIRT 2.5.2 `direc(method = "Stocking-Lord")`                       |           1.51 |   0.041× |
| Haebara linking of the common items                                                    | Two forms of 3PL items that share 10                                          |           0.058 | equateIRT 2.5.2 `direc(method = "Haebara")`                             |           1.58 |   0.037× |
| IRT true score equating of the linked forms                                            | Two forms of 3PL items that share 10                                          |           0.049 | equateIRT 2.5.2 `score(method = "TSE")`                                 |           2.62 |   0.019× |
| IRT observed score equating of the linked forms                                        | Two forms of 3PL items that share 10                                          |           0.289 | equateIRT 2.5.2 `score(method = "OSE")`                                 |           2.00 |    0.14× |
| Regression of item difficulties on three features by REML                              | 300 items, 3 features                                                         |           0.335 | metafor 5.2.1 `rma(method = "REML")`                                    |           18.0 |   0.019× |
| The linear logistic test model: the effects of three features on difficulty, by EM     | 1,000 persons, 20 items, 3 features                                           |            74.6 | mirt 1.48 `mirt(itemtype = "Rasch", itemdesign), Oakes standard errors` |            253 |    0.29× |
| Infit and outfit of the items                                                          | 2,000 persons, 40 items                                                       |            1.83 | mirt 1.48 `itemfit(fit_stats = "infit")`                                |           21.0 |   0.087× |
| Infit, outfit and the standardized log-likelihood of the persons                       | 2,000 persons, 40 items                                                       |            4.18 | mirt 1.48 `personfit`                                                   |           18.0 |    0.23× |
| The summed-score statistic S-X2 of the items                                           | 2,000 persons, 40 items                                                       |            5.35 | mirt 1.48 `itemfit(fit_stats = "S_X2")`                                 |           39.0 |    0.14× |
| Yen's Q3 of every pair of items                                                        | 2,000 persons, 40 items                                                       |            5.46 | mirt 1.48 `residuals(type = "Q3")`                                      |            115 |   0.047× |
| Variance components of a fully crossed design of three facets                          | 200 persons, 10 items, 4 raters: 8,000 scores                                 |            1.97 | lme4 2.1.0 `lmer, six random effects, REML`                             |            707 |  0.0028× |
| Scalability coefficients with standard errors                                          | 433 persons, 20 items of 5 categories                                         |            2.12 | mokken 3.1.2 `coefH(se = TRUE)`                                         |            196 |   0.011× |
| Automated item selection into Mokken scales                                            | 433 persons, 20 items of 5 categories                                         |           0.918 | mokken 3.1.2 `aisp`                                                     |           28.5 |   0.032× |
| The check of monotonicity of every item                                                | 433 persons, 20 items of 5 categories                                         |            1.89 | mokken 3.1.2 `check.monotonicity`                                       |           10.6 |    0.18× |
| The check of invariant item ordering, with selection of items                          | 433 persons, 20 items of 5 categories                                         |            12.7 | mokken 3.1.2 `check.iio(method = "MIIO")`                               |           66.0 |    0.19× |
| Cohen's kappa of two raters                                                            | 2,000 subjects, 5 categories                                                  |           0.050 | irr 0.85 `kappa2`                                                       |          0.255 |     0.2× |
| Fleiss' kappa of five raters                                                           | 500 subjects, 5 raters, 5 categories                                          |           0.065 | irr 0.85 `kappam.fleiss`                                                |           8.14 |   0.008× |
| The share of subjects on whom five raters agree exactly                                | 500 subjects, 5 raters, 5 categories                                          |           0.021 | irr 0.85 `agree(tolerance = 0)`                                         |          0.353 |   0.061× |
| The share of subjects whose ratings are within one category                            | 500 subjects, 5 raters, 5 categories                                          |           0.022 | irr 0.85 `agree(tolerance = 1)`                                         |          0.362 |   0.061× |
| The standardized mean difference of two raters                                         | 2,000 subjects, 2 raters                                                      |           0.195 | effsize 0.8.1 `cohen.d`                                                 |          0.168 |     1.2× |
| The six intraclass correlations of four raters                                         | 200 subjects, 4 raters                                                        |           0.016 | psych 2.6.9 `ICC`                                                       |           5.63 |  0.0028× |
| The many-facet Rasch model of items and raters, with standard errors                   | 200 persons, 10 items, 4 raters: 8,000 scores                                 |            58.9 | TAM 4.3.25 `tam.mml.mfr(formulaA = ~ item + rater + step)`              |           49.0 |     1.2× |
| The lognormal model of the response times                                              | 1,000 persons, 20 items                                                       |            6.40 | nlme 3.1.169 `lme(logtime ~ 0 + item, random = ~ 1                      |       person)` |      161 | 0.04× |
| The joint model of speed and accuracy, with the 2PL                                    | 1,000 persons, 20 items                                                       |             545 | none                                                                    |                |          |
| Speed of each person given the times of the items                                      | 1,000 persons, 20 items                                                       |           0.971 | none                                                                    |                |          |
| A conversion table of the posterior mean of theta given each summed score              | 40 items                                                                      |           0.305 | none                                                                    |                |          |
| A conversion table of the theta of each summed true score                              | 40 items                                                                      |           0.513 | none                                                                    |                |          |
| The cut score of a bookmark in an ordered item booklet                                 | 40 items                                                                      |           0.030 | none                                                                    |                |          |
| Accuracy and consistency of the classification by two cuts                             | 2,000 persons                                                                 |           0.345 | none                                                                    |                |          |
| Conversion of 2PL items to mirt's slope-intercept form                                 | 10,000 items                                                                  |            1.15 | none                                                                    |                |          |
| Conversion of 2PL items to mirt's IRT form                                             | 10,000 items                                                                  |            1.19 | none                                                                    |                |          |
| Simulation of the abilities of persons and their responses to 2PL items                | 2,000 persons, 40 items                                                       |            2.34 | mirt 1.48 `simdata(itemtype = "dich")`                                  |           6.22 |    0.38× |
| Calibration of the one-parameter model by EM                                           | 2,000 persons, 40 dichotomous items                                           |            80.9 | mirt 1.48 `mirt(itemtype = "Rasch")`                                    |           93.0 |    0.87× |
| Calibration of the two-parameter model by EM                                           | 2,000 persons, 40 dichotomous items                                           |             100 | mirt 1.48 `mirt(itemtype = "2PL")`                                      |            171 |    0.59× |
| Calibration of the three-parameter model by EM                                         | 2,000 persons, 40 dichotomous items                                           |             605 | mirt 1.48 `mirt(itemtype = "3PL")`                                      |          1,044 |    0.58× |
| The two-parameter model with cross-product standard errors                             | 2,000 persons, 40 dichotomous items                                           |             118 | mirt 1.48 `mirt(SE.type = "crossprod")`                                 |            360 |    0.33× |
| The two-parameter model with standard errors from the Hessian                          | 500 persons, 20 items                                                         |            94.8 | mirt 1.48 `mirt(SE.type = "Richardson")`                                |          5,972 |   0.016× |
| Calibration of the graded response model by EM                                         | 1,500 persons, 20 items of 5 categories                                       |            48.0 | mirt 1.48 `mirt(itemtype = "graded")`                                   |            153 |    0.31× |
| Calibration of the generalized partial credit model by EM                              | 1,500 persons, 20 items of 5 categories                                       |            80.5 | mirt 1.48 `mirt(itemtype = "gpcm")`                                     |            317 |    0.25× |
| Calibration of the testlet model by EM                                                 | 1,000 persons, 6 testlets of 5 items                                          |            57.2 | mirt 1.48 `bfactor`                                                     |         15,759 |  0.0036× |
| Ability by the posterior mean                                                          | 2,000 persons, the 40 items of the calibration                                |            21.7 | mirt 1.48 `fscores(method = "EAP")`                                     |           30.5 |    0.71× |
| Ability by the posterior mode                                                          | 2,000 persons, the 40 items of the calibration                                |            69.7 | mirt 1.48 `fscores(method = "MAP")`                                     |            943 |   0.074× |
| Ability by maximum likelihood                                                          | 2,000 persons, the 40 items of the calibration                                |            71.3 | mirt 1.48 `fscores(method = "ML")`                                      |            605 |    0.12× |
| Ability by weighted likelihood                                                         | 2,000 persons, the 40 items of the calibration                                |             132 | mirt 1.48 `fscores(method = "WLE")`                                     |          8,106 |   0.016× |
| Joint maximum likelihood calibration of the Rasch model                                | 2,000 persons, 40 items                                                       |            9.41 | TAM 4.3.25 `tam.jml`                                                    |           23.0 |    0.41× |
| The posterior mean of the ability given each summed score                              | 40 items                                                                      |           0.181 | none                                                                    |                |          |
| Information of a test at 121 values of theta                                           | 40 items                                                                      |           0.616 | mirt 1.48 `testinfo`                                                    |           1.35 |    0.46× |
| Calibration of a confirmatory two-dimensional 2PL by EM                                | 1,000 persons, 24 items, 2 correlated dimensions                              |             254 | mirt 1.48 `mirt(model, quadpts = 31)`                                   |            756 |    0.34× |
| Calibration of a confirmatory two-dimensional graded model by EM                       | 1,000 persons, 20 items of 4 categories, 2 correlated dimensions              |             310 | mirt 1.48 `mirt(itemtype = "graded", quadpts = 31)`                     |          1,781 |    0.17× |
| Calibration of an exploratory two-dimensional 2PL by EM                                | 1,000 persons, 20 items, 2 dimensions                                         |           1,816 | mirt 1.48 `mirt(data, 2, quadpts = 31)`                                 |          2,855 |    0.64× |
| Calibration of a five-dimensional graded model by Metropolis-Hastings Robbins-Monro    | 800 persons, 25 items of 6 categories, 5 correlated dimensions                |          11,308 | mirt 1.48 `mirt(method = "MHRM")`                                       |         15,542 |    0.73× |
| Abilities by the posterior mode, in two dimensions                                     | 1,000 persons, 24 items                                                       |            35.1 | mirt 1.48 `fscores(method = "MAP")`                                     |            753 |   0.047× |
| Abilities by the posterior mean, in two dimensions                                     | 1,000 persons, 24 items                                                       |            35.8 | mirt 1.48 `fscores(method = "EAP")`                                     |           47.0 |    0.76× |
| Oblimin rotation of the loadings of a calibration                                      | 20 items, 2 dimensions                                                        |           0.031 | GPArotation 2026.8.2 `oblimin`                                          |          0.226 |    0.14× |
| Varimax rotation of the loadings of a calibration                                      | 20 items, 2 dimensions                                                        |           0.024 | GPArotation 2026.8.2 `Varimax`                                          |          0.100 |    0.24× |

- Simulation of adaptive tests that select testlets as units, by their mean information: catR has no testlets, so no reference does this task.
- Posterior profiles of the persons of a calibration: personparm multiplies the log posterior that the fit stored by the profiles and returns the EAP classification; measurekit's calibration holds no posterior, so it computes each person's from the responses, and returns the posterior, the probability of each attribute and the MAP and EAP profiles.
- The lognormal model of the response times: measurekit also estimates the item discriminations.
- Calibration of an exploratory two-dimensional 2PL by EM: mirt stops at its limit of 500 cycles when it has not converged.
- Calibration of a five-dimensional graded model by Metropolis-Hastings Robbins-Monro: both draw the abilities, so the time depends on the seed; mirt runs with one.

### Knowledge tracing

| Task                                                                                      | Size                                            | measurekit (ms) | Reference                               | Reference (ms) |     Ratio |
| ----------------------------------------------------------------------------------------- | ----------------------------------------------- | --------------: | --------------------------------------- | -------------: | --------: |
| Fit of the skills of a log by EM from five starts                                         | 20,000 attempts of 200 learners at 5 skills     |             126 | pyBKT 1.4.3 `Model.fit`                 |         10,453 |    0.012× |
| Replay of the log through the fitted model, a prediction and an update for each attempt   | 20,000 attempts of 200 learners at 5 skills     |            9.71 | pyBKT 1.4.3 `Model.predict`             |            288 |    0.034× |
| Replay of a log through the learner-item Elo model                                        | 100,000 attempts of 2,000 learners at 400 items |            17.5 | base R, a loop over the attempts        |           30.5 |     0.58× |
| The area under the ROC curve                                                              | 100,000 predictions                             |            21.6 | pROC 1.19.1 `auc(roc(direction = "<"))` |           16.5 |      1.3× |
| The AUC, RMSE, log-loss and accuracy of the predictions                                   | 100,000 predictions                             |            25.8 | none                                    |                |           |
| Replay of the responses of learners, the grid posterior of the ability updated after each | 2,000 learners, 40 responses each               |             331 | mirt 1.48 `fscores(method = "EAP")`     |          1,735 |     0.19× |
| Fit of the basic local independence model of a chain of 12 items by EM                    | 1,000 persons, 12 items, 13 states              |            24.3 | pks 0.8.0 `blim`                        |           25.0 |     0.97× |
| Inductive item tree analysis of the responses                                             | 1,000 persons, 12 items                         |            1.10 | DAKS 2.1.3 `iita(v = 1)`                |         74,600 | 0.000015× |
| Fit of the additive factors model by maximum likelihood                                   | 20,000 attempts of 200 learners at 5 skills     |            16.9 | base R `glm`, binomial family           |            469 |    0.036× |
| The candidates ranked by their distance to the band of success of a learner               | 500 candidate items, an Elo model               |           0.047 | none                                    |                |           |
| A log built from and checked for events                                                   | 100,000 events                                  |            13.4 | none                                    |                |           |

- Replay of the responses of learners, the grid posterior of the ability updated after each: mirt scores each of the 40 prefixes of the responses from the first; measurekit updates the posterior by the one response and predicts it. A scoring of the final patterns alone takes a fortieth of the time.
- Fit of the additive factors model by maximum likelihood: the proficiency of a learner is a fixed effect, as measurekit's default is none.

<!-- /tables -->

## Iterations

The iterations that each EM calibration of the tables needed to reach its tolerance, on the data of the benchmark. A ratio
of times is the product of a ratio of iterations and a ratio of the time of one. The one-parameter to the generalized partial
credit calibrations, the multidimensional ones, the many-facet Rasch model and the linear logistic test model of
measurekit extrapolate EM with SQUAREM, as mirt does with Ramsay's method; the others run plain EM. Every EM iteration
counts, those that follow an extrapolation included. EM stops when an iteration changes no parameter by more than the
tolerance, which leaves the estimates some distance from the maximum, more where EM contracts slowly. Extrapolated, they
stop closer to it than plain EM does, by up to about 1e-3 at the default tolerance (the confirmatory 2PL is 1.3e-4 from
the maximum, was 4.8e-4); at a tolerance of 1e-8 both are at it.

| Calibration                         | measurekit | Reference |
| ----------------------------------- | ---------: | --------: |
| One-parameter model                 |         10 |         9 |
| Two-parameter model                 |         13 |        19 |
| Three-parameter model               |         81 |       183 |
| Graded response model               |         12 |        11 |
| Generalized partial credit model    |         19 |        37 |
| Confirmatory two-dimensional 2PL    |         12 |        24 |
| Confirmatory two-dimensional graded |         12 |        43 |
| Exploratory two-dimensional 2PL     |        100 |       135 |
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
