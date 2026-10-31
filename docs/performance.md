# Performance

## Ratios

<!-- charts -->

### Where measurekit is slower

```mermaid
%%{init: {"xyChart": {"height": 626, "xAxis": {"labelFontSize": 11}}}}%%
xychart-beta horizontal
  title "Tasks that take more than 1.5 times as long as the reference"
  x-axis ["irt.replay", "multi.estimateMean", "rater.manyFacetRasch", "cdm.attributeProfiles", "numerics.cholesky", "multi.calibrateExploratory", "data.scoreResponses", "irt.abilityPosteriorMean", "numerics.logistic", "numerics.normalQuantile", "irt.calibrate1pl", "cdm.calibrateGdina", "cdm.classificationAccuracy", "cdm.calibrateDino"]
  y-axis "Time of measurekit over the time of the reference" 0 --> 29
  bar [29, 24, 14, 9.8, 6.4, 5.3, 5.3, 5.2, 4.8, 4.2, 2.2, 2.1, 1.7, 1.6]
  line [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

### Every task

A bar above the line takes longer than the reference.

```mermaid
%%{init: {"xyChart": {"height": 320, "xAxis": {"labelFontSize": 11}}}}%%
xychart-beta horizontal
  title "Core: numerics, random"
  x-axis ["normalQuantile", "logistic", "cholesky", "uniform", "normal"]
  y-axis "Time of measurekit over the time of the reference" 0 --> 7
  bar [4.2, 4.8, 6.4, 1, 1.3]
  line [1, 1, 1, 1, 1]
```

```mermaid
%%{init: {"xyChart": {"height": 626, "xAxis": {"labelFontSize": 11}}}}%%
xychart-beta horizontal
  title "Psychometrics: irt"
  x-axis ["calibrate1pl", "calibrate2pl", "calibrate3pl", "calibrate2plCrossProduct", "calibrate2plHessian", "calibrateGraded", "calibrateGpcm", "calibrateTestlets", "abilityPosteriorMean", "abilityPosteriorMode", "abilityMaximumLikelihood", "abilityWeightedLikelihood", "jointMaximumLikelihood", "testInformation"]
  y-axis "Time of measurekit over the time of the reference" 0 --> 6
  bar [2.2, 1.4, 0.68, 0.42, 0.015, 0.45, 0.47, 0.013, 5.2, 0.2, 0.31, 0.045, 0.2, 0.23]
  line [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

```mermaid
%%{init: {"xyChart": {"height": 490, "xAxis": {"labelFontSize": 11}}}}%%
xychart-beta horizontal
  title "Psychometrics: multidimensional, cat"
  x-axis ["calibrateConfirmatory", "calibrateGraded", "calibrateExploratory", "estimateMode", "estimateMean", "rotateOblimin", "rotateVarimax", "simulateAdaptive", "simulateDiagnostic", "simulateMultidimensional"]
  y-axis "Time of measurekit over the time of the reference" 0 --> 24
  bar [0.68, 0.58, 5.3, 0.097, 24, 0.14, 0.25, 0.043, 0.028, 0.029]
  line [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

```mermaid
%%{init: {"xyChart": {"height": 660, "xAxis": {"labelFontSize": 11}}}}%%
xychart-beta horizontal
  title "Psychometrics: cdm, ctt, data"
  x-axis ["calibrateGdina", "calibrateDina", "calibrateDino", "calibrateAcdm", "calibrateLlm", "calibrateRrum", "attributeProfiles", "classificationAccuracy", "validateQMatrix", "nonparametric", "itemStatistics", "cronbachAlpha", "mcdonaldOmega", "subscoreValue", "scoreResponses"]
  y-axis "Time of measurekit over the time of the reference" 0 --> 10
  bar [2.1, 1.3, 1.6, 0.4, 0.38, 0.64, 9.8, 1.7, 1.1, 0.072, 0.22, 0.042, 0.63, 0.021, 5.3]
  line [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

```mermaid
%%{init: {"xyChart": {"height": 728, "xAxis": {"labelFontSize": 11}}}}%%
xychart-beta horizontal
  title "Psychometrics: dif, dimensionality, equating"
  x-axis ["mantelHaenszel", "logisticRegression", "lordChiSquare", "ordinalLogistic", "polychoricCorrelations", "parallelAnalysis", "linear", "equipercentile", "presmoothing", "bootstrapLinear", "tucker", "frequencyEstimation", "chainedEquipercentile", "stockingLordLinking", "haebaraLinking", "trueScore", "irtObservedScore"]
  y-axis "Time of measurekit over the time of the reference" 0 --> 2
  bar [0.077, 0.85, 0.88, 0.58, 0.013, 0.0086, 0.00072, 0.0076, 0.024, 0.081, 0.0044, 0.0039, 0.015, 0.13, 0.092, 0.056, 0.17]
  line [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

```mermaid
%%{init: {"xyChart": {"height": 388, "xAxis": {"labelFontSize": 11}}}}%%
xychart-beta horizontal
  title "Psychometrics: explanatory, fit, generalizability"
  x-axis ["itemFeatureRegression", "lltm", "itemFit", "personFit", "summedScoreFit", "yenQ3", "generalizabilityStudy"]
  y-axis "Time of measurekit over the time of the reference" 0 --> 2
  bar [0.02, 0.8, 0.23, 0.39, 0.097, 0.067, 0.0026]
  line [1, 1, 1, 1, 1, 1, 1]
```

```mermaid
%%{init: {"xyChart": {"height": 490, "xAxis": {"labelFontSize": 11}}}}%%
xychart-beta horizontal
  title "Psychometrics: mokken, rater, responseTime, sim"
  x-axis ["scalability", "automatedItemSelection", "monotonicity", "invariantItemOrdering", "cohenKappa", "fleissKappa", "intraclassCorrelation", "manyFacetRasch", "calibrateResponseTimes", "simulateResponses"]
  y-axis "Time of measurekit over the time of the reference" 0 --> 14
  bar [0.0081, 0.025, 0.22, 0.19, 0.2, 0.008, 0.0029, 14, 0.042, 0.77]
  line [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

```mermaid
%%{init: {"xyChart": {"height": 388, "xAxis": {"labelFontSize": 11}}}}%%
xychart-beta horizontal
  title "Knowledge tracing: bkt, elo, evaluate, irt, kst"
  x-axis ["fit", "replay", "replay", "auc", "replay", "blim", "iita"]
  y-axis "Time of measurekit over the time of the reference" 0 --> 29
  bar [0.012, 0.049, 1.3, 1.2, 29, 1.2, 0.000024]
  line [1, 1, 1, 1, 1, 1, 1]
```

```mermaid
%%{init: {"xyChart": {"height": 184, "xAxis": {"labelFontSize": 11}}}}%%
xychart-beta horizontal
  title "Knowledge tracing: logistic"
  x-axis ["afm"]
  y-axis "Time of measurekit over the time of the reference" 0 --> 2
  bar [0.041]
  line [1]
```

<!-- /charts -->

## Results

<!-- tables -->

Measured on Apple M5 with 10 cores, 16 GB of memory, Node.js 26.8.1 and R version 4.6.1 (2026-06-24).

### Core

| Task                         | Size                              | measurekit (ms) | Reference            | Reference (ms) |    Ratio |
| ---------------------------- | --------------------------------- | --------------: | -------------------- | -------------: | -------: |
| **Normal quantile function** | 1,000,000 probabilities           |            25.2 | stats 4.6.1 `qnorm`  |           6.00 | **4.2×** |
| **Logistic function**        | 1,000,000 points                  |            25.4 | stats 4.6.1 `plogis` |           5.30 | **4.8×** |
| **Cholesky factorization**   | A correlation matrix of order 200 |           0.757 | base 4.6.1 `chol`    |          0.118 | **6.4×** |
| Uniform draws                | 1,000,000 draws                   |            3.36 | stats 4.6.1 `runif`  |           3.29 |       1× |
| Normal draws                 | 1,000,000 draws                   |            16.7 | stats 4.6.1 `rnorm`  |           12.5 |     1.3× |

### Psychometrics

| Task                                                                                   | Size                                                             | measurekit (ms) | Reference                                                               | Reference (ms) |    Ratio |
| -------------------------------------------------------------------------------------- | ---------------------------------------------------------------- | --------------: | ----------------------------------------------------------------------- | -------------: | -------: |
| **Calibration of the one-parameter model by EM**                                       | 2,000 persons, 40 dichotomous items                              |             202 | mirt 1.48 `mirt(itemtype = "Rasch")`                                    |           91.0 | **2.2×** |
| Calibration of the two-parameter model by EM                                           | 2,000 persons, 40 dichotomous items                              |             235 | mirt 1.48 `mirt(itemtype = "2PL")`                                      |            164 |     1.4× |
| Calibration of the three-parameter model by EM                                         | 2,000 persons, 40 dichotomous items                              |             756 | mirt 1.48 `mirt(itemtype = "3PL")`                                      |          1,110 |    0.68× |
| The two-parameter model with cross-product standard errors                             | 2,000 persons, 40 dichotomous items                              |             250 | mirt 1.48 `mirt(SE.type = "crossprod")`                                 |            602 |    0.42× |
| The two-parameter model with standard errors from the Hessian                          | 500 persons, 20 items                                            |             115 | mirt 1.48 `mirt(SE.type = "Richardson")`                                |          7,740 |   0.015× |
| Calibration of the graded response model by EM                                         | 1,500 persons, 20 items of 5 categories                          |             105 | mirt 1.48 `mirt(itemtype = "graded")`                                   |            234 |    0.45× |
| Calibration of the generalized partial credit model by EM                              | 1,500 persons, 20 items of 5 categories                          |             169 | mirt 1.48 `mirt(itemtype = "gpcm")`                                     |            357 |    0.47× |
| Calibration of the testlet model by EM                                                 | 1,000 persons, 6 testlets of 5 items                             |             343 | mirt 1.48 `bfactor`                                                     |         27,374 |   0.013× |
| **Ability by the posterior mean**                                                      | 2,000 persons, the 40 items of the calibration                   |             272 | mirt 1.48 `fscores(method = "EAP")`                                     |           52.0 | **5.2×** |
| Ability by the posterior mode                                                          | 2,000 persons, the 40 items of the calibration                   |             301 | mirt 1.48 `fscores(method = "MAP")`                                     |          1,529 |     0.2× |
| Ability by maximum likelihood                                                          | 2,000 persons, the 40 items of the calibration                   |             294 | mirt 1.48 `fscores(method = "ML")`                                      |            950 |    0.31× |
| Ability by weighted likelihood                                                         | 2,000 persons, the 40 items of the calibration                   |             707 | mirt 1.48 `fscores(method = "WLE")`                                     |         15,729 |   0.045× |
| Joint maximum likelihood calibration of the Rasch model                                | 2,000 persons, 40 items                                          |            11.8 | TAM 4.3.25 `tam.jml`                                                    |           60.0 |     0.2× |
| The posterior mean of the ability given each summed score                              | 40 items                                                         |           0.245 | none                                                                    |                |          |
| Information of a test at 121 values of theta                                           | 40 items                                                         |           0.881 | mirt 1.48 `testinfo`                                                    |           3.77 |    0.23× |
| Calibration of a confirmatory two-dimensional 2PL by EM                                | 1,000 persons, 24 items, 2 correlated dimensions                 |             971 | mirt 1.48 `mirt(model, quadpts = 31)`                                   |          1,436 |    0.68× |
| Calibration of a confirmatory two-dimensional graded model by EM                       | 1,000 persons, 20 items of 4 categories, 2 correlated dimensions |           1,104 | mirt 1.48 `mirt(itemtype = "graded", quadpts = 31)`                     |          1,894 |    0.58× |
| **Calibration of an exploratory two-dimensional 2PL by EM**                            | 1,000 persons, 20 items, 2 dimensions                            |          16,789 | mirt 1.48 `mirt(data, 2, quadpts = 31)`                                 |          3,167 | **5.3×** |
| Abilities by the posterior mode, in two dimensions                                     | 1,000 persons, 24 items                                          |            75.6 | mirt 1.48 `fscores(method = "MAP")`                                     |            781 |   0.097× |
| **Abilities by the posterior mean, in two dimensions**                                 | 1,000 persons, 24 items                                          |           1,259 | mirt 1.48 `fscores(method = "EAP")`                                     |           52.0 |  **24×** |
| Oblimin rotation of the loadings of a calibration                                      | 20 items, 2 dimensions                                           |           0.033 | GPArotation 2026.8.2 `oblimin`                                          |          0.235 |    0.14× |
| Varimax rotation of the loadings of a calibration                                      | 20 items, 2 dimensions                                           |           0.025 | GPArotation 2026.8.2 `Varimax`                                          |          0.100 |    0.25× |
| Simulation of adaptive tests by maximum information and the posterior mean             | 200 persons, a pool of 500 3PL items, 30 items each              |             464 | catR 3.17 `randomCAT`                                                   |         10,892 |   0.043× |
| Simulation of diagnostic adaptive tests by the G-DINA discrimination index             | 200 persons, a pool of 20 items over 4 attributes, 10 items each |            13.2 | cdcatR 1.0.7 `cdcat`                                                    |            466 |   0.028× |
| Simulation of multidimensional adaptive tests by the determinant of the information    | 100 persons, a pool of 200 two-dimensional items, 20 items each  |             153 | mirtCAT 1.14 `mirtCAT(criteria = "Drule")`                              |          5,237 |   0.029× |
| **Calibration of the G-DINA model by EM**                                              | 1,000 persons, 20 items, 4 attributes                            |             211 | GDINA 2.13.2 `GDINA(model = "GDINA")`                                   |            101 | **2.1×** |
| Calibration of the DINA model by EM                                                    | 1,000 persons, 20 items, 4 attributes                            |            50.9 | GDINA 2.13.2 `GDINA(model = "DINA")`                                    |           40.5 |     1.3× |
| **Calibration of the DINO model by EM**                                                | 1,000 persons, 20 items, 4 attributes                            |             227 | GDINA 2.13.2 `GDINA(model = "DINO")`                                    |            143 | **1.6×** |
| Calibration of the A-CDM model by EM                                                   | 1,000 persons, 20 items, 4 attributes                            |            52.9 | GDINA 2.13.2 `GDINA(model = "ACDM")`                                    |            132 |     0.4× |
| Calibration of the LLM model by EM                                                     | 1,000 persons, 20 items, 4 attributes                            |            68.4 | GDINA 2.13.2 `GDINA(model = "LLM")`                                     |            181 |    0.38× |
| Calibration of the R-RUM model by EM                                                   | 1,000 persons, 20 items, 4 attributes                            |            70.2 | GDINA 2.13.2 `GDINA(model = "RRUM")`                                    |            110 |    0.64× |
| **Posterior profiles of the persons of a calibration**                                 | 1,000 persons, 20 items, 4 attributes                            |            1.54 | GDINA 2.13.2 `personparm(what = "EAP")`                                 |          0.156 | **9.8×** |
| **Accuracy and consistency of the classification**                                     | 1,000 persons, 20 items, 4 attributes                            |            1.73 | GDINA 2.13.2 `CA`                                                       |           1.04 | **1.7×** |
| Validation of the Q-matrix by the proportion of variance accounted for                 | 1,000 persons, 20 items, 4 attributes                            |            3.11 | GDINA 2.13.2 `Qval(method = "PVAF")`                                    |           2.91 |     1.1× |
| Nonparametric classification by the Hamming distance                                   | 1,000 persons, 20 items, 4 attributes                            |            1.53 | NPCD 1.0.11 `AlphaNP`                                                   |           21.3 |   0.072× |
| Item difficulty, discrimination and the statistics of the items                        | 3,000 persons, 50 items                                          |            10.5 | psych 2.6.9 `alpha`                                                     |           48.0 |    0.22× |
| Cronbach's alpha                                                                       | 3,000 persons, 50 items                                          |            2.04 | psych 2.6.9 `alpha`                                                     |           48.0 |   0.042× |
| McDonald's omega total of one factor                                                   | 3,000 persons, 20 items                                          |            5.20 | psych 2.6.9 `omega(nfactors = 1)`                                       |           8.20 |    0.63× |
| The value of five subscores of ten items over the total score                          | 3,000 persons, 50 items                                          |            2.31 | subscore 3.3 `CTTsub`                                                   |            108 |   0.021× |
| **Scoring of the choices of persons against a key**                                    | 2,000 persons, 50 items of 4 options                             |            3.81 | base 4.6.1 `t(t(choices) == key)`                                       |          0.720 | **5.3×** |
| The Mantel-Haenszel test of every item                                                 | 2,000 persons in two groups, 30 items                            |            6.54 | difR 6.1.0 `difMH`                                                      |           85.0 |   0.077× |
| Logistic regression for uniform and nonuniform DIF of every item                       | 2,000 persons in two groups, 30 items                            |            72.2 | difR 6.1.0 `difLogistic`                                                |           85.0 |    0.85× |
| Lord's chi-square of every item, with both groups calibrated by the 2PL                | 2,000 persons in two groups, 30 items                            |             888 | difR 6.1.0 `difLord`                                                    |          1,014 |    0.88× |
| Ordinal logistic regression of every item                                              | 1,000 persons in two groups, 10 items of 4 categories            |            54.1 | rms 8.2.0 `lrm, three models for an item`                               |           93.0 |    0.58× |
| Polychoric correlations of every pair of items                                         | 2,800 persons, 25 items of 6 categories                          |            19.5 | psych 2.6.9 `polychoric`                                                |          1,530 |   0.013× |
| Parallel analysis of components and factors from polychoric correlations, 20 data sets | 2,800 persons, 25 items of 6 categories                          |             297 | psych 2.6.9 `fa.parallel(cor = "poly")`                                 |         34,570 |  0.0086× |
| Linear equating, random groups                                                         | 4,329 and 4,152 persons, 40 items                                |           0.000 | equate 2.0.9 `equate(type = "linear")`                                  |          0.520 | 0.00072× |
| Equipercentile equating, random groups                                                 | 4,329 and 4,152 persons, 40 items                                |           0.002 | equate 2.0.9 `equate(type = "equipercentile")`                          |          0.320 |  0.0076× |
| Log-linear presmoothing of a score distribution, degree 3                              | 41 scores                                                        |           0.022 | equate 2.0.9 `presmoothing(smoothmethod = "loglinear")`                 |          0.882 |   0.024× |
| Standard errors of the linear equating by 100 bootstrap replications                   | 4,329 and 4,152 persons, 40 items                                |            24.5 | equate 2.0.9 `bootstrap`                                                |            301 |   0.081× |
| Tucker linear equating, nonequivalent groups                                           | 1,655 and 1,638 persons, 36 items, an anchor of 12               |           0.007 | equate 2.0.9 `equate(method = "tucker")`                                |           1.53 |  0.0044× |
| Frequency estimation equipercentile equating                                           | 1,655 and 1,638 persons, 36 items, an anchor of 12               |           0.009 | equate 2.0.9 `equate(method = "frequency estimation")`                  |           2.41 |  0.0039× |
| Chained equipercentile equating                                                        | 1,655 and 1,638 persons, 36 items, an anchor of 12               |           0.012 | equate 2.0.9 `equate(method = "chained")`                               |          0.824 |   0.015× |
| Stocking-Lord linking of the common items                                              | Two forms of 3PL items that share 10                             |           0.229 | equateIRT 2.5.2 `direc(method = "Stocking-Lord")`                       |           1.77 |    0.13× |
| Haebara linking of the common items                                                    | Two forms of 3PL items that share 10                             |           0.173 | equateIRT 2.5.2 `direc(method = "Haebara")`                             |           1.88 |   0.092× |
| IRT true score equating of the linked forms                                            | Two forms of 3PL items that share 10                             |           0.168 | equateIRT 2.5.2 `score(method = "TSE")`                                 |           3.00 |   0.056× |
| IRT observed score equating of the linked forms                                        | Two forms of 3PL items that share 10                             |           0.387 | equateIRT 2.5.2 `score(method = "OSE")`                                 |           2.29 |    0.17× |
| Regression of item difficulties on three features by REML                              | 300 items, 3 features                                            |           0.389 | metafor 5.2.1 `rma(method = "REML")`                                    |           19.0 |    0.02× |
| Infit and outfit of the items                                                          | 2,000 persons, 40 items                                          |            5.55 | mirt 1.48 `itemfit(fit_stats = "infit")`                                |           24.0 |    0.23× |
| Infit, outfit and the standardized log-likelihood of the persons                       | 2,000 persons, 40 items                                          |            8.27 | mirt 1.48 `personfit`                                                   |           21.0 |    0.39× |
| The summed-score statistic S-X2 of the items                                           | 2,000 persons, 40 items                                          |            5.61 | mirt 1.48 `itemfit(fit_stats = "S_X2")`                                 |           58.0 |   0.097× |
| Yen's Q3 of every pair of items                                                        | 2,000 persons, 40 items                                          |            8.66 | mirt 1.48 `residuals(type = "Q3")`                                      |            129 |   0.067× |
| Variance components of a fully crossed design of three facets                          | 200 persons, 10 items, 4 raters: 8,000 scores                    |            2.51 | lme4 2.1.0 `lmer, six random effects, REML`                             |            955 |  0.0026× |
| Scalability coefficients with standard errors                                          | 433 persons, 20 items of 5 categories                            |            2.25 | mokken 3.1.2 `coefH(se = TRUE)`                                         |            279 |  0.0081× |
| Automated item selection into Mokken scales                                            | 433 persons, 20 items of 5 categories                            |           0.959 | mokken 3.1.2 `aisp`                                                     |           39.0 |   0.025× |
| The check of monotonicity of every item                                                | 433 persons, 20 items of 5 categories                            |            3.02 | mokken 3.1.2 `check.monotonicity`                                       |           14.0 |    0.22× |
| The check of invariant item ordering, with selection of items                          | 433 persons, 20 items of 5 categories                            |            17.5 | mokken 3.1.2 `check.iio(method = "MIIO")`                               |           90.0 |    0.19× |
| Cohen's kappa of two raters                                                            | 2,000 subjects, 5 categories                                     |           0.051 | irr 0.85 `kappa2`                                                       |          0.255 |     0.2× |
| Fleiss' kappa of five raters                                                           | 500 subjects, 5 raters, 5 categories                             |           0.067 | irr 0.85 `kappam.fleiss`                                                |           8.33 |   0.008× |
| The six intraclass correlations of four raters                                         | 200 subjects, 4 raters                                           |           0.017 | psych 2.6.9 `ICC`                                                       |           5.83 |  0.0029× |
| **The many-facet Rasch model of items and raters, with standard errors**               | 200 persons, 10 items, 4 raters: 8,000 scores                    |             665 | TAM 4.3.25 `tam.mml.mfr(formulaA = ~ item + rater + step)`              |           48.0 |  **14×** |
| The lognormal model of the response times                                              | 1,000 persons, 20 items                                          |            6.62 | nlme 3.1.169 `lme(logtime ~ 0 + item, random = ~ 1                      |       person)` |      159 | 0.042× |
| The joint model of speed and accuracy, with the 2PL                                    | 1,000 persons, 20 items                                          |             599 | none                                                                    |                |          |
| Speed of each person given the times of the items                                      | 1,000 persons, 20 items                                          |            1.51 | none                                                                    |                |          |
| A conversion table of the posterior mean of theta given each summed score              | 40 items                                                         |           0.407 | none                                                                    |                |          |
| A conversion table of the theta of each summed true score                              | 40 items                                                         |            1.16 | none                                                                    |                |          |
| The cut score of a bookmark in an ordered item booklet                                 | 40 items                                                         |           0.039 | none                                                                    |                |          |
| Accuracy and consistency of the classification by two cuts                             | 2,000 persons                                                    |           0.442 | none                                                                    |                |          |
| Conversion of 2PL items to mirt's slope-intercept form                                 | 10,000 items                                                     |            1.69 | none                                                                    |                |          |
| Conversion of 2PL items to mirt's IRT form                                             | 10,000 items                                                     |            1.82 | none                                                                    |                |          |
| Simulation of the abilities of persons and their responses to 2PL items                | 2,000 persons, 40 items                                          |            4.52 | mirt 1.48 `simdata(itemtype = "dich")`                                  |           5.88 |    0.77× |
| The linear logistic test model: the effects of three features on difficulty, by EM     | 1,000 persons, 20 items, 3 features                              |             215 | mirt 1.48 `mirt(itemtype = "Rasch", itemdesign), Oakes standard errors` |            269 |     0.8× |

- Calibration of an exploratory two-dimensional 2PL by EM: mirt stops at its limit of 500 cycles when it has not converged.
- The lognormal model of the response times: measurekit also estimates the item discriminations.

### Knowledge tracing

| Task                                                                                          | Size                                            | measurekit (ms) | Reference                               | Reference (ms) |     Ratio |
| --------------------------------------------------------------------------------------------- | ----------------------------------------------- | --------------: | --------------------------------------- | -------------: | --------: |
| Fit of the skills of a log by EM from five starts                                             | 20,000 attempts of 200 learners at 5 skills     |             126 | pyBKT 1.4.3 `Model.fit`                 |         10,888 |    0.012× |
| Replay of the log through the fitted model, a prediction and an update for each attempt       | 20,000 attempts of 200 learners at 5 skills     |            15.7 | pyBKT 1.4.3 `Model.predict`             |            318 |    0.049× |
| Replay of a log through the learner-item Elo model                                            | 100,000 attempts of 2,000 learners at 400 items |            40.8 | base R, a loop over the attempts        |           32.5 |      1.3× |
| The area under the ROC curve                                                                  | 100,000 predictions                             |            23.4 | pROC 1.19.1 `auc(roc(direction = "<"))` |           19.3 |      1.2× |
| The AUC, RMSE, log-loss and accuracy of the predictions                                       | 100,000 predictions                             |            26.9 | none                                    |                |           |
| **Replay of the responses of learners, the grid posterior of the ability updated after each** | 2,000 learners, 40 responses each               |             987 | mirt 1.48 `fscores(method = "EAP")`     |           34.0 |   **29×** |
| Fit of the basic local independence model of a chain of 12 items by EM                        | 1,000 persons, 12 items, 13 states              |            33.5 | pks 0.8.0 `blim`                        |           29.0 |      1.2× |
| Inductive item tree analysis of the responses                                                 | 1,000 persons, 12 items                         |            2.01 | DAKS 2.1.3 `iita(v = 1)`                |         84,720 | 0.000024× |
| Fit of the additive factors model by maximum likelihood                                       | 20,000 attempts of 200 learners at 5 skills     |            22.0 | base R `glm`, binomial family           |            534 |    0.041× |
| The candidates ranked by their distance to the band of success of a learner                   | 500 candidate items, an Elo model               |           0.115 | none                                    |                |           |
| A log built from and checked for events                                                       | 100,000 events                                  |            18.0 | none                                    |                |           |

- Replay of the responses of learners, the grid posterior of the ability updated after each: mirt scores each pattern once; measurekit updates the posterior after every response.
- Fit of the additive factors model by maximum likelihood: the proficiency of a learner is a fixed effect, as measurekit's default is none.

<!-- /tables -->

## Iterations

The iterations that each EM calibration of the tables needed to reach its tolerance, on the data of the benchmark. A ratio
of times is the product of a ratio of iterations and a ratio of the time of one.

| Calibration                         | measurekit | Reference |
| ----------------------------------- | ---------: | --------: |
| One-parameter model                 |         25 |         9 |
| Two-parameter model                 |         29 |        19 |
| Three-parameter model               |         90 |       183 |
| Graded response model               |         22 |        11 |
| Generalized partial credit model    |         41 |        37 |
| Confirmatory two-dimensional 2PL    |         23 |        24 |
| Confirmatory two-dimensional graded |         22 |        43 |
| Exploratory two-dimensional 2PL     |        442 |       135 |
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

| Part       | How                                                                                                                                                                                                                                                    |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Data       | `validation/benchmarks.R` simulates the inputs with a fixed seed, or takes a public dataset, and writes them to `packages/<package>/test/fixtures/bench-<module>.json`. measurekit and its reference read the same data.                               |
| measurekit | `packages/<package>/test/<module>.bench.ts`, run by Vitest on Node.js with one thread. A benchmark takes at least 10 samples, or 5 when it takes a second or more, after a warm-up. The tables give the median of the medians of three runs.           |
| References | `validation/benchmarks/<package>/<module>.R`, run with one thread: `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` set R's BLAS. A task takes 15 samples under a second, 7 under 15 seconds, 3 under a minute and 2 above, after one call to warm up. |
| pyBKT      | `validation/benchmarks.py`, on the log of `bench-bkt.json`, with its processes off. It times what R does not.                                                                                                                                          |
| Settings   | A reference runs with the settings that make its work that of measurekit: the same quadrature, convergence tolerance and starts. The notes under a table name where they differ.                                                                       |
| Ratio      | The median of measurekit over the median of the reference.                                                                                                                                                                                             |

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
