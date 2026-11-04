# Coming to psychometrics from R

How the calls of mirt, TAM, catR and mirtCAT, GDINA, equate, equateIRT and kequate, difR, psych, the variance components
of gtheory and mokken map to `@measurekit/psychometrics`. Each R call stands beside the function that the tests compare
with it, the tolerance of that comparison from [validation](validation.md#tolerances), and the time of measurekit over the
time of the R call from [performance](performance.md#results), measured on Apple M5 with Node.js 26.8.1 and R 4.6.1 and
published there on 2026-11-04: below 1, measurekit takes less time.

```mermaid
flowchart LR
  R["An R session<br/>data.frame, mirt, difR, equate"] -->|"the same data as rows"| M["buildResponseMatrix"]
  M --> C["calibrate, mantelHaenszel,<br/>equipercentileEquating, ..."]
  C --> P["Plain objects: parameters in<br/>mirt's IRT form, D = 1"]
  P -->|"writeMirtIrt, writeMirtSlopeIntercept"| B["mirt's coefficients"]
```

## Side by side

A 2PL calibration and the posterior mean of the first person's theta, in R:

```r
library(mirt)
fit <- mirt(data, 1, itemtype = "2PL")
coef(fit, IRTpars = TRUE, simplify = TRUE)$items
coef(fit, IRTpars = FALSE, simplify = TRUE)$items
fscores(fit, method = "EAP", full.scores.SE = TRUE)[1, ]
```

The same in TypeScript, on 1,000 simulated persons:

```ts
import { createRandom } from "@measurekit/psychometrics";
import { calibrate, posteriorMeanEstimate, twoParameterLogistic } from "@measurekit/psychometrics/irt";
import { writeMirtSlopeIntercept } from "@measurekit/psychometrics/parameterization";
import { simulateItems, simulateResponses, simulateThetas } from "@measurekit/psychometrics/sim";

const random = createRandom(7);
const truth = simulateItems(twoParameterLogistic, 10, random);
const data = simulateResponses(truth, simulateThetas(1000, random), random);
const fit = calibrate(data, { model: twoParameterLogistic });
fit.items[0]?.parameters; // => { a: 0.83, b: -0.18, c: 0, d: 1 }: coef(fit, IRTpars = TRUE)
const coefficients = fit.items.map(writeMirtSlopeIntercept);
coefficients[0]; // => { a1: 0.83, d: 0.15, g: 0, u: 1 }: coef(fit, IRTpars = FALSE)
posteriorMeanEstimate(fit.items, data.responses[0] ?? []); // => { theta: 0.32, standardError: 0.59 }
```

## The data

| In R                                                 | In measurekit                                                                                                              |
| ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| A data.frame or matrix of scores, `NA` where missing | `buildResponseMatrix(rows, { itemIds })`, or `buildResponseMatrixFromColumns`, with `null` where missing ([data](data.md)) |
| Categories from 0 or from 1, by package              | Categories are integers from 0                                                                                             |
| A data.frame in long format                          | `buildResponseRecords` of `{ person, item, response }`, for sparse data                                                    |
| `set.seed(n)` before a simulation                    | `createRandom(n)`, another generator: a simulation does not reproduce R's draws                                            |
| mirt's item parameters, D = 1                        | The same IRT form; [parameterization](parameterization.md) converts to and from mirt's coefficients                        |
| An S4 object with methods and a print                | A plain object of the result's fields, which JSON serializes                                                               |
| `NA` for a statistic that the data do not define     | `null` in a table, and `DataError` from a function that returns only that statistic                                        |

## Calls

The tolerance is against the R call of the row unless the row names another reference; "none" in the time is a task that
[performance](performance.md#results) times against no R call.

### mirt

| R call                                            | measurekit                                                      | Tolerance                                                | Time over R |
| ------------------------------------------------- | --------------------------------------------------------------- | -------------------------------------------------------- | ----------: |
| `mirt(data, 1, itemtype = "2PL")`                 | `calibrate(matrix, { model: twoParameterLogistic })`            | 1e-5 absolute, the item parameters                       |       0.51× |
| `mirt(data, 1, itemtype = "graded")`              | `calibrate` with `gradedResponse`                               | 2e-4 absolute                                            |       0.26× |
| `mirt(data, 1, itemtype = "gpcm")`                | `calibrate` with `generalizedPartialCredit`                     | 2e-4 absolute                                            |       0.19× |
| `mirt(data, model, quadpts = 31)`, two dimensions | `calibrateMultidimensional`                                     | 1e-5 absolute, the M2PL's parameters                     |        0.3× |
| `bfactor(data, specific, quadpts = 31)`           | `calibrateBifactor`                                             | 5e-4 absolute                                            |      0.052× |
| `fscores(method = "EAP")`                         | `posteriorMeanEstimate`, or `abilityEstimates` for every person | 1e-10 absolute                                           |       0.62× |
| `fscores(method = "MAP")`                         | `posteriorModeEstimate`                                         | 1e-5 absolute                                            |      0.064× |
| `fscores(method = "ML")`                          | `maximumLikelihoodEstimate`                                     | 1e-5 absolute                                            |       0.11× |
| `fscores(method = "WLE")`                         | `weightedLikelihoodEstimate`                                    | 1e-10 absolute, against base R's root of Warm's equation |      0.015× |
| `testinfo`                                        | `testInformation`                                               | 1e-12 absolute                                           |       0.41× |
| `itemfit(fit_stats = "S_X2")`                     | `summedScoreFit`                                                | 1e-12 absolute                                           |       0.13× |
| `residuals(type = "Q3")`                          | `yenQ3`                                                         | 1e-12 absolute                                           |      0.044× |
| `M2`                                              | `limitedInformationFit`                                         | 1e-3 absolute                                            |       0.16× |
| `multipleGroup`, `DIF(scheme = "drop")`           | `irtLikelihoodRatioDif`                                         | 1e-5 absolute                                            |       0.24× |

### TAM

| R call                                          | measurekit                | Tolerance                                                       | Time over R |
| ----------------------------------------------- | ------------------------- | --------------------------------------------------------------- | ----------: |
| `tam.jml`                                       | `jointMaximumLikelihood`  | 1e-10 absolute, from TAM's estimates iterated by base R         |       0.33× |
| `IRT.RMSD`                                      | `rootMeanSquareDeviation` | 1e-8 absolute                                                   |        5.5× |
| `tam.mml.mfr(formulaA = ~ item + rater + step)` | `manyFacetRasch`          | 1e-8 absolute, against base R's maximum, which TAM's approaches |        1.2× |

### catR and mirtCAT

| R call                                  | measurekit                                                       | Tolerance                                                        | Time over R |
| --------------------------------------- | ---------------------------------------------------------------- | ---------------------------------------------------------------- | ----------: |
| catR's `nextItem(criterion = "MLWI")`   | `createAdaptiveTest` with `likelihoodWeightedInformation`        | 1e-12 relative, the scores of the items                          |    0.00017× |
| catR's `nextItem(criterion = "KL")`     | `createAdaptiveTest` with `kullbackLeibler`                      | 1e-12 relative                                                   |      0.008× |
| catR's `thetaEst` and `semTheta`, EAP   | The estimates of an adaptive test after each response            | 1e-6 absolute                                                    |   not timed |
| catR's `randomCAT`                      | `simulateAdaptiveTest`                                           | catR's item at every step of a run, through `createAdaptiveTest` |      0.017× |
| mirtCAT's `mirtCAT(criteria = "Drule")` | `simulateMultidimensionalAdaptiveTest` with `maximumDeterminant` | 1e-12 relative, against a base R replay                          |      0.012× |

### GDINA

| R call                       | measurekit                   | Tolerance                                                | Time over R |
| ---------------------------- | ---------------------------- | -------------------------------------------------------- | ----------: |
| `GDINA(model = "GDINA")`     | `calibrateCdm`               | 1e-5 absolute, the success probabilities and proportions |        0.8× |
| `GDINA(model = "DINA")`      | `calibrateCdm` with the DINA | 1e-5 absolute                                            |       0.48× |
| `modelcomp(method = "Wald")` | `compareItemModels`          | 1e-4 relative to max(1, statistic), the Wald statistics  |     0.0065× |
| `modelfit`                   | `cdmModelFit`                | 1e-3 absolute                                            |       0.58× |
| `personparm(what = "EAP")`   | `attributeProfiles`          | 1e-5 absolute                                            |        2.7× |
| `CA`                         | `cdmClassificationAccuracy`  | 1e-5 absolute                                            |       0.45× |

### equate, equateIRT and kequate

| R call                                             | measurekit                    | Tolerance                                                   | Time over R |
| -------------------------------------------------- | ----------------------------- | ----------------------------------------------------------- | ----------: |
| equate's `equate(type = "equipercentile")`         | `equipercentileEquating`      | 1e-6 absolute                                               |      0.008× |
| equate's `equate(method = "tucker")`               | `tuckerEquating`              | 1e-12 absolute                                              |     0.0044× |
| equate's `equate(method = "frequency estimation")` | `frequencyEstimationEquating` | 1e-6 absolute                                               |     0.0039× |
| equateIRT's `direc(method = "Stocking-Lord")`      | `stockingLordLinking`         | 1e-7 absolute                                               |       0.04× |
| equateIRT's `score(method = "TSE")`                | `trueScoreEquating`           | 1e-12 absolute, against base R, with which equateIRT agrees |      0.018× |
| kequate's `kequate("EG")`                          | `kernelEquating`              | 1e-5 absolute                                               |     0.0098× |

### difR

| R call        | measurekit              | Tolerance                                                          | Time over R |
| ------------- | ----------------------- | ------------------------------------------------------------------ | ----------: |
| `difMH`       | `mantelHaenszel`        | 1e-12 absolute                                                     |       0.06× |
| `difLogistic` | `logisticRegressionDif` | 1e-9 absolute                                                      |       0.53× |
| `difLord`     | `lordChiSquare`         | 1e-5 relative                                                      |       0.73× |
| `difRaju`     | `rajuAreaDif`           | 1e-8 absolute                                                      |       0.67× |
| `difSIBTEST`  | `sibtest`               | 1e-10 absolute, against mirt's `SIBTEST`, which difR's agrees with |       0.15× |

### psych

| R call                              | measurekit                        | Tolerance                                                      | Time over R |
| ----------------------------------- | --------------------------------- | -------------------------------------------------------------- | ----------: |
| `alpha`                             | `cronbachAlpha`, `itemStatistics` | 1e-10 absolute                                                 |      0.037× |
| `omega(nfactors = 1)`               | `mcdonaldOmega`                   | 1e-10 absolute against base R; psych's `omega.tot` within 1e-4 |       0.62× |
| `polychoric`                        | `polychoricCorrelations`          | 1e-4 absolute                                                  |     0.0064× |
| `fa.parallel(cor = "poly")`         | `parallelAnalysis`                | 4 Monte Carlo standard errors, the mean eigenvalues            |     0.0032× |
| `fa(fm = "ml", rotate = "oblimin")` | `exploratoryFactorAnalysis`       | 1e-4 absolute, the loadings                                    |       0.26× |
| `ICC`                               | `intraclassCorrelation`           | 1e-12 absolute                                                 |     0.0029× |

### gtheory

gtheory is archived on CRAN. Its G and D studies are compared with their definitions written out in base R and with
lme4's REML estimates.

| R call   | measurekit              | Tolerance                                                        |                    Time over R |
| -------- | ----------------------- | ---------------------------------------------------------------- | -----------------------------: |
| `gstudy` | `generalizabilityStudy` | 1e-10 relative against base R; 1e-5 absolute against lme4's REML | 0.0026×, against lme4's `lmer` |
| `dstudy` | `decisionStudy`         | 1e-10 relative                                                   |                      not timed |

### mokken

| R call                       | measurekit               | Tolerance                                   | Time over R |
| ---------------------------- | ------------------------ | ------------------------------------------- | ----------: |
| `coefH(se = TRUE)`           | `scalability`            | 1e-10, relative to max(1, value)            |      0.011× |
| `aisp`                       | `automatedItemSelection` | exact                                       |       0.03× |
| `check.monotonicity`         | `monotonicity`           | exact counts; 1e-8 absolute, the violations |       0.17× |
| `check.iio(method = "MIIO")` | `invariantItemOrdering`  | exact items removed; 1e-10 absolute, H^T    |       0.19× |

## Where measurekit differs

Each guide ends with its differences from the references, such as mirt's quadrature and its starting values. Two
differences reach every call: the generator of random numbers is not R's, so a simulation or a bootstrap gives other
draws from the same seed, and every iterative method reports `converged` and `iterations` instead of a warning.
