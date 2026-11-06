# Coming to psychometrics from R

How the calls of mirt, TAM, catR and mirtCAT, GDINA, equate, equateIRT and kequate, difR, psych, the variance components
of gtheory and mokken map to `@measurekit/psychometrics`. Each R call stands beside the function that the tests compare
with it and the tolerance of that comparison from [validation](validation.md#tolerances); [performance](performance.md#speed)
names the benchmarks that time each function against its R call.

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
const fit = calibrate(data, twoParameterLogistic);
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

The tolerance is against the R call of the row unless the row names another reference.

### mirt

| R call                                            | measurekit                                                      | Tolerance                                                |
| ------------------------------------------------- | --------------------------------------------------------------- | -------------------------------------------------------- |
| `mirt(data, 1, itemtype = "2PL")`                 | `calibrate(matrix, twoParameterLogistic)`                       | 1e-5 absolute, the item parameters                       |
| `mirt(data, 1, itemtype = "graded")`              | `calibrate` with `gradedResponse`                               | 2e-4 absolute                                            |
| `mirt(data, 1, itemtype = "gpcm")`                | `calibrate` with `generalizedPartialCredit`                     | 2e-4 absolute                                            |
| `mirt(data, model, quadpts = 31)`, two dimensions | `calibrateMultidimensional`                                     | 1e-5 absolute, the M2PL's parameters                     |
| `bfactor(data, specific, quadpts = 31)`           | `calibrateBifactor`                                             | 5e-4 absolute                                            |
| `fscores(method = "EAP")`                         | `posteriorMeanEstimate`, or `abilityEstimates` for every person | 1e-10 absolute                                           |
| `fscores(method = "MAP")`                         | `posteriorModeEstimate`                                         | 1e-5 absolute                                            |
| `fscores(method = "ML")`                          | `maximumLikelihoodEstimate`                                     | 1e-5 absolute                                            |
| `fscores(method = "WLE")`                         | `weightedLikelihoodEstimate`                                    | 1e-10 absolute, against base R's root of Warm's equation |
| `testinfo`                                        | `testInformation`                                               | 1e-12 absolute                                           |
| `itemfit(fit_stats = "S_X2")`                     | `summedScoreFit`                                                | 1e-12 absolute                                           |
| `residuals(type = "Q3")`                          | `yenQ3`                                                         | 1e-12 absolute                                           |
| `M2`                                              | `limitedInformationFit`                                         | 1e-3 absolute                                            |
| `multipleGroup`, `DIF(scheme = "drop")`           | `irtLikelihoodRatioDif`                                         | 1e-5 absolute                                            |

### TAM

| R call                                          | measurekit                | Tolerance                                                       |
| ----------------------------------------------- | ------------------------- | --------------------------------------------------------------- |
| `tam.jml`                                       | `jointMaximumLikelihood`  | 1e-10 absolute, from TAM's estimates iterated by base R         |
| `IRT.RMSD`                                      | `rootMeanSquareDeviation` | 1e-8 absolute                                                   |
| `tam.mml.mfr(formulaA = ~ item + rater + step)` | `manyFacetRasch`          | 1e-8 absolute, against base R's maximum, which TAM's approaches |

### catR and mirtCAT

| R call                                  | measurekit                                                       | Tolerance                                                        |
| --------------------------------------- | ---------------------------------------------------------------- | ---------------------------------------------------------------- |
| catR's `nextItem(criterion = "MLWI")`   | `createAdaptiveTest` with `likelihoodWeightedInformation`        | 1e-12 relative, the scores of the items                          |
| catR's `nextItem(criterion = "KL")`     | `createAdaptiveTest` with `kullbackLeibler`                      | 1e-12 relative                                                   |
| catR's `thetaEst` and `semTheta`, EAP   | The estimates of an adaptive test after each response            | 1e-6 absolute                                                    |
| catR's `randomCAT`                      | `simulateAdaptiveTest`                                           | catR's item at every step of a run, through `createAdaptiveTest` |
| mirtCAT's `mirtCAT(criteria = "Drule")` | `simulateMultidimensionalAdaptiveTest` with `maximumDeterminant` | 1e-12 relative, against a base R replay                          |

### GDINA

| R call                       | measurekit                   | Tolerance                                                |
| ---------------------------- | ---------------------------- | -------------------------------------------------------- |
| `GDINA(model = "GDINA")`     | `calibrateCdm`               | 1e-5 absolute, the success probabilities and proportions |
| `GDINA(model = "DINA")`      | `calibrateCdm` with the DINA | 1e-5 absolute                                            |
| `modelcomp(method = "Wald")` | `compareItemModels`          | 1e-4 relative to max(1, statistic), the Wald statistics  |
| `modelfit`                   | `cdmModelFit`                | 1e-3 absolute                                            |
| `personparm(what = "EAP")`   | `attributeProfiles`          | 1e-5 absolute                                            |
| `CA`                         | `cdmClassificationAccuracy`  | 1e-5 absolute                                            |

### equate, equateIRT and kequate

| R call                                             | measurekit                    | Tolerance                                                   |
| -------------------------------------------------- | ----------------------------- | ----------------------------------------------------------- |
| equate's `equate(type = "equipercentile")`         | `equipercentileEquating`      | 1e-6 absolute                                               |
| equate's `equate(method = "tucker")`               | `tuckerEquating`              | 1e-12 absolute                                              |
| equate's `equate(method = "frequency estimation")` | `frequencyEstimationEquating` | 1e-6 absolute                                               |
| equateIRT's `direc(method = "Stocking-Lord")`      | `stockingLordLinking`         | 1e-7 absolute                                               |
| equateIRT's `score(method = "TSE")`                | `trueScoreEquating`           | 1e-12 absolute, against base R, with which equateIRT agrees |
| kequate's `kequate("EG")`                          | `kernelEquating`              | 1e-5 absolute                                               |

### difR

| R call        | measurekit              | Tolerance                                                          |
| ------------- | ----------------------- | ------------------------------------------------------------------ |
| `difMH`       | `mantelHaenszel`        | 1e-12 absolute                                                     |
| `difLogistic` | `logisticRegressionDif` | 1e-9 absolute                                                      |
| `difLord`     | `lordChiSquare`         | 1e-5 relative                                                      |
| `difRaju`     | `rajuAreaDif`           | 1e-8 absolute                                                      |
| `difSIBTEST`  | `sibtest`               | 1e-10 absolute, against mirt's `SIBTEST`, which difR's agrees with |

### psych

| R call                              | measurekit                        | Tolerance                                                      |
| ----------------------------------- | --------------------------------- | -------------------------------------------------------------- |
| `alpha`                             | `cronbachAlpha`, `itemStatistics` | 1e-10 absolute                                                 |
| `omega(nfactors = 1)`               | `mcdonaldOmega`                   | 1e-10 absolute against base R; psych's `omega.tot` within 1e-4 |
| `polychoric`                        | `polychoricCorrelations`          | 1e-4 absolute                                                  |
| `fa.parallel(cor = "poly")`         | `parallelAnalysis`                | 4 Monte Carlo standard errors, the mean eigenvalues            |
| `fa(fm = "ml", rotate = "oblimin")` | `exploratoryFactorAnalysis`       | 1e-4 absolute, the loadings                                    |
| `ICC`                               | `intraclassCorrelation`           | 1e-12 absolute                                                 |

### gtheory

gtheory is archived on CRAN. Its G and D studies are compared with their definitions written out in base R and with
lme4's REML estimates.

| R call   | measurekit              | Tolerance                                                        |
| -------- | ----------------------- | ---------------------------------------------------------------- |
| `gstudy` | `generalizabilityStudy` | 1e-10 relative against base R; 1e-5 absolute against lme4's REML |
| `dstudy` | `decisionStudy`         | 1e-10 relative                                                   |

### mokken

| R call                       | measurekit               | Tolerance                                   |
| ---------------------------- | ------------------------ | ------------------------------------------- |
| `coefH(se = TRUE)`           | `scalability`            | 1e-10, relative to max(1, value)            |
| `aisp`                       | `automatedItemSelection` | exact                                       |
| `check.monotonicity`         | `monotonicity`           | exact counts; 1e-8 absolute, the violations |
| `check.iio(method = "MIIO")` | `invariantItemOrdering`  | exact items removed; 1e-10 absolute, H^T    |

## Where measurekit differs

Each guide ends with its differences from the references, such as mirt's quadrature and its starting values. Two
differences reach every call: the generator of random numbers is not R's, so a simulation or a bootstrap gives other
draws from the same seed, and every iterative method reports `converged` and `iterations` instead of a warning.
