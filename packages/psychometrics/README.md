# @measurekit/psychometrics

Psychometrics for JavaScript and TypeScript: classical test theory, item response theory (calibration in groups, on
sparse data and online; ability estimation; summed-score tables), cognitive diagnosis, adaptive and multistage testing
with exposure control and form assembly, item and person fit, differential item functioning, linking and equating,
score scaling and prediction, explanatory models of item difficulty, simulation, standard setting, rater agreement and effects, and response times. Every
statistic is validated against R. Its only dependency is `@measurekit/core`, which has none; it runs in any JavaScript
runtime.

```bash
npm install @measurekit/psychometrics
```

## Example

```ts
import {
  classifyDifficulty,
  classifyDiscrimination,
  cronbachAlpha,
  distractorAnalysis,
  itemStatistics,
  scoreResponses,
  standardErrorOfMeasurement,
} from "@measurekit/psychometrics";

// One row per person, one chosen option per item; null when no option was chosen.
const answers = [
  ["A", "C", "B", "D", "A"],
  ["A", "C", "B", "B", "A"],
  ["A", "B", "B", "D", "C"],
  ["B", "C", "D", "D", "A"],
  ["A", "A", "B", "C", "B"],
  ["C", "B", "A", "D", null],
  ["A", "C", "B", "D", "A"],
  ["B", "A", "C", "A", "B"],
];
const test = scoreResponses(answers, ["A", "C", "B", "D", "A"], { itemIds: ["q1", "q2", "q3", "q4", "q5"] });

itemStatistics(test)[0];
// { id: "q1", count: 8, mean: 0.625, sd: 0.52, itemTotal: 0.63, itemRest: 0.42, biserial: 0.84,
//   discriminationIndex: 0.53, alphaIfDeleted: 0.69 }
classifyDifficulty(0.625); // "moderate": the p-value is from 0.30 to 0.70
classifyDiscrimination(0.53); // "very-good": D is at least 0.40

cronbachAlpha(test); // 0.72, which is KR-20 for these 0/1 scores
standardErrorOfMeasurement(test); // 0.94 points

// Count, share and point-biserial correlation of every option, and its share in the upper and lower 27%.
distractorAnalysis(test)[0]?.options;
```

Already scored data goes into `responseMatrix(rows)`, or `responseMatrixFromColumns(columns)`, as integers from 0
with `null` for missing responses. Every function takes plain arrays and objects and returns plain objects.

The same scores calibrate an IRT model, which then estimates abilities. With few persons, priors keep the estimates
finite:

```ts
import {
  betaPrior,
  calibrate,
  logNormalPrior,
  threeParameterLogistic,
  weightedLikelihoodEstimate,
} from "@measurekit/psychometrics";

const irt = calibrate(test, {
  model: threeParameterLogistic,
  priors: { a: logNormalPrior(0, 0.5), c: betaPrior(5, 17) },
  standardErrors: "hessian",
});
irt.items[0]; // { id: "q1", model, parameters: { a: 0.85, b: -0.26, c: 0.2, d: 1 }, standardErrors: { a, b, c } }
weightedLikelihoodEstimate(irt.items, [1, 0, 1, 1, null]); // { theta: 0.64, standardError: 1.47 }
```

The calibrated items then make an adaptive test, which selects each next item from the responses so far:

```ts
import { createRandom } from "@measurekit/psychometrics";
import { adaptiveTest, allOf, anyOf, standardErrorThreshold, testLength } from "@measurekit/psychometrics/cat";

const cat = adaptiveTest({
  pool: irt.items,
  stopping: anyOf(testLength(4), allOf(testLength(2), standardErrorThreshold(0.5))),
});
const random = createRandom(1);
let state = cat.start(random); // state.next: the id of the first item
state = cat.answer(state, 1, random); // the response to it; then the next item, or null at the end
```

## API

| Import                                       | Contents                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `@measurekit/psychometrics/data`             | `responseMatrix`, `responseMatrixFromColumns`, `responseRecords` for sparse data, the `FacetedScore` type, `scoreResponses`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `@measurekit/psychometrics/ctt`              | `itemStatistics`, `distractorAnalysis`, `cronbachAlpha`, `kr20`, `kr21`, `splitHalf`, `mcdonaldOmega`, `standardErrorOfMeasurement`, `classifyDifficulty`, `classifyDiscrimination` and thresholds                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `@measurekit/psychometrics/irt`              | `calibrate` with groups, normal and histogram distributions of theta, testlets, priors by parameter or by item and two kinds of standard errors, `calibrateWithCovariance`, `jointMaximumLikelihood`; the 1PL to 4PL, GRM, GPCM, PCM, RSM and NRM models and the `IrtModel` interface; priors; `maximumLikelihoodEstimate`, `weightedLikelihoodEstimate`, `posteriorModeEstimate`, `posteriorMeanEstimate`, `thetaPosterior`, `abilityEstimates`; `summedScoreTable`; `itemInformation`, `testInformation`, `conditionalStandardError`; `marginalReliability`, `empiricalReliability`                                                                                                 |
| `@measurekit/psychometrics/cat`              | `adaptiveTest` with `maximumInformation`, `kullbackLeibler`, `aStratified` and `randomesque` selection, content balancing, enemy sets, Sympson-Hetter exposure control and `calibrateExposure`, and `testLength`, `standardErrorThreshold`, `informationThreshold`, `allOf`, `anyOf` stopping; `untilMixed`; `diagnosticAdaptiveTest` with `gdinaDiscriminationIndex`, `posteriorWeightedKullbackLeibler`, `modifiedPosteriorWeightedKullbackLeibler`, `diagnosticKullbackLeibler` and `shannonEntropy` selection and `posteriorThreshold`, `masteryThreshold` stopping; `multistageTest`; `simulateAdaptiveTest`, `simulateDiagnosticTest`, `simulateMultistageTest`; `assembleForm` |
| `@measurekit/psychometrics/cdm`              | `calibrateCdm` (G-DINA, DINA, DINO, A-CDM, LLM, R-RUM), `attributeProfiles`, `cdmClassificationAccuracy`, `validateQMatrix`, `nonparametricClassification`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `@measurekit/psychometrics/fit`              | `itemFit` and `personFit` (infit, outfit, lz), `summedScoreFit` (S-X²), `yenQ3`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `@measurekit/psychometrics/dif`              | `mantelHaenszel` with ETS categories, `logisticRegressionDif`, `mantelTest` with the standardized mean difference and ETS categories, `ordinalLogisticRegressionDif`, all on sparse data and with purification; `lordChiSquare`                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `@measurekit/psychometrics/equating`         | `meanMeanLinking`, `meanSigmaLinking`, `haebaraLinking`, `stockingLordLinking`, `transformItem`, `transformTheta`; `linearEquating`, `equipercentileEquating`, `logLinearPresmoothing`, `trueScoreEquating`; `tuckerEquating`, `levineEquating`, `chainedLinearEquating`, `chainedEquipercentileEquating`, `frequencyEstimationEquating`, `bivariateLogLinearPresmoothing`; `irtObservedScoreEquating`; `bootstrapEquating`, `irtEquatingStandardErrors`; `robustZ`                                                                                                                                                                                                                   |
| `@measurekit/psychometrics/explanatory`      | `itemFeatureRegression` of an item parameter on item features by two-stage REML; `predictItemParameter` for new items                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `@measurekit/psychometrics/scaling`          | `scaleScore`; `conversionTable` by the EAP of the summed score or the true score; `conditionalScaleError`; `scaleConversion`; `percentileRanks`; `predictFormScore`, `predictThetaScore`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `@measurekit/psychometrics/sim`              | `simulateThetas`, `simulateItems` of every built-in model, `simulateResponses` with missingness, `simulateCdmResponses`; `normalDistribution`, `logNormalDistribution`, `uniformDistribution`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `@measurekit/psychometrics/standard-setting` | `angoffCutScore`; `orderedItemBooklet`, `bookmarkCutScore`; `hofsteeCutScore`; `classificationAccuracy`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `@measurekit/psychometrics/rater`            | `cohenKappa`, unweighted, linear or quadratic weighted; `fleissKappa`; `intraclassCorrelation`; `manyFacetRasch` with `manyFacetEstimates`, `manyFacetFit` and `expectedRating`: rater severity, item difficulty and fair scores                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `@measurekit/psychometrics/response-time`    | `calibrateResponseTimes` (lognormal model), `speedEstimates`, `responseTimeFit`; `normativeThresholds`, `responseTimeEffort`, `effortModeratedEstimates`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `@measurekit/psychometrics/interop`          | Conversion to and from mirt's coefficients of every model                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `@measurekit/psychometrics`                  | All of the above, `createRandom` (a seeded random number generator), `DataError` and `MeasurekitError`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |

Invalid input throws `DataError`. A statistic that is undefined for the data, such as the correlation of an item
everyone answered correctly, is `null` in a table of statistics, and a `DataError` from a function that returns only
that statistic. Results never contain `NaN`.

## Documentation

- [Data](https://github.com/measurekit/measurekit/blob/main/docs/psychometrics/data.md): response matrices,
  response records for sparse data, missing responses, scoring against a key.
- [Classical test theory](https://github.com/measurekit/measurekit/blob/main/docs/psychometrics/ctt.md):
  every statistic, its formula, missing data and edge cases.
- [Item response theory](https://github.com/measurekit/measurekit/blob/main/docs/psychometrics/irt.md):
  models, calibration, groups and distributions of theta, online calibration, testlets, standard errors, joint
  maximum likelihood, ability estimation, summed-score tables, information, reliability, adding a model.
- [Cognitive diagnosis](https://github.com/measurekit/measurekit/blob/main/docs/psychometrics/cdm.md): G-DINA, DINA,
  DINO, A-CDM, LLM and R-RUM, attribute mastery, classification accuracy, Q-matrix validation, nonparametric
  classification.
- [Adaptive and multistage testing](https://github.com/measurekit/measurekit/blob/main/docs/psychometrics/cat.md):
  adaptive tests and adaptive cognitive diagnosis, item selection, content balancing, enemy sets, exposure control
  and its calibration, stopping rules, multistage panels, simulation, form assembly.
- [Item and person fit](https://github.com/measurekit/measurekit/blob/main/docs/psychometrics/fit.md): infit and
  outfit, S-X², lz, Yen's Q3.
- [Differential item functioning](https://github.com/measurekit/measurekit/blob/main/docs/psychometrics/dif.md):
  Mantel-Haenszel and the ETS categories, logistic regression, the Mantel test and ordinal logistic regression for
  polytomous items, sparse data, purification, Lord's chi-square.
- [Linking and equating](https://github.com/measurekit/measurekit/blob/main/docs/psychometrics/equating.md):
  scale linking from common items, observed-score equating with presmoothing, equating through common items,
  IRT true-score and observed-score equating, standard errors by bootstrap and the delta method, the anchor screen.
- [Simulation](https://github.com/measurekit/measurekit/blob/main/docs/psychometrics/sim.md): thetas, item parameters
  of every model and response matrices, from a seed.
- [Explanatory item response theory](https://github.com/measurekit/measurekit/blob/main/docs/psychometrics/explanatory.md):
  the regression of item difficulties and slopes on item features, and predictions for new items as calibration priors.
- [Score reporting and prediction](https://github.com/measurekit/measurekit/blob/main/docs/psychometrics/scaling.md):
  scales, conversion tables, conditional errors of scale scores, equated scores, percentile ranks, prediction of an
  official score.
- [Standard setting](https://github.com/measurekit/measurekit/blob/main/docs/psychometrics/standard-setting.md):
  modified Angoff, Bookmark and Hofstee cut scores, classification accuracy.
- [Rater agreement and effects](https://github.com/measurekit/measurekit/blob/main/docs/psychometrics/rater.md):
  Cohen's and Fleiss' kappa, intraclass correlations, the many-facet Rasch model.
- [Response times](https://github.com/measurekit/measurekit/blob/main/docs/psychometrics/response-time.md):
  the lognormal model, speed, aberrant times, rapid guessing.
- [Interoperability](https://github.com/measurekit/measurekit/blob/main/docs/psychometrics/interop.md):
  mirt's coefficients.
- [Validation](https://github.com/measurekit/measurekit/blob/main/docs/psychometrics/validation.md): R
  references, datasets and tolerances.
- [Architecture](https://github.com/measurekit/measurekit/blob/main/docs/psychometrics/architecture.md):
  design principles and layout, for contributors.
- [API reference](https://measurekit.github.io/measurekit/): every export, generated by TypeDoc.

## License

[Apache-2.0](LICENSE). Copyright Fariz Wakan; see
[NOTICE](NOTICE). [CONTRIBUTING.md](https://github.com/measurekit/measurekit/blob/main/CONTRIBUTING.md) describes how
to contribute.
