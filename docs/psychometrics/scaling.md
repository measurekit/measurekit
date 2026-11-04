# Score reporting and prediction

`@measurekit/psychometrics/scaling` turns theta and total scores into the scores a test reports, and predicts the
score a person will be reported on an operational form. It covers a linear scale with rounding and bounds, conversion
tables from total scores to scale scores, the conditional standard error of measurement of scale scores, the scale
scores of equated scores, percentile ranks and norms with their standard errors in a norm sample, predictions from a
posterior of theta, and joint draws of theta from separate posteriors. Every function is validated against R ([validation](validation.md)). The growth between the groups of a vertical scale, `scaleGrowth`,
has [its own guide](vertical-scaling.md).

## Quick start

Report two thetas on a scale of 200 to 800 in steps of 10:

```ts
import { scaleScore } from "@measurekit/psychometrics/scaling";

const scale = { slope: 100, intercept: 500, lowest: 200, highest: 800, step: 10 };
scaleScore(0.234, scale); // => 520
scaleScore(-4.2, scale); // => 200
```

## Scales

A scale reports slope θ + intercept, rounded to the nearest multiple of `step`, halves up, and bounded to `lowest` and
`highest`. `scaleScore(theta, scale)` applies it. A multiple of a decimal step such as 0.1 is the double nearest the
decimal, 52.3 rather than 523 · 0.1 = 52.300000000000004.

| Option      | Default     | Valid range              | Meaning                                           |
| ----------- | ----------- | ------------------------ | ------------------------------------------------- |
| `slope`     | required    | positive and finite      | Scale points per unit of theta                    |
| `intercept` | required    | finite                   | The scale score of theta 0                        |
| `lowest`    | `-Infinity` | a number below `highest` | Lower scores are reported as this one             |
| `highest`   | `Infinity`  | a number above `lowest`  | Higher scores are reported as this one            |
| `step`      | `1`         | finite, at least 0       | The rounding increment; 0 leaves scores unrounded |

The same options, without `slope` and `intercept`, are the `ScoreRounding` of `scaleConversion`. Scale scores do not
decrease with theta, and a score that rounds to -0 is reported as 0.

## Conversion tables

`conversionTable(items, scale, options)` converts each total score of a form to theta and to a scale score.

```ts
import { conversionTable } from "@measurekit/psychometrics/scaling";
import { twoParameterLogistic } from "@measurekit/psychometrics/irt";

const items = [
  { model: twoParameterLogistic, parameters: { a: 1.2, b: -0.8, c: 0, d: 1 } },
  { model: twoParameterLogistic, parameters: { a: 0.9, b: 0, c: 0, d: 1 } },
  { model: twoParameterLogistic, parameters: { a: 1.5, b: 0.9, c: 0, d: 1 } },
];
const scale = { slope: 100, intercept: 500, lowest: 200, highest: 800, step: 10 };
conversionTable(items, scale);
// => [{ score: 0, theta: -0.969, standardError: 0.787, scaleScore: 400, scaleStandardError: 47.5 },
//  { score: 1, theta: -0.297, standardError: 0.761, scaleScore: 470, scaleStandardError: 51.0 },
//  { score: 2, theta: 0.331, standardError: 0.756, scaleScore: 530, scaleStandardError: 53.2 },
//  { score: 3, theta: 1.081, standardError: 0.761, scaleScore: 610, scaleStandardError: 52.5 }]
```

The items are those of a calibration, `calibrate(...).items`, or any [items](irt.md#models) of any models.

| Option       | Default               | Meaning                                                                                                 |
| ------------ | --------------------- | ------------------------------------------------------------------------------------------------------- |
| `method`     | `"eap-sum"`           | `"eap-sum"` or `"true-score"`: how the theta of a total score is found                                  |
| `prior`      | `{ mean: 0, sd: 1 }`  | The prior of `"eap-sum"`: a finite mean and a positive sd                                               |
| `quadrature` | 61 rectangular points | The rule of `"eap-sum"`, placed on the prior, as [`summedScoreTable`](irt.md#summed-score-tables)       |
| `testlets`   | `[]`                  | `{ items, sd }` of `"eap-sum"`, whose effects it integrates out: [below](#tables-with-testlets)         |
| `range`      | `[-6, 6]`             | The thetas of the total scores that have none: a lower bound below an upper bound, at most 10,000 apart |

| Method         | Theta of a total score x                                                    | Standard error                                |
| -------------- | --------------------------------------------------------------------------- | --------------------------------------------- |
| `"eap-sum"`    | The posterior mean of theta given x (Thissen et al., 1995): mirt's `EAPsum` | The posterior standard deviation              |
| `"true-score"` | The root of the test characteristic curve τ(θ) = x (Lord, 1980)             | 1 / √I(θ), the test information at that theta |

- **Scores with no theta.** A score takes an end of `range`, with a `null` standard error: the lower end for a score
  below those that have a theta, the upper end otherwise. With `"eap-sum"` these are scores whose probability underflows
  to 0. With `"true-score"` they are scores at or below the sum of the lower asymptotes, which the curve approaches as
  theta falls, or at or above the sum of the upper asymptotes: for a 2PL form, the scores 0 and the highest.
- **True-score forms.** The curve must increase with theta: the built-in models other than the NRM, with positive
  slopes. Other forms throw `DataError` for `"true-score"`; `"eap-sum"` takes items of any model.
- **Scale scores.** `scaleScore` is `scaleScore(theta, scale)` of the row's theta. A table whose scores will be
  interpolated by `scaleConversion` is made with `step: 0`.
- **`scaleStandardError`.** The [conditional standard error](#conditional-standard-errors) at the row's theta of the
  table's own scale scores, rounded and bounded as reported.

## Conditional standard errors

`conditionalScaleError(items, conversion, theta, options)` returns the standard deviation of the scale score of a
person at theta (Kolen, Zeng and Hanson, 1996). With P(x | θ) the probability of the total score x, from the recursion
of Lord and Wingersky (1984), and c(x) its scale score, it is √(Σ c(x)² P(x | θ) - (Σ c(x) P(x | θ))²). With the option
`testlets`, P(x | θ) integrates each testlet's effect out over the rule of the option `quadrature`, 61 rectangular
points by default, as a [table with testlets](#tables-with-testlets) does.

```ts continues
import { conditionalScaleError } from "@measurekit/psychometrics/scaling";

// `items` is the form of the previous example and 400, 470, 530, 610 its table's scale scores.
conditionalScaleError(items, [400, 470, 530, 610], 0); // => 52.1
```

`conversion` holds one finite scale score for each total score from 0 to the form's highest, the sum of its items'
highest categories; any other length throws `DataError`. For the unrounded scale score of a theta estimate, the error
is the slope times `conditionalStandardError` of the [irt](irt.md#information) module. The recursion takes
O(items × scores) per theta, and a table does all its rows in one pass.

## Scale scores of equated scores

Equating gives each score of form X an equivalent on form Y, usually not a whole number. `scaleConversion(equivalents,
conversion, options)` reports it on Y's scale by linear interpolation between the neighboring scores of Y's table
(Kolen and Brennan, 2014), holding the end values for equivalents below 0 and above Y's highest score, and then rounds
and bounds as `options` say: the `lowest`, `highest` and `step` of [a scale](#scales).

```ts continues
import { linearEquating } from "@measurekit/psychometrics/equating";
import { conversionTable, scaleConversion } from "@measurekit/psychometrics/scaling";

// The number of persons at each total score of forms X and Y, 0 to 3, taken by random groups.
const equated = linearEquating([2, 8, 20, 10], [4, 11, 22, 8]); // => [-0.295, 0.757, 1.808, 2.860]
// Y's unrounded scale scores; `items` and `scale` are those of the first example, for form Y.
const unrounded = conversionTable(items, { ...scale, step: 0 }).map((row) => row.scaleScore);
const { lowest, highest, step } = scale;
scaleConversion(equated, unrounded, { lowest, highest, step }); // => [400, 450, 520, 600]
```

Interpolate unrounded scores: rounding Y's table first rounds twice. Any equating that returns equivalents on Y's raw
scale, including [NEAT equating](equating.md#equating-through-common-items), takes this route.

## Tables with testlets

A form whose items include a calibrated [testlet](irt.md#testlets), such as the questions on a passage, takes the
testlet response model for its table: `testlets` lists `{ items, sd }`, a calibration's `testlets`, and `"eap-sum"`
takes [`summedScoreTable`](irt.md#tables-with-testlets) with each testlet's effect integrated out, as does the error of
the scale scores in every row. Taken as independent, the items of a testlet overstate the precision of the score.

```ts
import { twoParameterLogistic } from "@measurekit/psychometrics/irt";
import { conditionalScaleError, conversionTable } from "@measurekit/psychometrics/scaling";

const item = (id: string, a: number, b: number) => ({
  id,
  model: twoParameterLogistic,
  parameters: { a, b, c: 0, d: 1 },
});
const form = [
  item("q1", 1.2, -0.5),
  item("q2", 0.8, 0.4),
  item("p1", 1.1, -1),
  item("p2", 0.9, 0),
  item("p3", 1.4, 0.6),
];
const testlets = [{ items: ["p1", "p2", "p3"], sd: 0.8 }];
const table = conversionTable(form, { slope: 100, intercept: 500, lowest: 200, highest: 800, step: 10 }, { testlets });
const conversion = table.map((row) => row.scaleScore); // => [380, 430, 470, 520, 560, 610]

conditionalScaleError(form, conversion, 0, { testlets }); // => 51.8
conditionalScaleError(form, conversion, 0); // => 47.7, the items taken as independent
```

`"true-score"` takes no testlets: its curve and its standard error 1 / √I(θ) take the items as independent, and the
information of a testlet sums over its response patterns, not its items.

## Norms

`percentileRanks(sample, scores)` gives the percentile rank in a norm sample of each score: the percentage of the
sample below it plus half the percentage at it, 100 (#{s < x} + #{s = x} / 2) / n. At integer scores it is Kolen and
Brennan's percentile rank, and the `px` of the equate package times 100.

```ts
import { percentileRanks } from "@measurekit/psychometrics/scaling";

const sample = [420, 480, 480, 510, 540, 600, 600, 600, 650, 720];
percentileRanks(sample, [480, 500, 600, 800]); // => [20, 30, 65, 100]
```

The sample is sorted once and each rank found by binary search, so n scores and m ranks take O((n + m) log n). It needs
at least one finite score, and the ranks are on the sample's own scale: norms of scale scores take a sample of scale
scores and, as `scores`, a table's.

A norm sample drawn by a survey design stands for its population only through its weights. With `personWeights`, a
positive survey weight for each score of the sample, the rank is the weighted share: 100 (W(s < x) + W(s = x) / 2) / W,
the mean of Hmisc's weighted distribution function `wtd.Ecdf` at x and just below it. Equal weights give the unweighted
ranks, a whole weight counts a person as many times, and only the ratios of the weights matter.

```mermaid
flowchart LR
  S["Norm sample: scores and person weights"] --> O["Sorted once: the weight below and at each score"]
  O --> R["Rank of x: 100 (W below + W at / 2) / W"]
  S --> N["normStatistics: weighted mean, SD, z, ranks, stanines"]
  N --> E["Standard errors: the delta method over the weights"]
```

```ts continues
// Three persons sampled from a stratum at a third of the others' rate stand for three persons each.
const personWeights = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1];
percentileRanks(sample, [480, 500, 600, 800], { personWeights }); // => [37.5, 56.25, 78.125, 100]
```

`normStatistics(scores, options)` gives the norms of a sample with the standard errors that a norm sample's size
leaves, as mokken's `check.norms` (Oosterhuis, van der Ark and Sijtsma, 2016), step 9 of a
[Mokken scale analysis](mokken.md). Each is an `EstimateInterval`: `estimate`, `standardError`, `lower` and `upper`.

| Field      | Estimate                                                                                                                        |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `mean`     | The mean of the scores                                                                                                          |
| `sd`       | Their standard deviation, with denominator n - 1                                                                                |
| `scores`   | For each distinct score from the lowest: `score`, its `z`, (score - mean) / SD, and its `percentile` rank, as `percentileRanks` |
| `stanines` | The eight boundaries of the stanines, mean + c SD for c from -1.75 to 1.75 by 0.5                                               |

| Option          | Default                      | Meaning                                                                              |
| --------------- | ---------------------------- | ------------------------------------------------------------------------------------ |
| `level`         | `0.95`                       | The level of the intervals, above 0 and below 1: estimate ± its normal quantile × SE |
| `personWeights` | every person weighs the same | A positive survey weight for each score, as `percentileRanks` takes it               |

The standard errors are the delta method over the numbers of persons with each score, taken as multinomial: for a
statistic f of the counts n_k, Var f = Σ n_k (∂f/∂n_k)² - (Σ n_k ∂f/∂n_k)² / N. The mean's is the SD with
denominator N over √N.

With `personWeights`, each person's weight w_l enters a statistic f through the total weight at each score, and
Var f = Σ_l (u_l - ū)² with u_l = w_l ∂f/∂w_l: with equal weights, the multinomial delta method above. It is the
linearization of a survey's estimators with one stage and no strata, as the survey package's `svymean` computes it,
without its factor n / (n - 1). The weighted variance is n / (n - 1) times the weighted mean of the squared deviations,
for n persons, as `svyvar` defines it. The standard errors are those of persons drawn independently with those weights:
the strata and clusters of a design change them, and need the design itself ([not included](#not-included)).

```ts continues
import { normStatistics } from "@measurekit/psychometrics/scaling";

const norms = normStatistics(sample);
norms.mean; // => { estimate: 560, standardError: 27.166, lower: 506.755, upper: 613.245 }
norms.sd.estimate; // => 90.554, with a standard error of 15.719
norms.scores[1]?.z; // => { estimate: -0.883, standardError: 0.314, lower: -1.499, upper: -0.267 }: of the score 480
norms.scores[1]?.percentile; // => { estimate: 20, standardError: 10.488, lower: -0.556, upper: 40.556 }
norms.stanines.map(({ estimate }) => estimate); // => [401.531, 446.808, 492.085, 537.362, 582.638, 627.915, 673.192, 718.469]
```

The scores are finite numbers, at least two and not all equal. A sample of ten gives wide intervals: the percentile of
480 above runs from -0.6 to 40.6, which is why norms need large samples.

```ts continues
const weighted = normStatistics(sample, { personWeights });
weighted.mean.estimate; // => 522.5, below 560 as the low scores weigh more
weighted.scores[1]?.percentile; // => { estimate: 37.5, standardError: 13.666, lower: 10.715, upper: 64.285 }
```

## Predicting an operational score

A learner practices on a bank of items and then takes the operational test. The practice responses give a posterior
distribution of theta, a `ThetaDistribution` of `nodes` and `weights` that sum to 1. The prediction is the distribution
of the score the operational test will report, treating theta as constant.

| Function                                        | Operational test                         | Prediction by                                                       |
| ----------------------------------------------- | ---------------------------------------- | ------------------------------------------------------------------- |
| `predictFormScore(posterior, form, conversion)` | A fixed form scored by total score       | P(x) = Σ_q w_q P(x \| θ_q), converted by the form's table           |
| `predictThetaScore(posterior, scale, options)`  | A test that reports theta, such as a CAT | The estimate θ + e, e ~ N(0, se²), rounded and bounded by the scale |

```ts
import { thetaPosterior, twoParameterLogistic } from "@measurekit/psychometrics/irt";
import { conversionTable, predictFormScore, predictThetaScore } from "@measurekit/psychometrics/scaling";

const item = (a: number, b: number) => ({ model: twoParameterLogistic, parameters: { a, b, c: 0, d: 1 } });
const practice = [item(1, -1), item(1.2, -0.2), item(0.9, 0.5), item(1.4, 1.1)];
const operational = [item(1.1, -0.5), item(1.3, 0.4), item(0.8, 1)];
const scale = { slope: 100, intercept: 500, lowest: 200, highest: 800, step: 10 };

const posterior = thetaPosterior(practice, [1, 1, 0, 0]);
const conversion = conversionTable(operational, scale).map((row) => row.scaleScore); // => [410, 480, 540, 600]
predictFormScore(posterior, operational, conversion);
// => { mean: 501.5, lower: 410, upper: 600,
//   distribution: [{ score: 410, probability: 0.180 }, { score: 480, probability: 0.373 },
//                  { score: 540, probability: 0.325 }, { score: 600, probability: 0.122 }] }
predictThetaScore(posterior, scale, { standardError: 0.3, level: 0.9 }); // => { mean: 508.5, lower: 380, upper: 640, ... }
```

| Option          | Of                  | Default | Valid range        | Meaning                                              |
| --------------- | ------------------- | ------- | ------------------ | ---------------------------------------------------- |
| `level`         | both                | `0.8`   | above 0, below 1   | The probability of the central interval              |
| `standardError` | `predictThetaScore` | `0`     | finite, at least 0 | The standard error of the operational theta estimate |

| Result         | Meaning                                                                                       |
| -------------- | --------------------------------------------------------------------------------------------- |
| `mean`         | The expected reported score                                                                   |
| `lower`        | The lowest score with at least (1 - `level`) / 2 of the probability at or below it            |
| `upper`        | The lowest score with at least (1 + `level`) / 2 of the probability at or below it            |
| `distribution` | `{ score, probability }` for each reportable score, in ascending order of score, summing to 1 |

- **Form scores.** `conversion` has one finite scale score for each total score of `form`. Total scores with equal scale
  scores pool their probabilities, so `distribution` lists the distinct scale scores.
- **Theta scores.** The reportable scores are `lowest`, `highest` and the multiples of `step` between them; the estimates
  that round and bound to a score s form an interval of theta, so that P(s) = Σ_q w_q [Φ((t_hi - θ_q) / se) - Φ((t_lo
  - θ_q) / se)]. With `standardError` 0 each node's weight goes to the score of its own theta. The scale needs finite
  `lowest` and `highest` and a `step` above 0 that give at most 10,000 reportable scores, about (`highest` - `lowest`) /
  `step`, the entries of `distribution`; a scale of more throws a `DataError`.
- **Posteriors.** Any distribution with finite `nodes` and non-negative `weights` that sum to 1 within 1e-6 serves. Its
  probabilities are scaled to sum to 1.

Where the posterior comes from:

| Source                      | Call                                                                                                  |
| --------------------------- | ----------------------------------------------------------------------------------------------------- |
| Practice responses          | `thetaPosterior(practiceItems, responses, { prior })` of [irt](irt.md#posterior-distribution)         |
| A knowledge tracing learner | `model.posterior(state)` of [`@measurekit/knowledge-tracing/irt`](../knowledge-tracing/irt.md#online) |
| No practice                 | A calibration's `groups[0]`, a `GroupDistribution`: the score of a person drawn from the population   |

```ts continues
import { irt } from "@measurekit/knowledge-tracing/irt";
import { predictFormScore } from "@measurekit/psychometrics/scaling";

// `operational` and `conversion` are those of the previous example; q1 to q4 are the practice items.
const model = irt({
  parameters: {
    items: { q1: { a: 1, b: -1 }, q2: { a: 1.2, b: -0.2 }, q3: { a: 0.9, b: 0.5 }, q4: { a: 1.4, b: 1.1 } },
  },
});
const answers: [string, number][] = [
  ["q1", 1],
  ["q2", 1],
  ["q3", 0],
  ["q4", 0],
];
let ana = model.initialState("ana");
answers.forEach(([item, outcome], time) => {
  ana = model.update(ana, { learner: "ana", item, skills: [], time, outcome });
});
predictFormScore(model.posterior(ana), operational, conversion).mean; // => 501.5, as thetaPosterior gives
```

The libraries meet in data, not imports: both return `{ nodes, weights }`. The prediction is valid only if the practice
items and the operational form are on one theta scale. Calibrate the practice items with operational items as anchors, or
link their scale first ([linking](equating.md#linking)); the prediction treats theta as constant between practice and
the test, so pass recent responses where learning is likely.

## Joint draws of theta

A host holds one posterior per section or person, computed apart, and needs the distribution of something that combines
them: a composite score, a pass on every section, the reliability of a composite. `drawJointThetas(posteriors,
correlation, draws, random)` draws thetas from all the posteriors at once, coupled by a Gaussian copula: each posterior
keeps its own distribution, and `correlation` says how the thetas move together.

```mermaid
flowchart LR
  E["independent normals e<br/>random.normal()"] --> Z["correlated scores z = F e<br/>F Fᵀ = correlation"]
  Z --> U["probabilities u = Φ(z)"]
  U --> T["theta = inverse distribution<br/>function of each posterior at u"]
  P["posteriors<br/>nodes and weights"] --> T
```

```ts
import { createRandom } from "@measurekit/psychometrics";
import { thetaPosterior, twoParameterLogistic } from "@measurekit/psychometrics/irt";
import { drawJointThetas } from "@measurekit/psychometrics/scaling";

const item = (a: number, b: number) => ({ model: twoParameterLogistic, parameters: { a, b, c: 0, d: 1 } });
const quadrature = { points: 241 };
const reading = thetaPosterior([item(1, -1), item(1.2, -0.2), item(0.9, 0.5), item(1.4, 1.1)], [1, 1, 0, 0], {
  quadrature,
});
const writing = thetaPosterior([item(1.1, -0.6), item(0.8, 0.3), item(1.3, 0.9)], [1, 0, 0], { quadrature });

// The mean and the SD of the sum of the two thetas at the correlations 0 and 1, which bound it for any correlation.
[0, 1].map((r) => {
  const draws = drawJointThetas(
    [reading, writing],
    [
      [1, r],
      [r, 1],
    ],
    10_000,
    createRandom(1),
  );
  const sums = draws.map(([x, y]) => (x ?? 0) + (y ?? 0));
  const mean = sums.reduce((total, sum) => total + sum, 0) / sums.length;
  const sd = Math.sqrt(sums.reduce((total, sum) => total + (sum - mean) ** 2, 0) / (sums.length - 1));
  return [mean, sd];
}); // => [[-0.123, 1.054], [-0.139, 1.474]]
```

| Argument      | Valid range                                                                                                       | Meaning                                             |
| ------------- | ----------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| `posteriors`  | at least one `ThetaDistribution`: finite, strictly ascending `nodes`, non-negative `weights`                      | The distribution of theta of each person or section |
| `correlation` | a symmetric matrix by rows, one row per posterior, 1 on the diagonal, entries from -1 to 1, positive semidefinite | The correlation of the normal scores z              |
| `draws`       | a positive integer                                                                                                | The number of draws                                 |
| `random`      | a `Random`, as `createRandom` returns it                                                                          | The source of the normals                           |

The result holds one array for each draw, with a theta for each posterior in the order given. All four arguments are
required. A draw takes as many normals from `random` as there are posteriors, whatever the correlation, so a seed gives the
same draws for the same posteriors.

| Correlation | Draws                                                                                   |
| ----------- | --------------------------------------------------------------------------------------- |
| 1           | One quantile in every posterior: the largest variance that a sum of the thetas can have |
| 0           | Independent draws                                                                       |
| -1          | Opposite quantiles: u in one posterior and 1 - u in the other (two posteriors)          |
| between     | Rank correlation (6 / π) asin(r / 2) for any posteriors                                 |

- **Reading a posterior.** The distribution function is linear within the cell of each node. A cell runs from the midpoint
  with the node below to the midpoint with the node above, and the outer cells are mirrored about their nodes; a node with
  no weight is never drawn, and one node gives that node. A cell of width h adds h² / 12 to the variance: 0.0033 on the 61
  points of spacing 0.2 that `thetaPosterior` returns, 3.7% of a variance of 0.09, and 0.0002 on 241 points. Pass 241 or
  more points for a posterior with a standard deviation under 0.5.
- **The correlation of the thetas.** `correlation` is that of the normal scores, which is the correlation of the thetas when
  the posteriors are near normal. A correlation of 1 is exact: the matrix is factored by its eigenvalues, and one below
  1e-10 in absolute value is taken as 0, so that rounding does not part the draws. A matrix with an eigenvalue below
  -1e-10 throws `DataError` naming it, and [`smoothCorrelations`](dimensionality.md) gives the nearest valid one.
- **What separate posteriors leave out.** A posterior of one section says nothing of another, so the copula carries only the
  correlation you give. Until one is estimated, report a forecast at the correlations 0 and 1. A joint model of the
  dimensions gives their joint posterior and covariance through the [multidimensional estimates](multidimensional.md#scoring);
  this function serves posteriors that were computed apart.

| Failure                                                                                                      | Result                                         |
| ------------------------------------------------------------------------------------------------------------ | ---------------------------------------------- |
| No posteriors; weights that are negative or do not sum to 1 within 1e-6; nodes that do not strictly ascend   | `DataError` naming the posterior               |
| A correlation that is not square of the posteriors' number, symmetric, 1 on the diagonal and within -1 and 1 | `DataError` naming the entry                   |
| An eigenvalue of the correlation below -1e-10                                                                | `DataError` naming it and `smoothCorrelations` |
| `draws` not a positive integer; `random` not a generator                                                     | `DataError`                                    |

The matrix is checked as symmetric and 1 on the diagonal to within 1e-8. A draw costs the square of the posteriors' number
plus a binary search in each, and 2,000 draws from 6 posteriors of 241 nodes take under a millisecond.

## Recipe: plausible values

A survey reports the distribution of theta in groups and its relation to other variables, not a score for each person. A
point estimate such as the EAP shrinks toward its prior, so the variance of EAPs is too small and their relation to a
variable the prior leaves out is too weak. Plausible values (Mislevy, 1991) are random draws from each person's posterior:
a statistic computed from sets of them is unbiased if the prior of every person conditions on all the variables that the
analysis uses (Mazzeo, 2017; Wu, 2005). The recipe is `calibrate` with `covariates`, `thetaPosterior` for the prior of
each person, and `drawJointThetas` with one posterior and the correlation `[[1]]`:

```mermaid
flowchart LR
  C["calibrate with covariates:<br/>items, β and σ"] --> P["Each person's prior N(xᵀβ, σ²)"]
  R["Responses"] --> T["thetaPosterior<br/>for each person"]
  P --> T
  T --> D["drawJointThetas([posterior], [[1]], m, random)"]
  D --> V["m plausible values for each person"]
  V --> S["The statistic of each set of values"]
  S --> B["Rubin's rules"]
```

```ts
import { createRandom } from "@measurekit/psychometrics";
import { calibrate, thetaPosterior, twoParameterLogistic } from "@measurekit/psychometrics/irt";
import { drawJointThetas } from "@measurekit/psychometrics/scaling";
import { normalDistribution, simulateItems, simulateResponses, simulateThetas } from "@measurekit/psychometrics/sim";

// 1,500 persons answer 20 items. Their theta is 0.6 x + e, e ~ N(0, 0.8²), with the covariate x centered.
const random = createRandom(2026);
const items = simulateItems(twoParameterLogistic, 20, random);
const raw = simulateThetas(1500, random);
const center = raw.reduce((sum, value) => sum + value, 0) / raw.length;
const x = raw.map((value) => value - center);
const noise = simulateThetas(1500, random, { distribution: normalDistribution(0, 0.8) });
const thetas = x.map((value, person) => 0.6 * value + (noise[person] ?? 0));
const responses = simulateResponses(items, thetas, random);

// Free slopes fix σ at 1, so β is in units of the residual SD: 0.6 / 0.8 = 0.75.
const fit = calibrate(responses, { model: twoParameterLogistic, covariates: { x } });
const coefficient = fit.regression?.coefficients.x ?? 0; // => 0.72
const residualSd = fit.regression?.residualSd ?? 1;

// Five plausible values for each person: draws from the posterior given the responses and the person's own prior.
const plausibleValues = responses.responses.map((row, person) => {
  const prior = { mean: coefficient * (x[person] ?? 0), sd: residualSd };
  const posterior = thetaPosterior(fit.items, row, { prior, quadrature: { points: 241 } });
  return drawJointThetas([posterior], [[1]], 5, random).map(([theta]) => theta ?? 0);
});
plausibleValues[0]; // => [-3.04, -2.41, -2.36, -2.92, -3.25]
```

| Choice          | Value                                                                                                                             |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Prior           | `{ mean, sd }` of the person: the group's `mean` and `sd` of a calibration with `groups`, or xᵀβ and `residualSd` of `regression` |
| Quadrature      | 241 points, as [joint draws](#joint-draws-of-theta) advise: a cell of width h adds h² / 12 to the variance of a draw              |
| Number of draws | 5 to 10 sets are usual for a survey; each set is analyzed apart                                                                   |
| Items           | Known: the uncertainty of the item parameters is not in the draws                                                                 |
| Weights         | The analysis applies them; the draws do not depend on them                                                                        |

Without the covariates the prior is N(0, 1) for everyone, and the relation to x is understated: the correlation of the first
set of values with x is 0.47 where the simulated thetas' own is 0.58, and 0.59 with the covariates. The prior conditions
on every variable of the analysis for that reason.

## Recipe: Rubin's rules

The analysis of each set of plausible values gives an estimate and the sampling variance of that estimate, as if the
values were the persons' thetas. Rubin's (1987) rules combine the m sets, with Ū the mean of the variances, B the variance
of the estimates and m the number of sets:

| Quantity           | Definition                                                                                         |
| ------------------ | -------------------------------------------------------------------------------------------------- |
| Estimate           | The mean of the estimates                                                                          |
| Within variance Ū  | The mean of the sampling variances                                                                 |
| Between variance B | The sample variance of the estimates, denominator m - 1                                            |
| Total variance     | Ū + (1 + 1 / m) B; its square root is the standard error                                           |
| Degrees of freedom | (m - 1) (1 + 1 / r)², with r = (1 + 1 / m) B / Ū, as `mice::pool` when the complete data are large |

```ts continues
const average = (values: readonly number[]) => values.reduce((sum, value) => sum + value, 0) / values.length;
const sampleVariance = (values: readonly number[]) => {
  const center = average(values);
  return values.reduce((sum, value) => sum + (value - center) ** 2, 0) / (values.length - 1);
};

/** Rubin's rules for the estimates of the sets and their sampling variances. */
function combine(estimates: readonly number[], variances: readonly number[]) {
  const sets = estimates.length;
  const within = average(variances);
  const between = sampleVariance(estimates);
  const total = within + (1 + 1 / sets) * between;
  const relativeIncrease = ((1 + 1 / sets) * between) / within;
  return {
    estimate: average(estimates),
    within,
    between,
    total,
    standardError: Math.sqrt(total),
    degreesOfFreedom: (sets - 1) * (1 + 1 / relativeIncrease) ** 2,
  };
}

// The mean of theta, with the sampling variance of a mean, in each of the five sets.
const sets = [0, 1, 2, 3, 4].map((set) => plausibleValues.map((values) => values[set] ?? 0));
combine(
  sets.map(average),
  sets.map((values) => sampleVariance(values) / values.length),
);
// => { estimate: 0.0075, within: 0.0010, between: 0.00016, total: 0.0012, standardError: 0.035, degreesOfFreedom: 156.8 }
```

The same function combines any statistic that has a sampling variance, such as the slope of theta on x with the variance of
a regression. With no difference between the sets, B is 0 and the degrees of freedom are infinite. Strata, clusters and
replicate weights of a survey design enter the variance of each set, which the caller computes ([not
included](#not-included)).

## Errors

Invalid input throws `DataError` naming the value found, also input of another type, as JavaScript or JSON passes it.

| Function                                                   | Error                                                                                                                                    | What to do                                                                       |
| ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `scaleScore`, `conversionTable`, `predictThetaScore`       | A scale without a positive finite slope and a finite intercept, `lowest` not below `highest`, or a negative `step`                       | Give the scale's slope, intercept and rounding                                   |
| `scaleScore`, `conditionalScaleError`                      | A theta that is not finite                                                                                                               | Give an estimate of theta                                                        |
| `conversionTable`                                          | No items, an invalid item, an invalid prior, quadrature or range, or `"true-score"` for a form whose curve does not rise                 | Use `"eap-sum"` for an NRM item or a slope a ≤ 0                                 |
| `conversionTable`                                          | `"true-score"` with testlets                                                                                                             | Use `"eap-sum"`, which integrates the testlets' effects out                      |
| `conversionTable`, `conditionalScaleError`                 | Testlets that are not an array of `{ items, sd }`, an sd not above 0, an item in two testlets, or items without ids                      | Give the items' ids and a calibration's `testlets`                               |
| `conditionalScaleError`, `predictFormScore`                | A conversion other than one finite scale score for each total score of the form                                                          | Give the scale scores of the totals 0 to the highest, as a conversion table does |
| `scaleConversion`                                          | Equivalents or a conversion that are not finite numbers, or an empty conversion                                                          | Give the equated scores and the form's unrounded scale scores                    |
| `scaleGrowth`                                              | Fewer than two groups, a label that is empty or repeats, a mean that is not finite, or a standard deviation not above 0                  | Give each grade its own label and moments, in the order of the scale             |
| `percentileRanks`, `normStatistics`                        | A sample without finite scores, or, for norms, fewer than two scores or scores that do not vary                                          | Give the norm sample's scores                                                    |
| `percentileRanks`, `normStatistics`                        | `personWeights` that are not a positive number for each score of the sample                                                              | Give a survey weight above 0 for each person                                     |
| `normStatistics`, `predictFormScore`, `predictThetaScore`  | A `level` not between 0 and 1                                                                                                            | Give a probability such as 0.95                                                  |
| `predictFormScore`, `predictThetaScore`, `drawJointThetas` | A distribution of theta whose nodes are not finite or whose weights do not sum to 1 within 1e-6                                          | Pass a posterior such as `thetaPosterior` returns                                |
| `predictThetaScore`                                        | A scale without finite `lowest` and `highest` and a `step` above 0, or of more than 10,000 reportable scores                             | Bound the scale and use a step that reports at most 10,000 scores                |
| `drawJointThetas`                                          | No posteriors, nodes that do not ascend, or a correlation that is not a symmetric matrix of one row per posterior with 1 on the diagonal | Give the correlation of the normal scores, one row per posterior                 |
| `drawJointThetas`                                          | A correlation with an eigenvalue below -1e-10                                                                                            | Pass it through `smoothCorrelations` of the dimensionality module                |

## Not included

| Not included                                            | Why                                                                                                                                                                                        |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Smoothed or continuous norming                          | Outside the sources: a clinical practice with competing models (`cNORM`); it needs a norming model, not a rank.                                                                            |
| The standard errors of a stratified or clustered design | No fit: the design's strata and clusters, which the survey package's `svydesign` describes, set them; the weights alone give those of independent draws.                                   |
| Growth during practice                                  | No fit: learning models live in `@measurekit/knowledge-tracing`.                                                                                                                           |
| Model-based percentiles Φ((θ - μ) / σ)                  | No fit: one line with a `GroupDistribution`'s `mean` and `sd`.                                                                                                                             |
| Scale anchoring                                         | No reference: descriptive work on a calibrated bank with no statistic of its own.                                                                                                          |
| Item mapping                                            | No reference: descriptive work on a calibrated bank with no statistic of its own.                                                                                                          |
| A copula other than the Gaussian                        | Outside the sources: the correlation bounds of comonotonicity and independence hold for any copula, but the Gaussian one is the usual choice for separate posteriors and needs one matrix. |
| Rubin's combining rules                                 | No fit: a few lines over the plausible values, [a recipe](#recipe-rubins-rules).                                                                                                           |
| Replicate weights                                       | No fit: a loop over `calibrate` that the program's design sets.                                                                                                                            |

## Differences from the references

| Topic                       | This package                                                    | Reference                                                                                                  |
| --------------------------- | --------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Rounding of halves          | Up: 525 to 530 with a step of 10                                | R's `round` rounds halves to even; testing programs differ                                                 |
| EAP of the summed score     | 61 rectangular points on the prior, any prior                   | mirt's `fscores(method = "EAPsum")`: the model's own latent distribution                                   |
| Tables with testlets        | The testlet model: one effect of SD σ added to theta            | mirt's `fscores(method = "EAPsum_2.0")`: a bifactor model, whose specific slopes a σ give the same tables  |
| True-score theta            | The root of the curve; the range's end where there is none      | A procedure of each program for scores below the guessing level                                            |
| Percentile ranks            | Of the sample's scores as given; a step function between scores | Kolen and Brennan interpolate linearly within each integer score                                           |
| Intervals of norms          | The normal quantile of the level                                | `check.norms` uses 1.96                                                                                    |
| Weighted standard errors    | The delta method over the weights, Σ (u_l - ū)²                 | survey's `svymean` and `svyvar` multiply the variance by n / (n - 1)                                       |
| Weighted SD                 | `svyvar`'s: n / (n - 1) × the weighted mean square              | Hmisc's `wtd.var(normwt = TRUE)` divides by W - Σ w² / W, as reliability weights do                        |
| Standard errors of z scores | The delta method                                                | `check.norms` subtracts the term (Σ n_k ∂f/∂n_k)² / N twice, which lowers them by up to 3e-6 of their size |
| Error of reported scores    | Of the rounded and bounded scores of the table                  | Kolen, Zeng and Hanson give the error before rounding too: slope × SEM(θ)                                  |
| Joint draws                 | A Gaussian copula over separate posteriors                      | mirt's `fscores(plausible.draws =)`: one model's joint posterior                                           |
| Plausible values            | The posterior on a grid of the person's own prior, exact        | mirt: Metropolis-Hastings, or a normal at the EAP with covariates; `tam.pv` adds 0.03 at its default grid  |

## References

- Mazzeo, J. (2017). Large-scale group-score assessments. In W. J. van der Linden (Ed.), Handbook of item response theory,
  volume 3: Applications (pp. 297-312). CRC Press.
- Mislevy, R. J. (1991). Randomization-based inference about latent variables from complex samples. Psychometrika, 56(2),
  177-196.
- Rubin, D. B. (1987). Multiple imputation for nonresponse in surveys. Wiley.
- Wu, M. (2005). The role of plausible values in large-scale surveys. Studies in Educational Evaluation, 31(2-3), 114-128.
- Kolen, M. J., & Brennan, R. L. (2014). Test equating, scaling, and linking: Methods and practices (3rd ed.). Springer.
- Kolen, M. J., Zeng, L., & Hanson, B. A. (1996). Conditional standard errors of measurement for scale scores using IRT.
  Journal of Educational Measurement, 33(2), 129-140.
- Cai, L. (2015). Lord-Wingersky algorithm version 2.0 for hierarchical item factor models with applications in test
  scoring, scale alignment, and model fit testing. Psychometrika, 80(2), 535-559.
- Cario, M. C., & Nelson, B. L. (1997). Modeling and generating random vectors with arbitrary marginal distributions and
  correlation matrix (technical report). Northwestern University.
- Dhaene, J., Denuit, M., Goovaerts, M. J., Kaas, R., & Vyncke, D. (2002). The concept of comonotonicity in actuarial science
  and finance: Theory. Insurance: Mathematics and Economics, 31(1), 3-33.
- Lord, F. M. (1980). Applications of item response theory to practical testing problems. Lawrence Erlbaum.
- Lord, F. M., & Wingersky, M. S. (1984). Comparison of IRT true-score and equipercentile observed-score "equatings".
  Applied Psychological Measurement, 8(4), 453-461.
- Nelsen, R. B. (2006). An introduction to copulas (2nd ed.). Springer.
- Oosterhuis, H. E. M., van der Ark, L. A., & Sijtsma, K. (2016). Sample size requirements for traditional and
  regression-based norms. Assessment, 23(2), 191-202.
- Petersen, N. S., Kolen, M. J., & Hoover, H. D. (1989). Scaling, norming, and equating. In R. L. Linn (Ed.),
  Educational measurement (3rd ed., pp. 221-262). American Council on Education.
- Thissen, D., Pommerich, M., Billeaud, K., & Williams, V. S. L. (1995). Item response theory for scores on tests
  including polytomous items with ordered responses. Applied Psychological Measurement, 19(1), 39-49.
