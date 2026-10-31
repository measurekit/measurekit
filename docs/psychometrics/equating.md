# Linking and equating

`@measurekit/psychometrics/equating` puts the item parameters of two calibrations on one theta scale (linking)
and converts the scores of one form to the scale of another (equating). Linking is validated against equateIRT and
base R, observed-score equating, with or without common items, and its bootstrap standard errors against the equate
package and base R, and true-score equating against base R and equateIRT ([validation](validation.md)).

## Linking

Each calibration fixes theta ~ N(0, 1) in its own group, so the same item gets different parameters from groups of
different ability. A linear transformation θ_to = A θ_from + B, a `ScaleTransformation` `{ slope, intercept }`, maps
one scale to the other. The linking functions estimate it from common items: `from[j]` and `to[j]` are the same item
calibrated on the two scales, with the same model and number of categories.

```ts
import { haebaraLinking, transformItem, transformTheta, trueScoreEquating } from "@measurekit/psychometrics/equating";
import { threeParameterLogistic } from "@measurekit/psychometrics/irt";

const item = (a: number, b: number, c: number) => ({ model: threeParameterLogistic, parameters: { a, b, c, d: 1 } });
// Two common items as calibrated with form X and with form Y; the forms' other items are left out here.
const commonOnX = [item(1.1, -0.4, 0.2), item(0.9, 0.6, 0.15)];
const commonOnY = [item(0.95, -0.2, 0.2), item(0.8, 0.85, 0.15)];
const formX = [...commonOnX, item(1.3, 0.1, 0.25)];
const formY = [...commonOnY, item(1.0, -0.9, 0.2)];

const transformation = haebaraLinking(commonOnX, commonOnY); // { slope, intercept, converged, iterations }
const formXOnY = formX.map((x) => transformItem(x, transformation));
transformTheta(0.5, transformation); // a theta of form X's calibration on form Y's scale
trueScoreEquating(formXOnY, formY); // the form Y true score of each form X score, 0 to 3
```

| Function                                 | A and B                                                                                |
| ---------------------------------------- | -------------------------------------------------------------------------------------- |
| `meanMeanLinking(from, to)`              | A = mean a_from / mean a_to; B = mean b_to - A mean b_from (Loyd and Hoover, 1980)     |
| `meanSigmaLinking(from, to)`             | A = sd b_to / sd b_from; B as mean-mean (Marco, 1977)                                  |
| `haebaraLinking(from, to, options)`      | Minimize Σ_θ w(θ) Σ_j Σ_k (P_jk(θ; to) - P_jk(θ; from moved))² (Haebara, 1980)         |
| `stockingLordLinking(from, to, options)` | Minimize Σ_θ w(θ) (Σ_j E_j(θ; to) - Σ_j E_j(θ; from moved))² (Stocking and Lord, 1983) |

### Moment methods

a is each item's slope; b its locations: the difficulty, every threshold of a GRM, GPCM or PCM item, and b + τ_k of an
RSM item, so mixed formats pool them (Kim and Lee, 2006). Standard deviations have denominator n - 1. NRM items have no
locations, so they throw `DataError`. For Rasch items, a = 1 on both scales, mean-mean gives A = 1.

### Characteristic curve methods

The sum runs over the quadrature nodes θ of the `to` scale with their N(0, 1) weights, `options.quadrature`: by default
61 rectangular points; equateIRT's are `{ method: "gauss-hermite", points: 30 }`. A `from` item moved to the `to` scale
gives P_jk((θ - B) / A), so the criteria work for items of any model, a custom one included. Haebara's criterion sums
every category, which for a dichotomous item is twice the usual term and does not move the minimum. BFGS minimizes over
ln A and B from A = 1, B = 0, with the criterion's exact gradient, and reports `converged` and `iterations`. It has
converged at a gradient of at most 1e-10, or, as R's `optim`, when rounding of the criterion leaves no step that lowers
it or its gradient: within about 1e-8 of the minimum.

### Moving items and thetas

`transformItem(item, transformation)` returns the item on the new scale: a person at A θ + B has the probabilities a
person at θ had. It keeps the item's other fields, such as `id`, and scales its `standardErrors` with their
parameters, treating A and B as known: a by 1 / A, locations and steps by A, asymptotes unchanged. An NRM item's
intercepts c_k - a_k B / A need the covariances of a_k and c_k, so with B ≠ 0 its standard errors become `null`. It
supports the built-in models:

| Model          | New parameters           |
| -------------- | ------------------------ |
| 1PL to 4PL     | a / A, A b + B, c, d     |
| GRM, GPCM, PCM | a / A, A b_k + B         |
| RSM            | a / A, A b + B, A τ_k    |
| NRM            | a_k / A, c_k - a_k B / A |

`transformTheta(theta, transformation)` returns A θ + B; a standard error of theta scales by A.

## Observed-score equating

Linear and equipercentile equating convert the scores of form X to the scale of form Y from random groups: one group
took X and an equivalent group took Y. Each form's score distribution is the number of persons at each score, from 0
to the highest, as the equate package's frequency tables; fractional counts, as from smoothing, are accepted. The
result has the equated score of each X score.

| Function                           | Equated score of x                                                                |
| ---------------------------------- | --------------------------------------------------------------------------------- |
| `linearEquating(from, to)`         | μ_Y + (σ_Y / σ_X)(x - μ_X), standard deviations with denominator n - 1, as equate |
| `equipercentileEquating(from, to)` | The Y score with x's percentile rank (Kolen and Brennan, 2014)                    |

`logLinearPresmoothing(distribution, degree)` smooths a distribution before equipercentile equating (Holland and
Thayer, 2000): the maximum likelihood fit of ln m_x = β0 + β1 x + ... + β_C x^C to the counts, a Poisson regression,
as the equate package's `presmoothing(smoothmethod = "loglinear")`. The fit keeps the number of persons and the first
C moments of the scores, and fills scores without persons. It returns the fitted `distribution`, `converged` and
`iterations`; pass the distribution to `equipercentileEquating`. Degrees from 2 to 6 are usual: compare the fits'
moments and plots, or the AIC of the Poisson fits, to choose one.

```ts
import { equipercentileEquating, logLinearPresmoothing } from "@measurekit/psychometrics/equating";

const from = [1, 3, 6, 10, 14, 15, 12, 9, 6, 3, 1];
const to = [2, 4, 8, 11, 13, 14, 11, 8, 5, 3, 1];
const smooth = (counts: number[]) => [...logLinearPresmoothing(counts, 3).distribution];
equipercentileEquating(smooth(from), smooth(to));
```

Equipercentile equating treats scores as continuous, each uniform over [x - 0.5, x + 0.5]. The percentile rank of x is
P = F_X(x - 1) + f_X(x) / 2, and its Y equivalent is y - 0.5 + (P - F_Y(y - 1)) / f_Y(y) for the smallest y with
F_Y(y) > P. Where Y has scores without persons the inverse is ambiguous; the result is the mean of that value and
y' + 0.5 + (P - F_Y(y')) / (F_Y(y' + 1) - F_Y(y')) for the largest y' with F_Y(y') < P, as equate. A rank of 0 gives
-0.5 and a rank of 1 the highest Y score plus 0.5. Ranks compare with cumulative shares exactly, by cross-multiplying
the counts.

## Equating through common items

When two groups of different ability took different forms, their score distributions differ for two reasons: the
forms and the groups. Common items, the anchor, tell the groups apart: both groups took them. A distribution here is
the number of persons at each total score and anchor score, a `BivariateDistribution`: a row for each total score from
0, each row holding a count for each anchor score from 0. `from` is the group that took the form to equate and `to`
the group that took the form to equate to. Both need at least two total scores and two anchor scores, the same
anchor scores, non-negative counts and at least one person; fractional counts, as from smoothing, are accepted. The
total score counts the anchor items for an internal anchor and not for an external one.

| Function                                         | Equated score of x                                                                                   |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| `tuckerEquating(from, to, options)`              | Linear in the synthetic population, with the regression slope of the total on the anchor score as γ  |
| `levineEquating(from, to, options)`              | As Tucker with Levine's γ, or by Levine's true-score method                                          |
| `chainedLinearEquating(from, to)`                | x to the anchor in `from`'s group by a linear equating, then the anchor to the total in `to`'s group |
| `chainedEquipercentileEquating(from, to)`        | The same with equipercentile equating                                                                |
| `frequencyEstimationEquating(from, to, options)` | Equipercentile equating of the two forms' score distributions in the synthetic population            |

```ts
import {
  chainedEquipercentileEquating,
  chainedLinearEquating,
  frequencyEstimationEquating,
  levineEquating,
  tuckerEquating,
} from "@measurekit/psychometrics/equating";

// Persons by total score on the form, 0 to 5 (rows), then by score on the anchor, 0 to 3 (columns).
const groupX = [
  [8, 3, 1, 0],
  [10, 8, 3, 0],
  [7, 12, 8, 2],
  [3, 10, 14, 6],
  [1, 4, 11, 9],
  [0, 1, 4, 7],
]; // took form X
const groupY = [
  [5, 2, 0, 0],
  [9, 5, 1, 0],
  [8, 11, 4, 1],
  [4, 12, 12, 4],
  [1, 6, 13, 10],
  [0, 1, 5, 9],
]; // took form Y

tuckerEquating(groupX, groupY); // [0.407, 1.372, 2.337, 3.301, 4.266, 5.231]: the form Y score of each form X score
levineEquating(groupX, groupY, { trueScore: true }); // [0.586, 1.48, 2.375, 3.27, 4.164, 5.059]
chainedLinearEquating(groupX, groupY); // [0.425, 1.383, 2.34, 3.298, 4.255, 5.213]
chainedEquipercentileEquating(groupX, groupY); // [0.298, 1.43, 2.459, 3.369, 4.186, 5.1]
frequencyEstimationEquating(groupX, groupY); // [0.305, 1.44, 2.438, 3.35, 4.234, 5.121]
```

### Synthetic population

Tucker, Levine and frequency estimation equate in a synthetic population: the two groups, weighted by w₁ for `from`'s
group and w₂ = 1 - w₁ for `to`'s (Braun and Holland, 1982). Each form's score distribution in it is estimated from
the group that took the form and the anchor scores of the other.

| Option      | Of                                                                | Default                       | Meaning                                                                                   |
| ----------- | ----------------------------------------------------------------- | ----------------------------- | ----------------------------------------------------------------------------------------- |
| `weight`    | `tuckerEquating`, `levineEquating`, `frequencyEstimationEquating` | `from`'s share of the persons | w₁, a number from 0 to 1; equate's `ws`, which is -1 for this default                     |
| `trueScore` | `levineEquating`                                                  | `false`                       | Levine's true-score method, which does not use `weight`; equate's `lts`                   |
| `anchor`    | `levineEquating`                                                  | `"internal"`                  | `"internal"` or `"external"`: whether the anchor counts in the total; equate's `internal` |

Moments have denominator n - 1, as equate. With μ, σ² and σ(X, V) the means, variances and covariance of a group's total
score X and anchor score V, Δ the difference of the groups' mean anchor scores, and γ₁, γ₂ the slopes of the groups:

| Quantity                  | Formula                                               |
| ------------------------- | ----------------------------------------------------- |
| Synthetic mean of X       | μ₁(X) - w₂ γ₁ Δ                                       |
| Synthetic mean of Y       | μ₂(Y) + w₁ γ₂ Δ                                       |
| Synthetic variance of X   | σ₁²(X) - w₂ γ₁² (σ₁²(V) - σ₂²(V)) + w₁ w₂ γ₁² Δ²      |
| Synthetic variance of Y   | σ₂²(Y) + w₁ γ₂² (σ₁²(V) - σ₂²(V)) + w₁ w₂ γ₂² Δ²      |
| Tucker γ                  | σ(X, V) / σ²(V) in each group                         |
| Levine γ, internal anchor | σ²(X) / σ(X, V)                                       |
| Levine γ, external anchor | (σ²(X) + σ(X, V)) / (σ²(V) + σ(X, V))                 |
| Levine true score         | y = (γ₂ / γ₁)(x - μ₁(X)) + μ₂(Y) + γ₂ (μ₁(V) - μ₂(V)) |

Linear equating in the synthetic population maps x to the Y score with the same distance from the synthetic mean in
standard deviations. Each group needs at least two persons and total and anchor scores with variance, Levine's methods a
positive covariance of the two, and the synthetic population total scores with variance, or the function throws
`DataError`.

Chained linear equating needs no synthetic population: a linear equating of the total score to the anchor score in
`from`'s group, followed by the linear equating of the anchor score to the total score in `to`'s.

### Equipercentile methods

Chained equipercentile equating equates the total score to the anchor score in `from`'s group by
`equipercentileEquating`. The anchor equivalent is not a whole score, so its percentile rank in `to`'s group
interpolates linearly between the cumulative shares at the ends of its score's interval; the Y score with that rank
in `to`'s group is the result.

Frequency estimation (Braun and Holland, 1982) estimates the distribution of each form in the synthetic population
from the group that took it and the other group's anchor distribution h:

- f(x) = w₁ f₁(x) + w₂ Σ_v f₁(x | v) h₂(v) for form X, where f₁(x | v) is the share of the persons of `from`'s
  group with the anchor score v who scored x;
- g(y) = w₁ Σ_v g₂(y | v) h₁(v) + w₂ g₂(y) for form Y.

`equipercentileEquating` of f and g gives the result. The conditional distribution given an anchor score needs persons
with it, so an anchor score that persons in one group have and none in the other have throws a `DataError` that names
the score; smooth the distributions first.

## Bivariate presmoothing

`bivariateLogLinearPresmoothing(distribution, { degrees, crossDegree })` smooths a bivariate distribution (Holland
and Thayer, 2000): the maximum likelihood fit of the Poisson regression

ln m_xv = β₀ + Σ_{i ≤ C_x} β_i xⁱ + Σ_{j ≤ C_v} γ_j vʲ + Σ_{i, j ≤ C_xv} δ_ij xⁱ vʲ

to the counts, as the equate package's `presmoothing(smoothmethod = "loglinear", degrees = list(c(C_x, C_v), C_xv))`.
The fit keeps the number of persons, the first C_x moments of the total scores, the first C_v of the anchor scores and
the cross moments E[xⁱ vʲ]. It fills cells without persons, which frequency estimation needs.

| Option        | Default | Meaning                                                                                                           |
| ------------- | ------- | ----------------------------------------------------------------------------------------------------------------- |
| `degrees`     | none    | `[total, anchor]`: integers from 1 to one less than the number of total scores and of anchor scores               |
| `crossDegree` | `1`     | The highest power of each score in the cross products, an integer from 0, for none, to the smaller of the degrees |

It returns the fitted `distribution`, `converged` and `iterations`, as `logLinearPresmoothing`. `converged` is `false`
when Newton's method does not reach the maximum within 200 iterations, as when cells without persons leave the
likelihood no finite maximum; the distribution is then the last iterate.

```ts
import { bivariateLogLinearPresmoothing, frequencyEstimationEquating } from "@measurekit/psychometrics/equating";

const groupX = [
  [8, 3, 1, 0],
  [10, 8, 3, 0],
  [7, 12, 8, 2],
  [3, 10, 14, 6],
  [1, 4, 11, 9],
  [0, 1, 4, 7],
];
const smooth = bivariateLogLinearPresmoothing(groupX, { degrees: [3, 2] });
smooth.converged; // true
smooth.distribution[0]?.map((count) => Math.round(count * 10) / 10); // [7.8, 3.6, 0.6, 0]: fitted persons at total score 0
```

## Standard errors by bootstrap

`bootstrapEquating(sample, equate, random, options)` returns the standard error of the equated score at each score of
the form to equate (Efron and Tibshirani, 1993; Kolen and Brennan, 2014). Each replication draws as many persons as the
sample has, with replacement, in proportion to the counts of the cells, applies `equate` to the resampled
distributions, and the standard error at a score is the standard deviation (denominator B - 1) of the B equated scores,
as the equate package's `bootstrap`.

The sample names its design, and groups are resampled independently:

| `design`          | Fields                                                                        | Resampled                                                    |
| ----------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `"random-groups"` | `from`, `to`: counts of each score                                            | The persons of each form separately                          |
| `"single-group"`  | `joint`: counts by score on the form to equate, then on the form to equate to | Pairs of scores, so that the correlation of the forms counts |
| `"neat"`          | `from`, `to`: bivariate distributions                                         | The persons of each group separately                         |

| Option         | Default | Meaning                                                                                                                             |
| -------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `replications` | `500`   | The number B of resamples, an integer from 2; the Monte Carlo error of a standard error is about 1 / √(2 (B - 1)) of it: 3% for 500 |

The counts of a sample are whole numbers of persons. `equate` takes a sample of the design's shape and returns the
equated score of every score of the form to equate, and every smoothing or other step belongs inside it, so that its
variability counts; a resample for which it throws, such as frequency estimation with an anchor score in one group only,
stops the bootstrap with that error. The same seed in `random` gives the same standard errors.

```ts
import { createRandom } from "@measurekit/psychometrics";
import { bootstrapEquating, tuckerEquating } from "@measurekit/psychometrics/equating";

const groupX = [
  [8, 3, 1, 0],
  [10, 8, 3, 0],
  [7, 12, 8, 2],
  [3, 10, 14, 6],
  [1, 4, 11, 9],
  [0, 1, 4, 7],
];
const groupY = [
  [5, 2, 0, 0],
  [9, 5, 1, 0],
  [8, 11, 4, 1],
  [4, 12, 12, 4],
  [1, 6, 13, 10],
  [0, 1, 5, 9],
];
const sample = { design: "neat", from: groupX, to: groupY } as const;
bootstrapEquating(sample, (s) => tuckerEquating(s.from, s.to), createRandom(1), { replications: 200 });
// [0.216, 0.168, 0.137, 0.133, 0.16, 0.205]: the standard error of each form X score's equivalent
```

A single group equates the marginal distributions of its two forms with `linearEquating` and `equipercentileEquating`;
only the bootstrap differs, as it resamples pairs of scores:

```ts
import { createRandom } from "@measurekit/psychometrics";
import { bootstrapEquating, linearEquating } from "@measurekit/psychometrics/equating";

const joint = [
  [6, 3, 1],
  [2, 8, 4],
  [0, 3, 7],
]; // persons by score on form X (rows), then on form Y (columns)
const sum = (counts: readonly number[]) => counts.reduce((total, count) => total + count, 0);
const marginals = (pairs: readonly (readonly number[])[]) =>
  [pairs.map(sum), (pairs[0] ?? []).map((_, y) => sum(pairs.map((row) => row[y] ?? 0)))] as const;

linearEquating(...marginals(joint)); // [0.129, 1.118, 2.106]
bootstrapEquating(
  { design: "single-group", joint },
  (sample) => linearEquating(...marginals(sample.joint)),
  createRandom(1),
  { replications: 200 },
); // [0.153, 0.116, 0.152]
```

## Anchor screen

Before linking or equating through common items, programs drop the items whose parameters drifted between
calibrations. `robustZ(from, to)` takes the common items on both calibrations, `from[j]` and `to[j]` being the same
item with the same model and number of categories, and returns for each item the `difference` d of its location (b, or
the mean of a polytomous item's thresholds) between the two, its robust z and whether it is `flagged` (Huynh and Meyer,
2010):

z = (d - median d) / (0.74 IQR d)

with R's `median` and `IQR` (quantiles by linear interpolation) over the common items, and an item flagged when |z|
exceeds 1.645. The differences of locations measure drift only when the scales differ by a shift, as the Rasch and
partial credit models do; for other models, move the `from` items to the `to` scale with `transformItem` first. The
interquartile range of the differences must be above 0: with a single common item, or when the middle half of the
differences are equal, there is no spread to judge drift by, and `robustZ` throws.

```ts
import { robustZ } from "@measurekit/psychometrics/equating";
import { twoParameterLogistic } from "@measurekit/psychometrics/irt";

const item = (b: number) => ({ model: twoParameterLogistic, parameters: { a: 1, b, c: 0, d: 1 } });
const from = [-1.2, -0.6, -0.1, 0.4, 0.9, 1.5].map(item);
const to = [-0.85, -0.34, 0.22, 0.67, 2.1, 1.81].map(item);
robustZ(from, to).map(({ z, flagged }) => [Math.round(z * 100) / 100, flagged]);
// [[0.76, false], [-1.19, false], [0.11, false], [-0.97, false], [19.14, true], [-0.11, false]]
```

## True-score equating

`trueScoreEquating(from, to)` takes the items of two forms on a common scale, as after linking, and returns the Y true
score of each X score from 0 to the highest (Lord, 1980; Kolen and Brennan, 2014):

1. A form's true score τ(θ) is the sum of its items' expected scores. It rises from the sum L of the items' limits as
   theta goes to -∞, Σ c for logistic items and 0 for the others, to the sum U of their limits at +∞, Σ d and the
   highest categories.
2. For L_X < x < U_X, solve τ_X(θ) = x and return τ_Y(θ).
3. Below L_X no theta gives x: x maps linearly from 0 to L_Y (Kolen and Brennan's procedure). Above U_X, which happens
   with 4PL items, x maps linearly from U_Y to Y's highest score.

The items' expected scores must increase with theta: built-in models with a > 0, not the NRM.

## Not included

- IRT observed-score equating, and standard errors of IRT equating.
- Braun-Holland linear, circle-arc and nominal weights equating, kernel equating, and cubic-spline postsmoothing:
  presmoothing covers smoothing, and the others see little use.
- Analytic standard errors of observed-score equating: the bootstrap is equate's method.

## Differences from the references

| Topic                          | This package                                                             | Reference                                                                                                                                                                                                 |
| ------------------------------ | ------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Rasch items                    | A estimated by every method; 1 by mean-mean                              | equateIRT fixes A = 1 for its Rasch model                                                                                                                                                                 |
| Polytomous items               | Every method; NRM by the characteristic curves only                      | equateIRT's `direc`: dichotomous items                                                                                                                                                                    |
| Percentile ranks               | Exact                                                                    | equate rounds them and the cumulative shares to 10 decimals: up to 1e-7 apart                                                                                                                             |
| X scores above Y's scale       | Rank 1 gives Y's highest score plus 0.5                                  | equate returns the X score itself                                                                                                                                                                         |
| True scores below Σ c          | Kolen and Brennan's line from 0                                          | equateIRT's `score` leaves them out                                                                                                                                                                       |
| Frequency estimation weights   | `weight` is w₁ of the synthetic population                               | equate's `ws` also multiplies the groups' numbers of persons, so its default weights `from` by n₁² / (n₁² + n₂²); `ws = w₁ n₂ / (w₁ n₂ + (1 - w₁) n₁)` gives w₁, and the two agree for groups of one size |
| Anchor score of one group only | `DataError` naming the score                                             | equate gives the persons at it their own group's weight alone, so the synthetic distributions lose mass                                                                                                   |
| Bivariate presmoothing degrees | `degrees: [total, anchor]` and `crossDegree`, at most the smaller degree | `degrees = list(c(total, anchor), cross)`, with the cross products as in `crossDegree`                                                                                                                    |
| Bootstrap                      | The seeded `Random` of the package; replications default to 500          | R's `sample`, 100 replications; the standard errors agree within their Monte Carlo error                                                                                                                  |
