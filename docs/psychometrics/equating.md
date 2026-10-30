# Linking and equating

`@measurekit/psychometrics/equating` puts the item parameters of two calibrations on one theta scale (linking) and converts
the scores of one form to the scale of another (equating). Linking is validated against equateIRT and base R,
observed-score equating against the equate package, and true-score equating against base R and equateIRT
([validation](validation.md)).

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

- Nonequivalent groups with anchor tests (Tucker, Levine, chained, frequency estimation), IRT observed-score
  equating, and standard errors of equating.

## Differences from the references

| Topic                    | This package                                        | Reference                                                                     |
| ------------------------ | --------------------------------------------------- | ----------------------------------------------------------------------------- |
| Rasch items              | A estimated by every method; 1 by mean-mean         | equateIRT fixes A = 1 for its Rasch model                                     |
| Polytomous items         | Every method; NRM by the characteristic curves only | equateIRT's `direc`: dichotomous items                                        |
| Percentile ranks         | Exact                                               | equate rounds them and the cumulative shares to 10 decimals: up to 1e-7 apart |
| X scores above Y's scale | Rank 1 gives Y's highest score plus 0.5             | equate returns the X score itself                                             |
| True scores below Σ c    | Kolen and Brennan's line from 0                     | equateIRT's `score` leaves them out                                           |
