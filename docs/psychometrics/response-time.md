# Response times

`@measurekit/psychometrics/response-time` models how long persons take on items: the lognormal model with its calibration,
person speed, aberrant times, and rapid guessing by normative thresholds, response time effort and effort-moderated
scoring. Each is validated against nlme, LNIRT, mirt or base R ([validation](validation.md)).

## Times

Times are in seconds, positive, aligned with the responses: `times` of a [response matrix](data.md), by person, then
item, `null` when missing; or `time` of a [response record](data.md#response-records). A time needs a response.

```ts
import { responseMatrix } from "@measurekit/psychometrics/data";

const matrix = responseMatrix(
  [
    [1, 0, 1],
    [0, 1, null],
  ],
  {
    times: [
      [41.2, 63.5, 22.8],
      [3.1, 70.4, null],
    ],
  },
);
```

## Lognormal model

The lognormal model (van der Linden, 2006) takes the log time of person j on item i as normal:

ln T_ij ~ N(β_i - τ_j, 1 / α_i²),

with β_i the item's time intensity, α_i its time discrimination, and τ_j the person's speed, τ ~ N(0, σ_τ²) in the
population: a faster person has a higher τ.

`calibrateResponseTimes(data, options)` estimates β, α and σ_τ by marginal maximum likelihood. τ integrates out in
closed form, so EM's E-step gives each person's normal posterior of τ and the M-step updates every parameter in
closed form; the log-likelihood is exact. The model is the linear mixed model of nlme's
`lme(log(t) ~ 0 + item, random = ~ 1 | person, weights = varIdent(form = ~ 1 | item), method = "ML")`. Each item needs
at least two times that differ: with equal times its variance would be 0 and the likelihood unbounded.

| Option          | Default | Meaning                                                              |
| --------------- | ------- | -------------------------------------------------------------------- |
| `tolerance`     | `1e-6`  | EM stops when no parameter changes by more than this in an iteration |
| `maxIterations` | `1000`  | The most EM iterations                                               |

| Result          | Meaning                                                                                    |
| --------------- | ------------------------------------------------------------------------------------------ |
| `items`         | Per item: `id`, `intensity` β, `discrimination` α and the `standardError` of the intensity |
| `speedSd`       | σ_τ                                                                                        |
| `logLikelihood` | The marginal log-likelihood of the log times                                               |
| `aic`, `bic`    | With 2 parameters per item and σ_τ; n the persons with times                               |
| `converged`     | Whether EM reached `tolerance`                                                             |
| `iterations`    | EM iterations run                                                                          |

The intensities' standard errors are the square roots of the diagonal of (Σ_j X_jᵀ V_j⁻¹ X_j)⁻¹, as lme's `vcov`. Each
item needs two times or more; responses without times are skipped.

```ts
import { createRandom } from "@measurekit/psychometrics";
import { responseMatrix } from "@measurekit/psychometrics/data";
import { calibrateResponseTimes, responseTimeFit, speedEstimates } from "@measurekit/psychometrics/response-time";
import { normalDistribution, simulateThetas } from "@measurekit/psychometrics/sim";

// 300 persons answer 3 items; ln T = β - τ + ε / α with β = 4, 3.5 and 3, α = 2 and speeds τ ~ N(0, 0.3²).
const random = createRandom(2026);
const intensities = [4, 3.5, 3];
const speeds = simulateThetas(300, random, normalDistribution(0, 0.3));
const times = speeds.map((speed) => intensities.map((intensity) => Math.exp(intensity - speed + random.normal() / 2)));
const data = responseMatrix(
  times.map((row) => row.map(() => 1)),
  { times },
);

const calibration = calibrateResponseTimes(data); // intensities 3.94, 3.46, 3.02; discriminations 2.24, 2.00, 1.91
calibration.speedSd; // 0.32
const options = { sd: calibration.speedSd };
speedEstimates(data, calibration.items, options)[0]; // { id: "person1", speed: -0.09, standardError: 0.21 }
responseTimeFit(data, calibration.items, options)[0]; // residuals [0.48, -0.01, -0.12], statistic 0.24
```

## Speed

`speedEstimates(data, items, { sd })` estimates each person's τ from the person's log times y over items of known
parameters, matched by id. With `sd`, σ_τ of the prior N(0, σ_τ²), such as a calibration's `speedSd`, it returns the
posterior mean Σ α_i² (β_i - y_i) / (1 / σ_τ² + Σ α_i²) and the posterior SD, lme's BLUP; without `sd`, the maximum
likelihood estimate Σ α_i² (β_i - y_i) / Σ α_i² and 1 / √Σ α_i². A person without times has `null` estimates.

## Aberrant times

`responseTimeFit(data, items, { sd })` returns, for each person, the speed as above and:

| Field              | Meaning                                                                                          |
| ------------------ | ------------------------------------------------------------------------------------------------ |
| `residuals`        | z_i = α_i (ln t_i - (β_i - τ̂)) per item, `null` without a time: a time far from the expected one |
| `statistic`        | Σ z_i², the person-level statistic of van der Linden and Guo (2008), as LNIRT's `personfitLN`    |
| `degreesOfFreedom` | The number of the person's times                                                                 |
| `significance`     | The probability of a chi-square at least as large with those degrees of freedom                  |

Under the model, each z_i is about N(0, 1) and the statistic about chi-square; with τ estimated from the same times
both are approximate. Large negative residuals flag times much shorter than the person's speed predicts, as from
preknowledge or rapid guessing; a large statistic flags a person's whole pattern.

## Rapid guessing

A rapid guess is a response faster than its item's threshold: too fast to have read and solved the item.

| Function                                                       | Result                                                                                                                   |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `normativeThresholds(data, { percentage, cap })`               | Each item's threshold: `percentage` of its mean time, at most `cap` seconds (Wise and Ma, 2012)                          |
| `responseTimeEffort(data, thresholds)`                         | Per person: `effort`, the share of timed responses at or above their threshold (Wise and Kong, 2005), and `rapidGuesses` |
| `effortModeratedEstimates(data, items, thresholds, estimator)` | Per person: theta with rapid guesses treated as missing (Wise and DeMars, 2006)                                          |

| Option       | Default | Meaning                                 |
| ------------ | ------- | --------------------------------------- |
| `percentage` | `10`    | The share of the item's mean time, NT10 |
| `cap`        | `10`    | The largest threshold in seconds        |

`thresholds` maps item ids to seconds, such as the result of `normativeThresholds`; items without one count every
response. `effortModeratedEstimates` estimates theta as `abilityEstimates` ([IRT](irt.md#ability-estimation)) does,
with `posteriorMeanEstimate` by default, from the person's responses that are not rapid guesses. Persons with low
effort, such as below 0.9, are often reported or left out of calibration.

```ts
import { responseMatrix } from "@measurekit/psychometrics/data";
import { twoParameterLogistic } from "@measurekit/psychometrics/irt";
import {
  effortModeratedEstimates,
  normativeThresholds,
  responseTimeEffort,
} from "@measurekit/psychometrics/response-time";

const data = responseMatrix(
  [
    [1, 0, 1],
    [0, 0, 1],
  ],
  {
    times: [
      [41, 63, 22],
      [2, 3, 35],
    ],
  },
);
const thresholds = normativeThresholds(data); // { item1: 2.15, item2: 3.3, item3: 2.85 }
responseTimeEffort(data, thresholds); // person2 guessed items 1 and 2: effort 1/3
const items = [
  { id: "item1", model: twoParameterLogistic, parameters: { a: 1.2, b: -0.5, c: 0, d: 1 } },
  { id: "item2", model: twoParameterLogistic, parameters: { a: 0.8, b: 0.3, c: 0, d: 1 } },
  { id: "item3", model: twoParameterLogistic, parameters: { a: 1.5, b: 1.1, c: 0, d: 1 } },
];
effortModeratedEstimates(data, items, thresholds); // person2 from item 3 alone
```

## Not included

- The joint hierarchical model of responses and times (van der Linden, 2007), which correlates ability and speed.
- Response times as collateral information in adaptive tests.

## References

- van der Linden, W. J. (2006). A lognormal model for response times on test items. Journal of Educational and
  Behavioral Statistics, 31, 181-204.
- van der Linden, W. J., & Guo, F. (2008). Bayesian procedures for identifying aberrant response-time patterns in
  adaptive testing. Psychometrika, 73, 365-384.
- Wise, S. L. (2017). Rapid-guessing behavior: Its identification, interpretation, and implications. Educational
  Measurement: Issues and Practice, 36(4), 52-61.
- Wise, S. L., & DeMars, C. E. (2006). An application of item response time: The effort-moderated IRT model. Journal of
  Educational Measurement, 43, 19-38.
- Wise, S. L., & Kong, X. (2005). Response time effort: A new measure of examinee motivation in computer-based tests.
  Applied Measurement in Education, 18, 163-183.
- Wise, S. L., & Ma, L. (2012). Setting response time thresholds for a CAT item pool: The normative threshold method.
  Annual meeting of the National Council on Measurement in Education, Vancouver.
