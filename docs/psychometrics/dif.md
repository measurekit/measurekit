# Differential item functioning

`@measurekit/psychometrics/dif` tests whether an item functions differently for two groups of persons of equal
ability: the Mantel-Haenszel statistics with the ETS categories, logistic regression and Lord's chi-square. The
first two work on sparse data and purify their matching criterion on request. Each is validated against difR, and
the sparse-data rule against base R ([validation](validation.md)).

## Groups

Every function takes a [response matrix](data.md) and one `DifGroup` per person: `"reference"` or `"focal"`, the group
whose items are studied. Both groups need persons.

| Function                                         | Items           | Matching                                         | Persons used                                        |
| ------------------------------------------------ | --------------- | ------------------------------------------------ | --------------------------------------------------- |
| `mantelHaenszel(matrix, groups, options)`        | Dichotomous     | [Matching score](#matching-score), item included | Per item: who answered it and another matching item |
| `logisticRegressionDif(matrix, groups, options)` | Dichotomous     | [Matching score](#matching-score), item included | Per item: who answered it and another matching item |
| `lordChiSquare(matrix, groups, options)`         | 2PL, 3PL or 4PL | Theta, by separate calibrations                  | All; calibration skips missing responses            |

```ts
import { createRandom } from "@measurekit/psychometrics";
import { responseMatrix } from "@measurekit/psychometrics/data";
import { type DifGroup, logisticRegressionDif, lordChiSquare, mantelHaenszel } from "@measurekit/psychometrics/dif";
import { twoParameterLogistic } from "@measurekit/psychometrics/irt";
import { simulateItems, simulateResponses, simulateThetas } from "@measurekit/psychometrics/sim";

// 500 persons per group answer 10 items; item 3 is one point harder for the focal group.
const random = createRandom(2026);
const items = simulateItems(twoParameterLogistic, 10, random);
const shifted = items.map((item, j) =>
  j === 2 ? { ...item, parameters: { ...item.parameters, b: item.parameters.b + 1 } } : item,
);
const reference = simulateResponses(items, simulateThetas(500, random), random);
const focal = simulateResponses(shifted, simulateThetas(500, random), random);
const matrix = responseMatrix([...reference.responses, ...focal.responses]);
const groups: DifGroup[] = matrix.personIds.map((_, person) => (person < 500 ? "reference" : "focal"));

mantelHaenszel(matrix, groups).items[2]; // { id: "item3", chiSquare: 33.6, ..., delta: -2.76, etsCategory: "C" }
// Purification flags item3 alone and leaves the other items as anchors.
logisticRegressionDif(matrix, groups, { purify: true }).purification; // { anchors, iterations: 1, converged: true }
lordChiSquare(matrix, groups, { model: twoParameterLogistic }).items[2]; // { id: "item3", chiSquare: 19.3, ... }
```

## Matching score

Persons are matched on their score over the matching items: all items, or the anchors after
[purification](#purification), and always the studied item. Each item uses the persons who answered it and at least one
other matching item. A person who answered n of the m other matching items, s of them correctly, has the score

x + round(m s / n),

with x the response to the studied item and halves rounded up: the proportion correct on the other matching items the
person answered, scaled to all of them. With every response present it is the total score over the matching items,
difR's criterion. Omitted responses that should count as wrong are scored 0 before the analysis
([data](data.md#scoring-against-a-key)); a missing response is one not presented, as an item not administered in an
adaptive test. With 2% of the responses missing at random over 30 items, 45% of the persons miss one; this rule keeps
them all.

## Purification

With `purify: true` (default `false`), the items flagged at the 0.05 level leave the anchors, and the statistics
are computed again on the items not flagged, until a round flags the items the previous one did, every item is
flagged, or 10 rounds have run, as difR's `purify` (Holland and Thayer, 1988). The Mantel-Haenszel test flags by
its chi-square, logistic regression by its test of both terms, difR's default `type`. The result's `purification`
holds the `anchors` of the final statistics, the `iterations` after the first statistics and whether they
`converged`; it is `null` without `purify`.

## Mantel-Haenszel

Persons are matched on their total score. Each score group gives a 2 × 2 table: A and B the reference group's correct
and incorrect responses, C and D the focal group's, T persons in all, with E(A) = (A + B)(A + C) / T and
Var(A) = (A + B)(C + D)(A + C)(B + D) / (T² (T - 1)). Score groups of one person are left out.

| Field                | Definition                                                                                         |
| -------------------- | -------------------------------------------------------------------------------------------------- |
| `chiSquare`          | (\|Σ (A - E(A))\| - 0.5)² / Σ Var(A), with one degree of freedom                                   |
| `significance`       | The probability of a chi-square at least as large without DIF                                      |
| `oddsRatio`          | α = Σ (AD / T) / Σ (BC / T): above 1 when the item favors the reference group                      |
| `delta`              | MH D-DIF = -2.35 ln α on the ETS delta scale: negative when the item is harder for the focal group |
| `deltaStandardError` | 2.35 times the square root of the Robins-Breslow-Greenland variance of ln α                        |
| `etsCategory`        | `"A"`, `"B"` or `"C"`, below                                                                       |

`continuityCorrection` (default `true`, as difR) subtracts 0.5 when \|Σ (A - E(A))\| is at least 0.5, as R's
`mantelhaen.test` does; `false` gives Σ (A - E(A))² / Σ Var(A).

### ETS categories

ETS's rule (Zwick, 2012), with SE the standard error of MH D-DIF:

| Category | DIF                | Rule                                                                                             |
| -------- | ------------------ | ------------------------------------------------------------------------------------------------ |
| `"A"`    | Negligible         | \|MH D-DIF\| < 1, or the chi-square is not significant at 0.05                                   |
| `"C"`    | Moderate to large  | \|MH D-DIF\| ≥ 1.5 and (\|MH D-DIF\| - 1) / SE > 1.645: significantly above 1, one-sided at 0.05 |
| `"B"`    | Slight to moderate | Otherwise                                                                                        |

The sign of `delta` tells which group the item favors: C- and B- items, negative, are harder for the focal group.

## Logistic regression

Three logistic regressions of a correct response on the total score S and the group G, 1 for the focal group
(Swaminathan and Rogers, 1990), each fitted by Newton's method, as glm:

| Model | Log odds                  |
| ----- | ------------------------- |
| 0     | β0 + β1 S                 |
| 1     | β0 + β1 S + β2 G          |
| 2     | β0 + β1 S + β2 G + β3 S G |

Each test is a `LikelihoodRatioTest`: `chiSquare`, the difference of the deviances; `degreesOfFreedom`;
`significance`; and `rSquaredChange`, the increase of Nagelkerke's R², (1 - e^((D - D0) / n)) / (1 - e^(-D0 / n)) with
D0 the deviance of the intercept alone, as difR's `deltaR2`.

| Test         | Models | Degrees of freedom | DIF                                             |
| ------------ | ------ | ------------------ | ----------------------------------------------- |
| `uniform`    | 1 vs 0 | 1                  | The item is harder for one group at every score |
| `nonuniform` | 2 vs 1 | 1                  | The difference changes with the score           |
| `both`       | 2 vs 0 | 2                  | Either                                          |

Where an estimate diverges (separation), the deviance reaches its infimum and the test is still computed. An item
its persons answered alike, or answered by one group only, has `null` tests.

## Lord's chi-square

`lordChiSquare` compares an item's IRT parameters between the groups (Lord, 1980), as difR's `difLord`:

1. It calibrates each group separately with `calibrate` and `options`: the [calibration options](irt.md#calibration)
   without `fixed`. The model is the 2PL, 3PL or 4PL.
2. It moves the focal group's estimates to the reference scale by the mean-sigma transformation over all items
   ([equating](equating.md#linking)): a / A and A b + B, and their covariance matrices with them.
3. For each item, with v the estimated parameters and Σ their covariance matrix from the calibration,
   χ² = (v_R - v_F)ᵀ (Σ_R + Σ_F)⁻¹ (v_R - v_F), with as many degrees of freedom as parameters.

The result holds `items`, `{ id, chiSquare, degreesOfFreedom, significance }` each, both calibrations, whose
`converged` reports their EM, and the `transformation`. difR's ltm uses 21 Gauss-Hermite points:
`{ quadrature: { method: "gauss-hermite", points: 21 } }` reproduces it.

## Not included

- Purification of Lord's chi-square, which would need a new linking each round.
- Polytomous items, the exact Mantel-Haenszel test, Wald tests and matching on a score other than the total.

## Differences from difR

| Topic                 | This package                                                 | difR                                                                                                          |
| --------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| Continuity correction | Subtracted when \|Σ (A - E(A))\| ≥ 0.5, as `mantelhaen.test` | Always subtracted: identical groups get (0.5)² / Σ Var(A), not 0                                              |
| ETS categories        | ETS's rule with its significance tests                       | `print` codes \|MH D-DIF\| alone: below 1 A, below 1.5 B, else C                                              |
| Lord's calibrations   | EM to `tolerance`                                            | ltm's default control, 40 EM cycles then BFGS, stops short: up to 2% in χ²                                    |
| Lord's 1PL            | Not supported: `oneParameterLogistic` estimates a common a   | a fixed at 1                                                                                                  |
| Missing responses     | The scaled matching score of each item's persons             | Sums the responses given; `difMH` counts a missing response to the studied item as both correct and incorrect |
