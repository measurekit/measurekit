# Item and person fit

`@measurekit/psychometrics/fit` checks how well items and persons fit an [IRT model](irt.md): Rasch infit and
outfit, the S-X² item fit statistic, the lz person fit statistic and Yen's Q3 for local dependence. Each is
validated against eRm or mirt ([validation](validation.md)).

## Inputs

Every function takes a [response matrix](data.md) and its items, one per column, of any model, such as a
calibration's `items`. The residual statistics also take one theta per person, finite, from any estimator; the
references use maximum likelihood estimates for Rasch items (eRm) and EAP estimates otherwise (mirt).

| Function                           | Result per                    | Persons used                                                |
| ---------------------------------- | ----------------------------- | ----------------------------------------------------------- |
| `itemFit(matrix, items, thetas)`   | item: `ItemFit`               | Persons without an extreme score; missing responses skipped |
| `personFit(matrix, items, thetas)` | person: `PersonFit`           | Each person's answered items                                |
| `summedScoreFit(matrix, items)`    | item: `SummedScoreFit`        | Persons who answered every item                             |
| `yenQ3(matrix, items, thetas)`     | pair of items: a matrix entry | For each pair, the persons who answered both                |

An extreme score has every answered response in its item's lowest category, or every one in its highest; its maximum
likelihood theta is infinite. A statistic undefined for the data is `null`.

```ts
import { responseMatrix } from "@measurekit/psychometrics/data";
import { itemFit, personFit, summedScoreFit, yenQ3 } from "@measurekit/psychometrics/fit";
import {
  calibrate,
  jointMaximumLikelihood,
  logNormalPrior,
  posteriorMeanEstimate,
  twoParameterLogistic,
  weightedLikelihoodEstimate,
} from "@measurekit/psychometrics/irt";

const matrix = responseMatrix([
  [1, 0, 1, 1],
  [0, 0, 1, 0],
  [1, 1, 1, 1],
  [0, 1, 0, 0],
  [1, 1, 0, 1],
  [0, 0, 0, 1],
  [1, 0, 1, 0],
  [1, 1, 1, 0],
]);

// Rasch items and their persons: infit and outfit.
const rasch = jointMaximumLikelihood(matrix);
const raschThetas = matrix.responses.map((row) => weightedLikelihoodEstimate(rasch.items, row).theta);
itemFit(matrix, rasch.items, raschThetas); // [{ id, infitMeanSquare, outfitMeanSquare, ... }, ...]

// Items of any model: S-X², lz and Q3. With few persons, a prior keeps the slopes finite.
const { items } = calibrate(matrix, { model: twoParameterLogistic, priors: { a: logNormalPrior(0, 0.5) } });
const thetas = matrix.responses.map((row) => posteriorMeanEstimate(items, row).theta);
summedScoreFit(matrix, items); // [{ id, chiSquare, degreesOfFreedom, significance }, ...]
personFit(matrix, items, thetas); // [{ id, ..., standardizedLogLikelihood }, ...]
yenQ3(matrix, items, thetas); // 4 × 4, symmetric, 1 on the diagonal
```

## Infit and outfit

For a response x at the person's theta, with E = Σ k P_k the expected score, W = Σ (k - E)² P_k its variance and
C = Σ (k - E)⁴ P_k its fourth central moment (Wright and Masters, 1982):

| Field                | Definition                                                                                |
| -------------------- | ----------------------------------------------------------------------------------------- |
| `outfitMeanSquare`   | The mean of (x - E)² / W over n responses: sensitive to unexpected responses far off      |
| `infitMeanSquare`    | Σ (x - E)² / Σ W: weighted by information, sensitive to responses near the person's level |
| `outfitStandardized` | (MS^(1/3) - 1)(3 / q) + q / 3, q² = Σ (C / W²) / n² - 1 / n (Wilson-Hilferty, ZSTD)       |
| `infitStandardized`  | The same with q² = Σ (C - W²) / (Σ W)²                                                    |

Mean squares near 1 fit. Above 1, responses are noisier than the model predicts; below 1, more predictable. The
standardized values are about N(0, 1) under fit. `itemFit` sums over persons, as eRm's `itemfit`; `personFit` over
items, as eRm's `personfit`, and gives `null` mean squares to persons with an extreme score. Both work for items of any
model, and are meant for the Rasch model and the partial credit models.

## S-X²

`summedScoreFit` computes Orlando and Thissen's (2000) S-X², in Kang and Chen's (2008) form for polytomous items, as
mirt's `itemfit(fit_stats = "S_X2")`:

1. Group the persons by total score s, from 1 to the highest score less 1. O(s, k) counts the persons of group s in
   category k of the item.
2. The expected count is N_s P(X = k | S = s): the integral over theta ~ N(0, 1) of P_k(θ) times the probability of a
   total of s - k on the other items, over the integral of the probability of s on all items. The totals' probabilities
   come from the recursion of Lord and Wingersky (1984), integrated with 61 rectangular points.
3. Collapse the table as mirt 1.48 does, to expected counts of at least 1: merge score groups with impossible cells
   into their neighbors, drop groups without persons, merge groups of one person into the next; for polytomous items,
   merge small categories within a group and rare last categories; then merge each group with a count between 0 and 1
   into a neighbor.
4. S-X² = Σ (O - E)² / E over the cells left, with cells - groups - parameters degrees of freedom; `null` when none
   are left.

The degrees of freedom count every parameter of the item's model as estimated, so for anchor items, which were not
estimated, they are too few by the anchor's parameters.

## lz

`standardizedLogLikelihood` in `PersonFit` is Drasgow, Levine and Williams's (1985) lz for items of any model, as
mirt's `personfit` computes `Zh`. With l0 = Σ ln P(x) over the answered items:

- E(l0) = Σ_i Σ_k P_ik ln P_ik;
- Var(l0) = Σ_i Σ_(k < h) P_ik P_ih (ln P_ik - ln P_ih)², a sum of non-negative terms;
- lz = (l0 - E(l0)) / √Var(l0), `null` when the variance is 0.

lz is about N(0, 1) under fit at the true theta; large negative values flag unexpected patterns.

## Yen's Q3

`yenQ3` correlates the residuals x - E(x | θ) of each pair of items over persons (Yen, 1984), as mirt's
`residuals(type = "Q3")`. Under local independence Q3 is near -1 / (k - 1) for k items; pairs well above the average
Q3 suggest dependence, as items sharing a stimulus.

## Differences from the references

| Topic                   | This package                                                | Reference                                                                    |
| ----------------------- | ----------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Missing responses       | Infit and outfit skip them in every sum                     | eRm's infit sums W over every person                                         |
| Small probabilities     | ln P of an observed category clamped at the smallest double | mirt's `personfit` counts an observed category with P below 2.2e-16 as P = 1 |
| Person mean squares     | As eRm                                                      | mirt's `personfit` raises W² below 1e-5 to 1e-5 and caps q at 1.4142         |
| S-X² degrees of freedom | Every parameter of the model                                | mirt: the freely estimated parameters                                        |
| An empty S-X² table     | `null` for the item                                         | mirt stops collapsing the remaining items                                    |
