# Multidimensional item response theory

`@measurekit/psychometrics/multidimensional` models a person by a vector of thetas, one per dimension, and an item by a
slope on each dimension: a high theta on one dimension makes up for a low one on another (compensatory models). It
calibrates items and the correlations between dimensions by marginal maximum likelihood, exploratory and confirmatory,
with EM for up to three dimensions and with MH-RM for any number, and bifactor and two-tier models by dimension
reduction; computes standard errors; rotates exploratory solutions; calibrates several groups of persons on one scale,
with fixed anchor items; estimates thetas by EAP and MAP in any number of dimensions; and computes information matrices.
Every estimate is validated against mirt, GPArotation and base R ([validation](validation.md)). The dimensions have mean
0 and variance 1, in the first group of a calibration of several groups.

```ts
import { createRandom } from "@measurekit/psychometrics";
import {
  calibrateMultidimensional,
  multidimensionalPosteriorModeEstimate,
  multidimensionalTwoParameterLogistic,
} from "@measurekit/psychometrics/multidimensional";
import { simulateMultidimensionalResponses } from "@measurekit/psychometrics/sim";

// Eight items, four on each of two correlated dimensions; 1,000 persons.
const random = createRandom(11);
const items = Array.from({ length: 8 }, (_, i) => ({
  model: multidimensionalTwoParameterLogistic,
  parameters: { a: [i < 4 ? 1.2 : 0, i < 4 ? 0 : 1.4], intercept: (i % 4) - 1.5, c: 0, d: 1 },
}));
const thetas = Array.from({ length: 1000 }, () => {
  const [z1, z2] = [random.normal(), random.normal()];
  return [z1, 0.5 * z1 + Math.sqrt(0.75) * z2];
});
const matrix = simulateMultidimensionalResponses(items, thetas, random);

// Confirmatory: the dimensions of each item by id.
const loadings = Object.fromEntries(matrix.itemIds.map((id, i) => [id, [i < 4 ? "verbal" : "quantitative"]]));
const fit = calibrateMultidimensional(matrix, {
  model: multidimensionalTwoParameterLogistic,
  loadings,
  standardErrors: "hessian",
});
fit.dimensions; // ["verbal", "quantitative"]
fit.items[0]?.parameters; // { a: [1.17, 0], intercept: -1.34, c: 0, d: 1 }
fit.items[0]?.standardErrors; // { a1: 0.18, intercept: 0.12 }
fit.correlations; // [[1, 0.46], [0.46, 1]]
fit.converged; // true

// A person's thetas, with the calibration's correlations as the prior.
const prior = { mean: [0, 0], covariance: fit.correlations };
multidimensionalPosteriorModeEstimate(fit.items, [1, 1, 0, 1, null, 1, 1, 0], { prior }).theta; // [0.61, 0.32]
```

## Models

Items have K ordered categories scored 0 to K - 1. With the slopes `a` and the thetas θ of a person, z = a · θ. The
parameters follow mirt's slope-intercept form, `coef(model, IRTpars = FALSE)`, with D = 1 ([parameterization](parameterization.md)
converts them); the intercept is mirt's `d`, named in full because `d` is the upper asymptote of the IRT form.

| Model                                    | Parameters                                 | P(X = 1) or P(X ≥ k)                                                 | mirt item type |
| ---------------------------------------- | ------------------------------------------ | -------------------------------------------------------------------- | -------------- |
| `multidimensionalTwoParameterLogistic`   | `{ a, intercept, c, d }` with c = 0, d = 1 | 1 / (1 + exp(-(z + intercept)))                                      | `2PL`          |
| `multidimensionalThreeParameterLogistic` | `{ a, intercept, c, d }` with d = 1        | c + (1 - c) / (1 + exp(-(z + intercept)))                            | `3PL`          |
| `multidimensionalGradedResponse`         | `{ a, intercepts: [d1, ..., d(K-1)] }`     | P(X ≥ k) = 1 / (1 + exp(-(z + d_k))), P(k) = P(X ≥ k) - P(X ≥ k + 1) | `graded`       |

Valid input: the slopes are finite numbers of any sign, one or more; the intercepts are finite; `c` is from 0 to below
1 and `d` is 1; the graded intercepts strictly decrease; thetas are finite. All items of a test have the same number of
dimensions, and the slopes of a dimension an item does not load on are 0.

The multidimensional discrimination of an item is MDISC = √(Σ a_d²), and its multidimensional difficulty, for a
dichotomous item, MDIFF = -intercept / MDISC.

A model implemented outside the package works as the built-in ones do if it has the members of
`MultidimensionalModel`. The first `dimensions` entries of its parameter vector are the slopes, `parameterDerivatives`
differentiates with respect to the vector, and `information` writes the Fisher information matrix.

## Calibration

```mermaid
flowchart TD
  M{"method"} -->|"em: the default for 1 to 3 dimensions"| E["EM: integrates theta over a grid of points ** D nodes"]
  M -->|"mhrm: the default for 4 or more"| H["MH-RM: draws theta from its posterior, no grid"]
  E --> EO["Deterministic; hessian and cross-product standard errors; at most 3 dimensions"]
  H --> HO["Seeded, with Monte Carlo error; hessian standard errors; any number of dimensions"]
```

|                      | EM                                                   | MH-RM                                                                               |
| -------------------- | ---------------------------------------------------- | ----------------------------------------------------------------------------------- |
| Dimensions           | 1 to 3                                               | Any number; the default above 3                                                     |
| Integration of theta | Quadrature on a grid of `points` nodes per dimension | Metropolis-Hastings draws, no grid                                                  |
| Result               | Deterministic, exact to `tolerance`                  | Seeded, within a Monte Carlo error                                                  |
| Standard errors      | `"hessian"` and `"cross-product"`                    | `"hessian"`                                                                         |
| Log-likelihood       | Exact on the grid                                    | Importance sampling                                                                 |
| Time                 | Grows with the responses times `points ** D`         | Grows with the persons, responses and parameters; slower than EM below 4 dimensions |

`calibrateMultidimensional(data, options)` takes a response matrix or response records, and a model, one for all items
or one per item. Missing responses are skipped, and persons without responses change nothing.

| Option           | Default                                   | Meaning                                                                                                              |
| ---------------- | ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `model`          | required                                  | One model, or one per item in item order                                                                             |
| `method`         | `"em"` up to 3 dimensions, `"mhrm"` above | `"em"` or `"mhrm"`                                                                                                   |
| `dimensions`     | none; one of `dimensions` and `loadings`  | Exploratory: the number of dimensions, a positive integer, at most 3 for EM; every item loads on each                |
| `loadings`       | none; one of `dimensions` and `loadings`  | Confirmatory: the names of the dimensions of each item, by item id, of every item                                    |
| `correlated`     | `true`                                    | Confirmatory: estimate the correlations, in every group. Ignored when exploratory                                    |
| `groups`         | one group                                 | EM only: each person's group label, aligned with the persons; [several groups on one scale](#groups-and-fixed-items) |
| `fixed`          | none                                      | EM only: parameters of anchor items by item id, which are not estimated and set the scale                            |
| `priors`         | none                                      | Priors by parameter name: `"a"` for every slope, `"intercept"` and `"c"`; estimates are posterior modes              |
| `points`         | `61`, `31`, `15` for 1, 2, 3 dimensions   | EM only: nodes per dimension of the grid, at least 2, from -6 to 6 standard deviations: mirt's `quadpts`             |
| `standardErrors` | `"none"`                                  | `"hessian"`, or `"cross-product"` for EM only                                                                        |
| `seed`           | `0`                                       | MH-RM only: a safe integer; the same seed gives the same calibration                                                 |
| `tolerance`      | `1e-4` for EM, `1e-3` for MH-RM           | Converged when no parameter changes by more than this: in an EM iteration, or in 3 successive MH-RM cycles           |
| `maxIterations`  | `500` for EM, `2000` for MH-RM            | The most EM iterations, or the most cycles of MH-RM's last stage, after the 250 of its first two stages              |

- **Confirmatory.** Each item loads on the dimensions named for it; the dimensions are named in order of first
  appearance among the items, in item order, and their slopes on the other dimensions are fixed at 0. With
  `correlated` the correlations are estimated, on a unit diagonal.
- **Exploratory.** Every item loads on every dimension and the dimensions are uncorrelated. Rotating the loadings
  leaves the likelihood unchanged, so the first D - 1 items' slopes beyond their own index are fixed at 0 (item j loads
  on the first j dimensions): the solution is then determined up to the signs of the dimensions. The dimensions are
  named `F1` to `FD`. EM starts from the first D eigenvectors of the items' correlations as loadings, converted to
  slopes by a = 1.702 λ / √(1 - h²). Use `rotateLoadings` for a simpler structure.
- **One dimension.** The model equals the one-dimensional model in the slope-intercept form: slope a, intercept
  -a b.
- **EM.** The grid is the product of `points` equally spaced nodes per dimension, weighted by the normal density of the
  correlations and normalized to sum to 1. The E-step accumulates each item's expected counts at every node; the M-step
  takes a Fisher-scoring step for each item, halved until the expected log-likelihood does not fall, and maximizes the
  correlations, which keep a unit diagonal and stay positive definite, with BFGS on the weights of the grid, so that
  it maximizes the reported likelihood. SQUAREM accelerates the iterations: every third one starts from a point
  extrapolated along the last two, shortened toward the plain steps where the log-likelihood falls. It reaches the
  fixed points of plain EM in a fraction of the iterations where EM contracts slowly, as for an exploratory model, and
  `iterations` counts every EM iteration.
- **MH-RM.** Each cycle draws every person's theta from its posterior given the current estimates, by 5 random-walk
  Metropolis-Hastings steps whose variance is tuned to an acceptance of 0.35, and moves the estimates by γ Γ⁻¹ g, with g
  the complete-data score at the draws and Γ the observed complete-data information. The stages, gains and tolerance
  are mirt's defaults (below). The dimensions are not integrated, so their number is not limited, and the cost grows with
  the persons, responses and parameters, not with a grid. The result is stochastic, with a Monte Carlo error that
  [validation](validation.md) measures, and seeded: `seed` fixes the draws. MH-RM does not compute the log-likelihood,
  which is estimated by importance sampling with 5,000 draws per person, from the normal approximation of each person's
  posterior.
- **Priors.** A prior of `"c"` is on the probability scale; mirt's `norm` prior is on logit c, which a custom
  `ParameterPrior` reproduces ([IRT](irt.md#priors)). A prior of `"a"` applies to every estimated slope.

```mermaid
flowchart LR
  A["Stage 1: 150 cycles, gain 0.2, tunes the walk"] --> B["Stage 2: 100 cycles, gain 0.2, keeps the estimates and information"]
  B --> C["Stage 3: gain (0.1 / t)^0.75, from their means, until 3 successive cycles move no estimate by more than tolerance"]
  C --> D["Log-likelihood: importance sampling"]
  C --> E["Standard errors: 400 cycles at the estimates"]
```

The result:

| Field                         | Content                                                                                                                        |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `dimensions`                  | The names of the dimensions, in the order of the slopes                                                                        |
| `items`                       | Each item's `id`, `model`, `parameters` and `standardErrors`                                                                   |
| `correlations`                | The correlations between dimensions, by rows; of the first group when there are several                                        |
| `groups`                      | With `groups` or `fixed`: each group's `group` label, `means` and `covariance`, in order of first appearance                   |
| `correlationStandardErrors`   | By rows with a zero diagonal; `null` if the correlations are not estimated or there are no standard errors                     |
| `loadings`                    | The standardized loadings, items by dimensions                                                                                 |
| `communalities`               | λᵀ Φ λ of each item                                                                                                            |
| `logLikelihood`, `aic`, `bic` | The marginal log-likelihood at the estimates, without priors, by importance sampling for MH-RM; -2 ln L + 2p; -2 ln L + p ln n |
| `converged`, `iterations`     | Whether `tolerance` was reached, and the iterations run: EM's, or the cycles of MH-RM's last stage                             |

The number of parameters p counts the estimated slopes, intercepts and asymptotes, and the correlations and, with
several groups or fixed items, the means and variances; n is the number of persons with a response.

The standardized loadings follow mirt's `summary`: λ = (a / 1.702) / √(1 + (a / 1.702)ᵀ Φ (a / 1.702)), where Φ is the
correlation matrix, and the communality is λᵀ Φ λ. With several groups, Φ is the correlation matrix of the first group
and the slopes are scaled by its standard deviations.

### Groups and fixed items

```mermaid
flowchart LR
  P["Persons, with a label in groups"] --> G1["First group: the reference"]
  P --> G2["Other groups"]
  I["Items: one set of parameters for all groups"] --> L["Every group's distribution of theta is integrated on the same grid"]
  G1 --> L
  G2 --> L
  F["fixed: anchor items"] -.->|"set the scale: no group is the reference"| L
```

`groups` calibrates persons from several populations on one scale, as `mirt::multipleGroup` with
`invariance = c("slopes", "intercepts", "free_means", "free_var")`: every item has the same parameters in every group,
and each group has its own distribution of theta, normal with a mean vector and a covariance matrix. `fixed` gives the
parameters of anchor items, which are not estimated; the other items are estimated on their scale.

| Entries of a group's distribution | The first group                        | The other groups                       | Every group, with `fixed`              |
| --------------------------------- | -------------------------------------- | -------------------------------------- | -------------------------------------- |
| Means                             | 0                                      | Estimated                              | Estimated                              |
| Variances                         | 1                                      | Estimated                              | Estimated                              |
| Correlations                      | Estimated with `correlated`, 0 without | Estimated with `correlated`, 0 without | Estimated with `correlated`, 0 without |

The first group is the first label in order of first appearance. The groups must share items, directly or through other
groups: a group that shares none with the first throws a `DataError`, as its distribution could not be put on the scale
of the others. With `fixed`, the fixed items set the scale of each set of groups that share items, so every dimension
needs a fixed item with a slope on it that the set answered; otherwise a `DataError`. A fixed item needs parameters of
the calibration's number of dimensions and as many categories as its responses reach. Calibrating persons of one group
onto the scale of fixed items is the case of `fixed` without `groups`: the result then has one group, labeled `"all"`.

```ts
import { createRandom } from "@measurekit/psychometrics";
import {
  calibrateMultidimensional,
  multidimensionalTwoParameterLogistic,
} from "@measurekit/psychometrics/multidimensional";
import { simulateMultidimensionalResponses } from "@measurekit/psychometrics/sim";

// Twenty items, ten on each of two dimensions; 1,000 persons in each of two years, the later one 0.4 higher.
const random = createRandom(5);
const items = Array.from({ length: 20 }, (_, i) => ({
  model: multidimensionalTwoParameterLogistic,
  parameters: { a: [i < 10 ? 1.2 : 0, i < 10 ? 0 : 1.4], intercept: (i % 10) * 0.25 - 1.1, c: 0, d: 1 },
}));
const thetas = Array.from({ length: 2000 }, (_, person) => {
  const shift = person < 1000 ? 0 : 0.4;
  const [z1, z2] = [random.normal(), random.normal()];
  return [shift + z1, shift + 0.5 * z1 + Math.sqrt(0.75) * z2];
});
const matrix = simulateMultidimensionalResponses(items, thetas, random);

const fit = calibrateMultidimensional(matrix, {
  model: multidimensionalTwoParameterLogistic,
  loadings: Object.fromEntries(matrix.itemIds.map((id, i) => [id, [i < 10 ? "verbal" : "quantitative"]])),
  groups: thetas.map((_, person) => (person < 1000 ? "2025" : "2026")),
});
fit.groups?.[0]; // { group: "2025", means: [0, 0], covariance: [[1, 0.5], [0.5, 1]] }
fit.groups?.[1]; // { group: "2026", means: [0.35, 0.41], covariance: [[0.98, 0.5], [0.5, 1.04]] }
fit.converged; // true
```

The groups' distributions can serve as priors of the scoring functions: pass `{ mean: group.means, covariance:
group.covariance }` as `prior`.

**Standard errors.** With `standardErrors`, the information covers the items and the entries of every group's
distribution, so an item's standard error accounts for the groups. Fixed items have none, and `correlationStandardErrors`
are those of the first group's correlations.

**Convergence.** A group's estimated distribution that runs to the edge of the grid, a mean outside it or a standard
deviation below the spacing of its nodes or above its range, leaves `converged` `false`: a few persons, or too coarse a
grid, give such a group no maximum. The default grid of three dimensions has 15 nodes per dimension; set `points: 21`
above 20 items per dimension, where the coarse grid biases the moments of the groups.

### Standard errors

| `standardErrors`  | Method                                                                                                                                                                                                                                                                       | Cost                                                       |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| `"hessian"`       | The inverse of the observed information of the log-posterior in unbounded coordinates, with the delta method for c, by Louis' formula: the complete-data information minus the covariance of the complete-data score under the posterior. For EM, mirt's `SE.type = "Oakes"` | EM: one pass over the posterior; MH-RM: 400 further cycles |
| `"cross-product"` | The inverse of the sum over persons of the outer products of their score vectors, plus the curvature of the priors; mirt's `"crossprod"`; EM only                                                                                                                            | One pass over the persons                                  |

Oakes' formula differentiates the EM map and needs two E-steps for each estimated parameter. Louis' formula gives the
same matrix from the posterior of one E-step: the complete-data information is the second derivative of the M-step's
objective, and the covariance of the scores is summed over the grid, for an item over the marginal of the posterior on
the dimensions the item loads on. MH-RM has no grid: the covariance is that of each person's score over the draws of 400
cycles at the estimates, summed over persons, and the information is the observed complete-data information at the
draws. The standard errors of MH-RM are therefore stochastic, as mirt's are; the person-wise covariance leaves out the
covariance between persons, which is 0 given the parameters and would add noise.

A standard error is listed by the name of the parameter: `a1` to `aD` for the slopes the item loads on, `intercept` or
`intercept1`, `intercept2`, ..., and `c`. Slopes fixed at 0 have none. All standard errors of a calibration are `null`
if the information matrix cannot be inverted or gives a variance that is not positive.

### Time

CPU time of one calibration on an Apple M5 with Node.js 26, simulated data, each item loading on one dimension, the
default `tolerance`. EM:

| Data                                                                       | `"none"` | `"cross-product"` | `"hessian"` |
| -------------------------------------------------------------------------- | -------- | ----------------- | ----------- |
| 2 dimensions, M2PL, 20 items, 2,000 persons, 31 points                     | 1.1 s    | 1.4 s             | 3.3 s       |
| 3 dimensions, M2PL, 30 items, 2,000 persons, 15 points                     | 7.3 s    | 7.6 s             | 11.6 s      |
| 3 dimensions, graded with 5 categories, 15 items, 1,000 persons, 15 points | 3.6 s    | 3.3 s             | 5.7 s       |

MH-RM, seeded, with the cycles of its last stage:

| Data                                                            | Cycles | `"none"` | `"hessian"` |
| --------------------------------------------------------------- | ------ | -------- | ----------- |
| 2 dimensions, M2PL, 20 items, 2,000 persons                     | 44     | 16.1 s   | 23.8 s      |
| 3 dimensions, M2PL, 30 items, 2,000 persons                     | 45     | 27.3 s   | 38.1 s      |
| 3 dimensions, graded with 5 categories, 15 items, 1,000 persons | 53     | 21.5 s   | 30.0 s      |
| 5 dimensions, M2PL, 30 items, 2,000 persons                     | 54     | 31.9 s   | 43.8 s      |
| 5 dimensions, graded with 5 categories, 25 items, 1,000 persons | 57     | 38.6 s   | 53.6 s      |

Differences under 10% are noise. The E-step of EM dominates and grows with the responses times the nodes: a grid of
`points` nodes in each of D dimensions has `points ** D` nodes. Fewer `points` is faster and, for a calibration, less
exact. MH-RM has no grid, so its time grows slowly with the dimensions; it runs 250 cycles before the last stage, and
the importance-sampled log-likelihood is part of every time. Use EM up to three dimensions, where it is exact and
faster. The exploratory model of a weak second dimension converges slowly with EM; `maxIterations` bounds it.
Calibration runs synchronously: in a browser, run it in a Web Worker ([IRT](irt.md#in-a-browser)).

## Bifactor and two-tier models

`calibrateBifactor(data, options)` calibrates models in which every item loads on the primary dimensions, one for a
bifactor model and two, correlated, for a two-tier model, and on at most one specific dimension, such as a passage or a
method factor. The specific dimensions are uncorrelated with everything. It is `calibrateMultidimensional` with the
dimensions integrated one group of items at a time, so the cost follows the primary dimensions plus one, however many
specific dimensions there are.

```mermaid
flowchart LR
  P["Primary dimensions: 1, or 2 correlated"] --> A["Items of specific dimension A"]
  P --> B["Items of specific dimension B"]
  P --> N["Items without a specific dimension"]
  SA["Specific dimension A"] --> A
  SB["Specific dimension B"] --> B
```

```ts
import { createRandom } from "@measurekit/psychometrics";
import {
  calibrateBifactor,
  multidimensionalEstimates,
  multidimensionalPosteriorModeEstimate,
  multidimensionalTwoParameterLogistic,
} from "@measurekit/psychometrics/multidimensional";
import { simulateMultidimensionalResponses } from "@measurekit/psychometrics/sim";

// Twelve items in three passages of four: each loads on a general dimension and on its passage's own; 2,000 persons.
const random = createRandom(5);
const items = Array.from({ length: 12 }, (_, i) => {
  const a = [1.2, 0, 0, 0];
  a[1 + Math.floor(i / 4)] = 0.9;
  return {
    model: multidimensionalTwoParameterLogistic,
    parameters: { a, intercept: ((i % 4) - 1.5) * 0.6, c: 0, d: 1 },
  };
});
const thetas = Array.from({ length: 2000 }, () => [random.normal(), random.normal(), random.normal(), random.normal()]);
const matrix = simulateMultidimensionalResponses(items, thetas, random);

// The specific dimension of each item by id; `null` for an item without one.
const specific = Object.fromEntries(matrix.itemIds.map((id, i) => [id, `passage${1 + Math.floor(i / 4)}`]));
const fit = calibrateBifactor(matrix, { specific, standardErrors: "hessian" });
fit.dimensions; // ["general", "passage1", "passage2", "passage3"]
fit.items[0]?.parameters; // { a: [1.19, 1.07, 0, 0], intercept: -0.96, c: 0, d: 1 }
fit.items[0]?.standardErrors; // { a1: 0.11, a2: 0.17, intercept: 0.08 }
fit.items[5]?.parameters.a; // [1.08, 0, 0.87, 0]
fit.logLikelihood; // -14468.29
fit.converged; // true

// A person's thetas on all dimensions, with the calibration's correlations as the prior.
const prior = { mean: [0, 0, 0, 0], covariance: fit.correlations };
const scores = multidimensionalEstimates(matrix, fit.items, (calibrated, responses) =>
  multidimensionalPosteriorModeEstimate(calibrated, responses, { prior }),
);
scores[0]?.estimate?.theta; // [0.63, 0.14, 0.14, 0.22]
```

| Option           | Default                                        | Meaning                                                                                                                                 |
| ---------------- | ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `specific`       | required                                       | Each item's specific dimension by item id, every item of the data: a name, or `null` for an item with none                              |
| `primary`        | one dimension, `"general"`, for every item     | Each item's primary dimensions by item id: one name for a bifactor model, up to two in all for a two-tier model                         |
| `model`          | `multidimensionalTwoParameterLogistic`         | The model of every item: also the 3PL and the graded model                                                                              |
| `priors`         | none                                           | Priors by parameter name, `"a"`, `"intercept"` and `"c"`: estimates are posterior modes, the usual way to estimate the 3PL's asymptotes |
| `points`         | `31` with one primary dimension, `21` with two | Nodes per integrated dimension, at least 2, from -6 to 6 standard deviations                                                            |
| `standardErrors` | `"none"`                                       | `"hessian"`, or `"cross-product"`, as in `calibrateMultidimensional`                                                                    |
| `tolerance`      | `1e-4`                                         | Converged when no parameter changes by more than this in an EM iteration                                                                |
| `maxIterations`  | `500`                                          | The most EM iterations                                                                                                                  |

The result is that of `calibrateMultidimensional`. Its dimensions are the primary ones, then the specific ones in order
of first appearance among the items, and an item has a slope of 0 on each specific dimension it does not load on. The
correlations are the identity, except between two primary dimensions, which are estimated and have a standard error.
The log-likelihood is exact on the grid, and the number of parameters of `aic` and `bic` counts the estimated slopes,
intercepts and asymptotes, and that correlation. Scoring needs no new function: `multidimensionalEstimates` works on the
result, on the grid of all dimensions. The posterior mean integrates over that grid, so it needs a small `points` with
many specific dimensions; the posterior mode does not.

| Rule                                | Definition                                                                                           |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Models                              | The multidimensional 2PL, 3PL and graded models, one for all items                                   |
| Two primary dimensions              | A two-tier model: the correlation of the two is estimated, the specific dimensions stay uncorrelated |
| Missing responses                   | Skipped; a person without responses changes nothing                                                  |
| An item in two specific dimensions  | `DataError`: `calibrateMultidimensional` with `loadings` fits it by MH-RM                            |
| A specific dimension of one item    | `DataError`: one item does not identify a specific dimension                                         |
| More than two primary dimensions    | `DataError`: `calibrateMultidimensional` with `loadings` fits them by MH-RM                          |
| A name that is primary and specific | `DataError`                                                                                          |
| A grid of more than 10⁶ nodes       | `DataError`: use fewer `points`                                                                      |

**How it is fast.** Given the primary dimensions, the groups of items of the specific dimensions are independent, so a
person's likelihood is a sum over the primary nodes of the weight times a product, over the groups, of a sum over each
group's own nodes. The sum of a group depends only on the responses to its items. Three things make a calibration take
a fraction of the time of the same model in full:

| Part                                | Effect                                                                                                       |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Distinct response patterns          | Persons with one pattern are one row of the E-step                                                           |
| Distinct sub-patterns of each group | A group's sum over its nodes is computed once for each pattern of the responses to its items, not per person |
| SQUAREM                             | The acceleration of the other EM calibrations of the module                                                  |

A group of items whose persons all have different sub-patterns is the worst case: the cost is then the persons times the
nodes squared times the items of the group, as with any bfactor program.

CPU time of one calibration at the default `tolerance`, on an Apple M5 with Node.js 26, against `mirt::bfactor` with
31 nodes on the same simulated data of 2,000 persons, each item loading on the general dimension and on one specific
dimension:

| Data                            | measurekit | mirt  | Ratio |
| ------------------------------- | ---------- | ----- | ----- |
| 12 items, 3 specific dimensions | 0.08 s     | 1.4 s | 17    |
| 40 items, 4 specific dimensions | 0.51 s     | 4.2 s | 8.3   |

## Rotation

`rotateLoadings(calibration, options)` rotates the standardized loadings of a calibration of uncorrelated dimensions,
such as an exploratory one, by gradient projection from the identity rotation.

| Option                       | Default         | Meaning                                                                                     |
| ---------------------------- | --------------- | ------------------------------------------------------------------------------------------- |
| `method`                     | `"oblimin"`     | `"varimax"`, orthogonal, or `"oblimin"`, oblique                                            |
| `normalize`                  | `false`         | Scale each item's loadings to unit length before rotating and back after (Kaiser)           |
| `gamma`                      | `0`             | The γ of oblimin: 0 is quartimin, 0.5 biquartimin. Ignored by varimax                       |
| `tolerance`, `maxIterations` | `1e-10`, `1000` | The norm of the projected gradient at which the rotation has converged; the most iterations |

For a rotation T, the rotated loadings are Λ (Tᵀ)⁻¹ and the correlations between the rotated dimensions TᵀT. The
rotated slopes are a (Tᵀ)⁻¹ = 1.702 λ / √(1 - h²), the communalities h² unchanged, and the intercepts are those of the
calibration, so the rotated `items` with the rotated `correlations` as the prior score as the original ones do, in the
rotated dimensions. The sign of each rotated dimension is chosen so that its largest loading in absolute value is
positive. The result also holds `rotation` (T), `converged` and `iterations`.

```ts
import { calibrateMultidimensional, rotateLoadings } from "@measurekit/psychometrics/multidimensional";

const exploratory = calibrateMultidimensional(matrix, { model: multidimensionalTwoParameterLogistic, dimensions: 2 });
const rotated = rotateLoadings(exploratory, { method: "oblimin" });
rotated.loadings.slice(0, 2); // [[0.61, -0.04], [0.51, 0.08]]
rotated.correlations; // [[1, 0.44], [0.44, 1]]
rotated.converged; // true
```

A calibration with correlated dimensions is not rotated: `rotateLoadings` throws `DataError` unless its correlations
are the identity and it has at least two dimensions.

## Scoring

| Function                                                           | Estimate                                                                                                       |
| ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| `multidimensionalPosteriorMeanEstimate(items, responses, options)` | EAP: the posterior mean, with the posterior covariance, by quadrature                                          |
| `multidimensionalPosteriorModeEstimate(items, responses, options)` | MAP: the posterior mode; the covariance is the inverse of the test information plus the precision of the prior |
| `multidimensionalEstimates(data, items, estimator)`                | The estimate of every person of a data set, `null` without responses                                           |

`items` are `{ model, parameters }` of one number of dimensions; `responses` has one response per item, `null` when
missing. An estimate has `theta`, `standardErrors` and `covariance`.

| Option   | Of       | Default                                                     | Meaning                                                                                              |
| -------- | -------- | ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `prior`  | EAP, MAP | N(0, I)                                                     | `{ mean, covariance }`: pass the calibration's `correlations` as the covariance                      |
| `points` | EAP      | `61`, `31`, `15`, `9`, `7` for 1 to 5 dimensions, `3` above | Nodes per dimension of the rectangular grid placed on the prior: `mean + sd * z` with z from -6 to 6 |

The covariance of the prior is symmetric and positive definite. EAP weights the grid nodes by the prior density times
the likelihood. A grid of more than 10⁶ nodes throws, naming MAP, which works in any number of dimensions: a scan of 5
points per dimension (3 above three dimensions) from -3 to 3 prior standard deviations picks the start of Fisher scoring
with halved steps, which runs to a gradient below 1e-10. Without responses the estimate is the prior, also without
items, when a `prior` gives the dimensions: an estimate from no items needs one.

`multidimensionalEstimates` matches the data's item ids to `items`, estimates each person from the person's answered
items only, and takes `multidimensionalPosteriorModeEstimate` unless given an estimator, a function of items and
responses such as `(items, responses) => multidimensionalPosteriorMeanEstimate(items, responses, { prior })`.

## Information

`multidimensionalInformation(items, theta)` returns the test information matrix at a vector of thetas: the sum over
the items of Σ_k ∇P_k ∇P_kᵀ / P_k. The information of an item in the direction of a unit vector u is uᵀ I u, and the
inverse of the matrix is the covariance of a maximum likelihood estimate of theta there.

## Simulation, parameterization

`simulateMultidimensionalResponses` ([simulation](sim.md#multidimensional-responses)) draws responses from a
vector of thetas per person; `toMirtMultidimensional` and `fromMirtMultidimensional` ([parameterization](parameterization.md#multidimensional-items))
convert items to and from mirt.

## Not included

- Noncompensatory models; multidimensional GPCM and nominal models.
- Bifactor models with an item in two specific dimensions, correlated specific dimensions or more than two primary
  dimensions: `calibrateMultidimensional` with `loadings` fits them by MH-RM.
- Multidimensional linking of separate calibrations: groups are calibrated together, on one scale.
- QMC-EM.
- Standard errors of the means and covariances of the groups; the items' standard errors account for them.
- Items whose parameters differ between groups; the groups share one set of item parameters.
- Multidimensional person fit and DIF.

## Differences from mirt

- **Exploratory models.** mirt leaves all slopes of an exploratory model free, so that the likelihood is flat along the
  rotations and its EM runs long: SAT12 scored with 31 points did not converge in 5,000 iterations at `TOL = 1e-10`. Here the first
  D - 1 items' later slopes are fixed at 0, as above. With the same constraint given to mirt, its EM stops short of the
  maximum that EM here and a base R optimizer reach: 14.5 log-likelihood units below on a simulated clear structure
  and 0.21 below on SAT12.
- **Where EM stops.** The M-step takes one Fisher-scoring step, not a full optimization, and the correlations
  maximize the likelihood on the grid. The estimates agree with mirt's within the tolerances in
  [validation](validation.md); mirt's EM stops short on flat likelihoods whatever its `TOL`.
- **Bifactor models.** `calibrateBifactor` is mirt's `bfactor` with `model2` for a two-tier model, on the same
  rectangular grid, and its estimates, standard errors and log-likelihood agree within the tolerances in
  [validation](validation.md). Without priors the asymptotes of the 3PL make the likelihood flat: on a simulation of 4,000
  persons and 12 items mirt reached its limit of 5,000 cycles at `TOL = 1e-10` short of the maximum, which EM here reaches
  (0.06 higher in log-likelihood), so the validation puts a prior on logit c.
- **Communalities.** mirt's `summary` sums the squared loadings, which is the communality only for uncorrelated
  dimensions; the communality here is λᵀ Φ λ.
- **MAP standard errors.** They come from the expected information, as `posteriorModeEstimate` does; mirt's come from the
  observed Hessian, which differs for the graded model.
- **Rotation.** GPArotation's default algorithm, Barzilai-Borwein steps with a nonmonotone line search; mirt's
  `summary` and `fscores` rotate to GPArotation's default tolerance, 1e-5.
- **Priors.** A prior of c is on the probability scale; mirt's `norm` on `g` is on its logit.
- **Dimensions.** EM calibrates up to three dimensions, MH-RM any number; scoring has no limit.
- **Groups.** The options `groups` and `fixed` stand for mirt's `invariance` and the `est` column of `pars`. mirt cannot
  constrain parameters that it does not estimate, so for fixed items it needs the other items named in `invariance`;
  here `fixed` frees the means and covariance of the first group as well. mirt's covariance of a group is free in
  every entry its model frees; here an uncorrelated model has a diagonal covariance in every group, as mirt's.
- **MH-RM estimates.** The stages, gains and tolerance are mirt's defaults, and the estimates agree within Monte Carlo
  error, not to the digits that EM does: on 800 persons, 25 graded items and 5 dimensions, mirt's four runs differ from
  their mean by up to 0.08 in a parameter, and a run here from that mean by up to 0.17.
- **MH-RM information.** A cycle's information is the observed complete-data information at the draws, as mirt's
  `Deriv` gives it. The expected information is about 10% larger and makes the estimates of weakly determined directions
  noisier, so it is not used.
- **MH-RM standard errors.** mirt accumulates the score and information over at least 400 cycles in a running average;
  here they are the person-wise Louis estimate above. On the same data the standard errors here are a median 1.03 of
  the mean of mirt's four runs, with a quarter of them within 0.96 to 1.08; mirt's own vary 20% from seed to seed
  at the median and 79% at most, and some are missing where its information is not positive definite.
- **MH-RM log-likelihood.** The importance-sampling estimate here agrees within about 1 with the exact log-likelihood
  of EM on three dimensions. On the five-dimensional data mirt's reported log-likelihood is 24 to 29 below the estimate
  here, at mirt's own parameters as well, so the two are not compared.

## References

- Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters: Application of an EM
  algorithm. Psychometrika, 46, 443-459. https://doi.org/10.1007/BF02293801
- Reckase, M. D. (2009). Multidimensional item response theory. Springer. https://doi.org/10.1007/978-0-387-89976-3
- Samejima, F. (1969). Estimation of latent ability using a response pattern of graded scores. Psychometrika Monograph
  Supplement, 17. https://doi.org/10.1007/BF03372160
- Chalmers, R. P. (2012). mirt: A multidimensional item response theory package for the R environment. Journal of
  Statistical Software, 48(6). https://doi.org/10.18637/jss.v048.i06
- Jennrich, R. I. (2001). A simple general procedure for orthogonal rotation. Psychometrika, 66, 289-306.
  https://doi.org/10.1007/BF02294840
- Jennrich, R. I. (2002). A simple general method for oblique rotation. Psychometrika, 67, 7-19.
  https://doi.org/10.1007/BF02294706
- Bernaards, C. A., & Jennrich, R. I. (2005). Gradient projection algorithms and software for arbitrary rotation
  criteria in factor analysis. Educational and Psychological Measurement, 65, 676-696.
  https://doi.org/10.1177/0013164404272507
- Camilli, G. (1994). Origin of the scaling constant d = 1.7 in item response theory. Journal of Educational and
  Behavioral Statistics, 19, 293-295. https://doi.org/10.3102/10769986019003293
- Robbins, H., & Monro, S. (1951). A stochastic approximation method. The Annals of Mathematical Statistics, 22,
  400-407. https://doi.org/10.1214/aoms/1177729586
- Louis, T. A. (1982). Finding the observed information matrix when using the EM algorithm. Journal of the Royal
  Statistical Society: Series B, 44, 226-233. https://doi.org/10.1111/j.2517-6161.1982.tb01203.x
- Cai, L. (2010). High-dimensional exploratory item factor analysis by a Metropolis-Hastings Robbins-Monro algorithm.
  Psychometrika, 75, 33-57. https://doi.org/10.1007/s11336-009-9136-x
- Cai, L. (2010). Metropolis-Hastings Robbins-Monro algorithm for confirmatory item factor analysis. Journal of
  Educational and Behavioral Statistics, 35, 307-335. https://doi.org/10.3102/1076998609353115
