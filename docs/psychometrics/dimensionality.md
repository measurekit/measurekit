# Dimensionality

`@measurekit/psychometrics/dimensionality` estimates the correlations of the normal variables behind item responses,
polychoric for items with several categories and tetrachoric for dichotomous items, and asks how many dimensions a test
has: by Horn's parallel analysis and by Velicer's minimum average partial test. It fits that many factors to the items by
the minimum residual method or maximum likelihood, rotated to a simple structure, with the fit of the factors, finds a
general factor behind them with omega hierarchical, and smooths correlation matrices that are not positive definite. The correlations are validated against psych and polycor,
the bivariate normal distribution against mvtnorm, and the eigenvalues, the factors, their fit and the partial
correlations against base R and psych ([validation](validation.md)).

```mermaid
flowchart LR
  R["Response matrix"] --> C["Correlations: Pearson or polychoric"]
  C --> PA["parallelAnalysis: how many dimensions, against random data"]
  C --> S["Smoothed if not positive definite"]
  S --> MAP["minimumAveragePartial: how many, by partial correlations"]
  S --> F["exploratoryFactorAnalysis: minres or maximum likelihood, rotation"]
  F --> L["Loadings, communalities, correlations of the factors, fit"]
  L --> SL["schmidLeiman, omegaHierarchical: a general factor"]
```

## Correlations

`polychoricCorrelations(matrix, options)` takes a [response matrix](data.md): responses are integers from 0, `null` when
missing, with any number of categories per item. Pearson correlations of scores treat the categories as equally spaced
numbers; a polychoric correlation instead takes each item to be a normal variable cut at thresholds into its categories,
and estimates the correlation of those variables.

| Option       | Default | Meaning                                                                                                         |
| ------------ | ------- | --------------------------------------------------------------------------------------------------------------- |
| `correction` | `0.5`   | A number of 0 or more added to each empty cell of a table of two items with two categories each; 0 adds nothing |

The result holds the `itemIds`, the `correlations` and each item's `thresholds`.

| Field          | Meaning                                                                                                      |
| -------------- | ------------------------------------------------------------------------------------------------------------ |
| `correlations` | A symmetric matrix by item, 1 on the diagonal; `null` where undefined, below                                 |
| `thresholds`   | Φ⁻¹ of each item's cumulative proportions over its observed categories but the last: one fewer than they are |

```ts
import { createRandom } from "@measurekit/psychometrics";
import { parallelAnalysis, polychoricCorrelations } from "@measurekit/psychometrics/dimensionality";
import { responseMatrix } from "@measurekit/psychometrics/data";

// 800 persons answer 12 items: items 1 to 6 measure one trait and items 7 to 12 another, independent of the first.
const random = createRandom(31);
const rows = Array.from({ length: 800 }, () => {
  const traits = [random.normal(), random.normal()];
  return Array.from({ length: 12 }, (_, j) => {
    const trait = traits[j < 6 ? 0 : 1] ?? 0;
    return random.uniform() < 1 / (1 + Math.exp(-1.7 * trait)) ? 1 : 0;
  });
});
const matrix = responseMatrix(rows);

const { correlations, thresholds } = polychoricCorrelations(matrix);
correlations[0]?.[1]; // 0.557: items 1 and 2 share a trait
correlations[0]?.[6]; // -0.031: items 1 and 7 do not
thresholds[0]; // [-0.006]: one for a dichotomous item
```

Each item's thresholds come from all the persons who answered it. For each pair of items, the correlation ρ maximizes
the log-likelihood Σ n_kl ln π_kl(ρ) of the table of the persons who answered both, where π_kl is the probability of
cell (k, l) under the bivariate normal distribution with correlation ρ between the thresholds, as Olsson's (1979)
two-step estimate. Newton's method with the exact derivatives, which are bivariate normal densities, runs inside a
bracket that bisection narrows, from the Pearson correlation of the category numbers. The bivariate normal distribution
function is Genz's (2004) algorithm, accurate to about 1e-15.

Categories are the responses an item has, in order: an item answered 0, 2 and 5 has three categories, like one answered
0, 1 and 2. Missing responses are skipped for the pairs they are in. The correction follows psych's `tetrachoric`: it
replaces an empty cell of a table of two dichotomous items by `correction` counts, for the pair's persons. Without it, a
table with an empty cell where the association is perfect gives 1 or -1, where the likelihood increases to the boundary.
Tables of items with more categories are not corrected, as psych's `polychoric` does not with its default thresholds
from all responses.

| Case                                            | Result                                              |
| ----------------------------------------------- | --------------------------------------------------- |
| An item with fewer than two observed categories | `null` in its row and column, the diagonal included |
| A pair of items that no person answered both    | `null`                                              |
| A table with one person                         | A number from -1 to 1, which carries no information |
| An item with more than 8 observed categories    | A `DataError`, as psych's `polychoric` stops        |

Cost grows with the pairs of items, the persons and the product of the items' categories: the 25 items of `bfi`, with
six categories and 2,800 persons, take about 20 ms, and a parallel analysis of 100 replications under a second. An item
has at most 8 observed categories, psych's `max.cat`, as a pair's search takes the bivariate normal distribution at
every pair of thresholds; items with more, such as scores, suit Pearson correlations, which every function here takes
with `correlation: "pearson"`.

## Parallel analysis

`parallelAnalysis(matrix, random, options)` asks how many dimensions the data support: it compares the eigenvalues of
the correlations with the eigenvalues of correlations of random data of the same size, and counts the leading ones that
exceed what chance gives. It needs at least three items, each with at least two observed categories, and a correlation
for every pair, so that every pair of items shares a person.

| Option         | Default        | Meaning                                                                                                                  |
| -------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `correlation`  | `"polychoric"` | `"polychoric"` for item responses; `"pearson"` for scores treated as continuous, over the persons who answered a pair    |
| `comparison`   | `"resampled"`  | `"resampled"`: each item's responses permuted among the persons; `"normal"`: standard normal data, with `"pearson"` only |
| `replications` | `100`          | The number of random data sets, a positive integer; psych's default is 20                                                |
| `quantile`     | `0.95`         | The quantile of the random eigenvalues that the observed ones must exceed, from 0 to 1                                   |
| `correction`   | `0.5`          | As for [`polychoricCorrelations`](#correlations)                                                                         |

Randomness comes from the `random` argument, so a seeded generator reproduces the result.

```ts continues
const result = parallelAnalysis(matrix, createRandom(1));
result.components.observed; // [3.671, 3.406, 0.647, 0.611, ...]: 12 eigenvalues of the correlation matrix
result.components.randomQuantile; // [1.4, 1.274, 1.206, 1.153, ...]: of random data
result.components.suggested; // 2
result.factors.observed; // [3.185, 2.423, 0.088, 0.035, ...]
result.factors.randomQuantile; // [1.049, 0.294, 0.231, 0.177, ...]
result.factors.suggested; // 2
```

The result has two `EigenvalueComparison`s, one for the components and one for the factors.

| Comparison   | Eigenvalues of                                                                                               |
| ------------ | ------------------------------------------------------------------------------------------------------------ |
| `components` | The correlation matrix, with 1 on its diagonal: the principal components                                     |
| `factors`    | The correlation matrix with the communalities λ² of the one-factor minimum residual solution on its diagonal |

| Field            | Meaning                                                                                           |
| ---------------- | ------------------------------------------------------------------------------------------------- |
| `observed`       | The eigenvalues of the data's correlations, largest first                                         |
| `randomMean`     | The mean of each eigenvalue over the random data sets                                             |
| `randomQuantile` | The `quantile` of each eigenvalue over the random data sets                                       |
| `suggested`      | The number of leading observed eigenvalues above the random quantile, up to the first that is not |

A matrix that is not positive definite, as polychoric correlations can be, is used as it is, without smoothing, so that
`observed` and the random eigenvalues can be negative. The sum of the component eigenvalues is the number of items.

The factor eigenvalues need the one-factor minres loadings: the λ that minimize Σ_(i<j) (r_ij - λ_i λ_j)². Each loading
is at most √0.995 in absolute value, as psych keeps each uniqueness at least 0.005. Where one factor does not fit, as in
random data, the residuals have no minimum without the bound, and with it a loading is often at the bound, so that the
first factor eigenvalue of random data is close to 1 in some data sets: the random quantile of the first factor in the
example is 1.049, far above the 0.294 of the second, so that the first factor of the data must exceed it to count. psych
behaves alike: its quantile on these data is 1.053. The residuals can have several local minima; the search starts from
the centroid loadings, as `mcdonaldOmega` does, and reaches one of them.

The `"resampled"` comparison permutes each item's responses among the persons, independently for each item, which keeps
each item's distribution and its missing responses and breaks the association between items, and computes the
correlations as for the data. The `"normal"` comparison draws standard normal scores for every person and item, as
psych's simulated data do, and takes their Pearson correlations: it suits `correlation: "pearson"`, and `"polychoric"`
with it throws `DataError`.

## Exploratory factor analysis

`exploratoryFactorAnalysis(matrix, options)` fits `factors` common factors to the items by the minimum residual (minres)
method, as `psych::fa(fm = "minres")`, or by maximum likelihood, as `fa(fm = "ml")`. It takes a
[response matrix](data.md), correlates the items, smooths the correlations if they are not positive definite, and
returns the rotated loadings, the communalities, the correlations between the factors and the fit of the factors.

| Option          | Default     | Meaning                                                                                                                 |
| --------------- | ----------- | ----------------------------------------------------------------------------------------------------------------------- |
| `factors`       | required    | The number of factors, a positive integer that the items identify: 12 items identify at most 7                          |
| `correlation`   | `"pearson"` | `"pearson"` for scores treated as continuous, over the persons who answered a pair; `"polychoric"` for item responses   |
| `extraction`    | `"minres"`  | `"minres"`, the least squared residuals; `"maximum-likelihood"`, Jöreskog's likelihood concentrated on the uniquenesses |
| `rotation`      | `"oblimin"` | `"none"`; `"varimax"`, orthogonal with Kaiser's normalization; `"oblimin"`, oblique. One factor is not rotated          |
| `tolerance`     | `1e-6`      | The largest change of a communality in an iteration at which the fit stops                                              |
| `maxIterations` | `100`       | The most iterations of the fit                                                                                          |
| `correction`    | `0.5`       | As for [`polychoricCorrelations`](#correlations)                                                                        |

| Field           | Meaning                                                                                                     |
| --------------- | ----------------------------------------------------------------------------------------------------------- |
| `itemIds`       | The item ids, in the matrix's order                                                                         |
| `loadings`      | Items by factors; the factors are in the order of the variance they account for                             |
| `communalities` | The part of each item's variance that the factors share                                                     |
| `correlations`  | Between the factors, by rows: the identity unless the rotation is oblique                                   |
| `converged`     | Whether the fit and the rotation both reached their tolerances                                              |
| `iterations`    | The iterations of the fit                                                                                   |
| `smoothed`      | Whether the correlations of the items were not positive definite and were smoothed                          |
| `fit`           | The fit of the factors, below; `null` for polychoric correlations, or a model that is not positive definite |

```ts continues
import { exploratoryFactorAnalysis } from "@measurekit/psychometrics/dimensionality";

// `matrix` is the response matrix of the example above: two independent traits, six items each.
const fit = exploratoryFactorAnalysis(matrix, { factors: 2, correlation: "polychoric" });
fit.loadings.slice(0, 2); // [[0.774, -0.021], [0.753, 0.025]]: items 1 and 2 load on the first factor
fit.loadings.slice(6, 8); // [[-0.01, 0.717], [-0.073, 0.718]]: items 7 and 8 on the second
fit.communalities.slice(0, 3); // [0.599, 0.569, 0.476]
fit.correlations; // [[1, 0.017], [0.017, 1]]: the traits are independent
fit.converged; // true
```

**The extraction.** The unknowns are the communalities h, which lie from 0 to 0.995: each item's uniqueness is at least
0.005, as in psych's `fa`. Each extraction measures what the factors leave out at some h:

| Extraction             | Objective                                                                                                                              | Loadings                                                                      |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `"minres"`             | Half the sum of the squared eigenvalues of R with h on its diagonal that the factors leave out: the squared residuals off the diagonal | The leading eigenvectors of that matrix, times the roots of their eigenvalues |
| `"maximum-likelihood"` | Σ (γ - ln γ - 1) over the eigenvalues γ of Ψ^(-1/2) R Ψ^(-1/2) that the factors leave out, with Ψ = 1 - h                              | Ψ^(1/2) times the leading eigenvectors, times √(γ - 1)                        |

Each iteration takes a Newton step on the communalities, with the Hessian that the objective has at a perfect fit, a
backtracking line search that keeps the bounds, and an eigendecomposition of the matrix. The search starts from the
squared multiple correlations. Convergence takes 5 to 20 iterations and a few milliseconds for 25 items. Fabrigar et
al. (1999) call maximum likelihood the best choice for data near the normal distribution, as it gives a test of the
fit, and principal axis factoring, whose limit is minres, the choice for data far from it.

**The solution.** The unrotated loadings, those of the table above, are rotated by gradient projection, with the oblimin criterion without normalization, as
`GPArotation::oblimin`, or the varimax criterion with each item's loadings scaled to unit length, as `stats::varimax`
does. Each factor's sign makes its loadings sum to a positive number, and the factors follow the variance they account
for, the diagonal of Φ ΛᵀΛ, as psych orders them. The communalities are the diagonal of Λ Φ Λᵀ for every rotation.

A communality that the fit pushes beyond 0.995, a Heywood case, stays at the bound, and the communality of the loadings
can exceed it, as it does in psych, for either extraction; `converged` is `true`. More factors than the items
identify throw `DataError`: the number of correlations must be at least the number of loadings less the rotations that
leave the model as it is.

**The fit.** For Pearson correlations, `fit` holds what `psych::fa` reports for either extraction, for p items, m
factors and the N persons of the matrix. The model's correlations M are Λ Φ Λᵀ off the diagonal and 1 on it.

| Field              | Definition                                                                                                                              |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| `chiSquare`        | (N - 1 - (2p + 5) / 6 - 2m / 3) F, at least 0, with F = tr(M⁻¹ R) - ln det(M⁻¹ R) - p: the likelihood ratio, with Bartlett's correction |
| `degreesOfFreedom` | p (p - 1) / 2 - p m + m (m - 1) / 2                                                                                                     |
| `significance`     | P(χ² > `chiSquare`)                                                                                                                     |
| `rmsea`            | √max(χ² / (df N) - 1 / (N - 1), 0)                                                                                                      |
| `rmseaInterval`    | Its 90% interval: √(λ / ((N - 1) df)) at the noncentralities λ that put χ² at the 95th and 5th percentiles                              |
| `tli`              | Tucker and Lewis's index against the independence model, whose F is -ln det R; it can exceed 1                                          |
| `bic`              | χ² - df ln N                                                                                                                            |
| `srmsr`            | The root mean square of the residual correlations R - M over the pairs of items, each weighted by the persons who answered it           |

With 0 degrees of freedom, as one factor of three items, the significance, the RMSEA, its interval and the TLI are
`null`, and the BIC is χ². The chi-square of polychoric correlations has no reference distribution, so their `fit` is
`null`, and so is that of a model whose correlations are not positive definite. Rotation leaves the model, and so the
fit, as it is.

```ts continues
const ml = exploratoryFactorAnalysis(matrix, { factors: 2, extraction: "maximum-likelihood" });
ml.loadings.slice(0, 2); // [[0.638, -0.018], [0.618, 0.02]]: of Pearson correlations, lower than polychoric
ml.fit;
// { chiSquare: 36.1, degreesOfFreedom: 43, significance: 0.762, rmsea: 0, rmseaInterval: [0, 0.0173], tli: 1.006,
//   bic: -251.3, srmsr: 0.0172 }
exploratoryFactorAnalysis(matrix, { factors: 1, extraction: "maximum-likelihood" }).fit?.rmsea; // 0.133: one factor misfits
```

**Which function.** `exploratoryFactorAnalysis` works on the correlations of the items, takes milliseconds and returns
the factors. `calibrateMultidimensional` with `dimensions` estimates a compensatory item response model from the
responses, by full information, and returns item parameters, a likelihood and standard errors, in seconds to minutes.
The two agree on what they share: on the 12 simulated tetrachoric items of the validation, the loadings of the first
factor average 0.710 here and 0.724 by full information after `rotateLoadings`, those of the second 0.606 and 0.618, and
the correlation of the factors 0.316 and 0.306, for the truth of 0.7, 0.6 and 0.3. Use the factors to explore and to
choose the number of dimensions, and the calibration when the item parameters matter.

## Omega hierarchical

A test whose items share a general trait and, within groups, narrower ones has a total score that the general trait
dominates to the degree omega hierarchical measures: the share of the total's variance that the general factor accounts
for. Omega total is the share that all the factors account for. `schmidLeiman(factors)` takes the oblique solution of
`exploratoryFactorAnalysis` and gives each item a loading on a general factor and on group factors orthogonal to it;
`omegaHierarchical(matrix, options)` runs both and returns the omegas, as `psych::omega`.

```mermaid
flowchart LR
  E["exploratoryFactorAnalysis, oblimin: loadings Λ, correlations Φ"] --> G["One factor of Φ by minres: g"]
  G --> S["schmidLeiman: general Λ g, group λ √(1 - g²)"]
  S --> O["omegaHierarchical: (Σ general)² / Σ R; omega total"]
```

| Quantity           | Definition, with R the correlations of the items                                                  |
| ------------------ | ------------------------------------------------------------------------------------------------- |
| General loading    | Σ_k λ_ik g_k, with g the one-factor minres loadings of Φ                                          |
| Group loading      | λ_ik √(1 - g_k²), with the sign of λ_ik; 0 where g_k² exceeds 1                                   |
| Omega hierarchical | (Σ_i general loading)² / Σ R                                                                      |
| Omega total        | 1 - Σ (1 - h_i²) / Σ R, with the communalities h² of the group factors: `mcdonaldOmega`'s formula |

The options are those of [`exploratoryFactorAnalysis`](#exploratory-factor-analysis) but `rotation`, which is
always oblimin, as the general factor comes from the correlations of oblique factors:

| Option    | Default | Meaning                                                                          |
| --------- | ------- | -------------------------------------------------------------------------------- |
| `factors` | `3`     | The group factors, at least 3: two do not identify the general factor's loadings |

| Field          | Meaning                                                                                                              |
| -------------- | -------------------------------------------------------------------------------------------------------------------- |
| `hierarchical` | Omega hierarchical                                                                                                   |
| `total`        | Omega total                                                                                                          |
| `loadings`     | The `SchmidLeiman` loadings: `itemIds`, `general`, `specific` by item and group factor, `communalities`, `converged` |
| `converged`    | Whether the factor analysis, its rotation and the general factor's fit converged                                     |

```ts
import { createRandom, responseMatrix } from "@measurekit/psychometrics";
import { cronbachAlpha } from "@measurekit/psychometrics/ctt";
import { omegaHierarchical } from "@measurekit/psychometrics/dimensionality";

// 1,000 persons answer 9 items scored 0 to 4: a general trait, and three group traits of three items each.
const random = createRandom(7);
const rows = Array.from({ length: 1000 }, () => {
  const general = random.normal();
  const groups = [random.normal(), random.normal(), random.normal()];
  return Array.from({ length: 9 }, (_, j) => {
    const latent = 0.6 * general + 0.45 * (groups[Math.floor(j / 3)] ?? 0) + 0.66 * random.normal();
    return Math.min(4, Math.max(0, Math.round(latent + 2)));
  });
});
const matrix = responseMatrix(rows);

const omega = omegaHierarchical(matrix);
omega.hierarchical; // 0.717: the general trait's share of the total's variance
omega.total; // 0.87
omega.loadings.general.slice(0, 3); // [0.606, 0.566, 0.606]
omega.loadings.specific[0]; // [0.016, 0.386, -0.008]: item 1 on the second group factor
cronbachAlpha(matrix); // 0.832
```

Items count as scored: a reversed item lowers both omegas, as it lowers alpha, so reverse-score it first. An orthogonal
solution has no general factor: `schmidLeiman` of varimax factors gives general loadings of 0. Fewer than three group
factors throw `DataError`, as two leave the general factor's loadings to a constraint. Omega hierarchical is at most
omega total; Revelle and Zinbarg (2009) and McNeish (2018) prefer the omegas to the greatest lower bound, which is not
built ([classical test theory](ctt.md#not-included)).

## Minimum average partial test

`minimumAveragePartial(matrix, options)` is Velicer's test for the number of factors. It removes the principal
components of the correlations one at a time and, after each, squares and averages the partial correlations of the
items. While the components remove common variance the average falls; when they start removing the variance of single
items it rises, and the number of components at the minimum is the number of factors.

| Option        | Default                 | Meaning                                                             |
| ------------- | ----------------------- | ------------------------------------------------------------------- |
| `correlation` | `"pearson"`             | As for [`exploratoryFactorAnalysis`](#exploratory-factor-analysis)  |
| `maxFactors`  | the number of items - 1 | The most components to remove, from 1 to the number of items less 1 |
| `correction`  | `0.5`                   | As for [`polychoricCorrelations`](#correlations)                    |

The result holds `map`, the mean squared partial correlation after 1, 2, ... components, `factors`, the number at the
lowest value, the first when tied, and `smoothed`.

```ts continues
import { minimumAveragePartial } from "@measurekit/psychometrics/dimensionality";

const { map, factors } = minimumAveragePartial(matrix, { correlation: "polychoric" });
map.slice(0, 4); // [0.0448, 0.0539, 0.063, 0.1071]: the lowest is after 1 component, the general trait
factors; // 1
```

`map` is the `map` of psych's `VSS`, which starts at one component, so the test never answers none; Velicer's original also
scores zero components. An item whose correlations with the others are all 0 is a component by itself; once it is
removed, the item's partial correlations are undefined, and the test throws `DataError`. With all the components but one removed, what is left of each pair is a correlation of 1 in
absolute value, so the last value is 1.

## Smoothing

`smoothCorrelations(correlations, size, options)` returns a correlation matrix that is positive definite. A matrix whose
least eigenvalue is below the machine epsilon has its eigenvalues below `tolerance` (`1e-12`) raised to 100 times it,
rescaled to sum to the number of variables, and the matrix rebuilt and rescaled to a correlation matrix, as
`psych::cor.smooth`. A matrix that is positive definite comes back as it is, in a copy.

Polychoric correlations come from pairs of items estimated apart and can fail to be positive definite: the tetrachoric
correlations of `SAT12`, 32 items, have a least eigenvalue of -0.09. `exploratoryFactorAnalysis` and
`minimumAveragePartial` smooth and report it in `smoothed`; `parallelAnalysis` does not, so that its eigenvalues can be
negative.

## Not included

- **Rotations other than varimax and oblimin, such as promax.** Outside the sources: Fabrigar et al. (1999) prefer no
  oblique rotation, as their results are alike; oblimin, `psych::fa`'s default, and varimax are built.
- **The asymptotic omega hierarchical, ECV and the omegas of the group factors.** No fit: each is one line over the
  Schmid-Leiman loadings, as psych's `omega` reports them.
- **WLS and WLSMV extraction.** Outside the sources: options of psych and lavaan.
- **Standard errors of loadings.** Outside the sources: a bootstrap over the persons gives them.
- **Factor scores.** Outside the sources: not confirmed in them; `multidimensionalEstimates` scores persons on the
  full-information route.
- **The Kaiser-Meyer-Olkin measure.** Outside the sources: not confirmed in them; parallel analysis, MAP and the fit
  decide the number of factors.
- **Bartlett's test of sphericity.** Outside the sources: not confirmed in them; it rejects at any useful sample size.
- **Very simple structure.** Outside the sources: a function of psych, beside MAP.
- **Principal components analysis.** No fit: general statistics; `parallelAnalysis` reports the eigenvalues, and
  Fabrigar et al. (1999) set it apart from factor analysis.
- **ICLUST and Bass-Ackward factoring.** Outside the sources: functions of psych.
- **Nonparametric conditional covariances.** Outside the sources: a function of sirt.
- **Statistical implicative analysis.** Outside the sources: a function of sirt.
- **Confirmatory factor analysis of ordinal items.** No fit: structural equation modeling, another domain's tool,
  lavaan.
- **Modification indices and score tests.** No fit: structural equation modeling, lavaan.
- **Multi-group confirmatory factor analysis and measurement invariance.** No fit: structural equation modeling,
  lavaan; `dif` tests items.
- **Structural paths, mediation and full-information maximum likelihood.** No fit: structural equation modeling,
  lavaan.
- **Invariance alignment.** No fit: structural equation modeling, sirt and Mplus.
- **Local structural equation models.** No fit: structural equation modeling, sirt.
- **DETECT.** Outside the sources: a method for clusters of items, beside the parallel analysis, MAP and factor analysis
  that are built.
- **DIMTEST.** No reference: no package that runs it was found.
- **Polyserial correlations of scores with categories.** Outside the sources: not confirmed in them; `itemRest` serves
  polytomous item analysis.
- **Standard errors of the correlations.** Outside the sources: a bootstrap over the persons gives them.
- **Weights of persons.** Outside the sources: the functions take responses.
- **A correlation matrix as input.** Outside the sources: the functions take responses.

## Differences from psych

| Topic                                   | This package                                                                      | psych                                                                                                                    |
| --------------------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Maximum of the likelihood               | Newton's method with derivatives, to 1e-13; the maximum of `optimize`             | `optimize` with its default tolerance, within about 2e-5 of the maximum                                                  |
| Correlation matrix                      | As estimated: not smoothed                                                        | `smooth = TRUE` by default, which makes the matrix positive definite                                                     |
| `correct`                               | Two items with two categories each, over the pair's persons                       | The same for `tetrachoric`; `polychoric` ignores it with global thresholds                                               |
| Thresholds                              | From all responses, over the observed categories                                  | The same (`global = TRUE`), over a range of categories shared by every item                                              |
| Categories of an item                   | At most 8 observed; an item of more throws                                        | `max.cat = 8` by default: `polychoric` stops above it                                                                    |
| Random data                             | Each item permuted among the persons (`resampled`), or normal data (`normal`)     | `fa.parallel` resamples each item with replacement, and also simulates normal                                            |
| Suggested numbers                       | From the quantile of the comparison you choose                                    | From the simulated normal data, also for polychoric correlations                                                         |
| Replications                            | 100                                                                               | 20 (`n.iter`)                                                                                                            |
| Factor eigenvalues                      | One-factor minres by Newton's method from the centroid loadings, to 1e-8          | `fa(fm = "minres")` by L-BFGS-B from the squared multiple correlations                                                   |
| Minres solution                         | Newton steps on the communalities from the squared multiple correlations          | L-BFGS-B on the same objective, within about 1e-5 of its minimum                                                         |
| Maximum likelihood                      | The same Newton steps, with the likelihood's Hessian at a perfect fit, to 1e-6    | L-BFGS-B on the uniquenesses from 0.005 to 1: loadings within 2e-5                                                       |
| Unweighted least squares                | Minres is the solution                                                            | `fm = "uls"` gives the minres solution: within 9e-6 on `bfi`                                                             |
| Principal axis factoring                | Minres is its limit                                                               | `fm = "pa"` run to `min.err = 1e-12` is within 4e-6 of minres on `bfi`; its default `min.err` of 0.001 stops 0.0017 away |
| Interval of the RMSEA                   | Roots in [0, χ²] and [0, max(N - 1, 5 χ²)]                                        | `uniroot` in [0, max(N, χ²) + 2N], to its default tolerance                                                              |
| Reversed items in the omegas            | Taken as scored                                                                   | `omega` reverses the items whose general loading is negative (`flip = TRUE`); the references run with `flip = FALSE`     |
| The general factor of the group factors | Minres, whatever the extraction of the items                                      | `schmid` uses the items' extraction; the same with 3 group factors, which the general factor fits exactly                |
| Two group factors                       | Throws                                                                            | `schmid` sets the two general loadings equal (`option = "equal"`) and warns                                              |
| Residuals of smoothed correlations      | Of the smoothed correlations that the factors fit                                 | Of the correlations before smoothing; the same for a matrix that needs none                                              |
| Varimax                                 | Gradient projection to a gradient of 1e-10, with Kaiser's normalization           | `stats::varimax`, which stops at 1e-5 of its criterion: loadings within 1e-2                                             |
| Smoothed correlations                   | Of the items' correlations in the factors and the test; not in `parallelAnalysis` | Of every correlation matrix `tetrachoric` and `polychoric` return                                                        |
| Items without variance                  | `null` in the correlations; an error in parallel analysis                         | Deleted with a message                                                                                                   |
| An item correlated with none            | `minimumAveragePartial` throws once a component holds all of its variance         | `VSS` stops with an error in `principal`                                                                                 |

## References

- Bock, R. D., Gibbons, R., and Muraki, E. (1988). Full-information item factor analysis. Applied Psychological
  Measurement, 12, 261-280.
- Fabrigar, L. R., Wegener, D. T., MacCallum, R. C., and Strahan, E. J. (1999). Evaluating the use of exploratory factor
  analysis in psychological research. Psychological Methods, 4, 272-299.
- Genz, A. (2004). Numerical computation of rectangular bivariate and trivariate normal and t probabilities. Statistics
  and Computing, 14, 251-260.
- Glorfeld, L. W. (1995). An improvement on Horn's parallel analysis methodology for selecting the correct number of
  factors to retain. Educational and Psychological Measurement, 55, 377-393.
- Horn, J. L. (1965). A rationale and test for the number of factors in factor analysis. Psychometrika, 30, 179-185.
- Jöreskog, K. G. (1967). Some contributions to maximum likelihood factor analysis. Psychometrika, 32, 443-482.
- McDonald, R. P. (1999). Test theory: A unified treatment. Lawrence Erlbaum.
- McNeish, D. (2018). Thanks coefficient alpha, we'll take it from here. Psychological Methods, 23, 412-433.
- Olsson, U. (1979). Maximum likelihood estimation of the polychoric correlation coefficient. Psychometrika, 44,
  443-460.
- Revelle, W., and Zinbarg, R. E. (2009). Coefficients alpha, beta, omega, and the glb: Comments on Sijtsma.
  Psychometrika, 74, 145-154.
- Schmid, J., and Leiman, J. M. (1957). The development of hierarchical factor solutions. Psychometrika, 22, 53-61.
- Velicer, W. F. (1976). Determining the number of components from the matrix of partial correlations. Psychometrika, 41,
  321-327.
- Wothke, W. (1993). Nonpositive definite matrices in structural modeling. In K. A. Bollen and J. S. Long (Eds.), Testing
  structural equation models (pp. 256-293). Sage.
