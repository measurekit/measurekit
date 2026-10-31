# Dimensionality

`@measurekit/psychometrics/dimensionality` estimates the correlations of the normal variables behind item responses,
polychoric for items with several categories and tetrachoric for dichotomous items, and runs Horn's parallel analysis on
them to suggest how many dimensions a test has. The correlations are validated against psych and polycor, the bivariate
normal distribution against mvtnorm, and the eigenvalues against base R and psych ([validation](validation.md)).

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

Cost grows with the pairs of items, the persons and the product of the items' categories: the 25 items of `bfi`, with
six categories and 2,800 persons, take about 20 ms, and a parallel analysis of 100 replications under a second. Items
with many categories, such as scores, suit Pearson correlations.

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

```ts
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

## Not included

- Exploratory factor analysis beyond the one-factor minres solution, Velicer's minimum average partial test, DETECT and
  DIMTEST.
- Smoothing of correlation matrices that are not positive definite.
- Polyserial correlations of scores with categories, and standard errors of the correlations.
- Weights of persons, and a correlation matrix as input: the functions take responses.

## Differences from psych

| Topic                     | This package                                                                  | psych                                                                         |
| ------------------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Maximum of the likelihood | Newton's method with derivatives, to 1e-13; the maximum of `optimize`         | `optimize` with its default tolerance, within about 2e-5 of the maximum       |
| Correlation matrix        | As estimated: not smoothed                                                    | `smooth = TRUE` by default, which makes the matrix positive definite          |
| `correct`                 | Two items with two categories each, over the pair's persons                   | The same for `tetrachoric`; `polychoric` ignores it with global thresholds    |
| Thresholds                | From all responses, over the observed categories                              | The same (`global = TRUE`), over a range of categories shared by every item   |
| Random data               | Each item permuted among the persons (`resampled`), or normal data (`normal`) | `fa.parallel` resamples each item with replacement, and also simulates normal |
| Suggested numbers         | From the quantile of the comparison you choose                                | From the simulated normal data, also for polychoric correlations              |
| Replications              | 100                                                                           | 20 (`n.iter`)                                                                 |
| Factor eigenvalues        | One-factor minres by Newton's method from the centroid loadings, to 1e-8      | `fa(fm = "minres")` by L-BFGS-B from the squared multiple correlations        |
| Items without variance    | `null` in the correlations; an error in parallel analysis                     | Deleted with a message                                                        |

## References

- Genz, A. (2004). Numerical computation of rectangular bivariate and trivariate normal and t probabilities. Statistics
  and Computing, 14, 251-260.
- Glorfeld, L. W. (1995). An improvement on Horn's parallel analysis methodology for selecting the correct number of
  factors to retain. Educational and Psychological Measurement, 55, 377-393.
- Horn, J. L. (1965). A rationale and test for the number of factors in factor analysis. Psychometrika, 30, 179-185.
- Olsson, U. (1979). Maximum likelihood estimation of the polychoric correlation coefficient. Psychometrika, 44,
  443-460.
