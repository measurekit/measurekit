# Rater agreement and effects

`@measurekit/psychometrics/rater` measures how far raters agree: exact and adjacent agreement, Cohen's kappa for two
raters, unweighted, linear or quadratic weighted, Fleiss' kappa for two or more, and intraclass correlations of
scores; how far two raters order the subjects alike, by the correlation of their ratings; and how far one rater is more
lenient than another, by the standardized mean difference. It also estimates what
raters do to scores: the many-facet Rasch model puts persons, items and raters on one scale, so that a rater's severity
is a number of logits and a person's score can be adjusted for the raters who gave it, and a bias analysis finds the
raters who are harsher or milder than usual on one item. The agreement statistics are validated against irr, psych and
effsize, the model against base R and TAM ([validation](validation.md)).

## Ratings

Every function takes ratings by subject: one row per rated subject, such as an essay, with one rating per rater,
`null` when a rater did not rate the subject. Agreement and kappa take categories, integers from 0, as item responses; the
correlation, the standardized mean difference and the intraclass correlations take any finite numbers. Each statistic uses the subjects every rater rated.

```ts
import { cohenKappa, fleissKappa, intraclassCorrelation } from "@measurekit/psychometrics/rater";

// Four essays scored 0 to 4 by three raters; rater 3 did not score essay 4.
const ratings = [
  [3, 3, 2],
  [1, 2, 1],
  [4, 4, 4],
  [2, 1, null],
];
cohenKappa(
  ratings.map((row) => row.slice(0, 2)),
  { weights: "quadratic" },
); // QWK of raters 1 and 2
fleissKappa(ratings); // over essays 1 to 3
intraclassCorrelation(ratings); // { oneWay, agreement, consistency, oneWayAverage, ... }
```

## Kappa

| Function                       | Value                                                      | irr             |
| ------------------------------ | ---------------------------------------------------------- | --------------- |
| `cohenKappa(ratings, options)` | (p_o - p_e) / (1 - p_e) of two raters (Cohen, 1960, 1968)  | `kappa2`        |
| `fleissKappa(ratings)`         | (P̄ - P_e) / (1 - P_e) of two or more raters (Fleiss, 1971) | `kappam.fleiss` |

For Cohen's kappa, p_o = Σ w_ij O_ij / n is the weighted observed agreement and p_e = Σ w_ij r_i c_j / n² the agreement
expected from the raters' margins r and c, with agreement weights w_ij by `weights`:

| `weights`          | w_ij                    | irr's `weight` |
| ------------------ | ----------------------- | -------------- |
| `"none"` (default) | 1 when i = j, else 0    | `"unweighted"` |
| `"linear"`         | 1 - \|i - j\| / (K - 1) | `"equal"`      |
| `"quadratic"`      | 1 - (i - j)² / (K - 1)² | `"squared"`    |

K - 1 is the range of the categories used. Quadratic weights give the quadratic weighted kappa (QWK) used to compare
automated with human scoring. The weights use the categories' values; irr numbers the categories used in order, which
gives the same weights when no category between the lowest and highest is unused.

For Fleiss' kappa, P̄ is the mean over subjects of the share of agreeing pairs of raters, Σ_k n_k (n_k - 1) / (m (m
- 1)) for m raters of whom n_k chose category k, and P_e = Σ_k p_k² from each category's share p_k of all ratings.

Kappa is 1 for perfect agreement and 0 for agreement at chance level. When every rating is in one category the chance
agreement is 1 and kappa is undefined: both functions throw `DataError`.

## Exact and adjacent agreement

`exactAgreement(ratings)` and `adjacentAgreement(ratings, options)` return the share of subjects on which the raters
agree: exactly, or within a number of categories. They are the simplest checks of how well automated scores or two
human raters match; kappa corrects them for chance.

```mermaid
flowchart LR
  R[Ratings by subject] --> C[Subjects every rater rated]
  C --> S["Span of each subject: highest rating less lowest"]
  S --> W{"Span at most the width?"}
  W -->|yes| A[The subject agrees]
  W -->|no| D[The subject does not agree]
  A --> P[Share of subjects that agree]
  D --> P
```

| Function                              | Agreement of a subject                                                                       | irr                        |
| ------------------------------------- | -------------------------------------------------------------------------------------------- | -------------------------- |
| `exactAgreement(ratings)`             | Every rater gave the same rating: a span of 0                                                | `agree(tolerance = 0)`     |
| `adjacentAgreement(ratings, options)` | The highest rating less the lowest is at most `width`: with two raters, a difference of that | `agree(tolerance = width)` |

| Option  | Default | Meaning                                                                                             |
| ------- | ------- | --------------------------------------------------------------------------------------------------- |
| `width` | `1`     | Of `adjacentAgreement`: the widest span, in categories, that counts as agreement; an integer from 0 |

Both return a proportion from 0 to 1, where irr reports a percentage. Ratings are categories, integers from 0, from two
raters or more; with more than two, every rating of a subject must lie within the width. They use the subjects every
rater rated. Not included: agreement corrected for chance, which kappa gives.

```ts
import { adjacentAgreement, exactAgreement } from "@measurekit/psychometrics/rater";

// Four essays scored 0 to 4 by three raters; rater 3 did not score essay 4.
const ratings = [
  [3, 3, 2],
  [1, 2, 1],
  [4, 4, 4],
  [2, 1, null],
];
exactAgreement(ratings); // 0.33: the three raters agree on essay 3 alone
adjacentAgreement(ratings); // 1: on essays 1 to 3 no two raters differ by more than 1
adjacentAgreement(
  ratings.map((row) => row.slice(0, 2)),
  { width: 0 },
); // 0.5: raters 1 and 2 agree exactly on two of four essays
```

## Intraclass correlation

`intraclassCorrelation(ratings)` returns the six coefficients of Shrout and Fleiss (1979) and McGraw and Wong (1996),
from the mean squares of an analysis of variance of n subjects by k raters: subjects MS_R, raters MS_C, residual MS_E
and within subjects MS_W.

| Field                | Form               | Value                                                       |
| -------------------- | ------------------ | ----------------------------------------------------------- |
| `oneWay`             | ICC(1)             | (MS_R - MS_W) / (MS_R + (k - 1) MS_W)                       |
| `agreement`          | ICC(A,1), ICC(2,1) | (MS_R - MS_E) / (MS_R + (k - 1) MS_E + k (MS_C - MS_E) / n) |
| `consistency`        | ICC(C,1), ICC(3,1) | (MS_R - MS_E) / (MS_R + (k - 1) MS_E)                       |
| `oneWayAverage`      | ICC(k)             | (MS_R - MS_W) / MS_R                                        |
| `agreementAverage`   | ICC(A,k), ICC(2,k) | (MS_R - MS_E) / (MS_R + (MS_C - MS_E) / n)                  |
| `consistencyAverage` | ICC(C,k), ICC(3,k) | (MS_R - MS_E) / MS_R                                        |

Use the single-rater forms when one rater scores each subject in practice, the average forms when the reported score
is the raters' mean; absolute agreement when raters' leniency matters, consistency when only their ordering does. A
coefficient whose denominator is 0, as when every rating is equal, is `null`.

## Correlation between two raters

`ratingCorrelation(ratings)` is the Pearson correlation between two raters' ratings: how far they order the subjects
alike, whatever their leniency. It is the companion of the [standardized mean difference](#standardized-mean-difference)
in the evaluation of automated scoring: a machine can correlate perfectly with a human rater and still rate everyone
higher, and each statistic shows what the other does not.

```mermaid
flowchart LR
  R[Ratings of two raters] --> C[Subjects both rated]
  C --> D["Deviations from each rater's mean"]
  D --> Q["r = Σ d₁ d₂ / √(Σ d₁² Σ d₂²)"]
```

It uses the subjects both raters rated, needs at least two, takes finite numbers, and is `null` when either rater's
ratings are constant, as R's `cor` gives `NA`. The value lies from -1 to 1, does not change with the order of the raters,
and is unchanged by a shift or a positive rescaling of either rater's ratings.

```ts
import { ratingCorrelation, standardizedMeanDifference } from "@measurekit/psychometrics/rater";

// Raters 1 and 3 of the four essays above; rater 3 did not score essay 4.
ratingCorrelation([
  [3, 2],
  [1, 1],
  [4, 4],
]); // 0.93

// A rater who scores everything 3 higher correlates perfectly and is 2.3 standard deviations harsher.
const shifted = [
  [2, 5],
  [3, 6],
  [1, 4],
  [4, 7],
];
ratingCorrelation(shifted); // 1
standardizedMeanDifference(shifted); // -2.32
```

Not included: Spearman's rank correlation, which ranks the ratings first and is a different statistic, and the correlations
among more than two raters, which pairs of raters give.

## Standardized mean difference

`standardizedMeanDifference(ratings)` measures how much more lenient the first of two raters is than the second, in
standard deviations: the difference of their mean ratings over their pooled standard deviation. It is the measure used
to compare automated with human scores, where a magnitude above 0.15 is commonly taken to flag a difference worth
acting on.

```mermaid
flowchart LR
  R[Ratings of two raters] --> C[Subjects both rated]
  C --> M["Means m₁ and m₂"]
  C --> V["Variances s₁² and s₂², on n - 1"]
  V --> P["Pooled SD, √((s₁² + s₂²) / 2)"]
  M --> D["Difference m₁ - m₂"]
  D --> Q["SMD = (m₁ - m₂) / pooled SD"]
  P --> Q
```

The result is positive when the first rater rates higher on average, and 0 when the means are equal; it does not depend
on the origin or the scale of the ratings. With the same subjects rated by both raters, the pooled standard deviation
is that of Cohen's d, so the value equals effsize's `cohen.d` of the two raters' ratings as independent samples.
Ratings are finite numbers, not only categories: automated scores are often continuous. It uses the subjects both
raters rated, needs at least two, and is `null` when both raters' ratings are constant.

```ts
import { standardizedMeanDifference } from "@measurekit/psychometrics/rater";

// Raters 1 and 3 of the four essays above; rater 3 did not score essay 4.
standardizedMeanDifference([
  [3, 2],
  [1, 1],
  [4, 4],
]); // 0.22: rater 1 is 0.22 standard deviations more lenient
```

Not included: the small-sample correction of Hedges' g, confidence intervals, and the differences among more than two
raters, which pairs of raters give. Where effsize treats the two sets of ratings as independent samples, so does this
function, though the same subjects were rated; it does not use the standard deviation of the differences, as a paired
effect size would.

## Many-facet Rasch model

`manyFacetRasch(scores, options)` estimates the many-facet Rasch model (Linacre, 1989) by marginal maximum likelihood.
Each score names the element of every facet it was given at: the person, the item or task, the rater, and any further
facet such as the occasion. The model says how much each element adds to the score, in logits.

```ts
import { expectedRating, manyFacetEstimates, manyFacetFit, manyFacetRasch } from "@measurekit/psychometrics/rater";

// Twelve essays scored 0 to 3 on content and style by two of three raters: essay, rater, content, style.
const rows = [
  ["e1", "r1", 1, 1],
  ["e1", "r2", 1, 2],
  ["e2", "r2", 2, 2],
  ["e2", "r3", 3, 2],
  ["e3", "r1", 3, 2],
  ["e3", "r3", 2, 3],
  ["e4", "r1", 1, 1],
  ["e4", "r2", 2, 1],
  ["e5", "r2", 3, 2],
  ["e5", "r3", 3, 1],
  ["e6", "r1", 2, 0],
  ["e6", "r3", 3, 3],
  ["e7", "r1", 0, 0],
  ["e7", "r2", 3, 2],
  ["e8", "r2", 2, 0],
  ["e8", "r3", 3, 0],
  ["e9", "r1", 1, 0],
  ["e9", "r3", 1, 1],
  ["e10", "r1", 2, 2],
  ["e10", "r2", 2, 2],
  ["e11", "r2", 1, 2],
  ["e11", "r3", 1, 2],
  ["e12", "r1", 2, 1],
  ["e12", "r3", 1, 2],
] as const;
const scores = rows.flatMap(([person, rater, content, style]) => [
  { levels: { person, item: "content", rater }, score: content },
  { levels: { person, item: "style", rater }, score: style },
]);

const model = manyFacetRasch(scores, { facets: ["item", "rater"] });
model.facets[1]?.elements.map(({ id, measure }) => [id, measure]); // [["r1", 0.77], ["r2", -0.31], ["r3", -0.46]]
model.facets[1]?.elements.map(({ fairAverage }) => fairAverage); // [1.15, 1.87, 1.96]: r1 is the severe rater
model.facets[1]?.reliability; // 0.68
expectedRating(model, 0.5, { rater: "r1" }); // 1.35: the score of a person of theta 0.5 from the severest rater
expectedRating(model, 0.5, { rater: "r3" }); // 2.12
const persons = manyFacetEstimates(scores, model); // EAP: e1 { theta: 0.03, standardError: 0.42 }
manyFacetFit(scores, model, persons).find(({ id }) => id === "r3")?.infitMeanSquare; // 1.22

// A bias for each rater on each item, [item, rater, bias]: no pair stands out in these few scores.
const biased = manyFacetRasch(scores, { facets: ["item", "rater"], interactions: [["item", "rater"]] });
biased.interactions?.map(({ levels, bias }) => [...levels, Math.round(bias * 100) / 100]);
// [["content", "r1", -0.13], ["style", "r1", 0.13], ["content", "r2", 0.07], ["style", "r2", -0.07], ["content", "r3", 0.05], ["style", "r3", -0.05]]
```

### Scores

`scores` is an array of `FacetedScore`s, `{ levels, score }`, one for each score in long format ([data](data.md)):
`levels` holds the element of each facet by facet name, as strings, and `score` an integer from 0.

| Rule                                 | Meaning                                                                                                                     |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| One element of each facet            | The person facet and the facets in `facets`; other keys of `levels` are ignored                                             |
| Connected                            | Each score links its person to its elements; all persons and elements must form one set, or the measures are not comparable |
| Categories 0 to the highest, no gaps | In each rating scale, at least two categories, and every category up to the highest used                                    |
| Two persons                          | At least two persons                                                                                                        |

A person may have several scores for one combination of elements, such as two ratings by one rater of the same item;
they count as independent observations. A design in which raters have no person or item in common is not connected,
and `manyFacetRasch` throws a `DataError` that names the subsets.

### Model

For a score in category k of a person with ability θ, with the measure δ of its element in each facet f, and the
thresholds τ of the rating scale s, the log odds of k against k - 1 are

θ - Σ_f δ_(f, element) - τ_(s, k),

and θ is normal with a mean μ and a standard deviation σ that are estimated. Each distinct combination of the elements
of the facets other than the person is a virtual item, so a person's scores are independent given θ. To identify the
scale, the measures of each facet sum to 0 and the thresholds of each set sum to 0; positive measures are harder items
and more severe raters, and μ carries the overall level.

The model is estimated by the EM algorithm of `calibrate`, over partial credit items with a = 1, with Newton's method
for all measures, thresholds, μ and σ together, and the extrapolation SQUAREM on the iterations. Standard errors come from the Hessian of the marginal log-likelihood,
and, for the element each facet's constraint leaves out and for the last threshold, from the delta method.

| Option           | Default                 | Meaning                                                                                                 |
| ---------------- | ----------------------- | ------------------------------------------------------------------------------------------------------- |
| `person`         | `"person"`              | The facet that names the persons                                                                        |
| `facets`         | required                | The other facets, such as `["item", "rater"]`: at least one, none repeated; facets beyond two add terms |
| `thresholdFacet` | one scale shared by all | A facet whose elements each get their own thresholds, as the partial credit model: one of `facets`      |
| `interactions`   | none                    | Pairs of `facets` with a bias for each cell, such as `[["item", "rater"]]`; none repeated               |
| `quadrature`     | 61 rectangular points   | The rule over theta, as `calibrate`                                                                     |
| `standardErrors` | `"hessian"`             | `"none"` leaves out the standard errors, and with them the separation, reliability and homogeneity test |
| `tolerance`      | `1e-4`                  | EM stops when no parameter changes by more than this                                                    |
| `maxIterations`  | `500`                   | The most EM iterations                                                                                  |

### Result

| Field                         | Meaning                                                                                                        |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `facets`                      | A `FacetSummary` for each facet in `facets`                                                                    |
| `thresholds`                  | The thresholds and their standard errors for the shared scale, or for each element of `thresholdFacet`         |
| `interactions`                | The bias of every cell of every interaction; absent without `interactions`                                     |
| `population`                  | The mean and the standard deviation of θ                                                                       |
| `logLikelihood`, `aic`, `bic` | The marginal log-likelihood; -2 ln L + 2p; -2 ln L + p ln n, p the parameters, μ and σ included, n the persons |
| `converged`, `iterations`     | Whether EM reached `tolerance`, and the iterations run                                                         |

A `FacetSummary` holds the facet's `elements`, in order of first appearance, and the statistics of the facet:

| Field                        | Meaning                                                                                                                  |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `elements[].measure`         | The element's measure in logits                                                                                          |
| `elements[].standardError`   | Its standard error: 0 for the one element of a facet, `null` without standard errors                                     |
| `elements[].count`           | Its number of scores                                                                                                     |
| `elements[].observedAverage` | The mean of its scores                                                                                                   |
| `elements[].fairAverage`     | Its expected score at θ = μ with the other facets at their means: the average the element would give to a typical person |
| `separation`                 | The standard deviation of the measures net of error over the root mean square error                                      |
| `reliability`                | The share of the variance of the measures that is not error                                                              |
| `homogeneity`                | The chi-square that all elements have the same measure, with its degrees of freedom and significance                     |

With V the variance of the measures (dividing by the number of elements) and M the mean of the squared standard errors,
the true variance is max(V - M, 0), the separation is its square root over √M, and the reliability is the true variance
over V (Wright and Masters, 1982). The homogeneity chi-square is Σ w d² - (Σ w d)² / Σ w with d the measures and
w = 1 / SE², on as many degrees of freedom as elements less 1, as FACETS' "fixed (all same)" test. All three are `null`
for a facet of one element and without standard errors.

A threshold set has K - 1 thresholds for categories 0 to K - 1. A threshold the constraint fixes, as the one threshold of
a two-category scale, has the standard error 0.

### Persons, fair scores and fit

| Function                                       | Returns                                                                        |
| ---------------------------------------------- | ------------------------------------------------------------------------------ |
| `manyFacetEstimates(scores, model, estimator)` | A `PersonEstimate` for each person in the scores, in order of first appearance |
| `expectedRating(model, theta, levels)`         | The expected rating of a person at θ on the elements in `levels`               |
| `manyFacetFit(scores, model, persons)`         | A `FacetFit` for each element of each facet                                    |

`manyFacetEstimates` turns each of a person's scores into a response to a partial credit item with a = 1, whose steps are
the sum of the measures of the score's elements plus the thresholds of its set, and passes them to an
[ability estimator](irt.md): the default is the posterior mean under the normal population of the model; any other,
such as `maximumLikelihoodEstimate`, works. The estimates account for the raters' and items' measures, so a person
rated by severe raters is not penalized. Scores need elements the model has, in categories its thresholds give.

`expectedRating` is the fair score, the severity-adjusted score that a platform reports for an essay: the rating a
person of ability θ would get from raters and items at the given elements. A facet not in `levels` is at its mean, a
measure of 0; with a `thresholdFacet` not in `levels`, the expected ratings under each of its elements' thresholds are
averaged.

`manyFacetFit` gives each element the infit and outfit mean squares and their standardized forms of the
[residual fit](fit.md#infit-and-outfit), computed over the scores of the element at the persons' estimates. Mean squares near 1
fit; above 1.5, a rater who departs from the others unpredictably; below 0.5, one whose scores are too predictable
(Linacre, 2002). Scores of persons without an estimate are skipped.

### Bias analysis

A rater can be harsher on one item than the rater's measure and the item's measure predict. `interactions` adds the
pair of facets to the model, and each cell, a rater scoring an item, gets a bias: the logits that the pair adds to the
sum of the two measures. Positive is harsher, or harder, than the measures predict. The example under Many-facet Rasch model
ends with one.

```mermaid
flowchart LR
  S["Scores"] --> C["Cells: the pairs of elements scored together"]
  C --> K["Each element's biases sum to 0 in both facets"]
  K --> L["One marginal likelihood: measures, thresholds and biases"]
  L --> B["Bias, standard error and z for each cell"]
  B --> F["Cells with a large |z| are the biased pairs"]
```

| Field of a cell | Meaning                                                                                          |
| --------------- | ------------------------------------------------------------------------------------------------ |
| `facets`        | The two facets, in the order of the option                                                       |
| `levels`        | The ids of the cell's elements in them                                                           |
| `bias`          | The bias in logits                                                                               |
| `standardError` | From the same Hessian as the measures, by the delta method; `null` with `standardErrors: "none"` |
| `z`             | The bias over its standard error; `null` where the standard error is `null` or 0                 |

The biases of the cells of an element sum to 0 in either facet, so that the measures keep their meaning; the cells of one
interaction have (n - 1) (m - 1) free biases for n by m elements. Cells that no score fills have no bias, and a table
with empty cells has as many free biases as its cells have cycles; an interaction whose cells leave none, such as raters
who each score a different item, throws a `DataError`. A pair may not repeat, in either order.

The package flags no cell: compare `z` with a cut-off of your choice. With many cells, some exceed 2 by chance, 5% of them
when no bias exists. The biases enter the same likelihood as the measures and the thresholds, so `aic` and `bic` count
them, and a model with the interaction can be compared with one without.

The biases enter `manyFacetEstimates` and `manyFacetFit`, which score each rating at its cell's location. They do not
enter `expectedRating` or the fair averages of a facet, which are at a facet's mean: a facet that is not in `levels` has
no cell.

### Not included

- **Joint maximum likelihood (FACETS).** Outside the sources: its measures are biased for short tests, and marginal
  maximum likelihood is consistent. Measures differ from FACETS' by its bias.
- **Rater drift over time.** No reference: it needs its own terms in the model, and no package that runs it was found.
- **The hierarchical rater model.** No reference: it needs its own terms in the model, and no package that runs it was
  found.
- **A person facet other than one.** No reference: the persons are the only random facet.

### Differences from FACETS

| Topic          | This package                                                         | FACETS                                                                  |
| -------------- | -------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Estimation     | Marginal maximum likelihood, theta normal with estimated mean and SD | Joint maximum likelihood, persons as parameters: biased for short tests |
| Identification | The measures of each facet and the thresholds of each set sum to 0   | Facets centered or anchored as specified; persons free                  |
| Separation     | The variance of the measures divided by the number of elements       | Reports it for the population (N) and for the sample (N - 1)            |
| Bias analysis  | The biases enter the likelihood with the measures                    | A second step with the other measures fixed: the biases differ          |

### Differences from TAM

| Topic                 | This package                                                                                         | TAM's `tam.mml.mfr`                                                                     |
| --------------------- | ---------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Quadrature            | Nodes μ + σ z on the 61 points of -6 to 6 standard deviations                                        | Fixed nodes in theta units, weighted by the normal density at the estimated mean and SD |
| Convergence           | EM with Newton's method for all parameters and SQUAREM: 27 to 51 iterations to 1e-11 on the fixtures | EM that stops short of the maximum within 3000 iterations on the fixtures               |
| Constraints           | Each facet and each set of thresholds sums to 0                                                      | The same, with `constraint = "items"`                                                   |
| Thresholds by a facet | Any facet                                                                                            | `rater:step` fails with more than three raters in TAM 4.3-25                            |
| Interaction           | `interactions`: biases within 4e-4 of TAM's on the fixture; standard errors from the Hessian         | `item:rater`: standard errors of its own parameters, 0.6 to 2.4 times the Hessian's     |

## References

- Cohen, J. (1960). A coefficient of agreement for nominal scales. Educational and Psychological Measurement, 20,
  37-46.
- Cohen, J. (1968). Weighted kappa. Psychological Bulletin, 70, 213-220.
- Eckes, T. (2023). Introduction to many-facet Rasch measurement: Analyzing and evaluating rater-mediated assessments
  (3rd ed.). Peter Lang.
- Fleiss, J. L. (1971). Measuring nominal scale agreement among many raters. Psychological Bulletin, 76, 378-382.
- Linacre, J. M. (1989). Many-facet Rasch measurement. MESA Press.
- Linacre, J. M. (2002). What do infit and outfit, mean-square and standardized mean? Rasch Measurement Transactions,
  16(2), 878.
- McGraw, K. O., & Wong, S. P. (1996). Forming inferences about some intraclass correlation coefficients.
  Psychological Methods, 1, 30-46.
- Myford, C. M., & Wolfe, E. W. (2003). Detecting and measuring rater effects using many-facet Rasch measurement: Part I.
  Journal of Applied Measurement, 4, 386-422.
- Myford, C. M., & Wolfe, E. W. (2004). Detecting and measuring rater effects using many-facet Rasch measurement: Part II.
  Journal of Applied Measurement, 5, 189-227.
- Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: Uses in assessing rater reliability. Psychological
  Bulletin, 86, 420-428.
- Wright, B. D., & Masters, G. N. (1982). Rating scale analysis. MESA Press.
