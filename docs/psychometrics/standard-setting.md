# Standard setting

`@measurekit/psychometrics/standard-setting` turns a panel's judgments into cut scores by the modified Angoff,
Bookmark, Hofstee and Beuk methods, the judgments of persons or of their work into cut scores by contrasting groups
and Body of Work, and examiners' global ratings at the stations of a performance test, such as an OSCE, into cut scores
by borderline regression and the borderline group. It reports how accurately and consistently cuts classify persons: on
theta by Rudner's method, and on total scores by Livingston and Lewis's. No R package among the references implements
the cut-score methods; each is validated against its definition written out in base R, with mirt's probabilities and
expected test scores for the Bookmark method, R's `glm` for contrasting groups and `lm` for borderline regression. The
classification indices are validated against cacIRT and betafunctions ([validation](validation.md)).

```mermaid
flowchart LR
  A["Ratings of borderline persons"] --> AC["angoffCutScore"]
  B["Items ordered by difficulty"] --> BC["orderedItemBooklet, bookmarkCutScore"]
  H["Bounds on the cut score and the failure rate"] --> HC["hofsteeCutScore"]
  K["Means and SDs of cut scores and pass rates"] --> KC["beukCutScore"]
  G["Scores and the categories judged"] --> GC["contrastingGroupsCutScores"]
  O["Checklist marks and global ratings at stations"] --> OC["borderlineRegressionCutScores, borderlineGroupCutScores"]
  AC --> C["Cut score on the total score"]
  BC --> C
  HC --> C
  KC --> C
  GC --> C
  OC --> C
  E["Estimates of theta and cuts on theta"] --> Q["classificationAccuracy"]
  T["Total scores, their reliability and cuts on them"] --> L["livingstonLewisClassification"]
```

Cut scores on the total score scale are not rounded: a person meets the standard with a total score at or above it.

## Modified Angoff

Each panelist rates each item with the expected score of a borderline person, the person just at the standard: for a
dichotomous item, the probability of a correct response (Angoff, 1971). `angoffCutScore(ratings)` takes the ratings
by panelist, then item.

```ts
import { angoffCutScore } from "@measurekit/psychometrics/standard-setting";

angoffCutScore([
  [0.6, 0.8, 2],
  [0.7, 0.9, 1.9],
  [0.5, 0.7, 2.1],
]); // { cutScore: 3.4, panelistCutScores: [3.4, 3.5, 3.3], sd: 0.1, standardError: 0.058 }
```

| Field               | Meaning                                                                |
| ------------------- | ---------------------------------------------------------------------- |
| `cutScore`          | The mean of the panelists' cut scores                                  |
| `panelistCutScores` | Each panelist's cut score: the sum of the panelist's ratings           |
| `sd`                | The standard deviation of the panelists' cut scores, denominator n - 1 |
| `standardError`     | sd / √n for n panelists, the standard error of the cut score           |

`sd` and `standardError` are `null` with one panelist. Each round of a standard-setting meeting is one call.

## Bookmark

The Bookmark method (Mitzel, Lewis, Patz and Green, 2001) orders the score points of a test's items by difficulty in
an ordered item booklet. Each panelist places a bookmark after the last page a borderline person masters, that is,
answers with a probability of at least the response probability (RP).

```ts
import { bookmarkCutScore, orderedItemBooklet } from "@measurekit/psychometrics/standard-setting";
import { gradedResponse, threeParameterLogistic } from "@measurekit/psychometrics/irt";

const items = [
  { id: "q1", model: threeParameterLogistic, parameters: { a: 1.2, b: -0.8, c: 0.2, d: 1 } },
  { id: "q2", model: threeParameterLogistic, parameters: { a: 0.9, b: 0.3, c: 0.25, d: 1 } },
  { id: "essay", model: gradedResponse, parameters: { a: 1.1, b: [-1.2, 0.4, 1.5] } },
];
orderedItemBooklet(items); // [{ id: "essay", category: 1, location }, { id: "q1", category: 1, location }, ...]
bookmarkCutScore(items, 3); // { theta, cutScore }: the location of page 3 and the expected total score there
```

- **Pages.** An item with K categories has a page for each score point k from 1 to K - 1, located at the theta where
  P(X ≥ k | θ) = RP: b + ln((RP - c) / (d - RP)) / a for a logistic item. Pages are in ascending order of location,
  ties in item order.
- **Cut scores.** `bookmarkCutScore(items, page)` takes the number of the last page mastered. The theta cut is that
  page's location; the raw cut score is the expected total score there, Σ E(X_i | θ) over the items, the test
  characteristic curve. A panel that places the bookmark on the first page not mastered gives that page's number
  less 1. Each panelist's bookmark is one call; the panel's cut is often the median of theirs.
- **Response probability.** `responseProbability` defaults to 0.67. For a Rasch item, RP 2/3 is where the information
  of a correct response, P(θ) I(θ), peaks; for a 3PL item that point is RP (2 + c) / 3 (Huynh, 1998, 2006).
- **Items.** Items of the built-in models whose expected score increases with theta: not the NRM, and a > 0. A
  logistic item needs c < RP < d, or its probability never equals RP.

## Hofstee

The Hofstee (1983) method asks the panel for the lowest and highest acceptable cut scores, k_min and k_max, and the
lowest and highest acceptable failure rates, f_min and f_max, usually the means of the panelists' answers. The cut
score is where the line from (k_min, f_max) to (k_max, f_min) crosses the observed failure rate F(c), the share of
persons with a total score below c.

```ts
import { hofsteeCutScore } from "@measurekit/psychometrics/standard-setting";

// The number of persons at each total score from 0 to 10.
const distribution = [1, 2, 4, 7, 10, 12, 11, 8, 5, 3, 1];
hofsteeCutScore(distribution, {
  minimumCutScore: 4,
  maximumCutScore: 7,
  minimumFailureRate: 0.1,
  maximumFailureRate: 0.4,
}); // { cutScore, failureRate }
```

F is taken at each score from 0 to the highest and is linear between them, so the cut score need not be a score. The
distribution is the number of persons at each total score, as in [observed-score equating](equating.md). When the line
does not cross F between k_min and k_max, the panel's bounds do not fit the data, and the function throws `DataError`.

## Beuk

The Beuk (1984) method asks each panelist for the lowest passing score and the share of persons expected to pass. The
means k̄ and p̄ give a point, and the standard deviations s_k and s_p a slope: the line through (k̄, p̄) of slope s_p / s_k
in pass rates crosses the observed pass rate, so the answer the panel agrees on more moves less. In failure rates, the
share of persons below c, the line falls and the observed failure rate rises, and they cross once.

```ts
import { beukCutScore } from "@measurekit/psychometrics/standard-setting";

// The number of persons at each total score from 0 to 10.
const distribution = [1, 2, 4, 7, 10, 12, 11, 8, 5, 3, 1];
beukCutScore(distribution, { cutScore: 6, passRate: 0.4, cutScoreSd: 1, passRateSd: 0.1 });
// { cutScore: 6.138, failureRate: 0.586 }, after rounding
```

| Judgment     | Meaning                                                                       |
| ------------ | ----------------------------------------------------------------------------- |
| `cutScore`   | The mean of the panelists' lowest passing scores, from 0 to the highest score |
| `passRate`   | The mean of the panelists' expected pass rates, from 0 to 1                   |
| `cutScoreSd` | The standard deviation of the lowest passing scores, positive                 |
| `passRateSd` | The standard deviation of the expected pass rates, positive                   |

The observed failure rate is taken at each score and is linear between them, as in the Hofstee method, with which it
shares its crossing. When the line does not cross it between the scores 0 and the highest, the function throws
`DataError`.

## Contrasting groups and Body of Work

`contrastingGroupsCutScores(scores, categories, options)` takes each person's score and the category a panel placed the
person in, and returns a cut score for each boundary between adjacent categories: the score where a logistic regression
of "at or above the boundary" on the score reaches `probability`, (logit(probability) - β₀) / β₁, as R's `glm(family =
binomial)`. Two categories, masters and non-masters, give the contrasting-groups method (Livingston and Zieky, 1982):

```ts
import { contrastingGroupsCutScores } from "@measurekit/psychometrics/standard-setting";

const scores = [12, 15, 17, 18, 20, 21, 21, 23, 24, 26, 27, 29, 31, 33];
const masters = [0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1]; // the panel's judgment of each person
contrastingGroupsCutScores(scores, masters); // { cutScores: [21.154], converged: true }, after rounding
contrastingGroupsCutScores(scores, masters, { probability: 0.67 }); // { cutScores: [22.816], ... }
```

More categories give the Body of Work method (Kingston, Kahl, Sweeney and Bay, 2001): a panel sorts collections of work
into the performance levels, each collection with its score, and each boundary gets its cut:

```ts
import { contrastingGroupsCutScores } from "@measurekit/psychometrics/standard-setting";

const work = [8, 12, 15, 19, 22, 24, 27, 29, 31, 34, 36, 38, 41, 43, 46, 49, 52, 55];
const levels = [0, 0, 0, 1, 0, 1, 1, 1, 2, 1, 2, 2, 2, 3, 2, 3, 3, 3]; // below basic to advanced
contrastingGroupsCutScores(work, levels).cutScores; // [20.031, 32.5, 44.76], after rounding
```

| Option        | Default | Meaning                                                          |
| ------------- | ------- | ---------------------------------------------------------------- |
| `probability` | `0.5`   | The probability of the higher category at a cut, between 0 and 1 |

Categories are integers from 0, and every category from 0 to the highest needs persons, so that each boundary has
persons on both sides. Where the scores of the two sides of a boundary do not overlap, the likelihood has no maximum:
that cut is `null` and `converged` is `false`. A cut need not be a score.

## Borderline regression and the borderline group

At each station of a performance test, such as an OSCE, an examiner gives each person a checklist mark and a global
rating of the whole performance, such as 1 fail, 2 borderline, 3 pass and 4 superior. Both methods set each station's
cut from those two after the test, and the test's cut is the sum of the stations' cuts: a person meets the standard with
a sum of marks at or above it. A cut on the mean of the stations' marks is that sum divided by the stations, and
classifies alike.

```mermaid
flowchart LR
  M["Marks and ratings, by person, then station"] --> R["Per station: lm of marks on ratings"]
  R --> RC["Mark on the line at the borderline grade,<br/>its standard error, R², slope"]
  M --> G["Per station: persons at a borderline grade"]
  G --> GC["Their mean or median mark"]
  RC --> T["Test's cut: the sum of the stations' cuts"]
  GC --> T
```

```ts
import { borderlineGroupCutScores, borderlineRegressionCutScores } from "@measurekit/psychometrics/standard-setting";

// Ten persons at three stations: each one's checklist marks and global ratings, 1 fail to 4 superior.
const marks = [
  [8, 11, 6],
  [12, 14, 9],
  [15, 13, 12],
  [11, 9, 10],
  [18, 17, 14],
  [14, 16, 11],
  [6, 8, 5],
  [16, 15, 13],
  [10, 12, 8],
  [13, 10, 12],
];
const ratings = [
  [1, 2, 1],
  [2, 3, 2],
  [3, 3, 3],
  [2, 2, 2],
  [4, 4, 4],
  [3, 3, 2],
  [1, 1, 1],
  [3, 4, 3],
  [2, 2, 2],
  [3, 2, 3],
];
const regression = borderlineRegressionCutScores(marks, ratings, 2);
regression.stations[0]; // { cutScore: 10.833, standardError: 0.369, rSquared: 0.925, intercept: 3.5, slope: 3.667 }
regression.cutScore; // 30.730, with standardError 0.686
borderlineGroupCutScores(marks, ratings, 2).cutScore; // 31: the means 11, 10.5 and 9.5
```

**Borderline regression** (Kramer et al., 2003) fits each station's least-squares line of the marks on the ratings, as
R's `lm`, and reads the mark at the borderline grade x₀. Every person counts, so no station needs many borderline
persons. Its fields are the quality metrics of Pell et al. (2010):

| Field           | Meaning                                                                                                                     |
| --------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `cutScore`      | ȳ + b (x₀ - x̄), the line's mark at the borderline grade                                                                     |
| `standardError` | s √(1/n + (x₀ - x̄)² / Sxx), s² = RSS / (n - 2), as `predict(se.fit = TRUE)` (Wood et al., 2006); `null` below three persons |
| `rSquared`      | 1 - RSS / TSS, how closely marks follow ratings; Pell et al. take above 0.5 as a reasonable relationship                    |
| `slope`         | The inter-grade discrimination b, the rise in marks per grade; Pell et al. recommend about a tenth of the station's maximum |
| `intercept`     | The line's mark at a rating of 0                                                                                            |

The test's `standardError` sums the covariances of the stations' cuts. A person's residuals at two stations correlate
(Homer et al., 2016), so the root of the sum of the squared standard errors, which assumes they do not, is off. Each cut
is Σᵢ wᵢ yᵢ with wᵢ = 1/n + (x₀ - x̄)(xᵢ - x̄) / Sxx, two cuts have the covariance σ_st Σᵢ w_si w_ti, and σ_st is
estimated by the residuals' cross products over n - 2, as for equations fitted one by one (Zellner, 1962). Like each
station's, it holds the ratings fixed.

**The borderline group** takes the marks of the persons rated borderline at the station: their mean, with its standard
error sd / √n, or with `statistic: "median"` their median, as Livingston and Zieky's method does. Several grades, such
as `[2, 3]` for a borderline fail and a borderline pass, give the modified borderline group. Each station's group
holds different persons, and a group of a few persons gives an unstable cut (McKinley and Norcini, 2014).

| Option      | Default  | Meaning                                     |
| ----------- | -------- | ------------------------------------------- |
| `statistic` | `"mean"` | `"mean"` or `"median"` of the group's marks |

- **Undefined cuts.** A station whose ratings all agree has no line, and one without a borderline person no group: its
  cut and standard error are `null`, and so is the test's cut. A station whose marks all agree has a cut and no
  `rSquared`.
- **Persons.** Every person needs a mark and a rating at every station; leave out those who missed one.
- **Limit.** Borderline regression sums over the persons for each pair of stations: persons × stations² is at most
  2^28. At the limit, 26,843 persons at 100 stations take 0.3 s and 671,088 persons at 20 stations 1.4 s.
- **A margin of error.** Programs that guard against false passes raise the cut by a multiple of the standard error of
  measurement of the total, `standardErrorOfMeasurement` of [classical test theory](ctt.md) (Homer et al., 2016).

## Classification accuracy

`classificationAccuracy(estimates, cuts)` takes ability estimates, `{ theta, standardError }` each, and cuts on theta in
increasing order, and returns Rudner's (2001, 2005) indices, as cacIRT's `class.Rud`. Each person's theta is taken as
normal around the estimate with its standard error: P_k = Φ((c_(k+1) - θ̂) / SE) - Φ((c_k - θ̂) / SE) for each
category k between cuts.

| Field         | Meaning                                                                                              |
| ------------- | ---------------------------------------------------------------------------------------------------- |
| `persons`     | Per person: `category`, the number of cuts at or below θ̂, and `probabilities`, P_k for each category |
| `accuracy`    | The mean of the probability of each person's estimated category                                      |
| `consistency` | The mean of Σ_k P_k²: the probability that two independent measurements agree                        |

With one cut, `probabilities` are a person's chances of failing and passing; with several, the indices are cacIRT's
simultaneous ones. The normal approximation suits ML, WLE and EAP estimates with their standard errors.

```ts
import { classificationAccuracy } from "@measurekit/psychometrics/standard-setting";

const result = classificationAccuracy(
  [
    { theta: -0.4, standardError: 0.35 },
    { theta: 0.6, standardError: 0.3 },
    { theta: 1.4, standardError: 0.4 },
  ],
  [0.5],
);
result.persons[1]; // { category: 1, probabilities: [0.37, 0.63] }: 0.63 = Φ((0.6 - 0.5) / 0.3)
```

## Livingston and Lewis

`livingstonLewisClassification(scores, cuts, options)` takes total scores of any items, their reliability and cuts on
the score scale, and returns Livingston and Lewis's (1995) accuracy and consistency, as betafunctions' `LL.CA` for one
cut and `LL.CA.MC` for several. Testing programs report it with BB-CLASS, which implements it; Rudner's indices above
suit cuts on theta with estimates and their standard errors, these suit cuts on a total score, which may sum
polytomous items or weighted parts.

```mermaid
flowchart LR
  S["Scores, reliability r"] --> N["Effective length N = ((μ - min)(max - μ) - r σ²) / (σ² (1 - r))"]
  N --> M["True-score moments by Lord's binomial relation"]
  M --> B{"Four-parameter beta permissible?"}
  B -->|"yes"| F["Beta on [lower, upper]"]
  B -->|"no"| T["Two-parameter beta on [0, 1]"]
  F --> I["Integrals over the beta of binomial(round(N), p) categories"]
  T --> I
  I --> A["Accuracy and consistency, per cut and over all"]
```

```ts
import { cronbachAlpha } from "@measurekit/psychometrics/ctt";
import { buildResponseMatrix } from "@measurekit/psychometrics/data";
import { livingstonLewisClassification } from "@measurekit/psychometrics/standard-setting";

// Eight persons' answers to eight items, and their total scores.
const answers = [
  [1, 1, 1, 0, 1, 1, 0, 1],
  [0, 1, 0, 0, 0, 1, 0, 0],
  [1, 1, 1, 1, 1, 1, 1, 0],
  [1, 0, 1, 0, 0, 1, 0, 0],
  [0, 0, 0, 0, 1, 0, 0, 0],
  [1, 1, 1, 1, 0, 1, 1, 1],
  [1, 1, 0, 1, 1, 0, 1, 0],
  [0, 1, 0, 0, 1, 1, 0, 0],
];
const reliability = cronbachAlpha(buildResponseMatrix(answers)); // 0.716
const scores = answers.map((row) => row.reduce((sum, x) => sum + x, 0));
const result = livingstonLewisClassification(scores, [4], { reliability, maximum: 8 });
result.accuracy; // 0.811
result.consistency; // 0.757
result.trueScore; // { alpha: 2.226, beta: 1.964, lower: 0, upper: 1, effectiveLength: 7.959 }: the fail-safe beta
```

| Option        | Default  | Meaning                                            |
| ------------- | -------- | -------------------------------------------------- |
| `reliability` | required | The reliability of the scores, above 0 and below 1 |
| `maximum`     | required | The highest possible score                         |
| `minimum`     | `0`      | The lowest possible score                          |

| Field         | Meaning                                                                                                  |
| ------------- | -------------------------------------------------------------------------------------------------------- |
| `cuts`        | Per cut: `cut`, and the `accuracy` and `consistency` of the classification below it and at or above it   |
| `accuracy`    | Over all categories: the probability that the observed category is the true one; with one cut, the cut's |
| `consistency` | Over all categories: the probability that two independent administrations give one category              |
| `trueScore`   | The beta's shapes `alpha`, `beta`, its bounds `lower`, `upper`, and the unrounded `effectiveLength`      |

| Step        | Rule                                                                                                                                                                                       |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Proportions | Each score as (x - min) / (max - min)                                                                                                                                                      |
| Length      | N from the mean, the variance (denominator n - 1) and the reliability, betafunctions' `ETL`; the number of items as reliable                                                               |
| Moments     | x′ = N times the proportion; m₁ = mean(x′) / N and m_i = mean(x′₍ᵢ₎) / N₍ᵢ₎ for i = 2 to 4, descending factorials                                                                          |
| True scores | The four-parameter beta of their mean, variance, skewness and kurtosis (Hanson, 1991)                                                                                                      |
| Fail-safe   | Where that beta has a bound outside [0, 1], a shape at most 0 or a value not finite, the two-parameter beta on [0, 1] of the mean and variance; `lower` 0 and `upper` 1 show it            |
| Rounding    | N to the nearest integer, halves to even, as R's `round`; only the binomial uses it                                                                                                        |
| Cuts        | A cut c is the true cut at p = (c - min) / (max - min) and the observed cut at ⌈N p⌉: a score at or above it is above; a cut within 1e-9 of a whole number of trials counts as that number |
| Indices     | Integrals over the beta of the binomial probability of each observed category, by adaptive Gauss-Kronrod quadrature to a relative 1e-12                                                    |

A shape below 1 makes the beta's density infinite at that end, as for two separate groups of persons; the integral there
is taken in t^α or (1 - t)^β, an exact change of variables, so that it converges. The time grows with the rounded
effective length, which may be at most 10,000: 2 ms for 3,000 scores of 40 items, 0.2 s at 4,639 trials.

| Error                                                                               | Meaning and remedy                                                                                                 |
| ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `Livingston and Lewis need an effective test length from 1 to 10000 after rounding` | The reliability does not fit the scores' variance: check it, and the range of the scores                           |
| `Livingston and Lewis find no true-score beta for these scores`                     | The true-score variance is at most 0, or not below m(1 - m): the reliability is too low or too high for the scores |
| `Livingston and Lewis need scores that vary`                                        | The variance is 0, so there is no effective length                                                                 |

Accuracy need not rise with the reliability: the observed cut ⌈N p⌉ moves with N, and the beta with the moments, so a
step up in the reliability can lower it by a few ten-thousandths, in betafunctions as here.

## Recipes

Two methods are one line over the functions above.

**Nedelsky.** Each panelist strikes the options of each multiple-choice item that a borderline person would rule out;
the item's rating is 1 over the options left, and the cut is the Angoff cut score of those ratings (Nedelsky, 1954):

```ts
import { angoffCutScore } from "@measurekit/psychometrics/standard-setting";

// The options left on each of four items, by two panelists.
const optionsLeft = [
  [2, 3, 1, 2],
  [2, 2, 1, 3],
];
angoffCutScore(optionsLeft.map((row) => row.map((options) => 1 / options))).cutScore; // 2.333
```

**Ebel.** The panel sorts the items into cells of relevance by difficulty and judges the share of each cell's items a
borderline person answers; the cut is the sum of the items times the share over the cells, the Angoff cut score of a
panel of one that rates each item with its cell's share (Ebel, 1972):

```ts
import { angoffCutScore } from "@measurekit/psychometrics/standard-setting";

const cells = [
  { items: 6, proportion: 0.9 },
  { items: 9, proportion: 0.75 },
  { items: 4, proportion: 0.55 },
];
const ratings = cells.flatMap(({ items, proportion }) => new Array<number>(items).fill(proportion));
angoffCutScore([ratings]).cutScore; // 14.35
```

## Not included

- **Nedelsky.** No fit: `angoffCutScore` of the reciprocals of the options left, a recipe.
- **Ebel.** No fit: a sum of items times proportions over the cells, a recipe.
- **A standard error of the borderline group's cut over stations.** Outside the sources: the stations' groups hold
  different persons, and no source gives the error of their sum; a bootstrap over the persons gives one.
- **A standard error of the borderline group's median.** Outside the sources: no source gives one; a bootstrap over the
  group gives one.
- **The bootstrap standard errors of borderline regression (Homer et al., 2016).** No fit: a resampling of the persons
  that calls `borderlineRegressionCutScores`, which also counts the variation of the ratings.
- **The objective borderline method (Shulruf et al., 2013).** Outside the sources: a recent variant whose assumption,
  that higher grades always bring higher marks, Homer, Pell and Fuller (2017) question.
- **Direct consensus.** No fit: `angoffCutScore` takes the sums of clusters.
- **Item-descriptor matching.** No fit: a panel procedure with no statistic beyond the Bookmark's.
- **The analytic judgment and other holistic methods.** Outside the sources: variants of Body of Work.
- **Vertically moderated standards.** No fit: a policy across grades, not a statistic.
- **Standards for alternate assessments.** No fit: a policy, not a statistic.
- **Lee's IRT classification indices.** Outside the sources: no technical report found uses them; programs use
  Livingston-Lewis and Rudner's, both built. cacIRT and irtQ have them.
- **The compound multinomial method (Lee, Brennan and Wan, 2009).** Outside the sources: a research extension of
  Livingston-Lewis for mixed formats.
- **Nonparametric classification (Lathrop and Cheng).** Outside the sources: a research method beside Rudner's and
  Livingston-Lewis.

## Errors

Invalid input throws `DataError` naming the value found, also input of another type, as JavaScript or JSON passes it.

| Function                                 | Error                                                                                                              | What to do                                                                                        |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| `angoffCutScore`                         | No panelists or items, panelists who rate different numbers of items, or a rating below 0                          | Give each panelist a rating of every item, from 0 to the item's highest score                     |
| `orderedItemBooklet`, `bookmarkCutScore` | No items, ids that are not strings or repeat, an NRM item or a slope a ≤ 0                                         | Give each item its own id, and models whose expected score rises with theta                       |
| `orderedItemBooklet`, `bookmarkCutScore` | `responseProbability` not between 0 and 1, or a logistic item with c ≥ RP or d ≤ RP                                | Choose a response probability that every item's curve passes, such as 0.67                        |
| `bookmarkCutScore`                       | A `page` that is not a whole number from 1 to the pages of the booklet                                             | Count the pages of `orderedItemBooklet` and place the bookmark after one of them                  |
| `hofsteeCutScore`                        | Bounds not ordered, 0 ≤ k_min < k_max ≤ the highest score and 0 ≤ f_min < f_max ≤ 1                                | Give the panel's bounds in order, on the scale of the distribution                                |
| `hofsteeCutScore`, `beukCutScore`        | A line that does not cross the observed failure rates                                                              | Widen the bounds, or check the panel's judgments against the distribution                         |
| `beukCutScore`                           | A mean cut score outside the scores, a pass rate outside [0, 1], or a standard deviation not above 0               | Give the means and the standard deviations of the panelists' answers                              |
| `hofsteeCutScore`, `beukCutScore`        | A distribution of fewer than two scores, a negative count, or no persons                                           | Give the persons at each score from 0 to the highest                                              |
| `contrastingGroupsCutScores`             | Scores and categories of different lengths, a category that is not an integer from 0, or fewer than two categories | Give each person a score and a category of 0 up                                                   |
| `contrastingGroupsCutScores`             | A category from 0 to the highest that has no person                                                                | Number the categories that persons hold from 0 without gaps                                       |
| Both borderline functions                | Marks or ratings that are not finite numbers, a row of another length than the first, or no person or station      | Give each person a mark and a rating at every station, by person, then station                    |
| Both borderline functions                | A borderline grade that is not a finite number, or an empty list of grades                                         | Give the grade on the scale of the ratings, such as 2                                             |
| `borderlineRegressionCutScores`          | More than 2^28 persons × stations²                                                                                 | Set the standard of fewer persons or stations at a time                                           |
| Both borderline functions                | A result that is not finite                                                                                        | Rescale the marks, the ratings or the grade, which are too large                                  |
| `borderlineGroupCutScores`               | A `statistic` other than `"mean"` and `"median"`                                                                   | Choose one of them                                                                                |
| `classificationAccuracy`                 | No estimates or cuts, a theta that is not finite, a standard error not above 0, or cuts that do not increase       | Give each estimate with its standard error, and the cuts in increasing order                      |
| `livingstonLewisClassification`          | Fewer than two scores, a score outside `minimum` to `maximum`, or scores that do not vary                          | Give the total scores of the persons and the range of the test                                    |
| `livingstonLewisClassification`          | A `reliability` not between 0 and 1, or cuts that are not finite and increasing                                    | Give a reliability such as `cronbachAlpha` of the items, and the cuts in increasing order         |
| `livingstonLewisClassification`          | An effective test length that rounds to below 1 or above 10,000 trials                                             | Check the reliability: a value near 1 or far too low for the scores' variance gives such a length |
| `livingstonLewisClassification`          | No permissible beta for the true scores                                                                            | Check the reliability against the scores' variance; use Rudner's indices on theta instead         |

## Differences from betafunctions

| Topic               | This package                                                                                    | betafunctions                                                                         |
| ------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Integrals           | Adaptive Gauss-Kronrod to a relative 1e-12, with a change of variables where a shape is below 1 | `integrate` at R's default tolerance, 1.2e-4: up to 7e-5 off where a shape is below 1 |
| One or several cuts | One function; each cut's indices and those over all categories                                  | `LL.CA` for one cut, `LL.CA.MC` for several                                           |
| Fail-safe           | Always, as `failsafe = TRUE`; `lower` 0 and `upper` 1 show it                                   | A warning, and an option                                                              |
| Specific rates      | Accuracy and consistency                                                                        | Also sensitivity, specificity, predictive values, Youden's J and kappa                |

## References

- Angoff, W. H. (1971). Scales, norms, and equivalent scores. In R. L. Thorndike (Ed.), Educational measurement (2nd
  ed., pp. 508-600). American Council on Education.
- Beuk, C. H. (1984). A method for reaching a compromise between absolute and relative standards in examinations.
  Journal of Educational Measurement, 21(2), 147-152.
- Cizek, G. J., & Bunch, M. B. (2007). Standard setting: A guide to establishing and evaluating performance standards
  on tests. Sage.
- Ebel, R. L. (1972). Essentials of educational measurement. Prentice-Hall.
- Hofstee, W. K. B. (1983). The case for compromise in educational selection and grading. In S. B. Anderson & J. S.
  Helmick (Eds.), On educational testing (pp. 109-127). Jossey-Bass.
- Homer, M., Pell, G., & Fuller, R. (2017). Problematizing the concept of the "borderline" group in performance
  assessments. Medical Teacher, 39(5), 469-475.
- Homer, M., Pell, G., Fuller, R., & Patterson, J. (2016). Quantifying error in OSCE standard setting for varying cohort
  sizes: A resampling approach to measuring assessment quality. Medical Teacher, 38(2), 181-188.
- Huynh, H. (1998). On score locations of binary and partial credit items and their applications to item mapping and
  criterion-referenced interpretation. Journal of Educational and Behavioral Statistics, 23, 35-56.
- Huynh, H. (2006). A clarification on the response probability criterion RP67 for standard settings based on
  bookmark and item mapping. Educational Measurement: Issues and Practice, 25(2), 19-20.
- Hanson, B. A. (1991). Method of moments estimates for the four-parameter beta compound binomial model and the
  calculation of classification consistency indexes. American College Testing.
- Kingston, N. M., Kahl, S. R., Sweeney, K. P., & Bay, L. (2001). Setting performance standards using the body of work
  method. In G. J. Cizek (Ed.), Setting performance standards (pp. 219-248). Lawrence Erlbaum.
- Kramer, A., Muijtjens, A., Jansen, K., Düsman, H., Tan, L., & van der Vleuten, C. (2003). Comparison of a rational
  and an empirical standard setting procedure for an OSCE. Medical Education, 37(2), 132-139.
- Livingston, S. A., & Lewis, C. (1995). Estimating the consistency and accuracy of classifications based on test
  scores. Journal of Educational Measurement, 32(2), 179-197.
- Livingston, S. A., & Zieky, M. J. (1982). Passing scores. Educational Testing Service.
- McKinley, D. W., & Norcini, J. J. (2014). How to set standards on performance-based examinations: AMEE Guide No. 85.
  Medical Teacher, 36(2), 97-110.
- Mitzel, H. C., Lewis, D. M., Patz, R. J., & Green, D. R. (2001). The Bookmark procedure: Psychological perspectives.
  In G. J. Cizek (Ed.), Setting performance standards (pp. 249-281). Lawrence Erlbaum.
- Nedelsky, L. (1954). Absolute grading standards for objective tests. Educational and Psychological Measurement, 14(1),
  3-19.
- Pell, G., Fuller, R., Homer, M., & Roberts, T. (2010). How to measure the quality of the OSCE: A review of metrics.
  AMEE guide no. 49. Medical Teacher, 32(10), 802-811.
- Rudner, L. M. (2005). Expected classification accuracy. Practical Assessment, Research & Evaluation, 10(13).
- Shulruf, B., Turner, R., Poole, P., & Wilkinson, T. (2013). The Objective Borderline method (OBM): A probability-based
  model for setting up an objective pass/fail cut-off score in medical programme assessments. Advances in Health
  Sciences Education, 18(2), 231-244.
- Wood, T. J., Humphrey-Murto, S. M., & Norman, G. R. (2006). Standard setting in a small scale OSCE: A comparison of
  the modified borderline-group method and the borderline regression method. Advances in Health Sciences Education,
  11(2), 115-122.
- Zellner, A. (1962). An efficient method of estimating seemingly unrelated regressions and tests for aggregation bias.
  Journal of the American Statistical Association, 57(298), 348-368.
