# Generalizability theory

`@measurekit/psychometrics/generalizability` separates the variance of scores into the effects of the objects of
measurement, such as persons, and of the facets of the measurement, such as items and raters, and then asks how a
procedure with other numbers of items and raters would measure. A **G study** estimates the variance components of a
balanced design by the analysis of variance; a **D study** combines them into error variances and the generalizability
and dependability coefficients of a procedure. Designs may cross and nest facets: p × i, p × r × i, (r:p) × i, i:p and
others. Components are validated against base R's `aov`, the expected mean squares solved as a linear system, and
lme4's REML estimates, and decision studies against the textbook formulas of each design
([validation](validation.md)).

## Designs

Scores are in long format: a `FacetedScore` has the element of each facet in `levels` and a `score`. The design names
the facet of the objects of measurement, `object`, and the others, `facets`. Facets are crossed unless `nested` says that
a facet is nested in others.

| Design        | `facets`            | `nested`                                  |
| ------------- | ------------------- | ----------------------------------------- |
| p × i         | `["item"]`          | none                                      |
| p × r × i     | `["rater", "item"]` | none                                      |
| i:p           | `["item"]`          | `{ item: ["person"] }`                    |
| (r:p) × i     | `["rater", "item"]` | `{ rater: ["person"] }`                   |
| r:(p × i)     | `["rater", "item"]` | `{ rater: ["person", "item"] }`           |
| i:r:p         | `["rater", "item"]` | `{ rater: ["person"], item: ["rater"] }`  |
| (i:p) × (r:p) | `["item", "rater"]` | `{ item: ["person"], rater: ["person"] }` |

The design must be balanced:

- every combination of elements has exactly one score;
- every facet has the same number of elements, at least 2, in each combination of the elements it is nested in;
- the object of measurement is not nested, and nesting is not circular;
- a design of k facets, the object included, has at least 2^k scores, and at most 20 facets are allowed;
- `levels` has a string element for the object and every facet; other entries are ignored.

An element of a nested facet is identified within the elements it is nested in. Raters may be labeled `r1` and `r2` for
every person, or each with a label of its own: the results are the same, and the first example below shows both.
Designs that are not balanced throw a `DataError` that names the first combination with more than one score or, failing
that, with none; incomplete rating designs are the case of the many-facet Rasch model (`rater` module).

## G study

`generalizabilityStudy(scores, design)` returns one `VarianceComponent` for each effect of the design, ordered by the
number of facets in the effect and then by the order of the facets: for p × i, the person, the item and the person by item
effect.

```ts
import { generalizabilityStudy } from "@measurekit/psychometrics/generalizability";

const persons = ["p1", "p2", "p3", "p4", "p5", "p6"];
const items = ["i1", "i2", "i3", "i4"];
const ratings = [
  [4, 5, 3, 4],
  [2, 3, 2, 2],
  [5, 5, 4, 5],
  [3, 4, 3, 2],
  [1, 2, 1, 2],
  [4, 4, 5, 4],
];
const scores = ratings.flatMap((row, p) =>
  row.map((score, i) => ({ levels: { person: persons[p] ?? "", item: items[i] ?? "" }, score })),
);
const study = generalizabilityStudy(scores, { facets: ["item"] });
study.sampleSizes; // { person: 6, item: 4 }
study.components.map(({ facets, degreesOfFreedom, meanSquare, variance }) => [
  facets.join(" x "),
  degreesOfFreedom,
  Math.round(meanSquare * 1000) / 1000,
  Math.round(variance * 1000) / 1000,
]);
// [["person", 5, 6.342, 1.506], ["item", 3, 0.819, 0.083], ["person x item", 15, 0.319, 0.319]], after rounding
```

| Field of a component | Meaning                                                                                            |
| -------------------- | -------------------------------------------------------------------------------------------------- |
| `facets`             | The effect's own facets: the rater and the item of the effect of ri:p                              |
| `nestedIn`           | The facets it is nested in, such as the person of r:p; none for a crossed effect                   |
| `degreesOfFreedom`   | The product of the sizes less 1 of its own facets and the sizes of those it is nested in           |
| `sumOfSquares`       | From the marginal means of the effect, with the scores centered                                    |
| `meanSquare`         | The sum of squares over the degrees of freedom                                                     |
| `variance`           | The analysis of variance estimate of the variance component; negative when the mean squares say so |

The highest-order effect, the one of all facets, is confounded with error: p × i has no residual of its own, and its last
component is σ²(pi, e).

### Estimation

The expected mean square of an effect is the sum, over the effects whose facets include its own, of their variance
components times the product of the sizes of the facets they lack (the Cornfield-Tukey rules for random effects). For p ×
i with n_p persons and n_i items:

| Effect | Expected mean square |
| ------ | -------------------- |
| p      | σ²(pi) + n_i σ²(p)   |
| i      | σ²(pi) + n_p σ²(i)   |
| pi     | σ²(pi)               |

The components are solved from the effect of the most facets down. Estimates are not constrained: a mean square below
that of the effects it is confounded with gives a negative component, which the G study reports as estimated and a D
study treats as 0.

### Nested designs

`nested` gives the facets each facet is nested in. The rater of the design (r:p) × i is nested in the person; each person
has their own raters, who each score every item.

```ts
import { generalizabilityStudy } from "@measurekit/psychometrics/generalizability";

// Persons by raters by items; person p has raters r1 and r2, who are not the raters of another person.
const byPerson = [
  [
    [4, 5, 3],
    [5, 5, 4],
  ],
  [
    [2, 3, 2],
    [3, 3, 2],
  ],
  [
    [5, 5, 4],
    [4, 5, 5],
  ],
  [
    [3, 4, 3],
    [2, 4, 3],
  ],
  [
    [1, 2, 1],
    [2, 2, 2],
  ],
];
const scores = byPerson.flatMap((raters, p) =>
  raters.flatMap((row, r) =>
    row.map((score, i) => ({ levels: { person: `p${p + 1}`, rater: `r${r + 1}`, item: `i${i + 1}` }, score })),
  ),
);
const design = { facets: ["rater", "item"], nested: { rater: ["person"] } };
const study = generalizabilityStudy(scores, design);
study.sampleSizes; // { person: 5, rater: 2, item: 3 }: raters per person
study.components.map(({ facets, nestedIn, degreesOfFreedom, variance }) => [
  facets.join(" x "),
  nestedIn.join(" x "),
  degreesOfFreedom,
  Math.round(variance * 1000) / 1000,
]);
// [["person", "", 4, 1.508], ["item", "", 2, 0.2], ["rater", "person", 5, 0.033], ["person x item", "", 8, 0],
//  ["rater x item", "person", 10, 0.233]], after rounding

// Labels of their own for each person's raters give the same study.
const labeled = scores.map(({ levels, score }) => ({
  levels: { ...levels, rater: `${levels.person}-${levels.rater}` },
  score,
}));
JSON.stringify(generalizabilityStudy(labeled, design)) === JSON.stringify(study); // true
```

The sample size of a nested facet is its number of elements within each element it is nested in.

## D study

`decisionStudy(study, options)` takes a study and returns the variances and coefficients of a measurement procedure that
averages over `sampleSizes` elements of each facet.

| Option        | Default           | Meaning                                                                                                               |
| ------------- | ----------------- | --------------------------------------------------------------------------------------------------------------------- |
| `sampleSizes` | the study's sizes | Positive integers, by facet name; for a nested facet, within each element it is nested in                             |
| `fixed`       | `[]`              | Facets fixed in the universe of generalization; a fixed facet's size is the number of conditions of its universe      |
| `nested`      | none              | Facets the decision study nests that the study crossed, such as `{ rater: ["person"] }`; added to the study's nesting |

Each effect of the decision study has a variance component, a divisor and a role:

- The **variance** of an effect is the study's component, or the sum of those the effect pools, with any below 0 counted
  as 0. Nesting a crossed facet pools components: σ²(r:p) = σ²(r) + σ²(pr), and σ²(ri:p) = σ²(ri) + σ²(pri).
- The **divisor** is the product of the decision study's sizes of the effect's facets other than the object.
- An effect with the object and only fixed facets besides is in the **universe score** variance, as are the object's own
  component and, for a fixed facet, its interactions with the object: σ²(τ) = σ²(p) + σ²(pi) / n_i for p × i with i fixed.
- An effect with the object and at least one random facet is in the **relative error** variance σ²(δ), which matters when
  persons are ranked or compared with each other.
- Those, and the effects without the object that have a random facet, are in the **absolute error** variance σ²(Δ), which
  matters when a score is read against a standard. Effects of fixed facets alone are in neither.

The coefficients are Eρ² = σ²(τ) / (σ²(τ) + σ²(δ)) and Φ = σ²(τ) / (σ²(τ) + σ²(Δ)). Φ is at most Eρ². They are undefined
when σ²(τ) and σ²(δ) are both 0, which the function reports as a `DataError`.

| Result field                                     | Meaning                                                                           |
| ------------------------------------------------ | --------------------------------------------------------------------------------- |
| `universeScoreVariance`                          | σ²(τ)                                                                             |
| `relativeErrorVariance`, `absoluteErrorVariance` | σ²(δ) and σ²(Δ)                                                                   |
| `generalizability`, `dependability`              | Eρ² and Φ                                                                         |
| `components`                                     | Each effect with its `variance`, `divisor`, `universe`, `relative` and `absolute` |

```ts
import { decisionStudy, generalizabilityStudy } from "@measurekit/psychometrics/generalizability";

// The study of the first example: 6 persons by 4 items.
const persons = ["p1", "p2", "p3", "p4", "p5", "p6"];
const items = ["i1", "i2", "i3", "i4"];
const ratings = [
  [4, 5, 3, 4],
  [2, 3, 2, 2],
  [5, 5, 4, 5],
  [3, 4, 3, 2],
  [1, 2, 1, 2],
  [4, 4, 5, 4],
];
const scores = ratings.flatMap((row, p) =>
  row.map((score, i) => ({ levels: { person: persons[p] ?? "", item: items[i] ?? "" }, score })),
);
const study = generalizabilityStudy(scores, { facets: ["item"] });

const same = decisionStudy(study); // the study's 4 items
Math.round(same.generalizability * 1000) / 1000; // 0.95
Math.round(same.dependability * 1000) / 1000; // 0.937
const longer = decisionStudy(study, { sampleSizes: { item: 8 } });
[
  longer.universeScoreVariance,
  longer.relativeErrorVariance,
  longer.absoluteErrorVariance,
  longer.generalizability,
  longer.dependability,
].map((value) => Math.round(value * 1000) / 1000); // [1.506, 0.04, 0.05, 0.974, 0.968]
```

Pooling and fixed facets work on any study. With the study of raters crossed with persons, a decision study in which each
person has raters of their own gives the components of (r:p) × i:

```ts
import { decisionStudy, generalizabilityStudy } from "@measurekit/psychometrics/generalizability";

const byPerson = [
  [
    [4, 5, 3],
    [5, 5, 4],
  ],
  [
    [2, 3, 2],
    [3, 3, 2],
  ],
  [
    [5, 5, 4],
    [4, 5, 5],
  ],
  [
    [3, 4, 3],
    [2, 4, 3],
  ],
  [
    [1, 2, 1],
    [2, 2, 2],
  ],
];
const scores = byPerson.flatMap((raters, p) =>
  raters.flatMap((row, r) =>
    row.map((score, i) => ({ levels: { person: `p${p + 1}`, rater: `r${r + 1}`, item: `i${i + 1}` }, score })),
  ),
);
const crossed = generalizabilityStudy(scores, { facets: ["rater", "item"] });
const nested = decisionStudy(crossed, { sampleSizes: { rater: 2, item: 3 }, nested: { rater: ["person"] } });
nested.components.map(({ facets, nestedIn, variance, divisor }) => [
  facets.join(" x "),
  nestedIn.join(" x "),
  Math.round(variance * 1000) / 1000,
  divisor,
]);
// [["person", "", 1.517, 1], ["item", "", 0.2, 3], ["rater", "person", 0.033, 2], ["person x item", "", 0, 3],
//  ["rater x item", "person", 0.233, 6]], after rounding

// With the items fixed, the effects of the item alone are not error and the person by item effect joins the universe.
const fixed = decisionStudy(crossed, {
  sampleSizes: { rater: 2, item: 3 },
  nested: { rater: ["person"] },
  fixed: ["item"],
});
[fixed.universeScoreVariance, fixed.relativeErrorVariance, fixed.generalizability].map(
  (value) => Math.round(value * 1000) / 1000,
);
// [1.517, 0.056, 0.965], after rounding
```

A study is plain data that JSON serializes; `decisionStudy` checks one that comes from storage: its design, a size for
every facet and one finite component for each effect.

## Not included

- Designs that are not balanced, such as incomplete rating designs: Henderson's methods and REML are not implemented, and
  the many-facet Rasch model of the `rater` module handles incomplete ratings.
- Multivariate generalizability theory, and standard errors and confidence intervals of the components.
- Mixed-model G studies: fixed facets are handled in the D study, from the random-effects components.
- Facets in the object of measurement: an object nested in another facet, such as persons within schools.

## Differences from the references

| Topic                         | This package                                                   | Reference                                                     |
| ----------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------- |
| Estimation                    | The analysis of variance, in closed form, for balanced designs | lme4 fits REML, which equals it when no component is negative |
| Negative components           | Reported as estimated; counted as 0 in a D study               | lme4's REML estimates are bounded at 0                        |
| Unbalanced designs            | `DataError`                                                    | lme4 estimates them                                           |
| Coefficients with no variance | `DataError` when σ²(τ) and σ²(δ) are both 0                    | The quotient 0 / 0 is `NaN` in R                              |
