# Classical test theory

`@measurekit/psychometrics/ctt` computes item statistics, distractor analysis, reliability, McDonald's omega and
the standard error of measurement from a [response matrix](data.md). Every statistic is validated against R
([validation](validation.md)).

## Missing responses

| Statistics                                                                                                        | Persons used                                     |
| ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| An item's `count`, `mean` and `sd`; an option's `count` and `proportion`                                          | Everyone who answered the item                   |
| Everything that uses the total score or other items: correlations, D, alpha if deleted, group shares, reliability | Persons who answered every item (complete cases) |

Persons without any response change nothing. Using complete cases keeps every coefficient consistent with its
definition: alpha is at most 1, correlations lie in [-1, 1] and KR-21 is at most KR-20, which pairwise deletion does
not guarantee. To keep persons who omitted items, score omissions as wrong ([data](data.md#scoring-against-a-key)).

## Item statistics

`itemStatistics(matrix, { groupFraction })` returns one `ItemStatistics` per item, in item order.

| Field                 | Definition                                                                                                                      |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `count`               | Persons who answered the item                                                                                                   |
| `mean`                | Mean score; the p-value (difficulty) of a 0/1 item                                                                              |
| `sd`                  | Standard deviation, denominator n - 1                                                                                           |
| `itemTotal`           | Pearson correlation with the total score; the point-biserial correlation of a 0/1 item                                          |
| `itemRest`            | Correlation with the total of the other items (corrected item-total correlation)                                                |
| `biserial`            | r √(pq) / φ(z), with r = `itemTotal`, p the share correct, q = 1 - p, φ(z) the normal density at the p quantile. 0/1 items only |
| `discriminationIndex` | D: mean score in the upper group minus mean score in the lower group; for a 0/1 item, p(upper) - p(lower)                       |
| `alphaIfDeleted`      | Cronbach's alpha of the other items; `null` with fewer than three items                                                         |

An item counts as dichotomous for `biserial` when every response to it is 0 or 1.

## Upper and lower groups

The upper and lower groups are the `groupFraction` of complete cases with the highest and the lowest total scores.
The default is 0.27 (Kelley, 1939); any value above 0 and at most 0.5 is accepted. Each group holds exactly
`groupFraction` × n persons: persons tied at a group's boundary share the remaining places equally, as fractions of a
person. With 10 persons and distinct totals, the upper group is the top two and 0.7 of the third. So D and the group
shares never depend on the order of persons.

## Distractor analysis

`distractorAnalysis(scored, { groupFraction })` takes the result of `scoreResponses` and returns, per item, its `id`,
its `key` and an `OptionStatistics` per option label:

| Field           | Definition                                                       |
| --------------- | ---------------------------------------------------------------- |
| `count`         | Persons who chose the option                                     |
| `proportion`    | `count` divided by the persons who answered the item             |
| `upper`         | Share of the upper group that chose the option                   |
| `lower`         | Share of the lower group that chose the option                   |
| `pointBiserial` | Correlation of choosing the option (1 or 0) with the total score |

A working distractor draws more of the lower group than of the upper group and has a negative point-biserial
correlation. The key's `pointBiserial` equals the item's `itemTotal`.

## Reliability

| Function                     | Value                                                                                                               |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `cronbachAlpha`              | k / (k - 1) · (1 - Σ sᵢ² / s²), with item variances sᵢ² and total score variance s²                                 |
| `kr20`                       | The same for 0/1 items; throws `DataError` for other scores                                                         |
| `kr21`                       | k / (k - 1) · (1 - M (k - M) / (k σ²)), with the total score mean M and variance σ² (denominator n); 0/1 items only |
| `splitHalf`                  | `correlation` r of the half totals, `spearmanBrown` 2r / (1 + r), `guttman` 4 cov(A, B) / var(A + B)                |
| `mcdonaldOmega`              | McDonald's omega total of one factor, 1 - Σ (1 - λ²) / Σ r, below                                                   |
| `standardErrorOfMeasurement` | s √(1 - reliability), in total score points; the reliability defaults to `cronbachAlpha`                            |

`splitHalf` splits the items `odd-even` (items 1, 3, 5, ... against 2, 4, 6, ...; the default) or `first-second`
(the first k / 2 items, rounded down, against the rest). The Spearman-Brown value assumes halves of equal variance;
Guttman's coefficient (Flanagan-Rulon) does not. KR-21 assumes items of equal difficulty and is at most KR-20.

`mcdonaldOmega` fits one factor to the Pearson correlations r of the items by minres, as psych's
`omega(nfactors = 1)`: the loadings λ minimize Σ_(i<j) (r_ij - λ_i λ_j)², by Newton's method from the centroid
loadings. Omega is 1 - Σ (1 - λ_i²) / Σ_ij r_ij, the diagonal included: psych's `omega.tot`, from correlations, so for
standardized item scores. It returns `omega`, the `loadings`, whose sum is positive, `converged` and `iterations`. It
needs three items and correlations whose sum Σ_ij r_ij, the variance of the standardized total, is above 0; a loading
above 1 (a Heywood case) shows that one factor does not fit. Alpha equals omega when the loadings are equal, and is
lower otherwise.

## Classification

`classifyDifficulty(p)` and `classifyDiscrimination(value)` label items. Both take optional thresholds; the defaults
are exported.

| `DIFFICULTY_THRESHOLDS` | Class       |
| ----------------------- | ----------- |
| p < 0.30 (`difficult`)  | `difficult` |
| 0.30 ≤ p ≤ 0.70         | `moderate`  |
| p > 0.70 (`easy`)       | `easy`      |

| `DISCRIMINATION_THRESHOLDS` (Ebel and Frisbie, 1991) | Class       |
| ---------------------------------------------------- | ----------- |
| value ≥ 0.40 (`veryGood`)                            | `very-good` |
| 0.30 ≤ value < 0.40 (`good`)                         | `good`      |
| 0.20 ≤ value < 0.30 (`marginal`)                     | `marginal`  |
| value < 0.20                                         | `poor`      |

```ts
import { classifyDifficulty, DIFFICULTY_THRESHOLDS } from "@measurekit/psychometrics/ctt";

classifyDifficulty(0.75, { ...DIFFICULTY_THRESHOLDS, easy: 0.8 }); // "moderate"
```

Ebel's bounds were set for D; they are often applied to `itemTotal` and `itemRest` as well.

## Undefined statistics and edge cases

A function that returns a table (`itemStatistics`, `distractorAnalysis`) sets a statistic that is undefined for the
data to `null`. A function that returns one statistic throws `DataError` with the reason instead. No result is `NaN`.

| Case                                                | Result                                                                                                          |
| --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| An item with one score for everyone                 | `sd` 0; `itemTotal`, `itemRest` and `biserial` `null`; D 0. Alpha counts it as an item of variance 0            |
| A person without responses                          | Skipped by every statistic                                                                                      |
| An item without responses                           | `count` 0, `mean` and `sd` `null`. Nobody answered every item, so the next row applies; remove the item         |
| Fewer than two persons answered every item          | Score-based fields `null`; reliability functions and `standardErrorOfMeasurement` throw `DataError`             |
| One item                                            | `itemTotal` 1 (or `null` without variance), `itemRest` and `alphaIfDeleted` `null`; reliability functions throw |
| Two persons                                         | Defined: correlations are 1, -1 or `null`; each group holds 0.54 of a person                                    |
| Total scores without variance                       | Correlations `null`; `cronbachAlpha`, `kr20`, `kr21` throw                                                      |
| A half score without variance, or halves at r = -1  | `splitHalf` throws                                                                                              |
| An item without variance, or fewer than three items | `mcdonaldOmega` throws                                                                                          |
| A reliability outside [0, 1]                        | `standardErrorOfMeasurement` throws                                                                             |

## References

- Cronbach, L. J. (1951). Coefficient alpha and the internal structure of tests. Psychometrika, 16, 297-334.
- Ebel, R. L., & Frisbie, D. A. (1991). Essentials of educational measurement (5th ed.). Prentice Hall.
- Guttman, L. (1945). A basis for analyzing test-retest reliability. Psychometrika, 10, 255-282.
- Kelley, T. L. (1939). The selection of upper and lower groups for the validation of test items. Journal of
  Educational Psychology, 30, 17-24.
- Kuder, G. F., & Richardson, M. W. (1937). The theory of the estimation of test reliability. Psychometrika, 2,
  151-160.
- Lord, F. M., & Novick, M. R. (1968). Statistical theories of mental test scores. Addison-Wesley.
- McDonald, R. P. (1999). Test theory: A unified treatment. Lawrence Erlbaum.
