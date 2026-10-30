# Rater agreement

`@measurekit/psychometrics/rater` measures how far raters agree: Cohen's kappa for two raters, unweighted, linear
or quadratic weighted, Fleiss' kappa for two or more, and intraclass correlations of scores. Each is validated
against irr and psych ([validation](validation.md)).

## Ratings

Every function takes ratings by subject: one row per rated subject, such as an essay, with one rating per rater,
`null` when a rater did not rate the subject. Kappa takes categories, integers from 0, as item responses; intraclass
correlations take any finite numbers. Each statistic uses the subjects every rater rated.

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

## References

- Cohen, J. (1960). A coefficient of agreement for nominal scales. Educational and Psychological Measurement, 20,
  37-46.
- Cohen, J. (1968). Weighted kappa. Psychological Bulletin, 70, 213-220.
- Fleiss, J. L. (1971). Measuring nominal scale agreement among many raters. Psychological Bulletin, 76, 378-382.
- McGraw, K. O., & Wong, S. P. (1996). Forming inferences about some intraclass correlation coefficients.
  Psychological Methods, 1, 30-46.
- Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: Uses in assessing rater reliability. Psychological
  Bulletin, 86, 420-428.
