# Interoperability

`@measurekit/psychometrics/interop` converts item parameters ([IRT](irt.md)) to and from mirt's coefficients.

## mirt

mirt reports an item's coefficients in two forms, both with D = 1: the IRT form, `coef(model, IRTpars = TRUE)`, and
the slope-intercept form, `coef(model, IRTpars = FALSE)`. The package's parameters are the IRT form.

| Function                                      | Conversion                                                       |
| --------------------------------------------- | ---------------------------------------------------------------- |
| `toMirtIrt(item)`                             | An item, such as a calibration's, to its IRT coefficients        |
| `toMirtSlopeIntercept(item)`                  | An item to its slope-intercept coefficients                      |
| `fromMirtIrt(coefficients, model)`            | IRT coefficients to an item `{ model, parameters }` of the model |
| `fromMirtSlopeIntercept(coefficients, model)` | Slope-intercept coefficients to an item of the model             |

`MirtCoefficients` is one row of `coef(model, simplify = TRUE)$items` with only the item's own columns.

| Model                                       | mirt item type          | IRT form            | Slope-intercept form                               |
| ------------------------------------------- | ----------------------- | ------------------- | -------------------------------------------------- |
| 1PL to 4PL                                  | `Rasch`, `2PL` to `4PL` | a, b, g = c, u = d  | a1 = a, d = -a b, g, u                             |
| `gradedResponse`                            | `graded`                | a, b1 to b(K-1)     | a1 = a, d_k = -a b_k                               |
| `generalizedPartialCredit`, `partialCredit` | `gpcm`, `Rasch`         | a, b1 to b(K-1)     | a1 = a, ak_k = k, d0 = 0, d_k = -a (b1 + ... + bk) |
| `ratingScale`                               | `rsm`                   | a1, b1 to b(K-1), c | the same as the IRT form                           |
| `nominalResponse`                           | `nominal`               | a1 to aK, c1 to cK  | a1, ak0 = 0 to ak(K-1) = K - 1, d0 = 0 to d(K-1)   |

A conversion from mirt throws `DataError` when a name of the form is missing or not finite, when the coefficients have
a name the form does not use, when it would divide by a slope a1 = 0, or when the parameters are invalid for the
model. Only the built-in models convert.

- **Rating scale.** mirt's thresholds are b_j - c / a1, with b_j shared by its rsm items. `fromMirtIrt` takes the
  location b as their mean and the steps around it; `toMirtIrt` writes the steps as b1 to b(K-1) and c = -a b, which
  give the same thresholds.
- **Nominal.** The IRT form sums the slopes and the intercepts to 0. The slope-intercept form fixes ak0 = 0,
  ak(K-1) = K - 1 and d0 = 0, so an item whose first and last slopes are equal has none.

In R, write the coefficients as one object per item; jsonlite leaves out the columns of other item types:

```r
library(jsonlite)
write_json(as.data.frame(coef(model, IRTpars = TRUE, simplify = TRUE)$items), "items.json", digits = I(17))
```

and read them as items for ability estimation, or take their `parameters` as anchors for calibration:

```ts
import { fromMirtIrt } from "@measurekit/psychometrics/interop";
import { generalizedPartialCredit, threeParameterLogistic } from "@measurekit/psychometrics/irt";

const items = [
  fromMirtIrt({ a: 1.09, b: -1.46, g: 0.2, u: 1 }, threeParameterLogistic),
  fromMirtIrt({ a: 0.86, b1: -3.27, b2: -2.89, b3: 1.54 }, generalizedPartialCredit),
];
items[1]; // { model: generalizedPartialCredit, parameters: { a: 0.86, b: [-3.27, -2.89, 1.54] } }
```
