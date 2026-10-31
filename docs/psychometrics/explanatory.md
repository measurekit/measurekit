# Explanatory item response theory

`@measurekit/psychometrics/explanatory` explains the parameters of items by their features, such as the number of steps
of a task or whether its response is open, and predicts the parameters of items that nobody has answered yet.

```mermaid
flowchart LR
  B[Bank of calibrated items] -->|"calibrate, standardErrors"| C[Estimates and sampling variances]
  C -->|itemFeatureRegression| R[Effects of the features and the residual between items]
  F[Features of a new item] --> P
  R -->|predictItemParameter| P["Mean and sd of its parameter"]
  P -->|"normalPrior or logNormalPrior in itemPriors"| N[Calibration of the new item on the bank's scale]
  D[Responses to the items] -.->|lltm| T[Do the features account for the difficulties?]
```

| Function                | Estimates                                                                   | From                          |
| ----------------------- | --------------------------------------------------------------------------- | ----------------------------- |
| `lltm`                  | The effects of features on difficulty, and the test against the Rasch model | Responses, by marginal ML     |
| `itemFeatureRegression` | The effects on a parameter and the residual between items                   | A calibration, by REML        |
| `predictItemParameter`  | The parameter of a new item, its mean and sd                                | A regression and the features |

The LLTM tests from the responses whether features account for the difficulties; the regression of a calibration's
estimates predicts a new item with a standard deviation that includes the variation between items, which becomes the
prior of the item's first calibration. The LLTM is validated against a base R maximum with mirt and eRm as cross-checks, the regression against
base R and metafor ([validation](validation.md)).

## The linear logistic test model

The LLTM (Fischer, 1973) makes the difficulty of each item the sum of the effects of its features: the probability of a
correct response is logistic(θ - b_i), b_i = x_iᵀη, with θ ~ N(0, σ²). `lltm(data, features, options)` estimates η and σ
by the Bock-Aitkin EM algorithm on the engine of the [many-facet Rasch model](rater.md), and tests the model against
the Rasch model that gives every item a difficulty of its own.

```ts
import { createRandom, lltm, simulateResponses, simulateThetas, twoParameterLogistic } from "@measurekit/psychometrics";

// Six items whose difficulty is -1.5 + 0.5 per step, answered by 500 simulated persons.
const random = createRandom(3);
const items = [1, 2, 3, 4, 5, 6].map((steps) => ({
  model: twoParameterLogistic,
  parameters: { a: 1, b: -1.5 + 0.5 * steps, c: 0, d: 1 },
}));
const responses = simulateResponses(items, simulateThetas(500, random), random);
const features = {
  item1: { steps: 1 },
  item2: { steps: 2 },
  item3: { steps: 3 },
  item4: { steps: 4 },
  item5: { steps: 5 },
  item6: { steps: 6 },
};
const model = lltm(responses, features);
model.effects; // (intercept) -1.56 (SE 0.11), steps 0.50 (SE 0.03)
model.sd; // 0.98
model.comparison; // { raschLogLikelihood: -1831.8, chiSquare: 2.43, degreesOfFreedom: 4, significance: 0.66 }
model.items[0]?.parameters; // { a: 1, b: -1.06, c: 0, d: 1 }
```

`data` is a [response matrix or response records](data.md) of dichotomous responses; `features` are the
[features by item id](#features) of every item in it.

| Option           | Default        | Valid range             | Meaning                                                              |
| ---------------- | -------------- | ----------------------- | -------------------------------------------------------------------- |
| `intercept`      | `true`         | `true` or `false`       | Whether the difficulties have an intercept                           |
| `quadrature`     | 61 rectangular | as `calibrate`          | The rule over theta                                                  |
| `standardErrors` | `"hessian"`    | `"none"` or `"hessian"` | The Hessian of the marginal log-likelihood, by central differences   |
| `tolerance`      | `1e-4`         | non-negative            | EM stops when no parameter changes by more than this in an iteration |
| `maxIterations`  | `500`          | positive integer        | The most EM iterations                                               |

| Result                        | Meaning                                                                                                                |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `effects`                     | `{ name, estimate, standardError }` for the intercept, named `"(intercept)"`, and each feature, on difficulty          |
| `sd`                          | σ, the standard deviation of theta, whose mean is 0                                                                    |
| `items`                       | Each item with `a = 1`, `b` its difficulty from the effects, `c = 0` and `d = 1`, as a calibration's items             |
| `comparison`                  | The likelihood-ratio test against the Rasch model                                                                      |
| `logLikelihood`, `aic`, `bic` | The marginal log-likelihood; -2 ℓ + 2p and -2 ℓ + p ln n, p the effects and σ, n the persons with a response           |
| `converged`, `iterations`     | Whether EM reached `tolerance` within `maxIterations` for the LLTM and for the Rasch model; the iterations of the LLTM |

- **Scale.** Discrimination is 1 and theta is N(0, σ²), so effects are in logits on the Rasch scale. With an intercept,
  the effect of a feature is its change of the difficulty for one unit, and the intercept is the difficulty of an item
  whose features are all 0, as theta has mean 0. Without an intercept, the difficulty is Σ feature · effect, and a bank
  whose features do not reproduce its average difficulty is rejected by the test.
- **Test.** The LLTM is nested in the Rasch model: χ² = 2 (ℓ_Rasch - ℓ_LLTM) with degrees of freedom the items less
  the effects, and `significance` the probability of a chi-square at least that large if the features account for the
  difficulties. A small value says that items with the same features still differ. The test is of difficulty only: it
  assumes the model of responses holds for both, and a calibration with 2PL slopes is a different model.
- **Items.** Every item needs responses in both categories and none above 1; a polytomous LLTM is not included. Persons
  without responses change nothing and are not counted in the BIC.
- **Valid input.** More items than effects, and effects that no combination of the others reproduces: a constant
  feature with an intercept, or a feature that is a multiple of another, throws `DataError` naming the effect. A feature
  that is 0 or 1 for every item belongs to the design as any other.
- **No residual between items.** Items with equal features get equal difficulties, so predictions for new items from an
  LLTM would be overconfident. When the test rejects the LLTM, or for a prediction with its uncertainty, regress a
  calibration's estimates with [`itemFeatureRegression`](#regressing-a-parameter-on-features).

### Features

`features` maps each item id to an object of feature values, `{ q1: { fractions: 1, steps: 3 } }`. Every item of the
data needs a finite number for each feature that any item has. A categorical feature is one 0/1 column for each of its
categories but one. The effects are ordered by their first appearance among the items, after the intercept, and
features of ids that are not items are ignored.

## The random-item model

An item's difficulty is its features' effect plus what the features leave unexplained: b_i = x_iᵀη + u_i, with
u_i ~ N(0, τ²) (Janssen, Schepers and Peres, 2004; De Boeck, 2008). Items with the same features still differ, by τ, so
the prediction for a new item has a standard deviation √(τ² + the uncertainty of η) instead of the overconfident zero
of a model that gives equal features equal difficulty.

The model is estimated in two stages. Calibrate the bank with any model and standard errors, then regress each
estimate y_i on the features, taking its sampling variance v_i from the calibration:

y_i = x_iᵀη + u_i + e_i, e_i ~ N(0, v_i).

η and τ² come from restricted maximum likelihood (REML). The covariance of the sampling errors between items is
neglected, as in the usual two-stage analysis (Raudenbush, 2009); a calibration's item estimates are nearly independent
given the common scale.

## Regressing a parameter on features

`itemFeatureRegression(items, features, options)` takes the `items` of a calibration made with standard errors and the
features by item id.

```ts
import {
  calibrate,
  createRandom,
  logNormalPrior,
  normalPrior,
  responseMatrix,
  simulateResponses,
  twoParameterLogistic,
} from "@measurekit/psychometrics";
import { itemFeatureRegression, predictItemParameter } from "@measurekit/psychometrics/explanatory";

// A bank of 16 items with two features each: the difficulty is -1.5 + 0.5 steps + 0.4 open + noise.
const rows = [
  { steps: 1, open: 0, noise: 0.2 },
  { steps: 2, open: 0, noise: -0.3 },
  { steps: 3, open: 0, noise: 0.1 },
  { steps: 4, open: 0, noise: 0.25 },
  { steps: 5, open: 0, noise: -0.2 },
  { steps: 6, open: 0, noise: 0.05 },
  { steps: 1, open: 1, noise: -0.1 },
  { steps: 2, open: 1, noise: 0.3 },
  { steps: 3, open: 1, noise: -0.25 },
  { steps: 4, open: 1, noise: 0.15 },
  { steps: 5, open: 1, noise: 0 },
  { steps: 6, open: 1, noise: -0.05 },
  { steps: 2, open: 0, noise: 0.1 },
  { steps: 3, open: 1, noise: -0.2 },
  { steps: 4, open: 0, noise: 0.3 },
  { steps: 5, open: 1, noise: -0.1 },
];
const difficulty = ({ steps, open, noise }: (typeof rows)[number]) => -1.5 + 0.5 * steps + 0.4 * open + noise;
const bank = rows.map((row, i) => ({
  id: `q${i + 1}`,
  model: twoParameterLogistic,
  parameters: { a: 1, b: difficulty(row), c: 0, d: 1 },
}));
const features = Object.fromEntries(rows.map(({ steps, open }, i) => [`q${i + 1}`, { steps, open }]));
// A new item with 4 steps, open, difficulty 0.9: 25 of 1,000 persons answer it.
const newItem = { id: "new1", model: twoParameterLogistic, parameters: { a: 1, b: 0.9, c: 0, d: 1 } };

const random = createRandom(11);
const thetas = Array.from({ length: 1000 }, () => random.normal());
const { responses } = simulateResponses([...bank, newItem], thetas, random);
const ids = [...bank, newItem].map(({ id }) => id);

// 1. Calibrate the bank, with standard errors, and regress its difficulties and slopes on the features.
const calibration = calibrate(
  responseMatrix(
    responses.map((row) => row.slice(0, 16)),
    { itemIds: ids.slice(0, 16) },
  ),
  { model: twoParameterLogistic, standardErrors: "hessian" },
);
const regression = itemFeatureRegression(calibration.items, features);
regression.effects; // (intercept) -1.40 (SE 0.14), steps 0.49 (0.03), open 0.21 (0.11)
regression.residualSd; // 0.18: the spread of difficulties that the features leave
const slopes = itemFeatureRegression(calibration.items, features, { parameter: "a" });

// 2. Predict the new item from its features: the prior of its difficulty and of its slope.
const itsFeatures = { steps: 4, open: 1 };
const b = predictItemParameter(regression, itsFeatures); // { mean: 0.78, sd: 0.20 }
const a = predictItemParameter(slopes, itsFeatures); // { mean: 0.05, sd: 0.04 }, the logarithm of the slope

// 3. Calibrate the new item on the bank's scale, with the prediction as its prior.
const sparse = responseMatrix(
  responses.map((row, person) => row.map((response, item) => (item === 16 && person >= 25 ? null : response))),
  { itemIds: ids },
);
const fixed = Object.fromEntries(calibration.items.map(({ id, parameters }) => [id, parameters]));
const itemPriors = { new1: { b: normalPrior(b.mean, b.sd), a: logNormalPrior(a.mean, a.sd) } };
calibrate(sparse, { model: twoParameterLogistic, fixed }).items[16]?.parameters; // a: 1.00, b: 0.59
calibrate(sparse, { model: twoParameterLogistic, fixed, itemPriors }).items[16]?.parameters; // a: 1.05, b: 0.75
```

| Option          | Default | Valid range                   | Meaning                                                          |
| --------------- | ------- | ----------------------------- | ---------------------------------------------------------------- |
| `parameter`     | `"b"`   | a parameter name of the items | The parameter regressed, as named in the items' `standardErrors` |
| `intercept`     | `true`  | `true` or `false`             | Whether the design has an intercept                              |
| `tolerance`     | `1e-10` | non-negative                  | Fisher scoring stops when τ² changes by less than this in a step |
| `maxIterations` | `100`   | positive integer              | The most Fisher scoring steps                                    |

| Result          | Meaning                                                                                        |
| --------------- | ---------------------------------------------------------------------------------------------- |
| `parameter`     | The regressed parameter's name                                                                 |
| `scale`         | `"log"` for the slope `a`, `"identity"` for any other parameter                                |
| `itemIds`       | The items in the regression, in item order                                                     |
| `effects`       | `{ name, estimate, standardError }` for the intercept, named `"(intercept)"`, and each feature |
| `covariance`    | The covariance matrix of the estimates of the effects, by rows, in the order of `effects`      |
| `residualSd`    | τ̂, the standard deviation of what the features leave between items                             |
| `logLikelihood` | The restricted log-likelihood at the estimates, with its constants, as metafor's `logLik`      |
| `converged`     | Whether Fisher scoring reached `tolerance` within `maxIterations`                              |
| `iterations`    | The Fisher scoring steps taken                                                                 |

- **Items.** The items are the `items` of a calibration, with an `id`, a model and parameters. Features are finite
  numbers by feature name for each item that enters; a categorical feature is one 0/1 column for each of its categories
  but one. Features of other items are ignored, and the effects are ordered by their first appearance among the items,
  after the intercept.
- **Scale.** The slope `a` is positive, so it is regressed as ln a with the variance (SE / a)² of the delta method, and
  needs positive estimates. Any other parameter, thresholds such as `"b1"` included, is regressed as it is.
- **Items left out.** Items without standard errors, such as the anchors fixed in the calibration, and items whose model
  has no parameter of that name, such as a graded item when `parameter` is `"b"`, do not enter. `itemIds` lists those
  that do. A parameter that the items of a model share, such as the `a` of the 1PL, does not vary by item and throws
  `DataError`, as does a standard error that is not positive.
- **Estimation.** The restricted log-likelihood of τ² is -(n - p) / 2 ln 2π + ½ ln |XᵀX| - ½ Σ ln(v_i + τ²) -
  ½ ln |XᵀWX| - ½ (y - Xη̂)ᵀ W (y - Xη̂), with W = diag(1 / (v_i + τ²)) and η̂ = (XᵀWX)⁻¹ XᵀWy. Fisher scoring on τ²
  starts at 0, halves a step that lowers the likelihood, and stays at 0 when the likelihood falls from there: the
  features then explain all the variation between items beyond the sampling error, and `residualSd` is 0. The
  covariance of η̂ is (XᵀWX)⁻¹.
- **Valid input.** More items with standard errors than effects, and effects that no combination of the others
  reproduces: a constant feature with an intercept, or a feature that is a multiple of another, throws `DataError`
  naming the effect. A regression on few items has an imprecise τ̂; ten items for three effects leave seven degrees of
  freedom.

## Predicting a new item

`predictItemParameter(regression, features)` returns `{ mean, sd }`: xᵀη̂ for the new item's features x, and
√(τ̂² + xᵀ Cov x), the standard deviation of the parameter of an item drawn from the model. The values are on the
regression's `scale`. The features need a finite number for each feature of the regression; other features are ignored.
The regression may be stored as JSON and read back.

| Parameter | Prior from the prediction                         | Passed as                                           |
| --------- | ------------------------------------------------- | --------------------------------------------------- |
| `"b"`     | `normalPrior(mean, sd)`                           | `itemPriors: { new1: { b: prior } }` of `calibrate` |
| `"a"`     | `logNormalPrior(mean, sd)`, as `scale` is `"log"` | `itemPriors: { new1: { a: prior } }`                |

`itemPriors` replaces the `priors` entry for that item and parameter ([priors](irt.md#priors)), so an item answered by
few persons is calibrated on the scale of the bank with its estimate pulled toward the prediction, by as much as the
prediction's sd is small against the item's own standard error. In the simulations of [validation](validation.md), the
90% interval mean ± 1.645 sd covers the difficulty of a held-out item between 85% and 95% of the time.

## Errors

Every function throws `DataError`, with a sentence that names the value, for options and arguments of the wrong type
and for these cases:

| Function                | Cases                                                                                                                                                                                                                                 |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lltm`                  | A response above 1, or an item with responses in one category; an item without features, or lacking a feature another has; a feature named `"(intercept)"`; no more items than effects; an effect that other effects reproduce        |
| `itemFeatureRegression` | No item with the parameter, or none with standard errors; a parameter shared by the items of a model; a standard error that is not positive; a slope that is not positive; repeated item ids; the cases of the features as for `lltm` |
| `predictItemParameter`  | A regression that is not an object with effects, a covariance and a residual sd; a feature the regression uses that the item lacks or gives a value that is not finite                                                                |

## Not included

- The polytomous LLTM (the linear partial credit and rating scale models): the engine supports them, and no use needs
  them yet.
- Item features computed from text: finding the features is the caller's job.
- Person covariates (latent regression), the one-stage random-item model estimated from the responses, and the 2PL
  model whose slopes are constrained by features: the two-stage regression of ln a gives the slopes' prior.
- Standard errors adjusted for few items (Knapp and Hartung), and the regression of several parameters at once.

## Differences from the references

| Topic                       | This package                                                                            | Reference                                                                                                                                      |
| --------------------------- | --------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Quadrature of the LLTM      | Nodes μ + σ z on the standard rule, so they follow the distribution                     | mirt keeps its nodes on [-6, 6] when it frees the variance: its estimates differ by up to 2e-5 on the fixtures                                 |
| Intercept of the LLTM       | The difficulty of an item with all features 0, theta having mean 0                      | mirt's `(Intercept)` is an easiness: its coefficients are minus these effects                                                                  |
| Variance of theta           | Reported as the standard deviation σ                                                    | mirt's `COV_11` is σ²                                                                                                                          |
| Standard errors of the LLTM | The Hessian of the marginal log-likelihood by central differences of its exact gradient | mirt supports only `SE.type = "Oakes"` or `"complete"` with an item design: the same covariance, to 1e-3                                       |
| Estimation of the LLTM      | Marginal maximum likelihood, theta random                                               | eRm's `LLTM` is conditional maximum likelihood, on easiness: the estimates agree up to sign within 0.05 on its example                         |
| Estimation of τ²            | Fisher scoring from 0, held at the boundary                                             | metafor's `rma(method = "REML")` agrees where it converges; for a design without an intercept it stopped at a lower likelihood on the fixtures |
| Standard errors of η̂        | (XᵀWX)⁻¹ at τ̂²                                                                          | metafor's default `vcov`, the same; its `test = "knha"` adjustment is not included                                                             |
| Model                       | Two stages: the calibration's estimates and standard errors are the data                | lme4's `glmer(y ~ 0 + X + (1 \| person) + (1 \| item))` estimates the effects from the responses at once, with Laplace                         |
| Estimates of other items    | Independent given their standard errors                                                 | The covariance between items' estimates is a property of the calibration, not used here                                                        |

## References

- Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters: Application of an EM
  algorithm. Psychometrika, 46(4), 443-459.
- De Boeck, P. (2008). Random item IRT models. Psychometrika, 73(4), 533-559.
- Fischer, G. H. (1973). The linear logistic test model as an instrument in educational research. Acta Psychologica,
  37(6), 359-374.
- Harville, D. A. (1977). Maximum likelihood approaches to variance component estimation and to related problems.
  Journal of the American Statistical Association, 72(358), 320-338.
- Janssen, R., Schepers, J., & Peres, D. (2004). Models with item and item group predictors. In P. De Boeck & M. Wilson
  (Eds.), Explanatory item response models (pp. 189-212). Springer.
- Raudenbush, S. W. (2009). Analyzing effect sizes: Random-effects models. In H. Cooper, L. V. Hedges, & J. C.
  Valentine (Eds.), The handbook of research synthesis and meta-analysis (2nd ed., pp. 295-315). Russell Sage Foundation.
- Viechtbauer, W. (2005). Bias and efficiency of meta-analytic variance estimators in the random-effects model. Journal
  of Educational and Behavioral Statistics, 30(3), 261-293.
- Viechtbauer, W. (2010). Conducting meta-analyses in R with the metafor package. Journal of Statistical Software,
  36(3), 1-48.
