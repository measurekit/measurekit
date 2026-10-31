# Cognitive diagnosis

`@measurekit/psychometrics/cdm` tells which attributes, such as the skills a test measures, each person has
mastered: the G-DINA, DINA, DINO, A-CDM, LLM and R-RUM models calibrated by marginal maximum likelihood, persons'
mastery probabilities and profiles, classification accuracy and consistency, Q-matrix validation, and a nonparametric
classification for small samples. Each is validated against GDINA or NPCD ([validation](validation.md)).

## Q-matrix

A `QMatrix` lists the attributes each item measures, by item id. It has the shape of the `QMatrix`
of `@measurekit/knowledge-tracing`, so one Q-matrix serves both packages:

```ts
import type { QMatrix } from "@measurekit/psychometrics/cdm";

const qMatrix: QMatrix = {
  q1: ["fractions"],
  q2: ["fractions", "borrow"],
  q3: ["borrow"],
};
```

Every item of the response matrix needs at least one attribute. The attributes take the order in which they first
appear over the items, which orders every profile in the results; K attributes give 2^K profiles, up to 20 attributes.
A profile is an array of 0 and 1 per attribute, 1 when mastered. Profiles come in GDINA's order: by the number mastered,
then those mastering earlier attributes first, 00, 10, 01, 11.

## Models

An item with K_j attributes splits the profiles into 2^K_j latent groups by its attributes alone. Each model sets the
probability of a correct response in each group.

| `model`   | Success probabilities                                               | Parameters per item |
| --------- | ------------------------------------------------------------------- | ------------------- |
| `"gdina"` | One per latent group (de la Torre, 2011), the saturated model       | 2^K_j               |
| `"dina"`  | 1 - slip mastering all of the item's attributes, guessing otherwise | 2                   |
| `"dino"`  | guessing mastering none of them, 1 - slip otherwise                 | 2                   |
| `"acdm"`  | P = δ₀ + Σ δ_k α_k, the additive model on the identity scale        | K_j + 1             |
| `"llm"`   | logit P = δ₀ + Σ δ_k α_k, the linear logistic model                 | K_j + 1             |
| `"rrum"`  | ln P = δ₀ + Σ δ_k α_k, the reduced reparameterized unified model    | K_j + 1             |

The DINA and DINO are the G-DINA with groups equated. The additive models give an item an intercept δ₀ and one effect
δ_k for each attribute k of the item, on the scale of their link: the identity, logit or log. They are the G-DINA
without interactions, and the same model for an item of one attribute. An effect is not constrained in sign: mastering
an attribute can lower the probability of success, which a monotonicity constraint, not included, would forbid. Each
latent group's probability stays within [1e-4, 1 - 1e-4], a bound that is linear in δ because the link is monotone.

The R-RUM's original parameters are π* = exp(δ₀ + Σ δ_k), the probability of success mastering every attribute of the
item, and r*_k = exp(-δ_k), the penalty for not mastering attribute k.

## Calibration

`calibrateCdm(matrix, qMatrix, options)` estimates the success probabilities and the joint distribution of the
attributes, a proportion for each profile (saturated), by EM over the profiles, as GDINA's `GDINA`. The E-step gives
each person's posterior over the profiles; the M-step sets each group's probability to its expected correct responses
over its expected persons, pooled over equated groups, and each profile's proportion to its expected persons. For an
additive item there is no such closed form: the M-step maximizes Σ_l [R_l ln P_l + (N_l - R_l) ln(1 - P_l)] over δ, for
the expected correct responses R_l and persons N_l of its latent groups l, exactly. The objective is concave in δ under
all three links and the bounds on P are linear in δ, so the maximum is unique; Newton's method with an active set finds
it, starting from the previous iteration's. As GDINA, a probability stays within [1e-4, 1 - 1e-4], a group with fewer
than 0.001 expected persons takes 0.0005 more correct responses over 0.001 more persons, and a proportion stays above
the machine epsilon. Responses are 0 or 1; missing ones are skipped.

| Option          | Default   | Meaning                                                                                                                       |
| --------------- | --------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `model`         | `"gdina"` | One model for every item, or an array with one per item, each of `"gdina"`, `"dina"`, `"dino"`, `"acdm"`, `"llm"` or `"rrum"` |
| `tolerance`     | `1e-4`    | EM stops when no probability or proportion changes by more than this                                                          |
| `maxIterations` | `2000`    | The most EM iterations                                                                                                        |

| Result          | Meaning                                                                                                                     |
| --------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `attributes`    | The attributes, in order                                                                                                    |
| `items`         | Per item: `id`, `attributes`, `model`, `probabilities` per latent group, `guessing`, `slip` and `coefficients`              |
| `profiles`      | Each profile with its `proportion`                                                                                          |
| `logLikelihood` | The marginal log-likelihood                                                                                                 |
| `aic`, `bic`    | With 2^K_j parameters per G-DINA item, 2 per DINA or DINO item, K_j + 1 per additive item, and the 2^K - 1 free proportions |
| `converged`     | Whether EM reached `tolerance`                                                                                              |
| `iterations`    | EM iterations run                                                                                                           |

`coefficients` are GDINA's `coef(what = "delta")`:

| Model                       | `coefficients`                                                                                                                                                           |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `"acdm"`, `"llm"`, `"rrum"` | δ₀, then δ_k for each attribute of the item, in the item's order, on the model's link scale                                                                              |
| `"gdina"`                   | The effects of the identity-link model with every interaction, one per latent group in the order of `probabilities`: δ₀, the main effects, the two-way interactions, ... |
| `"dina"`, `"dino"`          | The guessing, then the rise from it to 1 - slip                                                                                                                          |

```ts
import { calibrateCdm } from "@measurekit/psychometrics/cdm";
import { responseMatrix } from "@measurekit/psychometrics/data";

const matrix = responseMatrix([
  [1, 1, 0, 1],
  [0, 0, 1, 0],
  [1, 0, 0, 1],
  [1, 1, 1, 1],
  [0, 1, 0, 0],
  [0, 0, 0, 0],
]);
const qMatrix = { item1: ["a"], item2: ["a", "b"], item3: ["b"], item4: ["a"] };
const rrum = calibrateCdm(matrix, qMatrix, { model: "rrum" });
rrum.items[1]?.coefficients; // [-7.073, 1.098, 5.975]: δ₀, δ_a, δ_b, on the log scale
```

Profiles whose ideal responses agree on every item, frequent under the DINA and DINO with many attributes, cannot be
told apart: their proportions are identified only in sum, and so are the persons' posteriors among them. With many
attributes and few persons the G-DINA can have several maxima; compare the log-likelihoods of a few calibrations.

## Persons

`attributeProfiles(matrix, calibration)` returns, per person, as GDINA's `personparm`:

| Field       | Meaning                                                                          |
| ----------- | -------------------------------------------------------------------------------- |
| `posterior` | The posterior probability of each profile                                        |
| `mastery`   | The posterior probability of mastering each attribute                            |
| `map`       | The profile of largest posterior probability, the first in profile order on ties |
| `eap`       | Each attribute mastered when its probability is above 0.5                        |

Persons are classified from the success probabilities and proportions alone, so every model works unchanged. A calibration
from JSON may hold success probabilities of 0 or 1, which make some responses impossible. Such a
response's probability enters the likelihood as `Number.MIN_VALUE`, so that a person whose responses no profile can give
has a posterior on the profiles that the responses contradict least.

`cdmClassificationAccuracy(matrix, calibration, { classification })` returns GDINA's `CA` indices for `"map"`
(default) or `"eap"` classification: `accuracy` τ = Σ_c π_c τ_c, the probability that a person lands in the
true profile, with `profileAccuracy` τ_c, and `attributeAccuracy` τ_k; `consistency` γ, the mean of Σ_c post(c)²,
the chance that two measurements agree, and `attributeConsistency` γ_k. They use the persons with a response.

```ts
import { attributeProfiles, calibrateCdm, cdmClassificationAccuracy } from "@measurekit/psychometrics/cdm";
import { responseMatrix } from "@measurekit/psychometrics/data";

const matrix = responseMatrix([
  [1, 1, 0, 1],
  [0, 0, 1, 0],
  [1, 0, 0, 1],
  [1, 1, 1, 1],
  [0, 1, 0, 0],
  [0, 0, 0, 0],
]);
const qMatrix = { item1: ["a"], item2: ["a", "b"], item3: ["b"], item4: ["a"] };
const calibration = calibrateCdm(matrix, qMatrix, { model: "dina" });
attributeProfiles(matrix, calibration)[0]; // { id, mastery: [pa, pb], map, eap, posterior }
cdmClassificationAccuracy(matrix, calibration); // { accuracy, consistency, ... }
```

## Q-matrix validation

`validateQMatrix(matrix, calibration, { eps })` checks each item's attributes by the PVAF method of de la Torre and
Chiu (2016), as GDINA's `Qval(method = "PVAF")`. For a q-vector q, ς²(q) is the variance of the item's success
probabilities over the latent groups of q, weighted by the groups' proportions, each profile's probability its persons'
expected share correct; PVAF(q) is ς²(q) over ς² of all attributes. The suggestion is the q-vector with the fewest
attributes whose PVAF exceeds `eps`, default 0.95, of those the one of largest PVAF. An item whose success
probabilities do not vary over the profiles, such as an item without responses, has `pvaf` `null` and keeps its
attributes. A suggestion is evidence for experts to weigh, not a correction: calibrate the G-DINA before validating.

## Small samples

`nonparametricClassification(matrix, qMatrix, { rule })` classifies persons without calibration (Chiu and Douglas,
2013), as NPCD's `AlphaNP` with the Hamming distance. Each profile's ideal responses follow the Q-matrix by `rule`:
`"conjunctive"` (default) answers an item correctly mastering all its attributes, as the DINA; `"disjunctive"`
mastering any. A person takes the profile whose ideal responses differ from the person's on the fewest answered items,
the first in profile order on ties; the result reports the `distance` and the number of `ties`.

It needs no item parameters, so it suits a classroom of about 30 persons, where the G-DINA's 2^K_j probabilities per
item and 2^K - 1 proportions cannot be estimated. Prefer the G-DINA from several hundred persons, when slips and guesses
differ between items and the posterior's probabilities are wanted; nonparametric classification assumes the ideal
responses up to errors and gives no probabilities.

## Simulation

`simulateCdmResponses(model, profiles, random)` in [`@measurekit/psychometrics/sim`](sim.md) simulates responses from a
model's attributes and items, such as a calibration's, for one person per profile. It takes success probabilities, so
the probabilities of an additive item follow from its coefficients: g⁻¹(δ₀ + Σ δ_k α_k) for each of its latent groups.

## Differences from the references

| Reference | Difference                                                                                                                                                                                                                                                                                                                                                   |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| GDINA     | An additive M-step is maximized exactly. GDINA's solvers stop near 1e-6, so its estimates and log-likelihood fall slightly short: the two agree within 1e-5, and the log-likelihood here is at least GDINA's.                                                                                                                                                |
| GDINA     | The small-group correction, which GDINA applies to the G-DINA, DINA and DINO, applies to every model. It changes only groups with fewer than 0.001 expected persons.                                                                                                                                                                                         |
| GDINA     | GDINA starts from the best of three random starts; `calibrateCdm` is deterministic, starting at P = 0.2 + 0.6 m / K_j for m of K_j attributes mastered, and for an additive item at the least squares fit of that on the link scale. The likelihood of a model with several attributes can have several maxima: compare the log-likelihoods of calibrations. |
| GDINA     | No monotonicity constraint (`mono.constraint`), no item-level Wald tests (`modelcomp`) and no standard errors.                                                                                                                                                                                                                                               |
| CDM       | The `gdina(rule = "ACDM")` of the CDM package uses another M-step and is not a reference.                                                                                                                                                                                                                                                                    |

## Not included

- Item-level model comparison (Wald tests), which needs the covariance of the item parameters that `calibrateCdm` does
  not compute.
- Monotonicity constraints, sequential CDMs and hierarchical attribute structures.
- The R-RUM's original (π*, r*) parameters, which follow from `coefficients` as above.
- Adaptive cognitive diagnosis (CD-CAT), higher-order attribute models and polytomous attributes.

## References

- Chiu, C.-Y., & Douglas, J. (2013). A nonparametric approach to cognitive diagnosis by proximity to ideal response
  patterns. Journal of Classification, 30, 225-250.
- de la Torre, J. (2011). The generalized DINA model framework. Psychometrika, 76, 179-199.
- de la Torre, J., & Chiu, C.-Y. (2016). A general method of empirical Q-matrix validation. Psychometrika, 81, 253-273.
- Iaconangelo, C. (2017). Uses of classification error probabilities in the three-step approach to estimating
  cognitive diagnosis models. Rutgers University.
- Wang, W., Song, L., Chen, P., Meng, Y., & Ding, S. (2015). Attribute-level and pattern-level classification
  consistency and accuracy indices for cognitive diagnostic assessment. Journal of Educational Measurement, 52, 457-476.
