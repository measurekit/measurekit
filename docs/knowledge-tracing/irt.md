# Item response theory

`@measurekit/knowledge-tracing/irt` estimates learners' ability online from item parameters calibrated beforehand, as
by `@measurekit/psychometrics` or mirt: each outcome updates the posterior of the learner's θ, and the model
predicts the next outcome from it. Estimates are validated against mirt's `fscores` and a posterior written out in R
([validation](validation.md#item-response-theory)).

## The model

Item parameters use the IRT parameterization with D = 1, as `@measurekit/psychometrics` and mirt with `IRTpars = TRUE`:

| Item                                | Parameters                                 | Probability                                                     |
| ----------------------------------- | ------------------------------------------ | --------------------------------------------------------------- |
| Dichotomous, 1PL to 4PL             | `{ a, b, c, d }`; `c` default `0`, `d` `1` | P(correct) = c + (d - c) logistic(a (θ - b))                    |
| Graded response (Samejima)          | `{ model: "graded", a, b: [b_1, ...] }`    | P(category ≥ k) = logistic(a (θ - b_k)), b increasing for a > 0 |
| Generalized partial credit (Muraki) | `{ model: "gpcm", a, b: [b_1, ...] }`      | P(category k) ∝ exp(Σ over j ≤ k of a (θ - b_j))                |

An item `{ model, parameters }` of `@measurekit/psychometrics` gives its `parameters`, with `model: "graded"`
or `"gpcm"` added for the polytomous models. An item with K categories takes the outcomes k / (K - 1), as a rubric
score from 0 to K - 1 divided by K - 1; a dichotomous item takes 0 and 1. A prediction is the expected outcome on
that scale: P(correct) for a dichotomous item.

The posterior of θ starts from the prior, the population's distribution in the calibration, and each outcome multiplies
it by the outcome's probability. θ is assumed constant: the posterior narrows with every outcome and does not follow
learning, and an item answered twice counts twice. For learners who change while they practice, use a model of
learning, such as [Elo](elo.md) or [BKT](bkt.md).

## Scales and skills

A calibration places items on one scale, and the model keeps one θ per learner on it. Skills are not used: every item
measures the one θ. A test whose sections are calibrated separately, as the four skills of a language test, takes one
model per section, each with its items, and a learner has one state in each.

## Options

| Option       | Default              | Meaning                                                                                |
| ------------ | -------------------- | -------------------------------------------------------------------------------------- |
| `parameters` | required             | `{ items: { [item]: parameters } }`                                                    |
| `prior`      | `{ mean: 0, sd: 1 }` | The normal prior of θ                                                                  |
| `points`     | `61`                 | Grid points, equally spaced from 6 prior standard deviations below the mean to 6 above |

The prior's grid needs distinct, finite points, with finite squares of their spacing and of 1 / sd; an sd so small or a
mean so large that it has not throws `DataError`. The default grid is that of `@measurekit/psychometrics` for EAP,
and mirt's `fscores` with `quadpts = 61` and `theta_lim = c(-6, 6)` for the standard prior.

## Online

```ts
import { irt } from "@measurekit/knowledge-tracing/irt";

const model = irt({
  parameters: {
    items: {
      q1: { a: 1.2, b: -0.5 },
      q2: { a: 1.5, b: 0.4, c: 0.2 },
      essay: { model: "gpcm", a: 0.9, b: [-1, 0, 1.2] },
    },
  },
});
const attempt = { learner: "ana", item: "q1", skills: [], time: 1 };
let ana = model.update(model.initialState("ana"), { ...attempt, outcome: 1 });
ana = model.update(ana, { ...attempt, item: "essay", time: 2, outcome: 2 / 3 }); // category 2 of 0 to 3
model.ability(ana); // { theta: 0.508, standardError: 0.748 }, EAP
model.ability(ana, { estimator: "map" }); // the posterior mode
model.predict(ana, { ...attempt, item: "q2", time: 3 }); // 0.625
const { nodes, weights } = model.posterior(ana);
nodes.length; // 61, from -6 to 6
weights.reduce((sum, weight, q) => ((nodes[q] ?? 0) > 0 ? sum + weight : sum), 0); // 0.706, P(θ > 0)
```

`ability(state, { estimator })` returns:

- `"eap"` (default): the posterior mean on the grid and the posterior standard deviation.
- `"map"`: the posterior mode and 1 / √(-ℓ''(θ̂)), ℓ the log-posterior, or `Infinity` where ℓ is not curved down. The
  highest grid point locates the mode; the quintic polynomial matching ℓ and its first two derivatives at the two
  nearest points gives it between them. At the default grid, the mode is within 1e-5 of the exact one; the error grows
  with the fifth power of the spacing. A mode beyond the grid is reported at its end.

`posterior(state)` returns the learner's whole posterior of θ as a `ThetaDistribution`, `{ nodes, weights }`: the grid
and the state's density on it, normalized to sum to 1. `@measurekit/psychometrics` has the same shape, so a posterior
moves between the packages as plain data.

The state, `{ learner, logDensity, slope, curvature }`, holds ℓ and its first two derivatives at each grid point: 3 ×
`points` numbers, about 3.5 KB of JSON at the default grid, whatever the number of outcomes. ℓ is kept less its maximum,
so its largest value is 0. It is plain data; the model checks it. Each outcome's log-probability is computed from the
tail it is near, as ln logistic(±a (θ - b)) where an asymptote allows, so it stays finite where its probability rounds
to 0: a wrong answer to an easy item at the top of the grid. An item without parameters, or an outcome that is not one
of the item's categories, throws `DataError`.

## Fitting

Item parameters are given, not estimated: `fit(log)` returns the model unchanged, with each learner's state after the
log and the predictions before each event, so [evaluation](evaluate.md) runs as for any model. Its log-likelihood is
the marginal log-likelihood of the outcomes on the grid, the learners' θ integrated over the prior; `aic` and `bic`
are `null`, as the item parameters were estimated elsewhere.

## References

- Bock, R. D., & Mislevy, R. J. (1982). Adaptive EAP estimation of ability in a microcomputer environment. Applied
  Psychological Measurement, 6(4), 431-444.
- Chalmers, R. P. (2012). mirt: A multidimensional item response theory package for the R environment. Journal of
  Statistical Software, 48(6), 1-29.
- Muraki, E. (1992). A generalized partial credit model: Application of an EM algorithm. Applied Psychological
  Measurement, 16(2), 159-176.
- Samejima, F. (1969). Estimation of latent ability using a response pattern of graded scores. Psychometrika Monograph
  Supplement, 34(4, Pt. 2).
