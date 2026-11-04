# Bayesian knowledge tracing

`@measurekit/knowledge-tracing/bkt` traces each skill with Bayesian knowledge tracing (Corbett and Anderson, 1994)
and the extensions of pyBKT (Badrinath, Wang and Pardos, 2021): forgetting, item guess and slip, resource learning and
prior groups, pairs of resources, with fixed rates and bounds on the rates that EM estimates. Inference at fixed
parameters and EM fitting, with and without events that have no outcome, are validated against pyBKT, and bounds against
base R ([validation](validation.md#bayesian-knowledge-tracing)).

```mermaid
flowchart LR
  L["P(L)"] --> Q{outcome is null?}
  Q -->|no| P["predict P(correct)"]
  P --> C["condition P(L) on the outcome"]
  Q -->|yes| M["move P(L) by the learn and forget rates of the resource"]
  C --> M
  M --> N["P(L) at the next attempt"]
```

## Quick start

Trace a skill at given rates: predict an attempt, update the learner's state with its outcome, and read P(L):

```ts
import { bkt } from "@measurekit/knowledge-tracing/bkt";

const model = bkt({
  parameters: { skills: { fractions: { prior: 0.3, learn: 0.2, forget: 0, guess: 0.2, slip: 0.1 } } },
});
const ana = model.initialState("ana"); // => { learner: "ana", skills: {} }
const attempt = { learner: "ana", item: "q1", skills: ["fractions"], time: 1 };
model.predict(ana, attempt); // => 0.41
const next = model.update(ana, { ...attempt, outcome: 1 }); // => { learner: "ana", skills: { fractions: 0.73 } }
model.mastery(next, "fractions"); // => 0.73, P(L)
model.opportunitiesToMastery(next, "fractions"); // => 8
```

## The model

For each skill, a learner either knows it or not:

| Rate  | Field    | Meaning                                                                 |
| ----- | -------- | ----------------------------------------------------------------------- |
| P(L0) | `prior`  | The probability of knowing the skill before practicing it               |
| P(T)  | `learn`  | The probability of learning it from an attempt                          |
| P(F)  | `forget` | The probability of forgetting it after an attempt; 0 without forgetting |
| P(G)  | `guess`  | The probability of a correct answer without knowing it                  |
| P(S)  | `slip`   | The probability of an incorrect answer despite knowing it               |

With P(L), the probability that the learner knows the skill at their next attempt,

P(correct) = P(L) (1 - P(S)) + (1 - P(L)) P(G).

After the outcome, P(L) is conditioned on it by Bayes' rule, then moved by the rates of the resource just practiced:
P(L) ← P(L | outcome) (1 - P(F)) + (1 - P(L | outcome)) P(T).

- **Outcomes** are 0 or 1, and every event with an outcome needs at least one skill; other events throw `DataError`.
- **Events without an outcome.** An event with `outcome: null`, such as a lesson, shows nothing about P(L), so only the
  learn and forget rates apply: P(L) ← P(L) (1 - P(F)) + (1 - P(L)) P(T). It has no prediction, adds no guess or slip,
  and a skill without parameters stays as it is, online too. Its skills may be none.
- **Several skills.** An event with several skills is an event of each: each skill's P(L) updates as for that skill
  alone, the prediction is the mean of the skills' predictions, and fitting counts the event in every skill's
  sequences, as Elo does ([architecture](architecture.md#events-with-several-skills)).
- **Skills are independent.** Each has its own parameters and is fitted on its own events.

## Options

| Option             | Default | Meaning                                                                                           | pyBKT        |
| ------------------ | ------- | ------------------------------------------------------------------------------------------------- | ------------ |
| `forgetting`       | `false` | Estimate P(F); without it P(F) is 0                                                               | `forgets`    |
| `itemGuessSlip`    | `false` | Each item has its own P(G) and P(S)                                                               | `multigs`    |
| `resourceLearning` | `false` | Each resource has its own P(T), and P(F) with forgetting, applied after an attempt with it        | `multilearn` |
| `pairLearning`     | `false` | Learn and forget rates by the resource before an attempt and its own; not with `resourceLearning` | `multipair`  |
| `fixed`            | none    | Rates held at given values, by skill id; the others are estimated                                 | `fixed`      |
| `bounds`           | none    | A lower and an upper bound for each rate that EM estimates                                        | none         |
| `priorGroups`      | none    | The group of each learner, by learner id; each group has its own P(L0) per skill                  | `multiprior` |
| `restarts`         | `5`     | Random starts of EM for a skill without parameters; the highest log-likelihood is kept            | `num_fits`   |
| `seed`             | `0`     | The seed of the random starts; every skill draws its starts from a generator with this seed       | `seed`       |
| `tolerance`        | `1e-6`  | EM has converged when a skill's log-likelihood changes by at most this                            | `tol`        |
| `maxIterations`    | `500`   | The most EM iterations of a start                                                                 | `maxiter`    |
| `parameters`       | none    | The parameters to predict with and to start EM from                                               | `coef_`      |

The options combine freely. Guess and slip use the event's `item`: to share them among items, such as those of one
template, give their events the template's id as `item`. Learn and forget rates use the event's `resource`, which is its
item when it has none: a lesson, a video and an exercise can each have rates of their own.

## Parameters

`model.parameters` is plain data, `{ skills: { [skill]: BktSkill } }`. A skill has the five rates above and, with the
extensions, `groupPriors` by group, `resourceLearning` `{ learn, forget }` by resource, `pairLearning` `{ learn, forget }`
by the resource before an attempt and then its own, and `itemGuessSlip` `{ guess, slip }` by item. An item uses its own guess and slip, and a resource its own learn and forget rates, when the
skill has them and the skill's otherwise; a learner uses their group's prior when the skill has one and the skill's
otherwise. So items, resources and learners that a fit did not see, such as new items, use the skill's rates. Where
every item, resource or group has its own, the skill's rate is their pooled estimate: the expected count over the
expected opportunities of all of them.

## Fixed rates and bounds

```mermaid
flowchart LR
  E["E-step: expected counts"] --> R["each rate: expected count over expected opportunities"]
  R --> F{"fixed?"}
  F -->|yes| H["the given value"]
  F -->|no| C["clamped to its bounds"]
  H --> N["next iteration"]
  C --> N
```

| Option          | Holds                                                                                                                                       | Default                                            |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| `fixed[skill]`  | `prior`, `learn`, `forget`, `guess` and `slip` of the skill's own rates; `resources` the rates of single resources, with `resourceLearning` | none: every rate is estimated                      |
| `bounds.<rate>` | `[lower, upper]` for `prior`, `learn`, `forget`, `guess` or `slip`, of the skill's rate and of every group, resource, pair or item          | `[0, 1]`; a guess and a slip stay 2⁻⁵² off 0 and 1 |

- **Fixed.** A fixed rate is not estimated: a lesson's learn rate known from design can be held while practice's is fit,
  or a guess held at a chance level. `aic` and `bic` leave it out of k. With item or resource rates, a skill's own rate is
  their pooled estimate, and holding it fixes that value alone; hold single resources through `resources`.
- **Bounds.** Each rate's expected log-likelihood is a Bernoulli term, concave in the rate, so clamping the ratio of
  expected counts to the bounds is the exact constrained maximum of each M-step. Random starts are clamped to the
  bounds, and given parameters outside them are rejected.

## Pairs

With `pairLearning`, the learn and forget rates after an attempt depend on the resource before it and its own, as
pyBKT's `multipair`. A learner's first attempt at a skill uses the skill's own rates, pyBKT's `Default`; a pair the fit
has not seen uses them too. Four items give 16 pairs and the skill's rates. The state's `previous` holds the resource of
the latest attempt at each skill; a state without it, as a model without pairs writes, reads as one with no earlier
attempt.

## Degenerate fits

BKT's likelihood is flat along some directions, so EM can end at rates that fit as well as sensible ones and mean
something else: a guess above 0.5, or a slip and a guess that swap the meaning of knowing (Beck and Chang, 2007; Baker,
Corbett and Gowda, 2008). Signs are a P(G) or P(S) at 0.5 or above, a P(T) near 0 with a prior near 1, or fits from
different starts that agree in likelihood and differ in rates. `restarts` shows the second kind; `bounds` and `fixed` guard
against all three: Baker et al. cap the guess at 0.3 and the slip at 0.1, `bounds: { guess: [0, 0.3], slip: [0, 0.1] }`. A
fit that sits on a bound is a hint that the data want more than the bound allows, not a result to report as is.

## Fitting

`fit(log)` estimates each skill by expectation maximization on its learners' sequences, with the forward-backward
algorithm (Baum-Welch). A skill with parameters in the model starts from them; any other starts from `restarts` random
starts: P(L0) from U(0, 1), P(T) from U(0, 0.4), P(F) from U(0, 0.2) with forgetting, P(G) from U(0, 0.4) and P(S) from
U(0, 0.3), as pyBKT draws them except P(F). EM stops when the log-likelihood changes by at most `tolerance`. It keeps
P(G) and P(S) at least 2⁻⁵² from 0 and 1, so that a fitted model gives every outcome a positive probability: data whose
outcomes of a skill are all correct would otherwise fit P(S) = 0, and an incorrect answer would be impossible.

```ts
import { createRandom } from "@measurekit/knowledge-tracing";
import { bkt } from "@measurekit/knowledge-tracing/bkt";
import { buildEventLog } from "@measurekit/knowledge-tracing/data";

const random = createRandom(1);
const events = Array.from({ length: 30 }, (_, learner) =>
  Array.from({ length: 8 }, (_, attempt) => ({
    learner: `l${learner}`,
    item: `q${1 + (attempt % 4)}`,
    time: learner * 10 + attempt,
    outcome: random.uniform() < 0.3 + 0.08 * attempt ? 1 : 0,
  })),
).flat();
const qMatrix = { q1: ["fractions"], q2: ["fractions"], q3: ["decimals"], q4: ["decimals"] };
const log = buildEventLog(events, { qMatrix });
const fit = bkt({ forgetting: true, itemGuessSlip: true }).fit(log);
fit.model.parameters.skills.fractions;
// => { prior: 0.22, learn: 0.64, forget: 0.00, guess: 0.00, slip: 0.25,
//      itemGuessSlip: { q1: { guess: 0.00, slip: 0.12 }, q2: { guess: 0.00, slip: 0.33 } } }
fit.states.l0; // => { learner: "l0", skills: { fractions: 1.00, decimals: 1.00 } }: the learner's P(L) after the log
```

| Fit field                 | Meaning                                                                                                                                                                           |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `predictions`             | Each event's prediction by online inference with the fitted parameters; `null` for an event without an outcome                                                                    |
| `logLikelihood`           | The log-likelihood of those predictions: for events of one skill, the HMM log-likelihood                                                                                          |
| `aic`, `bic`              | With k the rates the events estimate: of each skill, its priors, learn rates, forget rates with forgetting, guess and slip rates in use; n of `bic` is the events with an outcome |
| `converged`, `iterations` | Whether every skill's chosen start converged, and the most iterations of any                                                                                                      |
| `skills`                  | By skill id, each skill's `converged`, `iterations`, `events` with an outcome and `logLikelihood`; a skill met only through lessons is absent                                     |

In EM, an event without an outcome has probability 1 in both states and adds no guess or slip count, while the learning
step after it counts for its resource: a lesson's P(T) is estimated from the outcomes that follow it. A transition with
no outcome after it in the sequence, such as a lesson at its end, teaches nothing and is not counted in `aic` and `bic`;
a skill met only through events without an outcome gets no parameters, and a lesson gets no guess or slip. `fit` throws
`DataError` for a log with no event with an outcome.

The skills fit apart, so a skill's entry of `skills` is the fit of that skill's events alone, as pyBKT's EM for one skill at a
time gives it. When every event has one skill, the log-likelihoods of `skills` add up to `logLikelihood`; an event with several
skills is predicted by the mean of their predictions, which `logLikelihood` scores once and each skill's fit scores apart.

A rate without expected opportunities in the data, such as P(T) of a skill whose learners each attempted it once, keeps
its starting value. Skills of the model absent from the log keep their parameters. A log too large for memory can be
fitted skill by skill and the results merged ([storage](storage.md#fitting-bkt-skill-by-skill)).

## Online

The [quick start](#quick-start) is online use, at given rates.

- `update(state, event)` returns a new state; `predict` and `mastery` read it. A skill the learner has not practiced
  has their prior. The model has no parameters for a skill it was not fitted to, and throws `DataError` for it.
- The state, `{ learner, skills: { [skill]: P(L) } }`, is plain data: store it as JSON and pass it back. Model methods
  check it.

## Mastery

`mastery(state, skill)` is P(L). `opportunitiesToMastery(state, skill, { threshold })` is the number of further
attempts at the skill after which P(L) is expected to reach the threshold (default `0.95`), before their outcomes are
known. Without new outcomes, the expected P(L) moves by the learn and forget rates alone, so after n attempts it is
limit + (P(L) - limit) (1 - P(T) - P(F))ⁿ with limit = P(T) / (P(T) + P(F)). The result is 0 when P(L) has reached the
threshold, and `Infinity` when the limit is at or below it or the number exceeds the largest double, as for a P(T) near
the smallest double. It uses the skill's P(T) and P(F), as the items to come are not known.

## Errors

Invalid input throws `DataError` naming the value found.

| Error                                                                                                                                         | What to do                                                                                                               |
| --------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| An outcome other than 0, 1 or `null`, or an event with an outcome and no skill                                                                | Score each attempt 0 or 1, and give its skills or a Q-matrix to `buildEventLog`; trace partial credit with [Elo](elo.md) |
| A skill the model has no parameters for, in `predict`, `update` or `mastery`                                                                  | Fit the model on a log with the skill, or give its rates in `parameters`                                                 |
| A rate that is not a probability, in `parameters` or `fixed`                                                                                  | Give each rate from 0 to 1                                                                                               |
| A bound that is not `[lower, upper]` of probabilities, or with the lower above the upper                                                      | Give each bound as two probabilities in order                                                                            |
| `parameters` or a fixed rate outside the bounds                                                                                               | Widen the bounds, or start from rates within them                                                                        |
| `forget` fixed or bounded without `forgetting: true`, `resources` without `resourceLearning: true`, or `pairLearning` with `resourceLearning` | Turn on the option that the rate needs, and choose one of the two resource options                                       |
| A prior group that is not a non-empty string                                                                                                  | Give each learner of `priorGroups` a group id                                                                            |
| An outcome of probability 0 under given parameters, as a correct answer with P(L) = 0 and P(G) = 0                                            | Give guess and slip rates above 0 and below 1, or fit them: fitted rates give every outcome a positive probability       |
| A log without an event with an outcome, for `fit`                                                                                             | Fit a log with outcomes                                                                                                  |
| A malformed state                                                                                                                             | Pass a state that `initialState` or `update` of the model returned, as stored                                            |

## Not included

| Not included       | Why                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Contextual BKT     | No reference: no package has it, not pyBKT, nor its R port `BKT`.                                                                                                                                                                                                                                                                                                                                                                                                           |
| Partial-credit BKT | No reference: no package has it, not pyBKT, nor its R port `BKT`.                                                                                                                                                                                                                                                                                                                                                                                                           |
| Conjunctive BKT    | No reference: no package has it, and the multi-skill form of `hmm-scalable` is compensatory.                                                                                                                                                                                                                                                                                                                                                                                |
| Individualized BKT | No reference: Yudelson et al. (2013) describe it, but no tool reaches the joint fit: `hmm-scalable` does not finish, and two `optim` starts end 62 nats apart. As written the model is not identified, since a constant moved between every offset and a skill's logit leaves the likelihood unchanged. On 200 seeded learners it does not beat standard BKT on held-out events. It can return with an L2 penalty, offsets of mean 0, and a joint reference that converges. |

## Differences from the references

The reference is pyBKT 1.4.3 ([validation](validation.md#bayesian-knowledge-tracing)).

| Topic                         | This package                                                                       | pyBKT                                                                                                                                                  |
| ----------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Prior groups                  | A group has its own P(L0) of each skill, and combines with item and resource rates | A group's prior is the learn rate of a phantom step before each learner's first attempt; `multiprior` combines with neither `multigs` nor `multilearn` |
| Resource rates and pairs      | `resourceLearning` and `pairLearning` refuse each other, as a rate has one key     | One resource handling at a time                                                                                                                        |
| Bounds                        | `bounds` clamps each rate that EM estimates, the exact maximum of each M-step      | None                                                                                                                                                   |
| Random starts with forgetting | P(F) from U(0, 0.2)                                                                | P(F) starts at 0, where EM keeps it                                                                                                                    |
| Guess and slip                | Kept 2⁻⁵² from 0 and 1, so that every outcome has a positive probability           | The ratio of the expected counts, which reaches 0 or 1                                                                                                 |

## References

- Baker, R. S. J. d., Corbett, A. T., & Gowda, S. M. (2008). More accurate student modeling through contextual
  estimation of slip and guess probabilities in Bayesian knowledge tracing. In Intelligent Tutoring Systems (ITS 2008).
- Beck, J. E., & Chang, K. (2007). Identifiability: A fundamental problem of student modeling. In User Modeling (UM 2007).
- Badrinath, A., Wang, F., & Pardos, Z. (2021). pyBKT: An accessible Python library of Bayesian knowledge tracing
  models. In Proceedings of the 14th International Conference on Educational Data Mining.
- Corbett, A. T., & Anderson, J. R. (1994). Knowledge tracing: Modeling the acquisition of procedural knowledge. User
  Modeling and User-Adapted Interaction, 4(4), 253-278.
- Pardos, Z. A., & Heffernan, N. T. (2010). Modeling individualization in a Bayesian networks implementation of
  knowledge tracing. In User Modeling, Adaptation, and Personalization (UMAP 2010).
- Pardos, Z. A., & Heffernan, N. T. (2011). KT-IDEM: Introducing item difficulty to the knowledge tracing model. In
  User Modeling, Adaption and Personalization (UMAP 2011).
