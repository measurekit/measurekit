# Elo

`@measurekit/knowledge-tracing/elo` estimates learners' abilities and items' difficulties with the Elo rating system as adapted to
education (Pelánek, 2016): each answer is a match between a learner and an item. It needs no fitting beyond a pass over
the data, updates after every answer, and tracks abilities that change. The updates are validated against the equations
written out in R, and a simulation checks that they recover the abilities and difficulties that generated the data
([validation](validation.md)).

## The models

A learner has an ability θ and an item a difficulty d, both 0 at first, and

P(correct) = logistic(θ - d).

After an event with outcome y, each estimate moves by its uncertainty times the error y - P:

θ ← θ + U(n_θ) (y - P), d ← d - U(n_d) (y - P), U(n) = a / (1 + b n),

where n counts the estimate's earlier updates, so updates are large for new learners and items and shrink as evidence
accumulates. b = 0 gives the classic constant K = a. Outcomes between 0 and 1 count as partial credit.

**Concept-specific Elo.** With `skills: true`, a learner also has an ability θ_k in each skill k, 0 until practiced,
and

P(correct) = logistic(θ + mean θ_k - d)

over the item's skills; after the event, each of those θ_k ← θ_k + γ U(n_k) (y - P), with γ = `skillWeight`. For items
with one skill this is the hierarchical model of Nižnan, Pelánek and Řihák (2015). For an item with several skills,
the mean makes an update move the prediction as much as for an item with one skill. Events without skills update only
θ and d.

| Option        | Default             | Meaning                                                                        |
| ------------- | ------------------- | ------------------------------------------------------------------------------ |
| `uncertainty` | `{ a: 1, b: 0.05 }` | The uncertainty function U(n) = a / (1 + b n); a > 0, b ≥ 0                    |
| `skills`      | `false`             | Whether learners have skill abilities: concept-specific Elo                    |
| `skillWeight` | `1`                 | γ, the size of skill updates relative to global ones; only with `skills: true` |
| `parameters`  | none                | Item ratings to start from, such as a fit's `model.parameters`                 |

The defaults of U are those Pelánek (2016) recommends as a starting point. To tune a, b or γ, compare candidates by
[cross-validation](evaluate.md#model-comparison).

## Batch: replaying a log

`fit(log)` replays a log in time order from the model's item ratings, every learner from 0, updating the learner and
the item after each event as online use would.

```ts
import { eventLog } from "@measurekit/knowledge-tracing/data";
import { elo } from "@measurekit/knowledge-tracing/elo";

const log = eventLog(
  [
    { learner: "ana", item: "q1", time: 1, outcome: 1 },
    { learner: "budi", item: "q2", time: 2, outcome: 0 },
    { learner: "ana", item: "q2", time: 3, outcome: 0.5 },
    { learner: "budi", item: "q1", time: 4, outcome: 1 },
  ],
  { qMatrix: { q1: ["fractions"], q2: ["fractions", "decimals"] } },
);
const fit = elo({ skills: true }).fit(log);
fit.model.parameters.items; // { q1: { difficulty: -1.09, count: 2 }, q2: { difficulty: 0.56, count: 2 } }
fit.states.ana; // { learner: "ana", ability: 0.44, count: 2, skills: { fractions, decimals } }
fit.predictions; // each event's prediction before its outcome, as online use would have made it
```

The fit's `logLikelihood` sums the log-likelihoods of these predictions; `aic` and `bic` are `null`, as Elo estimates
no fixed number of parameters; `converged` is `true` and `iterations` 1, a single pass.

## Online

An app keeps each learner's state and each item's rating as a row, and an answer needs only the two rows it touches:

```ts
// The rows as stored, here from the fit above.
const rows = JSON.parse(JSON.stringify({ q1: fit.model.parameters.items.q1, ana: fit.states.ana }));
const model = elo({ skills: true, parameters: { items: { q1: rows.q1 } } });
const attempt = { learner: "ana", item: "q1", skills: ["fractions"], time: 5 };
model.predict(rows.ana, attempt); // 0.88, P(correct), for choosing items
const event = { ...attempt, outcome: 0 };
const ana = model.update(rows.ana, event); // ana's new state: write it to the learner's row
const q1 = model.updateItem(rows.ana, event); // q1's new rating, { difficulty, count }: write it to its row
```

- `update(state, event)` returns the learner's new state, with the model's item ratings fixed.
- `updateItem(state, event)` returns the new rating of the event's item, from the learner's state before the event.
  With `update` on the same state and event, it is one step of a replay: running both after every event gives exactly
  `fit`'s results.
- Neither changes the state or the model it is given. A model built from the rows of the items at hand costs as little
  as the rows; [storage](storage.md) describes the whole setup.

Item updates shrink as an item's answers accumulate, so updating items online is optional: a system can instead fit
periodically and use `update` alone in between.

### State

| Field     | Meaning                                                                                        |
| --------- | ---------------------------------------------------------------------------------------------- |
| `learner` | The learner's id; `update` and `predict` reject an attempt of another learner                  |
| `ability` | θ, on the logit scale                                                                          |
| `count`   | n_θ, the events that have updated θ                                                            |
| `skills`  | `{ [skill]: { ability, count } }`, θ_k and n_k of each skill practiced; empty without `skills` |

The state is plain data: store it as JSON and pass it back. `update` and `predict` check it and throw `DataError` for
a malformed state, naming the field. Item ratings `{ difficulty, count }` are plain data alike.

## Mastery

Elo defines no mastery, so the model has no `mastery` member. `predict` gives the probability of answering a given
item, which practice decisions can use directly.

## References

- Nižnan, J., Pelánek, R., & Řihák, J. (2015). Student models for prior knowledge estimation. In Proceedings of the
  8th International Conference on Educational Data Mining (pp. 109-116).
- Pelánek, R. (2016). Applications of the Elo rating system in adaptive educational systems. Computers & Education, 98,
  169-179.
