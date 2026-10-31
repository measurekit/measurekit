# Data

Functions in `@measurekit/knowledge-tracing/data` build the logs that every model and evaluation takes, and split
them into folds. Logs are plain arrays of plain objects, so they can be built by hand or loaded from JSON;
every function that takes a log checks it.

## Events

An event is one attempt of a learner at an item, with its outcome.

| Field     | Meaning                                                                                                     |
| --------- | ----------------------------------------------------------------------------------------------------------- |
| `learner` | The learner's id: a non-empty string.                                                                       |
| `item`    | The item's id: a non-empty string.                                                                          |
| `skills`  | The ids of the skills (knowledge components) the item practices: distinct non-empty strings, possibly none. |
| `time`    | A finite number that orders the attempts, such as epoch milliseconds or a sequence number.                  |
| `outcome` | From 0 to 1: 1 correct, 0 incorrect, and between them partial credit, where a model supports it.            |

An attempt is an event without its outcome: what a model predicts from. Models take the attempt's skills and time, so
time-aware and skill-based models need nothing else.

## Logs

`eventLog(events, options)` checks and copies events, and orders them by time, keeping the given order of events with
equal times. The result is an `EventLog`: events in time order, for each learner and for all learners together, which
models that also update items, such as Elo, need. Every event counts as an attempt, so remove the repeated rows some
exports hold, as ASSISTments 2009 does: there, one row per skill of an event and copies of rows. Events with equal times
keep the order given, which another tool may not: order them by a unique time, such as a row number, to compare.

```ts
import { eventLog } from "@measurekit/knowledge-tracing/data";

const log = eventLog(
  [
    { learner: "budi", item: "q2", time: 1730190000000, outcome: 0 },
    { learner: "ana", item: "q1", time: 1730180000000, outcome: 1 },
    { learner: "ana", item: "q3", skills: ["decimals"], time: 1730200000000, outcome: 0.5 },
  ],
  { qMatrix: { q1: ["fractions"], q2: ["fractions", "decimals"] } },
);
log.map((event) => event.skills); // [["fractions"], ["fractions", "decimals"], ["decimals"]]
```

| Rule   | Detail                                                                                                       |
| ------ | ------------------------------------------------------------------------------------------------------------ |
| Size   | At least one event.                                                                                          |
| Skills | An event's own `skills`; without them, its item's row of the Q-matrix; without a Q-matrix, none.             |
| Order  | By `time`, stable. Functions that take a log reject one out of time order, which a log built by hand may be. |
| Copies | The events are copied, with only the fields above, so later changes to the input do not change the log.      |
| Parts  | Any subsequence of a log, such as `log.filter(...)`, is a log.                                               |

A malformed event throws `DataError` naming the first problem and the event's position in the input, e.g.
`Event 3 needs an outcome from 0 to 1, found 2.` `NaN`, infinite times, repeated skills, empty ids and values of
other types are rejected.

## Q-matrices

A `QMatrix` gives the skills of each item, `{ [item]: skills }`, for events that do not carry skills. It is the
`QMatrix` of `@measurekit/core`, which `@measurekit/psychometrics` shares; a row may be empty here, as an item may practice no skill,
where its cognitive diagnosis needs at least one attribute per item.

An event without skills whose item the Q-matrix lacks throws `DataError`:
`Event 7 has no skills, and the Q-matrix lacks its item "q9".` A skill no model has seen is not an error: models start
it from its initial value.

## Splits

Splits return folds `{ train, test }` of one log, for cross-validation without leakage ([evaluation](evaluate.md)).

| Function                             | Folds                                                                                       |
| ------------------------------------ | ------------------------------------------------------------------------------------------- |
| `splitByLearner(log, folds, random)` | Each learner's events are in one fold's test log and the other folds' training logs.        |
| `splitByItem(log, folds, random)`    | Each item's events are in one fold's test log: predictions of items the model has not seen. |
| `splitByTime(log, folds)`            | `folds + 1` consecutive blocks; fold k trains on blocks 1 to k and tests on block k + 1.    |

```ts
import { createRandom } from "@measurekit/knowledge-tracing";
import { splitByLearner, splitByTime } from "@measurekit/knowledge-tracing/data";

const byLearner = splitByLearner(log, 2, createRandom(42)); // [{ train, test }, { train, test }]
const byTime = splitByTime(log, 1); // [{ train: the earlier events, test: the later ones }]
```

- **By learner and by item.** Learners or items are dealt to folds in an order drawn from `random`, so folds differ in
  size by at most one learner or item, and the same seed gives the same folds. `folds` is at least 2 and at most the
  number of learners or items.
- **By time.** Blocks have about equal numbers of events. A boundary falls after the last event of its time, so a
  model never trains on an event at or after the time of one it predicts. `folds` is at least 1; one fold is a single
  split at the middle of the log. Too few distinct times for the blocks throw `DataError`.
