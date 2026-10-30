# Adaptive and multistage testing

`@measurekit/psychometrics/cat` runs computerized adaptive tests (CAT) and multistage tests (MST) on items of any [IRT
model](irt.md), simulates them, calibrates their exposure control and assembles fixed forms. Item selection, ability
estimates, content balancing and stopping decisions are validated against catR step by step
([validation](validation.md)).

## Adaptive tests

`adaptiveTest(options)` returns a test with two functions, `start` and `answer`. Each returns a new state and leaves
the one it was given unchanged.

```ts
import { createRandom } from "@measurekit/psychometrics";
import {
  adaptiveTest,
  allOf,
  anyOf,
  kullbackLeibler,
  standardErrorThreshold,
  testLength,
  untilMixed,
} from "@measurekit/psychometrics/cat";
import {
  generalizedPartialCredit,
  posteriorMeanEstimate,
  threeParameterLogistic,
  weightedLikelihoodEstimate,
} from "@measurekit/psychometrics/irt";

const pool = [
  { id: "q1", model: threeParameterLogistic, parameters: { a: 1.2, b: -1, c: 0.2, d: 1 }, contentArea: "algebra" },
  { id: "q2", model: threeParameterLogistic, parameters: { a: 0.9, b: -0.2, c: 0.15, d: 1 }, contentArea: "algebra" },
  { id: "q3", model: threeParameterLogistic, parameters: { a: 1.5, b: 0.4, c: 0.2, d: 1 }, contentArea: "algebra" },
  { id: "q4", model: threeParameterLogistic, parameters: { a: 1.1, b: 1.2, c: 0.25, d: 1 }, contentArea: "geometry" },
  { id: "q5", model: generalizedPartialCredit, parameters: { a: 1, b: [-0.8, 0.5] }, contentArea: "geometry" },
  { id: "q6", model: generalizedPartialCredit, parameters: { a: 1.3, b: [-0.2, 0.9, 1.6] }, contentArea: "geometry" },
];
const test = adaptiveTest({
  pool,
  selection: kullbackLeibler,
  estimator: untilMixed(posteriorMeanEstimate, weightedLikelihoodEstimate),
  stopping: anyOf(testLength(5), allOf(testLength(3), standardErrorThreshold(0.5))),
  contentTargets: { algebra: 0.6, geometry: 0.4 },
});

const answers: Record<string, number> = { q1: 1, q2: 1, q3: 0, q4: 0, q5: 2, q6: 1 };
const random = createRandom(42);
let state = test.start(random); // { items: [], responses: [], withheld: [], estimate, next: "q6" }
while (state.next !== null) state = test.answer(state, answers[state.next] ?? 0, random);
state.estimate; // { theta, standardError }
```

| Option           | Default                 | Meaning                                                                                             |
| ---------------- | ----------------------- | --------------------------------------------------------------------------------------------------- |
| `pool`           | required                | Items `{ id, model, parameters }`, with an optional `contentArea` and `exposure`.                   |
| `stopping`       | required                | A [stopping rule](#stopping-rules).                                                                 |
| `selection`      | `maximumInformation`    | An [item selection rule](#item-selection).                                                          |
| `estimator`      | `posteriorMeanEstimate` | An [estimator](#estimators).                                                                        |
| `contentTargets` | none                    | Target proportions by content area, for [content balancing](#content-balancing). Need not sum to 1. |
| `enemies`        | none                    | [Enemy sets](#enemy-sets): arrays of ids of items that must not appear in one test.                 |

A calibration's `items` are a valid pool. Ids must be unique. Before each item, the first included, the test:

1. estimates theta from the responses so far with `estimator`;
2. ends, with `next: null`, when `stopping` holds or no item is left: an item is left unless administered, withheld or
   an enemy of an administered item;
3. restricts the eligible items to one content area, with `contentTargets`;
4. scores the eligible items with `selection` and selects the highest, the first in pool order on ties;
5. administers it with probability `exposure`, or withholds it for the rest of the test and selects again.

### State

| Field       | Meaning                                                               |
| ----------- | --------------------------------------------------------------------- |
| `items`     | Ids of the administered items, in order                               |
| `responses` | The responses to them                                                 |
| `estimate`  | `{ theta, standardError }` from the responses                         |
| `withheld`  | Ids of items exposure control withheld; they are not administered     |
| `next`      | The id of the item to administer next, `null` when the test has ended |

The state is plain data. A server stores it as JSON between requests and passes it back to `answer`, which checks
it: its fields, its ids, that no item is administered or withheld twice, that no two administered items are enemies,
and every response. Randomness comes only from the `Random` passed to each call, so a generator seeded per
person and call reproduces a test.

### Estimators

An `AbilityEstimator`, from `@measurekit/psychometrics/irt`, is a function `(items, responses) => { theta, standardError }`:
one of `maximumLikelihoodEstimate`, `weightedLikelihoodEstimate`, `posteriorModeEstimate` and `posteriorMeanEstimate`
([IRT](irt.md#ability-estimation)), or one with options, `(items, responses) => posteriorModeEstimate(items, responses,
{ prior: { mean: 0, sd: 1.5 } })`.

`untilMixed(initial, mixed)` uses `initial` until one response is above its item's lowest category and one below its
highest, and `mixed` after: before that, the ML estimate is an end of the theta range. The test estimates before the
first response too, where ML and WLE throw, so they need `untilMixed`.

### Item selection

A rule scores every eligible item; the test selects the highest score.

| Rule                       | Score                                                                                  | catR                            |
| -------------------------- | -------------------------------------------------------------------------------------- | ------------------------------- |
| `maximumInformation`       | Fisher information at θ̂                                                                | `MFI`                           |
| `kullbackLeibler`          | ∫ L(θ) Σ_k P_k(θ̂) ln(P_k(θ̂) / P_k(θ)) dθ, L the likelihood of the responses so far     | `KL`                            |
| `aStratified(lengths)`     | -\|θ_max - θ̂\| in the scheduled stratum, θ_max where the item's information is largest | `thOpt` in `aStratified` strata |
| `randomesque(rule, count)` | A uniform draw for the `count` items `rule` scores highest, `-Infinity` for the others | the `randomesque` count         |

Every rule works for items of any model, through the model's methods.

- **KL** integrates by the trapezoid rule on 33 points from -4 to 4, catR's default `parInt`, with L scaled to a
  maximum of 1 on them.
- **a-stratified** (Chang and Ying, 1999) splits the pool into `lengths.length` strata of equal size by each item's
  maximum information: the first `lengths[0]` items come from the least informative stratum, the next `lengths[1]`
  from the next, and the last stratum after them. For the 1PL and 2PL, the maximum information is a² / 4 at θ = b, so
  the strata are Chang and Ying's by a and the match is on b; for other models this is maximum information
  stratification (Barrada et al., 2006). A stratum without eligible items passes to the next one, after the last to
  the nearest earlier one. Peaks are searched from -6 to 6. Strata are computed once per pool array, so a pool must
  not change after its first use.
- **Randomesque** (Kingsbury and Zara, 1989) draws from `context.random`. It wraps any rule:
  `randomesque(aStratified([5, 5, 10]), 3)`.

A custom rule implements `ItemSelectionRule`. `scores(context, out)` writes a score to `out[i]` for each eligible
item `i`; `-Infinity` excludes an item. The test throws `DataError` when a rule excludes every eligible item.

| Context field  | Meaning                                              |
| -------------- | ---------------------------------------------------- |
| `pool`         | The pool                                             |
| `administered` | Pool indices of the administered items, in order     |
| `responses`    | The responses to them                                |
| `estimate`     | The current `{ theta, standardError }`               |
| `eligible`     | A `Uint8Array`, 1 for each item that may be selected |
| `random`       | The `Random` of the call                             |

```ts
import type { ItemSelectionRule } from "@measurekit/psychometrics/cat";

/** catR's random criterion: any eligible item, with equal probability. */
export const randomSelection: ItemSelectionRule = {
  scores({ eligible, random }, out) {
    eligible.forEach((flag, i) => {
      if (flag === 1) out[i] = random.uniform();
    });
  },
};
```

### Content balancing

With `contentTargets`, every pool item needs one of their areas. Before each item, Kingsbury and Zara's rule picks an
area, as catR's `cbControl` does: an area without administered items, at random among several; otherwise the area
whose administered proportion falls furthest below its target, at random among ties. Only areas with eligible items
are picked. Selection then considers that area's items only.

### Enemy sets

`enemies` lists sets of item ids that must not appear in one test, such as items that give away each other's answers or
share a stimulus. Once an item of a set is administered, the other items of the set are not eligible. An item may be in
several sets. A withheld item does not exclude its enemies, as it was not administered.

### Exposure control

An item with an `exposure` parameter k in [0, 1] is subject to Sympson-Hetter control (Sympson and Hetter, 1985):
when selected, it is administered with probability k, otherwise withheld for the rest of the test, and the rule
selects again. Items without `exposure`, or with 1, take no random draw. The last eligible item is never withheld.

`calibrateExposure(test, thetas, random, options)` sets the exposure parameters by repeated simulation (Sympson and
Hetter, 1985). Every item starts at k = 1. Each round simulates `test`, the options of `adaptiveTest`, for persons with
the given thetas; with r the largest exposure rate allowed and P(S) the share of persons who had the item selected,
administered or withheld, it sets k = r / P(S) where P(S) > r and k = 1 elsewhere, and simulates again until no
exposure rate exceeds r by more than `tolerance`.

| Option          | Default  | Meaning                                                     |
| --------------- | -------- | ----------------------------------------------------------- |
| `maxRate`       | required | r, the largest exposure rate allowed, above 0 and at most 1 |
| `tolerance`     | `0.02`   | How far a rate may exceed r at convergence                  |
| `maxIterations` | `50`     | The most simulations                                        |

The result holds the `pool` with each item's `exposure`, ready for `adaptiveTest`, the `exposureRates` of the last
simulation, `converged` and `iterations`. Rates vary between simulations by √(r (1 - r) / n) for n thetas, and the
largest of many rates near r exceeds it by about 2.5 of these: with 2,000 thetas and r = 0.25, by 0.024. Take enough
thetas for the tolerance, or a tolerance for the thetas. When the test length over the pool size exceeds r, no
parameters meet r, and the result reports `converged: false`.

```ts
import { createRandom } from "@measurekit/psychometrics";
import { adaptiveTest, calibrateExposure, testLength } from "@measurekit/psychometrics/cat";
import { twoParameterLogistic } from "@measurekit/psychometrics/irt";
import { simulateItems, simulateThetas } from "@measurekit/psychometrics/sim";

const random = createRandom(3);
const pool = simulateItems(twoParameterLogistic, 100, random).map((item, i) => ({ ...item, id: `q${i + 1}` }));
const options = { pool, stopping: testLength(10) };
const { pool: controlled, converged } = calibrateExposure(options, simulateThetas(3000, random), random, {
  maxRate: 0.25,
});
const test = adaptiveTest({ ...options, pool: controlled });
```

### Stopping rules

The test checks its stopping rule before every item, the first included, and also ends when no item is left.

| Rule                                | Stops when                                              | catR                      |
| ----------------------------------- | ------------------------------------------------------- | ------------------------- |
| `testLength(count)`                 | `count` items have been administered                    | `length`                  |
| `standardErrorThreshold(threshold)` | the standard error is at most `threshold`               | `precision`               |
| `informationThreshold(threshold)`   | no eligible item has information above `threshold` at θ̂ | `minInfo`                 |
| `allOf(...rules)`                   | every rule stops                                        |                           |
| `anyOf(...rules)`                   | any rule stops                                          | `stop` with several rules |

`testLength` is a maximum length alone or in `anyOf`, and a minimum length in `allOf`:
`anyOf(testLength(30), allOf(testLength(10), standardErrorThreshold(0.3)))` gives 10 to 30 items. A custom rule
implements `StoppingRule`: `stops(context)` with the context of [item selection](#item-selection) without `random`,
`eligible` marking the items not administered or withheld.

## Multistage tests

`multistageTest(options)` administers a panel: stages of modules, one module per stage. Every person takes the single
module of the first stage; the module's cut points then route to a module of the next stage.

```ts
import { multistageTest } from "@measurekit/psychometrics/cat";
import { twoParameterLogistic } from "@measurekit/psychometrics/irt";

const module = (id: string, difficulties: number[], cuts?: number[]) => ({
  id,
  items: difficulties.map((b, j) => ({
    id: `${id}${j + 1}`,
    model: twoParameterLogistic,
    parameters: { a: 1, b, c: 0, d: 1 },
  })),
  ...(cuts && { cuts }),
});
const test = multistageTest({
  stages: [
    [module("routing", [-1, -0.3, 0.3, 1], [2, 3])],
    [module("easy", [-1.5, -1, -0.5]), module("medium", [-0.5, 0, 0.5]), module("hard", [0.5, 1, 1.5])],
  ],
  routing: "number-correct",
});

let state = test.start(); // { modules: [], items: [], responses: [], estimate, next: "routing" }
state = test.answer(state, [1, 1, 0, 0]); // score 2, from 2 to below 3: next is "medium"
state = test.answer(state, [1, 0, 1]); // the last stage: next is null
state.estimate;
```

| Option      | Default                 | Meaning                                                          |
| ----------- | ----------------------- | ---------------------------------------------------------------- |
| `stages`    | required                | Stages in order, each a list of modules; the first has one       |
| `routing`   | required                | `"number-correct"` or `"theta"`                                  |
| `estimator` | `posteriorMeanEstimate` | An [estimator](#estimators), used after each module and at start |

Modules are `{ id, items, cuts }`. Module and item ids are unique in the panel.

- **Cuts** of a module route to the next stage: one fewer than its modules, ascending. A person goes to module j,
  counting from 0, when exactly j cuts are at most the routing value, so a value at a cut goes up. Modules of the last
  stage have none. Equal cuts make a module unreachable from this one.
- **Routing values.** `number-correct`: the total score of all responses so far, the number correct for dichotomous
  items. `theta`: the estimate from them.

`answer(state, responses)` takes the responses to the items of `state.next`, in order. The state holds `modules`, the
ids of the administered modules, with `items`, `responses`, `estimate` and `next` as in an adaptive test, and is plain
data in the same way; `answer` also checks that it takes one module per stage, in order, with their items.

## Simulation

`simulateAdaptiveTest(test, thetas, random)` and `simulateMultistageTest(test, thetas, random)` administer a test to
one person per true theta, drawing each response from the item's category probabilities, as
[`simulateResponses`](sim.md#responses) does. The same `random` draws the responses and the test's random choices, so
a seed reproduces the result.

```ts
import { createRandom } from "@measurekit/psychometrics";
import {
  adaptiveTest,
  anyOf,
  simulateAdaptiveTest,
  standardErrorThreshold,
  testLength,
} from "@measurekit/psychometrics/cat";
import { twoParameterLogistic } from "@measurekit/psychometrics/irt";
import { simulateItems, simulateThetas } from "@measurekit/psychometrics/sim";

const random = createRandom(7);
const pool = simulateItems(twoParameterLogistic, 50, random).map((item, i) => ({ ...item, id: `q${i + 1}` }));
const result = simulateAdaptiveTest(
  adaptiveTest({ pool, stopping: anyOf(testLength(20), standardErrorThreshold(0.4)) }),
  simulateThetas(500, random),
  random,
);
result.rootMeanSquareError; // also bias, meanLength, exposureRates and persons
```

| Result                | Meaning                                                                                  |
| --------------------- | ---------------------------------------------------------------------------------------- |
| `persons`             | `{ theta, state }` per true theta, in order: the final state of each                     |
| `bias`                | Mean of the estimate minus the true theta                                                |
| `rootMeanSquareError` | √(mean of the squared differences)                                                       |
| `meanLength`          | Mean number of items administered                                                        |
| `exposureRates`       | `{ id, rate }` for every item of the pool or panel: the share of persons administered it |

## Form assembly

`assembleForm({ pool, length, content, target })` assembles a fixed form, such as a multistage module or a parallel
form, by Swanson and Stocking's (1993) weighted deviations method, a heuristic that needs no solver. It selects the
items one at a time; for each candidate it projects every property of the finished form, as the value of the items
selected, plus the candidate's, plus the mean of the remaining items times the places left, and selects the candidate
with the smallest weighted sum of deviations of the projections from their bounds.

| Option    | Default  | Meaning                                                                                                |
| --------- | -------- | ------------------------------------------------------------------------------------------------------ |
| `pool`    | required | Items with ids and `contentArea`s, as for an adaptive test                                             |
| `length`  | required | The number of items, from 1 to the pool size                                                           |
| `content` | none     | `{ area, minimum, maximum, weight }`: bounds on the number of items of an area; `weight` defaults to 1 |
| `target`  | none     | `{ theta, information, weight }`: the test information wanted at a theta; `weight` defaults to 1       |

The result holds the `items` in the order of selection, the `contentCounts` of the constrained areas and the test
`information` at each target theta, to check the form against its specification. A count deviates by items and
information by its own units: weight the targets, such as 1 / information, to balance them. The method meets feasible
constraints in practice but does not prove optimality or feasibility as integer programming does; the tests check the
constraints and the information of assembled forms rather than a reference solution.

```ts
import { createRandom } from "@measurekit/psychometrics";
import { assembleForm } from "@measurekit/psychometrics/cat";
import { twoParameterLogistic } from "@measurekit/psychometrics/irt";
import { simulateItems } from "@measurekit/psychometrics/sim";

const areas = ["algebra", "geometry", "number"];
const pool = simulateItems(twoParameterLogistic, 300, createRandom(4)).map((item, i) => ({
  ...item,
  id: `q${i + 1}`,
  contentArea: areas[i % 3],
}));
const form = assembleForm({
  pool,
  length: 30,
  content: [
    { area: "algebra", minimum: 12, maximum: 14 },
    { area: "geometry", minimum: 8, maximum: 10 },
  ],
  target: [
    { theta: -1, information: 6 },
    { theta: 0, information: 8 },
    { theta: 1, information: 6 },
  ],
});
form.contentCounts; // the number of items of each constrained area, within its bounds
```

## Differences from catR

| Topic             | This package                                                               | catR                                                                     |
| ----------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| ML, MAP, WLE      | Range [-6, 6]; the largest of several maxima; roots to 1e-12               | `range = c(-4, 4)`; one root by `uniroot`, tolerance about 1e-4          |
| EAP               | 61 points from -6 to 6 prior SDs, weights proportional to the prior        | Trapezoid rule on `parInt`, by default 33 points from -4 to 4            |
| Constant patterns | `untilMixed`                                                               | `constantPatt`, a fixed step or another estimator                        |
| Ties              | The first item in pool order                                               | At random                                                                |
| KL                | The likelihood scaled to a maximum of 1                                    | The likelihood as is: the same order                                     |
| a-stratification  | Strata in turn from the least informative, by maximum information; `thOpt` | `randomCAT(a.stratified)`: strata by a, balanced as content areas        |
| Content balancing | Only areas with eligible items                                             | Fails when the area picked has no item left                              |
| Exposure control  | Sympson-Hetter, calibrated by `calibrateExposure`                          | `simulateRespondents(rmax)`: the restricted and item eligibility methods |
| `minInfo`         | Over items not administered or withheld                                    | Over items not administered                                              |

## Implementation

- Selection rules write scores into a `Float64Array` with one entry per pool item; the built-ins loop over the pool
  through the model's methods, without allocating per item.
- Estimates come from the [IRT estimators](irt.md#ability-estimation), so CAT and MST support every model, a custom
  `IrtModel` included.
