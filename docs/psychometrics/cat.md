# Adaptive and multistage testing

`@measurekit/psychometrics/cat` runs computerized adaptive tests (CAT) and multistage tests (MST) on items of any [IRT
model](irt.md), adaptive tests that classify a person's attribute profile with the items of a [cognitive diagnosis
model](cdm.md), simulates them, calibrates their exposure control and assembles fixed forms. Item selection, ability
estimates, content balancing and stopping decisions are validated against catR step by step; diagnostic tests against a
replay of every rule written out in base R and against cdcatR ([validation](validation.md)).

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

An `AbilityEstimator`, from `@measurekit/psychometrics/irt`, is a
function `(items, responses) => { theta, standardError }`: one of
`maximumLikelihoodEstimate`, `weightedLikelihoodEstimate`, `posteriorModeEstimate` and
`posteriorMeanEstimate` ([IRT](irt.md#ability-estimation)), or one with
options, `(items, responses) => posteriorModeEstimate(items, responses, { prior: { mean: 0, sd: 1.5 } })`.

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

### Rules for other items and estimates

Contexts, rules and states are generic in the type of the pool's items and of the estimate, and the diagnostic and
other adaptive tests run on the same engine as `adaptiveTest`.

| Type                                                | Parameters                                | Defaults                      |
| --------------------------------------------------- | ----------------------------------------- | ----------------------------- |
| `AdaptiveTestContext`, `SelectionContext`           | `TItem extends PoolControls`, `TEstimate` | `PoolItem`, `AbilityEstimate` |
| `ItemSelectionRule`, `StoppingRule`                 | `TItem extends PoolControls`, `TEstimate` | `PoolItem`, `AbilityEstimate` |
| `TestState`                                         | `TEstimate`                               | `AbilityEstimate`             |
| `AdaptiveState`: a state with `withheld` and `next` | `TEstimate`                               | none                          |

`PoolControls` is what every adaptive test needs of an item: `id`, `contentArea` and `exposure`. `PoolItem` adds the
model and parameters of an IRT item. The context of a rule has the fields of the [table above](#item-selection) whatever
the types: `estimate` is the estimate the test computes from the responses, an `AbilityEstimate` for `adaptiveTest`.

`testLength`, `allOf`, `anyOf` and `randomesque` take the types of the test they are used in, or of the rules they
combine, so one `testLength(20)` stops a test of any kind. `standardErrorThreshold` stops a test of any estimate with a
`standardError`, or a `standardErrors` for each of its dimensions, once every one is at most the threshold. A rule
written for one kind of test, such as `maximumInformation`, has the types of that test and does not compile in another.

## Diagnostic adaptive tests

`diagnosticAdaptiveTest(options)` classifies which attributes a person has mastered. Its estimate is the posterior over
the 2^K profiles of K attributes, the prior times the probability of each response in the item's latent group of the
profile, from the items of a [cognitive diagnosis calibration](cdm.md#calibration). The rules use only each item's
success probabilities by latent group, so items of any model, the G-DINA, DINA and DINO included, work. Like
`adaptiveTest`, it returns a test with `start` and `answer`, each returning a new state.

```ts
import { createRandom } from "@measurekit/psychometrics";
import {
  anyOf,
  diagnosticAdaptiveTest,
  posteriorThreshold,
  simulateDiagnosticTest,
  testLength,
} from "@measurekit/psychometrics/cat";
import { calibrateCdm } from "@measurekit/psychometrics/cdm";
import { simulateCdmResponses } from "@measurekit/psychometrics/sim";

// A bank of 12 items over three attributes, and 1,000 persons with random profiles who answer all of them.
const attributes = ["add", "borrow", "convert"];
const single = (attribute: string, j: number) => ({
  id: `${attribute}${j}`,
  attributes: [attribute],
  probabilities: [0.15, 0.85],
});
const both = (id: string, pair: string[]) => ({ id, attributes: pair, probabilities: [0.1, 0.35, 0.35, 0.9] });
const bank = [
  ...attributes.flatMap((attribute) => [1, 2, 3].map((j) => single(attribute, j))),
  both("addBorrow", ["add", "borrow"]),
  both("borrowConvert", ["borrow", "convert"]),
  both("addConvert", ["add", "convert"]),
];
const random = createRandom(11);
const profiles = Array.from({ length: 1000 }, () => attributes.map(() => (random.uniform() < 0.5 ? 1 : 0)));
const matrix = simulateCdmResponses({ attributes, items: bank }, profiles, random);

// Calibrate the bank, then test from the calibration: its items are the pool and its profiles the prior.
const fit = calibrateCdm(matrix, Object.fromEntries(bank.map(({ id, attributes }) => [id, attributes])));
const test = diagnosticAdaptiveTest({
  pool: fit.items,
  attributes: fit.attributes,
  prior: fit.profiles,
  stopping: anyOf(testLength(8), posteriorThreshold(0.85)),
});

// The first person answers the items the test selects.
const answers = Object.fromEntries(matrix.itemIds.map((id, j) => [id, matrix.responses[0]?.[j] ?? 0]));
let state = test.start(random);
state.next; // "convert2", the first item
while (state.next !== null) state = test.answer(state, answers[state.next] ?? 0, random);
state.items; // ["convert2", "add2", "borrow3", "borrowConvert", "add1", "add3", "addConvert"]: stops at 0.87 >= 0.85
state.estimate.mastery; // [0.936, 0.946, 0.983] to three digits; the true profile is [1, 1, 1]
state.estimate.map; // [1, 1, 1]

const result = simulateDiagnosticTest(test, profiles.slice(0, 500), random);
result.patternAccuracy; // 0.826
result.attributeAccuracy; // [0.938, 0.934, 0.946]
result.meanLength; // 6.726
```

| Option           | Default                    | Meaning                                                                                                  |
| ---------------- | -------------------------- | -------------------------------------------------------------------------------------------------------- |
| `pool`           | required                   | Items `{ id, attributes, probabilities }`, with an optional `contentArea` and `exposure`; see below.     |
| `attributes`     | required                   | From 1 to 20 distinct attribute names, in the order of the profiles. A calibration's `attributes`.       |
| `prior`          | uniform over the profiles  | `{ profile, proportion }` for each profile: a calibration's `profiles`. A [hierarchy](#prior) has zeros. |
| `stopping`       | required                   | A [stopping rule](#stopping-of-diagnostic-tests).                                                        |
| `selection`      | `gdinaDiscriminationIndex` | An [item selection rule](#selection-of-diagnostic-tests).                                                |
| `contentTargets` | none                       | As for [`adaptiveTest`](#content-balancing).                                                             |
| `enemies`        | none                       | As for [`adaptiveTest`](#enemy-sets).                                                                    |

The pool is an array of calibrated items; a calibration's `items` are one. `attributes` of an item lists the attributes
it measures, each among the test's, and `probabilities` the probability of a correct response in each of its latent
groups: the profiles over its attributes as listed, by the number of attributes mastered, then earlier attributes first
(`00`, `10`, `01`, `11` for two). An item of one attribute has 2 probabilities, of three 8. Each probability is from 0
to 1. Content balancing, enemy sets and exposure control work as for [`adaptiveTest`](#adaptive-tests), and so do the
steps before each item; the first item is selected at the prior. Responses are 0 or 1.

### Estimate

The state's `estimate` is a `ProfileEstimate`, the shape of an [`AttributeProfile`](cdm.md#persons) without an id.

| Field       | Meaning                                                                                                                                              |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `posterior` | The posterior probability of each profile, in the order of the profiles: `000`, `100`, `010`, `001`, `110`, `101`, `011`, `111` for three attributes |
| `mastery`   | The posterior probability of mastering each attribute                                                                                                |
| `map`       | The profile of largest posterior probability, the first in that order on ties                                                                        |
| `eap`       | Each attribute mastered when its posterior probability is above 0.5                                                                                  |

A response enters the likelihood with a probability of at least `Number.MIN_VALUE`, so that a success probability of 0
or 1, as a calibration from JSON may hold, does not make the posterior 0 for every profile. The state is plain data in
the same way as an adaptive test's: `items`, `responses`, `estimate`, `withheld` and `next`, which `answer` checks.

### Prior

`prior` lists every profile once, matched to the profiles of `attributes` by its pattern, in any order, with a
proportion from 0 to 1 that are not all 0; the test normalizes them to sum to 1. A proportion of 0 excludes the profile,
as a hierarchy among the attributes does: its posterior stays 0 and the MAP profile is never it. Pass a calibration's
`profiles`, or `[{ profile: [0, 0], proportion: 0.5 }, ...]` for a prior of your own.

### Selection of diagnostic tests

A rule scores every eligible item at the posterior π over the profiles, and the test selects the highest score. Each is
computed over the item's latent groups, with π_l the posterior probability of the profiles in group l, P_l the success
probability of the group, α̂ the MAP profile and KL(p, q) = p ln(p / q) + (1 - p) ln((1 - p) / (1 - q)).

| Rule                                       | Score                                                                  | Name  |
| ------------------------------------------ | ---------------------------------------------------------------------- | ----- |
| `gdinaDiscriminationIndex`                 | Σ_l π_l (P_l - P̄)², P̄ = Σ_l π_l P_l                                    | GDI   |
| `modifiedPosteriorWeightedKullbackLeibler` | Σ_l Σ_m π_l π_m KL(P_l, P_m)                                           | MPWKL |
| `posteriorWeightedKullbackLeibler`         | Σ_l π_l KL(P_α̂, P_l)                                                   | PWKL  |
| `diagnosticKullbackLeibler`                | Σ_l n_l KL(P_α̂, P_l), n_l the profiles in group l: unweighted          | KL    |
| `shannonEntropy`                           | minus the expected Shannon entropy of the posterior after the response | SHE   |

Kaplan, de la Torre and Barrada (2015) found GDI and MPWKL select best; GDI is the default, as it needs no divergence.
KL sums over every profile, also one the prior excludes; PWKL and the others weight by the posterior. SHE (Tatsuoka
and Ferguson, 2003) ranks items as the Jensen-Shannon divergence rule does, which is not included. The cost of a rule is
proportional to the eligible items times the profiles, 2^K.

The rules are `ItemSelectionRule<DiagnosticTestItem, ProfileEstimate>`; `randomesque(rule, count)` wraps any of them. A
custom rule reads, as for [IRT items](#item-selection), `context.pool`, `administered`, `responses`, `estimate`,
`eligible` and `random`. Its pool items are `DiagnosticTestItem`s: the item as given, plus `groups`, the index of the
item's latent group in each profile of the test, so that `item.probabilities[item.groups[c]]` is the success probability
at profile `c` of `estimate.posterior`.

### Stopping of diagnostic tests

| Rule                    | Stops when                                                                      |
| ----------------------- | ------------------------------------------------------------------------------- |
| `posteriorThreshold(p)` | the largest posterior probability of a profile is at least p, from above 0 to 1 |
| `masteryThreshold(p)`   | every attribute's mastery is at least p or at most 1 - p, p above 0.5 to 1      |

`testLength(count)`, `allOf` and `anyOf` combine with them: `anyOf(testLength(20), posteriorThreshold(0.8))` gives a
fixed-precision test of at most 20 items. A custom rule implements `StoppingRule<DiagnosticTestItem, ProfileEstimate>`.

### Simulation of diagnostic tests

`simulateDiagnosticTest(test, profiles, random)` administers the test to one person per true profile. Each person's
response to every pool item is drawn first, correct with the success probability of the latent group the profile falls
in, as [`simulateCdmResponses`](sim.md#responses) draws them; the same `random` then makes the test's random choices, so
a seed reproduces the result.

| Result              | Meaning                                                                         |
| ------------------- | ------------------------------------------------------------------------------- |
| `persons`           | `{ profile, state }` per true profile, in order: the final state of each        |
| `patternAccuracy`   | The share of persons whose `map` profile equals the true profile                |
| `attributeAccuracy` | For each attribute, the share of persons whose `map` profile agrees on it       |
| `meanLength`        | Mean number of items administered                                               |
| `exposureRates`     | `{ id, rate }` for every item of the pool: the share of persons administered it |

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

[`simulateDiagnosticTest`](#simulation-of-diagnostic-tests) simulates diagnostic tests. `simulateAdaptiveTest(test,
thetas, random)` and `simulateMultistageTest(test, thetas, random)` administer a test to one person per true theta,
drawing each response from the item's category probabilities, as [`simulateResponses`](sim.md#responses) does. The same
`random` draws the responses and the test's random choices, so a seed reproduces the result.

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

## Not included

- Diagnostic tests of polytomous items or attributes, and tests that mix IRT and diagnostic items.
- Exposure calibration for diagnostic tests: `calibrateExposure` takes IRT tests only; set `exposure` on the items.
- The dual rule on the two largest posteriors, the Jensen-Shannon divergence rule, which ranks items as `shannonEntropy`
  does, and the nonparametric selection rules NPS and GNPS.
- Multidimensional adaptive tests.

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

## Differences from cdcatR

| Topic            | This package                                                | cdcatR                                                                                |
| ---------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Rules            | GDI, MPWKL, PWKL, KL and SHE                                | GDI, MPWKL, PWKL, JSD, NPS, GNPS and random                                           |
| PWKL's estimate  | The MAP profile, the first on ties                          | `sample(which(posterior == max(posterior)), 1)`: for a single mode m, a draw from 1:m |
| First item       | The item of the highest score at the prior                  | `startRule = "random"` by default; `"max"` selects as this package does               |
| Prior            | One `prior` for the first item and the posterior            | `att.prior` for the posterior and `initial.distr` for the first item                  |
| Stopping         | Any `StoppingRule`; `posteriorThreshold` is `precision.cut` | A fixed length, or `precision.cut` on the largest posterior                           |
| Constraints      | Content balancing and enemy sets                            | A minimum number of items per attribute (`itemConstraint = "attribute"`)              |
| Exposure control | Sympson-Hetter, with `exposure` set by hand                 | The progressive method                                                                |
| Estimates        | `ProfileEstimate`: posterior, mastery, MAP and EAP profiles | ML, MAP and EAP profiles by item in `est.cat`                                         |
| Parallel runs    | One test at a time; a seed reproduces a simulation          | Persons in parallel on sockets, `seed + i` per person                                 |

## Implementation

- Selection rules write scores into a `Float64Array` with one entry per pool item; the built-ins loop over the pool
  through the model's methods, without allocating per item.
- Estimates come from the [IRT estimators](irt.md#ability-estimation), so CAT and MST support every model, a custom
  `IrtModel` included.
