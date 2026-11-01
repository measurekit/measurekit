import { checkOptions, type Random } from "@measurekit/core";
import { checkFunction } from "../checks.js";
import { type AbilityEstimate, type AbilityEstimator, indexItems, posteriorMeanEstimate } from "../irt/internal.js";
import { type AdaptiveState, adaptiveEngine, checkRules } from "./engine.js";
import { checkResponse, type PoolItem } from "./items.js";
import { type ItemSelectionRule, maximumInformation } from "./selection.js";
import type { StoppingRule } from "./stopping.js";
import type { TestControlOptions } from "./test-controls.js";

/** Options of {@link adaptiveTest}. */
export interface AdaptiveTestOptions extends TestControlOptions {
  /** The item pool. */
  readonly pool: readonly PoolItem[];
  /** The rule that ends the test. The test also ends when no item is left or the selection rule scores every eligible item `-Infinity`. */
  readonly stopping: StoppingRule;
  /**
   * The rule that selects the next item.
   *
   * @defaultValue `maximumInformation`
   */
  readonly selection?: ItemSelectionRule;
  /**
   * Estimates theta after each response, and before the first, without responses.
   *
   * @defaultValue `posteriorMeanEstimate`
   */
  readonly estimator?: AbilityEstimator;
}

/** The state of an adaptive test: plain data, which JSON serializes, so that a test can resume from a stored state. */
export interface AdaptiveTestState extends AdaptiveState<AbilityEstimate> {}

/** An adaptive test: a pool with the rules that select its items, estimate theta and end the test. */
export interface AdaptiveTest {
  /** The item pool. */
  readonly pool: readonly PoolItem[];
  /**
   * Returns the state before the first response: the estimate without responses and the first item.
   *
   * @throws `DataError` when `random` is not a random number generator.
   */
  start(random: Random): AdaptiveTestState;
  /**
   * Returns the state after a response to the state's next item: the new estimate, and the next item unless the test
   * ends.
   *
   * @throws `DataError` when the test has ended, the state is malformed, names items the pool does not have or an item
   * twice, a response is not a category of its item, or `random` is not a random number generator.
   */
  answer(state: AdaptiveTestState, response: number, random: Random): AdaptiveTestState;
}

/**
 * Returns an adaptive test of a pool.
 *
 * @remarks
 * Before each item the test estimates theta from the responses so far and checks the stopping rule; unless the test
 * ends, content balancing restricts the eligible items to one content area, the selection rule selects among them, and
 * Sympson-Hetter exposure control administers or withholds the selected item. Items administered, withheld or enemies
 * of an administered item are not eligible. Items that share a `testlet` are administered together: see the guide.
 *
 * @param options - The pool, the stopping and selection rules, the estimator, and the content balancing, enemy sets and
 *   testlet order; each option has its default.
 * @returns The test: `start` gives the state before the first response, and `answer` the state after a response.
 * @throws `DataError` when `stopping`, `selection` or `estimator` is not a rule or function, an item's id or
 * parameters are invalid, ids repeat, an exposure parameter is outside [0, 1], a testlet is not a non-empty string, a
 * content target is invalid or missing for an item's area, an enemy set does not name two or more distinct pool items,
 * or `withinTestlet` is unknown.
 *
 * @example
 * ```ts
 * import { createRandom, twoParameterLogistic } from "@measurekit/psychometrics";
 * import { adaptiveTest, testLength } from "@measurekit/psychometrics/cat";
 *
 * const pool = [
 *   { id: "q1", model: twoParameterLogistic, parameters: { a: 1, b: -1, c: 0, d: 1 } },
 *   { id: "q2", model: twoParameterLogistic, parameters: { a: 1.2, b: 0, c: 0, d: 1 } },
 *   { id: "q3", model: twoParameterLogistic, parameters: { a: 0.8, b: 1, c: 0, d: 1 } },
 * ];
 * const test = adaptiveTest({ pool, stopping: testLength(2) });
 * const random = createRandom(1);
 * let state = test.start(random);
 * state.next; // "q2", the id of the first item
 * state = test.answer(state, 1, random); // the response to it
 * state.next; // "q3"; null when the test has ended
 * ```
 */
export function adaptiveTest(options: AdaptiveTestOptions): AdaptiveTest {
  checkOptions(options);
  const {
    pool,
    stopping,
    selection = maximumInformation,
    estimator = posteriorMeanEstimate,
    contentTargets,
    enemies,
    withinTestlet,
  } = options;
  checkRules(stopping, selection, "maximumInformation");
  checkFunction(
    estimator,
    "The option estimator must be a function of items and responses, such as posteriorMeanEstimate",
  );
  const engine = adaptiveEngine({
    pool,
    indices: indexItems(pool),
    stopping,
    selection,
    contentTargets,
    enemies,
    withinTestlet,
    estimate: estimator,
    checkResponse,
  });
  return { pool, ...engine };
}
