import { checkOptions, type FieldNames } from "@measurekit/core";
import { checkFunction } from "../checks.js";
import { type AbilityEstimator, indexItems, posteriorMeanEstimate } from "../irt/internal.js";
import { type AdaptiveTest, adaptiveEngine, checkRules } from "./engine.js";
import { checkResponse, type PoolItem } from "./items.js";
import { type ItemSelectionRule, maximumInformation } from "./selection.js";
import type { StoppingRule } from "./stopping.js";
import type { TestControlOptions } from "./test-controls.js";

/** Options of {@link createAdaptiveTest}. */
export interface AdaptiveTestOptions extends TestControlOptions {
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

const ADAPTIVE_TEST_OPTIONS: FieldNames<AdaptiveTestOptions> = {
  selection: true,
  estimator: true,
  contentTargets: true,
  enemies: true,
  withinTestlet: true,
};

/**
 * Returns an adaptive test of a pool.
 *
 * @remarks
 * Before each item the test estimates theta from the responses so far and checks the stopping rule; unless the test
 * ends, content balancing restricts the eligible items to one content area, the selection rule selects among them, and
 * Sympson-Hetter exposure control administers or withholds the selected item. Items administered, withheld or enemies
 * of an administered item are not eligible. Items that share a `testlet` are administered together: see the guide.
 *
 * @param pool - The item pool.
 * @param stopping - The rule that ends the test. The test also ends when no item is left or the selection rule scores
 *   every eligible item `-Infinity`.
 * @param options - The selection rule, the estimator, and the content balancing, enemy sets and testlet order; each
 *   option has its default.
 * @returns The test: `start` gives the state before the first response, and `answer` the state after a response.
 * @throws `DataError` when `stopping`, `selection` or `estimator` is not a rule or function, an item's id or
 * parameters are invalid, ids repeat, an exposure parameter is outside [0, 1], a testlet is not a non-empty string, a
 * content target is invalid or missing for an item's area, an enemy set does not name two or more distinct pool items,
 * or `withinTestlet` is unknown.
 *
 * @example
 * ```ts
 * import { createRandom, twoParameterLogistic } from "@measurekit/psychometrics";
 * import { createAdaptiveTest, testLength } from "@measurekit/psychometrics/cat";
 *
 * const pool = [
 *   { id: "q1", model: twoParameterLogistic, parameters: { a: 1, b: -1, c: 0, d: 1 } },
 *   { id: "q2", model: twoParameterLogistic, parameters: { a: 1.2, b: 0, c: 0, d: 1 } },
 *   { id: "q3", model: twoParameterLogistic, parameters: { a: 0.8, b: 1, c: 0, d: 1 } },
 * ];
 * const test = createAdaptiveTest(pool, testLength(2));
 * const random = createRandom(1);
 * let state = test.start(random);
 * state.next; // => "q2", the id of the first item
 * state = test.answer(state, 1, random); // the response to it
 * state.next; // => "q3"; null when the test has ended
 * ```
 */
export function createAdaptiveTest(
  pool: readonly PoolItem[],
  stopping: StoppingRule,
  options: AdaptiveTestOptions = {},
): AdaptiveTest {
  checkOptions(options, ADAPTIVE_TEST_OPTIONS);
  const {
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
