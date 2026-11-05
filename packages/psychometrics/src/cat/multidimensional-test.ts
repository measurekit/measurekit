import { checkOptions, type FieldNames } from "@measurekit/core";
import { checkFunction } from "../checks.js";
import {
  checkDimensions,
  checkMultidimensionalItems,
  indexMultidimensionalItems,
  type MultidimensionalEstimate,
  type MultidimensionalEstimator,
  multidimensionalPosteriorModeEstimate,
} from "../multidimensional/internal.js";
import { type AdaptiveTest, adaptiveEngine, checkRules } from "./engine.js";
import { checkCategory } from "./items.js";
import { maximumDeterminant } from "./maximum-determinant.js";
import type { MultidimensionalPoolItem } from "./multidimensional-items.js";
import type { ItemSelectionRule } from "./selection.js";
import type { StoppingRule } from "./stopping.js";
import type { TestControlOptions } from "./test-controls.js";

/** Options of {@link createMultidimensionalAdaptiveTest}. */
export interface MultidimensionalAdaptiveTestOptions extends TestControlOptions {
  /**
   * The rule that selects the next item.
   *
   * @defaultValue `maximumDeterminant()`
   */
  readonly selection?: ItemSelectionRule<MultidimensionalPoolItem, MultidimensionalEstimate>;
  /**
   * Estimates the thetas after each response, and before the first, without items or responses: it must take a prior,
   * which gives the dimensions then, as the built-in estimators do with `prior`.
   *
   * @defaultValue `multidimensionalPosteriorModeEstimate`, under a prior of N(0, I) of the pool's dimensions
   */
  readonly estimator?: MultidimensionalEstimator;
}

const MULTIDIMENSIONAL_ADAPTIVE_TEST_OPTIONS: FieldNames<MultidimensionalAdaptiveTestOptions> = {
  selection: true,
  estimator: true,
  contentTargets: true,
  enemies: true,
  withinTestlet: true,
};

/**
 * Returns an adaptive test of a pool of multidimensional items, which estimates a vector of thetas.
 *
 * @remarks
 * Before each item the test estimates the thetas from the responses so far and checks the stopping rule; unless the
 * test ends, content balancing restricts the eligible items to one content area, the selection rule selects among
 * them, and Sympson-Hetter exposure control administers or withholds the selected item. Items administered, withheld or
 * enemies of an administered item are not eligible. The estimator is called with the administered items and their
 * responses, and before the first response with neither: the built-in estimators then answer with the prior, which
 * gives them the dimensions. The default estimate and rule use a prior of N(0, I); give the same prior to both when the
 * dimensions correlate, such as a calibration's `correlations`.
 *
 * @param pool - The item pool: multidimensional items with ids, all on the same number of dimensions, such as a
 *   calibration's.
 * @param stopping - The rule that ends the test. The test also ends when no item is left or the selection rule scores
 *   every eligible item `-Infinity`.
 * @param options - The selection rule, the estimator, and the content balancing, enemy sets and testlet order; each
 *   option has its default.
 * @returns The test: `start` gives the state before the first response, and `answer` the state after a response.
 * @throws `DataError` when `stopping`, `selection` or `estimator` is not a rule or function, the pool is not an array
 * of at least one valid item with a string id, ids repeat, the items differ in their number of dimensions or have more
 * than 100, an exposure parameter is outside [0, 1], a content target is invalid or missing for an item's area, or an
 * enemy set does not name two or more distinct pool items. An estimator that takes no prior throws, from `start`, as it
 * cannot estimate without items.
 *
 * @example
 * ```ts
 * import { createRandom } from "@measurekit/psychometrics";
 * import { createMultidimensionalAdaptiveTest, testLength } from "@measurekit/psychometrics/cat";
 * import { multidimensionalTwoParameterLogistic } from "@measurekit/psychometrics/multidimensional";
 *
 * const item = (id: string, a: number[], intercept: number) => ({
 *   id,
 *   model: multidimensionalTwoParameterLogistic,
 *   parameters: { a, intercept, c: 0, d: 1 },
 * });
 * const pool = [item("q1", [1.2, 0], 0.3), item("q2", [0, 1.4], -0.2), item("q3", [0.8, 0.9], 0.1)];
 * const test = createMultidimensionalAdaptiveTest(pool, testLength(2));
 * const random = createRandom(1);
 * let state = test.start(random);
 * state.next; // => "q2", the id of the first item
 * state = test.answer(state, 1, random); // the response to it
 * state.estimate.theta; // => [0, 0.52], the estimate of each dimension
 * state.next; // => "q1"
 * ```
 */
export function createMultidimensionalAdaptiveTest(
  pool: readonly MultidimensionalPoolItem[],
  stopping: StoppingRule<MultidimensionalPoolItem, MultidimensionalEstimate>,
  options: MultidimensionalAdaptiveTestOptions = {},
): AdaptiveTest<MultidimensionalPoolItem, MultidimensionalEstimate> {
  checkOptions(options, MULTIDIMENSIONAL_ADAPTIVE_TEST_OPTIONS);
  const { selection = maximumDeterminant(), estimator, contentTargets, enemies, withinTestlet } = options;
  checkRules(stopping, selection, "maximumDeterminant()");
  if (estimator !== undefined) {
    checkFunction(
      estimator,
      "The option estimator must be a function of items and responses, such as multidimensionalPosteriorModeEstimate",
    );
  }
  const dimensions = checkMultidimensionalItems(pool, "A multidimensional adaptive test");
  checkDimensions(dimensions, "A multidimensional adaptive test");
  const standardPrior = {
    mean: Array.from({ length: dimensions }, () => 0),
    covariance: Array.from({ length: dimensions }, (_, i) =>
      Array.from({ length: dimensions }, (__, j) => (i === j ? 1 : 0)),
    ),
  };
  const estimate: MultidimensionalEstimator =
    estimator ??
    ((items, responses) => multidimensionalPosteriorModeEstimate(items, responses, { prior: standardPrior }));
  const engine = adaptiveEngine({
    pool,
    indices: indexMultidimensionalItems(pool),
    stopping,
    selection,
    contentTargets,
    enemies,
    withinTestlet,
    estimate,
    checkResponse: (item, response) => {
      checkCategory(item.id, item.model.categoryCount(item.parameters), response);
    },
  });
  return { pool, ...engine };
}
