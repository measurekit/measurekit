import { checkMethod, checkOptions, checkRandom, DataError, type Random } from "@measurekit/core";
import { checkFunction } from "../checks.js";
import { type AbilityEstimator, posteriorMeanEstimate } from "../irt/ability.js";
import { indexItems } from "../irt/items.js";
import { contentBalancing, restrictToArea } from "./content-balancing.js";
import { checkEnemies, enemyIndices, excludeEnemies } from "./enemies.js";
import { checkResponse, indexOf, itemById, type PoolItem } from "./items.js";
import { type AdaptiveTestContext, type ItemSelectionRule, maximumInformation } from "./selection.js";
import { checkDistinct, checkState, type TestState } from "./state.js";
import type { StoppingRule } from "./stopping.js";
import { checkExposure, selectItem } from "./sympson-hetter.js";

/** Options of {@link adaptiveTest}. */
export interface AdaptiveTestOptions {
  /** The item pool. */
  readonly pool: readonly PoolItem[];
  /** The rule that ends the test. The test also ends when no item is left. */
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
  /**
   * Target proportions of the content areas, by area, for Kingsbury-Zara content balancing; they need not sum to 1.
   * Every pool item then needs one of these areas.
   *
   * @defaultValue none: no content balancing
   */
  readonly contentTargets?: Readonly<Record<string, number>>;
  /**
   * Enemy sets: ids of items that must not be administered in the same test, such as items that give away each other's
   * answers. Once an item of a set is administered, the others are not eligible.
   *
   * @defaultValue none
   */
  readonly enemies?: readonly (readonly string[])[];
}

/** The state of an adaptive test: plain data, which JSON serializes, so that a test can resume from a stored state. */
export interface AdaptiveTestState extends TestState {
  /** The ids of the items exposure control withheld: they are not administered in this test. */
  readonly withheld: readonly string[];
  /** The id of the item to administer next, or `null` when the test has ended. */
  readonly next: string | null;
}

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
 * of an administered item are not eligible.
 *
 * @throws `DataError` when `stopping`, `selection` or `estimator` is not a rule or function, an item's id or
 * parameters are invalid, ids repeat, an exposure parameter is outside [0, 1], a content target is invalid or missing
 * for an item's area, or an enemy set does not name two or more distinct pool items.
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
    enemies: enemySets = [],
  } = options;
  checkMethod(stopping, "stops", "The option stopping must be a stopping rule, such as testLength(20)");
  checkMethod(selection, "scores", "The option selection must be a selection rule, such as maximumInformation");
  checkFunction(
    estimator,
    "The option estimator must be a function of items and responses, such as posteriorMeanEstimate",
  );
  const indices = indexItems(pool);
  pool.forEach(checkExposure);
  const balancing = contentTargets === undefined ? null : contentBalancing(pool, contentTargets);
  const enemies = enemyIndices(pool, indices, enemySets);

  const advance = (
    administered: readonly number[],
    responses: readonly number[],
    withheld: number[],
    random: Random,
  ): AdaptiveTestState => {
    const estimate = estimator(
      administered.flatMap((index) => pool[index] ?? []),
      responses,
    );
    const eligible = new Uint8Array(pool.length).fill(1);
    for (const index of administered) eligible[index] = 0;
    for (const index of withheld) eligible[index] = 0;
    excludeEnemies(enemies, administered, eligible);
    const context: AdaptiveTestContext = { pool, administered, responses, estimate, eligible };
    let next: number | null = null;
    if (eligible.includes(1) && !stopping.stops(context)) {
      const selectable = eligible.slice();
      if (balancing !== null) restrictToArea(balancing, administered, selectable, random);
      next = selectItem(selection, { ...context, eligible: selectable, random }, withheld);
    }
    const ids = (list: readonly number[]): string[] => list.flatMap((index) => pool[index]?.id ?? []);
    return {
      items: ids(administered),
      responses,
      estimate,
      withheld: ids(withheld),
      next: next === null ? null : (pool[next]?.id ?? null),
    };
  };

  return {
    pool,
    start(random) {
      checkRandom(random);
      return advance([], [], [], random);
    },
    answer(state, response, random) {
      checkRandom(random);
      const administered = checkState(state, pool, indices);
      if (!Array.isArray(state.withheld)) throw new DataError("An adaptive test state needs the array withheld.");
      if (state.next === null) throw new DataError("The test has ended; it takes no more responses.");
      const withheld = state.withheld.map((id) => indexOf(indices, id));
      const next = indexOf(indices, state.next);
      checkDistinct([...administered, ...withheld, next], pool);
      checkEnemies(enemies, [...administered, next], pool);
      checkResponse(itemById(pool, indices, state.next), response);
      return advance([...administered, next], [...state.responses, response], withheld, random);
    },
  };
}
