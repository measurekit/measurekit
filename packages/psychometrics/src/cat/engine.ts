import { checkMethod, checkRandom, DataError, type Random } from "@measurekit/core";
import { contentBalancing, restrictToArea } from "./content-balancing.js";
import { checkEnemies, enemyIndices, excludeEnemies } from "./enemies.js";
import { indexOf, itemById, type PoolControls } from "./items.js";
import type { AdaptiveTestContext, ItemSelectionRule, SelectionContext } from "./selection.js";
import { checkDistinct, checkState, type TestState } from "./state.js";
import type { StoppingRule } from "./stopping.js";
import { checkExposure, selectItem } from "./sympson-hetter.js";
import {
  checkTestletAreas,
  checkWithinTestlet,
  nextInTestlet,
  poolTestlets,
  remainingInTestlet,
  unitSelection,
  type WithinTestlet,
} from "./testlets.js";

/**
 * The state of an adaptive test of any kind: plain data, which JSON serializes, so that a test can resume from a stored
 * state.
 *
 * @typeParam TEstimate - The estimate of what the test measures.
 */
export interface AdaptiveState<TEstimate> extends TestState<TEstimate> {
  /** The ids of the items exposure control withheld: they are not administered in this test. */
  readonly withheld: readonly string[];
  /** The id of the item to administer next, or `null` when the test has ended. */
  readonly next: string | null;
}

/** What {@link adaptiveEngine} needs of a test: validated items, rules and the functions of the test's estimate. */
export interface AdaptiveEngineOptions<TItem extends PoolControls, TEstimate> {
  /** The pool, whose items the test has checked, and ids are unique. */
  readonly pool: readonly TItem[];
  /** The index of each pool item by id. */
  readonly indices: ReadonlyMap<string, number>;
  /** The rule that ends the test. */
  readonly stopping: StoppingRule<TItem, TEstimate>;
  /** The rule that selects the next item. */
  readonly selection: ItemSelectionRule<TItem, TEstimate>;
  /** The target proportions of the content areas, by area. */
  readonly contentTargets?: Readonly<Record<string, number>>;
  /**
   * The enemy sets.
   *
   * @defaultValue none
   */
  readonly enemies?: readonly (readonly string[])[];
  /**
   * How the items of a testlet are ordered. It applies to pool items that have a `testlet`.
   *
   * @defaultValue `"in-order"`
   */
  readonly withinTestlet?: WithinTestlet;
  /** Returns the estimate from the administered items and their responses, also without any. */
  readonly estimate: (administered: readonly TItem[], responses: readonly number[]) => TEstimate;
  /** Checks that a response is one the item can have. */
  readonly checkResponse: (item: TItem, response: number) => void;
}

/** The steps of an adaptive test: its state before the first response, and after each. */
export interface AdaptiveEngine<TEstimate> {
  /**
   * Returns the state before the first response.
   *
   * @throws `DataError` when `random` is not a random number generator.
   */
  start(random: Random): AdaptiveState<TEstimate>;
  /**
   * Returns the state after a response to the state's next item.
   *
   * @throws `DataError` when the test has ended, the state is malformed, names items the pool does not have or an item
   * twice, administers enemies, a response is not one of its item, or `random` is not a random number generator.
   */
  answer(state: AdaptiveState<TEstimate>, response: number, random: Random): AdaptiveState<TEstimate>;
}

/**
 * Checks the rules that an adaptive test takes.
 *
 * @param selectionExample - Names a selection rule for the message, such as `maximumInformation`.
 * @throws `DataError` when `stopping` is not a stopping rule or `selection` is not a selection rule.
 */
export function checkRules(stopping: unknown, selection: unknown, selectionExample: string): void {
  checkMethod(stopping, "stops", "The option stopping must be a stopping rule, such as testLength(20)");
  checkMethod(selection, "scores", `The option selection must be a selection rule, such as ${selectionExample}`);
}

/**
 * Returns the steps of an adaptive test that estimates, selects and stops by the rules it is given, which every kind
 * of adaptive test shares.
 *
 * @remarks
 * Before each item the engine estimates from the responses so far and checks the stopping rule; unless the test ends,
 * content balancing restricts the eligible items to one content area, the selection rule selects among them, and
 * Sympson-Hetter exposure control administers or withholds the selected item. Items administered, withheld or
 * enemies of an administered item are not eligible.
 *
 * Items of a testlet are administered together. The selection rule scores each testlet as a unit, by the mean of its
 * eligible items' scores, and the testlet's first item to be administered goes through content balancing, which takes
 * the testlet's area, and exposure control for the whole testlet. The items that follow are administered without the
 * stopping rule, content balancing or exposure control, in pool order or by the selection rule's scores among the
 * testlet's remaining items, until the testlet has none left.
 *
 * @throws `DataError` when an exposure parameter is outside [0, 1], a testlet is not a non-empty string, a content
 * target is invalid or missing for an item's area, the items of a testlet are in several content areas, an enemy set
 * does not name two or more distinct pool items, or `withinTestlet` is unknown.
 */
export function adaptiveEngine<TItem extends PoolControls, TEstimate>(
  options: AdaptiveEngineOptions<TItem, TEstimate>,
): AdaptiveEngine<TEstimate> {
  const {
    pool,
    indices,
    stopping,
    selection,
    contentTargets,
    enemies: enemySets = [],
    estimate,
    checkResponse,
  } = options;
  const { withinTestlet = "in-order" } = options;
  checkWithinTestlet(withinTestlet);
  pool.forEach(checkExposure);
  const testlets = poolTestlets(pool);
  const unitRule = testlets === null ? selection : unitSelection(selection, testlets, withinTestlet);
  const membersOf = (index: number): readonly number[] => {
    const testlet = testlets?.of[index] ?? -1;
    return testlet < 0 ? [index] : (testlets?.members[testlet] ?? [index]);
  };
  const balancing = contentTargets === undefined ? null : contentBalancing(pool, contentTargets);
  if (balancing !== null && testlets !== null) checkTestletAreas(pool, testlets);
  const enemies = enemyIndices(pool, indices, enemySets);

  /** Returns the item to administer next, or `null` when the test ends: the rest of a testlet, else a selection. */
  const selectNext = (context: SelectionContext<TItem, TEstimate>, withheld: number[]): number | null => {
    const { administered, eligible, random } = context;
    const remaining = testlets === null ? null : remainingInTestlet(testlets, administered, eligible);
    if (remaining !== null) return nextInTestlet(selection, context, remaining, withinTestlet);
    if (!eligible.includes(1) || stopping.stops(context)) return null;
    const selectable = eligible.slice();
    if (balancing !== null) restrictToArea(balancing, administered, selectable, random);
    return selectItem(unitRule, { ...context, eligible: selectable }, withheld, membersOf);
  };

  const advance = (
    administered: readonly number[],
    responses: readonly number[],
    withheld: number[],
    random: Random,
  ): AdaptiveState<TEstimate> => {
    const current = estimate(
      administered.flatMap((index) => pool[index] ?? []),
      responses,
    );
    const eligible = new Uint8Array(pool.length).fill(1);
    for (const index of administered) eligible[index] = 0;
    for (const index of withheld) eligible[index] = 0;
    excludeEnemies(enemies, administered, eligible);
    const context: AdaptiveTestContext<TItem, TEstimate> = {
      pool,
      administered,
      responses,
      estimate: current,
      eligible,
    };
    const next = selectNext({ ...context, random }, withheld);
    const ids = (list: readonly number[]): string[] => list.flatMap((index) => pool[index]?.id ?? []);
    return {
      items: ids(administered),
      responses,
      estimate: current,
      withheld: ids(withheld),
      next: next === null ? null : (pool[next]?.id ?? null),
    };
  };

  return {
    start(random) {
      checkRandom(random);
      return advance([], [], [], random);
    },
    answer(state, response, random) {
      checkRandom(random);
      const administered = checkState(state, pool, indices, checkResponse);
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
