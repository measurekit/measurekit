import { checkArray, checkChoice, checkOptions, checkRecord, DataError, describe } from "@measurekit/core";
import { checkFunction } from "../checks.js";
import { type AbilityEstimator, posteriorMeanEstimate } from "../irt/ability.js";
import { indexItems } from "../irt/items.js";
import type { TestItem } from "../irt/model.js";
import { formatList } from "../irt/thresholds.js";
import { checkResponse, itemById } from "./items.js";
import { checkState, type TestState } from "./state.js";

/**
 * What routes a person to a module of the next stage: `number-correct`, the total score of the responses so far, which
 * for dichotomous items is the number correct; `theta`, the estimate of theta from them.
 */
export type MultistageRouting = "number-correct" | "theta";

/** A module of a multistage test: items administered together. */
export interface MultistageModule {
  /** The module's id, unique in its panel. */
  readonly id: string;
  /** The module's items, in the order of their responses. */
  readonly items: readonly TestItem[];
  /**
   * The cut points that route from this module to the modules of the next stage: one fewer than those modules, in
   * ascending order. A person goes to module j of the next stage, counting from 0, when exactly j cuts are at most the
   * routing value. Modules of the last stage have none.
   */
  readonly cuts?: readonly number[];
}

/** Options of {@link multistageTest}. */
export interface MultistageTestOptions {
  /** The stages of the panel, in order, each a list of modules. The first stage has one module. */
  readonly stages: readonly (readonly MultistageModule[])[];
  /** What routes a person between stages. */
  readonly routing: MultistageRouting;
  /**
   * Estimates theta after each module, and before the first, without responses.
   *
   * @defaultValue `posteriorMeanEstimate`
   */
  readonly estimator?: AbilityEstimator;
}

/** The state of a multistage test: plain data, which JSON serializes, so that a test can resume from a stored state. */
export interface MultistageTestState extends TestState {
  /** The ids of the administered modules, in order. */
  readonly modules: readonly string[];
  /** The id of the module to administer next, or `null` when the test has ended. */
  readonly next: string | null;
}

/** A multistage test: a panel of stages of modules, with routing between stages. */
export interface MultistageTest {
  /** The stages of the panel. */
  readonly stages: readonly (readonly MultistageModule[])[];
  /** Returns the state before the first module: the estimate without responses and the first stage's module. */
  start(): MultistageTestState;
  /**
   * Returns the state after responses to the items of the state's next module, in their order: the new estimate, and
   * the module of the next stage that routing selects, or none after the last stage.
   *
   * @throws `DataError` when the test has ended, the state is malformed, names modules or items the panel does not have
   * or does not follow the stages, or the responses do not match the module's items.
   */
  answer(state: MultistageTestState, responses: readonly number[]): MultistageTestState;
}

const ROUTINGS: readonly MultistageRouting[] = ["number-correct", "theta"];

/**
 * Returns a multistage test of a panel. Every person takes the module of the first stage, then one module of each
 * following stage, chosen by the cut points of the module before.
 *
 * @throws `DataError` when the panel has no stages, a stage has no modules, the first stage has more than one, a module
 * has no items, ids of modules or items repeat, an item's parameters are invalid, cut points do not match the next
 * stage, the routing is unknown, or the estimator is not a function.
 */
export function multistageTest(options: MultistageTestOptions): MultistageTest {
  checkOptions(options);
  const { stages, routing, estimator = posteriorMeanEstimate } = options;
  checkChoice(routing, "The option routing", ROUTINGS);
  checkFunction(
    estimator,
    "The option estimator must be a function of items and responses, such as posteriorMeanEstimate",
  );
  const located = locateModules(stages);
  const items = stages.flat().flatMap((module) => module.items);
  const indices = indexItems(items);

  return {
    stages,
    start: () => ({
      modules: [],
      items: [],
      responses: [],
      estimate: estimator([], []),
      next: stages[0]?.[0]?.id ?? null,
    }),
    answer(state, responses) {
      checkState(state, items, indices);
      if (state.next === null) throw new DataError("The test has ended; it takes no more responses.");
      const { module, stage } = moduleById(located, state.next);
      checkPath(located, state, stage);
      checkArray(responses, "The responses must be an array");
      if (responses.length !== module.items.length) {
        throw new DataError(
          `Module "${module.id}" has ${module.items.length} items, found ${responses.length} responses.`,
        );
      }
      module.items.forEach((item, index) => {
        checkResponse(item, responses[index] ?? Number.NaN);
      });
      const ids = [...state.items, ...module.items.map((item) => item.id)];
      const allResponses = [...state.responses, ...responses];
      const estimate = estimator(
        ids.map((id) => itemById(items, indices, id)),
        allResponses,
      );
      const value = routing === "theta" ? estimate.theta : allResponses.reduce((sum, response) => sum + response, 0);
      const position = (module.cuts ?? []).filter((cut) => cut <= value).length;
      return {
        modules: [...state.modules, module.id],
        items: ids,
        responses: allResponses,
        estimate,
        next: stages[stage + 1]?.[position]?.id ?? null,
      };
    },
  };
}

/** A module with the index of its stage. */
interface LocatedModule {
  readonly module: MultistageModule;
  readonly stage: number;
}

/**
 * Checks the structure of a panel and returns its modules by id.
 *
 * @throws `DataError` when the panel has no stages, a stage has no modules, the first stage has more than one, a module
 * is not an object with a string id and at least one item, module ids repeat, or cut points do not match the
 * next stage.
 */
export function locateModules(stages: readonly (readonly MultistageModule[])[]): Map<string, LocatedModule> {
  checkArray(stages, "The option stages must be an array of stages");
  const valid = stages.length > 0 && stages.every((modules) => Array.isArray(modules) && modules.length > 0);
  if (!valid) {
    throw new DataError("A multistage test needs at least one stage, and every stage an array of at least one module.");
  }
  if (stages[0]?.length !== 1) {
    throw new DataError(`The first stage of a multistage test has one module, found ${stages[0]?.length}.`);
  }
  const located = new Map<string, LocatedModule>();
  stages.forEach((modules, stage) => {
    const following = stages[stage + 1]?.length ?? 1;
    for (const module of modules) {
      checkRecord(module, `The modules of stage ${stage + 1} must be objects { id, items, cuts }`);
      if (typeof module.id !== "string") {
        throw new DataError(`A module needs a string id, found ${describe(module.id)}.`);
      }
      if (located.has(module.id)) throw new DataError(`Module ids must be unique; "${module.id}" appears twice.`);
      if (!(Array.isArray(module.items) && module.items.length > 0)) {
        throw new DataError(`Module "${module.id}" needs an array of at least one item.`);
      }
      checkCuts(module, following);
      located.set(module.id, { module, stage });
    }
  });
  return located;
}

/** Checks that a module has one cut point fewer than the modules it routes to, in ascending order. */
function checkCuts({ id, cuts = [] }: MultistageModule, following: number): void {
  const ascending =
    Array.isArray(cuts) &&
    cuts.every((cut, j) => typeof cut === "number" && !Number.isNaN(cut) && (j === 0 || cut >= (cuts[j - 1] ?? cut)));
  if (!(ascending && cuts.length === following - 1)) {
    throw new DataError(
      `Module "${id}" needs an array of ${following - 1} cut points in ascending order to route to the next stage, found ${formatList(cuts)}.`,
    );
  }
}

/**
 * Checks that a state took one module of each stage before the stage of its next module, and their items in order.
 *
 * @throws `DataError` when it did not.
 */
function checkPath(located: ReadonlyMap<string, LocatedModule>, state: MultistageTestState, stage: number): void {
  if (!Array.isArray(state.modules)) throw new DataError("A multistage test state needs the array modules.");
  const taken = state.modules.map((id) => moduleById(located, id));
  const path = taken.flatMap(({ module }) => module.items.map((item) => item.id));
  const inOrder = taken.every((module, j) => module.stage === j) && stage === taken.length;
  if (!inOrder || path.length !== state.items.length || path.some((id, j) => id !== state.items[j])) {
    throw new DataError(
      `A multistage test state takes one module per stage, in order, with their items; found modules [${state.modules.join(", ")}] and then "${state.next}".`,
    );
  }
}

function moduleById(located: ReadonlyMap<string, LocatedModule>, id: string): LocatedModule {
  const module = located.get(id);
  if (module === undefined) throw new DataError(`The panel has no module ${describe(id)}.`);
  return module;
}
