import { checkRecord, DataError, describe, isUnitInterval } from "@measurekit/core";

/** An item's error rates in a basic local independence model. */
export interface KstItem {
  /** η, the probability of a lucky guess: a correct answer where the state does not master the item. */
  readonly guess: number;
  /** β, the probability of a careless error: a wrong answer where the state masters the item. */
  readonly slip: number;
}

/** The parameters of a KST model: a basic local independence model (BLIM) on its structure. */
export interface KstParameters {
  /** The probability of each knowledge state in the population, in the structure's order; they sum to 1. */
  readonly probabilities: readonly number[];
  /** Each item's guess and slip, by item id. */
  readonly items: Readonly<Record<string, KstItem>>;
}

/** The guess and slip that a fit starts an item without parameters from, as pks's `blim`. */
export const START_RATE = 0.1;

/** How far the probabilities of a distribution may sum from 1, for the rounding of a sum of many. */
const SUM_TOLERANCE = 1e-9;

/** Checks a list of probabilities over the states: non-negative, as many as the states, summing to 1. */
export function checkDistribution(values: unknown, count: number, label: string): void {
  if (!(Array.isArray(values) && values.length === count && values.every((p) => typeof p === "number" && p >= 0))) {
    throw new DataError(`${label} needs probabilities, ${count} numbers of at least 0, one per knowledge state.`);
  }
  const total = values.reduce((sum: number, p: number) => sum + p, 0);
  if (Math.abs(total - 1) > SUM_TOLERANCE) {
    throw new DataError(`${label} has probabilities that sum to ${total}, not 1.`);
  }
}

/** Checks parameters, which may come from storage, against the number of knowledge states. */
export function checkParameters(parameters: KstParameters, count: number): void {
  checkRecord(parameters, "KST parameters must be an object");
  checkDistribution(parameters.probabilities, count, "The KST parameters");
  checkRecord(parameters.items, "The KST parameters need items, an object of guess and slip by item id");
  for (const [id, item] of Object.entries(parameters.items)) {
    for (const name of ["guess", "slip"] as const) {
      const value = item?.[name];
      if (!isUnitInterval(value)) {
        throw new DataError(`Item "${id}" needs ${name}, a probability from 0 to 1, found ${describe(value)}.`);
      }
    }
  }
}
