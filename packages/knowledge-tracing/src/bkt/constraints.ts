import { checkRecord, DataError, describe, isUnitInterval } from "@measurekit/core";
import { entry, mapRecord } from "../data/internal.js";
import type { BktLearning, BktParameters, BktSkill } from "./parameters.js";

/** The rates of a skill that a fit estimates. */
type BktRate = "prior" | "learn" | "forget" | "guess" | "slip";

/** Rates that EM holds at given values for one skill: the others are estimated. */
export interface BktFixedRates {
  /** P(L0), the skill's own prior. */
  readonly prior?: number;
  /** P(T), the skill's own learn rate. */
  readonly learn?: number;
  /** P(F), the skill's own forget rate; needs `forgetting`. */
  readonly forget?: number;
  /** P(G), the skill's own guess rate. */
  readonly guess?: number;
  /** P(S), the skill's own slip rate. */
  readonly slip?: number;
  /**
   * The learn and forget rates of single resources, so that a lesson can be held while others are fitted; needs
   * `resourceLearning`.
   */
  readonly resources?: Readonly<Record<string, Partial<BktLearning>>>;
}

/** The bounds of the rates that EM estimates: a lower and an upper bound, from 0 to 1, for each rate that has any. */
export interface BktBounds {
  /** The bounds of P(L0), of the skill and of each group. */
  readonly prior?: readonly [number, number];
  /** The bounds of P(T), of the skill and of each resource or pair. */
  readonly learn?: readonly [number, number];
  /** The bounds of P(F); needs `forgetting`. */
  readonly forget?: readonly [number, number];
  /** The bounds of P(G), of the skill and of each item. */
  readonly guess?: readonly [number, number];
  /** The bounds of P(S), of the skill and of each item. */
  readonly slip?: readonly [number, number];
}

/** What a fit holds its estimates to: the fixed rates by skill and the bounds, which every skill has. */
export interface Constraints {
  readonly fixed: Readonly<Record<string, BktFixedRates>>;
  readonly bounds: BktBounds;
}

/**
 * The least distance of a fitted P(G) or P(S) from 0 and 1, 2⁻⁵², so that a fitted model gives every outcome a positive
 * probability: data with every outcome correct would otherwise fit P(S) = 0, and an incorrect answer would be impossible.
 */
const MIN_GUESS_SLIP = Number.EPSILON;

const RATES: readonly BktRate[] = ["prior", "learn", "forget", "guess", "slip"];

/**
 * Returns the bounds a rate is held to: its own, within 0 and 1, and for the estimate of a guess or slip, off 0 and 1.
 */
function limitsOf(bounds: BktBounds, rate: BktRate, estimate: boolean): readonly [number, number] {
  const [lower, upper] = bounds[rate] ?? [0, 1];
  const margin = estimate && (rate === "guess" || rate === "slip") ? MIN_GUESS_SLIP : 0;
  return [Math.max(lower, margin), Math.min(upper, 1 - margin)];
}

function clamp(value: number, [lower, upper]: readonly [number, number]): number {
  return Math.min(Math.max(value, lower), upper);
}

/**
 * Returns a skill with its fixed rates in place of its own and every other rate within the bounds: the groups' priors,
 * the resources' and pairs' learn and forget rates and the items' guess and slip are held to the bounds of their kind.
 * With `estimate`, a guess and a slip also keep 2⁻⁵² from 0 and 1, as an estimate must; a start does not.
 */
export function constrain(
  skill: BktSkill,
  fixed: BktFixedRates | undefined,
  bounds: BktBounds,
  estimate: boolean,
): BktSkill {
  const limits = (rate: BktRate): readonly [number, number] => limitsOf(bounds, rate, estimate);
  const own = (rate: BktRate): number => fixed?.[rate] ?? clamp(skill[rate], limits(rate));
  const learning = (rates: BktLearning, held: Partial<BktLearning> | undefined): BktLearning => ({
    learn: held?.learn ?? clamp(rates.learn, limits("learn")),
    forget: held?.forget ?? clamp(rates.forget, limits("forget")),
  });
  const { groupPriors, resourceLearning, pairLearning, itemGuessSlip } = skill;
  return {
    prior: own("prior"),
    learn: own("learn"),
    forget: own("forget"),
    guess: own("guess"),
    slip: own("slip"),
    ...(groupPriors !== undefined && {
      groupPriors: mapRecord(groupPriors, (prior) => clamp(prior, limits("prior"))),
    }),
    ...(resourceLearning !== undefined && {
      resourceLearning: mapRecord(resourceLearning, (rates, resource) =>
        learning(rates, entry(fixed?.resources, resource)),
      ),
    }),
    ...(pairLearning !== undefined && {
      pairLearning: mapRecord(pairLearning, (pairs) => mapRecord(pairs, (rates) => learning(rates, undefined))),
    }),
    ...(itemGuessSlip !== undefined && {
      itemGuessSlip: mapRecord(itemGuessSlip, ({ guess, slip }) => ({
        guess: clamp(guess, limits("guess")),
        slip: clamp(slip, limits("slip")),
      })),
    }),
  };
}

/** What the checks of the constraints need of the options. */
interface Structure {
  readonly forgetting: boolean;
  readonly resourceLearning: boolean;
  readonly pairLearning: boolean;
}

/**
 * Checks the fixed rates and the bounds, throwing a `DataError` that names the first problem found.
 *
 * @throws `DataError` when a rate is not a probability, a bound is not a pair of probabilities or its lower bound is above
 * its upper, a fixed rate is outside its bounds, or `forget` or `resources` is used without the option it needs.
 */
export function checkConstraints({ fixed, bounds }: Constraints, structure: Structure): void {
  checkRecord(fixed, "The option fixed must be an object of fixed rates by skill id");
  checkRecord(bounds, "The option bounds must be an object of bounds by rate");
  for (const rate of Object.keys(bounds)) {
    if (!RATES.includes(rate as BktRate)) {
      throw new DataError(`The option bounds has no rate "${rate}"; its rates are ${RATES.join(", ")}.`);
    }
  }
  for (const rate of RATES) {
    const pair = bounds[rate];
    if (pair === undefined) continue;
    if (!(Array.isArray(pair) && pair.length === 2 && pair.every(isUnitInterval))) {
      throw new DataError(
        `The bounds of ${rate} must be a pair of probabilities [lower, upper], found ${describe(pair)}.`,
      );
    }
    if (pair[0] > pair[1]) {
      throw new DataError(`The lower bound of ${rate}, ${pair[0]}, is above its upper bound, ${pair[1]}.`);
    }
    if (rate === "forget" && !structure.forgetting) throw new DataError("The bounds of forget need forgetting: true.");
  }
  for (const [skill, rates] of Object.entries(fixed)) {
    checkFixedRates(skill, rates, bounds, structure);
  }
}

function checkFixedRates(skill: string, rates: BktFixedRates, bounds: BktBounds, structure: Structure): void {
  const label = `The fixed rates of skill "${skill}"`;
  checkRecord(rates, `${label} must be an object`);
  for (const rate of RATES) checkHeld(label, rate, rates[rate], bounds, structure.forgetting);
  if (rates.resources === undefined) return;
  if (!structure.resourceLearning) throw new DataError(`${label} have resources, which need resourceLearning: true.`);
  checkRecord(rates.resources, `${label} need resources, an object of rates by resource id`);
  for (const [resource, held] of Object.entries(rates.resources)) {
    const where = `${label}, resource "${resource}",`;
    checkRecord(held, `${where} must be an object`);
    for (const rate of ["learn", "forget"] as const) checkHeld(where, rate, held[rate], bounds, structure.forgetting);
  }
}

/** Checks a fixed rate, if there is one: a probability within the bounds of its rate, and not a forget rate without forgetting. */
function checkHeld(label: string, rate: BktRate, value: unknown, bounds: BktBounds, forgetting: boolean): void {
  if (value === undefined) return;
  if (!isUnitInterval(value))
    throw new DataError(`${label} have ${rate} that is not a probability, found ${describe(value)}.`);
  if (rate === "forget" && !forgetting) throw new DataError(`${label} have forget, which needs forgetting: true.`);
  const [lower, upper] = bounds[rate] ?? [0, 1];
  if (value < lower || value > upper)
    throw new DataError(`${label} have ${rate} ${value}, outside its bounds [${lower}, ${upper}].`);
}

/** Lists the rates of a skill with the kind of each and where it is, for the message of an error. */
function listRates(skill: BktSkill): [BktRate, number, string][] {
  const learning = (rates: BktLearning, where: string): [BktRate, number, string][] => [
    ["learn", rates.learn, where],
    ["forget", rates.forget, where],
  ];
  return [
    ...RATES.map((rate): [BktRate, number, string] => [rate, skill[rate], ""]),
    ...Object.entries(skill.groupPriors ?? {}).map(([group, prior]): [BktRate, number, string] => [
      "prior",
      prior,
      ` of group "${group}"`,
    ]),
    ...Object.entries(skill.resourceLearning ?? {}).flatMap(([resource, rates]) =>
      learning(rates, ` of resource "${resource}"`),
    ),
    ...Object.entries(skill.pairLearning ?? {}).flatMap(([previous, pairs]) =>
      Object.entries(pairs).flatMap(([resource, rates]) =>
        learning(rates, ` of resource "${resource}" after "${previous}"`),
      ),
    ),
    ...Object.entries(skill.itemGuessSlip ?? {}).flatMap(([item, { guess, slip }]): [BktRate, number, string][] => [
      ["guess", guess, ` of item "${item}"`],
      ["slip", slip, ` of item "${item}"`],
    ]),
  ];
}

/** Checks that the rates of given parameters are within the bounds, as a model that starts from them needs. */
export function checkWithinBounds(parameters: BktParameters, bounds: BktBounds): void {
  for (const [id, skill] of Object.entries(parameters.skills)) {
    for (const [rate, value, where] of listRates(skill)) {
      const pair = bounds[rate];
      if (pair !== undefined && (value < pair[0] || value > pair[1])) {
        throw new DataError(
          `The BKT parameters of skill "${id}" have ${rate} ${value}${where}, outside the bounds [${pair[0]}, ${pair[1]}].`,
        );
      }
    }
  }
}
