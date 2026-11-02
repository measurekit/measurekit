import {
  backtrack,
  checkArray,
  checkOptions,
  checkRecord,
  checkStopping,
  cholesky,
  DataError,
  describe,
  isNonEmptyString,
  iterateToConvergence,
  maxAbs,
} from "@measurekit/core";
import { checkResponseMatrix, type ResponseMatrix } from "../data/internal.js";
import { invertPositiveDefinite } from "../math/internal.js";

/** Options of {@link linearChange}. */
export interface LinearChangeOptions {
  /**
   * Each person's group, one name for each person of the matrices.
   *
   * @defaultValue one group of all persons, named `"all"`
   */
  readonly groups?: readonly string[];
  /**
   * The group whose change is the trend: the effects of the other groups are relative to it.
   *
   * @defaultValue the group of the first person
   */
  readonly baseline?: string;
  /**
   * Sets of items that share one change parameter: the name of the set of each item, by item id, for every item.
   *
   * @defaultValue each item is a set of its own, named by its id
   */
  readonly itemGroups?: Readonly<Record<string, string>>;
  /**
   * Estimation has converged when no effect changes by more than this in a Newton step.
   *
   * @defaultValue `1e-8`
   */
  readonly tolerance?: number;
  /**
   * The most Newton steps to take.
   *
   * @defaultValue `100`
   */
  readonly maxIterations?: number;
}

/** The change of one set of items at one occasion, for one group. */
export interface LinearChangeEffect {
  /** The set of items: the name given in `itemGroups`, or the item's id. */
  readonly itemGroup: string;
  /** The occasion, counted from 1, whose change from the first occasion the effect is: from 2. */
  readonly occasion: number;
  /** The group; the baseline group's row is the trend, and the other groups' are relative to it. */
  readonly group: string;
  /** The effect, in logits. */
  readonly estimate: number;
  /** Its standard error from the conditional information; `null` where the estimate is not finite. */
  readonly standardError: number | null;
}

/** A linear logistic model of change estimated by conditional maximum likelihood. */
export interface LinearChange {
  /**
   * The effects by item group, then occasion, then group, the baseline group first: in each, the baseline group's row is
   * the trend, and the other groups' rows are the change beyond it.
   */
  readonly effects: readonly LinearChangeEffect[];
  /** The conditional log-likelihood at the estimates. */
  readonly logLikelihood: number;
  /** Whether no effect changed by more than the tolerance in the last step, within the most steps. */
  readonly converged: boolean;
  /** The Newton steps taken. */
  readonly iterations: number;
}

const DEFAULT_TOLERANCE = 1e-8;
const DEFAULT_MAX_ITERATIONS = 100;
/** The most occasions: the conditional likelihood of a person and an item sums over the 2ᵀ patterns of T responses. */
const MAX_OCCASIONS = 12;
/** The group of everyone when none are given. */
const EVERYONE = "all";
/**
 * The largest effect, in logits, that counts as finite: at e⁻²⁰ ≈ 2e-9 a pattern against the others, the likelihood is
 * flat to its rounding, so an estimate beyond it is a maximum at infinity, which the Newton steps only approach.
 */
const MAX_EFFECT = 20;
/**
 * The relative rise of the negative conditional log-likelihood that a step may have and still count as no rise: its
 * rounding. A Newton step near the maximum, even one of 1e-7, changes the log-likelihood by less than its digits hold, so
 * a search that rejects it halves the step, reports the shortened step as the change and stops with estimates off by the
 * rest of it.
 */
const LIKELIHOOD_ROUNDING = 1e-12;

/**
 * Estimates the linear logistic model for change (LLRA) of the same persons answering the same items at several
 * occasions, by conditional maximum likelihood.
 *
 * @remarks
 * Each person has a trait of their own on each item, which the model does not estimate: it conditions on the number of
 * correct responses of each person to each item, so that the trait drops out. The logit of a correct response is that
 * trait plus, from the second occasion, the effect of the item's set, the occasion and the person's group, and the
 * first occasion is the reference. The baseline group's effect is the trend, and another group's effect is its change
 * beyond the trend. A person who answered an item all correctly or all wrongly tells nothing about change, so with two
 * occasions only persons who changed on an item inform its effect, and the model is a logistic regression on them.
 *
 * Responses are 0 or 1, or `null` when missing: a person missing any occasion on an item is left out of that item. Every
 * occasion has the same persons in the same order and the same items. Newton's method with a line search maximizes the
 * concave conditional likelihood, and the standard errors are those of its information. Where the persons who changed
 * all changed one way, the estimate has no finite maximum: the steps stop beyond 20 logits and the estimation reports
 * `converged: false`.
 *
 * @example
 * ```ts
 * import {
 *   createRandom,
 *   linearChange,
 *   simulateResponses,
 *   simulateThetas,
 *   twoParameterLogistic,
 * } from "@measurekit/psychometrics";
 *
 * // 800 persons in two groups answer 4 items twice: the control group gains 0.3 logits, the treated group 0.8.
 * const random = createRandom(5);
 * const thetas = simulateThetas(800, random);
 * const group = thetas.map((_, person) => (person < 400 ? "control" : "treated"));
 * const items = [-0.5, 0, 0.5, 1].map((b) => ({ model: twoParameterLogistic, parameters: { a: 1, b, c: 0, d: 1 } }));
 * const before = simulateResponses(items, thetas, random);
 * const gain = (person: number) => (group[person] === "treated" ? 0.8 : 0.3);
 * const after = simulateResponses(items, thetas.map((theta, person) => theta + gain(person)), random);
 * const change = linearChange([before, after], {
 *   groups: group,
 *   baseline: "control",
 *   itemGroups: Object.fromEntries(before.itemIds.map((id) => [id, "test"])),
 * });
 * change.effects.map(({ group, estimate }) => [group, estimate]); // [["control", 0.35], ["treated", 0.35]]
 * ```
 *
 * @param occasions - One response matrix of dichotomous responses for each occasion, in time order, from two to 12.
 * @param options - The groups of persons, the baseline group, the sets of items and the stopping rules.
 * @returns The effects with their standard errors, and the conditional log-likelihood.
 * @throws `DataError` when the matrices or options are invalid, the occasions differ in persons or items, a response is
 * not 0, 1 or `null`, a group or a set is missing for a person or an item, or an effect has no unique estimate: no person
 * changed on it, or those who did make it a combination of the effects before it.
 * @see {@link https://doi.org/10.1007/978-1-4612-4230-7_9 | Fischer (1995)}
 */
export function linearChange(occasions: readonly ResponseMatrix[], options: LinearChangeOptions = {}): LinearChange {
  checkOptions(options);
  const { tolerance = DEFAULT_TOLERANCE, maxIterations = DEFAULT_MAX_ITERATIONS } = options;
  checkStopping(tolerance, maxIterations);
  const data = checkOccasions(occasions);
  const people = groupsOf(data.personIds.length, options);
  const sets = itemGroupsOf(data.itemIds, options.itemGroups);
  const layout = { sets: sets.names, groups: people.names, occasions: occasions.length };
  const strata = strataOf(occasions, people.slotOf, sets.setOf, layout);
  const problem = conditionalLikelihood(strata, layoutSize(layout), occasions.length);
  checkEstimable(problem, layout);
  const estimate = new Float64Array(layoutSize(layout));
  const run = iterateToConvergence(() => step(problem, estimate), { tolerance, maxIterations });
  const { iterations } = run;
  const converged = run.converged && maxAbs(estimate) <= MAX_EFFECT;
  const final = problem.evaluate(estimate);
  const covariance = invertPositiveDefinite(final.information, estimate.length);
  return {
    effects: effectsOf(layout, estimate, covariance),
    logLikelihood: final.logLikelihood,
    converged,
    iterations,
  };
}

/** The shape of the effects: the sets of items, the groups with the baseline first, and the number of occasions. */
interface Layout {
  readonly sets: readonly string[];
  readonly groups: readonly string[];
  readonly occasions: number;
}

/** Returns the number of effects: one for each set, each occasion after the first and each group. */
function layoutSize({ sets, groups, occasions }: Layout): number {
  return sets.length * (occasions - 1) * groups.length;
}

/** Returns the position of the effect of a set at a occasion from 1, 0-based, for a group slot, 0 the baseline. */
function effectIndex(layout: Layout, set: number, occasion: number, slot: number): number {
  return (set * (layout.occasions - 1) + (occasion - 1)) * layout.groups.length + slot;
}

/**
 * Checks the occasions and returns the persons' and items' ids they share.
 *
 * @throws `DataError` when they are not 2 to 12 response matrices of the same persons and items, with responses 0, 1 or
 * `null`.
 */
function checkOccasions(occasions: readonly ResponseMatrix[]): {
  personIds: readonly string[];
  itemIds: readonly string[];
} {
  checkArray(occasions, "The occasions must be an array of response matrices");
  if (occasions.length < 2 || occasions.length > MAX_OCCASIONS) {
    throw new DataError(`Linear change needs 2 to ${MAX_OCCASIONS} occasions, found ${occasions.length}.`);
  }
  occasions.forEach(checkResponseMatrix);
  const [first] = occasions;
  const personIds = first?.personIds ?? [];
  const itemIds = first?.itemIds ?? [];
  occasions.forEach((matrix, t) => {
    const same = (ids: readonly string[], expected: readonly string[]): boolean =>
      ids.length === expected.length && ids.every((id, i) => id === expected[i]);
    if (!same(matrix.personIds, personIds)) {
      throw new DataError(`Occasion ${t + 1} lists other persons than occasion 1, or in another order.`);
    }
    if (!same(matrix.itemIds, itemIds)) {
      throw new DataError(`Occasion ${t + 1} lists other items than occasion 1, or in another order.`);
    }
    matrix.responses.forEach((row, p) => {
      row.forEach((response, i) => {
        if (response !== null && response > 1) {
          throw new DataError(
            `The response of person "${personIds[p]}" to item "${itemIds[i]}" at occasion ${t + 1} must be 0, 1 or null, found ${response}.`,
          );
        }
      });
    });
  });
  return { personIds, itemIds };
}

/**
 * Returns the group names, the baseline first, and each person's slot among them, 0 for the baseline.
 *
 * @throws `DataError` when `groups` does not name every person or the baseline is not one of the groups.
 */
function groupsOf(persons: number, { groups, baseline }: LinearChangeOptions): { names: string[]; slotOf: Int32Array } {
  const given = groups ?? new Array<string>(persons).fill(EVERYONE);
  checkArray(given, "The option groups must be an array of group names");
  if (given.length !== persons || !given.every(isNonEmptyString)) {
    throw new DataError(`The option groups needs a group name for each of the ${persons} persons.`);
  }
  const appearing = [...new Set(given)];
  const first = baseline ?? appearing[0] ?? EVERYONE;
  if (!appearing.includes(first)) {
    throw new DataError(`The baseline ${describe(first)} is not the group of any person.`);
  }
  const names = [first, ...appearing.filter((name) => name !== first)];
  return { names, slotOf: Int32Array.from(given, (name) => names.indexOf(name)) };
}

/**
 * Returns the names of the sets of items, in the order of their first item, and the set of each item.
 *
 * @throws `DataError` when `itemGroups` is not an object that names a set for every item, or names an item that is not
 * one.
 */
function itemGroupsOf(
  itemIds: readonly string[],
  itemGroups?: Readonly<Record<string, string>>,
): { names: string[]; setOf: Int32Array } {
  if (itemGroups === undefined) return { names: [...itemIds], setOf: Int32Array.from(itemIds, (_, i) => i) };
  checkRecord(itemGroups, "The option itemGroups must be an object of set names by item id");
  for (const id of Object.keys(itemGroups)) {
    if (!itemIds.includes(id))
      throw new DataError(`The option itemGroups names ${describe(id)}, which is not an item.`);
  }
  const named = itemIds.map((id) => {
    const name: unknown = Object.hasOwn(itemGroups, id) ? itemGroups[id] : undefined;
    if (!isNonEmptyString(name)) {
      throw new DataError(`The option itemGroups needs a set name for item "${id}", found ${describe(name)}.`);
    }
    return name;
  });
  const names = [...new Set(named)];
  return { names, setOf: Int32Array.from(named, (name) => names.indexOf(name)) };
}

/**
 * The persons and items of one set, one group and one number of correct responses: the effects their patterns involve,
 * and the sum of the effect columns of the patterns observed.
 */
interface Stratum {
  /** The positions of the effects involved: the occasions after the first, for the baseline and for the group. */
  readonly effects: Int32Array;
  /** Whether the group is not the baseline, so that each occasion involves two effects. */
  readonly pairs: boolean;
  /** The number of correct responses of each pair of a person and an item in the stratum. */
  readonly sum: number;
  /** How many pairs of a person and an item. */
  count: number;
  /** The sum, over those pairs, of the effect columns of the pattern each gave. */
  readonly observed: Float64Array;
}

/** Writes the effect columns of a pattern, a bit for each occasion, 1 for a correct response, into `out`. */
function writeColumns(pattern: number, pairs: boolean, occasions: number, out: Float64Array): void {
  out.fill(0);
  const width = pairs ? 2 : 1;
  for (let t = 1; t < occasions; t++) {
    if (((pattern >>> t) & 1) === 0) continue;
    for (let j = 0; j < width; j++) out[(t - 1) * width + j] = 1;
  }
}

/** Returns the strata of the pairs of a person and an item that answered at every occasion with both responses. */
function strataOf(
  occasions: readonly ResponseMatrix[],
  slotOf: Int32Array,
  setOf: Int32Array,
  layout: Layout,
): Stratum[] {
  const strata = new Map<number, Stratum>();
  const count = occasions.length;
  const [first] = occasions;
  first?.personIds.forEach((_, p) => {
    for (let i = 0; i < setOf.length; i++) {
      const pattern = patternOf(occasions, p, i);
      if (pattern === null) continue;
      const sum = popcount(pattern);
      if (sum === 0 || sum === count) continue;
      const slot = slotOf[p] ?? 0;
      const set = setOf[i] ?? 0;
      const key = (set * layout.groups.length + slot) * (count + 1) + sum;
      let stratum = strata.get(key);
      if (stratum === undefined) {
        stratum = newStratum(layout, set, slot, sum);
        strata.set(key, stratum);
      }
      const columns = new Float64Array(stratum.observed.length);
      writeColumns(pattern, stratum.pairs, count, columns);
      stratum.count++;
      columns.forEach((value, j) => {
        stratum.observed[j] = (stratum.observed[j] ?? 0) + value;
      });
    }
  });
  return [...strata.values()];
}

/** Returns the pattern of a person's responses to an item as bits, occasion t at bit t, or `null` where one is missing. */
function patternOf(occasions: readonly ResponseMatrix[], person: number, item: number): number | null {
  let pattern = 0;
  for (let t = 0; t < occasions.length; t++) {
    const response = occasions[t]?.responses[person]?.[item];
    if (response === null || response === undefined) return null;
    pattern |= response << t;
  }
  return pattern;
}

function popcount(value: number): number {
  let bits = 0;
  for (let rest = value; rest !== 0; rest >>>= 1) bits += rest & 1;
  return bits;
}

/** Returns an empty stratum of a set, a group slot and a number of correct responses. */
function newStratum(layout: Layout, set: number, slot: number, sum: number): Stratum {
  const pairs = slot !== 0;
  const slots = pairs ? [0, slot] : [0];
  const effects = Int32Array.from({ length: (layout.occasions - 1) * slots.length }, (_, j) => {
    const occasion = Math.floor(j / slots.length) + 1;
    return effectIndex(layout, set, occasion, slots[j % slots.length] ?? 0);
  });
  return { effects, pairs, sum, count: 0, observed: new Float64Array(effects.length) };
}

/** The conditional log-likelihood, with its gradient and information, as a function of the effects. */
interface ConditionalLikelihood {
  evaluate(effects: Float64Array): { logLikelihood: number; gradient: Float64Array; information: Float64Array };
  /** The conditional log-likelihood alone. */
  value(effects: Float64Array): number;
}

/**
 * Returns the conditional likelihood of the strata: in each, the patterns of the observed sum are the outcomes, and the
 * logit of each is the sum of the effect columns of its correct responses.
 */
function conditionalLikelihood(strata: readonly Stratum[], size: number, occasions: number): ConditionalLikelihood {
  const patternsOfSum = Array.from({ length: occasions + 1 }, (_, sum) =>
    Array.from({ length: 2 ** occasions }, (_, pattern) => pattern).filter((pattern) => popcount(pattern) === sum),
  );
  const columns = new Float64Array(2 * (occasions - 1));
  const term = (stratum: Stratum, effects: Float64Array): StratumTerm => {
    const patterns = patternsOfSum[stratum.sum] ?? [];
    const logits = patterns.map((pattern) => {
      writeColumns(pattern, stratum.pairs, occasions, columns);
      return dotAt(columns, effects, stratum.effects);
    });
    const largest = Math.max(...logits);
    const weights = logits.map((logit) => Math.exp(logit - largest));
    const total = weights.reduce((sum, weight) => sum + weight, 0);
    const logLikelihood =
      dotAt(stratum.observed, effects, stratum.effects) - stratum.count * (largest + Math.log(total));
    return { patterns, probabilities: weights.map((weight) => weight / total), logLikelihood };
  };
  return {
    evaluate(effects) {
      const gradient = new Float64Array(size);
      const information = new Float64Array(size * size);
      let logLikelihood = 0;
      for (const stratum of strata) {
        const { patterns, probabilities, logLikelihood: part } = term(stratum, effects);
        logLikelihood += part;
        addDerivatives(stratum, patterns, probabilities, occasions, { gradient, information, size });
      }
      return { logLikelihood, gradient, information };
    },
    value(effects) {
      return strata.reduce((sum, stratum) => sum + term(stratum, effects).logLikelihood, 0);
    },
  };
}

/** The log-likelihood of a stratum, with the patterns of its sum and their conditional probabilities. */
interface StratumTerm {
  readonly patterns: readonly number[];
  readonly probabilities: readonly number[];
  readonly logLikelihood: number;
}

/** Returns Σ values[j] * effects[at[j]]. */
function dotAt(values: Float64Array, effects: Float64Array, at: Int32Array): number {
  let sum = 0;
  for (let j = 0; j < at.length; j++) sum += (values[j] ?? 0) * (effects[at[j] ?? 0] ?? 0);
  return sum;
}

/**
 * Adds the gradient and the information of a stratum to the totals: the observed columns less their expectation under the
 * conditional probabilities of the patterns, times the pairs, and the covariance of the columns times the pairs.
 */
function addDerivatives(
  stratum: Stratum,
  patterns: readonly number[],
  probabilities: readonly number[],
  occasions: number,
  total: { readonly gradient: Float64Array; readonly information: Float64Array; readonly size: number },
): void {
  const width = stratum.observed.length;
  const columns = new Float64Array(width);
  const expected = new Float64Array(width);
  const second = new Float64Array(width * width);
  patterns.forEach((pattern, k) => {
    writeColumns(pattern, stratum.pairs, occasions, columns);
    const probability = probabilities[k] ?? 0;
    for (let a = 0; a < width; a++) {
      expected[a] = (expected[a] ?? 0) + probability * (columns[a] ?? 0);
      for (let b = 0; b < width; b++) {
        second[a * width + b] = (second[a * width + b] ?? 0) + probability * (columns[a] ?? 0) * (columns[b] ?? 0);
      }
    }
  });
  for (let a = 0; a < width; a++) {
    const row = stratum.effects[a] ?? 0;
    total.gradient[row] = (total.gradient[row] ?? 0) + (stratum.observed[a] ?? 0) - stratum.count * (expected[a] ?? 0);
    for (let b = 0; b < width; b++) {
      const index = row * total.size + (stratum.effects[b] ?? 0);
      const covariance = (second[a * width + b] ?? 0) - (expected[a] ?? 0) * (expected[b] ?? 0);
      total.information[index] = (total.information[index] ?? 0) + stratum.count * covariance;
    }
  }
}

/**
 * Checks that the information at no effects is positive definite, so that every effect has a unique estimate.
 *
 * @throws `DataError` naming the first effect that is all zeros or a combination of those before it.
 */
function checkEstimable(problem: ConditionalLikelihood, layout: Layout): void {
  const size = layoutSize(layout);
  const { information } = problem.evaluate(new Float64Array(size));
  for (let leading = 1; leading <= size; leading++) {
    const block = new Float64Array(leading * leading);
    for (let j = 0; j < leading; j++)
      for (let k = 0; k < leading; k++) block[j * leading + k] = information[j * size + k] ?? 0;
    if (cholesky(block, leading) !== null) continue;
    const { itemGroup, occasion, group } = describeEffect(layout, leading - 1);
    throw new DataError(
      `The effect of item group ${describe(itemGroup)} at occasion ${occasion} in group ${describe(group)} has no unique estimate: no person changed on it, or those who did make it a combination of the effects before it.`,
    );
  }
}

/** Returns the set, the occasion and the group of the effect at `index`. */
function describeEffect(layout: Layout, index: number): { itemGroup: string; occasion: number; group: string } {
  const groups = layout.groups.length;
  const perSet = (layout.occasions - 1) * groups;
  const set = Math.floor(index / perSet);
  const occasion = Math.floor((index % perSet) / groups) + 2;
  return { itemGroup: layout.sets[set] ?? "", occasion, group: layout.groups[index % groups] ?? "" };
}

/**
 * Takes one Newton step with a line search from `effects`, in place, and returns the largest change of an effect. Where
 * the information is singular the step is not taken and the change is infinite, and where no length raises the
 * likelihood the effects are at the precision of the likelihood and the change is 0.
 */
function step(problem: ConditionalLikelihood, effects: Float64Array): number {
  const size = effects.length;
  const { logLikelihood, gradient, information } = problem.evaluate(effects);
  const inverse = invertPositiveDefinite(information, size);
  if (inverse === null) return Number.POSITIVE_INFINITY;
  const direction = new Float64Array(size);
  for (let j = 0; j < size; j++)
    for (let k = 0; k < size; k++)
      direction[j] = (direction[j] ?? 0) + (inverse[j * size + k] ?? 0) * (gradient[k] ?? 0);
  const trial = new Float64Array(size);
  const slope = -direction.reduce((sum, value, j) => sum + value * (gradient[j] ?? 0), 0);
  const length = backtrack(
    (t) => {
      for (let j = 0; j < size; j++) trial[j] = (effects[j] ?? 0) + t * (direction[j] ?? 0);
      return -problem.value(trial);
    },
    -logLikelihood,
    slope,
    { slack: LIKELIHOOD_ROUNDING * Math.abs(logLikelihood) },
  );
  if (length === null) return 0;
  effects.set(trial);
  return length * maxAbs(direction);
}

/** Returns the effects of the estimates in the order of the layout, with the standard errors of their covariance. */
function effectsOf(layout: Layout, estimate: Float64Array, covariance: Float64Array | null): LinearChangeEffect[] {
  const size = estimate.length;
  return Array.from(estimate, (value, index) => {
    const variance = covariance?.[index * size + index];
    return {
      ...describeEffect(layout, index),
      estimate: value,
      standardError:
        variance === undefined || !(variance >= 0) || !Number.isFinite(variance) ? null : Math.sqrt(variance),
    };
  });
}
