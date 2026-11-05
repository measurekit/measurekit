import {
  checkArray,
  checkChoice,
  checkFinite,
  checkOptions,
  checkRecord,
  DataError,
  describe,
  type FieldNames,
  isRecord,
  mean,
} from "@measurekit/core";
import { standardDeviation } from "../math/internal.js";
import { type AutomatedScore, checkAutomatedScores } from "./automated-scores.js";

/** How a mapping from machine to human scores is fitted. */
export type ScoreMappingMethod = "linear" | "isotonic";

/** Options of {@link scoreMapping}. */
export interface ScoreMappingOptions {
  /**
   * `"linear"`, the line that gives the machine scores the human scores' mean and standard deviation, as RSMTool scales
   * its predictions; or `"isotonic"`, the non-decreasing function nearest the human scores in least squares.
   *
   * @defaultValue `"linear"`
   */
  readonly method?: ScoreMappingMethod;
}

const SCORE_MAPPING_OPTIONS: FieldNames<ScoreMappingOptions> = { method: true };

/** A point of an isotonic mapping: a machine score and the human score it maps to. */
export interface ScoreKnot {
  /** The machine score. */
  readonly machine: number;
  /** The human score it maps to. */
  readonly human: number;
}

/** A linear mapping from machine to human scores: human = slope × machine + intercept. */
export interface LinearScoreMapping {
  /** The method that fitted it. */
  readonly method: "linear";
  /** The human SD over the machine SD. */
  readonly slope: number;
  /** The human mean less the slope times the machine mean. */
  readonly intercept: number;
}

/** An isotonic mapping from machine to human scores, linear between its knots and flat beyond them. */
export interface IsotonicScoreMapping {
  /** The method that fitted it. */
  readonly method: "isotonic";
  /** One knot for each distinct machine score of the fit, ascending, the human scores non-decreasing. */
  readonly knots: readonly ScoreKnot[];
}

/** A mapping from machine to human scores, as `scoreMapping` fits it and `mappedScore` applies it. */
export type ScoreMapping = LinearScoreMapping | IsotonicScoreMapping;

/**
 * Returns the mapping from an automated scorer's scores to the human scale, fitted to responses that both scored: a line
 * that matches the human scores' mean and SD, or an isotonic regression.
 *
 * @remarks
 * `"linear"` gives the machine scores the mean and standard deviation of the human ones, as RSMTool scales its
 * predictions, so that the SD ratio and the SMD of the mapped scores are 1 and 0 on the data it is fitted to; least
 * squares would shrink them toward the mean. `"isotonic"` is the non-decreasing function of the machine score nearest the
 * human scores in least squares, by the pool-adjacent-violators algorithm, as R's `isoreg`: subjects with the same
 * machine score share one value, their mean human score weighted by their number. Machine and human scores are finite
 * numbers; categories are not needed. Apply the mapping with `mappedScore`, then round to the scale.
 *
 * @param scores - The subjects: each one's machine and human score.
 * @param options - The method.
 * @returns The slope and intercept of the line, or the knots of the isotonic function.
 * @throws `DataError` when there are fewer than two subjects, a score is not a finite number, the method is unknown,
 * for `"linear"`, the machine or the human scores are all equal, or the scores are so large that the mapping is not
 * finite.
 * @see {@link https://doi.org/10.1111/j.1745-3992.2011.00223.x | Williamson, Xi and Breyer (2012)}
 * @see {@link https://doi.org/10.1214/aoms/1177728423 | Ayer et al. (1955)}
 *
 * @example
 * ```ts
 * import { mappedScore, scoreMapping } from "@measurekit/psychometrics/rater";
 *
 * // A rater whose scores run low at the top of a scale of 1 to 4.
 * const scores = [
 *   { machine: 1, human: 1 },
 *   { machine: 2, human: 2 },
 *   { machine: 2, human: 3 },
 *   { machine: 3, human: 4 },
 *   { machine: 3, human: 3 },
 *   { machine: 2.5, human: 4 },
 * ];
 * const isotonic = scoreMapping(scores, { method: "isotonic" });
 * isotonic.method === "isotonic" && isotonic.knots.map(({ human }) => human); // => [1, 2.5, 3.6667, 3.6667]
 * mappedScore(isotonic, 2.25); // => 3.0833, between the knots of 2 and 2.5
 * ```
 */
export function scoreMapping(scores: readonly AutomatedScore[], options: ScoreMappingOptions = {}): ScoreMapping {
  checkOptions(options, SCORE_MAPPING_OPTIONS);
  const { method = "linear" } = options;
  checkChoice(method, "The option method", ["linear", "isotonic"]);
  const { machine, human } = checkAutomatedScores(scores, { categories: false, minSubjects: 2 });
  if (method === "isotonic") {
    const knots = isotonicKnots(machine, human);
    if (!knots.every(({ human: value }) => Number.isFinite(value))) throw notFinite();
    return { method, knots };
  }
  const machineSd = standardDeviation(machine);
  const humanSd = standardDeviation(human);
  if (!(machineSd > 0 && humanSd > 0)) {
    throw new DataError("A linear mapping needs machine and human scores that vary.");
  }
  const slope = humanSd / machineSd;
  const intercept = mean(human) - slope * mean(machine);
  if (!(Number.isFinite(slope) && Number.isFinite(intercept))) throw notFinite();
  return { method, slope, intercept };
}

/** Returns the error of a mapping whose sums of scores overflow. */
function notFinite(): DataError {
  return new DataError("The score mapping is not finite: the scores are too large; rescale them.");
}

/**
 * Returns the knots of the isotonic regression of the human scores on the machine scores: the distinct machine scores,
 * each with the mean of its human scores weighted by their number, pooled with its neighbors while they decrease.
 */
function isotonicKnots(machine: Float64Array, human: Float64Array): ScoreKnot[] {
  const { xs, sums, counts } = distinctScores(machine, human);
  // Blocks of the pool-adjacent-violators algorithm: each its weighted sum, its weight and the knots it spans.
  const blocks = { sums: [] as number[], counts: [] as number[], spans: [] as number[] };
  for (let k = 0; k < xs.length; k++) {
    blocks.sums.push(sums[k] ?? 0);
    blocks.counts.push(counts[k] ?? 0);
    blocks.spans.push(1);
    while (blocks.sums.length > 1 && mergeable(blocks.sums, blocks.counts)) poolLast(blocks);
  }
  const knots: ScoreKnot[] = [];
  blocks.sums.forEach((sum, block) => {
    const value = sum / (blocks.counts[block] ?? 1);
    for (let k = 0; k < (blocks.spans[block] ?? 0); k++) knots.push({ machine: xs[knots.length] ?? 0, human: value });
  });
  return knots;
}

/**
 * Returns the distinct machine scores, ascending, with the sum and the number of the human scores at each. The scores
 * are sorted as numbers, natively, and each subject finds its score's place in a map, which costs less than sorting the
 * subjects by a comparison.
 */
function distinctScores(machine: Float64Array, human: Float64Array) {
  const sorted = Float64Array.from(machine).sort();
  const xs: number[] = [];
  const place = new Map<number, number>();
  for (const x of sorted) {
    if (xs.at(-1) === x) continue;
    place.set(x, xs.length);
    xs.push(x);
  }
  const sums = new Float64Array(xs.length);
  const counts = new Float64Array(xs.length);
  for (let subject = 0; subject < machine.length; subject++) {
    const k = place.get(machine[subject] ?? 0) ?? 0;
    sums[k] = (sums[k] ?? 0) + (human[subject] ?? 0);
    counts[k] = (counts[k] ?? 0) + 1;
  }
  return { xs, sums, counts };
}

/** Pools the last block into the one before it. */
function poolLast(blocks: { sums: number[]; counts: number[]; spans: number[] }): void {
  const sum = blocks.sums.pop() ?? 0;
  const count = blocks.counts.pop() ?? 0;
  const span = blocks.spans.pop() ?? 0;
  const last = blocks.sums.length - 1;
  blocks.sums[last] = (blocks.sums[last] ?? 0) + sum;
  blocks.counts[last] = (blocks.counts[last] ?? 0) + count;
  blocks.spans[last] = (blocks.spans[last] ?? 0) + span;
}

/** Returns whether the last block's mean is below the one before it, so the two pool. */
function mergeable(sums: readonly number[], counts: readonly number[]): boolean {
  const last = sums.length - 1;
  return (sums[last] ?? 0) * (counts[last - 1] ?? 0) < (sums[last - 1] ?? 0) * (counts[last] ?? 0);
}

/**
 * Returns the human score that a mapping gives a machine score: on the line, or, for an isotonic mapping, by linear
 * interpolation between its knots and flat beyond the first and the last, as R's `approx` with `rule = 2`.
 *
 * @param mapping - The mapping, as `scoreMapping` returns it.
 * @param score - The machine score, a finite number.
 * @returns The mapped score, unrounded.
 * @throws `DataError` when the mapping is not one that `scoreMapping` returns, with knots of finite numbers ascending in
 * the machine score and not descending in the human score, the score is not finite, or the line maps it beyond the
 * largest number.
 */
export function mappedScore(mapping: ScoreMapping, score: number): number {
  checkRecord(mapping, "The mapping must be one that scoreMapping returns");
  checkFinite(score, "The machine score");
  if (mapping.method === "linear") {
    checkFinite(mapping.slope, "The mapping's slope");
    checkFinite(mapping.intercept, "The mapping's intercept");
    const mapped = mapping.slope * score + mapping.intercept;
    if (!Number.isFinite(mapped)) {
      throw new DataError(
        `The mapped score of ${score} is not finite: the score or the slope is too large; rescale them.`,
      );
    }
    return mapped;
  }
  // A mapping from JSON may name another method, which the type does not allow.
  const method: unknown = mapping.method;
  if (method !== "isotonic") {
    throw new DataError(`The mapping's method must be "linear" or "isotonic", found ${describe(method)}.`);
  }
  const { knots } = mapping;
  checkKnots(knots);
  let low = 0;
  let high = knots.length - 1;
  const first = knots[0];
  const last = knots[high];
  if (first === undefined || last === undefined) throw new DataError("An isotonic mapping needs at least one knot.");
  if (score <= first.machine) return first.human;
  if (score >= last.machine) return last.human;
  while (high - low > 1) {
    const middle = (low + high) >>> 1;
    if ((knots[middle]?.machine ?? 0) <= score) low = middle;
    else high = middle;
  }
  const below = knots[low] ?? first;
  const above = knots[high] ?? last;
  return below.human + ((score - below.machine) * (above.human - below.human)) / (above.machine - below.machine);
}

/**
 * Checks the knots of an isotonic mapping.
 *
 * @throws `DataError` when they are not records of finite numbers, ascending in the machine score and not descending in
 * the human score.
 */
function checkKnots(knots: readonly ScoreKnot[]): void {
  checkArray(knots, "An isotonic mapping's knots must be an array of { machine, human }");
  knots.forEach((knot, k) => {
    const previous = knots[k - 1];
    const valid =
      isRecord(knot) &&
      Number.isFinite(knot.machine) &&
      Number.isFinite(knot.human) &&
      (previous === undefined || (knot.machine > previous.machine && knot.human >= previous.human));
    if (!valid) {
      throw new DataError(
        `Knot ${k + 1} of the mapping must have finite scores, its machine score above the knot before and its human score not below, found ${describe(knot)}.`,
      );
    }
  });
}
