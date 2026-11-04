import {
  checkChoice,
  checkFinite,
  checkOptions,
  chiSquareUpperTail,
  DataError,
  describe,
  type FieldNames,
  isRecord,
  normalQuantile,
} from "@measurekit/core";
import type { EventLog } from "../data/internal.js";
import { type Counterexamples, counterexampleData, fitRelation, type IitaVersion } from "./iita.js";
import { responsePatterns } from "./patterns.js";
import { type Prerequisites, transitiveClosure } from "./surmise.js";

const DEFAULT_TAIL = "two-sided";
const DEFAULT_MEAN = 0;
const DEFAULT_CONFIDENCE = 0.95;
const DEFAULT_VERSION = "minimized";

/** Options of {@link diffTest}. */
export interface DiffTestOptions {
  /**
   * The quasi-order to compare with, as `iita` returns it. The test is then of the difference of the two fit measures,
   * on the same learners.
   *
   * @defaultValue none: a test of the fit measure of one quasi-order
   */
  readonly alternative?: Prerequisites;
  /**
   * The alternative hypothesis: the estimate differs from `mean`, is below it or is above it.
   *
   * @defaultValue `"two-sided"`
   */
  readonly tail?: "two-sided" | "less" | "greater";
  /**
   * The value of the estimate under the null hypothesis.
   *
   * @defaultValue `0`
   */
  readonly mean?: number;
  /**
   * The confidence level of the interval, above 0 and below 1.
   *
   * @defaultValue `0.95`
   */
  readonly confidence?: number;
  /**
   * The fit measure: `"minimized"`, with the error rate that minimizes it, or `"corrected"`, with Sargin and Ünlü's
   * corrected expected counterexamples.
   *
   * @defaultValue `"minimized"`
   */
  readonly version?: Exclude<IitaVersion, "original">;
}

const DIFF_TEST_OPTIONS: FieldNames<DiffTestOptions> = {
  alternative: true,
  tail: true,
  mean: true,
  confidence: true,
  version: true,
};

/** The result of {@link diffTest}. */
export interface DiffTest {
  /** The fit measure `diff` over n², or the difference of the two, n the number of learners. */
  readonly estimate: number;
  /** The statistic: √n (estimate - mean) over the standard deviation of the estimate, standard normal under the null. */
  readonly z: number;
  /** The probability, under the null hypothesis, of a statistic at least as extreme in the direction of `tail`. */
  readonly pValue: number;
  /** The confidence interval of the estimate; one-sided for a `tail` of `"less"` or `"greater"`, with an infinite end. */
  readonly interval: readonly [number, number];
}

/**
 * Tests the fit measure `diff` of a quasi-order on learners' responses, or the difference of the measures of two
 * quasi-orders: the Z test of the DAKS package, with its variance corrected and the usual meaning of a tail.
 *
 * @remarks
 * The estimate is `diff` of inductive item tree analysis over n², n the number of learners. Its variance is that of the
 * delta method for the multinomial distribution of the response patterns, which is the mean squared influence of a
 * learner's pattern on the estimate, less the squared mean. Two quasi-orders are fitted to the same learners, so the
 * variance is that of the difference, with the covariance of the two estimates; DAKS adds the two variances. The
 * quasi-orders are given as prerequisites, which are closed under transitivity, and need at least one implication. With
 * `"greater"` the p-value is the probability of a larger statistic; DAKS returns the probability of a smaller one.
 * Each learner's first outcome at each item is used; every learner must have answered every item, with outcomes 0 or
 * 1, and every item must be solved by someone.
 *
 * @param log - The responses.
 * @param prerequisites - The quasi-order: each item of the log is a key, with its prerequisites.
 * @param options - The alternative, the tail, the mean to test against, the level of the interval and the version of
 *   the measure.
 * @returns The estimate, its z statistic, the probability and the interval.
 * @throws `DataError` when the log is malformed, a learner has not answered every item, an outcome is not 0 or 1, there
 * are fewer than 2 or more than 1,000 items, an item is solved by no learner, a quasi-order is not over the items of the log or has no
 * implication, an option is invalid, or the estimate has variance 0, as when every learner has the same answers or the
 * two quasi orders are the same.
 * @see {@link https://doi.org/10.18637/jss.v037.i02 | Ünlü and Sargin (2010)}
 */
export function diffTest(log: EventLog, prerequisites: Prerequisites, options: DiffTestOptions = {}): DiffTest {
  checkOptions(options, DIFF_TEST_OPTIONS);
  const { alternative, tail = DEFAULT_TAIL, mean = DEFAULT_MEAN, confidence = DEFAULT_CONFIDENCE } = options;
  const { version = DEFAULT_VERSION } = options;
  checkChoice(tail, "The option tail", ["two-sided", "less", "greater"]);
  checkChoice(version, "The option version", ["minimized", "corrected"]);
  checkFinite(mean, "The option mean");
  if (!(typeof confidence === "number" && confidence > 0 && confidence < 1)) {
    throw new DataError(`The option confidence must be above 0 and below 1, found ${describe(confidence)}.`);
  }
  const { items, patterns } = responsePatterns(log, "The difference test");
  if (items.length < 2) throw new DataError("The difference test needs at least 2 items.");
  const data = counterexampleData(items, patterns);
  const first = measure(relationOf(prerequisites, items, "The prerequisites"), data, version, patterns);
  const second =
    alternative === undefined
      ? null
      : measure(relationOf(alternative, items, "The alternative prerequisites"), data, version, patterns);
  const estimate = first.estimate - (second?.estimate ?? 0);
  const influence = first.influence.map((value, learner) => value - (second?.influence[learner] ?? 0));
  const variance = varianceOf(influence);
  if (!(variance > 0)) {
    throw new DataError(
      "The estimate has variance 0, as when every learner has the same responses or the two quasi orders are the same, so it cannot be tested.",
    );
  }
  const standardError = Math.sqrt(variance / data.n);
  const z = (estimate - mean) / standardError;
  return { estimate, z, pValue: pValueOf(z, tail), interval: intervalOf(estimate, standardError, confidence, tail) };
}

/**
 * Returns the relation of prerequisites as an m × m matrix over `items`: 1 at i × m + j when i is a prerequisite of j,
 * directly or through others; the diagonal is 1.
 */
function relationOf(prerequisites: Prerequisites, items: readonly string[], label: string): Uint8Array {
  if (!isRecord(prerequisites)) {
    throw new DataError(`${label} must be an object of lists of items by item, found ${describe(prerequisites)}.`);
  }
  const keys = Object.keys(prerequisites);
  for (const item of items) {
    if (!Object.hasOwn(prerequisites, item)) throw new DataError(`${label} have no key for item "${item}" of the log.`);
  }
  const stray = keys.find((key) => !items.includes(key));
  if (stray !== undefined) throw new DataError(`${label} have key "${stray}", which is not an item of the log.`);
  const below = transitiveClosure(prerequisites, keys);
  const m = items.length;
  const at = items.map((item) => keys.indexOf(item));
  const relation = new Uint8Array(m * m);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) relation[i * m + j] = below[(at[j] ?? 0) * m + (at[i] ?? 0)] ?? 0;
  }
  if (relation.reduce((sum, value) => sum + value, 0) === m) {
    throw new DataError(`${label} need at least one implication between two items.`);
  }
  return relation;
}

/** Returns the estimate `diff` / n² of a relation and each learner's influence on it. */
function measure(
  relation: Uint8Array,
  data: Counterexamples,
  version: Exclude<IitaVersion, "original">,
  patterns: readonly (readonly number[])[],
): { estimate: number; influence: Float64Array } {
  const { diff, errorRate } = fitRelation(relation, data, version);
  const { m, n } = data;
  const slopes = slopesOf(relation, data, version, errorRate);
  const influence = Float64Array.from(patterns, (pattern) => {
    let sum = 0;
    for (let j = 0; j < m; j++) {
      if (pattern[j] !== 1) continue;
      sum += slopes.rho[j] ?? 0;
      for (let i = 0; i < m; i++) if (pattern[i] === 0) sum += slopes.counter[i * m + j] ?? 0;
    }
    return sum;
  });
  return { estimate: diff / (n * n), influence };
}

/**
 * The partial derivatives of the estimate g = Σ (C_ij - e_ij)² / (m (m - 1)), with C_ij the proportion of learners who
 * fail i and solve j and e_ij the expected proportion. A learner with pattern t moves g by Σ_j `rho`_j t_j + Σ_ij
 * `counter`_ij (1 - t_i) t_j.
 */
interface Slopes {
  /** With respect to each solution rate. */
  readonly rho: Float64Array;
  /** With respect to each C_ij, at i × m + j. */
  readonly counter: Float64Array;
}

/** How a pair of items stands in a relation: i ⊑ j, only j ⊑ i, or neither. */
type Link = "forward" | "backward" | "none";

function linkOf(relation: Uint8Array, m: number, i: number, j: number): Link {
  if (relation[i * m + j] === 1) return "forward";
  return relation[j * m + i] === 1 ? "backward" : "none";
}

/** Returns the expected proportion e_ij of learners who fail i and solve j, given the error rate γ. */
function expectedProportion(link: Link, pi: number, pj: number, gamma: number): number {
  if (link === "forward") return gamma * pj;
  return link === "backward" ? pj - pi + gamma * pi : (1 - pi) * pj;
}

/**
 * Returns the slopes of g. The minimized error rate minimizes g, so by the envelope theorem its change adds nothing;
 * the corrected one, the mean of C_ij / ρ_j over the implications, adds its own.
 */
function slopesOf(
  relation: Uint8Array,
  data: Counterexamples,
  version: Exclude<IitaVersion, "original">,
  gamma: number,
): Slopes {
  const { b, p: rho, m, n } = data;
  const slopes: Slopes = { rho: new Float64Array(m), counter: new Float64Array(m * m) };
  let gammaSlope = 0;
  let implications = 0;
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) {
      if (i === j) continue;
      const pi = rho[i] ?? 0;
      const pj = rho[j] ?? 0;
      const link = linkOf(relation, m, i, j);
      const residual = (2 / (m * (m - 1))) * ((b[i * m + j] ?? 0) / n - expectedProportion(link, pi, pj, gamma));
      slopes.counter[i * m + j] = residual;
      gammaSlope -= residual * addExpectedSlopes(slopes.rho, link, { i, j, pi, pj, gamma }, residual);
      implications += link === "forward" ? 1 : 0;
    }
  }
  if (version === "corrected") addErrorRateSlopes(relation, data, gammaSlope / implications, slopes);
  return slopes;
}

/**
 * Subtracts from the slopes with respect to ρ the residual times the derivative of e_ij, and returns the derivative of e_ij
 * with respect to γ.
 */
function addExpectedSlopes(
  rho: Float64Array,
  link: Link,
  { i, j, pi, pj, gamma }: { i: number; j: number; pi: number; pj: number; gamma: number },
  residual: number,
): number {
  if (link === "forward") {
    rho[j] = (rho[j] ?? 0) - residual * gamma;
    return pj;
  }
  if (link === "backward") {
    rho[j] = (rho[j] ?? 0) - residual;
    rho[i] = (rho[i] ?? 0) - residual * (gamma - 1);
    return pi;
  }
  rho[i] = (rho[i] ?? 0) + residual * pj;
  rho[j] = (rho[j] ?? 0) - residual * (1 - pi);
  return 0;
}

/** Adds the part of the slopes that goes through the corrected error rate, the mean of C_ij / ρ_j over implications. */
function addErrorRateSlopes(
  relation: Uint8Array,
  { b, p: rho, m, n }: Counterexamples,
  perImplication: number,
  slopes: Slopes,
): void {
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) {
      if (i === j || relation[i * m + j] !== 1) continue;
      const pj = rho[j] ?? 0;
      slopes.counter[i * m + j] = (slopes.counter[i * m + j] ?? 0) + perImplication / pj;
      slopes.rho[j] = (slopes.rho[j] ?? 0) - (perImplication * (b[i * m + j] ?? 0)) / (n * pj * pj);
    }
  }
}

/** Returns the variance of the values about their mean, computed about the first so that equal values give 0. */
function varianceOf(values: Float64Array): number {
  const origin = values[0] ?? 0;
  let sum = 0;
  let squares = 0;
  for (const value of values) {
    sum += value - origin;
    squares += (value - origin) ** 2;
  }
  return squares / values.length - (sum / values.length) ** 2;
}

/** Returns the probability that a standard normal variable is at least `z`. */
function upperTail(z: number): number {
  const half = chiSquareUpperTail(z * z, 1) / 2;
  return z >= 0 ? half : 1 - half;
}

function pValueOf(z: number, tail: "two-sided" | "less" | "greater"): number {
  if (tail === "greater") return upperTail(z);
  if (tail === "less") return upperTail(-z);
  return chiSquareUpperTail(z * z, 1);
}

function intervalOf(
  estimate: number,
  standardError: number,
  confidence: number,
  tail: "two-sided" | "less" | "greater",
): readonly [number, number] {
  if (tail === "two-sided") {
    const half = normalQuantile(1 - (1 - confidence) / 2) * standardError;
    return [estimate - half, estimate + half];
  }
  const half = normalQuantile(confidence) * standardError;
  return tail === "greater" ? [estimate - half, Number.POSITIVE_INFINITY] : [Number.NEGATIVE_INFINITY, estimate + half];
}
