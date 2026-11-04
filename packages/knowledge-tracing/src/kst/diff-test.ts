import {
  checkChoice,
  checkFinite,
  checkOptions,
  DataError,
  describe,
  type FieldNames,
  isRecord,
  normalCdf,
  normalQuantile,
} from "@measurekit/core";
import type { EventLog } from "../data/internal.js";
import {
  type Counterexamples,
  fitRelation,
  forEachObservedImplication,
  type IitaVersion,
  isObservedPair,
  observedPairCount,
} from "./iita.js";
import { type Answers, answerSets } from "./patterns.js";
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
 * quasi-orders, from complete or incomplete response patterns: the Z test of the DAKS package, with its variance
 * corrected, the usual meaning of a tail, and missing responses taken pairwise.
 *
 * @remarks
 * The estimate is `diff` of inductive item tree analysis over n², n the number of learners: the mean, over the ordered
 * pairs of items, of the squared difference of the observed and the expected share of learners who fail one item and
 * solve the other. Learners may have answered some items only, as after an adaptive test or a booklet design: each
 * pair's share counts the learners who answered both, each item's solution rate the learners who answered it, and a pair
 * that no learner answered both of leaves the mean. With every item answered by every learner, this is DAKS's estimate.
 * Its variance is that of the delta method, the sum over learners of the squared influence of each learner's answers on
 * the estimate, over n²: the multinomial delta method of the response patterns when they are complete. The shares are
 * those of the population when whether a learner answered an item does not depend on the learner's answers, missing
 * completely at random; an adaptive test that chooses items by earlier answers biases them. Two quasi-orders are fitted to
 * the same learners, so the variance is that of the difference, with the covariance of the two estimates; DAKS adds the
 * two variances. The quasi-orders are given as prerequisites, which are closed under transitivity, and need an
 * implication whose two items some learner answered. With `"greater"` the p-value is the probability of a larger
 * statistic; DAKS returns the probability of a smaller one. Each learner's first outcome at each item is used, 0 or 1,
 * and every item must be solved by someone.
 *
 * @param log - The responses.
 * @param prerequisites - The quasi-order: each item of the log is a key, with its prerequisites.
 * @param options - The alternative, the tail, the mean to test against, the level of the interval and the version of
 *   the measure.
 * @returns The estimate, its z statistic, the probability and the interval.
 * @throws `DataError` when the log is malformed, an outcome is not 0 or 1, there are fewer than 2 or more than 1,000
 * items, an item is solved by no learner, a quasi-order is not over the items of the log or has no implication whose two
 * items some learner answered, an option is invalid, or the estimate has variance 0, as when every learner has the same
 * answers or the two quasi orders are the same.
 * @see {@link https://doi.org/10.18637/jss.v037.i02 | Ünlü and Sargin (2010)}
 * @see {@link https://doi.org/10.1002/9781119482260 | Little and Rubin (2019)}
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
  const { items, learners } = answerSets(log, "The difference test");
  if (items.length < 2) throw new DataError("The difference test needs at least 2 items.");
  const data = pairData(items, learners);
  const first = measure(relationOf(prerequisites, items, "The prerequisites"), data, version);
  const second =
    alternative === undefined
      ? null
      : measure(relationOf(alternative, items, "The alternative prerequisites"), data, version);
  const estimate = second === null ? first.estimate : first.estimate - second.estimate;
  // The slopes of the difference of two estimates are the differences of their slopes.
  const slopes =
    second === null
      ? first.slopes
      : {
          rho: first.slopes.rho.map((slope, k) => slope - (second.slopes.rho[k] ?? 0)),
          counter: first.slopes.counter.map((slope, k) => slope - (second.slopes.counter[k] ?? 0)),
        };
  const squares = squaredInfluence(learners, data, slopes);
  if (!(squares > 0)) {
    throw new DataError(
      "The estimate has variance 0, as when every learner has the same responses or the two quasi orders are the same, so it cannot be tested.",
    );
  }
  const standardError = Math.sqrt(squares) / data.n;
  const z = (estimate - mean) / standardError;
  return { estimate, z, pValue: pValueOf(z, tail), interval: intervalOf(estimate, standardError, confidence, tail) };
}

/** The counterexample data of the learners' answers, with the number of learners behind each share. */
interface PairData extends Counterexamples {
  /** At i × m + j, the learners who answered both i and j. */
  readonly pairLearners: Int32Array;
  /** The learners who answered each item. */
  readonly itemLearners: Int32Array;
}

/** The counts of learners' answers: by pair, those who answered both and those who fail the first and solve the second. */
interface AnswerCounts {
  /** At i × m + j, the learners who answered both i and j. */
  readonly pairLearners: Int32Array;
  /** At i × m + j, the learners who fail i and solve j. */
  readonly failSolve: Int32Array;
  /** The learners who answered each item. */
  readonly itemLearners: Int32Array;
  /** The learners who solved each item. */
  readonly solved: Int32Array;
}

/** Adds a learner's answers to the counts. */
function addAnswers(answers: Answers, counts: AnswerCounts, m: number): void {
  const { itemLearners, solved } = counts;
  let a = 0;
  for (const i of answers.items) {
    itemLearners[i] = (itemLearners[i] ?? 0) + 1;
    if (answers.outcomes[a] === 1) solved[i] = (solved[i] ?? 0) + 1;
    addPairs(answers, a, i * m, counts);
    a++;
  }
}

/** Adds to the counts the pairs of a learner's answer `a`, at the row `row` of the tables, with each other answer. */
function addPairs(
  { items, outcomes }: Answers,
  a: number,
  row: number,
  { pairLearners, failSolve }: AnswerCounts,
): void {
  const fails = outcomes[a] === 0;
  let c = 0;
  for (const j of items) {
    if (c !== a) {
      pairLearners[row + j] = (pairLearners[row + j] ?? 0) + 1;
      if (fails && outcomes[c] === 1) failSolve[row + j] = (failSolve[row + j] ?? 0) + 1;
    }
    c++;
  }
}

/**
 * Returns the counterexample data of learners' answers, complete or not: b_ij is n times the share of the learners who
 * answered both i and j who fail i and solve j, and p_i the share of those who answered i who solve it.
 *
 * @throws `DataError` when an item is solved by no learner.
 */
function pairData(items: readonly string[], learners: readonly Answers[]): PairData {
  const m = items.length;
  const n = learners.length;
  const counts: AnswerCounts = {
    pairLearners: new Int32Array(m * m),
    failSolve: new Int32Array(m * m),
    itemLearners: new Int32Array(m),
    solved: new Int32Array(m),
  };
  for (const answers of learners) addAnswers(answers, counts, m);
  const { pairLearners, failSolve, itemLearners, solved } = counts;
  const p = items.map((item, i) => {
    if (solved[i] === 0) throw new DataError(`Item "${item}" is solved by no learner, so IITA cannot weigh it.`);
    return (solved[i] ?? 0) / (itemLearners[i] ?? 1);
  });
  const complete = itemLearners.every((count) => count === n);
  // n b / n_ij is b itself when every learner answered both, so complete answers give DAKS's counts exactly.
  const b = Float64Array.from(failSolve, (count, k) => (count === 0 ? 0 : (count * n) / (pairLearners[k] ?? 1)));
  const observed = complete ? null : Uint8Array.from(pairLearners, (count) => (count > 0 ? 1 : 0));
  return { b, p, m, n, observed, pairLearners, itemLearners };
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

/**
 * Returns the estimate `diff` / n² of a relation and its slopes.
 *
 * @throws `DataError` when no learner answered both items of any implication of the relation.
 */
function measure(
  relation: Uint8Array,
  data: PairData,
  version: Exclude<IitaVersion, "original">,
): { estimate: number; slopes: Slopes } {
  if (!hasObservedImplication(relation, data)) {
    throw new DataError(
      "No learner answered both items of an implication of the prerequisites, so the responses cannot test them.",
    );
  }
  const { diff, errorRate } = fitRelation(relation, data, version);
  return { estimate: diff / (data.n * data.n), slopes: slopesOf(relation, data, version, errorRate) };
}

/** Returns whether some learner answered both items of an implication i ⊑ j, i ≠ j, of the relation. */
function hasObservedImplication(relation: Uint8Array, data: PairData): boolean {
  let found = false;
  forEachObservedImplication(relation, data, () => {
    found = true;
  });
  return found;
}

/**
 * Returns the sum over learners of the squared influence of their answers on the estimate, whose square root over n is
 * its standard error.
 */
function squaredInfluence(learners: readonly Answers[], data: PairData, slopes: Slopes): number {
  let sum = 0;
  for (const answers of learners) sum += influenceOf(answers, slopes, data) ** 2;
  return sum;
}

/**
 * Returns a learner's influence on the estimate. A learner who answered items i and j moves C_ij, a share of the n_ij
 * learners who answered both, by (x - C_ij) / n_ij, x 1 when the learner fails i and solves j; and moves ρ_j, a share of
 * the n_j who answered j, by (t_j - ρ_j) / n_j. Times n, these weigh the slopes into the influence.
 */
function influenceOf(answers: Answers, slopes: Slopes, data: PairData): number {
  const { p: rho, m, n, itemLearners } = data;
  let influence = 0;
  let a = 0;
  for (const i of answers.items) {
    const solved = answers.outcomes[a] === 1 ? 1 : 0;
    influence += ((slopes.rho[i] ?? 0) * n * (solved - (rho[i] ?? 0))) / (itemLearners[i] ?? 1);
    influence += pairInfluence(answers, a, i * m, slopes, data);
    a++;
  }
  return influence;
}

/** Returns the influence of a learner through the pairs of the learner's answer `a`, at the row `row`, with each other. */
function pairInfluence({ items, outcomes }: Answers, a: number, row: number, slopes: Slopes, data: PairData): number {
  const { b, n, pairLearners } = data;
  const fails = outcomes[a] === 0;
  let influence = 0;
  let c = 0;
  for (const j of items) {
    if (c !== a) {
      const counter = fails && outcomes[c] === 1 ? n : 0;
      influence += ((slopes.counter[row + j] ?? 0) * (counter - (b[row + j] ?? 0))) / (pairLearners[row + j] ?? 1);
    }
    c++;
  }
  return influence;
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
  const pairs = observedPairCount(data);
  let gammaSlope = 0;
  let implications = 0;
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) {
      if (i === j || !isObservedPair(data, i, j)) continue;
      const pi = rho[i] ?? 0;
      const pj = rho[j] ?? 0;
      const link = linkOf(relation, m, i, j);
      const residual = (2 / pairs) * ((b[i * m + j] ?? 0) / n - expectedProportion(link, pi, pj, gamma));
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
function addErrorRateSlopes(relation: Uint8Array, data: Counterexamples, perImplication: number, slopes: Slopes): void {
  const { b, p: rho, m, n } = data;
  forEachObservedImplication(relation, data, (i, j) => {
    const pj = rho[j] ?? 0;
    slopes.counter[i * m + j] = (slopes.counter[i * m + j] ?? 0) + perImplication / pj;
    slopes.rho[j] = (slopes.rho[j] ?? 0) - (perImplication * (b[i * m + j] ?? 0)) / (n * pj * pj);
  });
}

function pValueOf(z: number, tail: "two-sided" | "less" | "greater"): number {
  if (tail === "greater") return normalCdf(-z);
  if (tail === "less") return normalCdf(z);
  return 2 * normalCdf(-Math.abs(z));
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
