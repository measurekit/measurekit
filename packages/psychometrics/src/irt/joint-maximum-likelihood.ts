import {
  checkOptions,
  checkStopping,
  DataError,
  iterateToConvergence,
  type LogisticItem,
  largestChange,
  partialCreditProbabilities,
  type ThresholdItem,
} from "@measurekit/core";
import { checkResponseMatrix, type ResponseMatrix } from "../data/internal.js";
import { findIncreasingRoot, invert, type NewtonOptions, newton } from "../math/internal.js";
import type { CalibratedItem } from "./calibration.js";
import { oneParameterLogistic } from "./logistic.js";
import { partialCredit, partialCreditInformation, stepStarts } from "./partial-credit.js";
import { logProbability } from "./probability.js";
import { categoryTallies, checkCategories, compressResponses, NO_RESPONSES, responseCodes } from "./responses.js";

/** Options of {@link jointMaximumLikelihood}. */
export interface JointMaximumLikelihoodOptions {
  /**
   * Estimation has converged when no theta or threshold changes by more than this in an iteration.
   *
   * @defaultValue `1e-4`
   */
  readonly tolerance?: number;
  /**
   * The most iterations to run.
   *
   * @defaultValue `500`
   */
  readonly maxIterations?: number;
}

/** A person's theta from joint maximum likelihood, with its standard error. */
export interface PersonEstimate {
  /** The person id. */
  readonly id: string;
  /** The estimate on the Rasch scale; `null` for a person without responses. */
  readonly theta: number | null;
  /** 1 / √I(θ) from the information of the answered items; `null` for a person without responses. */
  readonly standardError: number | null;
}

/** Rasch and partial credit items and person abilities estimated jointly. */
export interface JointCalibration {
  /**
   * The items, in item order, on the Rasch scale (a = 1) after the bias correction: dichotomous items as 1PL items,
   * the others as PCM items. Standard errors are `null` for an item whose information matrix is not invertible.
   */
  readonly items: readonly CalibratedItem<LogisticItem | ThresholdItem>[];
  /** The persons, in person order, with the estimates of the joint solution, before the bias correction. */
  readonly persons: readonly PersonEstimate[];
  /** The joint log-likelihood of the responses at the joint solution, before the bias correction. */
  readonly logLikelihood: number;
  /** Whether the estimates reached `tolerance` within `maxIterations`. */
  readonly converged: boolean;
  /** Iterations run. */
  readonly iterations: number;
}

const DEFAULT_TOLERANCE = 1e-4;
const DEFAULT_MAX_ITERATIONS = 500;
/** The score points by which a total score of 0 or of the maximum moves inward: TAM's `adj` and Winsteps' default. */
const EXTREME_SCORE_ADJUSTMENT = 0.3;
/** Each iteration takes one Newton step with a line search on each item's thresholds. */
const ITEM_STEP: NewtonOptions = { tolerance: 0, maxIterations: 1 };
const NO_ITEMS = new Int32Array(0);
const NO_VALUES = new Float64Array(0);

/**
 * Estimates the Rasch model for dichotomous items and the partial credit model for the others jointly with the persons'
 * abilities by joint maximum likelihood (JML), as TAM's `tam.jml`.
 *
 * @remarks
 * Each iteration solves each person's score equation, Σ E(X_i | θ) = total score, centers theta on the persons, and
 * takes a Newton step on each item's thresholds. A total score of 0 or of the maximum over the answered items has no
 * finite estimate; it moves 0.3 points inward, and the person takes part in the estimation with that score. The
 * estimates are the fixed point of these steps: the thresholds solve their score equations given the thetas, and the
 * thetas solve theirs up to the shift that centers them, which is 0 unless the extreme scores moved by unequal totals.
 * JML item estimates are biased outward: the result multiplies the thresholds and their standard errors by (k - 1) / k
 * for k items. Item standard errors come from the inverse of each item's information matrix given the thetas. Missing
 * responses are skipped; persons without responses take no part.
 *
 * @param matrix - The responses of the persons.
 * @param options - The stopping rules.
 * @returns The items, the persons, the log-likelihood, whether the estimation converged and the iterations run.
 * @throws `DataError` when the matrix or an option is invalid, there are fewer than two items, or an item has
 * responses in fewer than two categories or none in a category below its highest.
 * @see {@link https://doi.org/10.1177/014662167700100216 | Wright and Douglas (1977)}
 */
export function jointMaximumLikelihood(
  matrix: ResponseMatrix,
  options: JointMaximumLikelihoodOptions = {},
): JointCalibration {
  checkResponseMatrix(matrix);
  checkOptions(options);
  const { tolerance = DEFAULT_TOLERANCE, maxIterations = DEFAULT_MAX_ITERATIONS } = options;
  checkStopping(tolerance, maxIterations);
  const { itemIds } = matrix;
  if (itemIds.length < 2) {
    throw new DataError(`Joint maximum likelihood needs at least two items, found ${itemIds.length}.`);
  }
  const data = setUp(matrix);
  const estimation = estimationSteps(data);
  const { theta, thresholds } = data;

  const previousTheta = new Float64Array(theta.length);
  const previousThresholds = new Float64Array(thresholds.length);
  const { converged, iterations } = iterateToConvergence(
    () => {
      previousTheta.set(theta);
      previousThresholds.set(thresholds);
      for (let group = 0; group < theta.length; group++) theta[group] = estimation.solveTheta(group);
      estimation.center();
      for (let item = 0; item < itemIds.length; item++) estimation.stepItem(item);
      return Math.max(largestChange(previousTheta, theta), largestChange(previousThresholds, thresholds));
    },
    { tolerance, maxIterations },
  );

  const correction = (itemIds.length - 1) / itemIds.length;
  return {
    items: itemIds.map((id, item) =>
      correctedItem(id, data.itemThresholds[item] ?? NO_VALUES, estimation.thresholdErrors(item), correction),
    ),
    persons: matrix.personIds.map((id, person) => {
      const group = data.personGroups[person] ?? -1;
      if (group < 0) return { id, theta: null, standardError: null };
      return { id, theta: theta[group] ?? 0, standardError: 1 / Math.sqrt(estimation.information(group)) };
    }),
    logLikelihood: estimation.logLikelihood(),
    converged,
    iterations,
  };
}

/**
 * The data of a JML estimation. Persons who answered the same items with the same total score share theta, as the
 * total score is sufficient for it: they form a group.
 */
interface JointData {
  /** The number of categories of each item. */
  readonly categories: Int32Array;
  /** All items' thresholds, item after item. */
  readonly thresholds: Float64Array;
  /** Each item's thresholds: views into `thresholds`. */
  readonly itemThresholds: readonly Float64Array[];
  /** The items each group answered. */
  readonly groupItems: readonly Int32Array[];
  /** The total score each group's theta solves for, extreme scores adjusted. */
  readonly targets: Float64Array;
  /** The number of persons in each group. */
  readonly sizes: Float64Array;
  /** Each group's theta. */
  readonly theta: Float64Array;
  /** Each person's group, -1 for a person without responses. */
  readonly personGroups: Int32Array;
  /** For each item, the number of persons of each group in each category, by group, then category. */
  readonly itemCounts: readonly Float64Array[];
}

function setUp(matrix: ResponseMatrix): JointData {
  const { itemIds } = matrix;
  const itemCount = itemIds.length;
  const byItem = categoryTallies(compressResponses(matrix));
  const responses = responseCodes(matrix);
  const proportions = itemIds.map((id, item) => checkCategories(id, partialCredit, byItem[item] ?? NO_RESPONSES));
  const categories = Int32Array.from(proportions, (shares) => shares.length);
  const thresholds = Float64Array.from(proportions.flatMap((shares) => stepStarts(shares)));
  let end = 0;
  const itemThresholds = Array.from(categories, (count) => {
    end += count - 1;
    return thresholds.subarray(end - count + 1, end);
  });

  const keys = new Map<string, number>();
  const groupItems: Int32Array[] = [];
  const targets: number[] = [];
  const starts: number[] = [];
  const sizes: number[] = [];
  const personGroups = Int32Array.from(matrix.responses, (_, person) => {
    const row = responses.subarray(person * itemCount, (person + 1) * itemCount);
    const answered = Int32Array.from(row.keys()).filter((item) => (row[item] ?? -1) >= 0);
    if (answered.length === 0) return -1;
    let score = 0;
    let maximum = 0;
    for (const item of answered) {
      score += row[item] ?? 0;
      maximum += (categories[item] ?? 0) - 1;
    }
    const key = `${answered.join()}:${score}`;
    const existing = keys.get(key);
    if (existing !== undefined) {
      sizes[existing] = (sizes[existing] ?? 0) + 1;
      return existing;
    }
    const target = adjustedScore(score, maximum);
    keys.set(key, groupItems.length);
    groupItems.push(answered);
    targets.push(target);
    starts.push(Math.log(target / (maximum - target)));
    sizes.push(1);
    return groupItems.length - 1;
  });

  const itemCounts = Array.from(categories, (count) => new Float64Array(groupItems.length * count));
  personGroups.forEach((group, person) => {
    if (group < 0) return;
    for (let item = 0; item < itemCount; item++) {
      const response = responses[person * itemCount + item] ?? -1;
      const table = itemCounts[item] ?? NO_VALUES;
      const index = group * (categories[item] ?? 0) + response;
      if (response >= 0) table[index] = (table[index] ?? 0) + 1;
    }
  });
  return {
    categories,
    thresholds,
    itemThresholds,
    groupItems,
    targets: Float64Array.from(targets),
    sizes: Float64Array.from(sizes),
    theta: Float64Array.from(starts),
    personGroups,
    itemCounts,
  };
}

/** Returns a total score moved inward by {@link EXTREME_SCORE_ADJUSTMENT} when it is 0 or `maximum`. */
function adjustedScore(score: number, maximum: number): number {
  if (score === 0) return EXTREME_SCORE_ADJUSTMENT;
  return score === maximum ? maximum - EXTREME_SCORE_ADJUSTMENT : score;
}

/** The steps of a JML estimation and the quantities computed at its estimates. */
interface JointEstimation {
  /** Returns the theta at which a group's expected total score equals its target. */
  solveTheta(group: number): number;
  /** Subtracts the persons' mean theta from every theta. */
  center(): void;
  /** Takes one Newton step with a line search on an item's thresholds. */
  stepItem(item: number): void;
  /** Returns the standard errors of an item's thresholds, or `null` when its information matrix is not invertible. */
  thresholdErrors(item: number): Float64Array | null;
  /** Returns the information of a group's answered items at its theta. */
  information(group: number): number;
  /** Returns the joint log-likelihood of the responses. */
  logLikelihood(): number;
}

function estimationSteps(data: JointData): JointEstimation {
  const { categories, itemThresholds, groupItems, targets, sizes, theta, itemCounts } = data;
  const largest = categories.reduce((most, count) => Math.max(most, count), 0);
  const probabilities = new Float64Array(largest);
  const tails = new Float64Array(largest);
  const gradient = new Float64Array(largest - 1);
  const hessian = new Float64Array((largest - 1) ** 2);
  // The group whose expected total score `excess` evaluates, and the item whose thresholds `itemObjective` evaluates.
  let currentGroup = 0;
  let currentItem = 0;

  /** Returns the expected total score of the current group at theta minus its target. */
  const excess = (value: number): number => {
    const items = groupItems[currentGroup] ?? NO_ITEMS;
    let expected = 0;
    for (let i = 0; i < items.length; i++) {
      const item = items[i] ?? 0;
      partialCreditProbabilities(1, 0, itemThresholds[item] ?? NO_VALUES, value, probabilities);
      for (let k = 1; k < (categories[item] ?? 0); k++) expected += k * (probabilities[k] ?? 0);
    }
    return expected - (targets[currentGroup] ?? 0);
  };

  /**
   * Returns minus the current item's log-likelihood at thresholds δ given the thetas, with its gradient
   * Σ ([x ≥ j] - P(X ≥ j)) and Hessian Σ (P(X ≥ max(j, m)) - P(X ≥ j) P(X ≥ m)) over the persons who answered it.
   */
  const itemObjective = (delta: Float64Array, out: Float64Array, curvature: Float64Array): number => {
    out.fill(0);
    curvature.fill(0);
    let value = 0;
    for (let group = 0; group < theta.length; group++) value += groupTerm(group, delta, out, curvature);
    return value;
  };

  /** Adds a group's terms of the current item's gradient and Hessian, and returns its term of the objective. */
  const groupTerm = (group: number, delta: Float64Array, out: Float64Array, curvature: Float64Array): number => {
    const count = categories[currentItem] ?? 0;
    const table = itemCounts[currentItem] ?? NO_VALUES;
    const offset = group * count;
    let persons = 0;
    for (let k = 0; k < count; k++) persons += table[offset + k] ?? 0;
    if (persons === 0) return 0;
    partialCreditProbabilities(1, 0, delta, theta[group] ?? 0, probabilities);
    let value = 0;
    for (let k = 0; k < count; k++) {
      const observed = table[offset + k] ?? 0;
      if (observed > 0) value -= observed * logProbability(probabilities[k] ?? 0);
    }
    addGradient(table, offset, count, persons, out);
    addCovariance(count, persons, curvature);
    return value;
  };

  /** Writes the tails P(X ≥ j) and adds a group's Σ [x ≥ j] - n P(X ≥ j) to the gradient, given its counts. */
  const addGradient = (
    table: Float64Array,
    offset: number,
    count: number,
    persons: number,
    out: Float64Array,
  ): void => {
    let tail = 0;
    let atLeast = 0;
    for (let k = count - 1; k >= 1; k--) {
      tail += probabilities[k] ?? 0;
      atLeast += table[offset + k] ?? 0;
      tails[k] = tail;
      out[k - 1] = (out[k - 1] ?? 0) + atLeast - persons * tail;
    }
  };

  /** Adds `persons` times the covariance matrix of the indicators [X ≥ j], from the tails, to the Hessian. */
  const addCovariance = (count: number, persons: number, curvature: Float64Array): void => {
    const size = count - 1;
    for (let j = 1; j < count; j++) {
      for (let m = 1; m < count; m++) {
        const index = (j - 1) * size + m - 1;
        const covariance = (tails[Math.max(j, m)] ?? 0) - (tails[j] ?? 0) * (tails[m] ?? 0);
        curvature[index] = (curvature[index] ?? 0) + persons * covariance;
      }
    }
  };

  return {
    solveTheta(group) {
      currentGroup = group;
      // The expected score increases with theta.
      return findIncreasingRoot(excess, theta[group] ?? 0);
    },
    center() {
      let total = 0;
      let persons = 0;
      for (let group = 0; group < theta.length; group++) {
        total += (sizes[group] ?? 0) * (theta[group] ?? 0);
        persons += sizes[group] ?? 0;
      }
      const mean = total / persons;
      for (let group = 0; group < theta.length; group++) theta[group] = (theta[group] ?? 0) - mean;
    },
    stepItem(item) {
      currentItem = item;
      const current = itemThresholds[item] ?? NO_VALUES;
      current.set(newton(itemObjective, current, ITEM_STEP).x);
    },
    thresholdErrors(item) {
      currentItem = item;
      const current = itemThresholds[item] ?? NO_VALUES;
      const size = current.length;
      itemObjective(current, gradient, hessian);
      const inverse = invert(hessian.subarray(0, size * size), size);
      if (inverse === null) return null;
      const errors = Float64Array.from({ length: size }, (_, j) => Math.sqrt(inverse[j * size + j] ?? 0));
      return errors.every((error) => error > 0) ? errors : null;
    },
    information(group) {
      const items = groupItems[group] ?? NO_ITEMS;
      let sum = 0;
      for (let i = 0; i < items.length; i++) {
        sum += partialCreditInformation(1, 0, itemThresholds[items[i] ?? 0] ?? NO_VALUES, theta[group] ?? 0);
      }
      return sum;
    },
    logLikelihood() {
      let sum = 0;
      for (let item = 0; item < categories.length; item++) {
        currentItem = item;
        sum -= itemObjective(itemThresholds[item] ?? NO_VALUES, gradient, hessian);
      }
      return sum;
    },
  };
}

/** Returns a JML item with its thresholds and their standard errors multiplied by the bias correction. */
function correctedItem(
  id: string,
  thresholds: Float64Array,
  errors: Float64Array | null,
  correction: number,
): CalibratedItem<LogisticItem | ThresholdItem> {
  const b = Array.from(thresholds, (threshold) => threshold * correction);
  const corrected = errors === null ? null : Array.from(errors, (error) => error * correction);
  if (b.length === 1) {
    return {
      id,
      model: oneParameterLogistic,
      parameters: { a: 1, b: b[0] ?? 0, c: 0, d: 1 },
      standardErrors: corrected === null ? null : { b: corrected[0] ?? 0 },
    };
  }
  return {
    id,
    model: partialCredit,
    parameters: { a: 1, b },
    standardErrors: corrected === null ? null : Object.fromEntries(corrected.map((error, j) => [`b${j + 1}`, error])),
  };
}
