import {
  checkChoice,
  checkOptions,
  checkStopping,
  DataError,
  type LogisticItem,
  type ThresholdItem,
} from "@measurekit/core";
import { checkResponseMatrix, type ResponseMatrix } from "../data/internal.js";
import { invertPositiveDefinite, multiply, transpose } from "../math/internal.js";
import type { CalibratedItem } from "./calibration.js";
import { oneParameterLogistic } from "./logistic.js";
import { partialCredit } from "./partial-credit.js";
import { raschScaleItem } from "./rasch-scale.js";
import { type RatingScaleParameters, ratingScale } from "./rating-scale.js";
import { checkCategories } from "./responses.js";
import {
  combine,
  itemFunctions,
  leaveOut,
  logFunction,
  NO_ITEMS,
  type SymmetricFunctions,
} from "./symmetric-functions.js";

/** A model of the Rasch family that conditional maximum likelihood estimates. */
export type ConditionalModel = "rasch" | "partial-credit" | "rating-scale";

/** Options of {@link conditionalMaximumLikelihood}. */
export interface ConditionalCalibrationOptions {
  /**
   * The model: the Rasch model of 0/1 items, the partial credit model, or the rating scale model, whose items share
   * their steps and have the same number of categories.
   *
   * @defaultValue `"rasch"` when every response is 0 or 1, otherwise `"partial-credit"`
   */
  readonly model?: ConditionalModel;
  /**
   * Newton's method has converged when no parameter changes by more than this.
   *
   * @defaultValue `1e-6`
   */
  readonly tolerance?: number;
  /**
   * The most Newton steps to take.
   *
   * @defaultValue `100`
   */
  readonly maxIterations?: number;
}

/** Item parameters of the Rasch family estimated by conditional maximum likelihood. */
export interface ConditionalCalibration {
  /**
   * The items, in item order, on the Rasch scale, a = 1, with the thresholds of all items summing to 0: a dichotomous
   * item as a 1PL item, the others as partial credit items or, under the rating scale model, rating scale items. The
   * standard errors are `null` when the information matrix is singular.
   */
  readonly items: readonly CalibratedItem<LogisticItem | ThresholdItem | RatingScaleParameters>[];
  /** The conditional log-likelihood at the estimates. */
  readonly logLikelihood: number;
  /** The number of estimated parameters: the thresholds, or the locations and steps, less 1 for the scale. */
  readonly parameters: number;
  /**
   * The persons the conditional likelihood uses: those who answered at least two items and whose total over them is
   * neither 0 nor its highest possible value.
   */
  readonly persons: number;
  /** Whether Newton's method reached `tolerance` within `maxIterations`. */
  readonly converged: boolean;
  /** Newton steps taken. */
  readonly iterations: number;
}

const MODELS: readonly ConditionalModel[] = ["rasch", "partial-credit", "rating-scale"];
const DEFAULT_TOLERANCE = 1e-6;
const DEFAULT_MAX_ITERATIONS = 100;
/** The most times a step that lowers the log-likelihood is halved. */
const MAX_HALVES = 30;
/** The relative decrease of the log-likelihood a step may show: the rounding error of a sum over all persons. */
const LIKELIHOOD_SLACK = 1e-12;

/**
 * Estimates the Rasch, partial credit or rating scale model by conditional maximum likelihood (CML), as eRm's `RM`,
 * `PCM` and `RSM`, without assuming a distribution of theta.
 *
 * @remarks
 * Given a person's total score r on the items the person answered, the responses do not depend on theta:
 * P(x | r) = Π_i e^(-η_(i x_i)) / γ_r, with η_ik the sum of item i's thresholds up to category k and γ_r the
 * elementary symmetric function of the answered items, the sum of that product over every pattern of total r. The
 * functions come from the summation algorithm, one item at a time, and those with one or two items left out give the
 * gradient and the Hessian; persons are grouped by the items they answered, each group with its own functions. Newton's
 * method maximizes the conditional log-likelihood from thresholds of 0, halving a step that lowers it, and the
 * standard errors come from the inverse of its Hessian. The scale is fixed by the thresholds of all items summing to 0.
 *
 * A person who answered fewer than two items, or whose total is 0 or the highest possible on the items answered, adds
 * nothing to the conditional likelihood and is left out. An item's categories run from 0 to its highest response in
 * the matrix, and the persons used need a response in each.
 *
 * @param matrix - The responses of the persons.
 * @param options - The model and the stopping rules.
 * @returns The items, the conditional log-likelihood, the number of parameters and of persons used, whether Newton's
 *   method converged and the steps taken.
 * @throws `DataError` when the matrix or an option is invalid, there are fewer than two items, the Rasch model meets a
 * response above 1, an item has no response from the persons used in a category up to its highest, as an item every
 * person answered right or wrong, or the items of the rating scale model differ in their number of categories.
 * @see {@link https://doi.org/10.1111/j.2517-6161.1972.tb00887.x | Andersen (1972)}
 * @see {@link https://doi.org/10.1007/978-1-4612-4230-7_3 | Molenaar (1995)}
 * @see {@link https://doi.org/10.18637/jss.v020.i09 | Mair and Hatzinger (2007)}
 */
export function conditionalMaximumLikelihood(
  matrix: ResponseMatrix,
  options: ConditionalCalibrationOptions = {},
): ConditionalCalibration {
  return conditionalFit(matrix, options).calibration;
}

/** A conditional calibration with the covariance matrix of the parameters it reports. */
export interface ConditionalFit {
  /** The calibration. */
  readonly calibration: ConditionalCalibration;
  /**
   * The covariance matrix by rows of the reported parameters: the thresholds of every item, in item order, or under the
   * rating scale model the locations and then the steps. `null` when the information matrix is singular.
   */
  readonly covariance: Float64Array | null;
}

/**
 * Estimates as {@link conditionalMaximumLikelihood} does and also returns the covariance matrix of the parameters.
 *
 * @throws `DataError` as {@link conditionalMaximumLikelihood} does.
 */
export function conditionalFit(matrix: ResponseMatrix, options: ConditionalCalibrationOptions = {}): ConditionalFit {
  checkResponseMatrix(matrix);
  checkOptions(options);
  const { tolerance = DEFAULT_TOLERANCE, maxIterations = DEFAULT_MAX_ITERATIONS } = options;
  checkStopping(tolerance, maxIterations);
  const { itemIds } = matrix;
  if (itemIds.length < 2) {
    throw new DataError(`Conditional maximum likelihood needs at least two items, found ${itemIds.length}.`);
  }
  const highest = highestResponses(matrix);
  const { model = highest.every((category) => category <= 1) ? "rasch" : "partial-credit" } = options;
  checkChoice(model, "The option model", MODELS);
  const data = conditionalData(matrix, highest, model);
  const layout = parameterLayout(highest, model);
  const maximum = maximize(data, layout, tolerance, maxIterations);
  const { values, covariance } = reportedValues(layout, maximum.estimates, maximum.covariance);
  return {
    calibration: {
      items: calibratedItems(itemIds, highest, model, values, covariance),
      logLikelihood: maximum.logLikelihood,
      parameters: layout.size,
      persons: data.persons,
      converged: maximum.converged,
      iterations: maximum.iterations,
    },
    covariance,
  };
}

/** Returns each item's highest response, at least 0. */
function highestResponses({ itemIds, responses }: ResponseMatrix): Int32Array {
  const highest = new Int32Array(itemIds.length);
  for (const row of responses) {
    row.forEach((response, item) => {
      if (response !== null && response > (highest[item] ?? 0)) highest[item] = response;
    });
  }
  return highest;
}

/** The persons who answered the same items. */
interface PatternGroup {
  /** The items, in item order. */
  readonly items: Int32Array;
  /** The number of persons with each total score, from 0 to the highest possible. */
  readonly counts: Float64Array;
}

/** The responses that enter the conditional likelihood. */
interface ConditionalData {
  /** The start of each item's category parameters η_i1 to η_im in the vector, and their number as the last entry. */
  readonly offsets: Int32Array;
  /** The persons by the items they answered. */
  readonly groups: readonly PatternGroup[];
  /** The responses in each category from 1 of each item, at the item's offset. */
  readonly responseCounts: Float64Array;
  /** The persons used. */
  readonly persons: number;
}

/**
 * Groups the persons the conditional likelihood uses by the items they answered, and counts their responses.
 *
 * @throws `DataError` when an item has a response above its model's categories, or no response from these persons in
 * a category up to its highest, or the items of the rating scale model differ in their number of categories.
 */
function conditionalData(matrix: ResponseMatrix, highest: Int32Array, model: ConditionalModel): ConditionalData {
  const offsets = new Int32Array(highest.length + 1);
  highest.forEach((categories, item) => {
    offsets[item + 1] = (offsets[item] ?? 0) + categories;
  });
  const responseCounts = new Float64Array(offsets[highest.length] ?? 0);
  const tallies = Array.from(highest, () => new Map<number, number>());
  const groups = new Map<string, PatternGroup>();
  let persons = 0;
  for (const row of matrix.responses) {
    const { answered, score, possible } = personScore(row, highest);
    if (answered.length < 2 || score === 0 || score === possible) continue;
    persons++;
    const key = answered.join();
    const group = groups.get(key) ?? { items: answered, counts: new Float64Array(possible + 1) };
    groups.set(key, group);
    group.counts[score] = (group.counts[score] ?? 0) + 1;
    answered.forEach((item) => {
      const response = row[item] ?? 0;
      tallies[item]?.set(response, (tallies[item]?.get(response) ?? 0) + 1);
      const slot = (offsets[item] ?? 0) + response - 1;
      if (response > 0) responseCounts[slot] = (responseCounts[slot] ?? 0) + 1;
    });
  }
  checkItems(matrix.itemIds, highest, model, tallies);
  return { offsets, groups: [...groups.values()], responseCounts, persons };
}

/** Returns the items a person answered, the total over them and its highest possible value. */
function personScore(
  row: readonly (number | null)[],
  highest: Int32Array,
): { readonly answered: Int32Array; readonly score: number; readonly possible: number } {
  const answered = Int32Array.from(row.keys()).filter((item) => row[item] !== null);
  let score = 0;
  let possible = 0;
  for (const item of answered) {
    score += row[item] ?? 0;
    possible += highest[item] ?? 0;
  }
  return { answered, score, possible };
}

/**
 * Checks each item's categories among the persons used, and that the items of the rating scale model have the same.
 *
 * @throws `DataError` as {@link conditionalData} does.
 */
function checkItems(
  itemIds: readonly string[],
  highest: Int32Array,
  model: ConditionalModel,
  tallies: readonly Map<number, number>[],
): void {
  const checked = model === "rasch" ? oneParameterLogistic : model === "rating-scale" ? ratingScale : partialCredit;
  itemIds.forEach((id, item) => {
    checkCategories(id, checked, { highest: highest[item] ?? 0, counts: tallies[item] ?? new Map() });
    if (model === "rating-scale" && highest[item] !== highest[0]) {
      throw new DataError(
        `The rating scale model needs items with the same number of categories; item "${itemIds[0]}" has ${(highest[0] ?? 0) + 1} and item "${id}" has ${(highest[item] ?? 0) + 1}.`,
      );
    }
  });
}

/** How the estimated parameters φ give the thresholds τ, the category parameters η and the reported parameters. */
interface ParameterLayout {
  /** The number of estimated parameters. */
  readonly size: number;
  /** η = J φ, by rows of η. */
  readonly cumulative: Float64Array;
  /** The reported parameters, by rows of R in R φ: each item's thresholds, or the locations and then the steps. */
  readonly reported: Float64Array;
}

/**
 * Returns the layout of the parameters. The thresholds of all items sum to 0: the last threshold, or the last location
 * and the last step, is minus the sum of the others.
 */
function parameterLayout(highest: Int32Array, model: ConditionalModel): ParameterLayout {
  const thresholdCount = highest.reduce((sum, categories) => sum + categories, 0);
  if (model !== "rating-scale") {
    const thresholds = sumToZero(thresholdCount);
    return {
      size: thresholdCount - 1,
      cumulative: cumulated(thresholds, highest, thresholdCount - 1),
      reported: thresholds,
    };
  }
  // Threshold (i, k) is location i plus step k; the locations fill the first columns, the steps the others.
  const itemCount = highest.length;
  const steps = highest[0] ?? 0;
  const locations = sumToZero(itemCount);
  const stepShares = sumToZero(steps);
  const size = itemCount - 1 + steps - 1;
  const thresholds = new Float64Array(thresholdCount * size);
  const reported = new Float64Array((itemCount + steps) * size);
  for (let i = 0; i < itemCount; i++) {
    for (let k = 0; k < steps; k++) {
      thresholds.set(locations.subarray(i * (itemCount - 1), (i + 1) * (itemCount - 1)), (i * steps + k) * size);
      thresholds.set(
        stepShares.subarray(k * (steps - 1), (k + 1) * (steps - 1)),
        (i * steps + k) * size + itemCount - 1,
      );
    }
    reported.set(locations.subarray(i * (itemCount - 1), (i + 1) * (itemCount - 1)), i * size);
  }
  for (let k = 0; k < steps; k++) {
    reported.set(stepShares.subarray(k * (steps - 1), (k + 1) * (steps - 1)), (itemCount + k) * size + itemCount - 1);
  }
  return { size, cumulative: cumulated(thresholds, highest, size), reported };
}

/** Returns the `count` by `count - 1` matrix that gives `count` values summing to 0 from all but the last. */
function sumToZero(count: number): Float64Array {
  const free = count - 1;
  const matrix = new Float64Array(count * free);
  for (let j = 0; j < free; j++) {
    matrix[j * free + j] = 1;
    matrix[free * free + j] = -1;
  }
  return matrix;
}

/** Returns the rows of the category parameters η: each item's thresholds summed up to each category. */
function cumulated(thresholds: Float64Array, highest: Int32Array, size: number): Float64Array {
  const cumulative = thresholds.slice();
  let first = 0;
  for (const categories of highest) {
    for (let e = first + 1; e < first + categories; e++) {
      for (let p = 0; p < size; p++)
        cumulative[e * size + p] = (cumulative[e * size + p] ?? 0) + (cumulative[(e - 1) * size + p] ?? 0);
    }
    first += categories;
  }
  return cumulative;
}

/** The conditional log-likelihood with its gradient and Hessian in η. */
interface Evaluation {
  readonly logLikelihood: number;
  readonly gradient: Float64Array;
  readonly hessian: Float64Array;
}

/**
 * Returns the conditional log-likelihood at η and, with `derivatives`, its gradient, -n_ik + Σ_r n_r P_ik(r), and its
 * Hessian, minus Σ_r n_r times the covariance of the category indicators given r: P_(ik,jl)(r) - P_ik(r) P_jl(r).
 */
function evaluate(data: ConditionalData, eta: Float64Array, derivatives: boolean): Evaluation {
  const { offsets, groups, responseCounts } = data;
  const size = responseCounts.length;
  const gradient = new Float64Array(size);
  const hessian = new Float64Array(derivatives ? size * size : 0);
  let logLikelihood = 0;
  responseCounts.forEach((count, e) => {
    logLikelihood -= count * (eta[e] ?? 0);
    gradient[e] = -count;
  });
  for (const group of groups) {
    const functions = Array.from(group.items, (item) =>
      itemFunctions(eta.subarray(offsets[item] ?? 0, offsets[item + 1] ?? 0)),
    );
    if (!derivatives) {
      const all = functions.reduce(combine, NO_ITEMS);
      group.counts.forEach((count, r) => {
        if (count > 0) logLikelihood -= count * logFunction(all, r);
      });
      continue;
    }
    logLikelihood -= addGroup(group, functions, eta, offsets, gradient, hessian, size);
  }
  return { logLikelihood, gradient, hessian };
}

/**
 * Adds a group's terms to the gradient and Hessian and returns Σ_r n_r ln γ_r. P_ik(r) = e^(-η_ik) γ_(r-k)(without i) /
 * γ_r and P_(ik,jl)(r) = e^(-η_ik - η_jl) γ_(r-k-l)(without i and j) / γ_r.
 */
function addGroup(
  { items, counts }: PatternGroup,
  functions: readonly SymmetricFunctions[],
  eta: Float64Array,
  offsets: Int32Array,
  gradient: Float64Array,
  hessian: Float64Array,
  size: number,
): number {
  const left = leaveOut(functions);
  const scores = Int32Array.from(counts.keys()).filter((r) => (counts[r] ?? 0) > 0);
  const logAll = Float64Array.from(scores, (r) => logFunction(left.all, r));
  let logSum = 0;
  scores.forEach((r, s) => {
    logSum += (counts[r] ?? 0) * (logAll[s] ?? 0);
  });
  // P[(e - first of the group's item) * scores + s]: the conditional probability of each category from 1 of each item.
  const starts = Int32Array.from(items, (item) => offsets[item] ?? 0);
  const ends = Int32Array.from(items, (item) => offsets[item + 1] ?? 0);
  const local = new Int32Array(items.length + 1);
  items.forEach((_, a) => {
    local[a + 1] = (local[a] ?? 0) + (ends[a] ?? 0) - (starts[a] ?? 0);
  });
  const probabilities = new Float64Array((local[items.length] ?? 0) * scores.length);
  const probability = (functionsOf: SymmetricFunctions, total: number, logWeight: number, s: number): number => {
    const r = scores[s] ?? 0;
    if (r - total < 0 || r - total >= functionsOf.values.length) return 0;
    return Math.exp(logFunction(functionsOf, r - total) - (logAll[s] ?? 0) + logWeight);
  };
  items.forEach((_, a) => {
    const without = left.without(a);
    const start = starts[a] ?? 0;
    const categories = (ends[a] ?? 0) - start;
    for (let k = 1; k <= categories; k++) {
      const e = start + k - 1;
      const row = ((local[a] ?? 0) + k - 1) * scores.length;
      scores.forEach((r, s) => {
        const value = probability(without, k, -(eta[e] ?? 0), s);
        probabilities[row + s] = value;
        gradient[e] = (gradient[e] ?? 0) + (counts[r] ?? 0) * value;
      });
    }
    // Within an item the categories exclude each other: the covariance is δ_kl P_k - P_k P_l.
    for (let k = 1; k <= categories; k++) {
      for (let l = 1; l <= categories; l++) {
        const rowK = ((local[a] ?? 0) + k - 1) * scores.length;
        const rowL = ((local[a] ?? 0) + l - 1) * scores.length;
        let covariance = 0;
        scores.forEach((r, s) => {
          const pk = probabilities[rowK + s] ?? 0;
          covariance += (counts[r] ?? 0) * ((k === l ? pk : 0) - pk * (probabilities[rowL + s] ?? 0));
        });
        const index = (start + k - 1) * size + start + l - 1;
        hessian[index] = (hessian[index] ?? 0) - covariance;
      }
    }
  });
  left.forEachPair((a, b, without) => {
    for (let k = 1; k <= (ends[a] ?? 0) - (starts[a] ?? 0); k++) {
      const e = (starts[a] ?? 0) + k - 1;
      const rowK = ((local[a] ?? 0) + k - 1) * scores.length;
      for (let l = 1; l <= (ends[b] ?? 0) - (starts[b] ?? 0); l++) {
        const f = (starts[b] ?? 0) + l - 1;
        const rowL = ((local[b] ?? 0) + l - 1) * scores.length;
        let covariance = 0;
        scores.forEach((r, s) => {
          const joint = probability(without, k + l, -(eta[e] ?? 0) - (eta[f] ?? 0), s);
          covariance += (counts[r] ?? 0) * (joint - (probabilities[rowK + s] ?? 0) * (probabilities[rowL + s] ?? 0));
        });
        hessian[e * size + f] = (hessian[e * size + f] ?? 0) - covariance;
        hessian[f * size + e] = (hessian[f * size + e] ?? 0) - covariance;
      }
    }
  });
  return logSum;
}

/** The maximum of the conditional likelihood. */
interface Maximum {
  /** The estimated parameters φ. */
  readonly estimates: Float64Array;
  /** The covariance matrix of φ by rows, or `null` when the information matrix is singular. */
  readonly covariance: Float64Array | null;
  readonly logLikelihood: number;
  readonly converged: boolean;
  readonly iterations: number;
}

/** Maximizes the conditional log-likelihood in φ by Newton's method from 0, halving a step that lowers it. */
function maximize(data: ConditionalData, layout: ParameterLayout, tolerance: number, maxIterations: number): Maximum {
  let estimates = new Float64Array(layout.size);
  let current = evaluate(data, categoryParameters(layout, estimates), true);
  let information = projectedInformation(current, layout);
  let converged = false;
  let iterations = 0;
  while (iterations < maxIterations && !converged) {
    const inverse = invertPositiveDefinite(information.matrix, layout.size);
    if (inverse === null) break;
    const direction = new Float64Array(layout.size);
    multiply(inverse, information.gradient, layout.size, layout.size, 1, direction);
    const step = halvedStep(data, layout, estimates, direction, current.logLikelihood);
    iterations++;
    estimates = estimates.map((value, p) => value + (step[p] ?? 0));
    current = evaluate(data, categoryParameters(layout, estimates), true);
    information = projectedInformation(current, layout);
    converged = step.every((change) => Math.abs(change) <= tolerance);
  }
  return {
    estimates,
    covariance: invertPositiveDefinite(information.matrix, layout.size),
    logLikelihood: current.logLikelihood,
    converged,
    iterations,
  };
}

/** Returns η = J φ. */
function categoryParameters({ size, cumulative }: ParameterLayout, estimates: Float64Array): Float64Array {
  const rows = cumulative.length / size;
  const eta = new Float64Array(rows);
  multiply(cumulative, estimates, rows, size, 1, eta);
  return eta;
}

/** Returns the Newton step, halved until the log-likelihood does not fall below `logLikelihood`. */
function halvedStep(
  data: ConditionalData,
  layout: ParameterLayout,
  estimates: Float64Array,
  direction: Float64Array,
  logLikelihood: number,
): Float64Array {
  let step = direction;
  for (let halves = 0; halves < MAX_HALVES; halves++) {
    const trial = estimates.map((value, p) => value + (step[p] ?? 0));
    const value = evaluate(data, categoryParameters(layout, trial), false).logLikelihood;
    if (value >= logLikelihood - LIKELIHOOD_SLACK * Math.abs(logLikelihood)) break;
    step = step.map((change) => change / 2);
  }
  return step;
}

/** Returns the gradient Jᵀ g and the information -Jᵀ H J of the log-likelihood in φ. */
function projectedInformation(
  { gradient, hessian }: Evaluation,
  { size, cumulative }: ParameterLayout,
): { readonly gradient: Float64Array; readonly matrix: Float64Array } {
  const rows = gradient.length;
  const transposed = transpose(cumulative, rows, size);
  const projected = new Float64Array(size);
  multiply(transposed, gradient, size, rows, 1, projected);
  const product = new Float64Array(rows * size);
  multiply(hessian, cumulative, rows, rows, size, product);
  const matrix = new Float64Array(size * size);
  multiply(transposed, product, size, rows, size, matrix);
  return { gradient: projected, matrix: matrix.map((value) => -value) };
}

/** Returns the reported parameters R φ and their covariance matrix R C Rᵀ. */
function reportedValues(
  { reported, size }: ParameterLayout,
  estimates: Float64Array,
  covariance: Float64Array | null,
): { readonly values: Float64Array; readonly covariance: Float64Array | null } {
  const count = reported.length / size;
  const values = new Float64Array(count);
  multiply(reported, estimates, count, size, 1, values);
  if (covariance === null) return { values, covariance: null };
  const product = new Float64Array(count * size);
  multiply(reported, covariance, count, size, size, product);
  const result = new Float64Array(count * count);
  multiply(product, transpose(reported, count, size), count, size, count, result);
  return { values, covariance: result };
}

function calibratedItems(
  itemIds: readonly string[],
  highest: Int32Array,
  model: ConditionalModel,
  values: Float64Array,
  covariance: Float64Array | null,
): CalibratedItem<LogisticItem | ThresholdItem | RatingScaleParameters>[] {
  const count = values.length;
  const errors =
    covariance === null
      ? null
      : Float64Array.from({ length: count }, (_, row) => Math.sqrt(covariance[row * count + row] ?? 0));
  if (model === "rating-scale") {
    const steps = highest[0] ?? 0;
    const stepValues = Array.from(values.subarray(itemIds.length));
    return itemIds.map((id, item) => ({
      id,
      model: ratingScale,
      parameters: { a: 1, b: values[item] ?? 0, steps: stepValues },
      standardErrors:
        errors === null
          ? null
          : Object.fromEntries([
              ["b", errors[item] ?? 0],
              ...Array.from({ length: steps - 1 }, (_, k) => [`step${k + 1}`, errors[itemIds.length + k] ?? 0]),
            ]),
    }));
  }
  let start = 0;
  return itemIds.map((id, item) => {
    const categories = highest[item] ?? 0;
    const thresholds = Array.from(values.subarray(start, start + categories));
    const se = errors === null ? null : Array.from(errors.subarray(start, start + categories));
    start += categories;
    return raschScaleItem(id, thresholds, se);
  });
}
