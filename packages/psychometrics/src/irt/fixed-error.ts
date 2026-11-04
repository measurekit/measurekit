import { DataError } from "@measurekit/core";
import { multiply, transpose } from "../math/internal.js";
import type { StandardErrorMethod } from "./calibration-options.js";
import type { ItemState, Setup } from "./calibration-setup.js";
import type { ExpectationStep } from "./expectation.js";
import { type Maximization, unboundedCoordinates, unboundedScales } from "./m-step.js";
import {
  checkCovariance,
  differenceStep,
  type ParameterCovariance,
  parameterEntry,
  VARIANCE_ROUNDING,
} from "./parameter-covariance.js";

/** Names the option in error messages. */
const NOUN = "the option fixedCovariance";

/** A fixed item's parameter whose sampling error a calibration carries. */
interface FixedTarget {
  /** The index of the item. */
  readonly item: number;
  /** The index of the parameter in the item's parameter vector. */
  readonly entry: number;
  /** The step of the central differences in the parameter. */
  readonly step: number;
}

/** The fixed parameters whose sampling error a calibration carries, with their covariance. */
export interface FixedError {
  readonly targets: readonly FixedTarget[];
  /** The covariance of the targets, by rows. */
  readonly matrix: Float64Array;
}

/**
 * Checks the option fixedCovariance against the items and returns the fixed parameters it lists, with their
 * covariance; `null` without the option.
 *
 * @throws `DataError` when the covariance is invalid, there are no standard errors, or a label names an estimated
 * item, a parameter its model does not have, a parameter listed before, or a parameter on a bound of its range.
 */
export function fixedErrorOf<TParameters>(
  covariance: ParameterCovariance | undefined,
  items: readonly ItemState<TParameters>[],
  standardErrors: StandardErrorMethod,
): FixedError | null {
  if (covariance === undefined) return null;
  const labels = checkCovariance(covariance, NOUN);
  if (standardErrors === "none") {
    throw new DataError('The option fixedCovariance needs the standard errors "hessian" or "cross-product".');
  }
  const indices = new Map(items.map((item, index) => [item.id, index]));
  const listed = new Set<string>();
  const used: number[] = [];
  const targets: FixedTarget[] = [];
  labels.forEach(({ item: id, name }, k) => {
    const index = indices.get(id);
    const item = index === undefined ? undefined : items[index];
    // A covariance of a whole pool may list items that the data do not have.
    if (index === undefined || item === undefined) return;
    if (item.estimated) {
      throw new DataError(
        `The option fixedCovariance has the parameter "${name}" of item "${id}", which is estimated here, not fixed.`,
      );
    }
    const key = JSON.stringify([id, name]);
    if (listed.has(key)) {
      throw new DataError(`The option fixedCovariance lists the parameter "${name}" of item "${id}" twice.`);
    }
    listed.add(key);
    const entry = parameterEntry(item, name, NOUN);
    targets.push({ item: index, entry, step: differenceStep(item, entry) });
    used.push(k);
  });
  const count = used.length;
  const matrix = new Float64Array(count * count);
  used.forEach((row, a) => {
    used.forEach((column, b) => {
      matrix[a * count + b] = covariance.matrix[row]?.[column] ?? 0;
    });
  });
  return { targets, matrix };
}

/**
 * Returns the covariance of the free parameters with the fixed parameters' sampling error added: V + J Σ Jᵀ, with V
 * the covariance as if they were known, Σ theirs and J = -V C the derivative of the estimates with respect to them, C
 * the derivative of the gradient of the negative log-posterior with respect to them.
 *
 * @remarks
 * This is the variance of a pseudo maximum likelihood estimate, whose nuisance parameters are estimates from data
 * independent of the score of the parameters estimated. V and C are on the parameters' own scale, as they meet at a
 * maximum, where the gradient vanishes.
 *
 * @throws `DataError` when Σ adds a negative variance beyond rounding, as a matrix that is not positive semidefinite
 * can.
 */
export function addFixedError<TParameters>(
  covariance: Float64Array,
  setup: Setup<TParameters>,
  mStep: Maximization<TParameters>,
  eStep: ExpectationStep,
  { targets, matrix }: FixedError,
): Float64Array {
  const size = setup.values.length;
  const count = targets.length;
  const derivative = new Float64Array(size * count);
  multiply(covariance, crossDerivatives(setup, mStep, eStep, targets), size, size, count, derivative);
  for (let i = 0; i < derivative.length; i++) derivative[i] = -(derivative[i] ?? 0);
  const weighted = new Float64Array(size * count);
  multiply(derivative, matrix, size, count, count, weighted);
  const added = new Float64Array(size * size);
  multiply(weighted, transpose(derivative, size, count), size, count, size, added);
  for (let j = 0; j < size; j++) checkAddedVariance(derivative.subarray(j * count, (j + 1) * count), matrix);
  return covariance.map((value, index) => value + (added[index] ?? 0));
}

/**
 * Returns C, the derivative of the gradient of the negative log-posterior on the parameters' own scale with respect to
 * each fixed parameter, by rows of the free parameters: central differences of the exact gradient at the estimates,
 * each with its own E-step.
 */
function crossDerivatives<TParameters>(
  { items, free, values }: Setup<TParameters>,
  mStep: Maximization<TParameters>,
  eStep: ExpectationStep,
  targets: readonly FixedTarget[],
): Float64Array {
  const size = values.length;
  const count = targets.length;
  const eta = unboundedCoordinates(values, free);
  const scales = unboundedScales(values, free);
  const above = new Float64Array(size);
  const below = new Float64Array(size);
  const cross = new Float64Array(size * count);
  targets.forEach(({ item: index, entry, step }, m) => {
    const item = items[index];
    if (item === undefined) return;
    const known = item.parameters;
    const vector = item.model.toVector(known);
    const center = vector[entry] ?? 0;
    const gradientAt = (value: number, gradient: Float64Array): void => {
      vector[entry] = value;
      item.parameters = item.model.fromVector(vector);
      eStep.run();
      mStep.objective(mStep.all, eta, gradient);
    };
    gradientAt(center + step, above);
    gradientAt(center - step, below);
    item.parameters = known;
    for (let j = 0; j < size; j++) {
      // The objective's gradient is on the unbounded scale: dividing by dx/dη puts it on the parameters' own.
      const scale = scales[j] ?? 0;
      cross[j * count + m] = scale > 0 ? ((above[j] ?? 0) - (below[j] ?? 0)) / (2 * step * scale) : 0;
    }
  });
  return cross;
}

/**
 * Checks that the variance a row of J adds, jᵀ Σ j, is not negative beyond rounding.
 *
 * @throws `DataError` when it is.
 */
function checkAddedVariance(row: Float64Array, matrix: Float64Array): void {
  const count = row.length;
  let value = 0;
  let terms = 0;
  for (let a = 0; a < count; a++) {
    for (let b = 0; b < count; b++) {
      const term = (row[a] ?? 0) * (matrix[a * count + b] ?? 0) * (row[b] ?? 0);
      value += term;
      terms += Math.abs(term);
    }
  }
  if (value < -VARIANCE_ROUNDING * terms) {
    throw new DataError(
      `The option fixedCovariance is not positive semidefinite: the variance it adds to an estimate is negative, ${value}.`,
    );
  }
}
