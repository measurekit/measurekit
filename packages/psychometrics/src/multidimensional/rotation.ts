import {
  checkArray,
  checkBoolean,
  checkChoice,
  checkFinite,
  checkOptions,
  checkRecord,
  checkStopping,
  DataError,
} from "@measurekit/core";
import { invert } from "../math/linear-algebra.js";
import type { MultidimensionalCalibration } from "./calibration.js";
import { gradientProjection, type RotationCriterion } from "./gradient-projection.js";
import { checkMultidimensionalItem } from "./items.js";
import { multiply, transpose } from "./matrix.js";
import type { MultidimensionalTestItem } from "./model.js";

/** The rotations: `"varimax"` is orthogonal, `"oblimin"` oblique. */
export type RotationMethod = "varimax" | "oblimin";

/** Options of {@link rotateLoadings}. */
export interface RotationOptions {
  /**
   * The rotation criterion.
   *
   * @defaultValue `"oblimin"`
   */
  readonly method?: RotationMethod;
  /**
   * Whether to scale each item's loadings to unit length before rotating and back after (Kaiser normalization).
   *
   * @defaultValue `false`
   */
  readonly normalize?: boolean;
  /**
   * The parameter γ of the oblimin criterion: 0 is quartimin and 0.5 biquartimin. Ignored by varimax.
   *
   * @defaultValue `0`
   */
  readonly gamma?: number;
  /**
   * The rotation has converged when the norm of the projected gradient falls to this.
   *
   * @defaultValue `1e-10`
   */
  readonly tolerance?: number;
  /**
   * The most iterations to run.
   *
   * @defaultValue `1000`
   */
  readonly maxIterations?: number;
}

/** A rotated solution of a calibration. */
export interface RotatedLoadings<TParameters> {
  /** The rotated standardized loadings, items by dimensions. */
  readonly loadings: readonly (readonly number[])[];
  /** The correlations between the rotated dimensions: the identity for an orthogonal rotation. */
  readonly correlations: readonly (readonly number[])[];
  /** The rotation matrix T: the rotated loadings are Λ (Tᵀ)⁻¹ and their correlations are Tᵀ T. */
  readonly rotation: readonly (readonly number[])[];
  /** The items with the slopes of the rotated solution, the intercepts and other parameters unchanged. */
  readonly items: readonly MultidimensionalTestItem<TParameters>[];
  /** Whether the norm of the projected gradient fell to the tolerance. */
  readonly converged: boolean;
  /** Iterations run. */
  readonly iterations: number;
}

const METHODS: readonly RotationMethod[] = ["varimax", "oblimin"];
const DEFAULT_TOLERANCE = 1e-10;
const DEFAULT_MAX_ITERATIONS = 1000;

/**
 * Rotates the standardized loadings of a calibration of uncorrelated dimensions, such as an exploratory one, to a
 * simpler structure.
 *
 * @remarks
 * Gradient projection from the identity rotation, with the varimax criterion for an orthogonal rotation and the oblimin
 * criterion for an oblique one. For an oblique rotation T, the loadings are Λ (Tᵀ)⁻¹ and the correlations Tᵀ T. The
 * slopes of the rotated solution are a (Tᵀ)⁻¹ = 1.702 λ / √(1 - h²), with the communalities h² unchanged, so that
 * the rotated `items` with the rotated `correlations` as the prior score as the original ones do, in the rotated
 * dimensions, as mirt's `fscores(rotate = )` does. The sign of each rotated dimension is chosen so that its largest
 * loading in absolute value is positive, as mirt's.
 *
 * @throws `DataError` when the options are invalid, the calibration has fewer than two dimensions or correlated
 * dimensions, or is not a calibration.
 *
 * @see {@link https://doi.org/10.1007/BF02294840 | Jennrich (2001)}
 * @see {@link https://doi.org/10.1007/BF02294706 | Jennrich (2002)}
 * @see {@link https://doi.org/10.1177/0013164404272507 | Bernaards and Jennrich (2005)}
 */
export function rotateLoadings<TParameters>(
  calibration: MultidimensionalCalibration<TParameters>,
  options: RotationOptions = {},
): RotatedLoadings<TParameters> {
  checkOptions(options);
  const {
    method = "oblimin",
    normalize = false,
    gamma = 0,
    tolerance = DEFAULT_TOLERANCE,
    maxIterations = DEFAULT_MAX_ITERATIONS,
  } = options;
  checkChoice(method, "The option method", METHODS);
  checkBoolean(normalize, "The option normalize");
  checkFinite(gamma, "The option gamma");
  checkStopping(tolerance, maxIterations);
  const dimensions = checkCalibration(calibration);
  const items = calibration.items.length;
  const unrotated = Float64Array.from(calibration.loadings.flat());
  const weights = Float64Array.from(calibration.loadings, (row) =>
    normalize ? Math.max(Math.hypot(...row), Number.EPSILON) : 1,
  );
  const scaled = unrotated.map((loading, k) => loading / (weights[Math.floor(k / dimensions)] ?? 1));
  const criterion: RotationCriterion = method === "varimax" ? { kind: "varimax" } : { kind: "oblimin", gamma };
  const result = gradientProjection(scaled, items, dimensions, criterion, tolerance, maxIterations);
  const rotation = signed(result.rotation, unrotated, items, dimensions);
  const inverse = invert(rotation, dimensions) ?? new Float64Array(dimensions * dimensions);
  const inverseTranspose = transpose(inverse, dimensions, dimensions);
  const rotated = new Float64Array(items * dimensions);
  multiply(unrotated, inverseTranspose, items, dimensions, dimensions, rotated);
  const correlations = new Float64Array(dimensions * dimensions);
  multiply(transpose(rotation, dimensions, dimensions), rotation, dimensions, dimensions, dimensions, correlations);
  return {
    loadings: rows(rotated, items, dimensions),
    correlations: rows(correlations, dimensions, dimensions),
    rotation: rows(rotation, dimensions, dimensions),
    items: calibration.items.map((item) => ({
      id: item.id,
      model: item.model,
      parameters: rotatedParameters(item, inverseTranspose, dimensions),
    })),
    converged: result.converged,
    iterations: result.iterations,
  };
}

/**
 * Checks that the calibration has at least two uncorrelated dimensions and consistent loadings, and returns the number
 * of dimensions.
 */
function checkCalibration(calibration: MultidimensionalCalibration<unknown>): number {
  checkRecord(calibration, "The calibration must be the result of calibrateMultidimensional");
  const { items, loadings, correlations } = calibration;
  checkArray(items, "The items of the calibration must be an array");
  checkArray(loadings, "The loadings of the calibration must be an array");
  checkArray(correlations, "The correlations of the calibration must be an array");
  const dimensions = correlations.length;
  if (dimensions < 2) throw new DataError(`Rotation needs at least two dimensions, found ${dimensions}.`);
  items.forEach((item, i) => {
    checkMultidimensionalItem(item, `Item ${i + 1}`);
  });
  const shaped =
    loadings.length === items.length &&
    loadings.every((row) => Array.isArray(row) && row.length === dimensions && row.every(Number.isFinite)) &&
    correlations.every(
      (row, i) =>
        Array.isArray(row) && row.length === dimensions && row.every((value, j) => value === (i === j ? 1 : 0)),
    );
  if (!shaped) {
    throw new DataError(
      `Rotation needs the loadings of ${items.length} items on ${dimensions} dimensions and the identity as the correlations, as an exploratory calibration has.`,
    );
  }
  return dimensions;
}

/** Flips the columns of the rotation so that each rotated dimension's largest loading in absolute value is positive. */
function signed(rotation: Float64Array, unrotated: Float64Array, items: number, dimensions: number): Float64Array {
  const inverseTranspose = transpose(invert(rotation, dimensions) ?? rotation, dimensions, dimensions);
  const loadings = new Float64Array(items * dimensions);
  multiply(unrotated, inverseTranspose, items, dimensions, dimensions, loadings);
  const flipped = Float64Array.from(rotation);
  for (let j = 0; j < dimensions; j++) {
    let largest = 0;
    for (let i = 0; i < items; i++)
      if (Math.abs(loadings[i * dimensions + j] ?? 0) > Math.abs(largest)) largest = loadings[i * dimensions + j] ?? 0;
    if (largest < 0)
      for (let i = 0; i < dimensions; i++) flipped[i * dimensions + j] = -(rotation[i * dimensions + j] ?? 0);
  }
  return flipped;
}

/** Returns an item's parameters with the slopes a (Tᵀ)⁻¹. */
function rotatedParameters<TParameters>(
  item: MultidimensionalTestItem<TParameters>,
  inverseTranspose: Float64Array,
  dimensions: number,
): TParameters {
  const vector = item.model.toVector(item.parameters);
  const slopes = new Float64Array(dimensions);
  multiply(vector.subarray(0, dimensions), inverseTranspose, 1, dimensions, dimensions, slopes);
  vector.set(slopes);
  return item.model.fromVector(vector, dimensions);
}

function rows(matrix: Float64Array, count: number, columns: number): number[][] {
  return Array.from({ length: count }, (_, i) => Array.from(matrix.subarray(i * columns, (i + 1) * columns)));
}
