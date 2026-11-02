import { checkArray, checkRecord, DataError } from "@measurekit/core";
import { multiply, type RotationOptions, resolveRotation, rotateLoadingMatrix } from "../math/internal.js";
import type { MultidimensionalCalibration } from "./calibration.js";
import { checkMultidimensionalItem } from "./items.js";
import type { MultidimensionalTestItem } from "./model.js";

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
 * @param calibration - A calibration of uncorrelated dimensions.
 * @param options - The rotation method, its normalization and parameter, and the stopping rules.
 * @returns The rotated loadings, the correlations of the dimensions, the rotation matrix, the items on the rotated
 *   axes, whether the rotation converged and the iterations run.
 * @throws `DataError` when the options are invalid, the calibration has fewer than two dimensions or correlated
 * dimensions, or is not a calibration.
 * @see {@link https://doi.org/10.1007/BF02294840 | Jennrich (2001)}
 * @see {@link https://doi.org/10.1007/BF02294706 | Jennrich (2002)}
 * @see {@link https://doi.org/10.1177/0013164404272507 | Bernaards and Jennrich (2005)}
 */
export function rotateLoadings<TParameters>(
  calibration: MultidimensionalCalibration<TParameters>,
  options: RotationOptions = {},
): RotatedLoadings<TParameters> {
  const resolved = resolveRotation(options);
  const dimensions = checkCalibration(calibration);
  const items = calibration.items.length;
  const result = rotateLoadingMatrix(
    Float64Array.from(calibration.loadings.flat()),
    items,
    dimensions,
    resolved,
    "largest",
  );
  return {
    loadings: rows(result.loadings, items, dimensions),
    correlations: rows(result.correlations, dimensions, dimensions),
    rotation: rows(result.rotation, dimensions, dimensions),
    items: calibration.items.map((item) => ({
      id: item.id,
      model: item.model,
      parameters: rotatedParameters(item, result.transform, dimensions),
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
