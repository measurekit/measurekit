import { checkBoolean, checkChoice, checkFinite, checkOptions, checkStopping } from "@measurekit/core";
import { gradientProjection, type RotationCriterion } from "./gradient-projection.js";
import { invert } from "./linear-algebra.js";
import { multiply, transpose } from "./matrix.js";

/** The rotations: `"varimax"` is orthogonal, `"oblimin"` oblique. */
export type RotationMethod = "varimax" | "oblimin";

/** Options of a rotation of loadings. */
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

/** The options of a rotation with their defaults. */
export type ResolvedRotation = Required<RotationOptions>;

/** The rule that fixes the sign of each rotated dimension, which a rotation leaves free. */
export type RotationSign =
  /** The dimension's largest loading in absolute value is positive, as mirt's. */
  | "largest"
  /** The dimension's loadings sum to a positive number, as psych's. */
  | "sum";

/** A rotated solution: the rotation matrix, the loadings and the correlations of the rotated dimensions. */
export interface RotatedMatrix {
  /** The rotation matrix T by rows: the rotated loadings are Λ (Tᵀ)⁻¹ and their correlations Tᵀ T. */
  readonly rotation: Float64Array;
  /** The inverse transpose of the rotation matrix, (Tᵀ)⁻¹, by rows. */
  readonly transform: Float64Array;
  /** The rotated loadings Λ (Tᵀ)⁻¹, rows by dimensions, by rows. */
  readonly loadings: Float64Array;
  /** The correlations Tᵀ T between the rotated dimensions by rows: the identity for an orthogonal rotation. */
  readonly correlations: Float64Array;
  /** Whether the norm of the projected gradient fell to the tolerance. */
  readonly converged: boolean;
  /** Iterations run. */
  readonly iterations: number;
}

const METHODS: readonly RotationMethod[] = ["varimax", "oblimin"];
const DEFAULT_TOLERANCE = 1e-10;
const DEFAULT_MAX_ITERATIONS = 1000;

/**
 * Checks the options of a rotation and returns them with their defaults.
 *
 * @throws `DataError` when an option is invalid.
 */
export function resolveRotation(options: RotationOptions = {}): ResolvedRotation {
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
  return { method, normalize, gamma, tolerance, maxIterations };
}

/**
 * Rotates the loadings of items on dimensions, `rows` by `columns` by rows, to a simpler structure.
 *
 * @remarks
 * Gradient projection from the identity rotation, with the varimax criterion for an orthogonal rotation and the oblimin
 * criterion for an oblique one. For an oblique rotation T the loadings are Λ (Tᵀ)⁻¹ and the correlations Tᵀ T. With
 * `normalize` each row is scaled to unit length for the search, and the rotation found applies to the loadings as they
 * are. The sign of each rotated dimension follows `sign`; the criteria do not depend on it.
 *
 * @see {@link https://doi.org/10.1007/BF02294840 | Jennrich (2001)}
 * @see {@link https://doi.org/10.1007/BF02294706 | Jennrich (2002)}
 * @see {@link https://doi.org/10.1177/0013164404272507 | Bernaards and Jennrich (2005)}
 */
export function rotateLoadingMatrix(
  unrotated: Float64Array,
  rows: number,
  columns: number,
  { method, normalize, gamma, tolerance, maxIterations }: ResolvedRotation,
  sign: RotationSign,
): RotatedMatrix {
  const weights = Float64Array.from({ length: rows }, (_, i) =>
    normalize ? Math.max(Math.hypot(...unrotated.subarray(i * columns, (i + 1) * columns)), Number.EPSILON) : 1,
  );
  const scaled = unrotated.map((loading, k) => loading / (weights[Math.floor(k / columns)] ?? 1));
  const criterion: RotationCriterion = method === "varimax" ? { kind: "varimax" } : { kind: "oblimin", gamma };
  const result = gradientProjection(scaled, rows, columns, criterion, tolerance, maxIterations);
  const rotation = signed(result.rotation, unrotated, rows, columns, sign);
  const transform = inverseTranspose(rotation, columns);
  const loadings = new Float64Array(rows * columns);
  multiply(unrotated, transform, rows, columns, columns, loadings);
  const correlations = new Float64Array(columns * columns);
  multiply(transpose(rotation, columns, columns), rotation, columns, columns, columns, correlations);
  return { rotation, transform, loadings, correlations, converged: result.converged, iterations: result.iterations };
}

function inverseTranspose(rotation: Float64Array, columns: number): Float64Array {
  return transpose(invert(rotation, columns) ?? new Float64Array(columns * columns), columns, columns);
}

/** Flips the columns of the rotation so that each rotated dimension has the sign that `sign` asks for. */
function signed(
  rotation: Float64Array,
  unrotated: Float64Array,
  rows: number,
  columns: number,
  sign: RotationSign,
): Float64Array {
  const loadings = new Float64Array(rows * columns);
  multiply(unrotated, inverseTranspose(rotation, columns), rows, columns, columns, loadings);
  const flipped = Float64Array.from(rotation);
  for (let j = 0; j < columns; j++) {
    if (anchor(loadings, rows, columns, j, sign) >= 0) continue;
    for (let i = 0; i < columns; i++) flipped[i * columns + j] = -(rotation[i * columns + j] ?? 0);
  }
  return flipped;
}

/** Returns the value of a dimension whose sign `sign` sets: its largest loading in absolute value, or their sum. */
function anchor(loadings: Float64Array, rows: number, columns: number, j: number, sign: RotationSign): number {
  let largest = 0;
  let sum = 0;
  for (let i = 0; i < rows; i++) {
    const loading = loadings[i * columns + j] ?? 0;
    sum += loading;
    if (Math.abs(loading) > Math.abs(largest)) largest = loading;
  }
  return sign === "largest" ? largest : sum;
}
