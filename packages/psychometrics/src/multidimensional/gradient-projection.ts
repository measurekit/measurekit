import { symmetricEigen } from "../math/eigen.js";
import { invert } from "../math/linear-algebra.js";
import { identity, multiply, transpose } from "../math/matrix.js";

/** The criterion of the varimax rotation, which is orthogonal, or of the oblimin rotation, which is oblique. */
export type RotationCriterion = { readonly kind: "varimax" } | { readonly kind: "oblimin"; readonly gamma: number };

/** The outcome of a gradient projection rotation. */
export interface Rotation {
  /** The rotation matrix T by rows: the rotated loadings are A (Tᵀ)⁻¹ and their correlations Tᵀ T. */
  readonly rotation: Float64Array;
  /** Iterations run. */
  readonly iterations: number;
  /** Whether the norm of the projected gradient fell to the tolerance. */
  readonly converged: boolean;
}

/** A criterion's value at loadings `L` (`rows` by `columns`, by rows) and its gradient with respect to them. */
type Criterion = (loadings: Float64Array, rows: number, columns: number, gradient: Float64Array) => number;

/**
 * The matrices T a rotation searches, and how it moves on them: the value of the criterion and its gradient with
 * respect to T, the projection of a gradient onto the directions that keep T admissible, and the return of a point
 * to the admissible matrices.
 */
interface Manifold {
  /** Writes the gradient of the criterion at T into `gradient` and returns its value. */
  evaluate(rotation: Float64Array, gradient: Float64Array): number;
  /** Writes the projection of `gradient` at T into `out`. */
  project(rotation: Float64Array, gradient: Float64Array, out: Float64Array): void;
  /** Writes the admissible matrix nearest to `x` into `out`. */
  retract(x: Float64Array, out: Float64Array): void;
}

/** The most halvings of the step in the line search, as GPArotation. */
const MAX_HALVINGS = 10;
/** The sufficient decrease of the line search, as a fraction of the squared norm of the projected gradient. */
const SUFFICIENT_DECREASE = 0.5;
/** The number of latest criterion values that a step may improve on, in the nonmonotone line search. */
const WINDOW = 10;
/** The bounds of the Barzilai-Borwein step. */
const MIN_STEP = 1e-10;
const MAX_STEP = 20;

/**
 * Rotates the loadings `A` (`rows` by `columns`, by rows) from the identity by gradient projection.
 *
 * @remarks
 * The orthogonal rotation keeps T orthogonal; the oblique one keeps the columns of T at unit length. The step along
 * the projected gradient is the Barzilai-Borwein step, halved until the criterion falls enough below the largest of its
 * latest values, as the default algorithm of GPArotation.
 *
 * @see {@link https://doi.org/10.1007/BF02294840 | Jennrich (2001)}
 * @see {@link https://doi.org/10.1007/BF02294706 | Jennrich (2002)}
 * @see {@link https://doi.org/10.1177/0013164404272507 | Bernaards and Jennrich (2005)}
 * @see {@link https://doi.org/10.1093/imanum/8.1.141 | Barzilai and Borwein (1988)}
 * @see {@link https://doi.org/10.1137/0723046 | Grippo, Lampariello and Lucidi (1986)}
 */
export function gradientProjection(
  loadings: Float64Array,
  rows: number,
  columns: number,
  criterion: RotationCriterion,
  tolerance: number,
  maxIterations: number,
): Rotation {
  const evaluate = criterion.kind === "varimax" ? varimax : oblimin(criterion.gamma);
  const manifold =
    criterion.kind === "varimax"
      ? orthogonal(loadings, rows, columns, evaluate)
      : oblique(loadings, rows, columns, evaluate);
  return descend(manifold, columns, tolerance, maxIterations);
}

function descend(manifold: Manifold, size: number, tolerance: number, maxIterations: number): Rotation {
  const entries = size * size;
  let rotation: Float64Array = identity(size);
  let gradient: Float64Array = new Float64Array(entries);
  let trial: Float64Array = new Float64Array(entries);
  let trialGradient: Float64Array = new Float64Array(entries);
  const candidate = new Float64Array(entries);
  const projected = new Float64Array(entries);
  const previousProjected = new Float64Array(entries);
  let previousRotation: Float64Array | null = null;
  const values = [manifold.evaluate(rotation, gradient)];
  let step = 1;
  for (let iteration = 0; iteration <= maxIterations; iteration++) {
    manifold.project(rotation, gradient, projected);
    const norm = Math.hypot(...projected);
    if (norm < tolerance) return { rotation, iterations: iteration, converged: true };
    if (previousRotation === null) step *= 2;
    else step = barzilaiBorwein(rotation, previousRotation, projected, previousProjected, step);
    const target = Math.max(...values.slice(-WINDOW));
    let value = 0;
    for (let halving = 0; halving <= MAX_HALVINGS; halving++) {
      for (let k = 0; k < entries; k++) candidate[k] = (rotation[k] ?? 0) - step * (projected[k] ?? 0);
      manifold.retract(candidate, trial);
      value = manifold.evaluate(trial, trialGradient);
      if (target - value > SUFFICIENT_DECREASE * norm * norm * step) break;
      step /= 2;
    }
    previousRotation = rotation;
    previousProjected.set(projected);
    [rotation, trial] = [trial, previousRotation];
    [gradient, trialGradient] = [trialGradient, gradient];
    values.push(value);
  }
  return { rotation, iterations: maxIterations, converged: false };
}

/** Returns the Barzilai-Borwein step Σ ΔT² / |Σ ΔT ΔGp|, within bounds; the step itself when the gradient is the same. */
function barzilaiBorwein(
  rotation: Float64Array,
  previous: Float64Array,
  projected: Float64Array,
  previousProjected: Float64Array,
  step: number,
): number {
  let changeSquares = 0;
  let gradientSquares = 0;
  let cross = 0;
  for (let k = 0; k < rotation.length; k++) {
    const change = (rotation[k] ?? 0) - (previous[k] ?? 0);
    const gradientChange = (projected[k] ?? 0) - (previousProjected[k] ?? 0);
    changeSquares += change * change;
    gradientSquares += gradientChange * gradientChange;
    cross += change * gradientChange;
  }
  return gradientSquares > 0 ? Math.max(MIN_STEP, Math.min(changeSquares / Math.abs(cross), MAX_STEP)) : step;
}

/** The orthogonal matrices: the loadings are A T, and the gradient Aᵀ Gq; a point returns by the polar factor. */
function orthogonal(a: Float64Array, rows: number, columns: number, criterion: Criterion): Manifold {
  const loadings = new Float64Array(rows * columns);
  const loadingsGradient = new Float64Array(rows * columns);
  const at = transpose(a, rows, columns);
  return {
    evaluate(rotation, gradient) {
      multiply(a, rotation, rows, columns, columns, loadings);
      const value = criterion(loadings, rows, columns, loadingsGradient);
      multiply(at, loadingsGradient, columns, rows, columns, gradient);
      return value;
    },
    project(rotation, gradient, out) {
      // G - T (Tᵀ G + Gᵀ T) / 2
      const product = new Float64Array(columns * columns);
      multiply(transpose(rotation, columns, columns), gradient, columns, columns, columns, product);
      const symmetric = Float64Array.from(
        product,
        (entry, k) => (entry + (product[transposedIndex(k, columns)] ?? 0)) / 2,
      );
      multiply(rotation, symmetric, columns, columns, columns, product);
      for (let k = 0; k < out.length; k++) out[k] = (gradient[k] ?? 0) - (product[k] ?? 0);
    },
    retract: (x, out) => out.set(polarFactor(x, columns)),
  };
}

/**
 * The matrices with columns of unit length: the loadings are A (Tᵀ)⁻¹ and the gradient -(Lᵀ Gq T⁻¹)ᵀ; a point returns
 * by scaling its columns.
 */
function oblique(a: Float64Array, rows: number, columns: number, criterion: Criterion): Manifold {
  const loadings = new Float64Array(rows * columns);
  const loadingsGradient = new Float64Array(rows * columns);
  return {
    evaluate(rotation, gradient) {
      const inverse = invert(rotation, columns) ?? identity(columns);
      multiply(a, transpose(inverse, columns, columns), rows, columns, columns, loadings);
      const value = criterion(loadings, rows, columns, loadingsGradient);
      const product = new Float64Array(columns * columns);
      multiply(transpose(loadings, rows, columns), loadingsGradient, columns, rows, columns, product);
      const full = new Float64Array(columns * columns);
      multiply(product, inverse, columns, columns, columns, full);
      gradient.set(transpose(full, columns, columns).map((entry) => -entry));
      return value;
    },
    project(rotation, gradient, out) {
      // G - T diag(column sums of T ∘ G)
      for (let j = 0; j < columns; j++) {
        let along = 0;
        for (let i = 0; i < columns; i++) along += (rotation[i * columns + j] ?? 0) * (gradient[i * columns + j] ?? 0);
        for (let i = 0; i < columns; i++) {
          out[i * columns + j] = (gradient[i * columns + j] ?? 0) - (rotation[i * columns + j] ?? 0) * along;
        }
      }
    },
    retract(x, out) {
      for (let j = 0; j < columns; j++) {
        let squares = 0;
        for (let i = 0; i < columns; i++) squares += (x[i * columns + j] ?? 0) ** 2;
        const norm = Math.sqrt(squares);
        for (let i = 0; i < columns; i++) out[i * columns + j] = (x[i * columns + j] ?? 0) / norm;
      }
    },
  };
}

/** The varimax criterion: f = -Σ (L² - column means of L²)² / 4. */
const varimax: Criterion = (loadings, rows, columns, gradient) => {
  let value = 0;
  for (let j = 0; j < columns; j++) {
    let mean = 0;
    for (let i = 0; i < rows; i++) mean += (loadings[i * columns + j] ?? 0) ** 2 / rows;
    for (let i = 0; i < rows; i++) {
      const l = loadings[i * columns + j] ?? 0;
      const centered = l * l - mean;
      gradient[i * columns + j] = -l * centered;
      value -= (centered * centered) / 4;
    }
  }
  return value;
};

/** The oblimin criterion with parameter gamma: f = Σ L² ∘ X / 4 with X the sums of the squares of the other columns. */
function oblimin(gamma: number): Criterion {
  return (loadings, rows, columns, gradient) => {
    const others = otherSquares(loadings, rows, columns);
    if (gamma !== 0) subtractColumnShares(others, rows, columns, gamma / rows);
    let value = 0;
    for (let k = 0; k < rows * columns; k++) {
      const l = loadings[k] ?? 0;
      gradient[k] = l * (others[k] ?? 0);
      value += l * l * (others[k] ?? 0);
    }
    return value / 4;
  };
}

/** Returns, for each loading, the sum of the squares of the other loadings of its row. */
function otherSquares(loadings: Float64Array, rows: number, columns: number): Float64Array {
  const others = new Float64Array(rows * columns);
  for (let i = 0; i < rows; i++) {
    let total = 0;
    for (let j = 0; j < columns; j++) total += (loadings[i * columns + j] ?? 0) ** 2;
    for (let j = 0; j < columns; j++) others[i * columns + j] = total - (loadings[i * columns + j] ?? 0) ** 2;
  }
  return others;
}

/** Subtracts `share` times the column's sum from each entry of the column. */
function subtractColumnShares(values: Float64Array, rows: number, columns: number, share: number): void {
  for (let j = 0; j < columns; j++) {
    let sum = 0;
    for (let i = 0; i < rows; i++) sum += values[i * columns + j] ?? 0;
    for (let i = 0; i < rows; i++) values[i * columns + j] = (values[i * columns + j] ?? 0) - share * sum;
  }
}

/** Returns the index of entry `k` of a `size` by `size` matrix in its transpose. */
function transposedIndex(k: number, size: number): number {
  return (k % size) * size + Math.floor(k / size);
}

/** Returns the orthogonal factor U Vᵀ of the singular value decomposition X = U D Vᵀ: X (Xᵀ X)^(-1/2). */
function polarFactor(x: Float64Array, size: number): Float64Array {
  const gram = new Float64Array(size * size);
  multiply(transpose(x, size, size), x, size, size, size, gram);
  const { values, vectors } = symmetricEigen(gram, size);
  const inverseRoot = new Float64Array(size * size);
  for (let k = 0; k < size; k++) {
    const scale = (values[k] ?? 0) > 0 ? 1 / Math.sqrt(values[k] ?? 1) : 0;
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        inverseRoot[i * size + j] =
          (inverseRoot[i * size + j] ?? 0) + scale * (vectors[k * size + i] ?? 0) * (vectors[k * size + j] ?? 0);
      }
    }
  }
  const result = new Float64Array(size * size);
  multiply(x, inverseRoot, size, size, size, result);
  return result;
}
