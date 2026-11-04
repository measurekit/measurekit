/** The eigenvalues and eigenvectors of a symmetric matrix. */
export interface SymmetricEigen {
  /** The eigenvalues, largest first. */
  readonly values: Float64Array;
  /**
   * The unit eigenvectors, stored by rows: row k, `vectors[k * size + i]`, belongs to `values[k]`, signed so that its
   * entries sum to at least 0, as psych's `principal`.
   */
  readonly vectors: Float64Array;
  /** QL iterations run, over all the eigenvalues. */
  readonly iterations: number;
  /**
   * Whether each eigenvalue was found within `MAX_ITERATIONS` QL iterations; false for an entry that is not finite, or so
   * near the largest double that the arithmetic overflows.
   */
  readonly converged: boolean;
}

/**
 * The most QL iterations for one eigenvalue, as EISPACK's imtql2 and LAPACK's dsteqr allow: Wilkinson's shift converges
 * cubically, in one to three iterations for most eigenvalues.
 */
const MAX_ITERATIONS = 30;

/**
 * Returns the eigendecomposition of the symmetric `size` by `size` matrix `a`, stored by rows, by Householder's
 * reduction to a tridiagonal matrix and the implicit QL algorithm with Wilkinson's shift.
 *
 * @remarks
 * The reduction takes `size - 2` reflections, each of which zeroes a row left of the subdiagonal, and their product is
 * the basis of the tridiagonal matrix; each QL iteration chases a shift down an unreduced block by plane rotations,
 * which turn two rows of the basis, so that the rows end as the eigenvectors. The cost is about 8 size³ operations.
 * Each eigenvalue is exact to rounding relative to the largest in absolute value, as LAPACK's are. The matrix must be
 * symmetric, which the function does not check; only its lower triangle is read. A matrix with an entry that is not
 * finite has no decomposition: its values and vectors are NaN, and it does not converge.
 *
 * @see {@link https://doi.org/10.1007/BF02161841 | Martin, Reinsch and Wilkinson (1968)}
 * @see {@link https://doi.org/10.1007/BF02161360 | Martin and Wilkinson (1968)}
 */
export function symmetricEigen(a: Float64Array, size: number): SymmetricEigen {
  const work = Float64Array.from(a);
  if (!work.every(Number.isFinite)) {
    return {
      values: new Float64Array(size).fill(Number.NaN),
      vectors: work.fill(Number.NaN),
      iterations: 0,
      converged: false,
    };
  }
  const { diagonal, offDiagonal, norms } = tridiagonalize(work, size);
  const basis = reflectionProduct(work, size, norms);
  const { iterations, converged } = diagonalize(diagonal, offDiagonal, basis, size);
  const order = Array.from({ length: size }, (_, k) => k).sort(
    (x, y) => (diagonal[y] ?? 0) - (diagonal[x] ?? 0) || x - y,
  );
  const values = Float64Array.from(order, (k) => diagonal[k] ?? 0);
  // The reflections in `work` are spent: it receives the eigenvectors, so that no third matrix is allocated. The method
  // leaves the sign of each arbitrary, and a start or a seeded draw that reads the vectors must not depend on it.
  order.forEach((k, row) => {
    const vector = basis.subarray(k * size, (k + 1) * size);
    const sign = vector.reduce((sum, entry) => sum + entry, 0) < 0 ? -1 : 1;
    for (let i = 0; i < size; i++) work[row * size + i] = sign * (vector[i] ?? 0);
  });
  return { values, vectors: work, iterations, converged };
}

/** A symmetric tridiagonal matrix Qᵀ A Q, and the reflections whose product is Q. */
interface Tridiagonal {
  /** The diagonal. */
  readonly diagonal: Float64Array;
  /** Entry i joins entries i and i + 1 of the diagonal; the last is 0. */
  readonly offDiagonal: Float64Array;
  /** The h of the reflection of each row, 0 where the row needs none. */
  readonly norms: Float64Array;
}

/**
 * Reduces `a` to a tridiagonal matrix by reflections P = I - u uᵀ / h, from the last row up, as EISPACK's tred2: row i
 * of `a` keeps the u that zeroes its entries left of the subdiagonal, in its first i entries.
 */
function tridiagonalize(a: Float64Array, size: number): Tridiagonal {
  const diagonal = new Float64Array(size);
  const offDiagonal = new Float64Array(size);
  const norms = new Float64Array(size);
  const product = new Float64Array(size);
  for (let i = size - 1; i > 0; i--) {
    diagonal[i] = a[i * size + i] ?? 0;
    const u = a.subarray(i * size, i * size + i);
    // The second row has only its subdiagonal entry left of the diagonal, and needs no reflection.
    const { entry, norm } = i === 1 ? { entry: u[0] ?? 0, norm: 0 } : householder(u);
    offDiagonal[i - 1] = entry;
    norms[i] = norm;
    if (norm > 0) reflectBlock(a, size, u, norm, product);
  }
  diagonal[0] = a[0] ?? 0;
  return { diagonal, offDiagonal, norms };
}

/**
 * Turns the entries x of a row left of its diagonal into the u of the reflection P = I - u uᵀ / h that maps x to a
 * multiple of its last unit vector, and returns that multiple, the row's subdiagonal entry, and h; h is 0, and x is left
 * as it is, when x is 0.
 */
function householder(x: Float64Array): { entry: number; norm: number } {
  const last = x.length - 1;
  let scale = 0;
  for (const value of x) scale += Math.abs(value);
  if (scale === 0) return { entry: x[last] ?? 0, norm: 0 };
  // Scaling by the sum of the magnitudes keeps the sum of squares from overflowing or underflowing.
  let squares = 0;
  for (let k = 0; k <= last; k++) {
    const scaled = (x[k] ?? 0) / scale;
    x[k] = scaled;
    squares += scaled * scaled;
  }
  const end = x[last] ?? 0;
  // The sign opposite the last entry's keeps u's last entry from cancelling.
  const length = end >= 0 ? -Math.sqrt(squares) : Math.sqrt(squares);
  x[last] = end - length;
  return { entry: scale * length, norm: squares - end * length };
}

/**
 * Replaces the leading block of `a` that is as wide as `u` with P A P, for the reflection P = I - u uᵀ / h: with
 * p = A u / h and q = p - (uᵀ p / 2h) u, P A P = A - q uᵀ - u qᵀ. Only the lower triangle is read and written.
 * `product` is work space at least as long as `u`.
 */
function reflectBlock(a: Float64Array, size: number, u: Float64Array, norm: number, product: Float64Array): void {
  const count = u.length;
  lowerProduct(a, size, u, product);
  let projection = 0;
  for (let j = 0; j < count; j++) {
    product[j] = (product[j] ?? 0) / norm;
    projection += (product[j] ?? 0) * (u[j] ?? 0);
  }
  const half = projection / (2 * norm);
  for (let j = 0; j < count; j++) product[j] = (product[j] ?? 0) - half * (u[j] ?? 0);
  for (let j = 0; j < count; j++) {
    const row = j * size;
    const qj = product[j] ?? 0;
    const uj = u[j] ?? 0;
    for (let k = 0; k <= j; k++) a[row + k] = (a[row + k] ?? 0) - qj * (u[k] ?? 0) - uj * (product[k] ?? 0);
  }
}

/**
 * Writes A u to `product`, for the leading block of `a` that is as wide as `u`, from its lower triangle: each entry
 * below the diagonal adds to two entries of the product.
 */
function lowerProduct(a: Float64Array, size: number, u: Float64Array, product: Float64Array): void {
  product.fill(0, 0, u.length);
  for (let j = 0; j < u.length; j++) {
    const row = j * size;
    const uj = u[j] ?? 0;
    let sum = 0;
    for (let k = 0; k < j; k++) {
      const entry = a[row + k] ?? 0;
      sum += entry * (u[k] ?? 0);
      product[k] = (product[k] ?? 0) + entry * uj;
    }
    product[j] = (product[j] ?? 0) + sum + (a[row + j] ?? 0) * uj;
  }
}

/**
 * Returns Qᵀ, by rows, for the product Q of the reflections that `tridiagonalize` left in `a`: the reflections multiply
 * it from the third row down, as the one of row i changes only the first i rows and columns of the product so far,
 * (4/3) size³ operations in all.
 */
function reflectionProduct(a: Float64Array, size: number, norms: Float64Array): Float64Array {
  const product = new Float64Array(size * size);
  for (let i = 0; i < size; i++) product[i * size + i] = 1;
  for (let i = 2; i < size; i++) {
    const norm = norms[i] ?? 0;
    if (norm > 0) reflectRows(product, size, a.subarray(i * size, i * size + i), norm);
  }
  return product;
}

/** Replaces each of the first rows of `product`, as many as `u` is long, with its product by I - u uᵀ / h. */
function reflectRows(product: Float64Array, size: number, u: Float64Array, norm: number): void {
  const count = u.length;
  for (let row = 0; row < count * size; row += size) {
    let projection = 0;
    for (let c = 0; c < count; c++) projection += (product[row + c] ?? 0) * (u[c] ?? 0);
    projection /= norm;
    for (let c = 0; c < count; c++) product[row + c] = (product[row + c] ?? 0) - projection * (u[c] ?? 0);
  }
}

/**
 * Diagonalizes the symmetric tridiagonal matrix of `diagonal` and `offDiagonal` by the implicit QL algorithm, as
 * EISPACK's imtql2, and turns the rows of `vectors` with each rotation, so that they end as the eigenvectors. The
 * diagonal receives the eigenvalues, unordered.
 */
function diagonalize(
  diagonal: Float64Array,
  offDiagonal: Float64Array,
  vectors: Float64Array,
  size: number,
): { iterations: number; converged: boolean } {
  let iterations = 0;
  for (let start = 0; start < size; start++) {
    let count = 0;
    for (let end = blockEnd(diagonal, offDiagonal, start); end > start; end = blockEnd(diagonal, offDiagonal, start)) {
      if (count === MAX_ITERATIONS) return { iterations, converged: false };
      count++;
      iterations++;
      chaseShift(diagonal, offDiagonal, vectors, size, start, end);
    }
  }
  return { iterations, converged: true };
}

/**
 * Returns the end of the unreduced block that starts at `start`: the first m from it whose off-diagonal entry is
 * negligible beside its two diagonal neighbors, EISPACK's test, or the last index.
 */
function blockEnd(diagonal: Float64Array, offDiagonal: Float64Array, start: number): number {
  const last = diagonal.length - 1;
  for (let m = start; m < last; m++) {
    const neighbors = Math.abs(diagonal[m] ?? 0) + Math.abs(diagonal[m + 1] ?? 0);
    if (Math.abs(offDiagonal[m] ?? 0) <= Number.EPSILON * neighbors) return m;
  }
  return last;
}

/**
 * Runs one implicit QL iteration on the block from `start` to `end`: the shift is the eigenvalue of the leading 2 by 2
 * block nearer its first diagonal entry, Wilkinson's, and plane rotations from the end up restore the tridiagonal form.
 */
function chaseShift(
  diagonal: Float64Array,
  offDiagonal: Float64Array,
  vectors: Float64Array,
  size: number,
  start: number,
  end: number,
): void {
  const first = diagonal[start] ?? 0;
  const coupling = offDiagonal[start] ?? 0;
  const ratio = ((diagonal[start + 1] ?? 0) - first) / (2 * coupling);
  const radius = hypotenuse(ratio, 1);
  let g = (diagonal[end] ?? 0) - first + coupling / (ratio + (ratio >= 0 ? radius : -radius));
  let sine = 1;
  let cosine = 1;
  let shift = 0;
  for (let i = end - 1; i >= start; i--) {
    const f = sine * (offDiagonal[i] ?? 0);
    const b = cosine * (offDiagonal[i] ?? 0);
    const r = hypotenuse(f, g);
    offDiagonal[i + 1] = r;
    if (r === 0) {
      // The rotation underflowed: the block splits at i + 1, and the next iteration starts on the shorter block.
      diagonal[i + 1] = (diagonal[i + 1] ?? 0) - shift;
      offDiagonal[end] = 0;
      return;
    }
    sine = f / r;
    cosine = g / r;
    g = (diagonal[i + 1] ?? 0) - shift;
    const t = ((diagonal[i] ?? 0) - g) * sine + 2 * cosine * b;
    shift = sine * t;
    diagonal[i + 1] = g + shift;
    g = cosine * t - b;
    rotateRows(vectors, size, i, cosine, sine);
  }
  diagonal[start] = first - shift;
  offDiagonal[start] = g;
  offDiagonal[end] = 0;
}

/**
 * Returns √(x² + y²) with no overflow or underflow of the squares, as LAPACK's dlapy2: `Math.hypot`, which takes any
 * number of arguments, costs a fifth of the whole decomposition at 25 items.
 */
function hypotenuse(x: number, y: number): number {
  const a = Math.abs(x);
  const b = Math.abs(y);
  if (a > b) {
    const ratio = b / a;
    return a * Math.sqrt(1 + ratio * ratio);
  }
  if (b === 0) return 0;
  const ratio = a / b;
  return b * Math.sqrt(1 + ratio * ratio);
}

/** Turns rows i and i + 1 of `vectors` by the plane rotation of `cosine` and `sine`. */
function rotateRows(vectors: Float64Array, size: number, i: number, cosine: number, sine: number): void {
  const upper = i * size;
  const lower = upper + size;
  for (let k = 0; k < size; k++) {
    const x = vectors[upper + k] ?? 0;
    const y = vectors[lower + k] ?? 0;
    vectors[lower + k] = sine * x + cosine * y;
    vectors[upper + k] = cosine * x - sine * y;
  }
}
