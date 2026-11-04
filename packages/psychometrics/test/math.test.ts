import {
  chiSquareUpperTail,
  createRandom,
  invertPositiveDefinite,
  logistic,
  normalCdf,
  normalDensity,
  normalQuantile,
} from "@measurekit/core";
import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  adaptiveIntegral,
  findIncreasingRoot,
  findRoot,
  gaussHermite,
  identity,
  invert,
  limitedInformationStatistic,
  mirror,
  multiply,
  newton,
  noncentralChiSquareCdf,
  numericalHessian,
  pseudoInverse,
  rectangular,
  resolveRotation,
  rotateLoadingMatrix,
  symmetricEigen,
  symmetrized,
  trace,
  traceOfProduct,
  transpose,
} from "../src/math/internal.js";
import { defined, expectClose } from "./assertions.js";
import { type DimensionalityFactorsFixture, readFixture } from "./fixtures.js";

const factors = readFixture<DimensionalityFactorsFixture>("dimensionality-factors.json");

interface MathEigenFixture {
  cases: { name: string; size: number; matrix: number[][]; values: number[]; vectors: (number[] | null)[] }[];
}
const eigen = readFixture<MathEigenFixture>("math-eigen.json");

// biome-ignore-start lint/style/useNamingConvention: the keys are those of the fixture file.
interface MathFunctionsFixture {
  p: number[];
  normalQuantile: number[];
  x: number[];
  normalDensity: number[];
  cumulativeX: number[];
  normalDistribution: number[];
  logistic: number[];
  chiSquare: { x: number[]; df: number[]; upperTail: number[] };
  chiSquareLarge: { x: number[]; df: number[]; upperTail: number[] };
}
// biome-ignore-end lint/style/useNamingConvention: the keys are those of the fixture file.
const functions = readFixture<MathFunctionsFixture>("math-functions.json");

interface MathGaussHermiteFixture {
  rules: { count: number; nodes: number[]; weights: number[] }[];
}
const hermite = readFixture<MathGaussHermiteFixture>("math-gauss-hermite.json");

interface MathNoncentralChiSquareFixture {
  x: number[];
  df: number[];
  noncentrality: number[];
  cdf: number[];
}
const noncentral = readFixture<MathNoncentralChiSquareFixture>("math-noncentral-chi-square.json");

/** Relative tolerance of the normal and logistic functions against R. */
const FUNCTION_TOLERANCE = 1e-14;
/** Relative tolerance of the chi-square upper tail against R's pchisq. */
const CHI_SQUARE_TOLERANCE = 1e-12;
/** The same with up to 10⁷ degrees of freedom, where the scale ln Γ(df / 2) has about 8 digits left. */
const LARGE_CHI_SQUARE_TOLERANCE = 1e-7;
/** Absolute tolerance of the noncentral chi-square distribution against R's pchisq. */
const NONCENTRAL_TOLERANCE = 1e-9;
/** Absolute tolerance of Gauss-Hermite nodes and weights against R's eigenvalue method. */
const QUADRATURE_TOLERANCE = 1e-12;
/**
 * Absolute tolerance of eigenvalues and, up to sign, eigenvectors against R's eigen, and of A v - λ v relative to the
 * largest absolute eigenvalue.
 */
const EIGEN_TOLERANCE = 1e-12;
/** The most items of a matrix of every pair of items, `MAX_MATRIX_ITEMS` of `data/`. */
const MATRIX_ITEMS = 1000;

function expectRelative(actual: number, expected: number, tolerance: number, label: string): void {
  expect(Math.abs(actual - expected), label).toBeLessThanOrEqual(tolerance * Math.abs(expected));
}

describe("chiSquareUpperTail", () => {
  it("matches R's pchisq(lower.tail = FALSE)", () => {
    const { x, df, upperTail } = functions.chiSquare;
    x.forEach((value, i) => {
      const expected = defined(upperTail[i]);
      const label = `pchisq(${value}, ${df[i]})`;
      expectRelative(chiSquareUpperTail(value, defined(df[i])), expected, CHI_SQUARE_TOLERANCE, label);
    });
  });

  it("matches it near the center with many degrees of freedom", () => {
    const { x, df, upperTail } = functions.chiSquareLarge;
    x.forEach((value, i) => {
      const expected = defined(upperTail[i]);
      const label = `pchisq(${value}, ${df[i]})`;
      expectRelative(chiSquareUpperTail(value, defined(df[i])), expected, LARGE_CHI_SQUARE_TOLERANCE, label);
    });
  });
});

describe("limitedInformationStatistic", () => {
  // Ξ = A Aᵀ + I of a seeded A, and Δ of three columns, the third a combination of the first two.
  const rows = 6;
  const random = createRandom(31);
  const a = Float64Array.from({ length: rows * rows }, () => random.uniform() - 0.5);
  const covariance = new Float64Array(rows * rows);
  multiply(a, transpose(a, rows, rows), rows, rows, rows, covariance);
  for (let i = 0; i < rows; i++) covariance[i * rows + i] = (covariance[i * rows + i] ?? 0) + 1;
  const residuals = Float64Array.from({ length: rows }, () => random.uniform() - 0.5);
  const independent = Float64Array.from({ length: rows * 2 }, () => random.uniform() - 0.5);
  const dependent = Float64Array.from({ length: rows * 3 }, (_, index) => {
    const [r, c] = [Math.floor(index / 3), index % 3];
    const first = independent[r * 2] ?? 0;
    const second = independent[r * 2 + 1] ?? 0;
    return c === 0 ? first : c === 1 ? second : 2 * first - 0.5 * second;
  });

  it("is N e′ (Ξ⁻¹ - Ξ⁻¹ Δ (Δ′ Ξ⁻¹ Δ)⁻¹ Δ′ Ξ⁻¹) e for a Δ of full rank", () => {
    const inverse = defined(invertPositiveDefinite(covariance, rows));
    const times = (m: Float64Array, v: Float64Array, columns: number) =>
      Float64Array.from({ length: m.length / columns }, (_, r) =>
        Array.from({ length: columns }, (__, c) => (m[r * columns + c] ?? 0) * (v[c] ?? 0)).reduce((x, y) => x + y, 0),
      );
    const solved = times(inverse, residuals, rows);
    const directions = new Float64Array(rows * 2);
    multiply(inverse, independent, rows, rows, 2, directions);
    const gram = new Float64Array(4);
    multiply(transpose(independent, rows, 2), directions, 2, rows, 2, gram);
    const projected = times(transpose(independent, rows, 2), solved, rows);
    const inner = times(defined(invert(gram, 2)), projected, 2);
    const expected =
      50 *
      (residuals.reduce((sum, e, i) => sum + e * (solved[i] ?? 0), 0) -
        (projected[0] ?? 0) * (inner[0] ?? 0) -
        (projected[1] ?? 0) * (inner[1] ?? 0));
    const { statistic, rank } = limitedInformationStatistic(residuals, covariance, independent, 2, 50);
    expectClose(statistic, expected, 1e-10);
    expect(rank).toBe(2);
  });

  it("leaves out a column that depends on the others, and counts it out of the rank", () => {
    const full = limitedInformationStatistic(residuals, covariance, independent, 2, 50);
    const deficient = limitedInformationStatistic(residuals, covariance, dependent, 3, 50);
    expectClose(deficient.statistic, full.statistic, 1e-10);
    expect(deficient.rank).toBe(2);
  });

  it("is null when Ξ is not positive definite", () => {
    expect(
      limitedInformationStatistic(residuals, new Float64Array(rows * rows), independent, 2, 50).statistic,
    ).toBeNull();
  });
});

describe("noncentralChiSquareCdf", () => {
  it("matches R's pchisq with a noncentrality", () => {
    const { x, df, noncentrality, cdf } = noncentral;
    x.forEach((value, i) => {
      const expected = defined(cdf[i]);
      const actual = noncentralChiSquareCdf(value, defined(df[i]), defined(noncentrality[i]));
      const label = `pchisq(${value}, ${df[i]}, ncp = ${noncentrality[i]})`;
      expectClose(actual, expected, NONCENTRAL_TOLERANCE, label);
    });
  });

  it("is 0 at 0 and below, and falls as the noncentrality grows", () => {
    expect(noncentralChiSquareCdf(0, 3, 2)).toBe(0);
    expect(noncentralChiSquareCdf(-1, 3, 2)).toBe(0);
    fc.assert(
      fc.property(
        fc.double({ min: 0.1, max: 500, noNaN: true }),
        fc.integer({ min: 1, max: 100 }),
        fc.double({ min: 0, max: 300, noNaN: true }),
        (x, df, lambda) => {
          const lower = noncentralChiSquareCdf(x, df, lambda);
          expect(lower).toBeGreaterThanOrEqual(0);
          expect(lower).toBeLessThanOrEqual(1);
          expect(noncentralChiSquareCdf(x, df, lambda + 1)).toBeLessThanOrEqual(lower + 1e-12);
        },
      ),
    );
  });
});

describe("symmetricEigen", () => {
  for (const { name, size, matrix, values, vectors } of eigen.cases) {
    it(`matches R's eigen for ${name}`, () => {
      const a = Float64Array.from(matrix.flat());
      const result = symmetricEigen(a, size);
      expect(result.converged).toBe(true);
      values.forEach((expected, k) => {
        expectClose(defined(result.values[k]), expected, EIGEN_TOLERANCE, `value ${k}`);
      });
      // R's sign is LAPACK's: compare with the sign that makes it agree. A vector whose eigenvalue is close to another's
      // is not unique, and is held to A v = λ v alone.
      vectors.forEach((expected, k) => {
        if (expected === null) return;
        const actual = result.vectors.subarray(k * size, (k + 1) * size);
        const sign = expected.reduce((sum, value, i) => sum + value * defined(actual[i]), 0) < 0 ? -1 : 1;
        expected.forEach((value, i) => {
          expectClose(sign * defined(actual[i]), value, EIGEN_TOLERANCE, `vector ${k}, entry ${i}`);
        });
      });
      expectEigenpairs(a, size, result);
    });
  }

  it("returns a diagonal matrix's entries in decreasing order with the unit vectors that belong to them", () => {
    const result = symmetricEigen(Float64Array.of(1, 0, 0, 0, 3, 0, 0, 0, 2), 3);
    expect([...result.values]).toEqual([3, 2, 1]);
    expect([...result.vectors]).toEqual([0, 1, 0, 0, 0, 1, 1, 0, 0]);
    expect(result).toMatchObject({ converged: true, iterations: 0 });
  });

  it("signs each eigenvector so that its entries sum to at least 0", () => {
    const { vectors } = symmetricEigen(Float64Array.of(2, -1, 0, -1, 2, -1, 0, -1, 2), 3);
    for (let k = 0; k < 3; k++) {
      const row = vectors.subarray(k * 3, (k + 1) * 3);
      expect(row.reduce((sum, entry) => sum + entry, 0)).toBeGreaterThanOrEqual(0);
    }
    // The eigenvector of 2 - √2 has entries of one sign, and the method alone may give them all negative.
    expect(defined(vectors[6])).toBeGreaterThan(0);
  });

  it("gives zeros and the identity for the zero matrix", () => {
    const result = symmetricEigen(new Float64Array(9), 3);
    expect([...result.values]).toEqual([0, 0, 0]);
    expect([...result.vectors]).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    expect(result).toMatchObject({ converged: true, iterations: 0 });
  });

  it("reads the lower triangle alone", () => {
    const lower = Float64Array.of(4, 0, 0, 1, 3, 0, 2, 0, 5);
    const full = Float64Array.of(4, 1, 2, 1, 3, 0, 2, 0, 5);
    expect(symmetricEigen(lower, 3)).toEqual(symmetricEigen(full, 3));
  });

  it("stays defined for entries of very different magnitudes", () => {
    for (const scale of [1e-150, 1e150]) {
      const result = symmetricEigen(Float64Array.of(2 * scale, scale, scale, 3 * scale), 2);
      expectClose(defined(result.values[0]) / scale, (5 + Math.sqrt(5)) / 2, 1e-12);
      expectClose(defined(result.values[1]) / scale, (5 - Math.sqrt(5)) / 2, 1e-12);
    }
    const stretched = symmetricEigen(Float64Array.of(1e-300, 1e-310, 1e-310, 1), 2);
    expect([...stretched.values]).toEqual([1, 1e-300]);
    // The reflections scale each row, so that its sum of squares neither underflows nor overflows.
    const matrix = Float64Array.of(4, 1, 2, 0.5, 1, 3, 0, 1, 2, 0, 5, 2, 0.5, 1, 2, 6);
    const { values } = symmetricEigen(matrix, 4);
    for (const scale of [1e-160, 1e160]) {
      const scaled = symmetricEigen(
        matrix.map((entry) => entry * scale),
        4,
      );
      expect(scaled.converged).toBe(true);
      values.forEach((value, k) => {
        expectClose(defined(scaled.values[k]) / scale, value, 1e-12 * defined(values[0]), `value ${k}`);
      });
    }
  });

  it("decomposes no matrix with an entry that is not finite, at once, and leaves it unchanged", () => {
    for (const entry of [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]) {
      const matrix = Float64Array.of(1, 0.5, 0, 0.5, 1, entry, 0, entry, 1);
      const result = symmetricEigen(matrix, 3);
      expect(result).toMatchObject({ converged: false, iterations: 0 });
      expect([...result.values, ...result.vectors].every(Number.isNaN)).toBe(true);
      expect(matrix[5]).toBe(entry);
    }
  });

  it("stops at the limit of QL iterations and reports it when the arithmetic of finite entries overflows", () => {
    const result = symmetricEigen(new Float64Array(9).fill(Number.MAX_VALUE), 3);
    expect(result).toMatchObject({ converged: false, iterations: 30 });
  });

  it("decomposes the correlations of 1,000 items, the most a matrix of items takes", () => {
    // Five factors of seeded loadings: five leading eigenvalues and the rest spread between the uniquenesses.
    const random = createRandom(29);
    const loadings = Float64Array.from({ length: MATRIX_ITEMS * 5 }, () => 0.4 * random.uniform());
    const r = new Float64Array(MATRIX_ITEMS * MATRIX_ITEMS);
    for (let i = 0; i < MATRIX_ITEMS; i++) {
      for (let j = 0; j < MATRIX_ITEMS; j++) {
        let common = 0;
        for (let k = 0; k < 5; k++) common += defined(loadings[i * 5 + k]) * defined(loadings[j * 5 + k]);
        r[i * MATRIX_ITEMS + j] = i === j ? 1 : common;
      }
    }
    const result = symmetricEigen(r, MATRIX_ITEMS);
    expect(result.converged).toBe(true);
    // About two QL iterations for each eigenvalue.
    expect(result.iterations).toBeLessThan(3 * MATRIX_ITEMS);
    expect(result.values.reduce((sum, value) => sum + value, 0)).toBeCloseTo(MATRIX_ITEMS, 9);
    // A v = λ v and orthonormality for every 37th eigenvector, the first and the last.
    const checked = Array.from({ length: MATRIX_ITEMS }, (_, k) => k).filter(
      (k) => k % 37 === 0 || k === MATRIX_ITEMS - 1,
    );
    expectEigenpairs(r, MATRIX_ITEMS, result, checked);
  });

  it("satisfies A v = λ v for orthonormal v and keeps the trace, for any symmetric matrix", () => {
    const symmetric = fc
      .integer({ min: 1, max: 12 })
      .chain((size) =>
        fc.array(fc.double({ min: -10, max: 10, noNaN: true }), { minLength: size * size, maxLength: size * size }),
      );
    fc.assert(
      fc.property(symmetric, (values) => {
        const size = Math.sqrt(values.length);
        const a = symmetrizedOf(values, size);
        const result = symmetricEigen(a, size);
        expect(result.converged).toBe(true);
        expectEigenpairs(a, size, result);
        let trace = 0;
        for (let i = 0; i < size; i++) trace += defined(a[i * size + i]);
        expectClose(
          result.values.reduce((sum, value) => sum + value, 0),
          trace,
          1e-12 * size * 10,
        );
      }),
    );
  });

  it("satisfies A v = λ v for matrices of repeated eigenvalues, zero rows and low rank", () => {
    // Entries of -1, 0 and 1 give zero rows, blocks that split and repeated eigenvalues; B Bᵀ of B with fewer columns
    // than rows has 0 as an eigenvalue of every multiplicity.
    const sparse = fc
      .integer({ min: 2, max: 12 })
      .chain((size) => fc.array(fc.constantFrom(-1, 0, 0, 1), { minLength: size * size, maxLength: size * size }))
      .map((values) => symmetrizedOf(values, Math.sqrt(values.length)));
    const lowRank = fc.tuple(fc.integer({ min: 2, max: 12 }), fc.integer({ min: 1, max: 3 })).chain(([size, rank]) =>
      fc
        .array(fc.double({ min: -2, max: 2, noNaN: true }), { minLength: size * rank, maxLength: size * rank })
        .map((entries) => {
          const a = new Float64Array(size * size);
          for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
              for (let k = 0; k < rank; k++) {
                a[i * size + j] =
                  defined(a[i * size + j]) + defined(entries[i * rank + k]) * defined(entries[j * rank + k]);
              }
            }
          }
          return a;
        }),
    );
    fc.assert(
      fc.property(fc.oneof(sparse, lowRank), (a) => {
        const size = Math.sqrt(a.length);
        const result = symmetricEigen(a, size);
        expect(result.converged).toBe(true);
        expectEigenpairs(a, size, result);
      }),
    );
  });
});

/** Returns the symmetric matrix (B + Bᵀ) / 2 of the `size` by `size` matrix B of `values`, by rows. */
function symmetrizedOf(values: readonly number[], size: number): Float64Array {
  const a = new Float64Array(size * size);
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++)
      a[i * size + j] = (defined(values[i * size + j]) + defined(values[j * size + i])) / 2;
  }
  return a;
}

/**
 * Expects the eigenvalues to decrease, and each eigenvector of `rows`, or of every row, to be a unit vector orthogonal to
 * the others of `rows` with A v = λ v, to `EIGEN_TOLERANCE` times the largest absolute eigenvalue, or 1.
 */
function expectEigenpairs(
  a: Float64Array,
  size: number,
  { values, vectors }: { values: Float64Array; vectors: Float64Array },
  rows: readonly number[] = Array.from({ length: size }, (_, k) => k),
): void {
  const scale = Math.max(1, Math.abs(defined(values[0])), Math.abs(defined(values[size - 1])));
  for (let k = 1; k < size; k++) expect(defined(values[k])).toBeLessThanOrEqual(defined(values[k - 1]));
  for (const k of rows) {
    for (let i = 0; i < size; i++) {
      let product = 0;
      for (let j = 0; j < size; j++) product += defined(a[i * size + j]) * defined(vectors[k * size + j]);
      const residual = product - defined(values[k]) * defined(vectors[k * size + i]);
      expect(Math.abs(residual), `A v - λ v of vector ${k}, entry ${i}`).toBeLessThanOrEqual(EIGEN_TOLERANCE * scale);
    }
    for (const l of rows) {
      if (l > k) continue;
      let product = 0;
      for (let i = 0; i < size; i++) product += defined(vectors[k * size + i]) * defined(vectors[l * size + i]);
      expectClose(product, k === l ? 1 : 0, EIGEN_TOLERANCE, `vectors ${k} and ${l}`);
    }
  }
}

describe("numericalHessian", () => {
  it("differentiates a gradient to the Hessian", () => {
    const hessian = numericalHessian((x, gradient) => rosenbrock(x, gradient), Float64Array.of(-1.2, 1));
    const exact = new Float64Array(4);
    rosenbrock(Float64Array.of(-1.2, 1), new Float64Array(2), exact);
    exact.forEach((value, i) => {
      expectClose(defined(hessian[i]), value, 1e-6 * Math.abs(value));
    });
    expect(hessian[1]).toBe(hessian[2]);
  });
});

describe("invert", () => {
  it("inverts a matrix", () => {
    const inverse = defined(invert(Float64Array.of(4, 7, 2, 6), 2));
    expect([...inverse].map((value) => Number(value.toFixed(12)))).toEqual([0.6, -0.7, -0.2, 0.4]);
  });

  it("returns the identity product for random well-conditioned matrices", () => {
    const matrix = fc
      .integer({ min: 1, max: 5 })
      .chain((size) =>
        fc.array(fc.double({ min: -1, max: 1, noNaN: true }), { minLength: size * size, maxLength: size * size }),
      );
    fc.assert(
      fc.property(matrix, (values) => {
        const size = Math.sqrt(values.length);
        // Diagonal dominance keeps the matrix well conditioned.
        const a = Float64Array.from(values, (value, i) => (i % (size + 1) === 0 ? value + size + 1 : value));
        const inverse = defined(invert(a, size));
        for (let i = 0; i < size; i++) {
          for (let j = 0; j < size; j++) {
            let sum = 0;
            for (let k = 0; k < size; k++) sum += defined(a[i * size + k]) * defined(inverse[k * size + j]);
            expect(Math.abs(sum - (i === j ? 1 : 0))).toBeLessThan(1e-12);
          }
        }
      }),
    );
  });

  it("returns null for a singular matrix", () => {
    expect(invert(Float64Array.of(1, 2, 2, 4), 2)).toBeNull();
    expect(invert(Float64Array.of(0), 1)).toBeNull();
  });
});

describe("invertPositiveDefinite", () => {
  it("gives the inverse of Gauss-Jordan elimination for symmetric positive definite matrices", () => {
    const factors = fc
      .integer({ min: 1, max: 6 })
      .chain((size) =>
        fc.array(fc.double({ min: -1, max: 1, noNaN: true }), { minLength: size * size, maxLength: size * size }),
      );
    fc.assert(
      fc.property(factors, (values) => {
        const size = Math.sqrt(values.length);
        // B Bᵀ plus the identity is symmetric positive definite.
        const a = new Float64Array(size * size);
        for (let i = 0; i < size; i++) {
          for (let j = 0; j < size; j++) {
            let sum = i === j ? 1 : 0;
            for (let k = 0; k < size; k++) sum += defined(values[i * size + k]) * defined(values[j * size + k]);
            a[i * size + j] = sum;
          }
        }
        const expected = defined(invert(a, size));
        defined(invertPositiveDefinite(a, size)).forEach((value, index) => {
          expectClose(value, defined(expected[index]), 1e-12 * Math.max(1, Math.abs(defined(expected[index]))));
        });
      }),
    );
  });
});

describe("pseudoInverse", () => {
  it("is the inverse of a positive definite matrix, derived by hand for a 2 by 2 matrix", () => {
    // [[2, 1], [1, 2]]⁻¹ = [[2, -1], [-1, 2]] / 3
    const inverse = pseudoInverse(Float64Array.from([2, 1, 1, 2]), 2);
    [2 / 3, -1 / 3, -1 / 3, 2 / 3].forEach((value, i) => {
      expectClose(defined(inverse[i]), value, 1e-14, `entry ${i}`);
    });
  });

  it("gives the Moore-Penrose inverse of a matrix of deficient rank", () => {
    // The projection onto (1, 1) / √2 scaled by 4 has the eigenvalues 4 and 0: its pseudo-inverse is a quarter of it.
    const inverse = pseudoInverse(Float64Array.from([2, 2, 2, 2]), 2);
    [0.125, 0.125, 0.125, 0.125].forEach((value, i) => {
      expectClose(defined(inverse[i]), value, 1e-14, `entry ${i}`);
    });
    expect(Array.from(pseudoInverse(new Float64Array(4), 2))).toEqual([0, 0, 0, 0]);
  });

  it("drops the eigenvalues below the square root of the machine epsilon times the largest, and keeps those above", () => {
    const dropped = pseudoInverse(Float64Array.from([1, 0, 0, 1e-9]), 2);
    expect(Array.from(dropped)).toEqual([1, 0, 0, 0]);
    const kept = pseudoInverse(Float64Array.from([1, 0, 0, 1e-6]), 2);
    expectClose(defined(kept[3]), 1e6, 1e-3, "kept");
  });

  it("treats an eigenvalue whose reciprocal overflows as 0", () => {
    expect(Array.from(pseudoInverse(Float64Array.from([5e-324]), 1))).toEqual([0]);
  });

  it("satisfies the four Moore-Penrose conditions for any semidefinite matrix", () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 1, max: 5 }),
        fc.integer({ min: 1, max: 5 }),
        fc.array(
          fc.double({ min: -2, max: 2, noNaN: true }).map((x) => (Math.abs(x) < 0.1 ? 0 : x)),
          { minLength: 25, maxLength: 25 },
        ),
        (size, rank, entries) => {
          // B' B for a rank by size matrix B is semidefinite of rank at most `rank`.
          const factor = entries.slice(0, rank * size);
          const a = new Float64Array(size * size);
          for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
              for (let k = 0; k < rank; k++) {
                a[i * size + j] = (a[i * size + j] ?? 0) + (factor[k * size + i] ?? 0) * (factor[k * size + j] ?? 0);
              }
            }
          }
          const g = pseudoInverse(a, size);
          const product = (x: Float64Array, y: Float64Array): Float64Array => {
            const out = new Float64Array(size * size);
            multiply(x, y, size, size, size, out);
            return out;
          };
          // Rounding is relative to the largest entry that a condition involves, which a nearly singular matrix's inverse enlarges.
          const close = (x: Float64Array, y: Float64Array): void => {
            const scale = Math.max(1, ...x.map(Math.abs), ...y.map(Math.abs));
            for (let i = 0; i < x.length; i++)
              expect(Math.abs(defined(x[i]) - defined(y[i]))).toBeLessThan(1e-6 * scale);
          };
          close(product(product(a, g), a), a);
          close(product(product(g, a), g), g);
          close(product(a, g), transpose(product(a, g), size, size));
          close(product(g, a), transpose(product(g, a), size, size));
        },
      ),
    );
  });
});

describe("matrix helpers", () => {
  const a = Float64Array.of(1, 2, 3, 4, 5, 6);
  const b = Float64Array.of(7, 8, 9, 10, 11, 12);

  it("multiplies a 2 by 3 matrix and a 3 by 2 matrix by rows", () => {
    const out = new Float64Array(4);
    multiply(a, b, 2, 3, 2, out);
    expect(Array.from(out)).toEqual([58, 64, 139, 154]);
  });

  it("transposes and symmetrizes", () => {
    expect(Array.from(transpose(a, 2, 3))).toEqual([1, 4, 2, 5, 3, 6]);
    expect(Array.from(symmetrized(Float64Array.of(1, 2, 4, 3), 2))).toEqual([1, 3, 3, 3]);
  });

  it("returns the identity, which multiplies to the same matrix", () => {
    expect(Array.from(identity(3))).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    const out = new Float64Array(6);
    multiply(a, identity(3), 2, 3, 3, out);
    expect(Array.from(out)).toEqual(Array.from(a));
  });

  it("sums the diagonal in trace", () => {
    expect(trace(Float64Array.of(1, 2, 3, 4, 5, 6, 7, 8, 9), 3)).toBe(15);
  });

  it("copies the upper triangle onto the lower entries that were not written, and no other", () => {
    const matrix = Float64Array.of(1, 2, 3, 0, 4, 5, 9, 0, 6);
    mirror(matrix, 3);
    expect(Array.from(matrix)).toEqual([1, 2, 3, 2, 4, 5, 9, 5, 6]);
  });

  it("takes tr(a b) as the trace of the product", () => {
    const first = Float64Array.of(1, 2, 3, 4);
    const second = Float64Array.of(5, 6, 7, 8);
    const product = new Float64Array(4);
    multiply(first, second, 2, 2, 2, product);
    expect(traceOfProduct(first, second, 2)).toBe((product[0] ?? 0) + (product[3] ?? 0));
    expect(traceOfProduct(first, second, 2)).toBe(69);
  });
});

describe("normal and logistic functions", () => {
  it("match R's qnorm, dnorm and plogis", () => {
    functions.p.forEach((p, i) => {
      expectRelative(normalQuantile(p), defined(functions.normalQuantile[i]), FUNCTION_TOLERANCE, `qnorm(${p})`);
    });
    functions.x.forEach((x, i) => {
      expectRelative(normalDensity(x), defined(functions.normalDensity[i]), FUNCTION_TOLERANCE, `dnorm(${x})`);
      expectRelative(logistic(x), defined(functions.logistic[i]), FUNCTION_TOLERANCE, `plogis(${x})`);
    });
  });

  it("matches R's pnorm in both tails and at the boundaries of Cody's ranges", () => {
    functions.cumulativeX.forEach((x, i) => {
      const expected = defined(functions.normalDistribution[i]);
      expectClose(normalCdf(x), expected, FUNCTION_TOLERANCE * Math.max(expected, Number.MIN_VALUE), `pnorm(${x})`);
    });
  });
});

/** The Rosenbrock function, minimal at (1, 1), with its gradient and Hessian. */
function rosenbrock(x: Float64Array, gradient: Float64Array, hessian?: Float64Array): number {
  const a = defined(x[0]);
  const b = defined(x[1]);
  gradient[0] = -2 * (1 - a) - 400 * a * (b - a * a);
  gradient[1] = 200 * (b - a * a);
  hessian?.set([2 - 400 * (b - 3 * a * a), -400 * a, -400 * a, 200]);
  return (1 - a) ** 2 + 100 * (b - a * a) ** 2;
}

describe("newton", () => {
  it("minimizes the Rosenbrock function", () => {
    const result = newton(rosenbrock, [-1.2, 1]);
    expect(result.converged).toBe(true);
    expect(defined(result.x[0])).toBeCloseTo(1, 7);
    expect(defined(result.x[1])).toBeCloseTo(1, 7);
    expect(result.value).toBeLessThan(1e-14);
  });

  it("solves a quadratic in one step", () => {
    const result = newton(
      (x, gradient, hessian) => {
        const v = defined(x[0]);
        gradient[0] = 2 * (v - 3);
        hessian[0] = 2;
        return (v - 3) ** 2;
      },
      [10],
    );
    expect(result).toMatchObject({ iterations: 1, converged: true, value: 0 });
    expect(result.x[0]).toBe(3);
  });

  it("follows the gradient where the Hessian is not positive definite", () => {
    // f(x) = x^4 - x^2 has a negative second derivative at 0.1 and minima at ±1/√2.
    const result = newton(
      (x, gradient, hessian) => {
        const v = defined(x[0]);
        gradient[0] = 4 * v ** 3 - 2 * v;
        hessian[0] = 12 * v * v - 2;
        return v ** 4 - v * v;
      },
      [0.1],
    );
    expect(result.converged).toBe(true);
    expect(defined(result.x[0])).toBeCloseTo(Math.SQRT1_2, 8);
  });

  it("takes in full a step whose decrease rounding hides, though the step is above the step tolerance", () => {
    // f = 1000 + 0.005 x² from x = 2e-6: the Newton step of 2e-6 lowers f by 2e-14, and a rounding error of one unit
    // at every other point, as the long sums of a likelihood make, hides that decrease from the line search.
    const start = 2e-6;
    const result = newton(
      (x, gradient, hessian) => {
        const v = defined(x[0]);
        gradient[0] = 0.01 * v;
        hessian[0] = 0.01;
        return 1000 + 0.005 * v * v + (v === start ? 0 : 1000 * Number.EPSILON);
      },
      [start],
      { tolerance: 0, stepTolerance: 1e-6, maxIterations: 50 },
    );
    expect(result).toMatchObject({ iterations: 1, converged: true });
    expect(result.x[0]).toBe(0);
  });

  it("reports non-convergence within the iteration limit", () => {
    const result = newton(rosenbrock, [-1.2, 1], { maxIterations: 2 });
    expect(result).toMatchObject({ iterations: 2, converged: false });
  });
});

describe("gaussHermite", () => {
  it("matches the Golub-Welsch rules computed in R", () => {
    for (const rule of hermite.rules) {
      const { nodes, weights } = gaussHermite(rule.count);
      expect(nodes).toHaveLength(rule.count);
      rule.nodes.forEach((node, i) => {
        expectClose(defined(nodes[i]), node, QUADRATURE_TOLERANCE, `node ${i} of ${rule.count}`);
      });
      rule.weights.forEach((weight, i) => {
        expectClose(defined(weights[i]), weight, QUADRATURE_TOLERANCE, `weight ${i} of ${rule.count}`);
      });
    }
  });

  it("integrates normal moments exactly up to degree 2n - 1", () => {
    for (const count of [1, 2, 3, 10, 40]) {
      const { nodes, weights } = gaussHermite(count);
      let doubleFactorial = 1;
      for (let degree = 0; degree <= 2 * count - 1; degree++) {
        const terms = Array.from(nodes, (node, i) => defined(weights[i]) * node ** degree);
        const moment = terms.reduce((sum, term) => sum + term, 0);
        const scale = terms.reduce((sum, term) => sum + Math.abs(term), 0);
        const expected = degree % 2 === 1 ? 0 : doubleFactorial;
        expect(Math.abs(moment - expected), `E[X^${degree}], ${count} points`).toBeLessThanOrEqual(1e-12 * scale);
        if (degree % 2 === 1) doubleFactorial *= degree;
      }
    }
  });
});

describe("rectangular", () => {
  it("spaces nodes evenly from -6 to 6 with normal density weights summing to 1, as mirt", () => {
    const { nodes, weights } = rectangular(61);
    expect(nodes[0]).toBe(-6);
    expect(nodes[60]).toBe(6);
    expectClose(defined(nodes[30]), 0, 1e-15);
    expectClose(defined(nodes[31]) - defined(nodes[30]), 0.2, 1e-14);
    expectClose(
      weights.reduce((sum, weight) => sum + weight, 0),
      1,
      1e-15,
    );
    expectClose(defined(weights[31]) / defined(weights[30]), normalDensity(0.2) / normalDensity(0), 1e-15);
  });
});

describe("adaptiveIntegral", () => {
  it("integrates a polynomial to rounding, its error taken as the difference from the 10-point Gauss rule", () => {
    const polynomial = (x: number): number => 32 * x ** 31 + 3 * x * x - 1;
    expectClose(adaptiveIntegral(polynomial, 0, 1, 1e-12), 1, 1e-14);
  });

  it("halves the intervals of a sharp peak and of a function with a kink until the relative tolerance holds", () => {
    const peak = (x: number): number => Math.exp(-((x - 0.3) ** 2) / (2 * 1e-4));
    expectClose(adaptiveIntegral(peak, 0, 1, 1e-12), Math.sqrt(2 * Math.PI) * 1e-2, 1e-14);
    expectClose(
      adaptiveIntegral((x) => Math.abs(x - 1 / 3), 0, 1, 1e-12),
      1 / 18 + 2 / 9,
      1e-12,
    );
  });

  it("stops at once for an integrand of 0, and on an interval of no width", () => {
    let calls = 0;
    expect(
      adaptiveIntegral(
        () => {
          calls++;
          return 0;
        },
        -1,
        1,
        1e-12,
      ),
    ).toBe(0);
    expect(calls).toBe(21);
    expect(adaptiveIntegral(() => 1, 2, 2, 1e-12)).toBe(0);
  });
});

describe("findRoot", () => {
  it("finds the root of a bracketing interval to the tolerance", () => {
    const f = (x: number): number => Math.exp(x) - 2;
    expectClose(findRoot(f, -1, 3, f(-1), f(3)), Math.LN2, 1e-12);
    const g = (x: number): number => Math.cos(x) - x;
    expectClose(findRoot(g, 0, 1, g(0), g(1)), 0.7390851332151607, 1e-12);
  });

  it("works with the signs in either order", () => {
    const f = (x: number): number => 1 - x ** 3;
    expectClose(findRoot(f, 0, 4, f(0), f(4)), 1, 1e-12);
    expectClose(
      findRoot((x) => -f(x), 0, 4, -f(0), -f(4)),
      1,
      1e-12,
    );
  });
});

describe("findIncreasingRoot", () => {
  it("brackets the root of an increasing function from a start on either side", () => {
    for (const start of [-50, 0, 3, 200]) expect(findIncreasingRoot((x) => Math.atan(x - 3), start)).toBeCloseTo(3, 12);
    expect(findIncreasingRoot((x) => x - 2, 2)).toBe(2);
  });

  it("returns the furthest point searched when the function keeps its sign over the finite numbers", () => {
    const point = findIncreasingRoot(() => -1, 0);
    expect(Number.isFinite(point)).toBe(true);
    expect(Number.isFinite(point * 2)).toBe(false);
  });
});

describe("rotateLoadingMatrix", () => {
  // The unrotated loadings of five factors of the bfi items: a rotation of them has structure to find.
  const unrotated = Float64Array.from(defined(factors.cases[3]).base.unrotated.flat());
  const rows = 25;
  const columns = 5;

  /** The sum and the largest loading in absolute value of a column of loadings by rows. */
  function summaries(loadings: Float64Array, column: number): { sum: number; largest: number } {
    let sum = 0;
    let largest = 0;
    for (let i = 0; i < rows; i++) {
      const loading = defined(loadings[i * columns + column]);
      sum += loading;
      if (Math.abs(loading) > Math.abs(largest)) largest = loading;
    }
    return { sum, largest };
  }

  it("gives each rotated dimension the sign of its rule, and the same loadings up to the signs", () => {
    for (const method of ["varimax", "oblimin"] as const) {
      const options = resolveRotation({ method });
      const bySum = rotateLoadingMatrix(unrotated, rows, columns, options, "sum");
      const byLargest = rotateLoadingMatrix(unrotated, rows, columns, options, "largest");
      for (let j = 0; j < columns; j++) {
        expect(summaries(bySum.loadings, j).sum, method).toBeGreaterThan(0);
        expect(summaries(byLargest.loadings, j).largest, method).toBeGreaterThan(0);
      }
      bySum.loadings.forEach((loading, k) => {
        expectClose(Math.abs(loading), Math.abs(defined(byLargest.loadings[k])), 1e-12, `${method} loading ${k}`);
      });
    }
  });

  it("gives loadings Λ (Tᵀ)⁻¹ and correlations Tᵀ T of the rotation it returns", () => {
    const rotated = rotateLoadingMatrix(unrotated, rows, columns, resolveRotation(), "sum");
    const product = new Float64Array(rows * columns);
    multiply(rotated.loadings, transpose(rotated.rotation, columns, columns), rows, columns, columns, product);
    unrotated.forEach((loading, k) => {
      expectClose(defined(product[k]), loading, 1e-10, `loading ${k}`);
    });
    const gram = new Float64Array(columns * columns);
    multiply(transpose(rotated.rotation, columns, columns), rotated.rotation, columns, columns, columns, gram);
    gram.forEach((value, k) => {
      expectClose(defined(rotated.correlations[k]), value, 1e-12, `correlation ${k}`);
    });
    expect(rotated.converged).toBe(true);
  });

  it("keeps the dimensions orthogonal for varimax, with or without Kaiser's normalization", () => {
    for (const normalize of [false, true]) {
      const { correlations } = rotateLoadingMatrix(
        unrotated,
        rows,
        columns,
        resolveRotation({ method: "varimax", normalize }),
        "sum",
      );
      correlations.forEach((value, k) => {
        expectClose(value, Math.floor(k / columns) === k % columns ? 1 : 0, 1e-10, `correlation ${k}`);
      });
    }
  });
});
