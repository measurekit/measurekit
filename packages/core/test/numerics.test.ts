import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  ARMIJO,
  backtrack,
  cholesky,
  type DescentPoint,
  dot,
  invertPositiveDefinite,
  iterateToConvergence,
  largestChange,
  logistic,
  logLogistic,
  maxAbs,
  mean,
  multiplyVector,
  normalCdf,
  normalDensity,
  normalizeLogs,
  normalQuantile,
  stepAlong,
  type ValueAndGradient,
} from "../src/index.js";

/** Relative tolerance of the sums that the normalization of logs rounds. */
const NORMALIZATION_TOLERANCE = 1e-12;

/** Absolute tolerance of L Lᵀ against the correlation matrix that was factored, whose entries are at most 1. */
const FACTORIZATION_TOLERANCE = 1e-14;

/** Returns the correlation matrix that decay by 0.9 with each step between two variables, 0.9 ** |i - j|, which is positive definite. */
function decayingCorrelations(size: number): Float64Array {
  const matrix = new Float64Array(size * size);
  for (let i = 0; i < size; i++) for (let j = 0; j < size; j++) matrix[i * size + j] = 0.9 ** Math.abs(i - j);
  return matrix;
}

/** Returns the largest absolute difference between L Lᵀ and the matrix. */
function largestError(lower: Float64Array, matrix: Float64Array, size: number): number {
  let largest = 0;
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      let sum = 0;
      for (let k = 0; k < size; k++) sum += (lower[i * size + k] ?? 0) * (lower[j * size + k] ?? 0);
      largest = Math.max(largest, Math.abs(sum - (matrix[i * size + j] ?? 0)));
    }
  }
  return largest;
}

describe("normalCdf and normalDensity", () => {
  // R's pnorm at each of Cody's ranges, and beyond the last, where the lower tail underflows to 0.
  const pnorm: readonly (readonly [number, number])[] = [
    [Number.NEGATIVE_INFINITY, 0],
    [-40, 0],
    [-37.5, 4.605353009581955e-308],
    [-8, 6.220960574271785e-16],
    [-5.6, 1.071759025831093e-8],
    [-0.7, 0.24196365222307298],
    [-1e-20, 0.5],
    [0.2, 0.579259709439103],
    [0.6744, 0.7499714786270594],
    [2, 0.9772498680518208],
    [5.7, 0.9999999940096286],
    [9, 1],
    [Number.POSITIVE_INFINITY, 1],
  ];

  it("matches R's pnorm in both tails and at the boundaries of Cody's ranges", () => {
    for (const [x, expected] of pnorm) {
      expect(Math.abs(normalCdf(x) - expected), `pnorm(${x})`).toBeLessThanOrEqual(1e-14 * expected);
    }
  });

  it("gives the two tails the same mass", () => {
    fc.assert(
      fc.property(fc.double({ min: -8, max: 8, noNaN: true }), (x) => {
        expect(Math.abs(normalCdf(x) + normalCdf(-x) - 1)).toBeLessThan(1e-15);
      }),
    );
  });

  it("matches R's dnorm", () => {
    expect(normalDensity(0)).toBe(0.3989422804014327);
    expect(Math.abs(normalDensity(1) - 0.24197072451914337)).toBeLessThan(1e-16);
    expect(normalDensity(-1)).toBe(normalDensity(1));
  });
});

describe("invertPositiveDefinite", () => {
  it("gives the matrix whose product with a symmetric positive definite matrix is the identity", () => {
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
            for (let k = 0; k < size; k++) sum += (values[i * size + k] ?? 0) * (values[j * size + k] ?? 0);
            a[i * size + j] = sum;
          }
        }
        const inverse = invertPositiveDefinite(a, size) ?? new Float64Array(size * size);
        for (let i = 0; i < size; i++) {
          for (let j = 0; j < size; j++) {
            let product = 0;
            for (let k = 0; k < size; k++) product += (a[i * size + k] ?? 0) * (inverse[k * size + j] ?? 0);
            expect(Math.abs(product - (i === j ? 1 : 0))).toBeLessThan(1e-10);
            expect(inverse[i * size + j]).toBe(inverse[j * size + i]);
          }
        }
      }),
    );
  });

  it("returns null for a matrix that is not positive definite", () => {
    expect(invertPositiveDefinite(Float64Array.of(1, 2, 2, 1), 2)).toBeNull();
    expect(invertPositiveDefinite(Float64Array.of(1, 1, 1, 1), 2)).toBeNull();
  });
});

describe("normalQuantile", () => {
  it("returns infinite quantiles at 0 and 1 and NaN outside [0, 1]", () => {
    expect(normalQuantile(0)).toBe(Number.NEGATIVE_INFINITY);
    expect(normalQuantile(1)).toBe(Number.POSITIVE_INFINITY);
    expect(normalQuantile(-0.1)).toBeNaN();
    expect(normalQuantile(Number.NaN)).toBeNaN();
  });

  it("is 0 at one half and odd around it", () => {
    expect(normalQuantile(0.5)).toBe(0);
    for (const p of [0.001, 0.025, 0.1, 0.3]) expect(normalQuantile(1 - p)).toBeCloseTo(-normalQuantile(p), 12);
  });
});

describe("logistic", () => {
  it("is the probability of a logit", () => {
    expect(logistic(0)).toBe(0.5);
    expect(logistic(Math.log(3))).toBeCloseTo(0.75, 15);
    expect(logistic(-Infinity)).toBe(0);
    expect(logistic(Infinity)).toBe(1);
  });

  it("gives its logarithm finite in both tails in logLogistic", () => {
    for (const x of [-3, -0.5, 0, 0.5, 3]) expect(logLogistic(x)).toBeCloseTo(Math.log(logistic(x)), 14);
    expect(logLogistic(-1000)).toBe(-1000);
    expect(logLogistic(40)).toBe(-Math.exp(-40));
  });
});

describe("cholesky", () => {
  it("factors a positive definite matrix into L Lᵀ", () => {
    const lower = cholesky(Float64Array.of(4, 2, 2, 5), 2);
    expect(Array.from(lower ?? [])).toEqual([2, 0, 1, 2]);
  });

  it("reproduces the matrix at every order, whether or not its columns and rows fill the blocks", () => {
    for (let size = 1; size <= 16; size++) {
      const matrix = decayingCorrelations(size);
      const lower = cholesky(matrix, size);
      expect(lower, `The factor of order ${size}`).not.toBeNull();
      expect(largestError(lower ?? new Float64Array(0), matrix, size), `The error of order ${size}`).toBeLessThan(
        FACTORIZATION_TOLERANCE,
      );
    }
  });

  it("leaves the entries above the diagonal at 0", () => {
    const lower = cholesky(decayingCorrelations(10), 10) ?? new Float64Array(0);
    for (let i = 0; i < 10; i++) for (let j = i + 1; j < 10; j++) expect(lower[i * 10 + j]).toBe(0);
  });

  it("returns null for a matrix that is singular or indefinite to working precision", () => {
    expect(cholesky(Float64Array.of(1, 1, 1, 1), 2)).toBeNull();
    expect(cholesky(Float64Array.of(1, 1, 1, 1 + 1e-17), 2)).toBeNull();
    expect(cholesky(Float64Array.of(1, 2, 2, 1), 2)).toBeNull();
    expect(cholesky(Float64Array.of(0), 1)).toBeNull();
  });

  it("returns null when the pivot fails at any column of any order", () => {
    for (let size = 1; size <= 10; size++) {
      for (let broken = 0; broken < size; broken++) {
        const matrix = decayingCorrelations(size);
        matrix[broken * size + broken] = -1;
        expect(cholesky(matrix, size), `Order ${size} with column ${broken} broken`).toBeNull();
      }
    }
  });
});

describe("normalizeLogs", () => {
  it("turns log weights into probabilities and returns the log of their sum", () => {
    const values = Float64Array.of(Math.log(1), Math.log(3));
    expect(normalizeLogs(values)).toBeCloseTo(Math.log(4), 14);
    expect(values[0]).toBeCloseTo(0.25, 14);
    expect(values[1]).toBeCloseTo(0.75, 14);
  });

  it("keeps weights that exp alone would underflow", () => {
    const values = Float64Array.of(-1000, -1000, -1000, -1000);
    expect(Math.abs(normalizeLogs(values) - (-1000 + Math.log(4)))).toBeLessThan(NORMALIZATION_TOLERANCE);
    expect(values[3]).toBeCloseTo(0.25, 14);
  });

  it("returns -Infinity and leaves the values when every weight is 0", () => {
    const values = Float64Array.of(Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY);
    expect(normalizeLogs(values)).toBe(Number.NEGATIVE_INFINITY);
    expect(Array.from(values)).toEqual([Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY]);
  });
});

describe("vectors", () => {
  it("multiplies and adds in dot", () => {
    expect(dot(Float64Array.of(1, 2, 3), Float64Array.of(4, -5, 6))).toBe(12);
    expect(dot(new Float64Array(0), new Float64Array(0))).toBe(0);
  });

  it("takes the largest absolute value in maxAbs, 0 for no values", () => {
    expect(maxAbs(Float64Array.of(1, -7, 3))).toBe(7);
    expect(maxAbs(new Float64Array(0))).toBe(0);
  });

  it("averages arrays and typed arrays alike in mean", () => {
    expect(mean([1, 2, 6])).toBe(3);
    expect(mean(Float64Array.of(0.5))).toBe(0.5);
  });

  it("multiplies a square matrix, by rows, and a vector in multiplyVector", () => {
    const out = new Float64Array(2);
    multiplyVector(Float64Array.of(1, 2, 3, 4), Float64Array.of(5, -1), out);
    expect(Array.from(out)).toEqual([3, 11]);
  });

  it("takes the largest absolute change in largestChange, 0 for no values", () => {
    expect(largestChange(Float64Array.of(1, 5, -2), Float64Array.of(1.5, 2, -2))).toBe(3);
    expect(largestChange(Float64Array.of(0, 0), Float64Array.of(0, -4))).toBe(4);
    expect(largestChange(new Float64Array(0), new Float64Array(0))).toBe(0);
  });
});

describe("iterateToConvergence", () => {
  it("stops at the first cycle whose change is at most the tolerance", () => {
    const changes = [4, 2, 1, 0.5, 0.25];
    let cycles = 0;
    const outcome = iterateToConvergence(() => changes[cycles++] ?? Number.NaN, { tolerance: 1, maxIterations: 10 });
    expect(outcome).toEqual({ converged: true, iterations: 3 });
    expect(cycles).toBe(3);
  });

  it("runs the most cycles and reports no convergence when the change stays above the tolerance", () => {
    let cycles = 0;
    const outcome = iterateToConvergence(
      () => {
        cycles++;
        return 2;
      },
      { tolerance: 1, maxIterations: 4 },
    );
    expect(outcome).toEqual({ converged: false, iterations: 4 });
    expect(cycles).toBe(4);
  });

  it("converges on the last cycle that is allowed", () => {
    expect(iterateToConvergence(() => 0, { tolerance: 0, maxIterations: 1 })).toEqual({
      converged: true,
      iterations: 1,
    });
  });

  it("never converges on a change that is NaN", () => {
    expect(iterateToConvergence(() => Number.NaN, { tolerance: 1, maxIterations: 3 })).toEqual({
      converged: false,
      iterations: 3,
    });
  });
});

describe("backtrack", () => {
  /** The objective (x - 2)² of a point x = scale · length along a direction from x = 0. */
  const objective = (scale: number) => (length: number) => (scale * length - 2) ** 2;

  it("accepts the full step when it decreases the objective enough", () => {
    // From x = 0 along +1: the value is 4, the slope -4, and a length of 1 gives 1 <= 4 - 1e-4 · 4.
    expect(backtrack(objective(1), 4, -4)).toBe(1);
  });

  it("halves the step until the decrease is sufficient", () => {
    // Along +10 the objective is (10 t - 2)²: 64 at t = 1, 9 at 1/2, 0.25 at 1/4, which is below 4 - 1e-4 · (1/4) · 40.
    expect(backtrack(objective(10), 4, -40)).toBe(0.25);
  });

  it("requires a decrease of c1 times the length times the slope, and returns null when no length gives it", () => {
    expect(backtrack(() => 5, 4, -1)).toBeNull();
    // A trial of 3.6 misses the bound 4 - 0.5 · 1 · 1 = 3.5 at length 1, and meets 4 - 0.5 · (1/2) · 1 = 3.75 at 1/2.
    expect(backtrack(() => 3.6, 4, -1, { decrease: 0.5 })).toBe(0.5);
    expect(backtrack(() => 3.6, 4, -1, { decrease: 0.5, maxHalvings: 0 })).toBeNull();
  });

  it("starts at the length it is given and halves from there", () => {
    const tried: number[] = [];
    backtrack(
      (length) => {
        tried.push(length);
        return Number.POSITIVE_INFINITY;
      },
      0,
      -1,
      { start: 0.5, maxHalvings: 3 },
    );
    expect(tried).toEqual([0.5, 0.25, 0.125, 0.0625]);
  });

  it("tries the first length and up to maxHalvings more, 39 by default", () => {
    let calls = 0;
    const evaluate = (): number => {
      calls++;
      return Number.POSITIVE_INFINITY;
    };
    expect(backtrack(evaluate, 0, -1, { maxHalvings: 0 })).toBeNull();
    expect(calls).toBe(1);
    calls = 0;
    backtrack(evaluate, 0, -1);
    expect(calls).toBe(40);
  });

  it("accepts a rise within the slack, and a trial that does not rise when no decrease is asked", () => {
    expect(backtrack(() => 4 + 1e-13, 4, -1, { decrease: 0, slack: 1e-12 })).toBe(1);
    expect(backtrack(() => 4 + 1e-11, 4, -1, { decrease: 0, slack: 1e-12, maxHalvings: 2 })).toBeNull();
    expect(backtrack(() => 4, 4, 0, { decrease: 0 })).toBe(1);
  });

  it("leaves the last trial at the accepted length", () => {
    let at = Number.NaN;
    const length = backtrack(
      (t) => {
        at = t;
        return t <= 0.125 ? -1 : 1;
      },
      0,
      -1,
    );
    expect(length).toBe(0.125);
    expect(at).toBe(0.125);
  });

  it("rejects a trial that is NaN, as it satisfies no comparison", () => {
    expect(backtrack(() => Number.NaN, 4, -1, { maxHalvings: 2 })).toBeNull();
  });

  it("asks the further test only of lengths that decrease the objective enough, and halves on past one it refuses", () => {
    const asked: number[] = [];
    // (10 t - 2)² is 64 at t = 1, 9 at 1/2, 0.25 at 1/4 and 0.25 at 1/8: the first two do not decrease enough.
    const length = backtrack(objective(10), 4, -40, {
      accept: (t) => {
        asked.push(t);
        return t < 0.25;
      },
    });
    expect(asked).toEqual([0.25, 0.125]);
    expect(length).toBe(0.125);
    expect(backtrack(objective(10), 4, -40, { accept: () => false, maxHalvings: 5 })).toBeNull();
  });

  it("takes the Armijo constant 1e-4 by default", () => {
    expect(ARMIJO).toBe(1e-4);
    // At slope -1 and length 1 the accepted bound is the value less 1e-4.
    expect(backtrack(() => 4 - 0.9e-4, 4, -1, { maxHalvings: 0 })).toBeNull();
    expect(backtrack(() => 4 - 1.1e-4, 4, -1, { maxHalvings: 0 })).toBe(1);
  });

  it("finds a length that meets the condition along any descent direction of a quadratic", () => {
    fc.assert(
      fc.property(
        fc.double({ min: 0.1, max: 4, noNaN: true }),
        fc.double({ min: -3, max: -0.01, noNaN: true }),
        (curvature, slope) => {
          const f = (t: number): number => 1 + slope * t + curvature * t * t;
          const length = backtrack(f, 1, slope);
          expect(length).not.toBeNull();
          expect(f(length ?? 0)).toBeLessThanOrEqual(1 + ARMIJO * (length ?? 0) * slope);
        },
      ),
    );
  });
});

describe("stepAlong", () => {
  /** The function (x - 2)² of one coordinate, with its gradient. */
  const square = (x: Float64Array, gradient: Float64Array): number => {
    gradient[0] = 2 * ((x[0] ?? 0) - 2);
    return ((x[0] ?? 0) - 2) ** 2;
  };
  const pointAt = (x: number): DescentPoint => {
    const point = { x: Float64Array.of(x), gradient: new Float64Array(1), value: 0 };
    point.value = square(point.x, point.gradient);
    return point;
  };

  it("leaves the accepted point in the trial after halving the step until the value falls enough", () => {
    const point = pointAt(0);
    const trial = { x: new Float64Array(1), gradient: new Float64Array(1), value: 0 };
    // A step of 10 along +1 overshoots to 10 and 5, and the length 1/4 lands at 2.5: (0.5)² = 0.25 <= 4 - 1e-4 · 4 · 10 / 4.
    expect(stepAlong(square, point, Float64Array.of(10), trial)).toBe(true);
    expect(Array.from(trial.x)).toEqual([2.5]);
    expect(trial.value).toBe(0.25);
    expect(Array.from(trial.gradient)).toEqual([1]);
  });

  it("refuses a direction that does not descend", () => {
    const point = pointAt(0);
    const trial = { x: new Float64Array(1), gradient: new Float64Array(1), value: 0 };
    expect(stepAlong(square, point, Float64Array.of(-1), trial)).toBe(false);
  });

  /** The same function with its value and gradient apart, recording the points at which each is asked. */
  const apart = (asked: { values: number[]; gradients: number[] }, undefinedAt?: number): ValueAndGradient => ({
    value: (x) => {
      asked.values.push(x[0] ?? 0);
      return ((x[0] ?? 0) - 2) ** 2;
    },
    gradient: (x, gradient) => {
      asked.gradients.push(x[0] ?? 0);
      gradient[0] = x[0] === undefinedAt ? Number.NaN : 2 * ((x[0] ?? 0) - 2);
    },
  });

  it("asks a function given apart for its value at each length tried, and for its gradient at the accepted one alone", () => {
    const asked = { values: [] as number[], gradients: [] as number[] };
    const trial = { x: new Float64Array(1), gradient: new Float64Array(1), value: 0 };
    expect(stepAlong(apart(asked), pointAt(0), Float64Array.of(10), trial)).toBe(true);
    expect(asked).toEqual({ values: [10, 5, 2.5], gradients: [2.5] });
    const together = { x: new Float64Array(1), gradient: new Float64Array(1), value: 0 };
    stepAlong(square, pointAt(0), Float64Array.of(10), together);
    expect(trial).toEqual(together);
  });

  it("halves on past a length where the gradient is not finite", () => {
    const asked = { values: [] as number[], gradients: [] as number[] };
    const trial = { x: new Float64Array(1), gradient: new Float64Array(1), value: 0 };
    expect(stepAlong(apart(asked, 2.5), pointAt(0), Float64Array.of(10), trial)).toBe(true);
    expect(asked).toEqual({ values: [10, 5, 2.5, 1.25], gradients: [2.5, 1.25] });
    expect(Array.from(trial.x)).toEqual([1.25]);
    expect(Array.from(trial.gradient)).toEqual([-1.5]);
  });
});
