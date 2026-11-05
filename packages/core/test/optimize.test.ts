import { describe, expect, it } from "vitest";
import { bfgs, type GradientFunction, type ValueAndGradient } from "../src/index.js";

/** The Rosenbrock function, minimal at (1, 1). */
const rosenbrock: GradientFunction = (x, gradient) => {
  const a = x[0] ?? 0;
  const b = x[1] ?? 0;
  gradient[0] = -2 * (1 - a) - 400 * a * (b - a * a);
  gradient[1] = 200 * (b - a * a);
  return (1 - a) ** 2 + 100 * (b - a * a) ** 2;
};

/** (v - 3)², minimal at 3. */
const parabola: GradientFunction = (x, gradient) => {
  const v = x[0] ?? 0;
  gradient[0] = 2 * (v - 3);
  return (v - 3) ** 2;
};

describe("bfgs", () => {
  it("minimizes the Rosenbrock function", () => {
    const result = bfgs(rosenbrock, [-1.2, 1]);
    expect(result.converged).toBe(true);
    expect(result.x[0]).toBeCloseTo(1, 7);
    expect(result.x[1]).toBeCloseTo(1, 7);
    expect(result.value).toBeLessThan(1e-14);
  });

  it("minimizes a function of one variable, and does not move from a point whose gradient is 0", () => {
    expect(bfgs(parabola, [10]).x[0]).toBeCloseTo(3, 8);
    expect(bfgs(parabola, [3])).toMatchObject({ iterations: 0, converged: true, value: 0 });
  });

  it("reports non-convergence within the iteration limit", () => {
    expect(bfgs(rosenbrock, [-1.2, 1], { maxIterations: 3 })).toMatchObject({ iterations: 3, converged: false });
  });

  it("stops at a gradient within the tolerance", () => {
    const loose = bfgs(rosenbrock, [-1.2, 1], { tolerance: 1e-2 });
    const tight = bfgs(rosenbrock, [-1.2, 1], { tolerance: 1e-8 });
    expect(loose.converged).toBe(true);
    expect(loose.iterations).toBeLessThan(tight.iterations);
  });

  it("stops where rounding hides any further decrease, below a tolerance that rounding does not let it reach", () => {
    // Minimal where e^x + y / 2 = 3 and sinh(y - 0.7) + x / 2 = 0, a point no double holds exactly.
    const smooth: GradientFunction = (x, gradient) => {
      const [a = 0, b = 0] = x;
      gradient[0] = Math.exp(a) - 3 + 0.5 * b;
      gradient[1] = Math.sinh(b - 0.7) + 0.5 * a;
      return Math.exp(a) - 3 * a + Math.cosh(b - 0.7) + 0.5 * a * b;
    };
    const result = bfgs(smooth, [2, -1], { tolerance: Number.MIN_VALUE });
    expect(result.converged).toBe(true);
    expect(result.iterations).toBeLessThan(100);
    const gradient = new Float64Array(2);
    smooth(result.x as Float64Array, gradient);
    // A step of a gradient g decreases the value by about g², which the rounding of a value near 1 hides below g = 1e-8.
    expect(Math.max(...gradient.map(Math.abs))).toBeLessThan(1e-6);
  });

  it("does not report convergence at a point where the value or the gradient is not finite", () => {
    const undefinedAtStart: GradientFunction = (_, gradient) => {
      gradient[0] = Number.NaN;
      return Number.NaN;
    };
    expect(bfgs(undefinedAtStart, [1])).toMatchObject({ iterations: 0, converged: false });
    const infiniteSlope: GradientFunction = (_, gradient) => {
      gradient[0] = Number.POSITIVE_INFINITY;
      return 0;
    };
    expect(bfgs(infiniteSlope, [1]).converged).toBe(false);
  });

  it("reaches the same point from a function given with its gradient apart, asking the gradient once a step", () => {
    let values = 0;
    let gradients = 0;
    const scratch = new Float64Array(2);
    const apart: ValueAndGradient = {
      value: (x) => {
        values++;
        return rosenbrock(x, scratch);
      },
      gradient: (x, gradient) => {
        gradients++;
        rosenbrock(x, gradient);
      },
    };
    const together = bfgs(rosenbrock, [-1.2, 1]);
    const result = bfgs(apart, [-1.2, 1]);
    expect(result).toEqual(together);
    // The start and each accepted step: the restarts of the approximation accept no step.
    expect(gradients).toBeLessThanOrEqual(result.iterations + 1);
    expect(values).toBeGreaterThan(gradients);
  });

  it("does not report convergence where the gradient given apart is not finite", () => {
    const undefinedSlope: ValueAndGradient = {
      value: () => 0,
      gradient: (_, gradient) => {
        gradient[0] = Number.NaN;
      },
    };
    expect(bfgs(undefinedSlope, [1])).toMatchObject({ iterations: 0, converged: false });
  });

  it("does not change its start", () => {
    const start = [-1.2, 1];
    bfgs(rosenbrock, start);
    expect(start).toEqual([-1.2, 1]);
  });
});
