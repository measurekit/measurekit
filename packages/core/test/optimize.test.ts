import { describe, expect, it } from "vitest";
import { bfgs, type GradientFunction } from "../src/index.js";

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

  it("does not change its start", () => {
    const start = [-1.2, 1];
    bfgs(rosenbrock, start);
    expect(start).toEqual([-1.2, 1]);
  });
});
