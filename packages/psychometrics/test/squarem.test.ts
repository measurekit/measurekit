import { describe, expect, it } from "vitest";
import { type Iteration, squarem } from "../src/irt/squarem.js";

/** An iteration of x → x + (target - x) (1 - rate) on one number, whose log-likelihood rises toward the target. */
function linear(target: number, rate: number, start: number): Iteration & { readonly state: () => number } {
  let x = start;
  return {
    step() {
      const next = x + (target - x) * (1 - rate);
      const change = Math.abs(next - x);
      const logLikelihood = -((target - x) ** 2);
      x = next;
      return { logLikelihood, change };
    },
    get: () => Float64Array.of(x),
    set: (state) => {
      x = state[0] ?? 0;
    },
    state: () => x,
  };
}

describe("squarem", () => {
  it("reaches the limit of a linearly convergent iteration in a few steps", () => {
    const iteration = linear(3, 0.9, 0);
    const { converged, iterations } = squarem(iteration, { tolerance: 1e-6, maxIterations: 500 });
    expect(converged).toBe(true);
    // The plain iteration needs about 130 steps for a change of 1e-6.
    expect(iterations).toBeLessThanOrEqual(10);
    expect(iteration.state()).toBeCloseTo(3, 4);
  });

  it("counts every EM iteration against maxIterations", () => {
    for (const maxIterations of [1, 2, 3, 4, 5, 6, 7]) {
      const { converged, iterations } = squarem(linear(3, 0.99, 0), { tolerance: 0, maxIterations });
      expect({ converged, iterations }).toEqual({ converged: false, iterations: maxIterations });
    }
  });

  it("falls back to the plain step when the jump lowers the log-likelihood", () => {
    // The log-likelihood falls at the point of an extrapolation beyond the target, so that the cycle goes to x2.
    let jumps = 0;
    let x = 0;
    const iteration: Iteration = {
      step() {
        const next = x + (3 - x) * 0.1;
        const logLikelihood = x > 3 ? Number.NEGATIVE_INFINITY : -((3 - x) ** 2);
        const change = Math.abs(next - x);
        x = next;
        return { logLikelihood, change };
      },
      get: () => Float64Array.of(x),
      set: (state) => {
        jumps++;
        x = state[0] ?? 0;
      },
    };
    const { converged } = squarem(iteration, { tolerance: 1e-6, maxIterations: 500 });
    expect(converged).toBe(true);
    expect(jumps).toBeGreaterThan(0);
    expect(x).toBeCloseTo(3, 4);
  });

  it("stops when an iteration changes nothing", () => {
    const { converged, iterations } = squarem(linear(0, 0.5, 0), { tolerance: 1e-6, maxIterations: 50 });
    expect({ converged, iterations }).toEqual({ converged: true, iterations: 1 });
  });
});
