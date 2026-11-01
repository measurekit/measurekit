import { logistic, normalQuantile } from "@measurekit/core";
import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { createRandom } from "../src/index.js";
import { expectationMaximization } from "../src/math/expectation-maximization.js";
import { forwardBackward } from "../src/math/hidden-markov.js";
import { choleskySolve } from "../src/math/linear-algebra.js";
import { outcomeLogLikelihood } from "../src/math/logistic.js";
import { logisticRegression } from "../src/math/logistic-regression.js";
import { permutation } from "../src/math/permutation.js";
import { quinticDerivative, quinticHermite, quinticMaximum } from "../src/math/quintic.js";
import { rank } from "../src/math/rank.js";
import type { SparseMatrix } from "../src/math/sparse-matrix.js";
import hiddenMarkov from "./fixtures/math-forward-backward.json" with { type: "json" };
import functions from "./fixtures/math-functions.json" with { type: "json" };
import regressions from "./fixtures/math-logistic-regression.json" with { type: "json" };
import { defined, expectAllClose, expectClose } from "./helpers.js";

/** Relative tolerance of the normal quantile and the logistic function against R. */
const FUNCTION_TOLERANCE = 1e-14;
/** Absolute tolerance of forward-backward quantities against enumeration. */
const FORWARD_BACKWARD_TOLERANCE = 1e-12;
/** Absolute tolerance of logistic regression coefficients and log-likelihoods against glm. */
const REGRESSION_TOLERANCE = 1e-10;

describe("normalQuantile and logistic", () => {
  it("match R's qnorm and plogis", () => {
    functions.p.forEach((p, i) => {
      const expected = defined(functions.normalQuantile[i]);
      expectClose(normalQuantile(p), expected, FUNCTION_TOLERANCE * Math.abs(expected), `qnorm(${p})`);
    });
    functions.x.forEach((x, i) => {
      const expected = defined(functions.logistic[i]);
      expectClose(logistic(x), expected, FUNCTION_TOLERANCE * expected, `plogis(${x})`);
    });
  });
});

describe("outcomeLogLikelihood", () => {
  it("weighs the log-probabilities of a correct and an incorrect outcome", () => {
    expect(outcomeLogLikelihood(0.8, 1)).toBe(Math.log(0.8));
    expect(outcomeLogLikelihood(0.8, 0)).toBe(Math.log1p(-0.8));
    expectClose(outcomeLogLikelihood(0.8, 0.25), 0.25 * Math.log(0.8) + 0.75 * Math.log(0.2), 1e-15);
  });

  it("is 0 for a certain prediction that is right and -Infinity for one that is wrong, never NaN", () => {
    expect(outcomeLogLikelihood(1, 1)).toBe(0);
    expect(outcomeLogLikelihood(0, 0)).toBe(0);
    expect(outcomeLogLikelihood(0, 1)).toBe(Number.NEGATIVE_INFINITY);
    expect(outcomeLogLikelihood(1, 0)).toBe(Number.NEGATIVE_INFINITY);
  });
});

describe("permutation", () => {
  it("returns every index once and reproduces from the same seed", () => {
    fc.assert(
      fc.property(fc.integer({ min: 0, max: 50 }), fc.integer(), (size, seed) => {
        const order = permutation(size, createRandom(seed));
        expect([...order].toSorted((a, b) => a - b)).toEqual(Array.from({ length: size }, (_, i) => i));
        expect(permutation(size, createRandom(seed))).toEqual(order);
      }),
    );
  });
});

describe("choleskySolve", () => {
  it("solves a positive definite system", () => {
    const out = new Float64Array(2);
    expect(choleskySolve(Float64Array.of(4, 2, 2, 3), 2, Float64Array.of(2, 1), out)).toBe(true);
    expectAllClose(out, [0.5, 0], 1e-15);
  });

  it("returns false for a matrix that is not positive definite", () => {
    const out = new Float64Array(2);
    expect(choleskySolve(Float64Array.of(1, 2, 2, 1), 2, Float64Array.of(1, 1), out)).toBe(false);
    expect(choleskySolve(Float64Array.of(1, 1, 1, 1), 2, Float64Array.of(1, 1), out)).toBe(false);
  });
});

/** Stores dense rows in compressed sparse row form, leaving out zeros. */
function sparse(rows: readonly (readonly number[])[]): SparseMatrix {
  const rowStarts = [0];
  const columns: number[] = [];
  const values: number[] = [];
  for (const row of rows) {
    row.forEach((value, column) => {
      if (value === 0) return;
      columns.push(column);
      values.push(value);
    });
    rowStarts.push(columns.length);
  }
  return {
    columnCount: rows[0]?.length ?? 0,
    rowStarts: Int32Array.from(rowStarts),
    columns: Int32Array.from(columns),
    values: Float64Array.from(values),
  };
}

describe("logisticRegression", () => {
  it("matches glm for 0/1 outcomes and outcomes in [0, 1]", () => {
    for (const [index, regression] of regressions.regressions.entries()) {
      const fit = logisticRegression(sparse(regression.design), Float64Array.from(regression.outcomes), {
        tolerance: 1e-10,
        maxIterations: 100,
      });
      expect(fit.converged).toBe(true);
      expectAllClose(fit.coefficients, regression.coefficients, REGRESSION_TOLERANCE, `regression ${index}`);
      expectClose(fit.logLikelihood, regression.logLikelihood, REGRESSION_TOLERANCE, `regression ${index}`);
    }
  });
});

describe("forwardBackward", () => {
  it("matches the posteriors, predictions and log-likelihood of enumerating every path", () => {
    for (const [index, sequence] of hiddenMarkov.sequences.entries()) {
      const result = forwardBackward({
        stateCount: sequence.stateCount,
        initial: Float64Array.from(sequence.initial),
        transitions: Float64Array.from(sequence.transitions),
        emissions: Float64Array.from(sequence.emissions),
      });
      const label = `sequence ${index}`;
      expectClose(result.logLikelihood, sequence.logLikelihood, FORWARD_BACKWARD_TOLERANCE, label);
      expectAllClose(result.posteriors, sequence.posteriors, FORWARD_BACKWARD_TOLERANCE, `${label} posteriors`);
      expectAllClose(
        result.transitionPosteriors,
        sequence.transitionPosteriors,
        FORWARD_BACKWARD_TOLERANCE,
        `${label} transitions`,
      );
      expectAllClose(result.predictions, sequence.predictions, FORWARD_BACKWARD_TOLERANCE, `${label} predictions`);
    }
  });

  it("keeps a sequence of 5,000 steps from underflowing", () => {
    const steps = 5000;
    const emissions = Float64Array.from({ length: 2 * steps }, (_, i) => (i % 3 === 0 ? 0.1 : 0.6));
    const transitions = Float64Array.from({ length: 4 * (steps - 1) }, (_, i) => [0.9, 0.1, 0.2, 0.8][i % 4] ?? 0);
    const result = forwardBackward({ stateCount: 2, initial: Float64Array.of(0.5, 0.5), transitions, emissions });
    expect(Number.isFinite(result.logLikelihood)).toBe(true);
    expect(result.logLikelihood).toBeLessThan(-1000);
    for (let t = 0; t < steps; t++) {
      expectClose(defined(result.posteriors[2 * t]) + defined(result.posteriors[2 * t + 1]), 1, 1e-12);
    }
  });

  it("smooths a long run where a state's filtered probability underflows to 0, as in log space", () => {
    // Knowledge tracing without forgetting: 800 correct answers drive P(unknown) below the smallest double, then 400
    // incorrect ones. Rabiner's scaled backward variables overflow here and give NaN.
    const [prior, learn, guess, slip] = [0.89, 0.005, 0.013, 0.0073];
    const outcomes = Array.from({ length: 1200 }, (_, t) => (t < 800 ? 1 : 0));
    const steps = outcomes.length;
    const transition = [1 - learn, learn, 0, 1];
    const emission = (y: number) => (y === 1 ? [guess, 1 - slip] : [1 - guess, slip]);
    const result = forwardBackward({
      stateCount: 2,
      initial: Float64Array.of(1 - prior, prior),
      transitions: Float64Array.from({ length: 4 * (steps - 1) }, (_, i) => transition[i % 4] ?? 0),
      emissions: Float64Array.from(outcomes.flatMap(emission)),
    });
    // The same posteriors from forward and backward variables in log space, by log-sum-exp.
    const logSum = (a: number, b: number) =>
      a === -Infinity ? b : Math.max(a, b) + Math.log1p(Math.exp(-Math.abs(a - b)));
    const logA = transition.map(Math.log);
    const alpha = [
      [Math.log(1 - prior) + Math.log(defined(emission(1)[0])), Math.log(prior) + Math.log(defined(emission(1)[1]))],
    ];
    for (let t = 1; t < steps; t++) {
      const e = emission(defined(outcomes[t])).map(Math.log);
      const previous = defined(alpha[t - 1]);
      alpha.push(
        [0, 1].map(
          (j) =>
            logSum(defined(previous[0]) + defined(logA[j]), defined(previous[1]) + defined(logA[2 + j])) +
            defined(e[j]),
        ),
      );
    }
    const beta: number[][] = Array.from({ length: steps }, () => [0, 0]);
    for (let t = steps - 2; t >= 0; t--) {
      const e = emission(defined(outcomes[t + 1])).map(Math.log);
      const next = defined(beta[t + 1]);
      beta[t] = [0, 1].map((i) =>
        logSum(
          defined(logA[2 * i]) + defined(e[0]) + defined(next[0]),
          defined(logA[2 * i + 1]) + defined(e[1]) + defined(next[1]),
        ),
      );
    }
    const exact = alpha.flatMap((a, t) => {
      const joint = [0, 1].map((i) => defined(a[i]) + defined(defined(beta[t])[i]));
      const total = logSum(defined(joint[0]), defined(joint[1]));
      return joint.map((value) => Math.exp(value - total));
    });
    expect(result.posteriors.every(Number.isFinite)).toBe(true);
    expectAllClose(result.posteriors, exact, 1e-10);
  });
});

describe("expectationMaximization", () => {
  // Observed 0/1 sequences of a two-state hidden Markov model, fitted by Baum-Welch: the use EM has in BKT.
  const observations = [
    [0, 0, 1, 1, 1],
    [0, 1, 0, 1, 1, 1],
    [1, 1, 1],
    [0, 0, 0, 1],
    [0, 1, 1, 0, 1, 1, 1],
  ];
  interface Parameters {
    readonly initial: readonly number[];
    readonly transition: readonly number[];
    readonly correct: readonly number[];
  }
  const baumWelch = (parameters: Parameters): { logLikelihood: number; parameters: Parameters } => {
    let logLikelihood = 0;
    const initial = [0, 0];
    const transitions = [0, 0, 0, 0];
    const visits = [0, 0];
    const corrects = [0, 0];
    for (const sequence of observations) {
      const result = forwardBackward({
        stateCount: 2,
        initial: Float64Array.from(parameters.initial),
        transitions: Float64Array.from(
          { length: 4 * (sequence.length - 1) },
          (_, i) => parameters.transition[i % 4] ?? 0,
        ),
        emissions: Float64Array.from(sequence.flatMap((y) => parameters.correct.map((p) => (y === 1 ? p : 1 - p)))),
      });
      logLikelihood += result.logLikelihood;
      sequence.forEach((y, t) => {
        for (let k = 0; k < 2; k++) {
          const posterior = defined(result.posteriors[2 * t + k]);
          if (t === 0) initial[k] = defined(initial[k]) + posterior;
          visits[k] = defined(visits[k]) + posterior;
          corrects[k] = defined(corrects[k]) + posterior * y;
        }
      });
      result.transitionPosteriors.forEach((value, i) => {
        transitions[i % 4] = defined(transitions[i % 4]) + value;
      });
    }
    const from = [defined(transitions[0]) + defined(transitions[1]), defined(transitions[2]) + defined(transitions[3])];
    return {
      logLikelihood,
      parameters: {
        initial: initial.map((value) => value / observations.length),
        transition: transitions.map((value, i) => value / defined(from[Math.floor(i / 2)])),
        correct: corrects.map((value, k) => value / defined(visits[k])),
      },
    };
  };
  const start: Parameters = { initial: [0.6, 0.4], transition: [0.7, 0.3, 0.1, 0.9], correct: [0.3, 0.8] };

  it("raises the log-likelihood at every iteration and reports the one of the parameters it returns", () => {
    const logLikelihoods: number[] = [];
    const result = expectationMaximization(
      start,
      (parameters) => {
        const step = baumWelch(parameters);
        logLikelihoods.push(step.logLikelihood);
        return step;
      },
      { tolerance: 1e-10, maxIterations: 1000 },
    );
    expect(result.converged).toBe(true);
    expect(result.iterations).toBe(logLikelihoods.length);
    for (let i = 1; i < logLikelihoods.length; i++) {
      expect(defined(logLikelihoods[i])).toBeGreaterThanOrEqual(defined(logLikelihoods[i - 1]) - 1e-12);
    }
    expect(baumWelch(result.parameters).logLikelihood).toBe(result.logLikelihood);
  });

  it("stops without convergence after the most iterations", () => {
    const result = expectationMaximization(start, baumWelch, { tolerance: 0, maxIterations: 3 });
    expect(result).toMatchObject({ iterations: 3, converged: false });
  });
});

describe("quintic Hermite interpolation", () => {
  const coefficient = fc.double({ min: -5, max: 5, noNaN: true });

  it("reproduces a quintic from its jets at 0 and 1", () => {
    fc.assert(
      fc.property(fc.array(coefficient, { minLength: 6, maxLength: 6 }), (values) => {
        const c = Float64Array.from(values);
        const value = (t: number) => c.reduce((sum, ci, i) => sum + ci * t ** i, 0);
        const jet = (t: number) => [value(t), quinticDerivative(c, t), quinticDerivative(c, t, 2)] as const;
        expectAllClose(quinticHermite(jet(0), jet(1)), c, 1e-12);
      }),
    );
  });

  it("finds the root of p' between a rise and a fall", () => {
    fc.assert(
      fc.property(
        fc.double({ min: 0.01, max: 0.99, noNaN: true }),
        fc.double({ min: 0.1, max: 5, noNaN: true }),
        (root, width) => {
          // p(t) = -width (t - root)² / 2 + (t - root)³ / 10: p'(root) = 0, rising before and falling after.
          const c = quinticHermite(
            [-(width * root ** 2) / 2 - root ** 3 / 10, width * root + (3 * root ** 2) / 10, -width - (6 * root) / 10],
            [
              -(width * (1 - root) ** 2) / 2 + (1 - root) ** 3 / 10,
              -width * (1 - root) + (3 * (1 - root) ** 2) / 10,
              -width + (6 * (1 - root)) / 10,
            ],
          );
          fc.pre(quinticDerivative(c, 0) > 0 && quinticDerivative(c, 1) <= 0);
          expect(Math.abs(quinticDerivative(c, quinticMaximum(c)))).toBeLessThan(1e-12);
        },
      ),
    );
  });
});

describe("rank", () => {
  const columns = (...values: number[][]): Float64Array[] => values.map((column) => Float64Array.from(column));

  it("counts the columns that the earlier ones do not span", () => {
    expect(rank(columns([1, 0, 0], [0, 1, 0], [0, 0, 1]))).toBe(3);
    expect(rank(columns([1, 2, 3], [2, 4, 6], [0, 1, 0]))).toBe(2);
    expect(rank(columns([1, 2], [3, 4], [5, 6]))).toBe(2);
    expect(rank([])).toBe(0);
  });

  it("gives a column of 0 no rank, and a column of a different scale its own tolerance", () => {
    expect(rank(columns([0, 0, 0], [1, 1, 0]))).toBe(1);
    expect(rank(columns([1e-12, 0], [0, 1e12]))).toBe(2);
  });

  it("calls a column dependent when under 1e-7 of its norm is left, as the tolerance of R's qr", () => {
    expect(rank(columns([1, 0], [1, 1e-9]))).toBe(1);
    expect(rank(columns([1, 0], [1, 1e-6]))).toBe(2);
  });

  it("finds the same rank in any order of the columns", () => {
    const independent = fc.uniqueArray(fc.nat({ max: 5 }), { minLength: 1, maxLength: 6 });
    const weights = fc.array(fc.array(fc.integer({ min: -3, max: 3 }), { minLength: 6, maxLength: 6 }), {
      maxLength: 6,
    });
    fc.assert(
      fc.property(independent, weights, fc.integer(), (chosen, mixes, seed) => {
        const unit = (at: number) => Array.from({ length: 6 }, (_, i) => (i === at ? 1 : 0));
        const mixed = mixes.map((mix) =>
          chosen
            .map((at, k) => unit(at).map((value) => value * (mix[k] ?? 0)))
            .reduce((sum, column) => sum.map((value, i) => value + (column[i] ?? 0)), new Array<number>(6).fill(0)),
        );
        const all = [...chosen.map(unit), ...mixed];
        expect(rank(columns(...all))).toBe(chosen.length);
        const shuffled = Array.from(permutation(all.length, createRandom(seed)), (at) => defined(all[at]));
        expect(rank(columns(...shuffled))).toBe(chosen.length);
      }),
    );
  });
});
