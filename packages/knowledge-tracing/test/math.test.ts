import { logistic, normalQuantile, shuffle } from "@measurekit/core";
import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { createRandom } from "../src/index.js";
import {
  choleskySolve,
  createHiddenMarkovBuffers,
  createProblem,
  expectationMaximization,
  type HiddenMarkovBuffers,
  logisticRegression,
  newtonCg,
  outcomeLogLikelihood,
  penalizedNewton,
  quinticDerivative,
  quinticHermite,
  quinticMaximum,
  rank,
  type SparseMatrix,
  smoothSequence,
} from "../src/math/internal.js";
import { defined, expectAllClose, expectClose } from "./assertions.js";
import { readFixture } from "./fixtures.js";

interface MathForwardBackwardFixture {
  sequences: {
    stateCount: number;
    initial: number[];
    transitions: number[];
    emissions: number[];
    logLikelihood: number;
    posteriors: number[];
    transitionPosteriors: number[];
    predictions: number[];
  }[];
}
const hiddenMarkov = readFixture<MathForwardBackwardFixture>("math-forward-backward.json");

interface MathFunctionsFixture {
  p: number[];
  normalQuantile: number[];
  x: number[];
  logistic: number[];
}
const functions = readFixture<MathFunctionsFixture>("math-functions.json");

interface MathLogisticRegressionFixture {
  regressions: { design: number[][]; outcomes: number[]; coefficients: number[]; logLikelihood: number }[];
}
const regressions = readFixture<MathLogisticRegressionFixture>("math-logistic-regression.json");

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

describe("newtonCg", () => {
  // f(x) = x'Ax / 2 - b'x, whose minimum solves A x = b: x = (2/9, 1/9, 13/9) for the A and b below.
  const matrix = [
    [4, 1, 0],
    [1, 3, 1],
    [0, 1, 2],
  ];
  const product = (v: Float64Array): number[] =>
    matrix.map((row) => row.reduce((sum, a, j) => sum + a * (v[j] ?? 0), 0));
  const b = [1, 2, 3];
  const quadratic = {
    value: (x: Float64Array) =>
      product(x).reduce((sum, ax, i) => sum + (ax * (x[i] ?? 0)) / 2 - (b[i] ?? 0) * (x[i] ?? 0), 0),
    linearize(x: Float64Array, gradient: Float64Array, diagonal: Float64Array) {
      product(x).forEach((ax, i) => {
        gradient[i] = ax - (b[i] ?? 0);
        diagonal[i] = defined(matrix[i]?.[i]);
      });
    },
    hessianVector(v: Float64Array, out: Float64Array) {
      product(v).forEach((value, i) => {
        out[i] = value;
      });
    },
  };

  it("minimizes a quadratic to its exact minimum", () => {
    const x = new Float64Array(3);
    const result = newtonCg(quadratic, x, { tolerance: 1e-12, maxIterations: 10 });
    expect(result.converged).toBe(true);
    expectAllClose(x, [2 / 9, 1 / 9, 13 / 9], 1e-12);
  });

  it("reports a start at the minimum as converged without iterating, and the iterations it was allowed as not converged", () => {
    expect(
      newtonCg(quadratic, Float64Array.from([2 / 9, 1 / 9, 13 / 9]), { tolerance: 1e-9, maxIterations: 5 }),
    ).toEqual({ iterations: 0, converged: true });
    const x = new Float64Array(3);
    expect(newtonCg(quadratic, x, { tolerance: 1e-30, maxIterations: 1 })).toEqual({ iterations: 1, converged: false });
  });
});

describe("logisticRegression with a ridge", () => {
  const regression = defined(regressions.regressions[0]);
  const design = sparse(regression.design);
  const outcomes = Float64Array.from(regression.outcomes);

  it("reaches the minimum that Newton's method with dense matrices does, with and without fixed intercepts", () => {
    const ridge = 2;
    const groups = Int32Array.from({ length: outcomes.length }, (_, row) => row % 7);
    const dense = createProblem(design, outcomes, undefined, 0);
    const beta = new Float64Array(design.columnCount);
    penalizedNewton(dense, beta, new Float64Array(0), {
      ridge,
      groupPenalty: 0,
      flats: [],
      tolerance: 1e-12,
      maxIterations: 100,
    });
    const plain = logisticRegression(design, outcomes, { ridge, tolerance: 1e-10, maxIterations: 100 });
    expect(plain.converged).toBe(true);
    expectAllClose(plain.coefficients, beta, 1e-8, "no groups");
    const grouped = createProblem(design, outcomes, groups, 7);
    const betaGrouped = new Float64Array(design.columnCount);
    const intercepts = new Float64Array(7);
    penalizedNewton(grouped, betaGrouped, intercepts, {
      ridge,
      groupPenalty: ridge,
      flats: [],
      tolerance: 1e-12,
      maxIterations: 100,
    });
    const withGroups = logisticRegression(design, outcomes, {
      groups: { indices: groups, count: 7, kind: "fixed" },
      ridge,
      tolerance: 1e-10,
      maxIterations: 100,
    });
    expectAllClose(withGroups.coefficients, betaGrouped, 1e-8, "groups");
    expectAllClose(withGroups.groupIntercepts, intercepts, 1e-8, "intercepts");
  });

  it("shrinks every coefficient toward 0 as the ridge grows", () => {
    const norms = [0.1, 1, 10].map((ridge) => {
      const fit = logisticRegression(design, outcomes, { ridge, tolerance: 1e-10, maxIterations: 100 });
      return Math.hypot(...fit.coefficients);
    });
    expect(norms[0]).toBeGreaterThan(defined(norms[1]));
    expect(norms[1]).toBeGreaterThan(defined(norms[2]));
  });

  it("gives no standard errors with a ridge", () => {
    const fit = logisticRegression(design, outcomes, {
      ridge: 1,
      standardErrors: true,
      tolerance: 1e-10,
      maxIterations: 100,
    });
    expect(fit.standardErrors).toBeNull();
  });
});

describe("logisticRegression standard errors", () => {
  const regression = defined(regressions.regressions[0]);
  const design = sparse(regression.design);
  const outcomes = Float64Array.from(regression.outcomes);

  it("are the square roots of the diagonal of the inverse of X'WX, with the binomial weight p (1 - p)", () => {
    const fit = logisticRegression(design, outcomes, { standardErrors: true, tolerance: 1e-12, maxIterations: 100 });
    const p = design.columnCount;
    const information = Array.from({ length: p }, () => new Array<number>(p).fill(0));
    for (let row = 0; row < outcomes.length; row++) {
      const entries = Array.from(
        { length: (design.rowStarts[row + 1] ?? 0) - (design.rowStarts[row] ?? 0) },
        (_, k) => (design.rowStarts[row] ?? 0) + k,
      );
      const eta = entries.reduce(
        (sum, e) => sum + defined(design.values[e]) * defined(fit.coefficients[defined(design.columns[e])]),
        0,
      );
      const weight = logistic(eta) * (1 - logistic(eta));
      for (const a of entries) {
        for (const c of entries) {
          const cell = defined(information[defined(design.columns[a])]);
          cell[defined(design.columns[c])] =
            defined(cell[defined(design.columns[c])]) + weight * defined(design.values[a]) * defined(design.values[c]);
        }
      }
    }
    const inverse = invert(information);
    expectAllClose(
      defined(fit.standardErrors).coefficients,
      inverse.map((row, j) => Math.sqrt(defined(row[j]))),
      1e-9,
    );
  });

  it("are null for a coefficient that no row uses", () => {
    const columns = Int32Array.from([0, 1, 0, 1]);
    const matrix: SparseMatrix = {
      columnCount: 3,
      rowStarts: Int32Array.from([0, 2, 4]),
      columns,
      values: Float64Array.from([1, 1, 1, 2]),
    };
    const fit = logisticRegression(matrix, Float64Array.from([0, 1]), {
      standardErrors: true,
      tolerance: 1e-9,
      maxIterations: 50,
    });
    expect(fit.standardErrors).toBeNull();
  });
});

/** Returns the inverse of a small square matrix by Gauss-Jordan elimination with partial pivoting. */
function invert(matrix: readonly (readonly number[])[]): number[][] {
  const n = matrix.length;
  const work = matrix.map((row, i) => [...row, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);
  for (let column = 0; column < n; column++) {
    const pivot = work.reduce(
      (best, row, i) =>
        i >= column && Math.abs(defined(row[column])) > Math.abs(defined(defined(work[best])[column])) ? i : best,
      column,
    );
    [work[column], work[pivot]] = [defined(work[pivot]), defined(work[column])];
    const scale = defined(defined(work[column])[column]);
    work[column] = defined(work[column]).map((value) => value / scale);
    for (let i = 0; i < n; i++) {
      if (i === column) continue;
      const factor = defined(defined(work[i])[column]);
      work[i] = defined(work[i]).map((value, j) => value - factor * defined(defined(work[column])[j]));
    }
  }
  return work.map((row) => row.slice(n));
}

/**
 * The forward-backward algorithm on one sequence, in buffers of its exact length: the oracle of {@link smoothSequence},
 * which a fit runs on the first steps of buffers made for its longest sequence.
 */
function forwardBackward(sequence: Pick<HiddenMarkovBuffers, "stateCount" | "initial" | "transitions" | "emissions">) {
  const { stateCount: k, emissions } = sequence;
  const steps = emissions.length / k;
  const pairs = k * k * Math.max(steps - 1, 0);
  const buffers: HiddenMarkovBuffers = {
    ...sequence,
    predictions: new Float64Array(steps * k),
    filtered: new Float64Array(steps * k),
    posteriors: new Float64Array(steps * k),
    transitionPosteriors: new Float64Array(pairs),
  };
  const logLikelihood = smoothSequence(buffers, steps);
  const { posteriors, transitionPosteriors, predictions } = buffers;
  return { logLikelihood, posteriors, transitionPosteriors, predictions };
}

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

  it("gives the same results in buffers that a longer sequence used before", () => {
    // A state predicted at 0, as with a prior and a learning rate of 0, has pairs of 0 where the longer sequence left
    // others.
    const never = {
      stateCount: 2,
      initial: Float64Array.of(1, 0),
      transitions: Float64Array.of(1, 0, 0.3, 0.7, 1, 0, 0.3, 0.7),
      emissions: Float64Array.of(0.8, 0.1, 0.2, 0.9, 0.8, 0.1),
    };
    const sequences = [...hiddenMarkov.sequences, never];
    const longest = Math.max(...sequences.map((sequence) => sequence.emissions.length / sequence.stateCount));
    const buffers = createHiddenMarkovBuffers(2, longest + 3);
    const used = () => {
      const steps = longest + 3;
      buffers.initial.set([0.4, 0.6]);
      buffers.transitions.set(Array.from({ length: 4 * (steps - 1) }, (_, i) => [0.6, 0.4, 0.1, 0.9][i % 4] ?? 0));
      buffers.emissions.set(Array.from({ length: 2 * steps }, (_, i) => (i % 3 === 0 ? 0.25 : 0.75)));
      smoothSequence(buffers, steps);
    };
    for (const sequence of sequences.filter(({ stateCount }) => stateCount === 2)) {
      const expected = forwardBackward({
        stateCount: 2,
        initial: Float64Array.from(sequence.initial),
        transitions: Float64Array.from(sequence.transitions),
        emissions: Float64Array.from(sequence.emissions),
      });
      used();
      const steps = sequence.emissions.length / 2;
      buffers.initial.set(sequence.initial);
      buffers.transitions.set(sequence.transitions);
      buffers.emissions.set(sequence.emissions);
      expect(smoothSequence(buffers, steps)).toBe(expected.logLikelihood);
      expect(buffers.posteriors.slice(0, 2 * steps)).toEqual(expected.posteriors);
      expect(buffers.transitionPosteriors.slice(0, 4 * (steps - 1))).toEqual(expected.transitionPosteriors);
      expect(buffers.predictions.slice(0, 2 * steps)).toEqual(expected.predictions);
    }
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
        const order = Uint32Array.from(all, (_, at) => at);
        shuffle(order, createRandom(seed));
        const shuffled = Array.from(order, (at) => defined(all[at]));
        expect(rank(columns(...shuffled))).toBe(chosen.length);
      }),
    );
  });
});
