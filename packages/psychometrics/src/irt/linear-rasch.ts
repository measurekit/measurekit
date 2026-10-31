import { iterateToConvergence, largestChange } from "@measurekit/core";
import { numericalHessian } from "../math/hessian.js";
import { invert } from "../math/linear-algebra.js";
import { type HessianFunction, newton } from "../math/optimize.js";
import type { Quadrature } from "../math/quadrature.js";
import { type ExpectationItem, type ExpectationStep, expectationStep } from "./expectation.js";
import type { Population } from "./latent.js";
import {
  type LocationRow,
  maximization,
  type RaschLayout,
  writeSteps,
  writeThresholds,
} from "./linear-rasch-maximization.js";
import { partialCredit, stepStarts } from "./partial-credit.js";
import type { CompressedResponses } from "./responses.js";

export type { LocationRow } from "./linear-rasch-maximization.js";

/**
 * A Rasch model for items of any number of categories whose locations are linear in parameters: item i gives a
 * person at theta category k with probability P_k ∝ exp(Σ_(j ≤ k) (θ - location_i - threshold_(s(i), j))), the
 * location a sum of coefficients times parameters, and the thresholds of each set s summing to 0.
 */
export interface LinearRaschModel {
  /** The responses: each person's answered items and categories. */
  readonly data: CompressedResponses;
  /** The location parameters of each item. */
  readonly locations: readonly LocationRow[];
  /** The number of location parameters. */
  readonly locationCount: number;
  /** The threshold set of each item. */
  readonly setOf: Int32Array;
  /** The number of categories of each set, at least 2; every item of a set has them. */
  readonly categoryCounts: Int32Array;
  /**
   * The mean of theta, held at this value rather than estimated, for locations that identify the scale themselves. It
   * has no standard error.
   *
   * @defaultValue estimated
   */
  readonly fixedMean?: number;
}

/** How a linear Rasch model is estimated. */
export interface LinearRaschOptions {
  /** The rule over a standard normal z: theta is μ + σ z. */
  readonly rule: Quadrature;
  /** Estimation has converged when no parameter changes by more than this in an iteration. */
  readonly tolerance: number;
  /** The most EM iterations. */
  readonly maxIterations: number;
  /** Whether to compute the covariance matrix of the parameters. */
  readonly standardErrors: boolean;
}

/** A linear Rasch model estimated by marginal maximum likelihood. */
export interface LinearRaschFit {
  /** The parameter vector: location parameters, free thresholds of each set, mean of theta, log of its SD. */
  readonly parameters: Float64Array;
  /** Where the parameters are in the vector. */
  readonly layout: RaschLayout;
  /** The thresholds of each set, summing to 0. */
  readonly thresholds: readonly Float64Array[];
  /** The mean of theta. */
  readonly mean: number;
  /** The standard deviation of theta. */
  readonly sd: number;
  /** The marginal log-likelihood at the estimates. */
  readonly logLikelihood: number;
  /** Whether the parameters changed by at most the tolerance within the maximum number of iterations. */
  readonly converged: boolean;
  /** EM iterations run. */
  readonly iterations: number;
  /** The covariance matrix of the parameter vector by rows, or `null` without standard errors or if it is singular. */
  readonly covariance: Float64Array | null;
}

/** The M-step stops when the gradient of the expected log-likelihood is this small. */
const M_STEP_TOLERANCE = 1e-10;
/**
 * The M-step takes a Newton step in full, and stops, when it moves no parameter by more than this. Newton's method
 * converges quadratically, so a step s leaves an error of the order of s², 1e-12 here, six orders below the EM tolerance
 * of 1e-4 and below the 1e-10 of the strictest calibrations. A smaller step is also one that the objective cannot
 * resolve: it changes by about λ s² / 2 with λ the curvature along the step, which is below the rounding of a value of
 * 1e4, 2e-12, for s under 2e-6 where λ is 1.
 */
const M_STEP_STEP_TOLERANCE = 1e-6;
const M_STEP_ITERATIONS = 50;

/**
 * Estimates a linear Rasch model by the Bock-Aitkin EM algorithm with a normal distribution of theta whose mean and
 * standard deviation are estimated, on the scale that the sum-to-zero constraints of the thresholds and the caller's
 * location parameters fix.
 *
 * @remarks
 * The E-step is that of `calibrate`, over partial credit items with a = 1 whose nodes are μ + σ z_q. The M-step is
 * Newton's method on all parameters together, with the exact gradient and Hessian of the expected complete-data
 * log-likelihood and a line search. Standard errors come from the Hessian of the marginal log-likelihood, by central
 * differences of its exact gradient, which is Fisher's identity: the gradient of the expected log-likelihood at the
 * parameters of the E-step.
 *
 * @see {@link https://doi.org/10.1007/BF02293801 | Bock and Aitkin (1981)}
 */
export function fitLinearRasch(model: LinearRaschModel, options: LinearRaschOptions): LinearRaschFit {
  const layout = layoutOf(model);
  const { rule, tolerance, maxIterations } = options;
  const nodes = Float64Array.from(rule.nodes);
  const { items, steps } = expectationItems(model, nodes.length);
  const step = expectationStep(model.data, items, nodes, standardNormalPopulation(rule, model.data), null, null);
  const objective = maximization(
    layout,
    nodes,
    items.map((item) => item.counts),
  );
  const held = model.fixedMean === undefined ? null : { index: layout.meanIndex, value: model.fixedMean };
  let parameters = startingValues(model, layout);
  if (held !== null) parameters[held.index] = held.value;
  const assign = assigner(layout, nodes, items, steps);
  const { converged, iterations } = iterateToConvergence(
    () => {
      assign(parameters);
      step.run();
      const next = maximize(objective, parameters, held);
      const change = largestChange(parameters, next);
      parameters = next;
      return change;
    },
    { tolerance, maxIterations },
  );
  assign(parameters);
  const logLikelihood = step.run();
  const covariance = options.standardErrors ? covarianceOf(layout, parameters, assign, step, objective, held) : null;
  return {
    parameters,
    layout,
    thresholds: Array.from(layout.thresholdCounts, (count, set) => {
      const out = new Float64Array(count);
      writeThresholds(layout, parameters, set, out);
      return out;
    }),
    mean: parameters[layout.meanIndex] ?? 0,
    sd: Math.exp(parameters[layout.logSdIndex] ?? 0),
    logLikelihood,
    converged,
    iterations,
    covariance,
  };
}

/** A parameter held at a value: its index in the parameter vector and the value. */
interface Held {
  readonly index: number;
  readonly value: number;
}

/** Returns the parameters that minimize the objective from `start` by Newton's method, a held parameter unchanged. */
export function maximize(objective: HessianFunction, start: Float64Array, held: Held | null): Float64Array {
  if (held === null) return descend(objective, start);
  const reduced = descend(withoutParameter(objective, start.length, held), removeAt(start, held.index));
  return insertAt(reduced, held.index, held.value);
}

/**
 * Takes Newton steps with a line search from `start` until the gradient or a full step is within its tolerance, or a
 * step does not lower the objective: where rounding of its value hides the decrease, the line search would otherwise
 * halve the step through every evaluation of every iteration without moving the point.
 */
function descend(objective: HessianFunction, start: Float64Array): Float64Array {
  const options = { tolerance: M_STEP_TOLERANCE, stepTolerance: M_STEP_STEP_TOLERANCE, maxIterations: 1 };
  const remembered = remembering(objective);
  let point = start;
  let value = remembered(start, new Float64Array(start.length), new Float64Array(start.length ** 2));
  for (let step = 0; step < M_STEP_ITERATIONS; step++) {
    const next = newton(remembered, point, options);
    if (next.converged) return next.x;
    if (!(next.value < value)) return next.value <= value ? next.x : point;
    point = next.x;
    value = next.value;
  }
  return point;
}

/**
 * Returns the objective, answering a call at the point of the last one from memory: one Newton step starts where the
 * line search of the one before it ended.
 */
function remembering(objective: HessianFunction): HessianFunction {
  let point = new Float64Array(0);
  let value = 0;
  let known = new Float64Array(0);
  let curvature = new Float64Array(0);
  return (x, gradient, hessian) => {
    if (x.length === point.length && x.every((entry, j) => entry === point[j])) {
      gradient.set(known);
      hessian.set(curvature);
      return value;
    }
    value = objective(x, gradient, hessian);
    point = Float64Array.from(x);
    known = Float64Array.from(gradient);
    curvature = Float64Array.from(hessian);
    return value;
  };
}

/** Returns the objective as a function of the parameters other than the held one, which keeps its value. */
function withoutParameter(objective: HessianFunction, size: number, held: Held): HessianFunction {
  const full = new Float64Array(size);
  full[held.index] = held.value;
  const fullGradient = new Float64Array(size);
  const fullHessian = new Float64Array(size * size);
  const kept = Array.from({ length: size }, (_, j) => j).filter((j) => j !== held.index);
  return (x, gradient, hessian) => {
    kept.forEach((j, a) => {
      full[j] = x[a] ?? 0;
    });
    const value = objective(full, fullGradient, fullHessian);
    kept.forEach((j, a) => {
      gradient[a] = fullGradient[j] ?? 0;
      kept.forEach((k, b) => {
        hessian[a * kept.length + b] = fullHessian[j * size + k] ?? 0;
      });
    });
    return value;
  };
}

function removeAt(values: ArrayLike<number>, index: number): Float64Array {
  return Float64Array.from(Array.from(values).filter((_, j) => j !== index));
}

function insertAt(values: ArrayLike<number>, index: number, value: number): Float64Array {
  const list = Array.from(values);
  list.splice(index, 0, value);
  return Float64Array.from(list);
}

function layoutOf(model: LinearRaschModel): RaschLayout {
  const thresholdCounts = Int32Array.from(model.categoryCounts, (categories) => categories - 1);
  const thresholdStarts = new Int32Array(thresholdCounts.length);
  let next = model.locationCount;
  thresholdCounts.forEach((count, set) => {
    thresholdStarts[set] = next;
    next += count - 1;
  });
  return {
    locations: model.locations,
    setOf: model.setOf,
    thresholdCounts,
    thresholdStarts,
    meanIndex: next,
    logSdIndex: next + 1,
    size: next + 2,
  };
}

/**
 * Returns the items of the E-step: partial credit items with a = 1, whose steps `assigner` sets in the arrays returned
 * with them, and whose nodes it sets as well.
 */
function expectationItems(model: LinearRaschModel, nodeCount: number): { items: ExpectationItem[]; steps: number[][] } {
  const steps = model.locations.map((_, item) =>
    new Array<number>((model.categoryCounts[model.setOf[item] ?? 0] ?? 2) - 1).fill(0),
  );
  const items = steps.map((b) => ({
    model: partialCredit,
    categoryCount: b.length + 1,
    nodes: new Float64Array(nodeCount),
    parameters: { a: 1, b },
    counts: new Float64Array(nodeCount * (b.length + 1)),
  }));
  return { items, steps };
}

/** Returns the population for the E-step: one group with the weights of the rule over z. */
function standardNormalPopulation(rule: Quadrature, data: CompressedResponses): Population {
  const persons = data.offsets.length - 1;
  return {
    groups: [
      {
        name: null,
        distribution: "standard-normal",
        logWeights: Float64Array.from(rule.weights, Math.log),
        counts: new Float64Array(rule.nodes.length),
        meanSlot: -1,
        sdSlot: -1,
      },
    ],
    groupOf: new Int32Array(persons),
  };
}

/** Returns a function that sets the items' steps and nodes to a parameter vector's. */
function assigner(
  layout: RaschLayout,
  z: Float64Array,
  items: readonly ExpectationItem[],
  steps: readonly number[][],
): (parameters: Float64Array) => void {
  const written = new Float64Array(layout.thresholdCounts.reduce((most, count) => Math.max(most, count), 1));
  return (parameters) => {
    const mean = parameters[layout.meanIndex] ?? 0;
    const sd = Math.exp(parameters[layout.logSdIndex] ?? 0);
    items.forEach((item, i) => {
      writeSteps(layout, parameters, i, written);
      const b = steps[i] ?? [];
      for (let j = 0; j < b.length; j++) b[j] = written[j] ?? 0;
      for (let q = 0; q < z.length; q++) item.nodes[q] = mean + sd * (z[q] ?? 0);
    });
  };
}

/**
 * Returns starting values: no location effects, thresholds from the shares of persons in the categories of each set,
 * and the mean that puts the average step at the middle of the distribution, with SD 1.
 */
function startingValues(model: LinearRaschModel, layout: RaschLayout): Float64Array {
  const { data, setOf, categoryCounts } = model;
  const shares = Array.from(categoryCounts, (categories) => new Float64Array(categories));
  data.codes.forEach((code, r) => {
    const counts = shares[setOf[data.items[r] ?? 0] ?? 0];
    if (counts !== undefined) counts[code] = (counts[code] ?? 0) + 1;
  });
  const parameters = new Float64Array(layout.size);
  let weightedMean = 0;
  let responses = 0;
  shares.forEach((counts, set) => {
    const total = counts.reduce((sum, count) => sum + count, 0);
    const steps = stepStarts(counts.map((count) => count / total));
    const average = steps.reduce((sum, step) => sum + step, 0) / steps.length;
    steps.slice(0, -1).forEach((step, j) => {
      parameters[(layout.thresholdStarts[set] ?? 0) + j] = step - average;
    });
    weightedMean -= total * average;
    responses += total;
  });
  parameters[layout.meanIndex] = weightedMean / responses;
  return parameters;
}

/**
 * Returns the covariance matrix of the parameters, or `null` when the information is singular. A held parameter has no
 * variance or covariance: its row and column are 0.
 */
function covarianceOf(
  layout: RaschLayout,
  parameters: Float64Array,
  assign: (parameters: Float64Array) => void,
  step: ExpectationStep,
  objective: ReturnType<typeof maximization>,
  held: Held | null,
): Float64Array | null {
  const { size } = layout;
  const kept = Array.from({ length: size }, (_, j) => j).filter((j) => j !== held?.index);
  const count = kept.length;
  const scratch = new Float64Array(size * size);
  const fullGradient = new Float64Array(size);
  const full = Float64Array.from(parameters);
  const information = numericalHessian(
    (point, gradient) => {
      kept.forEach((j, a) => {
        full[j] = point[a] ?? 0;
      });
      assign(full);
      step.run();
      const value = objective(full, fullGradient, scratch);
      kept.forEach((j, a) => {
        gradient[a] = fullGradient[j] ?? 0;
      });
      return value;
    },
    Float64Array.from(kept, (j) => parameters[j] ?? 0),
  );
  assign(parameters);
  const inverse = invert(information, count);
  if (inverse === null) return null;
  for (let a = 0; a < count; a++) if (!((inverse[a * count + a] ?? 0) > 0)) return null;
  const covariance = new Float64Array(size * size);
  kept.forEach((j, a) => {
    kept.forEach((k, b) => {
      covariance[j * size + k] = inverse[a * count + b] ?? 0;
    });
  });
  return covariance;
}
