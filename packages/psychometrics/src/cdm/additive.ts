import { backtrack, dot, logistic, maxAbs } from "@measurekit/core";
import { invert, invertPositiveDefinite } from "../math/linear-algebra.js";
import { profileIndices, profilesOf } from "./q-matrix.js";

/**
 * The link g of an additive model: a latent group's success probability is g⁻¹(δ₀ + Σ δ_k α_k) over the item's
 * attributes k. The identity gives the A-CDM, the logit the LLM and the log the R-RUM.
 */
export type CdmLink = "identity" | "logit" | "log";

/**
 * A link's functions. `slope` and `curvature` are the first and second derivatives, with respect to the linear
 * predictor, of a latent group's negative log-likelihood -R ln P - (N - R) ln(1 - P), given its expected correct
 * responses R, expected persons N and probability P. The negative log-likelihood is convex in the predictor under all
 * three links, so the curvature is not negative: it is the Bernoulli log-likelihood of a linear predictor under the
 * identity, and of the canonical predictor under the logit, and -ln(1 - e^η) is convex under the log.
 */
interface LinkFunctions {
  readonly link: (probability: number) => number;
  readonly inverse: (predictor: number) => number;
  readonly slope: (correct: number, persons: number, probability: number) => number;
  readonly curvature: (correct: number, persons: number, probability: number) => number;
}

const LINKS: Readonly<Record<CdmLink, LinkFunctions>> = {
  identity: {
    link: (p) => p,
    inverse: (eta) => eta,
    slope: (r, n, p) => (n - r) / (1 - p) - r / p,
    curvature: (r, n, p) => r / (p * p) + (n - r) / ((1 - p) * (1 - p)),
  },
  logit: {
    link: (p) => Math.log(p / (1 - p)),
    inverse: logistic,
    slope: (r, n, p) => n * p - r,
    curvature: (_r, n, p) => n * p * (1 - p),
  },
  log: {
    link: Math.log,
    inverse: Math.exp,
    slope: (r, n, p) => ((n - r) * p) / (1 - p) - r,
    curvature: (r, n, p) => ((n - r) * p) / ((1 - p) * (1 - p)),
  },
};

/** The Hessian's ridge, relative to one plus its largest diagonal entry, so that a group without curvature still has a step. */
const RIDGE = 1e-10;
/** The decrease the Newton step predicts, relative to one plus the objective, below which the maximum on the face is reached. */
const DECREMENT_TOLERANCE = 1e-15;
/** A multiplier below this, relative to one plus the largest gradient entry, is negative and releases its constraint. */
const MULTIPLIER_TOLERANCE = 1e-10;
/** The distance past a bound that a step may take a predictor, which is rounding and not a violation. */
const BOUND_SLACK = 1e-12;
/** The share of a vector that must lie outside the span of others for it to be independent of them. */
const INDEPENDENCE = 1e-6;
/** The decrease the line search accepts as rounding, relative to the objective. */
const ROUNDING = 1e-14;
const MAX_STEPS = 100;
/** The slack of a predictor outside its bounds that still counts as inside them: rounding of the last step. */
const FEASIBILITY_TOLERANCE = 1e-9;

/** An item's problem and its scratch space, which the steps of a maximization reuse. */
interface Problem {
  readonly functions: LinkFunctions;
  /** The probability bounds of the latent groups. */
  readonly bounds: readonly [number, number];
  /** The bounds on the link scale. */
  readonly lower: number;
  readonly upper: number;
  /** The rows of the design matrix, `[1, profile]` for each latent group, stored by rows. */
  readonly design: Float64Array;
  /** The number of coefficients, one more than the item's attributes. */
  readonly size: number;
  readonly groups: number;
  /** The predictors of the point being stepped from, and of the point tried. */
  readonly predictors: Float64Array;
  readonly trialPredictors: Float64Array;
  /** The gradient and Hessian of the negative log-likelihood at the point being stepped from. */
  readonly gradient: Float64Array;
  readonly hessian: Float64Array;
  /** The Newton step on the working face, and the change of each predictor it makes. */
  readonly step: Float64Array;
  readonly direction: Float64Array;
  readonly trial: Float64Array;
}

/** The maximization of an additive item's log-likelihood in its latent groups, kept between EM iterations. */
export interface AdditiveMaximizer {
  /** The coefficients δ on the link scale: the starting ones until `maximize` runs, the maximum after each call. */
  readonly coefficients: Float64Array;
  /**
   * Returns the success probability of each latent group at the coefficients that maximize the log-likelihood of the
   * expected correct responses and persons per group, found from the previous call's coefficients. The array is
   * overwritten by the next call.
   */
  maximize(correct: Float64Array, persons: Float64Array): Float64Array;
}

/**
 * Builds the constrained M-step of an additive model for an item with `attributeCount` attributes, as GDINA's:
 * maximizes Σ_l [R_l ln P_l + (N_l - R_l) ln(1 - P_l)] over δ, where P_l = g⁻¹(m_l δ), m_l = [1, profile of group l],
 * subject to `bounds[0] <= P_l <= bounds[1]`. Because g is monotone, the bounds are linear in δ, and the maximum is
 * unique.
 *
 * @remarks
 * Newton's method with the bounds that are met as equalities (the working set): each step is the maximum of the
 * quadratic approximation on the face of those bounds, shortened to the first other bound it reaches and by a
 * backtracking line search. When the step vanishes, a bound whose multiplier is negative is released; otherwise the
 * point is the maximum. The first call starts from the least squares fit of `start` on the link scale, and later calls
 * from the previous maximum, so that EM takes few steps. The Hessian's diagonal is ridged by 1e-10 of its largest entry,
 * which leaves a group without curvature, such as one of correct responses only under the log link, with a step.
 *
 * @param start - The success probability of each latent group to start from, in the order of the profiles.
 * @see {@link https://doi.org/10.1007/978-0-387-40065-5 | Nocedal and Wright (2006)}
 */
export function additiveMaximizer(
  link: CdmLink,
  attributeCount: number,
  bounds: readonly [number, number],
  start: ArrayLike<number>,
): AdditiveMaximizer {
  const functions = LINKS[link];
  const profiles = profilesOf(attributeCount);
  const size = attributeCount + 1;
  const groups = profiles.length;
  const problem: Problem = {
    functions,
    bounds,
    lower: functions.link(bounds[0]),
    upper: functions.link(bounds[1]),
    design: Float64Array.from(profiles.flatMap((profile) => [1, ...profile])),
    size,
    groups,
    predictors: new Float64Array(groups),
    trialPredictors: new Float64Array(groups),
    gradient: new Float64Array(size),
    hessian: new Float64Array(size * size),
    step: new Float64Array(size),
    direction: new Float64Array(groups),
    trial: new Float64Array(size),
  };
  const coefficients = leastSquares(problem, Float64Array.from(start, functions.link)) ?? constantStart(problem);
  const probabilities = new Float64Array(groups);
  return {
    coefficients,
    maximize(correct, persons) {
      if (!isFeasible(problem, coefficients)) coefficients.set(constantStart(problem));
      climb(problem, correct, persons, coefficients);
      predict(problem, coefficients, problem.predictors);
      for (let l = 0; l < groups; l++) probabilities[l] = probability(problem, problem.predictors[l] ?? 0);
      return probabilities;
    },
  };
}

/** Sets `out[l]` to the predictor of group l, the dot product of its design row and `coefficients`. */
function predict(problem: Problem, coefficients: Float64Array, out: Float64Array): void {
  const { design, size } = problem;
  for (let l = 0; l < out.length; l++) {
    let sum = 0;
    for (let i = 0; i < size; i++) sum += (design[l * size + i] ?? 0) * (coefficients[i] ?? 0);
    out[l] = sum;
  }
}

/** Returns a predictor's success probability, within the bounds. */
function probability(problem: Problem, predictor: number): number {
  return Math.min(Math.max(problem.functions.inverse(predictor), problem.bounds[0]), problem.bounds[1]);
}

/** Returns the coefficients of the constant predictor at probability 0.5, which is inside any bounds. */
function constantStart(problem: Problem): Float64Array {
  const coefficients = new Float64Array(problem.size);
  coefficients[0] = problem.functions.link(0.5);
  return coefficients;
}

/** Whether every predictor is within the bounds, up to the rounding of the steps that met them. */
function isFeasible(problem: Problem, coefficients: Float64Array): boolean {
  predict(problem, coefficients, problem.trialPredictors);
  return problem.trialPredictors.every(
    (predictor) =>
      predictor >= problem.lower - FEASIBILITY_TOLERANCE && predictor <= problem.upper + FEASIBILITY_TOLERANCE,
  );
}

/** Returns the least squares fit of the design to `targets`, or `null` when its normal equations are singular. */
function leastSquares(problem: Problem, targets: Float64Array): Float64Array | null {
  const { design, size, groups } = problem;
  const normal = new Float64Array(size * size);
  const moment = new Float64Array(size);
  for (let l = 0; l < groups; l++) {
    for (let i = 0; i < size; i++) {
      const row = design[l * size + i] ?? 0;
      moment[i] = (moment[i] ?? 0) + row * (targets[l] ?? 0);
      for (let j = 0; j < size; j++)
        normal[i * size + j] = (normal[i * size + j] ?? 0) + row * (design[l * size + j] ?? 0);
    }
  }
  const inverse = invert(normal, size);
  if (inverse === null) return null;
  return Float64Array.from({ length: size }, (_, i) => {
    let sum = 0;
    for (let j = 0; j < size; j++) sum += (inverse[i * size + j] ?? 0) * (moment[j] ?? 0);
    return sum;
  });
}

/** Returns the negative log-likelihood of the expected counts at `coefficients`. */
function objective(problem: Problem, correct: Float64Array, persons: Float64Array, coefficients: Float64Array): number {
  predict(problem, coefficients, problem.trialPredictors);
  let value = 0;
  for (let l = 0; l < problem.groups; l++) {
    const p = probability(problem, problem.trialPredictors[l] ?? 0);
    const r = correct[l] ?? 0;
    value -= r * Math.log(p) + ((persons[l] ?? 0) - r) * Math.log1p(-p);
  }
  return value;
}

/** Adds a group's slope to the gradient and its weight to the Hessian, over the group's design row. */
function accumulate(problem: Problem, group: number, slope: number, weight: number): void {
  const { design, size, gradient, hessian } = problem;
  for (let i = 0; i < size; i++) {
    const row = design[group * size + i] ?? 0;
    gradient[i] = (gradient[i] ?? 0) + slope * row;
    for (let j = 0; j < size; j++) {
      hessian[i * size + j] = (hessian[i * size + j] ?? 0) + weight * row * (design[group * size + j] ?? 0);
    }
  }
}

/** Sets the gradient and the ridged Hessian of the negative log-likelihood at the predictors in the problem. */
function derivatives(problem: Problem, correct: Float64Array, persons: Float64Array): void {
  const { size, groups, gradient, hessian, functions } = problem;
  gradient.fill(0);
  hessian.fill(0);
  for (let l = 0; l < groups; l++) {
    const p = probability(problem, problem.predictors[l] ?? 0);
    const r = correct[l] ?? 0;
    const n = persons[l] ?? 0;
    accumulate(problem, l, functions.slope(r, n, p), functions.curvature(r, n, p));
  }
  let largest = 0;
  for (let i = 0; i < size; i++) largest = Math.max(largest, hessian[i * size + i] ?? 0);
  for (let i = 0; i < size; i++) hessian[i * size + i] = (hessian[i * size + i] ?? 0) + RIDGE * (1 + largest);
}

/**
 * A bound is numbered 2l for the lower bound of group l and 2l + 1 for its upper bound. As a constraint c(δ) >= 0 its
 * normal is the group's design row for the lower bound and the row negated for the upper bound.
 */
function constraintNormal(problem: Problem, bound: number, out: Float64Array): void {
  const sign = bound % 2 === 0 ? 1 : -1;
  const start = (bound >> 1) * problem.size;
  for (let i = 0; i < problem.size; i++) out[i] = sign * (problem.design[start + i] ?? 0);
}

/** Removes from `vector` its components along the orthonormal `basis`, twice for accuracy. */
function projectOut(vector: Float64Array, basis: readonly Float64Array[]): void {
  for (let pass = 0; pass < 2; pass++) {
    for (const direction of basis) {
      const projection = dot(direction, vector);
      for (let i = 0; i < vector.length; i++) vector[i] = (vector[i] ?? 0) - projection * (direction[i] ?? 0);
    }
  }
}

/** Appends the unit vector along the part of `vector` outside the span of `basis`, if there is one, and says whether it did. */
function extend(basis: Float64Array[], vector: Float64Array): boolean {
  const length = Math.sqrt(dot(vector, vector));
  projectOut(vector, basis);
  const remaining = Math.sqrt(dot(vector, vector));
  if (!(remaining > INDEPENDENCE * length)) return false;
  basis.push(vector.map((value) => value / remaining));
  return true;
}

/**
 * Returns an orthonormal basis of the vectors orthogonal to `normals`, or `null` when the normals are dependent. Each
 * basis vector is the part of the unit vector with the most outside the span so far, so that steps along it stay
 * exactly within the face of the working bounds.
 */
function nullSpace(normals: readonly Float64Array[], size: number): Float64Array[] | null {
  const span: Float64Array[] = [];
  if (!normals.every((normal) => extend(span, Float64Array.from(normal)))) return null;
  const complement: Float64Array[] = [];
  while (span.length < size) {
    let best = new Float64Array(size);
    let bestLength = -1;
    for (let i = 0; i < size; i++) {
      const candidate = new Float64Array(size);
      candidate[i] = 1;
      projectOut(candidate, span);
      const length = dot(candidate, candidate);
      if (length > bestLength) {
        best = candidate;
        bestLength = length;
      }
    }
    const direction = best.map((value) => value / Math.sqrt(bestLength));
    span.push(direction);
    complement.push(direction);
  }
  return complement;
}

/** Returns the product of the symmetric `size` by `size` matrix `a` and a vector. */
function product(a: Float64Array, size: number, vector: Float64Array): Float64Array {
  return Float64Array.from({ length: size }, (_, i) => dot(a.subarray(i * size, (i + 1) * size), vector));
}

/**
 * Sets the Newton step on the face of the working bounds and returns the multipliers of those bounds, or `null` when
 * the bounds are dependent or the Hessian on the face is singular. With Z an orthonormal basis of the face, the step
 * is -Z (Zᵀ H Z)⁻¹ Zᵀ g, and with A the normals of the bounds the multipliers are (A Aᵀ)⁻¹ A g, which satisfy
 * g = Aᵀ λ where the step vanishes. The face's own basis keeps A s = 0 to rounding, however ill-conditioned H.
 */
function newtonStep(problem: Problem, working: readonly number[]): Float64Array | null {
  const { size, gradient, hessian, step } = problem;
  const normals = working.map((bound) => {
    const normal = new Float64Array(size);
    constraintNormal(problem, bound, normal);
    return normal;
  });
  const face = nullSpace(normals, size);
  if (face === null) return null;
  const curvature = face.map((direction) => product(hessian, size, direction));
  const reduced = new Float64Array(face.length ** 2);
  face.forEach((direction, a) => {
    curvature.forEach((other, b) => {
      reduced[a * face.length + b] = dot(direction, other);
    });
  });
  const inverse = invertPositiveDefinite(reduced, face.length);
  if (inverse === null && face.length > 0) return null;
  const slope = Float64Array.from(face, (direction) => dot(direction, gradient));
  const along = inverse === null ? slope : product(inverse, face.length, slope);
  step.fill(0);
  face.forEach((direction, a) => {
    for (let i = 0; i < size; i++) step[i] = (step[i] ?? 0) - (along[a] ?? 0) * (direction[i] ?? 0);
  });
  return multipliers(normals, gradient);
}

/** Returns λ = (A Aᵀ)⁻¹ A g for the rows `normals` of A, or `null` when A Aᵀ is singular. */
function multipliers(normals: readonly Float64Array[], gradient: Float64Array): Float64Array | null {
  const count = normals.length;
  const gram = new Float64Array(count * count);
  normals.forEach((normal, a) => {
    normals.forEach((other, b) => {
      gram[a * count + b] = dot(normal, other);
    });
  });
  const inverse = count === 0 ? new Float64Array(0) : invertPositiveDefinite(gram, count);
  if (inverse === null) return null;
  return product(
    inverse,
    count,
    Float64Array.from(normals, (normal) => dot(normal, gradient)),
  );
}

/** The first bound a step reaches: its number, and the fraction of the step at which it is reached. */
interface Blocking {
  readonly bound: number;
  readonly fraction: number;
}

/**
 * Returns the first bound outside the working set that the step moves a predictor to, or a fraction of `Infinity` for
 * none. A step may go `BOUND_SLACK` past a bound, so that the rounding of a predictor on a bound that is not in the
 * working set, because it depends on the bounds that are, does not stop the step.
 */
function firstBlocking(problem: Problem, working: readonly number[]): Blocking {
  predict(problem, problem.step, problem.direction);
  let blocking: Blocking = { bound: -1, fraction: Number.POSITIVE_INFINITY };
  for (let l = 0; l < problem.groups; l++) {
    const rate = problem.direction[l] ?? 0;
    if (rate === 0) continue;
    const predictor = problem.predictors[l] ?? 0;
    const bound = rate < 0 ? 2 * l : 2 * l + 1;
    const distance = rate < 0 ? predictor - problem.lower : problem.upper - predictor;
    const fraction = (Math.max(distance, 0) + BOUND_SLACK) / Math.abs(rate);
    if (fraction < blocking.fraction && !working.includes(bound)) blocking = { bound, fraction };
  }
  return blocking;
}

/** Returns the index of the most negative multiplier, or -1 when none is negative beyond rounding. */
function mostNegative(multipliers: Float64Array, gradient: Float64Array): number {
  const tolerance = MULTIPLIER_TOLERANCE * (1 + maxAbs(gradient));
  let found = -1;
  let least = -tolerance;
  multipliers.forEach((multiplier, a) => {
    if (multiplier < least) {
      least = multiplier;
      found = a;
    }
  });
  return found;
}

/** Moves `coefficients` along the step by a fraction. */
function advance(problem: Problem, coefficients: Float64Array, fraction: number): void {
  for (let i = 0; i < problem.size; i++) coefficients[i] = (coefficients[i] ?? 0) + fraction * (problem.step[i] ?? 0);
}

/**
 * Moves `coefficients` along the step by the largest of `limit`, `limit / 2`, ... that decreases the objective
 * enough, and returns the fraction, or 0 when none does. `slope` is the objective's derivative along the step.
 */
function lineSearch(
  problem: Problem,
  correct: Float64Array,
  persons: Float64Array,
  coefficients: Float64Array,
  value: number,
  limit: number,
): number {
  const { trial, step } = problem;
  const fraction = backtrack(
    (length) => {
      for (let i = 0; i < problem.size; i++) trial[i] = (coefficients[i] ?? 0) + length * (step[i] ?? 0);
      return objective(problem, correct, persons, trial);
    },
    value,
    dot(problem.gradient, step),
    { slack: ROUNDING * Math.abs(value), start: limit },
  );
  if (fraction === null) return 0;
  coefficients.set(trial);
  return fraction;
}

/**
 * Takes the next step of the maximization from `coefficients` and says whether to go on: whether the maximum is reached
 * or a step makes no progress ends it. A negative multiplier at the maximum on the face releases its bound instead.
 */
function advanceOnce(
  problem: Problem,
  correct: Float64Array,
  persons: Float64Array,
  coefficients: Float64Array,
  working: number[],
): boolean {
  predict(problem, coefficients, problem.predictors);
  derivatives(problem, correct, persons);
  const multipliers = newtonStep(problem, working);
  if (multipliers === null) return false;
  const value = objective(problem, correct, persons, coefficients);
  const settled = -dot(problem.gradient, problem.step) <= DECREMENT_TOLERANCE * (1 + Math.abs(value));
  const release = settled ? mostNegative(multipliers, problem.gradient) : -1;
  if (release >= 0) {
    working.splice(release, 1);
    return true;
  }
  const blocking = firstBlocking(problem, working);
  const limit = Math.min(1, blocking.fraction);
  if (settled) {
    advance(problem, coefficients, limit);
    return false;
  }
  const fraction = limit === 0 ? 0 : lineSearch(problem, correct, persons, coefficients, value, limit);
  if (fraction === blocking.fraction) working.push(blocking.bound);
  return fraction > 0 || limit === 0;
}

/** Maximizes the log-likelihood of the expected counts, starting from and leaving the result in `coefficients`. */
function climb(problem: Problem, correct: Float64Array, persons: Float64Array, coefficients: Float64Array): void {
  const working: number[] = [];
  for (let iteration = 0; iteration < MAX_STEPS; iteration++) {
    if (!advanceOnce(problem, correct, persons, coefficients, working)) return;
  }
}

/**
 * Returns the coefficients δ of a saturated model on the identity scale from its success probabilities per latent
 * group, in the order of the profiles: P = M δ over the intercept and every effect of the item's attributes, so that
 * δ_S is the sum over the subsets T of S of (-1)^(|S| - |T|) P_T. The order is the profiles', each coefficient being
 * the effect of the attributes its profile masters, as GDINA's `coef(what = "delta")` of the G-DINA model.
 */
export function saturatedCoefficients(probabilities: readonly number[], attributeCount: number): number[] {
  const indices = profileIndices(attributeCount);
  const effects = Float64Array.from(indices, (index) => probabilities[index] ?? 0);
  for (let bit = 1; bit < effects.length; bit *= 2) {
    for (let pattern = 0; pattern < effects.length; pattern++) {
      if ((pattern & bit) !== 0) effects[pattern] = (effects[pattern] ?? 0) - (effects[pattern ^ bit] ?? 0);
    }
  }
  const coefficients = new Array<number>(effects.length).fill(0);
  indices.forEach((index, pattern) => {
    coefficients[index] = effects[pattern] ?? 0;
  });
  return coefficients;
}
