import { backtrack, maxAbs } from "@measurekit/core";
import { invertPositiveDefinite } from "../math/internal.js";
import type { AnsweredPattern } from "./pattern.js";
import type { CheckedPrior } from "./prior.js";

/** The gradient norm at which the search for the posterior mode has converged. */
const MODE_TOLERANCE = 1e-10;
const MODE_MAX_ITERATIONS = 100;
/** A step may raise the negative log posterior by this much, relative to its magnitude: rounding. */
const ROUNDING = 1e-12;
/** The most times a step is halved. */
const MAX_HALVINGS = 40;
/** The scan for the starting point of the mode spans this many standard deviations of the prior around its mean. */
const SCAN_SPAN = 3;
/** Points per dimension of the scan, and above three dimensions. */
const SCAN_POINTS = 5;
const SCAN_POINTS_ABOVE = 3;
/**
 * The most dimensions whose scan is a grid, of 3 ** 5 = 243 points; a grid of 3 points per dimension has 1.6 million at
 * 13 dimensions and 3.5 billion at 20, so above this the scan runs along the axes.
 */
const MAX_GRID_SCAN_DIMENSIONS = 5;

/** The posterior mode of a response pattern under a prior, with the matrix of the information at it. */
export interface PosteriorMode {
  /** The mode. */
  readonly theta: Float64Array;
  /** The test information at the mode plus the precision of the prior, by rows. */
  readonly information: Float64Array;
}

/**
 * Finds the maximum of the posterior of a response pattern under a prior.
 *
 * @remarks
 * A scan of the log posterior on a grid of 5 points per dimension (3 above three dimensions) from -3 to 3 standard
 * deviations of the prior picks the start of Fisher scoring, whose steps -H⁻¹ g, with H the test information plus the
 * precision of the prior, are halved until the negative log posterior does not rise by more than rounding. It runs
 * until the gradient is below 1e-10. Above five dimensions the scan takes the prior mean and the points 3 standard
 * deviations either side of it on each axis, as a grid's points grow as 3 to the power of the dimensions.
 */
export function posteriorMode(pattern: AnsweredPattern, prior: CheckedPrior, dimensions: number): PosteriorMode {
  const objective = negativeLogPosterior(pattern, prior, dimensions);
  const theta = fisherScoring(objective, scanStart(prior, dimensions, objective.value));
  const information = new Float64Array(dimensions * dimensions);
  objective.scoring(theta, new Float64Array(dimensions), information);
  return { theta, information };
}

/** The negative log posterior up to a constant. */
interface Objective {
  /** Returns its value at theta. */
  value(theta: Float64Array): number;
  /** Returns its value and writes its gradient and the matrix Fisher scoring uses: the test information plus the precision. */
  scoring(theta: Float64Array, gradient: Float64Array, hessian: Float64Array): number;
}

function negativeLogPosterior(pattern: AnsweredPattern, prior: CheckedPrior, dimensions: number): Objective {
  const departure = new Float64Array(dimensions);
  const pull = new Float64Array(dimensions);

  /** Sets `departure` to θ - μ and `pull` to Ω (θ - μ), and returns (θ - μ)ᵀ Ω (θ - μ). */
  const priorTerm = (theta: Float64Array): number => {
    let quadratic = 0;
    for (let i = 0; i < dimensions; i++) departure[i] = (theta[i] ?? 0) - (prior.mean[i] ?? 0);
    for (let i = 0; i < dimensions; i++) {
      let sum = 0;
      for (let j = 0; j < dimensions; j++) sum += (prior.precision[i * dimensions + j] ?? 0) * (departure[j] ?? 0);
      pull[i] = sum;
      quadratic += sum * (departure[i] ?? 0);
    }
    return quadratic;
  };
  const value = (theta: Float64Array): number => -pattern.logLikelihood(theta) + priorTerm(theta) / 2;
  return {
    value,
    scoring(theta, gradient, hessian) {
      const result = value(theta);
      pattern.score(theta, gradient);
      for (let d = 0; d < dimensions; d++) gradient[d] = (pull[d] ?? 0) - (gradient[d] ?? 0);
      pattern.information(theta, hessian);
      for (let k = 0; k < hessian.length; k++) hessian[k] = (hessian[k] ?? 0) + (prior.precision[k] ?? 0);
      return result;
    },
  };
}

/** Returns the point of the scan with the lowest objective: of the grid, or along the axes above five dimensions. */
function scanStart(prior: CheckedPrior, dimensions: number, objective: (theta: Float64Array) => number): Float64Array {
  const theta = new Float64Array(dimensions);
  let best = Float64Array.from(prior.mean);
  let bestValue = Number.POSITIVE_INFINITY;
  const visit = (): void => {
    const value = objective(theta);
    if (value < bestValue) {
      bestValue = value;
      best = Float64Array.from(theta);
    }
  };
  if (dimensions <= MAX_GRID_SCAN_DIMENSIONS) {
    const perDimension = dimensions > 3 ? SCAN_POINTS_ABOVE : SCAN_POINTS;
    for (let q = 0; q < perDimension ** dimensions; q++) {
      scanPoint(q, perDimension, prior, theta);
      visit();
    }
    return best;
  }
  theta.set(prior.mean);
  visit();
  for (let d = 0; d < dimensions; d++) {
    const sd = Math.sqrt(prior.covariance[d * dimensions + d] ?? 0);
    for (const z of [-SCAN_SPAN, SCAN_SPAN]) {
      theta.set(prior.mean);
      theta[d] = (prior.mean[d] ?? 0) + sd * z;
      visit();
    }
  }
  return best;
}

/** Writes the point q of the scan grid into `theta`, the first dimension varying fastest. */
function scanPoint(q: number, perDimension: number, prior: CheckedPrior, theta: Float64Array): void {
  const dimensions = theta.length;
  let rest = q;
  for (let d = 0; d < dimensions; d++) {
    const z = -SCAN_SPAN + (2 * SCAN_SPAN * (rest % perDimension)) / (perDimension - 1);
    theta[d] = (prior.mean[d] ?? 0) + Math.sqrt(prior.covariance[d * dimensions + d] ?? 0) * z;
    rest = Math.floor(rest / perDimension);
  }
}

/** Minimizes the objective from `start` by Fisher scoring with halved steps, until its gradient is below the tolerance. */
function fisherScoring(objective: Objective, start: Float64Array): Float64Array {
  const size = start.length;
  const gradient = new Float64Array(size);
  const hessian = new Float64Array(size * size);
  const direction = new Float64Array(size);
  const trial = new Float64Array(size);
  const x = Float64Array.from(start);
  for (let iteration = 0; iteration < MODE_MAX_ITERATIONS; iteration++) {
    const value = objective.scoring(x, gradient, hessian);
    if (maxAbs(gradient) <= MODE_TOLERANCE) break;
    if (!scoringDirection(hessian, gradient, direction)) break;
    if (!halvedStep(objective.value, x, direction, value, trial)) break;
    x.set(trial);
  }
  return x;
}

/** Writes -H⁻¹ g into `direction`; returns `false` when H is not positive definite. */
function scoringDirection(hessian: Float64Array, gradient: Float64Array, direction: Float64Array): boolean {
  const size = gradient.length;
  const inverse = invertPositiveDefinite(hessian, size);
  if (inverse === null) return false;
  for (let i = 0; i < size; i++) {
    let sum = 0;
    for (let j = 0; j < size; j++) sum -= (inverse[i * size + j] ?? 0) * (gradient[j] ?? 0);
    direction[i] = sum;
  }
  return true;
}

/** Writes x + step * direction into `trial` for the first step in 1, 1/2, ... that does not raise the value; returns whether one did. */
function halvedStep(
  value: (theta: Float64Array) => number,
  x: Float64Array,
  direction: Float64Array,
  current: number,
  trial: Float64Array,
): boolean {
  const step = backtrack(
    (length) => {
      for (let k = 0; k < x.length; k++) trial[k] = (x[k] ?? 0) + length * (direction[k] ?? 0);
      return value(trial);
    },
    current,
    0,
    { decrease: 0, slack: ROUNDING * (1 + Math.abs(current)), maxHalvings: MAX_HALVINGS },
  );
  return step !== null;
}
