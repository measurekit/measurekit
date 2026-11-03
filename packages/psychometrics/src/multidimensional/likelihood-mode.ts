import { type BacktrackOptions, backtrack } from "@measurekit/core";
import { invert, invertPositiveDefinite } from "../math/internal.js";
import type { AnsweredPattern } from "./pattern.js";
import { posteriorMode, SCAN_SPAN, scanStart } from "./posterior-mode.js";
import { checkedPrior } from "./prior.js";

/** The largest score of the coordinates inside the box at which the maximum likelihood estimate has converged. */
const LIKELIHOOD_TOLERANCE = 1e-10;
/**
 * The same of the weighted score, whose correction comes from central differences of the gradients, good to about
 * 1e-10.
 */
const WEIGHTED_TOLERANCE = 1e-9;
/** The step of the central differences of the weighted score that give its Jacobian. */
const JACOBIAN_STEP = 1e-4;
const MAX_ITERATIONS = 100;
/** The most times a step is halved. */
const MAX_HALVINGS = 40;
/** A step may raise the criterion by this much, relative to its magnitude: rounding. */
const ROUNDING = 1e-12;

/** The box that the estimates of every dimension stay in. */
interface Box {
  readonly lower: number;
  readonly upper: number;
}

/**
 * Writes a step toward the root from theta, given the field there, into `direction` over the free coordinates and 0 for
 * the others; returns `false` where there is none.
 */
type Step = (theta: Float64Array, field: Float64Array, free: readonly number[], direction: Float64Array) => boolean;

/** The field whose root within the box a modal estimate is, its steps, and the criterion that the steps lower. */
interface Field {
  /** The largest field of the free coordinates at which the search has converged. */
  readonly tolerance: number;
  /** Writes the field at theta into `out`; returns `false` where it is not defined. */
  evaluate(theta: Float64Array, out: Float64Array): boolean;
  /** Fisher scoring's step: I⁻¹ times the field. */
  readonly scoring: Step;
  /**
   * Newton's step, for a field that is the gradient of no criterion: taken where scoring's step at full length does not
   * halve the criterion, which is then the length of the field.
   */
  readonly newton?: Step;
  /** The criterion that a step must not raise. */
  criterion(theta: Float64Array): number;
}

/**
 * Returns the maximum likelihood estimate of a pattern within the box of `range` in every dimension.
 *
 * @remarks
 * A scan of the box for the largest likelihood, on the points of the posterior mode's scan, picks the start of Fisher
 * scoring with the steps projected onto the box: a coordinate at a bound stays there while the score pushes it out.
 * Where the likelihood grows without limit, toward an extreme pattern's corner or along a dimension whose items were all
 * answered alike, the estimate is on that bound, as the one-dimensional estimate is at an end of its range.
 */
export function maximumLikelihoodMode(
  pattern: AnsweredPattern,
  dimensions: number,
  range: readonly [number, number],
): Float64Array {
  const box = { lower: range[0], upper: range[1] };
  const field: Field = {
    tolerance: LIKELIHOOD_TOLERANCE,
    evaluate(theta, out) {
      pattern.score(theta, out);
      return true;
    },
    scoring: scoringStep(pattern, dimensions),
    criterion: (theta) => -pattern.logLikelihood(theta),
  };
  return projectedRoot(field, boxStart(box, dimensions, field.criterion), box);
}

/**
 * Returns Warm's weighted likelihood estimate of a pattern within the box of `range` in every dimension: the root of the
 * score plus Warm's correction, ½ tr(I⁻¹ J_r) for dimension r.
 *
 * @remarks
 * The weighted score is the gradient of no function for the M3PL, or for a model of the caller's, so a step must lower
 * the length of the projected weighted score. Fisher scoring's step is kept where it halves that length; elsewhere, as
 * where the correction changes fast, Newton's step with the Jacobian of the weighted score by central differences is
 * taken, which costs 2D weighted scores. The search starts from the posterior mode under N(0, I), which lies near the
 * WLE: both pull the MLE toward the center, and both are finite for an extreme pattern.
 */
export function weightedLikelihoodMode(
  pattern: AnsweredPattern,
  dimensions: number,
  range: readonly [number, number],
): Float64Array {
  const box = { lower: range[0], upper: range[1] };
  const information = new Float64Array(dimensions * dimensions);
  const correction = new Float64Array(dimensions);
  const weightedScore = (theta: Float64Array, out: Float64Array): boolean => {
    pattern.score(theta, out);
    pattern.information(theta, information);
    const inverse = invertPositiveDefinite(information, dimensions);
    if (inverse === null) return false;
    pattern.warmCorrection(theta, inverse, correction);
    for (let d = 0; d < dimensions; d++) out[d] = (out[d] ?? 0) + (correction[d] ?? 0);
    return true;
  };
  const scratch = new Float64Array(dimensions);
  const field: Field = {
    tolerance: WEIGHTED_TOLERANCE,
    evaluate: weightedScore,
    scoring: scoringStep(pattern, dimensions),
    newton: newtonStep(weightedScore, dimensions),
    criterion(theta) {
      if (!weightedScore(theta, scratch)) return Number.POSITIVE_INFINITY;
      return projectedLength(theta, scratch, box);
    },
  };
  const start = posteriorMode(pattern, checkedPrior(undefined, dimensions), dimensions).theta;
  return projectedRoot(field, start, box);
}

/** Returns Fisher scoring's step: I⁻¹ times the field over the free coordinates, with the test information I. */
function scoringStep(pattern: AnsweredPattern, dimensions: number): Step {
  const information = new Float64Array(dimensions * dimensions);
  return (theta, field, free, direction) => {
    pattern.information(theta, information);
    return solveFree(information, field, free, direction, invertPositiveDefinite);
  };
}

/**
 * Returns Newton's step for the root of `evaluate`: -J⁻¹ g over the free coordinates, with the Jacobian J of the field
 * g by central differences.
 */
function newtonStep(evaluate: (theta: Float64Array, out: Float64Array) => boolean, dimensions: number): Step {
  const moved = new Float64Array(dimensions);
  const up = new Float64Array(dimensions);
  const down = new Float64Array(dimensions);
  /** -J, by rows: the information where the field is the score. */
  const slope = new Float64Array(dimensions * dimensions);
  return (theta, field, free, direction) => {
    moved.set(theta);
    for (const j of free) {
      const at = theta[j] ?? 0;
      moved[j] = at + JACOBIAN_STEP;
      if (!evaluate(moved, up)) return false;
      moved[j] = at - JACOBIAN_STEP;
      if (!evaluate(moved, down)) return false;
      moved[j] = at;
      for (let i = 0; i < dimensions; i++) {
        slope[i * dimensions + j] = -((up[i] ?? 0) - (down[i] ?? 0)) / (2 * JACOBIAN_STEP);
      }
    }
    return solveFree(slope, field, free, direction, invert);
  };
}

/** Returns the point of the scan of the box with the lowest objective. */
function boxStart(box: Box, dimensions: number, objective: (theta: Float64Array) => number): Float64Array {
  const center = new Float64Array(dimensions).fill((box.lower + box.upper) / 2);
  const scales = new Float64Array(dimensions).fill((box.upper - box.lower) / (2 * SCAN_SPAN));
  return scanStart(center, scales, objective);
}

/**
 * Finds the root of the field within the box from `start` by the field's steps on the coordinates that are free: Fisher
 * scoring's, halved until the criterion does not rise; or with Newton's step, scoring's at full length where it halves
 * the criterion, and Newton's, halved, where it does not.
 */
function projectedRoot(field: Field, start: Float64Array, box: Box): Float64Array {
  const size = start.length;
  const values = new Float64Array(size);
  const direction = new Float64Array(size);
  const trial = new Float64Array(size);
  const theta = start.map((value) => clamp(value, box));
  const moveTo = (length: number): number => {
    for (let d = 0; d < size; d++) trial[d] = clamp((theta[d] ?? 0) + length * (direction[d] ?? 0), box);
    return field.criterion(trial);
  };
  for (let iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
    if (!field.evaluate(theta, values)) break;
    const free = freeCoordinates(theta, values, box);
    if (free.every((d) => Math.abs(values[d] ?? 0) <= field.tolerance)) break;
    const current = field.criterion(theta);
    const halving: BacktrackOptions = {
      decrease: 0,
      slack: ROUNDING * (1 + Math.abs(current)),
      maxHalvings: MAX_HALVINGS,
    };
    const scored = field.scoring(theta, values, free, direction);
    const { newton } = field;
    let moved: boolean;
    if (newton === undefined) moved = scored && backtrack(moveTo, current, 0, halving) !== null;
    else if (scored && moveTo(1) <= current / 2) moved = true;
    else moved = newton(theta, values, free, direction) && backtrack(moveTo, current, 0, halving) !== null;
    if (!moved) break;
    theta.set(trial);
  }
  return theta;
}

/** Returns the coordinates not at a bound that the field pushes beyond. */
function freeCoordinates(theta: Float64Array, field: Float64Array, { lower, upper }: Box): number[] {
  return Array.from(theta.keys()).filter((d) => {
    const value = theta[d] ?? 0;
    const push = field[d] ?? 0;
    return !((value <= lower && push < 0) || (value >= upper && push > 0));
  });
}

/** Returns the length of the field over the coordinates that are free. */
function projectedLength(theta: Float64Array, field: Float64Array, box: Box): number {
  return Math.hypot(...freeCoordinates(theta, field, box).map((d) => field[d] ?? 0));
}

/**
 * Writes M⁻¹ v over the free coordinates into `direction`, 0 for the others, with `inverse` inverting M's free rows and
 * columns; returns `false` where it cannot.
 */
function solveFree(
  matrix: Float64Array,
  vector: Float64Array,
  free: readonly number[],
  direction: Float64Array,
  inverse: (matrix: Float64Array, size: number) => Float64Array | null,
): boolean {
  const size = vector.length;
  const count = free.length;
  const reduced = new Float64Array(count * count);
  free.forEach((i, a) => {
    free.forEach((j, b) => {
      reduced[a * count + b] = matrix[i * size + j] ?? 0;
    });
  });
  const inverted = inverse(reduced, count);
  if (inverted === null) return false;
  direction.fill(0);
  free.forEach((i, a) => {
    let sum = 0;
    free.forEach((j, b) => {
      sum += (inverted[a * count + b] ?? 0) * (vector[j] ?? 0);
    });
    direction[i] = sum;
  });
  return true;
}

function clamp(value: number, { lower, upper }: Box): number {
  return Math.min(upper, Math.max(lower, value));
}
