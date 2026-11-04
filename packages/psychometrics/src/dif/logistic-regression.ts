import { checkOptions, logistic, logLogistic } from "@measurekit/core";
import type { ResponseMatrix } from "../data/internal.js";
import { invert } from "../math/internal.js";
import { type DifGroup, dichotomousData, itemStatistics } from "./groups.js";
import { type LikelihoodRatioTest, likelihoodRatioTest, undefinedTest } from "./likelihood-ratio.js";
import {
  type DifAnalysis,
  PURIFICATION_OPTIONS,
  type PurificationOptions,
  purified,
  significant,
} from "./purification.js";

/** Logistic regression DIF tests of one item. */
export interface LogisticRegressionDif {
  /** The item id. */
  readonly id: string;
  /** Uniform DIF: the group term, model 1 against model 0. */
  readonly uniform: LikelihoodRatioTest;
  /** Nonuniform DIF: the interaction of score and group, model 2 against model 1. */
  readonly nonuniform: LikelihoodRatioTest;
  /** Both: the group and interaction terms, model 2 against model 0. */
  readonly both: LikelihoodRatioTest;
}

/**
 * Newton's method, as glm's iteratively reweighted least squares, stops when no coefficient changes by more than
 * this. It converges quadratically, so the deviance is then exact to working precision; a line search would stall
 * near the optimum, where the deviance changes less than its rounding error.
 */
export const STEP_TOLERANCE = 1e-10;
/** Bounds Newton's method where an estimate diverges (separation), by when the deviance has reached its infimum. */
export const MAX_ITERATIONS = 100;

/**
 * Returns the logistic regression DIF tests of every dichotomous item, as difR's `difLogistic` with likelihood-ratio
 * tests.
 *
 * @remarks
 * Three models of the log odds of a correct response, with the matching score S over the matching items, all items or,
 * with `purify`, the anchors, and the studied item, and G = 1 in the focal group and 0 in the reference group: model 0,
 * β0 + β1 S; model 1, adding β2 G; model 2, adding β3 S G. The R² change is Nagelkerke's, as difR's `deltaR2`. Each
 * item uses the persons who answered it and another matching item, their score scaled from the matching items they
 * answered (see the guide). A test is `null` for an item these persons answered alike, that has persons of one group
 * only or whose persons share one matching score, and where a model's terms are collinear, as when every focal person
 * has the same score. Purification flags items whose test of both terms is significant at 0.05, difR's default type.
 *
 * @param matrix - The responses of the persons, dichotomous.
 * @param groups - The group of each person, in the order of the matrix.
 * @param options - Whether to purify the matching score.
 * @returns The tests of every item, and the purification when it was asked for.
 * @throws `DataError` when the matrix or the groups are invalid, a response is neither 0 nor 1, either group has no
 * person with a response, or `purify` is not a boolean.
 * @see {@link https://doi.org/10.1111/j.1745-3984.1990.tb00754.x | Swaminathan and Rogers (1990)}
 */
export function logisticRegressionDif(
  matrix: ResponseMatrix,
  groups: readonly DifGroup[],
  options: PurificationOptions = {},
): DifAnalysis<LogisticRegressionDif> {
  checkOptions(options, PURIFICATION_OPTIONS);
  const { purify = false } = options;
  const data = dichotomousData(matrix, groups, "Logistic regression DIF");
  return purified(
    matrix.itemIds,
    purify,
    (anchors) =>
      itemStatistics(matrix.itemIds, data, anchors, ({ responses, scores, focal }) =>
        logisticTests({ outcomes: responses, scores, predictors: focal }),
      ),
    (item) => significant(item.both.significance),
  );
}

/** The persons of three nested logistic regressions of a 0/1 outcome on 1, S, P and S P. */
export interface LogisticCases {
  /** Each person's outcome, 0 or 1. */
  readonly outcomes: ArrayLike<number>;
  /** Each person's matching score S. */
  readonly scores: ArrayLike<number>;
  /** Each person's predictor P: the group in logistic regression DIF, the response in the discriminant analysis. */
  readonly predictors: ArrayLike<number>;
}

/** The likelihood-ratio tests of P (1 vs 0), of S P (2 vs 1) and of both (2 vs 0). */
type LogisticTests = Omit<LogisticRegressionDif, "id">;

/**
 * Returns the three likelihood-ratio tests of the regressions on S, then P, then S P: `null` all where the outcomes,
 * the scores or the predictors do not vary, and `null` where a model's terms are collinear.
 */
export function logisticTests(cases: LogisticCases): LogisticTests {
  const count = cases.scores.length;
  const defined = varies(cases.outcomes) && varies(cases.scores) && varies(cases.predictors);
  const [none = null, score = null, group = null, interaction = null] = defined
    ? [1, 2, 3, 4].map((terms) => deviance(cases, terms))
    : [];
  const test = (smaller: number | null, larger: number | null, degreesOfFreedom: number): LikelihoodRatioTest =>
    none !== null && smaller !== null && larger !== null
      ? likelihoodRatioTest(smaller, larger, degreesOfFreedom, none, count)
      : undefinedTest(degreesOfFreedom);
  return { uniform: test(score, group, 1), nonuniform: test(group, interaction, 1), both: test(score, interaction, 2) };
}

/** Returns whether the values are not all equal. */
function varies(values: ArrayLike<number>): boolean {
  for (let i = 1; i < values.length; i++) if (values[i] !== values[0]) return true;
  return false;
}

/**
 * Returns the deviance, -2 ln L, of the logistic regression of the outcomes on the first `terms` of 1, S, P and S P,
 * fitted by Newton's method from 0, or `null` when the Hessian is singular at the start: the terms are collinear.
 * A Hessian that turns singular later comes from separation, where the deviance has reached its infimum.
 */
function deviance(cases: LogisticCases, terms: number): number | null {
  const fit: LogisticFit = {
    terms,
    covariates: new Float64Array(4),
    beta: new Float64Array(terms),
    gradient: new Float64Array(terms),
    hessian: new Float64Array(terms * terms),
  };
  let value = evaluate(cases, fit);
  for (let iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
    const step = newtonStep(fit);
    if (step === null) {
      if (iteration === 0) return null;
      break;
    }
    value = evaluate(cases, fit);
    if (step <= STEP_TOLERANCE) break;
  }
  return 2 * value;
}

/** A logistic regression during fitting, with scratch space for one person's covariates 1, S, P and S P. */
interface LogisticFit {
  readonly terms: number;
  readonly covariates: Float64Array;
  readonly beta: Float64Array;
  readonly gradient: Float64Array;
  readonly hessian: Float64Array;
}

/** Writes the gradient and Hessian of -ln L at the fit's coefficients and returns -ln L. */
function evaluate(cases: LogisticCases, fit: LogisticFit): number {
  const { outcomes, scores, predictors } = cases;
  const { covariates } = fit;
  fit.gradient.fill(0);
  fit.hessian.fill(0);
  let value = 0;
  for (let person = 0; person < scores.length; person++) {
    const score = scores[person] ?? 0;
    const predictor = predictors[person] ?? 0;
    covariates[0] = 1;
    covariates[1] = score;
    covariates[2] = predictor;
    covariates[3] = score * predictor;
    value += addObservation(fit, outcomes[person] ?? 0);
  }
  return value;
}

/**
 * Adds a response y with the covariates in the fit's scratch space to the gradient and Hessian of -ln L, and returns
 * its term of -ln L: ln(1 + e^η) - y η.
 */
function addObservation(fit: LogisticFit, y: number): number {
  const { terms, covariates, beta, gradient, hessian } = fit;
  let eta = 0;
  for (let j = 0; j < terms; j++) eta += (beta[j] ?? 0) * (covariates[j] ?? 0);
  const p = logistic(eta);
  for (let j = 0; j < terms; j++) {
    const xj = covariates[j] ?? 0;
    gradient[j] = (gradient[j] ?? 0) + (p - y) * xj;
    for (let k = 0; k < terms; k++) {
      hessian[j * terms + k] = (hessian[j * terms + k] ?? 0) + p * (1 - p) * xj * (covariates[k] ?? 0);
    }
  }
  return -logLogistic(-eta) - y * eta;
}

/** Takes a Newton step on the fit's coefficients and returns its largest change, or `null` for a singular Hessian. */
function newtonStep(fit: LogisticFit): number | null {
  const { terms, beta, gradient, hessian } = fit;
  const inverse = invert(hessian, terms);
  if (inverse === null) return null;
  let largest = 0;
  for (let j = 0; j < terms; j++) {
    let step = 0;
    for (let k = 0; k < terms; k++) step += (inverse[j * terms + k] ?? 0) * (gradient[k] ?? 0);
    beta[j] = (beta[j] ?? 0) - step;
    largest = Math.max(largest, Math.abs(step));
  }
  return largest;
}
