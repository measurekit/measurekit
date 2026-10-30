import { checkOptions, logistic, logLogistic } from "@measurekit/core";
import type { ResponseMatrix } from "../data/response-matrix.js";
import { chiSquareUpperTail } from "../math/chi-square.js";
import { invert } from "../math/linear-algebra.js";
import { type DifGroup, dichotomousData, type ItemCases, itemCases } from "./groups.js";
import { type DifAnalysis, type PurificationOptions, purified, significant } from "./purification.js";

/** A likelihood-ratio test between two nested logistic regressions of an item. */
export interface LikelihoodRatioTest {
  /** The difference of the deviances. */
  readonly chiSquare: number | null;
  /** The number of parameters the larger model adds. */
  readonly degreesOfFreedom: number;
  /** The probability of a chi-square at least as large without the DIF tested. */
  readonly significance: number | null;
  /** The increase of Nagelkerke's R² from the smaller model to the larger: the effect size. */
  readonly rSquaredChange: number | null;
}

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
const STEP_TOLERANCE = 1e-10;
/** Bounds Newton's method where an estimate diverges (separation), by when the deviance has reached its infimum. */
const MAX_ITERATIONS = 100;

/**
 * Returns the logistic regression DIF tests of every dichotomous item, as difR's `difLogistic` with likelihood-ratio
 * tests.
 *
 * @remarks
 * Three models of the log odds of a correct response, with the matching score S over the matching items, all items or,
 * with `purify`, the anchors, and the studied item, and G = 1 in the focal group and 0 in the reference group: model 0,
 * β0 + β1 S; model 1, adding β2 G; model 2, adding β3 S G. The R² change is Nagelkerke's, as difR's `deltaR2`. Each
 * item uses the persons who answered it and another matching item, their score scaled from the matching items they
 * answered (see the guide). A test is `null` for an item these persons answered alike or that has persons of one group
 * only. Purification flags items whose test of both terms is significant at 0.05, difR's default type.
 *
 * @throws `DataError` when the matrix or the groups are invalid, a response is neither 0 nor 1, either group has no
 * person with a response, or `purify` is not a boolean.
 * @see {@link https://doi.org/10.1111/j.1745-3984.1990.tb00754.x | Swaminathan and Rogers (1990)}
 */
export function logisticRegressionDif(
  matrix: ResponseMatrix,
  groups: readonly DifGroup[],
  options: PurificationOptions = {},
): DifAnalysis<LogisticRegressionDif> {
  checkOptions(options);
  const { purify = false } = options;
  const data = dichotomousData(matrix, groups, "Logistic regression DIF");
  return purified(
    matrix.itemIds,
    purify,
    (anchors) => matrix.itemIds.map((id, item) => ({ id, ...itemTests(itemCases(data, item, anchors)) })),
    (item) => significant(item.both.significance),
  );
}

/** Returns the three likelihood-ratio tests of an item, `null` where the item's persons cannot show DIF. */
function itemTests(cases: ItemCases): Omit<LogisticRegressionDif, "id"> {
  const count = cases.scores.length;
  const correct = cases.responses.reduce((sum, response) => sum + response, 0);
  const focal = cases.focal.reduce((sum, flag) => sum + flag, 0);
  const defined = correct > 0 && correct < count && focal > 0 && focal < count;
  const [none = 0, score = 0, group = 0, interaction = 0] = defined
    ? [1, 2, 3, 4].map((terms) => deviance(cases, terms))
    : [];
  const test = (smaller: number, larger: number, degreesOfFreedom: number): LikelihoodRatioTest => {
    if (!defined) return { chiSquare: null, degreesOfFreedom, significance: null, rSquaredChange: null };
    // The larger model's deviance is at most the smaller's; rounding may break ties the other way.
    const chiSquare = Math.max(0, smaller - larger);
    return {
      chiSquare,
      degreesOfFreedom,
      significance: chiSquareUpperTail(chiSquare, degreesOfFreedom),
      rSquaredChange: nagelkerke(smaller - chiSquare, none, count) - nagelkerke(smaller, none, count),
    };
  };
  return { uniform: test(score, group, 1), nonuniform: test(group, interaction, 1), both: test(score, interaction, 2) };
}

/**
 * Returns Nagelkerke's R² of a model with deviance D, given the deviance D0 of the model with only an intercept:
 * (1 - e^((D - D0) / n)) / (1 - e^(-D0 / n)).
 */
function nagelkerke(modelDeviance: number, nullDeviance: number, count: number): number {
  return -Math.expm1((modelDeviance - nullDeviance) / count) / -Math.expm1(-nullDeviance / count);
}

/**
 * Returns the deviance, -2 ln L, of the logistic regression of an item's responses on the first `terms` of 1, S, G and
 * S G, fitted by Newton's method from 0.
 */
function deviance(cases: ItemCases, terms: number): number {
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
    if (step === null) break;
    value = evaluate(cases, fit);
    if (step <= STEP_TOLERANCE) break;
  }
  return 2 * value;
}

/** A logistic regression during fitting, with scratch space for one person's covariates 1, S, G and S G. */
interface LogisticFit {
  readonly terms: number;
  readonly covariates: Float64Array;
  readonly beta: Float64Array;
  readonly gradient: Float64Array;
  readonly hessian: Float64Array;
}

/** Writes the gradient and Hessian of -ln L at the fit's coefficients and returns -ln L. */
function evaluate(cases: ItemCases, fit: LogisticFit): number {
  const { responses, scores, focal } = cases;
  const { covariates } = fit;
  fit.gradient.fill(0);
  fit.hessian.fill(0);
  let value = 0;
  for (let person = 0; person < scores.length; person++) {
    const score = scores[person] ?? 0;
    const group = focal[person] ?? 0;
    covariates[0] = 1;
    covariates[1] = score;
    covariates[2] = group;
    covariates[3] = score * group;
    value += addObservation(fit, responses[person] ?? 0);
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
