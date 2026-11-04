import { backtrack, cholesky, DataError, invertPositiveDefinite, multiplyVector } from "@measurekit/core";
import { logDeterminantOfFactor, multiply, trace, traceOfProduct } from "../math/internal.js";
import { designCrossProducts } from "./cross-products.js";

/** The estimates of a regression with a random residual and known sampling variances. */
export interface RemlFit {
  /** The coefficients, by effect. */
  readonly coefficients: Float64Array;
  /** Their covariance matrix, by rows: (XᵀWX)⁻¹. */
  readonly covariance: Float64Array;
  /** The variance τ² of the random residual, at least 0. */
  readonly residualVariance: number;
  /** The restricted log-likelihood at the estimates. */
  readonly logLikelihood: number;
  /** Whether the Fisher scoring reached `tolerance` within `maxIterations`. */
  readonly converged: boolean;
  /** The Fisher scoring steps taken. */
  readonly iterations: number;
}

/** Quantities of the restricted likelihood at one τ². */
interface Evaluation {
  readonly logLikelihood: number;
  /** tr(P), P = W - W X (XᵀWX)⁻¹ XᵀW. */
  readonly traceP: number;
  /** tr(P²). */
  readonly traceP2: number;
  /** yᵀ P² y. */
  readonly quadratic: number;
  readonly coefficients: Float64Array;
  readonly covariance: Float64Array;
}

/** The relative decrease of the restricted log-likelihood that counts as rounding, not as a step that fails. */
const LIKELIHOOD_ROUNDING = 1e-12;
/** The most times a Fisher scoring step is halved, for a step that lowers the restricted log-likelihood. */
const MAX_HALVINGS = 60;

/**
 * Fits y = Xη + u + e by restricted maximum likelihood, u ~ N(0, τ²) the random residual of each item and e ~ N(0, v_i)
 * its estimate's sampling error with the known variance v_i.
 *
 * @remarks
 * The restricted log-likelihood is -(n - p)/2 ln 2π + ½ ln |XᵀX| - ½ Σ ln(v_i + τ²) - ½ ln |XᵀWX| - ½ (y - Xη̂)ᵀ W (y -
 * Xη̂), with W = diag(1 / (v_i + τ²)) and η̂ = (XᵀWX)⁻¹ XᵀWy. Fisher scoring on τ² takes the step (yᵀP²y - tr P) /
 * tr P², with P = W - W X (XᵀWX)⁻¹ XᵀW, halved while it lowers the likelihood and held at the boundary τ² = 0, where it
 * stops when the score is not positive. It starts at τ² = 0.
 *
 * @param design - X, by item, then effect, of full column rank.
 * @param items - The number of items.
 * @param effects - The number of effects.
 * @param y - The estimates.
 * @param variances - The sampling variances v_i, positive.
 * @param tolerance - The change of τ² in a step at which scoring has converged.
 * @param maxIterations - The most Fisher scoring steps.
 * @returns The coefficients, their covariance, τ², the restricted log-likelihood and the convergence.
 * @throws `DataError` when there are not more items than effects, or XᵀWX cannot be inverted.
 * @see {@link https://doi.org/10.1080/01621459.1977.10480998 | Harville (1977)}
 * @see {@link https://doi.org/10.3102/10769986030003261 | Viechtbauer (2005)}
 */
export function remlRegression(
  design: Float64Array,
  items: number,
  effects: number,
  y: Float64Array,
  variances: Float64Array,
  tolerance: number,
  maxIterations: number,
): RemlFit {
  const logDeterminantGram = logDeterminant(designCrossProducts(design, items, effects), effects);
  const constant = -((items - effects) / 2) * Math.log(2 * Math.PI) + logDeterminantGram / 2;
  const evaluate = (tau2: number): Evaluation => evaluation(design, items, effects, y, variances, tau2, constant);
  let tau2 = 0;
  let current = evaluate(tau2);
  let iterations = 0;
  let converged = false;
  while (!converged && iterations < maxIterations) {
    iterations++;
    const step = (current.quadratic - current.traceP) / current.traceP2;
    if (tau2 === 0 && !(step > 0)) {
      converged = true;
      break;
    }
    const proposed = Math.max(0, tau2 + step);
    let next = proposed;
    let trial = current;
    // The last step tried is taken when none lowers the likelihood.
    backtrack(
      (length) => {
        next = tau2 + length * (proposed - tau2);
        trial = evaluate(next);
        return -trial.logLikelihood;
      },
      -current.logLikelihood,
      0,
      { decrease: 0, slack: LIKELIHOOD_ROUNDING * Math.abs(current.logLikelihood), maxHalvings: MAX_HALVINGS },
    );
    converged = Math.abs(next - tau2) < tolerance;
    tau2 = next;
    current = trial;
  }
  return {
    coefficients: current.coefficients,
    covariance: current.covariance,
    residualVariance: tau2,
    logLikelihood: current.logLikelihood,
    converged,
    iterations,
  };
}

/** Evaluates the restricted likelihood and the quantities of its Fisher scoring step at τ². */
function evaluation(
  design: Float64Array,
  items: number,
  effects: number,
  y: Float64Array,
  variances: Float64Array,
  tau2: number,
  constant: number,
): Evaluation {
  const weights = Float64Array.from(variances, (v) => 1 / (v + tau2));
  const { normal, second, third, right } = weightedMoments(design, items, effects, y, weights);
  const covariance = invertPositiveDefinite(normal, effects);
  if (covariance === null) throw new DataError(UNDETERMINED);
  const coefficients = new Float64Array(effects);
  multiplyVector(covariance, right, coefficients);
  const { residualSquares, quadratic, logVariances } = residualSums(
    design,
    items,
    effects,
    y,
    variances,
    weights,
    coefficients,
    tau2,
  );
  const inverseSecond = product(covariance, second, effects);
  const inverseThird = product(covariance, third, effects);
  const traceP = sum(weights, 1) - trace(inverseSecond, effects);
  const traceP2 =
    sum(weights, 2) - 2 * trace(inverseThird, effects) + traceOfProduct(inverseSecond, inverseSecond, effects);
  const logLikelihood = constant - logVariances / 2 - logDeterminant(normal, effects) / 2 - residualSquares / 2;
  return { logLikelihood, traceP, traceP2, quadratic, coefficients, covariance };
}

/** The weighted sum of squared residuals, the sum of their squares times squared weights, and Σ ln(v_i + τ²). */
interface ResidualSums {
  readonly residualSquares: number;
  readonly quadratic: number;
  readonly logVariances: number;
}

function residualSums(
  design: Float64Array,
  items: number,
  effects: number,
  y: Float64Array,
  variances: Float64Array,
  weights: Float64Array,
  coefficients: Float64Array,
  tau2: number,
): ResidualSums {
  let residualSquares = 0;
  let quadratic = 0;
  let logVariances = 0;
  for (let i = 0; i < items; i++) {
    let residual = y[i] ?? 0;
    for (let j = 0; j < effects; j++) residual -= (design[i * effects + j] ?? 0) * (coefficients[j] ?? 0);
    const w = weights[i] ?? 0;
    residualSquares += w * residual * residual;
    quadratic += w * w * residual * residual;
    logVariances += Math.log((variances[i] ?? 0) + tau2);
  }
  return { residualSquares, quadratic, logVariances };
}

const UNDETERMINED =
  "The weighted design cannot be inverted: the sampling variances and the design leave an effect undetermined.";

/** Returns Σ w^power. */
function sum(weights: Float64Array, power: number): number {
  let total = 0;
  for (const w of weights) total += w ** power;
  return total;
}

/** The matrices XᵀW^kX for k = 1, 2, 3, each symmetric and by rows, and XᵀWy. */
interface WeightedMoments {
  readonly normal: Float64Array;
  readonly second: Float64Array;
  readonly third: Float64Array;
  readonly right: Float64Array;
}

function weightedMoments(
  design: Float64Array,
  items: number,
  effects: number,
  y: Float64Array,
  weights: Float64Array,
): WeightedMoments {
  const right = new Float64Array(effects);
  for (let i = 0; i < items; i++) {
    for (let j = 0; j < effects; j++) {
      right[j] = (right[j] ?? 0) + (weights[i] ?? 0) * (design[i * effects + j] ?? 0) * (y[i] ?? 0);
    }
  }
  return {
    normal: designCrossProducts(design, items, effects, weights, 1),
    second: designCrossProducts(design, items, effects, weights, 2),
    third: designCrossProducts(design, items, effects, weights, 3),
    right,
  };
}

/** Returns the product of two `size` by `size` matrices, by rows. */
function product(a: Float64Array, b: Float64Array, size: number): Float64Array {
  const result = new Float64Array(size * size);
  multiply(a, b, size, size, size, result);
  return result;
}

/** Returns ln |A| of a symmetric positive definite matrix, from its Cholesky factor. */
function logDeterminant(a: Float64Array, size: number): number {
  const factor = cholesky(a, size);
  if (factor === null)
    throw new DataError(
      "The weighted design cannot be inverted: the sampling variances and the design leave an effect undetermined.",
    );
  return logDeterminantOfFactor(factor, size);
}
