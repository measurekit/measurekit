import { checkChoice, checkOptions, DataError } from "@measurekit/core";
import type { ResponseMatrix } from "../data/internal.js";
import { checkThetaPrior, type IrtItem, logProbability, type ThetaPrior } from "../irt/internal.js";
import { checkItems, checkPersonThetas, isExtreme } from "./inputs.js";

/** The estimators whose thetas lz* corrects for. */
export type PersonFitEstimator = "maximum-likelihood" | "weighted-likelihood" | "posterior-mode";

const ESTIMATORS: readonly PersonFitEstimator[] = ["maximum-likelihood", "weighted-likelihood", "posterior-mode"];

/** Options of {@link correctedPersonFit}. */
export interface CorrectedPersonFitOptions {
  /**
   * The estimator of the thetas given, which sets Snijders' correction: `maximumLikelihoodEstimate`,
   * `weightedLikelihoodEstimate` or `posteriorModeEstimate`.
   *
   * @defaultValue `"maximum-likelihood"`
   */
  readonly estimator?: PersonFitEstimator;
  /**
   * The normal prior of the posterior mode, as the estimator took it; the other estimators have none.
   *
   * @defaultValue N(0, 1)
   */
  readonly prior?: ThetaPrior;
}

/** The corrected person fit of one person's responses. */
export interface CorrectedPersonFit {
  /** The person id. */
  readonly id: string;
  /**
   * lz*, about N(0, 1) under fit at an estimated theta; large negative values flag misfit. `null` for an extreme score,
   * or without variance.
   */
  readonly standardized: number | null;
}

/** One answered item at a person's theta: its response, P and Q, and the derivatives of P. */
interface ItemTerms {
  readonly response: number;
  readonly p: number;
  readonly q: number;
  readonly first: number;
  readonly second: number;
}

/**
 * Returns Snijders' lz* of each person: the lz statistic corrected for a theta estimated from the same responses, as
 * PerFit's `lzstar`.
 *
 * @remarks
 * For dichotomous items. With P the probability of a correct response at the person's theta, Q = 1 - P, P' and P'' its
 * derivatives, w = ln(P / Q) and r = P' / (P Q), over the answered items: W = Σ (x - P) w, c = Σ P' w / Σ P' r and
 * w̃ = w - c r; then lz* = (W + c r₀) / √(Σ w̃² P Q), where r₀ is the estimator's term: 0 for maximum likelihood,
 * Σ P' P'' / (P Q) over 2 Σ P'² / (P Q) for Warm's weighted likelihood, and (μ - θ) / σ² for the posterior mode under a
 * N(μ, σ²) prior. Missing responses are skipped. An extreme score, every answered item wrong or every one right, has no
 * value, as its maximum likelihood theta is infinite.
 *
 * @param matrix - The 0/1 responses of the persons.
 * @param items - The calibrated dichotomous items, in the order of the matrix's items.
 * @param thetas - Each person's theta from the estimator of `options`, in the order of the matrix.
 * @param options - The estimator and, for the posterior mode, its prior.
 * @returns For each person, lz*.
 * @throws `DataError` when the matrix, the items, the thetas or the options are invalid, an item is not dichotomous, or a
 * response is not a category of its item.
 * @see {@link https://doi.org/10.1007/BF02294437 | Snijders (2001)}
 * @see {@link https://doi.org/10.1177/01466210122031957 | Meijer and Sijtsma (2001)}
 */
export function correctedPersonFit(
  matrix: ResponseMatrix,
  items: readonly IrtItem[],
  thetas: readonly number[],
  options: CorrectedPersonFitOptions = {},
): CorrectedPersonFit[] {
  checkItems(matrix, items);
  checkPersonThetas(matrix, thetas);
  checkOptions(options);
  const { estimator = "maximum-likelihood", prior } = options;
  checkChoice(estimator, "The option estimator", ESTIMATORS);
  const { mean, sd } = checkThetaPrior(prior);
  items.forEach((item, column) => {
    const count = item.model.categoryCount(item.parameters);
    if (count !== 2) {
      throw new DataError(`lz* needs dichotomous items, but item "${matrix.itemIds[column]}" has ${count} categories.`);
    }
  });
  const probabilities = new Float64Array(2);
  const first = new Float64Array(2);
  const second = new Float64Array(2);
  return matrix.responses.map((row, person) => {
    const id = matrix.personIds[person] ?? "";
    const theta = thetas[person] ?? 0;
    if (isExtreme(items, row)) return { id, standardized: null };
    const terms = row.flatMap((response, column): ItemTerms[] => {
      const item = items[column];
      if (response === null || item === undefined) return [];
      item.model.probabilities(item.parameters, theta, probabilities);
      item.model.thetaDerivatives(item.parameters, theta, first, second);
      const [q = 0, p = 0] = probabilities;
      return [{ response, p, q, first: first[1] ?? 0, second: second[1] ?? 0 }];
    });
    const correction = estimatorTerm(estimator, terms, theta, mean, sd);
    const standardized = lzStar(terms, correction);
    return { id, standardized: Number.isFinite(standardized) ? standardized : null };
  });
}

/** Returns r₀, the term of the estimator in Snijders' correction. */
function estimatorTerm(
  estimator: PersonFitEstimator,
  terms: readonly ItemTerms[],
  theta: number,
  mean: number,
  sd: number,
): number {
  if (estimator === "posterior-mode") return (mean - theta) / (sd * sd);
  if (estimator === "maximum-likelihood") return 0;
  let curvature = 0;
  let information = 0;
  for (const { p, q, first, second } of terms) {
    curvature += (first * second) / (p * q);
    information += (first * first) / (p * q);
  }
  return curvature / (2 * information);
}

/** Returns lz*, (W + c r₀) / √(Σ w̃² P Q), from the terms of the answered items and r₀. */
function lzStar(terms: readonly ItemTerms[], correction: number): number {
  let residual = 0;
  let weighted = 0;
  let scaled = 0;
  for (const { response, p, q, first } of terms) {
    const w = logProbability(p) - logProbability(q);
    residual += (response - p) * w;
    weighted += first * w;
    scaled += (first * first) / (p * q);
  }
  const c = weighted / scaled;
  let variance = 0;
  for (const { p, q, first } of terms) {
    const tilde = logProbability(p) - logProbability(q) - (c * first) / (p * q);
    variance += tilde * tilde * p * q;
  }
  return (residual + c * correction) / Math.sqrt(variance);
}
