import { chiSquareUpperTail } from "../math/chi-square.js";

/** One element of a facet: a rater, an item, a task. */
export interface FacetElement {
  /** The element's id. */
  readonly id: string;
  /**
   * The measure in logits. Positive is harder or more severe. The measures of a facet sum to 0.
   */
  readonly measure: number;
  /**
   * The standard error of the measure: 0 for the one element of a facet, whose measure is fixed at 0; `null` without
   * standard errors.
   */
  readonly standardError: number | null;
  /** The number of scores of the element. */
  readonly count: number;
  /** The mean of the element's scores. */
  readonly observedAverage: number;
  /**
   * The expected score of the element when theta is at the mean of the population, the other facets are at their
   * means, and, with a threshold facet other than this one, the thresholds are averaged over its elements.
   */
  readonly fairAverage: number;
}

/** The chi-square test that all elements of a facet have the same measure. */
export interface HomogeneityTest {
  /** Σ w d² - (Σ w d)² / Σ w with w = 1 / SE². */
  readonly chiSquare: number;
  /** The number of elements less 1. */
  readonly degreesOfFreedom: number;
  /** The probability of a chi-square at least as large when the measures are equal. */
  readonly significance: number;
}

/** A facet's elements and how well the measures separate them. */
export interface FacetSummary {
  /** The facet's name. */
  readonly facet: string;
  /** The elements, in order of first appearance in the scores. */
  readonly elements: readonly FacetElement[];
  /**
   * The separation index: the standard deviation of the measures net of their errors over the root mean square error;
   * `null` with fewer than two elements or without standard errors.
   */
  readonly separation: number | null;
  /**
   * The reliability of separation: the share of the variance of the measures that is not error; `null` where the
   * separation is.
   */
  readonly reliability: number | null;
  /** The test that all elements have the same measure; `null` where the separation is. */
  readonly homogeneity: HomogeneityTest | null;
}

/**
 * Returns the summary of a facet's elements: the separation and reliability of Wright and Masters and the
 * homogeneity chi-square of FACETS' "fixed (all same)" test.
 *
 * @remarks
 * With the variance of the measures V (the population variance, dividing by the number of elements) and the mean square
 * error M = mean SE², the true variance is max(V - M, 0), the separation is its square root over √M, and the
 * reliability is the true variance over V.
 *
 * @see {@link https://www.rasch.org/rsa.htm | Wright and Masters (1982)}
 */
export function summarizeFacet(facet: string, elements: readonly FacetElement[]): FacetSummary {
  const errors = elements.map(({ standardError }) => standardError);
  const defined = elements.length >= 2 && errors.every((error) => error !== null && error > 0);
  if (!defined) return { facet, elements, separation: null, reliability: null, homogeneity: null };
  const measures = elements.map(({ measure }) => measure);
  const variances = errors.map((error) => (error ?? 0) ** 2);
  const mean = measures.reduce((sum, measure) => sum + measure, 0) / measures.length;
  const observed = measures.reduce((sum, measure) => sum + (measure - mean) ** 2, 0) / measures.length;
  const meanSquareError = variances.reduce((sum, variance) => sum + variance, 0) / variances.length;
  const trueVariance = Math.max(observed - meanSquareError, 0);
  return {
    facet,
    elements,
    separation: Math.sqrt(trueVariance / meanSquareError),
    reliability: observed > 0 ? trueVariance / observed : null,
    homogeneity: homogeneity(measures, variances),
  };
}

function homogeneity(measures: readonly number[], variances: readonly number[]): HomogeneityTest {
  const weights = variances.map((variance) => 1 / variance);
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  const weightedSum = weights.reduce((sum, weight, i) => sum + weight * (measures[i] ?? 0), 0);
  const weightedSquares = weights.reduce((sum, weight, i) => sum + weight * (measures[i] ?? 0) ** 2, 0);
  const chiSquare = Math.max(weightedSquares - (weightedSum * weightedSum) / total, 0);
  const degreesOfFreedom = measures.length - 1;
  return { chiSquare, degreesOfFreedom, significance: chiSquareUpperTail(chiSquare, degreesOfFreedom) };
}
