import { checkOptions, checkRecord, checkUnitInterval, type ThetaDistribution } from "@measurekit/core";
import { checkItemList } from "../irt/items.js";
import type { IrtItem } from "../irt/model.js";
import { quadratureRule } from "../irt/quadrature.js";
import { scoreLikelihoods } from "../irt/summed-score.js";
import { type CheckedThetaDistribution, checkThetaDistribution } from "../irt/theta-distribution.js";
import { equipercentileEquating } from "./observed-score.js";

/** The populations of the groups that took the two forms, on the scale of the forms' items. */
export interface EquatingPopulations {
  /**
   * The population of the group that took the form to equate.
   *
   * @defaultValue N(0, 1) on 61 rectangular points
   */
  readonly from?: ThetaDistribution;
  /**
   * The population of the group that took the form to equate to. A group linked to the scale of the other by A and B
   * has the nodes A θ + B of the other's, with the same weights.
   *
   * @defaultValue N(0, 1) on 61 rectangular points
   */
  readonly to?: ThetaDistribution;
}

/** Options of {@link irtObservedScoreEquating}. */
export interface IrtObservedScoreOptions {
  /** The populations of the two groups. */
  readonly populations?: EquatingPopulations;
  /**
   * The weight w₁ of the `from` group's population in the synthetic population, from 0 to 1; the `to` group's is 1 -
   * w₁.
   *
   * @defaultValue `0.5`
   */
  readonly weight?: number;
}

/** The populations and weight of an equating, checked. */
export interface ObservedScoreSetup {
  readonly from: CheckedThetaDistribution;
  readonly to: CheckedThetaDistribution;
  readonly weight: number;
}

/**
 * Returns the IRT observed-score equating of a form to another.
 *
 * @remarks
 * The forms' items must be on a common scale, as after linking, and `populations` are the groups' distributions of
 * theta on it. With P_X(x | θ) the probability of the summed score x on the form to equate at theta, from the recursion
 * of Lord and Wingersky, and g₁, g₂ the groups' populations, the form's score distribution in the synthetic population
 * is f(x) = w₁ Σ_q P_X(x | θ_q) g₁(θ_q) + w₂ Σ_q P_X(x | θ_q) g₂(θ_q), and the distribution g(y) of the form to equate to
 * likewise. Equipercentile equating of f and g gives the result, as {@link equipercentileEquating}. Items of any model
 * are supported, polytomous ones included; the summed score is the sum of the categories. It is the observed-score
 * method of the equateIRT package's `score(method = "OSE")`, which takes the populations N(0, 1) and N(B, A²) on
 * Gauss-Hermite points.
 *
 * @throws `DataError` when a form's items are not an array or there are none, an item is invalid, `populations` is not
 * an object of populations with finite nodes, non-negative weights that sum to 1 within 1e-6 and as many weights as
 * nodes, or `weight` is not a number from 0 to 1.
 * @see {@link https://doi.org/10.1177/014662168400800409 | Lord and Wingersky (1984)}
 * @see {@link https://doi.org/10.1007/978-1-4939-0317-7 | Kolen and Brennan (2014)}
 */
export function irtObservedScoreEquating(
  from: readonly IrtItem[],
  to: readonly IrtItem[],
  options: IrtObservedScoreOptions = {},
): number[] {
  checkItemList(from, "The from form");
  checkItemList(to, "The to form");
  return observedScoreEquivalents(from, to, observedScoreSetup(options));
}

/**
 * Checks the options of an IRT observed-score equating and returns its populations and weight.
 *
 * @throws `DataError` as {@link irtObservedScoreEquating} does for the options.
 */
export function observedScoreSetup(options: IrtObservedScoreOptions): ObservedScoreSetup {
  checkOptions(options);
  const { populations = {}, weight = 0.5 } = options;
  checkRecord(populations, "The option populations must be an object { from, to }");
  checkUnitInterval(weight, "The option weight");
  const { from, to } = populations;
  const standard = quadratureRule();
  return {
    from: from === undefined ? standard : checkThetaDistribution(from, "population of the from group"),
    to: to === undefined ? standard : checkThetaDistribution(to, "population of the to group"),
    weight,
  };
}

/** Returns the equivalents of the checked items of the forms. */
export function observedScoreEquivalents(
  from: readonly IrtItem[],
  to: readonly IrtItem[],
  { from: fromPopulation, to: toPopulation, weight }: ObservedScoreSetup,
): number[] {
  const synthetic = (items: readonly IrtItem[]): number[] => {
    const first = scoreDistribution(items, fromPopulation);
    const second = scoreDistribution(items, toPopulation);
    return Array.from(first, (probability, score) => weight * probability + (1 - weight) * (second[score] ?? 0));
  };
  return equipercentileEquating(synthetic(from), synthetic(to));
}

/** Returns the probability of each summed score of the items in a population: Σ_q g(θ_q) P(score | θ_q). */
function scoreDistribution(items: readonly IrtItem[], population: CheckedThetaDistribution): Float64Array {
  const likelihoods = scoreLikelihoods(items, population.nodes);
  const nodeCount = population.nodes.length;
  const distribution = new Float64Array(likelihoods.length / nodeCount);
  for (let score = 0; score < distribution.length; score++) {
    let probability = 0;
    for (let q = 0; q < nodeCount; q++) {
      probability += (population.weights[q] ?? 0) * (likelihoods[score * nodeCount + q] ?? 0);
    }
    distribution[score] = probability;
  }
  return distribution;
}
