import { checkOptions, DataError } from "@measurekit/core";
import { checkFunction } from "../checks.js";
import { checkFacetedScores, type FacetedScore } from "../data/internal.js";
import { type AbilityEstimator, type IrtItem, type PersonEstimate, posteriorMeanEstimate } from "../irt/internal.js";
import type { ManyFacetRasch } from "./many-facet-rasch.js";
import { checkManyFacetRasch, ratingStructure, scoredItems } from "./rating-model.js";

/** Options of {@link manyFacetEstimates}. */
export interface ManyFacetEstimatesOptions {
  /**
   * The estimator of each person.
   *
   * @defaultValue `posteriorMeanEstimate` under the normal prior of the model's population
   */
  readonly estimator?: AbilityEstimator;
}

/**
 * Returns the estimate of theta of every person in the scores, in order of first appearance, from the person's scores
 * on a model's items.
 *
 * @remarks
 * Each of a person's scores is a response to a partial credit item with a = 1 whose steps are the sum of the measures of
 * the score's elements plus the thresholds of its set; the items go to an existing ability estimator, so that EAP, MAP,
 * ML and WLE estimates all work. The estimates are on the model's scale, with the person's severity-adjusted position:
 * the raters' and items' measures are accounted for.
 *
 * @param scores - The scores in long format, each with the element of every facet in `levels`.
 * @param model - The many-facet Rasch model.
 * @param options - The estimator of each person.
 * @returns The estimate of theta and its standard error for each person, in order of first appearance.
 * @throws `DataError` when the model or the scores are invalid, a score has an element the model does not, a score is
 * not a category of its item, the options are not an object, the estimator is not a function, or it throws it.
 */
export function manyFacetEstimates(
  scores: readonly FacetedScore[],
  model: ManyFacetRasch,
  options: ManyFacetEstimatesOptions = {},
): readonly PersonEstimate[] {
  checkManyFacetRasch(model);
  checkOptions(options);
  const {
    estimator = (items: readonly IrtItem[], responses: readonly (number | null)[]) =>
      posteriorMeanEstimate(items, responses, { prior: model.population }),
  } = options;
  checkFunction(estimator, "The estimator must be a function of items and responses, such as posteriorMeanEstimate");
  checkFacetedScores(scores, [model.person, ...model.facets.map(({ facet }) => facet)]);
  const items = scoredItems(scores, ratingStructure(model));
  const byPerson = new Map<string, { items: IrtItem[]; responses: number[] }>();
  scores.forEach(({ levels, score }, index) => {
    const id = levels[model.person] ?? "";
    const entry = byPerson.get(id) ?? { items: [], responses: [] };
    const item = items[index];
    if (item === undefined) throw new DataError(`Score ${index + 1} has no item.`);
    entry.items.push(item);
    entry.responses.push(score);
    byPerson.set(id, entry);
  });
  return [...byPerson].map(([id, { items: answered, responses }]) => ({ id, ...estimator(answered, responses) }));
}
