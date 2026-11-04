import { checkArray, checkRecord, DataError, describe } from "@measurekit/core";
import { checkFacetedScores, type FacetedScore } from "../data/internal.js";
import { addResponse, type MeanSquareFit, meanSquares, SUM_COUNT, scoreMoments } from "../fit/internal.js";
import type { PersonEstimate } from "../irt/internal.js";
import type { ManyFacetRasch } from "./many-facet-rasch.js";
import { checkManyFacetRasch, ratingStructure, scoredItems } from "./rating-model.js";

/** The residual fit of one element of a facet. */
export interface FacetFit extends MeanSquareFit {
  /** The facet's name. */
  readonly facet: string;
  /** The element's id. */
  readonly id: string;
}

/**
 * Returns the infit and outfit mean squares of every element of every facet, with their standardized forms, from the
 * residuals of its scores at the persons' estimates.
 *
 * @remarks
 * Each score's expected value, variance and fourth central moment come from its item at the person's theta, as in
 * `itemFit`, and a score adds to the sums of its element in every facet. Scores of persons without an estimate, a `null`
 * theta, are skipped; persons with extreme scores are not left out, as the estimates are finite under a prior. A
 * statistic is `null` for an element without scores of persons with estimates.
 *
 * @param scores - The scores in long format, each with the element of every facet in `levels`.
 * @param model - The many-facet Rasch model.
 * @param persons - Estimates by person id, such as those of `manyFacetEstimates`.
 * @returns For each element of each facet, the infit and outfit mean squares and their standardized forms.
 * @throws `DataError` when the model, the scores or the estimates are invalid, a score has an element the model does
 * not, or a score is not a category of its item.
 */
export function manyFacetFit(
  scores: readonly FacetedScore[],
  model: ManyFacetRasch,
  persons: readonly PersonEstimate[],
): readonly FacetFit[] {
  checkManyFacetRasch(model);
  checkFacetedScores(scores, [model.person, ...model.facets.map(({ facet }) => facet)]);
  const thetas = personThetas(persons);
  const structure = ratingStructure(model);
  const items = scoredItems(scores, structure);
  const starts = [0];
  for (const { elements } of model.facets) starts.push((starts.at(-1) ?? 0) + elements.length);
  const indices = model.facets.map(({ elements }) => new Map(elements.map(({ id }, e) => [id, e])));
  const sums = new Float64Array((starts.at(-1) ?? 0) * SUM_COUNT);
  const probabilities = new Float64Array(items.reduce((most, item) => Math.max(most, item.parameters.b.length + 1), 0));
  const moments = new Float64Array(3);
  scores.forEach(({ levels, score }, index) => {
    const theta = thetas.get(levels[model.person] ?? "");
    const item = items[index];
    if (theta === undefined || theta === null || item === undefined) return;
    scoreMoments(item, theta, probabilities, moments);
    model.facets.forEach(({ facet }, f) => {
      const element = indices[f]?.get(levels[facet] ?? "") ?? 0;
      addResponse(sums, ((starts[f] ?? 0) + element) * SUM_COUNT, score - (moments[0] ?? 0), moments);
    });
  });
  return model.facets.flatMap(({ facet, elements }, f) =>
    elements.map(({ id }, e) => ({ facet, id, ...meanSquares(sums, ((starts[f] ?? 0) + e) * SUM_COUNT) })),
  );
}

/** Returns each person's theta by id, checking the estimates. */
function personThetas(persons: readonly PersonEstimate[]): Map<string, number | null> {
  checkArray(persons, "The persons must be an array of estimates { id, theta, standardError }");
  const thetas = new Map<string, number | null>();
  persons.forEach((person, index) => {
    checkRecord(person, `Estimate ${index + 1} must be an object { id, theta, standardError }`);
    if (typeof person.id !== "string" || !(person.theta === null || Number.isFinite(person.theta))) {
      throw new DataError(
        `Estimate ${index + 1} needs a string id and a theta that is a finite number or null, found ${describe(person.theta)}.`,
      );
    }
    thetas.set(person.id, person.theta);
  });
  return thetas;
}
