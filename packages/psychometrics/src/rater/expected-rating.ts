import { checkFinite, checkOptions, checkRecordById, DataError, describe, type FieldNames } from "@measurekit/core";
import type { ManyFacetRasch } from "./many-facet-rasch.js";
import { checkManyFacetRasch, expectedRatingAt, ratingStructure } from "./rating-model.js";

/** Options of {@link expectedRating}. */
export interface ExpectedRatingOptions {
  /**
   * The element of some facets by facet name, such as `{ rater: "r2" }`.
   *
   * @defaultValue `{}`: every facet at its mean
   */
  readonly levels?: Readonly<Record<string, string>>;
}

const EXPECTED_RATING_OPTIONS: FieldNames<ExpectedRatingOptions> = { levels: true };

/**
 * Returns the expected rating of a person at `theta` on the elements `levels`: the score a person of that ability would
 * get from raters and items of those measures, the severity-adjusted score that a platform reports for an essay.
 *
 * @remarks
 * Each facet absent from `levels` is at its mean, a measure of 0. With a threshold facet absent from `levels`, the
 * expected ratings under the thresholds of each of its elements are averaged, each weighing the same. With every facet
 * absent, it is the fair score of a person at theta on the average item and rater.
 *
 * @param model - The many-facet Rasch model.
 * @param theta - The ability of the person.
 * @param options - The element of some facets by facet name.
 * @returns The expected rating, in the categories of the model.
 * @throws `DataError` when the model is invalid, `theta` is not finite, the options are not an object, `levels` names a facet the model does not have
 * or an element it does not have.
 */
export function expectedRating(model: ManyFacetRasch, theta: number, options: ExpectedRatingOptions = {}): number {
  checkManyFacetRasch(model);
  checkFinite(theta, "Theta");
  checkOptions(options, EXPECTED_RATING_OPTIONS);
  const { levels = {} } = options;
  checkRecordById(levels, "The levels must be an object of elements by facet");
  const structure = ratingStructure(model);
  for (const [facet, id] of Object.entries(levels)) {
    if (!structure.facets.includes(facet)) {
      throw new DataError(
        `The levels name the facet ${describe(facet)}, but the model has ${structure.facets.map(describe).join(", ")}.`,
      );
    }
    if (typeof id !== "string") {
      throw new DataError(`The level of the facet "${facet}" must be an element id, found ${describe(id)}.`);
    }
  }
  return expectedRatingAt(
    structure,
    theta,
    // An own entry only, so that a facet such as "constructor" without a level is averaged over.
    structure.facets.map((facet) => (Object.hasOwn(levels, facet) ? (levels[facet] ?? null) : null)),
  );
}
