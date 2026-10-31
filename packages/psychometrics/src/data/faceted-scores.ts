import { checkArray, checkRecord, DataError, describe } from "@measurekit/core";

/**
 * One score with the element of each facet it was observed at: the long format of ratings by raters of persons on
 * tasks, and of other designs in which a score depends on several facets.
 */
export interface FacetedScore {
  /** The element of each facet by facet name, such as `{ person: "p1", item: "essay", rater: "r2" }`. */
  readonly levels: Readonly<Record<string, string>>;
  /** The score. */
  readonly score: number;
}

/**
 * Checks faceted scores for the facets an analysis uses.
 *
 * @remarks
 * Each score needs a finite `score` and, for every one of `facets`, an element that is a string. Other facets in
 * `levels` are ignored. What else a score must be, such as an integer, is for the analysis to check.
 *
 * @throws `DataError` naming the first score that is not an object, has no `levels` object or a score that is not
 * finite, or lacks an element of a facet.
 */
export function checkFacetedScores(scores: readonly FacetedScore[], facets: readonly string[]): void {
  checkArray(scores, "The scores must be an array of objects { levels, score }");
  scores.forEach((entry, index) => {
    const label = `Score ${index + 1}`;
    checkRecord(entry, `${label} must be an object { levels, score }`);
    checkRecord(entry.levels, `${label} needs levels, an object of elements by facet`);
    if (!Number.isFinite(entry.score)) {
      throw new DataError(`${label} needs a finite score, found ${describe(entry.score)}.`);
    }
    for (const facet of facets) {
      const element = entry.levels[facet];
      if (typeof element !== "string") {
        throw new DataError(`${label} needs a string element for the facet "${facet}", found ${describe(element)}.`);
      }
    }
  });
}
