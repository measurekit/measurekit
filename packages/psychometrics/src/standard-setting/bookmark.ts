import { checkArray, checkOptions, DataError, describe, type FieldNames } from "@measurekit/core";
import { scoreLimits } from "../equating/internal.js";
import { expectedScore, indexItems, largestCategoryCount, type TestItem } from "../irt/internal.js";
import { findIncreasingRoot } from "../math/internal.js";

/** Options of {@link orderedItemBooklet} and {@link bookmarkCutScore}. */
export interface BookmarkOptions {
  /**
   * The response probability (RP) with which a person masters a page, above 0 and below 1.
   *
   * @defaultValue `0.67`
   */
  readonly responseProbability?: number;
}

const BOOKMARK_OPTIONS: FieldNames<BookmarkOptions> = { responseProbability: true };

/** A page of an ordered item booklet: a score point of an item at its location. */
export interface BookletPage {
  /** The item id. */
  readonly id: string;
  /** The score point: the category a person reaches or exceeds on the item, from 1. */
  readonly category: number;
  /** The theta at which the probability of reaching `category` or above is the response probability. */
  readonly location: number;
}

/** The cut scores of a bookmark. */
export interface BookmarkCutScore {
  /** The theta cut: the location of the last page mastered. */
  readonly theta: number;
  /** The expected total score at the theta cut, Σ E(X_i | θ) over the items: the raw cut score. */
  readonly cutScore: number;
}

const DEFAULT_RESPONSE_PROBABILITY = 0.67;

/**
 * Returns the ordered item booklet of the Bookmark method: a page for each score point k from 1 of each item, located
 * at the theta where P(X ≥ k | θ) equals the response probability (RP), in ascending order of location, ties in item
 * order. A dichotomous item has one page, at P(θ) = RP.
 *
 * @param items - The items of the booklet.
 * @param options - The response probability that defines mastery.
 * @returns The pages, from the easiest to the hardest, each with its item, category and location.
 * @throws `DataError` without items, when an id is not a string or repeats, an item is invalid or not one of the
 * built-in models, its expected score does not increase with theta (an NRM item, or a ≤ 0), the response probability
 * is not between 0 and 1, or a logistic item's probability never equals it: c ≥ RP or d ≤ RP.
 * @see {@link https://doi.org/10.4324/9781410600264 | Mitzel, Lewis, Patz and Green (2001)}
 */
export function orderedItemBooklet(items: readonly TestItem[], options: BookmarkOptions = {}): readonly BookletPage[] {
  checkOptions(options, BOOKMARK_OPTIONS);
  const { responseProbability = DEFAULT_RESPONSE_PROBABILITY } = options;
  if (!(typeof responseProbability === "number" && responseProbability > 0 && responseProbability < 1)) {
    throw new DataError(
      `The option responseProbability must be a number above 0 and below 1, found ${describe(responseProbability)}.`,
    );
  }
  checkArray(items, "The items must be an array");
  if (items.length === 0) throw new DataError("An ordered item booklet needs at least one item.");
  indexItems(items);
  for (const item of items) checkReaches(item, responseProbability);
  const probabilities = new Float64Array(largestCategoryCount(items));
  const pages = items.flatMap((item) => {
    const count = item.model.categoryCount(item.parameters);
    return Array.from({ length: count - 1 }, (_, j) => {
      const category = j + 1;
      const reaching = (theta: number): number => {
        item.model.probabilities(item.parameters, theta, probabilities);
        let sum = 0;
        for (let k = category; k < count; k++) sum += probabilities[k] ?? 0;
        return sum - responseProbability;
      };
      return { id: item.id, category, location: findIncreasingRoot(reaching, 0) };
    });
  });
  // Array.prototype.sort is stable, which keeps ties in item order.
  return pages.sort((x, y) => x.location - y.location);
}

/**
 * Returns the cut scores of a bookmark placed after page `page` of the ordered item booklet of `items`: a borderline
 * person masters pages 1 to `page`, each with a probability of at least the response probability. The theta cut is
 * the location of page `page`, and the raw cut score is the expected total score there.
 *
 * @param items - The items of the booklet.
 * @param page - The number of the last page mastered, from 1 to the number of pages.
 * @param options - The response probability that defines mastery.
 * @returns The theta cut, the location of the last page mastered, and the expected total score there, the raw cut
 *   score.
 * @throws `DataError` when `page` is not one of the booklet's pages, or in the cases of {@link orderedItemBooklet}.
 */
export function bookmarkCutScore(
  items: readonly TestItem[],
  page: number,
  options: BookmarkOptions = {},
): BookmarkCutScore {
  const booklet = orderedItemBooklet(items, options);
  const mastered = Number.isSafeInteger(page) ? booklet[page - 1] : undefined;
  if (mastered === undefined) {
    throw new DataError(`A bookmark follows a page from 1 to ${booklet.length}, found ${describe(page)}.`);
  }
  const theta = mastered.location;
  const probabilities = new Float64Array(largestCategoryCount(items));
  let cutScore = 0;
  for (const item of items) cutScore += expectedScore(item, theta, probabilities);
  return { theta, cutScore };
}

/**
 * Checks that an item's probability of reaching each of its score points increases with theta and passes the response
 * probability.
 *
 * @throws `DataError` when it does not.
 */
function checkReaches(item: TestItem, responseProbability: number): void {
  const limits = scoreLimits(item, `Item "${item.id}"`);
  // P(X ≥ 1) of a dichotomous item has the limits of its expected score; P(X ≥ k) of the others runs from 0 to 1.
  const [lower, upper] = item.model.categoryCount(item.parameters) === 2 ? limits : [0, 1];
  if (!(lower < responseProbability && responseProbability < upper)) {
    throw new DataError(
      `Item "${item.id}" has probabilities of a correct response from ${lower} to ${upper}, which never equal the response probability ${responseProbability}.`,
    );
  }
}
