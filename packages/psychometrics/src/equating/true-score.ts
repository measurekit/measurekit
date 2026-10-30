import { checkArray, DataError } from "@measurekit/core";
import { expectedScore, largestCategoryCount } from "../irt/items.js";
import type { IrtItem } from "../irt/model.js";
import { findIncreasingRoot } from "../math/root.js";
import { scoreLimits } from "./models.js";

/**
 * Returns the IRT true-score equating of a form to another.
 *
 * @remarks
 * For each score x of `from`, from 0 to its highest, the true score of `to` at the theta where the true score of `from`
 * is x. The forms' items must be on a common scale, as after linking. A true score is the sum of the items' expected
 * scores; it rises from the sum L of their limits as theta goes to -∞, Σ c for logistic items and 0 for the others, to
 * the sum U of their limits as theta goes to +∞, Σ d and the highest scores. Scores outside (L, U) have no theta: below
 * L_from they map linearly from 0 to L_to (Kolen and Brennan's procedure), above U_from linearly from U_to to the
 * highest `to` score.
 *
 * @throws `DataError` when a form's items are not an array or there are none, an item is invalid or not one of the package's
 * built-in models, or its expected score does not increase with theta: an NRM item, or a slope a ≤ 0.
 * @see {@link https://doi.org/10.4324/9780203056615 | Lord (1980)}
 * @see {@link https://doi.org/10.1007/978-1-4939-0317-7 | Kolen and Brennan (2014)}
 */
export function trueScoreEquating(from: readonly IrtItem[], to: readonly IrtItem[]): number[] {
  const source = trueScoreForm(from, "from");
  const target = trueScoreForm(to, "to");
  return Array.from({ length: source.highest + 1 }, (_, x) => {
    if (x <= source.lower) return source.lower > 0 ? (x * target.lower) / source.lower : target.lower;
    if (x >= source.upper) {
      return source.upper < source.highest
        ? target.upper + ((x - source.upper) * (target.highest - target.upper)) / (source.highest - source.upper)
        : target.upper;
    }
    return target.trueScore(findIncreasingRoot((theta) => source.trueScore(theta) - x, 0));
  });
}

/** A form's true score function, its limits and its highest score. */
interface TrueScoreForm {
  /** The true score at theta. */
  trueScore(theta: number): number;
  /** The limit of the true score as theta goes to -∞. */
  readonly lower: number;
  /** The limit of the true score as theta goes to +∞. */
  readonly upper: number;
  /** The highest score. */
  readonly highest: number;
}

function trueScoreForm(items: readonly IrtItem[], label: string): TrueScoreForm {
  checkArray(items, `The items of the ${label} form must be an array`);
  if (items.length === 0) throw new DataError(`True-score equating needs items on the ${label} form.`);
  let lower = 0;
  let upper = 0;
  let highest = 0;
  items.forEach((item, j) => {
    const [itemLower, itemUpper] = scoreLimits(item, `Item ${j + 1} of the ${label} form`);
    lower += itemLower;
    upper += itemUpper;
    highest += item.model.categoryCount(item.parameters) - 1;
  });
  const probabilities = new Float64Array(largestCategoryCount(items));
  return {
    trueScore(theta) {
      let score = 0;
      for (const item of items) score += expectedScore(item, theta, probabilities);
      return score;
    },
    lower,
    upper,
    highest,
  };
}
