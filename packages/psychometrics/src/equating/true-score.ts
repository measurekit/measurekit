import { checkArray, DataError } from "@measurekit/core";
import { expectedScore, type IrtItem, largestCategoryCount } from "../irt/internal.js";
import { findIncreasingRoot } from "../math/internal.js";
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
 * @param from - The items of the form to equate.
 * @param to - The items of the form to equate to.
 * @returns The equated score at each score of the form to equate.
 * @throws `DataError` when a form's items are not an array or there are none, an item is invalid or not one of the
 * built-in models, or its expected score does not increase with theta: an NRM item, or a slope a ≤ 0.
 * @see {@link https://doi.org/10.4324/9780203056615 | Lord (1980)}
 * @see {@link https://doi.org/10.1007/978-1-4939-0317-7 | Kolen and Brennan (2014)}
 */
export function trueScoreEquating(from: readonly IrtItem[], to: readonly IrtItem[]): number[] {
  return trueScoreThetas(from, to).equated;
}

/** A true-score equating with the theta of each score, and the equating of items a small step away. */
export interface TrueScoreThetas {
  /** The equated score at each score of the form to equate. */
  readonly equated: number[];
  /**
   * Returns the equating of the same forms with parameters moved by a small step, such as that of a central difference:
   * one Newton step from each score's theta, which leaves an error of the order of the step squared.
   */
  nearby(from: readonly IrtItem[], to: readonly IrtItem[]): number[];
}

/**
 * Returns the true-score equating of {@link trueScoreEquating} with the theta of each score.
 *
 * @throws `DataError` as `trueScoreEquating` does.
 */
export function trueScoreThetas(from: readonly IrtItem[], to: readonly IrtItem[]): TrueScoreThetas {
  const source = trueScoreForm(from, "from");
  const target = trueScoreForm(to, "to");
  const thetas = Array.from({ length: source.highest + 1 }, (_, x) =>
    x <= source.lower || x >= source.upper ? null : findIncreasingRoot((theta) => source.trueScore(theta) - x, 0),
  );
  return {
    equated: thetas.map((theta, x) => (theta === null ? beyondLimits(source, target, x) : target.trueScore(theta))),
    nearby(movedFrom, movedTo) {
      const near = trueScoreForm(movedFrom, "from");
      const other = trueScoreForm(movedTo, "to");
      return thetas.map((theta, x) => {
        if (theta === null) return beyondLimits(near, other, x);
        const step = theta - (near.trueScore(theta) - x) / near.slope(theta);
        return other.trueScore(Number.isFinite(step) ? step : findIncreasingRoot((t) => near.trueScore(t) - x, 0));
      });
    },
  };
}

/**
 * Returns the equated score of a score with no theta: below the `from` form's lower limit, linear from 0 to the `to`
 * form's, and above its upper limit, linear from the `to` form's to its highest score.
 */
function beyondLimits(source: TrueScoreForm, target: TrueScoreForm, x: number): number {
  if (x <= source.lower) return source.lower > 0 ? (x * target.lower) / source.lower : target.lower;
  return source.upper < source.highest
    ? target.upper + ((x - source.upper) * (target.highest - target.upper)) / (source.highest - source.upper)
    : target.upper;
}

/** A form's true score function, its limits and its highest score. */
interface TrueScoreForm {
  /** The true score at theta. */
  trueScore(theta: number): number;
  /** The derivative of the true score with respect to theta. */
  slope(theta: number): number;
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
  const second = new Float64Array(probabilities.length);
  return {
    trueScore(theta) {
      let score = 0;
      for (const item of items) score += expectedScore(item, theta, probabilities);
      return score;
    },
    slope(theta) {
      let slope = 0;
      for (const item of items) {
        item.model.thetaDerivatives(item.parameters, theta, probabilities, second);
        const count = item.model.categoryCount(item.parameters);
        for (let k = 1; k < count; k++) slope += k * (probabilities[k] ?? 0);
      }
      return slope;
    },
    lower,
    upper,
    highest,
  };
}
