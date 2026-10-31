import { checkFinite } from "@measurekit/core";
import { checkItemList } from "../irt/items.js";
import type { IrtItem } from "../irt/model.js";
import { scoreLikelihoods } from "../irt/summed-score.js";
import { checkConversion, conditionalDeviation } from "./conversion.js";

/**
 * Returns the conditional standard error of measurement of scale scores at theta: the standard deviation of the scale
 * score of a person at theta.
 *
 * @remarks
 * With P(x | θ) the probability of the total score x, from the recursion of Lord and Wingersky, and c(x) the scale score
 * of x, the error is √(Σ c(x)² P(x | θ) - (Σ c(x) P(x | θ))²). `conversion` holds c(0) to c(highest): the table of a
 * form, rounded and bounded as it is reported. For the unrounded scale score of a θ estimate, the error is the
 * slope times `conditionalStandardError` of the `irt` module.
 *
 * Valid input: items of any models, a finite theta, and as many finite scale scores as the form has total scores,
 * from 0 to the sum of the items' highest categories.
 *
 * @throws `DataError` when the items are not an array, there are none, an item is invalid, theta is not finite, or the
 * conversion is not one finite number for each total score.
 * @see {@link https://doi.org/10.1111/j.1745-3984.1996.tb00485.x | Kolen, Zeng and Hanson (1996)}
 * @see {@link https://doi.org/10.1177/014662168400800409 | Lord and Wingersky (1984)}
 */
export function conditionalScaleError(items: readonly IrtItem[], conversion: readonly number[], theta: number): number {
  checkItemList(items, "A conditional error of scale scores");
  checkFinite(theta, "Theta");
  checkConversion(conversion, items, "The conversion");
  return conditionalDeviation(scoreLikelihoods(items, Float64Array.of(theta)), 1, 0, conversion);
}
