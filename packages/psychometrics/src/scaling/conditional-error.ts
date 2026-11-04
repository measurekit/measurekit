import { checkFinite, checkOptions, type FieldNames } from "@measurekit/core";
import {
  checkItemList,
  type IrtItem,
  type QuadratureOptions,
  scoreLikelihoods,
  scoreTestlets,
  type TestletOptions,
} from "../irt/internal.js";
import { checkConversion, conditionalDeviation } from "./conversion.js";

/** Options of {@link conditionalScaleError}. */
export interface ConditionalScaleErrorOptions extends TestletOptions {
  /**
   * The quadrature rule over each testlet's effect, placed on N(0, sd²); without testlets it is not used.
   *
   * @defaultValue 61 rectangular points
   */
  readonly quadrature?: QuadratureOptions;
}

const CONDITIONAL_SCALE_ERROR_OPTIONS: FieldNames<ConditionalScaleErrorOptions> = { testlets: true, quadrature: true };

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
 * With testlets, P(x | θ) integrates each testlet's effect out, as `summedScoreTable` does: the error is that of a
 * person at theta whose testlet effects vary, under the testlet response model.
 *
 * Valid input: items of any models, a finite theta, and as many finite scale scores as the form has total scores,
 * from 0 to the sum of the items' highest categories.
 *
 * @param items - The items of the form; with testlets, each with its id.
 * @param conversion - The scale score of each total score, from 0.
 * @param theta - The ability.
 * @param options - The testlets and the quadrature over their effects.
 * @returns The standard deviation of the scale score at theta, in scale points.
 * @throws `DataError` when the items are not an array, there are none, an item is invalid, theta is not finite, the
 * conversion is not one finite number for each total score, or the options, the testlets or the quadrature are invalid.
 * @see {@link https://doi.org/10.1111/j.1745-3984.1996.tb00485.x | Kolen, Zeng and Hanson (1996)}
 * @see {@link https://doi.org/10.1177/014662168400800409 | Lord and Wingersky (1984)}
 * @see {@link https://doi.org/10.1007/s11336-014-9411-3 | Cai (2015)}
 */
export function conditionalScaleError(
  items: readonly IrtItem[],
  conversion: readonly number[],
  theta: number,
  options: ConditionalScaleErrorOptions = {},
): number {
  checkItemList(items, "A conditional error of scale scores");
  checkFinite(theta, "Theta");
  checkConversion(conversion, items, "The conversion");
  checkOptions(options, CONDITIONAL_SCALE_ERROR_OPTIONS);
  const testlets = scoreTestlets(items, options.testlets, options.quadrature);
  return conditionalDeviation(scoreLikelihoods(items, Float64Array.of(theta), testlets), 1, 0, conversion);
}
