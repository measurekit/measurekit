import { checkChoice, checkOptions } from "@measurekit/core";
import { scoreLimits } from "../equating/models.js";
import { checkRange, type RangeOptions, type ThetaIntegrationOptions } from "../irt/ability.js";
import { conditionalStandardError } from "../irt/information.js";
import { checkItemList, expectedScore, largestCategoryCount } from "../irt/items.js";
import type { IrtItem } from "../irt/model.js";
import { scoreLikelihoods, summedScoreTable } from "../irt/summed-score.js";
import { findIncreasingRoot } from "../math/root.js";
import { conditionalDeviation, highestScore } from "./conversion.js";
import { checkScale, type ScoreScale, scoreOfTheta } from "./score-scale.js";

/**
 * How a conversion table finds the theta of a total score: `"eap-sum"`, the posterior mean of theta given the total
 * score; `"true-score"`, the theta at which the test characteristic curve equals the score.
 */
export type ConversionMethod = "eap-sum" | "true-score";

/** Options of {@link conversionTable}. */
export interface ConversionTableOptions extends ThetaIntegrationOptions, RangeOptions {
  /**
   * How the theta of a total score is found. `prior` and `quadrature` are the options of `"eap-sum"`; `"true-score"`
   * takes neither. `range` is the interval of the thetas of total scores that have none.
   *
   * @defaultValue `"eap-sum"`
   */
  readonly method?: ConversionMethod;
}

/** One total score of a form with its theta and its scale score. */
export interface ConversionRow {
  /** The total score, from 0 to the highest. */
  readonly score: number;
  /** The theta of the score. At an end of `range` where the score has none. */
  readonly theta: number;
  /**
   * The standard error of theta: the posterior standard deviation for `"eap-sum"`, 1 / √I(θ) for `"true-score"`.
   * `null` where theta is an end of `range`, and where the information is 0.
   */
  readonly standardError: number | null;
  /** The scale score of theta, rounded and bounded as the scale says. */
  readonly scaleScore: number;
  /** The conditional standard error of measurement of the table's scale scores at theta. */
  readonly scaleStandardError: number;
}

/** The thetas of a form's total scores, with their standard errors. */
interface ScoreThetas {
  readonly theta: number;
  readonly standardError: number | null;
}

const METHODS: readonly ConversionMethod[] = ["eap-sum", "true-score"];

/**
 * Returns the conversion table of a form: for each total score its theta, its scale score and the conditional standard
 * error of the scale scores there.
 *
 * @remarks
 * `"eap-sum"` takes the posterior mean of theta given the total score from `summedScoreTable`, as mirt's
 * `fscores(method = "EAPsum")`, with the posterior standard deviation as its standard error. `"true-score"` takes the
 * root of the test characteristic curve τ(θ) = x, with the standard error 1 / √I(θ), and needs a form whose curve
 * increases with theta: built-in models other than the NRM, with positive slopes. A score with no theta takes an
 * end of `range`, the lower for a score below those that have one, and its standard error is `null`: for
 * `"eap-sum"`, a score whose probability underflows to 0; for `"true-score"`, a score at or below the sum of the lower
 * asymptotes, which the curve approaches as theta falls, or at or above the sum of the upper ones.
 *
 * The scale score is {@link scaleScore} of theta. The standard error of the scale scores is
 * {@link conditionalScaleError} at the row's theta of the scale scores of the table, so it is that of the rounded and
 * bounded scores a person is reported. A table for {@link scaleConversion}, which interpolates between unrounded scores,
 * is made with a `step` of 0.
 *
 * Valid input: items of any models for `"eap-sum"`; a prior with a finite mean and positive sd, and a scale with a
 * positive slope.
 *
 * @example
 * ```ts
 * import { conversionTable, twoParameterLogistic } from "@measurekit/psychometrics";
 *
 * const items = [
 *   { model: twoParameterLogistic, parameters: { a: 1.2, b: -0.8, c: 0, d: 1 } },
 *   { model: twoParameterLogistic, parameters: { a: 0.9, b: 0, c: 0, d: 1 } },
 *   { model: twoParameterLogistic, parameters: { a: 1.5, b: 0.9, c: 0, d: 1 } },
 * ];
 * const table = conversionTable(items, { slope: 100, intercept: 500, lowest: 200, highest: 800, step: 10 });
 * table.map((row) => row.scaleScore); // [400, 470, 530, 610]
 * ```
 *
 * @throws `DataError` when the items are not an array, there are none, an item is invalid, the scale, the options, the
 * prior or the quadrature are invalid, or `"true-score"` is asked of a form whose curve does not increase.
 * @see {@link https://doi.org/10.1177/014662169501900105 | Thissen, Pommerich, Billeaud and Williams (1995)}
 * @see {@link https://doi.org/10.4324/9780203056615 | Lord (1980)}
 */
export function conversionTable(
  items: readonly IrtItem[],
  scale: ScoreScale,
  options: ConversionTableOptions = {},
): ConversionRow[] {
  checkItemList(items, "A conversion table");
  const checked = checkScale(scale);
  checkOptions(options);
  const { method = "eap-sum" } = options;
  checkChoice(method, "The option method", METHODS);
  const range = checkRange(options.range);
  const thetas = method === "eap-sum" ? eapSumThetas(items, options, range) : trueScoreThetas(items, range);
  const scaleScores = thetas.map(({ theta }) => scoreOfTheta(theta, checked));
  const nodes = Float64Array.from(thetas, ({ theta }) => theta);
  const likelihoods = scoreLikelihoods(items, nodes);
  return thetas.map(({ theta, standardError }, score) => ({
    score,
    theta,
    standardError,
    scaleScore: scaleScores[score] ?? 0,
    scaleStandardError: conditionalDeviation(likelihoods, nodes.length, score, scaleScores),
  }));
}

/** Returns the posterior mean and standard deviation of theta given each total score. */
function eapSumThetas(
  items: readonly IrtItem[],
  options: ThetaIntegrationOptions,
  range: readonly [number, number],
): ScoreThetas[] {
  const table = summedScoreTable(items, options);
  const first = table.findIndex(({ theta }) => theta !== null);
  return table.map(({ theta, standardError }, score) =>
    theta === null ? { theta: score < first ? range[0] : range[1], standardError: null } : { theta, standardError },
  );
}

/** Returns the theta at which the test characteristic curve equals each total score. */
function trueScoreThetas(items: readonly IrtItem[], range: readonly [number, number]): ScoreThetas[] {
  let lower = 0;
  let upper = 0;
  items.forEach((item, j) => {
    const [itemLower, itemUpper] = scoreLimits(item, `Item ${j + 1}`);
    lower += itemLower;
    upper += itemUpper;
  });
  const probabilities = new Float64Array(largestCategoryCount(items));
  const curve = (theta: number): number => {
    let score = 0;
    for (const item of items) score += expectedScore(item, theta, probabilities);
    return score;
  };
  return Array.from({ length: highestScore(items) + 1 }, (_, score) => {
    if (score <= lower) return { theta: range[0], standardError: null };
    if (score >= upper) return { theta: range[1], standardError: null };
    const theta = findIncreasingRoot((t) => curve(t) - score, 0);
    const standardError = conditionalStandardError(items, theta);
    return { theta, standardError: Number.isFinite(standardError) ? standardError : null };
  });
}
