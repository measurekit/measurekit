import { checkArray, checkFinite, checkOptions, DataError, describe } from "@measurekit/core";
import { checkRounding, roundScore, type ScoreRounding } from "./score-scale.js";

/**
 * Returns the scale scores of equated total scores, by interpolation in the conversion table of the form they are
 * equated to.
 *
 * @remarks
 * Equating gives each score of form X an equivalent on form Y, usually not an integer. The scale score of an equivalent
 * is interpolated linearly between the conversion's scores at the integers either side; an equivalent below 0 or above
 * the form's highest score takes the first or last scale score. The results are then rounded and bounded as `options`
 * say. `conversion` holds the scale scores of the total scores 0 to the highest of form Y, unrounded, as a table made
 * with a `step` of 0 gives them: rounding them first rounds twice.
 *
 * Valid input: finite equivalents and at least one finite scale score.
 *
 * @param equivalents - The equated score for each score, as `equipercentileEquating` gives them.
 * @param conversion - The scale score of each total score of the form equated to, from 0.
 * @param options - The rounding and the bounds of the scale scores.
 * @returns The scale score of each equated score.
 *
 * @example
 * ```ts
 * import { scaleConversion } from "@measurekit/psychometrics/scaling";
 *
 * // Form Y's scale scores of the total scores 0 to 3, and the equivalents on Y of form X's scores 0 to 3.
 * scaleConversion([0.4, 1.5, 2.2, 3.4], [300, 410, 520, 650], { step: 10 }); // [340, 470, 550, 650]
 * ```
 * @throws `DataError` when the equivalents or the conversion are not arrays of finite numbers, the conversion is empty,
 * or the rounding is invalid.
 * @see {@link https://doi.org/10.1007/978-1-4939-0317-7 | Kolen and Brennan (2014)}
 */
export function scaleConversion(
  equivalents: readonly number[],
  conversion: readonly number[],
  options: ScoreRounding = {},
): number[] {
  checkArray(equivalents, "The equivalents must be an array of total scores");
  checkArray(conversion, "The conversion must be an array of scale scores");
  checkOptions(options);
  if (conversion.length === 0) throw new DataError("The conversion needs at least one scale score.");
  const rounding = checkRounding(options, "The options");
  conversion.forEach((score, total) => {
    if (!Number.isFinite(score)) {
      throw new DataError(
        `The conversion needs finite scale scores, found ${describe(score)} for the total score ${total}.`,
      );
    }
  });
  const last = conversion.length - 1;
  return equivalents.map((equivalent, index) => {
    checkFinite(equivalent, `Equivalent ${index + 1}`);
    const position = Math.min(Math.max(equivalent, 0), last);
    const below = Math.floor(position);
    const lowerScore = conversion[below] ?? 0;
    const upperScore = conversion[Math.min(below + 1, last)] ?? 0;
    return roundScore(lowerScore + (position - below) * (upperScore - lowerScore), rounding);
  });
}
