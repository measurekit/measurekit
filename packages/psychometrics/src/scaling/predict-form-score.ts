import { checkOptions, type ThetaDistribution } from "@measurekit/core";
import { checkItemList, checkThetaDistribution, type IrtItem, scoreLikelihoods } from "../irt/internal.js";
import { checkConversion } from "./conversion.js";
import {
  checkLevel,
  SCORE_PREDICTION_OPTIONS,
  type ScorePrediction,
  type ScorePredictionOptions,
  summarize,
} from "./prediction.js";

/**
 * Returns the prediction of a person's scale score on an official form from a distribution of the person's theta.
 *
 * @remarks
 * The probability of the total score x is P(x) = Σ_q w_q P(x | θ_q) over the distribution's nodes θ_q and weights w_q,
 * with P(x | θ) from the recursion of Lord and Wingersky; the scale scores of total scores that `conversion` converts to
 * the same score pool their probabilities. The distribution is a posterior of theta given practice responses, as
 * `thetaPosterior` of the `irt` module gives it or the online IRT model of `@measurekit/knowledge-tracing` gives it for
 * a learner, or, with no practice, a calibration's `GroupDistribution`, which gives the score of a person drawn from
 * the population. The prediction treats theta as constant between practice and the official test and is valid only if
 * the practice items and the official form are on one scale: calibrate the practice items with official anchors, or link
 * them first.
 *
 * Valid input: items of any models; a distribution of finite nodes and non-negative weights that sum to 1 within 1e-6;
 * as many finite scale scores as the form has total scores; a level between 0 and 1.
 *
 * @param posterior - The distribution of the person's theta.
 * @param form - The items of the official form.
 * @param conversion - The scale score of each total score of the form, from 0.
 * @param options - The level of the interval.
 * @returns The expected scale score with its interval, and the distribution of the score.
 *
 * @example
 * ```ts
 * import { predictFormScore, thetaPosterior, twoParameterLogistic } from "@measurekit/psychometrics";
 *
 * const item = (b: number) => ({ model: twoParameterLogistic, parameters: { a: 1.1, b, c: 0, d: 1 } });
 * const practice = [item(-1), item(0), item(1)];
 * const official = [item(-0.5), item(0.5)];
 * const posterior = thetaPosterior(practice, [1, 1, 0]);
 * predictFormScore(posterior, official, [200, 500, 800]).mean; // 543.9
 * ```
 * @throws `DataError` when the distribution, the form, the conversion or the options are invalid.
 */
export function predictFormScore(
  posterior: ThetaDistribution,
  form: readonly IrtItem[],
  conversion: readonly number[],
  options: ScorePredictionOptions = {},
): ScorePrediction {
  checkItemList(form, "A score prediction");
  checkConversion(conversion, form, "The conversion");
  checkOptions(options, SCORE_PREDICTION_OPTIONS);
  const level = checkLevel(options);
  const { nodes, weights } = checkThetaDistribution(posterior, "posterior");
  const likelihoods = scoreLikelihoods(form, nodes);
  const pooled = new Map<number, number>();
  conversion.forEach((score, total) => {
    let probability = 0;
    for (let q = 0; q < nodes.length; q++) {
      probability += (weights[q] ?? 0) * (likelihoods[total * nodes.length + q] ?? 0);
    }
    pooled.set(score, (pooled.get(score) ?? 0) + probability);
  });
  const distribution = [...pooled].sort(([x], [y]) => x - y).map(([score, probability]) => ({ score, probability }));
  return summarize(distribution, level);
}
