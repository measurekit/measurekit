import { checkNonNegative, DataError } from "@measurekit/core";
import type { ThetaDistribution } from "../irt/ability.js";
import { checkThetaDistribution } from "../irt/theta-distribution.js";
import { normalCdf } from "../math/normal.js";
import {
  checkLevel,
  type ScorePrediction,
  type ScoreProbability,
  summarize,
  type ThetaScorePredictionOptions,
} from "./prediction.js";
import { type CheckedScale, checkScale, type ScoreScale } from "./score-scale.js";

const DEFAULT_STANDARD_ERROR = 0;

/**
 * Returns the prediction of a person's reported scale score when the official test scores theta directly, from a
 * distribution of the person's theta.
 *
 * @remarks
 * The official estimate is θ + e, e ~ N(0, se²), with se the `standardError`, and it is reported on `scale`. The scores
 * a person can be reported are the lowest, the highest and the multiples of `step` between them; the estimates that
 * round and bound to a score s are an interval [t_lo(s), t_hi(s)) of theta, so that P(s) = Σ_q w_q [Φ((t_hi - θ_q) /
 * se) - Φ((t_lo - θ_q) / se)] over the distribution's nodes θ_q and weights w_q. With `standardError` 0, each node's
 * weight goes to the score of its own theta. The prediction is valid only if the distribution and the official test
 * share a theta scale, and treats theta as constant between practice and the official test.
 *
 * Valid input: a distribution of finite nodes and non-negative weights that sum to 1 within 1e-6; a scale with a positive
 * slope, a finite intercept, finite `lowest` and `highest`, and a `step` above 0, since the reportable scores are
 * (`highest` - `lowest`) / `step` + 2 at most, the size of the result, so a scale of a few thousand scores or fewer;
 * a standard error from 0 and a level between 0 and 1.
 *
 * @example
 * ```ts
 * import { predictThetaScore } from "@measurekit/psychometrics/scaling";
 *
 * const posterior = { nodes: [-1, 0, 1], weights: [0.25, 0.5, 0.25] };
 * const scale = { slope: 100, intercept: 500, lowest: 200, highest: 800, step: 10 };
 * predictThetaScore(posterior, scale, { standardError: 0.3 }).mean; // 500
 * ```
 *
 * @throws `DataError` when the distribution, the scale or the options are invalid, or the scale lacks finite bounds or
 * a step above 0.
 */
export function predictThetaScore(
  posterior: ThetaDistribution,
  scale: ScoreScale,
  options: ThetaScorePredictionOptions = {},
): ScorePrediction {
  const checked = checkScale(scale);
  const level = checkLevel(options);
  const { standardError = DEFAULT_STANDARD_ERROR } = options;
  checkNonNegative(standardError, "The option standardError");
  const { nodes, weights } = checkThetaDistribution(posterior, "posterior");
  const { scores, firstMultiple, boundaries } = reportableScores(checked);
  const probabilities = new Float64Array(scores.length);
  nodes.forEach((theta, q) => {
    const weight = weights[q] ?? 0;
    if (standardError === 0) {
      const k = Math.round((checked.slope * theta + checked.intercept) / checked.step);
      const j = Math.min(Math.max(k - firstMultiple, 0), scores.length - 1);
      probabilities[j] = (probabilities[j] ?? 0) + weight;
      return;
    }
    let below = 0;
    for (let j = 0; j < scores.length; j++) {
      const above = j < boundaries.length ? normalCdf(((boundaries[j] ?? 0) - theta) / standardError) : 1;
      probabilities[j] = (probabilities[j] ?? 0) + weight * (above - below);
      below = above;
    }
  });
  const distribution: ScoreProbability[] = scores.map((score, j) => ({ score, probability: probabilities[j] ?? 0 }));
  return summarize(distribution, level);
}

/** The scores a scale reports, in ascending order, and the estimates of theta that round and bound to each. */
interface ReportableScores {
  readonly scores: readonly number[];
  /** The number k such that the multiple k · step is the largest at or below the lowest score. */
  readonly firstMultiple: number;
  /** The theta at which the estimate passes from each score to the next. */
  readonly boundaries: readonly number[];
}

/**
 * Returns the scores a scale reports: the lowest, the multiples of the step above it and below the highest, and the
 * highest. A score v rounds to the multiple k · step with k = round(v / step), and a multiple at or below the lowest
 * reports the lowest, at or above the highest the highest; so with k₀ the largest k with k · step at most the lowest,
 * the j-th score reports for v from (k₀ + j - 1/2) step to (k₀ + j + 1/2) step.
 *
 * @throws `DataError` when `lowest` or `highest` is not finite or `step` is not above 0.
 */
function reportableScores({ lowest, highest, step, slope, intercept }: CheckedScale): ReportableScores {
  if (!(Number.isFinite(lowest) && Number.isFinite(highest) && step > 0)) {
    throw new DataError(
      `Predicting a theta score needs a scale with a finite lowest and highest score and a step above 0, found ${lowest}, ${highest} and ${step}.`,
    );
  }
  const firstMultiple = Math.floor(lowest / step);
  const length = Math.ceil(highest / step) - firstMultiple + 1;
  return {
    scores: Array.from({ length }, (_, j) =>
      j === 0 ? lowest : j === length - 1 ? highest : (firstMultiple + j) * step,
    ),
    firstMultiple,
    boundaries: Array.from({ length: length - 1 }, (_, j) => ((firstMultiple + j + 0.5) * step - intercept) / slope),
  };
}
