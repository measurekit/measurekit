import { checkFinite, checkRecord, DataError } from "@measurekit/core";
import { checkDistribution } from "../equating/observed-score.js";

/** A panel's bounds in the Hofstee method: the acceptable cut scores and failure rates. */
export interface HofsteeBounds {
  /** The lowest acceptable cut score, k_min. */
  readonly minimumCutScore: number;
  /** The highest acceptable cut score, k_max. */
  readonly maximumCutScore: number;
  /** The lowest acceptable failure rate, f_min: a share of persons, from 0 to 1. */
  readonly minimumFailureRate: number;
  /** The highest acceptable failure rate, f_max. */
  readonly maximumFailureRate: number;
}

/** A Hofstee cut score with its failure rate. */
export interface HofsteeCutScore {
  /** The cut score on the total score scale. */
  readonly cutScore: number;
  /** The failure rate at the cut score. */
  readonly failureRate: number;
}

/**
 * Returns the Hofstee compromise cut score: where the line from (k_min, f_max) to (k_max, f_min) crosses the observed
 * failure rate F(c), the share of persons with a total score below c. F is taken at each score from 0 to the highest
 * and is linear between them, so the cut score need not be a score.
 *
 * @param distribution - The number of persons at each total score, from 0 to the highest.
 * @throws `DataError` when the distribution is invalid, the bounds are not ordered within the scores and [0, 1], or
 * the line does not cross the failure rates between k_min and k_max.
 * @see {@link https://openlibrary.org/works/OL19174477W | Hofstee (1983)}
 */
export function hofsteeCutScore(distribution: readonly number[], bounds: HofsteeBounds): HofsteeCutScore {
  const total = checkDistribution(distribution, "observed");
  const highest = distribution.length - 1;
  checkRecord(bounds, "The Hofstee bounds must be an object of the cut scores and failure rates");
  const { minimumCutScore, maximumCutScore, minimumFailureRate, maximumFailureRate } = bounds;
  checkFinite(minimumCutScore, "The bound minimumCutScore");
  checkFinite(maximumCutScore, "The bound maximumCutScore");
  checkFinite(minimumFailureRate, "The bound minimumFailureRate");
  checkFinite(maximumFailureRate, "The bound maximumFailureRate");
  if (!(0 <= minimumCutScore && minimumCutScore < maximumCutScore && maximumCutScore <= highest)) {
    throw new DataError(
      `The Hofstee cut score bounds need 0 ≤ k_min < k_max ≤ ${highest}, found ${minimumCutScore} and ${maximumCutScore}.`,
    );
  }
  if (!(0 <= minimumFailureRate && minimumFailureRate < maximumFailureRate && maximumFailureRate <= 1)) {
    throw new DataError(
      `The Hofstee failure rate bounds need 0 ≤ f_min < f_max ≤ 1, found ${minimumFailureRate} and ${maximumFailureRate}.`,
    );
  }
  // F at the scores from 0: the share of persons below each.
  const below: number[] = [0];
  for (let score = 1; score <= highest; score++) below.push((below[score - 1] ?? 0) + (distribution[score - 1] ?? 0));
  const failure = (score: number): number => (below[score] ?? 0) / total;
  const cutRange = maximumCutScore - minimumCutScore;
  const rateRange = maximumFailureRate - minimumFailureRate;
  const line = (c: number): number => maximumFailureRate - rateRange * ((c - minimumCutScore) / cutRange);
  // F, linear between scores, minus the line: it increases, so it is 0 at one cut score at most.
  const excess = (c: number): number => {
    const score = Math.min(Math.floor(c), highest - 1);
    return failure(score) + (c - score) * (failure(score + 1) - failure(score)) - line(c);
  };
  if (excess(minimumCutScore) > 0 || excess(maximumCutScore) < 0) {
    throw new DataError(
      "The Hofstee line does not cross the observed failure rates between the minimum and maximum cut scores.",
    );
  }
  let score = Math.floor(minimumCutScore);
  while (score + 1 < maximumCutScore && excess(score + 1) < 0) score++;
  // On the segment from score to score + 1, F(score) + (c - score) step = line(c) at c = k_min + t (k_max - k_min),
  // which stays finite however close the bounds; clamping t absorbs rounding.
  const step = failure(score + 1) - failure(score);
  const t = (maximumFailureRate - failure(score) - (minimumCutScore - score) * step) / (cutRange * step + rateRange);
  const share = Math.min(1, Math.max(0, t));
  // k_min + (k_max - k_min) can round above k_max, and the rate below f_min likewise.
  return {
    cutScore: Math.min(maximumCutScore, minimumCutScore + share * cutRange),
    failureRate: Math.max(minimumFailureRate, maximumFailureRate - share * rateRange),
  };
}
