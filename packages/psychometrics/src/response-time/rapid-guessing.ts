import { checkOptions, checkPositive, checkRecord } from "@measurekit/core";
import type { ResponseMatrix } from "../data/response-matrix.js";
import type { ResponseRecord, ResponseRecords } from "../data/response-records.js";
import { type AbilityEstimator, posteriorMeanEstimate } from "../irt/ability.js";
import { abilityEstimates } from "../irt/ability-estimates.js";
import type { PersonEstimate } from "../irt/joint-maximum-likelihood.js";
import type { TestItem } from "../irt/model.js";
import { timedResponses } from "./timed-responses.js";

const DEFAULT_PERCENTAGE = 10;
const DEFAULT_CAP = 10;

/** Options of {@link normativeThresholds}. */
export interface NormativeThresholdOptions {
  /**
   * The threshold's percentage of the item's mean time, 10 for the NT10 method.
   *
   * @defaultValue `10`
   */
  readonly percentage?: number;
  /**
   * The largest threshold in seconds.
   *
   * @defaultValue `10`
   */
  readonly cap?: number;
}

/** A person's response time effort. */
export interface ResponseTimeEffort {
  /** The person's id. */
  readonly id: string;
  /** The share of the person's timed responses at or above their item's threshold; `null` without any. */
  readonly effort: number | null;
  /** The number of the person's responses below their item's threshold: rapid guesses. */
  readonly rapidGuesses: number;
}

/**
 * Returns each item's rapid-guessing threshold by the normative threshold method: `percentage` of the mean of the
 * item's times, at most `cap` seconds; NT10 by default. A response faster than its item's threshold is a rapid guess.
 * Items without times have no threshold.
 *
 * @throws `DataError` when the data are malformed or have no times, or `percentage` or `cap` is not positive and
 * finite.
 * @see {@link https://www.researchgate.net/publication/265407579 | Wise and Ma (2012)}
 */
export function normativeThresholds(
  data: ResponseMatrix | ResponseRecords,
  options: NormativeThresholdOptions = {},
): Record<string, number> {
  checkOptions(options);
  const { percentage = DEFAULT_PERCENTAGE, cap = DEFAULT_CAP } = options;
  checkPositive(percentage, "The option percentage");
  checkPositive(cap, "The option cap");
  const timed = timedResponses(data, "Normative thresholds");
  const sums = new Float64Array(timed.itemIds.length);
  const counts = new Float64Array(timed.itemIds.length);
  timed.items.forEach((item, r) => {
    const time = timed.times?.[r] ?? 0;
    if (time === 0) return;
    sums[item] = (sums[item] ?? 0) + time;
    counts[item] = (counts[item] ?? 0) + 1;
  });
  return Object.fromEntries(
    timed.itemIds.flatMap((id, item) => {
      const count = counts[item] ?? 0;
      return count === 0 ? [] : [[id, Math.min(cap, ((percentage / 100) * (sums[item] ?? 0)) / count)]];
    }),
  );
}

/**
 * Returns each person's response time effort: the share of the person's responses with times at or above their item's
 * threshold, solution behavior rather than rapid guessing, over the items with thresholds.
 *
 * @throws `DataError` when the data are malformed or have no times, or a threshold is not a positive finite number.
 * @see {@link https://doi.org/10.1207/s15324818ame1802_2 | Wise and Kong (2005)}
 */
export function responseTimeEffort(
  data: ResponseMatrix | ResponseRecords,
  thresholds: Readonly<Record<string, number>>,
): ResponseTimeEffort[] {
  const timed = timedResponses(data, "Response time effort");
  const limits = thresholdsOf(timed.itemIds, thresholds);
  return timed.personIds.map((id, person) => {
    const flags: boolean[] = [];
    for (let r = timed.offsets[person] ?? 0; r < (timed.offsets[person + 1] ?? 0); r++) {
      const flag = isRapid(timed.times?.[r] ?? 0, limits[timed.items[r] ?? 0] ?? Number.NaN);
      if (flag !== null) flags.push(flag);
    }
    const rapidGuesses = flags.filter(Boolean).length;
    return { id, effort: flags.length === 0 ? null : 1 - rapidGuesses / flags.length, rapidGuesses };
  });
}

/**
 * Returns each person's effort-moderated estimate of theta: the person's rapid guesses, the responses faster than their
 * item's threshold, are treated as missing, and `estimator` estimates theta from the rest, as {@link abilityEstimates}
 * does. Responses without a time or a threshold count.
 *
 * @param estimator - The estimator of theta from the remaining responses.
 * @throws `DataError` when the data are malformed or have no times, a threshold is invalid, or in the cases of
 * {@link abilityEstimates}.
 * @defaultValue `estimator` is `posteriorMeanEstimate`
 * @see {@link https://doi.org/10.1111/j.1745-3984.2006.00002.x | Wise and DeMars (2006)}
 */
export function effortModeratedEstimates(
  data: ResponseMatrix | ResponseRecords,
  items: readonly TestItem[],
  thresholds: Readonly<Record<string, number>>,
  estimator: AbilityEstimator = posteriorMeanEstimate,
): PersonEstimate[] {
  const timed = timedResponses(data, "Effort-moderated scoring");
  const limits = thresholdsOf(timed.itemIds, thresholds);
  const records: ResponseRecord[] = [];
  timed.personIds.forEach((person, p) => {
    for (let r = timed.offsets[p] ?? 0; r < (timed.offsets[p + 1] ?? 0); r++) {
      const item = timed.items[r] ?? 0;
      if (isRapid(timed.times?.[r] ?? 0, limits[item] ?? Number.NaN) === true) continue;
      records.push({ person, item: timed.itemIds[item] ?? "", response: timed.codes[r] ?? 0 });
    }
  });
  return abilityEstimates({ personIds: timed.personIds, itemIds: timed.itemIds, records }, items, estimator);
}

/** Returns whether a response is a rapid guess, or `null` without a time or a threshold to tell. */
function isRapid(time: number, threshold: number): boolean | null {
  return time === 0 || Number.isNaN(threshold) ? null : time < threshold;
}

/**
 * Returns the threshold of each item in order, `NaN` for an item without one.
 *
 * @throws `DataError` when the thresholds are not an object or a threshold is not a positive finite number.
 */
function thresholdsOf(itemIds: readonly string[], thresholds: Readonly<Record<string, number>>): number[] {
  checkRecord(thresholds, "The thresholds must be an object of seconds by item id");
  for (const [id, threshold] of Object.entries(thresholds)) {
    checkPositive(threshold, `The threshold of item "${id}"`);
  }
  return itemIds.map((id) => thresholds[id] ?? Number.NaN);
}
