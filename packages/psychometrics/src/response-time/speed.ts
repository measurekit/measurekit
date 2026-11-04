import { checkOptions, checkPositive, chiSquareUpperTail, type FieldNames } from "@measurekit/core";
import type { ResponseMatrix, ResponseRecords } from "../data/internal.js";

import { posteriors } from "./lognormal.js";
import { forEachTime, itemsOfData, type TimedResponses, type TimeItem, timedResponses } from "./timed-responses.js";

/** Options of {@link speedEstimates} and {@link responseTimeFit}. */
export interface SpeedOptions {
  /**
   * σ_τ of the speed's normal prior, such as a calibration's `speedSd`, for posterior means.
   *
   * @defaultValue none: maximum likelihood estimates
   */
  readonly sd?: number;
}

const SPEED_OPTIONS: FieldNames<SpeedOptions> = { sd: true };

/** A person's estimated speed. */
export interface SpeedEstimate {
  /** The person's id. */
  readonly id: string;
  /** τ: the posterior mean, or the maximum likelihood estimate without a prior; `null` for a person without times. */
  readonly speed: number | null;
  /** The posterior SD, or 1 / √Σ α_i² without a prior; `null` with `speed`. */
  readonly standardError: number | null;
}

/** A person's log-time residuals and their person-level statistic. */
export interface PersonTimeFit extends SpeedEstimate {
  /** z_i = α_i (ln t_i - (β_i - τ)) for each item, in the data's item order; `null` without a time. */
  readonly residuals: readonly (number | null)[];
  /**
   * Σ z_i² over the person's times; `null` without times.
   *
   * @see {@link https://doi.org/10.1007/s11336-007-9046-8 | van der Linden and Guo (2008)}
   */
  readonly statistic: number | null;
  /** The number of times. */
  readonly degreesOfFreedom: number;
  /** The probability of a chi-square with that many degrees of freedom at least as large; `null` without times. */
  readonly significance: number | null;
}

/**
 * Returns each person's speed under the lognormal model, from the person's log times y over items of known
 * parameters: with a prior N(0, σ_τ²), the posterior mean Σ α_i² (β_i - y_i) / (1 / σ_τ² + Σ α_i²) and its SD;
 * without, the maximum likelihood estimate Σ α_i² (β_i - y_i) / Σ α_i² and 1 / √Σ α_i². Items are matched by id.
 *
 * @param data - The responses of the persons with their times.
 * @param items - The lognormal parameters of each item, matched to the data by id.
 * @param options - The standard deviation of the speed.
 * @returns The speed of each person and its standard error; `null` for a person without times.
 * @throws `DataError` when the data are malformed or have no times, an item of the data has no time item, or
 * the prior's sd is not positive and finite.
 */
export function speedEstimates(
  data: ResponseMatrix | ResponseRecords,
  items: readonly TimeItem[],
  options: SpeedOptions = {},
): readonly SpeedEstimate[] {
  const timed = timedResponses(data, "Speed estimation");
  const { means, variances, answered } = speedPosteriors(timed, items, options);
  return timed.personIds.map((id, person) =>
    (answered[person] ?? 0) === 0
      ? { id, speed: null, standardError: null }
      : { id, speed: means[person] ?? 0, standardError: Math.sqrt(variances[person] ?? 0) },
  );
}

/**
 * Returns each person's standardized log-time residuals at the estimated speed, z_i = α_i (ln t_i - (β_i - τ̂)), and
 * their sum of squares, the person-level statistic of van der Linden and Guo, as LNIRT's `personfitLN`.
 *
 * @remarks
 * About chi-square with one degree of freedom per time when the times fit, large for aberrant patterns such as
 * preknowledge or rapid guessing. A large residual flags a single time. The speed is estimated as by {@link
 * speedEstimates}.
 *
 * @param data - The responses of the persons with their times.
 * @param items - The lognormal parameters of each item, matched to the data by id.
 * @param options - The standard deviation of the speed.
 * @returns For each person, the speed estimate with its standard error, the standardized residuals of the times, and
 *   the chi-square statistic of the residuals with its degrees of freedom and probability.
 * @throws `DataError` in the cases of {@link speedEstimates}.
 * @see {@link https://doi.org/10.1007/s11336-007-9046-8 | van der Linden and Guo (2008)}
 */
export function responseTimeFit(
  data: ResponseMatrix | ResponseRecords,
  items: readonly TimeItem[],
  options: SpeedOptions = {},
): readonly PersonTimeFit[] {
  const timed = timedResponses(data, "Response time fit");
  return estimatesOf(timed, items, options);
}

/** Each person's posterior mean and variance of speed, the items' parameters, and the times each person has. */
interface SpeedPosteriors {
  readonly means: Float64Array;
  readonly variances: Float64Array;
  readonly answered: Int32Array;
  readonly intensity: Float64Array;
  readonly discrimination: Float64Array;
}

/**
 * Returns each person's posterior of speed and the number of times it rests on, in time that grows with the times: the
 * persons' residuals over every item, which a sparse log of many persons and items would make their product, are
 * {@link responseTimeFit}'s alone.
 */
function speedPosteriors(
  timed: TimedResponses,
  timeItems: readonly TimeItem[],
  options: SpeedOptions,
): SpeedPosteriors {
  checkOptions(options, SPEED_OPTIONS);
  const { sd } = options;
  if (sd !== undefined) checkPositive(sd, "The option sd");
  const items = itemsOfData(timed.itemIds, timeItems);
  const intensity = Float64Array.from(items, (item) => item.intensity);
  const discrimination = Float64Array.from(items, (item) => item.discrimination);
  const personCount = timed.personIds.length;
  const means = new Float64Array(personCount);
  const variances = new Float64Array(personCount);
  posteriors(timed, intensity, discrimination, sd === undefined ? Number.POSITIVE_INFINITY : sd * sd, means, variances);
  const answered = new Int32Array(personCount);
  forEachTime(timed, (person) => {
    answered[person] = (answered[person] ?? 0) + 1;
  });
  return { means, variances, answered, intensity, discrimination };
}

function estimatesOf(timed: TimedResponses, timeItems: readonly TimeItem[], options: SpeedOptions): PersonTimeFit[] {
  const { means, variances, intensity, discrimination } = speedPosteriors(timed, timeItems, options);
  const residuals = timed.personIds.map(() => new Array<number | null>(intensity.length).fill(null));
  forEachTime(timed, (person, item, logTime) => {
    const row = residuals[person];
    if (row !== undefined) {
      row[item] = (discrimination[item] ?? 0) * (logTime - (intensity[item] ?? 0) + (means[person] ?? 0));
    }
  });
  return timed.personIds.map((id, person) => {
    const row = residuals[person] ?? [];
    const answered = row.filter((value): value is number => value !== null);
    if (answered.length === 0) {
      return {
        id,
        speed: null,
        standardError: null,
        residuals: row,
        statistic: null,
        degreesOfFreedom: 0,
        significance: null,
      };
    }
    const statistic = answered.reduce((sum, value) => sum + value * value, 0);
    return {
      id,
      speed: means[person] ?? 0,
      standardError: Math.sqrt(variances[person] ?? 0),
      residuals: row,
      statistic,
      degreesOfFreedom: answered.length,
      significance: chiSquareUpperTail(statistic, answered.length),
    };
  });
}
