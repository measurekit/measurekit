import { checkOptions, checkPositive } from "@measurekit/core";
import type { ResponseMatrix } from "../data/response-matrix.js";
import type { ResponseRecords } from "../data/response-records.js";
import { chiSquareUpperTail } from "../math/chi-square.js";
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
 * @throws `DataError` when the data are malformed or have no times, an item of the data has no time item, or
 * the prior's sd is not positive and finite.
 */
export function speedEstimates(
  data: ResponseMatrix | ResponseRecords,
  items: readonly TimeItem[],
  options: SpeedOptions = {},
): SpeedEstimate[] {
  const timed = timedResponses(data, "Speed estimation");
  return estimatesOf(timed, items, options).map(({ id, speed, standardError }) => ({ id, speed, standardError }));
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
 * @throws `DataError` in the cases of {@link speedEstimates}.
 * @see {@link https://doi.org/10.1007/s11336-007-9046-8 | van der Linden and Guo (2008)}
 */
export function responseTimeFit(
  data: ResponseMatrix | ResponseRecords,
  items: readonly TimeItem[],
  options: SpeedOptions = {},
): PersonTimeFit[] {
  const timed = timedResponses(data, "Response time fit");
  return estimatesOf(timed, items, options);
}

function estimatesOf(timed: TimedResponses, timeItems: readonly TimeItem[], options: SpeedOptions): PersonTimeFit[] {
  checkOptions(options);
  const { sd } = options;
  if (sd !== undefined) checkPositive(sd, "The option sd");
  const items = itemsOfData(timed.itemIds, timeItems);
  const intensity = Float64Array.from(items, (item) => item.intensity);
  const discrimination = Float64Array.from(items, (item) => item.discrimination);
  const personCount = timed.personIds.length;
  const means = new Float64Array(personCount);
  const variances = new Float64Array(personCount);
  posteriors(timed, intensity, discrimination, sd === undefined ? Number.POSITIVE_INFINITY : sd * sd, means, variances);
  const residuals = timed.personIds.map(() => new Array<number | null>(items.length).fill(null));
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
