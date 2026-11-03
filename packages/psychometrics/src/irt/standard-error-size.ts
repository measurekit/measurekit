import { DataError } from "@measurekit/core";

/**
 * The most estimated parameters whose standard errors a calibration computes. Their information is a dense matrix of
 * every pair of them, held with its factor or inverse and the covariance reported, about 210 MB at 2,000, and inverting
 * it takes time with the cube of their number.
 */
const MAX_STANDARD_ERROR_PARAMETERS = 2_000;

/**
 * Returns whether a calibration computes the standard errors of `parameters` estimated parameters, at most
 * {@link MAX_STANDARD_ERROR_PARAMETERS}; a lognormal calibration of more items gives none.
 *
 * @param parameters - The estimated parameters the covariance spans.
 */
export function withinStandardErrorSize(parameters: number): boolean {
  return parameters <= MAX_STANDARD_ERROR_PARAMETERS;
}

/**
 * Checks that a calibration's standard errors are within their bound, before it estimates anything: unidimensional,
 * multidimensional, bifactor, longitudinal, unfolding and joint speed and accuracy calibrations alike.
 *
 * @param parameters - The estimated parameters the covariance spans, or the rows it reports where these are more.
 * @throws `DataError` when there are more than {@link MAX_STANDARD_ERROR_PARAMETERS}.
 */
export function checkStandardErrorSize(parameters: number): void {
  if (!withinStandardErrorSize(parameters)) {
    throw new DataError(
      `Standard errors take at most ${MAX_STANDARD_ERROR_PARAMETERS} estimated parameters, as their information holds every pair of them; found ${parameters}. Leave standardErrors "none", or calibrate fewer items at a time.`,
    );
  }
}
