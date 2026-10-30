/** The base class of every error that the measurekit packages throw. */
export class MeasurekitError extends Error {
  /** Error class name. */
  override name = "MeasurekitError";
}

/**
 * The error for invalid input: a malformed argument, a value out of range, or data for which the requested statistic is
 * undefined, such as the correlation of a constant.
 */
export class DataError extends MeasurekitError {
  /** Error class name. */
  override name = "DataError";
}
