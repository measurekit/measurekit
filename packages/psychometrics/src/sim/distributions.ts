import { checkFinite, checkPositive, DataError, type Random } from "@measurekit/core";

/**
 * A distribution to draw values from: a function that returns one draw, with its randomness from `random` only, so
 * that a seeded generator reproduces the draws. A custom distribution is such a function, such as the constant
 * `() => 1`.
 */
export type Distribution = (random: Random) => number;

/**
 * Returns the normal distribution N(mean, sd²).
 *
 * @throws `DataError` when `mean` is not finite or `sd` is not positive and finite.
 */
export function normalDistribution(mean: number, sd: number): Distribution {
  checkLocationScale("a normal distribution", ["mean", mean], ["sd", sd]);
  return (random) => mean + sd * random.normal();
}

/**
 * Returns the log-normal distribution: ln x ~ N(meanLog, sdLog²), as R's `rlnorm`.
 *
 * @throws `DataError` when `meanLog` is not finite or `sdLog` is not positive and finite.
 */
export function logNormalDistribution(meanLog: number, sdLog: number): Distribution {
  checkLocationScale("a log-normal distribution", ["meanLog", meanLog], ["sdLog", sdLog]);
  return (random) => Math.exp(meanLog + sdLog * random.normal());
}

/**
 * Returns the uniform distribution from `lower` to `upper`.
 *
 * @throws `DataError` unless `lower` and `upper` are finite and `lower` is below `upper`.
 */
export function uniformDistribution(lower: number, upper: number): Distribution {
  checkFinite(lower, "The lower bound of a uniform distribution");
  checkFinite(upper, "The upper bound of a uniform distribution");
  if (!(lower < upper)) {
    throw new DataError(`A uniform distribution needs the lower bound below the upper, found [${lower}, ${upper}].`);
  }
  return (random) => lower + (upper - lower) * random.uniform();
}

/** The standard normal distribution, N(0, 1). */
export const STANDARD_NORMAL = normalDistribution(0, 1);

function checkLocationScale(
  distribution: string,
  [locationName, location]: readonly [string, number],
  [scaleName, scale]: readonly [string, number],
): void {
  checkFinite(location, `The ${locationName} of ${distribution}`);
  checkPositive(scale, `The ${scaleName} of ${distribution}`);
}
