import { checkOptions, checkPositiveInteger, checkRandom, type Random } from "@measurekit/core";
import { checkFunction } from "../checks.js";
import { type Distribution, STANDARD_NORMAL } from "./distributions.js";
import { checkThetas } from "./responses.js";

/** Options of {@link simulateThetas}. */
export interface ThetaSimulationOptions {
  /**
   * The distribution of a theta.
   *
   * @defaultValue `STANDARD_NORMAL`, N(0, 1), the scale of calibration
   */
  readonly distribution?: Distribution;
}

/**
 * Draws `count` thetas from a distribution.
 *
 * @param count - The number of persons.
 * @param random - The generator of the draws.
 * @param options - The distribution of a theta.
 * @returns The thetas.
 * @throws `DataError` when `count` is not a positive integer, `random` not a random number generator, the options not an
 * object or `distribution` not a function, or a draw is not finite.
 */
export function simulateThetas(count: number, random: Random, options: ThetaSimulationOptions = {}): readonly number[] {
  checkPositiveInteger(count, "The number of thetas");
  checkRandom(random);
  checkOptions(options);
  const { distribution = STANDARD_NORMAL } = options;
  checkFunction(distribution, "The distribution must be a function of a random number generator");
  const thetas = Array.from({ length: count }, () => distribution(random));
  checkThetas(thetas);
  return thetas;
}
