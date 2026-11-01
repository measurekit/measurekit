import { checkPositiveInteger, checkRandom, type Random } from "@measurekit/core";
import { checkFunction } from "../checks.js";
import { type Distribution, STANDARD_NORMAL } from "./distributions.js";
import { checkThetas } from "./responses.js";

/**
 * Draws `count` thetas from a distribution.
 *
 * @param count - The number of persons.
 * @param random - The generator of the draws.
 * @param distribution - The distribution of a theta.
 * @returns The thetas.
 * @throws `DataError` when `count` is not a positive integer, `random` not a random number generator or `distribution`
 * not a function, or a draw is not finite.
 * @defaultValue `distribution` is `STANDARD_NORMAL`, N(0, 1), the scale of calibration
 */
export function simulateThetas(count: number, random: Random, distribution: Distribution = STANDARD_NORMAL): number[] {
  checkPositiveInteger(count, "The number of thetas");
  checkRandom(random);
  checkFunction(distribution, "The distribution must be a function of a random number generator");
  const thetas = Array.from({ length: count }, () => distribution(random));
  checkThetas(thetas);
  return thetas;
}
