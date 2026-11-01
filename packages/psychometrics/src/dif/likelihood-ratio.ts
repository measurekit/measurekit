import { chiSquareUpperTail } from "@measurekit/core";

/** A likelihood-ratio test between two nested regressions of an item. */
export interface LikelihoodRatioTest {
  /** The difference of the deviances. */
  readonly chiSquare: number | null;
  /** The number of parameters the larger model adds. */
  readonly degreesOfFreedom: number;
  /** The probability of a chi-square at least as large without the DIF tested. */
  readonly significance: number | null;
  /** The increase of Nagelkerke's R² from the smaller model to the larger: the effect size. */
  readonly rSquaredChange: number | null;
}

/** Returns the test of an item whose regressions cannot be fitted: every statistic is `null`. */
export function undefinedTest(degreesOfFreedom: number): LikelihoodRatioTest {
  return { chiSquare: null, degreesOfFreedom, significance: null, rSquaredChange: null };
}

/**
 * Returns the test of a larger model against a smaller one nested in it, from their deviances, the deviance of the
 * model with only an intercept or thresholds and the number of persons.
 */
export function likelihoodRatioTest(
  smaller: number,
  larger: number,
  degreesOfFreedom: number,
  nullDeviance: number,
  count: number,
): LikelihoodRatioTest {
  // The larger model's deviance is at most the smaller's; rounding may break ties the other way.
  const chiSquare = Math.max(0, smaller - larger);
  return {
    chiSquare,
    degreesOfFreedom,
    significance: chiSquareUpperTail(chiSquare, degreesOfFreedom),
    rSquaredChange: nagelkerke(smaller - chiSquare, nullDeviance, count) - nagelkerke(smaller, nullDeviance, count),
  };
}

/**
 * Returns Nagelkerke's R² of a model with deviance D, given the deviance D0 of the model with only an intercept:
 * (1 - e^((D - D0) / n)) / (1 - e^(-D0 / n)).
 */
function nagelkerke(modelDeviance: number, nullDeviance: number, count: number): number {
  return -Math.expm1((modelDeviance - nullDeviance) / count) / -Math.expm1(-nullDeviance / count);
}
