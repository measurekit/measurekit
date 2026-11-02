import { chiSquareUpperTail, logGamma } from "@measurekit/core";

/** The Poisson weight at which the mixture stops, on either side of its mode: below double precision. */
const NEGLIGIBLE_WEIGHT = 1e-17;

/**
 * Returns P(X ≤ x) of a noncentral chi-square variable, as R's `pchisq(x, df, ncp)`.
 *
 * @remarks
 * The distribution is the Poisson mixture of central chi-squares with df + 2j degrees of freedom, j ~ Poisson(λ / 2).
 * The sum starts at the mode of the weights and walks both ways. One central tail is computed; the others follow from
 * Q(a + 1, z) = Q(a, z) + z^a exp(-z) / Γ(a + 1), so that a call costs the terms of the sum, about 17 √(λ / 2).
 *
 * @param x - The statistic.
 * @param df - The degrees of freedom, at least 1.
 * @param noncentrality - The noncentrality λ, at least 0.
 * @returns The probability, from 0 to 1.
 */
export function noncentralChiSquareCdf(x: number, df: number, noncentrality: number): number {
  if (x <= 0) return 0;
  const mean = noncentrality / 2;
  if (mean === 0) return 1 - chiSquareUpperTail(x, df);
  const z = x / 2;
  const mode = Math.floor(mean);
  const shape = df / 2 + mode;
  const tail = chiSquareUpperTail(x, 2 * shape);
  const weight = Math.exp(mode * Math.log(mean) - mean - logGamma(mode + 1));
  const step = Math.exp(shape * Math.log(z) - z - logGamma(shape + 1));
  const above = sumAbove(mean, z, mode, shape, weight, tail, step);
  const below = sumBelow(mean, z, mode, shape, weight, tail, step);
  return Math.min(1, above + below);
}

/** Returns the terms from the mode up. `step` is z^shape exp(-z) / Γ(shape + 1), which raises the shape by 1. */
function sumAbove(
  mean: number,
  z: number,
  mode: number,
  shape: number,
  modeWeight: number,
  modeTail: number,
  modeStep: number,
): number {
  let sum = 0;
  let weight = modeWeight;
  let tail = modeTail;
  let step = modeStep;
  for (let j = 1; weight > NEGLIGIBLE_WEIGHT * sum || j === 1; j++) {
    sum += weight * (1 - tail);
    tail += step;
    step *= z / (shape + j);
    weight *= mean / (mode + j);
  }
  return sum;
}

/** Returns the terms below the mode. The step that lowers a shape s by 1 is z^(s - 1) exp(-z) / Γ(s). */
function sumBelow(
  mean: number,
  z: number,
  mode: number,
  shape: number,
  modeWeight: number,
  modeTail: number,
  modeStep: number,
): number {
  let sum = 0;
  let weight = modeWeight;
  let tail = modeTail;
  let step = modeStep;
  for (let j = 0; j < mode; j++) {
    step *= (shape - j) / z;
    tail -= step;
    weight *= (mode - j) / mean;
    sum += weight * (1 - tail);
    if (weight <= NEGLIGIBLE_WEIGHT * sum) break;
  }
  return sum;
}
