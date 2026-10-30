import { checkArray, checkOptions, checkRecord, DataError, describe } from "@measurekit/core";
import { type AbilityEstimate, checkThetaPrior, type ThetaIntegrationOptions } from "./ability.js";
import { checkItemList } from "./items.js";
import type { IrtItem } from "./model.js";
import { quadratureRule } from "./quadrature.js";

/**
 * Returns the marginal reliability of theta estimates from a test: the mean over the prior N(μ, σ²) of the reliability
 * at theta, σ² I(θ) / (σ² I(θ) + 1), with I(θ) the test information, by the quadrature of
 * {@link posteriorMeanEstimate}. For the standard normal it is mirt's `marginal_rxx`.
 *
 * @throws `DataError` without items, or when an item's parameters, the prior or the quadrature are invalid.
 */
export function marginalReliability(items: readonly IrtItem[], options: ThetaIntegrationOptions = {}): number {
  checkItemList(items, "Marginal reliability");
  checkOptions(options);
  const { mean, sd } = checkThetaPrior(options.prior);
  const { nodes, weights } = quadratureRule(options.quadrature, mean, sd);
  const variance = sd * sd;
  let reliability = 0;
  nodes.forEach((theta, q) => {
    let information = 0;
    for (const item of items) information += item.model.information(item.parameters, theta);
    reliability += ((weights[q] ?? 0) * variance * information) / (variance * information + 1);
  });
  return reliability;
}

/**
 * Returns the empirical reliability of theta estimates: var(θ̂) / (var(θ̂) + mean(SE²)), the variance with
 * denominator n - 1, as mirt's `empirical_rxx`. Pass the estimates of the persons of a sample, such as EAP estimates.
 *
 * @throws `DataError` with fewer than two estimates, when an estimate or standard error is not finite or a standard
 * error is negative, or when the estimates and standard errors are all 0.
 */
export function empiricalReliability(estimates: readonly AbilityEstimate[]): number {
  checkArray(estimates, "Empirical reliability needs an array of estimates");
  if (estimates.length < 2) throw new DataError("Empirical reliability needs at least two estimates.");
  let sum = 0;
  let errorVariance = 0;
  estimates.forEach((estimate, index) => {
    checkRecord(estimate, `Estimate ${index + 1} must be an object { theta, standardError }`);
    const { theta, standardError } = estimate;
    if (!(Number.isFinite(theta) && Number.isFinite(standardError) && standardError >= 0)) {
      throw new DataError(
        `Estimate ${index + 1} needs a finite theta and a finite, non-negative standard error, found ${describe(theta)} and ${describe(standardError)}.`,
      );
    }
    sum += theta;
    errorVariance += standardError * standardError;
  });
  const mean = sum / estimates.length;
  let squares = 0;
  for (const { theta } of estimates) squares += (theta - mean) ** 2;
  const variance = squares / (estimates.length - 1);
  errorVariance /= estimates.length;
  if (!(variance + errorVariance > 0)) {
    throw new DataError("Empirical reliability needs estimates or standard errors that are not all 0.");
  }
  return variance / (variance + errorVariance);
}
