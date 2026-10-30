import { checkFinite, checkMethod, checkPositive } from "@measurekit/core";

/**
 * A prior distribution of an item parameter, on the parameter's own scale. Calibration with priors maximizes the
 * log-likelihood plus the log prior densities: the posterior mode.
 */
export interface ParameterPrior {
  /** Returns the log density at `x`, up to an additive constant; `-Infinity` outside the support. */
  logDensity(x: number): number;
  /** Returns the derivative of the log density at `x`. */
  logDensityDerivative(x: number): number;
  /** Returns the second derivative of the log density at `x`. */
  logDensitySecondDerivative(x: number): number;
}

/**
 * Checks that a value implements {@link ParameterPrior}.
 *
 * @param label - Names the value in the error message, such as `The prior of "c"`.
 * @throws `DataError` when it does not.
 */
export function checkPrior(prior: unknown, label: string): void {
  for (const method of ["logDensity", "logDensityDerivative", "logDensitySecondDerivative"]) {
    checkMethod(prior, method, `${label} must be a prior such as normalPrior(0, 1)`);
  }
}

/**
 * Returns the normal prior N(mean, sd²), as mirt's `norm`.
 *
 * @throws `DataError` when `mean` is not finite or `sd` is not positive and finite.
 */
export function normalPrior(mean: number, sd: number): ParameterPrior {
  checkFinite(mean, "A normal prior's mean");
  checkPositive(sd, "A normal prior's sd");
  const variance = sd * sd;
  return {
    logDensity: (x) => (-0.5 * (x - mean) ** 2) / variance,
    logDensityDerivative: (x) => (mean - x) / variance,
    logDensitySecondDerivative: () => -1 / variance,
  };
}

/**
 * Returns the log-normal prior: log x ~ N(meanLog, sdLog²), as mirt's `lnorm`. Its support is x > 0.
 *
 * @throws `DataError` when `meanLog` is not finite or `sdLog` is not positive and finite.
 */
export function logNormalPrior(meanLog: number, sdLog: number): ParameterPrior {
  checkFinite(meanLog, "A log-normal prior's meanLog");
  checkPositive(sdLog, "A log-normal prior's sdLog");
  const variance = sdLog * sdLog;
  return {
    logDensity: (x) =>
      x > 0 ? -Math.log(x) - (0.5 * (Math.log(x) - meanLog) ** 2) / variance : Number.NEGATIVE_INFINITY,
    logDensityDerivative: (x) => (x > 0 ? -(1 + (Math.log(x) - meanLog) / variance) / x : 0),
    logDensitySecondDerivative: (x) => (x > 0 ? (1 + (Math.log(x) - meanLog - 1) / variance) / (x * x) : 0),
  };
}

/**
 * Returns the beta prior Beta(alpha, beta), with its mode at (alpha - 1) / (alpha + beta - 2), as mirt's `expbeta` on
 * `g` intends. Its support is 0 < x < 1.
 *
 * @throws `DataError` when `alpha` or `beta` is not positive and finite.
 */
export function betaPrior(alpha: number, beta: number): ParameterPrior {
  checkPositive(alpha, "A beta prior's alpha");
  checkPositive(beta, "A beta prior's beta");
  return {
    logDensity: (x) =>
      x > 0 && x < 1 ? (alpha - 1) * Math.log(x) + (beta - 1) * Math.log(1 - x) : Number.NEGATIVE_INFINITY,
    logDensityDerivative: (x) => (x > 0 && x < 1 ? (alpha - 1) / x - (beta - 1) / (1 - x) : 0),
    logDensitySecondDerivative: (x) => (x > 0 && x < 1 ? -(alpha - 1) / (x * x) - (beta - 1) / (1 - x) ** 2 : 0),
  };
}
