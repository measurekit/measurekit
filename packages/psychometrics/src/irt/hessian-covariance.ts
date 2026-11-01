import { numericalHessian } from "../math/hessian.js";
import { invert } from "../math/linear-algebra.js";
import type { Setup } from "./calibration-setup.js";
import type { ExpectationStep } from "./expectation.js";
import { type Maximization, unboundedCoordinates, unboundedScales } from "./m-step.js";

/**
 * Returns the covariance matrix of the free parameters on their own scale, by rows.
 *
 * @remarks
 * It is the inverse of the Hessian of the negative log-posterior in unbounded coordinates, by the delta method; `null`
 * when the Hessian cannot be inverted or gives a variance that is not positive. Each gradient evaluation runs its own
 * E-step: by Fisher's identity, the gradient of the expected complete-data log-likelihood at the parameters of the
 * E-step is the gradient of the marginal one.
 */
export function hessianCovariance<TParameters>(
  setup: Setup<TParameters>,
  mStep: Maximization<TParameters>,
  eStep: ExpectationStep,
): Float64Array | null {
  const { free, values } = setup;
  const eta = unboundedCoordinates(values, free);
  const hessian = numericalHessian((point, gradient) => {
    mStep.assign(mStep.all, point);
    eStep.run();
    return mStep.objective(mStep.all, point, gradient);
  }, eta);
  mStep.assign(mStep.all, eta);
  return covarianceFromInformation(hessian, values.length, unboundedScales(values, free));
}

/**
 * Returns the covariance of parameters on their own scale from the information on their unbounded scale: the inverse
 * scaled by dx/dη; `null` when the information cannot be inverted or gives a variance that is not positive.
 */
export function covarianceFromInformation(
  information: Float64Array,
  size: number,
  scales: Float64Array,
): Float64Array | null {
  const inverse = invert(information, size);
  if (inverse === null) return null;
  for (let j = 0; j < size; j++) {
    if (!((inverse[j * size + j] ?? 0) > 0)) return null;
    for (let k = 0; k < size; k++) {
      inverse[j * size + k] = (inverse[j * size + k] ?? 0) * (scales[j] ?? 1) * (scales[k] ?? 1);
    }
  }
  return inverse;
}
