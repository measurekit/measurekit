import { numericalHessian } from "../math/hessian.js";
import { invert } from "../math/linear-algebra.js";
import { toUnbounded, unboundedDerivative } from "./bounds.js";
import type { Setup } from "./calibration-setup.js";
import type { ExpectationStep } from "./expectation.js";
import { type Maximization, UNBOUNDED } from "./m-step.js";

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
  const size = values.length;
  const eta = Float64Array.from(values, (x, j) => {
    const { lower, upper } = free[j] ?? UNBOUNDED;
    return toUnbounded(x, lower, upper);
  });
  const hessian = numericalHessian((point, gradient) => {
    mStep.assign(mStep.all, point);
    eStep.run();
    return mStep.objective(mStep.all, point, gradient);
  }, eta);
  mStep.assign(mStep.all, eta);
  const inverse = invert(hessian, size);
  if (inverse === null) return null;
  const scales = Float64Array.from(values, (x, j) => {
    const { lower, upper } = free[j] ?? UNBOUNDED;
    return unboundedDerivative(x, lower, upper);
  });
  for (let j = 0; j < size; j++) {
    if (!((inverse[j * size + j] ?? 0) > 0)) return null;
    for (let k = 0; k < size; k++) {
      inverse[j * size + k] = (inverse[j * size + k] ?? 0) * (scales[j] ?? 0) * (scales[k] ?? 0);
    }
  }
  return inverse;
}
