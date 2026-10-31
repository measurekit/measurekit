import { toUnbounded, unboundedDerivative } from "../irt/bounds.js";
import { UNBOUNDED } from "../irt/m-step.js";
import { numericalHessian } from "../math/hessian.js";
import { invert } from "../math/linear-algebra.js";
import { expectation, type JointModel } from "./speed-accuracy-em.js";
import { timeGradient } from "./speed-accuracy-time.js";

/**
 * Returns the covariance matrix of the item parameters and the time parameters at the joint model's estimates, by rows:
 * the free parameters of the items on their own scale, then β, α, σ_τ and ρ; `null` when the information matrix cannot
 * be inverted or gives a variance that is not positive.
 *
 * @remarks
 * It is the inverse of the Hessian of the negative marginal log-likelihood plus the log priors, by central differences
 * of its exact gradient, in the unbounded coordinates of the items' parameters. By Fisher's identity the gradient is
 * that of the expected complete-data log-posterior at the parameters of the E-step, for the items' parameters through
 * the M-step's objective and for the times through the posterior mean and second moment of theta. The model is left at
 * its estimates.
 */
export function jointCovariance<TParameters>(joint: JointModel<TParameters>): Float64Array | null {
  const { setup, mStep, timed, sums, parameters, moments } = joint;
  const { free, values } = setup;
  const itemSize = values.length;
  const size = itemSize + parameters.length;
  const eta = Float64Array.from(values, (x, j) => {
    const { lower, upper } = free[j] ?? UNBOUNDED;
    return toUnbounded(x, lower, upper);
  });
  const itemGradient = new Float64Array(itemSize);
  const timeScores = new Float64Array(parameters.length);
  const point = new Float64Array(size);
  point.set(eta);
  point.set(parameters, itemSize);
  const information = numericalHessian((x, gradient) => {
    parameters.set(x.subarray(itemSize));
    mStep.assign(mStep.all, x.subarray(0, itemSize));
    expectation(joint);
    mStep.objective(mStep.all, x.subarray(0, itemSize), itemGradient);
    timeGradient(timed, sums, parameters, moments, timeScores);
    gradient.set(itemGradient);
    gradient.set(timeScores, itemSize);
    return 0;
  }, point);
  mStep.assign(mStep.all, eta);
  parameters.set(point.subarray(itemSize));
  expectation(joint);
  const inverse = invert(information, size);
  if (inverse === null) return null;
  const scales = Float64Array.from({ length: size }, (_, j) => {
    if (j >= itemSize) return 1;
    const { lower, upper } = free[j] ?? UNBOUNDED;
    return unboundedDerivative(values[j] ?? 0, lower, upper);
  });
  for (let j = 0; j < size; j++) {
    if (!((inverse[j * size + j] ?? 0) > 0)) return null;
    for (let k = 0; k < size; k++)
      inverse[j * size + k] = (inverse[j * size + k] ?? 0) * (scales[j] ?? 1) * (scales[k] ?? 1);
  }
  return inverse;
}
