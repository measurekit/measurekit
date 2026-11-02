import { covarianceFromInformation, unboundedCoordinates, unboundedScales } from "../irt/internal.js";
import { numericalHessian } from "../math/internal.js";
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
  const eta = unboundedCoordinates(values, free);
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
  const scales = new Float64Array(size).fill(1);
  scales.set(unboundedScales(values, free));
  return covarianceFromInformation(information, size, scales);
}
