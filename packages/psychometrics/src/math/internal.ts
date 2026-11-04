// Helpers that the modules and tests of this package share. They are not exported from the package.

export { bivariateNormalCdf, bivariateNormalDensity, bivariateNormalLogDensitySlope } from "./bivariate-normal.js";
export { symmetricEigen } from "./eigen.js";
export { numericalHessian } from "./hessian.js";
export {
  addPairDerivatives,
  addWeightDerivatives,
  type FirstOrderRows,
  independenceStatistic,
  limitedInformationStatistic,
  MAX_MOMENT_ITEMS,
  type MomentTables,
  modelMoments,
  pairRow,
  type QuadraticForm,
  residualCorrelationRoot,
  rmseaInterval,
  rowCount,
  type SampleMoments,
  type ScoreMoments,
  sampleMoments,
  tableMoments,
} from "./limited-information.js";
export { invert, logDeterminantOfFactor } from "./linear-algebra.js";
export {
  addOuterProduct,
  identity,
  mirror,
  multiply,
  symmetrized,
  trace,
  traceOfProduct,
  transpose,
} from "./matrix.js";
export {
  type CommunalityObjective,
  minimizeCommunalities,
  minres,
  minresLoadings,
  principalAxes,
  projectorHessian,
  startingCommunalities,
} from "./minres.js";
export { noncentralChiSquareCdf } from "./noncentral-chi-square.js";
export { type HessianFunction, type NewtonOptions, newton } from "./optimize.js";
export { orthonormalBasis } from "./orthonormal-basis.js";
export { pearsonMatrix } from "./pearson.js";
export { poissonFit } from "./poisson-fit.js";
export { pseudoInverse } from "./pseudo-inverse.js";
export { adaptiveIntegral, gaussHermite, type Quadrature, rectangular } from "./quadrature.js";
export { findIncreasingRoot, findRoot } from "./root.js";
export { type RotationMethod, type RotationOptions, resolveRotation, rotateLoadingMatrix } from "./rotation.js";
export {
  correlation,
  covariance,
  largestChange,
  roundHalfToEven,
  sampleQuantile,
  standardDeviation,
  variance,
} from "./statistics.js";
