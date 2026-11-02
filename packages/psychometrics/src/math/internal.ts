// Helpers that the modules and tests of this package share. They are not exported from the package.

export { bivariateNormalCdf, bivariateNormalDensity, bivariateNormalLogDensitySlope } from "./bivariate-normal.js";
export { symmetricEigen } from "./eigen.js";
export { numericalHessian } from "./hessian.js";
export { invert, invertPositiveDefinite } from "./linear-algebra.js";
export { identity, mirror, multiply, symmetrized, trace, traceOfProduct, transpose } from "./matrix.js";
export { minres, minresLoadings } from "./minres.js";
export { normalCdf, normalDensity } from "./normal.js";
export { type HessianFunction, type NewtonOptions, newton } from "./optimize.js";
export { pearsonMatrix } from "./pearson.js";
export { pseudoInverse } from "./pseudo-inverse.js";
export { gaussHermite, type Quadrature, rectangular } from "./quadrature.js";
export { findIncreasingRoot, findRoot } from "./root.js";
export { type RotationMethod, type RotationOptions, resolveRotation, rotateLoadingMatrix } from "./rotation.js";
export { correlation, covariance, standardDeviation, variance } from "./statistics.js";
