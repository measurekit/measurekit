// Helpers that the modules and tests of this package share. They are not exported from the package.

export { createProblem, type FlatDirection, linear, penalizedNewton } from "./arrow-system.js";
export { expectationMaximization } from "./expectation-maximization.js";
export { indexOfMax, minimum } from "./extremes.js";
export { forwardBackward } from "./hidden-markov.js";
export { choleskySolve } from "./linear-algebra.js";
export { outcomeLogLikelihood } from "./logistic.js";
export {
  type GroupIntercepts,
  type LogisticRegression,
  logisticRegression,
  MAX_DENSE_COEFFICIENTS,
} from "./logistic-regression.js";
export { newtonCg } from "./newton-cg.js";
export { permutation } from "./permutation.js";
export { type Jet, quinticDerivative, quinticHermite, quinticMaximum } from "./quintic.js";
export { rank } from "./rank.js";
export { reflectedWalk } from "./reflected-walk.js";
export type { SparseMatrix } from "./sparse-matrix.js";
