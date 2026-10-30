import { createProblem, type FlatDirection, logLikelihood, penalizedNewton } from "./arrow-system.js";
import { randomIntercepts } from "./laplace.js";
import type { SparseMatrix } from "./sparse-matrix.js";

/** An intercept for each group of rows, such as a learner's proficiency. */
export interface GroupIntercepts {
  /** The group of each row, from 0 to `count - 1`. */
  readonly indices: Int32Array;
  /** The number of groups. */
  readonly count: number;
  /**
   * `"fixed"`: a free intercept per group. `"random"`: intercepts from N(0, σ²), σ estimated by maximizing the Laplace
   * approximation of the marginal likelihood, as `lme4::glmer(nAGQ = 1)` does.
   */
  readonly kind: "fixed" | "random";
  /**
   * With fixed intercepts and no ridge: the directions along which the likelihood is flat, each raising some groups'
   * intercepts by c and the coefficients by c f. The fit sets the mean intercept of each one's groups to 0.
   */
  readonly flats?: readonly FlatDirection[];
}

/** Options of {@link logisticRegression}. */
export interface LogisticRegressionOptions {
  /** Intercepts by group of rows. */
  readonly groups?: GroupIntercepts;
  /**
   * λ ≥ 0 of the penalty λ / 2 Σ b², over the coefficients and fixed group intercepts.
   *
   * @defaultValue `0`
   */
  readonly ridge?: number;
  /** The largest absolute gradient component at which the fit has converged. */
  readonly tolerance: number;
  /** The most iterations. */
  readonly maxIterations: number;
}

/** A logistic regression fitted by maximum likelihood, penalized with a ridge. */
export interface LogisticRegression {
  /** The coefficients, one per column of the design. */
  readonly coefficients: Float64Array;
  /** The group intercepts: estimates when fixed, conditional modes when random; empty without groups. */
  readonly groupIntercepts: Float64Array;
  /** σ of random group intercepts, otherwise `null`. */
  readonly sd: number | null;
  /**
   * Σ y ln p + (1 - y) ln(1 - p) at the estimates, without the ridge penalty; with random intercepts, the Laplace
   * approximation of the marginal log-likelihood.
   */
  readonly logLikelihood: number;
  /** Newton iterations, or quasi-Newton iterations of the marginal likelihood with random intercepts. */
  readonly iterations: number;
  /** Whether the gradient reached the tolerance. */
  readonly converged: boolean;
}

/**
 * Fits P(y = 1) = logistic(x β + u_g) to outcomes y in [0, 1], one per row of `design`, with u_g the intercept of the
 * row's group, by Newton's method (iteratively reweighted least squares) from β = 0, u = 0.
 *
 * @remarks
 * The Hessian of fixed group intercepts and coefficients is an arrow: diagonal in the groups. Each step solves it by
 * the Schur complement of the group block, so time and memory grow linearly with the groups and rows and as the cube of
 * the coefficients.
 *
 * Random intercepts start from the penalized fit at σ = 1, then maximize the Laplace approximation of the marginal
 * likelihood over β and ln σ by quasi-Newton steps: the exact gradient, and the Schur complement as the curvature of β.
 */
export function logisticRegression(
  design: SparseMatrix,
  outcomes: Float64Array,
  options: LogisticRegressionOptions,
): LogisticRegression {
  const { groups, ridge = 0, tolerance, maxIterations } = options;
  const problem = createProblem(design, outcomes, groups?.indices, groups?.count ?? 0);
  if (groups?.kind === "random") {
    const fit = randomIntercepts(problem, ridge, tolerance, maxIterations);
    return { ...fit, groupIntercepts: fit.modes };
  }
  const beta = new Float64Array(design.columnCount);
  const intercepts = new Float64Array(problem.groupCount);
  const flats = ridge === 0 ? (groups?.flats ?? []) : [];
  const groupPenalty = groups === undefined ? 0 : ridge;
  const fit = penalizedNewton(problem, beta, intercepts, { ridge, groupPenalty, flats, tolerance, maxIterations });
  return {
    coefficients: beta,
    groupIntercepts: intercepts,
    sd: null,
    logLikelihood: logLikelihood(problem, beta, intercepts),
    ...fit,
  };
}
