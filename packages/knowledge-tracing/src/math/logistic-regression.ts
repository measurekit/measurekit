import { logistic } from "@measurekit/core";
import { createProblem, type FlatDirection, linear, logLikelihood, penalizedNewton, rowLoss } from "./arrow-system.js";
import { randomIntercepts } from "./laplace.js";
import { type NewtonObjective, newtonCg } from "./newton-cg.js";
import type { SparseMatrix } from "./sparse-matrix.js";
import { type StandardErrors, standardErrors } from "./standard-errors.js";

/**
 * The most coefficients that a fit solves with dense matrices: those without a ridge, which need the Hessian's
 * factorization, and those with random intercepts. A matrix of that many coefficients takes 200 MB.
 */
export const MAX_DENSE_COEFFICIENTS = 5_000;

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
  /**
   * Whether to give the standard errors, which a fit without a ridge has.
   *
   * @defaultValue `false`
   */
  readonly standardErrors?: boolean;
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
  /**
   * The standard errors of the coefficients, and of fixed group intercepts. `null` unless asked, with a ridge, and
   * where the information is singular; random group intercepts have none.
   */
  readonly standardErrors: StandardErrors | null;
}

/**
 * Fits P(y = 1) = logistic(x β + u_g) to outcomes y in [0, 1], one per row of `design`, with u_g the intercept of the
 * row's group, by Newton's method (iteratively reweighted least squares) from β = 0, u = 0.
 *
 * @remarks
 * Without a ridge, the Hessian of fixed group intercepts and coefficients is an arrow: diagonal in the groups. Each step
 * solves it by the Schur complement of the group block, so time and memory grow linearly with the groups and rows and
 * as the cube and the square of the coefficients. With a ridge, each step is solved approximately by conjugate
 * gradients on Hessian-vector products, so that the cost per step is that of passes over the rows, however many
 * coefficients there are.
 *
 * Random intercepts start from the penalized fit at σ = 1, then maximize the Laplace approximation of the marginal
 * likelihood over β and ln σ by quasi-Newton steps: the exact gradient, and the Schur complement as the curvature of β.
 */
export function logisticRegression(
  design: SparseMatrix,
  outcomes: Float64Array,
  options: LogisticRegressionOptions,
): LogisticRegression {
  const { groups, ridge = 0, standardErrors: asked = false, tolerance, maxIterations } = options;
  if (groups?.kind !== "random" && ridge > 0) return ridgeFit(design, outcomes, options);
  const problem = createProblem(design, outcomes, groups?.indices, groups?.count ?? 0);
  if (groups?.kind === "random") {
    const fit = randomIntercepts(problem, ridge, tolerance, maxIterations);
    const errors =
      asked && ridge === 0
        ? standardErrors(problem, fit.coefficients, fit.modes, {
            groupPenalty: 1 / (fit.sd * fit.sd),
            flats: [],
            groups: false,
          })
        : null;
    return { ...fit, groupIntercepts: fit.modes, standardErrors: errors };
  }
  const beta = new Float64Array(design.columnCount);
  const intercepts = new Float64Array(problem.groupCount);
  const flats = groups?.flats ?? [];
  const fit = penalizedNewton(problem, beta, intercepts, { ridge, groupPenalty: 0, flats, tolerance, maxIterations });
  return {
    coefficients: beta,
    groupIntercepts: intercepts,
    sd: null,
    logLikelihood: logLikelihood(problem, beta, intercepts),
    standardErrors: asked ? standardErrors(problem, beta, intercepts, { groupPenalty: 0, flats, groups: true }) : null,
    ...fit,
  };
}

/** Fits with a ridge, no random intercepts, by Newton-CG over the coefficients followed by the group intercepts. */
function ridgeFit(
  design: SparseMatrix,
  outcomes: Float64Array,
  options: LogisticRegressionOptions,
): LogisticRegression {
  const { groups, ridge = 0, tolerance, maxIterations } = options;
  const p = design.columnCount;
  const objective = ridgeObjective(design, outcomes, groups, ridge);
  const x = new Float64Array(p + (groups?.count ?? 0));
  const fit = newtonCg(objective, x, { tolerance, maxIterations });
  return {
    coefficients: x.slice(0, p),
    groupIntercepts: x.slice(p),
    sd: null,
    logLikelihood: -objective.loss(x),
    standardErrors: null,
    ...fit,
  };
}

/**
 * The objective of a ridge fit over x = (β, u): the negative log-likelihood + λ / 2 (|β|² + |u|²), with Hessian-vector
 * products, and the negative log-likelihood alone.
 */
function ridgeObjective(
  design: SparseMatrix,
  outcomes: Float64Array,
  groups: GroupIntercepts | undefined,
  ridge: number,
): NewtonObjective & { loss(x: Float64Array): number } {
  const p = design.columnCount;
  const weights = new Float64Array(outcomes.length);
  const indices = groups?.indices;
  /** Returns η = x β + u_g of a row for the vector `x` of coefficients followed by the group intercepts. */
  const predictor = (row: number, x: Float64Array): number =>
    linear(design, row, x) + (indices === undefined ? 0 : (x[p + (indices[row] ?? 0)] ?? 0));
  const loss = (x: Float64Array): number => {
    let sum = 0;
    for (let row = 0; row < outcomes.length; row++) sum += rowLoss(predictor(row, x), outcomes[row] ?? 0);
    return sum;
  };
  return {
    loss,
    value(x) {
      let penalty = 0;
      for (const value of x) penalty += value * value;
      return loss(x) + (ridge / 2) * penalty;
    },
    linearize(x, gradient, diagonal) {
      gradient.fill(0);
      diagonal.fill(0);
      for (let row = 0; row < outcomes.length; row++) {
        const p1 = logistic(predictor(row, x));
        const weight = p1 * (1 - p1);
        const residual = p1 - (outcomes[row] ?? 0);
        weights[row] = weight;
        scatter(design, row, residual, gradient);
        scatter(design, row, weight, diagonal, true);
        if (indices !== undefined) {
          const g = p + (indices[row] ?? 0);
          gradient[g] = (gradient[g] ?? 0) + residual;
          diagonal[g] = (diagonal[g] ?? 0) + weight;
        }
      }
      for (let j = 0; j < x.length; j++) {
        gradient[j] = (gradient[j] ?? 0) + ridge * (x[j] ?? 0);
        diagonal[j] = (diagonal[j] ?? 0) + ridge;
      }
    },
    hessianVector(v, out) {
      out.fill(0);
      for (let row = 0; row < outcomes.length; row++) {
        const scaled = (weights[row] ?? 0) * predictor(row, v);
        scatter(design, row, scaled, out);
        if (indices !== undefined) {
          const g = p + (indices[row] ?? 0);
          out[g] = (out[g] ?? 0) + scaled;
        }
      }
      for (let j = 0; j < v.length; j++) out[j] = (out[j] ?? 0) + ridge * (v[j] ?? 0);
    },
  };
}

/** Adds scale times a row's entries to `out`, or scale times their squares. */
function scatter(design: SparseMatrix, row: number, scale: number, out: Float64Array, squared = false): void {
  for (let e = design.rowStarts[row] ?? 0; e < (design.rowStarts[row + 1] ?? 0); e++) {
    const column = design.columns[e] ?? 0;
    const value = design.values[e] ?? 0;
    out[column] = (out[column] ?? 0) + scale * (squared ? value * value : value);
  }
}
