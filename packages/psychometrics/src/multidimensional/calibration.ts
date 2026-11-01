import { checkChoice, checkOptions, checkSeed, checkStopping, DataError } from "@measurekit/core";
import type { ResponseMatrix } from "../data/response-matrix.js";
import type { ResponseRecords } from "../data/response-records.js";
import type { StandardErrorMethod } from "../irt/calibration-options.js";
import { type CompressedResponses, compressResponses } from "../irt/responses.js";
import type { CalibrationMethod, MultidimensionalCalibrationOptions } from "./calibration-options.js";
import { type ItemState, setUp } from "./calibration-setup.js";
import { pairsOf } from "./correlation.js";
import { type Design, designOf } from "./design.js";
import { calibrateByEm, MAX_EM_DIMENSIONS } from "./em.js";
import { standardizedLoadings } from "./loadings.js";
import { calibrateByMhrm } from "./mhrm.js";
import type { MultidimensionalTestItem } from "./model.js";
import type { Settings, Solution } from "./solution.js";

/** An item with its calibrated parameters. */
export interface CalibratedMultidimensionalItem<TParameters> extends MultidimensionalTestItem<TParameters> {
  /**
   * Standard errors of the estimated parameters by name: `a1` to `aD`, `intercept` or `intercept1` upward, and `c`.
   * Slopes fixed at 0 by the pattern of loadings are not listed. `null` without standard errors: with the option
   * `standardErrors` `"none"`, and when the information matrix cannot be inverted or gives a variance that is not
   * positive.
   */
  readonly standardErrors: Readonly<Record<string, number>> | null;
}

/** The distribution of theta in one group of persons, estimated with the items. */
export interface MultidimensionalGroup {
  /** The group's label in the option `groups`; `"all"` for the persons of a calibration with fixed items and no groups. */
  readonly group: string;
  /** The mean of each dimension, in the order of the dimensions. */
  readonly means: readonly number[];
  /** The covariance matrix of the dimensions, by rows. */
  readonly covariance: readonly (readonly number[])[];
}

/** Item parameters and correlations of dimensions estimated by marginal maximum likelihood, with fit statistics. */
export interface MultidimensionalCalibration<TParameters> {
  /** The names of the dimensions, in the order of the slopes. */
  readonly dimensions: readonly string[];
  /** The items, in item order. */
  readonly items: readonly CalibratedMultidimensionalItem<TParameters>[];
  /**
   * The correlations between dimensions, by rows: the identity unless a confirmatory model estimates them. With several
   * groups or fixed items, those of the first group.
   */
  readonly correlations: readonly (readonly number[])[];
  /**
   * The standard errors of the correlations, by rows with 0 on the diagonal; `null` when the correlations are not
   * estimated or the calibration has no standard errors.
   */
  readonly correlationStandardErrors: readonly (readonly number[])[] | null;
  /**
   * The distribution of theta in each group, in order of first appearance in `groups`. The first group has means 0 and
   * variances 1 unless fixed items set the scale. Absent without the options `groups` and `fixed`.
   */
  readonly groups?: readonly MultidimensionalGroup[];
  /**
   * The standardized loadings, items by dimensions: λ = (a / 1.702) / √(1 + (a / 1.702)ᵀ Φ (a / 1.702)), with the
   * slopes scaled by the standard deviations of the dimensions in the first group.
   */
  readonly loadings: readonly (readonly number[])[];
  /** The communality of each item: λᵀ Φ λ, with Φ the correlations of the first group. */
  readonly communalities: readonly number[];
  /** The marginal log-likelihood at the estimates, without the priors. */
  readonly logLikelihood: number;
  /**
   * Akaike's information criterion: -2 logLikelihood + 2p, with p the number of estimated item parameters,
   * correlations and, with several groups or fixed items, means and variances.
   */
  readonly aic: number;
  /** The Bayesian information criterion: -2 logLikelihood + p ln n, with n the number of persons with at least one response. */
  readonly bic: number;
  /** Whether the calibration reached `tolerance` within `maxIterations`. */
  readonly converged: boolean;
  /** EM iterations run, or the cycles of the last stage of MH-RM. */
  readonly iterations: number;
}

/** The default `tolerance` of EM. */
export const EM_TOLERANCE = 1e-4;
/** The default `maxIterations` of EM. */
export const EM_MAX_ITERATIONS = 500;
/** The defaults of MH-RM: mirt's `TOL` and `NCYCLES`. */
const MHRM_TOLERANCE = 1e-3;
const MHRM_MAX_ITERATIONS = 2000;
const STANDARD_ERROR_METHODS: readonly StandardErrorMethod[] = ["none", "hessian", "cross-product"];
const METHODS: readonly CalibrationMethod[] = ["em", "mhrm"];

/**
 * Estimates the item parameters of a multidimensional model, and the correlations between its dimensions, by marginal
 * maximum likelihood: by the Bock-Aitkin EM algorithm with theta integrated by quadrature over a grid, up to three
 * dimensions, or by the Metropolis-Hastings Robbins-Monro algorithm, in any number.
 *
 * @remarks
 * Exploratory models (`dimensions`) have every item load on every dimension, and confirmatory ones (`loadings`) only
 * on the dimensions named for them; the dimensions have mean 0 and variance 1. With priors, the estimates are posterior
 * modes. Missing responses are skipped; persons without responses change nothing.
 *
 * EM: the grid has `points` equally spaced nodes per dimension from -6 to 6, weighted by the normal density of the
 * correlations, as mirt. The M-step takes a Fisher-scoring step for each item and maximizes the correlations, which keep
 * a unit diagonal and stay positive definite, with BFGS on the weights of the grid, so that it maximizes the reported
 * likelihood. SQUAREM accelerates the iterations, to the fixed points of plain EM. Standard errors come from Louis'
 * formula for the observed information of the marginal log-posterior, or from the cross-product of the persons' score
 * vectors.
 *
 * MH-RM: each cycle draws every person's theta from its posterior at the current estimates by Metropolis-Hastings and
 * moves the estimates by a Robbins-Monro step on the complete-data score, in the stages and with the gains of mirt's
 * defaults. The estimates carry Monte Carlo error, and a seed reproduces them. The log-likelihood is estimated by
 * importance sampling, and the standard errors, `"hessian"` only, by Louis' formula from 400 further cycles of draws.
 * `iterations` counts the cycles of the last stage, after the 250 of the first two.
 *
 * Groups: with `groups`, each group has its own means and covariance, and the items have the same parameters in every
 * group, as `mirt::multipleGroup` with invariant slopes and intercepts. The first group has means 0 and variances 1,
 * and the means, variances and, in a model with correlated dimensions, correlations of the others are estimated. With
 * `fixed`, the fixed items set the scale instead and every group's means and covariance are estimated. The groups must
 * share items, directly or through other groups, so that one scale holds them. A distribution that runs to the edge of
 * the grid, a mean outside it or a standard deviation below the spacing of its nodes or above its range, leaves
 * `converged` `false`.
 *
 * An exploratory solution is determined up to the signs of its dimensions; {@link rotateLoadings} rotates it.
 *
 * @throws `DataError` when the data or an option is invalid, there are no responses, neither or both of `dimensions`
 * and `loadings` are given, an item is not in `loadings` or lacks responses in a category up to its highest or has more
 * categories than its model allows; with EM, when there are more than 3 dimensions or the grid has more than 10⁶ nodes,
 * a group has no person with a response, the groups share no item or a fixed item is invalid, or the fixed items leave
 * a dimension without a scale; with MH-RM, when `points`, `groups`, `fixed` or the standard errors `"cross-product"`
 * are given.
 */
export function calibrateMultidimensional<TParameters>(
  data: ResponseMatrix | ResponseRecords,
  options: MultidimensionalCalibrationOptions<TParameters>,
): MultidimensionalCalibration<TParameters> {
  checkOptions(options);
  const responses = compressResponses(data);
  const { standardErrors = "none", seed = 0 } = options;
  checkChoice(standardErrors, "The option standardErrors", STANDARD_ERROR_METHODS);
  checkSeed(seed);
  const respondents = Array.from(responses.personIds, (_, person) => {
    return (responses.offsets[person + 1] ?? 0) > (responses.offsets[person] ?? 0);
  }).filter(Boolean).length;
  if (respondents === 0) throw new DataError("Calibration needs at least one response.");
  const design = designOf(options, responses.itemIds);
  const { method = defaultMethod(design.dimensionNames.length) } = options;
  checkChoice(method, "The option method", METHODS);
  const em = method === "em";
  const {
    tolerance = em ? EM_TOLERANCE : MHRM_TOLERANCE,
    maxIterations = em ? EM_MAX_ITERATIONS : MHRM_MAX_ITERATIONS,
  } = options;
  checkStopping(tolerance, maxIterations);
  const settings: Settings = { tolerance, maxIterations, seed, standardErrors, points: options.points };
  const solution = em
    ? calibrateByEm(responses, options, design, settings)
    : mhrm(responses, options, design, settings);
  return assemble(design, solution, respondents);
}

/** Estimates by MH-RM, which takes neither a grid nor the cross-product standard errors. */
function mhrm<TParameters>(
  responses: CompressedResponses,
  options: MultidimensionalCalibrationOptions<TParameters>,
  design: Design,
  settings: Settings,
): Solution<TParameters> {
  for (const name of ["points", "groups", "fixed"] as const) {
    if (options[name] !== undefined) {
      throw new DataError(`The option ${name} is for EM, which the method "mhrm" does not use.`);
    }
  }
  if (settings.standardErrors === "cross-product") {
    throw new DataError(
      'MH-RM has no posterior over a grid to cross-multiply: the option standardErrors needs "none" or "hessian" with the method "mhrm".',
    );
  }
  const items = setUp(responses, options, design, 0);
  const fit = calibrateByMhrm(responses, items, design, {
    tolerance: settings.tolerance,
    maxIterations: settings.maxIterations,
    standardErrors: settings.standardErrors === "hessian",
    seed: settings.seed,
  });
  return {
    items,
    correlations: fit.correlations.matrix,
    correlated: design.correlated,
    latentSize: design.correlated ? pairsOf(design.dimensionNames.length).length : 0,
    groups: null,
    covariance: fit.covariance,
    logLikelihood: fit.logLikelihood,
    converged: fit.converged,
    iterations: fit.iterations,
  };
}

/** EM up to the dimensions that it calibrates, MH-RM above. */
function defaultMethod(dimensions: number): CalibrationMethod {
  return dimensions > MAX_EM_DIMENSIONS ? "mhrm" : "em";
}

/** Builds the calibration from the solution of a method. */
export function assemble<TParameters>(
  design: Design,
  {
    items,
    correlations,
    correlated,
    latentSize,
    groups,
    covariance,
    logLikelihood,
    converged,
    iterations,
  }: Solution<TParameters>,
  respondents: number,
): MultidimensionalCalibration<TParameters> {
  const dimensions = design.dimensionNames.length;
  const itemSize = items.reduce((sum, item) => sum + item.free.length, 0);
  const size = itemSize + latentSize;
  const rows = matrixRows(correlations, dimensions);
  const [first] = groups ?? [];
  const { loadings, communalities } = standardizedLoadings(
    items.map((item) =>
      Array.from(
        item.vector.subarray(0, dimensions),
        (slope, d) => slope * Math.sqrt(first?.covariance[d * dimensions + d] ?? 1),
      ),
    ),
    rows,
  );
  return {
    dimensions: design.dimensionNames,
    items: items.map((item, i) => ({
      id: item.id,
      model: item.model,
      parameters: item.parameters,
      standardErrors: covariance === null || item.fixed ? null : itemErrors(item, i, items, covariance, size),
    })),
    correlations: rows,
    correlationStandardErrors:
      covariance === null || !correlated
        ? null
        : correlationErrors(pairsOf(dimensions), dimensions, covariance, size, itemSize),
    ...(groups === null
      ? {}
      : {
          groups: groups.map(({ group, means, covariance: matrix }) => ({
            group,
            means: Array.from(means),
            covariance: matrixRows(matrix, dimensions),
          })),
        }),
    loadings,
    communalities,
    logLikelihood,
    aic: -2 * logLikelihood + 2 * size,
    bic: -2 * logLikelihood + size * Math.log(respondents),
    converged,
    iterations,
  };
}

/** Returns the standard errors of an item's estimated entries by name. */
function itemErrors<TParameters>(
  item: ItemState<TParameters>,
  index: number,
  items: readonly ItemState<TParameters>[],
  covariance: Float64Array,
  size: number,
): Record<string, number> {
  let start = 0;
  for (let i = 0; i < index; i++) start += items[i]?.free.length ?? 0;
  return Object.fromEntries(
    Array.from(item.free, (j, l) => [item.names[j] ?? "", Math.sqrt(covariance[(start + l) * size + start + l] ?? 0)]),
  );
}

/** Returns the standard errors of the correlations as a matrix with a zero diagonal. */
function correlationErrors(
  pairs: readonly (readonly [number, number])[],
  dimensions: number,
  covariance: Float64Array,
  size: number,
  first: number,
): number[][] {
  const errors = Array.from({ length: dimensions }, () => new Array<number>(dimensions).fill(0));
  pairs.forEach(([i, j], p) => {
    const error = Math.sqrt(covariance[(first + p) * size + first + p] ?? 0);
    const row = errors[i];
    const column = errors[j];
    if (row !== undefined) row[j] = error;
    if (column !== undefined) column[i] = error;
  });
  return errors;
}

function matrixRows(matrix: Float64Array, dimensions: number): number[][] {
  return Array.from({ length: dimensions }, (_, i) =>
    Array.from(matrix.subarray(i * dimensions, (i + 1) * dimensions)),
  );
}
