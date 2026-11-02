import {
  checkChoice,
  checkNonNegative,
  checkOptions,
  checkPositiveInteger,
  checkStopping,
  DataError,
} from "@measurekit/core";
import type { ResponseMatrix } from "../data/internal.js";
import {
  minimizeCommunalities,
  minres,
  resolveRotation,
  rotateLoadingMatrix,
  startingCommunalities,
} from "../math/internal.js";
import type { CodedItem } from "./coded-items.js";
import { type FactorFit, factorFit } from "./factor-fit.js";
import { likelihoodObjective } from "./factor-likelihood.js";
import { ITEM_CORRELATIONS, type ItemCorrelation, positiveDefiniteCorrelations } from "./item-correlations.js";
import { DEFAULT_CORRECTION, type PolychoricOptions } from "./polychoric.js";

/** The rotation of the factors: none, orthogonal (Kaiser-normalized varimax) or oblique (oblimin). */
export type FactorRotation = "none" | "varimax" | "oblimin";

/** The extraction of the factors: minimum residuals, or maximum likelihood. */
export type FactorExtraction = "minres" | "maximum-likelihood";

/** Options of {@link exploratoryFactorAnalysis}. */
export interface ExploratoryFactorOptions extends PolychoricOptions {
  /** The number of factors, a positive integer, at most what the number of items identifies. */
  readonly factors: number;
  /**
   * The correlation of the items: `"pearson"` for scores treated as continuous, over the persons who answered both
   * items of each pair, or `"polychoric"` for item responses.
   *
   * @defaultValue `"pearson"`
   */
  readonly correlation?: ItemCorrelation;
  /**
   * The rotation of the factors: `"varimax"` is orthogonal, with each item's loadings scaled to unit length for the
   * search, and `"oblimin"` oblique. A solution of one factor is not rotated.
   *
   * @defaultValue `"oblimin"`
   */
  readonly rotation?: FactorRotation;
  /**
   * The extraction: `"minres"`, the least squared residuals of the correlations, or `"maximum-likelihood"`, Jöreskog's
   * likelihood concentrated on the uniquenesses.
   *
   * @defaultValue `"minres"`
   */
  readonly extraction?: FactorExtraction;
  /**
   * The largest change of a communality in an iteration at which the fit stops.
   *
   * @defaultValue `1e-6`
   */
  readonly tolerance?: number;
  /**
   * The most iterations of the fit.
   *
   * @defaultValue `100`
   */
  readonly maxIterations?: number;
}

/** The factors of a set of items. */
export interface ExploratoryFactors {
  /** The item ids, in the matrix's order. */
  readonly itemIds: readonly string[];
  /** The loadings, items by factors. The factors are in the order of the variance they account for. */
  readonly loadings: readonly (readonly number[])[];
  /** The communality of each item, the part of its variance that the factors share. */
  readonly communalities: readonly number[];
  /** The correlations between the factors, by rows: the identity unless the rotation is oblique. */
  readonly correlations: readonly (readonly number[])[];
  /** Whether the fit and the rotation both reached their tolerances within their iterations. */
  readonly converged: boolean;
  /** The iterations of the fit. */
  readonly iterations: number;
  /** Whether the correlations of the items were not positive definite and were smoothed. */
  readonly smoothed: boolean;
  /**
   * The fit of the factors, as `psych::fa` reports it for both extractions; `null` for the polychoric correlation, whose
   * chi-square has no reference distribution, and when the model's correlations are not positive definite.
   */
  readonly fit: FactorFit | null;
}

const ROTATIONS = ["none", "varimax", "oblimin"] as const;
const EXTRACTIONS = ["minres", "maximum-likelihood"] as const;
const DEFAULT_TOLERANCE = 1e-6;
const DEFAULT_MAX_ITERATIONS = 100;

/**
 * Returns the factors of the items of a response matrix by the minimum residual method or maximum likelihood, rotated,
 * with the fit of the factors.
 *
 * @remarks
 * The correlations of the items, Pearson's over the persons who answered both items of each pair or polychoric, are
 * smoothed when they are not positive definite, as psych's `cor.smooth` does. The loadings are those of
 * `psych::fa(fm = "minres")`: the communalities that leave the least squared residuals, with the uniqueness of an item
 * at least 0.005, and the loadings the leading eigenvectors of the correlations with those communalities on the
 * diagonal; or of `fa(fm = "ml")`: the uniquenesses from 0.005 to 1 that maximize Jöreskog's likelihood, by the same
 * projected Newton search with the likelihood's Hessian at a perfect fit. They are rotated by gradient projection from
 * the unrotated solution, oblimin without normalization or Kaiser-normalized varimax. The sign of each factor makes its
 * loadings sum to a positive number, and the factors are in the order of the variance they account for, the diagonal
 * of Φ ΛᵀΛ, as psych's. A solution beyond what the items identify, more factors than the largest number with at least as
 * many correlations as parameters, throws. The fit of Pearson correlations is psych's: the likelihood ratio statistic,
 * its RMSEA, TLI and BIC for the persons of the matrix, and the root mean square of the residual correlations.
 * `calibrateMultidimensional` with `dimensions` estimates the same model from the responses, by full information and far
 * more slowly, and gives the item parameters of the model.
 *
 * @param data - The responses of the persons.
 * @param options - The number of factors, the correlation, the extraction, the rotation, the stopping rules and the
 *   correction for continuity.
 * @returns The loadings and communalities of the items, the correlations of the factors, whether the extraction
 *   converged, the iterations run and the fit.
 * @throws `DataError` when the matrix is invalid, has fewer than three items or an item with fewer than two observed
 * categories, a pair of items has too few persons for a correlation, an option is invalid, or `factors` is more than the
 * items identify.
 * @see {@link https://doi.org/10.1037/1082-989X.4.3.272 | Fabrigar et al. (1999)}
 * @see {@link https://doi.org/10.1007/BF02289658 | Jöreskog (1967)}
 */
export function exploratoryFactorAnalysis(data: ResponseMatrix, options: ExploratoryFactorOptions): ExploratoryFactors {
  checkOptions(options);
  const {
    factors,
    correlation = "pearson",
    extraction = "minres",
    rotation = "oblimin",
    tolerance = DEFAULT_TOLERANCE,
    maxIterations = DEFAULT_MAX_ITERATIONS,
    correction = DEFAULT_CORRECTION,
  } = options;
  checkPositiveInteger(factors, "The option factors");
  checkChoice(correlation, "The option correlation", ITEM_CORRELATIONS);
  checkChoice(extraction, "The option extraction", EXTRACTIONS);
  checkChoice(rotation, "The option rotation", ROTATIONS);
  checkStopping(tolerance, maxIterations);
  checkNonNegative(correction, "The option correction");
  const { items, correlations, smoothed } = positiveDefiniteCorrelations(
    data,
    correlation,
    correction,
    "Exploratory factor analysis",
  );
  const size = items.length;
  checkIdentified(size, factors);
  const solution = factorSolution(correlations, size, factors, rotation, { tolerance, maxIterations }, extraction);
  const fit =
    correlation === "pearson"
      ? factorFit(correlations, size, modelOf(solution), factors, data.responses.length, pairCounts(items))
      : null;
  return { itemIds: data.itemIds, ...solution, smoothed, fit };
}

/** The factors of a correlation matrix. */
export type FactorSolution = Omit<ExploratoryFactors, "itemIds" | "smoothed" | "fit">;

/** Returns the model's correlations off the diagonal, Λ Φ Λᵀ, by rows. */
function modelOf({ loadings, correlations }: FactorSolution): Float64Array {
  const size = loadings.length;
  const model = new Float64Array(size * size);
  loadings.forEach((row, i) => {
    loadings.forEach((other, l) => {
      let sum = 0;
      row.forEach((a, j) => {
        other.forEach((b, k) => {
          sum += a * (correlations[j]?.[k] ?? 0) * b;
        });
      });
      model[i * size + l] = sum;
    });
  });
  return model;
}

/** Returns the number of persons who answered each pair of items, by rows; the diagonal those who answered each item. */
function pairCounts(items: readonly CodedItem[]): Float64Array {
  const size = items.length;
  const counts = new Float64Array(size * size);
  items.forEach(({ values: first }, i) => {
    items.forEach(({ values: second }, j) => {
      let count = 0;
      for (let person = 0; person < first.length; person++) {
        if (!Number.isNaN(first[person]) && !Number.isNaN(second[person])) count++;
      }
      counts[i * size + j] = count;
    });
  });
  return counts;
}

/**
 * Returns the minimum residual or maximum likelihood solution of a correlation matrix (`size` rows, by rows, 1 on the
 * diagonal), rotated, with each factor's sign and the order of the factors as `exploratoryFactorAnalysis` gives them.
 * `factors` is a positive integer that the matrix identifies, and the matrix is positive definite for maximum likelihood.
 */
export function factorSolution(
  correlations: Float64Array,
  size: number,
  factors: number,
  rotation: FactorRotation,
  stopping: { readonly tolerance: number; readonly maxIterations: number },
  extraction: FactorExtraction = "minres",
): FactorSolution {
  const fit =
    extraction === "minres"
      ? minres(correlations, size, factors, stopping)
      : minimizeCommunalities(
          likelihoodObjective(correlations, size, factors),
          startingCommunalities(correlations, size),
          size,
          factors,
          stopping,
        );
  const rotated = rotate(fit.loadings, size, factors, rotation);
  const order = byVariance(rotated.loadings, rotated.correlations, size, factors);
  return {
    loadings: Array.from({ length: size }, (_, i) => order.map((k) => rotated.loadings[i * factors + k] ?? Number.NaN)),
    communalities: Array.from(fit.communalities),
    correlations: order.map((j) => order.map((k) => rotated.correlations[j * factors + k] ?? Number.NaN)),
    converged: fit.converged && rotated.converged,
    iterations: fit.iterations,
  };
}

/**
 * Checks that the items give at least as many correlations as the factors have parameters: the loadings, less the
 * rotations that leave the model as it is.
 *
 * @throws `DataError` when they do not.
 */
function checkIdentified(items: number, factors: number): void {
  const most = Math.floor((2 * items + 1 - Math.sqrt(8 * items + 1)) / 2);
  if (factors > most) {
    throw new DataError(
      `${items} items identify at most ${most} factors, as there must be as many correlations as loadings less the rotations, found ${factors}.`,
    );
  }
}

/** The loadings after a rotation, with the correlations of the factors. */
interface Rotated {
  readonly loadings: Float64Array;
  readonly correlations: Float64Array;
  readonly converged: boolean;
}

/** Rotates the loadings; one factor, or none asked for, leaves them as they are. */
function rotate(loadings: Float64Array, size: number, factors: number, rotation: FactorRotation): Rotated {
  const identity = Float64Array.from({ length: factors * factors }, (_, k) => (k % (factors + 1) === 0 ? 1 : 0));
  if (factors === 1 || rotation === "none") return { loadings, correlations: identity, converged: true };
  const result = rotateLoadingMatrix(
    loadings,
    size,
    factors,
    resolveRotation({ method: rotation, normalize: rotation === "varimax" }),
    "sum",
  );
  return {
    loadings: result.loadings,
    correlations: rotation === "varimax" ? identity : result.correlations,
    converged: result.converged,
  };
}

/** Returns the factors in the order of the variance they account for, the diagonal of Φ ΛᵀΛ, largest first. */
function byVariance(loadings: Float64Array, correlations: Float64Array, size: number, factors: number): number[] {
  const accounted = Array.from({ length: factors }, (_, k) => {
    let total = 0;
    for (let j = 0; j < factors; j++) {
      let product = 0;
      for (let i = 0; i < size; i++) product += (loadings[i * factors + j] ?? 0) * (loadings[i * factors + k] ?? 0);
      total += (correlations[k * factors + j] ?? 0) * product;
    }
    return total;
  });
  return Array.from(accounted.keys()).sort((a, b) => (accounted[b] ?? 0) - (accounted[a] ?? 0) || a - b);
}
