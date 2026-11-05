import { DataError, describe, isRecord } from "@measurekit/core";
import { checkCorrelationMatrix, minres } from "../math/internal.js";
import type { ExploratoryFactors } from "./exploratory-factors.js";

/** The loadings of items on a general factor and on group factors orthogonal to it, by the Schmid-Leiman transformation. */
export interface SchmidLeiman {
  /** The item ids, in the order of the factors' items. */
  readonly itemIds: readonly string[];
  /** Each item's loading on the general factor. */
  readonly general: readonly number[];
  /** Items by group factors: each item's loading on each group factor, less what the general factor takes. */
  readonly specific: readonly (readonly number[])[];
  /** The communality of each item, which the transformation keeps. */
  readonly communalities: readonly number[];
  /** Whether the one-factor fit to the correlations of the group factors converged. */
  readonly converged: boolean;
}

/** The fewest group factors: with two, the general factor's loadings are not identified without a constraint. */
export const MIN_GROUP_FACTORS = 3;
/**
 * The change of a communality at which the general factor's fit stops: the matrix of the group factors is small, and
 * Newton's method reaches rounding in a few more steps.
 */
const GENERAL_TOLERANCE = 1e-12;

/**
 * Returns the Schmid-Leiman transformation of an oblique factor solution: a general factor and group factors orthogonal
 * to it and to each other, as psych's `schmid`.
 *
 * @remarks
 * The one-factor minres solution of the correlations Φ of the group factors gives each group factor's loading g_k on the
 * general factor. An item's general loading is Σ_k λ_ik g_k, and its loading on group factor k is λ_ik √(1 - g_k²), with
 * the sign of λ_ik, and 0 when g_k² exceeds 1. The items' communalities are those of the solution. An orthogonal
 * solution, whose Φ is the identity, has no general factor: its general loadings are 0.
 *
 * @param factors - An oblique solution of at least 3 factors, as `exploratoryFactorAnalysis` returns it.
 * @returns The general and group loadings and the communalities of the items.
 * @throws `DataError` when the solution is not an object with item ids, loadings of at least 3 factors, finite
 * communalities and a correlation matrix of the factors.
 * @see {@link https://doi.org/10.1007/BF02289209 | Schmid and Leiman (1957)}
 * @see {@link https://doi.org/10.4324/9781410601087 | McDonald (1999)}
 */
export function schmidLeiman(factors: ExploratoryFactors): SchmidLeiman {
  checkFactors(factors);
  return transformation(factors);
}

/** Returns the transformation of a valid solution of at least 3 factors. */
export function transformation({ itemIds, loadings, correlations, communalities }: ExploratoryFactors): SchmidLeiman {
  const size = correlations.length;
  const fit = minres(Float64Array.from(correlations.flat()), size, 1, { tolerance: GENERAL_TOLERANCE });
  const g = Array.from(fit.loadings);
  const shrink = g.map((loading) => Math.sqrt(Math.max(1 - loading * loading, 0)));
  return {
    itemIds,
    general: loadings.map((row) => row.reduce((sum, loading, k) => sum + loading * (g[k] ?? 0), 0)),
    specific: loadings.map((row) => row.map((loading, k) => loading * (shrink[k] ?? 0))),
    communalities,
    converged: fit.converged,
  };
}

/**
 * Checks a factor solution that may come from JSON.
 *
 * @throws `DataError` naming what is wrong.
 */
function checkFactors(factors: unknown): asserts factors is ExploratoryFactors {
  if (!isRecord(factors)) {
    throw new DataError(
      `The factors must be an object, as exploratoryFactorAnalysis returns them, found ${describe(factors)}.`,
    );
  }
  const { itemIds, loadings, correlations, communalities } = factors;
  if (!(Array.isArray(itemIds) && itemIds.every((id) => typeof id === "string"))) {
    throw new DataError(`The factors' itemIds must be an array of item ids, found ${describe(itemIds)}.`);
  }
  const items = itemIds.length;
  const count = checkRows(loadings, items, "loadings");
  if (count < MIN_GROUP_FACTORS) {
    throw new DataError(
      `The Schmid-Leiman transformation needs at least ${MIN_GROUP_FACTORS} group factors, as two do not identify the general factor, found ${count}.`,
    );
  }
  checkCorrelationMatrix(correlations, "the factors' correlations", count);
  if (!(Array.isArray(communalities) && communalities.length === items && communalities.every(Number.isFinite))) {
    throw new DataError(
      `The factors' communalities must be ${items} finite numbers, one per item, found ${describe(communalities)}.`,
    );
  }
}

/**
 * Checks that a value is an array of `rows` arrays of one length of finite numbers, and returns that length.
 *
 * @throws `DataError` naming the field when it is not.
 */
function checkRows(value: unknown, rows: number, field: string): number {
  if (!Array.isArray(value)) {
    throw new DataError(`The factors' ${field} must be an array of rows of numbers, found ${describe(value)}.`);
  }
  const list: readonly unknown[] = value;
  if (list.length !== rows) throw new DataError(`The factors' ${field} must have ${rows} rows, found ${list.length}.`);
  const width = Array.isArray(list[0]) ? list[0].length : 0;
  list.forEach((row, i) => {
    if (!Array.isArray(row) || row.length !== width || !row.every(Number.isFinite)) {
      throw new DataError(
        `Row ${i + 1} of the factors' ${field} must be ${width} finite numbers, found ${describe(row)}.`,
      );
    }
  });
  return width;
}
