import {
  checkChoice,
  checkNonNegative,
  checkOptions,
  checkPositiveInteger,
  checkRandom,
  checkUnitInterval,
  DataError,
  type Random,
} from "@measurekit/core";
import type { ResponseMatrix } from "../data/response-matrix.js";
import { type CodedItem, codeItems } from "./coded-items.js";
import { correlationEigenvalues } from "./eigenvalues.js";
import { pearsonMatrix } from "./pearson.js";
import { type PolychoricOptions, polychoricMatrix } from "./polychoric.js";

/** Options of {@link parallelAnalysis}. */
export interface ParallelAnalysisOptions extends PolychoricOptions {
  /**
   * The correlation of the items: `"polychoric"` for item responses, `"pearson"` for scores treated as continuous, over
   * the persons who answered both items of each pair.
   *
   * @defaultValue `"polychoric"`
   */
  readonly correlation?: "pearson" | "polychoric";
  /**
   * The random data: `"resampled"`, each item's responses permuted among the persons, which keeps its distribution and
   * its missing responses; `"normal"`, standard normal data for every person and item, with `correlation: "pearson"`.
   *
   * @defaultValue `"resampled"`
   */
  readonly comparison?: "resampled" | "normal";
  /**
   * The number of random data sets, at least 1.
   *
   * @defaultValue `100`
   */
  readonly replications?: number;
  /**
   * The quantile of the random eigenvalues that the observed ones must exceed, from 0 to 1.
   *
   * @defaultValue `0.95`
   */
  readonly quantile?: number;
}

/** The eigenvalues of the observed correlations against those of random data. */
export interface EigenvalueComparison {
  /** The observed eigenvalues, largest first. */
  readonly observed: readonly number[];
  /** The mean of each eigenvalue over the random data sets. */
  readonly randomMean: readonly number[];
  /** The `quantile` of each eigenvalue over the random data sets. */
  readonly randomQuantile: readonly number[];
  /** The number of leading observed eigenvalues above the random quantile, up to the first that is not. */
  readonly suggested: number;
}

/** The result of a parallel analysis: the comparison for components and for factors. */
export interface ParallelAnalysis {
  /** The eigenvalues of the correlation matrix, the principal components'. */
  readonly components: EigenvalueComparison;
  /** The eigenvalues of the correlation matrix with the one-factor minres communalities on its diagonal. */
  readonly factors: EigenvalueComparison;
}

const CORRELATIONS = ["pearson", "polychoric"] as const;
const COMPARISONS = ["resampled", "normal"] as const;
const DEFAULT_CORRECTION = 0.5;
const DEFAULT_REPLICATIONS = 100;
const DEFAULT_QUANTILE = 0.95;
const MIN_ITEMS = 3;

/**
 * Returns Horn's parallel analysis of the items of a response matrix: the eigenvalues of the observed correlations
 * against those of correlations of random data of the same size.
 *
 * @remarks
 * The observed correlations are polychoric or Pearson. Their eigenvalues are compared for the components, the
 * eigenvalues of the correlation matrix, and for the factors, those of the matrix with the communalities of the
 * one-factor minimum residual solution on its diagonal, as psych's `fa.parallel`. Each random data set gives the same
 * two sets of eigenvalues; the comparison reports their means and a quantile, and `suggested` counts the leading
 * observed eigenvalues above the quantile, the number of components or factors that exceed what chance gives. A matrix
 * that is not positive definite is used as it is, and negative eigenvalues are reported. Random permutations and normal
 * draws come from `random`, so a seeded generator reproduces the result. The cost of polychoric correlations grows with
 * the pairs of items and the product of their categories; items with many categories suit Pearson correlations.
 *
 * @throws `DataError` when the matrix is invalid, has fewer than three items or an item with fewer than two observed
 * categories, a pair of items has too few persons for a correlation, an option is invalid, `random` is not a random
 * number generator, or `comparison` is `"normal"` with `correlation` `"polychoric"`.
 * @see {@link https://doi.org/10.1007/BF02289447 | Horn (1965)}
 * @see {@link https://doi.org/10.1177/0013164495055003002 | Glorfeld (1995)}
 */
export function parallelAnalysis(
  matrix: ResponseMatrix,
  random: Random,
  options: ParallelAnalysisOptions = {},
): ParallelAnalysis {
  checkOptions(options);
  const {
    correlation = "polychoric",
    comparison = "resampled",
    replications = DEFAULT_REPLICATIONS,
    quantile = DEFAULT_QUANTILE,
    correction = DEFAULT_CORRECTION,
  } = options;
  checkChoice(correlation, "The option correlation", CORRELATIONS);
  checkChoice(comparison, "The option comparison", COMPARISONS);
  checkPositiveInteger(replications, "The option replications");
  checkUnitInterval(quantile, "The option quantile");
  checkNonNegative(correction, "The option correction");
  checkRandom(random);
  if (comparison === "normal" && correlation === "polychoric") {
    throw new DataError('The option comparison "normal" needs the option correlation "pearson", found "polychoric".');
  }
  const items = codeItems(matrix);
  checkItems(items);
  const size = items.length;
  const persons = matrix.responses.length;
  const correlate = (): Float64Array =>
    correlation === "pearson" ? pearsonMatrix(items.map((item) => item.values)) : polychoricMatrix(items, correction);

  const observed = correlationEigenvalues(checkedMatrix(correlate(), items, "The observed data"), size);
  const components = new Float64Array(replications * size);
  const factors = new Float64Array(replications * size);
  for (let replication = 0; replication < replications; replication++) {
    let randomCorrelations: Float64Array;
    if (comparison === "normal") {
      const columns = items.map(() => Float64Array.from({ length: persons }, () => random.normal()));
      randomCorrelations = pearsonMatrix(columns);
    } else {
      for (const item of items) shuffle(correlation === "pearson" ? item.values : item.ranks, random);
      randomCorrelations = correlate();
    }
    const eigenvalues = correlationEigenvalues(
      checkedMatrix(randomCorrelations, items, `Random data set ${replication + 1}`),
      size,
    );
    components.set(eigenvalues.components, replication * size);
    factors.set(eigenvalues.factors, replication * size);
  }
  return {
    components: compare(observed.components, components, replications, quantile),
    factors: compare(observed.factors, factors, replications, quantile),
  };
}

/**
 * Checks that there are at least three items and each has two observed categories.
 *
 * @throws `DataError` when not.
 */
function checkItems(items: readonly CodedItem[]): void {
  if (items.length < MIN_ITEMS) {
    throw new DataError(
      `Parallel analysis needs at least ${MIN_ITEMS} items, as one factor is not identified by two, found ${items.length}.`,
    );
  }
  for (const item of items) {
    if (item.levels < 2) {
      throw new DataError(
        `Parallel analysis needs items with at least two observed categories; item "${item.id}" has ${item.levels}.`,
      );
    }
  }
}

/**
 * Returns the correlations after checking that every pair has one.
 *
 * @throws `DataError` naming the first pair without a correlation and `what`.
 */
function checkedMatrix(correlations: Float64Array, items: readonly CodedItem[], what: string): Float64Array {
  const size = items.length;
  for (let i = 0; i < size; i++) {
    for (let j = i + 1; j < size; j++) {
      if (Number.isNaN(correlations[i * size + j])) {
        throw new DataError(
          `${what} gives no correlation of items "${items[i]?.id}" and "${items[j]?.id}": too few persons answered both.`,
        );
      }
    }
  }
  return correlations;
}

/** Permutes the entries of `values` in place by Fisher-Yates, uniformly at random. */
function shuffle(values: Int32Array | Float64Array, random: Random): void {
  for (let i = values.length - 1; i > 0; i--) {
    const j = Math.floor(random.uniform() * (i + 1));
    const held = values[i] ?? 0;
    values[i] = values[j] ?? 0;
    values[j] = held;
  }
}

/** Returns the comparison of observed eigenvalues with `samples`, one row of `observed.length` per replication. */
function compare(
  observed: Float64Array,
  samples: Float64Array,
  replications: number,
  quantile: number,
): EigenvalueComparison {
  const size = observed.length;
  const randomMean: number[] = [];
  const randomQuantile: number[] = [];
  for (let k = 0; k < size; k++) {
    const column = Float64Array.from(
      { length: replications },
      (_, replication) => samples[replication * size + k] ?? 0,
    );
    randomMean.push(column.reduce((sum, value) => sum + value, 0) / replications);
    randomQuantile.push(quantileOf(column.sort(), quantile));
  }
  const exceeded = observed.findIndex((value, k) => !(value > (randomQuantile[k] ?? Number.POSITIVE_INFINITY)));
  return {
    observed: Array.from(observed),
    randomMean,
    randomQuantile,
    suggested: exceeded < 0 ? size : exceeded,
  };
}

/** Returns the quantile of sorted values by linear interpolation between order statistics, R's default type 7. */
function quantileOf(sorted: Float64Array, probability: number): number {
  const position = (sorted.length - 1) * probability;
  const below = Math.floor(position);
  const lower = sorted[below] ?? 0;
  const upper = sorted[Math.min(below + 1, sorted.length - 1)] ?? lower;
  return lower + (position - below) * (upper - lower);
}
