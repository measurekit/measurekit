import { checkBoolean, checkChoice, checkOptions, checkUnitInterval, DataError } from "@measurekit/core";
import { type BivariateDistribution, type BivariateTable, margins, neatTables } from "./bivariate.js";

/** Options of the equating methods that average over a synthetic population. */
export interface SyntheticOptions {
  /**
   * The weight w₁ of the `from` group's population in the synthetic population, from 0 to 1; the `to` group's is 1 -
   * w₁.
   *
   * @defaultValue the `from` group's share of the persons
   */
  readonly weight?: number;
}

/** Options of {@link levineEquating}. */
export interface LevineOptions extends SyntheticOptions {
  /**
   * Whether to use Levine's true-score method rather than the observed-score method. The true-score method does not
   * use `weight`.
   *
   * @defaultValue `false`
   */
  readonly trueScore?: boolean;
  /**
   * Whether the anchor counts in the total score (`"internal"`) or not (`"external"`).
   *
   * @defaultValue `"internal"`
   */
  readonly anchor?: "internal" | "external";
}

/**
 * Returns the Tucker linear equating of the scores of a form to the scale of another through common items, from two
 * nonequivalent groups.
 *
 * @remarks
 * Each group's regression of the total score on the anchor score has the slope γ = σ(X, V) / σ²(V). With w₁ and w₂ the
 * weights of the two groups' populations and Δ the difference of the groups' mean anchor scores, the synthetic
 * population has the means μ(X) - w₂ γ₁ Δ and μ(Y) + w₁ γ₂ Δ, and the variances σ₁²(X) - w₂ γ₁² (σ₁²(V) - σ₂²(V)) + w₁
 * w₂ γ₁² Δ² and σ₂²(Y) + w₁ γ₂² (σ₁²(V) - σ₂²(V)) + w₁ w₂ γ₂² Δ²; linear equating in it gives the result, as the equate
 * package's `equate(type = "linear", method = "tucker")`. Moments have denominator n - 1. Each distribution is the
 * persons at each total score and anchor score; the `from` group took the form to equate and the `to` group the form
 * to equate to.
 *
 * @throws `DataError` when a distribution is invalid, the two have different numbers of anchor scores, a group has
 * fewer than two persons or total or anchor scores without variance, or the synthetic population's total scores have
 * none; or when `weight` is not a number from 0 to 1.
 * @see {@link https://doi.org/10.1007/978-1-4939-0317-7 | Kolen and Brennan (2014)}
 */
export function tuckerEquating(
  from: BivariateDistribution,
  to: BivariateDistribution,
  options: SyntheticOptions = {},
): number[] {
  checkOptions(options);
  const [source, target] = neatTables(from, to);
  const weight = syntheticWeight(options, source, target);
  const sourceMoments = groupMoments(source, "from");
  const targetMoments = groupMoments(target, "to");
  return syntheticLinear(
    sourceMoments,
    targetMoments,
    sourceMoments.covariance / sourceMoments.anchorVariance,
    targetMoments.covariance / targetMoments.anchorVariance,
    weight,
    source.rows,
  );
}

/**
 * Returns the Levine linear equating of the scores of a form to the scale of another through common items, from two
 * nonequivalent groups, by the observed-score or the true-score method.
 *
 * @remarks
 * The observed-score method averages over the synthetic population of {@link tuckerEquating} with the slopes γ =
 * σ²(X) / σ(X, V) for an internal anchor and γ = (σ²(X) + σ(X, V)) / (σ²(V) + σ(X, V)) for an external one. The
 * true-score method is free of the weights: y = (γ₂ / γ₁)(x - μ₁(X)) + μ₂(Y) + γ₂ (μ₁(V) - μ₂(V)). Both are the equate
 * package's `equate(type = "linear", method = "levine")`, with its `internal` and `lts`. Moments have denominator n -
 * 1.
 *
 * @throws `DataError` in the cases of {@link tuckerEquating}, or when a group's total and anchor scores do not have a
 * positive covariance; or when `anchor` is not `"internal"` or `"external"`.
 * @see {@link https://doi.org/10.1007/978-1-4939-0317-7 | Kolen and Brennan (2014)}
 */
export function levineEquating(
  from: BivariateDistribution,
  to: BivariateDistribution,
  options: LevineOptions = {},
): number[] {
  checkOptions(options);
  const [source, target] = neatTables(from, to);
  const weight = syntheticWeight(options, source, target);
  const { trueScore = false, anchor = "internal" } = options;
  checkBoolean(trueScore, "The option trueScore");
  checkChoice(anchor, "The option anchor", ["internal", "external"]);
  const sourceMoments = groupMoments(source, "from");
  const targetMoments = groupMoments(target, "to");
  const sourceSlope = levineSlope(sourceMoments, anchor, "from");
  const targetSlope = levineSlope(targetMoments, anchor, "to");
  if (!trueScore) return syntheticLinear(sourceMoments, targetMoments, sourceSlope, targetSlope, weight, source.rows);
  const slope = targetSlope / sourceSlope;
  const shift = targetMoments.totalMean + targetSlope * (sourceMoments.anchorMean - targetMoments.anchorMean);
  return Array.from({ length: source.rows }, (_, x) => slope * (x - sourceMoments.totalMean) + shift);
}

/**
 * Returns the chained linear equating of the scores of a form to the scale of another through common items, from two
 * nonequivalent groups.
 *
 * @remarks
 * The total scores of the `from` group are equated linearly to its anchor scores, and these to the total scores of the
 * `to` group by that group's linear equating, as the equate package's `equate(type = "linear", method = "chained")`.
 * Moments have denominator n - 1.
 *
 * @throws `DataError` when a distribution is invalid, the two have different numbers of anchor scores, or a group has
 * fewer than two persons or total or anchor scores without variance.
 * @see {@link https://doi.org/10.1007/978-1-4939-0317-7 | Kolen and Brennan (2014)}
 */
export function chainedLinearEquating(from: BivariateDistribution, to: BivariateDistribution): number[] {
  const [source, target] = neatTables(from, to);
  const sourceMoments = groupMoments(source, "from");
  const targetMoments = groupMoments(target, "to");
  const anchorSlope = Math.sqrt(sourceMoments.anchorVariance / sourceMoments.totalVariance);
  const totalSlope = Math.sqrt(targetMoments.totalVariance / targetMoments.anchorVariance);
  return Array.from({ length: source.rows }, (_, x) => {
    const anchorEquivalent = sourceMoments.anchorMean + anchorSlope * (x - sourceMoments.totalMean);
    return targetMoments.totalMean + totalSlope * (anchorEquivalent - targetMoments.anchorMean);
  });
}

/**
 * Returns the weight of the `from` group in a synthetic population: `options.weight`, checked, or the group's share of
 * the persons.
 *
 * @throws `DataError` when the weight is not a number from 0 to 1.
 */
export function syntheticWeight(options: SyntheticOptions, source: BivariateTable, target: BivariateTable): number {
  const { weight = source.persons / (source.persons + target.persons) } = options;
  checkUnitInterval(weight, "The option weight");
  return weight;
}

/** The moments of one group's total and anchor scores, with denominator n - 1. */
interface GroupMoments {
  readonly totalMean: number;
  readonly anchorMean: number;
  readonly totalVariance: number;
  readonly anchorVariance: number;
  readonly covariance: number;
}

/**
 * Returns the moments of a group's scores by two passes.
 *
 * @throws `DataError` when the group has fewer than two persons or its total or anchor scores have no variance.
 */
function groupMoments(table: BivariateTable, label: string): GroupMoments {
  if (!(table.persons > 1)) {
    throw new DataError(`Equating through an anchor needs at least two persons in the ${label} distribution.`);
  }
  const { rows, columns } = margins(table);
  const mean = (counts: Float64Array): number => {
    let sum = 0;
    counts.forEach((count, score) => {
      sum += count * score;
    });
    return sum / table.persons;
  };
  const totalMean = mean(rows);
  const anchorMean = mean(columns);
  const variance = (counts: Float64Array, center: number): number => {
    let squares = 0;
    counts.forEach((count, score) => {
      squares += count * (score - center) ** 2;
    });
    return squares / (table.persons - 1);
  };
  const totalVariance = variance(rows, totalMean);
  const anchorVariance = variance(columns, anchorMean);
  if (!(totalVariance > 0 && anchorVariance > 0)) {
    throw new DataError(
      `Equating through an anchor needs total and anchor scores with variance in the ${label} distribution.`,
    );
  }
  let products = 0;
  for (let x = 0; x < table.rows; x++) {
    for (let v = 0; v < table.columns; v++) {
      products += (table.counts[x * table.columns + v] ?? 0) * (x - totalMean) * (v - anchorMean);
    }
  }
  return { totalMean, anchorMean, totalVariance, anchorVariance, covariance: products / (table.persons - 1) };
}

/**
 * Returns Levine's slope γ of a group's total scores on its anchor scores.
 *
 * @throws `DataError` when the covariance of the total and anchor scores is not positive.
 */
function levineSlope(moments: GroupMoments, anchor: "internal" | "external", label: string): number {
  if (!(moments.covariance > 0)) {
    throw new DataError(
      `Levine equating needs total and anchor scores with a positive covariance in the ${label} distribution.`,
    );
  }
  return anchor === "internal"
    ? moments.totalVariance / moments.covariance
    : (moments.totalVariance + moments.covariance) / (moments.anchorVariance + moments.covariance);
}

/**
 * Returns the linear equating in the synthetic population of the groups, for the total scores 0 to `scores` - 1, the
 * regression slopes γ of the total on the anchor scores being given.
 *
 * @throws `DataError` when the synthetic population's total scores have no variance.
 */
function syntheticLinear(
  source: GroupMoments,
  target: GroupMoments,
  sourceSlope: number,
  targetSlope: number,
  weight: number,
  scores: number,
): number[] {
  const other = 1 - weight;
  const meanDifference = source.anchorMean - target.anchorMean;
  const varianceDifference = source.anchorVariance - target.anchorVariance;
  const sourceMean = source.totalMean - other * sourceSlope * meanDifference;
  const targetMean = target.totalMean + weight * targetSlope * meanDifference;
  const sourceVariance =
    source.totalVariance -
    other * sourceSlope ** 2 * varianceDifference +
    weight * other * sourceSlope ** 2 * meanDifference ** 2;
  const targetVariance =
    target.totalVariance +
    weight * targetSlope ** 2 * varianceDifference +
    weight * other * targetSlope ** 2 * meanDifference ** 2;
  if (!(sourceVariance > 0 && targetVariance > 0)) {
    throw new DataError("The synthetic population has total scores without variance; use another weight or method.");
  }
  const slope = Math.sqrt(targetVariance / sourceVariance);
  return Array.from({ length: scores }, (_, x) => targetMean + slope * (x - sourceMean));
}
