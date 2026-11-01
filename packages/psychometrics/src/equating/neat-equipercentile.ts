import { DataError } from "@measurekit/core";
import { type BivariateDistribution, type BivariateTable, margins, neatTables } from "./bivariate.js";
import { type SyntheticOptions, syntheticGroups } from "./neat.js";
import { cumulativeSums, equipercentileEquating, scoreAtPercentileRank } from "./observed-score.js";

/**
 * Returns the chained equipercentile equating of the scores of a form to the scale of another through common items,
 * from two nonequivalent groups.
 *
 * @remarks
 * The total scores of the `from` group are equated equipercentile to its anchor scores, which are continuous, and each
 * such anchor equivalent to the total scores of the `to` group through that group's anchor and total score
 * distributions. The percentile rank of an anchor equivalent in the `to` group's anchor distribution interpolates
 * linearly between the cumulative shares at the ends of its score's interval, as the equate package's `equate(type =
 * "equipercentile", method = "chained")`, which rounds the ranks to 10 decimals. Each distribution is the persons at
 * each total score and anchor score.
 *
 * @param from - The persons at each total score and anchor score of the form to equate.
 * @param to - The persons at each total score and anchor score of the form to equate to.
 * @returns The equated score at each score of the form to equate.
 * @throws `DataError` when a distribution is invalid or the two have different numbers of anchor scores.
 * @see {@link https://doi.org/10.1007/978-1-4939-0317-7 | Kolen and Brennan (2014)}
 */
export function chainedEquipercentileEquating(from: BivariateDistribution, to: BivariateDistribution): number[] {
  const [source, target] = neatTables(from, to);
  const sourceMargins = margins(source);
  const targetMargins = margins(target);
  const anchorEquivalents = equipercentileEquating(Array.from(sourceMargins.rows), Array.from(sourceMargins.columns));
  const anchorCumulative = cumulativeSums(targetMargins.columns);
  const totalCumulative = cumulativeSums(targetMargins.rows);
  const highest = target.rows - 1;
  return anchorEquivalents.map((anchor) => {
    const rank = anchorRank(anchor, anchorCumulative, target.persons);
    if (rank === 0) return -0.5;
    if (rank === 1) return highest + 0.5;
    return scoreAtPercentileRank(rank, totalCumulative, target.persons);
  });
}

/**
 * Returns the frequency estimation equipercentile equating of the scores of a form to the scale of another through
 * common items, from two nonequivalent groups.
 *
 * @remarks
 * Each form's total score distribution in the synthetic population mixes the group that took it with the other group
 * as the form would have scored it there: f(x) = w₁ f₁(x) + w₂ Σ_v f₁(x | v) h₂(v) and g(y) = w₁ Σ_v g₂(y | v) h₁(v) +
 * w₂ g₂(y), with f, g the groups' relative frequencies, h their anchor score frequencies and w₁, w₂ the weights of the
 * two groups' populations. Equipercentile equating of f and g gives the result. A group's conditional distribution
 * given an anchor score needs persons with it, so an anchor score that persons in only one group have throws; smooth
 * the distributions with {@link bivariateLogLinearPresmoothing} first. The equate package's `equate(type =
 * "equipercentile", method = "frequency estimation")` weighs the two components of each distribution by the groups'
 * sizes as well as w₁ and w₂, and drops the persons whose anchor score only one group has; it agrees with this
 * function when the groups have the same size.
 *
 * @param from - The persons at each total score and anchor score of the form to equate.
 * @param to - The persons at each total score and anchor score of the form to equate to.
 * @param options - The weight of the first group in the synthetic population.
 * @returns The equated score at each score of the form to equate.
 * @throws `DataError` when a distribution is invalid, the two have different numbers of anchor scores, an anchor score
 * has persons in only one group, or `weight` is not a number from 0 to 1.
 * @see {@link https://doi.org/10.1007/978-1-4939-0317-7 | Kolen and Brennan (2014)}
 */
export function frequencyEstimationEquating(
  from: BivariateDistribution,
  to: BivariateDistribution,
  options: SyntheticOptions = {},
): number[] {
  const { source, target, weight } = syntheticGroups(from, to, options);
  const sourceAnchor = margins(source).columns;
  const targetAnchor = margins(target).columns;
  checkSharedAnchorScores(sourceAnchor, targetAnchor);
  return equipercentileEquating(
    Array.from(syntheticTotals(source, sourceAnchor, targetAnchor, target.persons, weight)),
    Array.from(syntheticTotals(target, targetAnchor, sourceAnchor, source.persons, 1 - weight)),
  );
}

/**
 * Checks that persons in one group have an anchor score exactly when persons in the other do.
 *
 * @param from - The persons of the `from` group at each anchor score.
 * @param to - The persons of the `to` group at each anchor score.
 * @throws `DataError` when an anchor score has persons in one group only.
 */
function checkSharedAnchorScores(from: Float64Array, to: Float64Array): void {
  from.forEach((count, v) => {
    const inFrom = count > 0;
    if (inFrom !== (to[v] ?? 0) > 0) {
      throw new DataError(
        `Frequency estimation needs persons with the anchor score ${v} in both distributions or neither, but the ${inFrom ? "to" : "from"} distribution has none. Smooth the distributions with bivariateLogLinearPresmoothing.`,
      );
    }
  });
}

/**
 * Returns the relative frequency of each total score of a form in the synthetic population: `ownWeight` times the
 * group's own frequencies plus the other weight (1 - `ownWeight`) times the frequencies the form would have among the
 * other group's anchor scores, Σ_v f(x | v) h_other(v) with f(x | v) the group's count at x and v over its persons at v.
 */
function syntheticTotals(
  table: BivariateTable,
  ownAnchor: Float64Array,
  otherAnchor: Float64Array,
  otherPersons: number,
  ownWeight: number,
): Float64Array {
  const totals = new Float64Array(table.rows);
  for (let x = 0; x < table.rows; x++) {
    for (let v = 0; v < table.columns; v++) {
      const count = table.counts[x * table.columns + v] ?? 0;
      if (count === 0) continue;
      const spread = (otherAnchor[v] ?? 0) / ((ownAnchor[v] ?? 1) * otherPersons);
      totals[x] = (totals[x] ?? 0) + count * (ownWeight / table.persons + (1 - ownWeight) * spread);
    }
  }
  return totals;
}

/**
 * Returns the percentile rank of a continuous anchor score in a distribution: the cumulative share below the interval
 * of its score, plus the score's share in proportion to how far the value is into the interval.
 */
function anchorRank(value: number, cumulative: Float64Array, persons: number): number {
  const score = Math.floor(value + 0.5);
  const share = (k: number): number => (k < 0 ? 0 : k >= cumulative.length ? 1 : (cumulative[k] ?? 0) / persons);
  const below = share(score - 1);
  return below + (value - (score - 0.5)) * (share(score) - below);
}
