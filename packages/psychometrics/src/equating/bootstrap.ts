import {
  checkArray,
  checkChoice,
  checkOptions,
  checkPositiveInteger,
  checkRandom,
  checkRecord,
  DataError,
  describe,
  type Random,
} from "@measurekit/core";
import { checkFunction } from "../checks.js";
import { type BivariateDistribution, bivariateTable } from "./bivariate.js";
import { checkDistribution, cumulativeSums } from "./observed-score.js";

/** The scores of the persons a bootstrap resamples, by the design of the data collection. */
export type EquatingSample =
  | {
      /** Two equivalent groups, each of which took one form. */
      readonly design: "random-groups";
      /** The persons at each score on the form to equate, from 0. */
      readonly from: readonly number[];
      /** The persons at each score on the form to equate to, from 0. */
      readonly to: readonly number[];
    }
  | {
      /** One group, which took both forms. */
      readonly design: "single-group";
      /** The persons at each pair of scores: by score on the form to equate, then score on the form to equate to. */
      readonly joint: BivariateDistribution;
    }
  | {
      /** Two groups, each of which took one form and a common anchor. */
      readonly design: "neat";
      /** The persons of the group that took the form to equate, by total score, then anchor score. */
      readonly from: BivariateDistribution;
      /** The persons of the group that took the form to equate to, by total score, then anchor score. */
      readonly to: BivariateDistribution;
    };

/** Options of {@link bootstrapEquating}. */
export interface BootstrapOptions {
  /**
   * The number of bootstrap samples, an integer from 2. The Monte Carlo error of a standard error is about 1 / √(2 (B -
   * 1)) of it: 3% for 500.
   *
   * @defaultValue `500`
   */
  readonly replications?: number;
}

const DEFAULT_REPLICATIONS = 500;
const DESIGNS = ["random-groups", "single-group", "neat"];

/**
 * Returns the bootstrap standard error of the equated score at each score of the form to equate.
 *
 * @remarks
 * Each replication draws as many persons as the sample has, with replacement, from the cells of its distributions in
 * proportion to their counts, groups independently and a single group's pairs of scores together, and applies `equate`
 * to the resampled distributions. The standard error at a score is the standard deviation (denominator B - 1) of the
 * B equated scores, as the equate package's `bootstrap`. Smoothing and every other step belong inside `equate`, so that
 * their variability counts; `equate` also runs once on the sample, to fix the number of scores. A replication in which
 * `equate` throws, such as frequency estimation on a resample with an anchor score in one group only, stops the
 * bootstrap with that error. The same seed in `random` gives the same standard errors.
 *
 * @param sample - The persons' scores, as whole numbers of persons.
 * @param equate - Returns the equated score of each score of the form to equate, such as `(s) => tuckerEquating(s.from,
 * s.to)`.
 * @param random - The source of the draws.
 * @throws `DataError` when the sample is invalid or its counts are not whole numbers, `equate` is not a function or does
 * not return a finite equated score at each score, `random` is not a random number generator, or `replications` is
 * not an integer from 2.
 * @see {@link https://doi.org/10.1201/9780429246593 | Efron and Tibshirani (1993)}
 */
export function bootstrapEquating<TSample extends EquatingSample>(
  sample: TSample,
  equate: (sample: TSample) => readonly number[],
  random: Random,
  options: BootstrapOptions = {},
): number[] {
  checkRecord(sample, "The sample must be an object with a design");
  checkChoice(sample.design, "The design of the sample", DESIGNS);
  checkFunction(equate, "The equate argument must be a function of a sample");
  checkRandom(random);
  checkOptions(options);
  const { replications = DEFAULT_REPLICATIONS } = options;
  checkPositiveInteger(replications, "The option replications");
  if (replications < 2) {
    throw new DataError(`The option replications must be an integer from 2, found ${replications}.`);
  }
  const resample = resampler(sample);
  const scores = equated(equate(sample), undefined).length;
  const mean = new Float64Array(scores);
  const squares = new Float64Array(scores);
  for (let replication = 1; replication <= replications; replication++) {
    const result = equated(equate(resample(random)), scores);
    for (let x = 0; x < scores; x++) {
      const value = result[x] ?? 0;
      const delta = value - (mean[x] ?? 0);
      mean[x] = (mean[x] ?? 0) + delta / replication;
      squares[x] = (squares[x] ?? 0) + delta * (value - (mean[x] ?? 0));
    }
  }
  return Array.from(squares, (sum) => Math.sqrt(sum / (replications - 1)));
}

/**
 * Checks the equated scores of one equating: an array of finite numbers, `scores` of them when that is known.
 *
 * @throws `DataError` when they are not.
 */
function equated(result: readonly number[], scores: number | undefined): readonly number[] {
  checkArray(result, "The equate function must return an array of equated scores");
  if (scores !== undefined && result.length !== scores) {
    throw new DataError(`The equate function must return ${scores} equated scores every time, found ${result.length}.`);
  }
  result.forEach((score, x) => {
    if (!Number.isFinite(score)) {
      throw new DataError(`The equate function must return finite equated scores, found ${describe(score)} at ${x}.`);
    }
  });
  return result;
}

/** Returns a function that resamples the persons of a sample, checking the sample's distributions. */
function resampler<TSample extends EquatingSample>(sample: TSample): (random: Random) => TSample {
  if (sample.design === "random-groups") {
    checkDistribution(sample.from, "from");
    checkDistribution(sample.to, "to");
    const draw = [multinomial(sample.from, "from"), multinomial(sample.to, "to")] as const;
    return (random) => ({ ...sample, from: Array.from(draw[0](random)), to: Array.from(draw[1](random)) });
  }
  if (sample.design === "single-group") {
    const table = bivariateTable(sample.joint, "joint", {
      row: "score on the form to equate",
      column: "score on the form to equate to",
    });
    const draw = multinomial(table.counts, "joint");
    return (random) => ({ ...sample, joint: rows(draw(random), table.rows, table.columns) });
  }
  const from = bivariateTable(sample.from, "from");
  const to = bivariateTable(sample.to, "to");
  const draw = [multinomial(from.counts, "from"), multinomial(to.counts, "to")] as const;
  return (random) => ({
    ...sample,
    from: rows(draw[0](random), from.rows, from.columns),
    to: rows(draw[1](random), to.rows, to.columns),
  });
}

/** Returns the counts as rows of equal length. */
function rows(counts: Float64Array, count: number, columns: number): number[][] {
  return Array.from({ length: count }, (_, x) => Array.from(counts.subarray(x * columns, (x + 1) * columns)));
}

/**
 * Returns a function that draws as many persons as `counts` has, with replacement, in proportion to the counts, and
 * returns the number drawn in each cell.
 *
 * @throws `DataError` when a count is not a whole number.
 */
function multinomial(counts: ArrayLike<number>, label: string): (random: Random) => Float64Array {
  for (let cell = 0; cell < counts.length; cell++) {
    if (!Number.isSafeInteger(counts[cell])) {
      throw new DataError(
        `The ${label} distribution needs whole numbers of persons to resample, found ${describe(counts[cell])} in cell ${cell}.`,
      );
    }
  }
  const cumulative = cumulativeSums(counts);
  const persons = cumulative[cumulative.length - 1] ?? 0;
  return (random) => {
    const drawn = new Float64Array(counts.length);
    for (let person = 0; person < persons; person++) {
      const target = random.uniform() * persons;
      // The first cell whose cumulative count exceeds the draw: cells without persons are never drawn.
      let low = 0;
      let high = counts.length - 1;
      while (low < high) {
        const middle = (low + high) >> 1;
        if ((cumulative[middle] ?? 0) > target) high = middle;
        else low = middle + 1;
      }
      drawn[low] = (drawn[low] ?? 0) + 1;
    }
    return drawn;
  };
}
