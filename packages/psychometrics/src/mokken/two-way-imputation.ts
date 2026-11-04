import { checkIntegerFrom, checkOptions, checkRandom, DataError, type FieldNames, type Random } from "@measurekit/core";
import { checkResponseMatrix, MAX_RESPONSE, type ResponseMatrix } from "../data/internal.js";
import { roundHalfToEven } from "../math/internal.js";

/** Options of {@link twoWayImputation}. */
export interface TwoWayImputationOptions {
  /**
   * The lowest score an imputed response may take, an integer at most the lowest response of the matrix.
   *
   * @defaultValue the lowest response of the matrix
   */
  readonly minimum?: number;
  /**
   * The highest score an imputed response may take, an integer at least the highest response of the matrix and at most
   * 2³¹ - 1, the largest response.
   *
   * @defaultValue the highest response of the matrix
   */
  readonly maximum?: number;
}

const TWO_WAY_IMPUTATION_OPTIONS: FieldNames<TwoWayImputationOptions> = { minimum: true, maximum: true };

/**
 * Returns a copy of a response matrix with each missing response imputed by two-way imputation with a random error,
 * step 2 of a Mokken scale analysis, as the mokken package's `twoway`.
 *
 * @remarks
 * A missing response of person i to item j takes TW + e, rounded to an integer, halves to the even one as R's `round`,
 * and kept from `minimum` to `maximum`. TW = PM_i + IM_j - OM, with PM_i the mean of the person's responses, IM_j that
 * of the item's and OM that of all of them, and e a normal error of variance the sample variance of the observed
 * responses less their TW, drawn from `random` in the order of the persons, then the items. Responses that are present
 * are kept, and so are the ids and the response times. The error is van Ginkel, van der Ark and Sijtsma's, whose variance
 * `twoway` takes as its standard deviation. Each call draws a new completion: run it several times, with the analysis
 * on each, for multiple imputation.
 *
 * @param matrix - The responses of the persons, with `null` where missing.
 * @param random - The source of the errors, as `createRandom` returns it.
 * @param options - The bounds of an imputed response.
 * @returns A new matrix without missing responses; a copy of `matrix` when it has none.
 * @throws `DataError` when the matrix is invalid, a person or an item has no response, `random` is not a generator, or
 * a bound is not an integer from 0 to 2³¹ - 1 or does not hold every response.
 * @see {@link https://doi.org/10.1111/bmsp.12078 | Sijtsma and van der Ark (2017)}
 */
export function twoWayImputation(
  matrix: ResponseMatrix,
  random: Random,
  options: TwoWayImputationOptions = {},
): ResponseMatrix {
  checkResponseMatrix(matrix);
  checkRandom(random);
  checkOptions(options, TWO_WAY_IMPUTATION_OPTIONS);
  const observed = observedMoments(matrix);
  const { minimum = observed.lowest, maximum = observed.highest } = options;
  checkBound(minimum, "The option minimum", observed.lowest, "at most the lowest");
  checkBound(maximum, "The option maximum", observed.highest, "at least the highest");
  const { personMeans, itemMeans, mean } = observed;
  const twoWay = (person: number, item: number): number => (personMeans[person] ?? 0) + (itemMeans[item] ?? 0) - mean;
  const errorDeviation = Math.sqrt(residualVariance(matrix, twoWay));
  const responses = matrix.responses.map((row, person) =>
    row.map((response, item) => {
      if (response !== null) return response;
      const value = roundHalfToEven(twoWay(person, item) + errorDeviation * random.normal());
      return Math.min(maximum, Math.max(minimum, value));
    }),
  );
  const { personIds, itemIds, times } = matrix;
  return times === undefined ? { personIds, itemIds, responses } : { personIds, itemIds, responses, times };
}

/** The means of the observed responses by person, by item and over all, and their range. */
interface ObservedMoments {
  readonly personMeans: Float64Array;
  readonly itemMeans: Float64Array;
  readonly mean: number;
  readonly lowest: number;
  readonly highest: number;
}

/**
 * Returns the means of the observed responses, each a sum of integers over its count.
 *
 * @throws `DataError` naming the first person or item without a response.
 */
function observedMoments(matrix: ResponseMatrix): ObservedMoments {
  const items = matrix.itemIds.length;
  const itemSums = new Float64Array(items);
  const itemCounts = new Float64Array(items);
  let total = 0;
  let count = 0;
  let lowest = Number.POSITIVE_INFINITY;
  let highest = Number.NEGATIVE_INFINITY;
  const personMeans = Float64Array.from(matrix.responses, (row, person) => {
    let sum = 0;
    let answered = 0;
    row.forEach((response, item) => {
      if (response === null) return;
      sum += response;
      answered++;
      itemSums[item] = (itemSums[item] ?? 0) + response;
      itemCounts[item] = (itemCounts[item] ?? 0) + 1;
      lowest = Math.min(lowest, response);
      highest = Math.max(highest, response);
    });
    if (answered === 0) {
      throw new DataError(
        `Two-way imputation needs a response of every person; "${matrix.personIds[person]}" has none.`,
      );
    }
    total += sum;
    count += answered;
    return sum / answered;
  });
  const itemMeans = itemSums.map((sum, item) => {
    if (itemCounts[item] === 0) {
      throw new DataError(`Two-way imputation needs a response to every item; "${matrix.itemIds[item]}" has none.`);
    }
    return sum / (itemCounts[item] ?? 1);
  });
  return { personMeans, itemMeans, mean: total / count, lowest, highest };
}

/** Returns the sample variance of the observed responses less their two-way values, by two passes. */
function residualVariance(matrix: ResponseMatrix, twoWay: (person: number, item: number) => number): number {
  const residuals: number[] = [];
  matrix.responses.forEach((row, person) => {
    row.forEach((response, item) => {
      if (response !== null) residuals.push(response - twoWay(person, item));
    });
  });
  const center = residuals.reduce((sum, residual) => sum + residual, 0) / residuals.length;
  const squares = residuals.reduce((sum, residual) => sum + (residual - center) ** 2, 0);
  return residuals.length > 1 ? squares / (residuals.length - 1) : 0;
}

/**
 * Checks a bound of the imputed responses.
 *
 * @throws `DataError` when it is not a response, an integer from 0 to 2³¹ - 1, or lies inside the observed responses.
 */
function checkBound(bound: unknown, label: string, observed: number, relation: string): asserts bound is number {
  checkIntegerFrom(bound, label, 0);
  if (bound > MAX_RESPONSE) {
    throw new DataError(`${label} must be at most ${MAX_RESPONSE}, the largest response, found ${bound}.`);
  }
  if (relation.startsWith("at most") ? bound > observed : bound < observed) {
    throw new DataError(`${label} must be ${relation} response, ${observed}, found ${bound}.`);
  }
}
