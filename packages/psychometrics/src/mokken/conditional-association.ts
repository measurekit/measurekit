import { checkOptions, checkPositiveInteger, DataError, type FieldNames, normalCdf } from "@measurekit/core";
import type { ResponseMatrix } from "../data/internal.js";
import { sampleQuantile } from "../math/internal.js";
import { type RestScoreGroups, restScoreGroups } from "./groups.js";
import { type MokkenData, mokkenData, scoreOf } from "./scores.js";

/** Options of {@link conditionalAssociation}. */
export interface ConditionalAssociationOptions {
  /**
   * The fewest persons in a group of the rest score of a pair of items, mokken's `MINSIZE`; a positive integer, at most
   * the persons who answered every item.
   *
   * @defaultValue `4`
   */
  readonly minimumSize?: number;
}

const CONDITIONAL_ASSOCIATION_OPTIONS: FieldNames<ConditionalAssociationOptions> = { minimumSize: true };

/** The pairs and items whose index of conditional association is above its fence. */
export interface ConditionalAssociationFlags {
  /** The pairs [a, c] whose W1 is flagged: a's association with the other items, given c, is negative. */
  readonly w1: readonly (readonly [string, string])[];
  /** The items whose W2 is flagged. */
  readonly w2: readonly string[];
  /** The pairs [a, b], a before b in the matrix, whose W3 is flagged. */
  readonly w3: readonly (readonly [string, string])[];
}

/** The indices W1, W2 and W3 of conditional association of a set of items, step 5 of a Mokken scale analysis. */
export interface ConditionalAssociation {
  /** The item ids, which index the rows and columns of the indices. */
  readonly items: readonly string[];
  /** W1 by item a, then the conditioning item c; `null` on the diagonal. */
  readonly w1: readonly (readonly (number | null)[])[];
  /** W2 of each item. */
  readonly w2: readonly number[];
  /** W3 by item, then item, symmetric; `null` on the diagonal. */
  readonly w3: readonly (readonly (number | null)[])[];
  /** The pairs and items above the fence of each index, as `check.ca` flags them. */
  readonly flagged: ConditionalAssociationFlags;
}

const DEFAULT_MINIMUM_SIZE = 4;
/** mokken's `MINGROUP`: a group enters an index with at least 4 persons, as Fisher's z of its correlation needs N - 3 > 0. */
const MIN_GROUP = 4;
/** mokken's term that keeps Fisher's z of a correlation of 1 finite. */
const UNIT_CORRELATION_GUARD = 1e-40;
/** `check.ca` flags a value above Q3 + 3 (Q3 - median) of the values of its index. */
const FLAG_MULTIPLE = 3;

/**
 * Returns Straat, van der Ark and Sijtsma's indices of conditional association, W1, W2 and W3, with the items and pairs
 * they flag, as the first pass of the mokken package's `check.ca(Windex = TRUE)`.
 *
 * @remarks
 * Items that measure one trait are associated positively within every group of persons that agree on other items; a
 * negative covariance there points to local dependence. Within a group of n persons, the covariance of two items weighs
 * Φ(-z √(n - 3)), with z Fisher's transformation of their correlation: near 1 for a clearly negative correlation, 1/2
 * for none, and 0 where either item is constant or n is below 4. W1[a][c] sums the weights of a with every other item j
 * within the groups of persons with each score on c: high where a and c are positively dependent. W3[a][b] sums those of
 * a and b within the groups of their rest score, the sum of the other items, cut as `monotonicity` cuts it with at least
 * `minimumSize` persons a group; W2[a] is the sum of W3[a][b] over b: both are high where an item is negatively
 * dependent. An index flags a value above Q3 + 3 (Q3 - median) of all of its values, with quartiles by linear
 * interpolation. `check.ca` then removes the item with the most flags and runs again; to do so, drop it from the matrix
 * and call the function on the rest. The statistic uses the persons who answered every item, with the lowest score of
 * all items taken as 0.
 *
 * The time grows with the cube of the items times the persons: W1 sums over every item, conditioning item and third item.
 *
 * @param matrix - The responses of the persons.
 * @param options - The smallest group of a rest score.
 * @returns The indices by item and pair, and the items and pairs they flag.
 * @throws `DataError` when the matrix is invalid, has fewer than three items or more than 1,000, fewer than two persons
 * answered every item, the scores span more than 10 categories, or `minimumSize` is not a positive integer or exceeds
 * the persons who answered every item.
 * @see {@link https://doi.org/10.1111/bmsp.12078 | Sijtsma and van der Ark (2017)}
 */
export function conditionalAssociation(
  matrix: ResponseMatrix,
  options: ConditionalAssociationOptions = {},
): ConditionalAssociation {
  checkOptions(options, CONDITIONAL_ASSOCIATION_OPTIONS);
  const { minimumSize = DEFAULT_MINIMUM_SIZE } = options;
  checkPositiveInteger(minimumSize, "The option minimumSize");
  const data = mokkenData(matrix, "Conditional association", 3);
  if (minimumSize > data.persons) {
    throw new DataError(
      `The option minimumSize must be at most the ${data.persons} persons who answered every item, found ${minimumSize}.`,
    );
  }
  const { items, itemIds } = data;
  const w1 = conditioningIndex(data);
  const w3 = restScoreIndex(data, minimumSize);
  const w2 = Array.from({ length: items }, (_, a) => {
    let sum = 0;
    for (let b = 0; b < items; b++) if (b !== a) sum += w3[a * items + b] ?? 0;
    return sum;
  });
  const offDiagonal = (index: Float64Array): number[] =>
    Array.from(index).filter((_, entry) => Math.floor(entry / items) !== entry % items);
  const fence1 = fence(offDiagonal(w1));
  const fence2 = fence(w2);
  const fence3 = fence(offDiagonal(w3));
  const id = (item: number): string => itemIds[item] ?? "";
  const flagged1: [string, string][] = [];
  const flagged3: [string, string][] = [];
  for (let a = 0; a < items; a++) {
    for (let b = 0; b < items; b++) {
      if (a !== b && (w1[a * items + b] ?? 0) > fence1) flagged1.push([id(a), id(b)]);
      if (a < b && (w3[a * items + b] ?? 0) > fence3) flagged3.push([id(a), id(b)]);
    }
  }
  const square = (index: Float64Array): (number | null)[][] =>
    Array.from({ length: items }, (_, a) =>
      Array.from({ length: items }, (_, b) => (a === b ? null : (index[a * items + b] ?? 0))),
    );
  return {
    items: itemIds,
    w1: square(w1),
    w2,
    w3: square(w3),
    flagged: {
      w1: flagged1,
      w2: itemIds.filter((_, item) => (w2[item] ?? 0) > fence2),
      w3: flagged3,
    },
  };
}

/** Returns Q3 + 3 (Q3 - median) of the values. */
function fence(values: readonly number[]): number {
  const sorted = Float64Array.from(values).sort();
  const upper = sampleQuantile(sorted, 0.75);
  return upper + FLAG_MULTIPLE * (upper - sampleQuantile(sorted, 0.5));
}

/**
 * Returns mokken's weight of the covariance of two items in a group, Φ(-z √(n - 3)) for Fisher's z of their correlation,
 * from the group's size and sums; 0 for a group below 4 persons or an item without variance in it.
 */
function covarianceWeight(n: number, sumA: number, sumB: number, squaresA: number, squaresB: number, products: number) {
  if (n < MIN_GROUP) return 0;
  const varianceA = n * squaresA - sumA * sumA;
  const varianceB = n * squaresB - sumB * sumB;
  if (varianceA === 0 || varianceB === 0) return 0;
  const r = Math.max(-1, Math.min(1, (n * products - sumA * sumB) / Math.sqrt(varianceA * varianceB)));
  const z = 0.5 * Math.log((1 + r) / (1 - r + UNIT_CORRELATION_GUARD));
  return normalCdf(-z * Math.sqrt(n - 3));
}

/** Returns W1 by item a, then the conditioning item c: the weights of a with each other item j, within each score of c. */
function conditioningIndex(data: MokkenData): Float64Array {
  const { items, persons, categories } = data;
  const index = new Float64Array(items * items);
  const sums = new Float64Array(items);
  const products = new Float64Array(items * items);
  const members = new Int32Array(persons);
  for (let c = 0; c < items; c++) {
    for (let score = 0; score < categories; score++) {
      let n = 0;
      for (let person = 0; person < persons; person++) if (scoreOf(data, person, c) === score) members[n++] = person;
      if (n < MIN_GROUP) continue;
      crossProducts(data, members.subarray(0, n), sums, products);
      addConditionedWeights(index, items, c, n, sums, products);
    }
  }
  return index;
}

/** Adds to W1[a][c] the weight of a with every other item j but c, in a group of n persons with the sums given. */
function addConditionedWeights(
  index: Float64Array,
  items: number,
  c: number,
  n: number,
  sums: Float64Array,
  products: Float64Array,
): void {
  const product = (k: number, l: number): number => products[k * items + l] ?? 0;
  for (let a = 0; a < items; a++) {
    for (let j = a + 1; j < items; j++) {
      if (a === c || j === c) continue;
      const weight = covarianceWeight(n, sums[a] ?? 0, sums[j] ?? 0, product(a, a), product(j, j), product(a, j));
      index[a * items + c] = (index[a * items + c] ?? 0) + weight;
      index[j * items + c] = (index[j * items + c] ?? 0) + weight;
    }
  }
}

/** Writes the sums of each item's scores over `members` to `sums`, and the sums of products to the upper triangle of `products`. */
function crossProducts(data: MokkenData, members: Int32Array, sums: Float64Array, products: Float64Array): void {
  const { items } = data;
  sums.fill(0);
  products.fill(0);
  for (const person of members) {
    for (let a = 0; a < items; a++) {
      const x = scoreOf(data, person, a);
      sums[a] = (sums[a] ?? 0) + x;
      for (let j = a; j < items; j++)
        products[a * items + j] = (products[a * items + j] ?? 0) + x * scoreOf(data, person, j);
    }
  }
}

/** Returns W3 by item, then item: the weights of each pair within the groups of its rest score. */
function restScoreIndex(data: MokkenData, minimumSize: number): Float64Array {
  const { items, persons } = data;
  const index = new Float64Array(items * items);
  const totals = Float64Array.from({ length: persons }, (_, person) => {
    let total = 0;
    for (let item = 0; item < items; item++) total += scoreOf(data, person, item);
    return total;
  });
  for (let a = 0; a < items; a++) {
    for (let b = a + 1; b < items; b++) {
      const rest = totals.map((total, person) => total - scoreOf(data, person, a) - scoreOf(data, person, b));
      const weight = restScoreWeight(data, a, b, restScoreGroups(rest, minimumSize));
      index[a * items + b] = weight;
      index[b * items + a] = weight;
    }
  }
  return index;
}

/** Returns the sum of the weights of the covariance of items a and b within each group of their rest score. */
function restScoreWeight(data: MokkenData, a: number, b: number, groups: RestScoreGroups): number {
  // The size and the sums of a, b, a², b² and ab of each group.
  const moments = new Float64Array(groups.count * 6);
  for (let person = 0; person < data.persons; person++) {
    const at = (groups.members[person] ?? 0) * 6;
    const x = scoreOf(data, person, a);
    const y = scoreOf(data, person, b);
    moments[at] = (moments[at] ?? 0) + 1;
    moments[at + 1] = (moments[at + 1] ?? 0) + x;
    moments[at + 2] = (moments[at + 2] ?? 0) + y;
    moments[at + 3] = (moments[at + 3] ?? 0) + x * x;
    moments[at + 4] = (moments[at + 4] ?? 0) + y * y;
    moments[at + 5] = (moments[at + 5] ?? 0) + x * y;
  }
  let sum = 0;
  for (let at = 0; at < moments.length; at += 6) {
    const moment = (k: number): number => moments[at + k] ?? 0;
    sum += covarianceWeight(moment(0), moment(1), moment(2), moment(3), moment(4), moment(5));
  }
  return sum;
}
