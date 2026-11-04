import { checkOptions, type FieldNames } from "@measurekit/core";
import type { ResponseMatrix } from "../data/internal.js";
import type { IrtItem } from "../irt/internal.js";
import {
  checkComparedItems,
  checkModelInputs,
  type ModelInputs,
  matchProbability,
  personProbabilities,
  probabilitiesOf,
  readKey,
} from "./inputs.js";
import type { PersonPair } from "./pairs.js";

/** Options of {@link m4Index}. */
export interface M4IndexOptions {
  /**
   * The keyed category of each item, in the order of the matrix's items: the option that is correct.
   *
   * @defaultValue category 1 of every item, which needs every item dichotomous
   */
  readonly key?: readonly number[];
}

const M4_INDEX_OPTIONS: FieldNames<M4IndexOptions> = { key: true };

/** Maynes' M4 of one pair. */
export interface M4Index {
  /** The two persons. */
  readonly persons: PersonPair;
  /** The items both persons answered, which the index compares. */
  readonly compared: number;
  /** The compared items both answered with the keyed category. */
  readonly identicalCorrect: number;
  /** The compared items both answered with the same other category. */
  readonly identicalIncorrect: number;
  /**
   * The probability, without copying, of the observed counts or of counts as extreme: small values point to copying.
   */
  readonly significance: number;
}

/**
 * Returns Maynes' M4 of each pair: whether two persons gave identical correct and identical incorrect responses
 * together more often than two persons who answer alone at their abilities do.
 *
 * @remarks
 * Two persons who answer alone give item i the keyed response both with probability P_i = P_ik(θ₁) P_ik(θ₂), the
 * same other response with Q_i = Σ_(j≠k) P_ij(θ₁) P_ij(θ₂), and different responses otherwise. Over the items both
 * answered, the identical correct and identical incorrect counts follow the generalized trinomial distribution of
 * these probabilities, computed exactly. A pair of counts (x, y) is as extreme as the observed (m, n) when
 * P(X ≥ x, Y ≥ y) is at most P(X ≥ m, Y ≥ n), and the significance sums the probabilities of those pairs, as
 * CopyDetect does. The index is symmetric in the persons. With dichotomous items the key is category 1 and every
 * incorrect response is the same. Missing responses leave their item out of the pair. Time grows with the cube of the
 * items a pair compares, at most 1,000, and the table of counts holds their square: about 0.3 s and 16 MB at the limit.
 *
 * @param matrix - The responses: each person's category of each item, such as the option chosen, numbered from 0.
 * @param items - The calibrated items, in the order of the matrix's items, such as `nominalResponse` items.
 * @param thetas - Each person's theta, in the order of the matrix.
 * @param pairs - The pairs of persons.
 * @param options - The keyed category of each item.
 * @returns For each pair, the identical correct and incorrect responses and their significance.
 * @throws `DataError` when the matrix, the items, the thetas, the pairs or the key are invalid, a response is not a
 * category of its item, the key is absent while an item is not dichotomous, a pair names a person twice or a person the
 * matrix lacks, or a pair compares more than 1,000 items.
 * @see {@link https://doi.org/10.4324/9781315884677 | Maynes (2014)}
 * @see {@link https://doi.org/10.4324/9781315743097-3 | Maynes (2017)}
 *
 * @example
 * ```ts
 * import { buildResponseMatrix, twoParameterLogistic } from "@measurekit/psychometrics";
 * import { m4Index } from "@measurekit/psychometrics/similarity";
 *
 * const items = [-1, -0.5, 0, 0.5, 1, 1.5].map((b) => ({ model: twoParameterLogistic, parameters: { a: 1.2, b, c: 0, d: 1 } }));
 * const matrix = buildResponseMatrix([
 *   [1, 0, 0, 1, 0, 0],
 *   [1, 0, 0, 1, 0, 0],
 * ]);
 * const [pair] = m4Index(matrix, items, [0, 0], [["person1", "person2"]]);
 * pair?.identicalIncorrect; // => 4
 * pair?.significance; // => 0.0418, two identical correct and four identical incorrect responses
 * ```
 */
export function m4Index(
  matrix: ResponseMatrix,
  items: readonly IrtItem[],
  thetas: readonly number[],
  pairs: readonly PersonPair[],
  options: M4IndexOptions = {},
): readonly M4Index[] {
  const inputs = checkModelInputs(matrix, items, thetas, pairs);
  checkOptions(options, M4_INDEX_OPTIONS);
  const categories = (column: number): number => {
    const item = items[column];
    return item === undefined ? 0 : item.model.categoryCount(item.parameters);
  };
  const dichotomous = items.every((_, column) => categories(column) === 2);
  const key = readKey(
    options.key,
    matrix.itemIds,
    categories,
    dichotomous ? null : "M4 needs the option key when an item has more than two categories.",
  );
  const { personIds } = matrix;
  const trinomial = createTrinomial();
  const firstProbabilities = personProbabilities(inputs);
  const secondProbabilities = personProbabilities(inputs);
  return Array.from({ length: pairs.length }, (_, k): M4Index => {
    const first = inputs.pairs[2 * k] ?? 0;
    const second = inputs.pairs[2 * k + 1] ?? 0;
    const counts = pairCounts(
      inputs,
      key,
      first,
      second,
      probabilitiesOf(inputs, firstProbabilities, first),
      probabilitiesOf(inputs, secondProbabilities, second),
      trinomial,
    );
    return {
      persons: [personIds[first] ?? "", personIds[second] ?? ""],
      ...counts,
      significance: trinomialSignificance(trinomial, counts),
    };
  });
}

/** The generalized trinomial distribution of a pair's identical correct and incorrect counts, in reused arrays. */
interface Trinomial {
  /** The compared items. */
  items: number;
  /** Each compared item's P and Q, at 2i and 2i + 1. */
  probabilities: Float64Array;
  /** The probability of each pair of counts, x identical correct and y identical incorrect at x (items + 1) + y. */
  table: Float64Array;
  /** P(X ≥ x, Y ≥ y), at the place of (x, y) in `table`. */
  tails: Float64Array;
}

function createTrinomial(): Trinomial {
  return { items: 0, probabilities: new Float64Array(0), table: new Float64Array(0), tails: new Float64Array(0) };
}

/** The identical correct and incorrect responses of a pair and the items it compares. */
interface PairCounts {
  readonly compared: number;
  readonly identicalCorrect: number;
  readonly identicalIncorrect: number;
}

/**
 * Returns the counts of a pair and writes the probabilities P and Q of its compared items into `trinomial`.
 *
 * @throws `DataError` when the pair compares more than {@link MAX_COMPARED_ITEMS} items.
 */
function pairCounts(
  { matrix, offsets }: ModelInputs,
  key: Int32Array,
  first: number,
  second: number,
  p: Float64Array,
  q: Float64Array,
  trinomial: Trinomial,
): PairCounts {
  const firstRow = matrix.responses[first] ?? [];
  const secondRow = matrix.responses[second] ?? [];
  if (trinomial.probabilities.length < 2 * firstRow.length) {
    trinomial.probabilities = new Float64Array(2 * firstRow.length);
  }
  let compared = 0;
  for (let column = 0; column < firstRow.length; column++) {
    if (firstRow[column] === null || secondRow[column] === null) continue;
    const keyed = (offsets[column] ?? 0) + (key[column] ?? 0);
    writeItemProbabilities(trinomial.probabilities, compared, p, q, keyed, offsets, column);
    compared++;
  }
  checkComparedItems(compared, "M4");
  trinomial.items = compared;
  return { compared, ...identicalCounts(firstRow, secondRow, key) };
}

/** Returns the items two rows of responses both answered with the keyed category, and with the same other one. */
function identicalCounts(
  firstRow: readonly (number | null)[],
  secondRow: readonly (number | null)[],
  key: Int32Array,
): { readonly identicalCorrect: number; readonly identicalIncorrect: number } {
  let identicalCorrect = 0;
  let identicalIncorrect = 0;
  for (let column = 0; column < firstRow.length; column++) {
    const response = firstRow[column] ?? null;
    if (response === null || response !== secondRow[column]) continue;
    if (response === key[column]) identicalCorrect++;
    else identicalIncorrect++;
  }
  return { identicalCorrect, identicalIncorrect };
}

/**
 * Writes P and Q of an item, the probabilities that two persons who answer alone give both its keyed category, at
 * `keyed` of their probabilities, and both the same other one, into the place of the `compared`th compared item.
 */
function writeItemProbabilities(
  probabilities: Float64Array,
  compared: number,
  p: Float64Array,
  q: Float64Array,
  keyed: number,
  offsets: Int32Array,
  column: number,
): void {
  probabilities[2 * compared] = matchProbability(p, q, keyed, keyed + 1);
  probabilities[2 * compared + 1] =
    matchProbability(p, q, offsets[column] ?? 0, keyed) + matchProbability(p, q, keyed + 1, offsets[column + 1] ?? 0);
}

/** Returns the significance of a pair's counts under the trinomial distribution of its compared items. */
function trinomialSignificance(trinomial: Trinomial, { identicalCorrect, identicalIncorrect }: PairCounts): number {
  const { items, probabilities } = trinomial;
  const width = items + 1;
  if (trinomial.table.length < width * width) {
    trinomial.table = new Float64Array(width * width);
    trinomial.tails = new Float64Array(width * width);
  }
  const { table, tails } = trinomial;
  table.fill(0, 0, width * width);
  table[0] = 1;
  for (let step = 1; step <= items; step++) {
    addTrinomialItem(table, width, step, probabilities[2 * step - 2] ?? 0, probabilities[2 * step - 1] ?? 0);
  }
  writeUpperTails(table, tails, width, items);
  const observed = tails[identicalCorrect * width + identicalIncorrect] ?? 0;
  let significance = 0;
  for (let x = 0; x <= items; x++) {
    for (let cell = x * width; cell <= x * width + items - x; cell++) {
      if ((tails[cell] ?? 0) <= observed) significance += table[cell] ?? 0;
    }
  }
  return Math.min(significance, 1);
}

/**
 * Adds the `step`th item, of identical correct probability `correct` and identical incorrect probability `incorrect`,
 * to the table of the counts of the items before it: in place, from the highest counts down, so that each cell reads
 * the cells of one item fewer before they change.
 */
function addTrinomialItem(table: Float64Array, width: number, step: number, correct: number, incorrect: number): void {
  const neither = 1 - correct - incorrect;
  for (let x = step; x > 0; x--) {
    const row = x * width;
    for (let cell = row + step - x; cell > row; cell--) {
      table[cell] =
        neither * (table[cell] ?? 0) + correct * (table[cell - width] ?? 0) + incorrect * (table[cell - 1] ?? 0);
    }
    table[row] = neither * (table[row] ?? 0) + correct * (table[row - width] ?? 0);
  }
  for (let y = step; y > 0; y--) table[y] = neither * (table[y] ?? 0) + incorrect * (table[y - 1] ?? 0);
  table[0] = neither * (table[0] ?? 0);
}

/**
 * Writes P(X ≥ x, Y ≥ y) of each cell of the table into `tails`: by rows of y, then by columns of x, as sums of positive
 * terms, which round no tail away.
 */
function writeUpperTails(table: Float64Array, tails: Float64Array, width: number, items: number): void {
  for (let x = 0; x <= items; x++) {
    const row = x * width;
    let sum = 0;
    for (let cell = row + items - x; cell >= row; cell--) {
      sum += table[cell] ?? 0;
      tails[cell] = sum;
    }
  }
  for (let y = 0; y <= items; y++) {
    let sum = 0;
    for (let cell = (items - y) * width + y; cell >= 0; cell -= width) {
      sum += tails[cell] ?? 0;
      tails[cell] = sum;
    }
  }
}
