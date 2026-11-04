import { checkArray, checkOptions, chiSquareUpperTail, DataError, describe, type FieldNames } from "@measurekit/core";
import { checkResponseMatrix, type ResponseMatrix } from "../data/internal.js";
import { type ConditionalCalibrationOptions, type ConditionalFit, conditionalFit } from "../irt/internal.js";
import { normalCdf } from "../math/internal.js";

/**
 * How the persons are split into groups: at the median or the mean of the total score, into a lower group, at or below
 * it, and an upper group, or by a label for each person.
 */
export type RaschSplit = "median" | "mean" | readonly string[];

/** Options of {@link andersenTest} and {@link raschWaldTest}. */
export interface RaschTestOptions extends ConditionalCalibrationOptions {
  /**
   * The split of the persons. With missing responses, the median and the mean are those of the persons who answered
   * the same items, as their totals are over the same items. Labels form a group for each distinct label, in sorted
   * order.
   *
   * @defaultValue `"median"`
   */
  readonly split?: RaschSplit;
}

const RASCH_TEST_OPTIONS: FieldNames<RaschTestOptions> = {
  split: true,
  model: true,
  tolerance: true,
  maxIterations: true,
};

/** A likelihood-ratio test of the Rasch model, a partial credit model or a rating scale model. */
export interface RaschTest {
  /** The likelihood-ratio statistic: twice the gain in conditional log-likelihood of the larger model, at least 0. */
  readonly statistic: number;
  /** The parameters of the larger model less those of the smaller. */
  readonly degreesOfFreedom: number;
  /** The upper tail of the chi-square at the statistic. */
  readonly significance: number;
  /** The persons the test uses. */
  readonly persons: number;
  /** Whether every conditional maximum likelihood fit of the test converged. */
  readonly converged: boolean;
}

/** Andersen's likelihood-ratio test of the invariance of the item parameters across groups of persons. */
export interface AndersenTest extends RaschTest {
  /**
   * The ids of the items left out of the test, in item order: those without a response in one of their categories
   * from the persons a group's conditional likelihood uses, as their parameters there are infinite.
   */
  readonly excluded: readonly string[];
}

/**
 * How the items are split into sets: at the median or the mean of their total scores over the persons who answered every
 * item, into the items above it and the others, or as two or more sets of item ids that hold every item once.
 */
export type ItemSplit = "median" | "mean" | readonly (readonly string[])[];

/** Options of {@link martinLofTest}. */
export interface MartinLofOptions extends ConditionalCalibrationOptions {
  /**
   * The split of the items. Each set needs at least two items.
   *
   * @defaultValue `"median"`
   */
  readonly split?: ItemSplit;
}

const MARTIN_LOF_OPTIONS: FieldNames<MartinLofOptions> = {
  split: true,
  model: true,
  tolerance: true,
  maxIterations: true,
};

/** The Wald test of one item. */
export interface RaschWaldItem {
  /** The item id. */
  readonly id: string;
  /**
   * The difference of the item's location in the second group and in the first, over its standard error; positive
   * when the item is harder for the second group. `null` for an item left out, or without a standard error.
   */
  readonly z: number | null;
  /** The two-sided probability of the standard normal beyond z; `null` with `z`. */
  readonly significance: number | null;
}

/** The Wald test of each item between two groups of persons. */
export interface RaschWaldTest {
  /** One row for each item, in item order. */
  readonly items: readonly RaschWaldItem[];
  /** The ids of the items left out, in item order, as {@link AndersenTest}'s. */
  readonly excluded: readonly string[];
  /** The persons the two groups' conditional likelihoods use. */
  readonly persons: number;
  /** Whether both groups' conditional maximum likelihood fits converged. */
  readonly converged: boolean;
}

const DEFAULT_SPLIT = "median";

/** The groups of a split of the persons, as the rows of each group in row order, the groups in order. */
type PersonGroups = readonly Int32Array[];

/**
 * Returns Andersen's likelihood-ratio test that the items of the Rasch model, a partial credit model or a rating scale
 * model have the same parameters in groups of persons, as eRm's `LRtest`.
 *
 * @remarks
 * The model is estimated by conditional maximum likelihood (`conditionalMaximumLikelihood`) once on all the
 * persons and once in each group; with L the conditional likelihoods, the statistic is 2 (Σ ln L_group - ln L_all),
 * about chi-square on the parameters of the groups less those of all when the model holds. Conditioning on the total
 * score makes the test free of the distribution of theta, so groups may differ in ability, as they do when split by
 * the total score.
 *
 * An item with no response in one of its categories from the persons a group's likelihood uses has an infinite
 * parameter there; it leaves the test, and the persons are counted again without it until no such item is left.
 * Every fit then uses the remaining items.
 *
 * @param matrix - The responses of the persons.
 * @param options - The split of the persons, and the model and stopping rules of conditional maximum likelihood.
 * @returns The statistic, its degrees of freedom and significance, the persons used, the items left out, and whether
 *   every fit converged.
 * @throws `DataError` when the matrix or an option is invalid, the split leaves fewer than two groups with persons, fewer
 * than two items remain, or a fit throws, as `conditionalMaximumLikelihood` does.
 * @see {@link https://doi.org/10.1007/BF02291180 | Andersen (1973)}
 * @see {@link https://doi.org/10.18637/jss.v020.i09 | Mair and Hatzinger (2007)}
 */
export function andersenTest(matrix: ResponseMatrix, options: RaschTestOptions = {}): AndersenTest {
  const { groups, fitOptions } = checkedSplit(matrix, options);
  const kept = testedItems(matrix, groups);
  const all = subFit(matrix, null, kept, fitOptions).calibration;
  const parts = groups.map((rows) => subFit(matrix, rows, kept, fitOptions).calibration);
  const gain = parts.reduce((sum, part) => sum + part.logLikelihood, 0) - all.logLikelihood;
  const degreesOfFreedom = parts.reduce((sum, part) => sum + part.parameters, 0) - all.parameters;
  return {
    ...likelihoodRatio(gain, degreesOfFreedom),
    persons: all.persons,
    converged: [all, ...parts].every((fit) => fit.converged),
    excluded: matrix.itemIds.filter((_, item) => !kept.includes(item)),
  };
}

/**
 * Returns the Wald test of each item of the Rasch model, a partial credit model or a rating scale model between two
 * groups of persons, as eRm's `Waldtest` for 0/1 items.
 *
 * @remarks
 * The model is estimated by conditional maximum likelihood (`conditionalMaximumLikelihood`) in each group, each
 * on the scale where the thresholds of all items sum to 0. An item's location is its difficulty, or the mean of its
 * thresholds, and z = (location₂ - location₁) / √(SE₁² + SE₂²), about N(0, 1) when the item has the same parameters
 * in both groups. Items leave the test as in {@link andersenTest}.
 *
 * @param matrix - The responses of the persons.
 * @param options - The split of the persons into two groups, and the model and stopping rules of conditional maximum
 *   likelihood.
 * @returns z and its significance for each item, the items left out, the persons used, and whether both fits
 *   converged.
 * @throws `DataError` when the matrix or an option is invalid, the split does not give two groups with persons, fewer
 * than two items remain, or a fit throws, as `conditionalMaximumLikelihood` does.
 * @see {@link https://doi.org/10.1007/978-1-4612-4230-7_5 | Glas and Verhelst (1995)}
 * @see {@link https://doi.org/10.18637/jss.v020.i09 | Mair and Hatzinger (2007)}
 */
export function raschWaldTest(matrix: ResponseMatrix, options: RaschTestOptions = {}): RaschWaldTest {
  const { groups, fitOptions } = checkedSplit(matrix, options);
  if (groups.length !== 2) {
    throw new DataError(`The Wald test compares two groups of persons, found ${groups.length}.`);
  }
  const kept = testedItems(matrix, groups);
  const fits = groups.map((rows) => subFit(matrix, rows, kept, fitOptions));
  const [first, second] = fits.map(locations);
  const items = matrix.itemIds.map((id, item): RaschWaldItem => {
    const position = kept.indexOf(item);
    const a = first?.[position];
    const b = second?.[position];
    if (a === undefined || b === undefined || a.variance === null || b.variance === null) {
      return { id, z: null, significance: null };
    }
    const z = (b.value - a.value) / Math.sqrt(a.variance + b.variance);
    return Number.isFinite(z)
      ? { id, z, significance: 2 * normalCdf(-Math.abs(z)) }
      : { id, z: null, significance: null };
  });
  return {
    items,
    excluded: matrix.itemIds.filter((_, item) => !kept.includes(item)),
    persons: fits.reduce((sum, fit) => sum + fit.calibration.persons, 0),
    converged: fits.every((fit) => fit.calibration.converged),
  };
}

/**
 * Returns the Martin-Löf test that sets of items of the Rasch model, a partial credit model or a rating scale model
 * measure the same trait, as eRm's `MLoef`.
 *
 * @remarks
 * With n the persons, n_r the persons of each total score, n_(r₁…r_G) those of each combination of the sets' scores and
 * L the conditional likelihoods of all the items and of each set alone, the statistic is
 * 2 (Σ n_(r₁…r_G) ln(n_(r₁…r_G) / n) - Σ n_r ln(n_r / n) + Σ ln L_set - ln L_all), about chi-square when the model holds,
 * on the cells of the sets' scores less 1 plus the sets' parameters, less the total scores and the parameters of all
 * the items. The test uses the persons who answered every item.
 *
 * @param matrix - The responses of the persons.
 * @param options - The split of the items, and the model and stopping rules of conditional maximum likelihood.
 * @returns The statistic, its degrees of freedom and significance, the persons used, and whether every fit converged.
 * @throws `DataError` when the matrix or an option is invalid, no person answered every item, a set has fewer than two
 * items, or a fit throws, as `conditionalMaximumLikelihood` does.
 * @see {@link https://doi.org/10.1007/978-1-4612-4230-7_5 | Glas and Verhelst (1995)}
 * @see {@link https://doi.org/10.18637/jss.v020.i09 | Mair and Hatzinger (2007)}
 */
export function martinLofTest(matrix: ResponseMatrix, options: MartinLofOptions = {}): RaschTest {
  checkResponseMatrix(matrix);
  checkOptions(options, MARTIN_LOF_OPTIONS);
  const { split = DEFAULT_SPLIT, ...fitOptions } = options;
  const rows = Int32Array.from(matrix.responses.keys()).filter((row) => !matrix.responses[row]?.includes(null));
  if (rows.length === 0) throw new DataError("The Martin-Löf test needs persons who answered every item, found none.");
  const sets = itemSets(matrix, rows, split);
  const allItems = Int32Array.from(matrix.itemIds.keys());
  const all = subFit(matrix, rows, allItems, fitOptions).calibration;
  const parts = sets.map((set) => subFit(matrix, rows, set, fitOptions).calibration);
  const highest = Int32Array.from(allItems, (item) =>
    rows.reduce((top, row) => Math.max(top, matrix.responses[row]?.[item] ?? 0), 0),
  );
  const scoreRange = (set: Int32Array): number => set.reduce((sum, item) => sum + (highest[item] ?? 0), 0);
  const cells = sets.reduce((product, set) => product * (scoreRange(set) + 1), 1);
  const degreesOfFreedom =
    cells - 1 + parts.reduce((sum, part) => sum + part.parameters, 0) - (scoreRange(allItems) + all.parameters);
  const scores = (set: Int32Array, row: number): number =>
    set.reduce((sum, item) => sum + (matrix.responses[row]?.[item] ?? 0), 0);
  const tables = Array.from(rows, (row) => sets.map((set) => scores(set, row)).join());
  const totals = Array.from(rows, (row) => scores(allItems, row));
  const gain =
    countTerm(tables, rows.length) -
    countTerm(totals, rows.length) +
    parts.reduce((sum, part) => sum + part.logLikelihood, 0) -
    all.logLikelihood;
  return {
    ...likelihoodRatio(gain, degreesOfFreedom),
    persons: rows.length,
    converged: [all, ...parts].every((fit) => fit.converged),
  };
}

/** Returns the statistic 2 × gain, held at 0 against rounding, with its degrees of freedom and significance. */
function likelihoodRatio(
  gain: number,
  degreesOfFreedom: number,
): Pick<RaschTest, "statistic" | "degreesOfFreedom" | "significance"> {
  const statistic = Math.max(0, 2 * gain);
  return { statistic, degreesOfFreedom, significance: chiSquareUpperTail(statistic, degreesOfFreedom) };
}

/** Returns Σ n_c ln(n_c / n) over the distinct values c of `values`, n of them in all. */
function countTerm(values: readonly (string | number)[], n: number): number {
  const counts = new Map<string | number, number>();
  for (const value of values) counts.set(value, (counts.get(value) ?? 0) + 1);
  let sum = 0;
  for (const count of counts.values()) sum += count * Math.log(count / n);
  return sum;
}

/**
 * Checks the matrix and the options of a test that splits the persons, and returns the groups and the options of the
 * fits.
 *
 * @throws `DataError` when the matrix, the options or the split are invalid, or the split leaves fewer than two groups
 * with persons.
 */
function checkedSplit(
  matrix: ResponseMatrix,
  options: RaschTestOptions,
): { readonly groups: PersonGroups; readonly fitOptions: ConditionalCalibrationOptions } {
  checkResponseMatrix(matrix);
  checkOptions(options, RASCH_TEST_OPTIONS);
  const { split = DEFAULT_SPLIT, ...fitOptions } = options;
  const labels = personLabels(matrix, split);
  const distinct = [...new Set(labels)].sort();
  if (distinct.length < 2) {
    throw new DataError(`The split puts every person in one group; the test needs at least two groups of persons.`);
  }
  const groups = distinct.map((label) => Int32Array.from(labels.keys()).filter((row) => labels[row] === label));
  return { groups, fitOptions };
}

/** Lower and upper groups of a split at the median or the mean, in that order of their labels. */
const LOWER = "0 lower";
const UPPER = "1 upper";

/**
 * Returns the label of each person's group.
 *
 * @throws `DataError` when the split is neither `"median"`, `"mean"` nor one string for each person.
 */
function personLabels(matrix: ResponseMatrix, split: RaschSplit): readonly string[] {
  if (split === "median" || split === "mean") {
    // A total is compared with the center of the persons who answered the same items.
    const patterns = matrix.responses.map((row) => row.map((response) => (response === null ? "0" : "1")).join(""));
    const totals = matrix.responses.map((row) => row.reduce<number>((sum, response) => sum + (response ?? 0), 0));
    const byPattern = new Map<string, number[]>();
    patterns.forEach((pattern, row) => {
      const list = byPattern.get(pattern) ?? [];
      list.push(totals[row] ?? 0);
      byPattern.set(pattern, list);
    });
    const centers = new Map([...byPattern].map(([pattern, values]) => [pattern, center(values, split)]));
    return totals.map((total, row) => (total <= (centers.get(patterns[row] ?? "") ?? 0) ? LOWER : UPPER));
  }
  checkArray(split, 'The option split must be "median", "mean" or an array of labels');
  if (split.length !== matrix.personIds.length) {
    throw new DataError(
      `The option split must hold one label for each of the ${matrix.personIds.length} persons, found ${split.length}.`,
    );
  }
  split.forEach((label, row) => {
    if (typeof label !== "string") {
      throw new DataError(`The label of person "${matrix.personIds[row]}" must be a string, found ${describe(label)}.`);
    }
  });
  return split;
}

/** Returns the median, the mean of the middle two for an even count, or the mean of the values. */
function center(values: readonly number[], split: "median" | "mean"): number {
  if (split === "mean") return values.reduce((sum, value) => sum + value, 0) / values.length;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? (sorted[middle] ?? 0) : ((sorted[middle - 1] ?? 0) + (sorted[middle] ?? 0)) / 2;
}

/**
 * Returns the items the test keeps, in item order: those with a response in every category from 0 to their highest in
 * the matrix from the persons each group's conditional likelihood uses, which are counted again after each item left
 * out, as a person's total changes with the items.
 *
 * @throws `DataError` when fewer than two items remain.
 */
function testedItems(matrix: ResponseMatrix, groups: PersonGroups): Int32Array {
  const { responses, itemIds } = matrix;
  const highest = Int32Array.from(itemIds, (_, item) =>
    responses.reduce((top, row) => Math.max(top, row[item] ?? 0), 0),
  );
  // A group shows an item's categories only if it has more persons than the item's highest score, so an item of a far
  // higher score leaves at once, before its categories take room.
  const smallest = groups.reduce((least, rows) => Math.min(least, rows.length), Number.POSITIVE_INFINITY);
  let kept = Int32Array.from(itemIds.keys()).filter((item) => {
    const top = highest[item] ?? 0;
    return top >= 1 && top < smallest;
  });
  for (;;) {
    const seen = groups.map((rows) => categoriesSeen(responses, rows, kept, highest));
    const next = kept.filter((item) => seen.every((counts) => counts[item] === (highest[item] ?? 0) + 1));
    if (next.length === kept.length) break;
    kept = next;
  }
  if (kept.length < 2) {
    throw new DataError(
      `The test needs at least two items with a response in each of their categories in every group, found ${kept.length}.`,
    );
  }
  return kept;
}

/**
 * Returns, for each item, how many of its categories have a response from the persons of `rows` whom a conditional
 * likelihood of the `kept` items uses; 0 for an item not kept.
 */
function categoriesSeen(
  responses: ResponseMatrix["responses"],
  rows: Int32Array,
  kept: Int32Array,
  highest: Int32Array,
): Int32Array {
  const isKept = new Uint8Array(highest.length);
  for (const item of kept) isKept[item] = 1;
  const offsets = new Int32Array(highest.length + 1);
  highest.forEach((top, item) => {
    offsets[item + 1] = (offsets[item] ?? 0) + (isKept[item] === 1 ? top + 1 : 0);
  });
  const flags = new Uint8Array(offsets[highest.length] ?? 0);
  for (const row of rows) {
    const values = responses[row] ?? [];
    if (!isUsed(values, kept, highest)) continue;
    for (const item of kept) {
      const response = values[item] ?? null;
      if (response !== null) flags[(offsets[item] ?? 0) + response] = 1;
    }
  }
  return Int32Array.from(highest, (_, item) =>
    flags.subarray(offsets[item] ?? 0, offsets[item + 1] ?? 0).reduce((sum, flag) => sum + flag, 0),
  );
}

/**
 * Returns whether a conditional likelihood of the `kept` items uses a person: two or more answered items and a total
 * neither 0 nor its highest possible.
 */
function isUsed(values: readonly (number | null)[], kept: Int32Array, highest: Int32Array): boolean {
  let answered = 0;
  let total = 0;
  let possible = 0;
  for (const item of kept) {
    const response = values[item] ?? null;
    if (response === null) continue;
    answered++;
    total += response;
    possible += highest[item] ?? 0;
  }
  return answered >= 2 && total > 0 && total < possible;
}

/**
 * Returns the sets of item indices of a split of the items, by their totals over the persons of `rows`.
 *
 * @throws `DataError` when the split is invalid or a set has fewer than two items.
 */
function itemSets(matrix: ResponseMatrix, rows: Int32Array, split: ItemSplit): readonly Int32Array[] {
  const { itemIds, responses } = matrix;
  if (split === "median" || split === "mean") {
    const totals = itemIds.map((_, item) => rows.reduce((sum, row) => sum + (responses[row]?.[item] ?? 0), 0));
    const middle = center(totals, split);
    const sets = [false, true].map((above) =>
      Int32Array.from(itemIds.keys()).filter((item) => (totals[item] ?? 0) > middle === above),
    );
    return checkedSets(itemIds, sets);
  }
  checkArray(split, 'The option split must be "median", "mean" or an array of sets of item ids');
  if (split.length < 2) {
    throw new DataError(`The option split needs at least two sets of item ids, found ${split.length}.`);
  }
  const positions = new Map(itemIds.map((id, item) => [id, item]));
  const placed = new Set<string>();
  const sets = split.map((set, index) => {
    checkArray(set, `Set ${index + 1} of the option split must be an array of item ids`);
    return Int32Array.from(set, (id: unknown) => {
      const item = typeof id === "string" && !placed.has(id) ? positions.get(id) : undefined;
      if (item === undefined) {
        throw new DataError(
          `Set ${index + 1} of the option split names ${describe(id)}, which is not an item id or is in an earlier set.`,
        );
      }
      placed.add(itemIds[item] ?? "");
      return item;
    });
  });
  if (placed.size !== itemIds.length) {
    const missing = itemIds.find((id) => !placed.has(id));
    throw new DataError(`The option split must place every item in a set; item "${missing}" is in none.`);
  }
  return checkedSets(
    itemIds,
    sets.map((set) => set.sort()),
  );
}

/**
 * Checks that each set has at least two items, as a conditional likelihood needs.
 *
 * @throws `DataError` naming a set with fewer.
 */
function checkedSets(itemIds: readonly string[], sets: readonly Int32Array[]): readonly Int32Array[] {
  for (const set of sets) {
    if (set.length < 2) {
      const names = Array.from(set, (item) => `"${itemIds[item]}"`).join(", ") || "no items";
      throw new DataError(`Each set of items of the Martin-Löf test needs at least two items, found ${names}.`);
    }
  }
  return sets;
}

/** Fits the conditional likelihood of the responses of `rows` (every person for `null`) to the `items`. */
function subFit(
  matrix: ResponseMatrix,
  rows: Int32Array | null,
  items: Int32Array,
  options: ConditionalCalibrationOptions,
): ConditionalFit {
  const persons = rows ?? Int32Array.from(matrix.responses.keys());
  return conditionalFit(
    {
      personIds: Array.from(persons, (row) => matrix.personIds[row] ?? ""),
      itemIds: Array.from(items, (item) => matrix.itemIds[item] ?? ""),
      responses: Array.from(persons, (row) => Array.from(items, (item) => matrix.responses[row]?.[item] ?? null)),
    },
    options,
  );
}

/** An item's location and its variance, `null` without one. */
interface Location {
  readonly value: number;
  readonly variance: number | null;
}

/**
 * Returns each item's location, the mean of its thresholds, and its variance from the fit's covariance matrix, whose rows
 * are the thresholds item by item or, under the rating scale model, the locations and then the steps.
 */
function locations({ calibration, covariance }: ConditionalFit): readonly Location[] {
  const { items } = calibration;
  const count = covariance === null ? 0 : Math.round(Math.sqrt(covariance.length));
  const entry = (row: number, column: number): number => covariance?.[row * count + column] ?? 0;
  let start = 0;
  return items.map(({ parameters }, index) => {
    if ("steps" in parameters) {
      return { value: parameters.b, variance: covariance === null ? null : entry(index, index) };
    }
    const thresholds = typeof parameters.b === "number" ? [parameters.b] : parameters.b;
    const m = thresholds.length;
    let variance = 0;
    for (let k = start; k < start + m; k++) {
      for (let l = start; l < start + m; l++) variance += entry(k, l);
    }
    start += m;
    return {
      value: thresholds.reduce((sum, threshold) => sum + threshold, 0) / m,
      variance: covariance === null ? null : variance / (m * m),
    };
  });
}
