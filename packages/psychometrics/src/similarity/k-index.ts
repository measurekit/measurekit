import { checkOptions, DataError, type FieldNames } from "@measurekit/core";
import { checkOpenUnitInterval } from "../checks.js";
import { checkResponseMatrix, type ResponseMatrix } from "../data/internal.js";
import { orthonormalBasis, poissonFit } from "../math/internal.js";
import { readKey } from "./inputs.js";
import { type PersonPair, readPairs } from "./pairs.js";
import { binomialRange, poissonRange } from "./tails.js";

/** Options of {@link kIndex}. */
// biome-ignore lint/style/useNamingConvention: the K index is the name of this index in the field.
export interface KIndexOptions {
  /**
   * The keyed category of each item, in the order of the matrix's items: the option that is correct.
   *
   * @defaultValue category 1 of every item, which needs every response 0 or 1
   */
  readonly key?: readonly number[];
  /**
   * The probability of a correct guess, g of the weights of S2, between 0 and 1: 1 over the number of options.
   *
   * @defaultValue `0.2`, of five options, as CopyDetect takes for scored responses
   */
  readonly guessing?: number;
}

const K_INDEX_OPTIONS: FieldNames<KIndexOptions> = { key: true, guessing: true };

/** The K index and its variants of one pair. */
// biome-ignore lint/style/useNamingConvention: the K index is the name of this index in the field.
export interface KIndex {
  /** The suspected copier and the source. */
  readonly persons: readonly [copier: string, source: string];
  /** The source's incorrect responses, on which the indices count matches. */
  readonly sourceIncorrect: number;
  /** The source's incorrect responses that the copier gave too: the same category other than the key. */
  readonly identicalIncorrect: number;
  /**
   * Holland's K: the binomial probability of at least `identicalIncorrect` matches, at the matching rate of the
   * copier's number-incorrect group; `null`, as are the variants, when the source has no incorrect response.
   */
  readonly k: number | null;
  /** K1: K at the rate of a linear regression on the number-incorrect groups. */
  readonly k1: number | null;
  /** K2: K at the rate of a quadratic regression on the number-incorrect groups. */
  readonly k2: number | null;
  /** S1: the Poisson probability of `identicalIncorrect` to `sourceIncorrect` matches, at a log-linear mean. */
  readonly s1: number | null;
  /** S2: S1 of the identical incorrect and the weighted identical correct responses, up to the number of items. */
  readonly s2: number | null;
}

const DEFAULT_GUESSING = 0.2;
/** The most cells of the counts of each category of each item by number-incorrect group: 2²⁴ 32-bit counts, 64 MB. */
const MAX_COUNT_CELLS = 2 ** 24;
/** CopyDetect's bounds of the regression rates of K1 and K2, which a straight line or parabola may leave. */
const LOWEST_RATE = 0.001;
const HIGHEST_RATE = 0.999;

/**
 * Returns the K index and its variants K1, K2, S1 and S2 of each pair: whether the suspected copier, the first person,
 * shares more of the source's incorrect responses than persons with as many incorrect responses as the copier do.
 *
 * @remarks
 * The persons are grouped by their number of incorrect responses. For the source s with w_s incorrect responses, the
 * matching rate of a group is the mean number of the source's incorrect responses its members share, over w_s; the
 * source is in no group, and the copier is in its own. K is the upper tail of the binomial distribution of w_s trials
 * at the rate of the copier's group (Holland), K1 and K2 at the rate that a linear and a quadratic regression of the
 * groups' rates on their proportion of incorrect responses give, a rate of 1 or more taken as 0.999 and of 0 or less as
 * 0.001, as CopyDetect does (Sotaridona and Meijer). S1 is the Poisson probability of the identical incorrect responses
 * up to w_s, at the mean that a log-linear model of the groups' means on their number of incorrect responses gives; S2
 * adds to the count each identical correct response, weighted by how rarely the copier's group answers its item
 * correctly, ((1 + g) e / (1 - g))^(-(1 + g) p / g), rounded up, and counts up to the number of items. The indices need
 * no item parameters: their rates come from the other persons of the matrix, so the matrix holds every person who took
 * the test.
 *
 * A response other than the key is incorrect, and two incorrect responses are identical when they are the same
 * category: with scored responses every incorrect response is the same. A missing response is neither correct nor
 * incorrect and matches nothing; to count omissions as identical incorrect responses, give them a category. The counts
 * of each category that occurs in each item, by group, take at most 2²⁴ cells.
 *
 * @param matrix - The responses: each person's category of each item, such as the option chosen, numbered from 0.
 * @param pairs - The pairs: the suspected copier first and the source second.
 * @param options - The keyed category of each item and the probability of a correct guess.
 * @returns For each pair, the indices.
 * @throws `DataError` when the matrix, the pairs or the options are invalid, the key is absent while a response is
 * above 1, a pair names a person twice or a person the matrix lacks, or the counts would take more than 2²⁴ cells.
 * @see {@link https://doi.org/10.1002/j.2333-8504.1996.tb01685.x | Holland (1996)}
 * @see {@link https://doi.org/10.1111/j.1745-3984.2002.tb01138.x | Sotaridona and Meijer (2002)}
 * @see {@link https://doi.org/10.1111/j.1745-3984.2003.tb01096.x | Sotaridona and Meijer (2003)}
 *
 * @example
 * ```ts
 * import { buildResponseMatrix } from "@measurekit/psychometrics";
 * import { kIndex } from "@measurekit/psychometrics/similarity";
 *
 * // Option numbers of 8 persons on 6 items, whose keyed options are 0; the first person shares the second's 4 errors.
 * const matrix = buildResponseMatrix([
 *   [1, 2, 0, 3, 1, 0],
 *   [1, 2, 0, 3, 1, 0],
 *   [2, 0, 1, 0, 3, 2],
 *   [0, 3, 2, 1, 0, 0],
 *   [3, 1, 0, 2, 2, 0],
 *   [0, 0, 3, 0, 1, 1],
 *   [2, 0, 1, 3, 0, 0],
 *   [0, 1, 0, 0, 2, 3],
 * ]);
 * const [pair] = kIndex(matrix, [["person1", "person2"]], { key: [0, 0, 0, 0, 0, 0], guessing: 0.25 });
 * pair?.identicalIncorrect; // => 4
 * pair?.k; // => 0.0123, the probability of 4 of 4 matches at the copier's group's rate
 * ```
 */
export function kIndex(
  matrix: ResponseMatrix,
  pairs: readonly PersonPair[],
  options: KIndexOptions = {},
): readonly KIndex[] {
  checkResponseMatrix(matrix);
  const read = readPairs(matrix.personIds, pairs);
  checkOptions(options, K_INDEX_OPTIONS);
  const { guessing = DEFAULT_GUESSING } = options;
  checkOpenUnitInterval(guessing, "The option guessing");
  const { responses, itemIds } = matrix;
  const scored = responses.every((row) => row.every((response) => response === null || response <= 1));
  const key = readKey(
    options.key,
    itemIds,
    () => null,
    scored ? null : "The K index needs the option key when a response is above 1.",
  );
  const groups = numberIncorrectGroups(matrix, key, guessing);
  const rates = sourceRates(groups);
  const results = new Array<KIndex>(pairs.length);
  for (const k of pairsBySource(read, responses.length)) {
    const copier = read[2 * k] ?? 0;
    const source = read[2 * k + 1] ?? 0;
    loadSource(groups, rates, source);
    results[k] = pairIndices(groups, rates, copier, source);
  }
  return results;
}

/**
 * The persons grouped by their number of incorrect responses, with the count of each category of each item in each
 * group and the weights of S2 at each group's proportion correct of each item.
 */
interface Groups {
  readonly responses: readonly (readonly (number | null)[])[];
  readonly personIds: readonly string[];
  readonly key: Int32Array;
  readonly items: number;
  /** Each person's cell of each response in a group's counts, by person, then item; -1 for a missing response. */
  readonly cells: Int32Array;
  /** The cell of each item's keyed category in a group's counts. */
  readonly keyCells: Int32Array;
  /** The cells of a group's counts: the categories that occur, or are keyed, of all items. */
  readonly width: number;
  /** Each person's number of incorrect responses. */
  readonly incorrect: Int32Array;
  /** The group of each number of incorrect responses, from 0 to the items, or -1 when no person has it. */
  readonly slot: Int32Array;
  /** Each group's number of incorrect responses, ascending. */
  readonly numbers: Float64Array;
  /** Each group's persons. */
  readonly sizes: Int32Array;
  /** The persons of each group with each category of each item, by group, then cell. */
  readonly counts: Int32Array;
  /** S2's weight of each item at each group's proportion correct, by group, then item. */
  readonly weights: Float64Array;
  /** The exponent of S2's weight per unit of proportion correct: -(1 + g) / g × (ln((1 + g) / (1 - g)) + 1). */
  readonly decay: number;
}

/**
 * Groups the persons by their number of incorrect responses and counts the categories of each item in each group.
 *
 * @throws `DataError` when the counts would take more than {@link MAX_COUNT_CELLS} cells.
 */
function numberIncorrectGroups(matrix: ResponseMatrix, key: Int32Array, guessing: number): Groups {
  const { responses, personIds, itemIds } = matrix;
  const items = itemIds.length;
  const { cells, keyCells, width } = categoryCells(responses, key);
  const incorrect = incorrectResponses(cells, keyCells, responses.length);
  const byNumber = new Int32Array(items + 1);
  for (const count of incorrect) byNumber[count] = (byNumber[count] ?? 0) + 1;
  const slot = new Int32Array(items + 1).fill(-1);
  const numbers: number[] = [];
  byNumber.forEach((size, number) => {
    if (size > 0) slot[number] = numbers.push(number) - 1;
  });
  const groupCount = numbers.length;
  if (groupCount * width > MAX_COUNT_CELLS) {
    throw new DataError(
      `The K index counts each category of each item in each group of persons with as many incorrect responses, at most ${MAX_COUNT_CELLS} cells, found ${groupCount} groups of ${width} categories.`,
    );
  }
  const counts = new Int32Array(groupCount * width);
  incorrect.forEach((count, person) => {
    const base = (slot[count] ?? 0) * width;
    for (let column = 0; column < items; column++) {
      const cell = cells[person * items + column] ?? -1;
      if (cell >= 0) counts[base + cell] = (counts[base + cell] ?? 0) + 1;
    }
  });
  const sizes = Int32Array.from(numbers, (number) => byNumber[number] ?? 0);
  const decay = (-(1 + guessing) / guessing) * (Math.log((1 + guessing) / (1 - guessing)) + 1);
  const weights = new Float64Array(groupCount * items);
  weights.forEach((_, index) => {
    const group = Math.floor(index / items);
    const correct = counts[group * width + (keyCells[index - group * items] ?? 0)] ?? 0;
    weights[index] = Math.exp((decay * correct) / (sizes[group] ?? 1));
  });
  return {
    responses,
    personIds,
    key,
    items,
    cells,
    keyCells,
    width,
    incorrect,
    slot,
    numbers: Float64Array.from(numbers),
    sizes,
    counts,
    weights,
    decay,
  };
}

/** Returns each person's number of incorrect responses: answered, in a cell other than the keyed one. */
function incorrectResponses(cells: Int32Array, keyCells: Int32Array, persons: number): Int32Array {
  const items = keyCells.length;
  const incorrect = new Int32Array(persons);
  for (let person = 0; person < persons; person++) {
    let count = 0;
    for (let column = 0; column < items; column++) {
      const cell = cells[person * items + column] ?? -1;
      if (cell >= 0 && cell !== keyCells[column]) count++;
    }
    incorrect[person] = count;
  }
  return incorrect;
}

/**
 * Numbers the categories that occur in each item, and its keyed one, so that the counts hold a cell for each and none
 * for a category no person gives: a single response of 2³¹ - 1 costs one cell.
 *
 * @returns Each response's cell, by person, then item, -1 for a missing one; each item's keyed cell; and the cells.
 */
function categoryCells(
  responses: readonly (readonly (number | null)[])[],
  key: Int32Array,
): { readonly cells: Int32Array; readonly keyCells: Int32Array; readonly width: number } {
  const items = key.length;
  const cells = new Int32Array(responses.length * items).fill(-1);
  const keyCells = new Int32Array(items);
  let width = 0;
  for (let column = 0; column < items; column++) {
    const numbered = new Map<number, number>([[key[column] ?? 0, width]]);
    keyCells[column] = width;
    for (let person = 0; person < responses.length; person++) {
      const response = responses[person]?.[column] ?? null;
      if (response === null) continue;
      let cell = numbered.get(response);
      if (cell === undefined) {
        cell = width + numbered.size;
        numbered.set(response, cell);
      }
      cells[person * items + column] = cell;
    }
    width += numbered.size;
  }
  return { cells, keyCells, width };
}

/**
 * What the indices of one source take from the groups, which the source is in none of: each group's matching rate and
 * the rates and means the regressions fit to them.
 */
interface SourceRates {
  /** The source whose rates these are, or -1 before the first. */
  source: number;
  /** The source's incorrect responses. */
  sourceIncorrect: number;
  /** Each group's matching rate, K's; NaN for a group of the source alone. */
  readonly rates: Float64Array;
  /** The rate the linear and the quadratic regression give each group. */
  readonly linear: Float64Array;
  readonly quadratic: Float64Array;
  /** The mean identical incorrect responses, and with the weighted identical correct ones, the log-linear models give. */
  readonly poisson: Float64Array;
  readonly weightedPoisson: Float64Array;
  /** S2's weight of each item in the source's own group, without the source. */
  readonly ownWeights: Float64Array;
}

function sourceRates({ numbers, items }: Groups): SourceRates {
  const room = (): Float64Array => new Float64Array(numbers.length);
  return {
    source: -1,
    sourceIncorrect: 0,
    rates: room(),
    linear: room(),
    quadratic: room(),
    poisson: room(),
    weightedPoisson: room(),
    ownWeights: new Float64Array(items),
  };
}

/** Writes the rates of `source` into `rates`, unless they are already its. */
function loadSource(groups: Groups, rates: SourceRates, source: number): void {
  if (rates.source === source) return;
  rates.source = source;
  const sourceIncorrect = groups.incorrect[source] ?? 0;
  rates.sourceIncorrect = sourceIncorrect;
  if (sourceIncorrect === 0) return;
  const present: number[] = [];
  const weightedMeans: number[] = [];
  for (let group = 0; group < groups.numbers.length; group++) {
    const shared = sharedResponses(groups, rates, source, group);
    if (shared === null) {
      rates.rates[group] = Number.NaN;
      continue;
    }
    rates.rates[group] = shared.matches / (shared.persons * sourceIncorrect);
    present.push(group);
    weightedMeans.push(shared.weighted);
  }
  const points = Float64Array.from(present, (group) => groups.numbers[group] ?? 0);
  const observed = Float64Array.from(present, (group) => rates.rates[group] ?? 0);
  writeFitted(rates.linear, present, projection(points, observed, 1));
  writeFitted(rates.quadratic, present, projection(points, observed, 2));
  const means = observed.map((rate) => rate * sourceIncorrect);
  writeFitted(rates.poisson, present, logLinearMeans(points, means));
  const weightedCounts = means.map((mean, index) => mean + Math.ceil(weightedMeans[index] ?? 0));
  writeFitted(rates.weightedPoisson, present, logLinearMeans(points, weightedCounts));
}

/** What a group, without the source, shares with the source. */
interface SharedResponses {
  /** The group's persons other than the source. */
  readonly persons: number;
  /** The source's incorrect responses they give, summed over them. */
  readonly matches: number;
  /** The mean of their weighted correct responses on the items the source answered correctly. */
  readonly weighted: number;
}

/**
 * Returns what a group shares with the source, or `null` when the source is its only person. For the source's own
 * group, the weights of S2 at the proportions correct without the source go into `rates.ownWeights`.
 */
function sharedResponses(groups: Groups, rates: SourceRates, source: number, group: number): SharedResponses | null {
  const sourceIncorrect = groups.incorrect[source] ?? 0;
  const own = groups.slot[sourceIncorrect] === group;
  const persons = (groups.sizes[group] ?? 0) - (own ? 1 : 0);
  if (persons === 0) return null;
  const { matches, weighted } = sharedSums(groups, rates, source, group, own ? persons : 0);
  // The source shares each of its incorrect responses with itself.
  return { persons, matches: own ? matches - sourceIncorrect : matches, weighted };
}

/**
 * Returns the source's incorrect responses that a group gives, summed over its persons, and the sum over the source's
 * correct responses of the weight of S2 times the proportion correct. `ownPersons` is the persons of the source's own
 * group without it, whose weights go into `rates.ownWeights`, and 0 for another group.
 */
function sharedSums(
  { items, cells, keyCells, width, sizes, counts, weights, decay }: Groups,
  rates: SourceRates,
  source: number,
  group: number,
  ownPersons: number,
): { readonly matches: number; readonly weighted: number } {
  const base = group * width;
  let matches = 0;
  let weighted = 0;
  for (let column = 0; column < items; column++) {
    const cell = cells[source * items + column] ?? -1;
    if (cell < 0) continue;
    const count = counts[base + cell] ?? 0;
    if (cell !== keyCells[column]) matches += count;
    else if (ownPersons > 0) weighted += ownWeight(rates, column, (count - 1) / ownPersons, decay);
    else weighted += ((weights[group * items + column] ?? 0) * count) / (sizes[group] ?? 1);
  }
  return { matches, weighted };
}

/**
 * Writes the weight of S2 of an item in the source's group, without the source, into `rates.ownWeights`, and returns it
 * times the proportion correct.
 */
function ownWeight(rates: SourceRates, column: number, proportion: number, decay: number): number {
  const weight = Math.exp(decay * proportion);
  rates.ownWeights[column] = weight;
  return weight * proportion;
}

function writeFitted(out: Float64Array, groups: readonly number[], fitted: ArrayLike<number>): void {
  groups.forEach((group, index) => {
    out[group] = fitted[index] ?? 0;
  });
}

/**
 * Returns the least-squares fit of a polynomial of `degree` in the points to the values, as `lm`'s fitted values: their
 * projection on the polynomials, of a lower degree when there are too few points.
 */
function projection(points: Float64Array, values: Float64Array, degree: number): Float64Array {
  const terms = Math.min(degree, points.length - 1) + 1;
  const basis = orthonormalBasis(points, terms - 1);
  const fitted = new Float64Array(points.length);
  for (let term = 0; term < terms; term++) {
    let coefficient = 0;
    for (let point = 0; point < points.length; point++) {
      coefficient += (basis[point * terms + term] ?? 0) * (values[point] ?? 0);
    }
    for (let point = 0; point < points.length; point++) {
      fitted[point] = (fitted[point] ?? 0) + coefficient * (basis[point * terms + term] ?? 0);
    }
  }
  return fitted;
}

/**
 * Returns the means of the Poisson log-linear model ln μ = β₀ + β₁ x fitted to the counts by maximum likelihood, as
 * `glm(family = poisson)`: the counts themselves at one point, and 0 where every count is 0.
 */
function logLinearMeans(points: Float64Array, counts: Float64Array): ArrayLike<number> {
  let total = 0;
  for (const count of counts) total += count;
  if (points.length === 1 || total === 0) return counts;
  return poissonFit(counts, orthonormalBasis(points, 1), 2, total).distribution;
}

/** Returns the pair indices ordered by their source, so that each source's rates are found once. */
function pairsBySource(read: Int32Array, persons: number): Int32Array {
  const count = read.length / 2;
  const starts = new Int32Array(persons + 1);
  for (let k = 0; k < count; k++) {
    const next = (read[2 * k + 1] ?? 0) + 1;
    starts[next] = (starts[next] ?? 0) + 1;
  }
  for (let person = 0; person < persons; person++) {
    starts[person + 1] = (starts[person + 1] ?? 0) + (starts[person] ?? 0);
  }
  const order = new Int32Array(count);
  for (let k = 0; k < count; k++) {
    const source = read[2 * k + 1] ?? 0;
    order[starts[source] ?? 0] = k;
    starts[source] = (starts[source] ?? 0) + 1;
  }
  return order;
}

/** Bounds a regression's rate below 1 and above 0, as CopyDetect does. */
function boundedRate(rate: number): number {
  if (rate >= 1) return HIGHEST_RATE;
  return rate <= 0 ? LOWEST_RATE : rate;
}

/** Returns the indices of one pair from the rates of its source. */
function pairIndices(groups: Groups, rates: SourceRates, copier: number, source: number): KIndex {
  const { personIds, items, incorrect, slot } = groups;
  const { sourceIncorrect } = rates;
  const copierGroup = slot[incorrect[copier] ?? 0] ?? 0;
  const { identicalIncorrect, weightedCorrect } = pairMatches(groups, rates, copier, source, copierGroup);
  const persons: readonly [string, string] = [personIds[copier] ?? "", personIds[source] ?? ""];
  if (sourceIncorrect === 0) {
    return { persons, sourceIncorrect, identicalIncorrect, k: null, k1: null, k2: null, s1: null, s2: null };
  }
  const binomial = (rate: number): number => binomialRange(identicalIncorrect, sourceIncorrect, sourceIncorrect, rate);
  const weightedMatches = identicalIncorrect + Math.ceil(weightedCorrect);
  return {
    persons,
    sourceIncorrect,
    identicalIncorrect,
    k: binomial(rates.rates[copierGroup] ?? 0),
    k1: binomial(boundedRate(rates.linear[copierGroup] ?? 0)),
    k2: binomial(boundedRate(rates.quadratic[copierGroup] ?? 0)),
    s1: poissonRange(identicalIncorrect, sourceIncorrect, Math.min(rates.poisson[copierGroup] ?? 0, sourceIncorrect)),
    s2: poissonRange(weightedMatches, items, Math.min(rates.weightedPoisson[copierGroup] ?? 0, items)),
  };
}

/**
 * Returns the identical incorrect responses of a pair and the sum of the weights of S2, in the copier's group, of its
 * identical correct ones.
 */
function pairMatches(
  { responses, key, items, incorrect, slot, weights }: Groups,
  rates: SourceRates,
  copier: number,
  source: number,
  copierGroup: number,
): { readonly identicalIncorrect: number; readonly weightedCorrect: number } {
  const copied = responses[copier] ?? [];
  const given = responses[source] ?? [];
  // A copier in the source's group has the weights of the group without the source.
  const groupWeights =
    copierGroup === slot[incorrect[source] ?? 0] ? rates.ownWeights : weights.subarray(copierGroup * items);
  let identicalIncorrect = 0;
  let weightedCorrect = 0;
  for (let column = 0; column < items; column++) {
    const response = given[column] ?? null;
    if (response === null || response !== copied[column]) continue;
    if (response === key[column]) weightedCorrect += groupWeights[column] ?? 0;
    else identicalIncorrect++;
  }
  return { identicalIncorrect, weightedCorrect };
}
