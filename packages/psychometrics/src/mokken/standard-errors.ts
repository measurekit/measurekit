import { type MokkenData, scoreOf } from "./scores.js";
import { stepPairErrors } from "./weights.js";

/** The distinct response patterns of the persons and how many persons have each. */
interface Patterns {
  /** The number of distinct patterns. */
  readonly count: number;
  /** The number of items. */
  readonly items: number;
  /** The scores of each pattern by pattern, then item. */
  readonly scores: Uint8Array;
  /** The number of persons with each pattern. */
  readonly persons: Float64Array;
}

/** The standard errors of the scalability coefficients of a set of items; `null` where a coefficient is undefined. */
export interface CoefficientErrors {
  /** The standard error of each pair's coefficient by item, then item; `null` on the diagonal and where undefined. */
  readonly pairs: readonly (number | null)[][];
  /** The standard error of each item's coefficient. */
  readonly items: readonly (number | null)[];
  /** The standard error of the scale's coefficient. */
  readonly scale: number | null;
}

/** A coefficient as a ratio 1 - F / E of the Guttman errors observed to those expected, with its derivatives. */
class ErrorRatio {
  observed = 0;
  expected = 0;
  /** The derivatives of the observed and the expected errors with respect to the number of persons of each pattern. */
  readonly observedDerivative: Float64Array;
  readonly expectedDerivative: Float64Array;

  constructor(patterns: number) {
    this.observedDerivative = new Float64Array(patterns);
    this.expectedDerivative = new Float64Array(patterns);
  }

  /** Returns the standard error of 1 - F / E, or `null` when no error is expected. */
  standardError(patterns: Patterns): number | null {
    if (!(this.expected > 0)) return null;
    const ratio = this.observed / this.expected;
    let variance = 0;
    for (let k = 0; k < patterns.count; k++) {
      const derivative =
        -((this.observedDerivative[k] ?? 0) - ratio * (this.expectedDerivative[k] ?? 0)) / this.expected;
      variance += (patterns.persons[k] ?? 0) * derivative * derivative;
    }
    return Math.sqrt(variance);
  }
}

/** Returns the distinct patterns of the persons' scores. */
function patternsOf(data: MokkenData): Patterns {
  const found = new Map<string, number>();
  const firstPerson: number[] = [];
  const counts: number[] = [];
  for (let person = 0; person < data.persons; person++) {
    const key = String.fromCharCode(...data.scores.subarray(person * data.items, (person + 1) * data.items));
    const at = found.get(key);
    if (at === undefined) {
      found.set(key, firstPerson.length);
      firstPerson.push(person);
      counts.push(1);
    } else {
      counts[at] = (counts[at] ?? 0) + 1;
    }
  }
  const scores = new Uint8Array(firstPerson.length * data.items);
  firstPerson.forEach((person, k) => {
    for (let item = 0; item < data.items; item++) scores[k * data.items + item] = scoreOf(data, person, item);
  });
  return { count: firstPerson.length, items: data.items, scores, persons: Float64Array.from(counts) };
}

/** Returns the number of persons at each score of an item. */
function scoreCounts(data: MokkenData, item: number): Float64Array {
  const counts = new Float64Array(data.categories);
  for (let person = 0; person < data.persons; person++) {
    const score = scoreOf(data, person, item);
    counts[score] = (counts[score] ?? 0) + 1;
  }
  return counts;
}

/**
 * Returns the standard errors of the scalability coefficients of the items, by the delta method over the multinomial
 * distribution of the response patterns.
 *
 * @remarks
 * A coefficient is 1 - F / E: the Guttman errors observed over those expected if the items were independent, summed
 * over a pair, an item's pairs or all pairs. With the weights that count errors kept fixed, F and E are linear and
 * quadratic-over-N functions of the numbers of persons with each pattern, so a coefficient H has a derivative
 * dH = -(dF - (F / E) dE) / E with respect to each, and its variance is the sum over patterns of their persons times
 * dH squared (the term for the persons' total vanishes, as H does not change when every count is multiplied). It is the
 * computation of the mokken package's `coefH`. Items without variance are left out: their pairs expect no errors.
 *
 * @param usable - Whether each item has variance.
 */
export function coefficientErrors(data: MokkenData, usable: readonly boolean[]): CoefficientErrors {
  const { items, persons: total, categories } = data;
  const patterns = patternsOf(data);
  const counts = Array.from({ length: items }, (_, item) => scoreCounts(data, item));
  const pairRatios = new Map<number, ErrorRatio>();
  const itemRatios = Array.from({ length: items }, () => new ErrorRatio(patterns.count));
  const scaleRatio = new ErrorRatio(patterns.count);
  for (let i = 0; i < items - 1; i++) {
    for (let j = i + 1; j < items; j++) {
      if (!(usable[i] && usable[j])) continue;
      const pair = pairContribution(patterns, counts, categories, total, i, j);
      pairRatios.set(i * items + j, pair);
      for (const ratio of [itemRatios[i], itemRatios[j], scaleRatio]) addPair(ratio, pair);
    }
  }
  const pairs = Array.from({ length: items }, (_, i) =>
    Array.from({ length: items }, (_, j) => {
      if (i === j) return null;
      return pairRatios.get(Math.min(i, j) * items + Math.max(i, j))?.standardError(patterns) ?? null;
    }),
  );
  return {
    pairs,
    items: itemRatios.map((ratio, item) => (usable[item] ? ratio.standardError(patterns) : null)),
    scale: scaleRatio.standardError(patterns),
  };
}

/** Adds the errors of a pair of items, with their derivatives, to a ratio. */
function addPair(ratio: ErrorRatio | undefined, pair: ErrorRatio): void {
  if (ratio === undefined) return;
  ratio.observed += pair.observed;
  ratio.expected += pair.expected;
  for (let k = 0; k < pair.observedDerivative.length; k++) {
    ratio.observedDerivative[k] = (ratio.observedDerivative[k] ?? 0) + (pair.observedDerivative[k] ?? 0);
    ratio.expectedDerivative[k] = (ratio.expectedDerivative[k] ?? 0) + (pair.expectedDerivative[k] ?? 0);
  }
}

/** The errors expected of two independent items, and how much each score of each item adds to them. */
interface ExpectedErrors {
  readonly expected: number;
  readonly forFirst: Float64Array;
  readonly forSecond: Float64Array;
}

/** Returns the Guttman errors expected of a pair of items whose scores are independent, from the numbers of persons. */
function expectedErrors(
  errors: Float64Array,
  countsFirst: Float64Array,
  countsSecond: Float64Array,
  total: number,
): ExpectedErrors {
  const categories = countsFirst.length;
  const forFirst = new Float64Array(categories);
  const forSecond = new Float64Array(categories);
  let expected = 0;
  for (let a = 0; a < categories; a++) {
    for (let b = 0; b < categories; b++) {
      const error = errors[a * categories + b] ?? 0;
      forFirst[a] = (forFirst[a] ?? 0) + error * (countsSecond[b] ?? 0);
      forSecond[b] = (forSecond[b] ?? 0) + error * (countsFirst[a] ?? 0);
      expected += (error * (countsFirst[a] ?? 0) * (countsSecond[b] ?? 0)) / total;
    }
  }
  return { expected, forFirst, forSecond };
}

/** Returns the observed and expected Guttman errors of a pair of items and their derivatives by pattern. */
function pairContribution(
  patterns: Patterns,
  counts: readonly Float64Array[],
  categories: number,
  total: number,
  i: number,
  j: number,
): ErrorRatio {
  const countsI = counts[i] ?? new Float64Array(categories);
  const countsJ = counts[j] ?? new Float64Array(categories);
  const errors = stepPairErrors(countsI, countsJ);
  const { expected, forFirst, forSecond } = expectedErrors(errors, countsI, countsJ, total);
  const pair = new ErrorRatio(patterns.count);
  pair.expected = expected;
  for (let k = 0; k < patterns.count; k++) {
    const a = patterns.scores[k * patterns.items + i] ?? 0;
    const b = patterns.scores[k * patterns.items + j] ?? 0;
    const error = errors[a * categories + b] ?? 0;
    pair.observed += (patterns.persons[k] ?? 0) * error;
    pair.observedDerivative[k] = error;
    pair.expectedDerivative[k] = ((forFirst[a] ?? 0) + (forSecond[b] ?? 0) - expected) / total;
  }
  return pair;
}
