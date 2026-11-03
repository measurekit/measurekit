import { checkIntegerFrom, checkOptions, checkPositive, DataError, normalQuantile } from "@measurekit/core";
import type { ResponseMatrix } from "../data/internal.js";
import { type CategoryTally, type CompressedResponses, categoryTallies, compressResponses } from "../irt/internal.js";

/** The evaluation points of KernSmoothIRT's `ksIRT`. */
const DEFAULT_POINTS = 51;
/** Silverman's rule of thumb for scores of standard deviation 1, h = 1.06 n^(-1/5), as KernSmoothIRT's `"Silverman"`. */
const SILVERMAN_FACTOR = 1.06;
/** The smallest normal double: a sum of weights below it has lost precision to underflow. */
const MIN_NORMAL = 2 ** -1022;

/** Options of {@link kernelItemCurves}. */
export interface KernelCurveOptions {
  /**
   * The number of evaluation points, at least 2.
   *
   * @defaultValue `51`
   */
  readonly points?: number;
  /**
   * The bandwidth of the Gaussian kernel, on the scale of the normal scores.
   *
   * @defaultValue 1.06 n^(-1/5) for the n persons with a response, Silverman's rule as KernSmoothIRT's
   */
  readonly bandwidth?: number;
}

/** An item's curves, smoothed from its responses: the share of each category, and its standard error, at each node. */
export interface KernelItemCurve {
  /** The item id. */
  readonly id: string;
  /** The evaluation points on the scale of the normal scores, the same for every item. */
  readonly nodes: readonly number[];
  /** The probability of each category at each node, by category from 0 to the item's highest response, then node. */
  readonly probabilities: readonly (readonly number[])[];
  /** Ramsay's pointwise standard error of each probability, by category, then node. */
  readonly standardErrors: readonly (readonly number[])[];
}

/** One item's answers: the index of each person who answered it, and the response. */
interface ItemAnswers {
  readonly persons: Int32Array;
  readonly responses: Int32Array;
  readonly categories: number;
}

/** An item's curves as they are filled, by category, then node. */
interface CurveRows {
  readonly probabilities: Float64Array[];
  readonly standardErrors: Float64Array[];
}

/**
 * Returns kernel-smoothed curves of the categories of each item, as KernSmoothIRT's `ksIRT`.
 *
 * @remarks
 * Ramsay's nonparametric item curves. The persons with a response are ranked by their total score, ties in the order
 * of the persons, and rank r of n becomes the normal score Φ⁻¹(r / (n + 1)). At each node θ the share of category k is
 * the Nadaraya-Watson average P_k(θ) = Σ_j w_j [x_j = k] / Σ_j w_j over the persons j who answered the item, with
 * Gaussian weights w_j = exp(-((θ - θ_j) / h)² / 2), and its standard error √(Σ_j w_j² P_k(1 - P_k)) / Σ_j w_j. The
 * nodes are equally spaced from Φ⁻¹(1 / (n + 1)) to Φ⁻¹(n / (n + 1)). A missing response adds nothing to a total and
 * leaves the person out of that item's averages. An item's categories run from 0 to its highest response.
 *
 * @param matrix - The responses of the persons.
 * @param options - The number of nodes and the bandwidth.
 * @returns For each item, its nodes and its curves with their standard errors.
 * @throws `DataError` when the matrix or the options are invalid, fewer than two persons have a response, or an item
 * has no response or none in a category below its highest.
 * @see {@link https://doi.org/10.1007/BF02294494 | Ramsay (1991)}
 * @see {@link https://doi.org/10.18637/jss.v058.i06 | Mazza, Punzo and McGuire (2014)}
 */
export function kernelItemCurves(matrix: ResponseMatrix, options: KernelCurveOptions = {}): KernelItemCurve[] {
  checkOptions(options);
  const { points = DEFAULT_POINTS, bandwidth } = options;
  checkIntegerFrom(points, "The option points", 2);
  if (bandwidth !== undefined) checkPositive(bandwidth, "The option bandwidth");
  const data = compressResponses(matrix);
  const tallies = categoryTallies(data);
  tallies.forEach((tally, item) => {
    checkTally(data.itemIds[item] ?? "", tally);
  });
  const scores = normalScores(data);
  const persons = scores.filter((score) => !Number.isNaN(score)).length;
  if (persons < 2) {
    throw new DataError(`Kernel item curves need at least two persons with a response, found ${persons}.`);
  }
  const h = bandwidth ?? SILVERMAN_FACTOR * persons ** -0.2;
  const nodes = evaluationPoints(persons, points);
  const answers = itemAnswers(data, tallies);
  const curves = answers.map(
    ({ categories }): CurveRows => ({
      probabilities: Array.from({ length: categories }, () => new Float64Array(points)),
      standardErrors: Array.from({ length: categories }, () => new Float64Array(points)),
    }),
  );
  const weights = new Float64Array(scores.length);
  const own = new Float64Array(Math.max(...answers.map(({ persons }) => persons.length)));
  const shares = new Float64Array(Math.max(...answers.map(({ categories }) => categories)));
  nodes.forEach((node, q) => {
    relativeWeights(node, scores, h, weights);
    answers.forEach((item, i) => {
      const curve = curves[i];
      if (curve !== undefined) smooth(item, { node, h, scores, weights, own, shares }, curve, q);
    });
  });
  const shared = Array.from(nodes);
  return curves.map(({ probabilities, standardErrors }, i) => ({
    id: data.itemIds[i] ?? "",
    nodes: shared,
    probabilities: probabilities.map((row) => Array.from(row)),
    standardErrors: standardErrors.map((row) => Array.from(row)),
  }));
}

/**
 * Checks that an item has a response in every category up to its highest, so that its curves are as many as the
 * categories used.
 *
 * @throws `DataError` when it has no response or none in a category below its highest.
 */
function checkTally(id: string, { highest, counts }: CategoryTally): void {
  if (highest < 0) throw new DataError(`Item "${id}" has no responses; kernel item curves need at least one.`);
  if (counts.size <= highest) {
    let empty = 0;
    while (counts.has(empty)) empty++;
    throw new DataError(
      `Item "${id}" has no response in category ${empty}; kernel item curves need responses in every category up to the highest.`,
    );
  }
}

/**
 * Returns each person's normal score from the rank of the total score among the persons with a response, ties in the
 * order of the persons; `NaN` for a person without responses.
 */
function normalScores({ offsets, codes }: CompressedResponses): Float64Array {
  const count = offsets.length - 1;
  const totals = new Float64Array(count);
  const ranked: number[] = [];
  for (let person = 0; person < count; person++) {
    const start = offsets[person] ?? 0;
    const end = offsets[person + 1] ?? 0;
    if (end === start) continue;
    let total = 0;
    for (let r = start; r < end; r++) total += codes[r] ?? 0;
    totals[person] = total;
    ranked.push(person);
  }
  // Array.prototype.sort is stable, so persons of equal totals keep their order, as R's ties.method = "first".
  ranked.sort((a, b) => (totals[a] ?? 0) - (totals[b] ?? 0));
  const scores = new Float64Array(count).fill(Number.NaN);
  ranked.forEach((person, rank) => {
    scores[person] = normalQuantile((rank + 1) / (ranked.length + 1));
  });
  return scores;
}

/** Returns `points` nodes equally spaced from Φ⁻¹(1 / (n + 1)) to Φ⁻¹(n / (n + 1)), the last exactly, as R's `seq`. */
function evaluationPoints(persons: number, points: number): Float64Array {
  const from = normalQuantile(1 / (persons + 1));
  const to = normalQuantile(persons / (persons + 1));
  const step = (to - from) / (points - 1);
  return Float64Array.from({ length: points }, (_, q) => (q === points - 1 ? to : from + q * step));
}

/** Returns each item's answers: who gave them and what, with the item's number of categories. */
function itemAnswers(
  { offsets, items, codes, itemIds }: CompressedResponses,
  tallies: readonly CategoryTally[],
): ItemAnswers[] {
  const counts = new Int32Array(itemIds.length);
  for (const item of items) counts[item] = (counts[item] ?? 0) + 1;
  const answers = tallies.map(({ highest }, item) => ({
    persons: new Int32Array(counts[item] ?? 0),
    responses: new Int32Array(counts[item] ?? 0),
    categories: highest + 1,
  }));
  const filled = new Int32Array(itemIds.length);
  for (let person = 0; person < offsets.length - 1; person++) {
    for (let r = offsets[person] ?? 0; r < (offsets[person + 1] ?? 0); r++) {
      const item = items[r] ?? 0;
      const answer = answers[item];
      if (answer === undefined) continue;
      const slot = filled[item] ?? 0;
      answer.persons[slot] = person;
      answer.responses[slot] = codes[r] ?? 0;
      filled[item] = slot + 1;
    }
  }
  return answers;
}

/**
 * Writes the Gaussian weight exp(-((node - θ_j) / h)² / 2) of each person j in `persons`, or of every person without
 * it, into `out`, relative to the nearest one's, which is 1, so that a narrow bandwidth does not underflow them all. A
 * person without responses, whose score is `NaN`, gets 0.
 */
function relativeWeights(node: number, scores: Float64Array, h: number, out: Float64Array, persons?: Int32Array): void {
  const count = persons === undefined ? scores.length : persons.length;
  const scoreOf = (j: number): number => scores[persons === undefined ? j : (persons[j] ?? 0)] ?? Number.NaN;
  let nearest = Number.POSITIVE_INFINITY;
  for (let j = 0; j < count; j++) {
    const distance = (node - scoreOf(j)) ** 2;
    if (distance < nearest) nearest = distance;
  }
  for (let j = 0; j < count; j++) {
    const distance = (node - scoreOf(j)) ** 2;
    out[j] = Number.isNaN(distance) ? 0 : Math.exp(-(distance - nearest) / (2 * h * h));
  }
}

/** What smoothing at a node reads: the normal scores, the persons' weights there and room for an item's. */
interface NodeWeights {
  readonly node: number;
  readonly h: number;
  readonly scores: Float64Array;
  /** The weight of each person, relative to the nearest person's. */
  readonly weights: Float64Array;
  /** The weights of an item's persons, relative to the nearest of them, where theirs underflow. */
  readonly own: Float64Array;
  readonly shares: Float64Array;
}

/** The sums of an item's weights at a node: their total and the total of their squares. */
interface WeightSums {
  total: number;
  squares: number;
}

/**
 * Writes an item's shares and their standard errors at node `q`. The persons' weights serve every item; where those of
 * the item's persons underflow, as when they are far from the node and others near it, they are taken again relative to
 * the nearest of the item's persons.
 */
function smooth(
  { persons, responses, categories }: ItemAnswers,
  { node, h, scores, weights, own, shares }: NodeWeights,
  curve: CurveRows,
  q: number,
): void {
  const sums: WeightSums = { total: 0, squares: 0 };
  addShares(persons, responses, (j) => weights[persons[j] ?? 0] ?? 0, shares, sums);
  if (!(sums.total >= MIN_NORMAL)) {
    relativeWeights(node, scores, h, own, persons);
    addShares(persons, responses, (j) => own[j] ?? 0, shares, sums);
  }
  const { total, squares } = sums;
  for (let k = 0; k < categories; k++) {
    const p = (shares[k] ?? 0) / total;
    // 1 - p as the share of the other categories, which keeps its digits where p is within rounding of 1.
    let others = 0;
    for (let j = 0; j < categories; j++) if (j !== k) others += shares[j] ?? 0;
    const probabilities = curve.probabilities[k];
    const errors = curve.standardErrors[k];
    if (probabilities !== undefined) probabilities[q] = p;
    if (errors !== undefined) errors[q] = Math.sqrt(p * (others / total) * squares) / total;
  }
}

/**
 * Sums an item's weights, their squares and the weights of each category, in the order of the item's persons, into
 * `sums` and `shares`, both reset first.
 */
function addShares(
  persons: Int32Array,
  responses: Int32Array,
  weightOf: (j: number) => number,
  shares: Float64Array,
  sums: WeightSums,
): void {
  let total = 0;
  let squares = 0;
  shares.fill(0);
  for (let j = 0; j < persons.length; j++) {
    const w = weightOf(j);
    total += w;
    squares += w * w;
    const k = responses[j] ?? 0;
    shares[k] = (shares[k] ?? 0) + w;
  }
  sums.total = total;
  sums.squares = squares;
}
