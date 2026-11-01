import type { ExpectationItem } from "../irt/expectation.js";
import { logProbability } from "../irt/probability.js";
import type { CompressedResponses } from "../irt/responses.js";
import type { Grid } from "./grid.js";

/** The smallest total likelihood of a person that the E-step divides by: below it the person's posterior is skipped. */
const MIN_TOTAL = Number.MIN_VALUE;

/** A person's posterior at the parameters of the last run, as {@link OccasionExpectation.forEachPosterior} shows it. */
interface PersonPosterior {
  /** The probability of each node of the grid. */
  readonly joint: Float64Array;
  /** The marginal probability of each node of an occasion's axis, by occasion, then node. */
  readonly marginals: readonly Float64Array[];
}

/** The E-step of Bock-Aitkin EM for the persons of several occasions, one trait each, over a product grid. */
export interface OccasionExpectation {
  /**
   * Evaluates every person's posterior over the grid, from the prior's weights and the items' current parameters;
   * writes, for each item, the expected number of responses in each category at each node of an axis, pooled over the
   * occasions that the item was answered in, and `nodeCounts`; and returns the marginal log-likelihood: the sum over
   * persons of log Σ_q w_q Π_t Π_i P_i(x_it | θ_qt).
   */
  run(): number;
  /** The expected number of persons at each node of the grid, as of the last run. */
  readonly nodeCounts: Float64Array;
  /**
   * Calls `visit` with the person and the posterior of each person who responded, at the items' current parameters and
   * the prior's weights. The posterior is reused between calls.
   */
  forEachPosterior(visit: (person: number, posterior: PersonPosterior) => void): void;
}

/** The arrays that a person's posterior is computed in, reused from person to person. */
interface Workspace {
  /** The nodes on an axis. */
  readonly points: number;
  /** The number of occasions. */
  readonly dimensions: number;
  /** The prior's weights at the nodes of the grid. */
  readonly prior: Float64Array;
  /** The unnormalized, then normalized, posterior at the nodes of the grid. */
  readonly weights: Float64Array;
  /** The sums of `weights` over each run of nodes along the first axis. */
  readonly slabs: Float64Array;
  /** The likelihood of the person's responses of each occasion at the nodes of its axis, by occasion, then node. */
  readonly columns: readonly Float64Array[];
  /** The posterior marginal at the nodes of each axis. */
  readonly marginals: readonly Float64Array[];
}

/** The log-probability of each category of each item at the nodes of an axis, by item, then node, then category. */
interface LogTable {
  readonly values: Float64Array;
  readonly starts: Int32Array;
  readonly probabilities: Float64Array;
}

/**
 * Returns the E-step of persons answering the same items on several occasions.
 *
 * @remarks
 * Occasion t has axis t of the grid, the first axis varying fastest. A person's likelihood at a node is the product
 * over occasions of that occasion's likelihood at the node's coordinate, so it is tabulated once per occasion on the
 * axis, not on the grid: the work of an item is its responses times the points of an axis, and the work on the grid
 * is a few multiplications per node. An occasion without responses has likelihood 1.
 *
 * @param items - The items with nodes on an axis. An item's counts are filled over the occasions it was answered in.
 * @param logWeights - The log weights of the nodes of the grid, normalized to sum to 1, read at each run.
 */
export function occasionExpectation(
  occasions: readonly CompressedResponses[],
  items: readonly ExpectationItem[],
  grid: Grid,
  logWeights: Float64Array,
): OccasionExpectation {
  const { points, count } = grid;
  const dimensions = occasions.length;
  const persons = (occasions[0]?.offsets.length ?? 1) - 1;
  const nodeCounts = new Float64Array(count);
  const work: Workspace = {
    points,
    dimensions,
    prior: new Float64Array(count),
    weights: new Float64Array(count),
    slabs: new Float64Array(count / points),
    columns: Array.from({ length: dimensions }, () => new Float64Array(points)),
    marginals: Array.from({ length: dimensions }, () => new Float64Array(points)),
  };
  const table = logTable(items, points);
  const prepare = (): void => {
    tabulate(items, table);
    for (let q = 0; q < count; q++) work.prior[q] = Math.exp(logWeights[q] ?? 0);
  };
  /** Computes a person's posterior and returns its total, or `null` for a person without responses. */
  const posteriorOf = (person: number): { readonly logLikelihood: number; readonly total: number } | null => {
    const shifts = occasions.map((occasion, t) => likelihood(occasion, person, work.columns[t], items, table));
    if (shifts.every((shift) => shift === null)) return null;
    const total = weigh(work);
    const logShift = shifts.reduce<number>((sum, shift) => sum + (shift ?? 0), 0);
    return { logLikelihood: logShift + Math.log(Math.max(total, MIN_TOTAL)), total };
  };

  return {
    run() {
      prepare();
      for (const item of items) item.counts.fill(0);
      nodeCounts.fill(0);
      let logLikelihood = 0;
      for (let person = 0; person < persons; person++) {
        const found = posteriorOf(person);
        if (found === null) continue;
        logLikelihood += found.logLikelihood;
        // A person whose likelihood underflows at every node adds nothing to the counts rather than dividing by 0.
        if (!(found.total > MIN_TOTAL)) continue;
        normalize(work, 1 / found.total, nodeCounts);
        countResponses(occasions, person, items, work.marginals);
      }
      return logLikelihood;
    },
    nodeCounts,
    forEachPosterior(visit) {
      prepare();
      const posterior: PersonPosterior = { joint: work.weights, marginals: work.marginals };
      const discarded = new Float64Array(count);
      for (let person = 0; person < persons; person++) {
        const found = posteriorOf(person);
        if (found === null || !(found.total > MIN_TOTAL)) continue;
        normalize(work, 1 / found.total, discarded);
        visit(person, posterior);
      }
    },
  };
}

/** Returns the table of the items' log-probabilities, empty until {@link tabulate} fills it. */
function logTable(items: readonly ExpectationItem[], points: number): LogTable {
  const starts = new Int32Array(items.length + 1);
  items.forEach((item, i) => {
    starts[i + 1] = (starts[i] ?? 0) + points * item.categoryCount;
  });
  return {
    values: new Float64Array(starts[items.length] ?? 0),
    starts,
    probabilities: new Float64Array(items.reduce((most, item) => Math.max(most, item.categoryCount), 0)),
  };
}

/** Writes each item's log-probability of every category at every node of an axis into the table. */
function tabulate(items: readonly ExpectationItem[], { values, starts, probabilities }: LogTable): void {
  items.forEach((item, i) => {
    let index = starts[i] ?? 0;
    for (const theta of item.nodes) {
      item.model.probabilities(item.parameters, theta, probabilities);
      for (let k = 0; k < item.categoryCount; k++) values[index++] = logProbability(probabilities[k] ?? 0);
    }
  });
}

/**
 * Writes the likelihood of a person's responses of one occasion at each node of its axis into `out`, divided by its
 * largest so that none underflows, and returns the log of that largest; `null` when the person has no responses.
 */
function likelihood(
  occasion: CompressedResponses | undefined,
  person: number,
  out: Float64Array | undefined,
  items: readonly ExpectationItem[],
  table: LogTable,
): number | null {
  if (occasion === undefined || out === undefined) return null;
  if ((occasion.offsets[person + 1] ?? 0) === (occasion.offsets[person] ?? 0)) {
    out.fill(1);
    return null;
  }
  out.fill(0);
  for (let r = occasion.offsets[person] ?? 0; r < (occasion.offsets[person + 1] ?? 0); r++) {
    addResponse(out, occasion.items[r] ?? 0, occasion.codes[r] ?? 0, items, table);
  }
  const largest = out.reduce((most, value) => Math.max(most, value), Number.NEGATIVE_INFINITY);
  for (let j = 0; j < out.length; j++) out[j] = Math.exp((out[j] ?? 0) - largest);
  return largest;
}

/** Adds the log-probability of a response at each node of an axis to `out`. */
function addResponse(
  out: Float64Array,
  item: number,
  code: number,
  items: readonly ExpectationItem[],
  { values, starts }: LogTable,
): void {
  const stride = items[item]?.categoryCount ?? 1;
  const first = (starts[item] ?? 0) + code;
  for (let j = 0; j < out.length; j++) out[j] = (out[j] ?? 0) + (values[first + j * stride] ?? 0);
}

/** Returns the product of the occasions' likelihoods after the first at the nodes of block `b` of the first axis. */
function restOf({ columns, dimensions, points }: Workspace, block: number): number {
  let rest = 1;
  let remainder = block;
  for (let t = 1; t < dimensions; t++) {
    rest *= columns[t]?.[remainder % points] ?? 1;
    remainder = Math.floor(remainder / points);
  }
  return rest;
}

/**
 * Writes prior × likelihood at every node into `weights` and the sums of its runs along the first axis into `slabs`;
 * returns their total.
 */
function weigh(work: Workspace): number {
  const { prior, weights, slabs, points } = work;
  const first = work.columns[0] ?? new Float64Array(points);
  let total = 0;
  for (let b = 0; b < slabs.length; b++) {
    const rest = restOf(work, b);
    let sum = 0;
    for (let j = 0, q = b * points; j < points; j++, q++) {
      const weight = (prior[q] ?? 0) * rest * (first[j] ?? 0);
      weights[q] = weight;
      sum += weight;
    }
    slabs[b] = sum;
    total += sum;
  }
  return total;
}

/**
 * Scales `weights` to the person's posterior, adds it to `counts` and writes the posterior marginal at the nodes of each
 * axis.
 */
function normalize(work: Workspace, inverse: number, counts: Float64Array): void {
  const { weights, slabs, marginals, points } = work;
  for (const marginal of marginals) marginal.fill(0);
  const first = marginals[0];
  for (let b = 0; b < slabs.length; b++) {
    for (let j = 0, q = b * points; j < points && first !== undefined; j++, q++) {
      const posterior = (weights[q] ?? 0) * inverse;
      weights[q] = posterior;
      counts[q] = (counts[q] ?? 0) + posterior;
      first[j] = (first[j] ?? 0) + posterior;
    }
    addSlab(work, b, (slabs[b] ?? 0) * inverse);
  }
}

/** Adds the posterior mass of a block of nodes along the first axis to the marginals of the other axes. */
function addSlab({ marginals, points }: Workspace, block: number, mass: number): void {
  let remainder = block;
  for (let t = 1; t < marginals.length; t++) {
    const marginal = marginals[t];
    if (marginal !== undefined) marginal[remainder % points] = (marginal[remainder % points] ?? 0) + mass;
    remainder = Math.floor(remainder / points);
  }
}

/** Adds the person's marginals to the counts of the items answered, occasion by occasion. */
function countResponses(
  occasions: readonly CompressedResponses[],
  person: number,
  items: readonly ExpectationItem[],
  marginals: readonly Float64Array[],
): void {
  occasions.forEach((occasion, t) => {
    const marginal = marginals[t];
    for (let r = occasion.offsets[person] ?? 0; r < (occasion.offsets[person + 1] ?? 0); r++) {
      const item = items[occasion.items[r] ?? 0];
      if (item !== undefined && marginal !== undefined) addCounts(item, occasion.codes[r] ?? 0, marginal);
    }
  });
}

/** Adds a person's marginal posterior at the nodes of an axis to an item's counts of the category answered. */
function addCounts(item: ExpectationItem, code: number, marginal: Float64Array): void {
  marginal.forEach((probability, j) => {
    const index = j * item.categoryCount + code;
    item.counts[index] = (item.counts[index] ?? 0) + probability;
  });
}
