import { normalizeLogs } from "@measurekit/core";
import type { Population } from "./latent.js";
import type { IrtModel } from "./model.js";
import { logProbability } from "./probability.js";
import type { CompressedResponses } from "./responses.js";
import type { TestletDesign } from "./testlets.js";

/** An item as the E-step sees it: its model, its current parameters and the expected counts the E-step writes. */
export interface ExpectationItem {
  /** The item's model. */
  readonly model: IrtModel;
  /** The number of response categories. */
  readonly categoryCount: number;
  /** The thetas at which its probabilities enter: the quadrature nodes, or θ + γ for an item in a testlet. */
  readonly nodes: Float64Array;
  /** The current parameters. */
  parameters: unknown;
  /** The expected number of persons in each category at each of its nodes, by node, then category. */
  readonly counts: Float64Array;
}

/** The E-step of Bock-Aitkin EM. */
export interface ExpectationStep {
  /**
   * Evaluates every person's posterior over the quadrature nodes, under the weights of the person's group, or with a
   * regression of the person's covariates, and the items' current parameters; writes the items' expected counts and
   * each group's, or each covariate pattern's, expected number of persons at each node; and returns the marginal
   * log-likelihood: the sum over persons of log Σ_q w_q Π_i P_i(x_i | θ_q).
   */
  run(): number;
  /**
   * Calls `visit` with each person who answered an item and the person's posterior over the nodes, at the parameters
   * of the last run. `posterior` is reused between calls. Not with testlets, whose posteriors span θ and γ.
   */
  forEachPosterior(visit: (person: number, posterior: Float64Array) => void): void;
}

/**
 * Returns the E-step of Bock-Aitkin EM over answered responses.
 *
 * @remarks
 * Persons without responses change nothing. Each item's log-probabilities at its nodes sit in one array, by node, then
 * category, so that a response adds every node's term with a fixed stride. With testlets, each testlet's items enter by
 * the bifactor dimension reduction: a person's likelihood at θ_q multiplies Σ_g w_g Π_j P_j(x_j | θ_q + γ_g) over each
 * testlet, integrated once per run for each frequent pattern of responses to the testlet.
 *
 * `logTerms` adds a log-likelihood term the items do not give to a person's log joint density at each node, such as the
 * density of the person's response times given theta in a joint model of speed and accuracy. It is by person, then
 * node, with the persons of `data`, and it is read at each run, so that a caller whose term depends on its own
 * parameters rewrites it between runs. The log-likelihood a run returns includes it.
 *
 * @see {@link https://doi.org/10.1007/BF02295430 | Gibbons and Hedeker (1992)}
 */
export function expectationStep(
  data: CompressedResponses,
  items: readonly ExpectationItem[],
  nodes: Float64Array,
  { groups, groupOf, regression }: Population,
  design: TestletDesign | null = null,
  logTerms: Float64Array | null = null,
): ExpectationStep {
  const nodeCount = nodes.length;
  const { offsets, codes } = data;
  const responseItems = data.items;
  const personCount = offsets.length - 1;
  const categories = Int32Array.from(items, (item) => item.categoryCount);
  const tableStarts = new Int32Array(items.length + 1);
  items.forEach((item, i) => {
    tableStarts[i + 1] = (tableStarts[i] ?? 0) + item.nodes.length * item.categoryCount;
  });
  const testletOf = design?.testletOf ?? new Int32Array(items.length).fill(-1);
  const logTable = new Float64Array(tableStarts[items.length] ?? 0);
  const itemCounts = items.map((item) => item.counts);
  const posterior = new Float64Array(nodeCount);
  const probabilities = new Float64Array(categories.reduce((most, count) => Math.max(most, count), 0));

  /** Writes each item's log-probabilities at the nodes into the table. */
  const tabulate = (): void => {
    items.forEach((item, i) => {
      const size = item.categoryCount;
      let index = tableStarts[i] ?? 0;
      for (const theta of item.nodes) {
        item.model.probabilities(item.parameters, theta, probabilities);
        for (let k = 0; k < size; k++) logTable[index++] = logProbability(probabilities[k] ?? 0);
      }
    });
  };

  /** Adds the person's entries of `logTerms`, if there are any, to `posterior`. */
  const addTerm = (person: number): void => {
    if (logTerms === null) return;
    const first = person * nodeCount;
    for (let q = 0; q < nodeCount; q++) posterior[q] = (posterior[q] ?? 0) + (logTerms[first + q] ?? 0);
  };

  /** Starts `posterior` from the person's prior: the group's log weights, or the log weights of the covariate pattern. */
  const startFromPrior = (person: number, logWeights: Float64Array): void => {
    if (regression === undefined) {
      posterior.set(logWeights);
      return;
    }
    const first = (regression.patternOf[person] ?? 0) * nodeCount;
    for (let q = 0; q < nodeCount; q++) posterior[q] = regression.logWeights[first + q] ?? 0;
  };

  /** Writes a person's log joint densities over the nodes into `posterior`, from the weights of the person's prior. */
  const joint = (person: number, start: number, end: number, logWeights: Float64Array): void => {
    startFromPrior(person, logWeights);
    addTerm(person);
    for (let r = start; r < end; r++) {
      const item = responseItems[r] ?? 0;
      if ((testletOf[item] ?? -1) >= 0) continue;
      const column = (tableStarts[item] ?? 0) + (codes[r] ?? 0);
      gather(posterior, logTable, column, categories[item] ?? 0);
    }
  };

  /** Adds the posterior to the expected counts of the person's covariate pattern, if there is a regression. */
  const countPattern = (person: number): void => {
    if (regression === undefined) return;
    const first = (regression.patternOf[person] ?? 0) * nodeCount;
    for (let q = 0; q < nodeCount; q++) {
      regression.counts[first + q] = (regression.counts[first + q] ?? 0) + (posterior[q] ?? 0);
    }
  };

  /** Adds the posterior to the counts of the categories a person answered in and to those of the group or pattern. */
  const count = (person: number, start: number, end: number, groupCounts: Float64Array): void => {
    for (let r = start; r < end; r++) {
      const item = responseItems[r] ?? 0;
      const counts = itemCounts[item];
      if ((testletOf[item] ?? -1) >= 0 || counts === undefined) continue;
      scatter(counts, codes[r] ?? 0, categories[item] ?? 0, posterior);
    }
    for (let q = 0; q < nodeCount; q++) groupCounts[q] = (groupCounts[q] ?? 0) + (posterior[q] ?? 0);
    countPattern(person);
  };

  /** Calls `visit` with each person who answered an item, the bounds of the person's responses and the group. */
  const forEachRespondent = (visit: (person: number, start: number, end: number, groupIndex: number) => void): void => {
    for (let person = 0; person < personCount; person++) {
      const start = offsets[person] ?? 0;
      const end = offsets[person + 1] ?? 0;
      if (end > start) visit(person, start, end, groupOf[person] ?? 0);
    }
  };

  const reduction =
    design === null
      ? null
      : testletReduction(design, nodeCount, data, testletOf, categories, tableStarts, logTable, itemCounts);

  return {
    run() {
      tabulate();
      for (const counts of itemCounts) counts.fill(0);
      for (const group of groups) group.counts.fill(0);
      regression?.counts.fill(0);
      for (const testlet of design?.testlets ?? []) testlet.counts.fill(0);
      reduction?.prepare();
      let logLikelihood = 0;
      forEachRespondent((person, start, end, g) => {
        const group = groups[g];
        if (group === undefined) return;
        joint(person, start, end, group.logWeights);
        reduction?.integrate(person, start, end, posterior);
        logLikelihood += normalizeLogs(posterior);
        count(person, start, end, group.counts);
        reduction?.count(person, start, end, posterior);
      });
      reduction?.finish();
      return logLikelihood;
    },
    forEachPosterior(visit) {
      forEachRespondent((person, start, end, g) => {
        const group = groups[g];
        if (group === undefined) return;
        joint(person, start, end, group.logWeights);
        normalizeLogs(posterior);
        visit(person, posterior);
      });
    },
  };
}

/**
 * The testlets' part of the E-step. A testlet's likelihood over θ + γ depends only on the person's responses to its
 * items, so each frequent pattern of responses is integrated once per run, and its persons' expected counts are summed
 * over θ before they are spread over γ, once per pattern. Other persons are integrated and counted one at a time.
 */
interface TestletReduction {
  /** Integrates the frequent patterns at the items' current parameters and clears their sums. */
  prepare(): void;
  /** Adds ln Σ_g w_g Π_j P_j(x_j | θ_q + γ_g) of each testlet the person answered to `logPosterior` at each θ_q. */
  integrate(person: number, start: number, end: number, logPosterior: Float64Array): void;
  /**
   * Adds the person's expected counts to the testlet items' counts and to the testlets', given the posterior over θ.
   */
  count(person: number, start: number, end: number, posterior: Float64Array): void;
  /** Adds the frequent patterns' expected counts to the testlet items' counts and to the testlets'. */
  finish(): void;
}

/** The most patterns of one testlet integrated once per run, which bounds their memory to a few megabytes. */
const MAX_PATTERNS = 4096;

/** A testlet's frequent response patterns: each person's pattern slot, -1 for none, and a person with each pattern. */
interface TestletPatterns {
  readonly slotOf: Int32Array;
  readonly representatives: Int32Array;
}

function testletReduction(
  design: TestletDesign,
  nodeCount: number,
  data: CompressedResponses,
  testletOf: Int32Array,
  categories: Int32Array,
  tableStarts: Int32Array,
  logTable: Float64Array,
  itemCounts: readonly Float64Array[],
): TestletReduction {
  const { testlets } = design;
  const sumCount = design.sums.length;
  const patterns = testlets.map((_, t) => frequentPatterns(data, testletOf, t));
  // Per frequent pattern: Σ_g w_g exp(S_(q+g) - max S) at each θ_q, its log, max S, and the sum over its persons of
  // post(θ_q) / integral_q.
  const integrals = patterns.map(({ representatives }) => new Float64Array(representatives.length * nodeCount));
  const logIntegrals = patterns.map(({ representatives }) => new Float64Array(representatives.length * nodeCount));
  const largest = patterns.map(({ representatives }) => new Float64Array(representatives.length));
  const factorSums = patterns.map(({ representatives }) => new Float64Array(representatives.length * nodeCount));
  // Per testlet, for a person whose pattern is not frequent: exp(S_m - max S) over the sums θ_q + γ_g, the integral at
  // each θ_q, and whether the person answered it.
  const scaled = testlets.map(() => new Float64Array(sumCount));
  const personIntegrals = testlets.map(() => new Float64Array(nodeCount));
  const answered = new Uint8Array(testlets.length);
  const factors = new Float64Array(nodeCount);
  const totals = new Float64Array(sumCount);
  const sums = new Float64Array(sumCount);

  /** Writes exp(S - max S) of a person's responses to testlet t into `sums` and returns max S. */
  const scaledSums = (t: number, person: number): number => {
    sums.fill(0);
    for (let r = data.offsets[person] ?? 0; r < (data.offsets[person + 1] ?? 0); r++) {
      const item = data.items[r] ?? 0;
      if (testletOf[item] === t)
        gather(sums, logTable, (tableStarts[item] ?? 0) + (data.codes[r] ?? 0), categories[item] ?? 0);
    }
    return scaleSums(sums);
  };

  /** Adds `totals` to the counts of each item of testlet t at the person's response. */
  const countItems = (t: number, start: number, end: number): void => {
    for (let r = start; r < end; r++) {
      const item = data.items[r] ?? 0;
      const itemCount = itemCounts[item];
      if (testletOf[item] === t && itemCount !== undefined) {
        scatter(itemCount, data.codes[r] ?? 0, categories[item] ?? 0, totals);
      }
    }
  };

  return {
    prepare() {
      testlets.forEach(({ weights }, t) => {
        const representatives = patterns[t]?.representatives ?? new Int32Array(0);
        const integral = integrals[t];
        const logIntegral = logIntegrals[t];
        const maxima = largest[t];
        if (integral === undefined || logIntegral === undefined || maxima === undefined) return;
        representatives.forEach((person, slot) => {
          maxima[slot] = scaledSums(t, person);
          const values = integral.subarray(slot * nodeCount, (slot + 1) * nodeCount);
          integrateScaled(weights, sums, values);
          for (let q = 0; q < nodeCount; q++) logIntegral[slot * nodeCount + q] = Math.log(values[q] ?? 0);
        });
        factorSums[t]?.fill(0);
      });
    },
    integrate(person, start, end, logPosterior) {
      answered.fill(0);
      for (const values of scaled) values.fill(0);
      for (let r = start; r < end; r++) {
        const item = data.items[r] ?? 0;
        const t = testletOf[item] ?? -1;
        const values = scaled[t];
        if (values === undefined || (patterns[t]?.slotOf[person] ?? -1) >= 0) continue;
        answered[t] = 1;
        gather(values, logTable, (tableStarts[item] ?? 0) + (data.codes[r] ?? 0), categories[item] ?? 0);
      }
      testlets.forEach(({ weights }, t) => {
        const slot = patterns[t]?.slotOf[person] ?? -1;
        if (slot >= 0) {
          addLogIntegral(logPosterior, largest[t]?.[slot] ?? 0, logIntegrals[t], slot * nodeCount);
          return;
        }
        const values = scaled[t];
        const integral = personIntegrals[t];
        if (answered[t] !== 1 || values === undefined || integral === undefined) return;
        const maximum = scaleSums(values);
        integrateScaled(weights, values, integral);
        for (let q = 0; q < nodeCount; q++) {
          logPosterior[q] = (logPosterior[q] ?? 0) + maximum + Math.log(integral[q] ?? 0);
        }
      });
    },
    count(person, start, end, posterior) {
      testlets.forEach(({ weights, counts }, t) => {
        const slot = patterns[t]?.slotOf[person] ?? -1;
        if (slot >= 0) {
          addFactors(posterior, integrals[t], factorSums[t], slot * nodeCount);
          return;
        }
        const values = scaled[t];
        const integral = personIntegrals[t];
        if (answered[t] !== 1 || values === undefined || integral === undefined) return;
        factors.fill(0);
        addFactors(posterior, integral, factors, 0);
        countTestlet(weights, values, factors, totals, counts);
        countItems(t, start, end);
      });
    },
    finish() {
      testlets.forEach(({ weights, counts }, t) => {
        const sumsOfFactors = factorSums[t];
        if (sumsOfFactors === undefined) return;
        patterns[t]?.representatives.forEach((person, slot) => {
          scaledSums(t, person);
          countTestlet(weights, sums, sumsOfFactors.subarray(slot * nodeCount, (slot + 1) * nodeCount), totals, counts);
          countItems(t, data.offsets[person] ?? 0, data.offsets[person + 1] ?? 0);
        });
      });
    },
  };
}

/**
 * Returns testlet t's frequent response patterns: those of at least two persons, up to {@link MAX_PATTERNS} of the
 * most frequent. A pattern is the person's items of the testlet with their responses.
 */
function frequentPatterns(data: CompressedResponses, testletOf: Int32Array, t: number): TestletPatterns {
  const personCount = data.offsets.length - 1;
  const keys: (string | null)[] = [];
  const frequencies = new Map<string, { count: number; first: number }>();
  for (let person = 0; person < personCount; person++) {
    const key = patternKey(data, testletOf, t, person);
    keys.push(key);
    if (key === null) continue;
    const entry = frequencies.get(key);
    if (entry === undefined) frequencies.set(key, { count: 1, first: person });
    else entry.count++;
  }
  const frequent = [...frequencies.entries()]
    .filter(([, { count }]) => count >= 2)
    .sort(([, x], [, y]) => y.count - x.count || x.first - y.first)
    .slice(0, MAX_PATTERNS);
  const slots = new Map(frequent.map(([key], slot) => [key, slot]));
  return {
    slotOf: Int32Array.from(keys, (key) => (key === null ? -1 : (slots.get(key) ?? -1))),
    representatives: Int32Array.from(frequent, ([, { first }]) => first),
  };
}

/** Adds `table[start + i stride]` to `out[i]` for each entry of `out`: an item's log-probabilities of one response. */
function gather(out: Float64Array, table: Float64Array, start: number, stride: number): void {
  let index = start;
  for (let i = 0; i < out.length; i++, index += stride) out[i] = (out[i] ?? 0) + (table[index] ?? 0);
}

/** Adds `values[i]` to `out[start + i stride]` for each entry of `values`: one response's expected counts. */
function scatter(out: Float64Array, start: number, stride: number, values: Float64Array): void {
  let index = start;
  for (let i = 0; i < values.length; i++, index += stride) out[index] = (out[index] ?? 0) + (values[i] ?? 0);
}

/** Returns a person's items of testlet t with their responses, as a key, or `null` without responses to it. */
function patternKey(data: CompressedResponses, testletOf: Int32Array, t: number, person: number): string | null {
  const parts: number[] = [];
  for (let r = data.offsets[person] ?? 0; r < (data.offsets[person + 1] ?? 0); r++) {
    const item = data.items[r] ?? 0;
    if (testletOf[item] === t) parts.push(item, data.codes[r] ?? 0);
  }
  return parts.length === 0 ? null : parts.join(" ");
}

/** Turns a testlet's log-likelihoods `sums` over θ + γ into exp(S - max S) in place and returns max S. */
function scaleSums(sums: Float64Array): number {
  let largest = Number.NEGATIVE_INFINITY;
  for (const value of sums) largest = Math.max(largest, value);
  for (let m = 0; m < sums.length; m++) sums[m] = Math.exp((sums[m] ?? 0) - largest);
  return largest;
}

/** Writes Σ_g w_g e_(q+g) at each θ_q into `integral`, from the scaled likelihoods e over θ + γ. */
function integrateScaled(weights: Float64Array, scaled: Float64Array, integral: Float64Array): void {
  for (let q = 0; q < integral.length; q++) {
    let sum = 0;
    for (let g = 0; g < weights.length; g++) sum += (weights[g] ?? 0) * (scaled[q + g] ?? 0);
    integral[q] = sum;
  }
}

/** Adds max S + ln integral_q, from `logIntegral` at `offset`, to `logPosterior` at each θ_q. */
function addLogIntegral(
  logPosterior: Float64Array,
  largest: number,
  logIntegral: Float64Array | undefined,
  offset: number,
): void {
  for (let q = 0; q < logPosterior.length; q++) {
    logPosterior[q] = (logPosterior[q] ?? 0) + largest + (logIntegral?.[offset + q] ?? 0);
  }
}

/** Adds post(θ_q) / integral_q, 0 where the integral is 0, to `out` at `offset` at each θ_q. */
function addFactors(
  posterior: Float64Array,
  integral: Float64Array | undefined,
  out: Float64Array | undefined,
  offset: number,
): void {
  if (integral === undefined || out === undefined) return;
  for (let q = 0; q < posterior.length; q++) {
    const value = integral[offset + q] ?? 0;
    out[offset + q] = (out[offset + q] ?? 0) + (value > 0 ? (posterior[q] ?? 0) / value : 0);
  }
}

/**
 * Writes the expected counts over θ + γ, Σ_(q+g=m) f_q w_g e_(q+g), into `totals` and adds their margin over γ to the
 * testlet's `counts`, from the factors f_q, post(θ_q) / integral_q summed over persons, and the scaled likelihoods e.
 */
function countTestlet(
  weights: Float64Array,
  scaled: Float64Array,
  factors: Float64Array,
  totals: Float64Array,
  counts: Float64Array,
): void {
  totals.fill(0);
  for (let q = 0; q < factors.length; q++) {
    const factor = factors[q] ?? 0;
    for (let g = 0; g < weights.length; g++) {
      const joint = factor * (weights[g] ?? 0) * (scaled[q + g] ?? 0);
      totals[q + g] = (totals[q + g] ?? 0) + joint;
      counts[g] = (counts[g] ?? 0) + joint;
    }
  }
}
