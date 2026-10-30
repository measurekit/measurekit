import { checkChoice, DataError, describe, isRecord } from "@measurekit/core";
import { newton } from "../math/optimize.js";
import type { Quadrature } from "../math/quadrature.js";

/**
 * The distribution of theta in a group of persons during calibration: `"standard-normal"`, N(0, 1); `"normal"`, a
 * normal distribution whose mean and standard deviation are estimated; `"empirical-histogram"`, a weight at each
 * quadrature node, estimated.
 *
 * @see {@link https://doi.org/10.1007/BF02293801 | Bock and Aitkin (1981)}
 */
export type LatentDistribution = "standard-normal" | "normal" | "empirical-histogram";

/** The estimated distribution of theta in a group of persons. */
export interface GroupDistribution {
  /** The group's label in the `groups` option; `null` without groups. */
  readonly group: string | null;
  /** The form of the distribution. */
  readonly distribution: LatentDistribution;
  /** The mean of theta: 0 for the standard normal; the mean of the weights over the nodes for a histogram. */
  readonly mean: number;
  /** The standard deviation of theta: 1 for the standard normal; that of the weights for a histogram. */
  readonly sd: number;
  /** Standard errors of the mean and the standard deviation of an estimated normal distribution; `null` otherwise. */
  readonly standardErrors: {
    /** The standard error of the mean. */
    readonly mean: number;
    /** The standard error of the standard deviation. */
    readonly sd: number;
  } | null;
  /** The quadrature nodes. */
  readonly nodes: readonly number[];
  /** The weight of each node, summing to 1: the discrete distribution of theta that calibration integrates over. */
  readonly weights: readonly number[];
}

/** A group of persons during calibration. The E-step reads `logWeights` and writes `counts`. */
interface LatentGroup {
  readonly name: string | null;
  readonly distribution: LatentDistribution;
  /** The log weight of each node. */
  readonly logWeights: Float64Array;
  /** The expected number of the group's persons with responses at each node. */
  readonly counts: Float64Array;
  /** The free parameters of the mean and the standard deviation of a normal distribution; -1 otherwise. */
  readonly meanSlot: number;
  readonly sdSlot: number;
}

/** The groups of a calibration and each person's group. */
export interface Population {
  readonly groups: readonly LatentGroup[];
  /** The index of each person's group. */
  readonly groupOf: Int32Array;
}

/** The population options of a calibration: each person's group and the distribution of theta in each group. */
export interface PopulationOptions {
  readonly groups?: readonly string[] | undefined;
  readonly latent?: LatentDistribution | Readonly<Record<string, LatentDistribution>> | undefined;
}

/** Registers a free parameter with its lower bound and starting value, and returns its index. */
export type AddFreeParameter = (lower: number, start: number) => number;

const DISTRIBUTIONS: readonly LatentDistribution[] = ["standard-normal", "normal", "empirical-histogram"];

/**
 * Checks the population options and builds the groups with their starting weights, the standard normal's, and the
 * free parameters of their normal distributions.
 *
 * @param scaleFixed - Whether fixed items fix the theta scale.
 * @throws `DataError` when `groups` does not give each person a string label, `latent` names an unknown distribution or
 * group, a group has no person with a response, a distribution other than the standard normal is used with a rule
 * other than the rectangular one, or no fixed item and no standard normal group fixes the scale for the distributions
 * estimated.
 */
export function population(
  options: PopulationOptions,
  answered: readonly boolean[],
  rule: Quadrature,
  rectangular: boolean,
  scaleFixed: boolean,
  addFree: AddFreeParameter,
): Population {
  const labels = groupLabels(options.groups, answered.length);
  const names: (string | null)[] = labels === null ? [null] : [...new Set(labels)];
  const groupOf = Int32Array.from(answered, (_, person) => (labels === null ? 0 : names.indexOf(labels[person] ?? "")));
  const distributions = names.map((name) => distributionOf(options.latent, name, labels !== null));
  checkLatentNames(options.latent, labels);
  names.forEach((name, g) => {
    if (!answered.some((any, person) => any && groupOf[person] === g)) {
      throw new DataError(`Group "${name}" has no person with a response.`);
    }
  });
  const estimated = distributions.some((distribution) => distribution !== "standard-normal");
  if (estimated && !rectangular) {
    throw new DataError("Estimating a distribution of theta needs the rectangular quadrature rule.");
  }
  if (estimated && !scaleFixed && !distributions.includes("standard-normal")) {
    throw new DataError(
      "Estimating every group's distribution of theta needs fixed items or a standard normal group to fix the scale.",
    );
  }
  const logWeights = rule.weights.map(Math.log);
  const groups = names.map((name, g): LatentGroup => {
    const distribution = distributions[g] ?? "standard-normal";
    const normal = distribution === "normal";
    return {
      name,
      distribution,
      logWeights: logWeights.slice(),
      counts: new Float64Array(rule.nodes.length),
      meanSlot: normal ? addFree(Number.NEGATIVE_INFINITY, 0) : -1,
      sdSlot: normal ? addFree(0, 1) : -1,
    };
  });
  return { groups, groupOf };
}

function groupLabels(groups: readonly string[] | undefined, personCount: number): readonly string[] | null {
  if (groups === undefined) return null;
  if (!Array.isArray(groups) || groups.length !== personCount) {
    throw new DataError(`Found ${Array.isArray(groups) ? groups.length : "no"} groups for ${personCount} persons.`);
  }
  groups.forEach((label, person) => {
    if (typeof label !== "string") {
      throw new DataError(`The group of person ${person + 1} must be a string, found ${describe(label)}.`);
    }
  });
  return groups;
}

function distributionOf(
  latent: PopulationOptions["latent"],
  name: string | null,
  grouped: boolean,
): LatentDistribution {
  const byGroup = isRecord(latent);
  if (byGroup && !grouped) throw new DataError("Distributions of theta by group need the groups option.");
  const distribution = byGroup ? latent[name ?? ""] : latent;
  if (distribution === undefined) return "standard-normal";
  checkChoice(
    distribution,
    byGroup ? `The distribution of theta of group "${name}"` : "The option latent",
    DISTRIBUTIONS,
  );
  return distribution;
}

function checkLatentNames(latent: PopulationOptions["latent"], labels: readonly string[] | null): void {
  if (!isRecord(latent) || labels === null) return;
  for (const name of Object.keys(latent)) {
    if (!labels.includes(name)) {
      throw new DataError(`A distribution of theta is given for group "${name}", which no person is in.`);
    }
  }
}

/** Sets the log weights of each normal group from its mean and standard deviation in `values`. */
export function normalWeights({ groups }: Population, values: Float64Array, nodes: Float64Array): void {
  for (const group of groups) {
    if (group.distribution !== "normal") continue;
    const mean = values[group.meanSlot] ?? 0;
    const sd = values[group.sdSlot] ?? 1;
    const { logWeights } = group;
    for (let q = 0; q < nodes.length; q++) logWeights[q] = -0.5 * (((nodes[q] ?? 0) - mean) / sd) ** 2;
    normalizeLogWeights(logWeights);
  }
}

/** Shifts log weights so that the weights sum to 1. */
function normalizeLogWeights(logWeights: Float64Array): void {
  let largest = Number.NEGATIVE_INFINITY;
  for (const value of logWeights) largest = Math.max(largest, value);
  let sum = 0;
  for (const value of logWeights) sum += Math.exp(value - largest);
  const shift = largest + Math.log(sum);
  for (let q = 0; q < logWeights.length; q++) logWeights[q] = (logWeights[q] ?? 0) - shift;
}

/**
 * Updates each estimated distribution from the expected counts of the last E-step, maximizing Σ n_q ln w_q: a
 * histogram's weights become its counts over their sum; a normal group's mean and standard deviation become those whose
 * weights on the nodes have the mean and variance of the counts.
 *
 * @see {@link https://doi.org/10.1007/BF02293801 | Bock and Aitkin (1981)}
 */
export function maximizeDistributions(population: Population, values: Float64Array, nodes: Float64Array): void {
  for (const group of population.groups) {
    if (group.distribution === "standard-normal") continue;
    const { counts, logWeights } = group;
    let total = 0;
    let first = 0;
    let second = 0;
    for (let q = 0; q < nodes.length; q++) {
      const count = counts[q] ?? 0;
      const node = nodes[q] ?? 0;
      total += count;
      first += count * node;
      second += count * node * node;
    }
    if (group.distribution === "empirical-histogram") {
      for (let q = 0; q < nodes.length; q++) logWeights[q] = Math.log((counts[q] ?? 0) / total);
      continue;
    }
    const sd = values[group.sdSlot] ?? 1;
    const [eta1, eta2] = normalOnNodes(nodes, first / total, second / total, [
      (values[group.meanSlot] ?? 0) / (sd * sd),
      -0.5 / (sd * sd),
    ]);
    values[group.meanSlot] = -eta1 / (2 * eta2);
    values[group.sdSlot] = Math.sqrt(-0.5 / eta2);
  }
  normalWeights(population, values, nodes);
}

/**
 * Newton's method converges quadratically: a step below 1e-6 leaves an error near 1e-12, below the rounding of the
 * objective's value, which the line search would need to see.
 */
const MOMENT_STEP_TOLERANCE = 1e-6;
const MAX_MOMENT_ITERATIONS = 100;

/**
 * Returns the natural parameters η1 = μ / σ² and η2 = -1 / (2σ²) of the normal distribution whose weights on the
 * nodes, w_q ∝ exp(η1 θ_q + η2 θ_q²), an exponential family, have the first and second moments given: the minimum of
 * the convex ln Σ_q exp(η1 θ_q + η2 θ_q²) - η1 m1 - η2 m2, by Newton's method with a line search from `start`, the
 * group's current distribution. The line search keeps the steps safe when σ is below the spacing of the nodes, where
 * the weights sit on a few nodes and full Newton steps overshoot, and within η2 < 0, the normal distributions: counts
 * spread wider than any normal on the nodes, as an unbounded maximum of the likelihood gives, take σ ever larger.
 */
function normalOnNodes(nodes: Float64Array, first: number, second: number, start: readonly number[]): [number, number] {
  const result = newton(
    (eta, gradient, hessian) => momentObjective(nodes, first, second, eta, gradient, hessian),
    start,
    {
      tolerance: 0,
      stepTolerance: MOMENT_STEP_TOLERANCE,
      maxIterations: MAX_MOMENT_ITERATIONS,
    },
  );
  return [result.x[0] ?? 0, result.x[1] ?? -0.5];
}

/**
 * Returns ln Σ_q exp(η1 θ_q + η2 θ_q²) - η1 m1 - η2 m2 and writes its gradient, the weights' moments less m1 and m2,
 * and its Hessian, the weights' covariance matrix of θ and θ². Returns +∞ where η2 ≥ 0, which is no normal
 * distribution.
 */
function momentObjective(
  nodes: Float64Array,
  first: number,
  second: number,
  eta: Float64Array,
  gradient: Float64Array,
  hessian: Float64Array,
): number {
  const eta1 = eta[0] ?? 0;
  const eta2 = eta[1] ?? 0;
  if (!(eta2 < 0)) return Number.POSITIVE_INFINITY;
  let largest = Number.NEGATIVE_INFINITY;
  for (const node of nodes) largest = Math.max(largest, node * (eta1 + eta2 * node));
  // Σ w, Σ w θ, ..., Σ w θ⁴ of the unnormalized weights w.
  let total = 0;
  let m1 = 0;
  let m2 = 0;
  let m3 = 0;
  let m4 = 0;
  for (const node of nodes) {
    const weight = Math.exp(node * (eta1 + eta2 * node) - largest);
    total += weight;
    m1 += weight * node;
    m2 += weight * node * node;
    m3 += weight * node * node * node;
    m4 += weight * node * node * node * node;
  }
  const e1 = m1 / total;
  const e2 = m2 / total;
  gradient[0] = e1 - first;
  gradient[1] = e2 - second;
  hessian[0] = e2 - e1 * e1;
  hessian[1] = m3 / total - e1 * e2;
  hessian[2] = hessian[1];
  hessian[3] = m4 / total - e2 * e2;
  return largest + Math.log(total) - eta1 * first - eta2 * second;
}

/**
 * Returns -Σ n_q ln w_q of the normal groups, n_q the expected counts and w_q the normalized weights, and subtracts its
 * derivatives with respect to their means and standard deviations from `gradient`.
 *
 * @remarks
 * By Fisher's identity, at the parameters of the E-step, the derivatives of the marginal log-likelihood. With N = Σ
 * n_q, they are Σ (n_q - N w_q) (θ_q - μ) / σ² and Σ (n_q - N w_q) (θ_q - μ)² / σ³, which vanish when the weights have
 * the mean and variance of the counts.
 */
export function distributionObjective(
  { groups }: Population,
  values: Float64Array,
  nodes: Float64Array,
  gradient: Float64Array,
): number {
  let value = 0;
  for (const group of groups) {
    if (group.distribution !== "normal") continue;
    const { counts, logWeights, meanSlot, sdSlot } = group;
    const mean = values[meanSlot] ?? 0;
    const sd = values[sdSlot] ?? 1;
    let total = 0;
    for (const count of counts) total += count;
    let meanSlope = 0;
    let sdSlope = 0;
    for (let q = 0; q < nodes.length; q++) {
      const count = counts[q] ?? 0;
      const logWeight = logWeights[q] ?? 0;
      value -= count * logWeight;
      const excess = count - total * Math.exp(logWeight);
      const deviation = ((nodes[q] ?? 0) - mean) / sd;
      meanSlope += (excess * deviation) / sd;
      sdSlope += (excess * deviation * deviation) / sd;
    }
    gradient[meanSlot] = (gradient[meanSlot] ?? 0) - meanSlope;
    gradient[sdSlot] = (gradient[sdSlot] ?? 0) - sdSlope;
  }
  return value;
}

/**
 * Returns whether an estimated normal distribution has run to a boundary where the likelihood has no maximum.
 *
 * @remarks
 * The nodes show it as a mean outside their range, an SD below their spacing, where the weights sit on one node, or an
 * SD above their range, where the weights are flat. A few persons at one end of the scale, or whose likelihoods agree,
 * give such a distribution.
 */
export function hasDegenerateGroup({ groups }: Population, values: Float64Array, nodes: Float64Array): boolean {
  const lowest = nodes[0] ?? 0;
  const range = (nodes.at(-1) ?? 0) - lowest;
  const spacing = range / (nodes.length - 1);
  return groups.some((group) => {
    if (group.distribution !== "normal") return false;
    const mean = values[group.meanSlot] ?? 0;
    const sd = values[group.sdSlot] ?? 1;
    return !(mean >= lowest && mean <= lowest + range && sd >= spacing && sd <= range);
  });
}

/** Returns the largest change of the histograms' weights from `previous`, which then takes the current weights. */
export function histogramChange({ groups }: Population, previous: Float64Array): number {
  let largest = 0;
  let index = 0;
  for (const group of groups) {
    if (group.distribution !== "empirical-histogram") continue;
    for (const logWeight of group.logWeights) {
      const weight = Math.exp(logWeight);
      largest = Math.max(largest, Math.abs(weight - (previous[index] ?? 0)));
      previous[index++] = weight;
    }
  }
  return largest;
}

/** Returns the number of free weights of the histograms, which the information criteria count as parameters. */
export function histogramParameters({ groups }: Population): number {
  return groups.reduce(
    (sum, group) => sum + (group.distribution === "empirical-histogram" ? group.logWeights.length - 1 : 0),
    0,
  );
}

/**
 * Returns the distribution of each group.
 *
 * @param standardError - The standard error of a free parameter, or `null` when there are none.
 */
export function groupDistributions(
  { groups }: Population,
  values: Float64Array,
  nodes: Float64Array,
  standardError: ((slot: number) => number) | null,
): GroupDistribution[] {
  return groups.map(({ name, distribution, logWeights, meanSlot, sdSlot }) => {
    const weights = Array.from(logWeights, Math.exp);
    const nodeList = Array.from(nodes);
    const normal = distribution === "normal";
    const standardErrors =
      normal && standardError !== null ? { mean: standardError(meanSlot), sd: standardError(sdSlot) } : null;
    if (normal) {
      return {
        group: name,
        distribution,
        mean: values[meanSlot] ?? 0,
        sd: values[sdSlot] ?? 1,
        standardErrors,
        nodes: nodeList,
        weights,
      };
    }
    if (distribution === "standard-normal") {
      return { group: name, distribution, mean: 0, sd: 1, standardErrors, nodes: nodeList, weights };
    }
    const mean = weights.reduce((sum, weight, q) => sum + weight * (nodeList[q] ?? 0), 0);
    const variance = weights.reduce((sum, weight, q) => sum + weight * ((nodeList[q] ?? 0) - mean) ** 2, 0);
    return { group: name, distribution, mean, sd: Math.sqrt(variance), standardErrors, nodes: nodeList, weights };
  });
}
