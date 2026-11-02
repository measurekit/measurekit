import { checkArray, checkChoice, checkRecord, DataError, describe, isRecord } from "@measurekit/core";
import { invertPositiveDefinite, newton, type Quadrature } from "../math/internal.js";
import { personGroups } from "./responses.js";

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

/**
 * The regression of theta on the persons' covariates: θ ~ N(xᵀβ, σ²) with no intercept. Persons with the same
 * covariates share a pattern, whose prior over the nodes and expected counts the E-step and M-step keep once.
 */
interface Regression {
  /** The covariates' names, in the order of the coefficients. */
  readonly names: readonly string[];
  /** The pattern of each person; -1 for a person without responses. */
  readonly patternOf: Int32Array;
  /** The distinct covariate rows of the respondents, by pattern, then covariate. */
  readonly patterns: Float64Array;
  /** The number of respondents of each pattern. */
  readonly respondents: Float64Array;
  /** The inverse of XᵀX over the respondents, by rows. */
  readonly inverseGram: Float64Array;
  /** The free parameter of each coefficient. */
  readonly slots: Int32Array;
  /** The free parameter of the residual standard deviation; -1 when it is fixed at 1. */
  readonly sdSlot: number;
  /** The expected number of respondents of each pattern at each node, by pattern, then node: written by the E-step. */
  readonly counts: Float64Array;
  /** The log weight of each node under each pattern's prior, by pattern, then node. */
  readonly logWeights: Float64Array;
}

/** The groups of a calibration and each person's group; with a regression, one group whose prior it replaces. */
export interface Population {
  readonly groups: readonly LatentGroup[];
  /** The index of each person's group. */
  readonly groupOf: Int32Array;
  /** The regression of theta on covariates, if there is one. */
  readonly regression?: Regression | undefined;
}

/** The population options of a calibration: each person's group and the distribution of theta in each group. */
export interface PopulationOptions {
  readonly groups?: readonly string[] | undefined;
  readonly latent?: LatentDistribution | Readonly<Record<string, LatentDistribution>> | undefined;
  readonly covariates?: Readonly<Record<string, readonly number[]>> | undefined;
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
 * estimated, or covariates come with groups, a distribution other than the standard normal, or a rule other than the
 * rectangular one.
 */
export function population(
  options: PopulationOptions,
  answered: readonly boolean[],
  rule: Quadrature,
  rectangular: boolean,
  scaleFixed: boolean,
  addFree: AddFreeParameter,
): Population {
  if (options.covariates !== undefined) {
    return regressionPopulation(options, answered, rule, rectangular, scaleFixed, addFree);
  }
  const { labels, groupOf } = personGroups(options.groups, answered.length);
  const names: readonly (string | null)[] = labels ?? [null];
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
export function normalWeights(population: Population, values: Float64Array, nodes: Float64Array): void {
  if (population.regression !== undefined) refreshRegression(population.regression, values, nodes);
  for (const group of population.groups) {
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
  if (population.regression !== undefined) maximizeRegression(population.regression, values, nodes);
  for (const group of population.groups) maximizeGroup(group, values, nodes);
  normalWeights(population, values, nodes);
}

/** Updates a group's estimated distribution from the expected counts of the last E-step. */
function maximizeGroup(group: LatentGroup, values: Float64Array, nodes: Float64Array): void {
  if (group.distribution === "standard-normal") return;
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
    return;
  }
  const sd = values[group.sdSlot] ?? 1;
  const [eta1, eta2] = normalOnNodes(nodes, first / total, second / total, [
    (values[group.meanSlot] ?? 0) / (sd * sd),
    -0.5 / (sd * sd),
  ]);
  values[group.meanSlot] = -eta1 / (2 * eta2);
  values[group.sdSlot] = Math.sqrt(-0.5 / eta2);
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
  population: Population,
  values: Float64Array,
  nodes: Float64Array,
  gradient: Float64Array,
): number {
  let value =
    population.regression === undefined ? 0 : regressionObjective(population.regression, values, nodes, gradient);
  for (const group of population.groups) {
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
export function hasDegenerateGroup(
  { groups, regression }: Population,
  values: Float64Array,
  nodes: Float64Array,
): boolean {
  const lowest = nodes[0] ?? 0;
  const range = (nodes.at(-1) ?? 0) - lowest;
  const spacing = range / (nodes.length - 1);
  if (regression !== undefined && regression.sdSlot >= 0) {
    const sd = values[regression.sdSlot] ?? 1;
    if (!(sd >= spacing && sd <= range)) return true;
  }
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
 * Returns the distribution of each group; none with a regression, whose prior differs by person.
 *
 * @param standardError - The standard error of a free parameter, or `null` when there are none.
 */
export function groupDistributions(
  { groups, regression }: Population,
  values: Float64Array,
  nodes: Float64Array,
  standardError: ((slot: number) => number) | null,
): GroupDistribution[] {
  if (regression !== undefined) return [];
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

/** The regression of a calibration as it reports it. */
interface RegressionEstimate {
  readonly coefficients: Readonly<Record<string, number>>;
  readonly residualSd: number;
  readonly standardErrors: Readonly<Record<string, number>> | null;
}

/**
 * Returns the estimated coefficients and residual standard deviation, or `undefined` without a regression.
 *
 * @param standardError - The standard error of a free parameter, or `null` when there are none.
 */
export function regressionEstimate(
  { regression }: Population,
  values: Float64Array,
  standardError: ((slot: number) => number) | null,
): RegressionEstimate | undefined {
  if (regression === undefined) return undefined;
  const { names, slots, sdSlot } = regression;
  const entries = (read: (slot: number) => number): Record<string, number> =>
    Object.fromEntries(names.map((name, k) => [name, read(slots[k] ?? 0)]));
  return {
    coefficients: entries((slot) => values[slot] ?? 0),
    residualSd: sdSlot < 0 ? FIXED_RESIDUAL_SD : (values[sdSlot] ?? FIXED_RESIDUAL_SD),
    standardErrors: standardError === null ? null : entries(standardError),
  };
}

/** The residual standard deviation of a regression whose scale the items fix: theta keeps its N(0, 1) unit. */
const FIXED_RESIDUAL_SD = 1;

/**
 * Builds the population of a latent regression: one group whose prior the regression replaces, and the coefficients and
 * the residual standard deviation as free parameters. The residual standard deviation is estimated only when fixed
 * items fix the scale; otherwise the items' slopes carry it and it stays at 1.
 *
 * @throws `DataError` when the covariates are not an object of arrays of finite numbers, one for each person, with at
 * least one name; groups, a distribution other than the standard normal, or a rule other than the rectangular one come
 * with them; or the respondents' covariates are collinear.
 */
function regressionPopulation(
  options: PopulationOptions,
  answered: readonly boolean[],
  rule: Quadrature,
  rectangular: boolean,
  scaleFixed: boolean,
  addFree: AddFreeParameter,
): Population {
  if (options.groups !== undefined) {
    throw new DataError("Covariates need one group: a group is a covariate in a latent regression.");
  }
  if (options.latent !== undefined && options.latent !== "standard-normal") {
    throw new DataError(
      "Covariates replace the distribution of theta: the option latent must be absent or standard-normal.",
    );
  }
  if (!rectangular) throw new DataError("A latent regression needs the rectangular quadrature rule.");
  const { names, columns } = checkCovariates(options.covariates, answered.length);
  const { patternOf, patterns, respondents } = distinctPatterns(columns, answered);
  const size = names.length;
  const inverseGram = invertPositiveDefinite(gramMatrix(patterns, respondents, size), size);
  if (inverseGram === null) {
    throw new DataError("The covariates of the persons who answered are collinear: leave one of them out.");
  }
  const slots = Int32Array.from(names, () => addFree(Number.NEGATIVE_INFINITY, 0));
  const sdSlot = scaleFixed ? addFree(0, 1) : -1;
  const cells = respondents.length * rule.nodes.length;
  const regression: Regression = {
    names,
    patternOf,
    patterns,
    respondents,
    inverseGram,
    slots,
    sdSlot,
    counts: new Float64Array(cells),
    logWeights: new Float64Array(cells),
  };
  const standard = rule.weights.map(Math.log);
  for (let r = 0; r < respondents.length; r++) regression.logWeights.set(standard, r * standard.length);
  const group: LatentGroup = {
    name: null,
    distribution: "standard-normal",
    logWeights: standard,
    counts: new Float64Array(rule.nodes.length),
    meanSlot: -1,
    sdSlot: -1,
  };
  return { groups: [group], groupOf: new Int32Array(answered.length), regression };
}

/**
 * Checks the covariates against the number of persons.
 *
 * @throws `DataError` unless they are a non-empty object of arrays with a finite number for each person.
 */
function checkCovariates(
  covariates: PopulationOptions["covariates"],
  personCount: number,
): { readonly names: readonly string[]; readonly columns: readonly (readonly number[])[] } {
  checkRecord(covariates, "The option covariates must be an object of values by covariate name");
  const names = Object.keys(covariates ?? {});
  if (names.length === 0) throw new DataError("The option covariates needs at least one covariate.");
  const columns = names.map((name) => {
    const values = covariates?.[name];
    checkArray(values, `Covariate "${name}" must be an array with a value for each person`);
    if (values?.length !== personCount) {
      throw new DataError(`Covariate "${name}" has ${values?.length} values for ${personCount} persons.`);
    }
    values.forEach((value, person) => {
      if (!(typeof value === "number" && Number.isFinite(value))) {
        throw new DataError(
          `Covariate "${name}" of person ${person + 1} must be a finite number, found ${describe(value)}.`,
        );
      }
    });
    return values;
  });
  return { names, columns };
}

/** Groups the respondents by their covariates: each person's pattern, the patterns' rows and their respondents. */
function distinctPatterns(
  columns: readonly (readonly number[])[],
  answered: readonly boolean[],
): { readonly patternOf: Int32Array; readonly patterns: Float64Array; readonly respondents: Float64Array } {
  const rows: number[][] = [];
  const counts: number[] = [];
  const byKey = new Map<string, number>();
  const patternOf = new Int32Array(answered.length).fill(-1);
  answered.forEach((any, person) => {
    if (!any) return;
    const row = columns.map((column) => column[person] ?? 0);
    const key = row.join(",");
    let pattern = byKey.get(key);
    if (pattern === undefined) {
      pattern = rows.length;
      byKey.set(key, pattern);
      rows.push(row);
      counts.push(0);
    }
    counts[pattern] = (counts[pattern] ?? 0) + 1;
    patternOf[person] = pattern;
  });
  return { patternOf, patterns: Float64Array.from(rows.flat()), respondents: Float64Array.from(counts) };
}

/** Returns Σ_r N_r x_r x_rᵀ, by rows, over the patterns and their numbers of respondents. */
function gramMatrix(patterns: Float64Array, respondents: Float64Array, size: number): Float64Array {
  const gram = new Float64Array(size * size);
  for (let r = 0; r < respondents.length; r++) {
    const weight = respondents[r] ?? 0;
    for (let j = 0; j < size; j++) {
      for (let k = 0; k < size; k++) {
        gram[j * size + k] =
          (gram[j * size + k] ?? 0) + weight * (patterns[r * size + j] ?? 0) * (patterns[r * size + k] ?? 0);
      }
    }
  }
  return gram;
}

/** Returns the mean x_rᵀβ of a pattern's prior under the coefficients in `values`. */
function patternMean(regression: Regression, values: Float64Array, pattern: number): number {
  const size = regression.slots.length;
  let mean = 0;
  for (let k = 0; k < size; k++) {
    mean += (regression.patterns[pattern * size + k] ?? 0) * (values[regression.slots[k] ?? 0] ?? 0);
  }
  return mean;
}

/** Returns the residual standard deviation in `values`, or 1 where the items fix the scale. */
function residualSd(regression: Regression, values: Float64Array): number {
  return regression.sdSlot < 0 ? FIXED_RESIDUAL_SD : (values[regression.sdSlot] ?? FIXED_RESIDUAL_SD);
}

/** Sets each pattern's log weights from the coefficients and the residual standard deviation in `values`. */
function refreshRegression(regression: Regression, values: Float64Array, nodes: Float64Array): void {
  const sd = residualSd(regression, values);
  const nodeCount = nodes.length;
  for (let r = 0; r < regression.respondents.length; r++) {
    const mean = patternMean(regression, values, r);
    const row = regression.logWeights.subarray(r * nodeCount, (r + 1) * nodeCount);
    for (let q = 0; q < nodeCount; q++) row[q] = -0.5 * (((nodes[q] ?? 0) - mean) / sd) ** 2;
    normalizeLogWeights(row);
  }
}

/**
 * Sets the coefficients to the least squares fit to the respondents' posterior means, β = (XᵀX)⁻¹ Σ_r x_r Σ_q n_rq θ_q,
 * and, where the items fix the scale, the residual variance to the mean over respondents of the squared residual of the
 * posterior mean plus the posterior variance, Σ_r Σ_q n_rq (θ_q - x_rᵀβ)² / N.
 *
 * @see {@link https://doi.org/10.1007/BF02294457 | Mislevy (1985)}
 */
function maximizeRegression(regression: Regression, values: Float64Array, nodes: Float64Array): void {
  const { slots, patterns, counts, respondents, inverseGram } = regression;
  const size = slots.length;
  const nodeCount = nodes.length;
  const moments = new Float64Array(size);
  for (let r = 0; r < respondents.length; r++) {
    const total = rowDot(counts, r * nodeCount, nodes);
    for (let k = 0; k < size; k++) moments[k] = (moments[k] ?? 0) + (patterns[r * size + k] ?? 0) * total;
  }
  for (let j = 0; j < size; j++) {
    let coefficient = 0;
    for (let k = 0; k < size; k++) coefficient += (inverseGram[j * size + k] ?? 0) * (moments[k] ?? 0);
    values[slots[j] ?? 0] = coefficient;
  }
  if (regression.sdSlot >= 0) values[regression.sdSlot] = fitResidualSd(regression, values, nodes);
}

/** Returns the root of the mean over respondents of Σ_q n_rq (θ_q - x_rᵀβ)² at the coefficients in `values`. */
function fitResidualSd(regression: Regression, values: Float64Array, nodes: Float64Array): number {
  const { counts, respondents } = regression;
  const nodeCount = nodes.length;
  let squares = 0;
  let persons = 0;
  for (let r = 0; r < respondents.length; r++) {
    const mean = patternMean(regression, values, r);
    for (let q = 0; q < nodeCount; q++) squares += (counts[r * nodeCount + q] ?? 0) * ((nodes[q] ?? 0) - mean) ** 2;
    persons += respondents[r] ?? 0;
  }
  return Math.sqrt(squares / persons);
}

/** Returns Σ_q row[q] node[q] for the row of `table` starting at `start`. */
function rowDot(table: Float64Array, start: number, nodes: Float64Array): number {
  let sum = 0;
  for (let q = 0; q < nodes.length; q++) sum += (table[start + q] ?? 0) * (nodes[q] ?? 0);
  return sum;
}

/**
 * Returns -Σ n_rq ln w_rq of the patterns' priors and subtracts its derivatives with respect to the coefficients and
 * the residual standard deviation from `gradient`, as {@link distributionObjective} does for a normal group.
 */
function regressionObjective(
  regression: Regression,
  values: Float64Array,
  nodes: Float64Array,
  gradient: Float64Array,
): number {
  const { slots, patterns, counts, logWeights, respondents, sdSlot } = regression;
  const size = slots.length;
  const nodeCount = nodes.length;
  const sd = residualSd(regression, values);
  let value = 0;
  let sdSlope = 0;
  for (let r = 0; r < respondents.length; r++) {
    const mean = patternMean(regression, values, r);
    const total = respondents[r] ?? 0;
    let meanSlope = 0;
    for (let q = 0; q < nodeCount; q++) {
      const count = counts[r * nodeCount + q] ?? 0;
      const logWeight = logWeights[r * nodeCount + q] ?? 0;
      value -= count * logWeight;
      const excess = count - total * Math.exp(logWeight);
      const deviation = ((nodes[q] ?? 0) - mean) / sd;
      meanSlope += (excess * deviation) / sd;
      sdSlope += (excess * deviation * deviation) / sd;
    }
    for (let k = 0; k < size; k++) {
      const slot = slots[k] ?? 0;
      gradient[slot] = (gradient[slot] ?? 0) - meanSlope * (patterns[r * size + k] ?? 0);
    }
  }
  if (sdSlot >= 0) gradient[sdSlot] = (gradient[sdSlot] ?? 0) - sdSlope;
  return value;
}
