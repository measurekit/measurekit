import type { HessianFunction } from "../math/optimize.js";

/** An item's location as a sum of coefficients times parameters: `Σ coefficients[a] * parameters[indices[a]]`. */
export interface LocationRow {
  /** The indices of the parameters the item's location depends on, each once. */
  readonly indices: Int32Array;
  /** The coefficient of each of them. */
  readonly coefficients: Float64Array;
}

/**
 * Where the parameters of a linear Rasch model sit in its parameter vector: the location parameters, then the free
 * thresholds of each set, then the mean and the log standard deviation of theta.
 */
export interface RaschLayout {
  /** The location parameters of each item. */
  readonly locations: readonly LocationRow[];
  /** The threshold set of each item. */
  readonly setOf: Int32Array;
  /** The number of thresholds of each set: its categories less 1. */
  readonly thresholdCounts: Int32Array;
  /** The index of the first free threshold of each set. A set of K thresholds has K - 1 free ones. */
  readonly thresholdStarts: Int32Array;
  /** The index of the mean of theta. */
  readonly meanIndex: number;
  /** The index of the logarithm of the standard deviation of theta. */
  readonly logSdIndex: number;
  /** The length of the parameter vector. */
  readonly size: number;
}

/**
 * Writes the step thresholds of an item, its location plus the thresholds of its set, into `out`. The last threshold
 * of a set is minus the sum of the others, so that they sum to 0.
 */
export function writeSteps(layout: RaschLayout, parameters: Float64Array, item: number, out: Float64Array): void {
  const row = layout.locations[item];
  let location = 0;
  row?.indices.forEach((index, a) => {
    location += (row.coefficients[a] ?? 0) * (parameters[index] ?? 0);
  });
  writeThresholds(layout, parameters, layout.setOf[item] ?? 0, out);
  const count = layout.thresholdCounts[layout.setOf[item] ?? 0] ?? 1;
  for (let j = 0; j < count; j++) out[j] = location + (out[j] ?? 0);
}

/** Writes the thresholds of a set, which sum to 0, into `out`. */
export function writeThresholds(layout: RaschLayout, parameters: Float64Array, set: number, out: Float64Array): void {
  const count = layout.thresholdCounts[set] ?? 1;
  const start = layout.thresholdStarts[set] ?? 0;
  let sum = 0;
  for (let j = 0; j < count - 1; j++) {
    out[j] = parameters[start + j] ?? 0;
    sum += out[j] ?? 0;
  }
  out[count - 1] = 0 - sum;
}

/** Scratch space for one item: its steps, one node's category moments and the sums its derivatives are made of. */
interface Work {
  readonly steps: Float64Array;
  /** The logits of the categories of the node being processed. */
  readonly logits: Float64Array;
  readonly probabilities: Float64Array;
  /** At j from 1 to K: P(X ≥ j), Cov(1{X ≥ j}, X) and the expected number of persons with X ≥ j. */
  readonly tails: Float64Array;
  readonly covariances: Float64Array;
  readonly observedTails: Float64Array;
  /** The mean and the variance of the category. */
  readonly moments: Float64Array;
  /** Σ over nodes of the derivatives with respect to step j: the gradient, and the Hessian with the steps, theta, ... */
  readonly stepGradient: Float64Array;
  readonly stepSteps: Float64Array;
  readonly stepMean: Float64Array;
  readonly stepLogSd: Float64Array;
  /** Gradient with respect to the mean and the log standard deviation, then the Hessian entries mm, ms and ss. */
  readonly theta: Float64Array;
  /** The sums carried to the free thresholds: per step, row sums of the step Hessian, and the reduced forms. */
  readonly rowSums: Float64Array;
  readonly reducedGradient: Float64Array;
  readonly reducedRows: Float64Array;
  readonly reducedMean: Float64Array;
  readonly reducedLogSd: Float64Array;
  readonly reducedSteps: Float64Array;
}

function workFor(layout: RaschLayout): Work {
  const most = layout.thresholdCounts.reduce((largest, count) => Math.max(largest, count), 1);
  const vector = (): Float64Array => new Float64Array(most + 2);
  return {
    steps: vector(),
    logits: vector(),
    probabilities: vector(),
    tails: vector(),
    covariances: vector(),
    observedTails: vector(),
    moments: new Float64Array(2),
    stepGradient: vector(),
    stepSteps: new Float64Array(most * most),
    stepMean: vector(),
    stepLogSd: vector(),
    theta: new Float64Array(5),
    rowSums: vector(),
    reducedGradient: vector(),
    reducedRows: vector(),
    reducedMean: vector(),
    reducedLogSd: vector(),
    reducedSteps: new Float64Array(most * most),
  };
}

/**
 * Returns the negative expected complete-data log-likelihood of a linear Rasch model, -Σ n ln P, with the expected
 * counts n of the E-step, as a function for Newton's method.
 *
 * @remarks
 * Item i at node q gives category k the probability P_k ∝ exp(Σ_(j ≤ k) (θ_q - step_j)), θ_q = μ + σ z_q, with each
 * step the item's location plus a threshold. The log-likelihood is linear in theta and the steps up to the log
 * partition function, so its derivatives in them are moments of the category: with S_j = P(X ≥ j), E and Var the mean
 * and variance of X, C_j = Cov(1{X ≥ j}, X), and the total N and tails M_j of the counts,
 * g_j = N S_j - M_j, g_θ = Σ k n_k - N E, H_jl = N (S_j S_l - S_max(j, l)), H_jθ = N C_j and H_θθ = -N Var. The sums
 * over nodes are carried to the parameters through the item's location row and the free thresholds of its set.
 *
 * @param counts - The expected counts of each item, by node, then category.
 */
export function maximization(
  layout: RaschLayout,
  nodes: Float64Array,
  counts: readonly Float64Array[],
): HessianFunction {
  const work = workFor(layout);
  return (parameters, gradient, hessian) => {
    gradient.fill(0);
    hessian.fill(0);
    let value = 0;
    for (let item = 0; item < layout.locations.length; item++) {
      value += sumOverNodes(layout, parameters, nodes, counts[item] ?? new Float64Array(0), item, work);
      addItem(layout, item, work, gradient, hessian);
    }
    for (let j = 0; j < gradient.length; j++) gradient[j] = -(gradient[j] ?? 0);
    for (let j = 0; j < hessian.length; j++) hessian[j] = -(hessian[j] ?? 0);
    return -value;
  };
}

/** Returns an item's term of the log-likelihood and leaves the sums of its derivatives over the nodes in `work`. */
function sumOverNodes(
  layout: RaschLayout,
  parameters: Float64Array,
  nodes: Float64Array,
  counts: Float64Array,
  item: number,
  work: Work,
): number {
  writeSteps(layout, parameters, item, work.steps);
  const count = layout.thresholdCounts[layout.setOf[item] ?? 0] ?? 1;
  const mean = parameters[layout.meanIndex] ?? 0;
  const sd = Math.exp(parameters[layout.logSdIndex] ?? 0);
  work.stepGradient.fill(0);
  work.stepSteps.fill(0);
  work.stepMean.fill(0);
  work.stepLogSd.fill(0);
  work.theta.fill(0);
  let value = 0;
  nodes.forEach((z, q) => {
    const row = counts.subarray(q * (count + 1), (q + 1) * (count + 1));
    value += nodeValue(mean + sd * z, sd * z, row, count, work);
  });
  return value;
}

/**
 * Returns the node's term Σ_k n_k ln P_k and adds its derivatives to the item's sums, for a node at theta with the
 * counts `row`, where `scaled` = σ z is the derivative of theta with respect to the log standard deviation.
 */
function nodeValue(theta: number, scaled: number, row: Float64Array, count: number, work: Work): number {
  const { steps, logits, probabilities } = work;
  logits[0] = 0;
  let largest = 0;
  for (let k = 1; k <= count; k++) {
    logits[k] = (logits[k - 1] ?? 0) + theta - (steps[k - 1] ?? 0);
    largest = Math.max(largest, logits[k] ?? 0);
  }
  let total = 0;
  for (let k = 0; k <= count; k++) total += Math.exp((logits[k] ?? 0) - largest);
  const logTotal = largest + Math.log(total);
  let value = 0;
  let persons = 0;
  for (let k = 0; k <= count; k++) {
    probabilities[k] = Math.exp((logits[k] ?? 0) - logTotal);
    value += (row[k] ?? 0) * ((logits[k] ?? 0) - logTotal);
    persons += row[k] ?? 0;
  }
  if (persons > 0) {
    categoryMoments(count, work);
    tailSums(count, row, work);
    addStepSums(persons, scaled, count, work);
    addThetaSums(persons, scaled, row, work);
  }
  return value;
}

/** Writes the mean and the variance of the category into `work.moments`. */
function categoryMoments(count: number, { probabilities, moments }: Work): void {
  let mean = 0;
  for (let k = 0; k <= count; k++) mean += k * (probabilities[k] ?? 0);
  let variance = 0;
  for (let k = 0; k <= count; k++) variance += (k - mean) ** 2 * (probabilities[k] ?? 0);
  moments[0] = mean;
  moments[1] = variance;
}

/** Writes P(X ≥ j), Cov(1{X ≥ j}, X) and the expected counts at or above j, for j from 1 to the number of steps. */
function tailSums(
  count: number,
  row: Float64Array,
  { probabilities, moments, tails, covariances, observedTails }: Work,
): void {
  tails[count + 1] = 0;
  covariances[count + 1] = 0;
  observedTails[count + 1] = 0;
  for (let j = count; j >= 1; j--) {
    tails[j] = (tails[j + 1] ?? 0) + (probabilities[j] ?? 0);
    covariances[j] = (covariances[j + 1] ?? 0) + (probabilities[j] ?? 0) * (j - (moments[0] ?? 0));
    observedTails[j] = (observedTails[j + 1] ?? 0) + (row[j] ?? 0);
  }
}

/** Adds the node's gradient and Hessian with respect to the steps, and their cross terms with theta, to the sums. */
function addStepSums(persons: number, scaled: number, count: number, work: Work): void {
  const { tails, covariances, observedTails } = work;
  for (let j = 1; j <= count; j++) {
    work.stepGradient[j - 1] = (work.stepGradient[j - 1] ?? 0) + persons * (tails[j] ?? 0) - (observedTails[j] ?? 0);
    work.stepMean[j - 1] = (work.stepMean[j - 1] ?? 0) + persons * (covariances[j] ?? 0);
    work.stepLogSd[j - 1] = (work.stepLogSd[j - 1] ?? 0) + persons * (covariances[j] ?? 0) * scaled;
    for (let l = 1; l <= count; l++) {
      const at = (j - 1) * count + (l - 1);
      const curvature = (tails[j] ?? 0) * (tails[l] ?? 0) - (tails[Math.max(j, l)] ?? 0);
      work.stepSteps[at] = (work.stepSteps[at] ?? 0) + persons * curvature;
    }
  }
}

/** Adds the node's gradient and Hessian with respect to the mean and the log standard deviation of theta. */
function addThetaSums(persons: number, scaled: number, row: Float64Array, { moments, theta }: Work): void {
  let observed = 0;
  row.forEach((n, k) => {
    observed += k * n;
  });
  const gradient = observed - persons * (moments[0] ?? 0);
  const curvature = -persons * (moments[1] ?? 0);
  theta[0] = (theta[0] ?? 0) + gradient;
  theta[1] = (theta[1] ?? 0) + gradient * scaled;
  theta[2] = (theta[2] ?? 0) + curvature;
  theta[3] = (theta[3] ?? 0) + curvature * scaled;
  theta[4] = (theta[4] ?? 0) + curvature * scaled * scaled + gradient * scaled;
}

/** Adds `value` at (i, j) and (j, i) of a symmetric matrix of the given size, once on the diagonal. */
function addSymmetric(matrix: Float64Array, size: number, i: number, j: number, value: number): void {
  matrix[i * size + j] = (matrix[i * size + j] ?? 0) + value;
  if (i !== j) matrix[j * size + i] = (matrix[j * size + i] ?? 0) + value;
}

/** Carries an item's sums to the gradient and Hessian of the parameters. */
function addItem(layout: RaschLayout, item: number, work: Work, gradient: Float64Array, hessian: Float64Array): void {
  const count = layout.thresholdCounts[layout.setOf[item] ?? 0] ?? 1;
  reduceToThresholds(count, work);
  addLocations(layout, item, count, work, gradient, hessian);
  addThresholds(layout, item, count, work, gradient, hessian);
  const { meanIndex, logSdIndex, size } = layout;
  const t = work.theta;
  gradient[meanIndex] = (gradient[meanIndex] ?? 0) + (t[0] ?? 0);
  gradient[logSdIndex] = (gradient[logSdIndex] ?? 0) + (t[1] ?? 0);
  addSymmetric(hessian, size, meanIndex, meanIndex, t[2] ?? 0);
  addSymmetric(hessian, size, meanIndex, logSdIndex, t[3] ?? 0);
  addSymmetric(hessian, size, logSdIndex, logSdIndex, t[4] ?? 0);
}

/**
 * Writes the sums with respect to the K - 1 free thresholds from those with respect to the K steps: the last threshold
 * is minus the sum of the others, so the free threshold a stands for step a less the last step.
 */
function reduceToThresholds(count: number, work: Work): void {
  for (let j = 0; j < count; j++) {
    let sum = 0;
    for (let l = 0; l < count; l++) sum += work.stepSteps[j * count + l] ?? 0;
    work.rowSums[j] = sum;
  }
  for (let a = 0; a < count - 1; a++) reduceThreshold(a, count, work);
}

/** Writes the sums of the free threshold a: the step a less the last step. */
function reduceThreshold(a: number, count: number, work: Work): void {
  const last = count - 1;
  work.reducedGradient[a] = (work.stepGradient[a] ?? 0) - (work.stepGradient[last] ?? 0);
  work.reducedRows[a] = (work.rowSums[a] ?? 0) - (work.rowSums[last] ?? 0);
  work.reducedMean[a] = (work.stepMean[a] ?? 0) - (work.stepMean[last] ?? 0);
  work.reducedLogSd[a] = (work.stepLogSd[a] ?? 0) - (work.stepLogSd[last] ?? 0);
  for (let b = 0; b < last; b++) {
    work.reducedSteps[a * last + b] =
      (work.stepSteps[a * count + b] ?? 0) -
      (work.stepSteps[a * count + last] ?? 0) -
      (work.stepSteps[last * count + b] ?? 0) +
      (work.stepSteps[last * count + last] ?? 0);
  }
}

/** Adds the terms of the item's location parameters: each step depends on all of them with the same coefficients. */
function addLocations(
  layout: RaschLayout,
  item: number,
  count: number,
  work: Work,
  gradient: Float64Array,
  hessian: Float64Array,
): void {
  const row = layout.locations[item];
  if (row === undefined) return;
  const { size, meanIndex, logSdIndex } = layout;
  const start = layout.thresholdStarts[layout.setOf[item] ?? 0] ?? 0;
  let totalGradient = 0;
  let totalSteps = 0;
  let totalMean = 0;
  let totalLogSd = 0;
  for (let j = 0; j < count; j++) {
    totalGradient += work.stepGradient[j] ?? 0;
    totalSteps += work.rowSums[j] ?? 0;
    totalMean += work.stepMean[j] ?? 0;
    totalLogSd += work.stepLogSd[j] ?? 0;
  }
  row.indices.forEach((index, p) => {
    const c = row.coefficients[p] ?? 0;
    gradient[index] = (gradient[index] ?? 0) + c * totalGradient;
    row.indices.forEach((other, q) => {
      hessian[index * size + other] =
        (hessian[index * size + other] ?? 0) + c * (row.coefficients[q] ?? 0) * totalSteps;
    });
    addSymmetric(hessian, size, index, meanIndex, c * totalMean);
    addSymmetric(hessian, size, index, logSdIndex, c * totalLogSd);
    for (let a = 0; a < count - 1; a++) addSymmetric(hessian, size, index, start + a, c * (work.reducedRows[a] ?? 0));
  });
}

/** Adds the terms of the item's free thresholds. */
function addThresholds(
  layout: RaschLayout,
  item: number,
  count: number,
  work: Work,
  gradient: Float64Array,
  hessian: Float64Array,
): void {
  const { size, meanIndex, logSdIndex } = layout;
  const start = layout.thresholdStarts[layout.setOf[item] ?? 0] ?? 0;
  const last = count - 1;
  for (let a = 0; a < last; a++) {
    gradient[start + a] = (gradient[start + a] ?? 0) + (work.reducedGradient[a] ?? 0);
    addSymmetric(hessian, size, start + a, meanIndex, work.reducedMean[a] ?? 0);
    addSymmetric(hessian, size, start + a, logSdIndex, work.reducedLogSd[a] ?? 0);
    for (let b = 0; b < last; b++) {
      const at = (start + a) * size + start + b;
      hessian[at] = (hessian[at] ?? 0) + (work.reducedSteps[a * last + b] ?? 0);
    }
  }
}
