import { forEachTime, type TimedResponses } from "./timed-responses.js";

// The time model of the hierarchical model of speed and accuracy: ln T_ij ~ N(β_i - τ_j, 1 / α_i²), and given theta,
// τ ~ N(a θ, v) with a = ρ σ_τ and v = σ_τ² (1 - ρ²). Given theta the log times of a person are multivariate normal
// with mean β - a θ and covariance D + v 11ᵀ, D = diag(1 / α²), so with w_i = α_i² for the person's timed items,
// c_i = y_i - β_i, S = Σ w_i, A0 = Σ w_i c_i and B0 = Σ w_i c_i², by the matrix determinant lemma and the
// Sherman-Morrison formula
//   ln f(y | θ) = -1/2 (n ln 2π - Σ ln w_i + ln(1 + v S) + B0 + 2 a θ A0 + a² θ² S - v (A0 + a θ S)² / (1 + v S)),
// which is quadratic in theta. Everything the EM algorithm and the standard errors need of the posterior of theta is
// therefore its mean and its second moment.

/** The sums over each person's timed responses that the time model's formulas are made of. */
export interface PersonSums {
  /** n, the number of times. */
  readonly count: Float64Array;
  /** S = Σ α². */
  readonly weight: Float64Array;
  /** A0 = Σ α² (y - β). */
  readonly offset: Float64Array;
  /** B0 = Σ α² (y - β)². */
  readonly squares: Float64Array;
  /** Σ ln α². */
  readonly logWeight: Float64Array;
}

/** Returns room for the sums of `persons` persons. */
export function emptySums(persons: number): PersonSums {
  return {
    count: new Float64Array(persons),
    weight: new Float64Array(persons),
    offset: new Float64Array(persons),
    squares: new Float64Array(persons),
    logWeight: new Float64Array(persons),
  };
}

/** The parameters of the second level that the formulas use: a = ρ σ_τ, the slope of τ on theta, and v, its residual variance. */
export interface Conditional {
  readonly slope: number;
  readonly variance: number;
}

/** Returns the slope and the residual variance of τ given theta from σ_τ and ρ. */
export function conditional(speedSd: number, correlation: number): Conditional {
  return { slope: correlation * speedSd, variance: speedSd * speedSd * (1 - correlation * correlation) };
}

/** Fills `out` with the sums of each person's times at the intensities and discriminations. */
export function fillSums(
  timed: TimedResponses,
  intensity: ArrayLike<number>,
  discrimination: ArrayLike<number>,
  out: PersonSums,
): void {
  for (const array of Object.values(out)) array.fill(0);
  forEachTime(timed, (person, item, logTime) => {
    const weight = (discrimination[item] ?? 0) ** 2;
    const centered = logTime - (intensity[item] ?? 0);
    out.count[person] = (out.count[person] ?? 0) + 1;
    out.weight[person] = (out.weight[person] ?? 0) + weight;
    out.offset[person] = (out.offset[person] ?? 0) + weight * centered;
    out.squares[person] = (out.squares[person] ?? 0) + weight * centered * centered;
    out.logWeight[person] = (out.logWeight[person] ?? 0) + Math.log(weight);
  });
}

/**
 * Writes the log density of each person's times at each node into `out`, by person, then node: ln f(y | θ_q), or 0 for
 * a person without times.
 */
export function fillLogTerms(sums: PersonSums, second: Conditional, nodes: Float64Array, out: Float64Array): void {
  const { slope, variance } = second;
  const nodeCount = nodes.length;
  for (let person = 0; person < sums.count.length; person++) {
    const n = sums.count[person] ?? 0;
    const first = person * nodeCount;
    if (n === 0) {
      out.fill(0, first, first + nodeCount);
      continue;
    }
    const weight = sums.weight[person] ?? 0;
    const offset = sums.offset[person] ?? 0;
    const squares = sums.squares[person] ?? 0;
    const shrink = 1 + variance * weight;
    const constant = -0.5 * (n * Math.log(2 * Math.PI) - (sums.logWeight[person] ?? 0) + Math.log(shrink));
    for (let q = 0; q < nodeCount; q++) {
      const shift = slope * (nodes[q] ?? 0);
      const total = offset + shift * weight;
      const form = squares + 2 * shift * offset + shift * shift * weight - (variance * total * total) / shrink;
      out[first + q] = constant - 0.5 * form;
    }
  }
}

/**
 * The first two moments of a person's speed and theta given the data, from the posterior mean θ̄ and second moment θ̄²
 * of theta: given theta, τ is normal with precision P = 1 / v + S and mean (a θ / v - A0) / P.
 */
export interface SpeedMoments {
  /** E(τ). */
  mean: number;
  /** Var(τ). */
  variance: number;
  /** E(τ²). */
  square: number;
  /** E(τ θ). */
  product: number;
}

/** Writes the moments of person `person`'s speed into `out`; the person needs times. */
export function speedMoments(
  sums: PersonSums,
  person: number,
  second: Conditional,
  thetaMean: number,
  thetaSquare: number,
  out: SpeedMoments,
): void {
  const { slope, variance } = second;
  const offset = sums.offset[person] ?? 0;
  const precision = 1 / variance + (sums.weight[person] ?? 0);
  const ratio = slope / variance;
  out.mean = (ratio * thetaMean - offset) / precision;
  out.square =
    (ratio * ratio * thetaSquare - 2 * ratio * offset * thetaMean + offset * offset) / (precision * precision) +
    1 / precision;
  out.product = (ratio * thetaSquare - offset * thetaMean) / precision;
  out.variance = out.square - out.mean * out.mean;
}

/** The posterior mean and second moment of each person's theta, with the persons' indices of those with times. */
export interface ThetaMoments {
  readonly mean: Float64Array;
  readonly square: Float64Array;
}

/** The residual variance of τ given theta below which the correlation counts as 1, which keeps the likelihood finite. */
const MIN_VARIANCE = 1e-10;

/**
 * Updates the time parameters from the moments of the E-step, in closed form, and returns nothing: `parameters` is
 * [β, α, σ_τ, ρ].
 *
 * @remarks
 * β_i is the mean of ln t + E(τ), 1 / α_i² the mean of (ln t - β_i + E(τ))² + Var(τ), both over the item's times; the
 * slope a of τ on theta is Σ E(τ θ) / Σ E(θ²) and its residual variance v the mean of E(τ²) - 2 a E(τ θ) + a² E(θ²),
 * over the persons with times, and σ_τ = √(a² + v), ρ = a / σ_τ.
 */
export function maximizeTimes(
  timed: TimedResponses,
  sums: PersonSums,
  parameters: Float64Array,
  itemCounts: Float64Array,
  moments: ThetaMoments,
): void {
  const itemCount = itemCounts.length;
  const second = conditional(parameters[2 * itemCount] ?? 0, parameters[2 * itemCount + 1] ?? 0);
  const speed: SpeedMoments = { mean: 0, variance: 0, square: 0, product: 0 };
  const means = new Float64Array(sums.count.length);
  const variances = new Float64Array(sums.count.length);
  let product = 0;
  let thetaSquares = 0;
  let squares = 0;
  let persons = 0;
  for (let person = 0; person < means.length; person++) {
    if ((sums.count[person] ?? 0) === 0) continue;
    const thetaSquare = moments.square[person] ?? 0;
    speedMoments(sums, person, second, moments.mean[person] ?? 0, thetaSquare, speed);
    means[person] = speed.mean;
    variances[person] = speed.variance;
    product += speed.product;
    squares += speed.square;
    thetaSquares += thetaSquare;
    persons++;
  }
  const intensity = parameters.subarray(0, itemCount);
  const discrimination = parameters.subarray(itemCount, 2 * itemCount);
  const totals = new Float64Array(itemCount);
  forEachTime(timed, (person, item, logTime) => {
    totals[item] = (totals[item] ?? 0) + logTime + (means[person] ?? 0);
  });
  for (let i = 0; i < itemCount; i++) intensity[i] = (totals[i] ?? 0) / (itemCounts[i] ?? 1);
  const deviations = new Float64Array(itemCount);
  forEachTime(timed, (person, item, logTime) => {
    deviations[item] =
      (deviations[item] ?? 0) +
      (logTime - (intensity[item] ?? 0) + (means[person] ?? 0)) ** 2 +
      (variances[person] ?? 0);
  });
  for (let i = 0; i < itemCount; i++) discrimination[i] = Math.sqrt((itemCounts[i] ?? 1) / (deviations[i] ?? 1));
  const slope = product / thetaSquares;
  const residual = Math.max((squares - 2 * slope * product + slope * slope * thetaSquares) / persons, MIN_VARIANCE);
  const speedSd = Math.sqrt(slope * slope + residual);
  parameters[2 * itemCount] = speedSd;
  parameters[2 * itemCount + 1] = slope / speedSd;
}

/**
 * Writes the gradient of the negative marginal log-likelihood of the times with respect to [β, α, σ_τ, ρ] into
 * `gradient`, by Fisher's identity: the posterior mean of the score of ln f(y | θ), which needs only θ̄ and θ̄².
 *
 * @remarks
 * With u_i = w_i (g_i + h θ), g_i = c_i - v A0 / (1 + v S) and h = a / (1 + v S), the scores are ∂/∂β_i = u_i,
 * ∂/∂a = -θ A / (1 + v S) with A = A0 + a θ S, ∂/∂v = -1/2 (S / (1 + v S) - A² / (1 + v S)²) and
 * ∂/∂w_i = ((Σ⁻¹)_ii - u_i²) / (2 w_i²) with (Σ⁻¹)_ii = w_i - v w_i² / (1 + v S), chained to α = √w, σ_τ and ρ.
 */
export function timeGradient(
  timed: TimedResponses,
  sums: PersonSums,
  parameters: Float64Array,
  moments: ThetaMoments,
  gradient: Float64Array,
): void {
  const itemCount = (parameters.length - 2) / 2;
  const speedSd = parameters[2 * itemCount] ?? 1;
  const correlation = parameters[2 * itemCount + 1] ?? 0;
  const { slope, variance } = conditional(speedSd, correlation);
  gradient.fill(0);
  let slopeScore = 0;
  let varianceScore = 0;
  for (let person = 0; person < sums.count.length; person++) {
    if ((sums.count[person] ?? 0) === 0) continue;
    const weight = sums.weight[person] ?? 0;
    const offset = sums.offset[person] ?? 0;
    const thetaMean = moments.mean[person] ?? 0;
    const thetaSquare = moments.square[person] ?? 0;
    const shrink = 1 + variance * weight;
    slopeScore -= (offset * thetaMean + slope * weight * thetaSquare) / shrink;
    const quadratic =
      offset * offset + 2 * offset * slope * weight * thetaMean + slope * slope * weight * weight * thetaSquare;
    varianceScore -= 0.5 * (weight / shrink - quadratic / (shrink * shrink));
  }
  forEachTime(timed, (person, item, logTime) => {
    const weight = (parameters[itemCount + item] ?? 0) ** 2;
    const total = sums.weight[person] ?? 0;
    const shrink = 1 + variance * total;
    const centered = logTime - (parameters[item] ?? 0);
    const g = centered - (variance * (sums.offset[person] ?? 0)) / shrink;
    const h = slope / shrink;
    const thetaMean = moments.mean[person] ?? 0;
    const thetaSquare = moments.square[person] ?? 0;
    const first = weight * (g + h * thetaMean);
    const second = weight * weight * (g * g + 2 * g * h * thetaMean + h * h * thetaSquare);
    const inverse = weight - (variance * weight * weight) / shrink;
    gradient[item] = (gradient[item] ?? 0) + first;
    const slot = itemCount + item;
    gradient[slot] = (gradient[slot] ?? 0) + (2 * (parameters[slot] ?? 0) * (inverse - second)) / (2 * weight * weight);
  });
  gradient[2 * itemCount] = correlation * slopeScore + 2 * speedSd * (1 - correlation * correlation) * varianceScore;
  gradient[2 * itemCount + 1] = speedSd * slopeScore - 2 * speedSd * speedSd * correlation * varianceScore;
  for (let j = 0; j < gradient.length; j++) gradient[j] = -(gradient[j] ?? 0);
}
