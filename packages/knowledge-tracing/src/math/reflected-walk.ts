/**
 * The decay of the slowest mode of the walk past which it is uniform to rounding: e^-40 is 4e-18. That mode decays
 * as e^(-t (1 - cos(π / points))) after t jumps on average.
 */
const UNIFORM_DECAY = 40;
/**
 * The largest mean number of jumps of one part of a step. The Poisson weights are summed without their factor e^-t,
 * so that they reach about e^t, which stays finite below 709; a longer step splits into equal parts, as the walk
 * composes.
 */
const MAX_PART_JUMPS = 500;
/**
 * The terms of the Poisson series summed: up to t + 8 √(t + 1) + 12 jumps. The weights left out sum to at most 2e-16
 * for any mean t up to {@link MAX_PART_JUMPS}, by R's `ppois`.
 */
const TAIL_DEVIATIONS = 8;
const TAIL_TERMS = 12;

/**
 * Returns the step of a reflecting random walk on a grid of `points` points: it takes a log-density on the grid and a
 * mean number of jumps t, and returns the log-density after them.
 *
 * @remarks
 * The walk jumps to either neighbor at equal rates, and a jump past an end stays at the end. Over a gap with t jumps
 * on average, the density p becomes p' = e^-t Σ_k t^k / k! R^k p, where (Rp)_j = (p_{j-1} + p_{j+1}) / 2 and, at an
 * end, (p_0 + p_1) / 2: the chain's transition exp(Q Δt), computed by uniformization. Every term is non-negative, so
 * nothing cancels, and R conserves mass. The variance added is t h² for grid spacing h, away from the ends.
 *
 * The density is exp of the log-density, which underflows below -745, where far cells read 0. Since p' ≥ e^-t p, the
 * step returns max(ln p', ln p - t) at each point: a true lower bound, so a cell that underflowed keeps its own
 * log-density, less t, and never gains a false fat tail.
 *
 * A mean beyond 40 / (1 - cos(π / points)) is taken at that value, as the walk is then uniform to rounding, so that
 * a wrong unit of time costs at most about 940 parts of 500 jumps at 241 points, a fraction of a second.
 *
 * @see {@link https://doi.org/10.1080/03461238.1953.10419459 | Jensen (1953)}
 */
export function reflectedWalk(points: number): (logDensity: readonly number[], jumps: number) => number[] {
  let power = new Float64Array(points);
  let next = new Float64Array(points);
  const sum = new Float64Array(points);
  const cap = UNIFORM_DECAY / (1 - Math.cos(Math.PI / points));

  /** Advances the log-density in place by one part of t jumps on average, and returns its largest value. */
  const part = (logDensity: number[], t: number): number => {
    for (let q = 0; q < points; q++) {
      const density = Math.exp(logDensity[q] ?? 0);
      power[q] = density;
      sum[q] = density;
    }
    const terms = Math.ceil(t + TAIL_DEVIATIONS * Math.sqrt(t + 1) + TAIL_TERMS);
    let weight = 1;
    for (let k = 1; k <= terms; k++) {
      weight *= t / k;
      jump(power, next);
      for (let q = 0; q < points; q++) sum[q] = (sum[q] ?? 0) + weight * (next[q] ?? 0);
      [power, next] = [next, power];
    }
    // ln p' = ln(sum) - t, bounded below by ln p - t; the caller takes off the largest value, so t cancels.
    let top = Number.NEGATIVE_INFINITY;
    for (let q = 0; q < points; q++) {
      const value = Math.max(Math.log(sum[q] ?? 0), logDensity[q] ?? 0) - t;
      logDensity[q] = value;
      top = Math.max(top, value);
    }
    return top;
  };

  return (logDensity, jumps) => {
    const result = [...logDensity];
    const total = Math.min(jumps, cap);
    const parts = Math.ceil(total / MAX_PART_JUMPS);
    // Each part starts from a log-density whose largest value is 0, so that exp neither overflows nor underflows
    // more than it must; the offsets taken off are added back at the end.
    let offset = 0;
    for (let k = 0; k < parts; k++) {
      const top = part(result, total / parts);
      for (let q = 0; q < points; q++) result[q] = (result[q] ?? 0) - top;
      offset += top;
    }
    if (offset !== 0) for (let q = 0; q < points; q++) result[q] = (result[q] ?? 0) + offset;
    return result;
  };
}

/** Writes R p into `out`: each point the mean of its neighbors, and an end the mean of itself and its one neighbor. */
function jump(p: Float64Array, out: Float64Array): void {
  const last = p.length - 1;
  out[0] = ((p[0] ?? 0) + (p[1] ?? 0)) / 2;
  for (let q = 1; q < last; q++) out[q] = ((p[q - 1] ?? 0) + (p[q + 1] ?? 0)) / 2;
  out[last] = ((p[last - 1] ?? 0) + (p[last] ?? 0)) / 2;
}
