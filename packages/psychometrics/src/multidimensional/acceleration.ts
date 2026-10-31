/** An EM algorithm as a map on a vector of parameters, for {@link squarem}. */
export interface EmMap {
  /** The number of parameters. */
  readonly size: number;
  /** Writes the current parameters into `out`. */
  read(out: Float64Array): void;
  /** Sets the parameters. Returns `false`, leaving them unchanged, when they are not a valid point. */
  write(parameters: Float64Array): boolean;
  /**
   * Runs one EM iteration from the current parameters and leaves the parameters at its result. Returns the
   * log-likelihood at the parameters it started from and the largest change the iteration made.
   */
  step(): { readonly logLikelihood: number; readonly change: number };
}

/** The most times a cycle's extrapolated step is shortened toward the plain EM steps before the cycle gives it up. */
const MAX_SHORTENINGS = 5;
/** The log-likelihood that an extrapolated point may lose against the cycle's start: rounding of its sum over persons. */
const LOG_LIKELIHOOD_SLACK = 1e-9;

/** The outcome of an EM iteration: the log-likelihood it started from and the largest change it made. */
interface Iteration {
  readonly logLikelihood: number;
  readonly change: number;
}

/** The parameters of a cycle: where it started, after its first and second EM iterations, and the scratch point. */
interface Cycle {
  readonly start: Float64Array;
  readonly first: Float64Array;
  readonly second: Float64Array;
  readonly extrapolated: Float64Array;
}

/**
 * Runs an EM algorithm to convergence with SQUAREM (Varadhan and Roland, 2008, scheme S3): each cycle takes two EM
 * iterations, extrapolates the parameters along their steps, and takes one EM iteration from the extrapolated point.
 *
 * @remarks
 * The extrapolation goes a length α = -‖r‖/‖v‖ along the first step r, with v the change of step, and is shortened toward
 * the plain EM steps, α = -1, when the log-likelihood at the extrapolated point is below the cycle's start or the
 * point is invalid. It accelerates the linear convergence of EM, which is slow where much of the information is
 * missing, to the same fixed points. Every EM iteration counts as an iteration, and the algorithm has converged when
 * one of them changes no parameter by more than `tolerance`, as plain EM does.
 *
 * @see {@link https://doi.org/10.1111/j.1467-9469.2007.00585.x | Varadhan and Roland (2008)}
 */
export function squarem(
  map: EmMap,
  { tolerance, maxIterations }: { readonly tolerance: number; readonly maxIterations: number },
): { readonly converged: boolean; readonly iterations: number } {
  const cycle: Cycle = {
    start: new Float64Array(map.size),
    first: new Float64Array(map.size),
    second: new Float64Array(map.size),
    extrapolated: new Float64Array(map.size),
  };
  let iterations = 0;
  const step = (): Iteration => {
    iterations++;
    return map.step();
  };
  const remaining = (): number => maxIterations - iterations;
  map.read(cycle.start);
  while (remaining() > 0) {
    const initial = step();
    map.read(cycle.first);
    if (initial.change <= tolerance) return { converged: true, iterations };
    if (remaining() === 0) break;
    const next = step();
    map.read(cycle.second);
    if (next.change <= tolerance) return { converged: true, iterations };
    const length = extrapolationLength(cycle);
    const result = length === null ? null : extrapolate(map, cycle, length, initial.logLikelihood, { step, remaining });
    if (result !== null && result.change <= tolerance) return { converged: true, iterations };
    map.read(cycle.start);
  }
  return { converged: false, iterations };
}

/** Returns α = -‖r‖/‖v‖, at most -1, or `null` when the steps leave no direction to extrapolate along. */
function extrapolationLength({ start, first, second }: Cycle): number | null {
  let steps = 0;
  let changes = 0;
  for (let j = 0; j < start.length; j++) {
    const r = (first[j] ?? 0) - (start[j] ?? 0);
    const v = (second[j] ?? 0) - (first[j] ?? 0) - r;
    steps += r * r;
    changes += v * v;
  }
  if (!(changes > 0 && steps > 0)) return null;
  return Math.min(-Math.sqrt(steps / changes), -1);
}

/**
 * Takes an EM iteration from the extrapolated point of a cycle, shortening the extrapolation until the log-likelihood
 * there is not below the cycle's start. Returns that iteration, or `null` when no extrapolated point was accepted, with
 * the parameters at the cycle's second iteration.
 */
function extrapolate(
  map: EmMap,
  { start, first, second, extrapolated }: Cycle,
  length: number,
  startLogLikelihood: number,
  { step, remaining }: { readonly step: () => Iteration; readonly remaining: () => number },
): Iteration | null {
  let alpha = length;
  for (let attempt = 0; attempt < MAX_SHORTENINGS && alpha < -1 && remaining() > 0; attempt++) {
    for (let j = 0; j < start.length; j++) {
      const r = (first[j] ?? 0) - (start[j] ?? 0);
      const v = (second[j] ?? 0) - (first[j] ?? 0) - r;
      extrapolated[j] = (start[j] ?? 0) - 2 * alpha * r + alpha * alpha * v;
    }
    if (map.write(extrapolated)) {
      const result = step();
      if (result.logLikelihood >= startLogLikelihood - LOG_LIKELIHOOD_SLACK) return result;
    }
    alpha = (alpha - 1) / 2;
  }
  map.write(second);
  return null;
}
