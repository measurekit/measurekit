// The 21-point Gauss-Kronrod rule on [-1, 1] and its embedded 10-point Gauss rule, QUADPACK's qk21 rounded to doubles:
// the Kronrod nodes from the outermost to the center, their weights, and the Gauss weights of the odd-numbered nodes.
const KRONROD_NODES = [
  0.9956571630258081, 0.9739065285171717, 0.9301574913557082, 0.8650633666889845, 0.7808177265864169,
  0.6794095682990244, 0.5627571346686047, 0.4333953941292472, 0.2943928627014602, 0.14887433898163122, 0,
];
const KRONROD_WEIGHTS = [
  0.011694638867371874, 0.032558162307964725, 0.054755896574351995, 0.07503967481091996, 0.0931254545836976,
  0.10938715880229764, 0.12349197626206584, 0.13470921731147334, 0.14277593857706009, 0.14773910490133849,
  0.1494455540029169,
];
const GAUSS_WEIGHTS = [
  0.06667134430868814, 0.1494513491505806, 0.21908636251598204, 0.26926671930999635, 0.29552422471475287,
];
/** The most intervals of an adaptive integral: QUADPACK's `limit` that R's `integrate` passes, ten times over. */
const MAX_INTERVALS = 1000;

/** An interval of an adaptive integral with its estimate and the estimate's error. */
interface Interval {
  readonly lower: number;
  readonly upper: number;
  readonly value: number;
  readonly error: number;
}

/** Returns the 21-point Kronrod estimate of the integral of `f` over an interval, its error |K21 - G10| beside it. */
function kronrod(f: (x: number) => number, lower: number, upper: number): Interval {
  const center = (lower + upper) / 2;
  const half = (upper - lower) / 2;
  const middle = f(center);
  let value = (KRONROD_WEIGHTS[10] ?? 0) * middle;
  let gauss = 0;
  for (let i = 0; i < 10; i++) {
    const offset = half * (KRONROD_NODES[i] ?? 0);
    const pair = f(center - offset) + f(center + offset);
    value += (KRONROD_WEIGHTS[i] ?? 0) * pair;
    if (i % 2 === 1) gauss += (GAUSS_WEIGHTS[(i - 1) / 2] ?? 0) * pair;
  }
  return { lower, upper, value: value * half, error: Math.abs((value - gauss) * half) };
}

/**
 * Returns the integral of `f` from `lower` to `upper` by adaptive Gauss-Kronrod quadrature, as QUADPACK's qag with
 * the 21-point rule: the interval of the largest error is halved until the errors, each |K21 - G10|, sum to at most
 * `relativeTolerance` times the integral's magnitude, an interval can no longer be halved in floating point, or there
 * are 1,000 intervals.
 *
 * @remarks
 * `f` must be finite on the open interval; it is not evaluated at the ends. An integrable singularity at an end takes
 * many halvings, so a caller removes it by a change of variables first.
 */
export function adaptiveIntegral(
  f: (x: number) => number,
  lower: number,
  upper: number,
  relativeTolerance: number,
): number {
  const intervals = [kronrod(f, lower, upper)];
  let value = intervals[0]?.value ?? 0;
  let error = intervals[0]?.error ?? 0;
  while (error > relativeTolerance * Math.abs(value) && intervals.length < MAX_INTERVALS) {
    let worst = 0;
    intervals.forEach((interval, i) => {
      if (interval.error > (intervals[worst]?.error ?? 0)) worst = i;
    });
    const halved = intervals[worst];
    if (halved === undefined) break;
    const middle = (halved.lower + halved.upper) / 2;
    if (!(halved.lower < middle && middle < halved.upper) || halved.error === 0) break;
    const left = kronrod(f, halved.lower, middle);
    const right = kronrod(f, middle, halved.upper);
    intervals.splice(worst, 1, left, right);
    value += left.value + right.value - halved.value;
    error += left.error + right.error - halved.error;
  }
  return intervals.reduce((sum, interval) => sum + interval.value, 0);
}
