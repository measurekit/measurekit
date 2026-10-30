const INV_SQRT_2PI = 0.3989422804014327;

// Coefficients of Cody's (1993) rational approximations, as in R's pnorm, lowest degree last.
const CODY_A = [2.2352520354606837, 161.02823106855587, 1067.6894854603709, 18154.98125334356, 0.06568233791820745];
const CODY_B = [47.202581904688245, 976.0985517377767, 10260.932208618979, 45507.78933502673];
const CODY_C = [
  0.39894151208813466, 8.883149794388377, 93.50665613217785, 597.2702763948002, 2494.5375852903726, 6848.190450536283,
  11602.65143764735, 9842.714838383978, 1.0765576773720192e-8,
];
const CODY_D = [
  22.266688044328117, 235.387901782625, 1519.3775994075547, 6485.558298266761, 18615.571640885097, 34900.95272114598,
  38912.00328609327, 19685.429676859992,
];
const CODY_P = [
  0.215898534057957, 0.12740116116024736, 0.022235277870649807, 0.0014216191932278934, 2.9112874951168793e-5,
  0.023073441764940174,
];
const CODY_Q = [
  1.284260096144911, 0.4682382124808651, 0.06598813786892856, 0.0037823963320275824, 7.297515550839662e-5,
];
const SQRT_32 = 5.656854249492381;
/** Cody's bound of the central region: |x| at most this, the 0.75 quantile of the standard normal distribution. */
const CODY_CENTRAL_LIMIT = 0.67448975;
/** Beyond this y, Φ(-y) underflows to 0 in double precision. */
const CODY_UNDERFLOW = 37.5193;
/**
 * Returns the distribution function of the standard normal distribution at `x`, Φ(x), by Cody's algorithm as R's
 * `pnorm`: accurate to the last bits in both tails, so that Φ(-x) gives the upper tail without cancellation.
 *
 * @see {@link https://doi.org/10.1145/151271.151273 | Cody (1993)}
 */
export function normalCdf(x: number): number {
  const y = Math.abs(x);
  if (y <= CODY_CENTRAL_LIMIT) return 0.5 + centralTerm(x);
  const lower = y <= SQRT_32 ? middleTail(y) : y < CODY_UNDERFLOW ? farTail(y) : 0;
  return x > 0 ? 1 - lower : lower;
}

/** Returns Φ(x) - 1/2 for |x| at most the central limit. */
function centralTerm(x: number): number {
  const square = x * x;
  let numerator = 0;
  let denominator = 0;
  if (Math.abs(x) > Number.EPSILON / 2) {
    numerator = (CODY_A[4] ?? 0) * square;
    denominator = square;
    for (let i = 0; i < 3; i++) {
      numerator = (numerator + (CODY_A[i] ?? 0)) * square;
      denominator = (denominator + (CODY_B[i] ?? 0)) * square;
    }
  }
  return (x * (numerator + (CODY_A[3] ?? 0))) / (denominator + (CODY_B[3] ?? 0));
}

/** Returns Φ(-y) for y above the central limit and at most √32. */
function middleTail(y: number): number {
  let numerator = (CODY_C[8] ?? 0) * y;
  let denominator = y;
  for (let i = 0; i < 7; i++) {
    numerator = (numerator + (CODY_C[i] ?? 0)) * y;
    denominator = (denominator + (CODY_D[i] ?? 0)) * y;
  }
  return gaussianTail(y, (numerator + (CODY_C[7] ?? 0)) / (denominator + (CODY_D[7] ?? 0)));
}

/** Returns Φ(-y) for y above √32 and below the underflow bound. */
function farTail(y: number): number {
  const inverse = 1 / (y * y);
  let numerator = (CODY_P[5] ?? 0) * inverse;
  let denominator = inverse;
  for (let i = 0; i < 4; i++) {
    numerator = (numerator + (CODY_P[i] ?? 0)) * inverse;
    denominator = (denominator + (CODY_Q[i] ?? 0)) * inverse;
  }
  const ratio = (inverse * (numerator + (CODY_P[4] ?? 0))) / (denominator + (CODY_Q[4] ?? 0));
  return gaussianTail(y, (INV_SQRT_2PI - ratio) / y);
}

/** Returns exp(-y² / 2) times `factor`, splitting y² so that its rounding does not enter the exponential (R's do_del). */
function gaussianTail(y: number, factor: number): number {
  const rounded = Math.trunc(y * 16) / 16;
  const remainder = (y - rounded) * (y + rounded);
  return Math.exp(-rounded * rounded * 0.5) * Math.exp(-remainder * 0.5) * factor;
}

/** Returns the density of the standard normal distribution at `x`. */
export function normalDensity(x: number): number {
  return INV_SQRT_2PI * Math.exp(-0.5 * x * x);
}
