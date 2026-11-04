// Wichura's algorithm AS 241 (PPND16) splits: |p - 1/2| up to SPLIT1 uses the central rational function of CONST1 - q²;
// beyond it, with r = √(-ln min(p, 1 - p)), r up to SPLIT2 uses the near tail in r - CONST2, and beyond that the far tail
// in r - SPLIT2.
const SPLIT1 = 0.425;
const CONST1 = 0.180625;
const SPLIT2 = 5;
const CONST2 = 1.6;

/**
 * Returns the quantile of the standard normal distribution at probability `p`: `-Infinity` at 0, `Infinity` at 1 and
 * `NaN` outside [0, 1]. Uses Wichura's algorithm AS 241, as R's `qnorm` does.
 *
 * @param p - The probability, from 0 to 1.
 * @returns The quantile.
 */
export function normalQuantile(p: number): number {
  const q = p - 0.5;
  // The test of the central region comes first and is false for NaN, so that the common case needs no test of its range.
  if (Math.abs(q) <= SPLIT1) {
    const r = CONST1 - q * q;
    return (q * centralNumerator(r)) / centralDenominator(r);
  }
  if (!(p > 0 && p < 1)) {
    if (p === 0) return Number.NEGATIVE_INFINITY;
    return p === 1 ? Number.POSITIVE_INFINITY : Number.NaN;
  }
  const r = Math.sqrt(-Math.log(q < 0 ? p : 1 - p));
  const tail =
    r <= SPLIT2
      ? nearNumerator(r - CONST2) / nearDenominator(r - CONST2)
      : farNumerator(r - SPLIT2) / farDenominator(r - SPLIT2);
  return q < 0 ? -tail : tail;
}

// The numerators and denominators of the rational functions, each of degree 7, in Horner's form with their coefficients
// as literals, which the compiler makes into straight-line code: a loop over a table is several times as slow.

function centralNumerator(r: number): number {
  let sum = 2509.0809287301227;
  sum = sum * r + 33430.57558358813;
  sum = sum * r + 67265.7709270087;
  sum = sum * r + 45921.95393154987;
  sum = sum * r + 13731.69376550946;
  sum = sum * r + 1971.5909503065513;
  sum = sum * r + 133.14166789178438;
  sum = sum * r + 3.3871328727963665;
  return sum;
}

function centralDenominator(r: number): number {
  let sum = 5226.495278852546;
  sum = sum * r + 28729.085735721943;
  sum = sum * r + 39307.89580009271;
  sum = sum * r + 21213.794301586597;
  sum = sum * r + 5394.196021424751;
  sum = sum * r + 687.1870074920579;
  sum = sum * r + 42.31333070160091;
  sum = sum * r + 1;
  return sum;
}

function nearNumerator(x: number): number {
  let sum = 0.0007745450142783414;
  sum = sum * x + 0.022723844989269184;
  sum = sum * x + 0.2417807251774506;
  sum = sum * x + 1.2704582524523684;
  sum = sum * x + 3.6478483247632045;
  sum = sum * x + 5.769497221460691;
  sum = sum * x + 4.630337846156546;
  sum = sum * x + 1.4234371107496835;
  return sum;
}

function nearDenominator(x: number): number {
  let sum = 1.0507500716444169e-9;
  sum = sum * x + 0.0005475938084995345;
  sum = sum * x + 0.015198666563616457;
  sum = sum * x + 0.14810397642748008;
  sum = sum * x + 0.6897673349851;
  sum = sum * x + 1.6763848301838038;
  sum = sum * x + 2.053191626637759;
  sum = sum * x + 1;
  return sum;
}

function farNumerator(x: number): number {
  let sum = 2.0103343992922881e-7;
  sum = sum * x + 0.000027115555687434876;
  sum = sum * x + 0.0012426609473880784;
  sum = sum * x + 0.026532189526576124;
  sum = sum * x + 0.29656057182850487;
  sum = sum * x + 1.7848265399172913;
  sum = sum * x + 5.463784911164114;
  sum = sum * x + 6.657904643501103;
  return sum;
}

function farDenominator(x: number): number {
  let sum = 2.0442631033899397e-15;
  sum = sum * x + 1.421511758316446e-7;
  sum = sum * x + 0.000018463183175100548;
  sum = sum * x + 0.0007868691311456133;
  sum = sum * x + 0.014875361290850615;
  sum = sum * x + 0.1369298809227358;
  sum = sum * x + 0.599832206555888;
  sum = sum * x + 1;
  return sum;
}

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
 * @param x - The point, which may be infinite.
 * @returns The probability that a standard normal variable is at most `x`.
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

/**
 * Returns exp(-y² / 2) times `factor`, splitting y² so that its rounding does not enter the exponential (R's do_del).
 */
function gaussianTail(y: number, factor: number): number {
  const rounded = Math.trunc(y * 16) / 16;
  const remainder = (y - rounded) * (y + rounded);
  return Math.exp(-rounded * rounded * 0.5) * Math.exp(-remainder * 0.5) * factor;
}

/**
 * Returns the density of the standard normal distribution at `x`.
 *
 * @param x - The point.
 * @returns The density at `x`.
 */
export function normalDensity(x: number): number {
  return INV_SQRT_2PI * Math.exp(-0.5 * x * x);
}
