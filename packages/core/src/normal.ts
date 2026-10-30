// Coefficients of Wichura's algorithm AS 241 (PPND16), lowest degree first.
const CENTRAL_NUMERATOR = [
  3.3871328727963665, 133.14166789178438, 1971.5909503065513, 13731.69376550946, 45921.95393154987, 67265.7709270087,
  33430.57558358813, 2509.0809287301227,
];
const CENTRAL_DENOMINATOR = [
  1, 42.31333070160091, 687.1870074920579, 5394.196021424751, 21213.794301586597, 39307.89580009271, 28729.085735721943,
  5226.495278852546,
];
const NEAR_NUMERATOR = [
  1.4234371107496835, 4.630337846156546, 5.769497221460691, 3.6478483247632045, 1.2704582524523684, 0.2417807251774506,
  0.022723844989269184, 0.0007745450142783414,
];
const NEAR_DENOMINATOR = [
  1, 2.053191626637759, 1.6763848301838038, 0.6897673349851, 0.14810397642748008, 0.015198666563616457,
  0.0005475938084995345, 1.0507500716444169e-9,
];
const FAR_NUMERATOR = [
  6.657904643501103, 5.463784911164114, 1.7848265399172913, 0.29656057182850487, 0.026532189526576124,
  0.0012426609473880784, 0.000027115555687434876, 2.0103343992922881e-7,
];
const FAR_DENOMINATOR = [
  1, 0.599832206555888, 0.1369298809227358, 0.014875361290850615, 0.0007868691311456133, 0.000018463183175100548,
  1.421511758316446e-7, 2.0442631033899397e-15,
];
// Wichura's AS 241 splits: |p - 1/2| up to SPLIT1 uses the central rational function of CONST1 - q²; beyond it, with
// r = √(-ln min(p, 1 - p)), r up to SPLIT2 uses the near tail in r - CONST2, and beyond that the far tail in r - SPLIT2.
const SPLIT1 = 0.425;
const CONST1 = 0.180625;
const SPLIT2 = 5;
const CONST2 = 1.6;

/**
 * Returns the quantile of the standard normal distribution at probability `p`: `-Infinity` at 0, `Infinity` at 1 and
 * `NaN` outside [0, 1]. Uses Wichura's algorithm AS 241, as R's `qnorm` does.
 */
export function normalQuantile(p: number): number {
  if (!(p > 0 && p < 1)) {
    if (p === 0) return Number.NEGATIVE_INFINITY;
    return p === 1 ? Number.POSITIVE_INFINITY : Number.NaN;
  }
  const q = p - 0.5;
  if (Math.abs(q) <= SPLIT1) {
    const r = CONST1 - q * q;
    return (q * polynomial(CENTRAL_NUMERATOR, r)) / polynomial(CENTRAL_DENOMINATOR, r);
  }
  const r = Math.sqrt(-Math.log(q < 0 ? p : 1 - p));
  const tail =
    r <= SPLIT2
      ? polynomial(NEAR_NUMERATOR, r - CONST2) / polynomial(NEAR_DENOMINATOR, r - CONST2)
      : polynomial(FAR_NUMERATOR, r - SPLIT2) / polynomial(FAR_DENOMINATOR, r - SPLIT2);
  return q < 0 ? -tail : tail;
}

function polynomial(coefficients: readonly number[], x: number): number {
  let sum = 0;
  for (let i = coefficients.length - 1; i >= 0; i--) sum = sum * x + (coefficients[i] ?? 0);
  return sum;
}
