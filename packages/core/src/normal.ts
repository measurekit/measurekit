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
