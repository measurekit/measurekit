import { normalCdf } from "./normal.js";

/** Genz's choice of rule by |ρ|: 6 Gauss-Legendre points below 0.3, 12 below 0.75, and 20 above. */
const MEDIUM_CORRELATION = 0.3;
const HIGH_CORRELATION_RULE = 0.75;
/** Genz's switch: from this |ρ| the integral is taken over the complement of the correlation, with a closed form. */
const HIGH_CORRELATION = 0.925;
/** A term with an exponent below this, under 1e-43, is skipped in Genz's sums. */
const NEGLIGIBLE_EXPONENT = -100;
const TWO_PI = 2 * Math.PI;
/** The Newton step at which a root of a Legendre polynomial is exact to double precision. */
const ROOT_TOLERANCE = 1e-15;
const MAX_ROOT_ITERATIONS = 100;

/** The nodes of a Gauss-Legendre rule shifted to (0, 2), and its weights. */
interface Rule {
  readonly nodes: Float64Array;
  readonly weights: Float64Array;
}

/**
 * Returns the `count`-point Gauss-Legendre rule, `count` even, with the nodes 1 - x and 1 + x for each positive root x
 * of the Legendre polynomial, as Genz's algorithm uses them.
 */
function legendreRule(count: number): Rule {
  const half = count / 2;
  const nodes = new Float64Array(count);
  const weights = new Float64Array(count);
  // Returns P_count(x) and its derivative, by the three-term recurrence.
  const legendre = (x: number): readonly [number, number] => {
    let previous = 1;
    let current = x;
    for (let n = 2; n <= count; n++) {
      const next = ((2 * n - 1) * x * current - (n - 1) * previous) / n;
      previous = current;
      current = next;
    }
    return [current, (count * (x * current - previous)) / (x * x - 1)];
  };
  for (let i = 0; i < half; i++) {
    let x = Math.cos((Math.PI * (i + 0.75)) / (count + 0.5));
    for (let iteration = 0; iteration < MAX_ROOT_ITERATIONS; iteration++) {
      const [value, slope] = legendre(x);
      const step = value / slope;
      x -= step;
      if (Math.abs(step) <= ROOT_TOLERANCE) break;
    }
    const slope = legendre(x)[1];
    const weight = 2 / ((1 - x * x) * slope * slope);
    nodes[i] = 1 - x;
    nodes[half + i] = 1 + x;
    weights[i] = weight;
    weights[half + i] = weight;
  }
  return { nodes, weights };
}

const SIX_POINTS = legendreRule(6);
const TWELVE_POINTS = legendreRule(12);
const TWENTY_POINTS = legendreRule(20);

function ruleFor(absoluteRho: number): Rule {
  if (absoluteRho < MEDIUM_CORRELATION) return SIX_POINTS;
  return absoluteRho < HIGH_CORRELATION_RULE ? TWELVE_POINTS : TWENTY_POINTS;
}

/**
 * Returns P(X > h, Y > k) for standard normal X and Y with correlation `rho` in the open interval (-1, 1), by Genz's
 * improvement of the Drezner-Wesolowsky method; `h` and `k` are finite.
 */
function upperTail(h: number, k: number, rho: number): number {
  const rule = ruleFor(Math.abs(rho));
  return clamp(Math.abs(rho) < HIGH_CORRELATION ? moderateTail(h, k, rho, rule) : highTail(h, k, rho, rule));
}

/** Returns P(X > h, Y > k) for |ρ| below {@link HIGH_CORRELATION}: an integral over the angle asin(ρ). */
function moderateTail(h: number, k: number, rho: number, rule: Rule): number {
  const squares = (h * h + k * k) / 2;
  const angle = Math.asin(rho) / 2;
  let sum = 0;
  for (let i = 0; i < rule.nodes.length; i++) {
    const sine = Math.sin(angle * (rule.nodes[i] ?? 0));
    sum += (rule.weights[i] ?? 0) * Math.exp((sine * h * k - squares) / (1 - sine * sine));
  }
  return sum * (angle / TWO_PI) + normalCdf(-h) * normalCdf(-k);
}

/**
 * Returns P(X > h, Y > k) for |ρ| of at least {@link HIGH_CORRELATION}: the quadrature is over the complement, with a
 * Taylor-corrected closed form for the part the rule does not integrate well (Genz, 2004).
 */
function highTail(h: number, k: number, rho: number, rule: Rule): number {
  const kk = rho < 0 ? -k : k;
  const hk = rho < 0 ? -h * k : h * k;
  const as = (1 - rho) * (1 + rho);
  const a = Math.sqrt(as);
  const bs = (h - kk) ** 2;
  const c = (4 - hk) / 8;
  const d = (12 - hk) / 16;
  let closed = 0;
  if (-(bs / as + hk) / 2 > NEGLIGIBLE_EXPONENT) {
    closed = a * Math.exp(-(bs / as + hk) / 2) * (1 - (c * (bs - as) * (1 - (d * bs) / 5)) / 3 + (c * d * as * as) / 5);
  }
  if (hk > NEGLIGIBLE_EXPONENT) {
    const b = Math.sqrt(bs);
    const sp = Math.sqrt(TWO_PI) * normalCdf(-b / a);
    closed -= Math.exp(-hk / 2) * sp * b * (1 - (c * bs * (1 - (d * bs) / 5)) / 3);
  }
  const half = a / 2;
  let sum = 0;
  for (let i = 0; i < rule.nodes.length; i++) {
    const xs = (half * (rule.nodes[i] ?? 0)) ** 2;
    const exponent = -(bs / xs + hk) / 2;
    if (exponent > NEGLIGIBLE_EXPONENT) {
      const rs = Math.sqrt(1 - xs);
      const taylor = 1 + c * xs * (1 + d * xs);
      const exact = Math.exp((-(hk / 2) * xs) / (1 + rs) ** 2) / rs;
      sum += Math.exp(exponent) * (taylor - exact) * (rule.weights[i] ?? 0);
    }
  }
  const integral = (half * sum - closed) / TWO_PI;
  if (rho > 0) return integral + normalCdf(-Math.max(h, kk));
  return kk > h ? -integral + normalCdf(kk) - normalCdf(h) : -integral;
}

function clamp(probability: number): number {
  return Math.max(0, Math.min(1, probability));
}

/**
 * Returns P(X ≤ h, Y ≤ k) for standard normal X and Y with correlation `rho`: the bivariate normal distribution
 * function Φ₂(h, k; ρ).
 *
 * @remarks
 * The probability is computed by Genz's version of the Drezner-Wesolowsky method: Gauss-Legendre quadrature with 6, 12
 * or 20 points by |ρ|, and for |ρ| above 0.925 a closed form with the quadrature of its remainder. It is accurate to
 * about 1e-15. `h` and `k` may be infinite, and `rho` is from -1 to 1, where the distribution is concentrated on a
 * line: Φ(min(h, k)) at 1 and max(0, Φ(h) + Φ(k) - 1) at -1. The function assumes valid numbers, no NaN.
 *
 * @see {@link https://doi.org/10.1023/B:STCO.0000035301.49549.88 | Genz (2004)}
 * @see {@link https://doi.org/10.1080/00949658908811121 | Drezner and Wesolowsky (1990)}
 */
export function bivariateNormalCdf(h: number, k: number, rho: number): number {
  if (h === Number.NEGATIVE_INFINITY || k === Number.NEGATIVE_INFINITY) return 0;
  if (h === Number.POSITIVE_INFINITY) return normalCdf(k);
  if (k === Number.POSITIVE_INFINITY) return normalCdf(h);
  if (rho === 0) return normalCdf(h) * normalCdf(k);
  if (rho >= 1) return normalCdf(Math.min(h, k));
  if (rho <= -1) return Math.max(0, normalCdf(h) + normalCdf(k) - 1);
  return upperTail(-h, -k, rho);
}

/**
 * Returns the density of the bivariate normal distribution of standard normal X and Y with correlation `rho`, from -1
 * to 1 exclusive, at (x, y). The distribution function Φ₂ has this density as its derivative with respect to ρ.
 */
export function bivariateNormalDensity(x: number, y: number, rho: number): number {
  const complement = 1 - rho * rho;
  const quadratic = x * x - 2 * rho * x * y + y * y;
  return Math.exp(-quadratic / (2 * complement)) / (TWO_PI * Math.sqrt(complement));
}

/**
 * Returns ∂/∂ρ of the bivariate normal density at (x, y) divided by the density: [ρ (1 - ρ²) + (1 + ρ²) x y - ρ (x² +
 * y²)] / (1 - ρ²)². The second derivative of Φ₂ with respect to ρ is the density times this.
 */
export function bivariateNormalLogDensitySlope(x: number, y: number, rho: number): number {
  const complement = 1 - rho * rho;
  return (rho * complement + (1 + rho * rho) * x * y - rho * (x * x + y * y)) / (complement * complement);
}
