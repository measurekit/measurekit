/** The most steps of regula falsi; the Illinois variant converges faster than linearly and needs far fewer. */
const MAX_ITERATIONS = 200;

/**
 * Returns a root of `f` between `lower` and `upper`, given `fLower = f(lower) > 0 > fUpper = f(upper)` or the reverse.
 * Uses the Illinois variant of regula falsi, which keeps the root bracketed, and stops when the bracket is narrower
 * than `tolerance`.
 *
 * @see {@link https://doi.org/10.1007/BF01934364 | Dowell and Jarratt (1971)}
 */
export function findRoot(
  f: (x: number) => number,
  lower: number,
  upper: number,
  fLower: number,
  fUpper: number,
  tolerance = 1e-12,
): number {
  let a = lower;
  let b = upper;
  let fa = fLower;
  let fb = fUpper;
  // The end kept in the previous step: -1 for a, 1 for b. Halving the value at an end kept twice is the Illinois step.
  let kept = 0;
  for (let iteration = 0; iteration < MAX_ITERATIONS && Math.abs(b - a) > tolerance; iteration++) {
    const c = (a * fb - b * fa) / (fb - fa);
    const fc = f(c);
    if (fc === 0) return c;
    if (fc > 0 === fb > 0) {
      b = c;
      fb = fc;
      if (kept === -1) fa /= 2;
      kept = -1;
    } else {
      a = c;
      fa = fc;
      if (kept === 1) fb /= 2;
      kept = 1;
    }
  }
  return (a * fb - b * fa) / (fb - fa);
}

/**
 * Returns the root of an increasing function `f` that changes sign, searching from `start`.
 *
 * @remarks
 * Steps of 1, 2, 4, ... away from it toward the root bracket the root, and {@link findRoot} refines it. Where `f` keeps
 * its sign up to the largest finite step, as when rounding keeps a limit from crossing 0, the root lies beyond every
 * finite number and the result is the furthest point searched.
 */
export function findIncreasingRoot(f: (x: number) => number, start: number): number {
  const value = f(start);
  if (value === 0) return start;
  const direction = value > 0 ? -1 : 1;
  let step = 1;
  let other = start + direction;
  let otherValue = f(other);
  while (otherValue * value > 0) {
    const next = start + direction * step * 2;
    if (!Number.isFinite(next)) return other;
    step *= 2;
    other = next;
    otherValue = f(other);
  }
  return otherValue === 0 ? other : findRoot(f, start, other, value, otherValue);
}
