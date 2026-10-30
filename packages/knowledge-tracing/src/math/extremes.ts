/**
 * Returns the index of the largest value, the first on ties. A loop: spreading many values into `Math.max` overflows
 * the call stack, at about 120,000 values in V8.
 */
export function indexOfMax(values: ArrayLike<number>): number {
  let best = 0;
  for (let k = 1; k < values.length; k++) if ((values[k] ?? 0) > (values[best] ?? 0)) best = k;
  return best;
}

/** Returns the smallest value, by a loop for the reason of {@link indexOfMax}. */
export function minimum(values: ArrayLike<number>): number {
  let least = Number.POSITIVE_INFINITY;
  for (let k = 0; k < values.length; k++) least = Math.min(least, values[k] ?? Number.POSITIVE_INFINITY);
  return least;
}
