/**
 * Returns the index of the largest value, the first on ties. A loop: spreading many values into `Math.max` overflows
 * the call stack, at about 120,000 values in V8.
 */
export function indexOfMax(values: ArrayLike<number>): number {
  let best = 0;
  for (let k = 1; k < values.length; k++) if ((values[k] ?? 0) > (values[best] ?? 0)) best = k;
  return best;
}
