/**
 * Turns log weights into probabilities in place and returns the log of the weights' sum: `-Infinity` when every weight
 * is 0, which leaves the values unchanged.
 *
 * @remarks
 * The largest log weight is subtracted before `exp`, so that weights far below the largest do not make every term
 * underflow.
 */
export function normalizeLogs(values: Float64Array): number {
  let largest = Number.NEGATIVE_INFINITY;
  for (const value of values) largest = Math.max(largest, value);
  if (largest === Number.NEGATIVE_INFINITY) return largest;
  let total = 0;
  for (let k = 0; k < values.length; k++) {
    const weight = Math.exp((values[k] ?? 0) - largest);
    values[k] = weight;
    total += weight;
  }
  for (let k = 0; k < values.length; k++) values[k] = (values[k] ?? 0) / total;
  return largest + Math.log(total);
}
