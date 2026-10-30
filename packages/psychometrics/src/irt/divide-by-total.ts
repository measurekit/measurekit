// Divide-by-total models, the partial credit family and the nominal response model, give category k the probability
// exp(z_k) / Σ_j exp(z_j), with logits z_k linear in theta.

/**
 * Turns the logits at `values[k * stride]`, k < `count`, into probabilities in place. Subtracting the largest logit
 * keeps exp from overflowing.
 */
export function softmax(values: Float64Array, count: number, stride = 1): void {
  let largest = Number.NEGATIVE_INFINITY;
  for (let k = 0; k < count; k++) largest = Math.max(largest, values[k * stride] ?? 0);
  let total = 0;
  for (let k = 0; k < count; k++) {
    const weight = Math.exp((values[k * stride] ?? 0) - largest);
    values[k * stride] = weight;
    total += weight;
  }
  for (let k = 0; k < count; k++) values[k * stride] = (values[k * stride] ?? 0) / total;
}

/**
 * Turns the probabilities in `first` and the logit slopes s_k = dz_k/dθ in `second` into the derivatives of the
 * probabilities: dP_k/dθ = P_k (s_k - s̄) and d²P_k/dθ² = P_k ((s_k - s̄)² - V), with s̄ and V the mean and variance of
 * the slopes under the probabilities.
 */
export function divideByTotalThetaDerivatives(first: Float64Array, second: Float64Array, count: number): void {
  let mean = 0;
  for (let k = 0; k < count; k++) mean += (first[k] ?? 0) * (second[k] ?? 0);
  let variance = 0;
  for (let k = 0; k < count; k++) variance += (first[k] ?? 0) * ((second[k] ?? 0) - mean) ** 2;
  for (let k = 0; k < count; k++) {
    const probability = first[k] ?? 0;
    const deviation = (second[k] ?? 0) - mean;
    first[k] = probability * deviation;
    second[k] = probability * (deviation * deviation - variance);
  }
}
