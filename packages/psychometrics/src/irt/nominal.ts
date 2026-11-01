import { DataError, isRecord, softmax } from "@measurekit/core";
import { divideByTotalThetaDerivatives } from "./divide-by-total.js";
import type { IrtModel } from "./model.js";
import { formatList } from "./thresholds.js";

/**
 * Parameters of a nominal response item: a slope and an intercept per category, in mirt's IRT parameterization. Adding
 * a constant to every slope or every intercept leaves the probabilities unchanged; calibration reports both summing
 * to 0, as mirt does.
 */
export interface NominalParameters {
  /** The slope of each category: mirt's a1 to aK. */
  readonly a: readonly number[];
  /** The intercept of each category: mirt's c1 to cK. */
  readonly c: readonly number[];
}

/**
 * The nominal response model (NRM, Bock): P(k) ∝ exp(a_k θ + c_k) for unordered categories. This is mirt's `nominal`
 * item type with `IRTpars = TRUE`. The vector holds the slopes and the intercepts of all categories but the last,
 * named a1 to a(K-1) and c1 to c(K-1); the last ones are minus the sums of the others.
 */
export const nominalResponse: IrtModel<NominalParameters> = {
  name: "NRM",
  maxCategories: Number.POSITIVE_INFINITY,
  categoryCount: ({ a }) => a.length,
  parameters: (categoryCount) => [
    ...Array.from({ length: categoryCount - 1 }, (_, k) => ({ name: `a${k + 1}` })),
    ...Array.from({ length: categoryCount - 1 }, (_, k) => ({ name: `c${k + 1}` })),
  ],
  startingValues(proportions) {
    // The GPCM's start with a = 1: slopes 0 to K - 1 and intercepts ln p_k, both centered.
    const count = proportions.length;
    const logs = Array.from(proportions, Math.log);
    const meanLog = logs.reduce((sum, value) => sum + value, 0) / count;
    return {
      a: Array.from({ length: count }, (_, k) => k - (count - 1) / 2),
      c: logs.map((value) => value - meanLog),
    };
  },
  toVector(parameters) {
    const { a, c }: Partial<NominalParameters> = isRecord(parameters) ? parameters : {};
    const valid =
      Array.isArray(a) &&
      Array.isArray(c) &&
      a.length >= 2 &&
      a.length === c.length &&
      a.every(Number.isFinite) &&
      c.every(Number.isFinite);
    if (!valid) {
      throw new DataError(
        `An NRM item needs finite slopes a and intercepts c, one of each per category and at least two categories, found a = ${formatList(a)}, c = ${formatList(c)}.`,
      );
    }
    return Float64Array.from([...centered(a).slice(0, -1), ...centered(c).slice(0, -1)]);
  },
  fromVector(vector) {
    const size = vector.length / 2;
    return { a: completed(vector.subarray(0, size)), c: completed(vector.subarray(size)) };
  },
  probabilities({ a, c }, theta, out) {
    for (let k = 0; k < a.length; k++) out[k] = (a[k] ?? 0) * theta + (c[k] ?? 0);
    softmax(out, a.length);
  },
  thetaDerivatives({ a, c }, theta, first, second) {
    for (let k = 0; k < a.length; k++) {
      first[k] = (a[k] ?? 0) * theta + (c[k] ?? 0);
      second[k] = a[k] ?? 0;
    }
    softmax(first, a.length);
    divideByTotalThetaDerivatives(first, second, a.length);
  },
  parameterDerivatives({ a, c }, theta, out) {
    const count = a.length;
    const size = 2 * count - 2;
    const last = count - 1;
    for (let k = 0; k < count; k++) out[k * size] = (a[k] ?? 0) * theta + (c[k] ?? 0);
    softmax(out, count, size);
    // Entry m of the slopes and of the intercepts raises z_m by θ and 1 and lowers z_(K-1) by as much, so
    // dP_k/dc_m = P_k ([k = m] - [k = K - 1] - P_m + P_(K-1)) and dP_k/da_m = θ dP_k/dc_m. Column 0 holds the
    // probabilities until the derivatives with respect to a1 replace them last.
    const top = out[last * size] ?? 0;
    for (let m = count - 2; m >= 0; m--) {
      const shift = (out[m * size] ?? 0) - top;
      for (let k = 0; k < count; k++) {
        const derivative = (out[k * size] ?? 0) * ((k === m ? 1 : 0) - (k === last ? 1 : 0) - shift);
        out[k * size + m] = theta * derivative;
        out[k * size + last + m] = derivative;
      }
    }
  },
  information({ a, c }, theta) {
    // The variance of the slopes under the probabilities, by West's weighted update, which never goes below 0. The
    // slopes are taken from the first one: a variance does not change, and the first element then leaves a mean of 0
    // exactly, where a mean of the slopes themselves has a rounding error that, as slopes come closer, grows against
    // their variance.
    let largest = Number.NEGATIVE_INFINITY;
    for (let k = 0; k < a.length; k++) largest = Math.max(largest, (a[k] ?? 0) * theta + (c[k] ?? 0));
    const origin = a[0] ?? 0;
    let total = 0;
    let mean = 0;
    let squares = 0;
    for (let k = 0; k < a.length; k++) {
      const slope = (a[k] ?? 0) - origin;
      const weight = Math.exp((a[k] ?? 0) * theta + (c[k] ?? 0) - largest);
      // A weight that underflows adds nothing, and while the total is 0 it would make the mean 0 / 0.
      if (weight === 0) continue;
      total += weight;
      const deviation = slope - mean;
      mean += (deviation * weight) / total;
      squares += weight * deviation * (slope - mean);
    }
    return squares / total;
  },
};

/** Returns the values minus their mean. */
export function centered(values: readonly number[]): number[] {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  return values.map((value) => value - mean);
}

/** Returns the values followed by minus their sum. */
function completed(values: Float64Array): number[] {
  let last = 0;
  for (const value of values) last -= value;
  return [...values, last];
}
