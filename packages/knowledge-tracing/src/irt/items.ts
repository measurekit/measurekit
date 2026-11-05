import {
  checkRecord,
  DataError,
  describe,
  gradedProbabilities,
  hasOrderedThresholds,
  hasValidAsymptotes,
  logistic,
  logisticProbabilities,
  logLogistic,
  type NominalParameters,
  nominalProbabilities,
  partialCreditProbabilities,
  type ThresholdItem,
} from "@measurekit/core";

/**
 * An item's parameters in the IRT form of mirt and `@measurekit/psychometrics`, D = 1. A dichotomous item has
 * P(correct) = c + (d - c) logistic(a (θ - b)): the 1PL to 4PL. A polytomous item has categories 0 to K - 1: K - 1
 * thresholds, or a slope and an intercept per category.
 */
export type IrtItem = DichotomousItem | GradedItem | PartialCreditItem | NominalItem;

/** A dichotomous item: the 1PL to 4PL. */
export interface DichotomousItem {
  /** The discrimination. */
  readonly a: number;
  /** The difficulty. */
  readonly b: number;
  /**
   * The lower asymptote.
   *
   * @defaultValue `0`
   */
  readonly c?: number;
  /**
   * The upper asymptote.
   *
   * @defaultValue `1`
   */
  readonly d?: number;
}

/**
 * A graded response item (Samejima): P(category ≥ k) = logistic(a (θ - b_k)), with thresholds increasing when a > 0
 * and decreasing when a < 0.
 */
export interface GradedItem extends ThresholdItem {
  /** The model: `"graded"`. */
  readonly model: "graded";
}

/** A generalized partial credit item (Muraki): P(k) ∝ exp(Σ_{j ≤ k} a (θ - b_j)). */
export interface PartialCreditItem extends ThresholdItem {
  /** The model: `"gpcm"`. */
  readonly model: "gpcm";
}

/**
 * A nominal response item (Bock): P(k) ∝ exp(a_k θ + c_k), a slope and an intercept for each category, as
 * `nominalResponse` of `@measurekit/psychometrics` calibrates them and mirt's `nominal` item type gives them with
 * `IRTpars = TRUE`.
 */
export interface NominalItem extends NominalParameters {
  /** The model: `"nominal"`. */
  readonly model: "nominal";
}

/** Returns the number of categories of an item: 2 for a dichotomous one. */
export function categoryCount(item: IrtItem): number {
  if (!("model" in item)) return 2;
  return item.model === "nominal" ? item.a.length : item.b.length + 1;
}

/**
 * Writes into `out` the log-probability of category k at θ and its first and second derivatives with respect to θ.
 */
export function categoryTerms(item: IrtItem, k: number, theta: number, out: Float64Array): void {
  if (!("model" in item)) dichotomousTerms(item, k, theta, out);
  else if (item.model === "graded") gradedTerms(item, k, theta, out);
  else divideByTotalTerms(item, k, theta, out);
}

/**
 * Returns the expected outcome at θ: the expected category over K - 1, P(correct) for a dichotomous item. `terms` is
 * scratch space of at least 2 values for a dichotomous item, which callers evaluating many θ share.
 */
export function expectedOutcome(item: IrtItem, theta: number, terms: Float64Array): number {
  if (!("model" in item)) {
    const { a, b, c = 0, d = 1 } = item;
    logisticProbabilities(a * (theta - b), c, d, terms);
    return terms[1] ?? 0;
  }
  const count = categoryCount(item);
  const probabilities = scratchOf(count);
  if (item.model === "graded") gradedProbabilities(item.a, item.b, theta, probabilities);
  else if (item.model === "gpcm") partialCreditProbabilities(item.a, 0, item.b, theta, probabilities);
  else nominalProbabilities(item.a, item.c, theta, probabilities);
  let sum = 0;
  for (let k = 1; k < count; k++) sum += k * (probabilities[k] ?? 0);
  return sum / (count - 1);
}

/**
 * The logits, then the probabilities, of one polytomous item at a time, and the slopes of the logits beside them: the
 * calls are synchronous, so one pair of arrays serves them.
 */
let scratch = new Float64Array(0);
let slopeScratch = new Float64Array(0);

/** Returns the scratch space for `count` values, grown when it is too small. */
function scratchOf(count: number): Float64Array {
  if (scratch.length < count) scratch = new Float64Array(count);
  return scratch;
}

/** Returns the scratch space of the slopes for `count` values, grown when it is too small. */
function slopeScratchOf(count: number): Float64Array {
  if (slopeScratch.length < count) slopeScratch = new Float64Array(count);
  return slopeScratch;
}

/**
 * The probability of the outcome is c + (d - c) s for k = 1 and (1 - d) + (d - c) t for k = 0, s = logistic(z), t =
 * logistic(-z), z = a (θ - b): each from the tail it is near, so neither cancels to 0 where s rounds to 1. With no
 * asymptote below it, c = 0 for k = 1 or d = 1 for k = 0, the log-probability is ln(d - c) + ln logistic(±z) and its
 * derivatives a t, -a² s t (or -a s, -a² s t), finite for every θ.
 */
function dichotomousTerms(item: DichotomousItem, k: number, theta: number, out: Float64Array): void {
  const { a, b, c = 0, d = 1 } = item;
  const z = a * (theta - b);
  const s = logistic(z);
  const t = logistic(-z);
  const sign = k === 1 ? 1 : -1;
  const floor = k === 1 ? c : 1 - d;
  out[2] = -a * a * s * t;
  if (floor === 0) {
    out[0] = Math.log(d - c) + logLogistic(sign * z);
    out[1] = sign * a * (k === 1 ? t : s);
    return;
  }
  const q = floor + (d - c) * (k === 1 ? s : t);
  const first = (sign * (d - c) * a * s * t) / q;
  out[0] = Math.log(q);
  out[1] = first;
  out[2] = (sign * (d - c) * a * a * s * t * (t - s)) / q - first * first;
}

/**
 * P(k) = P*(k) - P*(k + 1), P*(j) = logistic(z_j), z_j = a (θ - b_j), P*(0) = 1 and P*(K) = 0, equals
 * logistic(z_k) logistic(-z_(k+1)) (1 - exp(z_(k+1) - z_k)). Its last factor does not depend on θ, so the
 * log-probability's derivatives are a (t_k - s_(k+1)) and -a² (s_k t_k + s_(k+1) t_(k+1)), s_j = logistic(z_j) and
 * t_j = logistic(-z_j): no difference of probabilities that round to 1.
 */
function gradedTerms(item: GradedItem, k: number, theta: number, out: Float64Array): void {
  const zOf = (j: number): number => {
    if (j === 0) return Number.POSITIVE_INFINITY;
    if (j > item.b.length) return Number.NEGATIVE_INFINITY;
    return item.a * (theta - (item.b[j - 1] ?? 0));
  };
  const upper = zOf(k);
  const lower = zOf(k + 1);
  const [su, tu, sl, tl] = [logistic(upper), logistic(-upper), logistic(lower), logistic(-lower)];
  out[0] = logLogistic(upper) + logLogistic(-lower) + Math.log(-Math.expm1(lower - upper));
  out[1] = item.a * (tu - sl);
  out[2] = -item.a * item.a * (su * tu + sl * tl);
}

/**
 * The divide-by-total models, the generalized partial credit and the nominal response model, give category k the
 * probability exp(z_k) / Σ_j exp(z_j), with logits z_j linear in θ of slopes s_j. ln P(k) = z_k - ln Σ exp(z_j), and
 * its derivatives are s_k - E[s] and -Var(s) under the probabilities. The variance is taken about the mean, in a second
 * pass, so that it stays at least 0 where the probabilities gather in one category.
 */
function divideByTotalTerms(item: PartialCreditItem | NominalItem, k: number, theta: number, out: Float64Array): void {
  const count = categoryCount(item);
  const z = scratchOf(count);
  const slopes = slopeScratchOf(count);
  writeLogits(item, theta, z, slopes);
  let top = Number.NEGATIVE_INFINITY;
  for (let j = 0; j < count; j++) top = Math.max(top, z[j] ?? 0);
  const logit = z[k] ?? 0;
  // The logits give way to their weights exp(z_j - top), which the second pass reads.
  let total = 0;
  let mean = 0;
  for (let j = 0; j < count; j++) {
    const weight = Math.exp((z[j] ?? 0) - top);
    z[j] = weight;
    total += weight;
    mean += weight * (slopes[j] ?? 0);
  }
  mean /= total;
  let variance = 0;
  for (let j = 0; j < count; j++) variance += (z[j] ?? 0) * ((slopes[j] ?? 0) - mean) ** 2;
  out[0] = logit - top - Math.log(total);
  out[1] = (slopes[k] ?? 0) - mean;
  out[2] = -variance / total;
}

/**
 * Writes the logits z_j of a divide-by-total item at θ and their slopes s_j = dz_j/dθ: Σ_{i ≤ j} a (θ - b_i) and j a
 * for the GPCM, a_j θ + c_j and a_j for the NRM.
 */
function writeLogits(
  item: PartialCreditItem | NominalItem,
  theta: number,
  z: Float64Array,
  slopes: Float64Array,
): void {
  if (item.model === "nominal") {
    for (let j = 0; j < item.a.length; j++) {
      z[j] = (item.a[j] ?? 0) * theta + (item.c[j] ?? 0);
      slopes[j] = item.a[j] ?? 0;
    }
    return;
  }
  z[0] = 0;
  slopes[0] = 0;
  for (let j = 1; j <= item.b.length; j++) {
    z[j] = (z[j - 1] ?? 0) + item.a * (theta - (item.b[j - 1] ?? 0));
    slopes[j] = j * item.a;
  }
}

/** Checks an item's parameters, throwing a `DataError` that names the item. */
export function checkItem(item: IrtItem, id: string): void {
  const label = `Item "${id}"`;
  checkRecord(item, `${label} must be an object`);
  if (!("model" in item)) checkDichotomous(item, label);
  else if (item.model === "nominal") checkNominal(item, label);
  else checkPolytomous(item, label);
}

function checkDichotomous({ a, b, c = 0, d = 1 }: DichotomousItem, label: string): void {
  if (!Number.isFinite(a)) throw new DataError(`${label} needs a, a finite number, found ${describe(a)}.`);
  if (!Number.isFinite(b)) throw new DataError(`${label} needs b, a finite number, found ${describe(b)}.`);
  if (!hasValidAsymptotes(c, d)) {
    throw new DataError(`${label} needs 0 ≤ c < d ≤ 1, found c = ${describe(c)} and d = ${describe(d)}.`);
  }
}

function checkNominal({ a, c }: NominalItem, label: string): void {
  const finite = (values: unknown): values is readonly number[] =>
    Array.isArray(values) && Array.from(values).every(Number.isFinite);
  if (!(finite(a) && finite(c) && a.length >= 2 && a.length === c.length)) {
    const shown = (values: unknown): string => (Array.isArray(values) ? `[${values.join(", ")}]` : describe(values));
    throw new DataError(
      `${label} is nominal, so needs a and c, arrays of finite numbers, a slope and an intercept for each of at least two categories, found a = ${shown(a)} and c = ${shown(c)}.`,
    );
  }
}

function checkPolytomous(item: GradedItem | PartialCreditItem, label: string): void {
  const { model } = item as { readonly model: unknown };
  if (model !== "graded" && model !== "gpcm") {
    throw new DataError(
      `${label} has the model ${describe(model)}; polytomous items are "graded", "gpcm" or "nominal".`,
    );
  }
  const { a, b } = item;
  if (!Number.isFinite(a)) throw new DataError(`${label} needs a, a finite number, found ${describe(a)}.`);
  if (!(Array.isArray(b) && b.length > 0 && Array.from(b).every(Number.isFinite))) {
    throw new DataError(`${label} needs b, a non-empty array of finite thresholds, found ${describe(b)}.`);
  }
  if (model === "graded" && !hasOrderedThresholds(a, b)) {
    throw new DataError(
      `${label} is graded, so needs thresholds b strictly increasing when a > 0 and decreasing when a < 0, found a = ${a}, b = ${b.join(", ")}.`,
    );
  }
}
