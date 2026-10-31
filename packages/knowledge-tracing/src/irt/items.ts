import { checkRecord, DataError, describe, isUnitInterval, logistic, logLogistic } from "@measurekit/core";
import { indexOfMax } from "../math/extremes.js";

/**
 * An item's parameters in the IRT form of mirt and `@measurekit/psychometrics`, D = 1. A dichotomous item has
 * P(correct) = c + (d - c) logistic(a (θ - b)): the 1PL to 4PL. A polytomous item has categories 0 to K - 1 and K - 1
 * thresholds.
 */
export type IrtItem = DichotomousItem | GradedItem | PartialCreditItem;

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
export interface GradedItem {
  /** The model: `"graded"`. */
  readonly model: "graded";
  /** The discrimination. */
  readonly a: number;
  /** The thresholds b_1 to b_{K-1}. */
  readonly b: readonly number[];
}

/** A generalized partial credit item (Muraki): P(k) ∝ exp(Σ_{j ≤ k} a (θ - b_j)). */
export interface PartialCreditItem {
  /** The model: `"gpcm"`. */
  readonly model: "gpcm";
  /** The discrimination. */
  readonly a: number;
  /** The step thresholds b_1 to b_{K-1}. */
  readonly b: readonly number[];
}

/** Returns the number of categories of an item: 2 for a dichotomous one. */
export function categoryCount(item: IrtItem): number {
  return "model" in item ? item.b.length + 1 : 2;
}

/**
 * Writes into `out` the log-probability of category k at θ and its first and second derivatives with respect to θ.
 */
export function categoryTerms(item: IrtItem, k: number, theta: number, out: Float64Array): void {
  if (!("model" in item)) dichotomousTerms(item, k, theta, out);
  else if (item.model === "graded") gradedTerms(item, k, theta, out);
  else partialCreditTerms(item, k, theta, out);
}

/**
 * Returns the log-probability of category k at θ: the first of the terms {@link categoryTerms} writes, without the
 * derivatives that a prediction does not use.
 */
function categoryLogProbability(item: IrtItem, k: number, theta: number, terms: Float64Array): number {
  if ("model" in item) {
    categoryTerms(item, k, theta, terms);
    return terms[0] ?? 0;
  }
  const { a, b, c = 0, d = 1 } = item;
  const z = a * (theta - b);
  const floor = k === 1 ? c : 1 - d;
  if (floor === 0) return Math.log(d - c) + logLogistic(k === 1 ? z : -z);
  return Math.log(floor + (d - c) * (k === 1 ? logistic(z) : logistic(-z)));
}

/**
 * Returns the expected outcome at θ: the expected category over K - 1, P(correct) for a dichotomous item. `terms` is
 * scratch space for a polytomous item's terms, which callers evaluating many θ share.
 */
export function expectedOutcome(item: IrtItem, theta: number, terms: Float64Array): number {
  const count = categoryCount(item);
  let sum = 0;
  for (let k = 1; k < count; k++) sum += k * Math.exp(categoryLogProbability(item, k, theta, terms));
  return sum / (count - 1);
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

function partialCreditTerms(item: PartialCreditItem, k: number, theta: number, out: Float64Array): void {
  // ln P(k) = z_k - ln Σ exp(z_j), z_j = Σ_{i ≤ j} a (θ - b_i); its derivatives are a (k - E[j]) and -a² Var(j).
  const count = item.b.length + 1;
  const z = new Float64Array(count);
  for (let j = 1; j < count; j++) z[j] = (z[j - 1] ?? 0) + item.a * (theta - (item.b[j - 1] ?? 0));
  const top = z[indexOfMax(z)] ?? 0;
  let total = 0;
  let mean = 0;
  let squares = 0;
  z.forEach((value, j) => {
    const weight = Math.exp(value - top);
    total += weight;
    mean += j * weight;
    squares += j * j * weight;
  });
  mean /= total;
  out[0] = (z[k] ?? 0) - top - Math.log(total);
  out[1] = item.a * (k - mean);
  out[2] = -item.a * item.a * (squares / total - mean * mean);
}

/** Checks an item's parameters, throwing a `DataError` that names the item. */
export function checkItem(item: IrtItem, id: string): void {
  const label = `Item "${id}"`;
  checkRecord(item, `${label} must be an object`);
  if (!Number.isFinite(item.a)) throw new DataError(`${label} needs a, a finite number, found ${describe(item.a)}.`);
  if ("model" in item) checkPolytomous(item, label);
  else checkDichotomous(item, label);
}

function checkDichotomous({ b, c = 0, d = 1 }: DichotomousItem, label: string): void {
  if (!Number.isFinite(b)) throw new DataError(`${label} needs b, a finite number, found ${describe(b)}.`);
  if (!(isUnitInterval(c) && isUnitInterval(d) && c < d)) {
    throw new DataError(`${label} needs 0 ≤ c < d ≤ 1, found c = ${describe(c)} and d = ${describe(d)}.`);
  }
}

function checkPolytomous(item: GradedItem | PartialCreditItem, label: string): void {
  const { model } = item as { readonly model: unknown };
  if (model !== "graded" && model !== "gpcm") {
    throw new DataError(`${label} has the model ${describe(model)}; polytomous items are "graded" or "gpcm".`);
  }
  const { a, b } = item;
  if (!(Array.isArray(b) && b.length > 0 && b.every(Number.isFinite))) {
    throw new DataError(`${label} needs b, a non-empty array of finite thresholds, found ${describe(b)}.`);
  }
  if (model === "graded" && !b.every((value, j) => j === 0 || a * (value - (b[j - 1] ?? 0)) > 0)) {
    throw new DataError(
      `${label} is graded, so needs thresholds b strictly increasing when a > 0 and decreasing when a < 0, found a = ${a}, b = ${b.join(", ")}.`,
    );
  }
}
