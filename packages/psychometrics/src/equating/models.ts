import { DataError } from "@measurekit/core";
import { gradedResponse } from "../irt/graded.js";
import { checkItem } from "../irt/items.js";
import {
  fourParameterLogistic,
  oneParameterLogistic,
  threeParameterLogistic,
  twoParameterLogistic,
} from "../irt/logistic.js";
import type { IrtItem, IrtModel } from "../irt/model.js";
import { nominalResponse } from "../irt/nominal.js";
import { generalizedPartialCredit, partialCredit } from "../irt/partial-credit.js";
import { ratingScale } from "../irt/rating-scale.js";

// Each built-in model's parameter vector changes with the theta scale in one of four ways. Under θ' = A θ + B:
// logistic, a, b, c, d → a / A, A b + B, c, d; threshold, a, b_k → a / A, A b_k + B; rating scale, a, b, τ_k → a / A,
// A b + B, A τ_k; nominal, slopes a_k and intercepts c_k → a_k / A, c_k - a_k B / A.
type ScaleKind = "logistic" | "threshold" | "rating-scale" | "nominal";

const KINDS = new Map<IrtModel, ScaleKind>([
  [oneParameterLogistic, "logistic"],
  [twoParameterLogistic, "logistic"],
  [threeParameterLogistic, "logistic"],
  [fourParameterLogistic, "logistic"],
  [gradedResponse, "threshold"],
  [generalizedPartialCredit, "threshold"],
  [partialCredit, "threshold"],
  [ratingScale, "rating-scale"],
  [nominalResponse, "nominal"],
]);

/**
 * Returns how an item's parameters change with the theta scale.
 *
 * @throws `DataError` when the item's parameters are invalid or its model is not one of the built-in models.
 */
function scaleKind(item: IrtItem, label: string): ScaleKind {
  checkItem(item, label);
  const kind = KINDS.get(item.model);
  if (kind === undefined) {
    throw new DataError(`${label} has the ${item.model.name} model; linking supports the built-in models only.`);
  }
  return kind;
}

/**
 * Returns an item with the parameters that give the same probabilities on the scale θ' = slope θ + intercept.
 *
 * @throws `DataError` when the item's parameters are invalid or its model is not one of the built-in models.
 */
export function rescaledItem<TParameters>(
  item: IrtItem<TParameters>,
  slope: number,
  intercept: number,
  label: string,
): IrtItem<TParameters> {
  const kind = scaleKind(item, label);
  const vector = item.model.toVector(item.parameters);
  if (kind === "nominal") rescaleNominal(vector, slope, intercept);
  else rescaleOrdered(vector, kind, slope, intercept);
  return { model: item.model, parameters: item.model.fromVector(vector) };
}

/**
 * Returns the standard errors of an item's estimated parameters, by name, on the scale θ' = slope θ + intercept,
 * treating slope and intercept as known: each scales as its parameter, a by 1 / A and locations and steps by A. The
 * intercepts c_k - a_k B / A of an NRM item need the covariances of a_k and c_k, so an NRM item moved by B ≠ 0
 * has none.
 *
 * @throws `DataError` when the item's parameters are invalid or its model is not one of the built-in models.
 */
export function rescaledErrors(
  item: IrtItem,
  errors: Readonly<Record<string, number>>,
  slope: number,
  intercept: number,
): Record<string, number> | null {
  const kind = scaleKind(item, "The item");
  if (kind === "nominal" && intercept !== 0) return null;
  const names = item.model.parameters(item.model.categoryCount(item.parameters)).map((parameter) => parameter.name);
  const half = names.length / 2;
  return Object.fromEntries(
    Object.entries(errors).map(([name, error]) => {
      const position = names.indexOf(name);
      const isSlope = kind === "nominal" ? position < half : position === 0;
      // The asymptotes of a logistic item and the intercepts of an NRM item moved by B = 0 do not change.
      const unchanged = (kind === "logistic" && position > 1) || (kind === "nominal" && !isSlope);
      if (isSlope) return [name, error / slope];
      return [name, unchanged ? error : error * slope];
    }),
  );
}

/** Rescales an NRM vector, slopes a_k then intercepts c_k: a_k / A and c_k - a_k B / A. */
function rescaleNominal(vector: Float64Array, slope: number, intercept: number): void {
  const half = vector.length / 2;
  for (let k = 0; k < half; k++) {
    vector[half + k] = (vector[half + k] ?? 0) - ((vector[k] ?? 0) * intercept) / slope;
    vector[k] = (vector[k] ?? 0) / slope;
  }
}

/** Rescales a vector that starts with the slope a, followed by its locations and, for the RSM, the steps. */
function rescaleOrdered(vector: Float64Array, kind: ScaleKind, slope: number, intercept: number): void {
  vector[0] = (vector[0] ?? 0) / slope;
  const locations = kind === "threshold" ? vector.length : 2;
  for (let j = 1; j < locations; j++) vector[j] = slope * (vector[j] ?? 0) + intercept;
  if (kind !== "rating-scale") return;
  for (let j = 2; j < vector.length; j++) vector[j] = slope * (vector[j] ?? 0);
}

/** An item's slope and its locations, the difficulty or every threshold, which moment methods average. */
export interface ItemMoments {
  /** The slope a. */
  readonly slope: number;
  /** The difficulty b, or the thresholds: b_k, or b + τ_k for the RSM. */
  readonly locations: readonly number[];
}

/**
 * Returns an item's slope and locations.
 *
 * @throws `DataError` when the item's parameters are invalid or its model is not one of the built-in models other
 * than the NRM, which has no locations.
 */
export function itemMoments(item: IrtItem, label: string): ItemMoments {
  const kind = scaleKind(item, label);
  if (kind === "nominal") throw new DataError(`${label} is an NRM item, which has no slope and locations to average.`);
  const vector = item.model.toVector(item.parameters);
  const slope = vector[0] ?? 0;
  if (kind === "logistic") return { slope, locations: [vector[1] ?? 0] };
  if (kind === "threshold") return { slope, locations: Array.from(vector.subarray(1)) };
  const b = vector[1] ?? 0;
  const steps = Array.from(vector.subarray(2));
  steps.push(-steps.reduce((sum, step) => sum + step, 0));
  return { slope, locations: steps.map((step) => b + step) };
}

/**
 * Returns the limits of an item's expected score as theta goes to -∞ and to +∞: c and d for a logistic item, 0 and
 * K - 1 for the others.
 *
 * @throws `DataError` when the item's parameters are invalid, its model is not one of the built-in models, or its
 * expected score does not increase with theta: an NRM item, or a slope a ≤ 0.
 */
export function scoreLimits(item: IrtItem, label: string): readonly [number, number] {
  const kind = scaleKind(item, label);
  const vector = item.model.toVector(item.parameters);
  if (kind === "nominal" || !((vector[0] ?? 0) > 0)) {
    throw new DataError(
      `${label} needs an expected score that increases with theta: a model other than the NRM, a > 0.`,
    );
  }
  if (kind === "logistic") return [vector[2] ?? 0, vector[3] ?? 1];
  return [0, item.model.categoryCount(item.parameters) - 1];
}
