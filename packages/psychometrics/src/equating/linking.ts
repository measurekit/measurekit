import { checkArray, checkFinite, checkPositive, checkRecord, DataError, describe, isRecord } from "@measurekit/core";
import { checkItem } from "../irt/items.js";
import type { IrtItem } from "../irt/model.js";
import { mean, standardDeviation } from "../math/statistics.js";
import { itemMoments, rescaledErrors, rescaledItem } from "./models.js";

/**
 * A linear transformation from one theta scale to another, θ_to = slope · θ_from + intercept: A and B of the linking
 * literature.
 */
export interface ScaleTransformation {
  /** The slope A. */
  readonly slope: number;
  /** The intercept B. */
  readonly intercept: number;
}

/**
 * Returns the mean-mean transformation of the common items' scale in `from` to their scale in `to`, `from[j]` and
 * `to[j]` being the same item.
 *
 * @remarks
 * A = mean a_from / mean a_to and B = mean b_to - A mean b_from, over the slopes of the items and their locations, the
 * difficulty or every threshold. For Rasch items, a = 1 on both scales, A is 1.
 *
 * @throws `DataError` when the item lists differ in length or are empty, a pair has different models or numbers of
 * categories, an item is invalid, an NRM item or not one of the built-in models, or a mean slope is not positive.
 * @see {@link https://doi.org/10.1111/j.1745-3984.1980.tb00825.x | Loyd and Hoover (1980)}
 */
export function meanMeanLinking(from: readonly IrtItem[], to: readonly IrtItem[]): ScaleTransformation {
  const [source, target] = commonMoments(from, to);
  if (!(mean(source.slopes) > 0 && mean(target.slopes) > 0)) {
    throw new DataError("Mean-mean linking needs common items with a positive mean slope on both scales.");
  }
  const slope = mean(source.slopes) / mean(target.slopes);
  return { slope, intercept: mean(target.locations) - slope * mean(source.locations) };
}

/**
 * Returns the mean-sigma transformation of the common items' scale in `from` to their scale in `to`: A = sd b_to / sd
 * b_from and B = mean b_to - A mean b_from, over the items' locations, the difficulty or every threshold.
 *
 * @throws `DataError` in the cases of {@link meanMeanLinking} other than the slopes, or when the locations on either
 * scale have no variance.
 * @see {@link https://doi.org/10.1111/j.1745-3984.1977.tb00033.x | Marco (1977)}
 */
export function meanSigmaLinking(from: readonly IrtItem[], to: readonly IrtItem[]): ScaleTransformation {
  const [source, target] = commonMoments(from, to);
  const sourceDeviation = source.locations.length >= 2 ? standardDeviation(source.locations) : 0;
  const targetDeviation = target.locations.length >= 2 ? standardDeviation(target.locations) : 0;
  if (!(sourceDeviation > 0 && targetDeviation > 0)) {
    throw new DataError("Mean-sigma linking needs common items whose locations vary on both scales.");
  }
  const slope = targetDeviation / sourceDeviation;
  return { slope, intercept: mean(target.locations) - slope * mean(source.locations) };
}

/**
 * Returns an item with its parameters on the new scale of a transformation.
 *
 * @remarks
 * The item gives a person at slope · θ + intercept the probabilities it gave a person at θ. Other fields, such as `id`,
 * are kept; standard errors by parameter name, as a calibration gives them, scale with their parameters, treating the
 * transformation as known: a by 1 / A, locations and steps by A. An NRM item moved by an intercept other than 0 has
 * `null` standard errors.
 *
 * @throws `DataError` when the transformation is invalid, or the item is invalid or not one of the built-in models.
 */
export function transformItem<TItem extends IrtItem>(item: TItem, transformation: ScaleTransformation): TItem {
  checkTransformation(transformation);
  const { slope, intercept } = transformation;
  const { parameters } = rescaledItem(item, slope, intercept, "The item");
  if (!("standardErrors" in item) || item.standardErrors === null || item.standardErrors === undefined) {
    return { ...item, parameters };
  }
  return {
    ...item,
    parameters,
    standardErrors: rescaledErrors(item, checkErrors(item.standardErrors), slope, intercept),
  };
}

/**
 * Checks standard errors by parameter name.
 *
 * @throws `DataError` unless they are an object of non-negative numbers.
 */
function checkErrors(errors: unknown): Readonly<Record<string, number>> {
  if (!isRecord(errors)) {
    throw new DataError("An item's standard errors must be an object of numbers by parameter name or null.");
  }
  const checked: Record<string, number> = {};
  for (const [name, error] of Object.entries(errors)) {
    if (!(typeof error === "number" && error >= 0)) {
      throw new DataError(`The standard error of "${name}" must be a non-negative number, found ${describe(error)}.`);
    }
    checked[name] = error;
  }
  return checked;
}

/**
 * Returns theta on the new scale of a transformation: slope · theta + intercept. A standard error of theta scales by
 * the slope.
 *
 * @throws `DataError` when theta is not finite or the transformation is invalid.
 */
export function transformTheta(theta: number, transformation: ScaleTransformation): number {
  checkFinite(theta, "Theta");
  checkTransformation(transformation);
  return transformation.slope * theta + transformation.intercept;
}

/**
 * Checks that the common items pair up: arrays of the same number of valid items on both scales, at least one, each
 * pair of the same model and number of categories.
 *
 * @throws `DataError` when they do not, or an item is invalid.
 */
export function checkCommonItems(from: readonly IrtItem[], to: readonly IrtItem[]): void {
  checkArray(from, "The common items on the from scale must be an array");
  checkArray(to, "The common items on the to scale must be an array");
  if (from.length === 0 || from.length !== to.length) {
    throw new DataError(
      `Linking needs the same common items on both scales, at least one; found ${from.length} and ${to.length}.`,
    );
  }
  from.forEach((item, j) => {
    const other = to[j];
    checkItem(item, `Common item ${j + 1} on the from scale`);
    checkItem(other, `Common item ${j + 1} on the to scale`);
    if (
      item.model !== other.model ||
      item.model.categoryCount(item.parameters) !== other.model.categoryCount(other.parameters)
    ) {
      throw new DataError(`Common item ${j + 1} needs the same model and number of categories on both scales.`);
    }
  });
}

/**
 * Checks that a transformation has a positive finite slope and a finite intercept.
 *
 * @throws `DataError` when it does not.
 */
function checkTransformation(transformation: ScaleTransformation): void {
  checkRecord(transformation, "A scale transformation must be an object { slope, intercept }");
  checkPositive(transformation.slope, "The slope of a scale transformation");
  checkFinite(transformation.intercept, "The intercept of a scale transformation");
}

/** The slopes and locations of the common items on one scale. */
interface Moments {
  readonly slopes: Float64Array;
  readonly locations: Float64Array;
}

function commonMoments(from: readonly IrtItem[], to: readonly IrtItem[]): [Moments, Moments] {
  checkCommonItems(from, to);
  const collect = (items: readonly IrtItem[], scale: string): Moments => {
    const moments = items.map((item, j) => itemMoments(item, `Common item ${j + 1} on the ${scale} scale`));
    return {
      slopes: Float64Array.from(moments, ({ slope }) => slope),
      locations: Float64Array.from(moments.flatMap(({ locations }) => locations)),
    };
  };
  return [collect(from, "from"), collect(to, "to")];
}
