import { checkArray, checkChoice, checkOptions, checkRecord, DataError } from "@measurekit/core";
import { type IrtItem, indexItems, type ParameterCovariance, type TestItem } from "../irt/internal.js";
import { haebaraLinking, type LinkingOptions, stockingLordLinking } from "./characteristic-curve.js";
import {
  bilinearForm,
  checkCovariance,
  checkedVariance,
  differenceGradient,
  moved,
  type ParameterTarget,
  parameterTargets,
} from "./irt-standard-errors.js";
import { chainTransformations, meanMeanLinking, meanSigmaLinking, type ScaleTransformation } from "./linking.js";

/** The items of a form calibrated on its own, with the covariance of their estimates. */
export interface CalibratedForm {
  /** The items of the form, with the ids of the covariance. */
  readonly items: readonly TestItem[];
  /** The covariance of the item parameters, as `calibrateWithCovariance` returns it. */
  readonly covariance: ParameterCovariance;
}

/** A method of linking two scales through common items. */
export type LinkMethod = "mean-mean" | "mean-sigma" | "haebara" | "stocking-lord";

/** Options of {@link linkCovariance}. */
export interface LinkErrorOptions extends LinkingOptions {
  /**
   * The linking method: {@link meanMeanLinking}, {@link meanSigmaLinking}, {@link haebaraLinking} or
   * {@link stockingLordLinking}, whose `quadrature` the characteristic curve methods take.
   *
   * @defaultValue `"stocking-lord"`
   */
  readonly linking?: LinkMethod;
}

/** The standard errors of a transformation's slope and intercept. */
export interface TransformationErrors {
  /** The standard error of the slope A. */
  readonly slope: number;
  /** The standard error of the intercept B. */
  readonly intercept: number;
}

/** The sampling errors of a transformation. */
export interface LinkErrors {
  /** The standard errors of A and B. */
  readonly standardErrors: TransformationErrors;
  /** The covariance matrix of (A, B), by rows: var A, cov(A, B), cov(A, B), var B. */
  readonly covariance: readonly [number, number, number, number];
}

/** A transformation of a chain of separately calibrated forms with its sampling errors. */
export interface LinkCovariance {
  /** The composed transformation of the last form to the first form's scale. */
  readonly transformation: ScaleTransformation;
  /** The standard errors and the covariance of the transformation's slope and intercept. */
  readonly errors: LinkErrors;
}

const LINK_METHODS: readonly LinkMethod[] = ["mean-mean", "mean-sigma", "haebara", "stocking-lord"];

/** A form checked: its items, their positions by id, and the labels and matrix of its covariance. */
export interface CheckedForm {
  readonly items: readonly TestItem[];
  readonly indices: Map<string, number>;
  readonly labels: readonly { readonly item: string; readonly name: string }[];
  readonly matrix: readonly (readonly number[])[];
}

/** The link of the common items of one scale to another. */
export type Linker = (from: readonly IrtItem[], to: readonly IrtItem[]) => ScaleTransformation;

/**
 * Returns the standard errors and covariance of the transformation that links a chain of separately calibrated forms,
 * by the delta method.
 *
 * @remarks
 * Each form after the first is linked to the one before it through the items they share, by id, and the links compose
 * into the transformation of the last form to the first form's scale, as {@link chainTransformations}; two forms are a
 * chain of one link. The transformation is a function of the item parameters of every form, differentiated by central
 * differences in each parameter of each form's covariance, so that a form between two links counts once, as
 * `equateIRT::chainec` does. Forms are calibrated apart, so the covariance of (A, B) is the sum over the forms of J Σ
 * Jᵀ, J the transformation's Jacobian in the form's parameters and Σ their covariance. A parameter that the covariance
 * lacks, such as one of a fixed item, is known; one of an item that no neighboring form shares does not enter a link.
 *
 * @param forms - The forms in order, each with its items and their covariance; at least two.
 * @param options - The linking method and the quadrature of the characteristic curve methods.
 * @returns The transformation of the last form to the first form's scale, with the standard errors and covariance of
 *   its slope and intercept.
 * @throws `DataError` when there are fewer than two forms, a form or the options are invalid, neighboring forms share
 * no item, no parameter of a covariance belongs to a common item, a parameter lies on a bound of its range, a linking
 * method throws for the common items, or a covariance gives a negative variance.
 * @see {@link https://doi.org/10.1177/01466216010251004 | Ogasawara (2001)}
 * @see {@link https://doi.org/10.18637/jss.v068.i07 | Battauz (2015)}
 */
export function linkCovariance(forms: readonly CalibratedForm[], options: LinkErrorOptions = {}): LinkCovariance {
  const link = linker(options);
  checkArray(forms, "The forms must be an array of calibrated forms { items, covariance }");
  if (forms.length < 2) throw new DataError(`A link needs at least two forms, found ${forms.length}.`);
  const checked = forms.map((form, k) => checkForm(form, `Form ${k + 1}`));
  // Pair k links form k + 1 to form k.
  const pairs = checked
    .slice(1)
    .map((form, k) => commonPositions(form, checked[k] ?? form, `forms ${k + 1} and ${k + 2}`));
  const items = checked.map((form) => form.items);
  const chain = (current: readonly (readonly IrtItem[])[]): ScaleTransformation =>
    chainTransformations(pairs.map((pair, k) => linkOf(current[k + 1] ?? [], current[k] ?? [], pair, link)).reverse());
  const transformation = chain(items);
  const covariance = [0, 0, 0];
  let parameters = 0;
  checked.forEach((form, f) => {
    const shared = new Set<number>([...(pairs[f - 1]?.from ?? []), ...(pairs[f]?.to ?? [])]);
    const targets = parameterTargets([form.items], [form.indices], form.labels);
    const used = targets.flatMap((found, k) => (found.some(({ item }) => shared.has(item)) ? [k] : []));
    parameters += used.length;
    const gradient = differenceGradient([form.items], targets, used, 2, (found, step) => {
      const { slope, intercept } = chain(movedForm(items, f, found, step));
      return [slope, intercept];
    });
    const slopes = gradient.subarray(0, used.length);
    const intercepts = gradient.subarray(used.length);
    covariance[0] = (covariance[0] ?? 0) + checkedVariance(slopes, used, form.matrix, "the slope");
    covariance[1] = (covariance[1] ?? 0) + bilinearForm(slopes, intercepts, used, form.matrix).value;
    covariance[2] = (covariance[2] ?? 0) + checkedVariance(intercepts, used, form.matrix, "the intercept");
  });
  if (parameters === 0) {
    throw new DataError("No covariance has a parameter of a common item; their labels must name the items by id.");
  }
  const [slopeVariance = 0, product = 0, interceptVariance = 0] = covariance;
  return {
    transformation: { slope: transformation.slope, intercept: transformation.intercept },
    errors: {
      standardErrors: { slope: Math.sqrt(slopeVariance), intercept: Math.sqrt(interceptVariance) },
      covariance: [slopeVariance, product, product, interceptVariance],
    },
  };
}

/**
 * Checks the linking options and returns the linking function they name.
 *
 * @throws `DataError` when the options or the method are invalid.
 */
export function linker(options: LinkErrorOptions): Linker {
  checkOptions(options);
  const { linking = "stocking-lord", quadrature } = options;
  checkChoice(linking, "The option linking", LINK_METHODS);
  const curves = quadrature === undefined ? {} : { quadrature };
  switch (linking) {
    case "mean-mean":
      return meanMeanLinking;
    case "mean-sigma":
      return meanSigmaLinking;
    case "haebara":
      return (from, to) => haebaraLinking(from, to, curves);
    default:
      return (from, to) => stockingLordLinking(from, to, curves);
  }
}

/**
 * Checks a calibrated form.
 *
 * @throws `DataError` when it is not an object of items with distinct string ids and a covariance.
 */
export function checkForm(form: CalibratedForm, label: string): CheckedForm {
  checkRecord(form, `${label} must be an object { items, covariance }`);
  const indices = indexItems(form.items);
  const labels = checkCovariance(form.covariance);
  return { items: form.items, indices, labels, matrix: form.covariance.matrix };
}

/** The positions of the common items of two forms, in the order of the first. */
export interface CommonPositions {
  readonly from: readonly number[];
  readonly to: readonly number[];
}

/**
 * Returns the positions in each form of the items they share by id.
 *
 * @throws `DataError` when they share none.
 */
export function commonPositions(from: CheckedForm, to: CheckedForm, label: string): CommonPositions {
  const fromPositions: number[] = [];
  const toPositions: number[] = [];
  for (const [id, j] of from.indices) {
    const k = to.indices.get(id);
    if (k === undefined) continue;
    fromPositions.push(j);
    toPositions.push(k);
  }
  if (fromPositions.length === 0) throw new DataError(`The ${label} share no item by id; a link needs common items.`);
  return { from: fromPositions, to: toPositions };
}

/** Returns the link of the common items of `from` to their scale in `to`. */
export function linkOf(
  from: readonly IrtItem[],
  to: readonly IrtItem[],
  common: CommonPositions,
  link: Linker,
): ScaleTransformation {
  const pick = (items: readonly IrtItem[], positions: readonly number[]): IrtItem[] =>
    positions.flatMap((position) => {
      const item = items[position];
      return item === undefined ? [] : [item];
    });
  return link(pick(from, common.from), pick(to, common.to));
}

/** Returns the forms' items with a parameter of form `f`, in each place it sits, moved by `step`. */
export function movedForm<TItem extends IrtItem>(
  forms: readonly (readonly TItem[])[],
  f: number,
  found: readonly ParameterTarget[],
  step: number,
): (readonly TItem[])[] {
  return forms.map((items, g) => (g === f ? (moved([items], found, step)[0] ?? items) : items));
}
