import { checkArray, checkChoice, checkOptions, checkRecord, DataError, type FieldNames } from "@measurekit/core";
import { type IrtItem, indexItems, type ParameterCovariance, type TestItem } from "../irt/internal.js";
import { curveLink, type LinkingOptions } from "./characteristic-curve.js";
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

const LINK_ERROR_OPTIONS: FieldNames<LinkErrorOptions> = { linking: true, quadrature: true };

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

/** The link of the common items of one scale to another, and the link of the same items a small step away. */
export interface Link {
  readonly transformation: ScaleTransformation;
  /**
   * Returns the link of the common items with parameters moved by a small step: the moment methods' own, and one Newton
   * step from the minimum for the characteristic curve methods, so that a central difference does not minimize again.
   */
  nearby(from: readonly IrtItem[], to: readonly IrtItem[]): ScaleTransformation;
}

/** Links the common items of one scale to another. */
export type Linker = (from: readonly IrtItem[], to: readonly IrtItem[]) => Link;

/**
 * Returns the standard errors and covariance of the transformation that links a chain of separately calibrated forms,
 * by the delta method.
 *
 * @remarks
 * Each form after the first is linked to the one before it through the items they share, by id, and the links compose
 * into the transformation of the last form to the first form's scale, as {@link chainTransformations}; two forms are a
 * chain of one link. The transformation is a function of the item parameters of every form, differentiated by central
 * differences in each parameter of each form's covariance, so that a form between two links counts once, as
 * `equateIRT::chainec` does; a characteristic curve link at a moved parameter is one Newton step from its minimum, the
 * implicit function's derivative, rather than a second minimization. Forms are calibrated apart, so the covariance of (A, B) is the sum over the forms of J Σ
 * Jᵀ, J the transformation's Jacobian in the form's parameters and Σ their covariance. A parameter that the covariance
 * lacks, such as one of a fixed item, is known; one of an item that no neighboring form shares does not enter a link.
 *
 * @param forms - The forms in order, each with its items and their covariance; at least two.
 * @param options - The linking method and the quadrature of the characteristic curve methods.
 * @returns The transformation of the last form to the first form's scale, with the standard errors and covariance of
 *   its slope and intercept.
 * @throws `DataError` when there are fewer than two forms, a form or the options are invalid, neighboring forms share
 * no item, no parameter of a covariance belongs to a common item, a parameter lies on a bound of its range, a linking
 * method throws for the common items, a characteristic curve criterion has no strict minimum at the link, or a
 * covariance gives a negative variance.
 * @see {@link https://doi.org/10.1177/01466216010251004 | Ogasawara (2001)}
 * @see {@link https://doi.org/10.18637/jss.v068.i07 | Battauz (2015)}
 */
export function linkCovariance(forms: readonly CalibratedForm[], options: LinkErrorOptions = {}): LinkCovariance {
  const link = linker(options, LINK_ERROR_OPTIONS);
  checkArray(forms, "The forms must be an array of calibrated forms { items, covariance }");
  if (forms.length < 2) throw new DataError(`A link needs at least two forms, found ${forms.length}.`);
  const checked = forms.map((form, k) => checkForm(form, `Form ${k + 1}`));
  // Pair k links form k + 1 to form k.
  const pairs = checked
    .slice(1)
    .map((form, k) => commonPositions(form, checked[k] ?? form, `forms ${k + 1} and ${k + 2}`));
  const items = checked.map((form) => form.items);
  const links = pairs.map((pair, k) => linkOf(items[k + 1] ?? [], items[k] ?? [], pair, link));
  const transformation = chainTransformations(links.map((base) => base.transformation).reverse());
  const covariance = [0, 0, 0];
  let parameters = 0;
  checked.forEach((form, f) => {
    const shared = new Set<number>([...(pairs[f - 1]?.from ?? []), ...(pairs[f]?.to ?? [])]);
    const targets = parameterTargets([form.items], [form.indices], form.labels);
    const used = targets.flatMap((found, k) => (found.some(({ item }) => shared.has(item)) ? [k] : []));
    parameters += used.length;
    const gradient = differenceGradient([form.items], targets, used, 2, (found, step) => {
      // A link depends only on the common items of its two forms, so a step that moves none of them leaves it as it
      // was: only the links of form f whose common items it moves are minimized again, one for most steps.
      const current = movedForm(items, f, found, step);
      const moves = (positions: readonly number[] = []) => found.some(({ item }) => positions.includes(item));
      const moved = links.map((base, k) => {
        const pair = pairs[k];
        const changed = pair !== undefined && ((k === f - 1 && moves(pair.from)) || (k === f && moves(pair.to)));
        return changed ? nearbyLink(base, current[k + 1] ?? [], current[k] ?? [], pair) : base.transformation;
      });
      const { slope, intercept } = chainTransformations(moved.reverse());
      return [slope, intercept];
    });
    const slopes = gradient.subarray(0, used.length);
    const intercepts = gradient.subarray(used.length);
    covariance[0] = (covariance[0] ?? 0) + checkedVariance(slopes, used, form.matrix, "the slope of the link");
    covariance[1] = (covariance[1] ?? 0) + bilinearForm(slopes, intercepts, used, form.matrix).value;
    covariance[2] = (covariance[2] ?? 0) + checkedVariance(intercepts, used, form.matrix, "the intercept of the link");
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
 * Checks the options against the names of the caller's, and returns the linking function they name.
 *
 * @throws `DataError` when the options are not an object or hold a name not in `names`, or the method is invalid.
 */
export function linker<TOptions extends LinkErrorOptions>(
  options: TOptions,
  names: FieldNames<NoInfer<TOptions>>,
): Linker {
  checkOptions(options, names);
  const { linking = "stocking-lord", quadrature } = options;
  checkChoice(linking, "The option linking", LINK_METHODS);
  const curves = quadrature === undefined ? {} : { quadrature };
  const moments =
    (method: (from: readonly IrtItem[], to: readonly IrtItem[]) => ScaleTransformation): Linker =>
    (from, to) => ({ transformation: method(from, to), nearby: method });
  switch (linking) {
    case "mean-mean":
      return moments(meanMeanLinking);
    case "mean-sigma":
      return moments(meanSigmaLinking);
    default:
      return (from, to) => {
        const { linking: transformation, nearby } = curveLink(from, to, curves, linking);
        return { transformation, nearby };
      };
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
export function linkOf(from: readonly IrtItem[], to: readonly IrtItem[], common: CommonPositions, link: Linker): Link {
  return link(pick(from, common.from), pick(to, common.to));
}

/** Returns the link of the common items of `from` and `to`, moved by a small step from those of `base`. */
export function nearbyLink(
  base: Link,
  from: readonly IrtItem[],
  to: readonly IrtItem[],
  common: CommonPositions,
): ScaleTransformation {
  return base.nearby(pick(from, common.from), pick(to, common.to));
}

/** Returns the items at the positions. */
function pick(items: readonly IrtItem[], positions: readonly number[]): IrtItem[] {
  return positions.flatMap((position) => {
    const item = items[position];
    return item === undefined ? [] : [item];
  });
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
