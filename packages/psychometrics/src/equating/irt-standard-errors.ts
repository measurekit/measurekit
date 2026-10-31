import { checkArray, checkChoice, checkFinite, checkOptions, checkRecord, DataError, describe } from "@measurekit/core";
import type { ParameterCovariance } from "../irt/calibration.js";
import { indexItems } from "../irt/items.js";
import type { IrtItem, TestItem } from "../irt/model.js";
import { type IrtObservedScoreOptions, observedScoreEquivalents, observedScoreSetup } from "./irt-observed-score.js";
import { trueScoreEquating } from "./true-score.js";

/** Options of {@link irtEquatingStandardErrors}. */
export interface IrtEquatingErrorOptions extends IrtObservedScoreOptions {
  /**
   * The equating whose standard errors to compute: `"true-score"` for {@link trueScoreEquating}, `"observed-score"` for
   * {@link irtObservedScoreEquating}, which also uses `populations` and `weight`.
   *
   * @defaultValue `"true-score"`
   */
  readonly method?: "true-score" | "observed-score";
}

/**
 * The step of the central differences, relative to a parameter of at least 1 in size: the cube root of the machine
 * epsilon, which balances truncation and rounding error.
 */
const DIFFERENCE_STEP = 6e-6;
/** The share of the room to a parameter's bound that a difference may use. */
const BOUND_ROOM = 0.5;
/** The negative variance, relative to the size of its terms, that rounding of a positive semidefinite matrix can give. */
const VARIANCE_ROUNDING = 1e-9;
const METHODS = ["true-score", "observed-score"];

/** Where one parameter of the covariance sits in the items of the forms. */
interface ParameterTarget {
  /** The form: 0 for the form to equate, 1 for the form equated to. */
  readonly form: 0 | 1;
  /** The index of the item in its form. */
  readonly item: number;
  /** The index of the parameter in the item's parameter vector. */
  readonly entry: number;
}

/**
 * Returns the standard error of the IRT equated score at each score of the form to equate, by the delta method.
 *
 * @remarks
 * An equated score e(x) is a function of the item parameters of the two forms. With g the gradient of e(x) with respect
 * to the parameters in the covariance and Σ their covariance matrix, the standard error is √(gᵀ Σ g). The gradient
 * comes from central differences. The covariance, as `calibrateWithCovariance` returns it, labels each
 * parameter by item id and name, and the parameters of the forms' items are matched to it by id and name; an item on
 * both forms, a common item, is one set of parameters, and must have the same parameters in both. The items need ids.
 * Parameters that the covariance lacks, such as those of fixed items, are known, and so are the populations of the
 * observed-score method. The standard errors are valid when both forms are on the scale of one calibration, as after a
 * concurrent calibration of both forms or a calibration of one form with the other's items fixed; the covariance of a
 * linking of two separate calibrations is not available.
 *
 * @param from - The items of the form to equate, with the ids of the covariance.
 * @param to - The items of the form to equate to.
 * @param covariance - The covariance of the item parameters.
 * @throws `DataError` when the options are invalid, an item list is not an array of items with distinct string ids, an
 * item on both forms differs between them, the covariance is not a square matrix of finite numbers with a label for each
 * row, a label names a parameter that the item's model does not estimate, no parameter of the covariance belongs to an
 * item of the forms, a parameter of the covariance lies on a bound of its range, such as c = 0, the covariance gives a
 * negative variance, or the equating throws for the items.
 * @see {@link https://doi.org/10.3102/10769986007003165 | Lord (1982)}
 * @see {@link https://doi.org/10.1177/01466216010251004 | Ogasawara (2001)}
 */
export function irtEquatingStandardErrors(
  from: readonly TestItem[],
  to: readonly TestItem[],
  covariance: ParameterCovariance,
  options: IrtEquatingErrorOptions = {},
): number[] {
  checkOptions(options);
  const { method = "true-score" } = options;
  checkChoice(method, "The option method", METHODS);
  const forms = [from, to] as const;
  const indices = forms.map((items) => indexItems(items));
  checkCommonItems(forms, indices);
  const labels = checkCovariance(covariance);
  const setup = method === "observed-score" ? observedScoreSetup(options) : null;
  const equate = (items: readonly [readonly IrtItem[], readonly IrtItem[]]): number[] =>
    setup === null ? trueScoreEquating(items[0], items[1]) : observedScoreEquivalents(items[0], items[1], setup);
  const targets = parameterTargets(forms, indices, labels);
  const used = targets.flatMap((found, k) => (found.length > 0 ? [k] : []));
  if (used.length === 0) {
    throw new DataError(
      "The covariance has no parameter of the items of the forms; its labels must name the items by id.",
    );
  }
  const scores = equate([from, to]).length;
  const gradient = new Float64Array(scores * used.length);
  used.forEach((k, column) => {
    const found = targets[k] ?? [];
    const step = differenceStep(forms, found);
    const above = equate(moved(forms, found, step));
    const below = equate(moved(forms, found, -step));
    for (let x = 0; x < scores; x++) {
      gradient[x * used.length + column] = ((above[x] ?? 0) - (below[x] ?? 0)) / (2 * step);
    }
  });
  return Array.from({ length: scores }, (_, x) => standardError(gradient, x, used, covariance.matrix));
}

/**
 * Checks that an item on both forms has the same model and parameters on both.
 *
 * @throws `DataError` when it differs.
 */
function checkCommonItems(
  forms: readonly [readonly TestItem[], readonly TestItem[]],
  indices: readonly Map<string, number>[],
): void {
  for (const [id, j] of indices[0] ?? []) {
    const k = indices[1]?.get(id);
    const first = forms[0][j];
    const second = k === undefined ? undefined : forms[1][k];
    if (first === undefined || second === undefined) continue;
    const vector = second.model.toVector(second.parameters);
    if (
      first.model !== second.model ||
      !first.model.toVector(first.parameters).every((value, i) => value === vector[i])
    ) {
      throw new DataError(
        `Item "${id}" is on both forms with different parameters; a common item has one set of parameters on the scale of one calibration.`,
      );
    }
  }
}

/**
 * Checks a covariance and returns its labels.
 *
 * @throws `DataError` when it is not an object with a label of two strings for each row and a square matrix of finite
 * numbers of that size.
 */
function checkCovariance(covariance: ParameterCovariance): readonly { readonly item: string; readonly name: string }[] {
  checkRecord(covariance, "The covariance must be an object { parameters, matrix }");
  const { parameters, matrix } = covariance;
  checkArray(parameters, "The parameters of the covariance must be an array of labels { item, name }");
  checkArray(matrix, "The matrix of the covariance must be an array of rows");
  if (matrix.length !== parameters.length) {
    throw new DataError(
      `The covariance needs a row for each of its ${parameters.length} parameters, found ${matrix.length}.`,
    );
  }
  parameters.forEach((label, k) => {
    checkRecord(label, `Parameter ${k + 1} of the covariance must be an object { item, name }`);
    if (!(typeof label.item === "string" && typeof label.name === "string")) {
      throw new DataError(
        `Parameter ${k + 1} of the covariance needs a string item and name, found ${describe(label.item)} and ${describe(label.name)}.`,
      );
    }
  });
  matrix.forEach((row, i) => {
    checkArray(row, `Row ${i + 1} of the covariance matrix must be an array of numbers`);
    if (row.length !== parameters.length) {
      throw new DataError(
        `The covariance matrix must be square: row ${i + 1} has ${row.length} entries for ${parameters.length} parameters.`,
      );
    }
    row.forEach((value, j) => {
      checkFinite(value, `Entry (${i + 1}, ${j + 1}) of the covariance matrix`);
    });
  });
  return parameters;
}

/**
 * Returns, for each parameter of the covariance, where it sits in the items of the forms: none if its item is on neither.
 *
 * @throws `DataError` when an item's model does not estimate the named parameter.
 */
function parameterTargets(
  forms: readonly [readonly TestItem[], readonly TestItem[]],
  indices: readonly Map<string, number>[],
  labels: readonly { readonly item: string; readonly name: string }[],
): ParameterTarget[][] {
  return labels.map(({ item: id, name }) => {
    const found: ParameterTarget[] = [];
    forms.forEach((items, form) => {
      const index = indices[form]?.get(id);
      const item = index === undefined ? undefined : items[index];
      if (index === undefined || item === undefined) return;
      const names = item.model.parameters(item.model.categoryCount(item.parameters)).map((parameter) => parameter.name);
      const entry = names.indexOf(name);
      if (entry < 0) {
        throw new DataError(
          `The covariance has the parameter "${name}" of item "${id}", which its ${item.model.name} model does not estimate; it estimates ${names.join(", ")}.`,
        );
      }
      found.push({ form: form === 0 ? 0 : 1, item: index, entry });
    });
    return found;
  });
}

/**
 * Returns the step of the central differences of a parameter, shrunk to stay inside the bounds of every copy of it.
 *
 * @throws `DataError` when a copy lies on a bound of its range, where the equating has no derivative to take.
 */
function differenceStep(
  forms: readonly [readonly TestItem[], readonly TestItem[]],
  found: readonly ParameterTarget[],
): number {
  let step = Infinity;
  for (const { form, item, entry } of found) {
    const target = forms[form][item];
    if (target === undefined) continue;
    const value = target.model.toVector(target.parameters)[entry] ?? 0;
    const parameter = target.model.parameters(target.model.categoryCount(target.parameters))[entry];
    const { lower = -Infinity, upper = Infinity } = parameter ?? {};
    const room = BOUND_ROOM * Math.min(value - lower, upper - value);
    if (!(room > 0)) {
      throw new DataError(
        `The parameter "${parameter?.name}" of item "${target.id}" is ${value}, on a bound of its range; the standard error needs it inside the range.`,
      );
    }
    step = Math.min(step, DIFFERENCE_STEP * Math.max(1, Math.abs(value)), room);
  }
  return step;
}

/** Returns the forms with a parameter, in each place it sits, moved by `step`. */
function moved(
  forms: readonly [readonly IrtItem[], readonly IrtItem[]],
  found: readonly ParameterTarget[],
  step: number,
): [IrtItem[], IrtItem[]] {
  const changed: [IrtItem[], IrtItem[]] = [[...forms[0]], [...forms[1]]];
  for (const { form, item, entry } of found) {
    const target = changed[form][item];
    if (target === undefined) continue;
    const vector = target.model.toVector(target.parameters);
    vector[entry] = (vector[entry] ?? 0) + step;
    changed[form][item] = { ...target, parameters: target.model.fromVector(vector) };
  }
  return changed;
}

/**
 * Returns √(gᵀ Σ g) for the gradient of the equated score at `score`, over the parameters `used` of the covariance.
 *
 * @throws `DataError` when the variance is negative beyond rounding.
 */
function standardError(
  gradient: Float64Array,
  score: number,
  used: readonly number[],
  matrix: readonly (readonly number[])[],
): number {
  const width = used.length;
  let variance = 0;
  let size = 0;
  for (let a = 0; a < width; a++) {
    const row = matrix[used[a] ?? 0];
    for (let b = 0; b < width; b++) {
      const term = (gradient[score * width + a] ?? 0) * (row?.[used[b] ?? 0] ?? 0) * (gradient[score * width + b] ?? 0);
      variance += term;
      size += Math.abs(term);
    }
  }
  if (variance < -VARIANCE_ROUNDING * size) {
    throw new DataError(
      `The covariance matrix is not positive semidefinite: the equated score of ${score} has a negative variance, ${variance}.`,
    );
  }
  return Math.sqrt(Math.max(0, variance));
}
