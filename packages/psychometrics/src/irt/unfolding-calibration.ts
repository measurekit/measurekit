import { checkChoice, checkOptions, checkStopping, DataError, type FieldNames } from "@measurekit/core";
import type { ResponseMatrix, ResponseRecords } from "../data/internal.js";
import { type Calibration, calibrateResponses } from "./calibration.js";
import { type CalibrationOptions, STANDARD_ERROR_METHODS, type StandardErrorMethod } from "./calibration-options.js";
import type { IrtModel } from "./model.js";
import { type QuadratureOptions, quadratureRule } from "./quadrature.js";
import {
  type CompressedResponses,
  categoryTallies,
  checkCategories,
  compressResponses,
  NO_RESPONSES,
  respondentsOf,
} from "./responses.js";
import { checkStandardErrorSize } from "./standard-error-size.js";
import { generalizedGradedUnfolding, type UnfoldingParameters, unitUnfoldingModel } from "./unfolding.js";
import { agreeCategory, unfoldingStarts } from "./unfolding-start.js";

/** Options of {@link calibrateUnfolding}. */
export interface UnfoldingCalibrationOptions {
  /**
   * The quadrature rule over theta.
   *
   * @defaultValue 61 rectangular points
   */
  readonly quadrature?: QuadratureOptions;
  /**
   * The method of the standard errors of the last stage: `"none"`, `"hessian"` or `"cross-product"`, as
   * `calibrate`'s.
   *
   * @defaultValue `"none"`
   */
  readonly standardErrors?: StandardErrorMethod;
  /**
   * Each stage's EM has converged when no estimated parameter changes by more than this in an iteration.
   *
   * @defaultValue `1e-4`
   */
  readonly tolerance?: number;
  /**
   * The most EM iterations of each stage.
   *
   * @defaultValue `500`
   */
  readonly maxIterations?: number;
}

const UNFOLDING_CALIBRATION_OPTIONS: FieldNames<UnfoldingCalibrationOptions> = {
  quadrature: true,
  standardErrors: true,
  tolerance: true,
  maxIterations: true,
};

/** A calibration of the generalized graded unfolding model, with the persons it used. */
export interface UnfoldingCalibration extends Calibration<UnfoldingParameters> {
  /**
   * The persons the calibration used: those with a response at or above an item's middle category, or its first agree
   * category when its number of categories is even. The others disagree with every item they answered, and are left
   * out.
   */
  readonly persons: number;
}

const DEFAULT_TOLERANCE = 1e-4;
const DEFAULT_MAX_ITERATIONS = 500;
/**
 * The most items of an unfolding calibration: its first stage shares the thresholds among the items, so each Newton step
 * of its M-step solves for the locations of every item at once, in time that grows with the cube of the items. On the
 * responses of two persons, the calibration takes about 12 s at 500 items and 77 s at 1,000; its starts, which decompose
 * the correlations of every pair of items, 0.3 s and 3 s.
 */
const MAX_ITEMS = 500;

/**
 * Calibrates the generalized graded unfolding model in the GGUM package's three stages, each starting from the last:
 * the graded unfolding model with a = 1 and thresholds shared by the items with the same number of categories, then
 * each item's thresholds with a = 1, then every parameter.
 *
 * @remarks
 * The first stage starts from Roberts and Laughlin's procedure: each location's distance from the most endorsed item
 * from the item's share of responses at or above its middle category, its sign from the item's loading on the leading
 * eigenvector of the pairwise item correlations, and the thresholds from the most endorsed item's share. Reflecting
 * theta and every b leaves the likelihood unchanged, so the starts' signs fix the orientation: the loadings sum to at
 * least 0. Each stage is the EM of {@link calibrate} with a standard normal theta. Persons who disagree with every item
 * they answered are left out, as the GGUM package and GGUM2004 leave them: they may sit beyond either end of the items.
 * The result's `converged` and `iterations` are the last stage's; a stage that does not converge starts the next.
 *
 * @param data - The responses of the persons, as a matrix or as records.
 * @param options - The quadrature, standard errors and stopping rules; each option has its default.
 * @returns The items with their parameters of {@link generalizedGradedUnfolding}, the group, the log-likelihood, AIC
 *   and BIC of the persons used, whether the last stage converged, its iterations, and the persons used.
 * @throws `DataError` when the data or an option is invalid, there are more than 500 items, or an item has responses in
 * fewer than two categories or none in a category below its highest.
 *
 * @see {@link https://doi.org/10.1177/01466216000241001 | Roberts, Donoghue and Laughlin (2000)}
 * @see {@link https://doi.org/10.1177/014662169602000305 | Roberts and Laughlin (1996)}
 */
export function calibrateUnfolding(
  data: ResponseMatrix | ResponseRecords,
  options: UnfoldingCalibrationOptions = {},
): UnfoldingCalibration {
  checkOptions(options, UNFOLDING_CALIBRATION_OPTIONS);
  const {
    quadrature,
    standardErrors = "none",
    tolerance = DEFAULT_TOLERANCE,
    maxIterations = DEFAULT_MAX_ITERATIONS,
  } = options;
  checkStopping(tolerance, maxIterations);
  checkChoice(standardErrors, "The option standardErrors", STANDARD_ERROR_METHODS);
  // Checks the quadrature before any stage runs.
  quadratureRule(quadrature);
  const all = compressResponses(data);
  if (all.itemIds.length > MAX_ITEMS) {
    throw new DataError(
      `Unfolding calibration takes at most ${MAX_ITEMS} items, as its first stage estimates the locations of every item at once; found ${all.itemIds.length}.`,
    );
  }
  const allTallies = categoryTallies(all);
  const categoryCounts = all.itemIds.map(
    (id, i) => checkCategories(id, generalizedGradedUnfolding, allTallies[i] ?? NO_RESPONSES).length,
  );
  // The last stage estimates a, b and a threshold for each category but the first, of every item.
  if (standardErrors !== "none") checkStandardErrorSize(categoryCounts.reduce((sum, count) => sum + count + 1, 0));
  const responses = withoutDisagreement(all, categoryCounts);
  const { respondents } = respondentsOf(responses);
  const proportions = keptProportions(responses, categoryCounts);

  const stage = (
    model: IrtModel<UnfoldingParameters> | IrtModel<UnfoldingParameters>[],
    starts: readonly UnfoldingParameters[],
    errors: StandardErrorMethod,
  ): Calibration<UnfoldingParameters> => {
    const stageOptions: CalibrationOptions<UnfoldingParameters> = {
      ...(quadrature !== undefined && { quadrature }),
      standardErrors: errors,
      tolerance,
      maxIterations,
    };
    return calibrateResponses(responses, model, stageOptions, starts).calibration;
  };
  const shared = new Map<number, IrtModel<UnfoldingParameters>>();
  const sharedModels = proportions.map(({ length }) => {
    const model = shared.get(length) ?? unitUnfoldingModel(true);
    shared.set(length, model);
    return model;
  });
  const first = stage(sharedModels, unfoldingStarts(responses, proportions), "none");
  const second = stage(unitUnfoldingModel(false), parametersOf(first), "none");
  const last = stage(generalizedGradedUnfolding, parametersOf(second), standardErrors);
  return { ...last, persons: respondents };
}

/**
 * Returns each item's share of the kept responses in each of its categories.
 *
 * @throws `DataError` when an item has no kept response in a category below its highest.
 */
function keptProportions(responses: CompressedResponses, categoryCounts: readonly number[]): Float64Array[] {
  const tallies = categoryTallies(responses);
  return responses.itemIds.map((id, i) => {
    const { counts } = tallies[i] ?? NO_RESPONSES;
    const categoryCount = categoryCounts[i] ?? 0;
    let empty = 0;
    while (counts.has(empty)) empty++;
    if (empty < categoryCount) {
      throw new DataError(
        `Item "${id}" has no response in category ${empty} from a person who agrees with an item; unfolding calibration leaves out the persons who disagree with every item they answered.`,
      );
    }
    let total = 0;
    for (const count of counts.values()) total += count;
    return Float64Array.from({ length: categoryCount }, (_, category) => (counts.get(category) ?? 0) / total);
  });
}

/** Returns the parameters of a calibration's items, in item order. */
function parametersOf(calibration: Calibration<UnfoldingParameters>): UnfoldingParameters[] {
  return calibration.items.map((item) => item.parameters);
}

/**
 * Returns the responses without those of the persons who disagree with every item they answered: whose every response
 * is below its item's agree category, given each item's number of categories. A person who gave an item its highest
 * response is kept, so some person is.
 */
function withoutDisagreement(responses: CompressedResponses, categoryCounts: readonly number[]): CompressedResponses {
  const { offsets, items, codes, times } = responses;
  const thresholds = categoryCounts.map(agreeCategory);
  const kept: number[] = [];
  const keptOffsets = new Int32Array(offsets.length);
  for (let person = 0; person < offsets.length - 1; person++) {
    const start = offsets[person] ?? 0;
    const end = offsets[person + 1] ?? 0;
    let agrees = false;
    for (let r = start; r < end && !agrees; r++) agrees = (codes[r] ?? 0) >= (thresholds[items[r] ?? 0] ?? 0);
    if (agrees) for (let r = start; r < end; r++) kept.push(r);
    keptOffsets[person + 1] = kept.length;
  }
  return {
    ...responses,
    offsets: keptOffsets,
    items: Int32Array.from(kept, (r) => items[r] ?? 0),
    codes: Int32Array.from(kept, (r) => codes[r] ?? 0),
    times: times === null ? null : Float64Array.from(kept, (r) => times[r] ?? 0),
  };
}
