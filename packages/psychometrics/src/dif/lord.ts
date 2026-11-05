import { checkBoolean, checkOptions, checkRecord, chiSquareUpperTail, DataError, describe } from "@measurekit/core";
import type { ResponseMatrix } from "../data/internal.js";
import type { ScaleTransformation } from "../equating/internal.js";
import { type Calibration, type IrtModel, isLogisticModel, type LogisticItem, modelsOf } from "../irt/internal.js";

import { invert } from "../math/internal.js";
import { checkGroups, type DifGroup } from "./groups.js";
import {
  LINKED_DIF_OPTIONS,
  type LinkedCalibrations,
  type LinkedDifOptions,
  linkedCalibrations,
  MINIMUM_LINK_ANCHORS,
} from "./linked-calibrations.js";
import { type DifAnalysis, purified, significant } from "./purification.js";

/** Lord's chi-square of one item. */
export interface LordChiSquare {
  /** The item id. */
  readonly id: string;
  /** (v_R - v_F)ᵀ (Σ_R + Σ_F)⁻¹ (v_R - v_F) of the item's estimated parameters v and their covariance matrices Σ. */
  readonly chiSquare: number | null;
  /**
   * The number of estimated parameters of the item: 2 for the 2PL, 3 for the 3PL, 4 for the 4PL, less the asymptotes
   * `fixedParameters` holds, so 2 for a 3PL item whose lower asymptote is held.
   */
  readonly degreesOfFreedom: number;
  /** The probability of a chi-square at least as large without DIF. */
  readonly significance: number | null;
}

/** Lord's chi-square of every item, with the calibrations it compares. */
export interface LordDif extends DifAnalysis<LordChiSquare> {
  /** The calibration of the reference group. */
  readonly reference: Calibration<LogisticItem>;
  /** The calibration of the focal group, on its own scale. */
  readonly focal: Calibration<LogisticItem>;
  /** The mean-sigma transformation of the focal scale to the reference scale, over the anchors of the final round. */
  readonly transformation: ScaleTransformation;
}

/**
 * Returns Lord's chi-square of every item, as difR's `difLord`, purified on request.
 *
 * @remarks
 * Each group is calibrated separately by `calibrate` with `model` and `options`; the focal group's estimates move to the reference
 * scale by the mean-sigma transformation over the anchors, a / A and A b + B, and their covariance matrices with them;
 * and each item's estimated parameters are compared. An asymptote `fixedParameters` holds is held at its value in both
 * groups and not compared, as difR's `c` holds the lower asymptotes of the 3PL; a and b may not be held, as the link
 * moves them. The anchors are all items, or with `purify` the items not flagged
 * by the round before, as difR's `purify`: each round links the scales again on the same two calibrations, and the
 * studied item, if it is not an anchor, is compared on the scale of the anchors. Purification stops when two rounds
 * flag the same items, when fewer than two items are left to link on, or after 10 rounds. Missing responses are
 * skipped, as in calibration.
 *
 * @param matrix - The responses of the persons, dichotomous.
 * @param groups - The group of each person, in the order of the matrix.
 * @param model - The model of every item, or one model per item, in item order: the 2PL, 3PL or 4PL.
 * @param options - The held parameters, priors and quadrature of the calibration of each group, its stopping rules and
 *   whether to purify the link.
 * @returns The calibration of each group, the transformation that links the focal group to the reference group, the
 *   chi-square of every item and the purification when it was asked for.
 * @throws `DataError` when the matrix or the groups are invalid, a group has no person, a model is not the 2PL, 3PL or
 * 4PL, `fixedParameters` is invalid for the items or holds a or b, `purify` is not a boolean, the locations of the
 * anchors do not vary in a group, or a group's calibration throws it.
 * @see {@link https://doi.org/10.4324/9780203056615 | Lord (1980)}
 */
export function lordChiSquare(
  matrix: ResponseMatrix,
  groups: readonly DifGroup[],
  model: IrtModel<LogisticItem> | readonly IrtModel<LogisticItem>[],
  options: LinkedDifOptions = {},
): LordDif {
  checkGroups(matrix, groups);
  checkOptions(options, LINKED_DIF_OPTIONS);
  const models = linkedModels(model, matrix.itemIds.length, "Lord's chi-square");
  const { purify = false, ...calibration } = options;
  checkBoolean(purify, "The option purify");
  const calibrations = linkedCalibrations(matrix, groups, models, calibration, "Lord's chi-square");
  const statistics = (anchors: Uint8Array): LordChiSquare[] => {
    const transformation = calibrations.link(anchors);
    return matrix.itemIds.map((id, item) => ({
      id,
      ...statistic(calibrations, item, transformation),
    }));
  };
  const analysis = purified(
    matrix.itemIds,
    purify,
    statistics,
    (item) => significant(item.significance),
    MINIMUM_LINK_ANCHORS,
  );
  const kept = new Set(analysis.purification?.anchors ?? matrix.itemIds);
  const transformation = calibrations.link(Uint8Array.from(matrix.itemIds, (id) => (kept.has(id) ? 1 : 0)));
  return { ...analysis, reference: calibrations.reference, focal: calibrations.focal, transformation };
}

/**
 * Checks that every model is the 2PL, 3PL or 4PL, and returns the model of each item.
 *
 * @param statistic - Names the statistic in error messages.
 * @throws `DataError` when a model is not one of them, or the number of models is not the number of items.
 */
export function linkedModels(
  model: IrtModel<LogisticItem> | readonly IrtModel<LogisticItem>[],
  itemCount: number,
  statistic: string,
): readonly IrtModel<LogisticItem>[] {
  return modelsOf(model, itemCount, (each) => {
    checkRecord(each, "The model must be the 2PL, 3PL or 4PL model");
    if (!isLogisticModel(each) || each.parameters(2).some((parameter) => parameter.shared === true)) {
      throw new DataError(`${statistic} needs the 2PL, 3PL or 4PL model, found the ${describe(each.name)} model.`);
    }
  });
}

function statistic(
  calibrations: LinkedCalibrations,
  item: number,
  transformation: ScaleTransformation,
): Omit<LordChiSquare, "id"> {
  const size = calibrations.estimated[item]?.length ?? 0;
  const undefinedStatistic = { chiSquare: null, degreesOfFreedom: size, significance: null };
  const estimates = calibrations.estimates(item, transformation);
  if (estimates === null) return undefinedStatistic;
  const { reference, focal, referenceCovariance, focalCovariance } = estimates;
  const sum = referenceCovariance.map((value, index) => value + (focalCovariance[index] ?? 0));
  const inverse = invert(sum, size);
  if (inverse === null) return undefinedStatistic;
  const difference = reference.map((value, j) => value - (focal[j] ?? 0));
  let chiSquare = 0;
  for (let j = 0; j < size; j++) {
    for (let k = 0; k < size; k++) {
      chiSquare += (difference[j] ?? 0) * (inverse[j * size + k] ?? 0) * (difference[k] ?? 0);
    }
  }
  return { chiSquare, degreesOfFreedom: size, significance: chiSquareUpperTail(chiSquare, size) };
}
