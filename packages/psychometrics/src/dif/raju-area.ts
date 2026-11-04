import {
  checkBoolean,
  checkOptions,
  checkRecord,
  chiSquareUpperTail,
  DataError,
  describe,
  logistic,
  logLogistic,
} from "@measurekit/core";
import type { ResponseMatrix } from "../data/internal.js";
import type { ScaleTransformation } from "../equating/internal.js";
import { twoParameterLogistic } from "../irt/internal.js";
import { checkGroups, type DifGroup } from "./groups.js";
import { type LinkedCalibrations, linkedCalibrations, MINIMUM_LINK_ANCHORS } from "./linked-calibrations.js";
import { LORD_OPTIONS, type LordOptions } from "./lord.js";
import { type DifAnalysis, purified, significant } from "./purification.js";

/** Raju's areas between an item's curves in the two groups, with their z tests. */
export interface RajuArea {
  /** The item id. */
  readonly id: string;
  /** b_F - b_R on the reference scale: positive when the item is harder for the focal group. */
  readonly signedArea: number | null;
  /** The area between the two curves, at least 0. */
  readonly unsignedArea: number | null;
  /** The signed area over its standard error. */
  readonly signedZ: number | null;
  /** The unsigned area over its standard error. */
  readonly unsignedZ: number | null;
  /** The two-sided probability of an unsigned z at least as large without DIF. */
  readonly significance: number | null;
}

/** The areas of an item whose calibrations give no covariance matrix. */
const UNDEFINED_AREA = {
  signedArea: null,
  unsignedArea: null,
  signedZ: null,
  unsignedZ: null,
  significance: null,
} as const;

/**
 * Returns Raju's signed and unsigned areas between the 2PL curves of every item in the two groups, with their z tests,
 * as difR's `difRaju`, purified on request.
 *
 * @remarks
 * Each group is calibrated separately and linked as {@link lordChiSquare} does: `calibrate` with `options` and the
 * observed information, and the focal group's estimates moved to the reference scale by the mean-sigma transformation
 * over the anchors, a / A and A b + B. With Δb = b_F - b_R and Y = a_F a_R Δb / (a_F - a_R), the signed area is Δb and
 * the unsigned area |2 (a_F - a_R) ln(1 + e^Y) / (a_F a_R) - Δb|, or |Δb| when the slopes are equal. Their standard
 * errors come from the covariance matrices of the estimates by the delta method. The significance is that of the
 * unsigned z, two-sided, and purification flags items whose significance is below 0.05. A 3PL item with unequal lower
 * asymptotes has an infinite unsigned area, so only the 2PL is taken. The areas are `null` where a group's information
 * matrix cannot be inverted, and the z tests where a standard error is 0.
 *
 * @param matrix - The responses of the persons, dichotomous.
 * @param groups - The group of each person, in the order of the matrix.
 * @param options - The model, the 2PL, the quadrature and stopping rules of each group's calibration and whether to
 *   purify the link.
 * @returns The areas of every item, and the purification when it was asked for.
 * @throws `DataError` when the matrix or the groups are invalid, a group has no person, a model is not the 2PL,
 * `purify` is not a boolean, the locations of the anchors do not vary in a group, or a group's calibration throws it.
 * @see {@link https://doi.org/10.1007/BF02294403 | Raju (1988)}
 * @see {@link https://doi.org/10.1177/014662169001400208 | Raju (1990)}
 */
export function rajuAreaDif(
  matrix: ResponseMatrix,
  groups: readonly DifGroup[],
  options: LordOptions,
): DifAnalysis<RajuArea> {
  checkGroups(matrix, groups);
  checkOptions(options, LORD_OPTIONS);
  checkModels(options.model);
  const { purify = false, ...calibration } = options;
  checkBoolean(purify, "The option purify");
  const calibrations = linkedCalibrations(matrix, groups, calibration, "Raju's area");
  return purified(
    matrix.itemIds,
    purify,
    (anchors) => {
      const transformation = calibrations.link(anchors);
      return matrix.itemIds.map((id, item) => ({ id, ...areas(calibrations, item, transformation) }));
    },
    (item) => significant(item.significance),
    MINIMUM_LINK_ANCHORS,
  );
}

/** Checks that every model is the 2PL. */
function checkModels(models: LordOptions["model"]): void {
  for (const model of [models].flat()) {
    checkRecord(model, "The option model must be the 2PL model");
    if (model !== twoParameterLogistic) {
      throw new DataError(`Raju's area needs the 2PL model, found the ${describe(model.name)} model.`);
    }
  }
}

/** The covariance matrix of a 2PL vector (a, b) by rows: its entries' variances and their covariance. */
interface Moments {
  readonly a: number;
  readonly b: number;
  readonly ab: number;
}

function moments(covariance: Float64Array): Moments {
  return { a: covariance[0] ?? 0, b: covariance[3] ?? 0, ab: covariance[1] ?? 0 };
}

function areas(
  calibrations: LinkedCalibrations,
  item: number,
  transformation: ScaleTransformation,
): Omit<RajuArea, "id"> {
  const estimates = calibrations.estimates(item, transformation);
  if (estimates === null) return UNDEFINED_AREA;
  const [referenceSlope = 0, referenceLocation = 0] = estimates.reference;
  const [focalSlope = 0, focalLocation = 0] = estimates.focal;
  const reference = moments(estimates.referenceCovariance);
  const focal = moments(estimates.focalCovariance);
  const signedArea = focalLocation - referenceLocation;
  const signedError = Math.sqrt(focal.b + reference.b);
  const y = (focalSlope * referenceSlope * signedArea) / (focalSlope - referenceSlope);
  // Equal slopes, or slopes so close that Y overflows, leave parallel curves: the unsigned area is |Δb|.
  let area = signedArea;
  let unsignedError = signedError;
  if (Number.isFinite(y)) {
    const softplus = -logLogistic(-y);
    const share = logistic(y);
    // The partial derivatives of the area before its absolute value, by b_F, b_R, a_F and a_R.
    const byFocalLocation = 2 * share - 1;
    const byFocalSlope = (2 * (softplus - y * share)) / focalSlope ** 2;
    const bySlope = -(focalSlope ** 2 / referenceSlope ** 2) * byFocalSlope;
    area = (2 * (focalSlope - referenceSlope) * softplus) / (focalSlope * referenceSlope) - signedArea;
    unsignedError = Math.sqrt(
      byFocalLocation ** 2 * (focal.b + reference.b) +
        byFocalSlope ** 2 * focal.a +
        bySlope ** 2 * reference.a +
        2 * byFocalLocation * byFocalSlope * focal.ab -
        2 * byFocalLocation * bySlope * reference.ab,
    );
  }
  const unsignedArea = Math.abs(area);
  const signedZ = signedArea / signedError;
  const unsignedZ = unsignedArea / unsignedError;
  if (!(Number.isFinite(signedArea) && Number.isFinite(unsignedArea))) return UNDEFINED_AREA;
  const tested = Number.isFinite(signedZ) && Number.isFinite(unsignedZ);
  return {
    signedArea,
    unsignedArea,
    signedZ: tested ? signedZ : null,
    unsignedZ: tested ? unsignedZ : null,
    significance: tested ? chiSquareUpperTail(unsignedZ ** 2, 1) : null,
  };
}
