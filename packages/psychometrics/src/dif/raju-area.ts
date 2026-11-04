import { checkBoolean, checkOptions, DataError, logistic, logLogistic, normalCdf } from "@measurekit/core";
import type { ResponseMatrix } from "../data/internal.js";
import type { ScaleTransformation } from "../equating/internal.js";
import { checkGroups, type DifGroup } from "./groups.js";
import {
  estimatedParameters,
  type LinkedCalibrations,
  linkedCalibrations,
  MINIMUM_LINK_ANCHORS,
} from "./linked-calibrations.js";
import { LORD_OPTIONS, type LordOptions, linkedModels } from "./lord.js";
import { type DifAnalysis, purified, significant } from "./purification.js";

/** Raju's areas between an item's curves in the two groups, with their z tests. */
export interface RajuArea {
  /** The item id. */
  readonly id: string;
  /** (d - c) (b_F - b_R) on the reference scale: positive when the item is harder for the focal group. */
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
 * Returns Raju's signed and unsigned areas between the curves of every item in the two groups, with their z tests, as
 * difR's `difRaju`, purified on request.
 *
 * @remarks
 * Each group is calibrated separately and linked as {@link lordChiSquare} does: `calibrate` with `options` and the
 * observed information, and the focal group's estimates moved to the reference scale by the mean-sigma transformation
 * over the anchors, a / A and A b + B. With Δb = b_F - b_R and Y = a_F a_R Δb / (a_F - a_R), the signed area of a 2PL
 * item is Δb and the unsigned area |2 (a_F - a_R) ln(1 + e^Y) / (a_F a_R) - Δb|, or |Δb| when the slopes are equal.
 * Their standard errors come from the covariance matrices of the estimates by the delta method. The significance is
 * that of the unsigned z, two-sided, and purification flags items whose significance is below 0.05. An item whose
 * asymptotes differ between the groups has an infinite unsigned area, so each item estimates its slope and location
 * alone: a 2PL item, or a 3PL or 4PL item whose asymptotes `fixedParameters` holds in both groups, as difR's `c` holds
 * the 3PL's, whose areas and standard errors are d - c times the 2PL's and whose z tests are the 2PL's. The areas are
 * `null` where a group's information matrix cannot be inverted, and the z tests where a standard error is 0.
 *
 * @param matrix - The responses of the persons, dichotomous.
 * @param groups - The group of each person, in the order of the matrix.
 * @param options - The model, the held parameters, the quadrature and stopping rules of each group's calibration and
 *   whether to purify the link.
 * @returns The areas of every item, and the purification when it was asked for.
 * @throws `DataError` when the matrix or the groups are invalid, a group has no person, a model is not the 2PL, 3PL or
 * 4PL, an item estimates a parameter other than its slope and location, `fixedParameters` is invalid for the items
 * or holds a or b, `purify` is not a boolean, the locations of the anchors do not vary in a group, or a group's
 * calibration throws it.
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
  const models = linkedModels(options.model, matrix.itemIds.length, "Raju's area");
  estimatedParameters(matrix.itemIds, models, options.fixedParameters, "Raju's area").forEach((names, item) => {
    if (names.join() !== AREA_PARAMETERS) {
      throw new DataError(
        `Raju's area needs items that estimate a and b alone, as the 2PL does, or the 3PL and 4PL with their asymptotes in the option fixedParameters; item "${matrix.itemIds[item]}" of the ${models[item]?.name} model estimates ${names.join(", ")}.`,
      );
    }
  });
  const { purify = false, ...calibration } = options;
  checkBoolean(purify, "The option purify");
  const calibrations = linkedCalibrations(matrix, groups, calibration, models, "Raju's area");
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

/** The estimated parameters of an item between whose curves the area is finite, joined by commas. */
const AREA_PARAMETERS = "a,b";

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
  const parameters = calibrations.reference.items[item]?.parameters;
  if (estimates === null || parameters === undefined) return UNDEFINED_AREA;
  // Curves c + (d - c) P(θ) of the same asymptotes enclose d - c times the area of P's.
  const height = parameters.d - parameters.c;
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
    signedArea: height * signedArea,
    unsignedArea: height * unsignedArea,
    signedZ: tested ? signedZ : null,
    unsignedZ: tested ? unsignedZ : null,
    significance: tested ? 2 * normalCdf(-Math.abs(unsignedZ)) : null,
  };
}
