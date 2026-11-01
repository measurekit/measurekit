import {
  checkBoolean,
  checkOptions,
  checkRecord,
  chiSquareUpperTail,
  DataError,
  describe,
  type LogisticItem,
} from "@measurekit/core";
import type { ResponseMatrix } from "../data/internal.js";
import { meanSigmaLinking, type ScaleTransformation, transformItem } from "../equating/internal.js";
import {
  type Calibration,
  type CalibrationOptions,
  calibrateWithCovariance,
  type IrtItem,
  isLogisticModel,
  type ParameterCovariance,
} from "../irt/internal.js";

import { invert } from "../math/linear-algebra.js";
import { checkGroups, type DifGroup } from "./groups.js";
import { type DifAnalysis, type PurificationOptions, purified, significant } from "./purification.js";

/** The fewest anchors that link two scales: the mean-sigma transformation needs two locations to have a spread. */
const MINIMUM_ANCHORS = 2;

/**
 * Options of {@link lordChiSquare}: the calibration of each group, without fixed items, groups, distributions of theta,
 * covariates or the choice of standard errors, which the statistic needs, and `purify`.
 */
export type LordOptions = Omit<
  CalibrationOptions<LogisticItem>,
  "fixed" | "groups" | "latent" | "covariates" | "standardErrors"
> &
  PurificationOptions;

/** Lord's chi-square of one item. */
export interface LordChiSquare {
  /** The item id. */
  readonly id: string;
  /** (v_R - v_F)ᵀ (Σ_R + Σ_F)⁻¹ (v_R - v_F) of the item's estimated parameters v and their covariance matrices Σ. */
  readonly chiSquare: number | null;
  /** The number of estimated parameters of the item: 2 for the 2PL, 3 for the 3PL, 4 for the 4PL. */
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
 * Each group is calibrated separately by `calibrate` with `options`; the focal group's estimates move to the reference
 * scale by the mean-sigma transformation over the anchors, a / A and A b + B, and their covariance matrices with them;
 * and each item's estimated parameters are compared. The anchors are all items, or with `purify` the items not flagged
 * by the round before, as difR's `purify`: each round links the scales again on the same two calibrations, and the
 * studied item, if it is not an anchor, is compared on the scale of the anchors. Purification stops when two rounds
 * flag the same items, when fewer than two items are left to link on, or after 10 rounds. Missing responses are
 * skipped, as in calibration.
 *
 * @param matrix - The responses of the persons, dichotomous.
 * @param groups - The group of each person, in the order of the matrix.
 * @param options - The model, priors and quadrature of the calibration of each group, its stopping rules and whether to
 *   purify the link.
 * @returns The calibration of each group, the transformation that links the focal group to the reference group, the
 *   chi-square of every item and the purification when it was asked for.
 * @throws `DataError` when the matrix or the groups are invalid, a group has no person, a model is not the 2PL, 3PL or
 * 4PL, `purify` is not a boolean, the locations of the anchors do not vary in a group, or a group's calibration
 * throws it.
 * @see {@link https://doi.org/10.4324/9780203056615 | Lord (1980)}
 */
export function lordChiSquare(matrix: ResponseMatrix, groups: readonly DifGroup[], options: LordOptions): LordDif {
  checkGroups(matrix, groups);
  checkOptions(options);
  checkModels(options.model);
  const { purify = false, ...calibration } = options;
  checkBoolean(purify, "The option purify");
  const groupOptions = { ...calibration, standardErrors: "hessian" } as const;
  const reference = calibrateWithCovariance(groupMatrix(matrix, groups, "reference"), groupOptions);
  const focal = calibrateWithCovariance(groupMatrix(matrix, groups, "focal"), groupOptions);
  const referenceBlocks = itemBlocks(reference.covariance);
  const focalBlocks = itemBlocks(focal.covariance);
  const link = (anchors: Uint8Array): ScaleTransformation =>
    meanSigmaLinking(anchored(focal.calibration.items, anchors), anchored(reference.calibration.items, anchors));
  const statistics = (anchors: Uint8Array): LordChiSquare[] => {
    const transformation = link(anchors);
    return matrix.itemIds.map((id, i) => ({
      id,
      ...statistic(
        { item: reference.calibration.items[i], covariance: referenceBlocks.get(id) ?? null },
        { item: focal.calibration.items[i], covariance: focalBlocks.get(id) ?? null },
        transformation,
      ),
    }));
  };
  const analysis = purified(
    matrix.itemIds,
    purify,
    statistics,
    (item) => significant(item.significance),
    MINIMUM_ANCHORS,
  );
  const kept = new Set(analysis.purification?.anchors ?? matrix.itemIds);
  const transformation = link(Uint8Array.from(matrix.itemIds, (id) => (kept.has(id) ? 1 : 0)));
  return { ...analysis, reference: reference.calibration, focal: focal.calibration, transformation };
}

/** Checks that every model is the 2PL, 3PL or 4PL. */
function checkModels(models: LordOptions["model"]): void {
  for (const model of [models].flat()) {
    checkRecord(model, "The option model must be the 2PL, 3PL or 4PL model");
    if (!isLogisticModel(model) || model.parameters(2).some((parameter) => parameter.shared === true)) {
      throw new DataError(
        `Lord's chi-square needs the 2PL, 3PL or 4PL model, found the ${describe(model.name)} model.`,
      );
    }
  }
}

/** Returns the items marked 1 in `anchors`. */
function anchored<TItem>(items: readonly TItem[], anchors: Uint8Array): TItem[] {
  return items.filter((_, i) => anchors[i] === 1);
}

/** An item's estimates in one group, with their covariance matrix. */
interface GroupEstimate {
  readonly item: IrtItem<LogisticItem> | undefined;
  readonly covariance: Float64Array | null;
}

/** Returns the covariance matrix of each item's estimated parameters by rows, in the order of its parameter names. */
function itemBlocks(covariance: ParameterCovariance | null): Map<string, Float64Array> {
  const rows = new Map<string, number[]>();
  covariance?.parameters.forEach(({ item }, row) => {
    const own = rows.get(item);
    if (own === undefined) rows.set(item, [row]);
    else own.push(row);
  });
  return new Map(
    [...rows].map(([item, own]) => [
      item,
      Float64Array.from({ length: own.length ** 2 }, (_, index) => {
        const row = own[Math.floor(index / own.length)] ?? 0;
        return covariance?.matrix[row]?.[own[index % own.length] ?? 0] ?? 0;
      }),
    ]),
  );
}

function statistic(
  reference: GroupEstimate,
  focal: GroupEstimate,
  transformation: ScaleTransformation,
): Omit<LordChiSquare, "id"> {
  const model = reference.item?.model;
  const size = model?.parameters(2).length ?? 0;
  const undefinedStatistic = { chiSquare: null, degreesOfFreedom: size, significance: null };
  if (model === undefined || reference.item === undefined || focal.item === undefined) return undefinedStatistic;
  if (reference.covariance === null || focal.covariance === null) return undefinedStatistic;
  const { slope } = transformation;
  // a / A and A b + B scale the covariances of a and b by 1 / A and A; c and d do not change.
  const scales = [1 / slope, slope, 1, 1];
  const sum = Float64Array.from(
    reference.covariance,
    (value, index) =>
      value + (focal.covariance?.[index] ?? 0) * (scales[Math.floor(index / size)] ?? 1) * (scales[index % size] ?? 1),
  );
  const inverse = invert(sum, size);
  if (inverse === null) return undefinedStatistic;
  const referenceVector = model.toVector(reference.item.parameters);
  const focalVector = model.toVector(transformItem(focal.item, transformation).parameters);
  const difference = referenceVector.map((value, j) => value - (focalVector[j] ?? 0));
  let chiSquare = 0;
  for (let j = 0; j < size; j++) {
    for (let k = 0; k < size; k++) {
      chiSquare += (difference[j] ?? 0) * (inverse[j * size + k] ?? 0) * (difference[k] ?? 0);
    }
  }
  return { chiSquare, degreesOfFreedom: size, significance: chiSquareUpperTail(chiSquare, size) };
}

/**
 * Returns the response matrix of one group's persons.
 *
 * @throws `DataError` when the group has no person.
 */
function groupMatrix(matrix: ResponseMatrix, groups: readonly DifGroup[], group: DifGroup): ResponseMatrix {
  const persons = matrix.personIds.flatMap((_, person) => (groups[person] === group ? [person] : []));
  if (persons.length === 0) throw new DataError(`Lord's chi-square needs persons in the ${group} group.`);
  return {
    personIds: persons.map((person) => matrix.personIds[person] ?? ""),
    itemIds: matrix.itemIds,
    responses: persons.map((person) => matrix.responses[person] ?? []),
  };
}
