import { DataError, type LogisticItem } from "@measurekit/core";
import type { ResponseMatrix } from "../data/internal.js";
import { meanSigmaLinking, type ScaleTransformation, transformItem } from "../equating/internal.js";
import {
  type Calibration,
  type CalibrationOptions,
  calibrateWithCovariance,
  type ParameterCovariance,
} from "../irt/internal.js";
import type { DifGroup } from "./groups.js";

/** The fewest anchors that link two scales: the mean-sigma transformation needs two locations to have a spread. */
export const MINIMUM_LINK_ANCHORS = 2;

/** The calibration options of each group in the statistics that compare separate calibrations. */
type GroupCalibrationOptions = Omit<
  CalibrationOptions<LogisticItem>,
  "fixed" | "fixedParameters" | "groups" | "latent" | "covariates" | "standardErrors" | "personWeights"
>;

/** An item's estimated parameters in both groups, the focal group's on the reference scale, with their covariances. */
interface LinkedEstimates {
  /** The reference group's parameter vector, in the order of the model's parameters. */
  readonly reference: Float64Array;
  /** The focal group's parameter vector, moved to the reference scale. */
  readonly focal: Float64Array;
  /** The covariance matrix of the reference group's vector, by rows. */
  readonly referenceCovariance: Float64Array;
  /** The covariance matrix of the focal group's vector on the reference scale, by rows. */
  readonly focalCovariance: Float64Array;
}

/** The separate calibrations of the two groups, linked over anchors by mean-sigma. */
export interface LinkedCalibrations {
  /** The calibration of the reference group. */
  readonly reference: Calibration<LogisticItem>;
  /** The calibration of the focal group, on its own scale. */
  readonly focal: Calibration<LogisticItem>;
  /** Returns the mean-sigma transformation of the focal scale to the reference scale over the items marked 1. */
  link(anchors: Uint8Array): ScaleTransformation;
  /** Returns an item's estimates on the reference scale, or `null` when a group's covariance matrix lacks them. */
  estimates(item: number, transformation: ScaleTransformation): LinkedEstimates | null;
}

/**
 * Calibrates each group separately with the observed information, for the statistics that compare an item's estimates.
 *
 * @param statistic - Names the statistic in error messages.
 * @throws `DataError` when a group has no person, or a group's calibration throws it.
 */
export function linkedCalibrations(
  matrix: ResponseMatrix,
  groups: readonly DifGroup[],
  options: GroupCalibrationOptions,
  statistic: string,
): LinkedCalibrations {
  const groupOptions = { ...options, standardErrors: "hessian" } as const;
  const reference = calibrateWithCovariance(groupMatrix(matrix, groups, "reference", statistic), groupOptions);
  const focal = calibrateWithCovariance(groupMatrix(matrix, groups, "focal", statistic), groupOptions);
  const referenceBlocks = itemBlocks(reference.covariance);
  const focalBlocks = itemBlocks(focal.covariance);
  return {
    reference: reference.calibration,
    focal: focal.calibration,
    link: (anchors) =>
      meanSigmaLinking(anchored(focal.calibration.items, anchors), anchored(reference.calibration.items, anchors)),
    estimates: (item, transformation) => {
      const referenceItem = reference.calibration.items[item];
      const focalItem = focal.calibration.items[item];
      const referenceCovariance = referenceBlocks.get(matrix.itemIds[item] ?? "");
      const focalCovariance = focalBlocks.get(matrix.itemIds[item] ?? "");
      if (referenceItem === undefined || focalItem === undefined) return null;
      if (referenceCovariance === undefined || focalCovariance === undefined) return null;
      const { model } = referenceItem;
      const size = model.parameters(2).length;
      const { slope } = transformation;
      // a / A and A b + B scale the covariances of a and b by 1 / A and A; c and d do not change.
      const scales = [1 / slope, slope, 1, 1];
      return {
        reference: model.toVector(referenceItem.parameters),
        focal: model.toVector(transformItem(focalItem, transformation).parameters),
        referenceCovariance,
        focalCovariance: focalCovariance.map(
          (value, index) => value * (scales[Math.floor(index / size)] ?? 1) * (scales[index % size] ?? 1),
        ),
      };
    },
  };
}

/** Returns the items marked 1 in `anchors`. */
function anchored<TItem>(items: readonly TItem[], anchors: Uint8Array): TItem[] {
  return items.filter((_, i) => anchors[i] === 1);
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

/**
 * Returns the response matrix of one group's persons.
 *
 * @throws `DataError` when the group has no person.
 */
function groupMatrix(
  matrix: ResponseMatrix,
  groups: readonly DifGroup[],
  group: DifGroup,
  statistic: string,
): ResponseMatrix {
  const persons = matrix.personIds.flatMap((_, person) => (groups[person] === group ? [person] : []));
  if (persons.length === 0) throw new DataError(`${statistic} needs persons in the ${group} group.`);
  return {
    personIds: persons.map((person) => matrix.personIds[person] ?? ""),
    itemIds: matrix.itemIds,
    responses: persons.map((person) => matrix.responses[person] ?? []),
  };
}
