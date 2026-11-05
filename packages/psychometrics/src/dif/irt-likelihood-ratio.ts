import { checkBoolean, checkOptions, chiSquareUpperTail, DataError, type FieldNames } from "@measurekit/core";
import type { ResponseMatrix } from "../data/internal.js";
import {
  type Calibration,
  type CalibrationOptions,
  calibrate,
  checkHeldParameters,
  checkModel,
  type FixedParameters,
  type IrtModel,
  modelsOf,
  readFixedParameters,
} from "../irt/internal.js";
import { checkGroups, type DifGroup } from "./groups.js";
import { type DifAnalysis, type PurificationOptions, purified, significant } from "./purification.js";

/**
 * Options of {@link irtLikelihoodRatioDif}: the parameters held in both groups (`fixedParameters`), the
 * quadrature and stopping rules of the calibrations, and `purify`. The groups and their distributions of theta are the
 * test's own, priors would turn the likelihood ratio into a ratio of posteriors, and person weights would turn it into
 * a ratio of pseudo-likelihoods, which is not chi-square.
 */
export type IrtDifOptions = Omit<
  CalibrationOptions<unknown>,
  | "fixed"
  | "fixedCovariance"
  | "groups"
  | "latent"
  | "covariates"
  | "standardErrors"
  | "testlets"
  | "priors"
  | "itemPriors"
  | "personWeights"
> &
  PurificationOptions;

const IRT_DIF_OPTIONS: FieldNames<IrtDifOptions> = {
  fixedParameters: true,
  quadrature: true,
  tolerance: true,
  maxIterations: true,
  purify: true,
};

/** The likelihood-ratio test of one item. */
export interface IrtDifTest {
  /** The item id. */
  readonly id: string;
  /**
   * 2 (ln L with the item free in each group - ln L with it equal across groups); `null` when the item lacks a category
   * in a group or a fit did not converge.
   */
  readonly chiSquare: number | null;
  /** The entries of the item's parameter vector that its model does not share among items and that are not held. */
  readonly degreesOfFreedom: number;
  /** The probability of a chi-square at least as large without DIF. */
  readonly significance: number | null;
}

/** The likelihood-ratio tests of every item, with the constrained calibration they compare with. */
export interface IrtDif<TParameters> extends DifAnalysis<IrtDifTest> {
  /**
   * The calibration with every item equal across groups but those flagged before the last round, each of which is two
   * items, `<id>:reference` and `<id>:focal`, after the item.
   */
  readonly constrained: Calibration<TParameters>;
}

/** The fewest items: an item free in each group is linked to the reference scale through the others. */
const MINIMUM_ITEMS = 2;
/**
 * The fewest anchors at which each anchor is tested on the others: one more than the anchor that fixes the focal
 * group's mean and standard deviation while the studied anchor is free.
 */
const MINIMUM_ANCHORS = 2;
/** The distribution of theta in each group: the reference group's fixes the scale. */
const LATENT = { focal: "normal" } as const;

/**
 * Returns the IRT likelihood-ratio DIF test of every item, as mirt's `DIF(scheme = "drop")` on a multiple-group model,
 * for items of any model, purified on request.
 *
 * @remarks
 * The constrained model calibrates both groups on one scale with every item equal across them: the reference group
 * N(0, 1) and the focal group normal, its mean and standard deviation estimated with the items. An item is tested by
 * replacing it with two items, one answered by each group, and calibrating again: χ² = 2 (ln L₁ - ln L₀), with as many
 * degrees of freedom as the item has parameters its model does not share. A parameter `fixedParameters` holds, such as a
 * 3PL item's lower asymptote, is held at its value in both groups and in every calibration, as mirt's `est = FALSE`,
 * and is not counted. Missing responses are skipped, as in calibration. An item's test is `null` when a group lacks one
 * of its categories from 0 to its highest response, as its parameters would not be identified in that group, or when
 * one of the two fits did not converge.
 *
 * With `purify`, the items flagged at 0.05 stay free in both models of the next round: an item not flagged is tested
 * by freeing it, a flagged one by constraining it, each against the model of that round. Purification stops, with
 * `converged` false, before fewer than two items are left constrained, as the free item of a test is linked to the
 * reference scale through the others. Each round runs one calibration more than there are items.
 *
 * @param matrix - The responses of the persons.
 * @param groups - The group of each person, in the order of the matrix.
 * @param model - The model of every item, or one model per item, in item order.
 * @param options - The held parameters, the quadrature, which must be rectangular, and the stopping rules of the
 *   calibrations, and whether to purify.
 * @returns The test of every item, the purification when it was asked for, and the constrained calibration of the last
 *   round.
 * @throws `DataError` when the matrix or the groups are invalid, there are fewer than two items, `fixedParameters` is
 * invalid for the items, `purify` is not a boolean, a calibration option is invalid, or the constrained calibration
 * throws it, as for the Gauss-Hermite rule or an item without responses in a category below its highest.
 * @see {@link https://doi.org/10.1177/014662169802200403 | Kim and Cohen (1998)}
 */
export function irtLikelihoodRatioDif<TParameters>(
  matrix: ResponseMatrix,
  groups: readonly DifGroup[],
  model: IrtModel<TParameters> | readonly IrtModel<TParameters>[],
  options: IrtDifOptions = {},
): IrtDif<TParameters> {
  checkGroups(matrix, groups);
  checkOptions(options, IRT_DIF_OPTIONS);
  const { purify = false, fixedParameters = {}, quadrature, tolerance, maxIterations } = options;
  checkBoolean(purify, "The option purify");
  const { itemIds } = matrix;
  if (itemIds.length < MINIMUM_ITEMS) {
    throw new DataError(`IRT likelihood-ratio DIF needs at least ${MINIMUM_ITEMS} items, found ${itemIds.length}.`);
  }
  const models = modelsOf(model, itemIds.length, (each) => {
    checkModel(each, "The model must be an IRT model such as twoParameterLogistic");
  });
  const highest = highestResponses(matrix);
  const heldOfItem = readFixedParameters(fixedParameters, itemIds, new Set());
  const degrees = models.map((each, item) => {
    const id = itemIds[item] ?? "";
    const specifications = each.parameters((highest[item] ?? 0) + 1);
    const held = heldOfItem.get(id) ?? new Map<string, number>();
    checkHeldParameters(id, each.name, specifications, held);
    return specifications.filter(({ name, shared }) => shared !== true && !held.has(name)).length;
  });
  const testable = itemIds.map((_, item) => answersEveryCategory(matrix, groups, item, highest[item] ?? 0));
  const fit = (free: Uint8Array): Calibration<TParameters> =>
    calibrate(
      splitMatrix(matrix, groups, free),
      models.flatMap((each, item) => (free[item] === 1 ? [each, each] : [each])),
      {
        fixedParameters: splitHeld(fixedParameters, itemIds, free),
        quadrature,
        tolerance,
        maxIterations,
        groups,
        latent: LATENT,
      },
    );
  let constrained: Calibration<TParameters> | undefined;
  const statistics = (anchors: Uint8Array): IrtDifTest[] => {
    const free = anchors.map((anchor) => 1 - anchor);
    const base = fit(free);
    constrained = base;
    return itemIds.map((id, item) => {
      const degreesOfFreedom = degrees[item] ?? 0;
      const undefinedTest = { id, chiSquare: null, degreesOfFreedom, significance: null };
      if (!testable[item]) return undefinedTest;
      const toggled = free.slice();
      toggled[item] = 1 - (free[item] ?? 0);
      const other = fit(toggled);
      const [larger, smaller] = free[item] === 1 ? [base, other] : [other, base];
      if (!(larger.converged && smaller.converged)) return undefinedTest;
      const chiSquare = Math.max(0, 2 * (larger.logLikelihood - smaller.logLikelihood));
      return { id, chiSquare, degreesOfFreedom, significance: chiSquareUpperTail(chiSquare, degreesOfFreedom) };
    });
  };
  const analysis = purified(itemIds, purify, statistics, (item) => significant(item.significance), MINIMUM_ANCHORS);
  if (constrained === undefined) throw new DataError("IRT likelihood-ratio DIF ran no calibration.");
  return { ...analysis, constrained };
}

/** Returns each item's highest response, at least 1, as a matrix declares no more. */
function highestResponses({ itemIds, responses }: ResponseMatrix): Int32Array {
  const highest = new Int32Array(itemIds.length).fill(1);
  for (const row of responses) {
    row.forEach((response, item) => {
      if (response !== null && response > (highest[item] ?? 1)) highest[item] = response;
    });
  }
  return highest;
}

/** Returns whether each group has a response in every category of an item from 0 to its highest. */
function answersEveryCategory(
  { responses }: ResponseMatrix,
  groups: readonly DifGroup[],
  item: number,
  highest: number,
): boolean {
  const categories = highest + 1;
  // Category k of the reference group at k, of the focal group at categories + k.
  const seen = new Uint8Array(2 * categories);
  responses.forEach((row, person) => {
    const response = row[item] ?? null;
    if (response !== null) seen[(groups[person] === "focal" ? categories : 0) + response] = 1;
  });
  return seen.every((answered) => answered === 1);
}

/**
 * Returns the held parameters of the items of {@link splitMatrix}: an item marked 1 in `free` holds them in both
 * groups.
 */
function splitHeld(fixedParameters: FixedParameters, itemIds: readonly string[], free: Uint8Array): FixedParameters {
  return Object.fromEntries(
    itemIds.flatMap((id, item) => {
      // An own entry only, so that an item id such as "constructor" holds nothing it was not given.
      const held = Object.hasOwn(fixedParameters, id) ? fixedParameters[id] : undefined;
      if (held === undefined) return [];
      return free[item] === 1
        ? [
            [`${id}:reference`, held],
            [`${id}:focal`, held],
          ]
        : [[id, held]];
    }),
  );
}

/**
 * Returns the matrix with each item marked 1 in `free` replaced by two items, `<id>:reference` with the reference
 * group's responses and `<id>:focal` with the focal group's, `null` for the other group.
 */
function splitMatrix(matrix: ResponseMatrix, groups: readonly DifGroup[], free: Uint8Array): ResponseMatrix {
  return {
    personIds: matrix.personIds,
    itemIds: matrix.itemIds.flatMap((id, item) => (free[item] === 1 ? [`${id}:reference`, `${id}:focal`] : [id])),
    responses: matrix.responses.map((row, person) => {
      const focal = groups[person] === "focal";
      return row.flatMap((response, item) => {
        if (free[item] !== 1) return [response];
        return focal ? [null, response] : [response, null];
      });
    }),
  };
}
