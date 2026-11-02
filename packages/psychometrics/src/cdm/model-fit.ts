import { checkChoice, checkOptions, chiSquareUpperTail, DataError } from "@measurekit/core";
import { checkOpenUnitInterval } from "../checks.js";
import type { ResponseMatrix } from "../data/internal.js";
import {
  addPairDerivatives,
  addWeightDerivatives,
  limitedInformationStatistic,
  MAX_MOMENT_ITEMS,
  type MomentTables,
  modelMoments,
  type QuadraticForm,
  residualCorrelationRoot,
  rmseaInterval,
  rowCount,
  type SampleMoments,
  type ScoreMoments,
  sampleMoments,
} from "../math/internal.js";
import { type CdmCalibration, type CdmModel, calibratedModel, LINKS, MODELS, parameterCount } from "./calibration.js";
import type { CdmState } from "./expectation.js";
import { type CdmDesign, profilesOf } from "./q-matrix.js";

/** Options of {@link cdmModelFit}. */
export interface CdmModelFitOptions {
  /**
   * The level of the RMSEA interval.
   *
   * @defaultValue `0.9`
   */
  readonly level?: number;
}

/** The fit of a calibrated cognitive diagnosis model to the first- and second-order margins of its test. */
export interface CdmModelFit {
  /** M2; `null` without degrees of freedom, or when Ξ is not positive definite. */
  readonly m2: number | null;
  /**
   * The moments less the rank of Δ: the items' parameters and the proportions of the profiles less 1, unless the
   * first- and second-order margins do not identify them all, as for the G-DINA and the A-CDM of GDINA's `sim10GDINA`.
   */
  readonly degreesOfFreedom: number;
  /** The probability of a chi-square at least as large when the model fits; `null` with `m2`. */
  readonly significance: number | null;
  /** The RMSEA, √(max(0, (M2 - df) / (N df))), as GDINA's `RMSEA2`; `null` with `m2`. */
  readonly rmsea: number | null;
  /** The interval of the RMSEA at `level`, as GDINA's; `null` with `m2`. */
  readonly rmseaInterval: readonly [lower: number, upper: number] | null;
  /** The root mean square of the residual correlations over the pairs of items; `null` for an item without variance. */
  readonly srmsr: number | null;
}

const DEFAULT_LEVEL = 0.9;

/**
 * Returns the limited-information fit of a calibrated cognitive diagnosis model: M2 with RMSEA and SRMSR, as GDINA's
 * `modelfit`.
 *
 * @remarks
 * M2 is N e′ (Ξ⁻¹ - Ξ⁻¹ Δ (Δ′ Ξ⁻¹ Δ)⁻¹ Δ′ Ξ⁻¹) e over the items' shares correct and the shares of each pair correct.
 * The moments are exact sums over the profiles, which are the nodes. Δ has a column for each parameter of an item on
 * its model's scale, as GDINA's design matrices give them, and one for each proportion of a profile but the last.
 * Where the margins do not identify every parameter, Δ's rank is below its columns, and the statistic uses the whole
 * complement of its span, with as many degrees of freedom as the moments less that rank. Uses the persons who answered
 * every item. A statistic needs fewer parameters than moments, which with 60 items at most leaves 10 attributes; above
 * that it is `null`, and SRMSR is still computed.
 *
 * @param matrix - The responses of the persons, dichotomous.
 * @param calibration - The calibration of the matrix's items, in its order.
 * @param options - The level of the RMSEA interval.
 * @returns M2, its degrees of freedom and significance, the RMSEA with its interval, and SRMSR.
 * @throws `DataError` when the matrix, the calibration or the option is invalid, the matrix's items are not the
 * calibration's, a response is neither 0 nor 1, there are more than 60 items, or fewer than two persons answered every
 * item.
 * @see {@link https://doi.org/10.1111/bmsp.12074 | Hansen, Cai, Monroe and Li (2016)}
 * @see {@link https://doi.org/10.1111/j.1745-3984.2012.00185.x | Chen, de la Torre and Zhang (2013)}
 */
export function cdmModelFit(
  matrix: ResponseMatrix,
  calibration: CdmCalibration,
  options: CdmModelFitOptions = {},
): CdmModelFit {
  checkOptions(options);
  const { level = DEFAULT_LEVEL } = options;
  checkOpenUnitInterval(level, "The option level");
  const model = checkedModel(matrix, calibration);
  const { design, state, models, sample } = model;
  const itemCount = models.length;
  const parameters =
    design.itemAttributes.reduce((sum, indices, j) => sum + parameterCount(models[j], indices.length), 0) +
    design.profiles.length -
    1;
  const moments = (itemCount * (itemCount + 1)) / 2;
  const srmsr = residualCorrelationRoot(sample, profileMoments(design, state));
  const { statistic: m2, rank } =
    moments > parameters ? statistic(model, parameters) : { statistic: null, rank: parameters };
  const degreesOfFreedom = moments - rank;
  if (m2 === null) {
    return { m2, degreesOfFreedom, significance: null, rmsea: null, rmseaInterval: null, srmsr };
  }
  const persons = sample.persons;
  return {
    m2,
    degreesOfFreedom,
    significance: chiSquareUpperTail(m2, degreesOfFreedom),
    rmsea: Math.sqrt(Math.max(0, (m2 - degreesOfFreedom) / (persons * degreesOfFreedom))),
    rmseaInterval: rmseaInterval(m2, degreesOfFreedom, persons, level),
    srmsr,
  };
}

/** A calibration checked against its matrix, with the moments of the persons who answered every item. */
export interface CheckedModel {
  readonly design: CdmDesign;
  readonly state: CdmState;
  readonly models: readonly CdmModel[];
  readonly sample: SampleMoments;
}

/**
 * Checks a calibration against its matrix and returns its parts and the sample's moments.
 *
 * @throws `DataError` as `calibratedModel` does, or when an item's model is unknown, there are more than 60 items, or
 * fewer than two persons answered every item.
 */
export function checkedModel(matrix: ResponseMatrix, calibration: CdmCalibration): CheckedModel {
  const { design, state } = calibratedModel(matrix, calibration);
  const models = calibration.items.map((item) => {
    checkChoice(item.model, `The model of item "${item.id}"`, MODELS);
    return item.model;
  });
  if (models.length > MAX_MOMENT_ITEMS) {
    throw new DataError(
      `The fit of a diagnosis model takes at most ${MAX_MOMENT_ITEMS} items, found ${models.length}.`,
    );
  }
  const complete = matrix.responses.filter((row) => !row.includes(null));
  if (complete.length < 2) {
    throw new DataError(
      `The fit of a diagnosis model needs at least two persons who answered every item, found ${complete.length}.`,
    );
  }
  return { design, state, models, sample: sampleMoments(complete, models.length) };
}

/** Returns the shares correct and the shares of each pair correct under the model: exact sums over the profiles. */
export function profileMoments(design: CdmDesign, state: CdmState): ScoreMoments {
  const itemCount = state.probabilities.length;
  const means = new Float64Array(itemCount);
  const products = new Float64Array(itemCount * itemCount);
  const success = new Float64Array(itemCount);
  state.proportions.forEach((proportion, c) => {
    if (proportion === 0) return;
    for (let j = 0; j < itemCount; j++) success[j] = successProbability(design, state, j, c);
    for (let i = 0; i < itemCount; i++) {
      const weighted = proportion * (success[i] ?? 0);
      means[i] = (means[i] ?? 0) + weighted;
      products[i * itemCount + i] = (products[i * itemCount + i] ?? 0) + weighted;
      for (let j = i + 1; j < itemCount; j++) {
        products[i * itemCount + j] = (products[i * itemCount + j] ?? 0) + weighted * (success[j] ?? 0);
      }
    }
  });
  return { means, products };
}

/** Returns item j's success probability for profile c. */
function successProbability(design: CdmDesign, state: CdmState, j: number, c: number): number {
  return state.probabilities[j]?.[design.groupOf[j]?.[c] ?? 0] ?? 0;
}

/** Returns M2 and the rank of Δ. */
function statistic({ design, state, models, sample }: CheckedModel, parameters: number): QuadraticForm {
  const itemCount = models.length;
  const profiles = design.profiles.length;
  const means = new Float64Array(profiles * itemCount);
  for (let c = 0; c < profiles; c++) {
    for (let j = 0; j < itemCount; j++) means[c * itemCount + j] = successProbability(design, state, j, c);
  }
  const first = { items: Int32Array.from(models, (_, j) => j), means, scores: means, squares: means };
  const tables: MomentTables = { weights: state.proportions, itemCount, means, squares: means, first };
  const moments = modelMoments(tables);
  const rows = rowCount(tables);
  const residuals = new Float64Array(rows);
  residuals.set(sample.means);
  let row = itemCount;
  for (let i = 0; i < itemCount; i++) {
    for (let j = i + 1; j < itemCount; j++) residuals[row++] = sample.products[i * itemCount + j] ?? 0;
  }
  for (let r = 0; r < rows; r++) residuals[r] = (residuals[r] ?? 0) - (moments.expected[r] ?? 0);
  const delta = new Float64Array(rows * parameters);
  let column = 0;
  models.forEach((model, j) => {
    for (const derivative of itemDerivatives(design, state, model, j)) {
      delta[j * parameters + column] = derivative.reduce(
        (sum, value, c) => sum + value * (state.proportions[c] ?? 0),
        0,
      );
      addPairDerivatives(tables, delta, parameters, column++, j, derivative);
    }
  });
  // The proportion of profile c, the last being 1 less the others: dw/dπ_c is 1 at c and -1 at the last profile.
  const derivative = new Float64Array(profiles);
  for (let c = 0; c < profiles - 1; c++) {
    derivative[c] = 1;
    derivative[profiles - 1] = -1;
    addWeightDerivatives(moments.conditional, delta, rows, parameters, column++, derivative);
    derivative[c] = 0;
  }
  return limitedInformationStatistic(residuals, moments.covariance, delta, parameters, sample.persons);
}

/**
 * Returns, for each parameter of item j on its model's scale, the derivative of its success probability at each
 * profile: GDINA's design matrix of the latent groups, times dP/dη of the model's link.
 */
function itemDerivatives(design: CdmDesign, state: CdmState, model: CdmModel, j: number): Float64Array[] {
  const groups = state.probabilities[j] ?? new Float64Array(0);
  const groupOf = design.groupOf[j] ?? new Int32Array(0);
  const reduced = profilesOf(design.itemAttributes[j]?.length ?? 0);
  const link = LINKS[model];
  // dP/dη: 1 for the identity, P (1 - P) for the logit, P for the log.
  const slope = (p: number): number => {
    if (link === "logit") return p * (1 - p);
    return link === "log" ? p : 1;
  };
  const columns = designColumns(model, reduced);
  return columns.map((entries) =>
    Float64Array.from(groupOf, (group) => (entries[group] ?? 0) * slope(groups[group] ?? 0)),
  );
}

/** Returns the columns of a model's design matrix over the latent groups of an item, whose profiles are `reduced`. */
function designColumns(model: CdmModel, reduced: readonly (readonly number[])[]): number[][] {
  const groups = reduced.length;
  const ones = new Array<number>(groups).fill(1);
  if (model === "gdina") return reduced.map((_, column) => reduced.map((__, group) => (group === column ? 1 : 0)));
  if (model === "dina") return [ones, reduced.map((profile) => (profile.every((value) => value === 1) ? 1 : 0))];
  if (model === "dino") return [ones, reduced.map((profile) => (profile.some((value) => value === 1) ? 1 : 0))];
  const attributes = reduced[0]?.length ?? 0;
  return [ones, ...Array.from({ length: attributes }, (_, k) => reduced.map((profile) => profile[k] ?? 0))];
}
