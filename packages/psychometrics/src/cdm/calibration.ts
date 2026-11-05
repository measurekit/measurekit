import {
  checkArray,
  checkChoice,
  checkOptions,
  checkRecord,
  checkStopping,
  checkUnitInterval,
  DataError,
  type FieldNames,
  isRecord,
  isUnitInterval,
  iterateToConvergence,
  type QMatrix,
} from "@measurekit/core";
import { checkDichotomous, checkResponseMatrix, type ResponseMatrix } from "../data/internal.js";
import { type AdditiveMaximizer, additiveMaximizer, type CdmLink, saturatedCoefficients } from "./additive.js";
import { type CdmState, type DichotomousRows, type ExpectedCounts, expectedCounts } from "./expectation.js";
import { type CdmDesign, cdmDesign, profilesOf } from "./q-matrix.js";
import { checkIncompleteSize, incompleteCovariances } from "./standard-errors.js";

/**
 * A cognitive diagnosis model of an item's success probabilities: `"gdina"`, one per latent group of its attributes;
 * `"dina"`, one for mastering all of them and one otherwise; `"dino"`, one for mastering none and one otherwise;
 * `"acdm"`, `"llm"` and `"rrum"`, an intercept and one effect per attribute on the identity, logit and log scale.
 *
 * @see {@link https://doi.org/10.1007/s11336-011-9207-7 | de la Torre (2011)}
 */
export type CdmModel = "gdina" | "dina" | "dino" | "acdm" | "llm" | "rrum";

/** Options of {@link calibrateCdm}. */
export interface CdmCalibrationOptions {
  /**
   * The model of every item, or one per item in item order.
   *
   * @defaultValue `"gdina"`
   */
  readonly model?: CdmModel | readonly CdmModel[];
  /**
   * EM has converged when no success probability or profile proportion changes by more than this.
   *
   * @defaultValue `1e-4`
   */
  readonly tolerance?: number;
  /**
   * The standard errors of the success probabilities: `"none"`, or `"incomplete"`, their covariance from the outer
   * product of the gradient over persons, as GDINA's `SE.type = 2`. `"incomplete"` needs the G-DINA model for every
   * item, and gives each item its `standardErrors` and `covariance`, which `compareItemModels` uses.
   *
   * @defaultValue `"none"`
   */
  readonly standardErrors?: "none" | "incomplete";
  /**
   * The most EM iterations.
   *
   * @defaultValue `2000`
   */
  readonly maxIterations?: number;
}

const CDM_CALIBRATION_OPTIONS: FieldNames<CdmCalibrationOptions> = {
  model: true,
  tolerance: true,
  standardErrors: true,
  maxIterations: true,
};

/** A calibrated item of a cognitive diagnosis model. */
export interface CdmItem {
  /** The item's id. */
  readonly id: string;
  /** Its attributes, in the design's order. */
  readonly attributes: readonly string[];
  /** Its model. */
  readonly model: CdmModel;
  /**
   * The probability of a correct response in each latent group: the reduced profiles over its attributes, by the
   * number mastered, then earlier attributes first.
   */
  readonly probabilities: readonly number[];
  /** The probability of a correct response mastering none of its attributes. */
  readonly guessing: number;
  /** The probability of an incorrect response mastering all of them. */
  readonly slip: number;
  /**
   * The coefficients δ on the model's link scale, as GDINA's `coef(what = "delta")`. The additive models have the
   * intercept, then one effect per attribute in the item's order. The G-DINA has the intercept, the main effects and
   * the interactions on the identity scale, one per latent group in the order of `probabilities`. The DINA and DINO
   * have the guessing and the rise from it to 1 - slip.
   */
  readonly coefficients: readonly number[];
  /**
   * The standard error of each success probability, in the order of `probabilities`; absent without the option
   * `standardErrors`.
   */
  readonly standardErrors?: readonly number[];
  /**
   * The covariance matrix of the success probabilities, by rows, in the order of `probabilities`; absent without the
   * option `standardErrors`.
   */
  readonly covariance?: readonly (readonly number[])[];
}

/** A profile with its estimated share of the population. */
export interface ProfileProportion {
  /** 1 for each mastered attribute, 0 otherwise, in the design's attribute order. */
  readonly profile: readonly number[];
  /** Its proportion. */
  readonly proportion: number;
}

/** A cognitive diagnosis model calibrated by marginal maximum likelihood. */
export interface CdmCalibration {
  /** The attributes, in order of first appearance in the Q-matrix. */
  readonly attributes: readonly string[];
  /** The items, in item order. */
  readonly items: readonly CdmItem[];
  /** Every profile with its proportion: the saturated joint distribution of the attributes. */
  readonly profiles: readonly ProfileProportion[];
  /** The marginal log-likelihood. */
  readonly logLikelihood: number;
  /**
   * -2 logLikelihood + 2p, p the items' parameters (2^K_j per G-DINA item, 2 per DINA or DINO item and K_j + 1 per
   * additive item, for K_j attributes) and the proportions less 1.
   */
  readonly aic: number;
  /** -2 logLikelihood + p ln n, n the persons with responses. */
  readonly bic: number;
  /** Whether EM reached `tolerance` within `maxIterations`. */
  readonly converged: boolean;
  /** EM iterations run. */
  readonly iterations: number;
}

export const MODELS: readonly CdmModel[] = ["gdina", "dina", "dino", "acdm", "llm", "rrum"];
const STANDARD_ERRORS = ["none", "incomplete"] as const;
/** The link of each additive model (de la Torre, 2011): the identity for the A-CDM, the logit for the LLM, the log for the R-RUM. */
export const LINKS: Readonly<Partial<Record<CdmModel, CdmLink>>> = { acdm: "identity", llm: "logit", rrum: "log" };
const DEFAULT_TOLERANCE = 1e-4;
const DEFAULT_MAX_ITERATIONS = 2000;
/** The starting success probability of a group mastering none of an item's attributes, and the rise to all mastered. */
const START_LOW = 0.2;
const START_RISE = 0.6;
/** GDINA's bounds of a success probability. */
const PROBABILITY_BOUNDS = [1e-4, 1 - 1e-4] as const;
/** GDINA's correction of a latent group with fewer expected persons than this: R + 0.0005 over N + 0.001. */
const SMALL_GROUP = 0.001;
/** GDINA's floor of a profile proportion. */
const MIN_PROPORTION = Number.EPSILON;

/**
 * Calibrates a cognitive diagnosis model of dichotomous items by marginal maximum likelihood with EM over the 2^K
 * profiles of K attributes, as GDINA's `GDINA`.
 *
 * @remarks
 * The E-step gives each person's posterior over the profiles; the M-step sets each latent group's success probability
 * to its expected correct responses over its expected persons, pooled over the groups the DINA or DINO model equates,
 * and each profile's proportion to its expected persons, the saturated joint distribution. The additive models
 * (A-CDM, LLM, R-RUM) have no such closed form: their M-step maximizes the expected log-likelihood over the intercept
 * and one effect per attribute, on the identity, logit or log scale, exactly. As GDINA, a success probability stays
 * within [1e-4, 1 - 1e-4], a group with fewer than 0.001 expected persons takes 0.0005 more correct over 0.001 more
 * persons, and a proportion stays above the machine epsilon. Missing responses are skipped; persons without responses
 * change nothing.
 *
 * The incomplete standard errors are computed at the estimates, from each person's posterior over the latent groups of
 * every item. They take O(n p²) for n persons and p success probabilities, and the inversion of the p by p information
 * O(p³) per sweep.
 *
 * @param matrix - The responses of the persons, dichotomous.
 * @param qMatrix - The attributes of each item, by item id.
 * @param options - The model, the standard errors and the stopping rules; each option has its default.
 * @returns The attributes, the items with their parameters, the proportion of each profile, the log-likelihood, AIC,
 *   BIC, whether the EM converged and the iterations run.
 * @throws `DataError` when the matrix, the Q-matrix or an option is invalid, a response is neither 0 nor 1, there is
 * no response, or `standardErrors` is `"incomplete"` and an item is not G-DINA or the items have more than 3,162
 * success probabilities.
 *
 * @example
 * ```ts
 * import { calibrateCdm } from "@measurekit/psychometrics/cdm";
 * import { buildResponseMatrix } from "@measurekit/psychometrics/data";
 *
 * const matrix = buildResponseMatrix([
 *   [1, 1, 0, 1],
 *   [0, 0, 1, 0],
 *   [1, 0, 0, 1],
 *   [1, 1, 1, 1],
 *   [0, 1, 0, 0],
 *   [0, 0, 0, 0],
 * ]);
 * const qMatrix = { item1: ["a"], item2: ["a", "b"], item3: ["b"], item4: ["a"] };
 * const calibration = calibrateCdm(matrix, qMatrix, { model: "dina" });
 * calibration.attributes; // => ["a", "b"]
 * calibration.converged; // => true
 * calibration.profiles.map(({ profile }) => profile); // => [[0, 0], [1, 0], [0, 1], [1, 1]]
 * calibration.items[1]?.attributes; // => ["a", "b"]
 * ```
 */
export function calibrateCdm(
  matrix: ResponseMatrix,
  qMatrix: QMatrix,
  options: CdmCalibrationOptions = {},
): CdmCalibration {
  checkResponseMatrix(matrix);
  checkOptions(options, CDM_CALIBRATION_OPTIONS);
  const { tolerance = DEFAULT_TOLERANCE, maxIterations = DEFAULT_MAX_ITERATIONS, standardErrors = "none" } = options;
  checkStopping(tolerance, maxIterations);
  checkChoice(standardErrors, "The option standardErrors", STANDARD_ERRORS);
  const design = cdmDesign(matrix.itemIds, qMatrix);
  const models = itemModels(options.model, matrix.itemIds.length);
  if (standardErrors === "incomplete") {
    checkSaturated(models, matrix.itemIds);
    checkIncompleteSize(design);
  }
  const data = dichotomousRows(matrix);
  if (data.respondents === 0) throw new DataError("Cognitive diagnosis needs at least one response.");
  const state = startingValues(design);
  const steps = additiveSteps(models, design, state);
  const { converged, iterations } = iterateToConvergence(
    () => maximize(models, steps, state, expectedCounts(data, design, state)),
    { tolerance, maxIterations },
  );
  const logLikelihood = expectedCounts(data, design, state).logLikelihood;
  const covariances = standardErrors === "incomplete" ? incompleteCovariances(data, design, state) : [];
  const parameters = parameterTotal(design, models);
  return {
    attributes: design.attributes,
    items: matrix.itemIds.map((id, j) => {
      const probabilities = Array.from(state.probabilities[j] ?? []);
      const model = models[j] ?? "gdina";
      const step = steps[j];
      return {
        id,
        attributes: (design.itemAttributes[j] ?? []).map((k) => design.attributes[k] ?? ""),
        model,
        probabilities,
        guessing: probabilities[0] ?? 0,
        slip: 1 - (probabilities.at(-1) ?? 1),
        coefficients: step
          ? Array.from(step.coefficients)
          : identityCoefficients(model, probabilities, design.itemAttributes[j]?.length ?? 0),
        ...itemErrors(covariances[j], probabilities.length),
      };
    }),
    profiles: design.profiles.map((profile, c) => ({ profile, proportion: state.proportions[c] ?? 0 })),
    logLikelihood,
    aic: -2 * logLikelihood + 2 * parameters,
    bic: -2 * logLikelihood + parameters * Math.log(data.respondents),
    converged,
    iterations,
  };
}

/**
 * Checks that every item is G-DINA, which the covariance of success probabilities is defined for.
 *
 * @throws `DataError` when an item is not.
 */
function checkSaturated(models: readonly CdmModel[], itemIds: readonly string[]): void {
  const index = models.findIndex((model) => model !== "gdina");
  if (index >= 0) {
    throw new DataError(
      `The option standardErrors "incomplete" needs the G-DINA model for every item, found "${models[index]}" for item "${itemIds[index]}".`,
    );
  }
}

/** Returns the standard errors and the covariance rows of an item from its flat covariance matrix, if it has one. */
function itemErrors(
  covariance: Float64Array | undefined,
  size: number,
): { standardErrors?: number[]; covariance?: number[][] } {
  if (covariance === undefined) return {};
  const rows = Array.from({ length: size }, (_, a) => Array.from(covariance.subarray(a * size, (a + 1) * size)));
  return { standardErrors: rows.map((row, a) => Math.sqrt(Math.max(row[a] ?? 0, 0))), covariance: rows };
}

/** Returns the number of parameters of an item with `attributeCount` attributes: its success probabilities. */
function parameterCount(model: CdmModel | undefined, attributeCount: number): number {
  if (model === "dina" || model === "dino") return 2;
  return model === undefined || model === "gdina" ? 2 ** attributeCount : attributeCount + 1;
}

/** Returns the parameters of a design's items under their models and the free proportions of its profiles. */
export function parameterTotal(design: CdmDesign, models: readonly CdmModel[]): number {
  const items = design.itemAttributes.reduce((sum, indices, j) => sum + parameterCount(models[j], indices.length), 0);
  return items + design.profiles.length - 1;
}

/** Returns the coefficients of a G-DINA, DINA or DINO item from its success probabilities. */
function identityCoefficients(model: CdmModel, probabilities: readonly number[], attributeCount: number): number[] {
  if (model === "gdina") return saturatedCoefficients(probabilities, attributeCount);
  const guessing = probabilities[0] ?? 0;
  return [guessing, (probabilities.at(-1) ?? 0) - guessing];
}

/**
 * Checks that every response is 0 or 1 and returns the responses.
 *
 * @throws `DataError` when a response is neither.
 */
export function dichotomousRows(matrix: ResponseMatrix): DichotomousRows {
  checkDichotomous(matrix, "Cognitive diagnosis");
  const itemCount = matrix.itemIds.length;
  const responses = new Int8Array(matrix.responses.length * itemCount).fill(-1);
  let respondents = 0;
  let complete = true;
  for (let person = 0; person < matrix.responses.length; person++) {
    const row = matrix.responses[person] ?? [];
    let answers = 0;
    for (let item = 0; item < row.length; item++) {
      const response = row[item];
      if (response === null || response === undefined) continue;
      responses[person * itemCount + item] = response;
      answers++;
    }
    if (answers > 0) respondents++;
    if (answers > 0 && answers < itemCount) complete = false;
  }
  return { itemCount, responses, respondents, complete };
}

function itemModels(model: CdmCalibrationOptions["model"], itemCount: number): readonly CdmModel[] {
  if (model === undefined || typeof model === "string") {
    const each = model ?? "gdina";
    checkChoice(each, "The option model", MODELS);
    return Array.from({ length: itemCount }, () => each);
  }
  checkArray(model, "The option model must be a cognitive diagnosis model or an array of one per item");
  if (model.length !== itemCount) throw new DataError(`Found ${model.length} models for ${itemCount} items.`);
  model.forEach((each, j) => {
    checkChoice(each, `The model of item ${j + 1}`, MODELS);
  });
  return model;
}

/** Starts at P = 0.2 + 0.6 m / K_j for a group mastering m of the item's K_j attributes, and at equal proportions. */
function startingValues(design: CdmDesign): CdmState {
  const probabilities = design.itemAttributes.map((indices) =>
    Float64Array.from(profilesOf(indices.length), (profile) => {
      const mastered = profile.reduce((sum, value) => sum + value, 0);
      return START_LOW + (START_RISE * mastered) / indices.length;
    }),
  );
  return { probabilities, proportions: new Float64Array(design.profiles.length).fill(1 / design.profiles.length) };
}

/** Builds the M-step of each additive item, starting from the item's starting probabilities. */
function additiveSteps(
  models: readonly CdmModel[],
  design: CdmDesign,
  state: CdmState,
): readonly (AdditiveMaximizer | undefined)[] {
  return models.map((model, j) => {
    const link = LINKS[model];
    if (link === undefined) return undefined;
    return additiveMaximizer(
      link,
      design.itemAttributes[j]?.length ?? 0,
      PROBABILITY_BOUNDS,
      state.probabilities[j] ?? [],
    );
  });
}

/** Runs the M-step and returns the largest change of a probability or proportion. */
function maximize(
  models: readonly CdmModel[],
  steps: readonly (AdditiveMaximizer | undefined)[],
  state: CdmState,
  counts: ExpectedCounts,
): number {
  let change = 0;
  state.probabilities.forEach((probabilities, j) => {
    const correct = counts.correct[j];
    const persons = counts.persons[j];
    const step = steps[j];
    const updated = step
      ? step.maximize(
          Float64Array.from(correct ?? [], (value, l) => correctedCorrect(value, persons?.[l] ?? 0)),
          Float64Array.from(persons ?? [], correctedPersons),
        )
      : groupProbabilities(models[j] ?? "gdina", correct, persons, probabilities.length);
    updated.forEach((value, l) => {
      change = Math.max(change, Math.abs(value - (probabilities[l] ?? 0)));
      probabilities[l] = value;
    });
  });
  let total = 0;
  for (const count of counts.profiles) total += count;
  let sum = 0;
  const proportions = Float64Array.from(counts.profiles, (count) => Math.max(count / total, MIN_PROPORTION));
  for (const proportion of proportions) sum += proportion;
  proportions.forEach((proportion, c) => {
    const value = proportion / sum;
    change = Math.max(change, Math.abs(value - (state.proportions[c] ?? 0)));
    state.proportions[c] = value;
  });
  return change;
}

/**
 * Returns an item's success probability per latent group from its expected correct responses and persons: each
 * group's own for the G-DINA, pooled over the groups below the last for the DINA and above the first for the DINO.
 */
function groupProbabilities(
  model: CdmModel,
  correct: Float64Array | undefined,
  persons: Float64Array | undefined,
  groups: number,
): number[] {
  const pool = (from: number, to: number): number => {
    let r = 0;
    let n = 0;
    for (let l = from; l < to; l++) {
      r += correct?.[l] ?? 0;
      n += persons?.[l] ?? 0;
    }
    return bounded(r, n);
  };
  if (model === "dina") {
    const below = pool(0, groups - 1);
    return Array.from({ length: groups }, (_, l) => (l === groups - 1 ? pool(groups - 1, groups) : below));
  }
  if (model === "dino") {
    const above = pool(1, groups);
    return Array.from({ length: groups }, (_, l) => (l === 0 ? pool(0, 1) : above));
  }
  return Array.from({ length: groups }, (_, l) => pool(l, l + 1));
}

/** Returns a group's expected correct responses, with GDINA's correction of a small group. */
function correctedCorrect(correct: number, persons: number): number {
  return persons < SMALL_GROUP ? correct + SMALL_GROUP / 2 : correct;
}

/** Returns a group's expected persons, with GDINA's correction of a small group. */
function correctedPersons(persons: number): number {
  return persons < SMALL_GROUP ? persons + SMALL_GROUP : persons;
}

/** Returns R / N, with GDINA's correction of a small group and its bounds. */
function bounded(correct: number, persons: number): number {
  const p = correctedCorrect(correct, persons) / correctedPersons(persons);
  return Math.min(Math.max(p, PROBABILITY_BOUNDS[0]), PROBABILITY_BOUNDS[1]);
}

/**
 * Checks that an item has one success probability from 0 to 1 per latent group of its attributes.
 *
 * @throws `DataError` when it does not.
 */
export function checkSuccessProbabilities(id: string, probabilities: readonly number[], groupCount: number): void {
  const valid =
    Array.isArray(probabilities) && probabilities.length === groupCount && probabilities.every(isUnitInterval);
  if (!valid) {
    throw new DataError(`Item "${id}" needs ${groupCount} probabilities from 0 to 1, one per latent group.`);
  }
}

/**
 * Checks a calibration, as from JSON, and returns its design and parameters for the persons of a matrix with the
 * calibration's items.
 *
 * @throws `DataError` when the matrix is invalid, its items are not the calibration's, a response is neither 0 nor 1,
 * or the calibration does not have the attributes, the success probabilities and the profile proportions of its design.
 */
export function calibratedModel(
  matrix: ResponseMatrix,
  calibration: CdmCalibration,
): { readonly data: DichotomousRows; readonly design: CdmDesign; readonly state: CdmState } {
  checkResponseMatrix(matrix);
  checkRecord(calibration, "A calibration must be an object from calibrateCdm");
  const { attributes, items, profiles } = calibration;
  checkArray(items, "A calibration needs the array items");
  items.forEach((item, j) => {
    checkRecord(item, `Calibrated item ${j + 1} must be an object { id, attributes, probabilities }`);
  });
  const ids = items.map((item) => item.id);
  if (ids.length !== matrix.itemIds.length || ids.some((id, j) => id !== matrix.itemIds[j])) {
    throw new DataError("The matrix needs the calibration's items, in its order.");
  }
  const design = cdmDesign(ids, Object.fromEntries(items.map((item) => [item.id, item.attributes])));
  const sameAttributes =
    Array.isArray(attributes) &&
    attributes.length === design.attributes.length &&
    attributes.every((attribute, k) => attribute === design.attributes[k]);
  if (!sameAttributes) {
    throw new DataError("A calibration's attributes must be its items', in order of first appearance.");
  }
  items.forEach((item, j) => {
    checkSuccessProbabilities(item.id, item.probabilities, 2 ** (design.itemAttributes[j]?.length ?? 0));
  });
  checkProfiles(profiles, design);
  const state = {
    probabilities: items.map((item) => Float64Array.from(item.probabilities)),
    proportions: Float64Array.from(profiles, (profile) => profile.proportion),
  };
  return { data: dichotomousRows(matrix), design, state };
}

/**
 * Checks a calibration's profiles: those of the design, in its order, with proportions from 0 to 1, not all 0.
 *
 * @throws `DataError` when they are not.
 */
function checkProfiles(profiles: readonly ProfileProportion[], design: CdmDesign): void {
  const valid =
    Array.isArray(profiles) &&
    profiles.length === design.profiles.length &&
    profiles.every((entry, c) => {
      const profile: unknown = isRecord(entry) ? entry.profile : undefined;
      const expected = design.profiles[c] ?? [];
      return (
        Array.isArray(profile) &&
        profile.length === expected.length &&
        expected.every((value, k) => profile[k] === value)
      );
    });
  if (!valid) {
    throw new DataError(
      `A calibration over ${design.attributes.length} attributes needs its ${design.profiles.length} profiles in order.`,
    );
  }
  let total = 0;
  profiles.forEach(({ proportion }, c) => {
    checkUnitInterval(proportion, `The proportion of profile ${c + 1}`);
    total += proportion;
  });
  if (!(total > 0)) throw new DataError("A calibration needs profile proportions that are not all 0.");
}
