import {
  checkArray,
  checkChoice,
  checkOptions,
  checkRecord,
  checkStopping,
  checkUnitInterval,
  DataError,
  isRecord,
  isUnitInterval,
  normalizeLogs,
} from "@measurekit/core";
import { checkDichotomous } from "../data/dichotomous.js";
import { checkResponseMatrix, type ResponseMatrix } from "../data/response-matrix.js";
import { type CdmDesign, cdmDesign, profilesOf, type QMatrix } from "./q-matrix.js";

/**
 * A cognitive diagnosis model of an item's success probabilities: `"gdina"`, one per latent group of its attributes;
 * `"dina"`, one for mastering all of them and one otherwise; `"dino"`, one for mastering none and one otherwise.
 *
 * @see {@link https://doi.org/10.1007/s11336-011-9207-7 | de la Torre (2011)}
 */
export type CdmModel = "gdina" | "dina" | "dino";

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
   * The most EM iterations.
   *
   * @defaultValue `2000`
   */
  readonly maxIterations?: number;
}

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
  /** -2 logLikelihood + 2p, p the items' probabilities, 2 per DINA or DINO item, and the proportions less 1. */
  readonly aic: number;
  /** -2 logLikelihood + p ln n, n the persons with responses. */
  readonly bic: number;
  /** Whether EM reached `tolerance` within `maxIterations`. */
  readonly converged: boolean;
  /** EM iterations run. */
  readonly iterations: number;
}

const MODELS: readonly CdmModel[] = ["gdina", "dina", "dino"];
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
 * and each profile's proportion to its expected persons, the saturated joint distribution. As GDINA, a success
 * probability stays within [1e-4, 1 - 1e-4], a group with fewer than 0.001 expected persons takes 0.0005 more correct
 * over 0.001 more persons, and a proportion stays above the machine epsilon. Missing responses are skipped; persons
 * without responses change nothing.
 *
 * @throws `DataError` when the matrix, the Q-matrix or an option is invalid, a response is neither 0 nor 1, or there is
 * no response.
 */
export function calibrateCdm(
  matrix: ResponseMatrix,
  qMatrix: QMatrix,
  options: CdmCalibrationOptions = {},
): CdmCalibration {
  checkResponseMatrix(matrix);
  checkOptions(options);
  const { tolerance = DEFAULT_TOLERANCE, maxIterations = DEFAULT_MAX_ITERATIONS } = options;
  checkStopping(tolerance, maxIterations);
  const design = cdmDesign(matrix.itemIds, qMatrix);
  const models = itemModels(options.model, matrix.itemIds.length);
  const data = dichotomousRows(matrix);
  if (data.respondents === 0) throw new DataError("Cognitive diagnosis needs at least one response.");
  const state = startingValues(design);
  let iterations = 0;
  let converged = false;
  while (!converged && iterations < maxIterations) {
    iterations++;
    const counts = expectedCounts(data, design, state);
    converged = maximize(models, state, counts) <= tolerance;
  }
  const logLikelihood = expectedCounts(data, design, state).logLikelihood;
  const parameters =
    models.reduce((sum, model, j) => sum + (model === "gdina" ? (state.probabilities[j]?.length ?? 0) : 2), 0) +
    design.profiles.length -
    1;
  return {
    attributes: design.attributes,
    items: matrix.itemIds.map((id, j) => {
      const probabilities = Array.from(state.probabilities[j] ?? []);
      return {
        id,
        attributes: (design.itemAttributes[j] ?? []).map((k) => design.attributes[k] ?? ""),
        model: models[j] ?? "gdina",
        probabilities,
        guessing: probabilities[0] ?? 0,
        slip: 1 - (probabilities.at(-1) ?? 1),
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

/** Responses by person, then item: 0, 1, or -1 when missing. */
export interface DichotomousRows {
  readonly itemCount: number;
  readonly responses: Int8Array;
  /** The number of persons with at least one response. */
  readonly respondents: number;
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
  matrix.responses.forEach((row, person) => {
    let answered = false;
    row.forEach((response, item) => {
      if (response === null) return;
      responses[person * itemCount + item] = response;
      answered = true;
    });
    if (answered) respondents++;
  });
  return { itemCount, responses, respondents };
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

/** The parameters during EM: each item's success probability per latent group, and the profile proportions. */
export interface CdmState {
  readonly probabilities: readonly Float64Array[];
  readonly proportions: Float64Array;
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

/** The expected counts of an E-step: correct responses and persons per item group, persons per profile. */
interface ExpectedCounts {
  readonly correct: readonly Float64Array[];
  readonly persons: readonly Float64Array[];
  readonly profiles: Float64Array;
  readonly logLikelihood: number;
}

/**
 * Runs the E-step: each person's posterior over the profiles, from the proportions and the success probabilities of
 * the answered items, gives the expected counts and the marginal log-likelihood. The loop over profiles is outside the
 * loop over a person's items, so that consecutive additions go to different counts; each count still receives its
 * terms in the order of persons, then profiles.
 */
function expectedCounts(data: DichotomousRows, design: CdmDesign, state: CdmState): ExpectedCounts {
  const { itemCount, responses } = data;
  const profileCount = design.profiles.length;
  // Every item's counts in one array, persons then correct responses per latent group; every item's groups in one.
  const starts = new Int32Array(itemCount + 1);
  state.probabilities.forEach((p, j) => {
    starts[j + 1] = (starts[j] ?? 0) + 2 * p.length;
  });
  const counts = new Float64Array(starts[itemCount] ?? 0);
  const groups = new Int32Array(itemCount * profileCount);
  design.groupOf.forEach((groupOf, j) => {
    groups.set(groupOf, j * profileCount);
  });
  const profiles = new Float64Array(profileCount);
  // For each count the current person adds to: its offset in `counts` and the offset of its item's groups.
  const countOffsets = new Int32Array(2 * itemCount);
  const groupOffsets = new Int32Array(2 * itemCount);
  let logLikelihood = 0;
  forEachPosterior(data, design, state, (person, posterior, logMarginal) => {
    logLikelihood += logMarginal;
    let size = 0;
    for (let j = 0; j < itemCount; j++) {
      const response = responses[person * itemCount + j] ?? -1;
      if (response < 0) continue;
      const start = starts[j] ?? 0;
      countOffsets[size] = start;
      groupOffsets[size++] = j * profileCount;
      if (response === 1) {
        countOffsets[size] = start + ((starts[j + 1] ?? 0) - start) / 2;
        groupOffsets[size++] = j * profileCount;
      }
    }
    addPosterior(posterior, profiles, counts, groups, countOffsets.subarray(0, size), groupOffsets);
  });
  return {
    persons: state.probabilities.map((p, j) => counts.subarray(starts[j], (starts[j] ?? 0) + p.length)),
    correct: state.probabilities.map((p, j) => counts.subarray((starts[j] ?? 0) + p.length, starts[j + 1])),
    profiles,
    logLikelihood,
  };
}

/**
 * Adds a person's posterior to the profile counts and, for each entry a of `countOffsets`, each profile's posterior to
 * the count at `countOffsets[a]` plus the profile's group, read at `groupOffsets[a]` in `groups`.
 */
function addPosterior(
  posterior: Float64Array,
  profiles: Float64Array,
  counts: Float64Array,
  groups: Int32Array,
  countOffsets: Int32Array,
  groupOffsets: Int32Array,
): void {
  for (let c = 0; c < profiles.length; c++) {
    const p = posterior[c] ?? 0;
    profiles[c] = (profiles[c] ?? 0) + p;
    for (let a = 0; a < countOffsets.length; a++) {
      const index = (countOffsets[a] ?? 0) + (groups[(groupOffsets[a] ?? 0) + c] ?? 0);
      counts[index] = (counts[index] ?? 0) + p;
    }
  }
}

/**
 * Calls `visit` with each person who answered an item, the posterior over the profiles and the log of the person's
 * marginal likelihood. `posterior` is reused between calls.
 */
export function forEachPosterior(
  data: DichotomousRows,
  design: CdmDesign,
  state: CdmState,
  visit: (person: number, posterior: Float64Array, logMarginal: number) => void,
): void {
  const { itemCount, responses } = data;
  const logProbabilities = state.probabilities.map((p) => ({
    correct: p.map(Math.log),
    incorrect: p.map((value) => Math.log(1 - value)),
  }));
  const logProportions = state.proportions.map(Math.log);
  const posterior = new Float64Array(design.profiles.length);
  const personCount = responses.length / itemCount;
  for (let person = 0; person < personCount; person++) {
    posterior.set(logProportions);
    let answered = false;
    for (let j = 0; j < itemCount; j++) {
      const response = responses[person * itemCount + j] ?? -1;
      const table = logProbabilities[j];
      const groupOf = design.groupOf[j];
      if (response < 0 || table === undefined || groupOf === undefined) continue;
      answered = true;
      addGroupValues(posterior, response === 1 ? table.correct : table.incorrect, groupOf);
    }
    if (answered) visit(person, posterior, normalizeLogs(posterior));
  }
}

/** Adds `values[group(c)]` to `out[c]` for each profile c. */
function addGroupValues(out: Float64Array, values: Float64Array, groupOf: Int32Array): void {
  for (let c = 0; c < out.length; c++) out[c] = (out[c] ?? 0) + (values[groupOf[c] ?? 0] ?? 0);
}

/** Runs the M-step and returns the largest change of a probability or proportion. */
function maximize(models: readonly CdmModel[], state: CdmState, counts: ExpectedCounts): number {
  let change = 0;
  state.probabilities.forEach((probabilities, j) => {
    const updated = groupProbabilities(
      models[j] ?? "gdina",
      counts.correct[j],
      counts.persons[j],
      probabilities.length,
    );
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
  if (model === "gdina") return Array.from({ length: groups }, (_, l) => pool(l, l + 1));
  if (model === "dina") {
    const below = pool(0, groups - 1);
    return Array.from({ length: groups }, (_, l) => (l === groups - 1 ? pool(groups - 1, groups) : below));
  }
  const above = pool(1, groups);
  return Array.from({ length: groups }, (_, l) => (l === 0 ? pool(0, 1) : above));
}

/** Returns R / N, with GDINA's correction of a small group and its bounds. */
function bounded(correct: number, persons: number): number {
  const p = persons < SMALL_GROUP ? (correct + SMALL_GROUP / 2) / (persons + SMALL_GROUP) : correct / persons;
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
