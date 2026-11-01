import { checkArray, checkChoice, checkOptions, checkRecord, chiSquareUpperTail, DataError } from "@measurekit/core";
import { pseudoInverse } from "../math/pseudo-inverse.js";
import { type CdmCalibration, type CdmModel, checkSuccessProbabilities } from "./calibration.js";
import { profilesOf } from "./q-matrix.js";

/** A model that an item can be reduced to: every model but the G-DINA. */
export type ReducedCdmModel = Exclude<CdmModel, "gdina">;

/** Options of {@link compareItemModels}. */
export interface ItemComparisonOptions {
  /**
   * The reduced models to test, in the order of their `tests`; each at most once.
   *
   * @defaultValue `["dina", "dino", "acdm", "llm", "rrum"]`
   */
  readonly models?: readonly ReducedCdmModel[];
}

/** The Wald test of one reduced model of an item against its G-DINA estimates. */
export interface ItemModelTest {
  /** The reduced model. */
  readonly model: ReducedCdmModel;
  /** The Wald statistic; `null` when it is not finite. */
  readonly statistic: number | null;
  /** The degrees of freedom: the success probabilities less those of the model. */
  readonly degreesOfFreedom: number;
  /** The probability of a statistic at least as large under the reduced model; `null` when `statistic` is. */
  readonly pValue: number | null;
}

/** The comparison of the reduced models of one item. */
export interface ItemModelComparison {
  /** The item's id. */
  readonly item: string;
  /** One test per model of the option `models`, in its order. */
  readonly tests: readonly ItemModelTest[];
  /**
   * The model that the tests select: the reduced model with the largest p-value above 0.05, or `"gdina"` when every
   * p-value is at most 0.05.
   */
  readonly selected: CdmModel;
}

const REDUCED_MODELS: readonly ReducedCdmModel[] = ["dina", "dino", "acdm", "llm", "rrum"];
/** The level of a Wald test: a reduced model that it rejects at this level is not selected (Ma et al., 2016). */
const SIGNIFICANCE_LEVEL = 0.05;

/** What an item needs for the test: its attributes' count, its G-DINA estimates and their covariance, by rows. */
interface ComparableItem {
  readonly id: string;
  readonly attributeCount: number;
  readonly probabilities: readonly number[];
  readonly covariance: Float64Array;
}

/**
 * Tests, for each item with two or more attributes, whether a reduced model fits as well as the G-DINA, by the Wald
 * test of its constraints on the item's success probabilities, and selects a model for each item.
 *
 * @remarks
 * The DINA model sets the success probabilities of every latent group but the last equal, the DINO model those of
 * every group but the first. The A-CDM sets the interactions of the identity scale to 0, the LLM those of the logit
 * scale and the R-RUM those of the log scale: constraints R g(P) = 0, with the link g, where P are the G-DINA's success
 * probabilities and V their covariance matrix. The statistic is (R g(P))' (R G V G' R')⁺ (R g(P)), G the derivative of
 * the link and ⁺ the Moore-Penrose inverse, on as many degrees of freedom as the model has fewer parameters than the
 * G-DINA. The p-values are not adjusted for the number of tests. The selected model is the one with the largest p-value, as long as it is above 0.05, and the G-DINA otherwise
 * (Ma et al., 2016). An item with one attribute has no reduced form and is left out.
 *
 * @param calibration - A G-DINA calibration.
 * @param options - The reduced models to test.
 * @returns For each item with two or more attributes, the Wald tests of the reduced models and the model selected.
 * @throws `DataError` when the calibration or the options are invalid, or an item with two or more attributes was not
 * calibrated as the G-DINA or has no covariance.
 * @see {@link https://doi.org/10.1111/jedm.12022 | de la Torre and Lee (2013)}
 * @see {@link https://doi.org/10.1177/0146621615621717 | Ma, Iaconangelo and de la Torre (2016)}
 */
export function compareItemModels(
  calibration: CdmCalibration,
  options: ItemComparisonOptions = {},
): ItemModelComparison[] {
  checkOptions(options);
  const { models = REDUCED_MODELS } = options;
  checkModels(models);
  return comparableItems(calibration).map((item) => {
    const tests = models.map((model) => testModel(item, model));
    return { item: item.id, tests, selected: selectModel(tests) };
  });
}

/**
 * Checks the models to test.
 *
 * @throws `DataError` when they are not a list of reduced models, or one repeats.
 */
function checkModels(models: readonly ReducedCdmModel[]): void {
  checkArray(models, 'The option models must be an array of reduced models, such as ["dina", "acdm"]');
  if (models.length === 0) throw new DataError("The option models needs at least one model.");
  models.forEach((model, index) => {
    checkChoice(model, `Model ${index + 1} of the option models`, REDUCED_MODELS);
    if (models.indexOf(model) !== index) throw new DataError(`The option models lists "${model}" twice.`);
  });
}

/**
 * Checks a calibration, as from JSON, and returns its items with two or more attributes.
 *
 * @throws `DataError` when such an item is not G-DINA, or its probabilities or covariance are not those of its latent
 * groups.
 */
function comparableItems(calibration: CdmCalibration): ComparableItem[] {
  checkRecord(calibration, "A calibration must be an object from calibrateCdm");
  checkArray(calibration.items, "A calibration needs the array items");
  const comparable: ComparableItem[] = [];
  calibration.items.forEach((item, j) => {
    checkRecord(item, `Calibrated item ${j + 1} must be an object from calibrateCdm`);
    checkArray(item.attributes, `Calibrated item ${j + 1} needs the array attributes`);
    const attributeCount = item.attributes.length;
    if (attributeCount < 2) return;
    if (item.model !== "gdina") {
      throw new DataError(`Item "${item.id}" was calibrated as "${item.model}"; the test needs the G-DINA estimates.`);
    }
    const groups = 2 ** attributeCount;
    checkSuccessProbabilities(item.id, item.probabilities, groups);
    comparable.push({
      id: item.id,
      attributeCount,
      probabilities: item.probabilities,
      covariance: covarianceOf(item, groups),
    });
  });
  return comparable;
}

/**
 * Returns the covariance matrix of an item by rows.
 *
 * @throws `DataError` when the item lacks one, or it is not a finite `groups` by `groups` matrix.
 */
function covarianceOf(item: CdmCalibration["items"][number], groups: number): Float64Array {
  const rows: unknown = item.covariance;
  const valid =
    Array.isArray(rows) &&
    rows.length === groups &&
    rows.every((row: unknown) => Array.isArray(row) && row.length === groups && row.every(Number.isFinite));
  if (!valid) {
    throw new DataError(
      `Item "${item.id}" needs the ${groups} by ${groups} covariance of its success probabilities, from calibrateCdm with the option standardErrors "incomplete".`,
    );
  }
  return Float64Array.from((item.covariance ?? []).flat());
}

/** The link scale on which a reduced model's constraints are linear, and its derivative. */
interface Link {
  readonly value: (p: number) => number;
  readonly slope: (p: number) => number;
}

const IDENTITY: Link = { value: (p) => p, slope: () => 1 };
const LOGIT: Link = { value: (p) => Math.log(p / (1 - p)), slope: (p) => 1 / (p * (1 - p)) };
const LOG: Link = { value: Math.log, slope: (p) => 1 / p };
const LINKS: Readonly<Record<ReducedCdmModel, Link>> = {
  dina: IDENTITY,
  dino: IDENTITY,
  acdm: IDENTITY,
  llm: LOGIT,
  rrum: LOG,
};

/** Runs the Wald test of one reduced model on an item. */
function testModel(item: ComparableItem, model: ReducedCdmModel): ItemModelTest {
  const rows = constraints(model, item.attributeCount);
  const link = LINKS[model];
  const groups = item.probabilities.length;
  const values = item.probabilities.map(link.value);
  const slopes = item.probabilities.map(link.slope);
  const scaled = new Float64Array(groups * groups);
  for (let a = 0; a < groups; a++) {
    for (let b = 0; b < groups; b++) {
      scaled[a * groups + b] = (slopes[a] ?? 0) * (slopes[b] ?? 0) * (item.covariance[a * groups + b] ?? 0);
    }
  }
  const differences = rows.map((row) => row.reduce((sum, weight, g) => sum + weight * (values[g] ?? 0), 0));
  const variance = new Float64Array(rows.length * rows.length);
  rows.forEach((row, a) => {
    rows.forEach((other, b) => {
      variance[a * rows.length + b] = bilinear(row, scaled, other, groups);
    });
  });
  const inverse = pseudoInverse(variance, rows.length);
  let statistic = 0;
  differences.forEach((x, a) => {
    differences.forEach((y, b) => {
      statistic += x * (inverse[a * rows.length + b] ?? 0) * y;
    });
  });
  const degreesOfFreedom = rows.length;
  if (!Number.isFinite(statistic)) return { model, statistic: null, degreesOfFreedom, pValue: null };
  return { model, statistic, degreesOfFreedom, pValue: chiSquareUpperTail(Math.max(statistic, 0), degreesOfFreedom) };
}

/** Returns x' M y for the `size` by `size` matrix `matrix`, stored by rows. */
function bilinear(x: readonly number[], matrix: Float64Array, y: readonly number[], size: number): number {
  let sum = 0;
  for (let a = 0; a < size; a++) {
    const weight = x[a] ?? 0;
    if (weight === 0) continue;
    for (let b = 0; b < size; b++) sum += weight * (matrix[a * size + b] ?? 0) * (y[b] ?? 0);
  }
  return sum;
}

/**
 * Returns the constraints of a model on the transformed success probabilities of the latent groups of an item with
 * `attributeCount` attributes, one row of weights per group: the independent equalities, which count the degrees of
 * freedom.
 */
function constraints(model: ReducedCdmModel, attributeCount: number): number[][] {
  const groups = 2 ** attributeCount;
  if (model === "dina" || model === "dino") {
    const first = model === "dina" ? 0 : 1;
    return Array.from({ length: groups - 2 }, (_, r) => difference(groups, first + r, first + r + 1));
  }
  return interactionConstraints(attributeCount);
}

/** Returns the row of weights `weight` 1 at the group `plus` and -1 at the group `minus`. */
function difference(groups: number, plus: number, minus: number): number[] {
  const row = new Array<number>(groups).fill(0);
  row[plus] = 1;
  row[minus] = -1;
  return row;
}

/**
 * Returns, for each group that masters two or more attributes, the constraint that moving to it adds what moving to its
 * parts adds: P(none) + P(S) - P(first of S) - P(S without its first), the interaction of the first attribute with the
 * others being 0.
 */
function interactionConstraints(attributeCount: number): number[][] {
  const profiles = profilesOf(attributeCount);
  const indexOf = new Map(profiles.map((profile, g) => [profile.join(""), g]));
  const rows: number[][] = [];
  profiles.forEach((profile, g) => {
    const mastered = profile.flatMap((value, k) => (value === 1 ? [k] : []));
    const [first, ...rest] = mastered;
    if (first === undefined || rest.length === 0) return;
    const alone = Array.from({ length: attributeCount }, (_, k) => (k === first ? 1 : 0)).join("");
    const others = profile.map((value, k) => (k === first ? 0 : value)).join("");
    const row = new Array<number>(profiles.length).fill(0);
    row[0] = 1;
    row[g] = 1;
    row[indexOf.get(alone) ?? 0] = -1;
    row[indexOf.get(others) ?? 0] = -1;
    rows.push(row);
  });
  return rows;
}

/** Returns the reduced model with the largest p-value above the level, the first of equal ones, or the G-DINA. */
function selectModel(tests: readonly ItemModelTest[]): CdmModel {
  let selected: CdmModel = "gdina";
  let largest = SIGNIFICANCE_LEVEL;
  for (const { model, pValue } of tests) {
    if (pValue !== null && pValue > largest) {
      selected = model;
      largest = pValue;
    }
  }
  return selected;
}
