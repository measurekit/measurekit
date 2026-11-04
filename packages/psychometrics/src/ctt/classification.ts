import {
  checkFields,
  checkFinite,
  checkOptions,
  checkRecord,
  checkUnitInterval,
  DataError,
  type FieldNames,
} from "@measurekit/core";

/** The difficulty class of an item by its p-value. */
export type DifficultyClass = "difficult" | "moderate" | "easy";

/** Bounds of the difficulty classes. */
export interface DifficultyThresholds {
  /** p-values below this are `difficult`. */
  readonly difficult: number;
  /** p-values above this are `easy`. */
  readonly easy: number;
}

const DIFFICULTY_THRESHOLD_FIELDS: FieldNames<DifficultyThresholds> = { difficult: true, easy: true };

/** The common difficulty bounds: below 0.30 difficult, above 0.70 easy, moderate between them inclusive. */
export const DIFFICULTY_THRESHOLDS: DifficultyThresholds = { difficult: 0.3, easy: 0.7 };

/** The discrimination class of an item, after Ebel's guidelines. */
export type DiscriminationClass = "very-good" | "good" | "marginal" | "poor";

/** Lower bounds of the discrimination classes; values below `marginal` are `poor`. */
export interface DiscriminationThresholds {
  /** Values from this up are `very-good`. */
  readonly veryGood: number;
  /** Values from this up are `good`. */
  readonly good: number;
  /** Values from this up are `marginal`. */
  readonly marginal: number;
}

const DISCRIMINATION_THRESHOLD_FIELDS: FieldNames<DiscriminationThresholds> = {
  veryGood: true,
  good: true,
  marginal: true,
};

/**
 * Ebel's bounds for the discrimination index D: 0.40 and up very good, 0.30 good, 0.20 marginal, below that poor. Often
 * applied to item-total correlations too.
 *
 * @see {@link https://openlibrary.org/works/OL4471505W | Ebel and Frisbie (1991)}
 */
export const DISCRIMINATION_THRESHOLDS: DiscriminationThresholds = { veryGood: 0.4, good: 0.3, marginal: 0.2 };

/** Options of {@link classifyDifficulty}. */
export interface DifficultyClassificationOptions {
  /**
   * The p-values that separate the classes.
   *
   * @defaultValue {@link DIFFICULTY_THRESHOLDS}
   */
  readonly thresholds?: DifficultyThresholds;
}

const DIFFICULTY_CLASSIFICATION_OPTIONS: FieldNames<DifficultyClassificationOptions> = { thresholds: true };

/**
 * Classifies a p-value from 0 to 1.
 *
 * @param p - The p-value of an item.
 * @param options - The p-values that separate the classes.
 * @returns The class.
 * @throws `DataError` when `p` is outside [0, 1], the options are not an object or the thresholds are not numbers
 * ordered within it.
 */
export function classifyDifficulty(p: number, options: DifficultyClassificationOptions = {}): DifficultyClass {
  checkOptions(options, DIFFICULTY_CLASSIFICATION_OPTIONS);
  const { thresholds = DIFFICULTY_THRESHOLDS } = options;
  checkRecord(thresholds, "Difficulty thresholds must be an object { difficult, easy }");
  checkFields(thresholds, DIFFICULTY_THRESHOLD_FIELDS, "The difficulty thresholds");
  const { difficult, easy } = thresholds;
  checkUnitInterval(difficult, "The difficult threshold");
  checkUnitInterval(easy, "The easy threshold");
  if (!(difficult <= easy)) {
    throw new DataError(`Difficulty thresholds must satisfy difficult ≤ easy, found ${difficult} and ${easy}.`);
  }
  checkUnitInterval(p, "A p-value");
  if (p < difficult) return "difficult";
  return p > easy ? "easy" : "moderate";
}

/** Options of {@link classifyDiscrimination}. */
export interface DiscriminationClassificationOptions {
  /**
   * The values that separate the classes.
   *
   * @defaultValue {@link DISCRIMINATION_THRESHOLDS}
   */
  readonly thresholds?: DiscriminationThresholds;
}

const DISCRIMINATION_CLASSIFICATION_OPTIONS: FieldNames<DiscriminationClassificationOptions> = { thresholds: true };

/**
 * Classifies a discrimination value, such as the discrimination index D or an item-total correlation.
 *
 * @param value - The discrimination index or correlation.
 * @param options - The values that separate the classes.
 * @returns The class.
 * @throws `DataError` when `value` is not a finite number, the options are not an object or the thresholds are not
 * finite numbers in order.
 */
export function classifyDiscrimination(
  value: number,
  options: DiscriminationClassificationOptions = {},
): DiscriminationClass {
  checkOptions(options, DISCRIMINATION_CLASSIFICATION_OPTIONS);
  const { thresholds = DISCRIMINATION_THRESHOLDS } = options;
  checkRecord(thresholds, "Discrimination thresholds must be an object { veryGood, good, marginal }");
  checkFields(thresholds, DISCRIMINATION_THRESHOLD_FIELDS, "The discrimination thresholds");
  const { veryGood, good, marginal } = thresholds;
  checkFinite(veryGood, "The very good threshold");
  checkFinite(good, "The good threshold");
  checkFinite(marginal, "The marginal threshold");
  if (!(marginal <= good && good <= veryGood)) {
    throw new DataError(
      `Discrimination thresholds must satisfy marginal ≤ good ≤ very good, found ${marginal}, ${good} and ${veryGood}.`,
    );
  }
  checkFinite(value, "A discrimination value");
  if (value >= veryGood) return "very-good";
  if (value >= good) return "good";
  return value >= marginal ? "marginal" : "poor";
}
