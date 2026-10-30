import { checkFinite, checkRecord, checkUnitInterval, DataError } from "@measurekit/core";

/** The difficulty class of an item by its p-value. */
export type DifficultyClass = "difficult" | "moderate" | "easy";

/** Bounds of the difficulty classes. */
export interface DifficultyThresholds {
  /** p-values below this are `difficult`. */
  readonly difficult: number;
  /** p-values above this are `easy`. */
  readonly easy: number;
}

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

/**
 * Ebel's bounds for the discrimination index D: 0.40 and up very good, 0.30 good, 0.20 marginal, below that poor. Often
 * applied to item-total correlations too.
 *
 * @see {@link https://openlibrary.org/works/OL4471505W | Ebel and Frisbie (1991)}
 */
export const DISCRIMINATION_THRESHOLDS: DiscriminationThresholds = { veryGood: 0.4, good: 0.3, marginal: 0.2 };

/**
 * Classifies a p-value from 0 to 1.
 *
 * @throws `DataError` when `p` is outside [0, 1] or the thresholds are not numbers ordered within it.
 */
export function classifyDifficulty(
  p: number,
  thresholds: DifficultyThresholds = DIFFICULTY_THRESHOLDS,
): DifficultyClass {
  checkRecord(thresholds, "Difficulty thresholds must be an object { difficult, easy }");
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

/**
 * Classifies a discrimination value, such as the discrimination index D or an item-total correlation.
 *
 * @throws `DataError` when `value` is not a finite number or the thresholds are not finite numbers in order.
 */
export function classifyDiscrimination(
  value: number,
  thresholds: DiscriminationThresholds = DISCRIMINATION_THRESHOLDS,
): DiscriminationClass {
  checkRecord(thresholds, "Discrimination thresholds must be an object { veryGood, good, marginal }");
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
