import { checkArray, checkOptions, DataError, describe, type FieldNames } from "@measurekit/core";
import { buildResponseMatrixOfRows, type ResponseMatrix, type ResponseMatrixOptions } from "./response-matrix.js";

/** Multiple-choice responses scored against a key, with the chosen options kept for distractor analysis. */
export interface ScoredResponses extends ResponseMatrix {
  /** Chosen option labels by person, then item; `null` when no option was chosen. */
  readonly choices: readonly (readonly (string | null)[])[];
  /** The keyed option label of each item. */
  readonly key: readonly string[];
  /** The option labels, in display order. */
  readonly optionLabels: readonly string[];
}

/** Ids and option labels of scored responses. */
export interface ScoringOptions extends ResponseMatrixOptions {
  /**
   * The option labels, in display order. Choices and keys must be among them.
   *
   * @defaultValue the labels found in the choices and the key, sorted
   */
  readonly optionLabels?: readonly string[];
}

const SCORING_OPTIONS: FieldNames<ScoringOptions> = { optionLabels: true, personIds: true, itemIds: true, times: true };

/**
 * Scores multiple-choice responses against a key: 1 for the keyed option, 0 for any other option, and `null` when no
 * option was chosen.
 *
 * @param choices - Chosen option labels by person, then item; `null` when no option was chosen.
 * @param key - The keyed option label of each item.
 * @param options - The labels of the options, the ids of the persons and items, and the response times.
 * @returns The scores as a response matrix, with the choices and the key.
 * @throws `DataError` when the choices do not form a matrix with one column per key, a label is not a non-empty string
 * among `optionLabels`, the option labels repeat, or the ids do not match.
 *
 * @example
 * ```ts
 * import { scoreResponses } from "@measurekit/psychometrics";
 *
 * const matrix = scoreResponses(
 *   [
 *     ["A", "C"],
 *     ["B", "C"],
 *     ["A", "D"],
 *   ],
 *   ["A", "C"],
 * );
 * matrix.responses; // => [[1, 1], [0, 1], [1, 0]]
 * ```
 */
export function scoreResponses(
  choices: readonly (readonly (string | null)[])[],
  key: readonly string[],
  options: ScoringOptions = {},
): ScoredResponses {
  checkArray(choices, "The choices must be an array of rows of option labels");
  checkArray(key, "The key must be an array of option labels");
  checkOptions(options, SCORING_OPTIONS);
  choices.forEach((row, person) => {
    if (!Array.isArray(row)) throw new DataError(`Row ${person + 1} of the choices must be an array of option labels.`);
    if (row.length !== key.length) {
      throw new DataError(`Row ${person + 1} has ${row.length} choices, but the key has ${key.length} items.`);
    }
  });
  const { optionLabels } = options;
  if (optionLabels !== undefined) {
    checkArray(optionLabels, "The option labels must be an array");
    for (const [index, label] of optionLabels.entries()) {
      if (!isOptionLabel(label)) throw unexpectedLabel(label, `option label ${index + 1}`);
    }
  }
  const accepted = acceptedLabels(optionLabels);
  if (optionLabels !== undefined && new Set(optionLabels).size !== optionLabels.length) {
    throw new DataError("Option labels must be unique.");
  }
  for (const [item, label] of key.entries()) {
    if (!accepted.accepts(label)) throw unexpectedLabel(label, `the key of item ${item + 1}`);
  }
  return {
    ...buildResponseMatrixOfRows(scoreRows(choices, key, accepted.accepts), options),
    choices: choices.map((row) => row.slice()),
    key: [...key],
    optionLabels: optionLabels === undefined ? [...accepted.labels].sort() : [...optionLabels],
  };
}

/**
 * The labels that a scan covers before the set is asked: a few options take nearly every choice, and a scan of them
 * costs less than a hash look-up.
 */
const SCANNED_LABELS = 8;

/**
 * Returns the option labels that scoring accepts, and a test of a label. Without option labels, the labels of the key and
 * the choices are the option labels, so the test adds each non-empty string it meets.
 */
function acceptedLabels(optionLabels: readonly string[] | undefined): {
  readonly labels: ReadonlySet<string>;
  readonly accepts: (label: unknown) => boolean;
} {
  const labels = new Set<string>();
  const scanned: string[] = [];
  const add = (label: string): void => {
    labels.add(label);
    if (scanned.length < SCANNED_LABELS) scanned.push(label);
  };
  for (const label of optionLabels ?? []) if (!labels.has(label)) add(label);
  const accepts = (label: unknown): boolean => {
    for (let i = 0; i < scanned.length; i++) if (scanned[i] === label) return true;
    if (typeof label === "string" && labels.has(label)) return true;
    if (optionLabels !== undefined || !isOptionLabel(label)) return false;
    add(label);
    return true;
  };
  return { labels, accepts };
}

/** Returns the scores of the choices: 1 for the keyed option, 0 for another option and `null` for no choice. */
function scoreRows(
  choices: readonly (readonly (string | null)[])[],
  key: readonly string[],
  accepts: (label: unknown) => boolean,
): (number | null)[][] {
  const rows = new Array<(number | null)[]>(choices.length);
  for (let person = 0; person < choices.length; person++) {
    const row = choices[person] ?? [];
    const scored = new Array<number | null>(row.length);
    for (let item = 0; item < row.length; item++) {
      const choice = row[item];
      if (choice === null) scored[item] = null;
      // The labels of the key are accepted, so a choice equal to the key needs no look-up.
      else if (choice === key[item]) scored[item] = 1;
      else if (accepts(choice)) scored[item] = 0;
      else throw unexpectedLabel(choice, `row ${person + 1}, item ${item + 1}`);
    }
    rows[person] = scored;
  }
  return rows;
}

function isOptionLabel(label: unknown): label is string {
  return typeof label === "string" && label !== "";
}

function unexpectedLabel(label: unknown, place: string): DataError {
  return new DataError(`Expected an option label in ${place}, found ${describe(label)}.`);
}
