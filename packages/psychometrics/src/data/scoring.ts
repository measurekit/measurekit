import { checkArray, checkOptions, DataError, describe } from "@measurekit/core";
import { type ResponseMatrix, type ResponseMatrixOptions, responseMatrixOfRows } from "./response-matrix.js";

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

/**
 * Scores multiple-choice responses against a key: 1 for the keyed option, 0 for any other option, and `null` when no
 * option was chosen.
 *
 * @param choices - Chosen option labels by person, then item; `null` when no option was chosen.
 * @param key - The keyed option label of each item.
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
 * matrix.responses; // [[1, 1], [0, 1], [1, 0]]
 * ```
 */
export function scoreResponses(
  choices: readonly (readonly (string | null)[])[],
  key: readonly string[],
  options: ScoringOptions = {},
): ScoredResponses {
  checkArray(choices, "The choices must be an array of rows of option labels");
  checkArray(key, "The key must be an array of option labels");
  checkOptions(options);
  choices.forEach((row, person) => {
    if (!Array.isArray(row)) throw new DataError(`Row ${person + 1} of the choices must be an array of option labels.`);
    if (row.length !== key.length) {
      throw new DataError(`Row ${person + 1} has ${row.length} choices, but the key has ${key.length} items.`);
    }
  });
  const { optionLabels } = options;
  if (optionLabels !== undefined) checkArray(optionLabels, "The option labels must be an array");
  const labels = new Set<string>();
  for (const label of optionLabels ?? []) if (isOptionLabel(label)) labels.add(label);
  if (optionLabels !== undefined && new Set(optionLabels).size !== optionLabels.length) {
    throw new DataError("Option labels must be unique.");
  }
  // Without option labels, the labels of the key and the choices are the option labels.
  const accepts = (label: unknown): boolean => {
    if (typeof label === "string" && labels.has(label)) return true;
    if (optionLabels !== undefined || !isOptionLabel(label)) return false;
    labels.add(label);
    return true;
  };
  for (const [item, label] of key.entries()) {
    if (!accepts(label)) throw unexpectedLabel(label, `the key of item ${item + 1}`);
  }
  const responses = choices.map((row, person) => {
    const scored: (number | null)[] = [];
    for (let item = 0; item < row.length; item++) {
      const choice = row[item];
      if (choice === null) scored.push(null);
      // The labels of the key are accepted, so a choice equal to the key needs no look-up.
      else if (choice === key[item]) scored.push(1);
      else if (accepts(choice)) scored.push(0);
      else throw unexpectedLabel(choice, `row ${person + 1}, item ${item + 1}`);
    }
    return scored;
  });
  return {
    ...responseMatrixOfRows(responses, options),
    choices: choices.map((row) => row.slice()),
    key: [...key],
    optionLabels: optionLabels === undefined ? [...labels].sort() : [...optionLabels],
  };
}

function isOptionLabel(label: unknown): label is string {
  return typeof label === "string" && label !== "";
}

function unexpectedLabel(label: unknown, place: string): DataError {
  return new DataError(`Expected an option label in ${place}, found ${describe(label)}.`);
}
