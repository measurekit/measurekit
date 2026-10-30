import type { ResponseMatrix } from "../data/response-matrix.js";

/** The persons who answered every item, with their responses and total scores. */
export interface CompleteCases {
  /** Indices of the persons in the response matrix. */
  readonly persons: readonly number[];
  /** Number of items. */
  readonly itemCount: number;
  /** Responses by person, then item. */
  readonly responses: Float64Array;
  /** Total score of each person. */
  readonly totals: Float64Array;
}

/** Returns the persons of a checked response matrix who answered every item. */
export function completeCases(matrix: ResponseMatrix): CompleteCases {
  const itemCount = matrix.itemIds.length;
  const persons = matrix.responses.flatMap((row, person) => (row.includes(null) ? [] : [person]));
  const responses = new Float64Array(persons.length * itemCount);
  const totals = new Float64Array(persons.length);
  persons.forEach((person, row) => {
    matrix.responses[person]?.forEach((response, item) => {
      responses[row * itemCount + item] = response ?? 0;
      totals[row] = (totals[row] ?? 0) + (response ?? 0);
    });
  });
  return { persons, itemCount, responses, totals };
}

/** Returns each complete case's summed responses to `items`: the item score for one item, a half score for a half. */
export function itemScores(cases: CompleteCases, items: readonly number[]): Float64Array {
  const scores = new Float64Array(cases.persons.length);
  for (let row = 0; row < scores.length; row++) {
    for (const item of items) scores[row] = (scores[row] ?? 0) + (cases.responses[row * cases.itemCount + item] ?? 0);
  }
  return scores;
}
