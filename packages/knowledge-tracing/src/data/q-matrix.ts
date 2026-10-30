import { DataError, describe, isRecord } from "@measurekit/core";
import { checkSkills } from "./event.js";
import { entry } from "./record.js";

/**
 * The skills of each item, by item id, such as `{ q1: ["fractions"], q2: ["fractions", "borrow"] }`. The `QMatrix` of
 * `@measurekit/psychometrics` has the same shape, so a Q-matrix moves between the two packages unchanged; a row may be
 * empty here, as an item may practice no skills.
 */
// biome-ignore lint/style/useNamingConvention: Q-matrix is the name of this matrix in the field.
export type QMatrix = Readonly<Record<string, readonly string[]>>;

/** Checks a Q-matrix: an object whose every row is a list of distinct skill ids. */
export function checkItemSkills(qMatrix: QMatrix): void {
  if (!isRecord(qMatrix)) {
    throw new DataError(`A Q-matrix must be an object of skill lists by item id, found ${describe(qMatrix)}.`);
  }
  for (const [item, skills] of Object.entries(qMatrix)) checkSkills(skills, `The Q-matrix row of item "${item}"`);
}

/**
 * Returns the skills of an item in a Q-matrix.
 *
 * @throws `DataError` when the Q-matrix lacks the item.
 */
export function skillsOf(qMatrix: QMatrix, item: string, label: string): readonly string[] {
  const skills = entry(qMatrix, item);
  if (skills === undefined) throw new DataError(`${label} has no skills, and the Q-matrix lacks its item "${item}".`);
  return skills;
}
