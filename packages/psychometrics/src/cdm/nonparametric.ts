import { checkChoice, checkOptions, type QMatrix } from "@measurekit/core";
import { checkResponseMatrix, type ResponseMatrix } from "../data/internal.js";
import { dichotomousRows } from "./calibration.js";
import { cdmDesign } from "./q-matrix.js";

/** Options of {@link nonparametricClassification}. */
export interface NonparametricOptions {
  /**
   * How an ideal response follows a profile: `"conjunctive"`, correct when every attribute of the item is mastered, as
   * the DINA; `"disjunctive"`, when any is, as the DINO.
   *
   * @defaultValue `"conjunctive"`
   */
  readonly rule?: "conjunctive" | "disjunctive";
}

const RULES: readonly ("conjunctive" | "disjunctive")[] = ["conjunctive", "disjunctive"];

/** A person's nonparametric classification. */
export interface NonparametricClassification {
  /** The person's id. */
  readonly id: string;
  /** The profile nearest the person's responses, the first in profile order on ties; attributes as in the design. */
  readonly profile: readonly number[];
  /** Its Hamming distance: the number of answered items whose response differs from the ideal one. */
  readonly distance: number;
  /** The number of profiles at that distance; above 1, the classification is a tie. */
  readonly ties: number;
}

/**
 * Classifies persons into attribute profiles without calibration, as NPCD's `AlphaNP` with the Hamming distance.
 *
 * @remarks
 * Each profile's ideal responses follow the Q-matrix by `rule`, and a person takes the profile whose ideal responses
 * differ from the person's on the fewest answered items. It needs no item parameters, so it suits small samples, such
 * as one classroom. Attributes are in order of first appearance in the Q-matrix; missing responses are skipped.
 *
 * @param matrix - The responses of the persons, dichotomous.
 * @param qMatrix - The attributes of each item, by item id.
 * @param options - The rule that combines the attributes of an item.
 * @returns For each person, the profile, its distance to the responses and the number of profiles tied with it.
 * @throws `DataError` when the matrix or the Q-matrix is invalid, the rule is unknown, or a response is neither 0
 * nor 1.
 * @see {@link https://doi.org/10.1007/s00357-013-9132-9 | Chiu and Douglas (2013)}
 */
export function nonparametricClassification(
  matrix: ResponseMatrix,
  qMatrix: QMatrix,
  options: NonparametricOptions = {},
): NonparametricClassification[] {
  checkOptions(options);
  const { rule = "conjunctive" } = options;
  checkChoice(rule, "The option rule", RULES);
  checkResponseMatrix(matrix);
  const design = cdmDesign(matrix.itemIds, qMatrix);
  const data = dichotomousRows(matrix);
  const ideal = design.profiles.map((profile) =>
    design.itemAttributes.map((indices) => {
      const mastered = indices.filter((k) => profile[k] === 1).length;
      return (rule === "conjunctive" ? mastered === indices.length : mastered > 0) ? 1 : 0;
    }),
  );
  return matrix.personIds.map((id, person) => {
    const distances = ideal.map((responses) => {
      let distance = 0;
      responses.forEach((expected, j) => {
        const response = data.responses[person * data.itemCount + j] ?? -1;
        if (response >= 0 && response !== expected) distance++;
      });
      return distance;
    });
    let best = 0;
    let ties = 0;
    distances.forEach((distance, c) => {
      if (distance < (distances[best] ?? 0)) {
        best = c;
        ties = 0;
      }
      if (distance === distances[best]) ties++;
    });
    return { id, profile: design.profiles[best] ?? [], distance: distances[best] ?? 0, ties };
  });
}
