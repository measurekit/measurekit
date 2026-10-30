import { DataError } from "@measurekit/core";
import { checkItemSkills, type QMatrix } from "../data/q-matrix.js";
import { compareStates, compile, holds, type KnowledgeStructure, seal } from "./structure.js";

/**
 * Returns the item structure delineated by a structure over skills: an item is in a state when all its skills are
 * (the conjunctive model). Items are the Q-matrix's, in its order; equal item states appear once. The structure is
 * frozen, and the functions of this module trust it instead of checking it again.
 *
 * @throws `DataError` when the structure or the Q-matrix is malformed, or an item has no skills or a skill outside the
 * structure's domain.
 */
export function delineate(structure: KnowledgeStructure, qMatrix: QMatrix): KnowledgeStructure {
  const compiled = compile(structure);
  checkItemSkills(qMatrix);
  const domain = Object.keys(qMatrix);
  const skillsOf = domain.map((item) => {
    const skills = qMatrix[item] ?? [];
    if (skills.length === 0) throw new DataError(`Item "${item}" needs at least one skill to be delineated.`);
    return skills.map((skill) => {
      const e = compiled.index.get(skill);
      if (e === undefined) {
        throw new DataError(`Item "${item}" has the skill "${skill}", which is not in the structure.`);
      }
      return e;
    });
  });
  const seen = new Set<string>();
  const states: number[][] = [];
  for (let s = 0; s < compiled.count; s++) {
    const state = domain.flatMap((_, i) => ((skillsOf[i] ?? []).every((e) => holds(compiled, s, e)) ? [i] : []));
    const key = state.join(",");
    if (!seen.has(key)) {
      seen.add(key);
      states.push(state);
    }
  }
  return seal({ domain, states: states.sort(compareStates).map((state) => state.map((i) => domain[i] ?? "")) });
}
