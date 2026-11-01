import { type CompiledStructure, compile, flipElement, holds, type KnowledgeStructure, keyOf } from "./structure.js";

/**
 * Returns, for each element, whether adding it to any state leaves a state: the structure is forward graded at the
 * element.
 *
 * @remarks
 * It is `is.forward.graded` of the pks package. A state that holds the element already stays as it is. The structure
 * need not be a knowledge space; a knowledge space is forward graded at every element.
 *
 * @throws `DataError` when the structure is malformed.
 * @see {@link https://doi.org/10.1007/978-3-642-01039-2 | Falmagne and Doignon (2011), chapter 2}
 */
export function forwardGraded(structure: KnowledgeStructure): Readonly<Record<string, boolean>> {
  return gradedAt(compile(structure), true);
}

/**
 * Returns, for each element, whether removing it from any state leaves a state: the structure is backward graded at the
 * element.
 *
 * @remarks
 * It is `is.backward.graded` of the pks package. A state without the element stays as it is.
 *
 * @throws `DataError` when the structure is malformed.
 * @see {@link https://doi.org/10.1007/978-3-642-01039-2 | Falmagne and Doignon (2011), chapter 2}
 */
export function backwardGraded(structure: KnowledgeStructure): Readonly<Record<string, boolean>> {
  return gradedAt(compile(structure), false);
}

/** Returns whether each element, put into every state (`adding`) or taken out of it, gives states of the structure. */
function gradedAt(compiled: CompiledStructure, adding: boolean): Record<string, boolean> {
  const { words, bits, stateOf } = compiled;
  const probe = new Uint32Array(words);
  return Object.fromEntries(
    compiled.domain.map((element, e) => {
      let graded = true;
      for (let s = 0; graded && s < compiled.count; s++) {
        // A state that already holds the element, or lacks it, is its own result.
        if (holds(compiled, s, e) === adding) continue;
        probe.set(bits.subarray(s * words, (s + 1) * words));
        flipElement(probe, e);
        graded = stateOf.has(keyOf(probe, 0, words));
      }
      return [element, graded];
    }),
  );
}
