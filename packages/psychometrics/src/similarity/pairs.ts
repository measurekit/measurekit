import {
  checkArray,
  checkIntegerFrom,
  checkOptions,
  checkRecord,
  DataError,
  describe,
  type FieldNames,
} from "@measurekit/core";
import { indexIds } from "../data/internal.js";

/**
 * Two persons by id. A directional index, such as ω or K, takes the first as the suspected copier and the second as
 * the source.
 */
export type PersonPair = readonly [first: string, second: string];

/** Options of {@link selectPairs}. */
export interface SelectPairsOptions {
  /**
   * The most persons that share a value and still form pairs, an integer from 2. A value that more persons share, such
   * as a hall or the network of a whole school, says little of contact: it forms no pairs, and the selection names it.
   *
   * @defaultValue `200`, whose persons form 19,900 pairs
   */
  readonly maxGroup?: number;
  /**
   * The most pairs, an integer from 1: a selection of more throws.
   *
   * @defaultValue `1_000_000`
   */
  readonly maxPairs?: number;
}

const SELECT_PAIRS_OPTIONS: FieldNames<SelectPairsOptions> = { maxGroup: true, maxPairs: true };

/** A value of a context that more persons share than `maxGroup`, which forms no pairs. */
export interface SkippedGroup {
  /** The name of the context, such as `"room"`. */
  readonly context: string;
  /** The value the persons share. */
  readonly value: string;
  /** The number of persons who share it. */
  readonly persons: number;
}

/** The pairs of persons who share a value of a context, and the values that formed none. */
export interface PairSelection {
  /**
   * Each pair once, its persons in the order of the ids, and the pairs by their first person, then their second.
   */
  readonly pairs: readonly PersonPair[];
  /** The values shared by more than `maxGroup` persons, by context, then by the first person who has the value. */
  readonly skipped: readonly SkippedGroup[];
}

const DEFAULT_MAX_GROUP = 200;
/** 2,000 rooms of 30 persons form 870,000 pairs. */
const DEFAULT_MAX_PAIRS = 1_000_000;
/** The most persons whose pairs a double numbers exactly, as first × persons + second: √(2⁵³). */
const MAX_SELECTED_PERSONS = 94_906_265;

/**
 * Returns the pairs of persons who share a value of any of the contexts, such as a room, a school, a network or a start
 * minute, each pair once.
 *
 * @remarks
 * A screening of answer similarity compares the persons who could have seen or told each other's answers, never every
 * pair: 64,160 persons make 2.06 × 10⁹ pairs. A value of a context forms the pairs of the persons who have it; a person
 * with `null` has no value of that context. A value shared by more than `maxGroup` persons forms none and is listed in
 * `skipped`. Pairs that two contexts form are kept once. Time and memory grow with the pairs: 1,000,000 pairs take
 * about 150 MB.
 *
 * The indices of answer similarity take the pairs as they are: ω and K, which are directional, take the first person of
 * a pair as the suspected copier. To screen both directions, pass each pair both ways.
 *
 * @param personIds - The ids of the persons.
 * @param contexts - The values of each context, by its name, as one value per person in the order of `personIds`: a
 * string, or `null` for none.
 * @param options - The largest group that forms pairs and the most pairs.
 * @returns The pairs and the values that formed none.
 * @throws `DataError` when the ids are not unique strings, a context does not have one string or `null` per person, an
 * option is out of range, or the contexts form more than `maxPairs` pairs.
 * @see {@link https://doi.org/10.4324/9781315743097 | Cizek and Wollack (2017)}
 *
 * @example
 * ```ts
 * import { selectPairs } from "@measurekit/psychometrics/similarity";
 *
 * const selection = selectPairs(["ana", "ben", "cy", "dee"], {
 *   room: ["r1", "r1", "r2", "r2"],
 *   network: ["n1", null, "n1", "n2"],
 * });
 * selection.pairs; // [["ana", "ben"], ["ana", "cy"], ["cy", "dee"]]
 * selectPairs(["ana", "ben", "cy"], { school: ["s1", "s1", "s1"] }, { maxGroup: 2 }).skipped;
 * // [{ context: "school", value: "s1", persons: 3 }]
 * ```
 */
export function selectPairs(
  personIds: readonly string[],
  contexts: Readonly<Record<string, readonly (string | null)[]>>,
  options: SelectPairsOptions = {},
): PairSelection {
  checkArray(personIds, "The person ids must be an array");
  indexIds("person", personIds);
  checkRecord(contexts, "The contexts must be an object of each context's values by person");
  checkOptions(options, SELECT_PAIRS_OPTIONS);
  const { maxGroup = DEFAULT_MAX_GROUP, maxPairs = DEFAULT_MAX_PAIRS } = options;
  checkIntegerFrom(maxGroup, "The option maxGroup", 2);
  checkIntegerFrom(maxPairs, "The option maxPairs", 1);
  const count = personIds.length;
  if (count > MAX_SELECTED_PERSONS) {
    throw new DataError(`A selection of pairs takes at most ${MAX_SELECTED_PERSONS} persons, found ${count}.`);
  }
  const codes = createPairCodes(maxPairs);
  const skipped: SkippedGroup[] = [];
  for (const [context, values] of Object.entries(contexts)) {
    for (const [value, members] of groupsOf(context, values, personIds)) {
      if (members.length > maxGroup) {
        skipped.push({ context, value, persons: members.length });
        continue;
      }
      addGroupPairs(codes, members, count);
    }
  }
  const pairs = Array.from(codes.finish(), (code): PersonPair => {
    const first = Math.floor(code / count);
    return [personIds[first] ?? "", personIds[code - first * count] ?? ""];
  });
  return { pairs, skipped };
}

/**
 * Returns the persons who have each value of a context, in the order of the persons, by value in the order of the first
 * person who has it.
 *
 * @throws `DataError` when the context does not have one string or `null` per person.
 */
function groupsOf(context: string, list: readonly unknown[], personIds: readonly string[]): Map<string, number[]> {
  checkArray(list, `The values of the context "${context}" must be an array`);
  if (list.length !== personIds.length) {
    throw new DataError(`The context "${context}" has ${list.length} values for ${personIds.length} persons.`);
  }
  const groups = new Map<string, number[]>();
  for (let person = 0; person < list.length; person++) {
    const value = list[person];
    if (value === null) continue;
    if (typeof value !== "string") {
      throw new DataError(
        `The value of the context "${context}" of person "${personIds[person]}" must be a string or null, found ${describe(value)}.`,
      );
    }
    const group = groups.get(value);
    if (group === undefined) groups.set(value, [person]);
    else group.push(person);
  }
  return groups;
}

/** Adds the pairs of the persons of a group, each as first × persons + second. */
function addGroupPairs(codes: PairCodes, members: readonly number[], persons: number): void {
  for (let a = 0; a < members.length; a++) {
    const first = (members[a] ?? 0) * persons;
    for (let b = a + 1; b < members.length; b++) codes.add(first + (members[b] ?? 0));
  }
}

/** The pairs selected so far, each as first × persons + second, which `finish` sorts with their repeats removed. */
interface PairCodes {
  add(code: number): void;
  finish(): Float64Array;
}

/**
 * Returns a set of pair codes that holds at most `maxPairs` distinct codes and throws past them. The codes go into an
 * array that is sorted and freed of repeats whenever it reaches twice `maxPairs`, so that contexts that form the same
 * pairs take no more memory than that.
 */
function createPairCodes(maxPairs: number): PairCodes {
  let codes = new Float64Array(1024);
  let size = 0;
  const compact = (): void => {
    const sorted = codes.subarray(0, size).sort();
    let unique = 0;
    for (let k = 0; k < size; k++) {
      const code = sorted[k] ?? 0;
      if (unique === 0 || code !== sorted[unique - 1]) sorted[unique++] = code;
    }
    size = unique;
    if (size > maxPairs) {
      throw new DataError(
        `The contexts form more than ${maxPairs} pairs, the option maxPairs: pass contexts of fewer persons, a lower maxGroup or a higher maxPairs.`,
      );
    }
  };
  return {
    add(code) {
      if (size === codes.length) {
        if (size >= 2 * maxPairs) compact();
        if (size === codes.length) {
          const grown = new Float64Array(2 * codes.length);
          grown.set(codes);
          codes = grown;
        }
      }
      codes[size++] = code;
    },
    finish() {
      compact();
      return codes.slice(0, size);
    },
  };
}

/**
 * Reads pairs of person ids against the ids of a response matrix.
 *
 * @returns The person indices of the pairs: the first and second person of pair k at 2k and 2k + 1.
 * @throws `DataError` when the pairs are not an array of two distinct person ids of the matrix each.
 */
export function readPairs(personIds: readonly string[], pairs: readonly PersonPair[]): Int32Array {
  checkArray(pairs, "The pairs must be an array of pairs of person ids");
  const indices = indexIds("person", personIds);
  const read = new Int32Array(2 * pairs.length);
  for (let k = 0; k < pairs.length; k++) {
    const pair: unknown = pairs[k];
    if (!(Array.isArray(pair) && pair.length === 2)) {
      throw new DataError(
        `Pair ${k + 1} must be an array of two person ids, found ${Array.isArray(pair) ? `an array of length ${pair.length}` : describe(pair)}.`,
      );
    }
    for (let side = 0; side < 2; side++) {
      const index = indices.get(pair[side]);
      if (index === undefined) {
        throw new DataError(`Pair ${k + 1} names ${describe(pair[side])}, which is not a person id of the matrix.`);
      }
      read[2 * k + side] = index;
    }
    if (read[2 * k] === read[2 * k + 1]) throw new DataError(`Pair ${k + 1} names the person "${pair[0]}" twice.`);
  }
  return read;
}
