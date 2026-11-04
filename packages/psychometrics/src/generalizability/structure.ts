import { checkArray, checkRecordById, DataError, describe, isNonEmptyString } from "@measurekit/core";

/**
 * The most facets, the object of measurement included, that a design may have: GENOVA's and urGENOVA's six. The effects
 * grow as 2^k and the marginal sums of a balanced design as 3^k for facets of two elements, so that 16 facets take
 * minutes and 20 tens of gigabytes; the unbalanced estimator reads the scores once for each pair of the effects.
 */
const MAX_FACETS = 6;

/** One effect of a balanced random-effects design, as sets of facets in bitmasks over the design's facets. */
export interface Effect {
  /** The facets that index the effect: its own and those it is nested in. */
  readonly mask: number;
  /** The facets the effect is a deviation in, such as the rater and the item of the effect of ri:p. */
  readonly own: number;
  /** The facets the effect is nested in, such as the person of r:p. */
  readonly nestedIn: number;
}

/** The facets of a design and how they nest: the object of measurement first, then the other facets in their order. */
export interface Structure {
  /** The facet names, the object of measurement first. */
  readonly names: readonly string[];
  /** For each facet, the facets it is nested in, directly or through others. */
  readonly ancestors: readonly number[];
}

/** Returns the indices of the facets in a mask, in design order. */
export function facetsOf(mask: number): number[] {
  const indices: number[] = [];
  for (let facet = 0; mask >> facet > 0; facet++) if ((mask >> facet) & 1) indices.push(facet);
  return indices;
}

/** Returns the names of the facets in a mask, in design order. */
export function namesOf(structure: Structure, mask: number): string[] {
  return facetsOf(mask).map((facet) => structure.names[facet] ?? "");
}

/**
 * Checks the facets and the nesting of a design and returns its structure.
 *
 * @param subject - Names the nesting in error messages, such as `The option nested of the design`.
 * @throws `DataError` when the object is not a non-empty string, the facets are not an array of distinct non-empty
 * strings other than the object, there are none or more than 6 with the object, or the nesting is not an object that
 * names facets of the design, nests the object or a facet in itself, or is circular.
 */
export function buildStructure(
  object: string,
  facets: readonly string[],
  nested: Readonly<Record<string, readonly string[]>>,
  subject: string,
): Structure {
  if (!isNonEmptyString(object)) {
    throw new DataError(`The object of measurement must be a non-empty string, found ${describe(object)}.`);
  }
  checkArray(facets, "The facets of the design must be an array of facet names");
  if (facets.length === 0)
    throw new DataError("The design needs at least one facet besides the object of measurement.");
  const names = [object, ...facets];
  if (names.length > MAX_FACETS) {
    throw new DataError(
      `A design has at most ${MAX_FACETS} facets with the object of measurement, found ${names.length}.`,
    );
  }
  names.forEach((name, index) => {
    if (!isNonEmptyString(name))
      throw new DataError(`A facet name must be a non-empty string, found ${describe(name)}.`);
    if (names.indexOf(name) !== index) throw new DataError(`Facet names must be distinct; "${name}" appears twice.`);
  });
  return { names, ancestors: ancestorsOf(names, directParents(names, nested, subject)) };
}

/** Returns the facets each facet is nested in as the nesting names them. */
function directParents(
  names: readonly string[],
  nested: Readonly<Record<string, readonly string[]>>,
  subject: string,
): number[] {
  checkRecordById(nested, `${subject} must be an object of the facets each facet is nested in`);
  const parents = names.map(() => 0);
  for (const [facet, within] of Object.entries(nested)) {
    const index = names.indexOf(facet);
    if (index < 0) throw new DataError(`${subject} names "${facet}", which is not a facet of the design.`);
    if (index === 0) throw new DataError(`The object of measurement "${facet}" cannot be nested in another facet.`);
    checkArray(within, `${subject} must give the facets that "${facet}" is nested in as an array`);
    for (const parent of within) {
      const found = names.indexOf(parent);
      if (found < 0)
        throw new DataError(`Facet "${facet}" is nested in ${describe(parent)}, which is not a facet of the design.`);
      if (found === index) throw new DataError(`Facet "${facet}" cannot be nested in itself.`);
      parents[index] = (parents[index] ?? 0) | (1 << found);
    }
  }
  return parents;
}

/** Returns each facet's facets it is nested in through others too, as the closure of the direct nesting. */
function ancestorsOf(names: readonly string[], direct: readonly number[]): number[] {
  const ancestors = [...direct];
  // A chain of nesting is at most one facet shorter than the design, so that many passes reach the closure.
  for (let pass = 0; pass < names.length; pass++) {
    for (let facet = 0; facet < names.length; facet++) {
      let reached = ancestors[facet] ?? 0;
      for (const parent of facetsOf(ancestors[facet] ?? 0)) reached |= ancestors[parent] ?? 0;
      ancestors[facet] = reached;
    }
  }
  ancestors.forEach((reached, facet) => {
    if ((reached >> facet) & 1) throw new DataError(`The nesting of the facets is circular at "${names[facet]}".`);
  });
  return ancestors;
}

/** Returns the effect with the facets of a mask, its facets that others are not nested in and those it is nested in. */
function effectOf(structure: Structure, mask: number): Effect {
  let nestedIn = 0;
  for (const facet of facetsOf(mask)) nestedIn |= structure.ancestors[facet] ?? 0;
  return { mask, own: mask & ~nestedIn, nestedIn };
}

/** Returns the facets of a mask and, with them, every facet they are nested in. */
export function closeOver(structure: Structure, mask: number): number {
  let closed = mask;
  for (const facet of facetsOf(mask)) closed |= structure.ancestors[facet] ?? 0;
  return closed;
}

/**
 * Returns the effects of the design: one for each set of facets that holds every facet its members are nested in,
 * ordered by their number of facets and then by the design's order of facets.
 */
export function effectsOf(structure: Structure): Effect[] {
  const effects: Effect[] = [];
  const all = 2 ** structure.names.length;
  for (let mask = 1; mask < all; mask++) {
    if (closeOver(structure, mask) === mask) effects.push(effectOf(structure, mask));
  }
  return effects.sort((a, b) => compareMasks(a.mask, b.mask));
}

function compareMasks(a: number, b: number): number {
  const first = facetsOf(a);
  const second = facetsOf(b);
  if (first.length !== second.length) return first.length - second.length;
  for (let k = 0; k < first.length; k++) {
    const difference = (first[k] ?? 0) - (second[k] ?? 0);
    if (difference !== 0) return difference;
  }
  return 0;
}
