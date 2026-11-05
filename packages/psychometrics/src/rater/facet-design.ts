import { DataError, describe } from "@measurekit/core";
import { checkFacetedScores, type FacetedScore, isResponse, MAX_RESPONSE } from "../data/internal.js";
import {
  type CompressedResponses,
  checkLinearRaschParameters,
  type LinearRaschModel,
  type LocationRow,
} from "../irt/internal.js";
import { type DesignInteraction, designInteraction } from "./facet-interactions.js";

/** The elements of one facet in a design, with the scores each element has. */
export interface DesignFacet {
  /** The facet's name. */
  readonly name: string;
  /** The element ids, in order of first appearance. */
  readonly ids: readonly string[];
  /** The index of the facet's first free measure in the location parameters: its elements less 1 are free. */
  readonly offset: number;
  /** The number of scores of each element. */
  readonly counts: Float64Array;
  /** The sum of the scores of each element. */
  readonly sums: Float64Array;
}

/** The faceted scores as a linear Rasch model: persons, facets, virtual items and threshold sets. */
export interface FacetDesign {
  /** The person ids, in order of first appearance. */
  readonly personIds: readonly string[];
  /** The facets other than the person facet, in the order of `facets`. */
  readonly facets: readonly DesignFacet[];
  /** The interactions of pairs of facets, in the order of the option `interactions`. */
  readonly interactions: readonly DesignInteraction[];
  /** The element of the threshold facet that owns each threshold set; `null` for the one shared set. */
  readonly sets: readonly (string | null)[];
  /** The model the engine estimates. */
  readonly model: LinearRaschModel;
}

/** The element of each facet by score, with the facets' ids: the scores indexed. */
interface Indexed {
  /** Facet names, the person facet first. */
  readonly names: readonly string[];
  /** The ids of each facet's elements, in order of first appearance. */
  readonly ids: readonly (readonly string[])[];
  /** The element index of each score in each facet. */
  readonly codes: readonly Int32Array[];
  readonly scores: Int32Array;
}

/** The model in the messages of the limits of its size. */
export const MODEL_NAME = "A many-facet model";
/** How many elements a message about a disconnected design lists for each subset. */
const LISTED_ELEMENTS = 3;
/** How many subsets a message about a disconnected design lists. */
const LISTED_SUBSETS = 3;

/**
 * Checks faceted scores and returns them as a linear Rasch model: each distinct combination of elements of the facets
 * other than the person facet is a virtual item, and its location the sum of the facet measures, each facet's summing
 * to 0, and of the biases of its cells in each interaction.
 *
 * @param interactions - Pairs of the facets, each with an interaction.
 * @throws `DataError` when the scores are malformed, a score is not an integer from 0, fewer than two persons were
 * scored, the elements do not form one connected design, a threshold set uses fewer than two categories or skips one,
 * an interaction has no estimable bias, or the model has more parameters than its estimation takes.
 */
export function facetDesign(
  scores: readonly FacetedScore[],
  person: string,
  facets: readonly string[],
  thresholdFacet: string | undefined,
  interactions: readonly (readonly [string, string])[],
): FacetDesign {
  const names = [person, ...facets];
  checkFacetedScores(scores, names);
  checkIntegerScores(scores);
  const indexed = indexScores(scores, names);
  const personIds = indexed.ids[0] ?? [];
  if (personIds.length < 2) throw new DataError(`The scores need at least two persons, found ${personIds.length}.`);
  checkConnected(indexed);
  const designFacets = designFacetsOf(indexed);
  const items = virtualItems(indexed);
  const sets = thresholdSets(indexed, items, thresholdFacet === undefined ? -1 : facets.indexOf(thresholdFacet) + 1);
  const measureCount = designFacets.reduce((sum, facet) => sum + Math.max(facet.ids.length - 1, 0), 0);
  // The parameters besides the measures and the biases: each set's free thresholds, and the mean and SD of theta.
  const others = sets.categoryCounts.reduce((sum, categories) => sum + categories - 2, 2);
  const checkSize = (locations: number): void => checkLinearRaschParameters(locations + others, MODEL_NAME);
  checkSize(measureCount);
  const designed = designInteractions(designFacets, items, interactions, measureCount, checkSize);
  const model: LinearRaschModel = {
    data: compressed(indexed, items),
    locations: items.elements.map((elements, item) =>
      locationRow(
        designFacets,
        elements,
        designed.map(({ rows, cellOf }) => rows[cellOf[item] ?? 0]),
      ),
    ),
    locationCount: designed.reduce((sum, { free }) => sum + free, measureCount),
    setOf: sets.setOf,
    categoryCounts: sets.categoryCounts,
  };
  return { personIds, facets: designFacets, interactions: designed, sets: sets.elements, model };
}

function checkIntegerScores(scores: readonly FacetedScore[]): void {
  scores.forEach(({ score }, index) => {
    if (!isResponse(score)) {
      throw new DataError(
        `Score ${index + 1} needs an integer score from 0 to ${MAX_RESPONSE}, found ${describe(score)}.`,
      );
    }
  });
}

function indexScores(scores: readonly FacetedScore[], names: readonly string[]): Indexed {
  const lookups = names.map(() => new Map<string, number>());
  const codes = names.map(() => new Int32Array(scores.length));
  scores.forEach(({ levels }, s) => {
    names.forEach((name, f) => {
      const lookup = lookups[f];
      const id = levels[name] ?? "";
      let index = lookup?.get(id);
      if (index === undefined) {
        index = lookup?.size ?? 0;
        lookup?.set(id, index);
      }
      const facetCodes = codes[f];
      if (facetCodes !== undefined) facetCodes[s] = index;
    });
  });
  return {
    names,
    ids: lookups.map((lookup) => [...lookup.keys()]),
    codes,
    scores: Int32Array.from(scores, ({ score }) => score),
  };
}

/** Throws unless the persons and elements are one connected set: each score links its person to its elements. */
function checkConnected(indexed: Indexed): void {
  const starts: number[] = [];
  let total = 0;
  for (const ids of indexed.ids) {
    starts.push(total);
    total += ids.length;
  }
  const parent = Int32Array.from({ length: total }, (_, node) => node);
  const root = (node: number): number => {
    let current = node;
    while (parent[current] !== current) {
      parent[current] = parent[parent[current] ?? current] ?? current;
      current = parent[current] ?? current;
    }
    return current;
  };
  for (let s = 0; s < indexed.scores.length; s++) {
    const personNode = (starts[0] ?? 0) + (indexed.codes[0]?.[s] ?? 0);
    for (let f = 1; f < indexed.names.length; f++) {
      parent[root((starts[f] ?? 0) + (indexed.codes[f]?.[s] ?? 0))] = root(personNode);
    }
  }
  const subsets = new Map<number, number[]>();
  for (let node = 0; node < total; node++) {
    const members = subsets.get(root(node)) ?? [];
    members.push(node);
    subsets.set(root(node), members);
  }
  if (subsets.size > 1) throw new DataError(disconnectedMessage([...subsets.values()], indexed, starts));
}

/** Returns the message for a design that falls into several subsets, naming a few elements of the first ones. */
function disconnectedMessage(subsets: readonly (readonly number[])[], indexed: Indexed, starts: readonly number[]) {
  const describeNode = (node: number): string => {
    const f = starts.findLastIndex((start) => start <= node);
    return `${indexed.names[f]} ${JSON.stringify(indexed.ids[f]?.[node - (starts[f] ?? 0)])}`;
  };
  const listed = subsets.slice(0, LISTED_SUBSETS).map((members, i) => {
    const named = members.slice(0, LISTED_ELEMENTS).map(describeNode).join(", ");
    const more = members.length > LISTED_ELEMENTS ? ` and ${members.length - LISTED_ELEMENTS} more` : "";
    return `subset ${i + 1}: ${named}${more}`;
  });
  const rest = subsets.length > LISTED_SUBSETS ? `; and ${subsets.length - LISTED_SUBSETS} more subsets` : "";
  return `The scores fall into ${subsets.length} subsets that share no person, item or rater, so their measures are not on one scale (${listed.join("; ")}${rest}).`;
}

function designFacetsOf(indexed: Indexed): DesignFacet[] {
  let offset = 0;
  return indexed.names.slice(1).map((name, i) => {
    const f = i + 1;
    const ids = indexed.ids[f] ?? [];
    const counts = new Float64Array(ids.length);
    const sums = new Float64Array(ids.length);
    indexed.codes[f]?.forEach((element, s) => {
      counts[element] = (counts[element] ?? 0) + 1;
      sums[element] = (sums[element] ?? 0) + (indexed.scores[s] ?? 0);
    });
    const facet = { name, ids, offset, counts, sums };
    offset += Math.max(ids.length - 1, 0);
    return facet;
  });
}

/** The distinct combinations of elements of the facets other than the person facet, and each score's combination. */
interface VirtualItems {
  /** The element of each facet but the person's for each virtual item. */
  readonly elements: readonly Int32Array[];
  /** The virtual item of each score. */
  readonly itemOf: Int32Array;
}

function virtualItems(indexed: Indexed): VirtualItems {
  const lookup = new Map<string, number>();
  const elements: Int32Array[] = [];
  const itemOf = new Int32Array(indexed.scores.length);
  for (let s = 0; s < itemOf.length; s++) {
    const combination = Int32Array.from({ length: indexed.names.length - 1 }, (_, f) => indexed.codes[f + 1]?.[s] ?? 0);
    const key = combination.join(" ");
    let item = lookup.get(key);
    if (item === undefined) {
      item = elements.length;
      lookup.set(key, item);
      elements.push(combination);
    }
    itemOf[s] = item;
  }
  return { elements, itemOf };
}

/**
 * Returns the interaction of each pair of facets, its parameters following the measures and the earlier interactions.
 *
 * @param checkSize - Checks the location parameters up to an interaction's last, before its biases are solved.
 */
function designInteractions(
  facets: readonly DesignFacet[],
  items: VirtualItems,
  pairs: readonly (readonly [string, string])[],
  start: number,
  checkSize: (locations: number) => void,
): DesignInteraction[] {
  let offset = start;
  return pairs.map((pair) => {
    const [f, g] = pair.map((name) => facets.findIndex((facet) => facet.name === name));
    const first = facets[f ?? -1];
    const second = facets[g ?? -1];
    if (first === undefined || second === undefined || f === undefined || g === undefined) {
      throw new DataError(`The interaction of ${pair.map(describe).join(" and ")} names a facet that the scores lack.`);
    }
    const interaction = designInteraction(
      [first, second],
      items.elements.map((elements) => [elements[f] ?? 0, elements[g] ?? 0]),
      offset,
      checkSize,
    );
    offset += interaction.free;
    return interaction;
  });
}

/**
 * The location parameters of a virtual item: the measure of its element in each facet, effect coded, and the bias of
 * its cell in each interaction.
 */
function locationRow(
  facets: readonly DesignFacet[],
  elements: Int32Array,
  cells: readonly (LocationRow | undefined)[],
): LocationRow {
  const indices: number[] = [];
  const coefficients: number[] = [];
  facets.forEach((facet, f) => {
    const free = facet.ids.length - 1;
    const element = elements[f] ?? 0;
    if (free < 1) return;
    if (element < free) {
      indices.push(facet.offset + element);
      coefficients.push(1);
      return;
    }
    for (let a = 0; a < free; a++) {
      indices.push(facet.offset + a);
      coefficients.push(-1);
    }
  });
  for (const cell of cells) {
    for (const index of cell?.indices ?? []) indices.push(index);
    for (const coefficient of cell?.coefficients ?? []) coefficients.push(coefficient);
  }
  return { indices: Int32Array.from(indices), coefficients: Float64Array.from(coefficients) };
}

/** The threshold sets of a design: each virtual item's set, each set's categories and owner. */
interface ThresholdSets {
  readonly setOf: Int32Array;
  readonly categoryCounts: Int32Array;
  readonly elements: readonly (string | null)[];
}

/** Returns the threshold sets: one shared by all items, or one for each element of the facet at `facet` in the names. */
function thresholdSets(indexed: Indexed, items: VirtualItems, facet: number): ThresholdSets {
  const owners: (string | null)[] = facet < 1 ? [null] : [...(indexed.ids[facet] ?? [])];
  const setOf = Int32Array.from(items.elements, (elements) => (facet < 1 ? 0 : (elements[facet - 1] ?? 0)));
  const used = owners.map(() => new Set<number>());
  items.itemOf.forEach((item, s) => {
    used[setOf[item] ?? 0]?.add(indexed.scores[s] ?? 0);
  });
  owners.forEach((owner, set) => {
    checkCategories(owner, indexed.names[facet] ?? "", used[set] ?? new Set());
  });
  return {
    setOf,
    categoryCounts: Int32Array.from(used, (categories) => highestOf(categories) + 1),
    elements: owners,
  };
}

/** Throws unless the categories used in a threshold set are 0 up to the highest, at least two of them. */
function checkCategories(owner: string | null, facet: string, used: ReadonlySet<number>): void {
  const scope = owner === null ? "the rating scale" : `the thresholds of the ${facet} "${owner}"`;
  const highest = highestOf(used);
  if (highest < 1) throw new DataError(`The scores of ${scope} need at least two categories, found only category 0.`);
  for (let k = 0; k < highest; k++) {
    if (!used.has(k)) {
      throw new DataError(
        `Category ${k} has no scores in ${scope}, but category ${highest} has: its threshold cannot be estimated.`,
      );
    }
  }
}

/** Returns each person's scores as the engine's responses to virtual items. */
function compressed(indexed: Indexed, items: VirtualItems): CompressedResponses {
  const personCodes = indexed.codes[0] ?? new Int32Array(0);
  const personIds = indexed.ids[0] ?? [];
  const offsets = new Int32Array(personIds.length + 1);
  for (const person of personCodes) offsets[person + 1] = (offsets[person + 1] ?? 0) + 1;
  for (let p = 0; p < personIds.length; p++) offsets[p + 1] = (offsets[p + 1] ?? 0) + (offsets[p] ?? 0);
  const next = Int32Array.from(offsets);
  const responseItems = new Int32Array(personCodes.length);
  const codes = new Int32Array(personCodes.length);
  personCodes.forEach((person, s) => {
    const at = next[person] ?? 0;
    next[person] = at + 1;
    responseItems[at] = items.itemOf[s] ?? 0;
    codes[at] = indexed.scores[s] ?? 0;
  });
  return {
    personIds,
    itemIds: items.elements.map((elements) => elements.join(" ")),
    offsets,
    items: responseItems,
    codes,
    times: null,
  };
}

/** Returns the highest of the categories, -Infinity for none, without passing each as an argument. */
function highestOf(categories: Iterable<number>): number {
  let highest = Number.NEGATIVE_INFINITY;
  for (const category of categories) highest = Math.max(highest, category);
  return highest;
}
