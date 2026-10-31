import { DataError } from "@measurekit/core";
import type { FacetedScore } from "../data/faceted-scores.js";
import { facetsOf, type Structure } from "./structure.js";

/** The scores of a balanced design as a table with one cell for each combination of the facets' elements. */
export interface BalancedScores {
  /** The number of elements of each facet, in the design's order: for a nested facet, within each element it is nested in. */
  readonly sizes: readonly number[];
  /** The scores less their mean, by cell: the facets in the design's order, the last one varying fastest. */
  readonly cells: Float64Array;
}

/** How the elements of the facets are told apart: a facet's elements are numbered within the elements it is nested in. */
interface Numbering {
  /** For each facet, the element numbers by label, for each combination of the numbers of the facets it is nested in. */
  readonly numbers: Map<string, Map<string, number>>[];
  /** For each facet, the labels in the order of their numbers, for each such combination. */
  readonly labels: Map<string, string[]>[];
}

/**
 * Arranges faceted scores in a table of the design.
 *
 * @remarks
 * The scores must have one score for every combination of elements: each facet has the same number of elements, at
 * least 2, in every combination of the elements it is nested in, and each cell has one score. An element of a nested
 * facet is identified within the elements it is nested in, so labels may repeat from one to the next. The scores are
 * centered at their mean, which leaves sums of squares of effects as they are and keeps them accurate for scores far
 * from 0.
 *
 * @throws `DataError` when the scores are fewer than the design needs, a facet has fewer than 2 elements, or the design
 * is not balanced, naming the first cell with more than one score or, failing that, the first with none.
 */
export function balancedScores(scores: readonly FacetedScore[], structure: Structure): BalancedScores {
  const { names, ancestors } = structure;
  const needed = 2 ** names.length;
  if (scores.length < needed) {
    throw new DataError(
      `A design of ${names.length} facets needs at least ${needed} scores, two elements of every facet, found ${scores.length}.`,
    );
  }
  const numbering: Numbering = { numbers: names.map(() => new Map()), labels: names.map(() => new Map()) };
  const order = names
    .map((_, facet) => facet)
    .sort((a, b) => facetsOf(ancestors[a] ?? 0).length - facetsOf(ancestors[b] ?? 0).length);
  const coordinates = scores.map((entry) => numberElements(entry, structure, order, numbering));
  const sizes = names.map((name, facet) => {
    const size = Math.max(...Array.from(numbering.labels[facet]?.values() ?? [], (list) => list.length));
    if (size < 2) {
      throw new DataError(
        `The facet "${name}" needs at least 2 elements${(ancestors[facet] ?? 0) > 0 ? " in each combination of the elements it is nested in" : ""}, found ${size}.`,
      );
    }
    return size;
  });
  const total = sizes.reduce((product, size) => product * size, 1);
  const cells = new Float64Array(total === scores.length ? total : 0);
  const describeCell = (cell: readonly number[]): string => cellDescription(cell, structure, numbering);
  if (total === scores.length) {
    const seen = new Uint8Array(total);
    coordinates.forEach((cell, index) => {
      const at = cellIndex(cell, sizes);
      if (seen[at] === 1) throw repeated(describeCell(cell));
      seen[at] = 1;
      cells[at] = scores[index]?.score ?? 0;
    });
    return { sizes, cells: centered(cells) };
  }
  throw unbalanced(coordinates, sizes, describeCell);
}

/** Numbers the elements of one score's facets, adding labels met for the first time, and returns the numbers. */
function numberElements(
  entry: FacetedScore,
  structure: Structure,
  order: readonly number[],
  numbering: Numbering,
): number[] {
  const cell: number[] = Array.from({ length: structure.names.length }, () => 0);
  for (const facet of order) {
    const key = facetsOf(structure.ancestors[facet] ?? 0)
      .map((parent) => cell[parent])
      .join(",");
    const label = entry.levels[structure.names[facet] ?? ""] ?? "";
    const byKey = numbering.numbers[facet] ?? new Map();
    const lists = numbering.labels[facet] ?? new Map();
    const known = byKey.get(key) ?? new Map<string, number>();
    const list = lists.get(key) ?? [];
    byKey.set(key, known);
    lists.set(key, list);
    let number = known.get(label);
    if (number === undefined) {
      number = list.length;
      known.set(label, number);
      list.push(label);
    }
    cell[facet] = number;
  }
  return cell;
}

/** Returns the position of a cell in the table, the last facet varying fastest. */
function cellIndex(cell: readonly number[], sizes: readonly number[]): number {
  let index = 0;
  for (let facet = 0; facet < sizes.length; facet++) index = index * (sizes[facet] ?? 1) + (cell[facet] ?? 0);
  return index;
}

/** Returns the cells less their mean, in a new array. */
function centered(cells: Float64Array): Float64Array {
  let sum = 0;
  for (const value of cells) sum += value;
  const mean = sum / cells.length;
  return cells.map((value) => value - mean);
}

/** Describes a cell by the element of each facet, such as `person "p2", rater "r1"`. */
function cellDescription(cell: readonly number[], structure: Structure, numbering: Numbering): string {
  return structure.names
    .map((name, facet) => {
      const key = facetsOf(structure.ancestors[facet] ?? 0)
        .map((parent) => cell[parent])
        .join(",");
      const label = numbering.labels[facet]?.get(key)?.[cell[facet] ?? 0];
      return label === undefined ? `${name} number ${(cell[facet] ?? 0) + 1}` : `${name} "${label}"`;
    })
    .join(", ");
}

function repeated(cell: string): DataError {
  return new DataError(`The design is not balanced: there is more than one score for ${cell}.`);
}

/** Returns the error for scores that do not fill the table: a repeated cell if there is one, else the first missing. */
function unbalanced(
  coordinates: readonly (readonly number[])[],
  sizes: readonly number[],
  describeCell: (cell: readonly number[]) => string,
): DataError {
  const present = new Set<string>();
  for (const cell of coordinates) {
    const key = cell.join(",");
    if (present.has(key)) return repeated(describeCell(cell));
    present.add(key);
  }
  // Fewer scores than cells, none repeated: one of the first `present.size + 1` cells in order has no score.
  const cell = sizes.map(() => 0);
  while (present.has(cell.join(","))) {
    let facet = sizes.length - 1;
    while (facet > 0 && (cell[facet] ?? 0) + 1 === (sizes[facet] ?? 1)) cell[facet--] = 0;
    cell[facet] = (cell[facet] ?? 0) + 1;
  }
  return new DataError(`The design is not balanced: there is no score for ${describeCell(cell)}.`);
}
