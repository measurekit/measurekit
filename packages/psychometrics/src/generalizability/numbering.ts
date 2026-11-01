import { DataError } from "@measurekit/core";
import type { FacetedScore } from "../data/faceted-scores.js";
import { facetsOf, type Structure } from "./structure.js";

/** How the elements of the facets are told apart: a facet's elements are numbered within the elements it is nested in. */
export interface Numbering {
  /** For each facet, the element numbers by label, for each combination of the numbers of the facets it is nested in. */
  readonly numbers: Map<string, Map<string, number>>[];
  /** For each facet, the labels in the order of their numbers, for each such combination. */
  readonly labels: Map<string, string[]>[];
}

/** The scores of a design with the number of the element each has in every facet. */
export interface NumberedScores {
  /** For each score, the numbers of its elements by facet, in the design's order. */
  readonly coordinates: readonly (readonly number[])[];
  /** The labels behind the numbers. */
  readonly numbering: Numbering;
}

/**
 * Numbers the elements of every facet in the order they appear, a nested facet's within the elements it is nested in.
 */
export function numberScores(scores: readonly FacetedScore[], structure: Structure): NumberedScores {
  const { names, ancestors } = structure;
  const numbering: Numbering = { numbers: names.map(() => new Map()), labels: names.map(() => new Map()) };
  const order = names
    .map((_, facet) => facet)
    .sort((a, b) => facetsOf(ancestors[a] ?? 0).length - facetsOf(ancestors[b] ?? 0).length);
  const coordinates = scores.map((entry) => numberElements(entry, structure, order, numbering));
  return { coordinates, numbering };
}

/**
 * Returns the most elements each facet has in any combination of the elements it is nested in.
 *
 * @throws `DataError` when a facet has fewer than 2 elements there.
 */
export function facetSizes(numbering: Numbering, structure: Structure): number[] {
  return structure.names.map((name, facet) => {
    const size = Math.max(...Array.from(numbering.labels[facet]?.values() ?? [], (list) => list.length));
    if (size < 2) {
      throw new DataError(
        `The facet "${name}" needs at least 2 elements${(structure.ancestors[facet] ?? 0) > 0 ? " in each combination of the elements it is nested in" : ""}, found ${size}.`,
      );
    }
    return size;
  });
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

/** Describes a cell by the element of each facet, such as `person "p2", rater "r1"`. */
export function describeElements(cell: readonly number[], structure: Structure, numbering: Numbering): string {
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
