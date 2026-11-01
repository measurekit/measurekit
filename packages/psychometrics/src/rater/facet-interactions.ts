import { DataError } from "@measurekit/core";
import type { LocationRow } from "../irt/linear-rasch.js";

/** The interaction of two facets of a design: the biases of the pairs of elements that were scored together. */
export interface DesignInteraction {
  /** The names of the two facets. */
  readonly facets: readonly [string, string];
  /** The ids of the elements of each cell, in order of first appearance in the virtual items. */
  readonly cells: readonly (readonly [string, string])[];
  /** Each cell's bias as a combination of the interaction's free parameters. */
  readonly rows: readonly LocationRow[];
  /** The cell of each virtual item. */
  readonly cellOf: Int32Array;
  /** The number of free parameters. */
  readonly free: number;
}

/** A facet of the design as the interaction reads it: its name and the ids of its elements. */
interface InteractionFacet {
  readonly name: string;
  readonly ids: readonly string[];
}

/** A sparse vector over the free parameters of an interaction, by parameter index. */
type Combination = Map<number, number>;

/**
 * Returns the interaction of two facets: the biases of the cells, the pairs of elements scored together, in effect form,
 * so that the biases of the cells of each element sum to 0 in either facet.
 *
 * @remarks
 * The cells and the elements are the edges and nodes of a bipartite graph. The constraints of each node fix a spanning
 * forest of the cells from the others, which are the free parameters: a complete table of n by m elements has
 * (n - 1) (m - 1) of them, as effect coding gives, and a table with empty cells has as many as there are cycles in
 * the graph. The biases of the forest's cells are found from its leaves up.
 *
 * @param facets - The two facets, by name and element ids.
 * @param elements - The elements of the two facets at each virtual item, as indices into their ids.
 * @param offset - The index of the first free parameter in the parameter vector.
 * @throws `DataError` when the cells leave no free parameter, because every bias would be 0.
 */
export function designInteraction(
  facets: readonly [InteractionFacet, InteractionFacet],
  elements: readonly (readonly [number, number])[],
  offset: number,
): DesignInteraction {
  const [first, second] = facets;
  const cellIndex = new Map<number, number>();
  const cellOf = new Int32Array(elements.length);
  const cellElements: [number, number][] = [];
  elements.forEach(([a, b], item) => {
    const key = a * second.ids.length + b;
    let cell = cellIndex.get(key);
    if (cell === undefined) {
      cell = cellElements.length;
      cellIndex.set(key, cell);
      cellElements.push([a, b]);
    }
    cellOf[item] = cell;
  });
  const combinations = cellCombinations(first.ids.length, second.ids.length, cellElements);
  const free = combinations.free;
  if (free === 0) {
    throw new DataError(
      `The interaction of the facets "${first.name}" and "${second.name}" cannot be estimated: its ${cellElements.length} cells are fixed by the ${first.ids.length} ${first.name} and ${second.ids.length} ${second.name} elements, so every bias would be 0. It needs elements scored together with at least two elements of the other facet.`,
    );
  }
  return {
    facets: [first.name, second.name],
    cells: cellElements.map(([a, b]) => [first.ids[a] ?? "", second.ids[b] ?? ""]),
    rows: combinations.rows.map((combination) => locationRow(combination, offset)),
    cellOf,
    free,
  };
}

/** Returns the combination of the free parameters that each cell is, and how many free parameters there are. */
function cellCombinations(
  rows: number,
  columns: number,
  cells: readonly (readonly [number, number])[],
): { rows: Combination[]; free: number } {
  const nodes = rows + columns;
  const parent = Int32Array.from({ length: nodes }, (_, node) => node);
  const root = (node: number): number => {
    let current = node;
    while (parent[current] !== current) current = parent[current] ?? current;
    return current;
  };
  const tree: number[] = [];
  const combinations: Combination[] = cells.map(() => new Map());
  let free = 0;
  cells.forEach(([a, b], cell) => {
    const rootA = root(a);
    const rootB = root(rows + b);
    if (rootA === rootB) {
      combinations[cell]?.set(free++, 1);
    } else {
      parent[rootA] = rootB;
      tree.push(cell);
    }
  });
  solveForest(rows, nodes, cells, tree, combinations);
  return { rows: combinations, free };
}

/**
 * Fills in the combinations of the cells of the spanning forest: from the leaves up, a leaf's one forest cell is minus
 * the sum of the other cells at its element, so that the cells of each element sum to 0.
 */
function solveForest(
  rows: number,
  nodes: number,
  cells: readonly (readonly [number, number])[],
  tree: readonly number[],
  combinations: Combination[],
): void {
  const incident: number[][] = Array.from({ length: nodes }, () => []);
  cells.forEach(([a, b], cell) => {
    incident[a]?.push(cell);
    incident[rows + b]?.push(cell);
  });
  const inTree = new Set(tree);
  const remaining = Int32Array.from(incident, (list) => list.filter((cell) => inTree.has(cell)).length);
  const pending = Array.from(remaining.keys()).filter((node) => remaining[node] === 1);
  const solved = new Set<number>();
  for (let node = pending.pop(); node !== undefined; node = pending.pop()) {
    const cell = (incident[node] ?? []).find((c) => inTree.has(c) && !solved.has(c));
    if (cell === undefined) continue;
    const others = (incident[node] ?? []).filter((c) => c !== cell);
    combinations[cell] = minusSum(others.map((c) => combinations[c] ?? new Map()));
    solved.add(cell);
    const [a, b] = cells[cell] ?? [0, 0];
    const far = node === a ? rows + b : a;
    remaining[far] = (remaining[far] ?? 0) - 1;
    if (remaining[far] === 1) pending.push(far);
  }
}

/** Returns minus the sum of combinations. */
function minusSum(combinations: readonly Combination[]): Combination {
  const sum: Combination = new Map();
  for (const combination of combinations) {
    for (const [index, coefficient] of combination) sum.set(index, (sum.get(index) ?? 0) - coefficient);
  }
  return sum;
}

/** Returns a combination as the location row of the parameters from `offset` on. */
function locationRow(combination: Combination, offset: number): LocationRow {
  const entries = [...combination].filter(([, coefficient]) => coefficient !== 0).sort(([a], [b]) => a - b);
  return {
    indices: Int32Array.from(entries, ([index]) => offset + index),
    coefficients: Float64Array.from(entries, ([, coefficient]) => coefficient),
  };
}
