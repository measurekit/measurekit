import type { CompressedResponses } from "../irt/responses.js";
import type { ItemGroup } from "./bifactor-structure.js";

/** The distinct responses to the items of one group, in lexicographic order. */
export interface GroupPatterns {
  /** The group of items. */
  readonly group: ItemGroup;
  /** The number of items of the group. */
  readonly width: number;
  /** The number of distinct sub-patterns. */
  readonly count: number;
  /** The responses of each sub-pattern, `count` rows of `width` entries, with -1 for an item left unanswered. */
  readonly codes: Int32Array;
  /** Whether each sub-pattern answers at least one item. */
  readonly answered: Uint8Array;
}

/** The distinct response patterns of the persons who answered at least one item, and their sub-patterns by group. */
export interface PatternTable {
  /** The number of distinct patterns. */
  readonly count: number;
  /** The number of persons with each pattern. */
  readonly frequency: Float64Array;
  /** The sub-pattern of each pattern in each group: pattern `p`, group `s` is `subPattern[p * groups.length + s]`. */
  readonly subPattern: Int32Array;
  /** The sub-patterns of each group. */
  readonly groups: readonly GroupPatterns[];
}

/**
 * Returns the distinct patterns of the responses, and for each group of items the distinct patterns of the responses to
 * its items.
 *
 * @remarks
 * A group's likelihood over the nodes depends only on the responses to its own items, so persons that differ elsewhere
 * share it. Sub-patterns are sorted, so that consecutive ones share the longest prefixes of items.
 */
export function patternsOf(responses: CompressedResponses, groups: readonly ItemGroup[]): PatternTable {
  const itemCount = responses.itemIds.length;
  const dense = new Int32Array(itemCount);
  const collectors = groups.map((group) => ({ group, ...collector(group.items.length) }));
  const patternIndex = new Map<string, number>();
  const patterns: { frequency: number; sub: number[] }[] = [];
  for (let person = 0; person < responses.offsets.length - 1; person++) {
    const start = responses.offsets[person] ?? 0;
    const end = responses.offsets[person + 1] ?? 0;
    if (end === start) continue;
    dense.fill(-1);
    for (let r = start; r < end; r++) dense[responses.items[r] ?? 0] = responses.codes[r] ?? 0;
    const sub = collectors.map(({ group, add }) => add(group.items.map((item) => dense[item] ?? -1)));
    const key = sub.join(",");
    const known = patternIndex.get(key);
    if (known === undefined) {
      patternIndex.set(key, patterns.length);
      patterns.push({ frequency: 1, sub });
    } else {
      const pattern = patterns[known];
      if (pattern !== undefined) pattern.frequency++;
    }
  }
  const sorted = collectors.map(({ group, sorted: sortRows }) => sortRows(group));
  const subPattern = new Int32Array(patterns.length * groups.length);
  patterns.forEach(({ sub }, p) => {
    sub.forEach((id, s) => {
      subPattern[p * groups.length + s] = sorted[s]?.rank[id] ?? 0;
    });
  });
  return {
    count: patterns.length,
    frequency: Float64Array.from(patterns, ({ frequency }) => frequency),
    subPattern,
    groups: sorted.map(({ patterns: group }) => group),
  };
}

/** The distinct sub-patterns of one group, collected in order of appearance. */
interface Collector {
  /** Returns the number of the row among the distinct rows, adding the row if it is new. */
  add(row: number[]): number;
  /** Returns the sub-patterns in lexicographic order and the position of each row of appearance in that order. */
  sorted(group: ItemGroup): { readonly patterns: GroupPatterns; readonly rank: Int32Array };
}

function collector(width: number): Collector {
  const index = new Map<string, number>();
  const rows: number[][] = [];
  return {
    add(row) {
      const key = row.join(",");
      const known = index.get(key);
      if (known !== undefined) return known;
      index.set(key, rows.length);
      rows.push(row);
      return rows.length - 1;
    },
    sorted(group) {
      const order = Array.from(rows.keys()).sort((a, b) => compareRows(rows[a] ?? [], rows[b] ?? []));
      const rank = new Int32Array(order.length);
      const codes = new Int32Array(order.length * width);
      const answered = new Uint8Array(order.length);
      order.forEach((original, position) => {
        rank[original] = position;
        const row = rows[original] ?? [];
        codes.set(row, position * width);
        answered[position] = row.some((code) => code >= 0) ? 1 : 0;
      });
      return { patterns: { group, width, count: order.length, codes, answered }, rank };
    },
  };
}

/** Compares two rows of the same length entry by entry. */
function compareRows(a: readonly number[], b: readonly number[]): number {
  for (let i = 0; i < a.length; i++) {
    const difference = (a[i] ?? 0) - (b[i] ?? 0);
    if (difference !== 0) return difference;
  }
  return 0;
}
