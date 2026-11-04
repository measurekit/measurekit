import { checkBoolean, type FieldNames } from "@measurekit/core";

/** Options of the DIF statistics whose matching criterion can be purified. */
export interface PurificationOptions {
  /**
   * Whether to purify the matching criterion, as difR's `purify`: the items flagged at 0.05 leave the anchors and the
   * statistics are computed again, until a round flags the items the previous one did, or for at most 10 rounds.
   *
   * @defaultValue `false`
   * @see {@link https://doi.org/10.4324/9780203056905-19 | Holland and Thayer (1988)}
   */
  readonly purify?: boolean;
}

export const PURIFICATION_OPTIONS: FieldNames<PurificationOptions> = { purify: true };

/** The purification of a matching criterion. */
export interface DifPurification {
  /**
   * The anchors of the final statistics, the items not flagged before them; with the studied item, its matching items.
   */
  readonly anchors: readonly string[];
  /** The rounds run after the first statistics, each on the items the previous one did not flag. */
  readonly iterations: number;
  /** Whether the last round flagged the items the round before it did. */
  readonly converged: boolean;
}

/**
 * DIF statistics of every item, with the purification of their matching criterion.
 *
 * @typeParam TItem - The statistics of one item.
 */
export interface DifAnalysis<TItem> {
  /** The statistics of each item, in item order. */
  readonly items: readonly TItem[];
  /** The purification, or `null` without the `purify` option. */
  readonly purification: DifPurification | null;
}

/** The significance level of the DIF tests: ETS's, and the level at which purification flags an item. */
export const SIGNIFICANCE_LEVEL = 0.05;
/** The most purification rounds, difR's `nrIter`. */
const MAX_ROUNDS = 10;
/** The fewest anchors a statistic needs to be computed: one, the studied item's matching item. */
const DEFAULT_MINIMUM_ANCHORS = 1;

/**
 * Returns the statistics of every item on all items as anchors and, with `purify`, purifies them: each round computes
 * them again with the items flagged before them out of the anchors. Stops when a round flags the items the previous
 * one did, when the items not flagged are fewer than `minimumAnchors`, or after the most rounds.
 *
 * @param statistics - Computes every item's statistics with the anchors marked 1.
 * @param flagged - Whether an item's statistics flag it.
 * @param minimumAnchors - The fewest anchors that `statistics` accepts.
 * @throws `DataError` when `purify` is not a boolean.
 */
export function purified<TItem>(
  itemIds: readonly string[],
  purify: boolean,
  statistics: (anchors: Uint8Array) => TItem[],
  flagged: (item: TItem) => boolean,
  minimumAnchors = DEFAULT_MINIMUM_ANCHORS,
): DifAnalysis<TItem> {
  checkBoolean(purify, "The option purify");
  let anchors = new Uint8Array(itemIds.length).fill(1);
  let items = statistics(anchors);
  if (!purify) return { items, purification: null };
  let flags = items.map(flagged);
  let iterations = 0;
  let converged = !flags.includes(true);
  while (!converged && iterations < MAX_ROUNDS && flags.filter((flag) => !flag).length >= minimumAnchors) {
    iterations++;
    anchors = Uint8Array.from(flags, (flag) => (flag ? 0 : 1));
    items = statistics(anchors);
    const next = items.map(flagged);
    converged = next.every((flag, i) => flag === flags[i]);
    flags = next;
  }
  return {
    items,
    purification: { anchors: itemIds.filter((_, i) => anchors[i] === 1), iterations, converged },
  };
}

/** Returns whether a significance is below the level of purification. */
export function significant(significance: number | null): boolean {
  return significance !== null && significance < SIGNIFICANCE_LEVEL;
}
