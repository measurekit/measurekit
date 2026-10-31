import type { Random } from "@measurekit/core";
import type { ItemState } from "./calibration-setup.js";
import type { Chain } from "./mhrm-chain.js";
import { type Correlations, latentObservedInformation, latentScore, secondMoments } from "./mhrm-latent.js";
import { type Draws, itemMoments, observedItemInformation, priorMoments, type Scratch } from "./mhrm-moments.js";

/** The observed complete-data information is computed every this many cycles; the cycles between reuse it. */
export const INFORMATION_INTERVAL = 4;
/** The Metropolis-Hastings steps per cycle: mirt's `MHDRAWS`. */
const SWEEPS_PER_CYCLE = 5;
/** The acceptance rate that the proposal variance is tuned to, in the middle of 0.2 to 0.5. */
const TARGET_ACCEPTANCE = 0.35;
/** The proposal variance changes by the factor exp(ADAPTATION * (acceptance - TARGET_ACCEPTANCE)) in each step. */
const ADAPTATION = 1;
/** The bounds of the proposal variance. */
const MIN_VARIANCE = 1e-4;
const MAX_VARIANCE = 100;

/** The parameters MH-RM updates, on their unbounded scale: the items' estimated entries, then the correlations. */
export interface Layout {
  readonly size: number;
  /** The slot of each item's first entry. */
  readonly offsets: Int32Array;
  /** The slot of the first correlation. */
  readonly latent: number;
  /** The number of correlations that are estimated. */
  readonly pairs: number;
}

/** Everything the cycles of a calibration share. */
export interface Run<TParameters> {
  readonly items: readonly ItemState<TParameters>[];
  readonly layout: Layout;
  readonly correlations: Correlations;
  readonly chain: Chain;
  readonly draws: Draws;
  readonly scratch: Scratch;
  readonly random: Random;
  readonly respondents: number;
}

/** The complete-data score of the parameters and the information in blocks: one per item, then the correlations. */
export interface Moments {
  readonly gradient: Float64Array;
  /** Each block's information by rows. */
  readonly blocks: Float64Array[];
  /** The views of `gradient` that each block fills. */
  readonly parts: Float64Array[];
}

/** What {@link computeMoments} adds to the complete-data score of the draws. */
export interface MomentOptions {
  /** Whether to add the log priors' derivatives. */
  readonly prior: boolean;
  /** Whether to write the observed complete-data information, by forward differences of the score. */
  readonly information: boolean;
}

/** Returns the layout of items with `pairs` estimated correlations. */
export function layoutOf<TParameters>(items: readonly ItemState<TParameters>[], pairs: number): Layout {
  const offsets = new Int32Array(items.length);
  let size = 0;
  items.forEach((item, i) => {
    offsets[i] = size;
    size += item.free.length;
  });
  return { size: size + pairs, offsets, latent: size, pairs };
}

/**
 * Makes one cycle's draws: refreshes the chain and takes the Metropolis-Hastings steps. With `adapt`, each step tunes
 * the proposal variance toward the target acceptance. Returns the variance.
 */
export function drawCycle<TParameters>(run: Run<TParameters>, variance: number, adapt: boolean): number {
  run.chain.refresh();
  let tuned = variance;
  for (let s = 0; s < SWEEPS_PER_CYCLE; s++) {
    const acceptance = run.chain.sweep(run.random, tuned);
    if (adapt) {
      tuned = Math.min(
        MAX_VARIANCE,
        Math.max(MIN_VARIANCE, tuned * Math.exp(ADAPTATION * (acceptance - TARGET_ACCEPTANCE))),
      );
    }
  }
  return tuned;
}

/** Returns space for the moments of the layout's blocks. */
export function createMoments<TParameters>({ items, layout }: Run<TParameters>): Moments {
  const gradient = new Float64Array(layout.size);
  const blocks: Float64Array[] = [];
  const parts: Float64Array[] = [];
  items.forEach((item, i) => {
    const size = item.free.length;
    blocks.push(new Float64Array(size * size));
    parts.push(gradient.subarray(layout.offsets[i] ?? 0, (layout.offsets[i] ?? 0) + size));
  });
  if (layout.pairs > 0) {
    blocks.push(new Float64Array(layout.pairs * layout.pairs));
    parts.push(gradient.subarray(layout.latent, layout.latent + layout.pairs));
  }
  return { gradient, blocks, parts };
}

/** Writes the complete-data score at the draws, and as asked the priors' derivatives and the observed information. */
export function computeMoments<TParameters>(run: Run<TParameters>, moments: Moments, options: MomentOptions): void {
  const { items, layout, correlations, draws, scratch, respondents } = run;
  items.forEach((item, i) => {
    const gradient = moments.parts[i];
    const information = moments.blocks[i];
    if (gradient === undefined || information === undefined) return;
    itemMoments(item, i, draws, scratch, gradient);
    if (options.information) observedItemInformation(item, i, draws, scratch, information, false);
    if (options.prior) priorMoments(item, gradient, options.information ? information : undefined);
  });
  const gradient = moments.parts[items.length];
  const information = moments.blocks[items.length];
  if (layout.pairs === 0 || gradient === undefined || information === undefined) return;
  const second = new Float64Array(correlations.dimensions ** 2);
  secondMoments(draws.theta, respondents, correlations.dimensions, second);
  latentScore(correlations, second, respondents, gradient);
  if (options.information) latentObservedInformation(correlations, second, respondents, information);
}

/** Writes the estimated parameters on their own scale into `out`: the items' estimated entries, then the correlations. */
export function estimatesOf<TParameters>({ items, layout, correlations }: Run<TParameters>, out: Float64Array): void {
  let slot = 0;
  for (const item of items) for (const j of item.free) out[slot++] = item.vector[j] ?? 0;
  if (layout.pairs > 0) {
    for (const [i, j] of correlations.pairs) out[slot++] = correlations.matrix[i * correlations.dimensions + j] ?? 0;
  }
}
