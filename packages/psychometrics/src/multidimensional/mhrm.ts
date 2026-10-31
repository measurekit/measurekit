import { backtrack, createRandom } from "@measurekit/core";
import type { CompressedResponses } from "../irt/responses.js";
import { invertPositiveDefinite } from "../math/linear-algebra.js";
import type { ItemState } from "./calibration-setup.js";
import type { Design } from "./design.js";
import { assignCoordinates, itemCoordinates } from "./m-step.js";
import { createChain, responsesByItem } from "./mhrm-chain.js";
import { type Correlations, createCorrelations } from "./mhrm-latent.js";
import { importanceLogLikelihood } from "./mhrm-likelihood.js";
import { createScratch } from "./mhrm-moments.js";
import {
  computeMoments,
  createMoments,
  drawCycle,
  estimatesOf,
  INFORMATION_INTERVAL,
  layoutOf,
  type Moments,
  type Run,
} from "./mhrm-run.js";
import { observedInformation } from "./mhrm-standard-errors.js";

/** The cycles of the first stage, in which the estimates move to the neighborhood of the solution: mirt's `BURNIN`. */
const BURNIN_CYCLES = 150;
/** The cycles of the second stage, whose estimates and information the third stage starts from: mirt's `SEMCYCLES`. */
const AVERAGING_CYCLES = 100;
/** The gain of the first two stages: mirt's `gamma <- 0.2`. */
const FIRST_STAGES_GAIN = 0.2;
/** The numerator and the exponent of the gain of the third stage, (0.1 / t)^0.75 in cycle t: mirt's default `gain`. */
const GAIN = [0.1, 0.75] as const;
/** The largest correction of an estimate in the first two stages: mirt clips its corrections to [-1, 1]. */
const MAX_CORRECTION = 1;
/** The successive cycles within the tolerance that end the third stage: mirt's `conv >= 3`. */
const CONVERGED_CYCLES = 3;

/** How an MH-RM calibration runs. */
export interface MhrmSettings {
  /** The largest change of an estimate, in three successive cycles, that ends the third stage. */
  readonly tolerance: number;
  /** The most cycles of the third stage. */
  readonly maxIterations: number;
  /** Whether to compute the covariance of the estimates by Louis' formula from the draws. */
  readonly standardErrors: boolean;
  /** The seed of the draws. */
  readonly seed: number;
}

/** The outcome of an MH-RM calibration, with the items at their estimates. */
export interface MhrmFit {
  readonly converged: boolean;
  /** The cycles of the third stage. */
  readonly iterations: number;
  /** The covariance of the estimates as {@link louisCovariance} lays it out, or `null`. */
  readonly covariance: Float64Array | null;
  readonly logLikelihood: number;
  readonly correlations: Correlations;
}

/**
 * Estimates the parameters of the items and the correlations by the Metropolis-Hastings Robbins-Monro algorithm.
 *
 * @remarks
 * Each cycle draws every person's theta from the posterior given the current estimates, by 5
 * random-walk Metropolis-Hastings steps, and moves the estimates by γ Γ⁻¹ g, with g the complete-data score at the draws
 * and Γ the complete-data information. The first stage, 150 cycles, takes steps of gain 0.2 with corrections clipped
 * to [-1, 1] and tunes the proposal variance. The second, 100 cycles, does the same and keeps the estimates and the
 * information; the third starts from their means, running the average of the information with the gain (0.1 / t)^0.75,
 * and stops when no estimate changes by more than the tolerance in three successive cycles or at `maxIterations`. The
 * information is the observed complete-data information at the draws, as in mirt: the derivative of the score by
 * differences, computed in the first cycle and every fourth after it, and averaged in between. Items move on the
 * unbounded scale of calibration, and the correlations are halved in their step until the matrix is positive definite.
 *
 * @see {@link https://doi.org/10.1007/s11336-009-9136-x | Cai (2010), exploratory item factor analysis}
 * @see {@link https://doi.org/10.3102/1076998609353115 | Cai (2010), confirmatory item factor analysis}
 */
export function calibrateByMhrm<TParameters>(
  responses: CompressedResponses,
  items: readonly ItemState<TParameters>[],
  design: Design,
  settings: MhrmSettings,
): MhrmFit {
  const dimensions = design.dimensionNames.length;
  const correlations = createCorrelations(dimensions);
  const data = responsesByItem(responses);
  const chain = createChain(items, data, correlations);
  const pairs = design.correlated ? correlations.pairs.length : 0;
  const run: Run<TParameters> = {
    items,
    layout: layoutOf(items, pairs),
    correlations,
    chain,
    draws: { theta: chain.theta, responses: data, dimensions },
    scratch: createScratch(items, dimensions),
    random: createRandom(settings.seed),
    respondents: data.count,
  };
  const stages = stochasticApproximation(run, settings);
  const { converged, iterations } = stages;
  const covariance = settings.standardErrors ? observedInformation(run, stages.variance) : null;
  const logLikelihood = importanceLogLikelihood(items, responses, correlations, run.random);
  return { converged, iterations, covariance, logLikelihood, correlations };
}

/** Runs the three stages and returns whether the third reached the tolerance, and its cycles. */
function stochasticApproximation<TParameters>(
  run: Run<TParameters>,
  settings: MhrmSettings,
): { converged: boolean; iterations: number; variance: number } {
  const moments = createMoments(run);
  const estimates = new Float64Array(run.layout.size);
  const kept = {
    values: [] as Float64Array[],
    information: moments.blocks.map((block) => new Float64Array(block.length)),
  };
  let variance = 1;
  for (let cycle = 1; cycle <= BURNIN_CYCLES + AVERAGING_CYCLES; cycle++) {
    const first = cycle <= BURNIN_CYCLES;
    variance = drawCycle(run, variance, first);
    computeMoments(run, moments, { prior: true, information: isDue(cycle) });
    update(run, moments, FIRST_STAGES_GAIN, true);
    if (!first) keep(run, moments, kept, estimates);
  }
  startThirdStage(run, kept, moments);
  return { ...thirdStage(run, moments, variance, settings), variance };
}

/** Writes Γ⁻¹ g of a block into `out`: by the inverse of the information, or by its diagonal where it is singular. */
function solveBlock(information: Float64Array, gradient: Float64Array, out: Float64Array): void {
  const size = gradient.length;
  const inverse = invertPositiveDefinite(information, size);
  for (let l = 0; l < size; l++) {
    if (inverse === null) {
      out[l] = (gradient[l] ?? 0) / Math.max(information[l * size + l] ?? 0, Number.EPSILON);
      continue;
    }
    let sum = 0;
    for (let m = 0; m < size; m++) sum += (inverse[l * size + m] ?? 0) * (gradient[m] ?? 0);
    out[l] = sum;
  }
}

/** Moves every parameter by `gain` times its block's correction, clipped to [-1, 1] when `clip`. */
function update<TParameters>(run: Run<TParameters>, moments: Moments, gain: number, clip: boolean): void {
  const { items, layout, correlations } = run;
  items.forEach((item, i) => {
    const gradient = moments.parts[i];
    const information = moments.blocks[i];
    if (gradient === undefined || information === undefined || item.free.length === 0) return;
    const correction = new Float64Array(item.free.length);
    solveBlock(information, gradient, correction);
    const eta = itemCoordinates(item);
    for (let l = 0; l < eta.length; l++) eta[l] = (eta[l] ?? 0) + gain * limited(correction[l] ?? 0, clip);
    assignCoordinates(item, eta);
  });
  if (layout.pairs === 0) return;
  const gradient = moments.parts[items.length];
  const information = moments.blocks[items.length];
  if (gradient === undefined || information === undefined) return;
  const correction = new Float64Array(layout.pairs);
  solveBlock(information, gradient, correction);
  moveCorrelations(
    correlations,
    correction.map((value) => gain * limited(value, clip)),
  );
}

function limited(value: number, clip: boolean): number {
  return clip ? Math.max(-MAX_CORRECTION, Math.min(MAX_CORRECTION, value)) : value;
}

/** Adds `change` to the correlations, halving it until the matrix is positive definite. */
function moveCorrelations(correlations: Correlations, change: Float64Array): void {
  const { dimensions, pairs, matrix } = correlations;
  backtrack(
    (length) => {
      const next = matrix.slice();
      pairs.forEach(([i, j], a) => {
        const value = (matrix[i * dimensions + j] ?? 0) + length * (change[a] ?? 0);
        next[i * dimensions + j] = value;
        next[j * dimensions + i] = value;
      });
      return correlations.set(next) ? 0 : Number.POSITIVE_INFINITY;
    },
    0,
    0,
    { decrease: 0 },
  );
}

/** Stores the second stage's estimates and information. */
function keep<TParameters>(
  run: Run<TParameters>,
  moments: Moments,
  kept: { values: Float64Array[]; information: Float64Array[] },
  scratch: Float64Array,
): void {
  estimatesOf(run, scratch);
  kept.values.push(Float64Array.from(scratch));
  moments.blocks.forEach((block, b) => {
    const sum = kept.information[b];
    if (sum !== undefined)
      for (let k = 0; k < block.length; k++) sum[k] = (sum[k] ?? 0) + (block[k] ?? 0) / AVERAGING_CYCLES;
  });
}

/** Sets the parameters to the means of the second stage's and the information to its mean. */
function startThirdStage<TParameters>(
  run: Run<TParameters>,
  kept: { values: Float64Array[]; information: Float64Array[] },
  moments: Moments,
): void {
  const mean = new Float64Array(run.layout.size);
  for (const values of kept.values)
    for (let k = 0; k < mean.length; k++) mean[k] = (mean[k] ?? 0) + (values[k] ?? 0) / kept.values.length;
  setOwnScale(run, mean);
  moments.blocks.forEach((block, b) => {
    block.set(kept.information[b] ?? block);
  });
}

/** Sets the parameters from their own-scale values. */
function setOwnScale<TParameters>({ items, layout, correlations }: Run<TParameters>, values: Float64Array): void {
  let slot = 0;
  for (const item of items) {
    for (const j of item.free) item.vector[j] = values[slot++] ?? 0;
    item.parameters = item.model.fromVector(item.vector, item.dimensions);
  }
  if (layout.pairs === 0) return;
  const next = correlations.matrix.slice();
  for (const [i, j] of correlations.pairs) {
    next[i * correlations.dimensions + j] = values[slot] ?? 0;
    next[j * correlations.dimensions + i] = values[slot++] ?? 0;
  }
  correlations.set(next);
}

/** The third stage: running information and a decreasing gain, until the tolerance is met in three cycles in a row. */
function thirdStage<TParameters>(
  run: Run<TParameters>,
  moments: Moments,
  start: number,
  settings: MhrmSettings,
): { converged: boolean; iterations: number } {
  const current = new Float64Array(run.layout.size);
  const previous = new Float64Array(run.layout.size);
  const cycle = createMoments(run);
  let within = 0;
  for (let t = 1; t <= settings.maxIterations; t++) {
    drawCycle(run, start, false);
    computeMoments(run, cycle, { prior: true, information: isDue(t) });
    const gain = (GAIN[0] / t) ** GAIN[1];
    if (isDue(t)) averageInformation(moments, cycle, gain);
    moments.gradient.set(cycle.gradient);
    estimatesOf(run, previous);
    update(run, moments, gain, false);
    estimatesOf(run, current);
    within = largestChange(previous, current) < settings.tolerance ? within + 1 : 0;
    if (within >= CONVERGED_CYCLES) return { converged: true, iterations: t };
  }
  return { converged: false, iterations: settings.maxIterations };
}

/** Whether the observed information is computed in a cycle: the first, and every {@link INFORMATION_INTERVAL}th after. */
function isDue(cycle: number): boolean {
  return (cycle - 1) % INFORMATION_INTERVAL === 0;
}

/** Moves the running information toward this cycle's by `gain`. */
function averageInformation(moments: Moments, cycle: Moments, gain: number): void {
  moments.blocks.forEach((block, b) => {
    const fresh = cycle.blocks[b];
    for (let k = 0; k < block.length; k++) block[k] = (block[k] ?? 0) + gain * ((fresh?.[k] ?? 0) - (block[k] ?? 0));
  });
}

function largestChange(previous: Float64Array, current: Float64Array): number {
  let largest = 0;
  for (let k = 0; k < current.length; k++)
    largest = Math.max(largest, Math.abs((current[k] ?? 0) - (previous[k] ?? 0)));
  return largest;
}
