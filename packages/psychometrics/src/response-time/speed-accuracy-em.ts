import { iterateToConvergence, largestChange } from "@measurekit/core";
import { type ExpectationStep, expectationStep, type Maximization, maximization, type Setup } from "../irt/internal.js";
import {
  conditional,
  emptySums,
  fillLogTerms,
  fillSums,
  maximizeTimes,
  type PersonSums,
  type ThetaMoments,
} from "./speed-accuracy-time.js";
import { forEachTime, type TimedResponses } from "./timed-responses.js";

/** The hierarchical model of speed and accuracy during estimation: the items and the time model with their E-step. */
export interface JointModel<TParameters> {
  readonly timed: TimedResponses;
  /** The IRT items and their free parameters. */
  readonly setup: Setup<TParameters>;
  readonly mStep: Maximization<TParameters>;
  readonly eStep: ExpectationStep;
  /** The quadrature nodes over theta, N(0, 1). */
  readonly nodes: Float64Array;
  /** Each person's log density of the times at each node, which the E-step adds to the responses'. */
  readonly logTerms: Float64Array;
  readonly sums: PersonSums;
  /** The time parameters: β, α, σ_τ and ρ. */
  readonly parameters: Float64Array;
  /** The number of times of each item. */
  readonly itemCounts: Float64Array;
  /** The posterior mean and second moment of each person's theta at the last E-step. */
  readonly moments: ThetaMoments;
}

/** Builds the joint model from the IRT setup and the time parameters to start from. */
export function jointModel<TParameters>(
  timed: TimedResponses,
  setup: Setup<TParameters>,
  nodes: Float64Array,
  parameters: Float64Array,
): JointModel<TParameters> {
  const persons = timed.personIds.length;
  const itemCounts = new Float64Array(timed.itemIds.length);
  forEachTime(timed, (_, item) => {
    itemCounts[item] = (itemCounts[item] ?? 0) + 1;
  });
  const logTerms = new Float64Array(persons * nodes.length);
  return {
    timed,
    setup,
    mStep: maximization(setup, nodes),
    eStep: expectationStep(timed, setup.items, nodes, setup.population, null, logTerms),
    nodes,
    logTerms,
    sums: emptySums(persons),
    parameters,
    itemCounts,
    moments: { mean: new Float64Array(persons), square: new Float64Array(persons) },
  };
}

/**
 * Runs the E-step at the current parameters and returns the marginal log-likelihood of the responses and times.
 *
 * @remarks
 * The time parameters give each person's log density of the times at each node, which the E-step adds to that of the
 * responses; the posterior of theta that results gives the moments the M-step and the gradient of the times need.
 */
export function expectation<TParameters>(joint: JointModel<TParameters>): number {
  const { timed, sums, parameters, nodes, logTerms, eStep, moments } = joint;
  const itemCount = joint.itemCounts.length;
  fillSums(timed, parameters.subarray(0, itemCount), parameters.subarray(itemCount, 2 * itemCount), sums);
  fillLogTerms(sums, conditional(parameters[2 * itemCount] ?? 1, parameters[2 * itemCount + 1] ?? 0), nodes, logTerms);
  const logLikelihood = eStep.run();
  moments.mean.fill(0);
  moments.square.fill(0);
  eStep.forEachPosterior((person, posterior) => {
    let mean = 0;
    let square = 0;
    for (let q = 0; q < nodes.length; q++) {
      const weight = posterior[q] ?? 0;
      mean += weight * (nodes[q] ?? 0);
      square += weight * (nodes[q] ?? 0) ** 2;
    }
    moments.mean[person] = mean;
    moments.square[person] = square;
  });
  return logLikelihood;
}

/**
 * Runs EM cycles until no parameter changes by more than `tolerance`, or `maxIterations` cycles: the E-step of
 * {@link expectation}, a Fisher-scoring step for each block of items, and the closed-form step for the time model.
 */
export function iterate<TParameters>(
  joint: JointModel<TParameters>,
  tolerance: number,
  maxIterations: number,
): { readonly converged: boolean; readonly iterations: number } {
  const { setup, mStep, timed, sums, parameters, itemCounts, moments } = joint;
  const previousItems = new Float64Array(setup.values.length);
  const previousTimes = new Float64Array(parameters.length);
  return iterateToConvergence(
    () => {
      expectation(joint);
      previousItems.set(setup.values);
      previousTimes.set(parameters);
      for (const block of mStep.blocks) mStep.maximize(block);
      maximizeTimes(timed, sums, parameters, itemCounts, moments);
      return Math.max(largestChange(previousItems, setup.values), largestChange(previousTimes, parameters));
    },
    { tolerance, maxIterations },
  );
}
