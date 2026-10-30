import { checkLearnerState } from "../data/learner.js";
import { checkDistribution } from "./parameters.js";

/** A learner's state in a KST model: the probability of each knowledge state, in the structure's order. */
export interface KstState {
  /** The learner's id. */
  readonly learner: string;
  /** The probability of each knowledge state; they sum to 1. */
  readonly probabilities: readonly number[];
}

/** Checks a state, which may come from storage, against the number of knowledge states. */
export function checkState(state: KstState, count: number): void {
  const learner = checkLearnerState(state, "KST");
  checkDistribution(state.probabilities, count, `The KST state of learner "${learner}"`);
}
