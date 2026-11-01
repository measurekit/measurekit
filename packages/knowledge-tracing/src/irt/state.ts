import { DataError, describe } from "@measurekit/core";
import { checkLearnerState } from "../data/learner.js";
import { indexOfMax } from "../math/extremes.js";

/**
 * A learner's state in an IRT model: the log-density of the posterior of θ at each point of the model's grid, up to a
 * constant. A model without drift keeps its first and second derivatives with respect to θ, which give the posterior
 * mode between grid points; a model with drift keeps the time of the latest event instead, from which it advances the
 * posterior. Plain data, which JSON serializes; its size depends only on the grid.
 */
export interface IrtState {
  /** The learner's id. */
  readonly learner: string;
  /** ln of the posterior density at each grid point, less its maximum. */
  readonly logDensity: readonly number[];
  /**
   * Its first derivative at each grid point.
   *
   * @defaultValue none in a state of a model with drift, which derives it when asked
   */
  readonly slope?: readonly number[];
  /**
   * Its second derivative at each grid point.
   *
   * @defaultValue none in a state of a model with drift, which derives it when asked
   */
  readonly curvature?: readonly number[];
  /**
   * The time of the latest event, in the units of the events' times, from which a model with drift advances the
   * posterior.
   *
   * @defaultValue none: in a state of a model without drift, and before the first event of a model with it
   */
  readonly time?: number;
}

/**
 * Checks a state, which may come from storage, against the number of grid points: a model without drift needs the
 * derivatives, and a model with drift a finite time, if any.
 */
export function checkState(state: IrtState, points: number, drift: boolean): void {
  const learner = checkLearnerState(state, "IRT");
  checkValues(state.logDensity, learner, "logDensity", points);
  if (!drift) {
    for (const name of ["slope", "curvature"] as const) {
      if (state[name] === undefined) {
        throw new DataError(
          `The IRT state of learner "${learner}" has no ${name}, which a model without the option drift needs: a state written by a model with drift is read by a model with it.`,
        );
      }
      checkValues(state[name], learner, name, points);
    }
  }
  if (state.logDensity[indexOfMax(state.logDensity)] !== 0) {
    throw new DataError(
      `The IRT state of learner "${learner}" needs logDensity less its maximum, so that its largest value is 0.`,
    );
  }
  if (drift && state.time !== undefined && !Number.isFinite(state.time)) {
    throw new DataError(
      `The IRT state of learner "${learner}" needs time, a finite number, or none before its first event, found ${describe(state.time)}.`,
    );
  }
}

/** Checks that a field of a state is an array of one finite number per grid point. */
function checkValues(values: unknown, learner: string, name: string, points: number): void {
  if (!(Array.isArray(values) && values.length === points && allFinite(values))) {
    throw new DataError(
      `The IRT state of learner "${learner}" needs ${name}, an array of ${points} finite numbers, one per grid point.`,
    );
  }
}

/** Returns whether every entry is a finite number; a loop is several times as fast as `every` for a state's few values. */
function allFinite(values: readonly unknown[]): boolean {
  for (let i = 0; i < values.length; i++) if (!Number.isFinite(values[i])) return false;
  return true;
}
