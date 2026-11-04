import { checkOptions, type FieldNames } from "./checks.js";

/**
 * The share of the decrease that a step along a descent direction must achieve, the constant c₁ of the Armijo
 * condition; 1e-4 is the usual value.
 *
 * @see {@link https://doi.org/10.1007/978-0-387-40065-5 | Nocedal and Wright (2006), section 3.1}
 */
export const ARMIJO = 1e-4;

/** The most times a step is halved by default: the shortest step is about 2⁻³⁹ of the first, under 1e-12. */
const DEFAULT_MAX_HALVINGS = 39;

/** Options of {@link backtrack}. */
export interface BacktrackOptions {
  /**
   * The constant c₁ of the sufficient decrease: a step of length t must lower the objective by c₁ t times the slope; 0
   * asks for no rise only.
   *
   * @defaultValue `ARMIJO`
   */
  readonly decrease?: number;
  /**
   * The rise of the objective that is accepted as rounding, in the units of the objective.
   *
   * @defaultValue `0`
   */
  readonly slack?: number;
  /**
   * The length of the first step.
   *
   * @defaultValue `1`
   */
  readonly start?: number;
  /**
   * The most times the step is halved: the search tries at most `maxHalvings + 1` lengths.
   *
   * @defaultValue `39`
   */
  readonly maxHalvings?: number;
  /**
   * A further test of a length whose objective decreases enough, such as that the gradient there is defined, for what
   * costs more than the objective: it is asked of those lengths alone, and a length it refuses is halved as one whose
   * objective does not decrease enough.
   *
   * @defaultValue every length whose objective decreases enough is accepted
   */
  readonly accept?: (length: number) => boolean;
}

const BACKTRACK_OPTIONS: FieldNames<BacktrackOptions> = {
  decrease: true,
  slack: true,
  start: true,
  maxHalvings: true,
  accept: true,
};

/**
 * Backtracking line search: returns the first of the lengths `start`, `start / 2`, ... at which the objective satisfies
 * the sufficient decrease, or `null` when none does.
 *
 * @remarks
 * A length is accepted when `evaluate(length)` is at most `value + decrease * length * slope + slack`, and `accept`
 * passes it: `value` is the objective at the point, and `slope` its derivative along the direction, below 0 for a
 * descent direction. `evaluate` moves to the trial point of the length and returns the objective there, so that after
 * an accepted length the caller's trial point is the accepted one, and after `null` it is the last one tried. A trial
 * that is infeasible or not finite is rejected by returning `Infinity`. The function does not allocate.
 *
 * @param evaluate - Moves to the point at a length along the direction and returns the objective there.
 * @param value - The objective at the point the search starts from.
 * @param slope - The derivative of the objective along the direction at the point.
 * @param options - The first length, the sufficient decrease, the most halvings and a further test.
 * @returns The accepted length, or `null` when none of the lengths tried satisfied the sufficient decrease.
 * @throws `DataError` when the options are not an object or hold a name that is not an option.
 */
export function backtrack(
  evaluate: (length: number) => number,
  value: number,
  slope: number,
  options: BacktrackOptions = {},
): number | null {
  checkOptions(options, BACKTRACK_OPTIONS);
  const { decrease = ARMIJO, slack = 0, start = 1, maxHalvings = DEFAULT_MAX_HALVINGS, accept } = options;
  let length = start;
  for (let halving = 0; halving <= maxHalvings; halving++) {
    if (evaluate(length) <= value + decrease * length * slope + slack && (accept === undefined || accept(length))) {
      return length;
    }
    length /= 2;
  }
  return null;
}
