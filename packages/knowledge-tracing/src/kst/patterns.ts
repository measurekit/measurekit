import { DataError } from "@measurekit/core";
import { checkLog, type EventLog, isObserved } from "../data/internal.js";

/** The items of a log, and each learner's response pattern over them: 1 for a correct first answer, 0 for a wrong one. */
export interface ResponsePatterns {
  /** The items, in the order of their first event with an outcome. */
  readonly items: string[];
  /** One pattern per learner, in the order of their first event: the outcome at each of `items`. */
  readonly patterns: number[][];
}

/**
 * Returns the items in order of their first event and each learner's first outcome at each, checking completeness.
 * `subject` names what needs the patterns, for the errors.
 *
 * @throws `DataError` when the log is malformed, an outcome is not 0 or 1, or a learner has not answered every item.
 */
export function responsePatterns(log: EventLog, subject: string): ResponsePatterns {
  checkLog(log);
  // A lesson has no response pattern, and an item met only through events without an outcome is not one either.
  const observed = log.filter(isObserved);
  const items = [...new Set(observed.map((event) => event.item))];
  const itemIndex = new Map(items.map((item, i) => [item, i]));
  const patterns = new Map<string, number[]>();
  for (const event of observed) {
    const pattern = patterns.get(event.learner) ?? new Array<number>(items.length).fill(-1);
    patterns.set(event.learner, pattern);
    const i = itemIndex.get(event.item) ?? 0;
    if (pattern[i] !== -1) continue;
    if (event.outcome !== 0 && event.outcome !== 1) {
      throw new DataError(
        `${subject} needs outcomes 0 or 1; learner "${event.learner}" has ${event.outcome} at "${event.item}".`,
      );
    }
    pattern[i] = event.outcome;
  }
  for (const [learner, pattern] of patterns) {
    const missing = pattern.indexOf(-1);
    if (missing !== -1) {
      throw new DataError(
        `${subject} needs every learner to answer every item; "${learner}" has no answer to "${items[missing]}".`,
      );
    }
  }
  return { items, patterns: [...patterns.values()] };
}

/** Returns b: at i × m + j, the number of learners who fail item i but solve item j. */
export function counterexamples(patterns: readonly (readonly number[])[], m: number): Float64Array {
  const b = new Float64Array(m * m);
  for (const pattern of patterns) {
    pattern.forEach((outcome, i) => {
      if (outcome === 1) return;
      pattern.forEach((other, j) => {
        if (other === 1) b[i * m + j] = (b[i * m + j] ?? 0) + 1;
      });
    });
  }
  return b;
}
