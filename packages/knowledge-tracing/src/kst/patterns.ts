import { DataError } from "@measurekit/core";
import { checkLog, type EventLog, isObserved } from "../data/internal.js";
import { checkElementCount } from "./element-limit.js";

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
 * @remarks
 * The first outcomes are collected by learner and checked before the patterns are laid out, so that a log of many
 * learners, each with few items of their own, is refused at the cost of its events, not of learners × items.
 *
 * @throws `DataError` when the log is malformed, an outcome is not 0 or 1, there are more than 1,000 items, or a learner
 * has not answered every item.
 */
export function responsePatterns(log: EventLog, subject: string): ResponsePatterns {
  checkLog(log);
  // A lesson has no response pattern, and an item met only through events without an outcome is not one either.
  const items = new Set<string>();
  const firsts = new Map<string, Map<string, number>>();
  for (const event of log) {
    if (!isObserved(event)) continue;
    items.add(event.item);
    const answers = firsts.get(event.learner) ?? new Map<string, number>();
    firsts.set(event.learner, answers);
    if (answers.has(event.item)) continue;
    if (event.outcome !== 0 && event.outcome !== 1) {
      throw new DataError(
        `${subject} needs outcomes 0 or 1; learner "${event.learner}" has ${event.outcome} at "${event.item}".`,
      );
    }
    answers.set(event.item, event.outcome);
  }
  const order = [...items];
  checkElementCount(order.length, subject, "items");
  for (const [learner, answers] of firsts) {
    const missing = answers.size < order.length ? order.find((item) => !answers.has(item)) : undefined;
    if (missing !== undefined) {
      throw new DataError(
        `${subject} needs every learner to answer every item; "${learner}" has no answer to "${missing}".`,
      );
    }
  }
  return { items: order, patterns: [...firsts.values()].map((answers) => order.map((item) => answers.get(item) ?? 0)) };
}

/**
 * Returns b: at i × m + j, the number of learners who fail item i but solve item j. The counts are integers below the
 * number of learners, so 32 bits hold them in half the bytes of doubles.
 */
export function counterexamples(patterns: readonly (readonly number[])[], m: number): Int32Array {
  const b = new Int32Array(m * m);
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
