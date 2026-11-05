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

/** One learner's first answers: the indices of the items answered, in the order answered, and the outcome of each. */
export interface Answers {
  /** The index of each item the learner answered, into the log's items. */
  readonly items: Int32Array;
  /** The first outcome at each, 0 or 1. */
  readonly outcomes: Uint8Array;
}

/** The items of a log, and each learner's first answers to those they answered. */
export interface AnswerSets {
  /** The items, in the order of their first event with an outcome. */
  readonly items: string[];
  /** Each learner's answers, in the order of their first event with an outcome. */
  readonly learners: readonly Answers[];
}

/**
 * Returns the items in order of their first event and each learner's first outcome at each item they answered, which
 * may be some of the items only. `subject` names what needs the answers, for the errors.
 *
 * @remarks
 * The answers are held by learner, at the cost of the events, not of learners × items.
 *
 * @throws `DataError` when the log is malformed, an outcome is not 0 or 1, or there are more than 1,000 items.
 */
export function answerSets(log: EventLog, subject: string): AnswerSets {
  checkLog(log);
  // A lesson has no response pattern, and an item met only through events without an outcome is not one either.
  const items = new Map<string, number>();
  const firsts = new Map<string, Map<number, number>>();
  for (const event of log) {
    if (!isObserved(event)) continue;
    let item = items.get(event.item);
    if (item === undefined) {
      item = items.size;
      items.set(event.item, item);
    }
    let answers = firsts.get(event.learner);
    if (answers === undefined) {
      answers = new Map<number, number>();
      firsts.set(event.learner, answers);
    }
    if (answers.has(item)) continue;
    if (event.outcome !== 0 && event.outcome !== 1) {
      throw new DataError(
        `${subject} needs outcomes 0 or 1; learner "${event.learner}" has ${event.outcome} at "${event.item}".`,
      );
    }
    answers.set(item, event.outcome);
  }
  checkElementCount(items.size, subject, "items");
  return { items: [...items.keys()], learners: Array.from(firsts.values(), answersOf) };
}

/** Lays out one learner's answers, an item and its outcome at a time, as the typed arrays' `from` of an iterator is slow. */
function answersOf(answered: ReadonlyMap<number, number>): Answers {
  const items = new Int32Array(answered.size);
  const outcomes = new Uint8Array(answered.size);
  let k = 0;
  for (const [item, outcome] of answered) {
    items[k] = item;
    outcomes[k++] = outcome;
  }
  return { items, outcomes };
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
  const { items, learners } = answerSets(log, subject);
  const m = items.length;
  const patterns = learners.map((answers, learner) => {
    if (answers.items.length < m) {
      const answered = new Set(answers.items);
      const missing = items.find((_, item) => !answered.has(item));
      throw new DataError(
        `${subject} needs every learner to answer every item; "${learnerOrder(log)[learner]}" has no answer to "${missing}".`,
      );
    }
    const pattern = new Array<number>(m).fill(0);
    answers.items.forEach((item, k) => {
      pattern[item] = answers.outcomes[k] ?? 0;
    });
    return pattern;
  });
  return { items, patterns };
}

/** Returns the learners of a log with an outcome, in the order of their first such event. */
function learnerOrder(log: EventLog): string[] {
  const learners = new Set<string>();
  for (const event of log) if (isObserved(event)) learners.add(event.learner);
  return [...learners];
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
