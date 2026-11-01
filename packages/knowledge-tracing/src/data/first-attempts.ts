import { checkChoice, checkOptions } from "@measurekit/core";
import type { Event } from "./event.js";
import { checkLog, type EventLog } from "./event-log.js";

/** What {@link firstAttempts} makes of a first attempt that took hints. */
export type HintedFirstAttempt = "incorrect" | "null" | "keep";

/** Options of {@link firstAttempts}. */
export interface FirstAttemptsOptions {
  /**
   * What a first attempt with hints counts as: `"incorrect"`, as DataShop does, outcome 0; `"null"`, no evidence,
   * outcome `null`; or `"keep"`, its outcome as recorded.
   *
   * @defaultValue `"incorrect"`
   */
  readonly hinted?: HintedFirstAttempt;
}

const HINTED: readonly HintedFirstAttempt[] = ["incorrect", "null", "keep"];
const DEFAULT_HINTED = "incorrect";

/**
 * Keeps the first attempt of each learner at each item: the events with `attemptNumber` 1 or, without an
 * `attemptNumber`, the learner's first observed event at the item. Events without an outcome stay.
 *
 * @remarks
 * A first attempt that took hints, `hints` above 0, is recoded by the option `hinted`: DataShop counts a step solved
 * with hints as incorrect. The result is a log in the order of the given one.
 *
 * @throws `DataError` when the log is malformed or an option is invalid.
 */
export function firstAttempts(log: EventLog, options: FirstAttemptsOptions = {}): EventLog {
  checkOptions(options);
  const { hinted = DEFAULT_HINTED } = options;
  checkChoice(hinted, "The option hinted", HINTED);
  checkLog(log);
  const seen = new Set<string>();
  const kept: Event[] = [];
  for (const event of log) {
    if (event.outcome === null) {
      kept.push(event);
      continue;
    }
    const key = `${event.learner}\u0000${event.item}`;
    const first = event.attemptNumber === undefined ? !seen.has(key) : event.attemptNumber === 1;
    seen.add(key);
    if (first) kept.push(recode(event, hinted));
  }
  return kept;
}

function recode(event: Event, hinted: HintedFirstAttempt): Event {
  if (hinted === "keep" || (event.hints ?? 0) === 0) return event;
  return { ...event, outcome: hinted === "incorrect" ? 0 : null };
}
