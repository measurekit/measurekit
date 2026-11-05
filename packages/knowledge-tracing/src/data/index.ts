/**
 * Events and logs: learners' attempts at items with their outcomes, or without one for a learning opportunity, in time
 * order; Q-matrices; first attempts; and splits of a log into folds by learner, item or time.
 *
 * @module @measurekit/knowledge-tracing/data
 */

export type { Attempt, Event } from "./event.js";
export { buildEventLog, type EventInput, type EventLog, type EventLogOptions } from "./event-log.js";
export { type FirstAttemptsOptions, type HintedFirstAttempt, selectFirstAttempts } from "./first-attempts.js";
export { type Fold, splitByItem, splitByLearner, splitByTime } from "./split.js";
