/**
 * Events and logs: learners' attempts at items with their outcomes, in time order; Q-matrices; and splits of a log into
 * folds by learner, item or time.
 *
 * @module @measurekit/knowledge-tracing/data
 */
export type { Attempt, Event } from "./event.js";
export { type EventInput, type EventLog, type EventLogOptions, eventLog } from "./event-log.js";
export type { QMatrix } from "./q-matrix.js";
export { type Fold, splitByItem, splitByLearner, splitByTime } from "./split.js";
