import { checkArray, checkOptions, checkRecord, DataError, type QMatrix } from "@measurekit/core";
import { checkEvent, type Event } from "./event.js";
import { checkItemSkills, skillsOf } from "./q-matrix.js";

/**
 * Events in time order, as {@link eventLog} returns them: a learner's events are in the order they happened, and so
 * are the events of all learners together, which models that update items, such as Elo, need. Any subsequence of a log,
 * such as `log.filter(...)`, is a log. Every function that takes a log checks it.
 */
export type EventLog = readonly Event[];

/** An event as {@link eventLog} takes it: without skills when a Q-matrix gives them. */
export interface EventInput extends Omit<Event, "skills"> {
  /**
   * The skills the item practices.
   *
   * @defaultValue the item's row of the Q-matrix, or none without a Q-matrix
   */
  readonly skills?: readonly string[];
}

/** Options of {@link eventLog}. */
export interface EventLogOptions {
  /**
   * The skills of each item, for events without skills of their own.
   *
   * @defaultValue none: an event without skills of its own practices none
   */
  readonly qMatrix?: QMatrix;
}

/**
 * Builds a log from events in any order: it checks and copies them, takes the skills of events without skills from the
 * Q-matrix, and orders them by time, keeping the given order of events with equal times.
 *
 * @throws `DataError` when there are no events, an event is malformed, the Q-matrix is malformed, or an event without
 * skills has an item the Q-matrix lacks.
 *
 * @example
 * ```ts
 * import { eventLog } from "@measurekit/knowledge-tracing/data";
 *
 * const log = eventLog(
 *   [
 *     { learner: "ana", item: "q1", time: 1, outcome: 1 },
 *     { learner: "budi", item: "q1", time: 2, outcome: 0 },
 *   ],
 *   { qMatrix: { q1: ["fractions"] } },
 * );
 * log[0]; // { learner: "ana", item: "q1", skills: ["fractions"], time: 1, outcome: 1 }
 * ```
 */
export function eventLog(events: readonly EventInput[], options: EventLogOptions = {}): EventLog {
  checkOptions(options);
  const { qMatrix } = options;
  checkEvents(events);
  if (qMatrix !== undefined) checkItemSkills(qMatrix);
  const log = events.map((input, index): Event => {
    const label = `Event ${index + 1}`;
    checkRecord(input, `${label} must be an object`);
    const { learner, item, time, outcome } = input;
    const skills = skillsOfInput(input, qMatrix, label);
    const event = { learner, item, skills, time, outcome };
    checkEvent(event, label);
    return { ...event, skills: [...skills] };
  });
  return log.sort((a, b) => a.time - b.time);
}

/** Returns an input event's skills: its own, its item's row of the Q-matrix, or none without a Q-matrix. */
function skillsOfInput(input: EventInput, qMatrix: QMatrix | undefined, label: string): readonly string[] {
  if (input.skills !== undefined) return input.skills;
  if (qMatrix === undefined || typeof input.item !== "string") return [];
  return skillsOf(qMatrix, input.item, label);
}

/**
 * Checks a log, throwing a `DataError` that names the first problem found: a log is a non-empty array of well-formed
 * events in time order.
 */
export function checkLog(log: EventLog): void {
  checkEvents(log);
  log.forEach((event, index) => {
    checkEvent(event, `Event ${index + 1}`);
    const previous = log[index - 1];
    if (previous !== undefined && event.time < previous.time) {
      throw new DataError(
        `Event ${index + 1}, at time ${event.time}, is earlier than event ${index}, at time ${previous.time}; a log is in time order.`,
      );
    }
  });
}

function checkEvents(events: readonly unknown[]): void {
  checkArray(events, "A log must be an array of events");
  if (events.length === 0) throw new DataError("A log needs at least one event.");
}
