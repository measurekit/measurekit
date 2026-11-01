import { DataError } from "@measurekit/core";
import { isObserved } from "../data/event.js";
import type { EventLog } from "../data/event-log.js";

/** The cells of a table by opportunity, from opportunity 1 at index 0: over all skills, and for each skill. */
export interface ByOpportunity<T> {
  readonly overall: T[];
  /** By skill, in the order the skills first appear. */
  readonly skills: Map<string, T[]>;
}

/**
 * Throws unless an event with an outcome of the log has skills: the opportunities of a log without skills are none.
 *
 * @param subject - Starts the message: what needs the events with skills, such as `A learning curve needs`.
 */
export function checkHasSkills(log: EventLog, subject: string): void {
  if (log.every((event) => !isObserved(event) || event.skills.length === 0)) {
    throw new DataError(`${subject} events with skills; no event of the log has any.`);
  }
}

/**
 * Collects a cell of each opportunity of each skill, by calling `add` with the cell and the position in the log of
 * every event with an outcome at it. An opportunity is the number of times the learner has practiced the skill, the
 * event included, counting the events with an outcome; an event with several skills is at one opportunity of each.
 * Only the learner and skill pairs with at least `minOpportunities` opportunities in the log take part.
 */
export function collectByOpportunity<T>(
  log: EventLog,
  minOpportunities: number,
  create: () => T,
  add: (cell: T, index: number) => void,
): ByOpportunity<T> {
  const totals = minOpportunities > 1 ? practiceCounts(log) : null;
  const running = new Map<string, Map<string, number>>();
  const overall: T[] = [];
  const skills = new Map<string, T[]>();
  log.forEach((event, index) => {
    if (!isObserved(event)) return;
    const counts = running.get(event.learner) ?? new Map<string, number>();
    running.set(event.learner, counts);
    for (const skill of event.skills) {
      const opportunity = (counts.get(skill) ?? 0) + 1;
      counts.set(skill, opportunity);
      if (totals !== null && (totals.get(event.learner)?.get(skill) ?? 0) < minOpportunities) continue;
      const own = skills.get(skill) ?? [];
      skills.set(skill, own);
      add(cellAt(overall, opportunity, create), index);
      add(cellAt(own, opportunity, create), index);
    }
  });
  return { overall, skills };
}

/** Returns the number of events with an outcome of each learner at each skill. */
function practiceCounts(log: EventLog): Map<string, Map<string, number>> {
  const totals = new Map<string, Map<string, number>>();
  for (const event of log) {
    if (!isObserved(event)) continue;
    const counts = totals.get(event.learner) ?? new Map<string, number>();
    totals.set(event.learner, counts);
    for (const skill of event.skills) counts.set(skill, (counts.get(skill) ?? 0) + 1);
  }
  return totals;
}

/** Returns the cell of an opportunity, made when it is the first event there: a learner reaches k only after k - 1. */
function cellAt<T>(cells: T[], opportunity: number, create: () => T): T {
  const existing = cells[opportunity - 1];
  if (existing !== undefined) return existing;
  const made = create();
  cells[opportunity - 1] = made;
  return made;
}
