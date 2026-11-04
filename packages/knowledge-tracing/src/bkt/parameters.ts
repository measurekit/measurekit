import { checkRecord, checkRecordById, DataError, describe, isUnitInterval } from "@measurekit/core";
import { entry } from "../data/internal.js";

/**
 * The parameters of a BKT model: plain data, which JSON serializes. Skills are independent, so the parameters of fits
 * to separate skills merge by spreading their `skills` into one object.
 */
export interface BktParameters {
  /** The parameters of each skill, by skill id. */
  readonly skills: Readonly<Record<string, BktSkill>>;
}

/**
 * The parameters of one skill. An item uses its own guess and slip, and a resource its own learn and forget rates, when
 * the skill has them and the skill's otherwise; a learner uses their group's prior when the skill has one and the
 * skill's otherwise.
 */
export interface BktSkill {
  /** P(L0), the probability that a learner knows the skill before practicing it. */
  readonly prior: number;
  /** P(T), the probability of learning the skill from an attempt. */
  readonly learn: number;
  /** P(F), the probability of forgetting the skill after an attempt; 0 without forgetting. */
  readonly forget: number;
  /** P(G), the probability of a correct answer without knowing the skill. */
  readonly guess: number;
  /** P(S), the probability of an incorrect answer despite knowing the skill. */
  readonly slip: number;
  /** The prior of each group of learners, by group; absent without prior groups. */
  readonly groupPriors?: Readonly<Record<string, number>>;
  /** The learn and forget rates of resources, by resource id; absent without resource learning. */
  readonly resourceLearning?: Readonly<Record<string, BktLearning>>;
  /**
   * The learn and forget rates of an attempt by the resource before it, then its own; absent without pair learning. A
   * learner's first attempt at the skill uses the skill's own rates.
   */
  readonly pairLearning?: Readonly<Record<string, Readonly<Record<string, BktLearning>>>>;
  /** The guess and slip rates of items, by item id; absent without item guess and slip. */
  readonly itemGuessSlip?: Readonly<Record<string, BktGuessSlip>>;
}

/** The learn and forget rates that follow an attempt with a resource. */
export interface BktLearning {
  /** P(T), the probability of learning the skill from the attempt. */
  readonly learn: number;
  /** P(F), the probability of forgetting the skill after the attempt. */
  readonly forget: number;
}

/** The guess and slip rates of an item. */
export interface BktGuessSlip {
  /** P(G), the probability of a correct answer without knowing the skill. */
  readonly guess: number;
  /** P(S), the probability of an incorrect answer despite knowing the skill. */
  readonly slip: number;
}

/** Returns the prior of a learner in a group, or of a learner without one. */
export function priorOf(skill: BktSkill, group: string | undefined): number {
  return (group === undefined ? undefined : entry(skill.groupPriors, group)) ?? skill.prior;
}

/**
 * Returns the learn and forget rates that follow an attempt with a resource: its pair with the resource before it, when
 * there is one and the skill has it, then the resource's own and the skill's.
 */
export function learningOf(skill: BktSkill, resource: string, previous?: string): BktLearning {
  const pair = previous === undefined ? undefined : entry(entry(skill.pairLearning, previous), resource);
  return pair ?? entry(skill.resourceLearning, resource) ?? skill;
}

/** Returns the guess and slip rates of an item. */
export function guessSlipOf(skill: BktSkill, item: string): BktGuessSlip {
  return entry(skill.itemGuessSlip, item) ?? skill;
}

/** Checks parameters, which may come from storage, throwing a `DataError` that names the first problem found. */
export function checkParameters(parameters: BktParameters): void {
  checkRecord(parameters, "BKT parameters must be an object");
  checkRecordById(parameters.skills, "BKT parameters need skills, an object of skill parameters by skill id");
  for (const [id, skill] of Object.entries(parameters.skills)) {
    const label = `The BKT parameters of skill "${id}"`;
    checkRecord(skill, `${label} must be an object`);
    const { prior, learn, forget, guess, slip, groupPriors, resourceLearning, pairLearning, itemGuessSlip } = skill;
    checkRates(label, { prior, learn, forget, guess, slip });
    for (const [group, groupPrior] of records(groupPriors, `${label} need groupPriors, an object of priors by group`)) {
      checkRates(`${label}, group "${group}",`, { prior: groupPrior });
    }
    for (const [resource, rates] of records(
      resourceLearning,
      `${label} need resourceLearning, an object of rates by resource id`,
    )) {
      checkRecord(rates, `${label}, resource "${resource}", must be an object`);
      checkRates(`${label}, resource "${resource}",`, { learn: rates.learn, forget: rates.forget });
    }
    for (const [previous, pairs] of records(
      pairLearning,
      `${label} need pairLearning, an object of rates by the resource before and the resource`,
    )) {
      checkRecordById(pairs, `${label}, after resource "${previous}", must be an object`);
      for (const [resource, rates] of Object.entries(pairs)) {
        checkRecord(rates, `${label}, resource "${resource}" after "${previous}", must be an object`);
        checkRates(`${label}, resource "${resource}" after "${previous}",`, {
          learn: rates.learn,
          forget: rates.forget,
        });
      }
    }
    for (const [item, rates] of records(itemGuessSlip, `${label} need itemGuessSlip, an object of rates by item id`)) {
      checkRecord(rates, `${label}, item "${item}", must be an object`);
      checkRates(`${label}, item "${item}",`, { guess: rates.guess, slip: rates.slip });
    }
  }
}

/** Returns the entries of an optional record, checking that it is one. */
function records<T>(record: Readonly<Record<string, T>> | undefined, message: string): [string, T][] {
  if (record === undefined) return [];
  checkRecordById(record, message);
  return Object.entries(record);
}

function checkRates(label: string, rates: Readonly<Record<string, unknown>>): void {
  for (const [name, value] of Object.entries(rates)) {
    if (!isUnitInterval(value)) {
      throw new DataError(`${label} ${name} must be a probability from 0 to 1, found ${describe(value)}.`);
    }
  }
}
