import { createRandom, DataError, type Random } from "@measurekit/core";
import type { EventLog } from "../data/event-log.js";
import { entry } from "../data/record.js";
import { type ExpectationMaximizationOptions, expectationMaximization } from "../math/expectation-maximization.js";
import { forwardBackward } from "../math/hidden-markov.js";
import { type BktGuessSlip, type BktLearning, type BktSkill, guessSlipOf, learningOf, priorOf } from "./parameters.js";

/** One learner's attempts at one skill, in time order. */
export interface Sequence {
  readonly learner: string;
  /** The learner's prior group, if they have one. */
  readonly group: string | undefined;
  readonly steps: { readonly item: string; readonly outcome: number }[];
}

/** Which rates of a skill have values by group or item, and how many rates the skill's events estimate. */
export interface Structure {
  readonly forgetting: boolean;
  /** The groups with a prior of their own: those of the skill's learners, with prior groups. */
  readonly groups: readonly string[];
  /** The items with learn and forget rates of their own: those followed by an attempt, with item learning. */
  readonly learningItems: readonly string[];
  /** The items with guess and slip rates of their own: all of the skill's, with item guess and slip. */
  readonly guessSlipItems: readonly string[];
  /** The number of rates the events estimate: k of the information criteria. */
  readonly parameterCount: number;
}

/** What the structure of a skill depends on. */
export interface StructureOptions {
  readonly forgetting: boolean;
  readonly itemLearning: boolean;
  readonly itemGuessSlip: boolean;
  readonly priorGroups: boolean;
}

/** A skill's estimates from one or more starts: the best by log-likelihood. */
export interface SkillEstimate {
  readonly skill: BktSkill;
  readonly logLikelihood: number;
  readonly iterations: number;
  readonly converged: boolean;
}

/** Returns the sequences of each skill, by skill id in the order the skills first appear in the log. */
export function sequencesBySkill(
  log: EventLog,
  priorGroups: Readonly<Record<string, string>> | undefined,
): Map<string, Sequence[]> {
  const bySkill = new Map<string, Map<string, Sequence>>();
  for (const { learner, item, skills, outcome } of log) {
    for (const skill of skills) {
      const learners = bySkill.get(skill) ?? new Map<string, Sequence>();
      bySkill.set(skill, learners);
      const sequence = learners.get(learner) ?? { learner, group: entry(priorGroups, learner), steps: [] };
      learners.set(learner, sequence);
      sequence.steps.push({ item, outcome });
    }
  }
  return new Map([...bySkill].map(([skill, learners]) => [skill, [...learners.values()]]));
}

/** Returns the structure of a skill's sequences. */
export function structureOf(sequences: readonly Sequence[], options: StructureOptions): Structure {
  const groups = options.priorGroups ? distinct(sequences.flatMap((sequence) => sequence.group ?? [])) : [];
  const followed = sequences.flatMap((sequence) => sequence.steps.slice(0, -1).map((step) => step.item));
  const learningItems = options.itemLearning ? distinct(followed) : [];
  const answered = sequences.flatMap((sequence) => sequence.steps.map((step) => step.item));
  const guessSlipItems = options.itemGuessSlip ? distinct(answered) : [];
  // The skill's own rate counts when an event uses it: a learner without a group, or any event without the option.
  const priors = groups.length + (sequences.some((sequence) => sequence.group === undefined) ? 1 : 0);
  const learning = options.itemLearning ? learningItems.length : Math.min(followed.length, 1);
  const guessSlip = options.itemGuessSlip ? guessSlipItems.length : 1;
  return {
    forgetting: options.forgetting,
    groups,
    learningItems,
    guessSlipItems,
    parameterCount: priors + learning * (options.forgetting ? 2 : 1) + 2 * guessSlip,
  };
}

/**
 * Returns a start from given parameters, with the structure's groups and items starting at their own values or, without
 * them, at the skill's, and the forget rate at 0 without forgetting.
 */
export function startFrom(skill: BktSkill, structure: Structure): BktSkill {
  const forget = (rate: number): number => (structure.forgetting ? rate : 0);
  return withRecords(
    { prior: skill.prior, learn: skill.learn, forget: forget(skill.forget), guess: skill.guess, slip: skill.slip },
    structure,
    (group) => priorOf(skill, group),
    (item) => {
      const { learn, forget: rate } = learningOf(skill, item);
      return { learn, forget: forget(rate) };
    },
    (item) => guessSlipOf(skill, item),
  );
}

/**
 * Returns `count` random starts from a generator seeded with `seed`, so that a skill's starts do not depend on the
 * other skills of a log.
 */
export function randomStarts(structure: Structure, count: number, seed: number): [BktSkill, ...BktSkill[]] {
  const random = createRandom(seed);
  return [randomStart(structure, random), ...Array.from({ length: count - 1 }, () => randomStart(structure, random))];
}

/**
 * The upper bounds of the uniform draws of a random start. pyBKT's `random_model_uni` draws P(L0) from U(0, 1), P(T)
 * from U(0, 0.4), P(G) from U(0, 0.4) and P(S) from U(0, 0.3), and has no forgetting; P(F) from U(0, 0.2) is the package's.
 */
const START_BOUNDS = { prior: 1, learn: 0.4, forget: 0.2, guess: 0.4, slip: 0.3 };

/** Returns a random start, drawn for the skill, then for each group and item in sorted order. */
function randomStart(structure: Structure, random: Random): BktSkill {
  const draw = (upper: number): number => upper * random.uniform();
  const learning = (): BktLearning => ({
    learn: draw(START_BOUNDS.learn),
    forget: structure.forgetting ? draw(START_BOUNDS.forget) : 0,
  });
  const guessSlip = (): BktGuessSlip => ({ guess: draw(START_BOUNDS.guess), slip: draw(START_BOUNDS.slip) });
  return withRecords(
    { prior: draw(START_BOUNDS.prior), ...learning(), ...guessSlip() },
    structure,
    () => draw(START_BOUNDS.prior),
    learning,
    guessSlip,
  );
}

function withRecords(
  skill: BktSkill,
  structure: Structure,
  prior: (group: string) => number,
  learning: (item: string) => BktLearning,
  guessSlip: (item: string) => BktGuessSlip,
): BktSkill {
  const { groups, learningItems, guessSlipItems } = structure;
  return {
    ...skill,
    ...(groups.length > 0 && { groupPriors: Object.fromEntries(groups.map((group) => [group, prior(group)])) }),
    ...(learningItems.length > 0 && {
      itemLearning: Object.fromEntries(learningItems.map((item) => [item, learning(item)])),
    }),
    ...(guessSlipItems.length > 0 && {
      itemGuessSlip: Object.fromEntries(guessSlipItems.map((item) => [item, guessSlip(item)])),
    }),
  };
}

/**
 * Estimates a skill's parameters by expectation maximization from each start, and returns the estimates with the
 * highest log-likelihood.
 *
 * @throws `DataError` when an outcome has probability 0 under a start, as with a guess or slip of 0.
 */
export function estimate(
  sequences: readonly Sequence[],
  starts: readonly [BktSkill, ...BktSkill[]],
  forgetting: boolean,
  options: ExpectationMaximizationOptions,
): SkillEstimate {
  const fits = starts.map((start): SkillEstimate => {
    const result = expectationMaximization(
      start,
      (skill) => {
        const { logLikelihood, counts } = expectation(skill, sequences);
        return { logLikelihood, parameters: maximization(skill, counts, forgetting) };
      },
      options,
    );
    const { parameters, logLikelihood, iterations, converged } = result;
    return { skill: parameters, logLikelihood, iterations, converged };
  });
  return fits.reduce((best, fit) => (fit.logLikelihood > best.logLikelihood ? fit : best));
}

/** Expected counts by class: a group or item, or `null` for the skill's own rate. */
type Counts = Map<string | null, Float64Array>;

/** The expected counts of an E-step. */
interface Expectation {
  /** By prior class: [learners who know the skill at their first attempt, learners]. */
  readonly prior: Counts;
  /** By learning class: [transitions from not knowing, of them to knowing, transitions from knowing, of them to not]. */
  readonly learning: Counts;
  /** By guess and slip class: [attempts not knowing, of them correct, attempts knowing, of them incorrect]. */
  readonly guessSlip: Counts;
}

/** The E-step: the log-likelihood of the sequences and the expected counts, by the forward-backward algorithm. */
function expectation(skill: BktSkill, sequences: readonly Sequence[]): { logLikelihood: number; counts: Expectation } {
  const counts: Expectation = { prior: new Map(), learning: new Map(), guessSlip: new Map() };
  const rates = {
    prior: classes((group) => entry(skill.groupPriors, group), skill.prior, counts.prior),
    learning: classes((item) => entry(skill.itemLearning, item), skill, counts.learning),
    guessSlip: classes((item) => entry(skill.itemGuessSlip, item), skill, counts.guessSlip),
  };
  let logLikelihood = 0;
  for (const sequence of sequences) logLikelihood += addSequence(sequence, rates);
  return { logLikelihood, counts };
}

/** A class of a rate: its value, and the sums of its expected counts. */
interface RateClass<T> {
  readonly rates: T;
  readonly sums: Float64Array;
}

/**
 * Returns a lookup of the class of a group or item: its own when the skill has a value for it, the skill's otherwise.
 * Classes are made once per E-step, so the steps of the sequences only look them up.
 */
function classes<T>(
  own: (key: string) => T | undefined,
  skillRates: T,
  counts: Counts,
): (key: string | undefined) => RateClass<T> {
  const made = new Map<string | undefined, RateClass<T>>();
  return (key) => {
    let found = made.get(key);
    if (found === undefined) {
      const rates = key === undefined ? undefined : own(key);
      const id = rates === undefined ? null : (key ?? null);
      const sums = counts.get(id) ?? new Float64Array(4);
      counts.set(id, sums);
      found = { rates: rates ?? skillRates, sums };
      made.set(key, found);
    }
    return found;
  };
}

/** The lookups of an E-step. */
interface Rates {
  readonly prior: (group: string | undefined) => RateClass<number>;
  readonly learning: (item: string) => RateClass<BktLearning>;
  readonly guessSlip: (item: string) => RateClass<BktGuessSlip>;
}

/** Adds a sequence's expected counts to its classes' sums and returns its log-likelihood. */
function addSequence({ learner, group, steps }: Sequence, rates: Rates): number {
  const last = steps.length - 1;
  const transitions = new Float64Array(4 * last);
  const emissions = new Float64Array(2 * steps.length);
  steps.forEach(({ item, outcome }, t) => {
    const { guess, slip } = rates.guessSlip(item).rates;
    emissions[2 * t] = outcome === 1 ? guess : 1 - guess;
    emissions[2 * t + 1] = outcome === 1 ? 1 - slip : slip;
    if (t === last) return;
    const { learn, forget } = rates.learning(item).rates;
    transitions[4 * t] = 1 - learn;
    transitions[4 * t + 1] = learn;
    transitions[4 * t + 2] = forget;
    transitions[4 * t + 3] = 1 - forget;
  });
  const prior = rates.prior(group);
  const initial = Float64Array.of(1 - prior.rates, prior.rates);
  const { logLikelihood, posteriors, transitionPosteriors } = forwardBackward({
    stateCount: 2,
    initial,
    transitions,
    emissions,
  });
  if (!Number.isFinite(logLikelihood)) {
    throw new DataError(
      `The outcomes of learner "${learner}" have probability 0 under the parameters, as with a guess or slip of 0 or 1.`,
    );
  }
  add(prior.sums, posteriors[1] ?? 0, 1);
  steps.forEach(({ item, outcome }, t) => {
    const unknown = posteriors[2 * t] ?? 0;
    const known = posteriors[2 * t + 1] ?? 0;
    add(rates.guessSlip(item).sums, unknown, unknown * outcome, known, known * (1 - outcome));
    if (t === last) return;
    const stay = transitionPosteriors[4 * t] ?? 0;
    const learned = transitionPosteriors[4 * t + 1] ?? 0;
    const forgot = transitionPosteriors[4 * t + 2] ?? 0;
    const kept = transitionPosteriors[4 * t + 3] ?? 0;
    add(rates.learning(item).sums, stay + learned, learned, forgot + kept, forgot);
  });
  return logLikelihood;
}

/** Adds up to four values to the first entries of `sums`. */
function add(sums: Float64Array, a: number, b: number, c = 0, d = 0): void {
  sums[0] = (sums[0] ?? 0) + a;
  sums[1] = (sums[1] ?? 0) + b;
  sums[2] = (sums[2] ?? 0) + c;
  sums[3] = (sums[3] ?? 0) + d;
}

/**
 * The least distance of a fitted P(G) or P(S) from 0 and 1, 2⁻⁵², so that a fitted model gives every outcome a positive
 * probability: data with every outcome correct would otherwise fit P(S) = 0, and an incorrect answer would be impossible.
 */
const MIN_GUESS_SLIP = Number.EPSILON;

/**
 * The M-step: each rate is its expected count over its expected opportunities, P(G) and P(S) bounded by
 * {@link MIN_GUESS_SLIP}, which for a single rate maximizes the expected log-likelihood within the bounds. A rate of the
 * skill used by no event, when every group or item has its own, is the ratio over all of them, the value for groups and
 * items not fitted. A rate without expected opportunities keeps its value.
 */
function maximization(skill: BktSkill, counts: Expectation, forgetting: boolean): BktSkill {
  const ratio = (sums: Float64Array | undefined, numerator: number, denominator: number, current: number): number => {
    const total = sums?.[denominator] ?? 0;
    // A posterior summed from rounded forward and backward variables can exceed its total by an ulp.
    return total > 0 ? Math.min(1, (sums?.[numerator] ?? 0) / total) : current;
  };
  const bounded = (rate: number): number => Math.min(Math.max(rate, MIN_GUESS_SLIP), 1 - MIN_GUESS_SLIP);
  const skillRatio = (classes: Counts, numerator: number, denominator: number, current: number): number =>
    ratio(classes.get(null) ?? pooled(classes), numerator, denominator, current);
  const forget = (rate: number): number => (forgetting ? rate : 0);
  return {
    prior: skillRatio(counts.prior, 0, 1, skill.prior),
    learn: skillRatio(counts.learning, 1, 0, skill.learn),
    forget: forget(skillRatio(counts.learning, 3, 2, skill.forget)),
    guess: bounded(skillRatio(counts.guessSlip, 1, 0, skill.guess)),
    slip: bounded(skillRatio(counts.guessSlip, 3, 2, skill.slip)),
    ...(skill.groupPriors !== undefined && {
      groupPriors: mapRecord(skill.groupPriors, (group, prior) => ratio(counts.prior.get(group), 0, 1, prior)),
    }),
    ...(skill.itemLearning !== undefined && {
      itemLearning: mapRecord(skill.itemLearning, (item, rates) => {
        const sums = counts.learning.get(item);
        return { learn: ratio(sums, 1, 0, rates.learn), forget: forget(ratio(sums, 3, 2, rates.forget)) };
      }),
    }),
    ...(skill.itemGuessSlip !== undefined && {
      itemGuessSlip: mapRecord(skill.itemGuessSlip, (item, rates) => {
        const sums = counts.guessSlip.get(item);
        return { guess: bounded(ratio(sums, 1, 0, rates.guess)), slip: bounded(ratio(sums, 3, 2, rates.slip)) };
      }),
    }),
  };
}

function pooled(classes: Counts): Float64Array {
  const sums = new Float64Array(4);
  for (const values of classes.values()) {
    values.forEach((value, index) => {
      sums[index] = (sums[index] ?? 0) + value;
    });
  }
  return sums;
}

function mapRecord<TValue, TResult>(
  record: Readonly<Record<string, TValue>>,
  map: (key: string, value: TValue) => TResult,
): Record<string, TResult> {
  return Object.fromEntries(Object.entries(record).map(([key, value]) => [key, map(key, value)]));
}

function distinct(values: readonly string[]): string[] {
  return [...new Set(values)].toSorted();
}
