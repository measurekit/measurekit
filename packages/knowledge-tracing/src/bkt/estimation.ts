import { createRandom, DataError, type Random, type Stopping } from "@measurekit/core";
import { type EventLog, entry, mapRecord, resourceOf } from "../data/internal.js";
import { expectationMaximization, forwardBackward } from "../math/internal.js";
import { type BktBounds, type BktFixedRates, constrain } from "./constraints.js";
import { type BktGuessSlip, type BktLearning, type BktSkill, guessSlipOf, learningOf, priorOf } from "./parameters.js";

/** One learner's attempts at one skill, in time order. */
export interface Sequence {
  readonly learner: string;
  /** The learner's prior group, if they have one. */
  readonly group: string | undefined;
  /**
   * The attempts: the item, the key of the class of the learn and forget rates that follow it, and the outcome, `null` for
   * an attempt with nothing observed. The key is the resource with resource learning, the resource before it and the
   * resource with pair learning, and `undefined` for the skill's own rates.
   */
  readonly steps: { readonly item: string; readonly learning: string | undefined; readonly outcome: number | null }[];
}

/** Returns the number of steps with an observed outcome in the sequences. */
export function observedSteps(sequences: readonly Sequence[]): number {
  return sequences.reduce((total, { steps }) => total + steps.filter(({ outcome }) => outcome !== null).length, 0);
}

/** Which rates of a skill have values by group or item, and how many rates the skill's events estimate. */
export interface Structure {
  readonly forgetting: boolean;
  /** The groups with a prior of their own: those of the skill's learners, with prior groups. */
  readonly groups: readonly string[];
  /** What the learn and forget rates are keyed by: the resource, the resource before it and the resource, or neither. */
  readonly learningKind: "resource" | "pair" | "none";
  /** The keys of the classes with learn and forget rates of their own: those followed by an outcome. */
  readonly learningClasses: readonly string[];
  /** The items with guess and slip rates of their own: those with an outcome, with item guess and slip. */
  readonly guessSlipItems: readonly string[];
  /** The number of rates the events estimate: k of the information criteria. */
  readonly parameterCount: number;
}

/** What the structure of a skill depends on. */
export interface StructureOptions {
  readonly forgetting: boolean;
  readonly resourceLearning: boolean;
  readonly pairLearning: boolean;
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

/** Returns the key of the class of the rates that follow an attempt with `resource` after one with `previous`. */
function learningKey(
  kind: Structure["learningKind"],
  previous: string | undefined,
  resource: string,
): string | undefined {
  if (kind === "resource") return resource;
  return kind === "pair" && previous !== undefined ? `${previous}${PAIR_SEPARATOR}${resource}` : undefined;
}

/** Separates the resource before an attempt from its own in the key of a pair; no id has a null character. */
const PAIR_SEPARATOR = "\u0000";

/**
 * Returns the sequences of each skill, by skill id in the order the skills first appear in the log. A sequence without
 * an observed outcome has nothing to estimate from, so a skill met only through such sequences is left out.
 */
export function sequencesBySkill(
  log: EventLog,
  priorGroups: Readonly<Record<string, string>> | undefined,
  kind: Structure["learningKind"],
): Map<string, Sequence[]> {
  const bySkill = new Map<string, Map<string, Sequence>>();
  const latest = new Map<Sequence, string>();
  for (const event of log) {
    const { learner, item, skills, outcome } = event;
    const resource = resourceOf(event);
    for (const skill of skills) {
      const learners = bySkill.get(skill) ?? new Map<string, Sequence>();
      bySkill.set(skill, learners);
      const sequence = learners.get(learner) ?? { learner, group: entry(priorGroups, learner), steps: [] };
      learners.set(learner, sequence);
      sequence.steps.push({ item, learning: learningKey(kind, latest.get(sequence), resource), outcome });
      latest.set(sequence, resource);
    }
  }
  const informative = [...bySkill].map(([skill, learners]): [string, Sequence[]] => [
    skill,
    [...learners.values()].filter(({ steps }) => steps.some(({ outcome }) => outcome !== null)),
  ]);
  return new Map(informative.filter(([, sequences]) => sequences.length > 0));
}

/** Returns the structure of a skill's sequences, with the rates that `fixed` holds left out of the count. */
export function structureOf(
  sequences: readonly Sequence[],
  options: StructureOptions,
  fixed: BktFixedRates | undefined,
): Structure {
  const groups = options.priorGroups ? distinct(sequences.flatMap((sequence) => sequence.group ?? [])) : [];
  // A transition teaches something when an outcome follows it, so steps after the last outcome count for nothing.
  const followed = sequences.flatMap(({ steps }) =>
    steps.slice(
      0,
      steps.findLastIndex(({ outcome }) => outcome !== null),
    ),
  );
  const learningKind = options.resourceLearning ? "resource" : options.pairLearning ? "pair" : "none";
  const learningClasses = distinct(followed.flatMap(({ learning }) => learning ?? []));
  const answered = sequences.flatMap(({ steps }) =>
    steps.filter(({ outcome }) => outcome !== null).map(({ item }) => item),
  );
  const guessSlipItems = options.itemGuessSlip ? distinct(answered) : [];
  // The skill's own rate counts when an event uses it: a learner without a group, a first attempt with pairs, or any
  // event without the option. A fixed rate is not estimated.
  const free = (rate: number | undefined): number => (rate === undefined ? 1 : 0);
  const priors = groups.length + (sequences.some((sequence) => sequence.group === undefined) ? free(fixed?.prior) : 0);
  const forgets = (held: Partial<BktLearning> | undefined): number => (options.forgetting ? free(held?.forget) : 0);
  const classRates = learningClasses.reduce((sum, key) => {
    const held = learningKind === "resource" ? entry(fixed?.resources, key) : undefined;
    return sum + free(held?.learn) + forgets(held);
  }, 0);
  const ownRates = followed.some(({ learning }) => learning === undefined) ? free(fixed?.learn) + forgets(fixed) : 0;
  const guessSlip = options.itemGuessSlip ? 2 * guessSlipItems.length : free(fixed?.guess) + free(fixed?.slip);
  return {
    forgetting: options.forgetting,
    groups,
    learningKind,
    learningClasses,
    guessSlipItems,
    parameterCount: priors + classRates + ownRates + guessSlip,
  };
}

/**
 * Returns a start from given parameters, with the structure's groups, classes and items starting at their own values
 * or, without them, at the skill's, and the forget rate at 0 without forgetting. The rates are held to the constraints.
 */
export function startFrom(
  skill: BktSkill,
  structure: Structure,
  fixed: BktFixedRates | undefined,
  bounds: BktBounds,
): BktSkill {
  const forget = (rate: number): number => (structure.forgetting ? rate : 0);
  const start = withRecords(
    { prior: skill.prior, learn: skill.learn, forget: forget(skill.forget), guess: skill.guess, slip: skill.slip },
    structure,
    (group) => priorOf(skill, group),
    (key) => {
      const [previous, resource] = structure.learningKind === "pair" ? splitPair(key) : [undefined, key];
      const { learn, forget: rate } = learningOf(skill, resource, previous);
      return { learn, forget: forget(rate) };
    },
    (item) => guessSlipOf(skill, item),
  );
  return constrain(start, fixed, bounds, false);
}

/** Returns the resource before and the resource that the key of a pair names. */
function splitPair(key: string): [string, string] {
  const [previous = "", resource = ""] = key.split(PAIR_SEPARATOR);
  return [previous, resource];
}

/**
 * Returns `count` random starts from a generator seeded with `seed`, so that a skill's starts do not depend on the
 * other skills of a log.
 */
export function randomStarts(
  structure: Structure,
  count: number,
  seed: number,
  fixed: BktFixedRates | undefined,
  bounds: BktBounds,
): [BktSkill, ...BktSkill[]] {
  const random = createRandom(seed);
  const start = (): BktSkill => constrain(randomStart(structure, random), fixed, bounds, false);
  return [start(), ...Array.from({ length: count - 1 }, start)];
}

/**
 * The upper bounds of the uniform draws of a random start. pyBKT's `random_model_uni` draws P(L0) from U(0, 1), P(T)
 * from U(0, 0.4), P(G) from U(0, 0.4) and P(S) from U(0, 0.3), and has no forgetting; P(F) from U(0, 0.2) is
 * added here.
 */
const START_BOUNDS = { prior: 1, learn: 0.4, forget: 0.2, guess: 0.4, slip: 0.3 };

/** Returns a random start, drawn for the skill, then for each group, class and item in sorted order. */
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
  learning: (key: string) => BktLearning,
  guessSlip: (item: string) => BktGuessSlip,
): BktSkill {
  const { groups, learningKind, learningClasses, guessSlipItems } = structure;
  const pairs: Record<string, Record<string, BktLearning>> = {};
  for (const key of learningKind === "pair" ? learningClasses : []) {
    const [previous, resource] = splitPair(key);
    pairs[previous] = { ...pairs[previous], [resource]: learning(key) };
  }
  return {
    ...skill,
    ...(groups.length > 0 && { groupPriors: Object.fromEntries(groups.map((group) => [group, prior(group)])) }),
    ...(learningKind === "resource" &&
      learningClasses.length > 0 && {
        resourceLearning: Object.fromEntries(learningClasses.map((key) => [key, learning(key)])),
      }),
    ...(learningKind === "pair" && learningClasses.length > 0 && { pairLearning: pairs }),
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
  { fixed, bounds }: { readonly fixed: BktFixedRates | undefined; readonly bounds: BktBounds },
  options: Stopping,
): SkillEstimate {
  const fits = starts.map((start): SkillEstimate => {
    const result = expectationMaximization(
      start,
      (skill) => {
        const { logLikelihood, counts } = expectation(skill, sequences);
        return { logLikelihood, parameters: constrain(maximization(skill, counts, forgetting), fixed, bounds, true) };
      },
      options,
    );
    const { parameters, logLikelihood, iterations, converged } = result;
    return { skill: parameters, logLikelihood, iterations, converged };
  });
  return fits.reduce((best, fit) => (fit.logLikelihood > best.logLikelihood ? fit : best));
}

/** Expected counts by class: a group, resource or item, or `null` for the skill's own rate. */
type Counts = Map<string | null, Float64Array>;

/** The expected counts of an E-step. */
interface Expectation {
  /** By prior class: [learners who know the skill at their first attempt, learners]. */
  readonly prior: Counts;
  /**
   * By learning class: [transitions from not knowing, of them to knowing, transitions from knowing, of them to not].
   */
  readonly learning: Counts;
  /** By guess and slip class: [attempts not knowing, of them correct, attempts knowing, of them incorrect]. */
  readonly guessSlip: Counts;
}

/** The E-step: the log-likelihood of the sequences and the expected counts, by the forward-backward algorithm. */
function expectation(skill: BktSkill, sequences: readonly Sequence[]): { logLikelihood: number; counts: Expectation } {
  const counts: Expectation = { prior: new Map(), learning: new Map(), guessSlip: new Map() };
  const rates = {
    prior: classes((group) => entry(skill.groupPriors, group), skill.prior, counts.prior),
    learning: classes((key) => ownLearning(skill, key), skill, counts.learning),
    guessSlip: classes((item) => entry(skill.itemGuessSlip, item), skill, counts.guessSlip),
  };
  let logLikelihood = 0;
  for (const sequence of sequences) logLikelihood += addSequence(sequence, rates);
  return { logLikelihood, counts };
}

/** Returns the learn and forget rates that the skill has for a class of resources or pairs, if it has any. */
function ownLearning(skill: BktSkill, key: string): BktLearning | undefined {
  if (skill.pairLearning === undefined) return entry(skill.resourceLearning, key);
  const [previous, resource] = splitPair(key);
  return entry(entry(skill.pairLearning, previous), resource);
}

/** A class of a rate: its value, and the sums of its expected counts. */
interface RateClass<T> {
  readonly rates: T;
  readonly sums: Float64Array;
}

/**
 * Returns a lookup of the class of a group, resource or item: its own when the skill has a value for it, the skill's
 * otherwise.
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
  readonly learning: (key: string | undefined) => RateClass<BktLearning>;
  readonly guessSlip: (item: string) => RateClass<BktGuessSlip>;
}

/** Adds a sequence's expected counts to its classes' sums and returns its log-likelihood. */
function addSequence({ learner, group, steps }: Sequence, rates: Rates): number {
  const last = steps.length - 1;
  const transitions = new Float64Array(4 * last);
  const emissions = new Float64Array(2 * steps.length);
  steps.forEach(({ item, learning, outcome }, t) => {
    if (outcome === null) {
      // An attempt with nothing observed has probability 1 in both states.
      emissions[2 * t] = 1;
      emissions[2 * t + 1] = 1;
    } else {
      const { guess, slip } = rates.guessSlip(item).rates;
      emissions[2 * t] = outcome === 1 ? guess : 1 - guess;
      emissions[2 * t + 1] = outcome === 1 ? 1 - slip : slip;
    }
    if (t === last) return;
    const { learn, forget } = rates.learning(learning).rates;
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
  steps.forEach(({ item, learning, outcome }, t) => {
    const unknown = posteriors[2 * t] ?? 0;
    const known = posteriors[2 * t + 1] ?? 0;
    if (outcome !== null) add(rates.guessSlip(item).sums, unknown, unknown * outcome, known, known * (1 - outcome));
    if (t === last) return;
    const stay = transitionPosteriors[4 * t] ?? 0;
    const learned = transitionPosteriors[4 * t + 1] ?? 0;
    const forgot = transitionPosteriors[4 * t + 2] ?? 0;
    const kept = transitionPosteriors[4 * t + 3] ?? 0;
    add(rates.learning(learning).sums, stay + learned, learned, forgot + kept, forgot);
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
 * The M-step: each rate is its expected count over its expected opportunities, which for a single rate maximizes the
 * expected log-likelihood; the caller holds the rates to the fixed values and the bounds, which for a single rate is the
 * exact constrained maximum. A rate of the skill used by no event, when every group, class or item has its own, is the
 * ratio over all of them, the value for groups, classes and items not fitted. A rate without expected opportunities
 * keeps its value.
 */
function maximization(skill: BktSkill, counts: Expectation, forgetting: boolean): BktSkill {
  const ratio = (sums: Float64Array | undefined, numerator: number, denominator: number, current: number): number => {
    const total = sums?.[denominator] ?? 0;
    // A posterior summed from rounded forward and backward variables can exceed its total by an ulp.
    return total > 0 ? Math.min(1, (sums?.[numerator] ?? 0) / total) : current;
  };
  const skillRatio = (classes: Counts, numerator: number, denominator: number, current: number): number =>
    ratio(classes.get(null) ?? pooled(classes), numerator, denominator, current);
  const forget = (rate: number): number => (forgetting ? rate : 0);
  const learning = (rates: BktLearning, key: string): BktLearning => {
    const sums = counts.learning.get(key);
    return { learn: ratio(sums, 1, 0, rates.learn), forget: forget(ratio(sums, 3, 2, rates.forget)) };
  };
  return {
    prior: skillRatio(counts.prior, 0, 1, skill.prior),
    learn: skillRatio(counts.learning, 1, 0, skill.learn),
    forget: forget(skillRatio(counts.learning, 3, 2, skill.forget)),
    guess: skillRatio(counts.guessSlip, 1, 0, skill.guess),
    slip: skillRatio(counts.guessSlip, 3, 2, skill.slip),
    ...(skill.groupPriors !== undefined && {
      groupPriors: mapRecord(skill.groupPriors, (prior, group) => ratio(counts.prior.get(group), 0, 1, prior)),
    }),
    ...(skill.resourceLearning !== undefined && {
      resourceLearning: mapRecord(skill.resourceLearning, learning),
    }),
    ...(skill.pairLearning !== undefined && {
      pairLearning: mapRecord(skill.pairLearning, (pairs, previous) =>
        mapRecord(pairs, (rates, resource) => learning(rates, `${previous}${PAIR_SEPARATOR}${resource}`)),
      ),
    }),
    ...(skill.itemGuessSlip !== undefined && {
      itemGuessSlip: mapRecord(skill.itemGuessSlip, (rates, item) => {
        const sums = counts.guessSlip.get(item);
        return { guess: ratio(sums, 1, 0, rates.guess), slip: ratio(sums, 3, 2, rates.slip) };
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

function distinct(values: readonly string[]): string[] {
  return [...new Set(values)].toSorted();
}
