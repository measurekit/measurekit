import {
  checkBoolean,
  checkOptions,
  checkPositiveInteger,
  checkSeed,
  checkStopping,
  checkUnitInterval,
  DataError,
  describe,
  isNonEmptyString,
  isRecord,
} from "@measurekit/core";
import {
  checkAttempt,
  checkEvent,
  checkHasSkills,
  checkObservedHasSkills,
  type Event,
  isObserved,
  resourceOf,
} from "../data/event.js";
import { checkLog, checkObservedEvents, type EventLog } from "../data/event-log.js";
import { checkLearner, checkLearnerId } from "../data/learner.js";
import { entry, parametersOf } from "../data/record.js";
import { replay, type Steps } from "../data/replay.js";
import type { LearnerModel, ModelFit } from "../model.js";
import {
  estimate,
  randomStarts,
  type SkillEstimate,
  type StructureOptions,
  sequencesBySkill,
  startFrom,
  structureOf,
} from "./estimation.js";
import { correctness, learnedMastery, nextMastery, opportunitiesToMastery } from "./inference.js";
import { type BktParameters, type BktSkill, checkParameters, guessSlipOf, learningOf, priorOf } from "./parameters.js";
import { type BktState, checkState } from "./state.js";

/** Options of {@link bkt}. */
export interface BktOptions {
  /**
   * Whether learners can forget a skill they know: a forget rate is estimated.
   *
   * @defaultValue `false`
   */
  readonly forgetting?: boolean;
  /**
   * Whether each item has guess and slip rates of its own, instead of its skill's.
   *
   * @defaultValue `false`
   */
  readonly itemGuessSlip?: boolean;
  /**
   * Whether each resource has learn and forget rates of its own, applied after an attempt with it. A resource is an
   * attempt's `resource`, or its item without one.
   *
   * @defaultValue `false`
   */
  readonly resourceLearning?: boolean;
  /**
   * The group of each learner, by learner id: the learners of a group share a prior for each skill (individualized
   * priors). Learners without a group have the skill's prior.
   *
   * @defaultValue none: every learner has the skill's prior
   */
  readonly priorGroups?: Readonly<Record<string, string>>;
  /**
   * The number of random starts of EM for a skill without parameters; the best by log-likelihood is kept.
   *
   * @defaultValue `5`
   */
  readonly restarts?: number;
  /**
   * The seed of the random starts; each skill draws its starts from a generator with this seed.
   *
   * @defaultValue `0`
   */
  readonly seed?: number;
  /**
   * The largest change of a skill's log-likelihood between EM iterations at which EM has converged.
   *
   * @defaultValue `1e-6`
   */
  readonly tolerance?: number;
  /**
   * The most EM iterations of a start.
   *
   * @defaultValue `500`
   */
  readonly maxIterations?: number;
  /**
   * The parameters to predict with, and to start EM from.
   *
   * @defaultValue none: no skill has parameters
   */
  readonly parameters?: BktParameters;
}

/** Options of {@link BktModel.opportunitiesToMastery}. */
export interface OpportunitiesToMasteryOptions {
  /**
   * P(L) at which a skill counts as mastered.
   *
   * @defaultValue `0.95`
   * @see {@link https://doi.org/10.1007/BF01099821 | Corbett and Anderson (1994)}
   */
  readonly threshold?: number;
}

/** A BKT model: a learner model with mastery and the opportunities expected to reach it. */
export interface BktModel extends LearnerModel<BktState, BktParameters> {
  /**
   * Fits each skill of a log by expectation maximization, independently of the others: from the model's parameters for
   * a skill that has them, otherwise from `restarts` random starts. The fit's `iterations` are the most of any skill's
   * chosen start, and it has `converged` when all of them have; `aic` and `bic` count the rates the events with an
   * outcome estimate, and n of `bic` is their number.
   *
   * @remarks
   * An event without an outcome applies its resource's learn and forget rates alone and adds no guess or slip, so a
   * lesson's rates are estimated from the outcomes that follow it. A skill met only through events without an outcome
   * gets no parameters.
   *
   * @throws `DataError` when the log is malformed or has no event with an outcome, an event with an outcome has no
   * skills or an outcome other than 0 or 1, or an outcome has probability 0 under the model's parameters.
   */
  fit(log: EventLog): BktFit;
  /**
   * Returns P(L), the probability that the learner knows the skill at their next attempt: their prior when they have
   * not practiced it.
   *
   * @throws `DataError` when the state is malformed or the model has no parameters for the skill.
   */
  mastery(state: BktState, skill: string): number;
  /**
   * Returns the number of further opportunities, attempts at the skill, after which P(L) is expected to reach the
   * threshold, before their outcomes are known.
   *
   * @remarks
   * The least n with limit + (P(L) - limit) (1 - P(T) - P(F))ⁿ ≥ threshold, limit = P(T) / (P(T) + P(F)), with the
   * skill's learn and forget rates. 0 when P(L) is at the threshold; `Infinity` when it never gets there or the number
   * exceeds the largest double.
   *
   * @throws `DataError` when the state is malformed, the model has no parameters for the skill, or the threshold is not
   * a probability.
   */
  opportunitiesToMastery(state: BktState, skill: string, options?: OpportunitiesToMasteryOptions): number;
}

/** The outcome of fitting a BKT model to a log. */
export interface BktFit extends ModelFit<BktState, BktParameters> {
  /** The model with the estimated parameters. */
  readonly model: BktModel;
}

const DEFAULT_RESTARTS = 5;
const DEFAULT_SEED = 0;
const DEFAULT_TOLERANCE = 1e-6;
const DEFAULT_MAX_ITERATIONS = 500;
const DEFAULT_THRESHOLD = 0.95;

/** The checked options. */
interface Settings extends StructureOptions {
  readonly groups: Readonly<Record<string, string>> | undefined;
  readonly restarts: number;
  readonly seed: number;
  readonly tolerance: number;
  readonly maxIterations: number;
}

/**
 * Creates a Bayesian knowledge tracing model.
 *
 * @remarks
 * For each skill, a learner either knows it or not, starts knowing it with probability P(L0), answers correctly with
 * probability 1 - P(S) when knowing it and P(G) when not, and after each attempt learns it with probability P(T) and,
 * with `forgetting`, forgets it with probability P(F). The options add pyBKT's extensions: item guess and slip,
 * resource learning and prior groups.
 *
 * An event with several skills is an event of each: each skill's P(L) is updated as for that skill alone, the
 * prediction is the mean of the skills' predictions, and fitting counts the event in every skill's sequences.
 *
 * An event with `outcome: null`, such as a lesson, moves each of its skills' P(L) to P(L) (1 - P(F)) + (1 - P(L))
 * P(T), with the rates of its resource, and a skill without parameters stays as it is.
 *
 * @throws `DataError` when an option is out of range or the parameters are malformed.
 * @see {@link https://doi.org/10.1007/BF01099821 | Corbett and Anderson (1994)}
 */
export function bkt(options: BktOptions = {}): BktModel {
  return create(options).model;
}

/** Builds a BKT model with its steps, which `fit` replays the log with. */
function create(options: BktOptions): { model: BktModel; steps: Steps<BktState> } {
  const settings = settingsOf(options);
  const { parameters = { skills: {} } } = options;
  checkParameters(parameters);
  const skillOf = (skill: string): BktSkill => parametersOf(parameters.skills, "skill", skill);
  const masteryOf = (state: BktState, skill: string): number =>
    entry(state.skills, skill) ?? priorOf(skillOf(skill), entry(settings.groups, state.learner));
  const steps: Steps<BktState> = {
    initialState: (learner) => ({ learner, skills: {} }),
    predict(state, attempt) {
      let sum = 0;
      for (const skill of attempt.skills) {
        sum += correctness(masteryOf(state, skill), guessSlipOf(skillOf(skill), attempt.item));
      }
      return sum / attempt.skills.length;
    },
    update(state, event) {
      const updates: [string, number][] = [];
      for (const skill of event.skills) {
        if (!isObserved(event)) {
          // Without parameters a skill has no prior to move, so a lesson leaves it alone.
          const known = entry(parameters.skills, skill);
          if (known !== undefined) {
            updates.push([skill, learnedMastery(masteryOf(state, skill), learningOf(known, resourceOf(event)))]);
          }
          continue;
        }
        const rates = skillOf(skill);
        const next = nextMastery(
          masteryOf(state, skill),
          event.outcome,
          guessSlipOf(rates, event.item),
          learningOf(rates, resourceOf(event)),
        );
        if (next === null) {
          throw new DataError(
            `The event's outcome has probability 0 for skill "${skill}" under the learner's P(L) and the item's guess and slip.`,
          );
        }
        updates.push([skill, next]);
      }
      return { learner: state.learner, skills: { ...state.skills, ...Object.fromEntries(updates) } };
    },
  };

  const model: BktModel = {
    parameters,
    fit(log) {
      checkLog(log);
      checkObservedEvents(log, "a BKT model");
      log.forEach((event, index) => {
        checkTraceable(event, `Event ${index + 1}`);
      });
      const estimates = new Map<string, SkillEstimate>();
      let parameterCount = 0;
      for (const [skill, sequences] of sequencesBySkill(log, settings.groups)) {
        const structure = structureOf(sequences, settings);
        const given = entry(parameters.skills, skill);
        const starts: readonly [BktSkill, ...BktSkill[]] =
          given === undefined
            ? randomStarts(structure, settings.restarts, settings.seed)
            : [startFrom(given, structure)];
        estimates.set(skill, estimate(sequences, starts, settings.forgetting, settings));
        parameterCount += structure.parameterCount;
      }
      const fitted = create({
        ...options,
        parameters: {
          skills: { ...parameters.skills, ...Object.fromEntries([...estimates].map(([id, { skill }]) => [id, skill])) },
        },
      });
      const { states, predictions, logLikelihood } = replay(log, fitted.steps);
      const fits = [...estimates.values()];
      return {
        model: fitted.model,
        states,
        predictions,
        logLikelihood,
        aic: 2 * parameterCount - 2 * logLikelihood,
        bic: parameterCount * Math.log(log.filter(isObserved).length) - 2 * logLikelihood,
        converged: fits.every((fit) => fit.converged),
        iterations: fits.reduce((most, fit) => Math.max(most, fit.iterations), 0),
      };
    },
    initialState(learner) {
      checkLearnerId(learner);
      return steps.initialState(learner);
    },
    update(state, event) {
      checkState(state);
      checkEvent(event, "The event");
      checkTraceable(event, "The event");
      checkLearner(state.learner, event);
      return steps.update(state, event);
    },
    predict(state, attempt) {
      checkState(state);
      checkAttempt(attempt, "The attempt");
      checkHasSkills(attempt, "The attempt", "BKT model");
      checkLearner(state.learner, attempt);
      return steps.predict(state, attempt);
    },
    mastery(state, skill) {
      checkState(state);
      return masteryOf(state, skill);
    },
    opportunitiesToMastery(state, skill, masteryOptions = {}) {
      checkOptions(masteryOptions);
      const { threshold = DEFAULT_THRESHOLD } = masteryOptions;
      checkState(state);
      checkUnitInterval(threshold, "The threshold");
      return opportunitiesToMastery(masteryOf(state, skill), skillOf(skill), threshold);
    },
  };
  return { model, steps };
}

/**
 * Checks that BKT can trace a well-formed event: with an outcome, at least one skill and an outcome of 0 or 1; without
 * one, anything.
 */
function checkTraceable(event: Event, label: string): void {
  checkObservedHasSkills(event, label, "BKT model");
  if (event.outcome !== null && event.outcome !== 0 && event.outcome !== 1) {
    throw new DataError(`${label} has the outcome ${event.outcome}; BKT needs outcomes of 0 or 1.`);
  }
}

function settingsOf(options: BktOptions): Settings {
  checkOptions(options);
  const {
    forgetting = false,
    itemGuessSlip = false,
    resourceLearning = false,
    priorGroups,
    restarts = DEFAULT_RESTARTS,
    seed = DEFAULT_SEED,
    tolerance = DEFAULT_TOLERANCE,
    maxIterations = DEFAULT_MAX_ITERATIONS,
  } = options;
  for (const [name, value] of Object.entries({ forgetting, itemGuessSlip, resourceLearning })) {
    checkBoolean(value, `The option ${name}`);
  }
  if (priorGroups !== undefined) checkPriorGroups(priorGroups);
  checkPositiveInteger(restarts, "The option restarts");
  checkSeed(seed);
  checkStopping(tolerance, maxIterations);
  const structure = { forgetting, itemGuessSlip, resourceLearning, priorGroups: priorGroups !== undefined };
  return { ...structure, groups: priorGroups, restarts, seed, tolerance, maxIterations };
}

function checkPriorGroups(priorGroups: Readonly<Record<string, string>>): void {
  if (!isRecord(priorGroups)) {
    throw new DataError(
      `The option priorGroups must be an object of groups by learner id, found ${describe(priorGroups)}.`,
    );
  }
  for (const [learner, group] of Object.entries(priorGroups)) {
    if (!isNonEmptyString(group)) {
      throw new DataError(
        `The prior group of learner "${learner}" must be a non-empty string, found ${describe(group)}.`,
      );
    }
  }
}
