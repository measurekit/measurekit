import {
  checkBoolean,
  checkNonNegative,
  checkOptions,
  checkPositive,
  checkRecord,
  DataError,
  logistic,
} from "@measurekit/core";
import {
  type Attempt,
  checkAttempt,
  checkEvent,
  checkLearner,
  checkLearnerId,
  checkLog,
  checkObservedEvents,
  type Event,
  type EventLog,
  entry,
  isObserved,
  type ObservedEvent,
  onePass,
} from "../data/internal.js";
import { outcomeLogLikelihood } from "../math/internal.js";
import type { LearnerModel, ModelFit } from "../model.js";
import { checkParameters, type EloItem, type EloParameters } from "./parameters.js";
import { checkState, type EloState, type SkillAbility } from "./state.js";

/** The uncertainty function U(n) = a / (1 + b n): the size of an update to an estimate that n events have updated. */
export interface Uncertainty {
  /** The size of the first update, U(0). */
  readonly a: number;
  /** How fast updates shrink with n; 0 gives a constant K = a. */
  readonly b: number;
}

/** Options of {@link elo}. */
export interface EloOptions {
  /**
   * The uncertainty function.
   *
   * @defaultValue `{ a: 1, b: 0.05 }`
   * @see {@link https://doi.org/10.1016/j.compedu.2016.03.017 | Pelánek (2016)}
   */
  readonly uncertainty?: Uncertainty;
  /**
   * Whether learners have an ability in each skill besides their global ability: the concept-specific (hierarchical)
   * Elo model of Nižnan, Pelánek and Řihák.
   *
   * @defaultValue `false`
   * @see {@link https://eric.ed.gov/?id=ED560514 | Nižnan, Pelánek and Řihák (2015)}
   */
  readonly skills?: boolean;
  /**
   * γ, the size of a skill ability's update relative to the global ability's; only with `skills`.
   *
   * @defaultValue `1`
   */
  readonly skillWeight?: number;
  /**
   * The item ratings to start from, such as a fit's.
   *
   * @defaultValue none: every item starts at difficulty 0
   */
  readonly parameters?: EloParameters;
}

/** An Elo model: a learner model whose item ratings can also be updated online. */
export interface EloModel extends LearnerModel<EloState, EloParameters> {
  /**
   * Replays a log in time order, updating learners and items after each event as online use would, from this model's
   * item ratings and every learner's initial state. An event without an outcome changes no rating and has no
   * prediction.
   *
   * @throws `DataError` when the log is malformed or has no event with an outcome.
   */
  fit(log: EventLog): EloFit;
  /**
   * Returns the rating of the event's item after the event, given the learner's state before it; the model is
   * unchanged. With {@link LearnerModel.update} on the same state and event, it makes one step of a replay. An app
   * stores the rating as the item's row and builds the next model from the rows it needs. An event without an outcome
   * leaves the item's rating as it is: its rating, or `{ difficulty: 0, count: 0 }` for an item the model does not
   * know.
   *
   * @throws `DataError` when the state or the event is malformed, or the event belongs to another learner.
   */
  updateItem(state: EloState, event: Event): EloItem;
}

/** The outcome of replaying a log through an Elo model. */
export interface EloFit extends ModelFit<EloState, EloParameters> {
  /** The model with the item ratings at the end of the replay. */
  readonly model: EloModel;
  /** Each learner's state after their events in the log, by learner id. */
  readonly states: Readonly<Record<string, EloState>>;
}

/**
 * The uncertainty function Pelánek recommends as a starting point.
 *
 * @see {@link https://doi.org/10.1016/j.compedu.2016.03.017 | Pelánek (2016)}
 */
const DEFAULT_UNCERTAINTY: Uncertainty = { a: 1, b: 0.05 };
const DEFAULT_SKILL_WEIGHT = 1;
const UNRATED: EloItem = { difficulty: 0, count: 0 };
const UNPRACTICED: SkillAbility = { ability: 0, count: 0 };

/** The checked settings of a model. */
interface Settings {
  readonly uncertainty: Uncertainty;
  readonly skills: boolean;
  readonly skillWeight: number;
}

/** One step of the Elo system: the prediction of an event, and the learner's and the item's ratings after it. */
interface Step {
  readonly prediction: number;
  readonly state: EloState;
  readonly item: EloItem;
}

/**
 * Creates an Elo learner model.
 *
 * @remarks
 * A learner has an ability θ, an item a difficulty d, both 0 at first, and P(correct) = logistic(θ - d). After an event
 * with outcome y, each estimate moves by its uncertainty U(n) times the error y - P: θ ← θ + U(n_θ)(y - P), d ← d -
 * U(n_d)(y - P), where n counts the estimate's earlier updates. Partial credit outcomes work as they are.
 *
 * With `skills`, a learner also has an ability θ_k in each skill k, and P(correct) = logistic(θ + mean θ_k - d) over
 * the item's skills, unpracticed skills at 0; each θ_k moves by γ U(n_k)(y - P). For an item with one skill this is the
 * hierarchical model of Nižnan, Pelánek and Řihák. The mean keeps the change of the prediction from the skill abilities
 * the same for an item with several skills as for one with a single skill.
 *
 * Elo defines no mastery, so the model has no `mastery`; `predict` gives the probability of answering an item.
 *
 * @param options - The uncertainty function, whether abilities are by skill and how skill abilities weigh in; each
 *   option has its default.
 * @returns The model: `fit` replays a log, `update` and `predict` serve online, and `updateItem` gives the item's new
 *   difficulty.
 * @throws `DataError` when an option is out of range or the parameters are malformed.
 * @see {@link https://doi.org/10.1016/j.compedu.2016.03.017 | Pelánek (2016)}
 * @see {@link https://eric.ed.gov/?id=ED560514 | Nižnan, Pelánek and Řihák (2015)}
 *
 * @example
 * ```ts
 * import { eventLog } from "@measurekit/knowledge-tracing/data";
 * import { elo } from "@measurekit/knowledge-tracing/elo";
 *
 * const log = eventLog([
 *   { learner: "ana", item: "q1", time: 1, outcome: 1 },
 *   { learner: "budi", item: "q1", time: 2, outcome: 0 },
 *   { learner: "ana", item: "q2", time: 3, outcome: 1 },
 * ]);
 * const { model, states } = elo().fit(log);
 * const ana = states.ana ?? model.initialState("ana");
 * const attempt = { learner: "ana", item: "q1", skills: [], time: 4 };
 * model.predict(ana, attempt); // 0.68, the probability of a correct answer
 * model.update(ana, { ...attempt, outcome: 0 }).ability; // 0.24, a new state; ana's was 0.86
 * ```
 */
export function elo(options: EloOptions = {}): EloModel {
  const settings = settingsOf(options);
  const { parameters = { items: {} } } = options;
  checkParameters(parameters);
  const items = new Map(Object.entries(parameters.items));

  return {
    parameters,
    fit(log) {
      checkLog(log);
      checkObservedEvents(log, "an Elo model");
      const ratings = new Map(items);
      const pass = onePass(log, initial, (state, event) => {
        const result = step(settings, state, ratings.get(event.item) ?? UNRATED, event);
        ratings.set(event.item, result.item);
        return {
          state: result.state,
          prediction: result.prediction,
          logLikelihood: outcomeLogLikelihood(result.prediction, event.outcome),
        };
      });
      return { model: elo({ ...options, parameters: { items: Object.fromEntries(ratings) } }), ...pass };
    },
    initialState(learner) {
      checkLearnerId(learner);
      return initial(learner);
    },
    update(state, event) {
      checkStep(state, event);
      if (!isObserved(event)) return copyOf(state);
      return step(settings, state, items.get(event.item) ?? UNRATED, event).state;
    },
    predict(state, attempt) {
      checkState(state);
      checkAttempt(attempt, "The attempt");
      checkLearner(state.learner, attempt);
      return logistic(logit(settings, state, items.get(attempt.item) ?? UNRATED, attempt));
    },
    updateItem(state, event) {
      checkStep(state, event);
      const item = items.get(event.item) ?? UNRATED;
      return isObserved(event) ? step(settings, state, item, event).item : { ...item };
    },
  };
}

function initial(learner: string): EloState {
  return { learner, ability: 0, count: 0, skills: {} };
}

/** Returns a copy of a state, for an event that changes nothing: a result never aliases its input. */
function copyOf(state: EloState): EloState {
  return { ...state, skills: { ...state.skills } };
}

function step(settings: Settings, state: EloState, item: EloItem, event: ObservedEvent): Step {
  const prediction = logistic(logit(settings, state, item, event));
  const error = event.outcome - prediction;
  const { a, b } = settings.uncertainty;
  const change = (count: number): number => (a / (1 + b * count)) * error;
  const skills = settings.skills
    ? {
        ...state.skills,
        ...Object.fromEntries(
          event.skills.map((skill): [string, SkillAbility] => {
            const { ability, count } = skillAbility(state, skill);
            return [skill, { ability: ability + settings.skillWeight * change(count), count: count + 1 }];
          }),
        ),
      }
    : state.skills;
  return {
    prediction,
    state: { learner: state.learner, ability: state.ability + change(state.count), count: state.count + 1, skills },
    item: { difficulty: item.difficulty - change(item.count), count: item.count + 1 },
  };
}

/** Returns θ + mean θ_k - d, the mean over the attempt's skills only with skill abilities. */
function logit(settings: Settings, state: EloState, item: EloItem, attempt: Attempt): number {
  let skillTerm = 0;
  if (settings.skills && attempt.skills.length > 0) {
    for (const skill of attempt.skills) skillTerm += skillAbility(state, skill).ability;
    skillTerm /= attempt.skills.length;
  }
  return state.ability + skillTerm - item.difficulty;
}

function skillAbility(state: EloState, skill: string): SkillAbility {
  return entry(state.skills, skill) ?? UNPRACTICED;
}

function checkStep(state: EloState, event: Event): void {
  checkState(state);
  checkEvent(event, "The event");
  checkLearner(state.learner, event);
}

function settingsOf(options: EloOptions): Settings {
  checkOptions(options);
  const { uncertainty = DEFAULT_UNCERTAINTY, skills = false, skillWeight } = options;
  checkRecord(uncertainty, "The uncertainty function must be an object { a, b }");
  checkPositive(uncertainty.a, "The uncertainty function's a");
  checkNonNegative(uncertainty.b, "The uncertainty function's b");
  checkBoolean(skills, "The option skills");
  if (skillWeight !== undefined) {
    if (!skills) throw new DataError("The option skillWeight needs skills: true.");
    checkPositive(skillWeight, "The option skillWeight");
  }
  return { uncertainty, skills, skillWeight: skillWeight ?? DEFAULT_SKILL_WEIGHT };
}
