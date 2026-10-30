import { checkChoice, checkOptions, checkStopping, DataError, describe } from "@measurekit/core";
import { type Attempt, checkAttempt, checkAttempts, checkEvent, checkHasSkills } from "../data/event.js";
import { checkLog, type EventLog } from "../data/event-log.js";
import { checkLearner, checkLearnerId } from "../data/learner.js";
import { entry, parametersOf } from "../data/record.js";
import { replay, type Steps } from "../data/replay.js";
import { indexOfMax } from "../math/extremes.js";
import type { LearnerModel, ModelFit } from "../model.js";
import { type Blim, type Group, maximumLikelihood, minimumDiscrepancy, type Row } from "./estimation.js";
import { checkParameters, type KstItem, type KstParameters } from "./parameters.js";
import { checkState, type KstState } from "./state.js";
import { type CompiledStructure, compile, elementsOf, holds, type KnowledgeStructure } from "./structure.js";

/** Options of {@link kst}. */
export interface KstOptions {
  /** The knowledge structure. */
  readonly structure: KnowledgeStructure;
  /**
   * What the structure's states hold: `"skills"`, so a state masters an item when it holds all of the attempt's skills,
   * or `"items"`, so it masters the items it holds.
   *
   * @defaultValue `"skills"`
   */
  readonly over?: "skills" | "items";
  /**
   * `"ml"`, maximum likelihood by EM, or `"md"`, minimum discrepancy in one pass.
   *
   * @defaultValue `"ml"`
   */
  readonly method?: "ml" | "md";
  /**
   * The change of the log-likelihood at which EM stops.
   *
   * @defaultValue `1e-8`
   */
  readonly tolerance?: number;
  /**
   * The most EM iterations.
   *
   * @defaultValue `10000`
   */
  readonly maxIterations?: number;
  /**
   * The parameters, which `fit` starts from. Without them, the states have equal probabilities, and items without
   * parameters start at guess and slip 0.1, as in pks.
   */
  readonly parameters?: KstParameters;
}

/** A knowledge state with its probability. */
export interface LikelyState {
  /** The elements of the state, in the domain's order. */
  readonly state: readonly string[];
  /** Its probability. */
  readonly probability: number;
}

/** Options of {@link KstModel.nextQuestion}. */
export interface NextQuestionOptions {
  /**
   * The probability of the most likely state at which the assessment stops.
   *
   * @defaultValue `0.9`
   */
  readonly threshold?: number;
}

/** A KST model: a learner model whose state is a distribution over knowledge states. */
export interface KstModel extends LearnerModel<KstState, KstParameters> {
  /**
   * Fits the BLIM to the log from this model's parameters: each learner's events, given a knowledge state, are
   * independent. The log-likelihood is the marginal one, the states summed over; the fit's states and predictions are
   * those of online inference with the fitted parameters.
   *
   * @throws `DataError` when the log is malformed, an event's item or skills are outside the structure, or the
   * outcomes have probability 0 under the parameters.
   */
  fit(log: EventLog): KstFit;
  /**
   * Returns the probability that the learner's state holds an element of the domain, a skill or an item.
   *
   * @throws `DataError` when the state is malformed or the element is not in the domain.
   */
  mastery(state: KstState, element: string): number;
  /**
   * Returns the learner's most likely knowledge state, the first in the structure's order on ties.
   *
   * @throws `DataError` when the state is malformed.
   */
  mostLikelyState(state: KstState): LikelyState;
  /**
   * Returns the candidate to ask next in an adaptive assessment, by half-split.
   *
   * @remarks
   * It is the one whose mastery is most uncertain, nearest 0.5, the first on ties; or `null` when the most likely
   * state's probability has reached the threshold or there is no candidate. Remove asked items from the candidates, or
   * an uncertain one is asked again.
   *
   * @throws `DataError` when the state, a candidate or the threshold is invalid.
   * @see {@link https://doi.org/10.1111/j.2044-8317.1988.tb00884.x | Falmagne and Doignon (1988)}
   */
  nextQuestion(state: KstState, candidates: readonly Attempt[], options?: NextQuestionOptions): Attempt | null;
}

/** The outcome of fitting a KST model. */
export interface KstFit extends ModelFit<KstState, KstParameters> {
  /** The model with the estimated parameters. */
  readonly model: KstModel;
}

/** The starting guess and slip of an item without parameters, as pks's `blim`. */
const START_RATE = 0.1;
const DEFAULT_OVER = "skills";
const DEFAULT_METHOD = "ml";
const DEFAULT_TOLERANCE = 1e-8;
const DEFAULT_MAX_ITERATIONS = 10_000;
const DEFAULT_THRESHOLD = 0.9;

/**
 * Creates a knowledge space theory learner model.
 *
 * @remarks
 * It is a basic local independence model on a knowledge structure, over skills or items. A learner's state is the
 * probability of each knowledge state; each event updates it by Bayes' rule, a correct answer having probability 1 -
 * slip where the state masters the item and guess where it does not. Outcomes between 0 and 1 count as partial credit.
 *
 * @throws `DataError` when the structure, the options or the parameters are invalid.
 */
export function kst(options: KstOptions): KstModel {
  return create(options).model;
}

/** Builds a KST model with its steps, which `fit` replays the log with. */
function create(options: KstOptions): { model: KstModel; steps: Steps<KstState> } {
  const settings = settingsOf(options);
  const compiled = compile(options.structure);
  const {
    parameters = { probabilities: Array.from({ length: compiled.count }, () => 1 / compiled.count), items: {} },
  } = options;
  checkParameters(parameters, compiled.count);
  const maskOf = attemptMasks(compiled, settings.over);
  const itemOf = (id: string): KstItem => parametersOf(parameters.items, "item", id);
  const checkStep = (state: KstState, attempt: Attempt): void => {
    checkState(state, compiled.count);
    checkLearner(state.learner, attempt);
  };
  const steps: Steps<KstState> = {
    initialState: (learner) => ({ learner, probabilities: [...parameters.probabilities] }),
    predict(state, attempt) {
      const { guess, slip } = itemOf(attempt.item);
      const mastered = sumOver(state.probabilities, maskOf(attempt));
      return mastered * (1 - slip) + (1 - mastered) * guess;
    },
    update(state, event) {
      const { guess, slip } = itemOf(event.item);
      const mask = maskOf(event);
      const y = event.outcome;
      const known = (1 - slip) ** y * slip ** (1 - y);
      const unknown = guess ** y * (1 - guess) ** (1 - y);
      // Plain loops into a packed array: this runs once per state for every event of a fit's replay.
      const probabilities: number[] = [];
      let total = 0;
      for (let s = 0; s < state.probabilities.length; s++) {
        const p = (state.probabilities[s] ?? 0) * (mask[s] === 1 ? known : unknown);
        probabilities.push(p);
        total += p;
      }
      if (!(total > 0)) {
        throw new DataError(
          `The event's outcome has probability 0 under the learner's state and the guess and slip of item "${event.item}".`,
        );
      }
      for (let s = 0; s < probabilities.length; s++) probabilities[s] = (probabilities[s] ?? 0) / total;
      return { learner: state.learner, probabilities };
    },
  };

  const model: KstModel = {
    parameters,
    fit(log) {
      checkLog(log);
      const items = [...new Set(log.map((event) => event.item))];
      const design = groupsOf(log, items, maskOf);
      const start: Blim = {
        probabilities: Float64Array.from(parameters.probabilities),
        guess: Float64Array.from(items, (id) => entry(parameters.items, id)?.guess ?? START_RATE),
        slip: Float64Array.from(items, (id) => entry(parameters.items, id)?.slip ?? START_RATE),
      };
      const result =
        settings.method === "ml" ? maximumLikelihood(design, start, settings) : minimumDiscrepancy(design, start);
      const fitted = create({
        ...options,
        parameters: {
          probabilities: [...result.parameters.probabilities],
          items: {
            ...parameters.items,
            ...Object.fromEntries(
              items.map((id, i) => [
                id,
                { guess: result.parameters.guess[i] ?? 0, slip: result.parameters.slip[i] ?? 0 },
              ]),
            ),
          },
        },
      });
      const { states, predictions } = replay(log, fitted.steps);
      const k = compiled.count - 1 + 2 * items.length;
      const ml = settings.method === "ml";
      return {
        model: fitted.model,
        states,
        predictions,
        logLikelihood: result.logLikelihood,
        aic: ml ? 2 * k - 2 * result.logLikelihood : null,
        bic: ml ? k * Math.log(log.length) - 2 * result.logLikelihood : null,
        converged: result.converged,
        iterations: result.iterations,
      };
    },
    initialState(learner) {
      checkLearnerId(learner);
      return steps.initialState(learner);
    },
    update(state, event) {
      checkEvent(event, "The event");
      checkStep(state, event);
      return steps.update(state, event);
    },
    predict(state, attempt) {
      checkAttempt(attempt, "The attempt");
      checkStep(state, attempt);
      return steps.predict(state, attempt);
    },
    mastery(state, element) {
      checkState(state, compiled.count);
      const e = compiled.index.get(element);
      if (e === undefined) throw new DataError(`"${element}" is not in the structure's domain.`);
      return state.probabilities.reduce((sum, p, s) => sum + (holds(compiled, s, e) ? p : 0), 0);
    },
    mostLikelyState(state) {
      checkState(state, compiled.count);
      const best = indexOfMax(state.probabilities);
      return { state: elementsOf(compiled, best), probability: state.probabilities[best] ?? 0 };
    },
    nextQuestion(state, candidates, questionOptions = {}) {
      checkOptions(questionOptions);
      const { threshold = DEFAULT_THRESHOLD } = questionOptions;
      checkState(state, compiled.count);
      if (!(threshold > 0 && threshold <= 1)) {
        throw new DataError(`The threshold must be above 0 and at most 1, found ${describe(threshold)}.`);
      }
      checkAttempts(candidates, "Candidate");
      if ((state.probabilities[indexOfMax(state.probabilities)] ?? 0) >= threshold) return null;
      let best: Attempt | null = null;
      let distance = Number.POSITIVE_INFINITY;
      for (const candidate of candidates) {
        checkLearner(state.learner, candidate);
        const gap = Math.abs(sumOver(state.probabilities, maskOf(candidate)) - 0.5);
        if (gap < distance) [best, distance] = [candidate, gap];
      }
      return best;
    },
  };
  return { model, steps };
}

/**
 * Returns a lookup of the states that master an attempt, as a 0/1 array over the states: those that hold all of its
 * skills, or its item. Masks depend only on the structure, so each distinct set of elements is computed once.
 */
function attemptMasks(compiled: CompiledStructure, over: "skills" | "items"): (attempt: Attempt) => Uint8Array {
  const masks = new Map<string, Uint8Array>();
  return (attempt) => {
    const elements = over === "items" ? [attempt.item] : [...new Set(attempt.skills)].sort();
    const key = elements.join("\u0000");
    let mask = masks.get(key);
    if (mask === undefined) {
      if (over === "skills") checkHasSkills(attempt, `The attempt at item "${attempt.item}"`, "KST model over skills");
      const indices = elements.map((element) => {
        const e = compiled.index.get(element);
        if (e === undefined) {
          throw new DataError(
            `The attempt at item "${attempt.item}" needs "${element}", which is not in the structure.`,
          );
        }
        return e;
      });
      mask = new Uint8Array(compiled.count);
      for (let s = 0; s < compiled.count; s++) mask[s] = indices.every((e) => holds(compiled, s, e)) ? 1 : 0;
      masks.set(key, mask);
    }
    return mask;
  };
}

/**
 * Returns the learners' answers as groups: each learner's events summed per item and mask, and learners with equal
 * sums merged.
 */
function groupsOf(log: EventLog, items: readonly string[], maskOf: (attempt: Attempt) => Uint8Array): Group[] {
  const index = new Map(items.map((id, i) => [id, i]));
  const masks = new Map<Uint8Array, number>();
  const learners = new Map<string, Map<string, Row>>();
  for (const event of log) {
    const mask = maskOf(event);
    if (!masks.has(mask)) masks.set(mask, masks.size);
    const rows = learners.get(event.learner) ?? new Map<string, Row>();
    learners.set(event.learner, rows);
    const item = index.get(event.item) ?? 0;
    const key = `${item}:${masks.get(mask)}`;
    const row = rows.get(key) ?? { item, mask, count: 0, sum: 0 };
    rows.set(key, { ...row, count: row.count + 1, sum: row.sum + event.outcome });
  }
  const groups = new Map<string, { weight: number; rows: Row[] }>();
  for (const rows of learners.values()) {
    const key = [...rows]
      .map(([at, row]) => `${at}:${row.count}:${row.sum}`)
      .sort()
      .join(",");
    const group = groups.get(key);
    if (group === undefined) groups.set(key, { weight: 1, rows: [...rows.values()] });
    else group.weight += 1;
  }
  return [...groups.values()];
}

function sumOver(probabilities: readonly number[], mask: Uint8Array): number {
  let sum = 0;
  for (let s = 0; s < probabilities.length; s++) if (mask[s] === 1) sum += probabilities[s] ?? 0;
  return sum;
}

interface Settings {
  readonly over: "skills" | "items";
  readonly method: "ml" | "md";
  readonly tolerance: number;
  readonly maxIterations: number;
}

function settingsOf(options: KstOptions): Settings {
  checkOptions(options);
  const {
    over = DEFAULT_OVER,
    method = DEFAULT_METHOD,
    tolerance = DEFAULT_TOLERANCE,
    maxIterations = DEFAULT_MAX_ITERATIONS,
  } = options;
  checkChoice(over, "The option over", ["skills", "items"]);
  checkChoice(method, "The option method", ["ml", "md"]);
  checkStopping(tolerance, maxIterations);
  return { over, method, tolerance, maxIterations };
}
