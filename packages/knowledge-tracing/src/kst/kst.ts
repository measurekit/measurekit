import { checkChoice, checkOptions, checkStopping, DataError, describe } from "@measurekit/core";
import {
  type Attempt,
  checkAttempt,
  checkAttempts,
  checkEvent,
  checkHasSkills,
  isObserved,
  type ObservedEvent,
} from "../data/event.js";
import { checkLog, checkObservedEvents, type EventLog } from "../data/event-log.js";
import { checkLearner, checkLearnerId } from "../data/learner.js";
import { entry, parametersOf } from "../data/record.js";
import { replay, type Steps } from "../data/replay.js";
import { indexOfMax } from "../math/extremes.js";
import type { LearnerModel, ModelFit } from "../model.js";
import {
  checkRestrictions,
  constraintsOf,
  type EqualRates,
  type FixedRates,
  type Restrictions,
} from "./constraints.js";
import {
  type Blim,
  type BlimFit,
  type Constraints,
  type Group,
  maximumLikelihood,
  meanDiscrepancy,
  minimumDiscrepancy,
  minimumDiscrepancyMaximumLikelihood,
  type Row,
} from "./estimation.js";
import { type GoodnessOfFit, goodnessOfFit, parameterCount } from "./goodness.js";
import { type Identifiability, localIdentifiability } from "./identifiability.js";
import { checkParameters, type KstItem, type KstParameters, START_RATE } from "./parameters.js";
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
   * `"ml"`, maximum likelihood by EM; `"md"`, minimum discrepancy in one pass; or `"mdml"`, maximum likelihood subject
   * to minimum discrepancy: EM in which each learner is spread over the states of least discrepancy only.
   *
   * @defaultValue `"ml"`
   */
  readonly method?: "ml" | "md" | "mdml";
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
   * The parameters, which `fit` starts from. Items without parameters start at guess and slip 0.1, as in pks.
   *
   * @defaultValue none: the states have equal probabilities
   */
  readonly parameters?: KstParameters;
  /**
   * Rates that `fit` holds at the given values, by item: `{ guess: { q1: 0 }, slip: { q2: 0.1 } }`. A fixed rate may be 0
   * or 1, which an estimated one may not.
   *
   * @defaultValue none: every rate is estimated
   */
  readonly fixed?: FixedRates;
  /**
   * Groups of items that `fit` gives one rate: `{ slip: [["q1", "q2"]] }` estimates one slip for both. An item is in one
   * group of a kind, and not in one with a fixed rate of that kind.
   *
   * @defaultValue none: every rate is estimated apart
   */
  readonly equal?: EqualRates;
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
   * those of online inference with the fitted parameters. Events without an outcome are left out of the fit, and
   * `aic` and `bic` count the events with one.
   *
   * @throws `DataError` when the log is malformed or has no event with an outcome, an event's item or skills are
   * outside the structure, `fixed` or `equal` names an item with no event with an outcome, or the outcomes have
   * probability 0 under the parameters.
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
  /**
   * Returns the local identifiability of the model at its parameters: the rank of the Jacobian of the probabilities of
   * the patterns of responses, against the number of parameters that are free.
   *
   * @remarks
   * A rank below the parameters means that some change of them leaves every pattern's probability as it was, so the
   * data cannot tell them apart. The model must be over items, as the patterns are over the elements of the structure;
   * an item without parameters is at guess and slip 0.1. The Jacobian has 2 ^ items rows and a column for each free
   * parameter, and at most 10 million cells (80 MB) in all.
   *
   * @throws `DataError` when the model is over skills, has equal rates, which no reference covers, or fixes the rate of
   * an item outside the structure, or the Jacobian would have more than 10 million cells.
   * @see {@link https://doi.org/10.1016/j.jmp.2016.07.008 | Heller (2017)}
   * @see {@link https://doi.org/10.3758/s13428-012-0187-z | Stefanutti et al. (2012)}
   */
  identifiability(): Identifiability;
}

/** The outcome of fitting a KST model. */
export interface KstFit extends ModelFit<KstState, KstParameters> {
  /** The model with the estimated parameters. */
  readonly model: KstModel;
  /**
   * G² against the model that gives each response pattern its own probability, with pks's degrees of freedom; `null`
   * unless every learner answered every item once with 0 or 1.
   */
  readonly goodnessOfFit: GoodnessOfFit | null;
  /** The mean number of items by which a learner's answers differ from the nearest state; `null` where `goodnessOfFit` is. */
  readonly discrepancy: number | null;
}

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
      // A lesson is not an item: it has no guess or slip, and it does not move the probabilities of the states.
      if (!isObserved(event)) return { learner: state.learner, probabilities: [...state.probabilities] };
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
    fit: (log) => fitLog({ options, settings, compiled, parameters, maskOf }, log),
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
    identifiability() {
      if (settings.over !== "items") {
        throw new DataError(
          'The local identifiability of a model needs responses to the elements of the structure: create the model with over: "items".',
        );
      }
      return localIdentifiability(compiled, parameters, settings.restrictions);
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

/** What a fit needs of the model it is made from. */
interface FitContext {
  readonly options: KstOptions;
  readonly settings: Settings;
  readonly compiled: CompiledStructure;
  readonly parameters: KstParameters;
  readonly maskOf: (attempt: Attempt) => Uint8Array;
}

/** Fits the BLIM of a model to a log, and replays the log through the model with the estimates. */
function fitLog({ options, settings, compiled, parameters, maskOf }: FitContext, log: EventLog): KstFit {
  checkLog(log);
  checkObservedEvents(log, "a KST model");
  const observed = log.filter(isObserved);
  const items = [...new Set(observed.map((event) => event.item))];
  const design = groupsOf(observed, items, maskOf);
  const start: Blim = {
    probabilities: Float64Array.from(parameters.probabilities),
    guess: Float64Array.from(items, (id) => entry(parameters.items, id)?.guess ?? START_RATE),
    slip: Float64Array.from(items, (id) => entry(parameters.items, id)?.slip ?? START_RATE),
  };
  const constraints = constraintsOf(settings.restrictions, items, start, "has no event with an outcome in the log");
  const result = estimate(design, start, constraints, settings);
  const fitted = create({
    ...options,
    parameters: {
      probabilities: [...result.parameters.probabilities],
      items: {
        ...parameters.items,
        ...Object.fromEntries(
          items.map((id, i) => [id, { guess: result.parameters.guess[i] ?? 0, slip: result.parameters.slip[i] ?? 0 }]),
        ),
      },
    },
  });
  const { states, predictions } = replay(log, fitted.steps);
  const k = parameterCount(compiled.count, constraints);
  const ml = settings.method === "ml";
  const complete = answeredOnce(observed, items.length);
  return {
    model: fitted.model,
    states,
    predictions,
    logLikelihood: result.logLikelihood,
    aic: ml ? 2 * k - 2 * result.logLikelihood : null,
    bic: ml ? k * Math.log(observed.length) - 2 * result.logLikelihood : null,
    converged: result.converged,
    iterations: result.iterations,
    goodnessOfFit: complete ? goodnessOfFit(design, result.logLikelihood, k, items.length) : null,
    discrepancy: complete ? meanDiscrepancy(design, result.parameters, constraints) : null,
  };
}

/** Fits the BLIM by the method of the settings. */
function estimate(groups: readonly Group[], start: Blim, constraints: Constraints, settings: Settings): BlimFit {
  if (settings.method === "md") return minimumDiscrepancy(groups, start, constraints);
  return settings.method === "ml"
    ? maximumLikelihood(groups, start, constraints, settings)
    : minimumDiscrepancyMaximumLikelihood(groups, start, constraints, settings);
}

/** Returns whether every learner answered every item once, with 0 or 1: the response patterns of a complete table. */
function answeredOnce(observed: readonly ObservedEvent[], itemCount: number): boolean {
  const answered = new Map<string, Set<string>>();
  for (const { learner, item, outcome } of observed) {
    if (outcome !== 0 && outcome !== 1) return false;
    const items = answered.get(learner) ?? new Set<string>();
    if (items.has(item)) return false;
    answered.set(learner, items.add(item));
  }
  return [...answered.values()].every((items) => items.size === itemCount);
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
function groupsOf(
  log: readonly ObservedEvent[],
  items: readonly string[],
  maskOf: (attempt: Attempt) => Uint8Array,
): Group[] {
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
  readonly method: "ml" | "md" | "mdml";
  readonly tolerance: number;
  readonly maxIterations: number;
  readonly restrictions: Restrictions;
}

function settingsOf(options: KstOptions): Settings {
  checkOptions(options);
  const {
    over = DEFAULT_OVER,
    method = DEFAULT_METHOD,
    tolerance = DEFAULT_TOLERANCE,
    maxIterations = DEFAULT_MAX_ITERATIONS,
    fixed = {},
    equal = {},
  } = options;
  checkChoice(over, "The option over", ["skills", "items"]);
  checkChoice(method, "The option method", ["ml", "md", "mdml"]);
  checkStopping(tolerance, maxIterations);
  return { over, method, tolerance, maxIterations, restrictions: checkRestrictions(fixed, equal) };
}
