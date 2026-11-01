import {
  checkChoice,
  checkNonNegative,
  checkOptions,
  checkStopping,
  mean,
  logistic as sigmoid,
} from "@measurekit/core";
import {
  type Attempt,
  checkAttempt,
  checkEvent,
  checkHasSkills,
  checkObservedHasSkills,
  type Event,
  isObserved,
} from "../data/event.js";
import { checkLog, checkObservedEvents, type EventLog } from "../data/event-log.js";
import { checkLearner, checkLearnerId } from "../data/learner.js";
import { entry, parametersOf } from "../data/record.js";
import { replay, type Steps } from "../data/replay.js";
import type { LearnerModel, ModelFit } from "../model.js";
import { estimate } from "./estimation.js";
import {
  checkFeatures,
  decaysOf,
  exposureDecaysOf,
  featureValue,
  type LogisticFeature,
  usesExposures,
} from "./features.js";
import {
  checkParameters,
  type LogisticParameters,
  type LogisticSkill,
  PROFICIENCIES,
  type Proficiency,
} from "./parameters.js";
import { checkOrder, checkState, type LogisticState, nextExposure, nextHistory } from "./state.js";

/** Options of {@link logistic}. */
export interface LogisticOptions {
  /**
   * The features of each skill, each with a coefficient per skill besides the skill's intercept.
   *
   * @defaultValue `[]`
   */
  readonly features?: readonly LogisticFeature[];
  /**
   * Learners' proficiency θ: `"none"`; `"fixed"`, a free θ per learner; or `"random"`, θ from N(0, σ²) with σ
   * estimated, fitted by the Laplace approximation as `lme4::glmer(nAGQ = 1)`.
   *
   * @defaultValue `"none"`
   */
  readonly proficiency?: Proficiency;
  /**
   * λ of the penalty λ / 2 Σ b² over the coefficients and fixed proficiencies, as LKT's LiblineaR fits use.
   *
   * @defaultValue `0`
   */
  readonly ridge?: number;
  /**
   * The largest absolute gradient component at which the fit has converged.
   *
   * @defaultValue `1e-8`
   */
  readonly tolerance?: number;
  /**
   * The most iterations of the fit.
   *
   * @defaultValue `100`
   */
  readonly maxIterations?: number;
  /**
   * The parameters to predict with.
   *
   * @defaultValue none: no skill has parameters
   */
  readonly parameters?: LogisticParameters;
}

/** Options of {@link afm}: those of {@link logistic} but the features, which AFM fixes. */
export interface AfmOptions extends Omit<LogisticOptions, "features" | "proficiency"> {
  /**
   * Learners' proficiency θ, as {@link LogisticOptions.proficiency}.
   *
   * @defaultValue `"fixed"`
   */
  readonly proficiency?: Proficiency;
}

/** A logistic learner model: AFM, PFA or a model of LKT features. */
export interface LogisticModel extends LearnerModel<LogisticState, LogisticParameters> {
  /**
   * Fits the coefficients and proficiencies to a log by maximum likelihood, penalized with the ridge. The parameters of
   * the fit cover the skills and learners of the log.
   *
   * @throws `DataError` when the log is malformed or an event has no skills.
   */
  fit(log: EventLog): LogisticFit;
}

/** The outcome of fitting a logistic model to a log. */
export interface LogisticFit extends ModelFit<LogisticState, LogisticParameters> {
  /** The model with the estimated parameters. */
  readonly model: LogisticModel;
}

/**
 * Creates a logistic learner model.
 *
 * @remarks
 * The model is logit P(correct) = θ + Σ over the attempt's skills k of (β_k + Σ_f w_kf x_kf), with θ the learner's
 * proficiency, β_k the skill's intercept, and x_kf the features of the learner's history with the skill before the
 * attempt, each with a coefficient w_kf per skill. A learner the fit did not see has the mean proficiency of the fitted
 * learners with fixed proficiency, and 0, the population mean, with a random intercept.
 *
 * @throws `DataError` when an option is out of range, a feature is unknown, or the parameters are malformed or lack a
 * feature's coefficients.
 */
export function logistic(options: LogisticOptions = {}): LogisticModel {
  return create(options).model;
}

const DEFAULT_PROFICIENCY: Proficiency = "none";
/**
 * AFM's proficiency: the model has a learner parameter θ.
 *
 * @see {@link https://doi.org/10.1007/11774303_17 | Cen, Koedinger and Junker (2006)}
 */
const AFM_PROFICIENCY: Proficiency = "fixed";
const DEFAULT_RIDGE = 0;
const DEFAULT_TOLERANCE = 1e-8;
const DEFAULT_MAX_ITERATIONS = 100;

/** Builds a logistic model with its steps, which `fit` replays the log with. */
function create(options: LogisticOptions): { model: LogisticModel; steps: Steps<LogisticState> } {
  checkOptions(options);
  const {
    features = [],
    proficiency = DEFAULT_PROFICIENCY,
    ridge = DEFAULT_RIDGE,
    tolerance = DEFAULT_TOLERANCE,
    maxIterations = DEFAULT_MAX_ITERATIONS,
  } = options;
  const specs = checkFeatures(features);
  checkChoice(proficiency, "The option proficiency", PROFICIENCIES);
  checkNonNegative(ridge, "The option ridge");
  checkStopping(tolerance, maxIterations);
  const { parameters = { skills: {} } } = options;
  const labels = specs.map((spec) => spec.label);
  checkParameters(parameters, labels);
  const decays = decaysOf(specs);
  const exposureDecays = exposureDecaysOf(specs);
  const exposed = usesExposures(specs);
  const proficiencies = proficiency === "none" ? undefined : parameters.learners;
  const fitted = Object.values(proficiencies ?? {});
  const absent = proficiency === "fixed" && fitted.length > 0 ? mean(fitted) : 0;
  const skillOf = (skill: string): LogisticSkill => parametersOf(parameters.skills, "skill", skill);

  /** Returns P(correct) of an attempt, the state and attempt checked. */
  const probability = (state: LogisticState, attempt: Attempt): number => {
    let terms = 0;
    for (const skill of attempt.skills) {
      const { intercept, coefficients } = skillOf(skill);
      const history = entry(state.skills, skill);
      terms += intercept;
      for (const spec of specs) {
        const value = featureValue(spec, history, entry(state.exposures, skill), attempt.time);
        terms += value * (coefficients[spec.label] ?? 0);
      }
    }
    const theta = proficiency === "none" ? 0 : (entry(proficiencies, state.learner) ?? absent);
    return sigmoid(theta + terms);
  };

  /** Returns the state after an event, the state and event checked, the event not earlier than the state's attempts. */
  const advance = (state: LogisticState, event: Event): LogisticState => {
    if (isObserved(event)) {
      const updates = event.skills.map((skill) => [
        skill,
        nextHistory(entry(state.skills, skill), event.outcome, event.time, decays),
      ]);
      return carry(state.learner, { ...state.skills, ...Object.fromEntries(updates) }, state.exposures);
    }
    // Without a feature that reads exposures, an event without an outcome leaves the learner as they were.
    const updates = exposed
      ? event.skills.map((skill) => [skill, nextExposure(entry(state.exposures, skill), event.time, exposureDecays)])
      : [];
    const exposures = updates.length === 0 ? state.exposures : { ...state.exposures, ...Object.fromEntries(updates) };
    return carry(state.learner, state.skills, exposures);
  };
  const steps: Steps<LogisticState> = {
    initialState: (learner) => ({ learner, skills: {} }),
    predict: probability,
    update: advance,
  };

  const model: LogisticModel = {
    parameters,
    fit(log) {
      checkLog(log);
      checkObservedEvents(log, "a logistic model");
      log.forEach((event, index) => {
        checkObservedHasSkills(event, `Event ${index + 1}`, "logistic model");
      });
      const fitted = estimate(log, { specs, decays, exposureDecays, proficiency, ridge, tolerance, maxIterations });
      const built = create({ ...options, parameters: fitted.parameters });
      const replayed = replay(log, built.steps);
      const { states, predictions } = replayed;
      const logLikelihood = fitted.marginal ?? replayed.logLikelihood;
      const k = fitted.parameterCount;
      const n = log.filter(isObserved).length;
      return {
        model: built.model,
        states,
        predictions,
        logLikelihood,
        aic: ridge > 0 ? null : 2 * k - 2 * logLikelihood,
        bic: ridge > 0 ? null : k * Math.log(n) - 2 * logLikelihood,
        converged: fitted.converged,
        iterations: fitted.iterations,
      };
    },
    initialState(learner) {
      checkLearnerId(learner);
      return steps.initialState(learner);
    },
    update(state, event) {
      checkState(state, decays, exposureDecays);
      checkEvent(event, "The event");
      checkObservedHasSkills(event, "The event", "logistic model");
      checkLearner(state.learner, event);
      checkOrder(state, event, "The event");
      return advance(state, event);
    },
    predict(state, attempt) {
      checkState(state, decays, exposureDecays);
      checkAttempt(attempt, "The attempt");
      checkHasSkills(attempt, "The attempt", "logistic model");
      checkLearner(state.learner, attempt);
      checkOrder(state, attempt, "The attempt");
      return probability(state, attempt);
    },
  };
  return { model, steps };
}

/** Returns a state with copies of its records, which are present only when they have been given. */
function carry(learner: string, skills: LogisticState["skills"], exposures: LogisticState["exposures"]): LogisticState {
  return { learner, skills: { ...skills }, ...(exposures !== undefined && { exposures: { ...exposures } }) };
}

/**
 * Creates the additive factors model: logit P = θ + Σ_k (β_k + γ_k T_k), with T_k the learner's earlier attempts at
 * skill k.
 *
 * @throws `DataError` in the cases listed for {@link logistic}.
 * @see {@link https://doi.org/10.1007/11774303_17 | Cen, Koedinger and Junker (2006)}
 */
export function afm(options: AfmOptions = {}): LogisticModel {
  checkOptions(options);
  const { proficiency = AFM_PROFICIENCY } = options;
  return logistic({ ...options, proficiency, features: ["opportunities"] });
}

/**
 * Creates performance factors analysis: logit P = θ + Σ_k (β_k + γ_k s_k + ρ_k f_k), with s_k and f_k the learner's
 * earlier successes and failures at skill k. As published, it has no learner parameter unless the `proficiency` option
 * adds one.
 *
 * @throws `DataError` in the cases listed for {@link logistic}.
 * @see {@link https://doi.org/10.3233/978-1-60750-028-5-531 | Pavlik, Cen and Koedinger (2009)}
 */
export function pfa(options: Omit<LogisticOptions, "features"> = {}): LogisticModel {
  checkOptions(options);
  return logistic({ ...options, features: ["successes", "failures"] });
}
