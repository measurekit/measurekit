import {
  checkBoolean,
  checkChoice,
  checkNonNegative,
  checkOptions,
  checkStopping,
  DataError,
  mean,
  logistic as sigmoid,
} from "@measurekit/core";
import {
  type Attempt,
  checkAttempt,
  checkEvent,
  checkHasSkills,
  checkObservedHasSkills,
  isObserved,
} from "../data/event.js";
import { checkLog, checkObservedEvents, type EventLog } from "../data/event-log.js";
import { checkLearner, checkLearnerId } from "../data/learner.js";
import { entry, parametersOf } from "../data/record.js";
import { replay, type Steps } from "../data/replay.js";
import type { LearnerModel, ModelFit } from "../model.js";
import { estimate } from "./estimation.js";
import type { LogisticFeature } from "./features.js";
import {
  checkParameters,
  type LogisticParameters,
  type LogisticSkill,
  PROFICIENCIES,
  type Proficiency,
} from "./parameters.js";
import { advanceState, checkDuration, checkOrder, checkState, type LogisticState } from "./state.js";
import {
  checkCovariateNames,
  checkCovariates,
  checkIntercepts,
  checkTerms,
  compileTerms,
  type InterceptKind,
  type LogisticTerm,
  type RowVisitor,
  visitRow,
} from "./terms.js";

/** Options of {@link logistic}. */
export interface LogisticOptions {
  /**
   * The terms of the model: features of the learner's history, each over the skills of an attempt, over its item or over
   * all the learner's attempts, with a coefficient for each skill or item or one for all. A feature on its own is a term
   * over skills with a coefficient for each, besides the skill's intercept.
   *
   * @defaultValue `[]`
   */
  readonly features?: readonly (LogisticFeature | LogisticTerm)[];
  /**
   * The intercepts of the model: one for each skill and, besides, one for each item as Best-LR has them. Both need a
   * ridge above 0, as the items' intercepts are otherwise not identified apart from the skills'.
   *
   * @defaultValue `["skill"]`
   */
  readonly intercepts?: readonly InterceptKind[];
  /**
   * The names of numbers that each attempt carries in its `covariates`, each with one coefficient, such as LKT's
   * `numer`. A fit and a prediction need the covariate on every attempt.
   *
   * @defaultValue `[]`
   */
  readonly covariates?: readonly string[];
  /**
   * Learners' proficiency θ: `"none"`; `"fixed"`, a free θ per learner; or `"random"`, θ from N(0, σ²) with σ
   * estimated, fitted by the Laplace approximation as `lme4::glmer(nAGQ = 1)`.
   *
   * @defaultValue `"none"`
   */
  readonly proficiency?: Proficiency;
  /**
   * λ of the penalty λ / 2 Σ b² over the coefficients and fixed proficiencies, as LKT's LiblineaR fits use. A fit with a
   * ridge has no standard errors and, with fixed or no proficiency, solves with Newton-CG: its cost does not grow with
   * the square of the coefficients.
   *
   * @defaultValue `0`
   */
  readonly ridge?: number;
  /**
   * Whether `fit` returns the standard errors of the coefficients and, with fixed proficiency, of the proficiencies.
   * They need no ridge.
   *
   * @defaultValue `false`
   */
  readonly standardErrors?: boolean;
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
   * the fit cover the skills, items and learners of the log.
   *
   * @throws `DataError` when the log is malformed, an event has no skills, a covariate is missing from an event with an
   * outcome, or the fit has more coefficients than a fit without a ridge or with random proficiency solves.
   */
  fit(log: EventLog): LogisticFit;
}

/** The outcome of fitting a logistic model to a log. */
export interface LogisticFit extends ModelFit<LogisticState, LogisticParameters> {
  /** The model with the estimated parameters. */
  readonly model: LogisticModel;
  /**
   * The standard errors in the shape of the parameters, from the inverse of the Hessian at the estimates, with the
   * binomial variance p (1 - p) as `glm` has it. `null` unless the option `standardErrors` is set, with a ridge, and
   * where the information is singular, as with features that are linear combinations of each other or a coefficient
   * that no event uses. Random proficiencies and `learnerSd` have none. A coefficient the model does not have, such as the
   * skill intercepts of a model with item intercepts alone, has 0.
   */
  readonly standardErrors: LogisticParameters | null;
}

/**
 * Creates a logistic learner model.
 *
 * @remarks
 * The model is logit P(correct) = θ + β_item + Σ over the attempt's skills k of (β_k + Σ_f w_kf x_kf) + Σ_g w_g x_g +
 * Σ_c w_c z_c, with θ the learner's proficiency, β_k and β_item the skill's and the item's intercepts, and x the features
 * of the learner's history before the attempt: x_kf those with skill k, with a coefficient w_kf for each skill, and x_g
 * those with a coefficient shared across skills, items or the learner's attempts. z_c are the covariates. A learner the
 * fit did not see has the mean proficiency of the fitted learners with fixed proficiency, and 0, the population mean,
 * with a random intercept. An item the fit did not see has the intercept 0 and the coefficients 0; a skill it did not see
 * throws.
 *
 * @throws `DataError` when an option is out of range, a feature or term is unknown, skill and item intercepts are asked
 * for without a ridge, or the parameters are malformed or lack a coefficient the model has.
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
const DEFAULT_INTERCEPTS: readonly InterceptKind[] = ["skill"];
const DEFAULT_RIDGE = 0;
const DEFAULT_TOLERANCE = 1e-8;
const DEFAULT_MAX_ITERATIONS = 100;

/** Builds a logistic model with its steps, which `fit` replays the log with. */
function create(options: LogisticOptions): { model: LogisticModel; steps: Steps<LogisticState> } {
  checkOptions(options);
  const {
    features = [],
    intercepts = DEFAULT_INTERCEPTS,
    covariates = [],
    proficiency = DEFAULT_PROFICIENCY,
    ridge = DEFAULT_RIDGE,
    standardErrors = false,
    tolerance = DEFAULT_TOLERANCE,
    maxIterations = DEFAULT_MAX_ITERATIONS,
  } = options;
  const specs = checkTerms(features);
  const kinds = checkIntercepts(intercepts);
  const names = checkCovariateNames(covariates);
  checkChoice(proficiency, "The option proficiency", PROFICIENCIES);
  checkNonNegative(ridge, "The option ridge");
  checkBoolean(standardErrors, "The option standardErrors");
  checkStopping(tolerance, maxIterations);
  if (kinds.skill && kinds.item && ridge === 0) {
    throw new DataError(
      'The option intercepts has both "skill" and "item", which needs a ridge above 0: items nest in skills, so without one the pair adds nothing to predictions.',
    );
  }
  const { parameters = { skills: {} } } = options;
  const terms = compileTerms(specs, kinds, names);
  checkParameters(parameters, terms.labels);
  const { needs } = terms;
  const proficiencies = proficiency === "none" ? undefined : parameters.learners;
  const fitted = Object.values(proficiencies ?? {});
  const absent = proficiency === "fixed" && fitted.length > 0 ? mean(fitted) : 0;
  const skillOf = (skill: string): LogisticSkill => parametersOf(parameters.skills, "skill", skill);

  // The sum of the terms of the attempt being predicted, which the visitor adds to as visitRow reaches each block.
  let total = 0;
  /** Adds a block's intercept and its values times the coefficients with the labels of its terms. */
  const addBlock = (
    found: LogisticSkill,
    values: Float64Array,
    intercept: boolean,
    labels: readonly string[],
  ): void => {
    const first = intercept ? 1 : 0;
    if (intercept) total += found.intercept;
    for (let j = 0; j < labels.length; j++) {
      total += (values[first + j] ?? 0) * (found.coefficients[labels[j] ?? ""] ?? 0);
    }
  };
  const visitor: RowVisitor = {
    skill(skill, values) {
      if (values.length > 0) addBlock(skillOf(skill), values, kinds.skill, terms.labels.skills);
    },
    item(item, values) {
      const found = entry(parameters.items, item);
      if (found !== undefined) addBlock(found, values, kinds.item, terms.labels.items);
    },
    shared(values) {
      const { shared } = parameters;
      if (shared === undefined) {
        throw new DataError(
          `The model has no parameters for its shared coefficients ${terms.labels.shared.join(", ")}.`,
        );
      }
      for (let j = 0; j < terms.labels.shared.length; j++) {
        total += (values[j] ?? 0) * (entry(shared, terms.labels.shared[j] ?? "") ?? 0);
      }
    },
  };

  /** Returns P(correct) of an attempt, the state and attempt checked. */
  const probability = (state: LogisticState, attempt: Attempt): number => {
    total = 0;
    visitRow(terms, state, attempt, visitor);
    const theta = proficiency === "none" ? 0 : (entry(proficiencies, state.learner) ?? absent);
    return sigmoid(theta + total);
  };

  const steps: Steps<LogisticState> = {
    initialState: (learner) => ({ learner, skills: {} }),
    predict: probability,
    update: (state, event) => advanceState(state, event, needs),
  };

  const model: LogisticModel = {
    parameters,
    fit(log) {
      checkLog(log);
      checkObservedEvents(log, "a logistic model");
      log.forEach((event, index) => {
        const label = `Event ${index + 1}`;
        checkObservedHasSkills(event, label, "logistic model");
        checkDuration(event, label, needs);
        if (isObserved(event)) checkCovariates(event, label, terms.covariates);
      });
      const fitted = estimate(log, { terms, proficiency, ridge, standardErrors, tolerance, maxIterations });
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
        standardErrors: fitted.standardErrors,
        converged: fitted.converged,
        iterations: fitted.iterations,
      };
    },
    initialState(learner) {
      checkLearnerId(learner);
      return steps.initialState(learner);
    },
    update(state, event) {
      checkState(state, needs);
      checkEvent(event, "The event");
      checkObservedHasSkills(event, "The event", "logistic model");
      checkDuration(event, "The event", needs);
      checkLearner(state.learner, event);
      checkOrder(state, event, "The event", needs);
      return advanceState(state, event, needs);
    },
    predict(state, attempt) {
      checkState(state, needs);
      checkAttempt(attempt, "The attempt");
      checkHasSkills(attempt, "The attempt", "logistic model");
      checkCovariates(attempt, "The attempt", terms.covariates);
      checkLearner(state.learner, attempt);
      checkOrder(state, attempt, "The attempt", needs);
      return probability(state, attempt);
    },
  };
  return { model, steps };
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
