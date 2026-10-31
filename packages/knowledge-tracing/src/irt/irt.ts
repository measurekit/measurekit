import {
  checkChoice,
  checkFinite,
  checkIntegerFrom,
  checkOptions,
  checkPositive,
  checkRecord,
  DataError,
  type ThetaDistribution,
} from "@measurekit/core";
import { checkAttempt, checkEvent, type Event } from "../data/event.js";
import { checkLog, type EventLog } from "../data/event-log.js";
import { checkLearner, checkLearnerId } from "../data/learner.js";
import { parametersOf } from "../data/record.js";
import { indexOfMax } from "../math/extremes.js";
import type { LearnerModel, ModelFit } from "../model.js";
import { type Ability, average, posteriorMean, posteriorMode } from "./ability.js";
import { categoryCount, categoryTerms, expectedOutcome, type IrtItem } from "./items.js";
import { checkParameters, type IrtParameters } from "./parameters.js";
import { checkState, type IrtState } from "./state.js";

/** A normal prior of θ. */
export interface IrtPrior {
  /** The mean. */
  readonly mean: number;
  /** The standard deviation. */
  readonly sd: number;
}

/** Options of {@link irt}. */
export interface IrtOptions {
  /** The items' parameters. */
  readonly parameters: IrtParameters;
  /**
   * The prior of θ, the population's distribution in the calibration.
   *
   * @defaultValue `{ mean: 0, sd: 1 }`
   */
  readonly prior?: IrtPrior;
  /**
   * The number of grid points, equally spaced from 6 prior standard deviations below the mean to 6 above, as in the EAP
   * of `@measurekit/psychometrics`; at least 3.
   *
   * @defaultValue `61`
   */
  readonly points?: number;
}

/** Options of {@link IrtModel.ability}. */
export interface AbilityOptions {
  /**
   * `"eap"`, the posterior mean, or `"map"`, the posterior mode.
   *
   * @defaultValue `"eap"`
   */
  readonly estimator?: "eap" | "map";
}

/** An IRT model: a learner model that updates the posterior of each learner's θ from given item parameters. */
export interface IrtModel extends LearnerModel<IrtState, IrtParameters> {
  /**
   * Runs the log through the model and returns it unchanged, with the learners' states: item parameters are given, not
   * estimated. The log-likelihood is the marginal log-likelihood of the outcomes on the grid.
   *
   * @throws `DataError` when the log is malformed, an item has no parameters, or an outcome is not a category.
   */
  fit(log: EventLog): IrtFit;
  /**
   * Returns the learner's ability: EAP, the posterior mean on the grid, or MAP, the posterior mode, found between grid
   * points from the log-posterior and its first two derivatives at the nearest points.
   *
   * @throws `DataError` when the state is malformed or the estimator unknown.
   */
  ability(state: IrtState, options?: AbilityOptions): Ability;
  /**
   * Returns the posterior of the learner's θ on the model's grid: the state's density, normalized to sum to 1.
   *
   * @throws `DataError` when the state is malformed.
   */
  posterior(state: IrtState): ThetaDistribution;
}

/** The outcome of running a log through an IRT model. */
export interface IrtFit extends ModelFit<IrtState, IrtParameters> {
  /** The model, unchanged. */
  readonly model: IrtModel;
}

/** The half-width of the grid in prior standard deviations. */
const GRID_HALF_WIDTH = 6;
const DEFAULT_PRIOR: IrtPrior = { mean: 0, sd: 1 };
const ESTIMATORS = ["eap", "map"] as const;
const DEFAULT_ESTIMATOR = "eap";
const DEFAULT_POINTS = 61;
const MIN_POINTS = 3;

/** How far from a category k / (K - 1) an outcome may be, for rounding in an outcome computed from a score. */
const CATEGORY_TOLERANCE = 1e-9;

/**
 * Creates an IRT learner model from calibrated item parameters.
 *
 * @remarks
 * One θ per learner, on the scale of the calibration, with a posterior on a grid that each outcome updates. A
 * dichotomous item takes the outcomes 0 and 1, an item with K categories the outcomes k / (K - 1), and a prediction is
 * the expected outcome. Skills are not used: every item measures the one θ, so a test calibrated in sections takes one
 * model per section.
 *
 * @throws `DataError` when an item's parameters, the prior or the number of points are invalid.
 */
export function irt(options: IrtOptions): IrtModel {
  checkOptions(options);
  const { parameters, prior = DEFAULT_PRIOR, points = DEFAULT_POINTS } = options;
  checkParameters(parameters);
  checkRecord(prior, "The prior must be an object { mean, sd }");
  checkFinite(prior.mean, "The mean of the prior");
  checkPositive(prior.sd, "The sd of the prior");
  checkIntegerFrom(points, "The option points", MIN_POINTS);
  const z = Array.from({ length: points }, (_, q) => GRID_HALF_WIDTH * ((2 * q) / (points - 1) - 1));
  const grid = z.map((value) => prior.mean + prior.sd * value);
  checkGrid(grid, prior);
  const itemOf = (id: string): IrtItem => parametersOf(parameters.items, "item", id);

  const initial = (learner: string): IrtState => ({
    learner,
    logDensity: shift(z.map((value) => -(value * value) / 2)),
    slope: grid.map((theta) => (prior.mean - theta) / (prior.sd * prior.sd)),
    curvature: z.map(() => -1 / (prior.sd * prior.sd)),
  });
  const weightsOf = new Float64Array(points);
  const terms = new Float64Array(3);
  /** Writes the unnormalized posterior of a state, exp of its log-density, into the scratch weights. */
  const weightsOfState = (state: IrtState): Float64Array => {
    for (let q = 0; q < points; q++) weightsOf[q] = Math.exp(state.logDensity[q] ?? 0);
    return weightsOf;
  };
  /**
   * Returns the state after an outcome in category k, writing P(k | θ) at each grid point into `likelihood` if given.
   */
  const advance = (state: IrtState, item: IrtItem, k: number, likelihood?: Float64Array): IrtState => {
    const logDensity: number[] = [];
    const slope: number[] = [];
    const curvature: number[] = [];
    for (let q = 0; q < points; q++) {
      categoryTerms(item, k, grid[q] ?? 0, terms);
      if (likelihood !== undefined) likelihood[q] = Math.exp(terms[0] ?? 0);
      logDensity.push((state.logDensity[q] ?? 0) + (terms[0] ?? 0));
      slope.push((state.slope[q] ?? 0) + (terms[1] ?? 0));
      curvature.push((state.curvature[q] ?? 0) + (terms[2] ?? 0));
    }
    return { learner: state.learner, logDensity: shift(logDensity), slope, curvature };
  };

  const model: IrtModel = {
    parameters,
    fit(log) {
      checkLog(log);
      const states = new Map<string, IrtState>();
      const likelihood = new Float64Array(points);
      let logLikelihood = 0;
      const predictions = log.map((event) => {
        const state = states.get(event.learner) ?? initial(event.learner);
        const item = itemOf(event.item);
        const weights = weightsOfState(state);
        const prediction = expectation(weights, item, grid, terms);
        states.set(event.learner, advance(state, item, categoryOf(item, event), likelihood));
        logLikelihood += Math.log(average(weights, (q) => likelihood[q] ?? 0));
        return prediction;
      });
      return {
        model,
        states: Object.fromEntries(states),
        predictions,
        logLikelihood,
        aic: null,
        bic: null,
        converged: true,
        iterations: 1,
      };
    },
    initialState(learner) {
      checkLearnerId(learner);
      return initial(learner);
    },
    update(state, event) {
      checkState(state, points);
      checkEvent(event, "The event");
      checkLearner(state.learner, event);
      const item = itemOf(event.item);
      return advance(state, item, categoryOf(item, event));
    },
    predict(state, attempt) {
      checkState(state, points);
      checkAttempt(attempt, "The attempt");
      checkLearner(state.learner, attempt);
      const item = itemOf(attempt.item);
      return expectation(weightsOfState(state), item, grid, terms);
    },
    ability(state, abilityOptions = {}) {
      checkOptions(abilityOptions);
      const { estimator = DEFAULT_ESTIMATOR } = abilityOptions;
      checkState(state, points);
      checkChoice(estimator, "The estimator", ESTIMATORS);
      return estimator === "eap" ? posteriorMean(state, grid) : posteriorMode(state, grid);
    },
    posterior(state) {
      checkState(state, points);
      const weights = state.logDensity.map(Math.exp);
      const total = weights.reduce((sum, weight) => sum + weight, 0);
      return { nodes: [...grid], weights: weights.map((weight) => weight / total) };
    },
  };
  return model;
}

/**
 * Checks that a prior's grid is in the range of numbers: distinct, finite points, with finite squares of their spacing
 * and of 1 / sd, which the posterior's derivatives and its mode use.
 */
function checkGrid(grid: readonly number[], prior: IrtPrior): void {
  const spacing = (grid[1] ?? 0) - (grid[0] ?? 0);
  const distinct = grid.every((theta, q) => Number.isFinite(theta) && (q === 0 || theta > (grid[q - 1] ?? 0)));
  if (!(distinct && Number.isFinite(spacing ** 2) && Number.isFinite(1 / prior.sd ** 2))) {
    throw new DataError(
      `The prior needs a mean and an sd whose grid has distinct, finite points, with finite squares of their spacing and of 1 / sd, found mean ${prior.mean} and sd ${prior.sd}.`,
    );
  }
}

/** Returns the category of an event's outcome: k for the outcome k / (K - 1). */
function categoryOf(item: IrtItem, event: Event): number {
  const last = categoryCount(item) - 1;
  const k = Math.round(event.outcome * last);
  if (Math.abs(event.outcome * last - k) > CATEGORY_TOLERANCE) {
    throw new DataError(
      `The outcome ${event.outcome} of item "${event.item}" is not one of its ${last + 1} categories, k / ${last}.`,
    );
  }
  return k;
}

/** Returns the mean over the grid, weighted by the posterior weights, of the item's expected outcome. */
function expectation(weights: ArrayLike<number>, item: IrtItem, grid: readonly number[], terms: Float64Array): number {
  return average(weights, (q) => expectedOutcome(item, grid[q] ?? 0, terms));
}

/** Returns the values less their largest, in place. */
function shift(values: number[]): number[] {
  const top = values[indexOfMax(values)] ?? 0;
  for (let q = 0; q < values.length; q++) values[q] = (values[q] ?? 0) - top;
  return values;
}
