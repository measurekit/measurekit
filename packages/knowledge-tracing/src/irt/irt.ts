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
import {
  checkAttempt,
  checkEvent,
  checkLearner,
  checkLearnerId,
  checkLog,
  checkObservedEvents,
  type Event,
  type EventLog,
  isObserved,
  type ObservedEvent,
  onePass,
  parametersOf,
} from "../data/internal.js";
import { indexOfMax } from "../math/extremes.js";
import { reflectedWalk } from "../math/reflected-walk.js";
import type { LearnerModel, ModelFit } from "../model.js";
import { type Ability, average, type Derivatives, differences, posteriorMean, posteriorMode } from "./ability.js";
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

/** The drift of θ: a random walk whose variance grows by `variance` every `per` units of time. */
export interface IrtDrift {
  /** The variance θ gains in `per` units of time: above 0. */
  readonly variance: number;
  /** The span of time in which θ gains `variance`, in the units of the events' times: above 0. */
  readonly per: number;
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
   * of `@measurekit/psychometrics`: at least 3, or at least 6 with `drift`, which requires it.
   *
   * @remarks
   * With drift, 241 points are recommended, and 481 for items of discrimination above 2 with dense evidence: see the
   * tolerances per number of points in the guide.
   *
   * @defaultValue `61` without `drift`
   */
  readonly points?: number;
  /**
   * The drift of θ between events, so that the posterior follows a learner who changes: θ is a random walk whose
   * variance grows by `variance` every `per` units of time.
   *
   * @defaultValue none: θ is constant
   */
  readonly drift?: IrtDrift;
}

/** Options of {@link IrtModel.posterior}. */
export interface PosteriorOptions {
  /**
   * The time of the posterior, in the units of the events' times, to which a model with drift advances the state. A
   * time before the state's is taken as the state's time, as is any time for a state without one; a model without
   * drift has the same posterior at every time.
   *
   * @defaultValue the state's time
   */
  readonly at?: number;
}

/** Options of {@link IrtModel.ability}. */
export interface AbilityOptions extends PosteriorOptions {
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
   * estimated. The log-likelihood is the marginal log-likelihood of the outcomes on the grid, and with drift the
   * forward likelihood, each outcome predicted from the posterior advanced to its time. An event without an outcome has
   * no prediction, and changes no state but its time.
   *
   * @throws `DataError` when the log is malformed or has no event with an outcome, an item has no parameters, or an
   * outcome is not a category.
   */
  fit(log: EventLog): IrtFit;
  /**
   * Returns the learner's ability: EAP, the posterior mean on the grid, or MAP, the posterior mode, found between grid
   * points from the log-posterior and its first two derivatives at the nearest points. With drift, the derivatives are
   * fourth-order differences of the log-posterior.
   *
   * @throws `DataError` when the state is malformed, the estimator unknown or `at` not a finite number.
   */
  ability(state: IrtState, options?: AbilityOptions): Ability;
  /**
   * Returns the posterior of the learner's θ on the model's grid: the state's density, advanced to `at` by the drift,
   * normalized to sum to 1.
   *
   * @throws `DataError` when the state is malformed or `at` not a finite number.
   */
  posterior(state: IrtState, options?: PosteriorOptions): ThetaDistribution;
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
/** The fewest grid points with drift: the differences that give the derivatives at an end take 6. */
const MIN_DRIFT_POINTS = 6;

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
 * With `drift`, θ is a random walk between events, and the grid's continuous-time chain carries the posterior across
 * each gap: a reflecting walk with t = variance Δt / (per h²) jumps on average for grid spacing h, which adds exactly
 * variance Δt / per to the posterior's variance away from the grid's ends. A state then keeps the time of its latest
 * event: an event, an attempt or an `at` before it takes no drift, and leaves the time as it was.
 *
 * @param options - The calibrated item parameters, the prior, the grid and the drift of ability; each option has its
 *   default.
 * @returns The model: `fit` replays a log, `update` and `predict` serve online, and `ability` and `posterior` read a
 *   state.
 * @throws `DataError` when an item's parameters, the prior, the number of points or the drift are invalid, or `drift`
 * is given without `points`.
 * @see {@link https://doi.org/10.1214/12-AOAS608 | Wang, Berger and Burdick (2013)}
 */
export function irt(options: IrtOptions): IrtModel {
  checkOptions(options);
  const { parameters, prior = DEFAULT_PRIOR, drift } = options;
  const { points = drift === undefined ? DEFAULT_POINTS : undefined } = options;
  checkParameters(parameters);
  checkRecord(prior, "The prior must be an object { mean, sd }");
  checkFinite(prior.mean, "The mean of the prior");
  checkPositive(prior.sd, "The sd of the prior");
  if (drift !== undefined) {
    checkRecord(drift, "The option drift must be an object { variance, per }");
    checkPositive(drift.variance, "The variance of the drift");
    checkPositive(drift.per, "The per of the drift");
  }
  if (points === undefined) {
    throw new DataError(
      "The option drift needs the option points: 241 is recommended, or the number of points of stored states.",
    );
  }
  checkIntegerFrom(points, "The option points", drift === undefined ? MIN_POINTS : MIN_DRIFT_POINTS);
  const z = Array.from({ length: points }, (_, q) => GRID_HALF_WIDTH * ((2 * q) / (points - 1) - 1));
  const grid = z.map((value) => prior.mean + prior.sd * value);
  checkGrid(grid, prior);
  const spacing = (grid[1] ?? 0) - (grid[0] ?? 0);
  const itemOf = (id: string): IrtItem => parametersOf(parameters.items, "item", id);
  const drifting = drift !== undefined;
  const walk = reflectedWalk(points);
  /** The mean number of jumps of the walk per unit of time. */
  const jumpRate = drifting ? drift.variance / (drift.per * spacing * spacing) : 0;

  const initial = (learner: string): IrtState => {
    const logDensity = shift(z.map((value) => -(value * value) / 2));
    if (drifting) return { learner, logDensity };
    return {
      learner,
      logDensity,
      slope: grid.map((theta) => (prior.mean - theta) / (prior.sd * prior.sd)),
      curvature: z.map(() => -1 / (prior.sd * prior.sd)),
    };
  };
  /**
   * Returns the log-density of a state at a time: advanced by the drift from the state's time, or as it is without
   * drift, without a time in the state, or at a time not after it.
   */
  const logDensityAt = (state: IrtState, at: number | undefined): readonly number[] => {
    if (!drifting || state.time === undefined || at === undefined || !(at > state.time)) return state.logDensity;
    return shift(walk(state.logDensity, jumpRate * (at - state.time)));
  };
  /** Returns the time of a state of a model with drift after an event at a time: the later of the two. */
  const timeAfter = (state: IrtState, time: number): number =>
    state.time === undefined ? time : Math.max(state.time, time);
  /** Returns the state of a model with drift advanced to an event's time, with nothing observed. */
  const idle = (state: IrtState, event: Event): IrtState => ({
    learner: state.learner,
    logDensity: [...logDensityAt(state, event.time)],
    time: timeAfter(state, event.time),
  });
  const weightsOf = new Float64Array(points);
  const terms = new Float64Array(3);
  /** Writes the unnormalized posterior, exp of a log-density, into the scratch weights. */
  const weightsOfDensity = (logDensity: readonly number[]): Float64Array => {
    for (let q = 0; q < points; q++) weightsOf[q] = Math.exp(logDensity[q] ?? 0);
    return weightsOf;
  };
  /**
   * Returns the state of a model without drift after an outcome in category k, writing P(k | θ) at each grid point into
   * `likelihood` if given.
   */
  const advance = (state: IrtState, item: IrtItem, k: number, likelihood?: Float64Array): IrtState => {
    const { slope: slopeBefore = [], curvature: curvatureBefore = [] } = state;
    const logDensity: number[] = [];
    const slope: number[] = [];
    const curvature: number[] = [];
    for (let q = 0; q < points; q++) {
      categoryTerms(item, k, grid[q] ?? 0, terms);
      if (likelihood !== undefined) likelihood[q] = Math.exp(terms[0] ?? 0);
      logDensity.push((state.logDensity[q] ?? 0) + (terms[0] ?? 0));
      slope.push((slopeBefore[q] ?? 0) + (terms[1] ?? 0));
      curvature.push((curvatureBefore[q] ?? 0) + (terms[2] ?? 0));
    }
    return { learner: state.learner, logDensity: shift(logDensity), slope, curvature };
  };
  /**
   * Returns the state of a model with drift after an outcome in category k at a time, from its log-density at that time,
   * writing P(k | θ) at each grid point into `likelihood` if given.
   */
  const advanceDrifting = (
    state: IrtState,
    logDensityBefore: readonly number[],
    time: number,
    item: IrtItem,
    k: number,
    likelihood?: Float64Array,
  ): IrtState => {
    const logDensity: number[] = [];
    for (let q = 0; q < points; q++) {
      categoryTerms(item, k, grid[q] ?? 0, terms);
      if (likelihood !== undefined) likelihood[q] = Math.exp(terms[0] ?? 0);
      logDensity.push((logDensityBefore[q] ?? 0) + (terms[0] ?? 0));
    }
    return { learner: state.learner, logDensity: shift(logDensity), time: timeAfter(state, time) };
  };
  /** Returns the state after an outcome in category k at a time, from the state's log-density at that time. */
  const observe = (
    state: IrtState,
    logDensity: readonly number[],
    time: number,
    item: IrtItem,
    k: number,
    likelihood?: Float64Array,
  ): IrtState =>
    drifting ? advanceDrifting(state, logDensity, time, item, k, likelihood) : advance(state, item, k, likelihood);
  /** Returns the derivatives of a log-density: the state's own without drift, and differences with it. */
  const derivativesOf = (state: IrtState, logDensity: readonly number[]): Derivatives =>
    drifting ? differences(logDensity, spacing) : (q) => [state.slope?.[q] ?? 0, state.curvature?.[q] ?? 0];
  /** Checks a state and the options of a posterior, and returns the state's log-density at the time asked. */
  const posteriorOf = (state: IrtState, posteriorOptions: PosteriorOptions): readonly number[] => {
    checkState(state, points, drifting);
    const { at = state.time } = posteriorOptions;
    if (at !== undefined) checkFinite(at, "The option at");
    return logDensityAt(state, at);
  };

  const model: IrtModel = {
    parameters,
    fit(log) {
      checkLog(log);
      checkObservedEvents(log, "an IRT model");
      const likelihood = new Float64Array(points);
      const pass = onePass(
        log,
        initial,
        (state, event) => {
          const item = itemOf(event.item);
          const logDensity = logDensityAt(state, event.time);
          const weights = weightsOfDensity(logDensity);
          const prediction = expectation(weights, item, grid, terms);
          const next = observe(state, logDensity, event.time, item, categoryOf(item, event), likelihood);
          return { state: next, prediction, logLikelihood: Math.log(average(weights, (q) => likelihood[q] ?? 0)) };
        },
        drifting ? idle : undefined,
      );
      return { model, ...pass };
    },
    initialState(learner) {
      checkLearnerId(learner);
      return initial(learner);
    },
    update(state, event) {
      checkState(state, points, drifting);
      checkEvent(event, "The event");
      checkLearner(state.learner, event);
      // A lesson is not an item: it has no parameters, and it does not move the posterior of θ but by the drift.
      if (!isObserved(event)) {
        if (drifting) return idle(state, event);
        return {
          learner: state.learner,
          logDensity: [...state.logDensity],
          slope: [...(state.slope ?? [])],
          curvature: [...(state.curvature ?? [])],
        };
      }
      const item = itemOf(event.item);
      const k = categoryOf(item, event);
      return observe(state, logDensityAt(state, event.time), event.time, item, k);
    },
    predict(state, attempt) {
      checkState(state, points, drifting);
      checkAttempt(attempt, "The attempt");
      checkLearner(state.learner, attempt);
      const item = itemOf(attempt.item);
      return expectation(weightsOfDensity(logDensityAt(state, attempt.time)), item, grid, terms);
    },
    ability(state, abilityOptions = {}) {
      checkOptions(abilityOptions);
      const logDensity = posteriorOf(state, abilityOptions);
      const { estimator = DEFAULT_ESTIMATOR } = abilityOptions;
      checkChoice(estimator, "The estimator", ESTIMATORS);
      return estimator === "eap"
        ? posteriorMean(logDensity, grid)
        : posteriorMode(logDensity, grid, derivativesOf(state, logDensity));
    },
    posterior(state, posteriorOptions = {}) {
      checkOptions(posteriorOptions);
      const weights = posteriorOf(state, posteriorOptions).map(Math.exp);
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
function categoryOf(item: IrtItem, event: ObservedEvent): number {
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
