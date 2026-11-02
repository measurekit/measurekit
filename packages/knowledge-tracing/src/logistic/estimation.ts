import { DataError, logistic } from "@measurekit/core";
import { type EventLog, isObserved, type ObservedEvent } from "../data/internal.js";
import {
  type FlatDirection,
  type GroupIntercepts,
  type LogisticRegression,
  linear,
  logisticRegression,
  MAX_DENSE_COEFFICIENTS,
  outcomeLogLikelihood,
  type SparseMatrix,
} from "../math/internal.js";
import type { LogisticParameters, LogisticSkill, Proficiency } from "./parameters.js";
import { advanceInPlace, finishState, type LogisticState, startState, type WorkingState } from "./state.js";
import { blockWidth, type InterceptKind, type TermSpec, type Terms, visitRow } from "./terms.js";

/** What a fit depends on: the checked options of the model. */
export interface FitSettings {
  readonly terms: Terms;
  readonly proficiency: Proficiency;
  readonly ridge: number;
  /** Whether to return the standard errors, which a fit without a ridge has. */
  readonly standardErrors: boolean;
  readonly tolerance: number;
  readonly maxIterations: number;
}

/** The estimates of a fit, with what the fit reports of them. */
export interface Estimates {
  readonly parameters: LogisticParameters;
  /** The standard errors in the shape of the parameters; `null` unless asked, with a ridge and where singular. */
  readonly standardErrors: LogisticParameters | null;
  readonly parameterCount: number;
  /** Each learner's state after the log, by learner id. */
  readonly states: Record<string, LogisticState>;
  /** Each event's prediction at the estimates, made before the event; `null` for an event without an outcome. */
  readonly predictions: (number | null)[];
  /** The sum over events with an outcome of y ln p + (1 - y) ln(1 - p), with p the event's prediction. */
  readonly logLikelihood: number;
  /** The Laplace approximation of the marginal log-likelihood, with a random intercept. */
  readonly marginal: number | null;
  readonly converged: boolean;
  readonly iterations: number;
}

/**
 * The design of a log: a sparse matrix with a row per event with an outcome, and the indices of the skills, items and
 * learners of those events, with each learner's state after the log.
 */
interface Design {
  readonly matrix: SparseMatrix;
  /** Each learner's state after the log, in order of first appearance. */
  readonly states: ReadonlyMap<string, WorkingState>;
  /** The first column of the block of each skill and of each item, in order of first appearance. */
  readonly skills: ReadonlyMap<string, number>;
  readonly items: ReadonlyMap<string, number>;
  readonly learners: ReadonlyMap<string, number>;
  /** The learner of each row. */
  readonly rowLearners: Int32Array;
}

/**
 * Builds the design of a log, one row per event. The columns are the shared coefficients' and the covariates', then the
 * block of each skill and each item as it first appears: the intercept, if the model has one, and a column per term with
 * a coefficient for each, from the learner's history before the event. Each learner's state advances in place, so the
 * pass copies no records.
 */
function designOf(log: EventLog, observed: number, terms: Terms): Design {
  const widths = { skill: blockWidth(terms, "skill"), item: blockWidth(terms, "item") };
  const sharedCount = terms.labels.shared.length;
  const skills = new Map<string, number>();
  const items = new Map<string, number>();
  const learners = new Map<string, number>();
  const states = new Map<string, WorkingState>();
  const rowStarts = [0];
  const columns: number[] = [];
  const values: number[] = [];
  const rowLearners = new Int32Array(observed);
  let columnCount = sharedCount;
  let row = 0;
  /** Returns the first column of a block, adding it at the first appearance. */
  const blockOf = (blocks: Map<string, number>, id: string, size: number): number => {
    let base = blocks.get(id);
    if (base === undefined) {
      base = columnCount;
      columnCount += size;
      blocks.set(id, base);
    }
    return base;
  };
  const push = (first: number, block: Float64Array): void => {
    block.forEach((value, j) => {
      if (value === 0) return;
      columns.push(first + j);
      values.push(value);
    });
  };
  for (const event of log) {
    let state = states.get(event.learner);
    if (state === undefined) {
      state = startState(event.learner);
      states.set(event.learner, state);
    }
    if (isObserved(event)) {
      const learner = learners.get(event.learner) ?? learners.size;
      learners.set(event.learner, learner);
      rowLearners[row++] = learner;
      visitRow(terms, state, event, {
        skill: (skill, block) => push(blockOf(skills, skill, widths.skill), block),
        item: (item, block) => push(blockOf(items, item, widths.item), block),
        shared: (block) => push(0, block),
      });
      rowStarts.push(columns.length);
    }
    advanceInPlace(state, event, terms.needs);
  }
  const matrix: SparseMatrix = {
    columnCount,
    rowStarts: Int32Array.from(rowStarts),
    columns: Int32Array.from(columns),
    values: Float64Array.from(values),
  };
  return { matrix, states, skills, items, learners, rowLearners };
}

/**
 * Returns the flat directions of fixed proficiencies. Events connect learners and the skills, or the items, whose
 * intercepts the model has into sets that share none: in each set whose events all have m of them, raising its
 * learners' θ by c and their intercepts by -c/m changes no prediction. Each set thus has its own direction, and its θ are
 * comparable only among themselves.
 */
function flatDirections(log: readonly ObservedEvent[], design: Design, kind: InterceptKind): FlatDirection[] {
  const { matrix, learners, rowLearners } = design;
  const units = kind === "skill" ? design.skills : design.items;
  const ordinals = new Map([...units.keys()].map((id, i) => [id, i]));
  const unitsOf = (event: ObservedEvent): readonly string[] => (kind === "skill" ? event.skills : [event.item]);
  // Union-find over the learners, then the units, with path halving so that long chains of sets stay fast.
  const parent = Int32Array.from({ length: learners.size + units.size }, (_, i) => i);
  const find = (x: number): number => {
    let root = x;
    while (parent[root] !== root) {
      parent[root] = parent[parent[root] ?? root] ?? root;
      root = parent[root] ?? root;
    }
    return root;
  };
  log.forEach((event, row) => {
    for (const id of unitsOf(event))
      parent[find(learners.size + (ordinals.get(id) ?? 0))] = find(rowLearners[row] ?? 0);
  });
  // The number of units of each set's events, or null when they differ.
  const unitCounts = new Map<number, number | null>();
  log.forEach((event, row) => {
    const root = find(rowLearners[row] ?? 0);
    const count = unitsOf(event).length;
    const seen = unitCounts.has(root) ? unitCounts.get(root) : count;
    unitCounts.set(root, seen === count ? seen : null);
  });
  const directions = new Map<number, { coefficients: Float64Array; groups: number[] }>();
  for (let learner = 0; learner < learners.size; learner++) {
    const root = find(learner);
    if (unitCounts.get(root) === null) continue;
    const direction = directions.get(root) ?? { coefficients: new Float64Array(matrix.columnCount), groups: [] };
    directions.set(root, direction);
    direction.groups.push(learner);
  }
  for (const [id, column] of units) {
    const root = find(learners.size + (ordinals.get(id) ?? 0));
    const direction = directions.get(root);
    if (direction !== undefined) direction.coefficients[column] = -1 / (unitCounts.get(root) ?? 1);
  }
  return [...directions.values()].map(({ coefficients, groups }) => ({
    coefficients,
    groups: Int32Array.from(groups),
  }));
}

/** Returns the kind of intercept that fixed proficiencies are not identified apart from, or `null` for none. */
function interceptOf({ intercepts }: Terms): InterceptKind | null {
  if (intercepts.skill) return "skill";
  return intercepts.item ? "item" : null;
}

/**
 * Throws when a fit that solves with dense matrices has more coefficients than {@link MAX_DENSE_COEFFICIENTS}: those
 * without a ridge, and with random proficiency.
 */
function checkDense(coefficients: number, proficiency: Proficiency, ridge: number): void {
  if (coefficients <= MAX_DENSE_COEFFICIENTS) return;
  if (proficiency === "random") {
    throw new DataError(
      `The fit has ${coefficients} coefficients, more than the ${MAX_DENSE_COEFFICIENTS} that a fit with random proficiency solves. Use the proficiency "fixed" or "none" with a ridge.`,
    );
  }
  if (ridge === 0) {
    throw new DataError(
      `The fit has ${coefficients} coefficients, more than the ${MAX_DENSE_COEFFICIENTS} that a fit without a ridge solves. Add a ridge, which a fit of any size solves without the matrix of the coefficients, but which has no standard errors.`,
    );
  }
}

/** Returns the values of a vector over the design's columns in the shape of the parameters, without the learners. */
function shapeOf(
  design: Design,
  terms: Terms,
  values: ArrayLike<number>,
): Pick<LogisticParameters, "skills" | "items" | "shared"> {
  const block = (base: number, intercept: boolean, eachTerms: readonly TermSpec[]): LogisticSkill => ({
    intercept: intercept ? (values[base] ?? 0) : 0,
    coefficients: Object.fromEntries(
      eachTerms.map((term, j) => [term.label, values[base + (intercept ? 1 : 0) + j] ?? 0]),
    ),
  });
  const blocks = (
    map: ReadonlyMap<string, number>,
    kind: InterceptKind,
    eachTerms: readonly TermSpec[],
  ): Record<string, LogisticSkill> =>
    Object.fromEntries([...map].map(([id, base]) => [id, block(base, terms.intercepts[kind], eachTerms)]));
  const hasItems = terms.intercepts.item || terms.items.length > 0;
  return {
    skills: blocks(design.skills, "skill", terms.skills),
    ...(hasItems && { items: blocks(design.items, "item", terms.items) }),
    ...(terms.labels.shared.length > 0 && {
      shared: Object.fromEntries(terms.labels.shared.map((label, j) => [label, values[j] ?? 0])),
    }),
  };
}

/**
 * Returns each event's prediction at a fit, the probability of a correct outcome from its row of the design and its
 * learner's intercept, made before the event as a replay of the log through the fitted model makes it, and the
 * log-likelihood of the outcomes.
 */
function predictionsOf(
  log: EventLog,
  design: Design,
  fit: Pick<LogisticRegression, "coefficients" | "groupIntercepts">,
  proficiency: Proficiency,
): { predictions: (number | null)[]; logLikelihood: number } {
  let row = 0;
  let logLikelihood = 0;
  const predictions = log.map((event) => {
    if (!isObserved(event)) return null;
    const theta = proficiency === "none" ? 0 : (fit.groupIntercepts[design.rowLearners[row] ?? 0] ?? 0);
    const prediction = logistic(theta + linear(design.matrix, row++, fit.coefficients));
    logLikelihood += outcomeLogLikelihood(prediction, event.outcome);
    return prediction;
  });
  return { predictions, logLikelihood };
}

/** Fits the model's coefficients and learners' proficiencies to the events with an outcome of a log. */
export function estimate(log: EventLog, settings: FitSettings): Estimates {
  const { terms, proficiency, ridge, standardErrors: asked, tolerance, maxIterations } = settings;
  const observed = log.filter(isObserved);
  const design = designOf(log, observed.length, terms);
  const { matrix, learners, rowLearners } = design;
  checkDense(matrix.columnCount, proficiency, ridge);
  const kind = interceptOf(terms);
  const flats = proficiency === "fixed" && ridge === 0 && kind !== null ? flatDirections(observed, design, kind) : [];
  const grouping: GroupIntercepts | undefined =
    proficiency === "none" ? undefined : { indices: rowLearners, count: learners.size, kind: proficiency, flats };
  const fit = logisticRegression(
    matrix,
    Float64Array.from(observed, (event) => event.outcome),
    { ...(grouping && { groups: grouping }), ridge, standardErrors: asked, tolerance, maxIterations },
  );
  const learnerValues = (values: ArrayLike<number>): Record<string, number> =>
    Object.fromEntries([...learners].map(([learner, index]) => [learner, values[index] ?? 0]));
  const parameters: LogisticParameters = {
    ...shapeOf(design, terms, fit.coefficients),
    ...(proficiency !== "none" && { learners: learnerValues(fit.groupIntercepts) }),
    ...(fit.sd !== null && { learnerSd: fit.sd }),
  };
  const errors = fit.standardErrors;
  const replayed = predictionsOf(log, design, fit, proficiency);
  const learnerCount = { none: 0, fixed: learners.size - flats.length, random: 1 }[proficiency];
  return {
    parameters,
    standardErrors:
      errors === null
        ? null
        : {
            ...shapeOf(design, terms, errors.coefficients),
            ...(proficiency === "fixed" && { learners: learnerValues(errors.groups) }),
          },
    parameterCount: matrix.columnCount + learnerCount,
    states: Object.fromEntries([...design.states].map(([learner, state]) => [learner, finishState(state)])),
    ...replayed,
    marginal: proficiency === "random" ? fit.logLikelihood : null,
    converged: fit.converged,
    iterations: fit.iterations,
  };
}
