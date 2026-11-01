import { isObserved, type ObservedEvent } from "../data/event.js";
import type { EventLog } from "../data/event-log.js";
import { entry } from "../data/record.js";
import type { FlatDirection } from "../math/arrow-system.js";
import { type GroupIntercepts, logisticRegression } from "../math/logistic-regression.js";
import type { SparseMatrix } from "../math/sparse-matrix.js";
import { type FeatureSpec, featureValue } from "./features.js";
import type { LogisticParameters, Proficiency } from "./parameters.js";
import { advanceState, type HistoryNeeds, type LogisticState } from "./state.js";

/** What a fit depends on: the checked options of the model. */
export interface FitSettings {
  readonly specs: readonly FeatureSpec[];
  /** What the features need a state to track. */
  readonly needs: HistoryNeeds;
  readonly proficiency: Proficiency;
  readonly ridge: number;
  readonly tolerance: number;
  readonly maxIterations: number;
}

/** The estimates of a fit, with what the fit reports of them. */
export interface Estimates {
  readonly parameters: LogisticParameters;
  readonly parameterCount: number;
  /** The Laplace approximation of the marginal log-likelihood, with a random intercept. */
  readonly marginal: number | null;
  readonly converged: boolean;
  readonly iterations: number;
}

/**
 * The design of a log: a sparse matrix with a row per event with an outcome, and the indices of the skills and learners
 * of those events.
 */
interface Design {
  readonly matrix: SparseMatrix;
  /** The block of columns of each skill, in order of first appearance: the intercept, then one per feature. */
  readonly skills: ReadonlyMap<string, number>;
  readonly learners: ReadonlyMap<string, number>;
  /** The learner of each row. */
  readonly rowLearners: Int32Array;
}

/**
 * Builds the design of a log, one row per event: for each of its skills, a column for the intercept and one per
 * feature, from the learner's history with the skill before the event.
 */
function designOf(log: EventLog, observed: number, { specs, needs }: FitSettings): Design {
  const width = specs.length + 1;
  const skills = new Map<string, number>();
  const learners = new Map<string, number>();
  const histories = new Map<string, LogisticState>();
  const rowStarts = [0];
  const columns: number[] = [];
  const values: number[] = [];
  const rowLearners = new Int32Array(observed);
  let row = 0;
  for (const event of log) {
    const state = histories.get(event.learner) ?? { learner: event.learner, skills: {} };
    if (!isObserved(event)) {
      histories.set(event.learner, advanceState(state, event, needs));
      continue;
    }
    const learner = learners.get(event.learner) ?? learners.size;
    learners.set(event.learner, learner);
    rowLearners[row++] = learner;
    for (const skill of event.skills) {
      const index = skills.get(skill) ?? skills.size;
      skills.set(skill, index);
      const history = entry(state.skills, skill);
      const exposure = entry(state.exposures, skill);
      columns.push(index * width);
      values.push(1);
      specs.forEach((spec, f) => {
        const value = featureValue(spec, history, exposure, event.time, state.practice ?? 0);
        if (value === 0) return;
        columns.push(index * width + f + 1);
        values.push(value);
      });
    }
    rowStarts.push(columns.length);
    histories.set(event.learner, advanceState(state, event, needs));
  }
  const matrix: SparseMatrix = {
    columnCount: skills.size * width,
    rowStarts: Int32Array.from(rowStarts),
    columns: Int32Array.from(columns),
    values: Float64Array.from(values),
  };
  return { matrix, skills, learners, rowLearners };
}

/**
 * Returns the flat directions of fixed proficiencies. Events connect learners and skills into sets that share none: in
 * each set whose events all have m skills, raising its learners' θ by c and its skills' intercepts by -c/m changes no
 * prediction. Each set thus has its own direction, and its θ are comparable only among themselves.
 */
function flatDirections(log: readonly ObservedEvent[], design: Design, width: number): FlatDirection[] {
  const { matrix, skills, learners, rowLearners } = design;
  // Union-find over the learners, then the skills, with path halving so that long chains of sets stay fast.
  const parent = Int32Array.from({ length: learners.size + skills.size }, (_, i) => i);
  const find = (x: number): number => {
    let root = x;
    while (parent[root] !== root) {
      parent[root] = parent[parent[root] ?? root] ?? root;
      root = parent[root] ?? root;
    }
    return root;
  };
  log.forEach((event, row) => {
    for (const skill of event.skills) {
      parent[find(learners.size + (skills.get(skill) ?? 0))] = find(rowLearners[row] ?? 0);
    }
  });
  // The number of skills of each set's events, or null when they differ.
  const skillCounts = new Map<number, number | null>();
  log.forEach((event, row) => {
    const root = find(rowLearners[row] ?? 0);
    const seen = skillCounts.has(root) ? skillCounts.get(root) : event.skills.length;
    skillCounts.set(root, seen === event.skills.length ? seen : null);
  });
  const directions = new Map<number, { coefficients: Float64Array; groups: number[] }>();
  for (let learner = 0; learner < learners.size; learner++) {
    const root = find(learner);
    if (skillCounts.get(root) === null) continue;
    const direction = directions.get(root) ?? { coefficients: new Float64Array(matrix.columnCount), groups: [] };
    directions.set(root, direction);
    direction.groups.push(learner);
  }
  for (const index of skills.values()) {
    const root = find(learners.size + index);
    const direction = directions.get(root);
    if (direction !== undefined) direction.coefficients[index * width] = -1 / (skillCounts.get(root) ?? 1);
  }
  return [...directions.values()].map(({ coefficients, groups }) => ({
    coefficients,
    groups: Int32Array.from(groups),
  }));
}

/** Fits the model's coefficients and learners' proficiencies to the events with an outcome of a log. */
export function estimate(log: EventLog, settings: FitSettings): Estimates {
  const { specs, proficiency, ridge, tolerance, maxIterations } = settings;
  const observed = log.filter(isObserved);
  const { matrix, skills, learners, rowLearners } = designOf(log, observed.length, settings);
  const width = specs.length + 1;
  const flats =
    proficiency === "fixed" && ridge === 0
      ? flatDirections(observed, { matrix, skills, learners, rowLearners }, width)
      : [];
  const grouping: GroupIntercepts | undefined =
    proficiency === "none" ? undefined : { indices: rowLearners, count: learners.size, kind: proficiency, flats };
  const fit = logisticRegression(
    matrix,
    Float64Array.from(observed, (event) => event.outcome),
    { ...(grouping && { groups: grouping }), ridge, tolerance, maxIterations },
  );
  const coefficient = (index: number, f: number): number => fit.coefficients[index * width + f] ?? 0;
  const parameters: LogisticParameters = {
    skills: Object.fromEntries(
      [...skills].map(([skill, index]) => [
        skill,
        {
          intercept: coefficient(index, 0),
          coefficients: Object.fromEntries(specs.map((spec, f) => [spec.label, coefficient(index, f + 1)])),
        },
      ]),
    ),
    ...(proficiency !== "none" && {
      learners: Object.fromEntries([...learners].map(([learner, index]) => [learner, fit.groupIntercepts[index] ?? 0])),
    }),
    ...(fit.sd !== null && { learnerSd: fit.sd }),
  };
  const learnerCount = { none: 0, fixed: learners.size - flats.length, random: 1 }[proficiency];
  return {
    parameters,
    parameterCount: matrix.columnCount + learnerCount,
    marginal: proficiency === "random" ? fit.logLikelihood : null,
    converged: fit.converged,
    iterations: fit.iterations,
  };
}
