import { bkt, buildEventLog, createRandom as createTracingRandom } from "@measurekit/knowledge-tracing";
import {
  abilityEstimates,
  calibrate,
  createRandom,
  cronbachAlpha,
  simulateItems,
  simulateResponses,
  simulateThetas,
  twoParameterLogistic,
} from "@measurekit/psychometrics";
import { buildResponseMatrix } from "@measurekit/psychometrics/data";

/** What every runtime must give: plain data, the same on each. */
export interface Results {
  /** A 2PL calibration of simulated responses: its items, the fit and the first persons' abilities. */
  readonly calibration: {
    readonly converged: boolean;
    readonly logLikelihood: number;
    readonly alpha: number;
    readonly items: readonly {
      readonly id: string;
      readonly a: number;
      readonly b: number;
      readonly aError: number | null;
      readonly bError: number | null;
    }[];
    readonly thetas: readonly (number | null)[];
  };
  /** A BKT fit of a simulated practice log: its rates for each skill, the first predictions and a learner's mastery. */
  readonly tracing: {
    readonly converged: boolean;
    readonly logLikelihood: number;
    readonly skills: Readonly<
      Record<string, { readonly prior: number; readonly learn: number; readonly guess: number; readonly slip: number }>
    >;
    readonly predictions: readonly (number | null)[];
    readonly mastery: Readonly<Record<string, number>>;
  };
}

/**
 * The decimals that every number of the results keeps. The functions of `Math` may differ in their last bit from one engine to
 * another, and an estimate that stops within its tolerance of 1e-8 keeps that difference far below this, which is what the
 * results of the runtimes are compared by.
 */
const DECIMALS = 6;

const round = (value: number): number => Math.round(value * 10 ** DECIMALS) / 10 ** DECIMALS;
const roundOrNull = (value: number | null | undefined): number | null =>
  value === null || value === undefined ? null : round(value);

/** The persons and items of the calibration. */
const PERSONS = 400;
const ITEMS = 15;

/** Simulates 15 items and 400 persons with a seed, calibrates the 2PL by EM with its standard errors, and estimates abilities. */
function calibration(): Results["calibration"] {
  const random = createRandom(2026);
  const items = simulateItems(twoParameterLogistic, ITEMS, random);
  const { responses } = simulateResponses(items, simulateThetas(PERSONS, random), random);
  const matrix = buildResponseMatrix(responses);
  const fit = calibrate(matrix, { model: twoParameterLogistic, standardErrors: "hessian" });
  return {
    converged: fit.converged,
    logLikelihood: round(fit.logLikelihood),
    alpha: round(cronbachAlpha(matrix)),
    items: fit.items.map(({ id, parameters, standardErrors }) => ({
      id,
      a: round(parameters.a),
      b: round(parameters.b),
      aError: roundOrNull(standardErrors?.a),
      bError: roundOrNull(standardErrors?.b),
    })),
    thetas: abilityEstimates(matrix, fit.items)
      .slice(0, 5)
      .map(({ theta }) => roundOrNull(theta)),
  };
}

/** The learners, the attempts of each and the items of the practice log. */
const LEARNERS = 60;
const ATTEMPTS = 15;
const QUESTIONS = 12;
const SKILLS = ["fractions", "decimals", "percentages"] as const;

/** Simulates a practice log of 60 learners with a seed, fits BKT to it, and predicts and traces a learner. */
function tracing(): Results["tracing"] {
  const random = createTracingRandom(20261029);
  const qMatrix = Object.fromEntries(
    Array.from({ length: QUESTIONS }, (_, i) => [`q${i + 1}`, [SKILLS[i % SKILLS.length] ?? ""]]),
  );
  const events = Array.from({ length: LEARNERS }, (_, learner) => {
    const ability = random.normal();
    // Each learner practices 15 items at random, and practice raises the chance of a correct answer.
    return Array.from({ length: ATTEMPTS }, (_, attempt) => {
      const item = `q${1 + Math.floor(random.uniform() * QUESTIONS)}`;
      const chance = 1 / (1 + Math.exp(-(ability + 0.15 * attempt - 0.3)));
      return {
        learner: `learner${learner + 1}`,
        item,
        time: learner * 100 + attempt,
        outcome: random.uniform() < chance ? 1 : 0,
      };
    });
  }).flat();
  const fit = bkt().fit(buildEventLog(events, { qMatrix }));
  const state = fit.states?.learner7 ?? fit.model.initialState("learner7");
  return {
    converged: fit.converged,
    logLikelihood: round(fit.logLikelihood),
    skills: Object.fromEntries(
      Object.entries(fit.model.parameters.skills).map(([skill, rates]) => [
        skill,
        { prior: round(rates.prior), learn: round(rates.learn), guess: round(rates.guess), slip: round(rates.slip) },
      ]),
    ),
    predictions: fit.predictions.slice(0, 5).map(roundOrNull),
    mastery: Object.fromEntries(SKILLS.map((skill) => [skill, round(fit.model.mastery?.(state, skill) ?? Number.NaN)])),
  };
}

/** Runs both scenarios. */
export function runAll(): Results {
  return { calibration: calibration(), tracing: tracing() };
}
