import { cholesky, logistic, mean } from "@measurekit/core";
import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { createMultidimensionalAdaptiveTest, testLength } from "../src/cat/index.js";
import { buildResponseMatrix, buildResponseRecords, type ResponseMatrix } from "../src/data/index.js";
import { pearsonMatrix } from "../src/dimensionality/internal.js";
import { createRandom, DataError } from "../src/index.js";
import {
  betaPrior,
  calibrate,
  generalizedPartialCredit,
  gradedResponse,
  logNormalPrior,
  maximumLikelihoodEstimate,
  normalPrior,
  oneParameterLogistic,
  type ParameterPrior,
  posteriorMeanEstimate,
  posteriorModeEstimate,
  threeParameterLogistic,
  twoParameterLogistic,
  weightedLikelihoodEstimate,
} from "../src/irt/index.js";
import { compressResponses } from "../src/irt/internal.js";
import {
  calibrateBifactor,
  calibrateLongitudinal,
  calibrateMultidimensional,
  type LoadingPattern,
  type LongitudinalCalibration,
  type MultidimensionalCalibration,
  type MultidimensionalCalibrationOptions,
  type MultidimensionalGradedParameters,
  type MultidimensionalItem,
  type MultidimensionalLogisticParameters,
  type MultidimensionalModel,
  type MultidimensionalTestItem,
  multidimensionalDifficulty,
  multidimensionalDiscrimination,
  multidimensionalEstimates,
  multidimensionalGradedResponse,
  multidimensionalInformation,
  multidimensionalMaximumLikelihoodEstimate,
  multidimensionalPosteriorMeanEstimate,
  multidimensionalPosteriorModeEstimate,
  multidimensionalThreeParameterLogistic,
  multidimensionalTwoParameterLogistic,
  multidimensionalWeightedLikelihoodEstimate,
  type RotationOptions,
  rotateLoadings,
} from "../src/multidimensional/index.js";
import {
  createChain,
  createCorrelations,
  createScratch,
  designOf,
  type ExpectationItem,
  expectationStep,
  gridLikelihood,
  itemCorrelations,
  itemMoments,
  latentDistribution,
  latentObservedInformation,
  latentScore,
  observedItemInformation,
  personLatentScores,
  productGrid,
  responsesByItem,
  responsesByPerson,
  secondMoments,
  setUp,
} from "../src/multidimensional/internal.js";
import { simulateMultidimensionalResponses } from "../src/sim/index.js";
import { defined, expectClose, expectParameters } from "./assertions.js";
import { MODELS as FIXTURE_MODELS } from "./fixture-items.js";
import { readFixture } from "./fixtures.js";

interface LongitudinalFixture {
  two: {
    source: string;
    model: string;
    points: number;
    occasions: { personIds: string[]; itemIds: string[]; responses: number[][] }[];
    items: { id: string; parameters: { a: number; b: number; c: number; d: number } }[];
    standardErrors: number[];
    crossProductStandardErrors: number[];
    means: number[];
    sds: number[];
    correlations: number[][];
    covariance: number[][];
    changes: { from: number; to: number; mean: number; sd: number; correlationWithStart: number }[];
    logLikelihood: number;
    iterations: number;
  };
  three: {
    source: string;
    model: string;
    points: number;
    occasions: { personIds: string[]; itemIds: string[]; responses: number[][] }[];
    items: { id: string; parameters: { a: number; b: number; c: number; d: number } }[];
    means: number[];
    sds: number[];
    correlations: number[][];
    covariance: number[][];
    changes: { from: number; to: number; mean: number; sd: number; correlationWithStart: number }[];
    logLikelihood: number;
    iterations: number;
  };
  anchored: {
    source: string;
    model: string;
    points: number;
    occasions: { personIds: string[]; itemIds: string[]; responses: number[][] }[];
    items: { id: string; parameters: { a: number; b: number; c: number; d: number } }[];
    means: number[];
    sds: number[];
    correlations: number[][];
    covariance: number[][];
    changes: { from: number; to: number; mean: number; sd: number; correlationWithStart: number }[];
    logLikelihood: number;
    iterations: number;
  };
  graded: {
    source: string;
    model: string;
    points: number;
    occasions: { personIds: string[]; itemIds: string[]; responses: number[][] }[];
    items: { id: string; parameters: { a: number; b: number[] } }[];
    means: number[];
    sds: number[];
    correlations: number[][];
    covariance: number[][];
    changes: { from: number; to: number; mean: number; sd: number; correlationWithStart: number }[];
    logLikelihood: number;
    iterations: number;
  };
  rasch: {
    source: string;
    model: string;
    points: number;
    occasions: { personIds: string[]; itemIds: string[]; responses: number[][] }[];
    items: { id: string; parameters: { a: number; b: number; c: number; d: number } }[];
    tam: {
      slope: number;
      difficulties: number[];
      means: number[];
      sds: number[];
      correlations: number[][];
      logLikelihood: number;
    };
    means: number[];
    sds: number[];
    correlations: number[][];
    covariance: number[][];
    changes: { from: number; to: number; mean: number; sd: number; correlationWithStart: number }[];
    logLikelihood: number;
    iterations: number;
  };
}
const longitudinal = readFixture<LongitudinalFixture>("longitudinal.json");

interface LongitudinalRecoveryFixture {
  source: string;
  truth: { means: number[]; sds: number[]; rho: number };
  recovery: {
    persons: number;
    occasions: number;
    points: number;
    fits: { seed: number; mean: number; sd: number; correlation: number; iterations: number; converged: boolean }[];
    worst: { mean: number; sd: number; correlation: number };
  }[];
  grid: {
    persons: number;
    occasions: number;
    points: number;
    fits: { seed: number; mean: number; sd: number; correlation: number; iterations: number; converged: boolean }[];
    worst: { mean: number; sd: number; correlation: number };
  }[];
}
const longitudinalRecovery = readFixture<LongitudinalRecoveryFixture>("longitudinal-recovery.json");

interface MultidimensionalExploratoryFixture {
  flat: {
    source: string;
    model: string;
    points: number;
    dimensions: number;
    itemIds: string[];
    responses: number[][];
    items: { id: string; parameters: { a: number[]; intercept: number; c: number; d: number } }[];
    logLikelihood: number;
    aic: number;
    bic: number;
    iterations: number;
    baseMaximum: { slopes: number[][]; intercepts: number[]; logLikelihood: number; gradient: number };
  };
}
const exploratory = readFixture<MultidimensionalExploratoryFixture>("multidimensional-exploratory.json");

interface MultidimensionalGradedFixture {
  source: string;
  model: string;
  points: number;
  itemIds: string[];
  responses: number[][];
  loadings: Record<string, string[]>;
  items: {
    id: string;
    parameters: { a: number[]; intercepts: number[] };
    standardErrors: {
      a1?: number;
      intercept1: number;
      intercept2: number;
      intercept3: number;
      intercept4: number;
      intercept5: number;
      a2?: number;
    };
  }[];
  correlations: number[][];
  correlationStandardErrors: number[][];
  logLikelihood: number;
  aic: number;
  bic: number;
  iterations: number;
  standardized: { loadings: number[][]; communalities: number[] };
  scoring: {
    patterns: (number | null)[][];
    points: number;
    eap: { theta: number[]; standardErrors: number[] }[];
    map: { theta: number[]; standardErrors: number[]; expectedStandardErrors: number[] }[];
  };
}
const graded = readFixture<MultidimensionalGradedFixture>("multidimensional-graded.json");

interface MultidimensionalGraded3Fixture {
  source: string;
  model: string;
  points: number;
  itemIds: string[];
  responses: number[][];
  loadings: Record<string, string[]>;
  items: { id: string; parameters: { a: number[]; intercepts: number[] } }[];
  correlations: number[][];
  logLikelihood: number;
  aic: number;
  bic: number;
  iterations: number;
}
const graded3 = readFixture<MultidimensionalGraded3Fixture>("multidimensional-graded3.json");

interface MultidimensionalGroupsFixture {
  correlated: {
    source: string;
    model: string;
    points: number;
    itemIds: string[];
    responses: number[][];
    personGroups: string[];
    loadings: { q02: string[] };
    items: {
      id: string;
      parameters: { a: number[]; intercept: number; c: number; d: number };
      standardErrors: { a1?: number; intercept: number; a2?: number };
    }[];
    groupMoments: { group: string; means: number[]; covariance: number[][] }[];
    correlations: number[][];
    correlationStandardErrors: number[][];
    logLikelihood: number;
    aic: number;
    bic: number;
    iterations: number;
  };
  anchors: {
    source: string;
    model: string;
    points: number;
    itemIds: string[];
    responses: number[][];
    personGroups: string[];
    loadings: { q02: string[] };
    fixed: { q02: { a: number[]; intercept: number; c: number; d: number } };
    items: {
      id: string;
      parameters: { a: number[]; intercept: number; c: number; d: number };
      standardErrors?: { a1?: number; intercept: number; a2?: number };
    }[];
    groupMoments: { group: string; means: number[]; covariance: number[][] }[];
    correlations: number[][];
    logLikelihood: number;
    aic: number;
    bic: number;
    iterations: number;
  };
  uncorrelated: {
    source: string;
    model: string;
    points: number;
    itemIds: string[];
    responses: (number | null)[][];
    personGroups: string[];
    loadings: { q02: string[] };
    correlated: boolean;
    items: {
      id: string;
      parameters: { a: number[]; intercept: number; c: number; d: number };
      standardErrors: { a1?: number; intercept: number; a2?: number };
    }[];
    groupMoments: { group: string; means: number[]; covariance: number[][] }[];
    correlations: number[][];
    logLikelihood: number;
    aic: number;
    bic: number;
    iterations: number;
  };
  graded: {
    source: string;
    model: string;
    points: number;
    itemIds: string[];
    responses: number[][];
    personGroups: string[];
    loadings: { q02: string[] };
    items: { id: string; parameters: { a: number[]; intercepts: number[] } }[];
    groupMoments: { group: string; means: number[]; covariance: number[][] }[];
    correlations: number[][];
    logLikelihood: number;
    aic: number;
    bic: number;
    iterations: number;
  };
  three: {
    source: string;
    model: string;
    points: number;
    itemIds: string[];
    responses: number[][];
    personGroups: string[];
    loadings: { q02: string[] };
    items: { id: string; parameters: { a: number[]; intercept: number; c: number; d: number } }[];
    groupMoments: { group: string; means: number[]; covariance: number[][] }[];
    correlations: number[][];
    logLikelihood: number;
    aic: number;
    bic: number;
    iterations: number;
  };
}
const groups = readFixture<MultidimensionalGroupsFixture>("multidimensional-groups.json");

interface MultidimensionalInformationFixture {
  source: string;
  cases: {
    model: string;
    dimensions: number;
    items: { a: number[]; intercept?: number; c?: number; d?: number; intercepts?: number[] }[];
    thetas: number[][];
    matrices: number[][][][];
    discrimination: number[];
    difficulty: number[][];
  }[];
}
const information = readFixture<MultidimensionalInformationFixture>("multidimensional-information.json");

interface MultidimensionalLikelihoodFixture {
  source: string;
  range: number[];
  cases: {
    model: string;
    dimensions: number;
    items: { a: number[]; intercept?: number; c?: number; d?: number; intercepts?: number[] }[];
    estimates: {
      pattern: (number | null)[];
      ml: number[];
      mlStandardErrors: number[];
      wle: number[];
      wleStandardErrors: number[];
      mirtMl: (number | null)[];
      mirtWle: number[];
    }[];
  }[];
}
const likelihood = readFixture<MultidimensionalLikelihoodFixture>("multidimensional-likelihood.json");

interface MultidimensionalM2plFixture {
  source: string;
  model: string;
  points: number;
  itemIds: string[];
  responses: number[][];
  loadings: { q02: string[] };
  items: {
    id: string;
    parameters: { a: number[]; intercept: number; c: number; d: number };
    standardErrors: { a1?: number; intercept: number; a2?: number };
  }[];
  correlations: number[][];
  correlationStandardErrors: number[][];
  logLikelihood: number;
  aic: number;
  bic: number;
  iterations: number;
  standardized: { loadings: number[][]; communalities: number[] };
  crossProduct: {
    standardErrors: { a1?: number; intercept: number; a2?: number }[];
    correlationStandardErrors: number[][];
  };
  scoring: {
    patterns: (number | null)[][];
    points: number;
    eap: { theta: number[]; standardErrors: number[] }[];
    map: { theta: number[]; standardErrors: number[]; expectedStandardErrors: number[] }[];
  };
}
const m2pl = readFixture<MultidimensionalM2plFixture>("multidimensional-m2pl.json");

interface MultidimensionalM3plFixture {
  source: string;
  model: string;
  points: number;
  itemIds: string[];
  responses: number[][];
  loadings: { q02: string[] };
  priors: { c: { type: string; mean: number; sd: number } };
  items: {
    id: string;
    parameters: { a: number[]; intercept: number; c: number; d: number };
    standardErrors: { a1?: number; intercept: number; c: number; a2?: number };
  }[];
  correlations: number[][];
  correlationStandardErrors: number[][];
  logLikelihood: number;
  aic: number;
  bic: number;
  iterations: number;
}
const m3pl = readFixture<MultidimensionalM3plFixture>("multidimensional-m3pl.json");

interface MultidimensionalHeldFixture {
  source: string;
  models: ("m2pl" | "m3pl")[];
  points: number;
  itemIds: string[];
  responses: number[][];
  loadings: Record<string, string[]>;
  fixedParameters: Record<string, Record<string, number>>;
  items: {
    id: string;
    parameters: { a: number[]; intercept: number; c: number; d: number };
    standardErrors: Record<string, number>;
  }[];
  estimated: number;
  correlations: number[][];
  correlationStandardErrors: number[][];
  logLikelihood: number;
  aic: number;
  bic: number;
  iterations: number;
  crossProduct: { standardErrors: Record<string, number>[]; correlationStandardErrors: number[][] };
}
const held = readFixture<MultidimensionalHeldFixture>("multidimensional-fixed-parameters.json");

interface MultidimensionalMhrmFixture {
  source: string;
  model: string;
  itemIds: string[];
  responses: number[][];
  loadings: Record<string, string[]>;
  runs: {
    seed: number;
    items: {
      id: string;
      parameters: { a: number[]; intercepts: number[] };
      standardErrors: {
        a1?: number;
        intercept1: number;
        intercept2: number;
        intercept3: number;
        intercept4: number;
        intercept5: number | null;
        a2?: number;
        a3?: number;
        a5?: number;
      };
    }[];
    correlations: number[][];
    correlationStandardErrors: number[][];
    logLikelihood: number;
    aic: number;
    bic: number;
    iterations: number;
  }[];
  thetas: number[][];
  completeData: { item: string; parameters: number[]; responses: number[]; gradient: number[]; hessian: number[][] }[];
}
const mhrm = readFixture<MultidimensionalMhrmFixture>("multidimensional-mhrm.json");

interface MultidimensionalMissingFixture {
  source: string;
  model: string;
  points: number;
  itemIds: string[];
  responses: (number | null)[][];
  loadings: { q02: string[] };
  items: {
    id: string;
    parameters: { a: number[]; intercept: number; c: number; d: number };
    standardErrors: { a1?: number; intercept: number; a2?: number };
  }[];
  correlations: number[][];
  correlationStandardErrors: number[][];
  logLikelihood: number;
  aic: number;
  bic: number;
  iterations: number;
}
const missing = readFixture<MultidimensionalMissingFixture>("multidimensional-missing.json");

interface MultidimensionalNotAtRandomFixture {
  source: string;
  model: string;
  points: number;
  itemIds: string[];
  responses: (number | null)[][];
  loadings: { q02: string[] };
  items: { id: string; parameters: { a: number[]; intercept: number; c: number; d: number } }[];
  correlations: number[][];
  logLikelihood: number;
  aic: number;
  bic: number;
  truth: { correlation: number; share: number; slopes: number[]; intercepts: number[] };
}
const notAtRandom = readFixture<MultidimensionalNotAtRandomFixture>("multidimensional-not-at-random.json");

interface MultidimensionalRotationFixture {
  source: string;
  slopes: number[][];
  intercepts: number[];
  loadings: number[][];
  rotations: {
    options: { method: string; normalize?: boolean; gamma?: number };
    loadings: number[][];
    correlations: number[][];
    rotation: number[][];
    iterations: number;
    slopes: number[][];
  }[];
  scoring: { patterns: (number | null)[][]; points: number; eap: { theta: number[]; standardErrors: number[] }[] };
}
const rotation = readFixture<MultidimensionalRotationFixture>("multidimensional-rotation.json");

/** The built-in models by the names that fixtures give them. */
const MODELS = {
  m2pl: multidimensionalTwoParameterLogistic,
  m3pl: multidimensionalThreeParameterLogistic,
  mgrm: multidimensionalGradedResponse,
} as const;

type ModelName = keyof typeof MODELS;
type Parameters = MultidimensionalLogisticParameters | MultidimensionalGradedParameters;

/** Item parameters in a fixture: the logistic and the graded names, whichever the model has. */
interface FixtureParameters {
  readonly a: readonly number[];
  readonly intercept?: number;
  readonly intercepts?: readonly number[];
  readonly c?: number;
}

/** One calibrated item of a fixture: its parameters and, where mirt gave them, its standard errors. */
interface FixtureItem {
  readonly id: string;
  readonly parameters: FixtureParameters;
  readonly standardErrors?: Readonly<Record<string, number | undefined>>;
}

/** A calibration of mirt against which the tests compare. */
interface Fixture {
  readonly model: string;
  readonly points: number;
  readonly itemIds: readonly string[];
  readonly responses: readonly (readonly (number | null)[])[];
  readonly loadings: LoadingPattern;
  readonly items: readonly FixtureItem[];
  readonly correlations: readonly (readonly number[])[];
  readonly correlationStandardErrors?: readonly (readonly number[])[];
  readonly standardized?: {
    readonly loadings: readonly (readonly number[])[];
    readonly communalities: readonly number[];
  };
  readonly logLikelihood: number;
  readonly aic: number;
  readonly bic: number;
}

/** One item of a run of mirt's MH-RM: its standard error is `null` where mirt's information was not positive definite. */
interface MhrmItem {
  readonly id: string;
  readonly parameters: FixtureParameters;
  readonly standardErrors?: Readonly<Record<string, number | null | undefined>>;
}

/** Runs of mirt's MH-RM on one dataset, with seeds 1 to 4, and the complete-data score and Hessian of two items at given thetas. */
interface MhrmFixture extends Pick<Fixture, "model" | "itemIds" | "responses" | "loadings"> {
  readonly runs: readonly {
    readonly items: readonly MhrmItem[];
    readonly correlations: readonly (readonly number[])[];
  }[];
  readonly thetas: readonly (readonly number[])[];
  readonly completeData: readonly {
    readonly item: string;
    readonly parameters: readonly number[];
    readonly gradient: readonly number[];
    readonly hessian: readonly (readonly number[])[];
  }[];
}

const mirtMhrm: MhrmFixture = mhrm;
/**
 * The time a calibration of several groups may take under coverage, which CI's job of Node.js 22 runs and which makes it
 * about five times as long: the slowest took 79 to 213 s under it, and 20 s without it.
 */
const UNDER_COVERAGE = 600_000;

function modelOf(name: string): MultidimensionalModel<Parameters> {
  return defined(MODELS[name as ModelName]);
}

function parametersOf(model: string, { a, intercept = 0, intercepts = [], c = 0 }: FixtureParameters): Parameters {
  return model === "mgrm" ? { a, intercepts } : { a, intercept, c, d: 1 };
}

/** The items of a fixture, as scoring takes them. */
function itemsOf(fixture: Pick<Fixture, "model" | "items">): MultidimensionalTestItem<Parameters>[] {
  return fixture.items.map(({ id, parameters }) => ({
    id,
    model: modelOf(fixture.model),
    parameters: parametersOf(fixture.model, parameters),
  }));
}

/** A normal prior on logit(x), as mirt's `norm` on g. Implemented here as user code. */
function logitNormalPrior(mean: number, sd: number): ParameterPrior {
  const logit = (x: number): number => Math.log(x / (1 - x));
  const variance = sd * sd;
  return {
    logDensity: (x) => (-0.5 * (logit(x) - mean) ** 2) / variance,
    logDensityDerivative: (x) => -(logit(x) - mean) / (variance * x * (1 - x)),
    logDensitySecondDerivative: (x) => -(1 - (logit(x) - mean) * (1 - 2 * x)) / (variance * (x * (1 - x)) ** 2),
  };
}

/** Compares a calibration with mirt's: the parameters, correlations and fit statistics, and the standard errors. */
function expectCalibration(
  fit: MultidimensionalCalibration<Parameters>,
  fixture: Omit<Fixture, "model" | "points" | "itemIds" | "responses" | "loadings">,
  tolerances: { parameters: number; errors: number; likelihood: number },
  label: string,
): void {
  fixture.items.forEach((expected, i) => {
    const found = defined(fit.items[i]);
    const parameters = found.parameters;
    const target = expected.parameters;
    const what = `${label} ${expected.id}`;
    expect(found.id).toBe(expected.id);
    target.a.forEach((slope, d) => {
      expectClose(defined(parameters.a[d]), slope, tolerances.parameters, `${what} a${d + 1}`);
    });
    if ("intercepts" in parameters) {
      (target.intercepts ?? []).forEach((value, k) => {
        expectClose(defined(parameters.intercepts[k]), value, tolerances.parameters, `${what} intercept${k + 1}`);
      });
    } else {
      expectClose(parameters.intercept, defined(target.intercept), tolerances.parameters, `${what} intercept`);
      expectClose(parameters.c, defined(target.c), tolerances.parameters, `${what} c`);
    }
    for (const [name, error] of Object.entries(expected.standardErrors ?? {})) {
      const actual = found.standardErrors?.[name] ?? Number.NaN;
      expectClose(actual / defined(error), 1, tolerances.errors, `${what} standard error of ${name}`);
    }
  });
  fixture.correlations.forEach((row, i) => {
    row.forEach((value, j) => {
      expectClose(
        fit.correlations[i]?.[j] ?? Number.NaN,
        value,
        tolerances.parameters,
        `${label} correlation ${i},${j}`,
      );
    });
  });
  fixture.correlationStandardErrors?.forEach((row, i) => {
    row.forEach((value, j) => {
      if (i === j) return;
      const actual = fit.correlationStandardErrors?.[i]?.[j] ?? Number.NaN;
      expectClose(actual / value, 1, tolerances.errors, `${label} standard error of correlation ${i},${j}`);
    });
  });
  fixture.standardized?.loadings.forEach((row, i) => {
    row.forEach((value, d) => {
      expectClose(defined(fit.loadings[i]?.[d]), value, tolerances.parameters, `${label} loading ${i + 1},${d + 1}`);
    });
    expectClose(
      defined(fit.communalities[i]),
      defined(fixture.standardized?.communalities[i]),
      tolerances.parameters,
      `${label} communality ${i + 1}`,
    );
  });
  expectClose(fit.logLikelihood, fixture.logLikelihood, tolerances.likelihood, `${label} log-likelihood`);
  expectClose(fit.aic, fixture.aic, tolerances.likelihood, `${label} AIC`);
  expectClose(fit.bic, fixture.bic, tolerances.likelihood, `${label} BIC`);
}

function calibrateFixture(
  fixture: Fixture,
  standardErrors: "none" | "hessian" | "cross-product",
  extra: { readonly priors?: Readonly<Record<string, ParameterPrior>>; readonly tolerance: number },
): MultidimensionalCalibration<Parameters> {
  return calibrateMultidimensional(
    buildResponseMatrix(fixture.responses, { itemIds: [...fixture.itemIds] }),
    modelOf(fixture.model),
    {
      loadings: fixture.loadings,
      points: fixture.points,
      standardErrors,
      maxIterations: 2000,
      ...extra,
    },
  );
}

describe("models", () => {
  const slope = fc.double({ min: -2.5, max: 3, noNaN: true });
  const location = fc.double({ min: -3, max: 3, noNaN: true });
  const thetaFor = (dimensions: number): fc.Arbitrary<number[]> =>
    fc.array(fc.double({ min: -4, max: 4, noNaN: true }), { minLength: dimensions, maxLength: dimensions });

  /** Items of every built-in model on one to three dimensions, with a theta. */
  const itemAndTheta: fc.Arbitrary<{ item: MultidimensionalItem; theta: number[] }> = fc
    .integer({ min: 1, max: 3 })
    .chain((dimensions) => {
      const slopes = fc.array(slope, { minLength: dimensions, maxLength: dimensions });
      const logistic = fc
        .record({
          a: slopes,
          intercept: location,
          c: fc.double({ min: 0, max: 0.4, noNaN: true }),
          guessing: fc.boolean(),
        })
        .map(({ a, intercept, c, guessing }) => ({
          model: guessing ? multidimensionalThreeParameterLogistic : multidimensionalTwoParameterLogistic,
          parameters: { a, intercept, c: guessing ? c : 0, d: 1 },
        }));
      const gradedItem = fc
        .record({
          a: slopes,
          first: location,
          gaps: fc.array(fc.double({ min: 0.05, max: 2, noNaN: true }), { minLength: 1, maxLength: 4 }),
        })
        .map(({ a, first, gaps }) => {
          const intercepts = [first];
          for (const gap of gaps) intercepts.push((intercepts.at(-1) ?? 0) - gap);
          return { model: multidimensionalGradedResponse, parameters: { a, intercepts } };
        });
      return fc.record({ item: fc.oneof(logistic, gradedItem), theta: thetaFor(dimensions) });
    });

  /** Returns the category probabilities of an item at a theta. */
  function probabilities(item: MultidimensionalItem, theta: readonly number[]): Float64Array {
    const out = new Float64Array(item.model.categoryCount(item.parameters));
    item.model.probabilities(item.parameters, Float64Array.from(theta), out);
    return out;
  }

  it("gives probabilities that sum to 1", () => {
    fc.assert(
      fc.property(itemAndTheta, ({ item, theta }) => {
        const p = probabilities(item, theta);
        expect(p.every((value) => value >= 0 && value <= 1)).toBe(true);
        expect(p.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 12);
      }),
    );
  });

  it("differentiates the probabilities with respect to theta and to the parameter vector", () => {
    const step = 1e-6;
    fc.assert(
      fc.property(itemAndTheta, ({ item, theta }) => {
        const { model, parameters } = item;
        const dimensions = theta.length;
        const categories = model.categoryCount(parameters);
        const gradients = new Float64Array(categories * dimensions);
        model.thetaGradients(parameters, Float64Array.from(theta), gradients);
        for (let d = 0; d < dimensions; d++) {
          const up = probabilities(
            item,
            theta.map((value, k) => (k === d ? value + step : value)),
          );
          const down = probabilities(
            item,
            theta.map((value, k) => (k === d ? value - step : value)),
          );
          for (let k = 0; k < categories; k++) {
            expect(gradients[k * dimensions + d] ?? 0).toBeCloseTo(((up[k] ?? 0) - (down[k] ?? 0)) / (2 * step), 6);
          }
        }
        const vector = model.toVector(parameters);
        const derivatives = new Float64Array(categories * vector.length);
        model.parameterDerivatives(parameters, Float64Array.from(theta), derivatives);
        for (let j = 0; j < vector.length; j++) {
          const moved = (shift: number): Float64Array => {
            const changed = Float64Array.from(vector);
            changed[j] = (changed[j] ?? 0) + shift;
            return probabilities({ model, parameters: model.fromVector(changed, dimensions) }, theta);
          };
          const up = moved(step);
          const down = moved(-step);
          for (let k = 0; k < categories; k++) {
            expect(derivatives[k * vector.length + j] ?? 0).toBeCloseTo(
              ((up[k] ?? 0) - (down[k] ?? 0)) / (2 * step),
              6,
            );
          }
        }
      }),
    );
  });

  it("gives an information matrix that is symmetric, positive semidefinite and the sum of the outer products of the gradients", () => {
    fc.assert(
      fc.property(itemAndTheta, ({ item, theta }) => {
        const { model, parameters } = item;
        const dimensions = theta.length;
        const point = Float64Array.from(theta);
        const matrix = new Float64Array(dimensions * dimensions);
        model.information(parameters, point, matrix);
        const p = probabilities(item, theta);
        const gradients = new Float64Array(p.length * dimensions);
        model.thetaGradients(parameters, point, gradients);
        for (let i = 0; i < dimensions; i++) {
          for (let j = 0; j < dimensions; j++) {
            let sum = 0;
            for (let k = 0; k < p.length; k++) {
              if ((p[k] ?? 0) > 1e-12)
                sum += ((gradients[k * dimensions + i] ?? 0) * (gradients[k * dimensions + j] ?? 0)) / (p[k] ?? 1);
            }
            const entry = matrix[i * dimensions + j] ?? 0;
            expect(Math.abs(entry - sum)).toBeLessThanOrEqual(1e-9 * Math.max(1, Math.abs(sum)));
            expect(entry).toBe(matrix[j * dimensions + i] ?? 0);
          }
          expect(matrix[i * dimensions + i] ?? 0).toBeGreaterThanOrEqual(0);
        }
      }),
    );
  });

  it("round-trips the parameter vector", () => {
    fc.assert(
      fc.property(itemAndTheta, ({ item, theta }) => {
        const vector = item.model.toVector(item.parameters);
        expect(item.model.fromVector(vector, theta.length)).toEqual(item.parameters);
        const names = item.model.parameters(item.model.categoryCount(item.parameters), theta.length);
        expect(names).toHaveLength(vector.length);
      }),
    );
  });

  it("names the estimated parameters with the slopes first", () => {
    expect(multidimensionalTwoParameterLogistic.parameters(2, 3).map(({ name }) => name)).toEqual([
      "a1",
      "a2",
      "a3",
      "intercept",
    ]);
    expect(multidimensionalThreeParameterLogistic.parameters(2, 2)).toEqual([
      { name: "a1" },
      { name: "a2" },
      { name: "intercept" },
      { name: "c", lower: 0, upper: 1 },
    ]);
    expect(multidimensionalGradedResponse.parameters(4, 2).map(({ name }) => name)).toEqual([
      "a1",
      "a2",
      "intercept1",
      "intercept2",
      "intercept3",
    ]);
  });

  it("reads the numbers of categories and dimensions from the parameters", () => {
    expect(multidimensionalTwoParameterLogistic.categoryCount({ a: [1, 2], intercept: 0, c: 0, d: 1 })).toBe(2);
    expect(multidimensionalGradedResponse.categoryCount({ a: [1], intercepts: [1, 0, -1] })).toBe(4);
    expect(multidimensionalGradedResponse.dimensionCount({ a: [1, 2, 3], intercepts: [0] })).toBe(3);
  });

  it("starts from the shares of the categories and the given slopes", () => {
    const start = multidimensionalThreeParameterLogistic.startingValues(
      Float64Array.of(0.7, 0.3),
      Float64Array.of(1, 0),
    );
    expect(start.a).toEqual([1, 0]);
    expect(start.c).toBeGreaterThan(0);
    expect(start.c).toBeLessThan(0.3);
    const intercepts = multidimensionalGradedResponse.startingValues(
      Float64Array.of(0.2, 0.3, 0.5),
      Float64Array.of(1),
    ).intercepts;
    expect(intercepts[0]).toBeCloseTo(Math.log(0.8 / 0.2), 12);
    expect(intercepts[1]).toBeCloseTo(Math.log(0.5 / 0.5), 12);
  });

  it("keeps small probabilities precise in the tails", () => {
    const tail = probabilities(
      { model: multidimensionalTwoParameterLogistic, parameters: { a: [1, 1], intercept: 0, c: 0, d: 1 } },
      [30, 30],
    );
    expect(tail[0]).toBeGreaterThan(0);
    expect(tail[0]).toBeCloseTo(Math.exp(-60), 40);
    const item = { model: multidimensionalGradedResponse, parameters: { a: [2], intercepts: [3, -3] } };
    expect(probabilities(item, [40])[0]).toBeGreaterThan(0);
  });

  it("rejects invalid parameters", () => {
    expect(() => multidimensionalTwoParameterLogistic.toVector({ a: [], intercept: 0, c: 0, d: 1 })).toThrow(
      "An M2PL item needs finite slopes a, at least one",
    );
    expect(() => multidimensionalTwoParameterLogistic.toVector({ a: [1], intercept: 0, c: 0.1, d: 1 })).toThrow(
      "An M2PL item has c = 0, found c = 0.1.",
    );
    expect(() => multidimensionalTwoParameterLogistic.toVector({ a: [1], intercept: 0, c: 0, d: 0.9 })).toThrow(
      "An M2PL item has d = 1, found d = 0.9.",
    );
    expect(() => multidimensionalThreeParameterLogistic.toVector({ a: [1], intercept: 0, c: 1, d: 1 })).toThrow(
      DataError,
    );
    expect(() => multidimensionalTwoParameterLogistic.toVector({ a: [Number.NaN], intercept: 0, c: 0, d: 1 })).toThrow(
      DataError,
    );
    expect(() => multidimensionalGradedResponse.toVector({ a: [1, 1], intercepts: [0, 0] })).toThrow(
      "An MGRM item needs strictly decreasing intercepts, found [0, 0].",
    );
    expect(() => multidimensionalGradedResponse.toVector({ a: [1], intercepts: [-1, 1] })).toThrow(DataError);
    expect(() => multidimensionalGradedResponse.toVector({ a: [1], intercepts: [] })).toThrow(
      "An MGRM item needs finite slopes a, at least one, and at least one finite intercept",
    );
    expect(() => multidimensionalGradedResponse.toVector(JSON.parse("null"))).toThrow(DataError);
    expect(() => multidimensionalTwoParameterLogistic.toVector(JSON.parse("null"))).toThrow(DataError);
  });
});

describe("latent distribution", () => {
  /** The expected number of persons at each node of a grid under correlations `matrix`, for 1,000 persons. */
  function counts(dimensions: number, points: number, matrix: Float64Array): Float64Array {
    const latent = latentDistribution(productGrid(dimensions, points));
    expect(latent.set(matrix)).toBe(true);
    return latent.logWeights.map((logWeight) => 1000 * Math.exp(logWeight));
  }

  const Correlations3 = Float64Array.of(1, 0.5, -0.2, 0.5, 1, 0.3, -0.2, 0.3, 1);

  it("weights the nodes by the normal density, summing to 1", () => {
    const latent = latentDistribution(productGrid(2, 31));
    latent.set(Float64Array.of(1, 0.4, 0.4, 1));
    expect(latent.logWeights.reduce((sum, value) => sum + Math.exp(value), 0)).toBeCloseTo(1, 12);
    // The grid is symmetric, so the weights are as well, and positive correlation weights the diagonal more.
    const grid = productGrid(2, 31);
    const weight = (z1: number, z2: number): number => {
      const q = Array.from({ length: grid.count }, (_, k) => k).find(
        (k) => Math.abs((grid.nodes[2 * k] ?? 0) - z1) < 1e-9 && Math.abs((grid.nodes[2 * k + 1] ?? 0) - z2) < 1e-9,
      );
      return Math.exp(latent.logWeights[defined(q)] ?? 0);
    };
    expect(weight(1.2, 1.2)).toBeCloseTo(weight(-1.2, -1.2), 14);
    expect(weight(1.2, 1.2)).toBeGreaterThan(weight(1.2, -1.2));
  });

  it("differentiates the expected negative log-likelihood with respect to the correlations", () => {
    const grid = productGrid(3, 7);
    const latent = latentDistribution(grid);
    const nodeCounts = counts(3, 7, Correlations3).map((value, q) => value * (1 + 0.3 * Math.sin(q)));
    latent.set(Float64Array.of(1, 0.2, 0.1, 0.2, 1, -0.3, 0.1, -0.3, 1));
    const gradient = new Float64Array(3);
    latent.gradient(nodeCounts, gradient);
    const objective = (matrix: Float64Array): number => {
      latent.set(matrix);
      return -nodeCounts.reduce((sum, value, q) => sum + value * (latent.logWeights[q] ?? 0), 0);
    };
    const step = 1e-6;
    latent.pairs.forEach(([i, j], p) => {
      const at = (shift: number): number => {
        const matrix = Float64Array.of(1, 0.2, 0.1, 0.2, 1, -0.3, 0.1, -0.3, 1);
        matrix[i * 3 + j] = (matrix[i * 3 + j] ?? 0) + shift;
        matrix[j * 3 + i] = (matrix[j * 3 + i] ?? 0) + shift;
        return objective(matrix);
      };
      expectClose(
        gradient[p] ?? Number.NaN,
        (at(step) - at(-step)) / (2 * step),
        1e-5 * Math.max(1, Math.abs(gradient[p] ?? 0)),
        `pair ${p}`,
      );
    });
  });

  it("recovers the correlations that generated the node counts", () => {
    for (const [dimensions, points, matrix] of [
      [2, 15, Float64Array.of(1, 0.7, 0.7, 1)],
      [3, 9, Correlations3],
    ] as const) {
      const latent = latentDistribution(productGrid(dimensions, points));
      latent.maximize(counts(dimensions, points, matrix));
      matrix.forEach((value, k) => {
        expectClose(latent.matrix[k] ?? Number.NaN, value, 1e-7, `entry ${k} of ${dimensions} dimensions`);
      });
    }
  });

  it("keeps a unit diagonal and stays positive definite when the counts ask for more correlation than there can be", () => {
    const grid = productGrid(2, 15);
    const latent = latentDistribution(grid);
    // All persons on the diagonal: the maximum likelihood correlation is 1, which the matrix cannot reach.
    const nodeCounts = Float64Array.from({ length: grid.count }, (_, q) =>
      Math.abs((grid.nodes[2 * q] ?? 0) - (grid.nodes[2 * q + 1] ?? 0)) < 1e-9 ? 10 : 0,
    );
    latent.maximize(nodeCounts);
    expect(latent.matrix[0]).toBe(1);
    expect(latent.matrix[3]).toBe(1);
    expect(latent.matrix[1] ?? 0).toBeGreaterThan(0.99);
    expect(latent.matrix[1] ?? 2).toBeLessThan(1);
  });

  it("rejects a matrix that is not positive definite and keeps the weights", () => {
    const latent = latentDistribution(productGrid(2, 9));
    const before = Float64Array.from(latent.logWeights);
    expect(latent.set(Float64Array.of(1, 1, 1, 1))).toBe(false);
    expect(latent.logWeights).toEqual(before);
    expect(latent.matrix).toEqual(Float64Array.of(1, 0, 0, 1));
  });

  it("limits the nodes of a grid", () => {
    expect(() => productGrid(5, 20)).toThrow("A grid of 20 points in 5 dimensions has more than 1000000 nodes");
  });
});

describe("calibration", () => {
  it("matches mirt for a confirmatory model of two correlated dimensions", () => {
    const fit = calibrateFixture(m2pl, "hessian", { tolerance: 1e-9 });
    expectCalibration(fit, m2pl, { parameters: 1e-5, errors: 1e-4, likelihood: 1e-6 }, "M2PL");
    expect(fit.converged).toBe(true);
    expect(fit.dimensions).toEqual(["F1", "F2"]);
  });

  it("matches mirt's cross-product standard errors", () => {
    const fit = calibrateFixture(m2pl, "cross-product", { tolerance: 1e-9 });
    const expected = m2pl.crossProduct;
    m2pl.items.forEach((item, i) => {
      const errors = defined(expected.standardErrors[i]);
      for (const [name, value] of Object.entries(errors)) {
        const actual = defined(fit.items[i]?.standardErrors)[name] ?? Number.NaN;
        expectClose(actual / (value as number), 1, 1e-3, `${item.id} ${name}`);
      }
    });
    expectClose(
      (fit.correlationStandardErrors?.[0]?.[1] ?? Number.NaN) / defined(expected.correlationStandardErrors[0]?.[1]),
      1,
      1e-3,
      "correlation",
    );
  });

  it("matches mirt with missing responses", () => {
    const fit = calibrateFixture(missing, "hessian", { tolerance: 1e-9 });
    expectCalibration(fit, missing, { parameters: 1e-5, errors: 1e-4, likelihood: 1e-6 }, "missing");
  });

  it("matches mirt for the M3PL with normal priors on logit c", () => {
    const { mean, sd } = m3pl.priors.c;
    const fit = calibrateFixture(m3pl, "hessian", { priors: { c: logitNormalPrior(mean, sd) }, tolerance: 1e-7 });
    expectCalibration(fit, m3pl, { parameters: 2e-4, errors: 1e-3, likelihood: 1e-4 }, "M3PL");
  });

  it("matches mirt for the graded model on two dimensions", () => {
    const fit = calibrateFixture(graded, "hessian", { tolerance: 1e-7 });
    expectCalibration(fit, graded, { parameters: 2e-4, errors: 1e-3, likelihood: 1e-4 }, "graded");
  });

  it("matches mirt for the graded model on three dimensions", () => {
    const fit = calibrateFixture(graded3, "none", { tolerance: 1e-8 });
    expectCalibration(fit, graded3, { parameters: 2e-4, errors: 1e-3, likelihood: 1e-4 }, "graded, three dimensions");
    expect(fit.correlations).toHaveLength(3);
    expect(fit.correlationStandardErrors).toBeNull();
    expect(fit.items.every(({ standardErrors }) => standardErrors === null)).toBe(true);
  });

  describe("exploratory models", () => {
    /** Returns the sign (1 or -1) that aligns each dimension of the fit with the reference slopes. */
    function signs(fit: MultidimensionalCalibration<Parameters>, slopes: readonly (readonly number[])[]): number[] {
      return [0, 1].map((d) =>
        Math.sign(
          fit.items.reduce((sum, { parameters }, i) => sum + defined(parameters.a[d]) * (slopes[i]?.[d] ?? 0), 0),
        ),
      );
    }

    for (const [name, fixture] of Object.entries(exploratory)) {
      it(`reaches the maximum of the likelihood that mirt's EM stops short of: ${name}`, () => {
        const fit = calibrateMultidimensional(
          buildResponseMatrix(fixture.responses, { itemIds: fixture.itemIds }),
          multidimensionalTwoParameterLogistic,
          {
            dimensions: 2,
            points: fixture.points,
            tolerance: 1e-8,
            maxIterations: 20000,
          },
        );
        const { baseMaximum } = fixture;
        expect(fit.converged).toBe(true);
        // The likelihood is not lower than mirt's, and equals the maximum found in base R.
        expect(fit.logLikelihood).toBeGreaterThanOrEqual(fixture.logLikelihood);
        expectClose(fit.logLikelihood, baseMaximum.logLikelihood, 1e-6, "log-likelihood");
        const sign = signs(fit, baseMaximum.slopes);
        fit.items.forEach(({ parameters }, i) => {
          sign.forEach((s, d) => {
            expectClose(
              s * defined(parameters.a[d]),
              defined(baseMaximum.slopes[i]?.[d]),
              2e-4,
              `item ${i + 1} a${d + 1}`,
            );
          });
          expectClose(
            (parameters as MultidimensionalLogisticParameters).intercept,
            defined(baseMaximum.intercepts[i]),
            2e-4,
            `item ${i + 1} intercept`,
          );
        });
        // The first item loads on the first dimension only, which fixes the rotation.
        expect(fit.items[0]?.parameters.a[1]).toBe(0);
        expect(fit.dimensions).toEqual(["F1", "F2"]);
        expect(fit.correlations).toEqual([
          [1, 0],
          [0, 1],
        ]);
        expect(fit.correlationStandardErrors).toBeNull();
      }, 300_000);
    }
  });

  it("standardizes the loadings of an item on several dimensions with the correlations of the dimensions", () => {
    // mirt's summary leaves the correlations out of the norm, which differs only for items that load on several
    // dimensions: the variance of a · θ is aᵀ Φ a.
    const loadings: Record<string, string[]> = { ...m2pl.loadings };
    loadings[defined(m2pl.itemIds[5])] = ["F1", "F2"];
    const fit = calibrateMultidimensional(
      buildResponseMatrix(m2pl.responses.slice(0, 400), { itemIds: [...m2pl.itemIds] }),
      multidimensionalTwoParameterLogistic,
      {
        loadings,
        points: 15,
        tolerance: 1e-6,
      },
    );
    const phi = fit.correlations;
    const quadratic = (x: readonly number[]): number =>
      x.reduce((sum, xi, p) => sum + xi * x.reduce((inner, xj, q) => inner + (phi[p]?.[q] ?? 0) * xj, 0), 0);
    expect(defined(phi[0]?.[1])).not.toBe(0);
    fit.items.forEach(({ parameters }, i) => {
      const a = parameters.a.map((value) => value / 1.702);
      const norm = Math.sqrt(1 + quadratic(a));
      const row = defined(fit.loadings[i]);
      a.forEach((value, d) => {
        expectClose(defined(row[d]), value / norm, 1e-14, `loading ${i + 1},${d + 1}`);
      });
      expectClose(defined(fit.communalities[i]), quadratic(row), 1e-14, `communality ${i + 1}`);
      expect(fit.communalities[i]).toBeGreaterThan(0);
      expect(fit.communalities[i]).toBeLessThan(1);
    });
    expect(defined(fit.loadings[5]).every((value) => value !== 0)).toBe(true);
  });

  it("reports the log-likelihood, AIC and BIC from the number of estimated parameters", () => {
    const fit = calibrateFixture(m2pl, "none", { tolerance: 1e-6 });
    // 12 slopes, 12 intercepts and one correlation; 1,500 persons.
    expectClose(fit.aic, -2 * fit.logLikelihood + 2 * 25, 1e-9);
    expectClose(fit.bic, -2 * fit.logLikelihood + 25 * Math.log(1500), 1e-9);
  });

  it("fixes the correlations at 0 when they are not estimated", () => {
    const free = calibrateFixture(m2pl, "none", { tolerance: 1e-6 });
    const fit = calibrateMultidimensional(
      buildResponseMatrix(m2pl.responses, { itemIds: [...m2pl.itemIds] }),
      multidimensionalTwoParameterLogistic,
      {
        loadings: m2pl.loadings,
        correlated: false,
        points: 31,
        tolerance: 1e-6,
      },
    );
    expect(fit.correlations).toEqual([
      [1, 0],
      [0, 1],
    ]);
    expect(fit.correlationStandardErrors).toBeNull();
    expect(fit.logLikelihood).toBeLessThan(free.logLikelihood);
    expectClose(fit.aic, -2 * fit.logLikelihood + 2 * 24, 1e-9);
  });

  it("estimates one dimension as the two-parameter logistic model does, in the slope-intercept form", () => {
    const columns = m2pl.itemIds.slice(0, 6);
    const data = buildResponseMatrix(
      m2pl.responses.map((row) => row.slice(0, 6)),
      { itemIds: [...columns] },
    );
    const one = calibrateMultidimensional(data, multidimensionalTwoParameterLogistic, {
      dimensions: 1,
      standardErrors: "hessian",
      tolerance: 1e-10,
    });
    const reference = calibrate(data, twoParameterLogistic, { standardErrors: "hessian", tolerance: 1e-10 });
    expect(one.dimensions).toEqual(["F1"]);
    expect(one.correlations).toEqual([[1]]);
    expectClose(one.logLikelihood, reference.logLikelihood, 1e-8, "log-likelihood");
    reference.items.forEach(({ parameters, standardErrors }, i) => {
      const found = defined(one.items[i]);
      const { a, b } = parameters;
      expectClose(defined(found.parameters.a[0]), a, 1e-6, `a of item ${i + 1}`);
      expectClose(
        (found.parameters as MultidimensionalLogisticParameters).intercept,
        -a * b,
        1e-6,
        `intercept of item ${i + 1}`,
      );
      // The slope's standard error does not depend on the parameterization of the difficulty.
      expectClose(
        defined(found.standardErrors?.a1) / defined(standardErrors?.a),
        1,
        1e-4,
        `standard error of a ${i + 1}`,
      );
    });
  });

  it("gives the same estimates for sparse records as for a matrix with missing responses", () => {
    const records = missing.responses.flatMap((row, person) =>
      row.flatMap((response, item) =>
        response === null ? [] : [{ person: `p${person}`, item: defined(missing.itemIds[item]), response }],
      ),
    );
    const sparse = calibrateMultidimensional(buildResponseRecords(records), multidimensionalTwoParameterLogistic, {
      loadings: missing.loadings,
      points: 15,
      tolerance: 1e-8,
    });
    const dense = calibrateMultidimensional(
      buildResponseMatrix(missing.responses, { itemIds: [...missing.itemIds] }),
      multidimensionalTwoParameterLogistic,
      {
        loadings: missing.loadings,
        points: 15,
        tolerance: 1e-8,
      },
    );
    expectClose(sparse.logLikelihood, dense.logLikelihood, 1e-8);
    dense.items.forEach((item, i) => {
      expectClose(defined(sparse.items[i]?.parameters.a[0]), defined(item.parameters.a[0]), 1e-8);
    });
  });

  it("ignores persons without responses", () => {
    const rows = [...missing.responses.slice(0, 300)];
    const base = calibrateMultidimensional(
      buildResponseMatrix(rows, { itemIds: [...missing.itemIds] }),
      multidimensionalTwoParameterLogistic,
      {
        loadings: missing.loadings,
        points: 15,
        tolerance: 1e-9,
      },
    );
    const padded = calibrateMultidimensional(
      buildResponseMatrix([...rows, new Array<number | null>(12).fill(null)], { itemIds: [...missing.itemIds] }),
      multidimensionalTwoParameterLogistic,
      { loadings: missing.loadings, points: 15, tolerance: 1e-9 },
    );
    expect(padded.logLikelihood).toBe(base.logLikelihood);
    expect(padded.bic).toBe(base.bic);
  });

  it("estimates posterior modes with priors", () => {
    const columns = [...m2pl.itemIds];
    const data = buildResponseMatrix(m2pl.responses.slice(0, 200), { itemIds: columns });
    const options = {
      loadings: m2pl.loadings,
      points: 15,
      tolerance: 1e-8,
    };
    const plain = calibrateMultidimensional(data, multidimensionalTwoParameterLogistic, options);
    const prior = calibrateMultidimensional(data, multidimensionalTwoParameterLogistic, {
      ...options,
      priors: { a: normalPrior(0.5, 0.2), intercept: normalPrior(0, 0.3) },
    });
    const slope = (fit: MultidimensionalCalibration<Parameters>): number =>
      fit.items.reduce((sum, { parameters }, i) => sum + defined(parameters.a[i < 6 ? 0 : 1]), 0) / 12;
    const intercept = (fit: MultidimensionalCalibration<Parameters>): number =>
      fit.items.reduce(
        (sum, { parameters }) => sum + Math.abs((parameters as MultidimensionalLogisticParameters).intercept),
        0,
      ) / 12;
    expect(slope(prior)).toBeLessThan(slope(plain));
    expect(intercept(prior)).toBeLessThan(intercept(plain));
    const lognormal = calibrateMultidimensional(data, multidimensionalTwoParameterLogistic, {
      ...options,
      priors: { a: logNormalPrior(0, 0.3) },
    });
    expect(lognormal.items.every(({ parameters }) => (parameters.a[0] ?? 0) >= 0)).toBe(true);
  });

  it("starts exploratory models from the principal components and uses the dimensions of each item once", () => {
    const fit = calibrateMultidimensional(
      buildResponseMatrix(m2pl.responses.slice(0, 300), { itemIds: [...m2pl.itemIds] }),
      multidimensionalTwoParameterLogistic,
      {
        dimensions: 2,
        points: 11,
        maxIterations: 1,
      },
    );
    expect(fit.iterations).toBe(1);
    expect(fit.converged).toBe(false);
    expect(fit.items[0]?.parameters.a[1]).toBe(0);
    expect(fit.items[1]?.parameters.a[1]).not.toBe(0);
  });

  it("correlates the items of sparse data over their common respondents, as the pairwise correlations of a matrix", () => {
    const cell = fc.oneof(fc.constant(null), fc.integer({ min: 0, max: 3 }));
    const rows = fc
      .integer({ min: 2, max: 7 })
      .chain((items) =>
        fc.array(fc.array(cell, { minLength: items, maxLength: items }), { minLength: 1, maxLength: 30 }),
      );
    fc.assert(
      fc.property(rows, (responses) => {
        const columns = Array.from(defined(responses[0]).keys(), (j) =>
          Float64Array.from(responses, (row) => row[j] ?? Number.NaN),
        );
        const dense = pearsonMatrix(columns).map((value) => (Number.isNaN(value) ? 0 : value));
        expect(itemCorrelations(compressResponses(buildResponseMatrix(responses)))).toEqual(dense);
      }),
    );
  });

  it("names the dimensions of a confirmatory model in order of first appearance among the items", () => {
    const data = buildResponseMatrix(m2pl.responses.slice(0, 150), { itemIds: [...m2pl.itemIds] });
    const loadings: Record<string, string[]> = {};
    m2pl.itemIds.forEach((id, i) => {
      loadings[id] = i < 6 ? ["verbal"] : ["quantitative"];
    });
    loadings[defined(m2pl.itemIds[5])] = ["verbal", "quantitative"];
    const fit = calibrateMultidimensional(data, multidimensionalTwoParameterLogistic, {
      loadings,
      points: 11,
      tolerance: 1e-4,
    });
    expect(fit.dimensions).toEqual(["verbal", "quantitative"]);
    expect(fit.items[5]?.parameters.a.every((slope) => slope !== 0)).toBe(true);
    expect(fit.items[0]?.parameters.a[1]).toBe(0);
    expect(fit.items[6]?.parameters.a[0]).toBe(0);
  });

  it("reports standard errors only for the estimated parameters", () => {
    const fit = calibrateFixture(m2pl, "cross-product", { tolerance: 1e-6 });
    expect(Object.keys(defined(fit.items[0]?.standardErrors)).sort()).toEqual(["a1", "intercept"]);
    expect(Object.keys(defined(fit.items[6]?.standardErrors)).sort()).toEqual(["a2", "intercept"]);
    const gradedFit = calibrateMultidimensional(
      buildResponseMatrix(graded.responses.slice(0, 300), { itemIds: [...graded.itemIds] }),
      multidimensionalGradedResponse,
      {
        loadings: graded.loadings,
        points: 11,
        standardErrors: "cross-product",
        tolerance: 1e-5,
      },
    );
    expect(Object.keys(defined(gradedFit.items[0]?.standardErrors))).toEqual([
      "a1",
      "intercept1",
      "intercept2",
      "intercept3",
      "intercept4",
      "intercept5",
    ]);
  });

  it("stops at the largest number of iterations and says so", () => {
    const short = calibrateMultidimensional(
      buildResponseMatrix(m2pl.responses.slice(0, 200), { itemIds: [...m2pl.itemIds] }),
      multidimensionalTwoParameterLogistic,
      {
        loadings: m2pl.loadings,
        points: 11,
        maxIterations: 3,
      },
    );
    expect(short.converged).toBe(false);
    expect(short.iterations).toBe(3);
  });
});

describe("fixed parameters", () => {
  const matrix = buildResponseMatrix(held.responses, { itemIds: held.itemIds });
  const optionsModel = held.models.map(modelOf);
  const options = {
    loadings: held.loadings,
    fixedParameters: held.fixedParameters,
    points: held.points,
    tolerance: 1e-9,
    maxIterations: 5000,
  };
  const hessian = calibrateMultidimensional(matrix, optionsModel, { ...options, standardErrors: "hessian" });

  it("match mirt with est = FALSE on a shared 3PL item's asymptote and a 2PL item's slope", () => {
    expectCalibration(hessian, held, { parameters: 1e-5, errors: 1e-5, likelihood: 1e-8 }, "held");
    expect(hessian.converged).toBe(true);
    expect(hessian.dimensions).toEqual(["F1", "F2"]);
  });

  it("match mirt's cross-product standard errors", () => {
    const fit = calibrateMultidimensional(matrix, optionsModel, { ...options, standardErrors: "cross-product" });
    held.crossProduct.standardErrors.forEach((errors, i) => {
      const found = defined(fit.items[i]?.standardErrors);
      expect(Object.keys(found)).toEqual(Object.keys(errors));
      for (const [name, value] of Object.entries(errors)) {
        expectClose(defined(found[name]) / value, 1, 1e-5, `${held.itemIds[i]} ${name}`);
      }
    });
    expectClose(
      defined(fit.correlationStandardErrors?.[0]?.[1]) / defined(held.crossProduct.correlationStandardErrors[0]?.[1]),
      1,
      1e-5,
      "correlation",
    );
  });

  it("keep each held value exactly, list it in no standard errors, and count it in no criterion", () => {
    const parameters = (id: string): MultidimensionalLogisticParameters =>
      defined(hessian.items.find((item) => item.id === id)).parameters as MultidimensionalLogisticParameters;
    expect(parameters("q01").c).toBe(0.25);
    expect(parameters("q02").c).toBe(0.2);
    expect(parameters("q06").c).toBe(0.25);
    expect(parameters("q07").c).toBe(0.25);
    expect(parameters("q05").a[0]).toBe(1);
    expect(parameters("q06").a.every((slope) => slope > 0.3)).toBe(true);
    held.items.forEach(({ standardErrors }, i) => {
      expect(Object.keys(defined(hessian.items[i]?.standardErrors))).toEqual(Object.keys(standardErrors));
    });
    expect(hessian.aic).toBeCloseTo(-2 * hessian.logLikelihood + 2 * held.estimated, 9);
    expect(hessian.bic).toBeCloseTo(-2 * hessian.logLikelihood + held.estimated * Math.log(held.responses.length), 9);
  });

  it("give the free calibration when a graded item's intercept is held at its estimate", () => {
    const data = buildResponseMatrix(graded.responses.slice(0, 300), { itemIds: [...graded.itemIds] });
    const base = {
      loadings: graded.loadings,
      points: 11,
      tolerance: 1e-7,
      maxIterations: 2000,
    };
    const free = calibrateMultidimensional(data, multidimensionalGradedResponse, base);
    const [first, second] = free.items;
    const intercept = (defined(second).parameters as MultidimensionalGradedParameters).intercepts[1];
    const fit = calibrateMultidimensional(data, multidimensionalGradedResponse, {
      ...base,
      fixedParameters: { [defined(second).id]: { intercept2: defined(intercept) } },
    });
    expect((defined(fit.items[1]).parameters as MultidimensionalGradedParameters).intercepts[1]).toBe(intercept);
    free.items.forEach(({ id, parameters }, i) => {
      defined(fit.items[i]).parameters.a.forEach((slope, d) => {
        expectClose(slope, defined(parameters.a[d]), 1e-4, `${id} a${d + 1}`);
      });
    });
    expect(first?.id).toBe(graded.itemIds[0]);
    expectClose(fit.logLikelihood, free.logLikelihood, 1e-6, "log-likelihood");
    expectClose(fit.aic, free.aic - 2, 1e-5, "AIC");
    expect(() =>
      calibrateMultidimensional(data, multidimensionalGradedResponse, {
        ...base,
        fixedParameters: { [defined(second).id]: { intercept1: 0, intercept2: 1 } },
      }),
    ).toThrow(/with its held parameters: An MGRM item needs strictly decreasing intercepts/);
  });

  it("hold with anchors and groups too, which set the scale as they do without them", () => {
    const id = defined(hessian.items[0]).id;
    const anchors = Object.fromEntries(
      [hessian.items[2], hessian.items[8]].map((item) => [defined(item).id, defined(item).parameters]),
    );
    const groups = held.responses.map((_, p) => (p % 2 === 0 ? "reference" : "focal"));
    const fit = calibrateMultidimensional(matrix, optionsModel, {
      ...options,
      tolerance: 1e-5,
      fixed: anchors,
      groups,
    });
    expect((defined(fit.items[0]).parameters as MultidimensionalLogisticParameters).c).toBe(0.25);
    expect(fit.items[0]?.id).toBe(id);
    expect(fit.groups?.map(({ group }) => group)).toEqual(["reference", "focal"]);
    fit.groups?.forEach(({ means }) => {
      for (const value of means) expect(Math.abs(value)).toBeLessThan(0.3);
    });
  });

  it("hold with MH-RM, near the solution of EM within its Monte Carlo error", () => {
    const fit = calibrateMultidimensional(matrix, optionsModel, {
      loadings: options.loadings,
      fixedParameters: options.fixedParameters,
      method: "mhrm",
      seed: 1,
      standardErrors: "hessian",
    });
    expect(fit.converged).toBe(true);
    held.items.forEach(({ id, parameters, standardErrors }, i) => {
      const found = defined(fit.items[i]);
      const logistic = found.parameters as MultidimensionalLogisticParameters;
      expect(logistic.c).toBe(held.fixedParameters[id]?.c ?? 0);
      parameters.a.forEach((slope, d) => {
        expect(Math.abs(defined(logistic.a[d]) - slope), `${id} a${d + 1}`).toBeLessThan(0.15);
      });
      expect(Math.abs(logistic.intercept - parameters.intercept), `${id} intercept`).toBeLessThan(0.15);
      expect(Object.keys(defined(found.standardErrors))).toEqual(Object.keys(standardErrors));
    });
    expect(fit.aic).toBeCloseTo(-2 * fit.logLikelihood + 2 * held.estimated, 9);
  }, 120_000);

  it("change nothing when empty", () => {
    const small = buildResponseMatrix(m2pl.responses.slice(0, 200), { itemIds: [...m2pl.itemIds] });
    const plain = {
      loadings: m2pl.loadings,
      points: 11,
    };
    const fit = calibrateMultidimensional(small, multidimensionalTwoParameterLogistic, plain);
    expect(
      calibrateMultidimensional(small, multidimensionalTwoParameterLogistic, { ...plain, fixedParameters: {} }),
    ).toEqual(fit);
    expect(
      calibrateMultidimensional(small, multidimensionalTwoParameterLogistic, {
        ...plain,
        fixedParameters: { q01: {} },
      }),
    ).toEqual(fit);
  });

  it("reject unknown and fixed items, parameters the item lacks, every parameter, and invalid values", () => {
    const small = buildResponseMatrix(held.responses.slice(0, 40), { itemIds: held.itemIds });
    const base = { loadings: held.loadings, points: 5, maxIterations: 1 };
    const rejected = (
      value: object | string,
      extra: object = {},
      model: MultidimensionalModel<Parameters> | readonly MultidimensionalModel<Parameters>[] = optionsModel,
    ): string | undefined => {
      try {
        const fixedParameters = typeof value === "string" ? JSON.parse(value) : value;
        calibrateMultidimensional(small, model, { ...base, ...extra, fixedParameters });
      } catch (error) {
        return error instanceof DataError ? error.message : undefined;
      }
      return undefined;
    };
    expect(rejected({ zz: { c: 0.25 } })).toBe(
      'The option fixedParameters holds parameters of "zz", which is not an item id.',
    );
    const anchor = { q03: defined(held.items[2]).parameters };
    expect(rejected({ q03: { a1: 1 } }, { fixed: anchor })).toBe(
      'The option fixedParameters holds parameters of item "q03", which the option fixed holds whole.',
    );
    expect(rejected({ q03: { c: 0.25 } })).toBe(
      'The option fixedParameters holds "c" of item "q03", whose M2PL model has a1, intercept.',
    );
    expect(rejected({ q01: { g: 0.25 } })).toBe(
      'The option fixedParameters holds "g" of item "q01", whose M3PL model has a1, intercept, c.',
    );
    expect(rejected({ q01: { a2: 1 } })).toBe(
      'The option fixedParameters holds "a2" of item "q01", a slope that the pattern of loadings fixes at 0.',
    );
    expect(rejected({ q01: { a2: 0 } })).toBe(
      'The option fixedParameters holds "a2" of item "q01", a slope that the pattern of loadings fixes at 0.',
    );
    expect(
      rejected({ q01: { a2: 1 } }, { loadings: undefined, dimensions: 2 }, multidimensionalThreeParameterLogistic),
    ).toBe('The option fixedParameters holds "a2" of item "q01", a slope that the pattern of loadings fixes at 0.');
    expect(rejected({ q03: { a1: 1, intercept: 0 } })).toBe(
      'The option fixedParameters holds every parameter of item "q03"; give its parameters in the option fixed.',
    );
    expect(rejected({ q01: { c: 1 } })).toMatch(
      /^Item "q01" with its held parameters: An M3PL item needs finite slopes a, at least one, a finite intercept and 0 ≤ c < d ≤ 1, found .* c = 1, d = 1\.$/,
    );
    const everyAsymptote = Object.fromEntries(
      held.models.flatMap((name, i) => (name === "m3pl" ? [[held.itemIds[i], { c: 0.25 }]] : [])),
    );
    expect(rejected(everyAsymptote, { priors: { c: betaPrior(5, 17) } })).toBe(
      'A prior is given for "c", which no estimated item has.',
    );
    expect(rejected("null")).toBe(
      "The option fixedParameters must be an object of values by item id and parameter name, found null.",
    );
    expect(rejected('{ "q01": [] }')).toBe(
      'The held parameters of item "q01" must be an object of values by parameter name, found an array.',
    );
    expect(rejected('{ "q01": { "c": "0.25" } }')).toBe(
      'The held value of "c" of item "q01" must be a finite number, found "0.25".',
    );
    expect(rejected('{ "constructor": {} }')).toBe(
      'The option fixedParameters holds parameters of "constructor", which is not an item id.',
    );
  });
});

describe("MH-RM", () => {
  /** The calibration of a fixture by MH-RM. */
  function byMhrm(
    fixture: Pick<Fixture, "model" | "itemIds" | "responses" | "loadings">,
    extra: {
      readonly seed: number;
      readonly priors?: Readonly<Record<string, ParameterPrior>>;
      readonly standardErrors?: "none" | "hessian";
    },
  ): MultidimensionalCalibration<Parameters> {
    return calibrateMultidimensional(
      buildResponseMatrix(fixture.responses, { itemIds: [...fixture.itemIds] }),
      modelOf(fixture.model),
      {
        loadings: fixture.loadings,
        method: "mhrm",
        ...extra,
      },
    );
  }

  /** Returns the largest absolute difference of the slopes and intercepts from those of the fixture. */
  function largestDifference(fit: MultidimensionalCalibration<Parameters>, fixture: Pick<Fixture, "items">): number {
    let largest = 0;
    fixture.items.forEach(({ parameters: target }, i) => {
      const found = defined(fit.items[i]).parameters;
      target.a.forEach((slope, d) => {
        if (slope !== 0) largest = Math.max(largest, Math.abs(defined(found.a[d]) - slope));
      });
      if ("intercepts" in found) {
        (target.intercepts ?? []).forEach((value, k) => {
          largest = Math.max(largest, Math.abs(defined(found.intercepts[k]) - value));
        });
      } else {
        largest = Math.max(largest, Math.abs(found.intercept - defined(target.intercept)));
      }
    });
    return largest;
  }

  // The estimates of MH-RM carry Monte Carlo error. Its size was measured over seeds 1 to 6 on the case of two correlated
  // dimensions, at the default settings: the largest difference from the solution of EM was 0.057, 0.34 to 0.41 of the
  // largest standard error, and the standard errors were within 7%. The tolerances are about twice as large.
  it("agrees with the EM solution on two correlated dimensions within its Monte Carlo error", () => {
    const fit = byMhrm(m2pl, { seed: 1, standardErrors: "hessian" });
    expect(fit.converged).toBe(true);
    expect(fit.dimensions).toEqual(["F1", "F2"]);
    expect(largestDifference(fit, m2pl)).toBeLessThan(0.12);
    expectClose(defined(fit.correlations[0]?.[1]), defined(m2pl.correlations[0]?.[1]), 0.05, "correlation");
    m2pl.items.forEach((expected, i) => {
      for (const [name, error] of Object.entries(expected.standardErrors ?? {})) {
        const actual = defined(fit.items[i]?.standardErrors)[name] ?? Number.NaN;
        expectClose(actual / defined(error), 1, 0.15, `standard error of ${name} of ${expected.id}`);
      }
    });
    expectClose(
      defined(fit.correlationStandardErrors?.[0]?.[1]) / defined(m2pl.correlationStandardErrors?.[0]?.[1]),
      1,
      0.15,
      "standard error of the correlation",
    );
    // The log-likelihood is estimated by importance sampling, within about one unit of EM's here.
    expectClose(fit.logLikelihood, m2pl.logLikelihood, 4, "log-likelihood");
    expectClose(fit.aic, -2 * fit.logLikelihood + 2 * 25, 1e-9, "AIC");
  }, 120_000);

  it("agrees with the EM solution with missing responses and with the M3PL and its priors", () => {
    const withMissing = byMhrm(missing, { seed: 1 });
    expect(largestDifference(withMissing, missing)).toBeLessThan(0.15);
    const { mean, sd } = m3pl.priors.c;
    const guessing = byMhrm(m3pl, { seed: 1, priors: { c: logitNormalPrior(mean, sd) } });
    expect(largestDifference(guessing, m3pl)).toBeLessThan(0.2);
    m3pl.items.forEach(({ parameters: target }, i) => {
      const c = (defined(guessing.items[i]).parameters as MultidimensionalLogisticParameters).c;
      expectClose(c, defined(target.c), 0.05, `c of item ${i + 1}`);
      expect(c).toBeGreaterThan(0);
    });
  }, 120_000);

  // mirt's MH-RM was run with seeds 1 to 4 on the first 800 complete cases of the 25 items of bfi: its estimates differ from
  // their mean over the seeds by up to 0.076, and measurekit's, over seeds 1 to 4, from that mean by up to 0.17, which is
  // about one standard error of the least precise intercepts. The tolerance is 0.3. mirt's standard errors vary 20% from
  // seed to seed (79% at most); measurekit's are compared with their mean over the seeds.
  it("agrees with mirt's MH-RM on five dimensions", () => {
    const fit = byMhrm(mirtMhrm, { seed: 1, standardErrors: "hessian" });
    expect(fit.converged).toBe(true);
    const slopes = (parameters: FixtureParameters): number[] => parameters.a.filter((value) => value !== 0);
    mirtMhrm.itemIds.forEach((_, i) => {
      const found = defined(fit.items[i]).parameters as MultidimensionalGradedParameters;
      const references = mirtMhrm.runs.map((run) => defined(run.items[i]).parameters);
      slopes(found).forEach((slope, k) => {
        const reference =
          references.reduce((sum, parameters) => sum + defined(slopes(parameters)[k]), 0) / references.length;
        expectClose(slope, reference, 0.3, `slope of item ${i + 1}`);
      });
      found.intercepts.forEach((value, k) => {
        const reference =
          references.reduce((sum, parameters) => sum + defined(parameters.intercepts?.[k]), 0) / references.length;
        expectClose(value, reference, 0.3, `intercept ${k + 1} of item ${i + 1}`);
      });
    });
    const meanCorrelation = (i: number, j: number): number =>
      mirtMhrm.runs.reduce((sum, run) => sum + defined(run.correlations[i]?.[j]), 0) / mirtMhrm.runs.length;
    fit.correlations.forEach((row, i) => {
      row.forEach((value, j) => {
        expectClose(value, meanCorrelation(i, j), 0.08, `correlation ${i},${j}`);
      });
    });
    const ratios: number[] = [];
    mirtMhrm.runs[0]?.items.forEach((item, i) => {
      for (const name of Object.keys(item.standardErrors ?? {})) {
        const reference = mirtMhrm.runs.map((run) => run.items[i]?.standardErrors?.[name]);
        if (!reference.every((value) => typeof value === "number")) continue;
        const mean = reference.reduce((sum, value) => sum + (value ?? 0), 0) / reference.length;
        ratios.push(defined(fit.items[i]?.standardErrors?.[name]) / mean);
      }
    });
    ratios.sort((a, b) => a - b);
    expect(ratios.length).toBeGreaterThan(100);
    expectClose(ratios[Math.floor(ratios.length / 2)] ?? Number.NaN, 1, 0.1, "median ratio of the standard errors");
    expect(ratios[0]).toBeGreaterThan(0.6);
    expect(ratios.at(-1)).toBeLessThan(1.6);
  }, 300_000);

  it("estimates the log-likelihood that EM computes, on three dimensions", () => {
    const fit = byMhrm(graded3, { seed: 1 });
    expect(fit.converged).toBe(true);
    expectClose(fit.logLikelihood, graded3.logLikelihood, 3, "log-likelihood");
    expect(largestDifference(fit, graded3)).toBeLessThan(0.25);
  }, 120_000);

  describe("small data", () => {
    const rows = m2pl.responses.slice(0, 250);
    const data = buildResponseMatrix(rows, { itemIds: [...m2pl.itemIds] });
    const options = { loadings: m2pl.loadings, method: "mhrm" } as const;

    it("gives the same calibration for the same seed and another for another seed", () => {
      // 50 persons and 10 cycles of the last stage: the seed fixes the draws of every stage and of the standard errors.
      const few = buildResponseMatrix(rows.slice(0, 50), { itemIds: [...m2pl.itemIds] });
      const short = { ...options, maxIterations: 10 };
      const first = calibrateMultidimensional(few, multidimensionalTwoParameterLogistic, {
        ...short,
        seed: 5,
        standardErrors: "hessian",
      });
      expect(first.items[0]?.standardErrors).not.toBeNull();
      expect(
        calibrateMultidimensional(few, multidimensionalTwoParameterLogistic, {
          ...short,
          seed: 5,
          standardErrors: "hessian",
        }),
      ).toEqual(first);
      const other = calibrateMultidimensional(few, multidimensionalTwoParameterLogistic, { ...short, seed: 6 });
      expect(other.items[0]?.parameters).not.toEqual(first.items[0]?.parameters);
    });

    it("reports the cycles of its last stage and whether the tolerance was met", () => {
      const short = calibrateMultidimensional(data, multidimensionalTwoParameterLogistic, {
        ...options,
        maxIterations: 3,
      });
      expect(short.converged).toBe(false);
      expect(short.iterations).toBe(3);
      const long = calibrateMultidimensional(data, multidimensionalTwoParameterLogistic, {
        ...options,
        tolerance: 0.05,
      });
      expect(long.converged).toBe(true);
      expect(long.iterations).toBeGreaterThanOrEqual(3);
      expect(long.iterations).toBeLessThan(2000);
    });

    it("reports AIC and BIC from the same number of parameters as EM", () => {
      const fit = calibrateMultidimensional(data, multidimensionalTwoParameterLogistic, {
        ...options,
        maxIterations: 5,
      });
      expectClose(fit.aic, -2 * fit.logLikelihood + 2 * 25, 1e-9);
      expectClose(fit.bic, -2 * fit.logLikelihood + 25 * Math.log(250), 1e-9);
    });

    it("fixes the correlations at 0 when they are not estimated and has none in one dimension", () => {
      const uncorrelated = calibrateMultidimensional(data, multidimensionalTwoParameterLogistic, {
        ...options,
        correlated: false,
        maxIterations: 5,
      });
      expect(uncorrelated.correlations).toEqual([
        [1, 0],
        [0, 1],
      ]);
      expect(uncorrelated.correlationStandardErrors).toBeNull();
      const one = calibrateMultidimensional(
        buildResponseMatrix(
          rows.map((row) => row.slice(0, 6)),
          { itemIds: m2pl.itemIds.slice(0, 6) },
        ),
        multidimensionalTwoParameterLogistic,
        { dimensions: 1, method: "mhrm", standardErrors: "hessian" },
      );
      expect(one.correlations).toEqual([[1]]);
      expect(one.items.every(({ standardErrors }) => standardErrors !== null)).toBe(true);
    });

    it("estimates exploratory models with the first items' later slopes fixed at 0", () => {
      const fit = calibrateMultidimensional(data, multidimensionalTwoParameterLogistic, {
        dimensions: 2,
        method: "mhrm",
        maxIterations: 10,
      });
      expect(fit.items[0]?.parameters.a[1]).toBe(0);
      expect(fit.items[1]?.parameters.a[1]).not.toBe(0);
      expect(fit.correlations).toEqual([
        [1, 0],
        [0, 1],
      ]);
    });

    it("applies priors", () => {
      const plain = calibrateMultidimensional(data, multidimensionalTwoParameterLogistic, { ...options, seed: 2 });
      const shrunk = calibrateMultidimensional(data, multidimensionalTwoParameterLogistic, {
        ...options,
        seed: 2,
        priors: { intercept: normalPrior(0, 0.3) },
      });
      const size = (fit: MultidimensionalCalibration<Parameters>): number =>
        fit.items.reduce(
          (sum, { parameters }) => sum + Math.abs((parameters as MultidimensionalLogisticParameters).intercept),
          0,
        );
      expect(size(shrunk)).toBeLessThan(size(plain));
    });

    it("calibrates sparse records", () => {
      const records = rows.flatMap((row, person) =>
        row.flatMap((response, item) =>
          response === null ? [] : [{ person: `p${person}`, item: defined(m2pl.itemIds[item]), response }],
        ),
      );
      const fit = calibrateMultidimensional(buildResponseRecords(records), multidimensionalTwoParameterLogistic, {
        ...options,
        maxIterations: 5,
      });
      expect(fit.items).toHaveLength(12);
      expect(fit.logLikelihood).toBeLessThan(0);
    });

    it("chooses EM up to three dimensions and MH-RM above", () => {
      // Only EM has a grid: the option points is accepted by the default method up to three dimensions and rejected above.
      expect(() =>
        calibrateMultidimensional(data, multidimensionalTwoParameterLogistic, {
          loadings: m2pl.loadings,
          points: 5,
          maxIterations: 1,
        }),
      ).not.toThrow();
      const four = Object.fromEntries(m2pl.itemIds.map((id, i) => [id, [`D${(i % 4) + 1}`]]));
      expect(() =>
        calibrateMultidimensional(data, multidimensionalTwoParameterLogistic, { loadings: four, points: 5 }),
      ).toThrow('The option points is for EM, which the method "mhrm" does not use.');
      const fit = calibrateMultidimensional(data, multidimensionalTwoParameterLogistic, {
        loadings: four,
        maxIterations: 3,
      });
      expect(fit.dimensions).toEqual(["D1", "D2", "D3", "D4"]);
      expect(fit.correlations).toHaveLength(4);
    });
  });

  it("gives the same calibration for the same seed, whatever the seed", () => {
    const data = buildResponseMatrix(
      m2pl.responses.slice(0, 40).map((row) => row.slice(0, 6)),
      { itemIds: m2pl.itemIds.slice(0, 6) },
    );
    fc.assert(
      fc.property(fc.integer({ min: 0, max: 2 ** 31 }), (seed) => {
        const options = {
          dimensions: 1,
          method: "mhrm",
          seed,
          maxIterations: 2,
        } as const;
        expect(calibrateMultidimensional(data, multidimensionalTwoParameterLogistic, options)).toEqual(
          calibrateMultidimensional(data, multidimensionalTwoParameterLogistic, options),
        );
      }),
      { numRuns: 4 },
    );
  }, 120_000);
});

describe("MH-RM internals", () => {
  const items5 = mirtMhrm.itemIds;
  const responses = compressResponses(buildResponseMatrix(mirtMhrm.responses, { itemIds: [...items5] }));

  it("computes the complete-data score and information of mirt's Deriv", () => {
    const design = designOf({ loadings: mirtMhrm.loadings }, items5);
    const states = setUp(responses, multidimensionalGradedResponse, {}, design, 0);
    const theta = Float64Array.from(mirtMhrm.thetas.flat());
    const data = responsesByItem(responses);
    const scratch = createScratch(states, 5);
    for (const reference of mirtMhrm.completeData) {
      const index = items5.indexOf(reference.item);
      const item = defined(states[index]);
      item.parameters = { a: reference.parameters.slice(0, 5), intercepts: reference.parameters.slice(5) };
      item.vector.set(item.model.toVector(item.parameters));
      // The estimated entries are indices in mirt's vector: the slopes a1 to a5 of the loaded dimensions, then d1 to d5.
      const own = Array.from(item.free);
      const draws = { theta, responses: data, dimensions: 5 };
      const size = own.length;
      const gradient = new Float64Array(size);
      itemMoments(item, index, draws, scratch, gradient);
      const central = new Float64Array(size * size);
      observedItemInformation(item, index, draws, scratch, central, true);
      const forward = new Float64Array(size * size);
      observedItemInformation(item, index, draws, scratch, forward, false);
      own.forEach((entry, l) => {
        const slot = entry;
        expectClose(
          gradient[l] ?? Number.NaN,
          defined(reference.gradient[slot]),
          1e-10 * Math.max(1, Math.abs(defined(reference.gradient[slot]))),
          `${reference.item} score ${l}`,
        );
        own.forEach((other, m) => {
          const expected = -defined(reference.hessian[slot]?.[other]);
          expectClose(
            central[l * size + m] ?? Number.NaN,
            expected,
            1e-8 * Math.max(1, Math.abs(expected)),
            `${reference.item} information ${l},${m}`,
          );
          expectClose(
            forward[l * size + m] ?? Number.NaN,
            expected,
            2e-4 * Math.max(1, Math.abs(expected)),
            `${reference.item} forward information ${l},${m}`,
          );
        });
      });
    }
  });

  it("lists the responses by item and by person, each once, with a person's in item order", () => {
    fc.assert(
      fc.property(
        fc.array(fc.array(fc.option(fc.integer({ min: 0, max: 1 }), { nil: null }), { minLength: 4, maxLength: 4 }), {
          minLength: 1,
          maxLength: 12,
        }),
        (rows) => {
          const compressed = compressResponses(buildResponseMatrix(rows));
          const data = responsesByItem(compressed);
          const people = responsesByPerson(data);
          const answered = rows.flat().filter((value) => value !== null).length;
          expect(data.respondents).toHaveLength(answered);
          expect(people.responses).toHaveLength(answered);
          expect(new Set(people.responses).size).toBe(answered);
          for (let p = 0; p < data.count; p++) {
            const list = Array.from(people.responses.subarray(people.starts[p], people.starts[p + 1]));
            const itemsOfPerson = list.map((r) => people.itemOf[r] ?? -1);
            expect(itemsOfPerson).toEqual([...itemsOfPerson].sort((a, b) => a - b));
            expect(list.every((r) => data.respondents[r] === p)).toBe(true);
          }
        },
      ),
    );
  });

  it("draws a person's theta from the exact posterior", () => {
    const itemIds = m2pl.itemIds.slice(0, 6);
    const loadings = Object.fromEntries(itemIds.map((id, i) => [id, [i < 3 ? "F1" : "F2"]]));
    const pattern = [1, 0, 1, 1, 0, 1];
    // Starting values need every category answered, which one person's responses do not.
    const calibrated = compressResponses(
      buildResponseMatrix(
        m2pl.responses.slice(0, 250).map((row) => row.slice(0, 6)),
        { itemIds },
      ),
    );
    const states = setUp(calibrated, multidimensionalTwoParameterLogistic, {}, designOf({ loadings }, itemIds), 0);
    const parameters = [
      { a: [1.2, 0], intercept: 0.3 },
      { a: [0.9, 0], intercept: -0.4 },
      { a: [1.5, 0], intercept: 0.1 },
      { a: [0, 1.1], intercept: 0.5 },
      { a: [0, 1.4], intercept: -0.2 },
      { a: [0, 0.8], intercept: 0.2 },
    ];
    states.forEach((item, i) => {
      item.parameters = { ...defined(parameters[i]), c: 0, d: 1 };
    });
    const correlations = createCorrelations(2);
    correlations.set(Float64Array.of(1, 0.5, 0.5, 1));
    const chain = createChain(
      states,
      responsesByItem(compressResponses(buildResponseMatrix([pattern], { itemIds }))),
      correlations,
    );
    const random = createRandom(11);
    chain.refresh();
    for (let k = 0; k < 200; k++) chain.sweep(random, 0.5);
    const sums = new Float64Array(2);
    const products = new Float64Array(4);
    const sweeps = 40_000;
    let accepted = 0;
    for (let k = 0; k < sweeps; k++) {
      accepted += chain.sweep(random, 0.5);
      for (let i = 0; i < 2; i++) {
        sums[i] = (sums[i] ?? 0) + (chain.theta[i] ?? 0);
        for (let j = 0; j < 2; j++)
          products[i * 2 + j] = (products[i * 2 + j] ?? 0) + (chain.theta[i] ?? 0) * (chain.theta[j] ?? 0);
      }
    }
    const exact = multidimensionalPosteriorMeanEstimate(
      states.map((item) => ({ model: item.model, parameters: item.parameters })),
      pattern,
      {
        prior: {
          mean: [0, 0],
          covariance: [
            [1, 0.5],
            [0.5, 1],
          ],
        },
        points: 61,
      },
    );
    // The draws are correlated, so the bound is several times the standard error of independent draws.
    for (let i = 0; i < 2; i++) {
      expectClose((sums[i] ?? 0) / sweeps, defined(exact.theta[i]), 0.04, `mean ${i + 1}`);
      const variance = (products[i * 2 + i] ?? 0) / sweeps - ((sums[i] ?? 0) / sweeps) ** 2;
      expectClose(variance, defined(exact.covariance[i]?.[i]), 0.04, `variance ${i + 1}`);
    }
    expect(accepted / sweeps).toBeGreaterThan(0.1);
    expect(accepted / sweeps).toBeLessThan(0.9);
  });

  describe("correlations of the draws", () => {
    const dimensions = 3;
    const matrix = Float64Array.of(1, 0.5, -0.2, 0.5, 1, 0.3, -0.2, 0.3, 1);
    const correlations = createCorrelations(dimensions);
    expect(correlations.set(matrix)).toBe(true);
    const theta = Float64Array.from({ length: 7 * dimensions }, (_, k) => Math.sin(k * 1.7) * 1.3 + Math.cos(k * 0.9));
    const persons = 7;

    /** The log-likelihood of the draws under N(0, Φ) for the matrix with the pair (i, j) shifted by `shift`. */
    function logLikelihood(shifts: number[]): number {
      const shifted = createCorrelations(dimensions);
      const next = Float64Array.from(matrix);
      correlations.pairs.forEach(([i, j], a) => {
        next[i * dimensions + j] = (matrix[i * dimensions + j] ?? 0) + (shifts[a] ?? 0);
        next[j * dimensions + i] = (matrix[j * dimensions + i] ?? 0) + (shifts[a] ?? 0);
      });
      shifted.set(next);
      const second = new Float64Array(dimensions * dimensions);
      secondMoments(theta, persons, dimensions, second);
      let trace = 0;
      for (let i = 0; i < dimensions; i++)
        for (let j = 0; j < dimensions; j++)
          trace += (shifted.precision[i * dimensions + j] ?? 0) * (second[j * dimensions + i] ?? 0);
      return -0.5 * persons * shifted.logDeterminant() - 0.5 * trace;
    }

    it("gives the score and the observed information as derivatives of the log-likelihood", () => {
      const pairs = correlations.pairs.length;
      const second = new Float64Array(dimensions * dimensions);
      secondMoments(theta, persons, dimensions, second);
      const score = new Float64Array(pairs);
      latentScore(correlations, second, persons, score);
      const information = new Float64Array(pairs * pairs);
      latentObservedInformation(correlations, second, persons, information);
      const step = 1e-5;
      const at = (a: number, b: number, sa: number, sb: number): number => {
        const shifts = new Array<number>(pairs).fill(0);
        shifts[a] = (shifts[a] ?? 0) + sa;
        shifts[b] = (shifts[b] ?? 0) + sb;
        return logLikelihood(shifts);
      };
      for (let a = 0; a < pairs; a++) {
        expectClose(defined(score[a]), (at(a, a, step, 0) - at(a, a, -step, 0)) / (2 * step), 1e-6, `score ${a}`);
        for (let b = 0; b < pairs; b++) {
          const second2 =
            a === b
              ? (at(a, b, step, 0) - 2 * logLikelihood(new Array<number>(pairs).fill(0)) + at(a, b, -step, 0)) /
                step ** 2
              : (at(a, b, step, step) - at(a, b, step, -step) - at(a, b, -step, step) + at(a, b, -step, -step)) /
                (4 * step ** 2);
          expectClose(defined(information[a * pairs + b]), -second2, 1e-4, `information ${a},${b}`);
        }
      }
    });

    it("gives the sum over persons of the persons' scores as the score of the draws", () => {
      const pairs = correlations.pairs.length;
      const second = new Float64Array(dimensions * dimensions);
      secondMoments(theta, persons, dimensions, second);
      const total = new Float64Array(pairs);
      latentScore(correlations, second, persons, total);
      const each = new Float64Array(persons * pairs);
      personLatentScores(correlations, theta, persons, each);
      for (let a = 0; a < pairs; a++) {
        let sum = 0;
        for (let p = 0; p < persons; p++) sum += each[p * pairs + a] ?? 0;
        expectClose(sum, defined(total[a]), 1e-12, `pair ${a}`);
      }
    });

    it("rejects a matrix that is not positive definite and keeps the old one", () => {
      const fresh = createCorrelations(2);
      expect(fresh.set(Float64Array.of(1, 1, 1, 1))).toBe(false);
      expect(fresh.matrix).toEqual(Float64Array.of(1, 0, 0, 1));
      expect(fresh.logDeterminant()).toBe(0);
    });
  });
});

/** A calibration of mirt in several groups: the data, the group of each person and mirt's estimates. */
interface GroupsFixture extends Fixture {
  readonly personGroups: readonly string[];
  readonly groupMoments: readonly {
    readonly group: string;
    readonly means: readonly number[];
    readonly covariance: readonly (readonly number[])[];
  }[];
  readonly fixed?: Readonly<Record<string, FixtureParameters>>;
  readonly correlated?: boolean;
}

describe("multiple groups", () => {
  const twoGroups: GroupsFixture = groups.correlated;
  const anchored: GroupsFixture = groups.anchors;
  // The likelihood is flat in the moments of the groups, and mirt's EM stops before its estimates settle: measurekit's,
  // run to a tolerance of 1e-11, stay where they were at 1e-9 and reach a log-likelihood at least as high as mirt's.
  const mirtTolerances = { parameters: 2e-4, errors: 5e-4, likelihood: 5e-6 };

  /** Calibrates a fixture's groups, whatever its model and anchors, as mirt's multipleGroup does. */
  function calibrateGroups(
    fixture: GroupsFixture,
    standardErrors: "none" | "hessian" | "cross-product",
    extra: Partial<MultidimensionalCalibrationOptions<Parameters>> = {},
  ): MultidimensionalCalibration<Parameters> {
    const { fixed, correlated } = fixture;
    return calibrateMultidimensional(
      buildResponseMatrix(fixture.responses, { itemIds: [...fixture.itemIds] }),
      modelOf(fixture.model),
      {
        loadings: fixture.loadings,
        points: fixture.points,
        groups: [...fixture.personGroups],
        ...(fixed === undefined
          ? {}
          : {
              fixed: Object.fromEntries(Object.entries(fixed).map(([id, p]) => [id, parametersOf(fixture.model, p)])),
            }),
        ...(correlated === undefined ? {} : { correlated }),
        standardErrors,
        tolerance: 1e-9,
        maxIterations: 2000,
        ...extra,
      },
    );
  }

  /** Compares the means and covariance of each group with mirt's. */
  function expectGroups(fit: MultidimensionalCalibration<Parameters>, fixture: GroupsFixture, tolerance: number): void {
    expect(fit.groups?.map(({ group }) => group)).toEqual(fixture.groupMoments.map(({ group }) => group));
    fixture.groupMoments.forEach((expected, g) => {
      const found = defined(fit.groups?.[g]);
      expected.means.forEach((value, d) => {
        expectClose(defined(found.means[d]), value, tolerance, `mean ${d + 1} of group ${expected.group}`);
      });
      expected.covariance.forEach((row, i) => {
        row.forEach((value, j) => {
          expectClose(
            defined(found.covariance[i]?.[j]),
            value,
            tolerance,
            `covariance ${i + 1},${j + 1} of group ${expected.group}`,
          );
        });
      });
    });
  }

  it("matches mirt's multipleGroup for two groups of correlated dimensions, with standard errors", () => {
    const fit = calibrateGroups(twoGroups, "hessian");
    expectCalibration(fit, twoGroups, { parameters: 5e-5, errors: 1e-4, likelihood: 1e-6 }, "two groups");
    expectGroups(fit, twoGroups, 1e-4);
    expect(fit.converged).toBe(true);
    // The first group is the reference: means 0 and variances 1.
    expect(fit.groups?.[0]?.means).toEqual([0, 0]);
    expect(fit.groups?.[0]?.covariance.map((row, i) => row[i])).toEqual([1, 1]);
  }, 120_000);

  it(
    "matches mirt when fixed items set the scale and every group's means and covariance are free",
    () => {
      const fit = calibrateGroups(anchored, "hessian");
      expectCalibration(fit, anchored, mirtTolerances, "fixed items");
      expectGroups(fit, anchored, 5e-4);
      // Fixed items keep their parameters and have no standard errors.
      for (const [id, parameters] of Object.entries(defined(anchored.fixed))) {
        const item = defined(fit.items.find((each) => each.id === id));
        expect(item.parameters).toEqual(parametersOf(anchored.model, parameters));
        expect(item.standardErrors).toBeNull();
      }
      expect(defined(fit.groups?.[0]).means.some((mean) => mean !== 0)).toBe(true);
    },
    UNDER_COVERAGE,
  );

  it(
    "matches mirt for three groups of uncorrelated dimensions with items missing by design",
    () => {
      const uncorrelated: GroupsFixture = groups.uncorrelated;
      const fit = calibrateGroups(uncorrelated, "hessian");
      expectCalibration(fit, uncorrelated, mirtTolerances, "three groups");
      expectGroups(fit, uncorrelated, 5e-4);
      expect(fit.correlationStandardErrors).toBeNull();
      // The covariance of every group is diagonal.
      expect(fit.groups?.every(({ covariance }) => covariance[0]?.[1] === 0)).toBe(true);
    },
    UNDER_COVERAGE,
  );

  it("matches mirt for the graded model in two groups", () => {
    const graded: GroupsFixture = groups.graded;
    const fit = calibrateGroups(graded, "none");
    expectCalibration(fit, graded, mirtTolerances, "graded groups");
    expectGroups(fit, graded, 5e-4);
  }, 120_000);

  it(
    "matches mirt for two groups on three dimensions",
    () => {
      const three: GroupsFixture = groups.three;
      const fit = calibrateGroups(three, "none");
      expectCalibration(fit, three, mirtTolerances, "three dimensions");
      expectGroups(fit, three, 5e-4);
    },
    UNDER_COVERAGE,
  );

  it("gives the same calibration with the cross-product standard errors for the same groups", () => {
    const hessian = calibrateGroups(twoGroups, "hessian");
    const cross = calibrateGroups(twoGroups, "cross-product");
    hessian.items.forEach((item, i) => {
      for (const [name, error] of Object.entries(defined(item.standardErrors))) {
        expectClose(defined(cross.items[i]?.standardErrors?.[name]) / error, 1, 0.2, `${item.id} ${name}`);
      }
    });
  }, 120_000);

  it("reports no groups without the options groups and fixed", () => {
    const { itemIds, responses, loadings, points, model } = twoGroups;
    const fit = calibrateMultidimensional(buildResponseMatrix(responses, { itemIds: [...itemIds] }), modelOf(model), {
      loadings,
      points,
    });
    expect(fit.groups).toBeUndefined();
  });

  it("names the group of all persons when fixed items come without groups", () => {
    const { itemIds, responses, loadings, points, model, fixed } = anchored;
    const fit = calibrateMultidimensional(buildResponseMatrix(responses, { itemIds: [...itemIds] }), modelOf(model), {
      loadings,
      points,
      fixed: Object.fromEntries(Object.entries(defined(fixed)).map(([id, p]) => [id, parametersOf(model, p)])),
    });
    expect(fit.groups?.map(({ group }) => group)).toEqual(["all"]);
    expect(fit.converged).toBe(true);
  });

  it("orders the groups by first appearance and reads a group label as a string", () => {
    const labels = [...twoGroups.personGroups].reverse();
    const fit = calibrateGroups(
      { ...twoGroups, personGroups: labels, responses: [...twoGroups.responses].reverse() },
      "none",
      {
        tolerance: 1e-6,
      },
    );
    expect(fit.groups?.map(({ group }) => group)).toEqual(["B", "A"]);
    // The first group is now the reference: the other group's mean is the opposite of the earlier difference.
    expect(defined(fit.groups?.[1]).means[0]).toBeLessThan(0);
  });

  it("estimates the same groups from sparse records as from a matrix", () => {
    const small: GroupsFixture = {
      ...twoGroups,
      responses: twoGroups.responses.filter((_, person) => person % 3 === 0),
      personGroups: twoGroups.personGroups.filter((_, person) => person % 3 === 0),
    };
    const matrix = buildResponseMatrix(small.responses, { itemIds: [...small.itemIds] });
    const records = buildResponseRecords(
      matrix.responses.flatMap((row, person) =>
        row.flatMap((response, item) =>
          response === null
            ? []
            : [{ person: matrix.personIds[person] ?? "", item: matrix.itemIds[item] ?? "", response }],
        ),
      ),
      { personIds: [...matrix.personIds], itemIds: [...matrix.itemIds] },
    );
    const optionsModel = modelOf(small.model);
    const options = {
      loadings: small.loadings,
      points: 15,
      groups: [...small.personGroups],
    };
    const fromMatrix = calibrateMultidimensional(matrix, optionsModel, options);
    const fromRecords = calibrateMultidimensional(records, optionsModel, options);
    expect(fromRecords.groups).toEqual(fromMatrix.groups);
    expect(fromRecords.logLikelihood).toBeCloseTo(fromMatrix.logLikelihood, 8);
  }, 120_000);

  it("keeps the estimates when the persons are shuffled together with their groups", () => {
    const first = twoGroups.personGroups.map((_, k) => k).filter((k) => twoGroups.personGroups[k] === "A");
    const second = twoGroups.personGroups.map((_, k) => k).filter((k) => twoGroups.personGroups[k] === "B");
    // Alternating the groups keeps A first, so the reference group and the scale stay the same.
    const order = first.flatMap((k, n) => (second[n] === undefined ? [k] : [k, defined(second[n])]));
    const shuffled: GroupsFixture = {
      ...twoGroups,
      responses: order.map((k) => defined(twoGroups.responses[k])),
      personGroups: order.map((k) => defined(twoGroups.personGroups[k])),
    };
    const expected = calibrateGroups(twoGroups, "none");
    const fit = calibrateGroups(shuffled, "none");
    expectGroups(fit, { ...twoGroups, groupMoments: defined(expected.groups).map((g) => g) }, 1e-6);
    expectClose(fit.logLikelihood, expected.logLikelihood, 1e-6, "log-likelihood");
  }, 120_000);

  it("stops at the largest number of iterations and says so", () => {
    const fit = calibrateGroups(twoGroups, "none", { maxIterations: 3, tolerance: 1e-12 });
    expect(fit.converged).toBe(false);
    expect(fit.iterations).toBe(3);
  });

  it("does not claim convergence when a group's standard deviation is below the spacing of the nodes", () => {
    const fit = calibrateGroups(twoGroups, "none", { points: 5, tolerance: 1e-6 });
    expect(fit.iterations).toBeLessThan(2000);
    expect(fit.converged).toBe(false);
  }, 120_000);

  describe("recovery", () => {
    const dimensions = 3;
    const perDimension = 40;
    const truth = [
      {
        group: "A",
        means: [0, 0, 0],
        persons: 700,
        covariance: [
          [1, 0.3, 0.2],
          [0.3, 1, 0.4],
          [0.2, 0.4, 1],
        ],
      },
      {
        group: "B",
        means: [0.4, -0.3, 0.5],
        persons: 600,
        covariance: [
          [1.4, 0.4, 0.3],
          [0.4, 0.7, 0.2],
          [0.3, 0.2, 1.2],
        ],
      },
      {
        group: "C",
        means: [-0.5, 0.3, 0.2],
        persons: 600,
        covariance: [
          [0.8, 0.2, 0.1],
          [0.2, 1.3, 0.5],
          [0.1, 0.5, 0.9],
        ],
      },
    ];

    /** Simulates responses to 40 items on each of 3 dimensions in three groups of the given moments. */
    function simulate(seed: number) {
      const random = createRandom(seed);
      const items = Array.from({ length: dimensions * perDimension }, (_, i) => ({
        model: multidimensionalTwoParameterLogistic,
        parameters: {
          a: Array.from({ length: dimensions }, (__, d) =>
            d === Math.floor(i / perDimension) ? 0.8 + random.uniform() : 0,
          ),
          intercept: random.normal() * 0.7,
          c: 0,
          d: 1,
        } satisfies MultidimensionalLogisticParameters,
      }));
      const thetas: number[][] = [];
      const labels: string[] = [];
      for (const { group, means, persons, covariance } of truth) {
        const factor = defined(cholesky(Float64Array.from(covariance.flat()), dimensions));
        for (let p = 0; p < persons; p++) {
          const z = Array.from({ length: dimensions }, () => random.normal());
          thetas.push(
            means.map(
              (mean, i) =>
                mean + z.reduce((sum, value, j) => sum + (j <= i ? (factor[i * dimensions + j] ?? 0) * value : 0), 0),
            ),
          );
          labels.push(group);
        }
      }
      return { matrix: simulateMultidimensionalResponses(items, thetas, random), labels };
    }

    // Under coverage this recovery took 541 to 880 s.
    it(
      "recovers the means and covariances of three groups on three dimensions from 40 items per dimension",
      () => {
        const { matrix, labels } = simulate(20261031);
        const fit = calibrateMultidimensional(matrix, multidimensionalTwoParameterLogistic, {
          loadings: Object.fromEntries(matrix.itemIds.map((id, i) => [id, [`D${Math.floor(i / perDimension) + 1}`]])),
          groups: labels,
          points: 21,
          maxIterations: 1000,
        });
        expect(fit.converged).toBe(true);
        // The first group is the reference and has the moments of its scale. On this seed the errors are at most 0.13 for a
        // mean and 0.18 for a covariance at 21 nodes, and 0.24 and 0.61 at the default 15, which the tolerances exclude.
        truth.forEach(({ group, means, covariance }, g) => {
          const found = defined(fit.groups?.[g]);
          expect(found.group).toBe(group);
          means.forEach((value, d) => {
            expectClose(defined(found.means[d]), value, 0.18, `mean ${d + 1} of group ${group}`);
          });
          covariance.forEach((row, i) => {
            row.forEach((value, j) => {
              expectClose(
                defined(found.covariance[i]?.[j]),
                value,
                0.25,
                `covariance ${i + 1},${j + 1} of group ${group}`,
              );
            });
          });
        });
      },
      3 * UNDER_COVERAGE,
    );
  });
});

describe("latent distribution of a group", () => {
  const free = { correlations: true, variances: true, means: true } as const;
  const Covariance3 = Float64Array.of(1.4, 0.4, -0.2, 0.4, 0.7, 0.3, -0.2, 0.3, 1.1);
  const Mean3 = [0.5, -0.3, 0.2];

  /** The expected number of persons at each node of a grid under a mean and covariance, for 1,000 persons. */
  function counts(
    grid: ReturnType<typeof productGrid>,
    covariance: Float64Array,
    mean: readonly number[],
  ): Float64Array {
    const latent = latentDistribution(grid);
    expect(latent.set(covariance, mean)).toBe(true);
    return latent.logWeights.map((logWeight) => 1000 * Math.exp(logWeight));
  }

  it("weights the nodes by the normal density of its mean and covariance, summing to 1", () => {
    const grid = productGrid(3, 9);
    const latent = latentDistribution(grid, free);
    expect(latent.set(Covariance3, Mean3)).toBe(true);
    expect(latent.logWeights.reduce((sum, value) => sum + Math.exp(value), 0)).toBeCloseTo(1, 12);
    expect(Array.from(latent.mean)).toEqual(Mean3);
    expect(Array.from(latent.standardDeviations)).toEqual([Math.sqrt(1.4), Math.sqrt(0.7), Math.sqrt(1.1)]);
    // The weighted mean of the nodes is near the mean, up to the truncation of the grid.
    const weighted = [0, 0, 0].map((_, d) =>
      latent.logWeights.reduce((sum, logWeight, q) => sum + Math.exp(logWeight) * (grid.nodes[3 * q + d] ?? 0), 0),
    );
    weighted.forEach((value, d) => {
      expectClose(value, defined(Mean3[d]), 0.15, `weighted mean ${d + 1}`);
    });
  });

  it("differentiates the expected negative log-likelihood with respect to its entries", () => {
    const grid = productGrid(3, 7);
    const latent = latentDistribution(grid, free);
    const nodeCounts = counts(grid, Covariance3, Mean3).map((value, q) => value * (1 + 0.3 * Math.sin(q)));
    latent.set(Float64Array.of(1.2, 0.2, 0.1, 0.2, 0.9, -0.3, 0.1, -0.3, 1), [0.1, 0.2, -0.1]);
    const start = latent.entries();
    expect(start.length).toBe(9);
    const gradient = new Float64Array(9);
    latent.gradient(nodeCounts, gradient);
    const objective = (entries: Float64Array): number => {
      latent.assignEntries(entries);
      return -nodeCounts.reduce((sum, value, q) => sum + value * (latent.logWeights[q] ?? 0), 0);
    };
    const step = 1e-6;
    start.forEach((_, e) => {
      const at = (shift: number): number => {
        const moved = Float64Array.from(start);
        moved[e] = (moved[e] ?? 0) + shift;
        return objective(moved);
      };
      expectClose(
        gradient[e] ?? Number.NaN,
        (at(step) - at(-step)) / (2 * step),
        1e-5 * Math.max(1, Math.abs(gradient[e] ?? 0)),
        `entry ${e}`,
      );
    });
  });

  it("gives scores that average to zero under its weights and match the gradient of the log weights", () => {
    const grid = productGrid(2, 9);
    const latent = latentDistribution(grid, free);
    latent.set(Float64Array.of(1.3, 0.3, 0.3, 0.8), [0.2, -0.4]);
    const scores = new Float64Array(latent.size * grid.count);
    latent.scores(scores);
    for (let e = 0; e < latent.size; e++) {
      let expected = 0;
      for (let q = 0; q < grid.count; q++)
        expected += Math.exp(latent.logWeights[q] ?? 0) * (scores[e * grid.count + q] ?? 0);
      expectClose(expected, 0, 1e-12, `entry ${e}`);
    }
    const start = latent.entries();
    const logWeights = (entries: Float64Array): Float64Array => {
      latent.assignEntries(entries);
      return latent.logWeights.slice();
    };
    const step = 1e-6;
    start.forEach((_, e) => {
      const up = Float64Array.from(start);
      const down = Float64Array.from(start);
      up[e] = (up[e] ?? 0) + step;
      down[e] = (down[e] ?? 0) - step;
      const high = logWeights(up);
      const low = logWeights(down);
      for (let q = 0; q < grid.count; q += 5) {
        expectClose(
          scores[e * grid.count + q] ?? Number.NaN,
          ((high[q] ?? 0) - (low[q] ?? 0)) / (2 * step),
          1e-6,
          `entry ${e} node ${q}`,
        );
      }
    });
  });

  it("recovers the mean and covariance that generated the node counts", () => {
    const grid = productGrid(3, 11);
    const latent = latentDistribution(grid, free);
    latent.maximize(counts(grid, Covariance3, Mean3));
    Covariance3.forEach((value, k) => {
      expectClose(latent.matrix[k] ?? Number.NaN, value, 1e-6, `covariance entry ${k}`);
    });
    Mean3.forEach((value, d) => {
      expectClose(latent.mean[d] ?? Number.NaN, value, 1e-6, `mean ${d + 1}`);
    });
  });

  it("estimates only the entries it is given as free", () => {
    const grid = productGrid(2, 15);
    const source = Float64Array.of(1.5, 0.4, 0.4, 0.6);
    const nodeCounts = counts(grid, source, [0.3, -0.2]);
    const means = latentDistribution(grid, { correlations: false, variances: false, means: true });
    expect(means.size).toBe(2);
    means.maximize(nodeCounts);
    expect(Array.from(means.matrix)).toEqual([1, 0, 0, 1]);
    expect(means.mean[0] ?? 0).toBeGreaterThan(0.1);
    const diagonal = latentDistribution(grid, { correlations: false, variances: true, means: true });
    expect(diagonal.size).toBe(4);
    diagonal.maximize(nodeCounts);
    expect(diagonal.matrix[1]).toBe(0);
    expect(diagonal.matrix[0] ?? 0).toBeGreaterThan(1);
    const fixed = latentDistribution(grid, { correlations: false, variances: false, means: false });
    expect(fixed.size).toBe(0);
    fixed.maximize(nodeCounts);
    expect(Array.from(fixed.matrix)).toEqual([1, 0, 0, 1]);
  });

  it("keeps the mean and standard deviation of a pinned dimension and estimates the others", () => {
    const grid = productGrid(3, 11);
    const pinned = latentDistribution(grid, { ...free, pinned: [0] });
    // Three correlations, two standard deviations and two means.
    expect(pinned.size).toBe(7);
    const covariance = Float64Array.of(1, 0.54, 0.27, 0.54, 1.44, 0.54, 0.27, 0.54, 0.81);
    pinned.maximize(counts(grid, covariance, [0, 0.4, -0.3]));
    covariance.forEach((value, k) => {
      expectClose(pinned.matrix[k] ?? Number.NaN, value, 1e-6, `covariance entry ${k}`);
    });
    expectClose(pinned.mean[1] ?? Number.NaN, 0.4, 1e-6, "mean 2");
    expectClose(pinned.mean[2] ?? Number.NaN, -0.3, 1e-6, "mean 3");
    // Counts of a first dimension with another mean and spread leave the pinned entries where they were set.
    pinned.maximize(counts(grid, Float64Array.of(1.5, 0.5, 0.2, 0.5, 1, 0.1, 0.2, 0.1, 1), [0.3, 0, 0]));
    expect(pinned.mean[0]).toBe(0);
    expect(pinned.standardDeviations[0]).toBe(1);
    expect(pinned.entries().length).toBe(7);
  });

  it("round-trips its entries and its parameters", () => {
    const grid = productGrid(3, 7);
    const latent = latentDistribution(grid, free);
    latent.set(Covariance3, Mean3);
    const entries = latent.entries();
    const parameters = latent.parameters();
    const matrix = latent.matrix.slice();
    latent.set(Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1), [0, 0, 0]);
    expect(latent.assignEntries(entries)).toBe(true);
    latent.matrix.forEach((value, k) => {
      expectClose(value, matrix[k] ?? Number.NaN, 1e-12, `entry ${k}`);
    });
    latent.set(Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1), [0, 0, 0]);
    expect(latent.assign(parameters)).toBe(true);
    latent.matrix.forEach((value, k) => {
      expectClose(value, matrix[k] ?? Number.NaN, 1e-12, `parameter entry ${k}`);
    });
  });

  it("rejects a covariance that is not positive definite and keeps the distribution", () => {
    const latent = latentDistribution(productGrid(2, 9), free);
    const before = Float64Array.from(latent.logWeights);
    expect(latent.set(Float64Array.of(1, 2, 2, 1), [0.5, 0.5])).toBe(false);
    expect(latent.set(Float64Array.of(0, 0, 0, 1))).toBe(false);
    expect(latent.logWeights).toEqual(before);
    expect(Array.from(latent.mean)).toEqual([0, 0]);
  });
});

describe("expectation step with groups", () => {
  const grid = productGrid(1, 11);
  const model = multidimensionalTwoParameterLogistic;
  const rows: readonly (readonly (number | null)[])[] = [
    [1, 0, 1],
    [0, 0, 1],
    [1, 1, 1],
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 0],
  ];

  function items(): ExpectationItem[] {
    return [0.4, -0.2, 0.7].map((intercept) => ({
      model,
      categoryCount: 2,
      parameters: { a: [1.3], intercept, c: 0, d: 1 },
      counts: new Float64Array(grid.count * 2),
    }));
  }

  /** The log weights of a normal distribution of the given mean on the grid. */
  function weights(mean: number): Float64Array {
    const latent = latentDistribution(grid, { correlations: false, variances: false, means: true });
    latent.set(Float64Array.of(1), [mean]);
    return latent.logWeights;
  }

  /** Runs the E-step of the rows of one group under its weights alone. */
  function alone(subset: readonly (readonly (number | null)[])[], logWeights: Float64Array) {
    const each = items();
    const step = expectationStep(compressResponses(buildResponseMatrix(subset)), each, grid, logWeights);
    return {
      logLikelihood: step.run(),
      nodeCounts: step.nodeCounts.slice(),
      counts: each.map((item) => item.counts.slice()),
    };
  }

  for (const [name, data] of [
    ["every person answers every item", rows],
    ["some responses are missing", rows.map((row, p) => (p === 1 ? [0, null, 1] : row))],
  ] as const) {
    it(`counts the persons of each group under the weights of the group when ${name}`, () => {
      const low = weights(-0.5);
      const high = weights(0.6);
      const each = items();
      const step = expectationStep(
        compressResponses(buildResponseMatrix(data)),
        each,
        grid,
        [low, high],
        [0, 0, 0, 1, 1, 1],
      );
      const logLikelihood = step.run();
      const first = alone(data.slice(0, 3), low);
      const second = alone(data.slice(3), high);
      expect(logLikelihood).toBeCloseTo(first.logLikelihood + second.logLikelihood, 10);
      step.groupCounts[0]?.forEach((value, q) => {
        expect(value).toBeCloseTo(first.nodeCounts[q] ?? Number.NaN, 10);
      });
      step.groupCounts[1]?.forEach((value, q) => {
        expect(value).toBeCloseTo(second.nodeCounts[q] ?? Number.NaN, 10);
      });
      step.nodeCounts.forEach((value, q) => {
        expect(value).toBeCloseTo((first.nodeCounts[q] ?? 0) + (second.nodeCounts[q] ?? 0), 10);
      });
      each.forEach((item, i) => {
        item.counts.forEach((value, k) => {
          expect(value).toBeCloseTo((first.counts[i]?.[k] ?? 0) + (second.counts[i]?.[k] ?? 0), 10);
        });
      });
    });
  }

  it("keeps the counts of one group as the counts of all persons", () => {
    const step = expectationStep(compressResponses(buildResponseMatrix(rows)), items(), grid, weights(0));
    step.run();
    expect(step.groupCounts).toHaveLength(1);
    expect(step.groupCounts[0]).toBe(step.nodeCounts);
  });
});

describe("multiple groups errors", () => {
  const { personGroups, responses, itemIds, loadings, model } = groups.correlated;
  /** Forty persons of each group, answering every item, on a coarse grid. */
  const keep = [...Array.from({ length: 40 }, (_, k) => k), ...Array.from({ length: 40 }, (_, k) => 900 + k)];
  const matrix = (rows: readonly (readonly (number | null)[])[] = keep.map((k) => defined(responses[k]))) =>
    buildResponseMatrix(rows, { itemIds: [...itemIds] });
  const labels = keep.map((k) => defined(personGroups[k]));
  const baseModel = modelOf(model);
  const base = { loadings, points: 7 };

  it("needs one string label for each person", () => {
    expect(() => calibrateMultidimensional(matrix(), baseModel, { ...base, groups: ["A", "B"] })).toThrow(
      "Found 2 groups for 80 persons.",
    );
    expect(() => calibrateMultidimensional(matrix(), baseModel, { ...base, groups: JSON.parse('"A"') })).toThrow(
      "Found no groups for 80 persons.",
    );
    expect(() =>
      calibrateMultidimensional(matrix(), baseModel, {
        ...base,
        groups: JSON.parse(JSON.stringify([1, ...labels.slice(1)])),
      }),
    ).toThrow("The group of person 1 must be a string, found 1.");
    expect(() => calibrateMultidimensional(matrix(), baseModel, { ...base, groups: JSON.parse("null") })).toThrow(
      "Found no groups for 80 persons.",
    );
  });

  it("needs a person with a response in each group", () => {
    const rows = keep.map((k, p) => (p < 40 ? defined(responses[k]) : defined(responses[k]).map(() => null)));
    expect(() => calibrateMultidimensional(matrix(rows), baseModel, { ...base, groups: labels })).toThrow(
      'Group "B" has no person with a response.',
    );
  });

  it("needs more persons than correlated dimensions in each group, whose correlations otherwise run to ±1", () => {
    const one = keep.map((k, p) =>
      p === 0 || p >= 40 ? defined(responses[k]) : defined(responses[k]).map(() => null),
    );
    expect(() => calibrateMultidimensional(matrix(one), baseModel, { ...base, groups: labels })).toThrow(
      'Group "A" has 1 person with a response, no more than its 2 correlated dimensions, whose correlations then run to ±1: estimate the correlations from more persons, or set correlated to false.',
    );
    expect(
      calibrateMultidimensional(matrix(one), baseModel, { ...base, groups: labels, correlated: false }).groups,
    ).toHaveLength(2);
    const three = keep.map((k, p) =>
      p < 3 || p >= 40 ? defined(responses[k]) : defined(responses[k]).map(() => null),
    );
    expect(calibrateMultidimensional(matrix(three), baseModel, { ...base, groups: labels }).groups).toHaveLength(2);
    // Two persons of opposite responses, so that every item has both categories.
    const first = defined(responses[0]).map((response) => response ?? 0);
    const two = buildResponseMatrix([first, first.map((response) => 1 - response)], { itemIds: [...itemIds] });
    for (const method of ["em", "mhrm"] as const) {
      expect(() => calibrateMultidimensional(two, baseModel, { loadings, method })).toThrow(
        "The calibration has 2 persons with a response, no more than its 2 correlated dimensions",
      );
    }
  });

  it("needs groups that share items", () => {
    const rows = keep.map((k, p) =>
      defined(responses[k]).map((response, i) => (p < 40 ? (i < 6 ? response : null) : i < 6 ? null : response)),
    );
    expect(() => calibrateMultidimensional(matrix(rows), baseModel, { ...base, groups: labels })).toThrow(
      'The group "B" shares no item with the group "A", directly or through other groups',
    );
  });

  it("links groups through other groups", () => {
    const three = [...labels.slice(0, 40), ...labels.slice(40, 70), ...labels.slice(70).map(() => "C")];
    const rows = keep.map((k, p) =>
      defined(responses[k]).map((response, i) => (p >= 70 ? (i >= 3 && i < 9 ? response : null) : response)),
    );
    const fit = calibrateMultidimensional(matrix(rows), baseModel, { ...base, groups: three });
    expect(fit.groups?.map(({ group }) => group)).toEqual(["A", "B", "C"]);
  });

  it("needs fixed items that set the scale of every dimension", () => {
    const fixed = { [defined(itemIds[0])]: { a: [1, 0], intercept: 0.2, c: 0, d: 1 } };
    expect(() => calibrateMultidimensional(matrix(), baseModel, { ...base, groups: labels, fixed })).toThrow(
      'No fixed item with a slope on dimension "F2" was answered by the group "A" or the groups linked to it',
    );
    expect(() => calibrateMultidimensional(matrix(), baseModel, { ...base, fixed })).toThrow(
      'No fixed item with a slope on dimension "F2" was answered by the persons',
    );
  });

  it("checks the fixed items", () => {
    const parameters = { a: [1, 0], intercept: 0.2, c: 0, d: 1 };
    const both = { [defined(itemIds[0])]: parameters, [defined(itemIds[6])]: { a: [0, 1], intercept: 0, c: 0, d: 1 } };
    expect(() => calibrateMultidimensional(matrix(), baseModel, { ...base, fixed: { nobody: parameters } })).toThrow(
      'Fixed parameters are given for "nobody", which is not an item id.',
    );
    expect(() =>
      calibrateMultidimensional(matrix(), baseModel, {
        ...base,
        fixed: { ...both, [defined(itemIds[1])]: { ...parameters, a: [1] } },
      }),
    ).toThrow('Fixed item "q02" has 1 dimensions, but the calibration has 2.');
    expect(() =>
      calibrateMultidimensional(matrix(), baseModel, {
        ...base,
        fixed: { ...both, [defined(itemIds[1])]: { ...parameters, c: 2 } },
      }),
    ).toThrow('Fixed item "q02"');
    expect(() => calibrateMultidimensional(matrix(), baseModel, { ...base, fixed: JSON.parse("null") })).toThrow(
      "The option fixed must be an object of item parameters by item id, found null.",
    );
    expect(() =>
      calibrateMultidimensional(matrix(), baseModel, { ...base, fixed: { ...both, q02: JSON.parse("null") } }),
    ).toThrow('Fixed item "q02"');
    const graded = buildResponseMatrix(
      [
        [0, 1],
        [1, 2],
        [2, 0],
      ],
      { itemIds: ["g1", "g2"] },
    );
    expect(() =>
      calibrateMultidimensional(graded, multidimensionalGradedResponse, {
        loadings: { g1: ["D"], g2: ["D"] },
        fixed: { g1: { a: [1], intercepts: [0.5] } },
      }),
    ).toThrow('Item "g1" has a response of 2, but its fixed parameters have 2 categories.');
  });

  it("is for EM and not for MH-RM", () => {
    expect(() =>
      calibrateMultidimensional(matrix(), baseModel, { ...base, points: undefined, method: "mhrm", groups: labels }),
    ).toThrow('The option groups is for EM, which the method "mhrm" does not use.');
    expect(() =>
      calibrateMultidimensional(matrix(), baseModel, {
        ...base,
        points: undefined,
        method: "mhrm",
        fixed: { [defined(itemIds[0])]: { a: [1, 0], intercept: 0, c: 0, d: 1 } },
      }),
    ).toThrow('The option fixed is for EM, which the method "mhrm" does not use.');
  });
});

/** The moments that abilities over several occasions are drawn with. */
interface TraitMoments {
  readonly means: readonly number[];
  readonly sds: readonly number[];
  readonly rho: number;
}

/**
 * Draws the abilities of persons on several occasions, normal with the means and standard deviations and an
 * autoregressive correlation, and their responses to the same 2PL items on every occasion, with slopes from 0.8 to 1.8
 * and difficulties of standard deviation 0.8.
 */
function simulateTrait(
  seed: number,
  persons: number,
  itemCount: number,
  { means, sds, rho }: TraitMoments,
): { readonly occasions: ResponseMatrix[]; readonly thetas: number[][] } {
  const random = createRandom(seed);
  const slopes = Array.from({ length: itemCount }, () => 0.8 + random.uniform());
  const difficulties = Array.from({ length: itemCount }, () => 0.8 * random.normal());
  const ids = Array.from({ length: itemCount }, (_, i) => `q${i + 1}`);
  const thetas = Array.from({ length: persons }, () => {
    let state = random.normal();
    return means.map((mean, t) => {
      if (t > 0) state = rho * state + Math.sqrt(1 - rho * rho) * random.normal();
      return mean + (sds[t] ?? 1) * state;
    });
  });
  const occasions = means.map((_, t) =>
    buildResponseMatrix(
      thetas.map((theta) =>
        ids.map((__, i) =>
          random.uniform() < logistic(defined(slopes[i]) * ((theta[t] ?? 0) - defined(difficulties[i]))) ? 1 : 0,
        ),
      ),
      { itemIds: ids },
    ),
  );
  return { occasions, thetas };
}

/** Draws the responses of {@link simulateTrait}, without the abilities. */
function simulateOccasions(seed: number, persons: number, itemCount: number, moments: TraitMoments): ResponseMatrix[] {
  return simulateTrait(seed, persons, itemCount, moments).occasions;
}

/** Persons, items and moments small enough for many calibrations: 2 occasions, 300 persons, 6 items. */
const SMALL = { means: [0, 0.4], sds: [1, 1.2], rho: 0.8 } as const;

/** Expects two calibrations to agree on the log-likelihood, the items and the moments. */
function expectSameFit(
  actual: LongitudinalCalibration,
  expected: LongitudinalCalibration,
  ids: readonly string[] = expected.items.map(({ id }) => id),
): void {
  expectClose(actual.logLikelihood, expected.logLikelihood, 1e-6, "log-likelihood");
  for (const id of ids) {
    const found = defined(actual.items.find((item) => item.id === id)).parameters;
    expectParameters(found, defined(expected.items.find((item) => item.id === id)).parameters, 1e-5, `item ${id}`);
  }
  expected.occasions.forEach(({ mean: expectedMean, sd }, t) => {
    expectClose(defined(actual.occasions[t]).mean, expectedMean, 1e-5, `mean ${t}`);
    expectClose(defined(actual.occasions[t]).sd, sd, 1e-5, `sd ${t}`);
  });
}

/** A case of the longitudinal fixture: the occasions' responses and the estimates of mirt. */
interface LongitudinalCase {
  readonly model: string;
  readonly points: number;
  readonly occasions: readonly {
    readonly personIds: readonly string[];
    readonly itemIds: readonly string[];
    readonly responses: readonly (readonly (number | null)[])[];
  }[];
  readonly items: readonly { readonly id: string; readonly parameters: object }[];
  readonly means: readonly number[];
  readonly sds: readonly number[];
  readonly correlations: readonly (readonly number[])[];
  readonly changes: readonly {
    readonly from: number;
    readonly to: number;
    readonly mean: number;
    readonly sd: number;
    readonly correlationWithStart: number;
  }[];
  readonly logLikelihood: number;
}

describe("longitudinal calibration", () => {
  /** How far each case may be from mirt: mirt's own EM stops where its estimates have not settled. */
  const Tolerances = { two: 1e-4, anchored: 1e-4, graded: 1e-4, three: 1e-3, rasch: 5e-4 } as const;
  type CaseName = keyof typeof Tolerances;
  const cases: Readonly<Record<CaseName, LongitudinalCase>> = {
    two: longitudinal.two,
    three: longitudinal.three,
    anchored: longitudinal.anchored,
    graded: longitudinal.graded,
    rasch: longitudinal.rasch,
  };
  const names = Object.keys(Tolerances) as CaseName[];
  const matricesOf = (source: LongitudinalCase): ResponseMatrix[] =>
    source.occasions.map(({ personIds, itemIds, responses }) =>
      buildResponseMatrix(responses, { personIds: [...personIds], itemIds: [...itemIds] }),
    );
  const fits = new Map<CaseName, LongitudinalCalibration>();
  /** Calibrates a case at a tolerance tight enough for its estimates to settle, once for all tests. */
  function fitOf(name: CaseName): LongitudinalCalibration {
    const source = cases[name];
    const found =
      fits.get(name) ??
      calibrateLongitudinal(matricesOf(source), {
        model: defined(FIXTURE_MODELS[source.model]),
        points: source.points,
        tolerance: 1e-6,
        maxIterations: 5000,
      });
    fits.set(name, found);
    return found;
  }

  it.each(names)(
    "matches mirt on the %s case: items, moments, changes and log-likelihood",
    (name) => {
      const fit = fitOf(name);
      const source = cases[name];
      const tolerance = Tolerances[name];
      expect(fit.converged).toBe(true);
      expect(fit.items.map(({ id }) => id)).toEqual(source.items.map(({ id }) => id));
      fit.items.forEach(({ id, parameters }, i) => {
        expectParameters(parameters, source.items[i]?.parameters, tolerance, `item ${id}`);
      });
      source.means.forEach((mean, t) => {
        expectClose(defined(fit.occasions[t]).mean, mean, tolerance, `mean ${t}`);
        expectClose(defined(fit.occasions[t]).sd, defined(source.sds[t]), tolerance, `sd ${t}`);
      });
      source.correlations.forEach((row, i) => {
        row.forEach((value, j) => {
          expectClose(defined(fit.correlations[i]?.[j]), value, tolerance, `correlation ${i},${j}`);
        });
      });
      source.changes.forEach((change, k) => {
        const found = defined(fit.changes[k]);
        expect([found.from, found.to]).toEqual([change.from, change.to]);
        expectClose(found.mean, change.mean, 2 * tolerance, `mean of change ${k}`);
        expectClose(found.sd, change.sd, 2 * tolerance, `sd of change ${k}`);
        expectClose(
          found.correlationWithStart,
          change.correlationWithStart,
          2 * tolerance,
          `correlation with start ${k}`,
        );
      });
      expectClose(fit.logLikelihood, source.logLikelihood, 1e-2, "log-likelihood");
      // Where mirt's EM stops short of the maximum its log-likelihood is lower; it is never higher.
      expect(fit.logLikelihood).toBeGreaterThanOrEqual(source.logLikelihood - 1e-6);
    },
    120_000,
  );

  it("matches TAM's Rasch model, whose scale is that of the first occasion's standard deviation", () => {
    const { tam } = longitudinal.rasch;
    const fit = fitOf("rasch");
    fit.items.forEach(({ id, parameters }, i) => {
      expectParameters(parameters, { a: tam.slope, b: tam.difficulties[i], c: 0, d: 1 }, 5e-3, `item ${id}`);
    });
    tam.means.forEach((mean, t) => {
      expectClose(defined(fit.occasions[t]).mean, mean, 5e-3, `mean ${t}`);
      expectClose(defined(fit.occasions[t]).sd, defined(tam.sds[t]), 5e-3, `sd ${t}`);
    });
    tam.correlations.forEach((row, i) => {
      row.forEach((value, j) => {
        expectClose(defined(fit.correlations[i]?.[j]), value, 5e-3, `correlation ${i},${j}`);
      });
    });
    expectClose(fit.logLikelihood, tam.logLikelihood, 1e-2, "log-likelihood");
  }, 120_000);

  describe("standard errors", () => {
    const { two } = longitudinal;
    const run = (standardErrors: "hessian" | "cross-product") =>
      calibrateLongitudinal(matricesOf(two), {
        points: two.points,
        tolerance: 1e-8,
        maxIterations: 5000,
        standardErrors,
      });

    it("matches mirt's Oakes errors by the observed information, which the Hessian of the gradient gives", () => {
      const errors = errorVector(run("hessian"));
      // Ten items' slopes and difficulties, then the correlation, the standard deviation and the mean of the second occasion.
      expect(errors).toHaveLength(23);
      two.standardErrors.forEach((se, k) => {
        expectClose(defined(errors[k]), se, 1e-5, `standard error ${k}`);
      });
    }, 120_000);

    it("matches mirt's cross-product errors by the persons' scores", () => {
      const errors = errorVector(run("cross-product"));
      two.crossProductStandardErrors.forEach((se, k) => {
        expectClose(defined(errors[k]), se, 1e-5, `standard error ${k}`);
      });
    }, 120_000);
  });

  describe("grid", () => {
    const small = simulateOccasions(5, 200, 6, SMALL);
    const three = simulateOccasions(6, 150, 6, { means: [0, 0.3, 0.6], sds: [1, 1.1, 1.2], rho: 0.8 });

    it("has 31 nodes per occasion for two occasions and 21 for three by default", () => {
      expectSameFit(calibrateLongitudinal(small), calibrateLongitudinal(small, { points: 31 }));
      expectSameFit(calibrateLongitudinal(three), calibrateLongitudinal(three, { points: 21 }));
    }, 120_000);

    it("changes the estimates a little with the number of nodes on short tests", () => {
      const coarse = calibrateLongitudinal(small, { points: 21 });
      const fine = calibrateLongitudinal(small, { points: 41 });
      expectClose(coarse.logLikelihood, fine.logLikelihood, 0.5);
      expectClose(defined(coarse.occasions[1]).mean, defined(fine.occasions[1]).mean, 0.02);
    });
  });
});

describe("longitudinal recovery", () => {
  const moments = { means: [0, 0.3, 0.6], sds: [1, 1.15, 1.3], rho: 0.85 } as const;
  /** The thresholds are rounded up to a multiple of this. */
  const Step = 0.005;
  /**
   * The largest errors of the means, standard deviations and correlations that the recovery of 5,000 persons may have:
   * 1.5 times the worst of the three seeds of mirt on the same design, from the fixture, rounded up to a multiple of
   * 0.005. With more items per occasion the errors are smaller, so the same thresholds hold for 40 items.
   */
  function thresholds(occasionCount: number) {
    const { worst } = defined(longitudinalRecovery.recovery.find((row) => row.occasions === occasionCount));
    const limit = (error: number): number => Math.ceil((1.5 * error) / Step - 1e-9) * Step;
    return { mean: limit(worst.mean), sd: limit(worst.sd), correlation: limit(worst.correlation) };
  }
  const column = (thetas: readonly (readonly number[])[], t: number): number[] =>
    thetas.map((theta) => defined(theta[t]));
  const spread = (values: readonly number[]): number => {
    const center = mean(values);
    return Math.sqrt(values.reduce((sum, value) => sum + (value - center) ** 2, 0) / (values.length - 1));
  };
  const correlate = (x: readonly number[], y: readonly number[]): number => {
    const [cx, cy] = [mean(x), mean(y)];
    const cross = x.reduce((sum, value, i) => sum + (value - cx) * (defined(y[i]) - cy), 0);
    return cross / Math.sqrt(x.reduce((s, v) => s + (v - cx) ** 2, 0) * y.reduce((s, v) => s + (v - cy) ** 2, 0));
  };

  /**
   * Returns the largest distance of the estimates from the moments of the abilities that the persons were given, so
   * that the sampling of the persons is no part of it: the means and standard deviations of the occasions after the
   * first, in units of the first occasion's sample standard deviation and from its sample mean, and the correlations.
   */
  function errorsOf(fit: LongitudinalCalibration, thetas: readonly (readonly number[])[]) {
    const first = column(thetas, 0);
    const errors = { mean: 0, sd: 0, correlation: 0 };
    fit.occasions.forEach(({ mean: estimated, sd }, t) => {
      const values = column(thetas, t);
      if (t > 0) {
        errors.mean = Math.max(errors.mean, Math.abs(estimated - (mean(values) - mean(first)) / spread(first)));
        errors.sd = Math.max(errors.sd, Math.abs(sd - spread(values) / spread(first)));
      }
      for (let u = 0; u < t; u++) {
        errors.correlation = Math.max(
          errors.correlation,
          Math.abs(defined(fit.correlations[t]?.[u]) - correlate(values, column(thetas, u))),
        );
      }
    });
    return errors;
  }

  /** Calibrates the persons of a seed at the default nodes, or at `points`. */
  function recover(seed: number, occasionCount: number, itemCount: number, points?: number) {
    const { occasions, thetas } = simulateTrait(seed, 5000, itemCount, {
      ...moments,
      means: moments.means.slice(0, occasionCount),
      sds: moments.sds.slice(0, occasionCount),
    });
    const fit = calibrateLongitudinal(occasions, { ...(points === undefined ? {} : { points }), maxIterations: 1000 });
    return { fit, errors: errorsOf(fit, thetas) };
  }

  it.each([
    [2, 10],
    [2, 40],
    [3, 10],
    [3, 40],
  ] as const)(
    "recovers the moments of %i occasions from 5,000 persons and %i items on each",
    (occasionCount, itemCount) => {
      for (const seed of [1, 2, 3]) {
        const { fit, errors } = recover(seed, occasionCount, itemCount);
        expect(fit.converged).toBe(true);
        const limits = thresholds(occasionCount);
        expect(errors.mean, `mean, seed ${seed}`).toBeLessThanOrEqual(limits.mean);
        expect(errors.sd, `sd, seed ${seed}`).toBeLessThanOrEqual(limits.sd);
        expect(errors.correlation, `correlation, seed ${seed}`).toBeLessThanOrEqual(limits.correlation);
      }
    },
    600_000,
  );

  it("misses the thresholds at 40 items on three occasions with 15 nodes, which is why 21 is the default", () => {
    const { errors } = recover(1, 3, 40, 15);
    expect(errors.mean).toBeGreaterThan(thresholds(3).mean);
    expect(errors.sd).toBeGreaterThan(thresholds(3).sd);
  }, 600_000);

  /** Returns the largest difference between the means, standard deviations and correlations of two calibrations. */
  function momentDistance(a: LongitudinalCalibration, b: LongitudinalCalibration): number {
    const differences = a.occasions.flatMap(({ mean: m, sd }, t) => [
      Math.abs(m - defined(b.occasions[t]).mean),
      Math.abs(sd - defined(b.occasions[t]).sd),
    ]);
    for (const [i, row] of a.correlations.entries()) {
      for (const [j, value] of row.entries()) differences.push(Math.abs(value - defined(b.correlations[i]?.[j])));
    }
    return Math.max(...differences);
  }

  it("settles by 21 nodes per occasion at 40 items and 3 occasions, where 15 are far from 31", () => {
    for (const seed of [101, 102]) {
      const { occasions } = simulateTrait(seed, 2000, 40, moments);
      const [coarse, usual, fine] = [15, 21, 31].map((points) =>
        calibrateLongitudinal(occasions, { points, maxIterations: 1000 }),
      );
      expect(defined(fine).converged).toBe(true);
      expect(momentDistance(defined(usual), defined(fine)), `21 against 31, seed ${seed}`).toBeLessThan(0.03);
      expect(momentDistance(defined(coarse), defined(fine)), `15 against 31, seed ${seed}`).toBeGreaterThan(0.05);
    }
  }, 900_000);
});

/**
 * Returns the standard errors of a longitudinal calibration in the order of mirt's: each item's by its parameters, the
 * correlations of the pairs (1, 0), (2, 0), (2, 1) of occasions, then the standard deviations and the means of the
 * occasions after the first.
 */
function errorVector({ items, correlationStandardErrors, occasions }: LongitudinalCalibration): number[] {
  const correlations = defined(correlationStandardErrors).flatMap((row, i) => row.slice(0, i));
  const later = occasions.slice(1).map(({ standardErrors }) => defined(standardErrors));
  return [
    ...items.flatMap(({ standardErrors }) => Object.values(defined(standardErrors))),
    ...correlations,
    ...later.map(({ sd }) => sd),
    ...later.map(({ mean }) => mean),
  ];
}

describe("longitudinal calibration data", () => {
  const occasions = simulateOccasions(1, 300, 6, SMALL);
  const base = { points: 21, tolerance: 1e-8 } as const;
  const fit = calibrateLongitudinal(occasions, base);

  it("converges and fixes the first occasion at N(0, 1)", () => {
    expect(fit.converged).toBe(true);
    expect(fit.occasions[0]).toEqual({ mean: 0, sd: 1, standardErrors: null });
    expect(fit.correlations[0]?.[0]).toBe(1);
    expect(fit.correlations[0]?.[1]).toBe(fit.correlations[1]?.[0]);
    expect(fit.items.map(({ id }) => id)).toEqual(["q1", "q2", "q3", "q4", "q5", "q6"]);
    expect(fit.items.every(({ standardErrors }) => standardErrors === null)).toBe(true);
    expect(fit.occasions.map(({ standardErrors }) => standardErrors)).toEqual([null, null]);
    expect(fit.correlationStandardErrors).toBeNull();
  });

  it("names the parameters each item estimated, leaving out those its model fixes", () => {
    // The 2PL holds c at 0 and d at 1; the 1PL's slope, which the items share, is estimated at each.
    expect(fit.items.map(({ estimated }) => estimated)).toEqual(Array.from({ length: 6 }, () => ["a", "b"]));
    fit.items.forEach(({ parameters }) => {
      expect(parameters).toMatchObject({ c: 0, d: 1 });
    });
    const shared = calibrateLongitudinal(occasions, {
      ...base,
      model: oneParameterLogistic,
      standardErrors: "hessian",
    });
    shared.items.forEach(({ estimated, standardErrors }) => {
      expect(estimated).toEqual(["a", "b"]);
      expect(Object.keys(defined(standardErrors))).toEqual(estimated);
    });
  });

  it("reports the change from the moments of the two occasions", () => {
    const [first, second] = fit.occasions;
    const rho = defined(fit.correlations[0]?.[1]);
    const change = defined(fit.changes[0]);
    expect([change.from, change.to]).toEqual([0, 1]);
    expectClose(change.mean, defined(second).mean - defined(first).mean, 1e-12);
    expectClose(change.sd ** 2, defined(first).sd ** 2 + defined(second).sd ** 2 - 2 * rho * defined(second).sd, 1e-12);
    expectClose(
      change.correlationWithStart,
      (rho * defined(second).sd - defined(first).sd) / change.sd,
      1e-12,
      "correlation with the start",
    );
  });

  it("matches persons and items by id, whatever their order", () => {
    const [first, second] = occasions;
    const rows = defined(second)
      .personIds.map((_, p) => p)
      .reverse();
    const columns = [5, 3, 1, 0, 2, 4];
    const reordered = buildResponseMatrix(
      rows.map((p) => columns.map((i) => defined(defined(second).responses[p]?.[i]))),
      {
        personIds: rows.map((p) => defined(defined(second).personIds[p])),
        itemIds: columns.map((i) => defined(defined(second).itemIds[i])),
      },
    );
    expectSameFit(calibrateLongitudinal([defined(first), reordered], base), fit);
  });

  it("treats a person who did not take part in an occasion as one without responses there", () => {
    const [first, second] = occasions;
    const kept = defined(second).personIds.slice(0, 250);
    const shortened = buildResponseMatrix(defined(second).responses.slice(0, 250), {
      personIds: kept,
      itemIds: [...defined(second).itemIds],
    });
    const blanked = buildResponseMatrix(
      defined(second).responses.map((row, p) => (p < 250 ? row : row.map(() => null))),
      { personIds: [...defined(second).personIds], itemIds: [...defined(second).itemIds] },
    );
    const absent = calibrateLongitudinal([defined(first), shortened], base);
    expectSameFit(calibrateLongitudinal([defined(first), blanked], base), absent);
    expect(absent.converged).toBe(true);
  });

  it("changes nothing for a person without a response on any occasion", () => {
    const withExtra = occasions.map((occasion) =>
      buildResponseMatrix([...occasion.responses, occasion.itemIds.map(() => null)], {
        personIds: [...occasion.personIds, "nobody"],
        itemIds: [...occasion.itemIds],
      }),
    );
    expect(withExtra).toHaveLength(2);
    expectSameFit(calibrateLongitudinal(withExtra, base), fit);
  });

  it("estimates items that only one occasion has", () => {
    const [first, second] = occasions;
    // The second occasion's q6 becomes q7: q6 is on the first occasion only and q7 on the second only.
    const renamed = buildResponseMatrix(defined(second).responses, {
      personIds: [...defined(second).personIds],
      itemIds: [...defined(second).itemIds.slice(0, 5), "q7"],
    });
    const anchored = calibrateLongitudinal([defined(first), renamed], base);
    expect(anchored.items.map(({ id }) => id)).toEqual(["q1", "q2", "q3", "q4", "q5", "q6", "q7"]);
    expect(anchored.converged).toBe(true);
    expectClose(defined(anchored.occasions[1]).mean, 0.4, 0.25);
  });

  it("links occasions through the others", () => {
    const three = simulateOccasions(2, 200, 6, { means: [0, 0.3, 0.6], sds: [1, 1.1, 1.2], rho: 0.8 });
    const keep = (matrix: ResponseMatrix, from: number, to: number): ResponseMatrix =>
      buildResponseMatrix(
        matrix.responses.map((row) => row.slice(from, to)),
        { personIds: [...matrix.personIds], itemIds: matrix.itemIds.slice(from, to) },
      );
    const chained = [keep(defined(three[0]), 0, 4), keep(defined(three[1]), 2, 6), keep(defined(three[2]), 3, 6)];
    // The first and the second share q3 and q4, the second and the third q4 to q6; the first and third share none.
    const chain = calibrateLongitudinal(chained, { points: 11 });
    expect(chain.occasions).toHaveLength(3);
    expect(chain.correlations).toHaveLength(3);
  });

  it("gives the slope that the 1PL's items share at each item, and each item's errors by name", () => {
    const shared = calibrateLongitudinal(occasions, {
      model: oneParameterLogistic,
      points: 21,
      standardErrors: "hessian",
    });
    const separate = calibrateLongitudinal(occasions, { points: 21, standardErrors: "hessian" });
    for (const fitted of [shared, separate]) {
      expect(fitted.items.map(({ standardErrors }) => Object.keys(defined(standardErrors)))).toEqual(
        Array.from({ length: 6 }, () => ["a", "b"]),
      );
      const [first, second] = fitted.occasions;
      expect(first?.standardErrors).toBeNull();
      expect(defined(second?.standardErrors).mean).toBeGreaterThan(0);
      expect(defined(second?.standardErrors).sd).toBeGreaterThan(0);
      const [[diagonal, below] = [], [above] = []] = defined(fitted.correlationStandardErrors);
      expect(diagonal).toBe(0);
      expect(above).toBeGreaterThan(0);
      expect(above).toBe(below);
    }
    const slopes = (fitted: LongitudinalCalibration) => fitted.items.map(({ standardErrors }) => standardErrors?.a);
    expect(new Set(slopes(shared)).size).toBe(1);
    expect(new Set(slopes(separate)).size).toBe(6);
  });

  it("gives Hessian and cross-product standard errors of one size, all positive", () => {
    const hessian = errorVector(calibrateLongitudinal(occasions, { points: 21, standardErrors: "hessian" }));
    const product = errorVector(calibrateLongitudinal(occasions, { points: 21, standardErrors: "cross-product" }));
    expect(product).toHaveLength(hessian.length);
    hessian.forEach((error, k) => {
      expect(error).toBeGreaterThan(0);
      expect(defined(product[k]) / error, `standard error ${k}`).toBeGreaterThan(0.7);
      expect(defined(product[k]) / error, `standard error ${k}`).toBeLessThan(1.4);
    });
  });

  it("stops at maxIterations without claiming convergence", () => {
    const stopped = calibrateLongitudinal(occasions, { points: 11, maxIterations: 2 });
    expect(stopped.converged).toBe(false);
    expect(stopped.iterations).toBe(2);
  });

  it("does not claim convergence when a distribution is narrower than the spacing of the nodes", () => {
    expect(calibrateLongitudinal(occasions, { points: 3 }).converged).toBe(false);
  });
});

describe("longitudinal calibration errors", () => {
  const occasions = simulateOccasions(3, 60, 4, SMALL);
  const first = defined(occasions[0]);
  const second = defined(occasions[1]);
  const options = { points: 7 } as const;
  const only = (matrix: ResponseMatrix, ids: readonly string[]): ResponseMatrix =>
    buildResponseMatrix(
      matrix.responses.map((row) => ids.map((id) => defined(row[matrix.itemIds.indexOf(id)]))),
      { personIds: [...matrix.personIds], itemIds: [...ids] },
    );

  it("takes two or three occasions", () => {
    expect(() => calibrateLongitudinal([first], options)).toThrow(
      "Longitudinal calibration takes 2 or 3 occasions, found 1",
    );
    expect(() => calibrateLongitudinal([first, second, first, second], options)).toThrow(
      "Longitudinal calibration takes 2 or 3 occasions, found 4",
    );
    expect(() => calibrateLongitudinal([], options)).toThrow("found 0");
  });

  it("needs response matrices with responses", () => {
    const empty = (matrix: ResponseMatrix): ResponseMatrix =>
      buildResponseMatrix(
        matrix.responses.map((row) => row.map(() => null)),
        { personIds: [...matrix.personIds], itemIds: [...matrix.itemIds] },
      );
    expect(() => calibrateLongitudinal([empty(first), empty(second)], options)).toThrow(
      "Calibration needs at least one response.",
    );
    expect(() => calibrateLongitudinal(JSON.parse('[{"responses": 1}, null]'), options)).toThrow(DataError);
  });

  it("needs occasions that share an item, directly or through the others", () => {
    const apart = only(second, ["q3", "q4"]);
    const renamed = buildResponseMatrix(apart.responses, { personIds: [...apart.personIds], itemIds: ["x1", "x2"] });
    expect(() => calibrateLongitudinal([first, renamed], options)).toThrow(
      "Occasion 2 shares no item with occasion 1, directly or through the other occasions",
    );
    const third = only(first, ["q1", "q2"]);
    expect(() => calibrateLongitudinal([first, third, renamed], options)).toThrow(
      "Occasion 3 shares no item with occasion 1",
    );
  });

  it("needs an item with responses in two categories, and every category up to its highest", () => {
    const constant = buildResponseMatrix(
      first.responses.map((row) => row.map((response, i) => (i === 0 ? 0 : response))),
      { personIds: [...first.personIds], itemIds: [...first.itemIds] },
    );
    const alike = buildResponseMatrix(
      second.responses.map((row) => row.map((response, i) => (i === 0 ? 0 : response))),
      { personIds: [...second.personIds], itemIds: [...second.itemIds] },
    );
    expect(() => calibrateLongitudinal([constant, alike], options)).toThrow(
      'Item "q1" has responses in fewer than two categories',
    );
  });

  it("takes only the models that it validates", () => {
    expect(() => calibrateLongitudinal([first, second], { model: threeParameterLogistic })).toThrow(
      "The option model must be oneParameterLogistic, twoParameterLogistic or gradedResponse",
    );
    expect(() => calibrateLongitudinal([first, second], { model: generalizedPartialCredit })).toThrow(DataError);
  });

  it("checks its options", () => {
    expect(() => calibrateLongitudinal([first, second], { points: 1 })).toThrow(
      "The option points must be an integer from 2, found 1.",
    );
    expect(() => calibrateLongitudinal([first, second], { points: 2.5 })).toThrow(DataError);
    expect(() => calibrateLongitudinal([first, second], { points: 1001 })).toThrow("has more than 1000000 nodes");
    expect(() => calibrateLongitudinal([first, second], { ...options, tolerance: -1 })).toThrow(DataError);
    expect(() => calibrateLongitudinal([first, second], { ...options, maxIterations: 0 })).toThrow(DataError);
    expect(() => calibrateLongitudinal([first, second], { ...options, standardErrors: JSON.parse('"louis"') })).toThrow(
      'The option standardErrors must be "none", "hessian" or "cross-product", found "louis".',
    );
  });

  it("rejects occasions and options of other types", () => {
    expect(() => calibrateLongitudinal(JSON.parse("null"))).toThrow(
      "The occasions must be an array of response matrices",
    );
    expect(() => calibrateLongitudinal(JSON.parse('{"length": 2}'))).toThrow(DataError);
    expect(() => calibrateLongitudinal([first, second], JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => calibrateLongitudinal([first, second], JSON.parse('{"verbose":1}'))).toThrow(
      'There is no option "verbose"',
    );
    expect(() => calibrateLongitudinal([first, second], { points: JSON.parse('"7"') })).toThrow(DataError);
    expect(() => calibrateLongitudinal([first, second], { ...options, tolerance: JSON.parse('"1e-4"') })).toThrow(
      DataError,
    );
    expect(() => calibrateLongitudinal([first, second], { model: JSON.parse('{"name":"2PL"}') })).toThrow(DataError);
  });
});

describe("rotation", () => {
  /** An exploratory calibration with the base R maximum of the fixture: loadings and items, uncorrelated. */
  const calibration: MultidimensionalCalibration<Parameters> = {
    dimensions: ["F1", "F2"],
    items: rotation.slopes.map((a, i) => ({
      id: `q${i + 1}`,
      model: multidimensionalTwoParameterLogistic,
      parameters: { a, intercept: defined(rotation.intercepts[i]), c: 0, d: 1 },
      standardErrors: null,
    })),
    correlations: [
      [1, 0],
      [0, 1],
    ],
    correlationStandardErrors: null,
    loadings: rotation.loadings,
    communalities: rotation.loadings.map((row) => row.reduce((sum, value) => sum + value * value, 0)),
    logLikelihood: 0,
    aic: 0,
    bic: 0,
    converged: true,
    iterations: 0,
  };

  function expectMatrix(
    actual: readonly (readonly number[])[],
    expected: readonly (readonly number[])[],
    tolerance: number,
    label: string,
  ): void {
    expected.forEach((row, i) => {
      row.forEach((value, j) => {
        expectClose(defined(actual[i]?.[j]), value, tolerance, `${label} ${i},${j}`);
      });
    });
  }

  for (const reference of rotation.rotations) {
    it(`matches GPArotation: ${JSON.stringify(reference.options)}`, () => {
      // The options of a fixture are untyped data, which rotateLoadings validates.
      const result = rotateLoadings(calibration, reference.options as RotationOptions);
      expect(result.converged).toBe(true);
      expectMatrix(result.loadings, reference.loadings, 1e-6, "loadings");
      expectMatrix(result.correlations, reference.correlations, 1e-6, "correlations");
      expectMatrix(result.rotation, reference.rotation, 1e-6, "rotation");
      result.items.forEach(({ parameters, id }, i) => {
        expect(id).toBe(`q${i + 1}`);
        expectMatrix(
          [(parameters as MultidimensionalLogisticParameters).a],
          [defined(reference.slopes[i])],
          1e-6,
          `slopes of item ${i + 1}`,
        );
      });
    });
  }

  it("keeps the intercepts and the communalities, and leaves the dimensions uncorrelated, under an orthogonal rotation", () => {
    const result = rotateLoadings(calibration, { method: "varimax" });
    expectMatrix(
      result.correlations,
      [
        [1, 0],
        [0, 1],
      ],
      1e-12,
      "correlations",
    );
    result.items.forEach(({ parameters }, i) => {
      expect((parameters as MultidimensionalLogisticParameters).intercept).toBe(defined(rotation.intercepts[i]));
      const row = defined(result.loadings[i]);
      expectClose(
        row.reduce((sum, value) => sum + value * value, 0),
        defined(calibration.communalities[i]),
        1e-12,
      );
    });
  });

  it("gives the same probabilities in the rotated dimensions", () => {
    for (const options of [{ method: "varimax" }, { method: "oblimin" }] as const) {
      const result = rotateLoadings(calibration, options);
      const [[t11, t12], [t21, t22]] = result.rotation as [[number, number], [number, number]];
      for (const theta of [
        [0.3, -0.7],
        [-1.4, 1.1],
        [2, 0.5],
      ] as const) {
        // The rotated dimensions are Tᵀ theta.
        const rotated = [t11 * theta[0] + t21 * theta[1], t12 * theta[0] + t22 * theta[1]];
        calibration.items.forEach((item, i) => {
          const original = new Float64Array(2);
          item.model.probabilities(item.parameters, Float64Array.from(theta), original);
          const other = defined(result.items[i]);
          const moved = new Float64Array(2);
          other.model.probabilities(other.parameters, Float64Array.from(rotated), moved);
          expectClose(moved[1] ?? Number.NaN, original[1] ?? 0, 1e-12, `item ${i + 1}`);
        });
      }
    }
  });

  it("scores in the rotated dimensions as mirt's fscores with a rotation does", () => {
    const result = rotateLoadings(calibration, { method: "oblimin" });
    rotation.scoring.patterns.forEach((pattern, p) => {
      const estimate = multidimensionalPosteriorMeanEstimate(result.items, pattern, {
        prior: { mean: [0, 0], covariance: result.correlations },
        points: rotation.scoring.points,
      });
      const expected = defined(rotation.scoring.eap[p]);
      // mirt rotates to its default tolerance of 1e-5.
      expected.theta.forEach((value, d) => {
        expectClose(defined(estimate.theta[d]), value, 1e-4, `pattern ${p + 1} theta ${d + 1}`);
        expectClose(
          defined(estimate.standardErrors[d]),
          defined(expected.standardErrors[d]),
          1e-4,
          `pattern ${p + 1} se ${d + 1}`,
        );
      });
    });
  });

  it("rotates with Kaiser normalization", () => {
    const result = rotateLoadings(calibration, { method: "varimax", normalize: true });
    expect(result.converged).toBe(true);
    expect(result.iterations).toBeLessThan(1000);
  });

  it("stops at the largest number of iterations and says so", () => {
    const result = rotateLoadings(calibration, { method: "oblimin", maxIterations: 1, tolerance: 1e-14 });
    expect(result.converged).toBe(false);
    expect(result.iterations).toBe(1);
  });

  it("flips each dimension so that its largest loading is positive", () => {
    for (const options of [{ method: "varimax" }, { method: "oblimin" }] as const) {
      const result = rotateLoadings(calibration, options);
      for (let d = 0; d < 2; d++) {
        const column = result.loadings.map((row) => defined(row[d]));
        const largest = column.reduce((best, value) => (Math.abs(value) > Math.abs(best) ? value : best), 0);
        expect(largest).toBeGreaterThan(0);
      }
    }
  });

  it("rejects options and calibrations it cannot rotate", () => {
    expect(() => rotateLoadings(calibration, { method: "promax" as never })).toThrow(
      'The option method must be "varimax" or "oblimin", found "promax".',
    );
    expect(() => rotateLoadings(calibration, JSON.parse("null"))).toThrow("The options must be an object, found null.");
    expect(() => rotateLoadings(calibration, JSON.parse('{"verbose":1}'))).toThrow('There is no option "verbose"');
    expect(() => rotateLoadings(calibration, { normalize: JSON.parse('"yes"') })).toThrow(DataError);
    expect(() => rotateLoadings(calibration, { gamma: JSON.parse('"0"') })).toThrow(DataError);
    expect(() => rotateLoadings(calibration, { tolerance: JSON.parse("null") })).toThrow(DataError);
    expect(() => rotateLoadings(calibration, { maxIterations: 0 })).toThrow(DataError);
    expect(() => rotateLoadings(JSON.parse("null"))).toThrow(
      "The calibration must be the result of calibrateMultidimensional, found null.",
    );
    expect(() => rotateLoadings({ ...calibration, items: JSON.parse("null") })).toThrow(DataError);
    expect(() => rotateLoadings({ ...calibration, loadings: JSON.parse("null") })).toThrow(DataError);
    expect(() => rotateLoadings({ ...calibration, correlations: JSON.parse("null") })).toThrow(DataError);
    expect(() => rotateLoadings({ ...calibration, correlations: [[1]], loadings: [[1]] })).toThrow(
      "Rotation needs at least two dimensions, found 1.",
    );
    expect(() =>
      rotateLoadings({
        ...calibration,
        correlations: [
          [1, 0.3],
          [0.3, 1],
        ],
      }),
    ).toThrow("Rotation needs the loadings of 12 items on 2 dimensions and the identity as the correlations");
    expect(() => rotateLoadings({ ...calibration, loadings: [[1, 0]] })).toThrow(DataError);
    expect(() => rotateLoadings({ ...calibration, items: [JSON.parse("null")] })).toThrow(DataError);
  });
});

describe("scoring", () => {
  const scored = [
    ["M2PL", m2pl],
    ["graded", graded],
  ] as const;

  for (const [name, fixture] of scored) {
    const items = itemsOf(fixture);
    const prior = { mean: [0, 0], covariance: fixture.correlations };

    it(`matches mirt's EAP for the ${name} model`, () => {
      fixture.scoring.patterns.forEach((pattern, p) => {
        const estimate = multidimensionalPosteriorMeanEstimate(items, pattern, {
          prior,
          points: fixture.scoring.points,
        });
        const expected = defined(fixture.scoring.eap[p]);
        expected.theta.forEach((value, d) => {
          expectClose(defined(estimate.theta[d]), value, 1e-8, `pattern ${p + 1} theta ${d + 1}`);
          expectClose(
            defined(estimate.standardErrors[d]),
            defined(expected.standardErrors[d]),
            1e-8,
            `pattern ${p + 1} se ${d + 1}`,
          );
        });
      });
    });

    it(`matches mirt's MAP for the ${name} model, and its standard errors from the expected information`, () => {
      fixture.scoring.patterns.forEach((pattern, p) => {
        const estimate = multidimensionalPosteriorModeEstimate(items, pattern, { prior });
        const expected = defined(fixture.scoring.map[p]);
        expected.theta.forEach((value, d) => {
          // mirt maximizes with nlm, to a tolerance of about 1e-5.
          expectClose(defined(estimate.theta[d]), value, 1e-5, `pattern ${p + 1} theta ${d + 1}`);
          expectClose(
            defined(estimate.standardErrors[d]),
            defined(expected.expectedStandardErrors[d]),
            1e-4,
            `pattern ${p + 1} se ${d + 1}`,
          );
        });
      });
    });
  }

  it("matches mirt's MAP standard errors where the observed and the expected information are equal", () => {
    const items = itemsOf(m2pl);
    m2pl.scoring.patterns.forEach((pattern, p) => {
      const estimate = multidimensionalPosteriorModeEstimate(items, pattern, {
        prior: { mean: [0, 0], covariance: m2pl.correlations },
      });
      const expected = defined(m2pl.scoring.map[p]);
      expected.standardErrors.forEach((value, d) => {
        expectClose(defined(estimate.standardErrors[d]), value, 1e-4, `pattern ${p + 1} se ${d + 1}`);
      });
    });
  });

  it("returns the covariance of the estimate with the standard errors on its diagonal", () => {
    const items = itemsOf(m2pl);
    const pattern = defined(m2pl.scoring.patterns[0]);
    for (const estimate of [
      multidimensionalPosteriorMeanEstimate(items, pattern),
      multidimensionalPosteriorModeEstimate(items, pattern),
    ]) {
      expect(estimate.covariance).toHaveLength(2);
      expect(estimate.covariance[0]?.[1]).toBeCloseTo(estimate.covariance[1]?.[0] ?? Number.NaN, 12);
      estimate.standardErrors.forEach((se, d) => {
        expect(se ** 2).toBeCloseTo(defined(estimate.covariance[d]?.[d]), 12);
      });
    }
  });

  it("agrees with the one-dimensional estimators on one dimension, under any prior", () => {
    const one = [
      { model: multidimensionalTwoParameterLogistic, parameters: { a: [1.3], intercept: -0.4, c: 0, d: 1 } },
      { model: multidimensionalTwoParameterLogistic, parameters: { a: [0.8], intercept: 0.6, c: 0, d: 1 } },
      { model: multidimensionalTwoParameterLogistic, parameters: { a: [1.9], intercept: 1.1, c: 0, d: 1 } },
    ];
    // Slope-intercept to the IRT form: b = -intercept / a.
    const unidimensional = one.map(({ parameters: { a, intercept } }) => ({
      model: twoParameterLogistic,
      parameters: { a: defined(a[0]), b: -intercept / defined(a[0]), c: 0, d: 1 },
    }));
    for (const pattern of [
      [1, 0, 1],
      [0, 0, 1],
      [1, 1, 1],
      [null, 0, 1],
    ]) {
      for (const [mean, sd] of [
        [0, 1],
        [0.4, 1.3],
        [-0.7, 0.6],
      ] as const) {
        const eap = multidimensionalPosteriorMeanEstimate(one, pattern, {
          prior: { mean: [mean], covariance: [[sd * sd]] },
        });
        const expectedEap = posteriorMeanEstimate(unidimensional, pattern, { prior: { mean, sd } });
        expectClose(defined(eap.theta[0]), expectedEap.theta, 1e-12, `EAP ${mean},${sd}`);
        expectClose(defined(eap.standardErrors[0]), expectedEap.standardError, 1e-12, `EAP se ${mean},${sd}`);
        const map = multidimensionalPosteriorModeEstimate(one, pattern, {
          prior: { mean: [mean], covariance: [[sd * sd]] },
        });
        const expectedMap = posteriorModeEstimate(unidimensional, pattern, { prior: { mean, sd } });
        expectClose(defined(map.theta[0]), expectedMap.theta, 1e-8, `MAP ${mean},${sd}`);
        expectClose(defined(map.standardErrors[0]), expectedMap.standardError, 1e-8, `MAP se ${mean},${sd}`);
      }
    }
  });

  it("returns the prior when there are no items, as the prior gives the dimensions", () => {
    const prior = {
      mean: [0.5, -0.5, 0.25],
      covariance: [
        [1, 0.3, 0],
        [0.3, 2, 0.1],
        [0, 0.1, 1.5],
      ],
    };
    const mode = multidimensionalPosteriorModeEstimate([], [], { prior });
    expect(mode.theta).toEqual(prior.mean);
    prior.covariance.forEach((row, i) => {
      row.forEach((value, j) => {
        expectClose(defined(mode.covariance[i]?.[j]), value, 1e-12, `mode covariance ${i + 1}, ${j + 1}`);
      });
    });
    expect(mode.standardErrors.map((value) => value ** 2)).toEqual(
      mode.covariance.map((row, i) => defined(row[i])).map((value) => expect.closeTo(value, 12)),
    );
    const mean = multidimensionalPosteriorMeanEstimate([], [], { prior, points: 15 });
    prior.mean.forEach((value, d) => {
      expectClose(defined(mean.theta[d]), value, 1e-12, `mean ${d + 1}`);
    });
    expectClose(defined(mean.covariance[0]?.[1]), 0.3, 1e-6);
  });

  it("needs a prior to estimate without items, as it gives the dimensions", () => {
    const message = "Scoring needs at least one item, or a prior, which gives the dimensions.";
    expect(() => multidimensionalPosteriorModeEstimate([], [])).toThrow(message);
    expect(() => multidimensionalPosteriorMeanEstimate([], [])).toThrow(message);
    expect(() => multidimensionalPosteriorModeEstimate([], [], { prior: { mean: [], covariance: [] } })).toThrow(
      message,
    );
    expect(() => multidimensionalPosteriorModeEstimate([], [1])).toThrow(message);
    const prior = {
      mean: [0, 0],
      covariance: [
        [1, 0],
        [0, 1],
      ],
    };
    expect(() => multidimensionalPosteriorModeEstimate([], [1], { prior })).toThrow("Found 1 responses to 0 items.");
    expect(() =>
      multidimensionalPosteriorModeEstimate([], [], { prior: JSON.parse('{"mean":[0,0],"covariance":"x"}') }),
    ).toThrow("The covariance of the prior must be an array of rows");
  });

  it("returns the prior when there are no responses", () => {
    const items = itemsOf(m2pl);
    const prior = {
      mean: [0.5, -0.5],
      covariance: [
        [1, 0.3],
        [0.3, 2],
      ],
    };
    const none = items.map(() => null);
    const mode = multidimensionalPosteriorModeEstimate(items, none, { prior });
    expect(mode.theta).toEqual([0.5, -0.5]);
    expectClose(defined(mode.covariance[1]?.[1]), 2, 1e-12);
    const mean = multidimensionalPosteriorMeanEstimate(items, none, { prior });
    expectClose(defined(mean.theta[0]), 0.5, 1e-12);
    expectClose(defined(mean.covariance[0]?.[1]), 0.3, 1e-6);
  });

  it("keeps finite estimates for extreme patterns", () => {
    const items = itemsOf(graded);
    const lowest = items.map(() => 0);
    const highest = items.map((item) => item.model.categoryCount(item.parameters) - 1);
    for (const pattern of [lowest, highest]) {
      const mode = multidimensionalPosteriorModeEstimate(items, pattern);
      const mean = multidimensionalPosteriorMeanEstimate(items, pattern);
      for (const estimate of [mode, mean]) {
        expect(estimate.theta.every(Number.isFinite)).toBe(true);
        expect(estimate.standardErrors.every((se) => Number.isFinite(se) && se > 0)).toBe(true);
      }
    }
  });

  it("estimates three dimensions and more with the posterior mode", () => {
    const item = (a: number[], intercept: number): MultidimensionalTestItem<Parameters> => ({
      id: `q${intercept}`,
      model: multidimensionalTwoParameterLogistic,
      parameters: { a, intercept, c: 0, d: 1 },
    });
    const items = Array.from({ length: 12 }, (_, i) =>
      item([i % 4 === 0 ? 1.2 : 0, i % 4 === 1 ? 1 : 0, i % 4 === 2 ? 1.5 : 0, i % 4 === 3 ? 0.9 : 0], (i - 6) / 4),
    );
    const estimate = multidimensionalPosteriorModeEstimate(
      items,
      items.map((_, i) => i % 2),
    );
    expect(estimate.theta).toHaveLength(4);
    expect(estimate.theta.every(Number.isFinite)).toBe(true);
    const grid = multidimensionalPosteriorMeanEstimate(
      items,
      items.map((_, i) => i % 2),
      { points: 15 },
    );
    expect(grid.theta).toHaveLength(4);
    // Close to the mode, for items of this much information.
    estimate.theta.forEach((value, d) => {
      expect(Math.abs(value - defined(grid.theta[d]))).toBeLessThan(0.15);
    });
  });

  it("finds the posterior mode in many dimensions with a scan that grows with them, not as a power of them", () => {
    // Items of one dimension each under N(0, I): the posterior is a product, and its mode the modes of the dimensions.
    const dimensions = 12;
    let evaluations = 0;
    const counted: MultidimensionalModel<MultidimensionalLogisticParameters> = {
      ...multidimensionalTwoParameterLogistic,
      probabilities(parameters, theta, out) {
        evaluations++;
        multidimensionalTwoParameterLogistic.probabilities(parameters, theta, out);
      },
    };
    const slopes = (j: number, size: number, at: number): number[] =>
      Array.from({ length: size }, (_, d) => (d === at ? 1 + (j % 3) / 4 : 0));
    const items = Array.from({ length: 2 * dimensions }, (_, j) => ({
      model: counted,
      parameters: { a: slopes(j, dimensions, j % dimensions), intercept: (j % 5) / 2 - 1, c: 0, d: 1 },
    }));
    const responses = items.map((_, j) => (j % 3 === 0 ? 0 : 1));
    const { theta } = multidimensionalPosteriorModeEstimate(items, responses);
    // A grid of 3 points per dimension would evaluate each item at 3 ** 12 = 531,441 points.
    expect(evaluations).toBeLessThan(100_000);
    theta.forEach((value, d) => {
      const own = [d, d + dimensions].map((j) => ({
        model: multidimensionalTwoParameterLogistic,
        parameters: { a: slopes(j, 1, 0), intercept: (j % 5) / 2 - 1, c: 0, d: 1 },
      }));
      const alone = multidimensionalPosteriorModeEstimate(
        own,
        [d, d + dimensions].map((j) => defined(responses[j])),
      );
      expectClose(value, defined(alone.theta[0]), 1e-8, `dimension ${d + 1}`);
    });
  });

  it("limits the grid of the posterior mean and names the posterior mode instead", () => {
    const item = {
      model: multidimensionalTwoParameterLogistic,
      parameters: { a: new Array<number>(8).fill(1), intercept: 0, c: 0, d: 1 },
    };
    expect(() => multidimensionalPosteriorMeanEstimate([item], [1], { points: 15 })).toThrow(
      "A grid of 15 points in 8 dimensions has more than 1000000 nodes; use fewer points or multidimensionalPosteriorModeEstimate.",
    );
    expect(multidimensionalPosteriorModeEstimate([item], [1]).theta).toHaveLength(8);
  });

  describe("estimates of the persons of a data set", () => {
    const items = itemsOf(m2pl);
    const data = buildResponseMatrix(
      [
        [1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0],
        new Array<number | null>(12).fill(null),
        [0, 0, 0, null, null, 1, 1, 1, 0, null, 1, 1],
      ],
      { itemIds: [...m2pl.itemIds] },
    );

    it("matches the estimator person by person, with null for a person without responses", () => {
      const results = multidimensionalEstimates(data, items);
      expect(results.map(({ id }) => id)).toEqual(data.personIds);
      expect(results[1]?.estimate).toBeNull();
      const direct = multidimensionalPosteriorModeEstimate(items, defined(data.responses[2]));
      expect(results[2]?.estimate).toEqual(direct);
    });

    it("gives the posterior means of the persons exactly as the estimator gives them one at a time", () => {
      const results = multidimensionalEstimates(data, items, { estimator: multidimensionalPosteriorMeanEstimate });
      expect(results[1]?.estimate).toBeNull();
      for (const p of [0, 2]) {
        expect(results[p]?.estimate).toEqual(
          multidimensionalPosteriorMeanEstimate(items, defined(data.responses[p]), { points: 31 }),
        );
      }
    });

    it("keeps the sums of log-likelihoods whether or not the log probabilities of the items are kept", () => {
      const grid = productGrid(2, 9);
      const kept = gridLikelihood(grid, Number.POSITIVE_INFINITY);
      const recomputed = gridLikelihood(grid, 0);
      const sums = [new Float64Array(grid.count), new Float64Array(grid.count)];
      for (const [item, response] of [
        [defined(items[0]), 1],
        [defined(items[3]), 0],
        [defined(items[0]), 0],
      ] as const) {
        kept.add(item, response, defined(sums[0]));
        recomputed.add(item, response, defined(sums[1]));
      }
      expect(sums[0]).toEqual(sums[1]);
      expect(defined(sums[0])[0]).toBeLessThan(0);
    });

    it("takes the estimator as an argument and sparse records", () => {
      const estimator = (it: readonly MultidimensionalItem[], responses: readonly (number | null)[]) =>
        multidimensionalPosteriorMeanEstimate(it, responses, { points: 21 });
      const dense = multidimensionalEstimates(data, items, { estimator });
      const records = buildResponseRecords([
        { person: "a", item: defined(m2pl.itemIds[0]), response: 1 },
        { person: "a", item: defined(m2pl.itemIds[7]), response: 0 },
        { person: "b", item: defined(m2pl.itemIds[3]), response: 1 },
      ]);
      const sparse = multidimensionalEstimates(records, items, { estimator });
      expect(sparse.map(({ id }) => id)).toEqual(["a", "b"]);
      const manual = multidimensionalPosteriorMeanEstimate([defined(items[0]), defined(items[7])], [1, 0], {
        points: 21,
      });
      expect(sparse[0]?.estimate).toEqual(manual);
      expect(dense).toHaveLength(3);
    });

    it("rejects items that do not match the data and an estimator that is not a function", () => {
      expect(() => multidimensionalEstimates(data, items.slice(1))).toThrow(
        `The data have item "${m2pl.itemIds[0]}", which is not among the items.`,
      );
      expect(() => multidimensionalEstimates(data, items, JSON.parse("null"))).toThrow(
        "The options must be an object, found null.",
      );
      expect(() => multidimensionalEstimates(data, items, JSON.parse('{"verbose":1}'))).toThrow(
        'There is no option "verbose"',
      );
      expect(() => multidimensionalEstimates(data, items, { estimator: JSON.parse("null") })).toThrow(DataError);
      expect(() => multidimensionalEstimates(data, [...items, defined(items[0])])).toThrow(
        `Item ids must be unique; "${m2pl.itemIds[0]}" appears twice.`,
      );
      expect(() => multidimensionalEstimates(data, JSON.parse("null"))).toThrow(
        "The items must be an array, found null.",
      );
    });
  });
});

/** One pattern of multidimensional-likelihood.json: its base R MLE and WLE and mirt's, `null` where mirt's is infinite. */
interface LikelihoodPattern {
  readonly pattern: readonly (number | null)[];
  readonly ml: readonly number[];
  readonly mlStandardErrors: readonly number[];
  readonly wle: readonly number[];
  readonly wleStandardErrors: readonly number[];
  readonly mirtMl: readonly (number | null)[];
  readonly mirtWle: readonly (number | null)[];
}

/** A case of multidimensional-likelihood.json: items set in mirt and the estimates of their patterns. */
interface LikelihoodCase {
  readonly model: string;
  readonly dimensions: number;
  readonly items: readonly FixtureParameters[];
  readonly estimates: readonly LikelihoodPattern[];
}

const likelihoodCases: readonly LikelihoodCase[] = likelihood.cases;
const LOWEST = defined(likelihood.range[0]);
const HIGHEST = defined(likelihood.range[1]);

function likelihoodItems({ model, items }: LikelihoodCase): MultidimensionalItem<Parameters>[] {
  const name = model.replace(/3$/, "");
  return items.map((parameters) => ({ model: modelOf(name), parameters: parametersOf(name, parameters) }));
}

/** The log-likelihood of a pattern at theta, written out from the probabilities of the categories. */
function patternLogLikelihood(
  items: readonly MultidimensionalItem<Parameters>[],
  pattern: readonly (number | null)[],
  theta: readonly number[],
): number {
  const probabilities = new Float64Array(8);
  const at = Float64Array.from(theta);
  return items.reduce((sum, item, i) => {
    const response = pattern[i];
    if (response === null || response === undefined) return sum;
    item.model.probabilities(item.parameters, at, probabilities);
    return sum + Math.log(defined(probabilities[response]));
  }, 0);
}

describe("maximum and weighted likelihood estimates", () => {
  for (const fixture of likelihoodCases) {
    const items = likelihoodItems(fixture);

    it(`matches the base R MLE and WLE in the range, and their standard errors: ${fixture.model}`, () => {
      fixture.estimates.forEach(({ pattern, ml, mlStandardErrors, wle, wleStandardErrors }, p) => {
        const maximum = multidimensionalMaximumLikelihoodEstimate(items, pattern);
        const weighted = multidimensionalWeightedLikelihoodEstimate(items, pattern);
        ml.forEach((value, d) => {
          expectClose(defined(maximum.theta[d]), value, 1e-8, `pattern ${p + 1} MLE ${d + 1}`);
          expectClose(
            defined(maximum.standardErrors[d]),
            defined(mlStandardErrors[d]),
            1e-8,
            `pattern ${p + 1} MLE se`,
          );
        });
        wle.forEach((value, d) => {
          expectClose(defined(weighted.theta[d]), value, 1e-8, `pattern ${p + 1} WLE ${d + 1}`);
          expectClose(
            defined(weighted.standardErrors[d]),
            defined(wleStandardErrors[d]),
            1e-8,
            `pattern ${p + 1} WLE se`,
          );
        });
      });
    });

    it(`matches mirt's MLE inside the range, and is on a bound of the range where mirt's is not: ${fixture.model}`, () => {
      fixture.estimates.forEach(({ pattern, mirtMl }, p) => {
        const { theta } = multidimensionalMaximumLikelihoodEstimate(items, pattern);
        // mirt gives -Inf, Inf or NaN, which the fixture writes as null, or stops short of convergence at ±20.
        if (mirtMl.every((value) => value !== null && value > LOWEST && value < HIGHEST)) {
          // mirt maximizes with nlm, to a tolerance of about 1e-5.
          mirtMl.forEach((value, d) => {
            expectClose(defined(theta[d]), defined(value), 1e-5, `pattern ${p + 1} theta ${d + 1}`);
          });
        } else {
          expect(
            theta.some((value) => value === LOWEST || value === HIGHEST),
            `pattern ${p + 1}`,
          ).toBe(true);
        }
      });
    });

    if (fixture.model !== "m3pl") {
      it(`matches mirt's WLE where it is Warm's estimator: ${fixture.model}`, () => {
        fixture.estimates.forEach(({ pattern, mirtWle }, p) => {
          const { theta } = multidimensionalWeightedLikelihoodEstimate(items, pattern);
          mirtWle.forEach((value, d) => {
            expectClose(defined(theta[d]), defined(value), 1e-5, `pattern ${p + 1} theta ${d + 1}`);
          });
        });
      });
    }
  }

  it("differs from mirt's WLE for the M3PL, which maximizes the likelihood times √det I", () => {
    const fixture = defined(likelihoodCases.find(({ model }) => model === "m3pl"));
    const items = likelihoodItems(fixture);
    const { pattern, mirtWle } = defined(fixture.estimates[0]);
    const { theta } = multidimensionalWeightedLikelihoodEstimate(items, pattern);
    expect(Math.abs(defined(theta[0]) - defined(mirtWle[0]))).toBeGreaterThan(0.05);
  });

  it("agrees with the one-dimensional MLE and WLE on one dimension, in any range", () => {
    const one: MultidimensionalItem<Parameters>[] = [
      { model: multidimensionalThreeParameterLogistic, parameters: { a: [1.3], intercept: -0.4, c: 0.2, d: 1 } },
      { model: multidimensionalTwoParameterLogistic, parameters: { a: [0.8], intercept: 0.6, c: 0, d: 1 } },
      { model: multidimensionalGradedResponse, parameters: { a: [1.6], intercepts: [1.5, 0.2, -1.1] } },
      { model: multidimensionalTwoParameterLogistic, parameters: { a: [1.9], intercept: 1.1, c: 0, d: 1 } },
    ];
    // Slope-intercept to the IRT form: b = -intercept / a.
    const unidimensional = [
      { model: threeParameterLogistic, parameters: { a: 1.3, b: 0.4 / 1.3, c: 0.2, d: 1 } },
      { model: twoParameterLogistic, parameters: { a: 0.8, b: -0.6 / 0.8, c: 0, d: 1 } },
      { model: gradedResponse, parameters: { a: 1.6, b: [-1.5 / 1.6, -0.2 / 1.6, 1.1 / 1.6] } },
      { model: twoParameterLogistic, parameters: { a: 1.9, b: -1.1 / 1.9, c: 0, d: 1 } },
    ];
    for (const pattern of [
      [1, 0, 2, 1],
      [0, 1, 1, null],
      [0, 0, 0, 0],
      [1, 1, 3, 1],
      [null, null, 3, 1],
    ]) {
      for (const range of [undefined, [-3, 2.5] as const]) {
        const options = range === undefined ? {} : { range };
        const ml = multidimensionalMaximumLikelihoodEstimate(one, pattern, options);
        const expectedMl = maximumLikelihoodEstimate(unidimensional, pattern, options);
        expectClose(defined(ml.theta[0]), expectedMl.theta, 1e-8, `MLE ${pattern}`);
        expectClose(defined(ml.standardErrors[0]), expectedMl.standardError, 1e-8, `MLE se ${pattern}`);
        const wle = multidimensionalWeightedLikelihoodEstimate(one, pattern, options);
        const expectedWle = weightedLikelihoodEstimate(unidimensional, pattern, options);
        expectClose(defined(wle.theta[0]), expectedWle.theta, 1e-8, `WLE ${pattern}`);
        expectClose(defined(wle.standardErrors[0]), expectedWle.standardError, 1e-8, `WLE se ${pattern}`);
      }
    }
  });

  it("puts the MLE of an extreme pattern at a corner of the range, and keeps its WLE inside", () => {
    const fixture = defined(likelihoodCases.find(({ model }) => model === "mgrm3"));
    const items = likelihoodItems(fixture);
    const range = [-4, 5] as const;
    const lowest = items.map(() => 0);
    const highest = items.map((item) => item.model.categoryCount(item.parameters) - 1);
    expect(multidimensionalMaximumLikelihoodEstimate(items, lowest, { range }).theta).toEqual([-4, -4, -4]);
    expect(multidimensionalMaximumLikelihoodEstimate(items, highest, { range }).theta).toEqual([5, 5, 5]);
    for (const pattern of [lowest, highest]) {
      const { theta, standardErrors } = multidimensionalWeightedLikelihoodEstimate(items, pattern, { range });
      expect(theta.every((value) => value > -4 && value < 5)).toBe(true);
      expect(standardErrors.every((se) => Number.isFinite(se) && se > 0)).toBe(true);
    }
  });

  it("finds the estimates in many dimensions, where the scan runs along the axes", () => {
    // Items of one dimension each: the likelihood is a product, and its maximum the maxima of the dimensions.
    const dimensions = 7;
    const slopes = (size: number, at: number, j: number): number[] =>
      Array.from({ length: size }, (_, d) => (d === at ? 0.8 + (j % 4) / 4 : 0));
    const items = Array.from({ length: 3 * dimensions }, (_, j) => ({
      model: multidimensionalTwoParameterLogistic,
      parameters: { a: slopes(dimensions, j % dimensions, j), intercept: (j % 5) / 2 - 1, c: 0, d: 1 },
    }));
    const responses = items.map((_, j) => (j % 4 === 0 ? 0 : 1));
    const ml = multidimensionalMaximumLikelihoodEstimate(items, responses);
    const wle = multidimensionalWeightedLikelihoodEstimate(items, responses);
    for (let d = 0; d < dimensions; d++) {
      const own = [d, d + dimensions, d + 2 * dimensions];
      const alone = own.map((j) => ({
        model: multidimensionalTwoParameterLogistic,
        parameters: { a: slopes(1, 0, j), intercept: (j % 5) / 2 - 1, c: 0, d: 1 },
      }));
      const answers = own.map((j) => defined(responses[j]));
      const expectedMl = multidimensionalMaximumLikelihoodEstimate(alone, answers);
      const expectedWle = multidimensionalWeightedLikelihoodEstimate(alone, answers);
      expectClose(defined(ml.theta[d]), defined(expectedMl.theta[0]), 1e-8, `MLE ${d + 1}`);
      expectClose(defined(wle.theta[d]), defined(expectedWle.theta[0]), 1e-8, `WLE ${d + 1}`);
    }
  });

  it("estimates the persons of a data set", () => {
    const items = itemsOf(m2pl);
    const data = buildResponseMatrix(
      [
        [1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0],
        [0, 0, 0, null, null, 1, 1, 1, 0, null, 1, 1],
      ],
      { itemIds: [...m2pl.itemIds] },
    );
    for (const estimator of [multidimensionalMaximumLikelihoodEstimate, multidimensionalWeightedLikelihoodEstimate]) {
      const results = multidimensionalEstimates(data, items, { estimator });
      results.forEach(({ estimate }, p) => {
        expect(estimate).toEqual(estimator(items, defined(data.responses[p])));
      });
    }
  });

  it("needs a response, and answered items whose slopes span every dimension", () => {
    const items = likelihoodItems(defined(likelihoodCases[0]));
    const none = items.map(() => null);
    expect(() => multidimensionalMaximumLikelihoodEstimate(items, none)).toThrow(
      "Maximum likelihood estimation needs at least one response.",
    );
    expect(() => multidimensionalWeightedLikelihoodEstimate([], [])).toThrow(
      "Weighted likelihood estimation needs at least one response.",
    );
    // The first five items load on the first dimension alone.
    const first = items.map((_, i) => (i < 5 ? 1 : null));
    for (const estimate of [multidimensionalMaximumLikelihoodEstimate, multidimensionalWeightedLikelihoodEstimate]) {
      expect(() => estimate(items, first)).toThrow(
        "estimation needs answered items whose slopes span the 2 dimensions: along a dimension they leave out, the likelihood is flat.",
      );
    }
    const all = items.map(() => 1);
    expect(() => multidimensionalMaximumLikelihoodEstimate(items, all, { range: [1, 1] })).toThrow(
      "The range needs the lower bound below the upper, found [1, 1].",
    );
    expect(() => multidimensionalWeightedLikelihoodEstimate(items, all, { range: [0, Number.NaN] })).toThrow(DataError);
  });

  it("has the largest likelihood of the range at the MLE", () => {
    const fixture = defined(likelihoodCases[0]);
    const items = likelihoodItems(fixture);
    fc.assert(
      fc.property(
        fc.array(fc.constantFrom(0, 1, null), { minLength: items.length, maxLength: items.length }),
        fc.array(fc.double({ min: LOWEST, max: HIGHEST, noNaN: true }), { minLength: 2, maxLength: 2 }),
        (pattern, other) => {
          let theta: readonly number[];
          try {
            theta = multidimensionalMaximumLikelihoodEstimate(items, pattern).theta;
          } catch (error) {
            expect(error).toBeInstanceOf(DataError);
            return;
          }
          expect(theta.every((value) => value >= LOWEST && value <= HIGHEST)).toBe(true);
          const at = patternLogLikelihood(items, pattern, theta);
          expect(at).toBeGreaterThanOrEqual(patternLogLikelihood(items, pattern, other) - 1e-12);
        },
      ),
      { numRuns: 200 },
    );
  });
});

/** Item parameters against mirt's two-dimensional model of the responses and their indicators. */
const NOT_AT_RANDOM_TOLERANCE = 1e-3;

describe("the recipe for responses missing not at random", () => {
  /** The recipe of multidimensional.md: the responses and an indicator item of whether each was answered. */
  function withIndicators(data: ResponseMatrix) {
    const itemIds = [...data.itemIds, ...data.itemIds.map((id) => `answered:${id}`)];
    const combined = data.responses.map((row) => [...row, ...row.map((response) => (response === null ? 0 : 1))]);
    const loadings = Object.fromEntries(
      itemIds.map((id, i) => [id, [i < data.itemIds.length ? "ability" : "propensity"]]),
    );
    return { matrix: buildResponseMatrix(combined, { itemIds }), loadings };
  }

  const items = notAtRandom.itemIds.length / 2;
  const omitted = buildResponseMatrix(
    notAtRandom.responses.map((row) => row.slice(0, items)),
    { itemIds: notAtRandom.itemIds.slice(0, items) },
  );
  const recipe = withIndicators(omitted);

  it("builds the combined responses and the loadings of the fixture", () => {
    expect(recipe.matrix.responses).toEqual(notAtRandom.responses);
    expect(Object.values(recipe.loadings)).toEqual(Object.values(notAtRandom.loadings));
    expect(omitted.responses.flat().filter((response) => response === null).length).toBeGreaterThan(1000);
  });

  const fit = calibrateMultidimensional(recipe.matrix, multidimensionalTwoParameterLogistic, {
    loadings: recipe.loadings,
    points: notAtRandom.points,
    tolerance: 1e-9,
    maxIterations: 2000,
  });

  it("matches the items, the correlation and the likelihood of mirt's two-dimensional model", () => {
    const fixture: Fixture = { ...notAtRandom, itemIds: recipe.matrix.itemIds };
    expectCalibration(
      fit,
      fixture,
      { parameters: NOT_AT_RANDOM_TOLERANCE, errors: NOT_AT_RANDOM_TOLERANCE, likelihood: NOT_AT_RANDOM_TOLERANCE },
      "not at random",
    );
    expect(fit.converged).toBe(true);
  });

  it("finds the correlation of the ability and the propensity that omits, near the 0.5 that made the data", () => {
    expectClose(defined(fit.correlations[0]?.[1]), notAtRandom.truth.correlation, 0.1, "correlation");
  });

  it("estimates the intercepts of the items nearer their truth than a calibration that skips the omissions", () => {
    const skipping = calibrateMultidimensional(omitted, multidimensionalTwoParameterLogistic, {
      loadings: Object.fromEntries(omitted.itemIds.map((id) => [id, ["ability"]])),
      points: notAtRandom.points,
    });
    const meanError = (calibration: MultidimensionalCalibration<Parameters>) =>
      mean(
        notAtRandom.truth.intercepts.map((truth, i) => {
          const parameters = defined(calibration.items[i]).parameters;
          return Math.abs(("intercept" in parameters ? parameters.intercept : Number.NaN) - truth);
        }),
      );
    expect(meanError(fit)).toBeLessThan(0.5 * meanError(skipping));
  });
});

describe("information", () => {
  for (const fixture of information.cases) {
    it(`matches the derivatives of the probabilities written out in base R: ${fixture.model}`, () => {
      const model = modelOf(fixture.model.replace(/3$/, ""));
      fixture.items.forEach((parameters, i) => {
        fixture.thetas.forEach((theta, j) => {
          const matrix = multidimensionalInformation(
            [{ model, parameters: parametersOf(fixture.model.replace(/3$/, ""), parameters) }],
            theta,
          );
          const expected = defined(fixture.matrices[i]?.[j]);
          expected.forEach((row, a) => {
            row.forEach((value, b) => {
              expectClose(defined(matrix[a]?.[b]), value, 1e-12, `item ${i + 1} at ${j + 1}: ${a},${b}`);
            });
          });
        });
      });
    });
  }

  it("sums the information of the items", () => {
    const items = itemsOf(graded).slice(0, 4);
    const theta = [0.2, -0.3];
    const total = multidimensionalInformation(items, theta);
    const parts = items.map((item) => multidimensionalInformation([item], theta));
    total.forEach((row, a) => {
      row.forEach((value, b) => {
        expectClose(
          value,
          parts.reduce((sum, part) => sum + defined(part[a]?.[b]), 0),
          1e-12,
        );
      });
    });
  });

  it("gives the information of the one-dimensional model on one dimension", () => {
    const [matrix] = multidimensionalInformation(
      [{ model: multidimensionalTwoParameterLogistic, parameters: { a: [1.4], intercept: -0.7, c: 0, d: 1 } }],
      [0.5],
    );
    const a = 1.4;
    const p = 1 / (1 + Math.exp(-(a * 0.5 - 0.7)));
    expectClose(defined(matrix?.[0]), a * a * p * (1 - p), 1e-14);
  });

  it("is 0 where the probabilities underflow, however large the slopes", () => {
    const steep = [
      { model: multidimensionalTwoParameterLogistic, parameters: { a: [1e200, 1e200], intercept: 0, c: 0, d: 1 } },
      { model: multidimensionalGradedResponse, parameters: { a: [1e200, 1], intercepts: [1, -1] } },
    ];
    for (const item of steep)
      expect(multidimensionalInformation([item], [1, 1])).toEqual([
        [0, 0],
        [0, 0],
      ]);
  });

  it("rejects items and thetas it cannot use", () => {
    const item = defined(itemsOf(m2pl)[0]);
    expect(() => multidimensionalInformation([], [0, 0])).toThrow("Information needs at least one item.");
    expect(() => multidimensionalInformation(JSON.parse("null"), [0, 0])).toThrow(
      "Information needs an array of items, found null.",
    );
    expect(() => multidimensionalInformation([item], JSON.parse("null"))).toThrow(
      "Theta must be an array, found null.",
    );
    expect(() => multidimensionalInformation([item], [0])).toThrow(
      "Theta needs 2 values for items on 2 dimensions, found 1.",
    );
    expect(() => multidimensionalInformation([item], [0, JSON.parse('"1"')])).toThrow(DataError);
    expect(() => multidimensionalInformation([item], [0, Number.NaN])).toThrow(DataError);
    expect(() =>
      multidimensionalInformation(
        [item, { model: multidimensionalTwoParameterLogistic, parameters: { a: [1], intercept: 0, c: 0, d: 1 } }],
        [0, 0],
      ),
    ).toThrow("Item 2 has 1 dimensions, but item 1 has 2.");
    expect(() => multidimensionalInformation([JSON.parse("null")], [0, 0])).toThrow(
      "Item 1 must be an item with a model and parameters, found null.",
    );
    expect(() => multidimensionalInformation([JSON.parse('{ "model": {}, "parameters": {} }')], [0, 0])).toThrow(
      "Item 1 needs a model such as multidimensionalTwoParameterLogistic, found an object.",
    );
    expect(() =>
      multidimensionalInformation(
        [{ model: multidimensionalTwoParameterLogistic, parameters: JSON.parse("null") }],
        [0, 0],
      ),
    ).toThrow("Item 1: An M2PL item needs finite slopes a");
  });
});

describe("multidimensional discrimination and difficulty", () => {
  for (const fixture of information.cases) {
    it(`matches mirt's MDISC and MDIFF: ${fixture.model}`, () => {
      const model = modelOf(fixture.model.replace(/3$/, ""));
      fixture.items.forEach((parameters, i) => {
        const item = { model, parameters: parametersOf(fixture.model.replace(/3$/, ""), parameters) };
        expectClose(multidimensionalDiscrimination(item), defined(fixture.discrimination[i]), 1e-15, `MDISC ${i + 1}`);
        const difficulty = multidimensionalDifficulty(item);
        expect(difficulty).toHaveLength(defined(fixture.difficulty[i]).length);
        difficulty.forEach((value, k) => {
          expectClose(value, defined(fixture.difficulty[i]?.[k]), 1e-15, `MDIFF ${i + 1}, ${k + 1}`);
        });
      });
    });
  }

  it("gives the slope and the difficulty of the one-dimensional model on one dimension", () => {
    const item = { model: multidimensionalTwoParameterLogistic, parameters: { a: [-1.6], intercept: 0.8, c: 0, d: 1 } };
    expect(multidimensionalDiscrimination(item)).toBe(1.6);
    expect(multidimensionalDifficulty(item)).toEqual([-0.5]);
  });

  it("gives the discrimination of a model implemented outside the package, and needs a built-in one for difficulty", () => {
    const custom = { ...multidimensionalTwoParameterLogistic, name: "Custom" };
    const item = { model: custom, parameters: { a: [3, 4], intercept: 1, c: 0, d: 1 } };
    expect(multidimensionalDiscrimination(item)).toBe(5);
    expect(() => multidimensionalDifficulty(item)).toThrow(
      'Only the built-in models have the intercepts of a multidimensional difficulty, found the "Custom" model.',
    );
  });

  it("refuses the difficulty of an item without slopes, and items it cannot read", () => {
    const flat = { model: multidimensionalGradedResponse, parameters: { a: [0, 0], intercepts: [1, -1] } };
    expect(multidimensionalDiscrimination(flat)).toBe(0);
    expect(() => multidimensionalDifficulty(flat)).toThrow(
      "An item whose slopes are all 0 has no direction, and so no multidimensional difficulty.",
    );
    expect(() => multidimensionalDiscrimination(JSON.parse("null"))).toThrow(
      "The item must be an item with a model and parameters, found null.",
    );
    expect(() =>
      multidimensionalDifficulty({ model: multidimensionalTwoParameterLogistic, parameters: JSON.parse('{"a": "1"}') }),
    ).toThrow("The item: An M2PL item needs finite slopes a");
  });
});

describe("errors", () => {
  const data = buildResponseMatrix(m2pl.responses.slice(0, 40), { itemIds: [...m2pl.itemIds] });
  const loadings: LoadingPattern = m2pl.loadings;
  const model = multidimensionalTwoParameterLogistic;

  it("takes at most 100 dimensions in calibration, so that every calibration can be scored", () => {
    const message = "Calibration takes at most 100 dimensions, found 101.";
    expect(() => calibrateMultidimensional(data, model, { dimensions: 101 })).toThrow(message);
    expect(() => calibrateMultidimensional(data, model, { dimensions: 101, method: "mhrm" })).toThrow(message);
    const names = Array.from({ length: 101 }, (_, d) => `D${d + 1}`);
    const wide = Object.fromEntries(m2pl.itemIds.map((id, i) => [id, i === 0 ? names : ["D1"]]));
    expect(() => calibrateMultidimensional(data, model, { loadings: wide, method: "mhrm" })).toThrow(message);
    // 100 specific dimensions of two items each, and the general one.
    const itemIds = Array.from({ length: 200 }, (_, i) => `q${i + 1}`);
    const testlets = buildResponseMatrix(
      Array.from({ length: 4 }, (_, p) => itemIds.map((_, i) => (p + i) % 2)),
      { itemIds },
    );
    const specific = Object.fromEntries(itemIds.map((id, i) => [id, `s${Math.floor(i / 2) + 1}`]));
    expect(() => calibrateBifactor(testlets, specific)).toThrow(
      "A bifactor or two-tier calibration takes at most 100 dimensions, found 101.",
    );
  });

  it("takes at most 100 dimensions in scoring, information and adaptive tests", () => {
    const item = (id: string, dimensions: number) => ({
      id,
      model,
      parameters: { a: new Array<number>(dimensions).fill(0.3), intercept: 0, c: 0, d: 1 },
    });
    const wide = [item("q1", 101), item("q2", 101)];
    const message = "Scoring takes at most 100 dimensions, found 101.";
    for (const estimate of [
      multidimensionalPosteriorModeEstimate,
      multidimensionalPosteriorMeanEstimate,
      multidimensionalMaximumLikelihoodEstimate,
      multidimensionalWeightedLikelihoodEstimate,
    ]) {
      expect(() => estimate(wide, [1, 0])).toThrow(message);
    }
    const prior = {
      mean: new Array<number>(101).fill(0),
      covariance: Array.from({ length: 101 }, (_, i) => Array.from({ length: 101 }, (__, j) => (i === j ? 1 : 0))),
    };
    expect(() => multidimensionalPosteriorModeEstimate([], [], { prior })).toThrow(message);
    expect(() => multidimensionalEstimates(buildResponseMatrix([[1, 0]], { itemIds: ["q1", "q2"] }), wide)).toThrow(
      message,
    );
    expect(() => multidimensionalInformation(wide, new Array<number>(101).fill(0))).toThrow(
      "Information takes at most 100 dimensions, found 101.",
    );
    expect(() => createMultidimensionalAdaptiveTest(wide, testLength(1))).toThrow(
      "A multidimensional adaptive test takes at most 100 dimensions, found 101.",
    );
    const limit = [item("q1", 100), item("q2", 100)];
    expect(multidimensionalPosteriorModeEstimate(limit, [1, 0]).theta).toHaveLength(100);
    expect(multidimensionalInformation(limit, new Array<number>(100).fill(0))).toHaveLength(100);
  });

  it("bounds standard errors to 2,000 parameters before it estimates anything, by EM, MH-RM and over occasions", () => {
    // Three persons answer 701 items with both categories each: two slopes and an intercept each, 2,103 parameters.
    const wide = buildResponseMatrix([0, 1, 2].map((k) => Array.from({ length: 701 }, (_, j) => (j + k) % 2)));
    const message = "Standard errors take at most 2000 estimated parameters, as their information holds every pair";
    for (const standardErrors of ["hessian", "cross-product"] as const) {
      expect(() => calibrateMultidimensional(wide, model, { dimensions: 2, standardErrors })).toThrow(message);
    }
    expect(() =>
      calibrateMultidimensional(wide, model, { dimensions: 2, method: "mhrm", standardErrors: "hessian" }),
    ).toThrow(message);
    // Two occasions of 1,001 items each: their 2PL items alone have 2,002 parameters.
    const occasion = (seed: number) =>
      buildResponseMatrix([0, 1, 2].map((k) => Array.from({ length: 1001 }, (_, j) => (j + k + seed) % 2)));
    expect(() => calibrateLongitudinal([occasion(0), occasion(1)], { standardErrors: "cross-product" })).toThrow(
      message,
    );
  });

  it("needs exactly one of dimensions and loadings", () => {
    expect(() => calibrateMultidimensional(data, model)).toThrow(
      "Calibration needs the option dimensions, for an exploratory model, or loadings, for a confirmatory one.",
    );
    expect(() => calibrateMultidimensional(data, model, { dimensions: 2, loadings })).toThrow(
      "Calibration takes the option dimensions or the option loadings, not both.",
    );
  });

  it("limits the probabilities of the items' categories that EM keeps at the nodes, before it allocates them", () => {
    // 17 dichotomous items on a grid of 10⁶ nodes would keep 3.4e7 probabilities, above 2 ** 24: 544 MB of tables.
    const wide = buildResponseMatrix([
      Array.from({ length: 17 }, (_, j) => j % 2),
      Array.from({ length: 17 }, (_, j) => 1 - (j % 2)),
    ]);
    const single = Object.fromEntries(wide.itemIds.map((id) => [id, ["F1"]]));
    expect(() =>
      calibrateMultidimensional(wide, model, { loadings: single, points: 1_000_000, maxIterations: 1 }),
    ).toThrow(
      "The items of the calibration would keep 34000000 values at the nodes of their grids, more than 16777216; use fewer points.",
    );
  });

  it("limits EM to three dimensions and needs at least as many items as dimensions", () => {
    expect(() => calibrateMultidimensional(data, model, { method: "em", dimensions: 4 })).toThrow(
      "EM calibrates up to 3 dimensions, found 4.",
    );
    const names = Object.fromEntries(m2pl.itemIds.map((id, i) => [id, [`d${i % 4}`]]));
    expect(() => calibrateMultidimensional(data, model, { method: "em", loadings: names })).toThrow(
      "EM calibrates up to 3 dimensions, found 4.",
    );
    expect(() =>
      calibrateMultidimensional(
        buildResponseMatrix([
          [0, 1],
          [1, 0],
          [1, 1],
        ]),
        model,
        { dimensions: 3 },
      ),
    ).toThrow("An exploratory model of 3 dimensions needs at least 3 items, found 2.");
  });

  it("rejects the method, the seed and the options that a method does not take", () => {
    const options = { loadings };
    expect(() => calibrateMultidimensional(data, model, { ...options, method: "mcmc" as never })).toThrow(
      'The option method must be "em" or "mhrm", found "mcmc".',
    );
    expect(() => calibrateMultidimensional(data, model, { ...options, method: JSON.parse("null") })).toThrow(DataError);
    expect(() => calibrateMultidimensional(data, model, { ...options, seed: 1.5 })).toThrow(
      "A seed must be a safe integer, found 1.5.",
    );
    expect(() => calibrateMultidimensional(data, model, { ...options, seed: JSON.parse('"7"') })).toThrow(DataError);
    expect(() => calibrateMultidimensional(data, model, { ...options, method: "mhrm", points: 15 })).toThrow(
      'The option points is for EM, which the method "mhrm" does not use.',
    );
    expect(() =>
      calibrateMultidimensional(data, model, { ...options, method: "mhrm", standardErrors: "cross-product" }),
    ).toThrow(
      'MH-RM has no posterior over a grid to cross-multiply: the option standardErrors needs "none" or "hessian" with the method "mhrm".',
    );
  });

  it("checks the pattern of loadings", () => {
    const first = defined(m2pl.itemIds[0]);
    expect(() => calibrateMultidimensional(data, model, { loadings: { ...loadings, nowhere: ["F1"] } })).toThrow(
      'Loadings are given for "nowhere", which is not an item id.',
    );
    const { [first]: _, ...without } = loadings;
    expect(() => calibrateMultidimensional(data, model, { loadings: without })).toThrow(
      `The loadings of item "${first}" must be an array of at least one dimension name, found undefined.`,
    );
    expect(() => calibrateMultidimensional(data, model, { loadings: { ...loadings, [first]: [] } })).toThrow(
      `The loadings of item "${first}" must be an array of at least one dimension name, found an array.`,
    );
    expect(() => calibrateMultidimensional(data, model, { loadings: { ...loadings, [first]: ["F1", "F1"] } })).toThrow(
      `Item "${first}" loads on a dimension twice: F1, F1.`,
    );
    expect(() => calibrateMultidimensional(data, model, { loadings: JSON.parse("null") })).toThrow(
      "The option loadings must be an object of dimension names by item id, found null.",
    );
    expect(() =>
      calibrateMultidimensional(data, model, { loadings: { ...loadings, [first]: JSON.parse('"F1"') } }),
    ).toThrow(DataError);
    expect(() =>
      calibrateMultidimensional(data, model, { loadings: { ...loadings, [first]: [JSON.parse("1")] } }),
    ).toThrow(DataError);
  });

  it("checks the other options", () => {
    const options = { loadings };
    expect(() => calibrateMultidimensional(data, model, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => calibrateMultidimensional(data, model, JSON.parse('{"verbose":1}'))).toThrow(
      'There is no option "verbose"',
    );
    expect(() => calibrateMultidimensional(JSON.parse("null"), model, options)).toThrow(DataError);
    expect(() =>
      calibrateMultidimensional(data, model, { ...options, dimensions: undefined, loadings: undefined }),
    ).toThrow(DataError);
    expect(() => calibrateMultidimensional(data, model, { dimensions: 1.5 })).toThrow(DataError);
    expect(() => calibrateMultidimensional(data, model, { dimensions: JSON.parse("null") })).toThrow(DataError);
    expect(() => calibrateMultidimensional(data, model, { ...options, correlated: JSON.parse('"yes"') })).toThrow(
      'The option correlated must be a boolean, found "yes".',
    );
    expect(() => calibrateMultidimensional(data, model, { ...options, points: 1 })).toThrow(
      "The option points must be an integer from 2, found 1.",
    );
    expect(() => calibrateMultidimensional(data, model, { ...options, points: JSON.parse('"31"') })).toThrow(DataError);
    expect(() => calibrateMultidimensional(data, model, { ...options, points: 2000 })).toThrow(
      "A grid of 2000 points in 2 dimensions has more than 1000000 nodes; use fewer points.",
    );
    expect(() => calibrateMultidimensional(data, model, { ...options, standardErrors: "bootstrap" as never })).toThrow(
      'The option standardErrors must be "none", "hessian" or "cross-product", found "bootstrap".',
    );
    expect(() => calibrateMultidimensional(data, model, { ...options, tolerance: -1 })).toThrow(DataError);
    expect(() => calibrateMultidimensional(data, model, { ...options, maxIterations: 0 })).toThrow(DataError);
    expect(() => calibrateMultidimensional(data, model, { ...options, priors: JSON.parse("null") })).toThrow(
      "The option priors must be an object of priors by parameter name, found null.",
    );
    expect(() => calibrateMultidimensional(data, model, { ...options, priors: { a: JSON.parse("{}") } })).toThrow(
      'The prior of "a" must be a prior such as normalPrior(0, 1), found an object.',
    );
    expect(() => calibrateMultidimensional(data, model, { ...options, priors: { d: normalPrior(1, 1) } })).toThrow(
      'A prior is given for "d", which no estimated item has.',
    );
  });

  it("checks the models", () => {
    const options = { loadings };
    expect(() => calibrateMultidimensional(data, JSON.parse("null"), { ...options })).toThrow(
      "The model must be a multidimensional model such as multidimensionalTwoParameterLogistic, found null.",
    );
    expect(() => calibrateMultidimensional(data, [model, model], { ...options })).toThrow(
      "Found 2 models for 12 items.",
    );
    expect(() => calibrateMultidimensional(data, JSON.parse("{}"), { ...options })).toThrow(DataError);
    // Binary data cannot be a graded item with a single category, and graded data need every category up to the highest.
    const polytomous = buildResponseMatrix([
      [0, 1],
      [1, 0],
      [2, 1],
      [2, 0],
    ]);
    expect(() => calibrateMultidimensional(polytomous, model, { dimensions: 1 })).toThrow(
      'Item "item1" has a response of 2, but the M2PL model allows responses from 0 to 1.',
    );
    const gap = buildResponseMatrix([
      [0, 1],
      [3, 0],
      [3, 1],
      [0, 0],
    ]);
    expect(() => calibrateMultidimensional(gap, multidimensionalGradedResponse, { dimensions: 1 })).toThrow(
      'Item "item1" has no response in category 1',
    );
    const constant = buildResponseMatrix([
      [0, 0],
      [0, 1],
      [0, 0],
    ]);
    expect(() => calibrateMultidimensional(constant, model, { dimensions: 1 })).toThrow(
      'Item "item1" has responses in fewer than two categories',
    );
  });

  it("needs at least one response", () => {
    expect(() =>
      calibrateMultidimensional(
        buildResponseMatrix([
          [null, null],
          [null, null],
        ]),
        model,
        { dimensions: 1 },
      ),
    ).toThrow("Calibration needs at least one response.");
  });
});

describe("input of other types", () => {
  const item = defined(itemsOf(m2pl)[0]);
  const items = itemsOf(m2pl);

  it("rejects options, items, responses and priors that are not what they should be", () => {
    expect(() => multidimensionalPosteriorMeanEstimate(items, [], JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => multidimensionalPosteriorMeanEstimate(items, [], JSON.parse('{"verbose":1}'))).toThrow(
      'There is no option "verbose"',
    );
    expect(() => multidimensionalPosteriorModeEstimate(items, [], JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => multidimensionalPosteriorModeEstimate(items, [], JSON.parse('{"verbose":1}'))).toThrow(
      'There is no option "verbose"',
    );
    expect(() => multidimensionalPosteriorModeEstimate(JSON.parse("null"), [])).toThrow(
      "Scoring needs an array of items, found null.",
    );
    expect(() => multidimensionalPosteriorModeEstimate([item], JSON.parse("null"))).toThrow(
      "The responses must be an array, found null.",
    );
    expect(() => multidimensionalPosteriorModeEstimate([item], [1, 0])).toThrow("Found 2 responses to 1 items.");
    expect(() => multidimensionalPosteriorModeEstimate([item], [2])).toThrow(
      "The response to item 1 must be null or an integer from 0 to 1, found 2.",
    );
    expect(() => multidimensionalPosteriorModeEstimate([item], [JSON.parse('"1"')])).toThrow(DataError);
    expect(() => multidimensionalPosteriorModeEstimate([item], [0.5])).toThrow(DataError);
    expect(() => multidimensionalPosteriorMeanEstimate([item], [1], { prior: JSON.parse("null") })).toThrow(
      "The option prior must be an object { mean, covariance }, found null.",
    );
    expect(() =>
      multidimensionalPosteriorMeanEstimate([item], [1], { prior: { mean: JSON.parse("null"), covariance: [[1]] } }),
    ).toThrow("The mean of the prior must be an array, found null.");
    expect(() =>
      multidimensionalPosteriorMeanEstimate([item], [1], { prior: { mean: [0, 0], covariance: JSON.parse("null") } }),
    ).toThrow("The covariance of the prior must be an array of rows, found null.");
    expect(() =>
      multidimensionalPosteriorMeanEstimate([item], [1], {
        prior: {
          mean: [0],
          covariance: [
            [1, 0],
            [0, 1],
          ],
        },
      }),
    ).toThrow("The prior needs a mean of 2 values and a covariance of 2 rows.");
    expect(() =>
      multidimensionalPosteriorMeanEstimate([item], [1], {
        prior: {
          mean: [0, JSON.parse('"0"')],
          covariance: [
            [1, 0],
            [0, 1],
          ],
        },
      }),
    ).toThrow(DataError);
    expect(() =>
      multidimensionalPosteriorMeanEstimate([item], [1], {
        prior: { mean: [0, 0], covariance: [JSON.parse("null"), [0, 1]] },
      }),
    ).toThrow(DataError);
    expect(() =>
      multidimensionalPosteriorMeanEstimate([item], [1], { prior: { mean: [0, 0], covariance: [[1], [0, 1]] } }),
    ).toThrow("Row 1 of the covariance of the prior needs 2 values, found 1.");
    expect(() =>
      multidimensionalPosteriorModeEstimate([item], [1], {
        prior: {
          mean: [0, 0],
          covariance: [
            [1, 0.5],
            [0.4, 1],
          ],
        },
      }),
    ).toThrow("The covariance of the prior must be a symmetric positive definite matrix.");
    expect(() =>
      multidimensionalPosteriorModeEstimate([item], [1], {
        prior: {
          mean: [0, 0],
          covariance: [
            [1, 2],
            [2, 1],
          ],
        },
      }),
    ).toThrow("The covariance of the prior must be a symmetric positive definite matrix.");
    expect(() => multidimensionalPosteriorMeanEstimate([item], [1], { points: 1 })).toThrow(
      "The option points must be an integer from 2, found 1.",
    );
    expect(() => multidimensionalPosteriorMeanEstimate([item], [1], { points: JSON.parse('"9"') })).toThrow(DataError);
  });
});

describe("properties", () => {
  /** Small simulated data: a matrix of persons whose order the properties permute. */
  const rows = m2pl.responses.slice(0, 120);
  const columns = [...m2pl.itemIds];

  it("gives estimates that do not depend on the order of the persons", () => {
    const options = {
      loadings: m2pl.loadings,
      points: 11,
      tolerance: 1e-10,
    };
    const fit = calibrateMultidimensional(
      buildResponseMatrix(rows, { itemIds: columns }),
      multidimensionalTwoParameterLogistic,
      options,
    );
    fc.assert(
      fc.property(fc.shuffledSubarray(rows, { minLength: rows.length, maxLength: rows.length }), (shuffled) => {
        const other = calibrateMultidimensional(
          buildResponseMatrix(shuffled, { itemIds: columns }),
          multidimensionalTwoParameterLogistic,
          options,
        );
        expectClose(other.logLikelihood, fit.logLikelihood, 1e-8);
        other.items.forEach(({ parameters }, i) => {
          expectClose(defined(parameters.a[i < 6 ? 0 : 1]), defined(fit.items[i]?.parameters.a[i < 6 ? 0 : 1]), 1e-6);
        });
        expectClose(defined(other.correlations[0]?.[1]), defined(fit.correlations[0]?.[1]), 1e-6);
      }),
      { numRuns: 3 },
    );
  });

  it("gives longitudinal estimates that do not depend on the order of the persons", () => {
    const data = simulateOccasions(11, 120, 5, SMALL);
    const options = { points: 15, tolerance: 1e-9 };
    const fit = calibrateLongitudinal(data, options);
    fc.assert(
      fc.property(
        fc.shuffledSubarray(
          Array.from({ length: 120 }, (_, p) => p),
          { minLength: 120, maxLength: 120 },
        ),
        (order) => {
          const shuffled = data.map((occasion) =>
            buildResponseMatrix(
              order.map((p) => defined(occasion.responses[p])),
              { personIds: order.map((p) => defined(occasion.personIds[p])), itemIds: [...occasion.itemIds] },
            ),
          );
          expectSameFit(calibrateLongitudinal(shuffled, options), fit);
        },
      ),
      { numRuns: 3 },
    );
  });

  it("gives estimates that do not depend on the order of the items", () => {
    const items = itemsOf(m2pl);
    fc.assert(
      fc.property(
        fc.array(fc.constantFrom(0, 1), { minLength: 12, maxLength: 12 }),
        fc.shuffledSubarray([...items.keys()], { minLength: 12, maxLength: 12 }),
        (pattern, order) => {
          const prior = { mean: [0, 0], covariance: m2pl.correlations };
          const estimate = multidimensionalPosteriorModeEstimate(items, pattern, { prior });
          const permuted = multidimensionalPosteriorModeEstimate(
            order.map((i) => defined(items[i])),
            order.map((i) => defined(pattern[i])),
            { prior },
          );
          estimate.theta.forEach((value, d) => {
            expectClose(defined(permuted.theta[d]), value, 1e-8);
          });
        },
      ),
      { numRuns: 20 },
    );
  });

  it("keeps the posterior mode inside the range of the data for a standard normal prior", () => {
    const items = itemsOf(m2pl);
    fc.assert(
      fc.property(fc.array(fc.constantFrom(0, 1), { minLength: 12, maxLength: 12 }), (pattern) => {
        const estimate = multidimensionalPosteriorModeEstimate(items, pattern);
        expect(estimate.theta.every((value) => Math.abs(value) < 6)).toBe(true);
        expect(estimate.standardErrors.every((se) => se > 0 && se <= 1)).toBe(true);
      }),
      { numRuns: 30 },
    );
  });
});
