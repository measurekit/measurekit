import { chiSquareUpperTail, normalQuantile, type ThetaDistribution, type ThresholdItem } from "@measurekit/core";
import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { createAdaptiveTest, simulateAdaptiveTest, testLength } from "../src/cat/index.js";
import { buildResponseMatrix, buildResponseRecords } from "../src/data/index.js";
import { itemFit } from "../src/fit/index.js";
import { createRandom, DataError } from "../src/index.js";
import {
  type AbilityEstimate,
  abilityEstimates,
  betaPrior,
  type Calibration,
  type CalibrationOptions,
  type ConditionalCalibration,
  calibrate,
  calibrateUnfolding,
  calibrateWithCovariance,
  conditionalMaximumLikelihood,
  conditionalStandardError,
  empiricalReliability,
  fourParameterLogistic,
  generalizedGradedUnfolding,
  generalizedPartialCredit,
  gradedResponse,
  type IrtItem,
  type IrtModel,
  itemInformation,
  jointMaximumLikelihood,
  type LatentDistribution,
  type LogisticItem,
  logNormalPrior,
  marginalReliability,
  maximumLikelihoodEstimate,
  nominalResponse,
  normalPrior,
  oneParameterLogistic,
  type ParameterCovariance,
  type ParameterPrior,
  partialCredit,
  posteriorMeanEstimate,
  posteriorModeEstimate,
  type RatingScaleParameters,
  ratingScale,
  type StandardErrorMethod,
  summedScoreTable,
  testInformation,
  thetaPosterior,
  threeParameterLogistic,
  twoParameterLogistic,
  type UnfoldingCalibration,
  type UnfoldingParameters,
  weightedLikelihoodEstimate,
} from "../src/irt/index.js";
import {
  categoryTallies,
  checkCategories,
  combine,
  compressResponses,
  type ExpectationItem,
  expectationStep,
  type Iteration,
  itemFunctions,
  logFunction,
  maximize,
  NO_ITEMS,
  NO_RESPONSES,
  population,
  quadratureRule,
  rescaledWeights,
  squarem,
  startAtHeldParameters,
  unfoldingStarts,
} from "../src/irt/internal.js";
import { rectangular } from "../src/math/internal.js";
import { normalDistribution, simulateItems, simulateResponses, simulateThetas } from "../src/sim/index.js";
import {
  arbitraryItem,
  arbitraryLocation,
  arbitraryLocations,
  arbitrarySlope,
  arbitraryUnfoldingItem,
} from "./arbitraries.js";
import { defined, expectClose, expectParameters } from "./assertions.js";
import { definedEntries, type FixtureErrors, type FixtureParameters, MODELS } from "./fixture-items.js";
import { type IrtFixedParametersFixture, readFixture } from "./fixtures.js";

interface IrtConditionalFixture {
  source: string;
  cases: {
    name: string;
    model: string;
    responses: (number | null)[][];
    logLikelihood: number;
    parameters: number;
    thresholds: (number | number[])[];
    standardErrors: (number | number[])[];
    locations?: number[];
    steps?: number[];
    locationErrors?: number[];
    stepErrors?: number[];
    erm?: { logLikelihood: number; atItsEstimates: number; thresholds: (number | number[])[] };
  }[];
}
const conditional = readFixture<IrtConditionalFixture>("irt-conditional.json");

// biome-ignore-start lint/style/useNamingConvention: the keys are those of the fixture file.
interface IrtCovarianceFixture {
  source: string;
  cases: {
    data: string;
    name: string;
    quadrature: { method: string; points: number };
    models: string[];
    covariance: { parameters: { item: string; name: string }[]; matrix: number[][] };
    priors?: { a: { type: string; meanLog: number; sdLog: number } };
    itemPriors?: {
      "Item.4": { a: { type: string; meanLog: number; sdLog: number } };
      "Item.2": { b: { type: string; mean: number; sd: number } };
    };
    items?: { parameters: { a: number; b: number; c: number; d: number } }[];
    logLikelihood?: number;
  }[];
}
// biome-ignore-end lint/style/useNamingConvention: the keys are those of the fixture file.
const covariance = readFixture<IrtCovarianceFixture>("irt-covariance.json");

interface IrtCrossProductFixture {
  source: string;
  cases: {
    itemIds: string[];
    responses: (number | null)[][];
    name: string;
    quadrature: { method: string; points: number };
    models: string[];
    items: {
      parameters: { a: number; b: number | number[]; c?: number; d?: number };
      standardErrors: { a: number; b?: number; b1?: number; b2?: number; b3?: number };
    }[];
    logLikelihood: number;
    aic: number;
    bic: number;
  }[];
}
const crossProduct = readFixture<IrtCrossProductFixture>("irt-cross-product.json");

const fixedParameters = readFixture<IrtFixedParametersFixture>("irt-fixed-parameters.json");

interface IrtLatentFixture {
  source: string;
  calibrations: {
    name: string;
    itemIds: string[];
    responses: (number | null)[][];
    groups: string[] | null;
    fixed: {
      q01?: { a: number; b: number; c: number; d: number };
      q02?: { a: number; b: number; c: number; d: number };
      q03?: { a: number; b: number; c: number; d: number };
      q04?: { a: number; b: number; c: number; d: number };
      op01?: { a: number; b: number; c: number; d: number };
      op02?: { a: number; b: number; c: number; d: number };
      op03?: { a: number; b: number; c: number; d: number };
      op04?: { a: number; b: number; c: number; d: number };
      op05?: { a: number; b: number; c: number; d: number };
      op06?: { a: number; b: number; c: number; d: number };
      op07?: { a: number; b: number; c: number; d: number };
      op08?: { a: number; b: number; c: number; d: number };
      op09?: { a: number; b: number; c: number; d: number };
      op10?: { a: number; b: number; c: number; d: number };
      op12?: { a: number; b: number; c: number; d: number };
      op13?: { a: number; b: number; c: number; d: number };
      op14?: { a: number; b: number; c: number; d: number };
      op15?: { a: number; b: number; c: number; d: number };
      op16?: { a: number; b: number; c: number; d: number };
      op17?: { a: number; b: number; c: number; d: number };
      op18?: { a: number; b: number; c: number; d: number };
      op19?: { a: number; b: number; c: number; d: number };
      op20?: { a: number; b: number; c: number; d: number };
      op21?: { a: number; b: number; c: number; d: number };
      op22?: { a: number; b: number; c: number; d: number };
      op23?: { a: number; b: number; c: number; d: number };
      op24?: { a: number; b: number; c: number; d: number };
      op25?: { a: number; b: number; c: number; d: number };
      op26?: { a: number; b: number; c: number; d: number };
      op29?: { a: number; b: number; c: number; d: number };
      op30?: { a: number; b: number; c: number; d: number };
      op31?: { a: number; b: number; c: number; d: number };
      op32?: { a: number; b: number; c: number; d: number };
      op33?: { a: number; b: number; c: number; d: number };
      op34?: { a: number; b: number; c: number; d: number };
      op35?: { a: number; b: number; c: number; d: number };
      op36?: { a: number; b: number; c: number; d: number };
      op37?: { a: number; b: number; c: number; d: number };
      op38?: { a: number; b: number; c: number; d: number };
      op39?: { a: number; b: number; c: number; d: number };
      op40?: { a: number; b: number; c: number; d: number };
    };
    distributions: { all?: string; reference?: string; focal?: string };
    items: { parameters: { a: number; b: number; c: number; d: number }; standardErrors?: { a: number; b: number } }[];
    groupDistributions: { group: string; mean: number; sd: number; standardErrors: { mean: number; sd: number } }[];
    logLikelihood: number;
    aic: number;
    bic: number;
  }[];
  histogram: {
    name: string;
    itemIds: string[];
    responses: (number | null)[][];
    fixed: Record<string, { a: number; b: number; c: number; d: number }>;
    tolerance: number;
    iterations: number;
    weights: number[];
    logLikelihood: number;
  };
}
const latent = readFixture<IrtLatentFixture>("irt-latent.json");

interface IrtFixedErrorCase {
  name: string;
  itemIds: string[];
  responses: (number | null)[][];
  latent: "normal" | "standard-normal";
  fixed: Record<string, LogisticItem>;
  fixedCovariance: ParameterCovariance;
  items: { id: string; a: number; b: number }[];
  groups: { mean: number; sd: number } | null;
  covariance: { parameters: { item: string; name: string }[]; matrix: number[][] };
  knownCovariance: number[][];
  groupErrors: { mean: number; sd: number } | null;
  knownGroupErrors: { mean: number; sd: number } | null;
}
const fixedError = readFixture<{ cases: IrtFixedErrorCase[] }>("irt-fixed-error.json");

interface IrtLatentRegressionFixture {
  source: string;
  cases: {
    name: string;
    model: string;
    itemIds: string[];
    responses: (number | null)[][];
    covariates: { x?: number[]; z?: number[]; group?: number[] };
    fixed: {
      q01?: { a: number; b: number; c: number; d: number };
      q02?: { a: number; b: number; c: number; d: number };
      q03?: { a: number; b: number; c: number; d: number };
      q04?: { a: number; b: number; c: number; d: number };
      q05?: { a: number; b: number; c: number; d: number };
      q06?: { a: number; b: number; c: number; d: number };
    };
    mirt: {
      items: { a: number; b: number }[];
      coefficients: { x?: number; z?: number; group?: number };
      residualVariance: number;
      logLikelihood: number;
    };
    hessianStandardErrors?: { group: number };
    tam?: { difficulties: number[]; coefficient: number; residualVariance: number };
  }[];
  polytomous: {
    name: string;
    model: string;
    itemIds: string[];
    responses: number[][];
    covariates: { x: number[] };
    fixed: Record<string, never>;
    mirt: {
      items: { a: number; b: number[] }[];
      coefficients: { x: number };
      residualVariance: number;
      logLikelihood: number;
    };
  };
}
const latentRegression = readFixture<IrtLatentRegressionFixture>("irt-latent-regression.json");

// biome-ignore-start lint/style/useNamingConvention: the keys are those of the fixture file.
interface IrtLsat7Fixture {
  source: string;
  itemIds: string[];
  responses: number[][];
  calibrations: {
    name: string;
    quadrature: { method: string; points: number };
    models: string[];
    items: {
      parameters: { a: number; b: number; c: number; d: number };
      standardErrors?: { a: number; b: number; c?: number };
    }[];
    logLikelihood: number;
    aic?: number;
    bic?: number;
    fixed?: {
      "Item.1": { a: number; b: number; c: number; d: number };
      "Item.2": { a: number; b: number; c: number; d: number };
    };
    priors?: { c: { type: string; alpha: number; beta: number } };
  }[];
  abilities: {
    models: string[];
    parameters: { a: number; b: number; c: number; d: number }[];
    patterns: (number | null)[][];
    posteriorMean: { theta: number; standardError: number }[];
    posteriorMode: { theta: number; standardError: number }[];
    maximumLikelihood: { theta: number; standardError: number }[];
    weightedLikelihood: { theta: number; standardError: number }[];
  }[];
  information: {
    models: string[];
    parameters: { a: number; b: number; c: number; d: number }[];
    theta: number[];
    items: number[][];
    test: number[];
  }[];
}
// biome-ignore-end lint/style/useNamingConvention: the keys are those of the fixture file.
const lsat7 = readFixture<IrtLsat7Fixture>("irt-lsat7.json");

interface IrtNestedModelsFixture {
  slopes: {
    source: string;
    itemIds: string[];
    responses: number[][];
    restricted: { logLikelihood: number; parameters: number };
    general: { logLikelihood: number; parameters: number };
    chiSquare: number;
    degreesOfFreedom: number;
    pValue: number;
  };
  credit: {
    source: string;
    itemIds: string[];
    responses: number[][];
    restricted: { logLikelihood: number; parameters: number };
    general: { logLikelihood: number; parameters: number };
    chiSquare: number;
    degreesOfFreedom: number;
    pValue: number;
  };
}
const nestedModels = readFixture<IrtNestedModelsFixture>("irt-nested-models.json");

interface IrtPersonWeightsFixture {
  source: string;
  cases: {
    twoPl: {
      description: string;
      itemIds: string[];
      responses: (number | null)[][];
      weights: number[];
      items: { a: number; b: number }[];
      hessianErrors: { a: number; b: number }[];
      crossProductErrors: { a: number; b: number }[];
      logLikelihood: number;
      bic: number;
    };
    graded: {
      description: string;
      itemIds: string[];
      responses: number[][];
      weights: number[];
      items: { a: number; b: number[] }[];
      logLikelihood: number;
    };
    groups: {
      description: string;
      itemIds: string[];
      responses: (number | null)[][];
      weights: number[];
      groups: string[];
      items: { a: number; b: number }[];
      focal: { mean: number; sd: number };
      logLikelihood: number;
      mirt: { focal: { mean: number; sd: number }; logLikelihood: number };
    };
    rasch: {
      description: string;
      itemIds: string[];
      responses: number[][];
      weights: number[];
      covariates: { x: number[] };
      tam: { difficulties: number[]; coefficient: number; residualVariance: number };
    };
    regression: {
      description: string;
      itemIds: string[];
      responses: (number | null)[][];
      weights: number[];
      covariates: { x: number[] };
      items: { a: number; b: number }[];
      coefficient: number;
      logLikelihood: number;
      mirtSurveyWeights: { coefficient: number; logLikelihood: number };
    };
  };
}
const weighted = readFixture<IrtPersonWeightsFixture>("irt-person-weights.json");

interface IrtPosteriorFixture {
  source: string;
  forms: {
    models: string[];
    parameters: { a: number; b: number | number[]; c?: number; d?: number }[];
    patterns: (number | null)[][];
    priors: { prior: { mean: number; sd: number }; nodes: number[]; weights: number[][] }[];
  }[];
}
const posterior = readFixture<IrtPosteriorFixture>("irt-posterior.json");

interface IrtScienceFixture {
  source: string;
  itemIds: string[];
  responses: number[][];
  calibrations: {
    name: string;
    quadrature: { method: string; points: number };
    models: string[];
    items: {
      parameters: { a: number | number[]; b?: number[] | number; steps?: number[]; c?: number[] };
      standardErrors: {
        a?: number;
        b1?: number;
        b2?: number;
        b3?: number;
        b?: number;
        step1?: number;
        step2?: number;
        a1?: number;
        a2?: number;
        c1?: number;
        c2?: number;
      };
    }[];
    logLikelihood: number;
    aic: number;
    bic: number;
  }[];
  abilities: {
    models: string[];
    parameters: { a: number | number[]; b?: number[] | number; steps?: number[]; c?: number[] }[];
    patterns: (number | null)[][];
    posteriorMean: { theta: number; standardError: number }[];
    posteriorMode: { theta: number; standardError: number }[];
    maximumLikelihood: { theta: number; standardError: number }[];
    weightedLikelihood: { theta: number; standardError: number }[];
  }[];
  information: {
    models: string[];
    parameters: { a: number | number[]; b?: number[] | number; steps?: number[]; c?: number[] }[];
    theta: number[];
    items: number[][];
    test: number[];
  }[];
}
const science = readFixture<IrtScienceFixture>("irt-science.json");

interface IrtScoringFixture {
  source: string;
  forms: {
    models: string[];
    parameters: { a: number; b: number | number[]; c?: number; d?: number }[];
    tables: {
      prior: { mean: number; sd: number };
      estimates: { score: number; theta: number; standardError: number; probability: number }[];
    }[];
    marginalReliability: { prior: { mean: number; sd: number }; value: number }[];
  }[];
  empiricalReliability: { estimates: { theta: number; standardError: number }[]; value: number };
}
const scoring = readFixture<IrtScoringFixture>("irt-scoring.json");

interface IrtSimulated1plFixture {
  source: string;
  itemIds: string[];
  responses: (number | null)[][];
  calibrations: {
    name: string;
    quadrature: { method: string; points: number };
    models: string[];
    items: { parameters: { a: number; b: number; c: number; d: number }; standardErrors: { a: number; b: number } }[];
    logLikelihood: number;
    aic: number;
    bic: number;
  }[];
  abilities: {
    models: string[];
    parameters: { a: number; b: number; c: number; d: number }[];
    patterns: (number | null)[][];
    posteriorMean: { theta: number; standardError: number }[];
    posteriorMode: { theta: number; standardError: number }[];
    maximumLikelihood: { theta: number; standardError: number }[];
    weightedLikelihood: { theta: number; standardError: number }[];
  }[];
}
const simulated1pl = readFixture<IrtSimulated1plFixture>("irt-simulated-1pl.json");

interface IrtSimulated4plFixture {
  source: string;
  itemIds: string[];
  responses: number[][];
  calibrations: {
    name: string;
    quadrature: { method: string; points: number };
    priors: { c: { type: string; mean: number; sd: number }; d: { type: string; mean: number; sd: number } };
    models: string[];
    items: {
      parameters: { a: number; b: number; c: number; d: number };
      standardErrors: { a: number; b: number; c: number; d: number };
    }[];
    logLikelihood: number;
    aic: number;
    bic: number;
  }[];
  abilities: {
    models: string[];
    parameters: { a: number; b: number; c: number; d: number }[];
    patterns: number[][];
    posteriorMean: { theta: number; standardError: number }[];
    posteriorMode: { theta: number; standardError: number }[];
    maximumLikelihood: { theta: number; standardError: number }[];
    weightedLikelihood: { theta: number; standardError: number }[];
  }[];
  information: {
    models: string[];
    parameters: { a: number; b: number; c: number; d: number }[];
    theta: number[];
    items: number[][];
    test: number[];
  }[];
}
const simulated4pl = readFixture<IrtSimulated4plFixture>("irt-simulated-4pl.json");

interface IrtSimulatedMixedFixture {
  source: string;
  itemIds: string[];
  responses: number[][];
  calibrations: {
    name: string;
    quadrature: { method: string; points: number };
    fixed: { q01: { a: number; b: number; c: number; d: number }; q10: { a: number; b: number[] } };
    priors: { c: { type: string; mean: number; sd: number } };
    models: string[];
    items: {
      parameters: { a: number; b: number | number[]; c?: number; d?: number };
      standardErrors?: { a: number; b?: number; c?: number; b1?: number; b2?: number; b3?: number };
    }[];
    logLikelihood: number;
    aic: number;
    bic: number;
  }[];
  abilities: {
    models: string[];
    parameters: { a: number; b: number | number[]; c?: number; d?: number }[];
    patterns: number[][];
    posteriorMean: { theta: number; standardError: number }[];
    posteriorMode: { theta: number; standardError: number }[];
    maximumLikelihood: { theta: number; standardError: number }[];
    weightedLikelihood: { theta: number; standardError: number }[];
  }[];
  information: {
    models: string[];
    parameters: { a: number; b: number | number[]; c?: number; d?: number }[];
    theta: number[];
    items: number[][];
    test: number[];
  }[];
}
const simulatedMixed = readFixture<IrtSimulatedMixedFixture>("irt-simulated-mixed.json");

interface IrtSimulatedPcmFixture {
  source: string;
  itemIds: string[];
  responses: (number | null)[][];
  jointMaximumLikelihood: {
    name: string;
    items: {
      model: string;
      parameters: { a: number; b: number | number[]; c?: number; d?: number };
      standardErrors: { b?: number; b1?: number; b2?: number; b3?: number };
    }[];
    persons: { theta: number; standardError: number }[];
    logLikelihood: number;
  };
}
const simulatedPcm = readFixture<IrtSimulatedPcmFixture>("irt-simulated-pcm.json");

// biome-ignore-start lint/style/useNamingConvention: the keys are those of the fixture file.
interface IrtTestletFixture {
  source: string;
  points: number;
  cases: {
    name: string;
    itemIds: string[];
    responses: number[][];
    testlets: { t1?: string[]; t2?: string[]; t3?: string[]; A?: string[]; C?: string[] };
    items: { parameters: { a: number; b: number; c: number; d: number }; standardErrors: { a: number; b: number } }[];
    sd: number[];
    sdStandardErrors: number[];
    logLikelihood: number;
    abilities: { theta: number; standardError: number }[];
  }[];
}
// biome-ignore-end lint/style/useNamingConvention: the keys are those of the fixture file.
const testlet = readFixture<IrtTestletFixture>("irt-testlet.json");

interface IrtTestletTablesFixture {
  source: string;
  forms: {
    name: string;
    items: { id: string; model: string; parameters: FixtureParameters }[];
    testlets: { items: string[]; sd: number }[];
  }[];
  tables: {
    form: string;
    tables: {
      prior: { mean: number; sd: number };
      estimates: { score: number; theta: number; standardError: number; probability: number }[];
    }[];
  }[];
}
const testletTables = readFixture<IrtTestletTablesFixture>("irt-testlet-tables.json");

interface IrtTestletGroupsFixture {
  source: string;
  points: number;
  cases: {
    name: string;
    itemIds: string[];
    responses: number[][];
    groups: string[];
    testlets: { t1: string[]; t2: string[]; t3: string[] };
    latent: { b: string; c?: string; a?: string };
    fixed: {
      q01?: { a: number; b: number; c: number; d: number };
      q02?: { a: number; b: number; c: number; d: number };
      q05?: { a: number; b: number; c: number; d: number };
      q09?: { a: number; b: number; c: number; d: number };
    };
    items: {
      parameters: { a: number; b: number; c: number; d: number };
      standardErrors: { a: number; b: number } | null;
    }[];
    sd: number[];
    sdStandardErrors: number[];
    distributions: { mean: number; sd: number; standardErrors: { mean: number; sd: number } | null }[];
    parameterCount: number;
    logLikelihood: number;
    mirtLogLikelihood?: number;
  }[];
  histogram: {
    name: string;
    itemIds: string[];
    responses: number[][];
    groups: string[];
    testlets: { t1: string[]; t2: string[]; t3: string[] };
    fixed: Record<string, { a: number; b: number; c: number; d: number }>;
    tolerance: number;
    iterations: number;
    weights: number[];
    sd: number[];
    logLikelihood: number;
  };
}
const testletGroups = readFixture<IrtTestletGroupsFixture>("irt-testlet-groups.json");

interface IrtUnfoldingFixture {
  source: string;
  cases: {
    mixed: {
      responses: string[];
      truth: { a: number; b: number; t: number[] }[];
      persons: number;
      expected: {
        items: {
          a: number;
          b: number;
          t: number[];
          standardErrors: { a: number; b: number; t1: number; t2?: number; t3?: number };
        }[];
        logLikelihood: number;
        crossProductErrors: { a: number; b: number; t1: number; t2?: number; t3?: number }[];
      };
    };
  };
  starts: Record<string, { b: number[]; t: number[][] }>;
  defaultTolerance: {
    case: string;
    items: { a: number; b: number; t: number[] }[];
    logLikelihood: number;
    tightLogLikelihood: number;
  };
}
const unfolding = readFixture<IrtUnfoldingFixture>("irt-unfolding.json");

// Tolerances against R, explained in docs/validation.md.
/** Item parameters against mirt, ltm and the base R posterior mode. */
const PARAMETER_TOLERANCE = 1e-5;
/** Relative tolerance of standard errors: numerical Hessians on both sides. */
const STANDARD_ERROR_TOLERANCE = 1e-4;
/** Log-likelihood, AIC and BIC at the estimates. */
const LOG_LIKELIHOOD_TOLERANCE = 1e-6;
/** The 4PL, polytomous and mixed-format calibrations: mirt's EM stops short on their flat likelihoods. */
const FLAT_PARAMETER_TOLERANCE = 2e-4;
const FLAT_STANDARD_ERROR_TOLERANCE = 1e-3;
const FLAT_LOG_LIKELIHOOD_TOLERANCE = 1e-4;
/** The log-likelihood at given parameters, with the same quadrature as mirt. */
const FIXED_LOG_LIKELIHOOD_TOLERANCE = 1e-8;
/** ML and MAP theta and standard errors against mirt's nlm, which uses numerical derivatives. */
const MODAL_TOLERANCE = 1e-5;
/** EAP and WLE against references with the same quadrature or an exact root. */
const EXACT_ABILITY_TOLERANCE = 1e-10;
/** Information against mirt; absolute, as mirt's 1 - P loses relative precision in the tails. */
const INFORMATION_TOLERANCE = 1e-12;
/** Calibrations with person weights against mirt and TAM, and a weighted latent regression against repeated persons. */
const WEIGHTED_TOLERANCE = 1e-3;
/** Integer weights against the persons repeated by them: the same sums, rounded in another order. */
const EXPANSION_TOLERANCE = 1e-9;
/** Relative tolerance of the standard errors of integer weights: numerical Hessians of two scaled likelihoods. */
const EXPANSION_STANDARD_ERROR_TOLERANCE = 1e-8;
/** Relative tolerance of weights times a constant, rescaled: a few rounding errors in each weight and in their sum. */
const RESCALED_WEIGHTS_TOLERANCE = 1e-13;
/** CML estimates, standard errors and log-likelihood against eRm, whose nlm stops short of the maximum. */
const ERM_TOLERANCE = 2e-4;
/** CML against base R's conditional likelihood by every pattern, maximized by optim to 1e-15. */
const CONDITIONAL_TOLERANCE = 1e-6;
/** JML estimates, standard errors and log-likelihood against the base R solution of TAM's steps. */
const JOINT_TOLERANCE = 1e-10;
/** EM tolerance of the fixture comparisons, mirt's TOL in validation/psychometrics/irt.R. */
const EM_TOLERANCE = 1e-10;
/** Estimated distributions of theta against the base R maximum: EM stops within 1e-8 of it. */
const LATENT_PARAMETER_TOLERANCE = 1e-7;
/** Relative tolerance of their standard errors: numerical Hessians on both sides. */
const LATENT_STANDARD_ERROR_TOLERANCE = 1e-6;
/** Items, coefficients and residual variances of a latent regression against mirt's EM, which stops near its TOL. */
const REGRESSION_TOLERANCE = 1e-3;
/** The standard error of a coefficient against base R's Hessian of the marginal log-likelihood, relative: mirt's estimates differ in the 5th digit. */
const REGRESSION_STANDARD_ERROR_TOLERANCE = 1e-3;
/** A histogram on fixed items against base R's EM with the same steps. */
const HISTOGRAM_TOLERANCE = 1e-12;
/** EAP summed-score tables against mirt and the base R recursion with the same nodes. */
const TABLE_TOLERANCE = 1e-12;
/** Marginal reliability by 61 nodes against integrate. */
const MARGINAL_RELIABILITY_TOLERANCE = 1e-8;
/** Testlet calibrations against the base R maximum, whose gradient is by central differences: EM stops within 1e-5. */
const TESTLET_PARAMETER_TOLERANCE = 2e-5;
/** Their standard errors, relative: numerical Hessians on both sides. */
const TESTLET_STANDARD_ERROR_TOLERANCE = 1e-4;
/** How far above measurekit's log-likelihood mirt's may be at its EM tolerance of 1e-8 on the parameters. */
const MIRT_LOG_LIKELIHOOD_SLACK = 1e-6;

/** Covariance matrices against the delta method on mirt's vcov and the base R Hessian, relative to √(Σ_ii Σ_jj). */
const COVARIANCE_TOLERANCE = 1e-4;
/** The GPCM's, where mirt's EM stops short of the optimum. */
const FLAT_COVARIANCE_TOLERANCE = 1e-3;
/** Posteriors of theta against the base R weights. */
const POSTERIOR_TOLERANCE = 1e-12;

/**
 * Unfolding calibrations against the GGUM package at tol = 1e-6, which integrates over 30 nodes from -4 to 4: they differ
 * by up to 3e-3 on an item far from the persons.
 */
const UNFOLDING_PARAMETER_TOLERANCE = 1e-2;
/** Their log-likelihood against base R at the GGUM package's estimates and measurekit's nodes. */
const UNFOLDING_LOG_LIKELIHOOD_TOLERANCE = 1e-2;
/** Their cross-product standard errors, relative, against base R at the GGUM package's estimates. */
const UNFOLDING_STANDARD_ERROR_TOLERANCE = 1e-2;
/** The 97.5% quantile of the standard normal distribution. */
const NORMAL_95 = 1.959963984540054;

/** Models whose calibrations mirt stops short of the optimum. */
const FLAT_MODELS = new Set(["4pl", "grm", "gpcm", "pcm", "rsm", "nrm"]);

function model(name: string): IrtModel {
  return defined(MODELS[name]);
}

/** A distribution of theta named in a fixture. */
function latentDistribution(name: string): LatentDistribution {
  if (name !== "standard-normal" && name !== "normal" && name !== "empirical-histogram") {
    throw new Error(`Unknown distribution ${name}.`);
  }
  return name;
}

/** A prior of the fixtures: a beta prior, or a normal prior on the logit scale. */
interface PriorDescription {
  readonly type: string;
  readonly alpha?: number;
  readonly beta?: number;
  readonly mean?: number;
  readonly sd?: number;
}

/**
 * A normal prior on logit(x), as mirt's `norm` on g and u. Implemented here as user code: its log density as a function
 * of x is -((logit(x) - mean) / sd)² / 2.
 */
function logitNormalPrior(mean: number, sd: number): ParameterPrior {
  const logit = (x: number): number => Math.log(x / (1 - x));
  const variance = sd * sd;
  return {
    logDensity: (x) => (-0.5 * (logit(x) - mean) ** 2) / variance,
    logDensityDerivative: (x) => -(logit(x) - mean) / (variance * x * (1 - x)),
    logDensitySecondDerivative: (x) => -(1 - (logit(x) - mean) * (1 - 2 * x)) / (variance * (x * (1 - x)) ** 2),
  };
}

function prior(description: PriorDescription): ParameterPrior {
  const { type, alpha = 0, beta = 0, mean = 0, sd = 0 } = description;
  return type === "beta" ? betaPrior(alpha, beta) : logitNormalPrior(mean, sd);
}

/** A prior of the covariance fixture: a normal or a log-normal prior. */
interface DescribedPrior {
  readonly type: string;
  readonly mean?: number;
  readonly sd?: number;
  readonly meanLog?: number;
  readonly sdLog?: number;
}

function describedPrior({ type, mean = 0, sd = 1, meanLog = 0, sdLog = 1 }: DescribedPrior): ParameterPrior {
  return type === "log-normal" ? logNormalPrior(meanLog, sdLog) : normalPrior(mean, sd);
}

/** Expects a covariance to have the labels of the reference and entries within `tolerance` of its correlation scale. */
function expectCovariance(
  actual: ParameterCovariance | null,
  expected: {
    readonly parameters: readonly { item: string; name: string }[];
    readonly matrix: readonly (readonly number[])[];
  },
  tolerance: number,
): void {
  const result = defined(actual);
  expect(result.parameters).toEqual(expected.parameters);
  expect(result.matrix).toHaveLength(expected.matrix.length);
  expected.matrix.forEach((row, i) => {
    row.forEach((value, j) => {
      const scale = Math.sqrt(defined(expected.matrix[i]?.[i]) * defined(expected.matrix[j]?.[j]));
      expectClose(defined(result.matrix[i]?.[j]), value, tolerance * scale, `covariance (${i + 1}, ${j + 1})`);
    });
  });
}

/** A calibration of a fixture. */
interface CalibrationFixture {
  readonly name: string;
  readonly models: readonly string[];
  readonly quadrature: { readonly method: string; readonly points: number };
  readonly fixed?: Readonly<Record<string, FixtureParameters>>;
  readonly priors?: Readonly<Record<string, PriorDescription>>;
  readonly items: readonly {
    readonly parameters: FixtureParameters;
    readonly standardErrors?: FixtureErrors;
  }[];
  readonly logLikelihood: number;
  readonly aic?: number;
  readonly bic?: number;
}

/** Items with one model per item, as the fixtures give them. */
interface FixturePool {
  readonly models: readonly string[];
  readonly parameters: readonly FixtureParameters[];
}

/** A fixture: responses with calibrations, abilities and information computed in R. */
interface IrtFixture {
  readonly source: string;
  readonly itemIds: readonly string[];
  readonly responses: readonly (readonly (number | null)[])[];
  readonly calibrations: readonly CalibrationFixture[];
  readonly abilities: readonly (FixturePool & {
    readonly patterns: readonly (readonly (number | null)[])[];
    readonly posteriorMean: readonly AbilityEstimate[];
    readonly posteriorMode: readonly AbilityEstimate[];
    readonly maximumLikelihood: readonly AbilityEstimate[];
    readonly weightedLikelihood: readonly AbilityEstimate[];
  })[];
  readonly information?: readonly (FixturePool & {
    readonly theta: readonly number[];
    readonly items: readonly (readonly number[])[];
    readonly test: readonly number[];
  })[];
}

const FIXTURES: readonly IrtFixture[] = [lsat7, simulated1pl, simulated4pl, science, simulatedMixed];

function fixtureModels(fixture: CalibrationFixture): IrtModel<unknown>[] {
  return fixture.models.map(model);
}

function calibrationOptions(fixture: CalibrationFixture): CalibrationOptions<unknown> {
  return {
    ...quadratureOf(fixture),
    ...(fixture.fixed && { fixed: fixture.fixed }),
    ...(fixture.priors && {
      priors: Object.fromEntries(Object.entries(fixture.priors).map(([name, value]) => [name, prior(value)])),
    }),
    tolerance: EM_TOLERANCE,
    maxIterations: 10_000,
    standardErrors: "hessian",
  };
}

function quadratureOf(fixture: CalibrationFixture): CalibrationOptions<unknown> {
  const method = fixture.quadrature.method === "gauss-hermite" ? "gauss-hermite" : "rectangular";
  return { quadrature: { method, points: fixture.quadrature.points } };
}

function fixtureItems({ models, parameters }: FixturePool): IrtItem[] {
  return models.map((name, i) => ({ model: model(name), parameters: defined(parameters[i]) }));
}

/** Names the models of a pool, such as `3PL and GPCM`. */
function modelNames({ models }: FixturePool): string {
  return [...new Set(models)].map((name) => name.toUpperCase()).join(" and ");
}

/** Returns the paths of the numbers in a result that are not finite, skipping the items' models. */
function nonFinite(value: unknown, path = "result"): string[] {
  if (typeof value === "number") return Number.isFinite(value) ? [] : [path];
  if (typeof value !== "object" || value === null) return [];
  return Object.entries(value).flatMap(([key, entry]) => (key === "model" ? [] : nonFinite(entry, `${path}.${key}`)));
}

/** Simulates responses to items with a seeded generator, theta ~ N(0, 1). */
function simulate(seed: number, persons: number, items: readonly IrtItem[]): readonly (readonly (number | null)[])[] {
  const random = createRandom(seed);
  return simulateResponses(items, simulateThetas(persons, random), random).responses;
}

const PARAMETERS_2PL: readonly LogisticItem[] = [
  { a: 1.2, b: -1, c: 0, d: 1 },
  { a: 0.8, b: 0, c: 0, d: 1 },
  { a: 1.5, b: 0.5, c: 0, d: 1 },
  { a: 1, b: 1.2, c: 0, d: 1 },
];
const ITEMS_2PL: readonly IrtItem<LogisticItem>[] = PARAMETERS_2PL.map((parameters) => ({
  model: twoParameterLogistic,
  parameters,
}));
const ITEMS_GPCM: readonly IrtItem<ThresholdItem>[] = [
  { model: generalizedPartialCredit, parameters: { a: 1.1, b: [-1, 0.3, 1.2] } },
  { model: generalizedPartialCredit, parameters: { a: 0.8, b: [-0.5, 0.9] } },
  { model: generalizedPartialCredit, parameters: { a: 1.4, b: [0.2, -0.6, 1.5] } },
];

describe("ability estimation", () => {
  const estimators = [
    ["posteriorMean", posteriorMeanEstimate, EXACT_ABILITY_TOLERANCE],
    ["posteriorMode", posteriorModeEstimate, MODAL_TOLERANCE],
    ["maximumLikelihood", maximumLikelihoodEstimate, MODAL_TOLERANCE],
    ["weightedLikelihood", weightedLikelihoodEstimate, EXACT_ABILITY_TOLERANCE],
  ] as const;

  for (const fixture of FIXTURES) {
    for (const abilities of fixture.abilities) {
      it(`matches R's EAP, MAP, ML and WLE estimates for the ${modelNames(abilities)} items of ${fixture.source}`, () => {
        const pool = fixtureItems(abilities);
        for (const [method, estimate, tolerance] of estimators) {
          abilities.patterns.forEach((pattern, row) => {
            const expected = defined(abilities[method][row]);
            const actual = estimate(pool, pattern);
            expectClose(actual.theta, expected.theta, tolerance, `${method} theta, pattern ${row + 1}`);
            expectClose(actual.standardError, expected.standardError, tolerance, `${method} SE, pattern ${row + 1}`);
          });
        }
      });
    }
  }

  it("puts the ML estimate at the end of the range when every response is in the lowest or highest category", () => {
    const high = maximumLikelihoodEstimate(ITEMS_2PL, [1, 1, 1, 1]);
    expect(high.theta).toBe(6);
    expectClose(high.standardError, conditionalStandardError(ITEMS_2PL, 6), 1e-12);
    expect(maximumLikelihoodEstimate(ITEMS_2PL, [0, 0, null, 0]).theta).toBe(-6);
    expect(maximumLikelihoodEstimate(ITEMS_2PL, [1, 1, 1, 1], { range: [-4, 4] }).theta).toBe(4);
    const mixed = [...ITEMS_2PL.slice(0, 2), ...ITEMS_GPCM];
    expect(maximumLikelihoodEstimate(mixed, [1, 1, 3, 2, 3]).theta).toBe(6);
    expect(maximumLikelihoodEstimate(mixed, [0, null, 0, 0, 0]).theta).toBe(-6);
  });

  it("gives a finite WLE for extreme 2PL patterns, inside the range", () => {
    for (const pattern of [
      [1, 1, 1, 1],
      [0, 0, 0, 0],
    ]) {
      const { theta, standardError } = weightedLikelihoodEstimate(ITEMS_2PL, pattern);
      expect(Math.abs(theta)).toBeLessThan(6);
      expectClose(standardError, conditionalStandardError(ITEMS_2PL, theta), 1e-12);
    }
  });

  it("gives the prior without responses for MAP and EAP, and rejects ML and WLE", () => {
    const none = [null, null, null, null];
    const thetaPrior = { mean: 0.5, sd: 2 };
    const map = posteriorModeEstimate(ITEMS_2PL, none, { prior: thetaPrior });
    expectClose(map.theta, 0.5, 1e-12);
    expectClose(map.standardError, 2, 1e-12);
    const eap = posteriorMeanEstimate(ITEMS_2PL, none, {
      prior: thetaPrior,
      quadrature: { method: "gauss-hermite", points: 21 },
    });
    expectClose(eap.theta, 0.5, 1e-12);
    expectClose(eap.standardError, 2, 1e-12);
    expect(() => maximumLikelihoodEstimate(ITEMS_2PL, none)).toThrow(
      "Maximum likelihood estimation needs at least one response.",
    );
    expect(() => weightedLikelihoodEstimate([], [])).toThrow(DataError);
  });

  it("estimates from a single item", () => {
    const one = [{ model: twoParameterLogistic, parameters: { a: 1, b: 0.5, c: 0, d: 1 } }];
    expect(maximumLikelihoodEstimate(one, [1]).theta).toBe(6);
    const wle = weightedLikelihoodEstimate(one, [1]);
    // For one 2PL item, Warm's equation (x - P) a + a (1 - 2P) / 2 = 0 has the root P = 3 / 4.
    expectClose(1 / (1 + Math.exp(-(wle.theta - 0.5))), 0.75, 1e-12);
  });

  it("estimates every person from response records or rows, by their answered items", () => {
    const items = ITEMS_2PL.map((item, i) => ({ ...item, id: `q${i + 1}` }));
    const rows = [
      [1, 0, null, 1],
      [null, null, null, null],
      [0, 0, 1, null],
    ];
    const matrix = buildResponseMatrix(rows, { itemIds: ["q4", "q3", "q2", "q1"] });
    const records = buildResponseRecords(
      [
        { person: "person1", item: "q4", response: 1 },
        { person: "person3", item: "q2", response: 1 },
        { person: "person1", item: "q1", response: 1 },
        { person: "person3", item: "q4", response: 0 },
        { person: "person1", item: "q3", response: 0 },
        { person: "person3", item: "q3", response: 0 },
      ],
      { personIds: matrix.personIds, itemIds: matrix.itemIds },
    );
    const expected = [
      { id: "person1", ...weightedLikelihoodEstimate([items[3], items[2], items[0]].map(defined), [1, 0, 1]) },
      { id: "person2", theta: null, standardError: null },
      { id: "person3", ...weightedLikelihoodEstimate([items[3], items[2], items[1]].map(defined), [0, 0, 1]) },
    ];
    expect(abilityEstimates(matrix, items, { estimator: weightedLikelihoodEstimate })).toEqual(expected);
    expect(abilityEstimates(records, items, { estimator: weightedLikelihoodEstimate })).toEqual(expected);
    expect(abilityEstimates(matrix, items)[0]).toEqual({
      id: "person1",
      ...posteriorMeanEstimate([items[3], items[2], items[0]].map(defined), [1, 0, 1]),
    });
    expect(() => abilityEstimates(matrix, items.slice(1))).toThrow(
      'The data have item "q1", which is not among the items.',
    );
    // Items are matched by id: a repeated id would leave one of them unused.
    expect(() => abilityEstimates(matrix, [...items, defined(items[0])])).toThrow(
      'Item ids must be unique; "q1" appears twice.',
    );
    expect(() =>
      buildResponseRecords([
        { person: "p", item: "q1", response: 1 },
        { person: "p", item: "q1", response: 0 },
      ]),
    ).toThrow('Person "p" has two records of item "q1"; a person answers an item once.');
    expect(() => buildResponseRecords([{ person: "p", item: "q1", response: 0.5 }])).toThrow(DataError);
  });

  it("gives every person the estimate of posteriorMeanEstimate by default, for dichotomous and polytomous items", () => {
    const named = (items: readonly IrtItem[]): (IrtItem & { id: string })[] =>
      items.map((item, i) => ({ ...item, id: `q${i + 1}` }));
    for (const items of [named(ITEMS_2PL), named(ITEMS_GPCM)]) {
      const patterns = items.reduce<(number | null)[][]>(
        (rows, item) =>
          rows.flatMap((row) =>
            [null, ...Array.from({ length: item.model.categoryCount(item.parameters) }, (_, k) => k)].map(
              (response) => [...row, response],
            ),
          ),
        [[]],
      );
      const estimates = abilityEstimates(
        buildResponseMatrix(patterns, { itemIds: items.map((item) => item.id) }),
        items,
      );
      patterns.forEach((pattern, person) => {
        const answered = items.flatMap((item, i) => (pattern[i] === null ? [] : [item]));
        const responses = pattern.filter((response) => response !== null);
        const estimate = estimates[person];
        if (answered.length === 0)
          expect(estimate).toEqual({ id: `person${person + 1}`, theta: null, standardError: null });
        else expect(estimate).toEqual({ id: `person${person + 1}`, ...posteriorMeanEstimate(answered, responses) });
      });
    }
    const items = named(ITEMS_2PL);
    expect(() =>
      abilityEstimates(buildResponseMatrix([[2, 0, 1, 0]], { itemIds: ["q1", "q2", "q3", "q4"] }), items),
    ).toThrow("The response to item 1 must be null or an integer from 0 to 1, found 2.");
  });

  it("rejects invalid items, responses, ranges and priors", () => {
    expect(() => maximumLikelihoodEstimate(ITEMS_2PL, [1, 0])).toThrow("Found 2 responses to 4 items.");
    expect(() => maximumLikelihoodEstimate(ITEMS_2PL, [1, 0, 2, 1])).toThrow(
      "The response to item 3 must be null or an integer from 0 to 1, found 2.",
    );
    expect(() => maximumLikelihoodEstimate(ITEMS_GPCM, [1, 3, 0])).toThrow(
      "The response to item 2 must be null or an integer from 0 to 2, found 3.",
    );
    const guessing = [{ model: twoParameterLogistic, parameters: { a: 1, b: 0, c: 0.2, d: 1 } }];
    expect(() => posteriorMeanEstimate(guessing, [1])).toThrow("Item 1: A 2PL item has c = 0, found c = 0.2.");
    expect(() => maximumLikelihoodEstimate(ITEMS_2PL, [1, 0, 1, 0], { range: [2, -2] })).toThrow(DataError);
    expect(() => posteriorModeEstimate(ITEMS_2PL, [1, 0, 1, 0], { prior: { mean: 0, sd: 0 } })).toThrow(DataError);
    expect(() => posteriorMeanEstimate(ITEMS_2PL, [1, 0, 1, 0], { quadrature: { points: 1 } })).toThrow(DataError);
    const withoutModel: IrtItem[] = JSON.parse('[{ "parameters": { "a": 1, "b": 0 } }]');
    expect(() => posteriorMeanEstimate(withoutModel, [1])).toThrow(
      "Item 1 needs a model such as twoParameterLogistic, found undefined.",
    );
    expect(() => abilityEstimates(buildResponseMatrix([[1]]), [], JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => abilityEstimates(buildResponseMatrix([[1]]), [], JSON.parse('{"verbose":1}'))).toThrow(
      'There is no option "verbose"',
    );
    expect(() => abilityEstimates(buildResponseMatrix([[1]]), [], { estimator: JSON.parse("null") })).toThrow(
      "The estimator must be a function of items and responses, such as posteriorMeanEstimate, found null.",
    );
  });

  it("rejects items, responses and options of other types from JavaScript", () => {
    const responses = [1, 0, 1, 0];
    expect(() => posteriorMeanEstimate(JSON.parse("null"), [])).toThrow("The items must be an array, found null.");
    expect(() => posteriorMeanEstimate(ITEMS_2PL, JSON.parse("null"))).toThrow(
      "The responses must be an array, found null.",
    );
    expect(() => posteriorMeanEstimate(JSON.parse("[null]"), [1])).toThrow(
      "Item 1 must be an item with a model and parameters, found null.",
    );
    const emptyModel: IrtItem[] = JSON.parse('[{ "model": {}, "parameters": {} }]');
    expect(() => posteriorMeanEstimate(emptyModel, [1])).toThrow(
      "Item 1 needs a model such as twoParameterLogistic, found an object.",
    );
    expect(() => posteriorMeanEstimate([{ model: twoParameterLogistic, parameters: JSON.parse("null") }], [1])).toThrow(
      "Item 1: Logistic item parameters need finite a and b",
    );
    expect(() => posteriorMeanEstimate(ITEMS_2PL, responses, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => posteriorMeanEstimate(ITEMS_2PL, responses, JSON.parse('{"verbose":1}'))).toThrow(
      'There is no option "verbose"',
    );
    expect(() => posteriorMeanEstimate(ITEMS_2PL, responses, { prior: JSON.parse("null") })).toThrow(
      "The option prior must be an object { mean, sd }, found null.",
    );
    expect(() => posteriorMeanEstimate(ITEMS_2PL, responses, { quadrature: JSON.parse("null") })).toThrow(
      "The option quadrature must be an object { method, points }, found null.",
    );
    expect(() => posteriorMeanEstimate(ITEMS_2PL, responses, { quadrature: JSON.parse('{"nodes":21}') })).toThrow(
      'The option quadrature may hold only method and points, found "nodes".',
    );
    expect(() =>
      posteriorMeanEstimate(ITEMS_2PL, responses, { prior: JSON.parse('{"mean":0,"sd":1,"df":3}') }),
    ).toThrow('The option prior may hold only mean and sd, found "df".');
    expect(() => posteriorMeanEstimate(ITEMS_2PL, responses, { testlets: JSON.parse("{}") })).toThrow(
      "The option testlets must be an array of testlets, found an object.",
    );
    expect(() => maximumLikelihoodEstimate(ITEMS_2PL, responses, { range: JSON.parse("null") })).toThrow(
      "The option range must be an array [lower, upper], found null.",
    );
    expect(() => weightedLikelihoodEstimate(ITEMS_2PL, responses, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => weightedLikelihoodEstimate(ITEMS_2PL, responses, JSON.parse('{"verbose":1}'))).toThrow(
      'There is no option "verbose"',
    );
    expect(() => posteriorModeEstimate(ITEMS_2PL, responses, { range: JSON.parse('[-6, "6"]') })).toThrow(
      'The upper bound of the range must be a finite number, found "6".',
    );
  });

  it("rejects more than 1,000 quadrature points, and takes 1,000", () => {
    expect(() => posteriorMeanEstimate(ITEMS_2PL, [1, 0, 1, 0], { quadrature: { points: 1001 } })).toThrow(
      "The quadrature points must be an integer from 2 to 1000, found 1001.",
    );
    const eap = posteriorMeanEstimate(ITEMS_2PL, [1, 0, 1, 0], { quadrature: { points: 1000 } });
    expect(Number.isFinite(eap.theta) && Number.isFinite(eap.standardError)).toBe(true);
  });

  it("gives each Gauss-Hermite rule its own nodes, whatever was asked for before, and no rule aliases the one kept", () => {
    const quadrature = (points: number) => ({ quadrature: { method: "gauss-hermite", points } }) as const;
    const nine = posteriorMeanEstimate(ITEMS_2PL, [1, 0, 1, 0], quadrature(9));
    const fifteen = posteriorMeanEstimate(ITEMS_2PL, [1, 0, 1, 0], quadrature(15));
    expect(fifteen.theta).not.toBe(nine.theta);
    expect(posteriorMeanEstimate(ITEMS_2PL, [1, 0, 1, 0], quadrature(9))).toEqual(nine);
    expect(posteriorMeanEstimate(ITEMS_2PL, [1, 0, 1, 0], quadrature(15))).toEqual(fifteen);
    const rule = quadratureRule({ method: "gauss-hermite", points: 9 });
    rule.nodes.fill(0);
    rule.weights.fill(0);
    expect(posteriorMeanEstimate(ITEMS_2PL, [1, 0, 1, 0], quadrature(9))).toEqual(nine);
  });

  it("rejects a range wider than 10,000, whose scan would not end", () => {
    // The width of [-1e308, 1e308] overflows to Infinity, and the scan in steps of 0.25 would never reach its end.
    for (const range of [
      [-5000.5, 5000],
      [-1e308, 1e308],
    ] as const) {
      expect(() => maximumLikelihoodEstimate(ITEMS_2PL, [1, 0, 1, 0], { range })).toThrow(
        `The range must be at most 10000 wide, found [${range[0]}, ${range[1]}].`,
      );
    }
    expect(maximumLikelihoodEstimate(ITEMS_2PL, [1, 1, 1, 1], { range: [-5000, 5000] }).theta).toBeGreaterThan(6);
  });
});

describe("thetaPosterior", () => {
  for (const form of posterior.forms) {
    it(`matches the base R posterior of theta for the ${modelNames(form)} items`, () => {
      const pool = fixtureItems(form);
      for (const { prior, nodes, weights } of form.priors) {
        form.patterns.forEach((pattern, row) => {
          const actual = thetaPosterior(pool, pattern, { prior });
          expect(actual.nodes).toHaveLength(nodes.length);
          nodes.forEach((node, q) => {
            expectClose(defined(actual.nodes[q]), node, 1e-13, `node ${q}`);
          });
          defined(weights[row]).forEach((weight, q) => {
            expectClose(defined(actual.weights[q]), weight, POSTERIOR_TOLERANCE, `pattern ${row + 1}, node ${q}`);
          });
        });
      }
    });
  }

  it("has the mean and standard deviation of the EAP estimate, with any prior and rule", () => {
    const quadratures = [{}, { method: "gauss-hermite", points: 21 }, { points: 15 }] as const;
    const patterns = fc.tuple(...ITEMS_2PL.map(() => fc.constantFrom(0, 1, null)));
    fc.assert(
      fc.property(
        patterns,
        fc.double({ min: -1, max: 1, noNaN: true }),
        fc.double({ min: 0.3, max: 2, noNaN: true }),
        fc.constantFrom(...quadratures),
        (responses, mean, sd, quadrature) => {
          const options = { prior: { mean, sd }, quadrature };
          const { nodes, weights } = thetaPosterior(ITEMS_2PL, responses, options);
          expectClose(
            weights.reduce((sum, weight) => sum + weight, 0),
            1,
            1e-12,
          );
          expect(weights.every((weight) => weight >= 0)).toBe(true);
          const eap = posteriorMeanEstimate(ITEMS_2PL, responses, options);
          const theta = weights.reduce((sum, weight, q) => sum + weight * defined(nodes[q]), 0);
          const variance = weights.reduce((sum, weight, q) => sum + weight * (defined(nodes[q]) - theta) ** 2, 0);
          expectClose(theta, eap.theta, 1e-12);
          expectClose(Math.sqrt(variance), eap.standardError, 1e-12);
        },
      ),
    );
  });

  it("is the prior without responses, as a calibration's group distribution describes it", () => {
    const sample = simulate(20261029, 100, ITEMS_2PL);
    const group: ThetaDistribution = defined(calibrate(buildResponseMatrix(sample), twoParameterLogistic).groups[0]);
    const none = thetaPosterior(ITEMS_2PL, [null, null, null, null]);
    group.nodes.forEach((node, q) => {
      expectClose(defined(none.nodes[q]), node, 1e-14);
    });
    group.weights.forEach((weight, q) => {
      expectClose(defined(none.weights[q]), weight, 1e-15);
    });
  });

  it("concentrates where the responses point, and narrows with more of them", () => {
    const high = thetaPosterior(ITEMS_2PL, [1, 1, 1, 1]);
    const low = thetaPosterior(ITEMS_2PL, [0, 0, 0, 0]);
    const spread = (distribution: ThetaDistribution): number => {
      const theta = distribution.weights.reduce((sum, weight, q) => sum + weight * defined(distribution.nodes[q]), 0);
      return distribution.weights.reduce(
        (sum, weight, q) => sum + weight * (defined(distribution.nodes[q]) - theta) ** 2,
        0,
      );
    };
    const mode = (distribution: ThetaDistribution): number =>
      defined(distribution.nodes[distribution.weights.indexOf(Math.max(...distribution.weights))]);
    expect(mode(high)).toBeGreaterThan(0.5);
    expect(mode(low)).toBeLessThan(-0.5);
    expect(spread(thetaPosterior(ITEMS_2PL, [1, 0, 1, 0]))).toBeLessThan(
      spread(thetaPosterior(ITEMS_2PL, [1, null, null, null])),
    );
  });

  it("is plain data that JSON serializes", () => {
    const result = thetaPosterior(ITEMS_GPCM, [1, null, 3]);
    expect(JSON.parse(JSON.stringify(result))).toEqual(result);
  });

  it("rejects invalid items, responses, prior, quadrature and options", () => {
    expect(() => thetaPosterior(ITEMS_2PL, [1, 0])).toThrow("Found 2 responses to 4 items.");
    expect(() => thetaPosterior(ITEMS_2PL, [1, 0, 2, 1])).toThrow(
      "The response to item 3 must be null or an integer from 0 to 1, found 2.",
    );
    expect(() => thetaPosterior(ITEMS_2PL, [1, 0, 1, 0], { prior: { mean: 0, sd: 0 } })).toThrow(DataError);
    expect(() => thetaPosterior(ITEMS_2PL, [1, 0, 1, 0], { quadrature: { points: 1 } })).toThrow(
      "The quadrature points must be an integer from 2, found 1.",
    );
    expect(() => thetaPosterior(JSON.parse("null"), [])).toThrow("The items must be an array, found null.");
    expect(() => thetaPosterior(ITEMS_2PL, JSON.parse("null"))).toThrow("The responses must be an array, found null.");
    expect(() => thetaPosterior(ITEMS_2PL, [1, 0, 1, 0], JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => thetaPosterior(ITEMS_2PL, [1, 0, 1, 0], JSON.parse('{"verbose":1}'))).toThrow(
      'There is no option "verbose"',
    );
    expect(() => thetaPosterior(ITEMS_2PL, [1, 0, 1, 0], { prior: JSON.parse('"normal"') })).toThrow(
      'The option prior must be an object { mean, sd }, found "normal".',
    );
  });
});

describe("calibrate", () => {
  for (const fixture of FIXTURES) {
    const matrix = buildResponseMatrix(fixture.responses, { itemIds: fixture.itemIds });
    for (const reference of fixture.calibrations) {
      const flat = reference.models.some((name) => FLAT_MODELS.has(name));
      it(`matches ${reference.name} on ${fixture.source}`, () => {
        const result = calibrate(matrix, fixtureModels(reference), calibrationOptions(reference));
        expect(result.converged).toBe(true);
        reference.items.forEach((expected, i) => {
          const actual = defined(result.items[i]);
          expect(actual.id).toBe(fixture.itemIds[i]);
          expect(actual.model).toBe(model(defined(reference.models[i])));
          const tolerance = flat ? FLAT_PARAMETER_TOLERANCE : PARAMETER_TOLERANCE;
          expectParameters(actual.parameters, expected.parameters, tolerance, actual.id);
          if (reference.fixed?.[actual.id] !== undefined) {
            expect(actual.parameters).toEqual(reference.fixed[actual.id]);
            expect(actual.standardErrors).toBeNull();
            return;
          }
          const errors = defined(actual.standardErrors);
          expect(Object.keys(errors)).toEqual(Object.keys(defined(expected.standardErrors)));
          for (const [name, value = 0] of Object.entries(defined(expected.standardErrors))) {
            const relative = (flat ? FLAT_STANDARD_ERROR_TOLERANCE : STANDARD_ERROR_TOLERANCE) * value;
            expectClose(defined(errors[name]), value, relative, `${actual.id} SE of ${name}`);
          }
        });
        const tolerance = flat ? FLAT_LOG_LIKELIHOOD_TOLERANCE : LOG_LIKELIHOOD_TOLERANCE;
        expectClose(result.logLikelihood, reference.logLikelihood, tolerance, "log-likelihood");
        if (reference.aic !== undefined) expectClose(result.aic, reference.aic, 2 * tolerance, "AIC");
        if (reference.bic !== undefined) expectClose(result.bic, reference.bic, 2 * tolerance, "BIC");
      });

      it(`gives ${reference.name}'s log-likelihood at its estimates on ${fixture.source}`, () => {
        const fixed = Object.fromEntries(
          reference.items.map(({ parameters }, i) => [defined(fixture.itemIds[i]), parameters]),
        );
        const result = calibrate(matrix, fixtureModels(reference), { ...quadratureOf(reference), fixed });
        expectClose(result.logLikelihood, reference.logLikelihood, FIXED_LOG_LIKELIHOOD_TOLERANCE, "log-likelihood");
      });
    }
  }

  for (const reference of crossProduct.cases) {
    it(`matches the standard errors of ${reference.name}`, () => {
      const matrix = buildResponseMatrix(reference.responses, { itemIds: reference.itemIds });
      const result = calibrate(matrix, fixtureModels(reference), {
        ...calibrationOptions(reference),
        standardErrors: "cross-product",
      });
      const flat = reference.models.some((name) => FLAT_MODELS.has(name));
      reference.items.forEach((expected, i) => {
        const errors = defined(defined(result.items[i]).standardErrors);
        for (const [name, value = 0] of Object.entries(defined(expected.standardErrors))) {
          const relative = (flat ? FLAT_STANDARD_ERROR_TOLERANCE : STANDARD_ERROR_TOLERANCE) * value;
          expectClose(defined(errors[name]), value, relative, `item ${i + 1} SE of ${name}`);
        }
      });
    });
  }

  const rows = simulate(20261029, 300, ITEMS_2PL);

  it("gives identical calibrations from response records and from the matrix", () => {
    const sparse = rows.map((row, person) =>
      row.map((response, item) => ((person + item) % 3 === 0 ? null : response)),
    );
    const matrix = buildResponseMatrix(sparse);
    const records = buildResponseRecords(
      matrix.responses
        .flatMap((row, person) =>
          row.flatMap((response, item) =>
            response === null ? [] : [{ person: `person${person + 1}`, item: `item${item + 1}`, response }],
          ),
        )
        .reverse(),
      { personIds: matrix.personIds, itemIds: matrix.itemIds },
    );
    for (const standardErrors of ["hessian", "cross-product"] as const) {
      const options = {
        latent: "normal",
        fixed: { item1: defined(PARAMETERS_2PL[0]) },
        standardErrors,
      } as const;
      expect(calibrate(records, twoParameterLogistic, options)).toEqual(
        calibrate(matrix, twoParameterLogistic, options),
      );
    }
  });

  it("reports non-convergence within the cycle limit", () => {
    const result = calibrate(buildResponseMatrix(rows), twoParameterLogistic, { maxIterations: 2 });
    expect(result).toMatchObject({ converged: false, iterations: 2 });
  });

  it("ignores persons without responses, also in the BIC", () => {
    const base = calibrate(buildResponseMatrix(rows), twoParameterLogistic);
    const padded = calibrate(buildResponseMatrix([...rows, [null, null, null, null]]), twoParameterLogistic);
    expect(padded.items.map((item) => item.parameters)).toEqual(base.items.map((item) => item.parameters));
    expectClose(padded.logLikelihood, base.logLikelihood, 1e-9);
    expectClose(padded.bic, base.bic, 1e-9);
    expectClose(base.bic, -2 * base.logLikelihood + 8 * Math.log(300), 1e-9);
  });

  it("evaluates the log-likelihood when every item is fixed", () => {
    const fixed = Object.fromEntries(PARAMETERS_2PL.map((item, i) => [`item${i + 1}`, item]));
    const result = calibrate(buildResponseMatrix(rows), twoParameterLogistic, { fixed });
    expect(result.items.map((item) => item.parameters)).toEqual(PARAMETERS_2PL);
    expect(result.items.every((item) => item.standardErrors === null)).toBe(true);
    expect(result.aic).toBe(-2 * result.logLikelihood);
  });

  it("takes one model per item and shares a only among the 1PL items", () => {
    const models = [oneParameterLogistic, oneParameterLogistic, twoParameterLogistic, twoParameterLogistic];
    const result = calibrate(buildResponseMatrix(rows), models);
    const [first, second, third, fourth] = result.items.map((item) => item.parameters.a);
    expect(first).toBe(second);
    expect(third).not.toBe(fourth);
    expectClose(result.aic, -2 * result.logLikelihood + 2 * 7, 1e-9);
  });

  it("calibrates multiple-choice and constructed-response items together, typed by their parameters", () => {
    const models: IrtModel<LogisticItem | ThresholdItem>[] = [
      threeParameterLogistic,
      threeParameterLogistic,
      twoParameterLogistic,
      generalizedPartialCredit,
      generalizedPartialCredit,
    ];
    const pool = [...ITEMS_2PL.slice(0, 3), ...ITEMS_GPCM.slice(0, 2)];
    const result = calibrate(buildResponseMatrix(simulate(20261029, 600, pool)), models, {
      priors: { c: betaPrior(5, 17) },
      standardErrors: "hessian",
    });
    expect(result.converged).toBe(true);
    expect(result.items.map((item) => Object.keys(item.standardErrors ?? {}))).toEqual([
      ["a", "b", "c"],
      ["a", "b", "c"],
      ["a", "b"],
      ["a", "b1", "b2", "b3"],
      ["a", "b1", "b2"],
    ]);
    expectClose(result.aic, -2 * result.logLikelihood + 2 * 15, 1e-9);
  });

  it("shares a among PCM items, and a and the steps among RSM items", () => {
    const matrix = buildResponseMatrix(science.responses, { itemIds: science.itemIds });
    const pcm = calibrate(matrix, partialCredit, { standardErrors: "cross-product" });
    expect(new Set(pcm.items.map((item) => item.parameters.a)).size).toBe(1);
    expect(Object.keys(defined(pcm.items[0]?.standardErrors))).toEqual(["a", "b1", "b2", "b3"]);
    const rsm = calibrate(matrix, ratingScale, { standardErrors: "cross-product" });
    expect(new Set(rsm.items.map((item) => item.parameters.a)).size).toBe(1);
    expect(new Set(rsm.items.map((item) => item.parameters.steps.join())).size).toBe(1);
    expect(new Set(rsm.items.map((item) => item.parameters.b)).size).toBe(4);
    expect(Object.keys(defined(rsm.items[0]?.standardErrors))).toEqual(["a", "b", "step1", "step2"]);
    expectClose(rsm.aic, -2 * rsm.logLikelihood + 2 * 7, 1e-9);
  });

  it("applies priors to the parameters they name", () => {
    const weak = calibrate(buildResponseMatrix(rows), twoParameterLogistic);
    const strong = calibrate(buildResponseMatrix(rows), twoParameterLogistic, {
      priors: { a: logNormalPrior(0, 0.05), b: normalPrior(0, 0.1) },
    });
    for (const [i, item] of strong.items.entries()) {
      const free = defined(weak.items[i]).parameters;
      expect(Math.abs(Math.log(item.parameters.a))).toBeLessThan(Math.abs(Math.log(free.a)));
      expect(Math.abs(item.parameters.b)).toBeLessThan(Math.abs(free.b));
    }
  });

  it("gives the same estimates with any standard errors, and none by default", () => {
    const matrix = buildResponseMatrix(rows);
    const options = {
      latent: { all: "normal" },
      groups: rows.map(() => "all"),
      fixed: { item1: defined(PARAMETERS_2PL[0]) },
    } as const;
    const none = calibrate(matrix, twoParameterLogistic, options);
    expect(none.items.every((item) => item.standardErrors === null)).toBe(true);
    expect(none.groups[0]?.standardErrors).toBeNull();
    for (const standardErrors of ["hessian", "cross-product"] as const) {
      const result = calibrate(matrix, twoParameterLogistic, { ...options, standardErrors });
      expect(result.items.map((item) => item.parameters)).toEqual(none.items.map((item) => item.parameters));
      expect(result.groups[0]?.standardErrors).not.toBeNull();
      expect(result.items[1]?.standardErrors).not.toBeNull();
    }
  });

  it("reports one standard normal group without groups", () => {
    const [group] = calibrate(buildResponseMatrix(rows), twoParameterLogistic).groups;
    expect(group).toMatchObject({ group: null, distribution: "standard-normal", mean: 0, sd: 1, standardErrors: null });
    expect(defined(group).nodes).toHaveLength(61);
    expectClose(
      defined(group).weights.reduce((sum, weight) => sum + weight, 0),
      1,
      1e-15,
    );
  });

  it("gives finite results for a single item and for two persons", () => {
    const single = calibrate(buildResponseMatrix(rows.map((row) => row.slice(0, 1))), twoParameterLogistic);
    const pair = calibrate(
      buildResponseMatrix([
        [1, 0],
        [0, 1],
      ]),
      twoParameterLogistic,
      { standardErrors: "hessian" },
    );
    for (const result of [single, pair]) expect(nonFinite(result)).toEqual([]);
    // Two persons with opposite responses say nothing about a: the information matrix is singular.
    expect(pair.items.map((item) => item.standardErrors)).toEqual([null, null]);
  });

  it("bounds standard errors to 2,000 parameters, counting a shared one for each item, before it estimates anything", () => {
    // Three persons answer 1,001 items with both categories each: 2,002 parameters of the 2PL, and 1,002 of the 1PL,
    // whose covariance lists the shared a for each of its items.
    const rows = [0, 1, 2].map((k) => Array.from({ length: 1001 }, (_, j) => (j + k) % 2));
    const matrix = buildResponseMatrix(rows);
    for (const [model, standardErrors] of [
      [twoParameterLogistic, "hessian"],
      [oneParameterLogistic, "cross-product"],
    ] as const) {
      expect(() => calibrate(matrix, model, { standardErrors })).toThrow(
        "Standard errors take at most 2000 estimated parameters, as their information holds every pair of them; found 2002.",
      );
    }
    expect(calibrate(matrix, twoParameterLogistic, { maxIterations: 1 }).items).toHaveLength(1001);
  });

  it("rejects RSM items with different numbers of categories", () => {
    const matrix = buildResponseMatrix([
      [0, 0],
      [1, 1],
      [2, 2],
      [1, 3],
    ]);
    expect(() => calibrate(matrix, ratingScale)).toThrow(
      'Items "item1" and "item2" of the RSM model share parameters, so they need the same number of categories; found 3 and 4.',
    );
  });

  it("rejects items it cannot calibrate and invalid options", () => {
    const matrix = buildResponseMatrix([
      [0, 0, 1],
      [0, 1, 0],
      [0, null, 2],
    ]);
    expect(() => calibrate(matrix, twoParameterLogistic)).toThrow(
      'Item "item1" has responses in fewer than two categories; calibration needs at least two.',
    );
    expect(() => calibrate(matrix, twoParameterLogistic, { fixed: { item1: { a: 1, b: 0, c: 0, d: 1 } } })).toThrow(
      'Item "item3" has a response of 2, but the 2PL model allows responses from 0 to 1.',
    );
    const binary = buildResponseMatrix(rows);
    const binaryItem = defined(PARAMETERS_2PL[0]);
    expect(() => calibrate(binary, twoParameterLogistic, { fixed: { q1: binaryItem } })).toThrow(
      'Fixed parameters are given for "q1", which is not an item id.',
    );
    expect(() => calibrate(binary, twoParameterLogistic, { fixed: { item1: { a: 1, b: 0, c: 0.1, d: 1 } } })).toThrow(
      'Fixed item "item1": A 2PL item has c = 0, found c = 0.1.',
    );
    expect(() => calibrate(binary, twoParameterLogistic, { priors: { g: betaPrior(5, 17) } })).toThrow(
      'A prior is given for "g", which no estimated item has.',
    );
    expect(() => calibrate(binary, [twoParameterLogistic])).toThrow("Found 1 models for 4 items.");
    expect(() => calibrate(binary, JSON.parse("null"))).toThrow(
      "The model must be an IRT model such as twoParameterLogistic, found null.",
    );
    expect(() => calibrate(binary, twoParameterLogistic, { standardErrors: JSON.parse('"full"') })).toThrow(
      'The option standardErrors must be "none", "hessian" or "cross-product", found "full".',
    );
    const empty = buildResponseMatrix([[null, null]]);
    expect(() => calibrate(empty, twoParameterLogistic, { fixed: { item1: binaryItem, item2: binaryItem } })).toThrow(
      "Calibration needs at least one response.",
    );
    expect(() => calibrate(binary, twoParameterLogistic, { tolerance: -1 })).toThrow(
      "The option tolerance must be a non-negative number, found -1.",
    );
    expect(calibrate(binary, twoParameterLogistic, { tolerance: 0, maxIterations: 2 }).iterations).toBe(2);
    expect(() => calibrate(binary, twoParameterLogistic, { maxIterations: 0.5 })).toThrow(DataError);
    expect(() =>
      calibrate(binary, twoParameterLogistic, { quadrature: JSON.parse('{ "method": "trapezoid" }') }),
    ).toThrow('The quadrature method must be "rectangular" or "gauss-hermite", found "trapezoid".');
  });

  it("rejects options of other types from JavaScript", () => {
    const binary = buildResponseMatrix(rows);
    expect(() => calibrate(binary, twoParameterLogistic, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => calibrate(binary, twoParameterLogistic, JSON.parse('{"verbose":1}'))).toThrow(
      'There is no option "verbose"',
    );
    expect(() => calibrate(binary, JSON.parse('{ "name": "2PL" }'))).toThrow(
      "The model must be an IRT model such as twoParameterLogistic, found an object.",
    );
    expect(() => calibrate(binary, twoParameterLogistic, { fixed: JSON.parse("null") })).toThrow(
      "The option fixed must be an object of item parameters by item id, found null.",
    );
    expect(() => calibrate(binary, twoParameterLogistic, { priors: JSON.parse('{ "a": null }') })).toThrow(
      'The prior of "a" must be a prior such as normalPrior(0, 1), found null.',
    );
    expect(() => calibrate(binary, twoParameterLogistic, { testlets: JSON.parse("null") })).toThrow(
      "The option testlets must be an object of item id lists by testlet id, found null.",
    );
    expect(() => calibrate(binary, twoParameterLogistic, { maxIterations: JSON.parse('"10"') })).toThrow(
      'The option maxIterations must be a positive integer, found "10".',
    );
  });

  it("calibrates items whose ids are names of the properties of every object, none of them fixed", () => {
    const named = buildResponseMatrix(
      simulate(20261029, 300, ITEMS_2PL).map((row) => row.slice(0, 4)),
      {
        itemIds: ["__proto__", "constructor", "toString", "valueOf"],
      },
    );
    const result = calibrate(named, twoParameterLogistic);
    expect(result.items.map(({ id }) => id)).toEqual(["__proto__", "constructor", "toString", "valueOf"]);
    expect(result.items.every(({ parameters }) => Number.isFinite(parameters.a) && parameters.a > 0)).toBe(true);
  });

  it("rejects an item without responses or without a response in a lower category", () => {
    const matrix = buildResponseMatrix([
      [1, null],
      [1, null],
      [null, null],
    ]);
    expect(() => calibrate(matrix, fourParameterLogistic)).toThrow(
      'Item "item1" has no response in category 0; calibration needs responses in every category up to the highest.',
    );
    const fixed = { item1: defined(PARAMETERS_2PL[0]) };
    expect(() => calibrate(matrix, fourParameterLogistic, { fixed })).toThrow(
      'Item "item2" has responses in fewer than two categories; calibration needs at least two.',
    );
    expect(() => calibrate(matrix, gradedResponse, { fixed: { item1: { a: 1, b: [0] } } })).toThrow(
      'Item "item2" has responses in fewer than two categories; calibration needs at least two.',
    );
  });

  it("finds an empty category below a response far above the others without a table up to it", () => {
    const highest = 2 ** 31 - 1;
    const matrix = buildResponseMatrix([[0], [1], [highest], [1]]);
    for (const model of [gradedResponse, generalizedPartialCredit]) {
      expect(() => calibrate(matrix, model)).toThrow(
        'Item "item1" has no response in category 2; calibration needs responses in every category up to the highest.',
      );
    }
    expect(() => calibrate(matrix, twoParameterLogistic)).toThrow(
      `Item "item1" has a response of ${highest}, but the 2PL model allows responses from 0 to 1.`,
    );
  }, 5_000);
});

describe("calibrateWithCovariance", () => {
  const sample = simulate(20261029, 300, ITEMS_2PL);
  const options = { standardErrors: "hessian" } as const;

  for (const reference of covariance.cases) {
    const source = reference.data === "irt-lsat7" ? lsat7 : science;
    it(`matches the covariance matrix of ${reference.name}`, () => {
      const matrix = buildResponseMatrix(source.responses, { itemIds: source.itemIds });
      const result = calibrateWithCovariance(matrix, reference.models.map(model), {
        quadrature: { method: "rectangular", points: reference.quadrature.points },
        ...(reference.priors && {
          priors: Object.fromEntries(
            Object.entries(reference.priors).map(([name, value]) => [name, describedPrior(value)]),
          ),
        }),
        ...(reference.itemPriors && {
          itemPriors: Object.fromEntries(
            Object.entries(reference.itemPriors).map(([id, byName]) => [
              id,
              Object.fromEntries(Object.entries(byName).map(([name, value]) => [name, describedPrior(value)])),
            ]),
          ),
        }),
        tolerance: EM_TOLERANCE,
        maxIterations: 10_000,
        standardErrors: "hessian",
      });
      expect(result.calibration.converged).toBe(true);
      const tolerance = reference.models.some((name) => FLAT_MODELS.has(name))
        ? FLAT_COVARIANCE_TOLERANCE
        : COVARIANCE_TOLERANCE;
      expectCovariance(result.covariance, reference.covariance, tolerance);
    });
  }

  it("calibrates as calibrate does", () => {
    const matrix = buildResponseMatrix(sample);
    expect(calibrateWithCovariance(matrix, twoParameterLogistic, options).calibration).toEqual(
      calibrate(matrix, twoParameterLogistic, options),
    );
  });

  it("has the squares of the standard errors on its diagonal, with either method", () => {
    const matrix = buildResponseMatrix(sample);
    for (const standardErrors of ["hessian", "cross-product"] as const) {
      const { calibration, covariance: result } = calibrateWithCovariance(matrix, twoParameterLogistic, {
        ...options,
        standardErrors,
      });
      const { parameters, matrix: entries } = defined(result);
      parameters.forEach(({ item, name }, k) => {
        const errors = defined(defined(calibration.items.find(({ id }) => id === item)).standardErrors);
        expectClose(Math.sqrt(defined(entries[k]?.[k])), defined(errors[name]), 1e-15, `${item} ${name}`);
      });
    }
  });

  it("is symmetric and covers the estimated items in item order, leaving out the fixed ones", () => {
    const fixed = { item1: defined(PARAMETERS_2PL[0]), item3: defined(PARAMETERS_2PL[2]) };
    const { covariance: result } = calibrateWithCovariance(buildResponseMatrix(sample), twoParameterLogistic, {
      ...options,
      fixed,
    });
    const { parameters, matrix } = defined(result);
    expect(parameters).toEqual([
      { item: "item2", name: "a" },
      { item: "item2", name: "b" },
      { item: "item4", name: "a" },
      { item: "item4", name: "b" },
    ]);
    matrix.forEach((row, i) => {
      row.forEach((value, j) => {
        expect(value).toBe(defined(matrix[j]?.[i]));
      });
    });
    expect(defined(matrix[0]?.[2])).not.toBe(0);
  });

  it("lists a parameter that items share once for each item, with correlation 1", () => {
    const { covariance: result } = calibrateWithCovariance(
      buildResponseMatrix(science.responses, { itemIds: science.itemIds }),
      ratingScale,
      {
        standardErrors: "hessian",
      },
    );
    const { parameters, matrix } = defined(result);
    expect(parameters.map(({ item }) => item)).toEqual(science.itemIds.flatMap((id) => [id, id, id, id]));
    const shared = parameters.flatMap(({ name }, k) => (name === "a" ? [k] : []));
    expect(shared).toHaveLength(4);
    for (const j of shared) for (const k of shared) expect(matrix[j]?.[k]).toBe(matrix[0]?.[0]);
  });

  it("is null without standard errors, with every item fixed, and where the information matrix is singular", () => {
    const matrix = buildResponseMatrix(sample);
    expect(calibrateWithCovariance(matrix, twoParameterLogistic).covariance).toBeNull();
    const fixed = Object.fromEntries(ITEMS_2PL.map((item, i) => [`item${i + 1}`, item.parameters]));
    const normal = calibrateWithCovariance(matrix, twoParameterLogistic, { ...options, fixed, latent: "normal" });
    expect(normal.covariance).toBeNull();
    expect(normal.calibration.groups[0]?.standardErrors).not.toBeNull();
    const pair = buildResponseMatrix([
      [1, 0],
      [0, 1],
    ]);
    expect(calibrateWithCovariance(pair, twoParameterLogistic, options).covariance).toBeNull();
  });

  it("is plain data that JSON serializes", () => {
    const { covariance: result } = calibrateWithCovariance(buildResponseMatrix(sample), twoParameterLogistic, options);
    expect(JSON.parse(JSON.stringify(result))).toEqual(result);
  });

  it("rejects options of other types from JavaScript", () => {
    const matrix = buildResponseMatrix(sample);
    expect(() => calibrateWithCovariance(matrix, twoParameterLogistic, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => calibrateWithCovariance(matrix, twoParameterLogistic, JSON.parse('{"verbose":1}'))).toThrow(
      'There is no option "verbose"',
    );
    expect(() =>
      calibrateWithCovariance(matrix, twoParameterLogistic, { ...options, standardErrors: JSON.parse('"full"') }),
    ).toThrow('The option standardErrors must be "none", "hessian" or "cross-product", found "full".');
  });
});

describe("item priors", () => {
  const sample = simulate(20261029, 300, ITEMS_2PL);
  const matrix = buildResponseMatrix(sample);

  it("replace the priors of the option for the parameter they name, and only for that item", () => {
    const priors = { a: logNormalPrior(0.3, 0.5), b: normalPrior(0, 0.5) };
    const general = calibrate(matrix, twoParameterLogistic, { priors });
    const replaced = calibrate(matrix, twoParameterLogistic, {
      priors,
      itemPriors: { item2: { b: normalPrior(1, 0.01) } },
    });
    expect(Math.abs(defined(replaced.items[1]).parameters.b - 1)).toBeLessThan(0.05);
    expect(defined(general.items[1]).parameters.b).toBeLessThan(0.5);
  });

  it("give the result of the option priors when every item has the same prior", () => {
    const prior = logNormalPrior(0.2, 0.4);
    const itemPriors = Object.fromEntries(ITEMS_2PL.map((_, i) => [`item${i + 1}`, { a: prior }]));
    const byItem = calibrate(matrix, twoParameterLogistic, { itemPriors, standardErrors: "hessian" });
    expect(byItem).toEqual(
      calibrate(matrix, twoParameterLogistic, { priors: { a: prior }, standardErrors: "hessian" }),
    );
  });

  it("change nothing when empty", () => {
    expect(calibrate(matrix, twoParameterLogistic, { itemPriors: {} })).toEqual(
      calibrate(matrix, twoParameterLogistic),
    );
    expect(calibrate(matrix, twoParameterLogistic, { itemPriors: { item1: {} } })).toEqual(
      calibrate(matrix, twoParameterLogistic),
    );
  });

  it("apply to the parameters of models that do not share them", () => {
    const grades = buildResponseMatrix(science.responses, { itemIds: science.itemIds });
    const free = calibrate(grades, generalizedPartialCredit);
    const tight = calibrate(grades, generalizedPartialCredit, {
      itemPriors: { [defined(science.itemIds[0])]: { a: logNormalPrior(0, 0.05) } },
    });
    expect(Math.abs(Math.log(defined(tight.items[0]).parameters.a))).toBeLessThan(
      Math.abs(Math.log(defined(free.items[0]).parameters.a)),
    );
  });

  it("reject items that are unknown or fixed, parameters the model lacks or shares, and priors of other types", () => {
    const fixed = { item1: defined(PARAMETERS_2PL[0]) };
    expect(() => calibrate(matrix, twoParameterLogistic, { itemPriors: { q1: { a: normalPrior(1, 1) } } })).toThrow(
      'Priors are given for "q1", which is not an item id.',
    );
    expect(() =>
      calibrate(matrix, twoParameterLogistic, { fixed, itemPriors: { item1: { a: normalPrior(1, 1) } } }),
    ).toThrow('Priors are given for item "item1", which is fixed.');
    expect(() => calibrate(matrix, twoParameterLogistic, { itemPriors: { item2: { c: betaPrior(5, 17) } } })).toThrow(
      'A prior is given for "c" of item "item2", whose model has no such parameter.',
    );
    const grades = buildResponseMatrix(science.responses, { itemIds: science.itemIds });
    expect(() =>
      calibrate(grades, ratingScale, {
        itemPriors: { [defined(science.itemIds[1])]: { a: normalPrior(1, 1) } },
      }),
    ).toThrow(`A prior is given for "a" of item "${science.itemIds[1]}", whose model shares it among items`);
    const rejected = (value: string): DataError | undefined => {
      try {
        calibrate(matrix, twoParameterLogistic, { itemPriors: JSON.parse(value) });
      } catch (error) {
        return error instanceof DataError ? error : undefined;
      }
      return undefined;
    };
    expect(rejected("null")?.message).toBe(
      "The option itemPriors must be an object of priors by item id and parameter name, found null.",
    );
    expect(rejected('{ "item1": null }')?.message).toBe(
      'The priors of item "item1" must be an object of priors by parameter name, found null.',
    );
    expect(rejected('{ "item1": { "a": null } }')?.message).toBe(
      'The prior of "a" of item "item1" must be a prior such as normalPrior(0, 1), found null.',
    );
  });

  it("treat an item id such as constructor as an id", () => {
    expect(() => calibrate(matrix, twoParameterLogistic, { itemPriors: JSON.parse('{ "constructor": {} }') })).toThrow(
      'Priors are given for "constructor", which is not an item id.',
    );
  });
});

describe("the sampling error of fixed items", () => {
  const options = (reference: IrtFixedErrorCase) => ({
    fixed: reference.fixed,
    ...(reference.latent === "normal" && { latent: "normal" as const }),
    standardErrors: "hessian" as const,
    tolerance: EM_TOLERANCE,
    maxIterations: 10_000,
  });
  const matrixOf = (reference: IrtFixedErrorCase) =>
    buildResponseMatrix(reference.responses, { itemIds: reference.itemIds });
  const [independent] = fixedError.cases;

  for (const reference of fixedError.cases) {
    it(`matches base R's V + J S J' with ${reference.name}`, () => {
      const known = calibrateWithCovariance(matrixOf(reference), twoParameterLogistic, options(reference));
      const { calibration, covariance } = calibrateWithCovariance(matrixOf(reference), twoParameterLogistic, {
        ...options(reference),
        fixedCovariance: reference.fixedCovariance,
      });
      expect(calibration.converged).toBe(true);
      for (const expected of reference.items) {
        const actual = defined(calibration.items.find((item) => item.id === expected.id));
        expectParameters(
          actual.parameters,
          { a: expected.a, b: expected.b, c: 0, d: 1 },
          LATENT_PARAMETER_TOLERANCE,
          actual.id,
        );
      }
      expect(defined(covariance).parameters).toEqual(reference.covariance.parameters);
      const compare = (actual: readonly (readonly number[])[], expected: number[][], what: string) => {
        expected.forEach((row, i) => {
          row.forEach((value, j) => {
            const scale = Math.sqrt((expected[i]?.[i] ?? 0) * (expected[j]?.[j] ?? 0));
            expectClose(
              defined(actual[i]?.[j]),
              value,
              LATENT_STANDARD_ERROR_TOLERANCE * scale,
              `${what} (${i}, ${j})`,
            );
          });
        });
      };
      compare(defined(known.covariance).matrix, reference.knownCovariance, "known covariance");
      compare(defined(covariance).matrix, reference.covariance.matrix, "covariance");
      if (reference.groupErrors === null || reference.knownGroupErrors === null) return;
      const [group] = calibration.groups;
      const errors = defined(defined(group).standardErrors);
      expectClose(errors.mean, reference.groupErrors.mean, LATENT_STANDARD_ERROR_TOLERANCE * errors.mean, "SE of mean");
      expectClose(errors.sd, reference.groupErrors.sd, LATENT_STANDARD_ERROR_TOLERANCE * errors.sd, "SE of sd");
      const knownErrors = defined(defined(known.calibration.groups[0]).standardErrors);
      expectClose(
        knownErrors.mean,
        reference.knownGroupErrors.mean,
        LATENT_STANDARD_ERROR_TOLERANCE * knownErrors.mean,
      );
    });
  }

  it("leaves the estimates as they are and adds as much to the variances as the covariance has", () => {
    const reference = defined(independent);
    const without = calibrateWithCovariance(matrixOf(reference), twoParameterLogistic, options(reference));
    const scaled = (factor: number): ParameterCovariance => ({
      parameters: reference.fixedCovariance.parameters,
      matrix: reference.fixedCovariance.matrix.map((row) => row.map((value) => value * factor)),
    });
    const run = (factor: number) =>
      calibrateWithCovariance(matrixOf(reference), twoParameterLogistic, {
        ...options(reference),
        fixedCovariance: scaled(factor),
      });
    const zero = run(0);
    const once = run(1);
    const four = run(4);
    expect(zero.calibration.items).toEqual(without.calibration.items);
    expect(zero.covariance).toEqual(without.covariance);
    expect(once.calibration.items.map((item) => item.parameters)).toEqual(
      without.calibration.items.map((item) => item.parameters),
    );
    const base = defined(without.covariance).matrix;
    defined(once.covariance).matrix.forEach((row, i) => {
      row.forEach((value, j) => {
        const added = value - (base[i]?.[j] ?? 0);
        const fourfold = (defined(four.covariance).matrix[i]?.[j] ?? 0) - (base[i]?.[j] ?? 0);
        expectClose(fourfold, 4 * added, 1e-9 * Math.max(1, Math.abs(fourfold)), `(${i}, ${j})`);
      });
      expect(row[i]).toBeGreaterThan(base[i]?.[i] ?? 0);
    });
  });

  it("carries the fixed items' error with the cross-product standard errors too", () => {
    const reference = defined(independent);
    const crossProduct = { ...options(reference), standardErrors: "cross-product" as const };
    const without = calibrate(matrixOf(reference), twoParameterLogistic, crossProduct);
    const withError = calibrate(matrixOf(reference), twoParameterLogistic, {
      ...crossProduct,
      fixedCovariance: reference.fixedCovariance,
    });
    without.items.forEach((item, i) => {
      if (item.standardErrors === null) return;
      expect(defined(withError.items[i]?.standardErrors?.b)).toBeGreaterThan(defined(item.standardErrors.b));
    });
  });

  it("skips the parameters of items the data do not have", () => {
    const reference = defined(independent);
    const pool: ParameterCovariance = {
      parameters: [...reference.fixedCovariance.parameters, { item: "elsewhere", name: "a" }],
      matrix: [
        ...reference.fixedCovariance.matrix.map((row) => [...row, 0]),
        [...reference.fixedCovariance.matrix.map(() => 0), 0.01],
      ],
    };
    const listed = calibrateWithCovariance(matrixOf(reference), twoParameterLogistic, {
      ...options(reference),
      fixedCovariance: pool,
    });
    const exact = calibrateWithCovariance(matrixOf(reference), twoParameterLogistic, {
      ...options(reference),
      fixedCovariance: reference.fixedCovariance,
    });
    expect(listed.covariance).toEqual(exact.covariance);
  });

  /**
   * Replicates a pretest under fixed items calibrated before, and counts the 95% intervals of the pretest items' a and b
   * and of the mean of theta that cover the truth, with the fixed items taken as known and with their error carried.
   */
  const coverage = (design: {
    readonly samePersons: boolean;
    readonly replications: number;
    readonly seed: number;
  }) => {
    const random = createRandom(design.seed);
    const operational = 10;
    const items = Array.from({ length: operational + 4 }, (_, j) => ({
      id: `q${String(j + 1).padStart(2, "0")}`,
      model: twoParameterLogistic,
      parameters: { a: 0.8 + 1.2 * random.uniform(), b: random.normal(), c: 0, d: 1 },
    }));
    const itemIds = items.map((item) => item.id);
    const z = normalQuantile(0.975);
    const counts = { known: { a: 0, b: 0, mean: 0 }, carried: { a: 0, b: 0, mean: 0 }, items: 0, means: 0 };
    for (let replication = 0; replication < design.replications; replication++) {
      const persons = (count: number, mean: number, sd: number) =>
        simulateThetas(count, random).map((theta) => mean + sd * theta);
      const thetas = design.samePersons ? persons(1000, 0, 1) : persons(1000, 0.3, 1.1);
      const rows = simulateResponses(items, thetas, random).responses;
      const first = design.samePersons
        ? rows.map((row) => row.slice(0, operational))
        : simulateResponses(items.slice(0, operational), persons(250, 0, 1), random).responses;
      const calibration = calibrateWithCovariance(
        buildResponseMatrix(first, { itemIds: itemIds.slice(0, operational) }),
        twoParameterLogistic,
        { standardErrors: "hessian" },
      );
      const base = {
        fixed: Object.fromEntries(calibration.calibration.items.map((item) => [item.id, item.parameters])),
        ...(design.samePersons ? {} : { latent: "normal" as const }),
        standardErrors: "hessian" as const,
      };
      const matrix = buildResponseMatrix(rows, { itemIds });
      const fits = {
        known: calibrate(matrix, twoParameterLogistic, base),
        carried: calibrate(matrix, twoParameterLogistic, { ...base, fixedCovariance: defined(calibration.covariance) }),
      };
      for (const [name, fit] of Object.entries(fits) as [keyof typeof fits, Calibration<LogisticItem>][]) {
        items.slice(operational).forEach((truth, k) => {
          const estimate = defined(fit.items[operational + k]);
          const errors = defined(estimate.standardErrors);
          const parameters = estimate.parameters as LogisticItem;
          if (Math.abs(parameters.a - truth.parameters.a) <= z * defined(errors.a)) counts[name].a++;
          if (Math.abs(parameters.b - truth.parameters.b) <= z * defined(errors.b)) counts[name].b++;
        });
        const group = defined(fit.groups[0]);
        if (!design.samePersons && Math.abs(group.mean - 0.3) <= z * defined(group.standardErrors).mean) {
          counts[name].mean++;
        }
      }
      counts.items += 4;
      counts.means += 1;
    }
    return counts;
  };

  it("covers the pretest items and the mean when the fixed items were calibrated on other persons", () => {
    // Thresholds fixed before the first run: with the error carried, 95% intervals of a and b cover the truth 0.92 to
    // 0.98 of the time and those of the mean at least 0.88; taken as known, b's cover at most 0.90 and the mean's 0.85.
    const counts = coverage({ samePersons: false, replications: 60, seed: 20261104 });
    const share = (count: number, total: number) => count / total;
    expect(share(counts.carried.a, counts.items)).toBeGreaterThanOrEqual(0.92);
    expect(share(counts.carried.a, counts.items)).toBeLessThanOrEqual(0.98);
    expect(share(counts.carried.b, counts.items)).toBeGreaterThanOrEqual(0.92);
    expect(share(counts.carried.b, counts.items)).toBeLessThanOrEqual(0.98);
    expect(share(counts.carried.mean, counts.means)).toBeGreaterThanOrEqual(0.88);
    expect(share(counts.known.b, counts.items)).toBeLessThanOrEqual(0.9);
    expect(share(counts.known.mean, counts.means)).toBeLessThanOrEqual(0.85);
  }, 120_000);

  it("covers the pretest items when the fixed items were calibrated on the same persons, theta standard normal", () => {
    // Thresholds fixed before the first run: with the error carried, 95% intervals of a and b cover the truth 0.92 to
    // 0.98 of the time.
    const counts = coverage({ samePersons: true, replications: 40, seed: 20261105 });
    for (const count of [counts.carried.a, counts.carried.b]) {
      expect(count / counts.items).toBeGreaterThanOrEqual(0.92);
      expect(count / counts.items).toBeLessThanOrEqual(0.98);
    }
  }, 120_000);

  it("refuses a covariance without standard errors, of estimated items or of parameters it cannot move", () => {
    const reference = defined(independent);
    const matrix = matrixOf(reference);
    const covariance = (item: string, name: string, variance = 0.01): ParameterCovariance => ({
      parameters: [{ item, name }],
      matrix: [[variance]],
    });
    expect(() =>
      calibrate(matrix, twoParameterLogistic, {
        ...options(reference),
        standardErrors: "none",
        fixedCovariance: covariance("q01", "a"),
      }),
    ).toThrow('The option fixedCovariance needs the standard errors "hessian" or "cross-product".');
    expect(() =>
      calibrate(matrix, twoParameterLogistic, { ...options(reference), fixedCovariance: covariance("q09", "a") }),
    ).toThrow('The option fixedCovariance has the parameter "a" of item "q09", which is estimated here, not fixed.');
    expect(() =>
      calibrate(matrix, twoParameterLogistic, { ...options(reference), fixedCovariance: covariance("q01", "slope") }),
    ).toThrow(
      'The option fixedCovariance has the parameter "slope" of item "q01", which its 2PL model does not estimate; it estimates a, b.',
    );
    const twice: ParameterCovariance = {
      parameters: [
        { item: "q01", name: "a" },
        { item: "q01", name: "a" },
      ],
      matrix: [
        [0.01, 0],
        [0, 0.01],
      ],
    };
    expect(() => calibrate(matrix, twoParameterLogistic, { ...options(reference), fixedCovariance: twice })).toThrow(
      'The option fixedCovariance lists the parameter "a" of item "q01" twice.',
    );
    expect(() =>
      calibrate(matrix, twoParameterLogistic, {
        ...options(reference),
        fixedCovariance: covariance("q01", "a", -0.01),
      }),
    ).toThrow(
      "The option fixedCovariance is not positive semidefinite: the variance it adds to an estimate is negative",
    );
    const guessing = { ...options(reference), fixed: { ...reference.fixed } };
    const pretest = reference.itemIds.filter((id) => !(id in reference.fixed));
    expect(() =>
      calibrate(
        matrix,
        reference.itemIds.map((id) => (pretest.includes(id) ? twoParameterLogistic : threeParameterLogistic)),
        {
          ...guessing,
          fixedCovariance: covariance("q01", "c"),
        },
      ),
    ).toThrow(
      'The parameter "c" of item "q01" is 0, on a bound of its range; the standard error needs it inside the range.',
    );
  });

  it("rejects a covariance of another type from JavaScript", () => {
    const reference = defined(independent);
    const matrix = matrixOf(reference);
    expect(() =>
      calibrate(matrix, twoParameterLogistic, { ...options(reference), fixedCovariance: JSON.parse("null") }),
    ).toThrow("The option fixedCovariance must be an object { parameters, matrix }, found null.");
    expect(() =>
      calibrate(matrix, twoParameterLogistic, {
        ...options(reference),
        fixedCovariance: JSON.parse('{"parameters":[{"item":"q01","name":"a"}],"matrix":[["1"]]}'),
      }),
    ).toThrow('Entry (1, 1) of the matrix of the option fixedCovariance must be a finite number, found "1".');
  });
});

describe("fixed parameters", () => {
  const matrix = buildResponseMatrix(fixedParameters.responses, { itemIds: fixedParameters.itemIds });
  const optionsModel = fixedParameters.models.map(model);
  const options = {
    fixedParameters: fixedParameters.fixedParameters,
    tolerance: EM_TOLERANCE,
    maxIterations: 10_000,
  };

  for (const reference of fixedParameters.calibrations) {
    const standardErrors = reference.standardErrors === "cross-product" ? "cross-product" : "hessian";
    it(`match mirt with est = FALSE on the held parameters, with ${standardErrors} standard errors`, () => {
      const result = calibrate(matrix, optionsModel, { ...options, standardErrors });
      expect(result.converged).toBe(true);
      reference.items.forEach((expected, i) => {
        const actual = defined(result.items[i]);
        expectParameters(actual.parameters, expected.parameters, PARAMETER_TOLERANCE, actual.id);
        const errors = defined(actual.standardErrors);
        expect(Object.keys(errors)).toEqual(Object.keys(expected.standardErrors));
        for (const [name, value = 0] of Object.entries(expected.standardErrors)) {
          expectClose(defined(errors[name]), value, STANDARD_ERROR_TOLERANCE * value, `${actual.id} SE of ${name}`);
        }
      });
      expectClose(result.logLikelihood, reference.logLikelihood, LOG_LIKELIHOOD_TOLERANCE, "log-likelihood");
      expectClose(result.aic, reference.aic, 2 * LOG_LIKELIHOOD_TOLERANCE, "AIC");
      expectClose(result.bic, reference.bic, 2 * LOG_LIKELIHOOD_TOLERANCE, "BIC");
    });
  }

  it("keep each held value exactly, and leave it out of the standard errors and the covariance", () => {
    const { calibration, covariance: result } = calibrateWithCovariance(matrix, optionsModel, {
      ...options,
      standardErrors: "hessian",
    });
    expect(defined(calibration.items[0]).parameters).toMatchObject({ c: 0.25 });
    expect(defined(calibration.items[7]).parameters).toMatchObject({ c: 0.2 });
    expect(defined(calibration.items[11]).parameters).toMatchObject({ a: 1 });
    const labels = defined(result).parameters;
    expect(labels).toHaveLength(defined(fixedParameters.calibrations[0]).estimated);
    expect(labels.slice(0, 2)).toEqual([
      { item: "item1", name: "a" },
      { item: "item1", name: "b" },
    ]);
    expect(labels.at(-1)).toEqual({ item: "item12", name: "b" });
    labels.forEach(({ item, name }, row) => {
      const error = defined(calibration.items.find(({ id }) => id === item)?.standardErrors?.[name]);
      expectClose(defined(defined(result).matrix[row]?.[row]), error ** 2, 1e-12 * error ** 2, `${item} ${name}`);
    });
  });

  it("give the free calibration when a parameter is held at its estimate", () => {
    const grades = buildResponseMatrix(science.responses, { itemIds: science.itemIds });
    const free = calibrate(grades, generalizedPartialCredit, { tolerance: EM_TOLERANCE, maxIterations: 10_000 });
    const second = defined(free.items[1]);
    const held = calibrate(grades, generalizedPartialCredit, {
      fixedParameters: { [second.id]: { b2: defined(second.parameters.b[1]) } },
      tolerance: EM_TOLERANCE,
      maxIterations: 10_000,
    });
    free.items.forEach((item, i) => {
      expectParameters(defined(held.items[i]).parameters, item.parameters, 1e-6, item.id);
    });
    expectClose(held.logLikelihood, free.logLikelihood, 1e-8, "log-likelihood");
    expectClose(held.aic, free.aic - 2, 1e-7, "AIC");
  });

  it("move the starts of the other ordered parameters to fit the held ones, and no others", () => {
    const start = Float64Array.of(1, -2, -1, 0, 1);
    const specifications = [
      { name: "a" },
      { name: "b1", ordered: true },
      { name: "b2", ordered: true },
      { name: "b3", ordered: true },
      { name: "b4", ordered: true },
    ];
    startAtHeldParameters(specifications, new Map([["b2", 4]]), start);
    expect(Array.from(start)).toEqual([1, 3, 4, 5, 6]);
    startAtHeldParameters(
      specifications,
      new Map([
        ["b1", 0],
        ["b4", 9],
      ]),
      start,
    );
    expect(Array.from(start)).toEqual([1, 0, 3, 6, 9]);
    const unordered = Float64Array.of(1, -2, -1);
    startAtHeldParameters(
      [{ name: "a" }, { name: "b1" }, { name: "b2" }],
      new Map([
        ["a", 2],
        ["b2", 5],
      ]),
      unordered,
    );
    expect(Array.from(unordered)).toEqual([2, -2, 5]);
  });

  it("give the free calibration when a threshold of a graded item is held at its estimate, whatever the starts", () => {
    const grades = buildResponseMatrix(science.responses, { itemIds: science.itemIds });
    const free = calibrate(grades, gradedResponse, { tolerance: EM_TOLERANCE, maxIterations: 10_000 });
    const second = defined(free.items[1]);
    const b = second.parameters.b;
    for (const fixedParameters of [
      { [second.id]: { b1: defined(b[0]) } },
      { [second.id]: { b1: defined(b[0]), b3: defined(b[2]) } },
      { [second.id]: { b3: defined(b[2]) } },
    ]) {
      const held = calibrate(grades, gradedResponse, {
        fixedParameters,
        tolerance: EM_TOLERANCE,
        maxIterations: 10_000,
      });
      expect(held.converged).toBe(true);
      free.items.forEach((item, i) => {
        expectParameters(defined(held.items[i]).parameters, item.parameters, 1e-5, item.id);
      });
      expectClose(held.logLikelihood, free.logLikelihood, 1e-7, "log-likelihood");
    }
  });

  it("agree with the near-degenerate beta prior that held the guessing parameter", () => {
    // A beta prior of mode c and SD about 0.004 for each held c: the estimates move by little more than that.
    const itemPriors = Object.fromEntries(
      Object.entries(fixedParameters.fixedParameters).flatMap(([id, held]) =>
        "c" in held ? [[id, { c: betaPrior(1 + 10_000 * held.c, 1 + 10_000 * (1 - held.c)) }]] : [],
      ),
    );
    const prior = calibrate(matrix, fixedParameters.models.map(model), {
      itemPriors,
      fixedParameters: { item12: { a: 1 } },
    });
    const held = calibrate(matrix, optionsModel, { ...options, tolerance: 1e-4, maxIterations: 500 });
    held.items.forEach((item, i) => {
      expectParameters(defined(prior.items[i]).parameters, item.parameters, 0.02, item.id);
    });
  });

  it("change nothing when empty", () => {
    const sample = buildResponseMatrix(simulate(20261029, 300, ITEMS_2PL));
    const plain = calibrate(sample, twoParameterLogistic);
    expect(calibrate(sample, twoParameterLogistic, { fixedParameters: {} })).toEqual(plain);
    expect(calibrate(sample, twoParameterLogistic, { fixedParameters: { item1: {} } })).toEqual(plain);
  });

  it("reject unknown and fixed items, parameters the model lacks or shares, every parameter, priors and values", () => {
    const sample = buildResponseMatrix(simulate(20261029, 300, ITEMS_2PL));
    const fixed = { item1: defined(PARAMETERS_2PL[0]) };
    const rejected = (
      value: object | string,
      extra: object = {},
      model: IrtModel<LogisticItem> = twoParameterLogistic,
    ): string | undefined => {
      try {
        const held = typeof value === "string" ? JSON.parse(value) : value;
        calibrate(sample, model, { ...extra, fixedParameters: held });
      } catch (error) {
        return error instanceof DataError ? error.message : undefined;
      }
      return undefined;
    };
    expect(rejected({ q1: { a: 1 } })).toBe(
      'The option fixedParameters holds parameters of "q1", which is not an item id.',
    );
    expect(rejected({ item1: { b: 0 } }, { fixed })).toBe(
      'The option fixedParameters holds parameters of item "item1", which the option fixed holds whole.',
    );
    expect(rejected({ item2: { c: 0.25 } })).toBe(
      'The option fixedParameters holds "c" of item "item2", whose 2PL model has a, b.',
    );
    expect(rejected({ item2: { a: 1, b: 0 } })).toBe(
      'The option fixedParameters holds every parameter of item "item2"; give its parameters in the option fixed.',
    );
    expect(rejected({ item2: { a: 1 } }, { itemPriors: { item2: { a: logNormalPrior(0, 0.5) } } })).toBe(
      'A prior is given for "a" of item "item2", which the option fixedParameters holds.',
    );
    const allSlopes = Object.fromEntries(ITEMS_2PL.map((_, i) => [`item${i + 1}`, { a: 1 }]));
    expect(rejected(allSlopes, { priors: { a: logNormalPrior(0, 0.5) } })).toBe(
      'A prior is given for "a", which no estimated item has.',
    );
    expect(rejected({ item2: { a: 1 } }, {}, oneParameterLogistic)).toBe(
      'The option fixedParameters holds "a" of item "item2", whose 1PL model shares it among items.',
    );
    const thresholds = buildResponseMatrix(science.responses, { itemIds: science.itemIds });
    expect(() =>
      calibrate(thresholds, gradedResponse, {
        fixedParameters: { [defined(science.itemIds[0])]: { b1: 1, b2: 0 } },
      }),
    ).toThrow(/with its held parameters: A GRM item .* thresholds b strictly increasing/);
    expect(rejected({ item2: { c: 1 } }, {}, threeParameterLogistic)).toMatch(
      /^Item "item2" with its held parameters: Logistic item parameters need finite a and b and 0 ≤ c < d ≤ 1, found .* c = 1, d = 1\.$/,
    );
    expect(rejected("null")).toBe(
      "The option fixedParameters must be an object of values by item id and parameter name, found null.",
    );
    expect(rejected('{ "item1": [] }')).toBe(
      'The held parameters of item "item1" must be an object of values by parameter name, found an array.',
    );
    expect(rejected('{ "item1": { "a": "1" } }')).toBe(
      'The held value of "a" of item "item1" must be a finite number, found "1".',
    );
    expect(rejected('{ "constructor": {} }')).toBe(
      'The option fixedParameters holds parameters of "constructor", which is not an item id.',
    );
  });
});

describe("conditionalMaximumLikelihood", () => {
  /** A case of the fixture: its expected parameters by item, one threshold or a list. */
  type ConditionalCase = (typeof conditional.cases)[number];
  const listed = (value: number | readonly number[] | undefined): readonly number[] =>
    value === undefined ? [] : typeof value === "number" ? [value] : value;

  function expectConditional(result: ConditionalCalibration, expected: ConditionalCase, tolerance: number): void {
    expect(result.converged).toBe(true);
    expect(result.parameters).toBe(expected.parameters);
    expectClose(result.logLikelihood, expected.logLikelihood, tolerance);
    result.items.forEach((item, i) => {
      const errors = defined(item.standardErrors);
      if (expected.locations !== undefined) {
        const parameters = item.parameters as RatingScaleParameters;
        const steps = defined(expected.steps);
        const stepErrors = defined(expected.stepErrors);
        expect(item.model).toBe(ratingScale);
        expectClose(parameters.b, expected.locations[i] ?? null, tolerance, `location ${i + 1}`);
        parameters.steps.forEach((step, k) => {
          expectClose(step, steps[k] ?? null, tolerance, `step ${k + 1}`);
        });
        expectClose(errors.b ?? null, defined(expected.locationErrors)[i] ?? null, tolerance, `location SE ${i + 1}`);
        Object.entries(errors)
          .filter(([name]) => name.startsWith("step"))
          .forEach(([, error], k) => {
            expectClose(error, stepErrors[k] ?? null, tolerance, `step SE ${k + 1}`);
          });
        return;
      }
      const thresholds = listed(expected.thresholds[i]);
      const parameters = item.parameters as LogisticItem | ThresholdItem;
      expect(listed(parameters.b).length).toBe(thresholds.length);
      listed(parameters.b).forEach((b, k) => {
        expectClose(b, thresholds[k] ?? null, tolerance, `${item.id} b${k + 1}`);
      });
      const expectedErrors = listed(expected.standardErrors[i]);
      Object.values(errors).forEach((error, k) => {
        expectClose(error, expectedErrors[k] ?? null, tolerance, `${item.id} SE ${k + 1}`);
      });
    });
  }

  for (const expected of conditional.cases) {
    const reference = expected.erm === undefined ? "eRm" : "base R's conditional likelihood";
    it(`matches ${reference} on ${expected.name}`, () => {
      const model =
        expected.model === "rating-scale" ? "rating-scale" : expected.model === "rasch" ? "rasch" : "partial-credit";
      const result = conditionalMaximumLikelihood(buildResponseMatrix(expected.responses), { model });
      expectConditional(result, expected, expected.erm === undefined ? ERM_TOLERANCE : CONDITIONAL_TOLERANCE);
    });
  }

  it("maximizes the conditional likelihood of each person's answered items, where eRm's PCM does not on missing ones", () => {
    const expected = defined(conditional.cases.find((candidate) => candidate.erm !== undefined));
    const erm = defined(expected.erm);
    const { logLikelihood } = conditionalMaximumLikelihood(buildResponseMatrix(expected.responses));
    expect(logLikelihood).toBeGreaterThan(erm.atItsEstimates + 1);
    expect(Math.abs(erm.logLikelihood - erm.atItsEstimates)).toBeGreaterThan(1);
  });

  it("gives the values of the guide's sample, where item 3's thresholds are -ln 2 and ln 2", () => {
    const result = conditionalMaximumLikelihood(
      buildResponseMatrix([
        [1, 0, 2],
        [0, 0, 1],
        [1, 1, 2],
        [0, 1, 0],
        [1, 1, 1],
        [0, 0, 0],
      ]),
    );
    const item3 = defined(result.items[2]);
    expect(item3.model).toBe(partialCredit);
    const [lower = 0, upper = 0] = (item3.parameters as ThresholdItem).b;
    expect([lower + Math.LN2, upper - Math.LN2].map((value) => Math.abs(value) < 1e-9)).toEqual([true, true]);
    expect(Object.values(defined(item3.standardErrors)).map((error) => Number(error.toPrecision(3)))).toEqual([
      1.12, 1.12,
    ]);
    expect(result.persons).toBe(4);
  });

  it("chooses the Rasch model for 0/1 responses and the partial credit model otherwise, as 1PL and PCM items", () => {
    const rasch = conditionalMaximumLikelihood(buildResponseMatrix(defined(conditional.cases[0]).responses));
    expect(rasch.items.every((item) => item.model === oneParameterLogistic && item.parameters.a === 1)).toBe(true);
    const mixed = conditionalMaximumLikelihood(buildResponseMatrix(defined(conditional.cases.at(-1)).responses));
    expect(mixed.items.map((item) => item.model.name)).toEqual(["1PL", "1PL", "1PL", "PCM", "PCM"]);
    const sum = mixed.items.flatMap((item) => listed((item.parameters as LogisticItem | ThresholdItem).b));
    expectClose(
      sum.reduce((total, b) => total + b, 0),
      0,
      1e-12,
    );
  });

  it("recovers the difficulties of 2,000 persons, with 95% intervals that cover them", () => {
    // Thresholds fixed before the first run: RMSE at most 0.06, coverage from 0.92 to 0.98 over 40 sets of 15 items.
    const random = createRandom(2026);
    let squares = 0;
    let covered = 0;
    let count = 0;
    for (let set = 0; set < 40; set++) {
      const difficulties = Array.from({ length: 15 }, (_, i) => -2 + (4 * i) / 14);
      const items = difficulties.map((b) => ({ model: oneParameterLogistic, parameters: { a: 1, b, c: 0, d: 1 } }));
      const { items: estimates } = conditionalMaximumLikelihood(
        simulateResponses(items, simulateThetas(2000, random), random),
      );
      estimates.forEach((item, i) => {
        const error = (item.parameters as LogisticItem).b - (difficulties[i] ?? 0);
        squares += error ** 2;
        if (Math.abs(error) <= normalQuantile(0.975) * (item.standardErrors?.b ?? 0)) covered++;
        count++;
      });
    }
    expect(Math.sqrt(squares / count)).toBeLessThanOrEqual(0.06);
    expect(covered / count).toBeGreaterThanOrEqual(0.92);
    expect(covered / count).toBeLessThanOrEqual(0.98);
  });

  it("leaves out persons with an extreme score or one answered item, and gives items that itemFit reads", () => {
    const rows = [
      [1, 0, 1],
      [0, 1, 0],
      [1, 1, 0],
      [0, 0, 1],
      [1, 1, 1],
      [0, 0, 0],
      [1, null, null],
      [1, 0, null],
    ];
    const matrix = buildResponseMatrix(rows);
    const result = conditionalMaximumLikelihood(matrix);
    // Persons 5 and 6 are extreme, person 7 answered one item: 5 persons remain.
    expect(result.persons).toBe(5);
    const thetas = rows.map(() => 0);
    expect(itemFit(matrix, result.items, thetas)).toHaveLength(3);
  });

  it("reports a fit that ran out of steps, and the rating scale model of dichotomous items as the Rasch model", () => {
    const expected = defined(conditional.cases[0]);
    const stopped = conditionalMaximumLikelihood(buildResponseMatrix(expected.responses), { maxIterations: 1 });
    expect([stopped.converged, stopped.iterations]).toEqual([false, 1]);
    const rasch = conditionalMaximumLikelihood(buildResponseMatrix(expected.responses));
    const rating = conditionalMaximumLikelihood(buildResponseMatrix(expected.responses), { model: "rating-scale" });
    rating.items.forEach((item, i) => {
      const parameters = item.parameters as RatingScaleParameters;
      expect(parameters.steps).toEqual([0]);
      expectClose(parameters.b, (defined(rasch.items[i]).parameters as LogisticItem).b, 1e-10);
    });
  });

  it("rejects an item every used person answered alike, too few items, and responses its model cannot take", () => {
    const allRight = buildResponseMatrix([
      [1, 0, 1],
      [1, 1, 0],
      [1, 0, 0],
    ]);
    expect(() => conditionalMaximumLikelihood(allRight)).toThrow('Item "item1" has no response in category 0');
    // Item 1's one wrong response is the extreme person's, whom the conditional likelihood leaves out.
    const rightAmongUsed = buildResponseMatrix([
      [0, 0, 0],
      [1, 0, 1],
      [1, 1, 0],
      [1, 0, 0],
    ]);
    expect(() => conditionalMaximumLikelihood(rightAmongUsed)).toThrow('Item "item1" has no response in category 0');
    expect(() => conditionalMaximumLikelihood(buildResponseMatrix([[0], [1]]))).toThrow(
      "Conditional maximum likelihood needs at least two items, found 1.",
    );
    const polytomous = buildResponseMatrix([
      [0, 1, 2],
      [1, 2, 0],
      [2, 0, 1],
      [1, 1, 1],
    ]);
    expect(() => conditionalMaximumLikelihood(polytomous, { model: "rasch" })).toThrow(DataError);
    const unequal = buildResponseMatrix([
      [0, 1, 2],
      [1, 0, 0],
      [1, 1, 1],
      [0, 0, 2],
    ]);
    expect(() => conditionalMaximumLikelihood(unequal, { model: "rating-scale" })).toThrow(
      'The rating scale model needs items with the same number of categories; item "item1" has 2 and item "item3" has 3.',
    );
  });

  it("refuses a response far above the others from the categories that occur, before it sizes the parameters", () => {
    // A response of 2³¹ - 1 overflowed the 32-bit offsets of the parameters, and one of 2³⁰ sized gigabytes for them.
    for (const far of [2 ** 31 - 1, 2 ** 30]) {
      const rows = Array.from({ length: 50 }, (_, p) => Array.from({ length: 5 }, (_, i) => (p + i) % 2));
      defined(rows[0])[0] = far;
      expect(() => conditionalMaximumLikelihood(buildResponseMatrix(rows))).toThrow(
        'Item "item1" has no response in category 2; calibration needs responses in every category up to the highest.',
      );
    }
  }, 5_000);

  it("takes at most 2^29 products of the functions with two items left out in a Newton step", () => {
    // Three persons answer 284 items: the pairs of one group take Σ j² (284 - j) products, 0.7 s a step at 283 items.
    const rows = [0, 1, 2].map((k) => Array.from({ length: 284 }, (_, j) => (j + k) % 2));
    expect(() => conditionalMaximumLikelihood(buildResponseMatrix(rows), { maxIterations: 1 })).toThrow(
      "Conditional maximum likelihood would take 542109140 products of the functions with two items left out in each Newton step, more than 536870912,",
    );
  });

  it("rejects options of other types", () => {
    const matrix = buildResponseMatrix(defined(conditional.cases[0]).responses);
    expect(() => conditionalMaximumLikelihood(matrix, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => conditionalMaximumLikelihood(matrix, JSON.parse('{"verbose":1}'))).toThrow(
      'There is no option "verbose"',
    );
    expect(() => conditionalMaximumLikelihood(matrix, { model: JSON.parse('"RSM"') })).toThrow(DataError);
    expect(() => conditionalMaximumLikelihood(matrix, { tolerance: JSON.parse('"small"') })).toThrow(DataError);
    expect(() => conditionalMaximumLikelihood(JSON.parse("[]"))).toThrow(DataError);
  });
});

describe("expectationStep", () => {
  const rule = rectangular(61);
  const rows = simulate(5, 30, ITEMS_2PL).map((row, person) => (person === 7 ? row.map(() => null) : row));
  const data = compressResponses(buildResponseMatrix(rows));
  const answered = rows.map((row) => row.some((response) => response !== null));

  function items(): ExpectationItem[] {
    return ITEMS_2PL.map(({ model: itemModel, parameters }) => ({
      model: itemModel,
      categoryCount: 2,
      nodes: rule.nodes,
      parameters,
      counts: new Float64Array(rule.nodes.length * 2),
    }));
  }

  function groups(logWeights?: Float64Array) {
    const result = population({}, answered, rule, true, false, () => 0, null);
    if (logWeights !== undefined) result.groups[0]?.logWeights.set(logWeights);
    return result;
  }

  const logWeights = rule.weights.map(Math.log);
  const flat = new Float64Array(rule.nodes.length);
  const respondents = answered.flatMap((any, person) => (any ? [person] : []));

  it("treats a term of the log prior weights, by node, as the prior weights", () => {
    const standard = items();
    const standardGroups = groups();
    const expected = expectationStep(data, standard, rule.nodes, standardGroups).run();
    const terms = new Float64Array(rows.length * rule.nodes.length);
    for (let person = 0; person < rows.length; person++) terms.set(logWeights, person * rule.nodes.length);
    const termItems = items();
    const termGroups = groups(flat);
    const actual = expectationStep(data, termItems, rule.nodes, termGroups, null, terms).run();
    expectClose(actual, expected, 1e-10);
    termItems.forEach((item, i) => {
      item.counts.forEach((count, k) => {
        expectClose(count, defined(standard[i]?.counts[k]), 1e-10);
      });
    });
    termGroups.groups[0]?.counts.forEach((count, q) => {
      expectClose(count, defined(standardGroups.groups[0]?.counts[q]), 1e-10);
    });
  });

  it("adds a term that is constant over the nodes to the log-likelihood and changes nothing else", () => {
    const base = items();
    const baseLogLikelihood = expectationStep(data, base, rule.nodes, groups()).run();
    const terms = new Float64Array(rows.length * rule.nodes.length);
    const shifts = rows.map((_, person) => (person % 5) - 2);
    shifts.forEach((shift, person) => {
      terms.fill(shift, person * rule.nodes.length, (person + 1) * rule.nodes.length);
    });
    const shifted = items();
    const logLikelihood = expectationStep(data, shifted, rule.nodes, groups(), null, terms).run();
    expectClose(
      logLikelihood,
      baseLogLikelihood + respondents.reduce((sum, person) => sum + defined(shifts[person]), 0),
      1e-10,
    );
    shifted.forEach((item, i) => {
      item.counts.forEach((count, k) => {
        expectClose(count, defined(base[i]?.counts[k]), 1e-10);
      });
    });
  });

  it("reads the term at each run and ignores it for persons without responses", () => {
    const terms = new Float64Array(rows.length * rule.nodes.length);
    const step = expectationStep(data, items(), rule.nodes, groups(), null, terms);
    const first = step.run();
    terms.fill(3, 7 * rule.nodes.length, 8 * rule.nodes.length);
    expect(step.run()).toBe(first);
    terms.fill(-2, 0, rule.nodes.length);
    expectClose(step.run(), first - 2, 1e-10);
  });

  it("gives posteriors that follow the term", () => {
    const terms = new Float64Array(rows.length * rule.nodes.length).fill(-1000);
    for (let person = 0; person < rows.length; person++) terms[person * rule.nodes.length + 40] = 0;
    const step = expectationStep(data, items(), rule.nodes, groups(), null, terms);
    step.run();
    let visited = 0;
    step.forEachPosterior((person, posteriorWeights) => {
      visited++;
      expectClose(defined(posteriorWeights[40]), 1, 1e-12, `person ${person}`);
    });
    expect(visited).toBe(respondents.length);
  });
});

describe("gradedResponse", () => {
  it("names the slope a and one threshold b for each boundary between categories", () => {
    expect(gradedResponse.parameters(4)).toEqual([
      { name: "a" },
      { name: "b1", ordered: true },
      { name: "b2", ordered: true },
      { name: "b3", ordered: true },
    ]);
    expect(gradedResponse.categoryCount({ a: 1, b: [0, 1] })).toBe(3);
  });

  it("starts at thresholds that give a person at θ = 0 the shares of the categories", () => {
    const shares = Float64Array.of(0.1, 0.2, 0.3, 0.4);
    const start = gradedResponse.startingValues(shares);
    const probabilities = new Float64Array(4);
    gradedResponse.probabilities(start, 0, probabilities);
    probabilities.forEach((p, k) => {
      expectClose(p, defined(shares[k]), 1e-15);
    });
  });

  it("keeps the precision of small probabilities in both tails", () => {
    const item = { a: 1, b: [-1, 0, 1] };
    const probabilities = new Float64Array(4);
    gradedResponse.probabilities(item, 30, probabilities);
    // P(X = 0) = 1 / (1 + exp(a (θ - b1))) = 1 / (1 + exp(31)).
    expectClose(defined(probabilities[0]) / (1 / (1 + Math.exp(31))), 1, 1e-14);
    gradedResponse.probabilities(item, -30, probabilities);
    expectClose(defined(probabilities[3]) / (1 / (1 + Math.exp(31))), 1, 1e-14);
  });

  it("rejects thresholds out of order for the sign of a", () => {
    expect(() => gradedResponse.toVector({ a: 1, b: [0, -1] })).toThrow(
      "A GRM item with more than two categories needs a nonzero a and thresholds b strictly increasing when a > 0 and strictly decreasing when a < 0, found a = 1, b = [0, -1].",
    );
    expect([...gradedResponse.toVector({ a: -1, b: [0, -1] })]).toEqual([-1, 0, -1]);
    expect(() => gradedResponse.toVector({ a: 0, b: [0, 1] })).toThrow(DataError);
    expect(() => gradedResponse.toVector({ a: 1, b: [] })).toThrow(
      "A GRM item needs a finite a and at least one finite threshold b, found a = 1, b = [].",
    );
  });
});

describe("information", () => {
  for (const fixture of FIXTURES) {
    for (const reference of fixture.information ?? []) {
      it(`matches mirt's iteminfo and testinfo for the ${modelNames(reference)} items of ${fixture.source}`, () => {
        const pool = fixtureItems(reference);
        reference.theta.forEach((theta, j) => {
          pool.forEach((item, i) => {
            const expected = defined(reference.items[i]?.[j]);
            expectClose(itemInformation(item, theta), expected, INFORMATION_TOLERANCE, `item ${i + 1}`);
          });
          const test = defined(reference.test[j]);
          expectClose(testInformation(pool, theta), test, INFORMATION_TOLERANCE, `test at ${theta}`);
          const standardError = 1 / Math.sqrt(test);
          expectClose(conditionalStandardError(pool, theta), standardError, 1e-9 * standardError, `SE at ${theta}`);
        });
      });
    }
  }

  it("gives an infinite standard error where there is no information", () => {
    const flat = [{ model: twoParameterLogistic, parameters: { a: 0, b: 0, c: 0, d: 1 } }];
    expect(testInformation(flat, 1)).toBe(0);
    expect(conditionalStandardError(flat, 1)).toBe(Number.POSITIVE_INFINITY);
    expect(testInformation([], 0)).toBe(0);
  });

  it("stays finite for steep divide-by-total items whose lowest category underflows", () => {
    // At θ = 0.5 the GPCM's category 2 has logit 1000 more than category 0, whose weight exp(-1000) is 0.
    const gpcm = { model: generalizedPartialCredit, parameters: { a: 1000, b: [-0.5, 0] } };
    const nrm = { model: nominalResponse, parameters: { a: [0, 1000, 2000], c: [0, 0, 0] } };
    for (const item of [gpcm, nrm]) {
      for (const theta of [-0.5, 0.5, 3]) expect(Number.isFinite(itemInformation(item, theta))).toBe(true);
      expect(Number.isFinite(marginalReliability([item]))).toBe(true);
    }
    // Category 1's share is logistic(-500) ≈ 7e-218 at θ = 0.5, so the variance of the category is about that.
    expect(itemInformation(gpcm, 0.5)).toBeCloseTo(0, 10);
  });

  it("rejects invalid theta and items", () => {
    const item = { model: twoParameterLogistic, parameters: { a: 1, b: 0, c: 0, d: 1 } };
    expect(() => itemInformation(item, Number.NaN)).toThrow("Theta must be a finite number, found NaN.");
    expect(() => testInformation([item, { ...item, parameters: { a: 1, b: Number.NaN, c: 0, d: 1 } }], 0)).toThrow(
      "Item 2: Logistic item parameters need finite a and b and 0 ≤ c < d ≤ 1, found a = 1, b = NaN, c = 0, d = 1.",
    );
    expect(() => testInformation(JSON.parse("null"), 0)).toThrow("The items must be an array, found null.");
    expect(() => itemInformation(item, JSON.parse('"0"'))).toThrow('Theta must be a finite number, found "0".');
  });
});

describe("jointMaximumLikelihood", () => {
  const reference = simulatedPcm.jointMaximumLikelihood;

  it(`matches ${reference.name} on ${simulatedPcm.source}`, () => {
    const matrix = buildResponseMatrix(simulatedPcm.responses, { itemIds: simulatedPcm.itemIds });
    const result = jointMaximumLikelihood(matrix, { tolerance: 1e-12, maxIterations: 10_000 });
    expect(result.converged).toBe(true);
    reference.items.forEach((expected, i) => {
      const actual = defined(result.items[i]);
      expect(actual.id).toBe(simulatedPcm.itemIds[i]);
      expect(actual.model).toBe(model(expected.model));
      expectParameters(actual.parameters, expected.parameters, JOINT_TOLERANCE, actual.id);
      const errors = defined(actual.standardErrors);
      expect(Object.keys(errors)).toEqual(Object.keys(expected.standardErrors));
      for (const [name, value] of Object.entries(expected.standardErrors)) {
        expectClose(defined(errors[name]), value, JOINT_TOLERANCE, `${actual.id} SE of ${name}`);
      }
    });
    reference.persons.forEach((expected, p) => {
      const actual = defined(result.persons[p]);
      expectClose(actual.theta, expected.theta, JOINT_TOLERANCE, `person ${p + 1} theta`);
      expectClose(actual.standardError, expected.standardError, JOINT_TOLERANCE, `person ${p + 1} SE`);
    });
    expectClose(result.logLikelihood, reference.logLikelihood, JOINT_TOLERANCE, "log-likelihood");
  });

  const rows = [
    [1, 1, 2],
    [0, 0, 0],
    [1, 0, 1],
    [0, 1, 0],
    [null, null, null],
    [1, null, 2],
    [0, 1, 1],
  ];

  it("gives finite thetas at extreme scores and none to persons without responses", () => {
    const result = jointMaximumLikelihood(buildResponseMatrix(rows));
    const [high, low, , , none, partial] = result.persons;
    expect(none).toEqual({ id: "person5", theta: null, standardError: null });
    for (const person of [high, low, partial]) expect(Number.isFinite(person?.theta)).toBe(true);
    expect(defined(high?.theta)).toBeGreaterThan(defined(low?.theta));
    expect(result.items.map((item) => item.model)).toEqual([oneParameterLogistic, oneParameterLogistic, partialCredit]);
    expect(nonFinite(result)).toEqual([]);
  });

  it("reports non-convergence within the iteration limit", () => {
    expect(jointMaximumLikelihood(buildResponseMatrix(rows), { maxIterations: 1 })).toMatchObject({
      converged: false,
      iterations: 1,
    });
  });

  it("rejects fewer than two items, items it cannot estimate and invalid options", () => {
    expect(() => jointMaximumLikelihood(buildResponseMatrix([[0], [1]]))).toThrow(
      "Joint maximum likelihood needs at least two items, found 1.",
    );
    expect(() => jointMaximumLikelihood(buildResponseMatrix(rows.map(([a, b]) => [a ?? null, b ?? null, 2])))).toThrow(
      'Item "item3" has no response in category 0; calibration needs responses in every category up to the highest.',
    );
    expect(() => jointMaximumLikelihood(buildResponseMatrix(rows), { tolerance: -1 })).toThrow(DataError);
    expect(() => jointMaximumLikelihood(buildResponseMatrix(rows), { maxIterations: 0 })).toThrow(
      "The option maxIterations must be a positive integer, found 0.",
    );
    expect(() => jointMaximumLikelihood(buildResponseMatrix(rows), JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => jointMaximumLikelihood(buildResponseMatrix(rows), JSON.parse('{"verbose":1}'))).toThrow(
      'There is no option "verbose"',
    );
  });
});

describe("distributions of theta", () => {
  for (const reference of latent.calibrations) {
    it(`matches ${reference.name}`, () => {
      const groups = reference.groups ?? undefined;
      const distributions: Readonly<Record<string, LatentDistribution>> = Object.fromEntries(
        Object.entries(reference.distributions).map(([group, name]) => [group, latentDistribution(name)]),
      );
      const result = calibrate(
        buildResponseMatrix(reference.responses, { itemIds: reference.itemIds }),
        twoParameterLogistic,
        {
          fixed: definedEntries(reference.fixed),
          ...(groups === undefined ? { latent: defined(distributions.all) } : { groups, latent: distributions }),
          tolerance: EM_TOLERANCE,
          maxIterations: 10_000,
          standardErrors: "hessian",
        },
      );
      expect(result.converged).toBe(true);
      reference.items.forEach((expected, i) => {
        const actual = defined(result.items[i]);
        expectParameters(actual.parameters, expected.parameters, LATENT_PARAMETER_TOLERANCE, actual.id);
        const errors = "standardErrors" in expected ? expected.standardErrors : undefined;
        if (errors === undefined) {
          expect(actual.standardErrors).toBeNull();
          return;
        }
        for (const [name, value] of Object.entries(errors)) {
          const relative = LATENT_STANDARD_ERROR_TOLERANCE * value;
          expectClose(defined(actual.standardErrors?.[name]), value, relative, `${actual.id} SE of ${name}`);
        }
      });
      for (const expected of reference.groupDistributions) {
        const actual = defined(result.groups.find((group) => (group.group ?? "all") === expected.group));
        expect(actual.distribution).toBe("normal");
        expectClose(actual.mean, expected.mean, LATENT_PARAMETER_TOLERANCE, "mean");
        expectClose(actual.sd, expected.sd, LATENT_PARAMETER_TOLERANCE, "sd");
        const errors = defined(actual.standardErrors);
        expectClose(
          errors.mean,
          expected.standardErrors.mean,
          LATENT_STANDARD_ERROR_TOLERANCE * errors.mean,
          "SE of mean",
        );
        expectClose(errors.sd, expected.standardErrors.sd, LATENT_STANDARD_ERROR_TOLERANCE * errors.sd, "SE of sd");
      }
      expectClose(result.logLikelihood, reference.logLikelihood, LOG_LIKELIHOOD_TOLERANCE, "log-likelihood");
      expectClose(result.aic, reference.aic, 2 * LOG_LIKELIHOOD_TOLERANCE, "AIC");
      expectClose(result.bic, reference.bic, 2 * LOG_LIKELIHOOD_TOLERANCE, "BIC");
    });
  }

  it(`matches ${latent.histogram.name}`, () => {
    const { histogram } = latent;
    const result = calibrate(
      buildResponseMatrix(histogram.responses, { itemIds: histogram.itemIds }),
      twoParameterLogistic,
      {
        fixed: histogram.fixed,
        latent: "empirical-histogram",
        tolerance: histogram.tolerance,
        maxIterations: 10_000,
      },
    );
    expect(result).toMatchObject({ converged: true, iterations: histogram.iterations });
    const group = defined(result.groups[0]);
    group.weights.forEach((weight, q) => {
      expectClose(weight, defined(histogram.weights[q]), HISTOGRAM_TOLERANCE);
    });
    expectClose(result.logLikelihood, histogram.logLikelihood, 1e-8, "log-likelihood");
    expectClose(result.aic, -2 * result.logLikelihood + 2 * 60, 1e-9, "AIC");
    const mean = group.weights.reduce((sum, weight, q) => sum + weight * defined(group.nodes[q]), 0);
    expectClose(group.mean, mean, 1e-15);
  });

  it("recovers pretest items seeded into simulated adaptive tests, with a normal distribution or a histogram", () => {
    const random = createRandom(20261029);
    const pool = simulateItems(twoParameterLogistic, 60, random).map((item, i) => ({ ...item, id: `op${i + 1}` }));
    const pretest = simulateItems(twoParameterLogistic, 4, random);
    const thetas = simulateThetas(3000, random).map((theta) => 0.4 + 1.2 * theta);
    const sessions = simulateAdaptiveTest(createAdaptiveTest(pool, testLength(12)), thetas, random).persons;
    const seeded = simulateResponses(pretest, thetas, random).responses;
    const itemIds = [...pool.map((item) => item.id), "pre1", "pre2", "pre3", "pre4"];
    // Each session answers its adaptive items and two of the four pretest items.
    const rows = sessions.map(({ state }, person) => {
      const row: (number | null)[] = itemIds.map(() => null);
      state.items.forEach((id, j) => {
        row[itemIds.indexOf(id)] = defined(state.responses[j]);
      });
      for (const k of [person % 4, (person + 1) % 4]) row[pool.length + k] = defined(seeded[person]?.[k]);
      return row;
    });
    const matrix = buildResponseMatrix(rows, { itemIds });
    const fixed = Object.fromEntries(pool.map((item) => [item.id, item.parameters]));
    for (const distribution of ["normal", "empirical-histogram"] as const) {
      const result = calibrate(matrix, twoParameterLogistic, {
        fixed,
        latent: distribution,
        standardErrors: "cross-product",
      });
      expect(result.converged).toBe(true);
      pretest.forEach((item, k) => {
        const estimate = defined(result.items[pool.length + k]);
        const errors = defined(estimate.standardErrors);
        expect(Math.abs(estimate.parameters.a - item.parameters.a)).toBeLessThan(4 * defined(errors.a));
        expect(Math.abs(estimate.parameters.b - item.parameters.b)).toBeLessThan(4 * defined(errors.b));
      });
      const group = defined(result.groups[0]);
      expect(Math.abs(group.mean - 0.4)).toBeLessThan(0.1);
      expect(Math.abs(group.sd - 1.2)).toBeLessThan(0.1);
    }
  });

  it("gives the estimates of one group to groups that all keep the standard normal", () => {
    const rows = simulate(7, 400, ITEMS_2PL);
    const groups = rows.map((_, person) => (person % 2 === 0 ? "even" : "odd"));
    const one = calibrate(buildResponseMatrix(rows), twoParameterLogistic);
    const two = calibrate(buildResponseMatrix(rows), twoParameterLogistic, { groups });
    two.items.forEach((item, i) => {
      expectParameters(item.parameters, defined(one.items[i]).parameters, 1e-12, item.id);
    });
    expect(two.groups.map((group) => group.group)).toEqual(["even", "odd"]);
    expectClose(two.logLikelihood, one.logLikelihood, 1e-9);
  });

  it("estimates the distribution of a group on fixed items alone", () => {
    const rows = simulate(11, 500, ITEMS_2PL);
    const fixed = Object.fromEntries(PARAMETERS_2PL.map((item, i) => [`item${i + 1}`, item]));
    const result = calibrate(buildResponseMatrix(rows), twoParameterLogistic, {
      fixed,
      latent: "normal",
      standardErrors: "hessian",
    });
    const group = defined(result.groups[0]);
    expect(result.converged).toBe(true);
    expect(Math.abs(group.mean)).toBeLessThan(4 * defined(group.standardErrors).mean);
    expectClose(result.aic, -2 * result.logLikelihood + 4, 1e-9);
  });

  it("estimates a normal distribution whose SD falls below the spacing of the nodes", () => {
    // Four persons whose likelihoods agree on one theta: the SD's maximum likelihood estimate tends to 0.
    const matrix = buildResponseMatrix([
      [1, 0, 1],
      [0, 1, 1],
      [1, 1, 0],
      [0, 0, 1],
    ]);
    const fixed = {
      item1: { a: 1.2, b: 0.32, c: 0, d: 1 },
      item2: { a: 0.99, b: 0.82, c: 0, d: 1 },
      item3: { a: 1.21, b: -0.22, c: 0, d: 1 },
    };
    const result = calibrate(matrix, twoParameterLogistic, { fixed, latent: "normal", maxIterations: 2000 });
    const group = defined(result.groups[0]);
    expect(Number.isFinite(group.mean)).toBe(true);
    expect(group.sd).toBeGreaterThan(0);
    expect(group.sd).toBeLessThan(0.2);
    // The SD is below the spacing of the nodes, 0.2: the likelihood has no maximum there.
    expect(result.converged).toBe(false);
    const first = calibrate(matrix, twoParameterLogistic, { fixed, latent: "normal", maxIterations: 1 });
    expect(result.logLikelihood).toBeGreaterThan(first.logLikelihood);
  });

  it("keeps a normal distribution whose maximum lies beyond the nodes finite, without standard errors", () => {
    // Three of the focal group's four persons answer the item in its top category: its mean runs to +∞.
    const responses = [[2], [2], [1], [0], [1], [2], [1], [2], [1]];
    const groups = responses.map((_, person) => (person % 2 === 1 ? "focal" : "reference"));
    const result = calibrate(buildResponseMatrix(responses), generalizedPartialCredit, {
      groups,
      latent: { focal: "normal" },
      standardErrors: "hessian",
    });
    const focal = defined(result.groups[1]);
    expect(Number.isFinite(focal.mean) && Number.isFinite(focal.sd) && Number.isFinite(result.logLikelihood)).toBe(
      true,
    );
    expect(focal.mean).toBeGreaterThan(6);
    expect(focal.standardErrors).toBeNull();
    expect(result.converged).toBe(false);
  });

  it("takes groups named as the properties of every object as groups without a distribution of their own", () => {
    const rows = simulate(9, 200, ITEMS_2PL);
    const groups = rows.map((_, person) => ["constructor", "toString", "reference"][person % 3] ?? "reference");
    const result = calibrate(buildResponseMatrix(rows), twoParameterLogistic, {
      groups,
      latent: { reference: "normal" },
      fixed: { item1: defined(PARAMETERS_2PL[0]) },
    });
    expect(result.groups.map(({ group, distribution }) => [group, distribution])).toEqual([
      ["constructor", "standard-normal"],
      ["toString", "standard-normal"],
      ["reference", "normal"],
    ]);
  });

  it("checks many groups in time that grows with the persons", () => {
    // 100,000 persons in 50,000 groups: comparing each person with every group would take 5 billion steps, twice.
    const rows = simulate(12, 100_000, ITEMS_2PL.slice(0, 2));
    const groups = rows.map((_, person) => `g${person >> 1}`);
    const options = { groups, quadrature: { points: 5 }, maxIterations: 1 };
    const result = calibrate(buildResponseMatrix(rows), twoParameterLogistic, options);
    expect(result.groups).toHaveLength(50_000);
    expect(result.groups[49_999]?.group).toBe("g49999");
  }, 3_000);

  it("rejects groups and distributions that do not fit the data or leave the scale free", () => {
    const rows = simulate(3, 50, ITEMS_2PL);
    const matrix = buildResponseMatrix(rows);
    const labels = rows.map((_, person) => (person < 25 ? "a" : "b"));
    expect(() => calibrate(matrix, twoParameterLogistic, { groups: ["a"] })).toThrow("Found 1 groups for 50 persons.");
    expect(() =>
      calibrate(matrix, twoParameterLogistic, {
        groups: JSON.parse(JSON.stringify(labels.map((_, i) => (i === 3 ? 3 : "a")))),
      }),
    ).toThrow("The group of person 4 must be a string, found 3.");
    const sparse = [...labels];
    delete sparse[7];
    expect(() => calibrate(matrix, twoParameterLogistic, { groups: sparse })).toThrow(
      "The group of person 8 must be a string, found undefined.",
    );
    expect(() => calibrate(matrix, twoParameterLogistic, { latent: "normal" })).toThrow(
      "Estimating every group's distribution of theta needs fixed items or a standard normal group to fix the scale.",
    );
    expect(() => calibrate(matrix, twoParameterLogistic, { latent: { a: "normal" } })).toThrow(
      "Distributions of theta by group need the groups option.",
    );
    expect(() => calibrate(matrix, twoParameterLogistic, { groups: labels, latent: { c: "normal" } })).toThrow(
      'A distribution of theta is given for group "c", which no person is in.',
    );
    expect(() =>
      calibrate(matrix, twoParameterLogistic, { groups: labels, latent: JSON.parse('{ "b": "uniform" }') }),
    ).toThrow(
      'The distribution of theta of group "b" must be "standard-normal", "normal" or "empirical-histogram", found "uniform".',
    );
    const fixed = { item1: defined(PARAMETERS_2PL[0]) };
    expect(() =>
      calibrate(matrix, twoParameterLogistic, {
        fixed,
        latent: "normal",
        quadrature: { method: "gauss-hermite", points: 21 },
      }),
    ).toThrow("Estimating a distribution of theta needs the rectangular quadrature rule.");
    const silent = buildResponseMatrix([...rows, [null, null, null, null]]);
    expect(() => calibrate(silent, twoParameterLogistic, { groups: [...labels, "c"] })).toThrow(
      'Group "c" has no person with a response.',
    );
  });
});

describe("latent regression", () => {
  type Reference = (typeof latentRegression.cases)[number];
  const covariatesOf = (reference: Reference) => definedEntries(reference.covariates);
  const calibrateReference = (reference: Reference, standardErrors: StandardErrorMethod = "none") =>
    calibrate(
      buildResponseMatrix(reference.responses, { itemIds: reference.itemIds }),
      reference.model === "1pl" ? oneParameterLogistic : twoParameterLogistic,
      {
        fixed: definedEntries(reference.fixed),
        covariates: covariatesOf(reference),
        tolerance: 1e-8,
        maxIterations: 5000,
        standardErrors,
      },
    );

  for (const reference of latentRegression.cases.filter(({ model }) => model === "2pl")) {
    it(`matches mirt's items, coefficients and residual variance of ${reference.name}`, () => {
      const result = calibrateReference(reference);
      expect(result.converged).toBe(true);
      reference.mirt.items.forEach((expected, i) => {
        const actual = defined(result.items[i]).parameters as LogisticItem;
        expectClose(actual.a, expected.a, REGRESSION_TOLERANCE, `a of item ${i + 1}`);
        expectClose(actual.b, expected.b, REGRESSION_TOLERANCE, `b of item ${i + 1}`);
      });
      const regression = defined(result.regression);
      for (const [name, expected] of Object.entries(reference.mirt.coefficients)) {
        expectClose(defined(regression.coefficients[name]), expected, REGRESSION_TOLERANCE, `coefficient of ${name}`);
      }
      const fixedItems = Object.keys(reference.fixed).length;
      expectClose(
        regression.residualSd ** 2,
        fixedItems === 0 ? 1 : reference.mirt.residualVariance,
        REGRESSION_TOLERANCE,
      );
      expect(result.groups).toEqual([]);
    });
  }

  it("matches mirt's slopes, thresholds and coefficient of the generalized partial credit model", () => {
    const reference = latentRegression.polytomous;
    const result = calibrate(
      buildResponseMatrix(reference.responses, { itemIds: reference.itemIds }),
      generalizedPartialCredit,
      {
        covariates: reference.covariates,
        tolerance: 1e-8,
        maxIterations: 5000,
      },
    );
    expect(result.converged).toBe(true);
    reference.mirt.items.forEach((expected, i) => {
      const actual = defined(result.items[i]).parameters as ThresholdItem;
      expectClose(actual.a, expected.a, REGRESSION_TOLERANCE, `a of item ${i + 1}`);
      expected.b.forEach((threshold, k) => {
        expectClose(actual.b[k] ?? null, threshold, REGRESSION_TOLERANCE, `threshold ${k + 1} of item ${i + 1}`);
      });
    });
    const regression = defined(result.regression);
    expectClose(defined(regression.coefficients.x), defined(reference.mirt.coefficients.x), REGRESSION_TOLERANCE);
    expect(regression.residualSd).toBe(1);
  });

  it("matches mirt's and TAM's Rasch model after moving the 1PL, whose slope a is estimated, to the Rasch scale", () => {
    const reference = defined(latentRegression.cases.find(({ model }) => model === "1pl"));
    const result = calibrateReference(reference);
    expect(result.converged).toBe(true);
    const a = (defined(result.items[0]).parameters as LogisticItem).a;
    const regression = defined(result.regression);
    expect(regression.residualSd).toBe(1);
    const tam = defined(
      (reference as { tam?: { difficulties: number[]; coefficient: number; residualVariance: number } }).tam,
    );
    expectClose(a * defined(regression.coefficients.x), defined(reference.mirt.coefficients.x), REGRESSION_TOLERANCE);
    expectClose(a * defined(regression.coefficients.x), tam.coefficient, REGRESSION_TOLERANCE);
    expectClose(a * a, reference.mirt.residualVariance, REGRESSION_TOLERANCE);
    expectClose(a * a, tam.residualVariance, REGRESSION_TOLERANCE);
    reference.mirt.items.forEach((expected, i) => {
      expectClose(a * (defined(result.items[i]).parameters as LogisticItem).b, expected.b, REGRESSION_TOLERANCE);
    });
  });

  it("matches the Hessian of the marginal log-likelihood in base R for the standard error of a coefficient", () => {
    const reference = defined(latentRegression.cases.find(({ name }) => name === "2PL, a binary covariate"));
    const result = calibrateReference(reference, "hessian");
    const expected = defined((reference as { hessianStandardErrors?: { group: number } }).hessianStandardErrors).group;
    const actual = defined(defined(defined(result.regression).standardErrors).group);
    expectClose(actual, expected, REGRESSION_STANDARD_ERROR_TOLERANCE * expected);
  });

  it("recovers the coefficient, in units of the residual SD with free slopes and in theta's with fixed items", () => {
    const random = createRandom(11);
    const items = simulateItems(twoParameterLogistic, 25, random);
    const x = simulateThetas(1500, random);
    const noise = simulateThetas(1500, random, { distribution: normalDistribution(0, 0.8) });
    const responses = simulateResponses(
      items,
      x.map((value, person) => 0.6 * value + (noise[person] ?? 0)),
      random,
    );
    const free = defined(calibrate(responses, twoParameterLogistic, { covariates: { x } }).regression);
    expect(free.residualSd).toBe(1);
    expectClose(defined(free.coefficients.x), 0.6 / 0.8, 0.07);
    const fixed = Object.fromEntries(items.slice(0, 8).map((item, j) => [`item${j + 1}`, item.parameters]));
    const anchored = defined(calibrate(responses, twoParameterLogistic, { covariates: { x }, fixed }).regression);
    expectClose(defined(anchored.coefficients.x), 0.6, 0.06);
    expectClose(anchored.residualSd, 0.8, 0.06);
  });

  it("reports standard errors only when asked, and none of a distribution of theta", () => {
    const reference = defined(latentRegression.cases[0]);
    expect(calibrateReference(reference).regression?.standardErrors).toBeNull();
    const result = calibrateReference(reference, "hessian");
    expect(Object.keys(defined(result.regression?.standardErrors))).toEqual(Object.keys(reference.covariates));
    const matrix = buildResponseMatrix(reference.responses, { itemIds: reference.itemIds });
    expect(calibrate(matrix, twoParameterLogistic).regression).toBeUndefined();
  });

  it("counts the coefficients, and the residual SD when estimated, among the parameters", () => {
    const [free, anchored] = [defined(latentRegression.cases[0]), defined(latentRegression.cases.at(-1))];
    const withoutCovariates = calibrate(
      buildResponseMatrix(free.responses, { itemIds: free.itemIds }),
      twoParameterLogistic,
    );
    const withCovariates = calibrateReference(free);
    const parameters = (aic: number, logLikelihood: number) => (aic + 2 * logLikelihood) / 2;
    const base = parameters(withoutCovariates.aic, withoutCovariates.logLikelihood);
    expect(parameters(withCovariates.aic, withCovariates.logLikelihood)).toBeCloseTo(base + 2, 9);
    const fixedCount = Object.keys(anchored.fixed).length;
    const result = calibrateReference(anchored);
    expect(parameters(result.aic, result.logLikelihood)).toBeCloseTo(base - 2 * fixedCount + 3, 9);
  });

  it("gives the same estimates in any order of the persons, and for persons without responses whatever their covariates", () => {
    const reference = defined(latentRegression.cases[0]);
    const order = reference.responses.map((_, person) => (person * 7 + 3) % reference.responses.length);
    expect(new Set(order).size).toBe(order.length);
    const options = { tolerance: 1e-9, maxIterations: 5000 } as const;
    const original = calibrate(
      buildResponseMatrix(reference.responses, { itemIds: reference.itemIds }),
      twoParameterLogistic,
      {
        ...options,
        covariates: covariatesOf(reference),
      },
    );
    const permuted = calibrate(
      buildResponseMatrix(
        order.map((person) => defined(reference.responses[person])),
        { itemIds: reference.itemIds },
      ),
      twoParameterLogistic,
      {
        ...options,
        covariates: Object.fromEntries(
          Object.entries(covariatesOf(reference)).map(([name, values]) => [
            name,
            order.map((person) => defined(values[person])),
          ]),
        ),
      },
    );
    expectClose(defined(permuted.regression?.coefficients.x), defined(original.regression?.coefficients.x), 1e-7);
    const silent = [...reference.responses, reference.itemIds.map(() => null)];
    const withSilent = calibrate(buildResponseMatrix(silent, { itemIds: reference.itemIds }), twoParameterLogistic, {
      ...options,
      covariates: Object.fromEntries(
        Object.entries(covariatesOf(reference)).map(([name, values]) => [name, [...values, 1e6]]),
      ),
    });
    expectClose(defined(withSilent.regression?.coefficients.x), defined(original.regression?.coefficients.x), 1e-7);
  });

  it("rejects covariates that conflict with other options or do not give a number to every person", () => {
    const reference = defined(latentRegression.cases[0]);
    const matrix = buildResponseMatrix(reference.responses, { itemIds: reference.itemIds });
    const covariates = covariatesOf(reference);
    const x = defined(covariates.x);
    const persons = reference.responses.length;
    const options = { covariates } as const;
    expect(() =>
      calibrate(matrix, twoParameterLogistic, { ...options, groups: reference.responses.map(() => "a") }),
    ).toThrow("Covariates need one group: a group is a covariate in a latent regression.");
    expect(() => calibrate(matrix, twoParameterLogistic, { ...options, latent: "normal" })).toThrow(
      "Covariates replace the distribution of theta: the option latent must be absent or standard-normal.",
    );
    expect(() =>
      calibrate(matrix, twoParameterLogistic, { ...options, quadrature: { method: "gauss-hermite" } }),
    ).toThrow("A latent regression needs the rectangular quadrature rule.");
    expect(() => calibrate(matrix, twoParameterLogistic, { ...options, standardErrors: "cross-product" })).toThrow(
      'Covariates need the standard errors "none" or "hessian".',
    );
    expect(() =>
      calibrate(matrix, twoParameterLogistic, { ...options, testlets: { t1: [defined(reference.itemIds[0])] } }),
    ).toThrow("Covariates cannot be used with testlets.");
    expect(() => calibrate(matrix, twoParameterLogistic, { ...options, covariates: {} })).toThrow(
      "The option covariates needs at least one covariate.",
    );
    expect(() => calibrate(matrix, twoParameterLogistic, { ...options, covariates: { x: x.slice(1) } })).toThrow(
      `Covariate "x" has ${persons - 1} values for ${persons} persons.`,
    );
    expect(() =>
      calibrate(matrix, twoParameterLogistic, {
        ...options,
        covariates: { x: x.map((v, i) => (i === 4 ? Number.NaN : v)) },
      }),
    ).toThrow('Covariate "x" of person 5 must be a finite number, found NaN.');
    expect(() =>
      calibrate(matrix, twoParameterLogistic, { ...options, covariates: { x: x.map(() => 1), y: x.map(() => 1) } }),
    ).toThrow("The covariates of the persons who answered are collinear: leave one of them out.");
  });

  it("rejects covariates of other types, as JavaScript or JSON pass them", () => {
    const reference = defined(latentRegression.cases[0]);
    const matrix = buildResponseMatrix(reference.responses, { itemIds: reference.itemIds });
    expect(() => calibrate(matrix, twoParameterLogistic, { covariates: JSON.parse("null") })).toThrow(
      "The option covariates must be an object of values by covariate name, found null.",
    );
    expect(() => calibrate(matrix, twoParameterLogistic, { covariates: JSON.parse('{ "x": "high" }') })).toThrow(
      'Covariate "x" must be an array with a value for each person, found "high".',
    );
    expect(() => calibrate(matrix, twoParameterLogistic, { covariates: JSON.parse('{ "x": { "0": 1 } }') })).toThrow(
      'Covariate "x" must be an array with a value for each person, found an object.',
    );
    const column = reference.responses.map((_, person) => (person === 2 ? "2" : person % 5));
    expect(() =>
      calibrate(matrix, twoParameterLogistic, { covariates: JSON.parse(JSON.stringify({ x: column })) }),
    ).toThrow('Covariate "x" of person 3 must be a finite number, found "2".');
  });
});

describe("the M-step of the linear Rasch model", () => {
  const target = Float64Array.of(0.25, -0.5);

  /** A quadratic of curvature 1 around `target` on top of `offset`, and the number of times it was evaluated. */
  function quadratic(offset: number, rounded?: Float64Array) {
    let evaluations = 0;
    const objective = (x: Float64Array, gradient: Float64Array, hessian: Float64Array): number => {
      evaluations++;
      hessian.fill(0);
      let value = offset;
      for (let j = 0; j < x.length; j++) {
        const distance = (x[j] ?? 0) - (target[j] ?? 0);
        value += 0.5 * distance * distance;
        gradient[j] = distance;
        hessian[j * x.length + j] = 1;
      }
      // The value at the point where the descent starts rounds lower than at every point near it, as the sum of
      // thousands of terms does at some point: no step looks like a decrease, though the gradient still points at the minimum.
      const lucky = rounded !== undefined && x.every((entry, j) => entry === rounded[j]);
      return lucky ? value - 1e-6 : value;
    };
    return { objective, evaluations: () => evaluations };
  }

  it("reaches the minimum of a smooth objective in a few evaluations", () => {
    const { objective, evaluations } = quadratic(0);
    const x = maximize(objective, Float64Array.of(1, 2), null);
    expectClose(defined(x[0]), 0.25, 1e-12);
    expectClose(defined(x[1]), -0.5, 1e-12);
    expect(evaluations()).toBeLessThanOrEqual(4);
  });

  it("keeps a held parameter at its value", () => {
    const { objective } = quadratic(0);
    const x = maximize(objective, Float64Array.of(1, 2), { index: 1, value: 3 });
    expectClose(defined(x[0]), 0.25, 1e-12);
    expect(x[1]).toBe(3);
  });

  it("stops at a step whose decrease the rounding of the objective hides, which would take all its iterations", () => {
    // A step of 1e-5 is above the step tolerance, and 40 halvings bring it below the spacing of the numbers near 0.25.
    const start = Float64Array.of(0.25 + 1e-5, -0.5 - 1e-5);
    const { objective, evaluations } = quadratic(1e9, start);
    const x = maximize(objective, start, null);
    // One line search of 40 halvings, and not 50 of them.
    expect(evaluations()).toBeLessThan(50);
    expect(Math.abs((x[0] ?? 0) - 0.25)).toBeLessThanOrEqual(1.1e-5);
    expect(Math.abs((x[1] ?? 0) + 0.5)).toBeLessThanOrEqual(1.1e-5);
  });
});

describe("logistic models", () => {
  it("name their estimated parameters, with c and d bounded and a shared by 1PL items", () => {
    expect(oneParameterLogistic.parameters(2)).toEqual([{ name: "a", shared: true }, { name: "b" }]);
    expect(twoParameterLogistic.parameters(2).map(({ name }) => name)).toEqual(["a", "b"]);
    expect(fourParameterLogistic.parameters(2)).toEqual([
      { name: "a" },
      { name: "b" },
      { name: "c", lower: 0, upper: 1 },
      { name: "d", lower: 0, upper: 1 },
    ]);
  });

  it("convert parameters to vectors and back", () => {
    const item = { a: 1.4, b: -0.3, c: 0.2, d: 1 };
    expect([...threeParameterLogistic.toVector(item)]).toEqual([1.4, -0.3, 0.2]);
    expect(threeParameterLogistic.fromVector(threeParameterLogistic.toVector(item))).toEqual(item);
    expect(twoParameterLogistic.fromVector(Float64Array.of(1, 2))).toEqual({ a: 1, b: 2, c: 0, d: 1 });
  });

  it("start inside the bounds, with b where the share correct is reached", () => {
    for (const p of [0.05, 0.5, 0.97]) {
      const start = fourParameterLogistic.startingValues(Float64Array.of(1 - p, p));
      expect(start.c).toBeGreaterThan(0);
      expect(start.d).toBeLessThan(1);
      const share = new Float64Array(2);
      fourParameterLogistic.probabilities(start, start.b, share);
      expect(defined(share[1])).toBeCloseTo((start.c + start.d) / 2, 12);
    }
  });

  it("reject parameters outside the model", () => {
    expect(() => twoParameterLogistic.toVector({ a: 1, b: 0, c: 0, d: 0.9 })).toThrow(
      "A 2PL item has d = 1, found d = 0.9.",
    );
    expect(() => threeParameterLogistic.toVector({ a: 1, b: 0, c: 0.3, d: 0.9 })).toThrow(
      "A 3PL item has d = 1, found d = 0.9.",
    );
    expect(() => fourParameterLogistic.toVector({ a: 1, b: 0, c: 0.5, d: 0.5 })).toThrow(DataError);
    expect(() => fourParameterLogistic.toVector({ a: Number.POSITIVE_INFINITY, b: 0, c: 0, d: 1 })).toThrow(DataError);
  });
});

describe("nominalResponse", () => {
  it("names the slopes and intercepts of all categories but the last", () => {
    expect(nominalResponse.parameters(3).map(({ name }) => name)).toEqual(["a1", "a2", "c1", "c2"]);
  });

  it("centers the slopes and intercepts, which leaves the probabilities unchanged", () => {
    const item = { a: [1, 2, 3], c: [0, 1, 5] };
    const centered = nominalResponse.fromVector(nominalResponse.toVector(item));
    expect(centered).toEqual({ a: [-1, 0, 1], c: [-2, -1, 3] });
    const before = new Float64Array(3);
    const after = new Float64Array(3);
    nominalResponse.probabilities(item, 0.7, before);
    nominalResponse.probabilities(centered, 0.7, after);
    before.forEach((p, k) => {
      expectClose(defined(after[k]), p, 1e-15);
    });
  });

  it("computes the information of almost equal slopes to rounding, whatever their size", () => {
    // Two categories give (a2 - a1)² p (1 - p) with p the logistic function of (a2 - a1) θ. The slopes differ by 6e-5,
    // so their variance is 1e-9 and a rounding error of their mean, 4e-16, would show in its 7th digit.
    const [low, high] = [-2.9999997615268814, -2.9999389350413916];
    const item = { a: [low, high], c: [0, 0] };
    const gap = high - low;
    for (let theta = -6; theta <= 6; theta += 0.05) {
      const p = 1 / (1 + Math.exp(-gap * theta));
      expectClose(nominalResponse.information(item, theta) / (gap * gap * p * (1 - p)), 1, 1e-12, `theta ${theta}`);
    }
  });

  it("rejects slopes and intercepts of different lengths", () => {
    expect(() => nominalResponse.toVector({ a: [0, 1], c: [0, 1, 2] })).toThrow(
      "An NRM item needs finite slopes a and intercepts c, one of each per category and at least two categories, found a = [0, 1], c = [0, 1, 2].",
    );
    expect(() => nominalResponse.toVector({ a: [1], c: [0] })).toThrow(DataError);
  });
});

describe("partial credit models", () => {
  it("name a, shared by PCM items, and the thresholds", () => {
    expect(partialCredit.parameters(3)).toEqual([{ name: "a", shared: true }, { name: "b1" }, { name: "b2" }]);
    expect(generalizedPartialCredit.parameters(3)).toEqual([{ name: "a" }, { name: "b1" }, { name: "b2" }]);
  });

  it("start at thresholds that give a person at θ = 0 the shares of the categories", () => {
    const shares = Float64Array.of(0.4, 0.1, 0.5);
    const probabilities = new Float64Array(3);
    generalizedPartialCredit.probabilities(generalizedPartialCredit.startingValues(shares), 0, probabilities);
    probabilities.forEach((p, k) => {
      expectClose(p, defined(shares[k]), 1e-15);
    });
  });

  it("convert items with more thresholds than a spread of arguments holds", () => {
    const b = Array.from({ length: 200_000 }, () => 0);
    expect(generalizedPartialCredit.toVector({ a: 1, b })).toHaveLength(200_001);
  });

  it("reject items without thresholds", () => {
    expect(() => partialCredit.toVector({ a: 1, b: [] })).toThrow(
      "A PCM item needs a finite a and at least one finite threshold b, found a = 1, b = [].",
    );
    expect(() => generalizedPartialCredit.toVector({ a: 1, b: [Number.NaN] })).toThrow(DataError);
  });

  it("reject a hole among the parameters of any model, which every would skip and leave a NaN", () => {
    const hole = (values: number[]): number[] => {
      const sparse = [...values];
      delete sparse[1];
      return sparse;
    };
    expect(() => generalizedPartialCredit.toVector({ a: 1, b: hole([-1, 0, 1]) })).toThrow(DataError);
    expect(() => gradedResponse.toVector({ a: 1, b: hole([-1, 0, 1]) })).toThrow(DataError);
    expect(() => ratingScale.toVector({ a: 1, b: 0, steps: hole([-1, 0, 1]) })).toThrow(DataError);
    expect(() => nominalResponse.toVector({ a: hole([-1, 0, 1]), c: [0, 0, 0] })).toThrow(DataError);
    expect(() => nominalResponse.toVector({ a: [-1, 0, 1], c: hole([0, 0, 0]) })).toThrow(DataError);
    expect(() => generalizedGradedUnfolding.toVector({ a: 1, b: 0, t: hole([-1, -0.5, -0.2]) })).toThrow(DataError);
  });
});

describe("priors", () => {
  it("have derivatives consistent with their log densities", () => {
    const cases: [ParameterPrior, number][] = [
      [normalPrior(0.5, 2), -1.3],
      [logNormalPrior(0, 0.5), 1.7],
      [betaPrior(5, 17), 0.23],
    ];
    const h = 1e-5;
    for (const [density, x] of cases) {
      const slope = (density.logDensity(x + h) - density.logDensity(x - h)) / (2 * h);
      expectClose(density.logDensityDerivative(x), slope, 1e-7);
      const curvature = (density.logDensityDerivative(x + h) - density.logDensityDerivative(x - h)) / (2 * h);
      expectClose(density.logDensitySecondDerivative(x), curvature, 1e-6);
    }
  });

  it("are -Infinity outside their support", () => {
    expect(logNormalPrior(0, 1).logDensity(0)).toBe(Number.NEGATIVE_INFINITY);
    expect(betaPrior(2, 2).logDensity(1)).toBe(Number.NEGATIVE_INFINITY);
  });

  it("reject invalid arguments", () => {
    expect(() => normalPrior(0, -1)).toThrow("A normal prior's sd must be a positive number, found -1.");
    expect(() => normalPrior(Number.NaN, 1)).toThrow(DataError);
    expect(() => logNormalPrior(0, 0)).toThrow(DataError);
    expect(() => betaPrior(0, 1)).toThrow(DataError);
    expect(() => normalPrior(JSON.parse('"0"'), 1)).toThrow(
      'A normal prior\'s mean must be a finite number, found "0".',
    );
    expect(() => logNormalPrior(0, JSON.parse("null"))).toThrow(
      "A log-normal prior's sdLog must be a positive number, found null.",
    );
  });
});

describe("ratingScale", () => {
  it("names a and the steps, shared by its items, and the location b", () => {
    expect(ratingScale.parameters(4)).toEqual([
      { name: "a", shared: true },
      { name: "b" },
      { name: "step1", shared: true },
      { name: "step2", shared: true },
    ]);
    expect(ratingScale.parameters(2).map(({ name }) => name)).toEqual(["a", "b"]);
  });

  it("moves the mean of the steps into the location, which leaves the probabilities unchanged", () => {
    const item = { a: 1.2, b: 0.5, steps: [-1, 0, 2] };
    const centered = ratingScale.fromVector(ratingScale.toVector(item));
    expectClose(centered.b, 0.5 + 1 / 3, 1e-15);
    expect(centered.steps.reduce((sum, step) => sum + step, 0)).toBeCloseTo(0, 15);
    const before = new Float64Array(4);
    const after = new Float64Array(4);
    ratingScale.probabilities(item, -0.4, before);
    ratingScale.probabilities(centered, -0.4, after);
    before.forEach((p, k) => {
      expectClose(defined(after[k]), p, 1e-15);
    });
    expect(ratingScale.fromVector(Float64Array.of(1, 0.3))).toEqual({ a: 1, b: 0.3, steps: [0] });
  });

  it("rejects an item without steps or with steps that are not finite", () => {
    expect(() => ratingScale.toVector({ a: 1, b: 0, steps: [] })).toThrow(
      "An RSM item needs finite a and b and at least one step, found a = 1, b = 0, steps = [].",
    );
    expect(() => ratingScale.toVector({ a: 1, b: 0, steps: [0, Number.POSITIVE_INFINITY] })).toThrow(
      "An RSM item needs finite steps, found [0, Infinity].",
    );
  });
});

const arbitraryTheta = fc.double({ min: -8, max: 8, noNaN: true });

/** Returns the response matrix of persons whose responses are written as one string of digits each. */
function unfoldingMatrix(rows: readonly string[]): ReturnType<typeof buildResponseMatrix> {
  return buildResponseMatrix(rows.map((row) => Array.from(row, Number)));
}

/** Returns 1 when calibrated locations have the signs of the true ones, -1 when the calibration reflects them. */
function orientation(
  items: readonly { readonly parameters: UnfoldingParameters }[],
  truth: readonly { readonly b: number }[],
): number {
  return items.reduce((sum, item, i) => sum + item.parameters.b * defined(truth[i]).b, 0) < 0 ? -1 : 1;
}

/** Returns an item's category probabilities, their theta derivatives and its information at theta. */
function evaluate({ model: itemModel, parameters }: IrtItem, theta: number): number[][] {
  const count = itemModel.categoryCount(parameters);
  const probabilities = new Float64Array(count);
  const first = new Float64Array(count);
  const second = new Float64Array(count);
  itemModel.probabilities(parameters, theta, probabilities);
  itemModel.thetaDerivatives(parameters, theta, first, second);
  return [[...probabilities], [...first], [...second], [itemModel.information(parameters, theta)]];
}

describe("reliability", () => {
  for (const form of scoring.forms) {
    it(`matches mirt's marginal_rxx and base R for the ${modelNames(form)} items`, () => {
      const items = fixtureItems(form);
      for (const { prior, value } of form.marginalReliability) {
        expectClose(marginalReliability(items, { prior }), value, MARGINAL_RELIABILITY_TOLERANCE);
      }
    });
  }

  it("matches mirt's empirical_rxx of EAP estimates", () => {
    const { estimates, value } = scoring.empiricalReliability;
    expectClose(empiricalReliability(estimates), value, 1e-12);
  });

  it("gives 0 for estimates without variance and 1 for estimates without error", () => {
    expect(
      empiricalReliability([
        { theta: 0.5, standardError: 0.4 },
        { theta: 0.5, standardError: 0.3 },
      ]),
    ).toBe(0);
    expect(
      empiricalReliability([
        { theta: -1, standardError: 0 },
        { theta: 1, standardError: 0 },
      ]),
    ).toBe(1);
  });

  it("rejects too few or invalid estimates and forms without items", () => {
    expect(() => empiricalReliability([{ theta: 0, standardError: 1 }])).toThrow(
      "Empirical reliability needs at least two estimates.",
    );
    expect(() =>
      empiricalReliability([
        { theta: 0, standardError: Infinity },
        { theta: 1, standardError: 1 },
      ]),
    ).toThrow("Estimate 1 needs a finite theta and a finite, non-negative standard error, found 0 and Infinity.");
    expect(() =>
      empiricalReliability([
        { theta: 0, standardError: 0 },
        { theta: 0, standardError: 0 },
      ]),
    ).toThrow(DataError);
    expect(() => marginalReliability([])).toThrow("Marginal reliability needs at least one item.");
    expect(() => marginalReliability(ITEMS_2PL, { prior: { mean: 0, sd: -1 } })).toThrow(DataError);
    expect(() => marginalReliability(JSON.parse("null"))).toThrow(
      "Marginal reliability needs an array of items, found null.",
    );
    expect(() => marginalReliability(ITEMS_2PL, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => marginalReliability(ITEMS_2PL, JSON.parse('{"verbose":1}'))).toThrow('There is no option "verbose"');
    expect(() => empiricalReliability(JSON.parse("null"))).toThrow(
      "Empirical reliability needs an array of estimates, found null.",
    );
    expect(() => empiricalReliability(JSON.parse("[null, null]"))).toThrow(
      "Estimate 1 must be an object { theta, standardError }, found null.",
    );
  });
});

/** Returns the entries of each person repeated as often as the person's integer weight. */
function repeated<T>(values: readonly T[], weights: readonly number[]): T[] {
  return values.flatMap((value, person) => Array.from({ length: weights[person] ?? 0 }, () => value));
}

describe("person weights", () => {
  const { twoPl, graded, groups, rasch, regression } = weighted.cases;
  const matrixOf = (reference: {
    readonly itemIds: readonly string[];
    readonly responses: readonly (readonly (number | null)[])[];
  }) => buildResponseMatrix(reference.responses, { itemIds: reference.itemIds });
  const tight = { tolerance: 1e-8, maxIterations: 5000 } as const;

  it("matches mirt's survey weights for the 2PL: items, log-likelihood, BIC and both kinds of standard errors", () => {
    const methods = [
      ["hessian", twoPl.hessianErrors],
      ["cross-product", twoPl.crossProductErrors],
    ] as const;
    for (const [standardErrors, expectedErrors] of methods) {
      const result = calibrate(matrixOf(twoPl), twoParameterLogistic, {
        personWeights: twoPl.weights,
        standardErrors,
        ...tight,
      });
      expect(result.converged).toBe(true);
      twoPl.items.forEach((expected, i) => {
        const item = defined(result.items[i]);
        expectParameters(
          { a: (item.parameters as LogisticItem).a, b: (item.parameters as LogisticItem).b },
          expected,
          WEIGHTED_TOLERANCE,
          `item ${i + 1}`,
        );
        expectParameters(item.standardErrors, expectedErrors[i], WEIGHTED_TOLERANCE, `${standardErrors} ${i + 1}`);
      });
      expectClose(result.logLikelihood, twoPl.logLikelihood, WEIGHTED_TOLERANCE);
      expectClose(result.bic, twoPl.bic, WEIGHTED_TOLERANCE);
    }
  });

  it("matches mirt's survey weights for the graded response model", () => {
    const result = calibrate(matrixOf(graded), gradedResponse, { personWeights: graded.weights, ...tight });
    expect(result.converged).toBe(true);
    graded.items.forEach((expected, i) => {
      expectParameters(defined(result.items[i]).parameters, expected, WEIGHTED_TOLERANCE, `item ${i + 1}`);
    });
    expectClose(result.logLikelihood, graded.logLikelihood, WEIGHTED_TOLERANCE);
  });

  it("matches base R's maximum of the weighted likelihood in two groups, which mirt's survey weights stop short of", () => {
    const result = calibrate(matrixOf(groups), twoParameterLogistic, {
      personWeights: groups.weights,
      groups: groups.groups,
      latent: { focal: "normal" },
      ...tight,
    });
    expect(result.converged).toBe(true);
    groups.items.forEach((expected, i) => {
      const { a, b } = defined(result.items[i]).parameters as LogisticItem;
      expectParameters({ a, b }, expected, WEIGHTED_TOLERANCE, `item ${i + 1}`);
    });
    const focal = defined(result.groups.find(({ group }) => group === "focal"));
    expectClose(focal.mean, groups.focal.mean, WEIGHTED_TOLERANCE);
    expectClose(focal.sd, groups.focal.sd, WEIGHTED_TOLERANCE);
    expectClose(result.logLikelihood, groups.logLikelihood, WEIGHTED_TOLERANCE);
    expect(result.logLikelihood).toBeGreaterThan(groups.mirt.logLikelihood);
    expectClose(focal.sd, groups.mirt.focal.sd, WEIGHTED_TOLERANCE);
  });

  it("matches TAM's pweights for the Rasch model with a latent regression, on the Rasch scale", () => {
    const result = calibrate(matrixOf(rasch), oneParameterLogistic, {
      personWeights: rasch.weights,
      covariates: rasch.covariates,
      ...tight,
    });
    expect(result.converged).toBe(true);
    const a = (defined(result.items[0]).parameters as LogisticItem).a;
    expectClose(a * defined(result.regression?.coefficients.x), rasch.tam.coefficient, WEIGHTED_TOLERANCE);
    expectClose(a * a, rasch.tam.residualVariance, WEIGHTED_TOLERANCE);
    rasch.tam.difficulties.forEach((difficulty, i) => {
      expectClose(a * (defined(result.items[i]).parameters as LogisticItem).b, difficulty, WEIGHTED_TOLERANCE);
    });
  });

  it("weights the latent regression as the repeated persons do, where mirt's survey weights leave it out", () => {
    const result = calibrate(matrixOf(regression), twoParameterLogistic, {
      personWeights: regression.weights,
      covariates: regression.covariates,
      ...tight,
    });
    expect(result.converged).toBe(true);
    regression.items.forEach((expected, i) => {
      const { a, b } = defined(result.items[i]).parameters as LogisticItem;
      expectParameters({ a, b }, expected, WEIGHTED_TOLERANCE, `item ${i + 1}`);
    });
    expectClose(defined(result.regression?.coefficients.x), regression.coefficient, WEIGHTED_TOLERANCE);
    expectClose(result.logLikelihood, regression.logLikelihood, WEIGHTED_TOLERANCE);
    // Base R's weighted log-likelihood at mirt's estimates with survey.weights is lower, so they are not its maximum.
    expect(result.logLikelihood).toBeGreaterThan(regression.mirtSurveyWeights.logLikelihood + 1);
  });

  describe("integer weights give the calibration of the persons repeated by them", () => {
    const rows = simulate(31, 240, ITEMS_2PL).map((row, person) => (person === 5 ? row.map(() => null) : row));
    const weights = rows.map((_, person) => 1 + (person % 3));
    const labels = rows.map((_, person) => (person % 2 === 0 ? "reference" : "focal"));
    const x = rows.map((_, person) => ((person % 7) - 3) / 2);
    const anchor = { item1: defined(PARAMETERS_2PL[0]) };
    const testletCase = defined(testlet.cases[0]);
    const testletRows = testletCase.responses.slice(0, 300);
    const cases: readonly {
      readonly name: string;
      readonly rows: readonly (readonly (number | null)[])[];
      readonly itemIds?: readonly string[];
      readonly model: IrtModel<unknown>;
      readonly options?: CalibrationOptions<unknown>;
      readonly byPerson?: (weights: readonly number[]) => Partial<CalibrationOptions<unknown>>;
    }[] = [
      {
        name: "the 2PL with errors from the Hessian",
        rows,
        model: twoParameterLogistic,
        options: { standardErrors: "hessian" },
      },
      {
        name: "the 2PL with cross-product errors",
        rows,
        model: twoParameterLogistic,
        options: { standardErrors: "cross-product" },
      },
      { name: "the GPCM", rows: simulate(32, 240, ITEMS_GPCM), model: generalizedPartialCredit },
      {
        name: "two groups, the focal group's normal estimated",
        rows,
        model: twoParameterLogistic,
        options: { latent: { focal: "normal" }, standardErrors: "hessian" },
        byPerson: (each) => ({ groups: repeated(labels, each) }),
      },
      {
        name: "a histogram of theta on a fixed item's scale",
        rows,
        model: twoParameterLogistic,
        options: { fixed: anchor, latent: "empirical-histogram", maxIterations: 5000 },
      },
      {
        name: "a latent regression on a fixed item's scale",
        rows,
        model: twoParameterLogistic,
        options: { fixed: anchor, standardErrors: "hessian" },
        byPerson: (each) => ({ covariates: { x: repeated(x, each) } }),
      },
      {
        name: "testlets",
        rows: testletRows,
        itemIds: testletCase.itemIds,
        model: twoParameterLogistic,
        options: {
          testlets: definedEntries(testletCase.testlets),
          standardErrors: "hessian",
        },
      },
    ];

    for (const { name, rows: data, itemIds, model, options, byPerson } of cases) {
      it(`with ${name}`, () => {
        const each = data.map((_, person) => 1 + (person % 3));
        const ones = data.map(() => 1);
        const ids = itemIds === undefined ? {} : { itemIds };
        const fit = calibrate(buildResponseMatrix(data, ids), model, {
          ...options,
          ...byPerson?.(ones),
          personWeights: each,
        });
        const expanded = calibrate(buildResponseMatrix(repeated(data, each), ids), model, {
          ...options,
          ...byPerson?.(each),
        });
        const answered = data.map((row) => row.some((response) => response !== null));
        const respondents = answered.filter(Boolean).length;
        const ratio = each.reduce((sum, weight, person) => sum + (answered[person] ? weight : 0), 0) / respondents;
        expect(fit.converged).toBe(true);
        expect(expanded.converged).toBe(true);
        expect(fit.iterations).toBe(expanded.iterations);
        fit.items.forEach((item, i) => {
          const twin = defined(expanded.items[i]);
          expectParameters(item.parameters, twin.parameters, EXPANSION_TOLERANCE, `item ${i + 1}`);
          if (twin.standardErrors === null) {
            expect(item.standardErrors).toBeNull();
            return;
          }
          const scaled = Object.fromEntries(
            Object.entries(defined(item.standardErrors)).map(([key, error]) => [key, error / Math.sqrt(ratio)]),
          );
          expectParameters(
            scaled,
            twin.standardErrors,
            EXPANSION_STANDARD_ERROR_TOLERANCE,
            `errors ${i + 1}`,
            Math.abs,
          );
        });
        fit.groups.forEach((group, g) => {
          const twin = defined(expanded.groups[g]);
          expectParameters(
            { mean: group.mean, sd: group.sd, weights: group.weights },
            { mean: twin.mean, sd: twin.sd, weights: twin.weights },
            EXPANSION_TOLERANCE,
            `group ${g + 1}`,
          );
        });
        expectParameters(fit.regression?.coefficients, expanded.regression?.coefficients, EXPANSION_TOLERANCE, "β");
        fit.testlets.forEach((estimate, t) => {
          expectClose(estimate.sd, defined(expanded.testlets[t]).sd, EXPANSION_TOLERANCE, `testlet ${t + 1}`);
        });
        expectClose(fit.logLikelihood * ratio, expanded.logLikelihood, EXPANSION_TOLERANCE * -expanded.logLikelihood);
      });
    }

    it("gives the unweighted calibration, to the last digit, when every weight is equal", () => {
      const matrix = buildResponseMatrix(rows);
      const options = { standardErrors: "cross-product" } as const;
      expect(calibrate(matrix, twoParameterLogistic, { ...options, personWeights: rows.map(() => 2.5) })).toEqual(
        calibrate(matrix, twoParameterLogistic, options),
      );
    });

    it("ignores the weight of a person without responses and rescales weights of any size", () => {
      const matrix = buildResponseMatrix(rows);
      const options = { ...tight } as const;
      const base = calibrate(matrix, twoParameterLogistic, { ...options, personWeights: weights });
      const heavy = weights.map((weight, person) => (person === 5 ? 1e300 : weight));
      expect(calibrate(matrix, twoParameterLogistic, { ...options, personWeights: heavy })).toEqual(base);
      for (const scale of [1e306, 1e-310]) {
        const result = calibrate(matrix, twoParameterLogistic, {
          ...options,
          personWeights: weights.map((weight) => weight * scale),
        });
        expect(nonFinite(result)).toEqual([]);
        result.items.forEach((item, i) => {
          expectParameters(item.parameters, defined(base.items[i]).parameters, EXPANSION_TOLERANCE, `item ${i + 1}`);
        });
      }
    });
  });

  it("aligns the weights with the persons of records as with the rows of a matrix", () => {
    const matrix = matrixOf(twoPl);
    const records = buildResponseRecords(
      matrix.responses
        .flatMap((row, person) =>
          row.flatMap((response, item) =>
            response === null
              ? []
              : [{ person: defined(matrix.personIds[person]), item: defined(matrix.itemIds[item]), response }],
          ),
        )
        .reverse(),
      { personIds: matrix.personIds, itemIds: matrix.itemIds },
    );
    const options = { personWeights: twoPl.weights } as const;
    expect(calibrate(records, twoParameterLogistic, options)).toEqual(calibrate(matrix, twoParameterLogistic, options));
  });

  it("rejects weights that are not a positive number for each person, from JavaScript or JSON", () => {
    const matrix = buildResponseMatrix(simulate(33, 20, ITEMS_2PL));
    const fit = (personWeights: unknown) =>
      calibrate(matrix, twoParameterLogistic, { ...JSON.parse(JSON.stringify({ personWeights })) });
    const at = (value: unknown) => Array.from({ length: 20 }, (_, person) => (person === 2 ? value : 1));
    const third = (value: unknown) =>
      new DataError(`The weight of person 3 must be a positive number, found ${value}.`);
    expect(() => fit([1, 2])).toThrow(new DataError("Found 2 person weights for 20 persons."));
    expect(() => fit(at(0))).toThrow(third(0));
    expect(() => fit(at(-1))).toThrow(third(-1));
    expect(() => fit(at("2"))).toThrow(third('"2"'));
    expect(() => fit(at(null))).toThrow(third(null));
    for (const value of [Number.NaN, Number.POSITIVE_INFINITY]) {
      const personWeights = Array.from({ length: 20 }, (_, person) => (person === 2 ? value : 1));
      expect(() => calibrate(matrix, twoParameterLogistic, { personWeights })).toThrow(third(value));
    }
    const notAnArray = "The option personWeights must be an array with a weight for each person, found";
    expect(() => fit(null)).toThrow(new DataError(`${notAnArray} null.`));
    expect(() => fit({ 0: 1 })).toThrow(new DataError(`${notAnArray} an object.`));
  });
});

/** An iteration of x → x + (target - x) (1 - rate) on one number, whose log-likelihood rises toward the target. */
function linear(target: number, rate: number, start: number): Iteration & { readonly state: () => number } {
  let x = start;
  return {
    step() {
      const next = x + (target - x) * (1 - rate);
      const change = Math.abs(next - x);
      const logLikelihood = -((target - x) ** 2);
      x = next;
      return { logLikelihood, change };
    },
    get: () => Float64Array.of(x),
    set: (state) => {
      x = state[0] ?? 0;
    },
    state: () => x,
  };
}

describe("squarem", () => {
  it("reaches the limit of a linearly convergent iteration in a few steps", () => {
    const iteration = linear(3, 0.9, 0);
    const { converged, iterations } = squarem(iteration, { tolerance: 1e-6, maxIterations: 500 });
    expect(converged).toBe(true);
    // The plain iteration needs about 130 steps for a change of 1e-6.
    expect(iterations).toBeLessThanOrEqual(10);
    expect(iteration.state()).toBeCloseTo(3, 4);
  });

  it("counts every EM iteration against maxIterations", () => {
    for (const maxIterations of [1, 2, 3, 4, 5, 6, 7]) {
      const { converged, iterations } = squarem(linear(3, 0.99, 0), { tolerance: 0, maxIterations });
      expect({ converged, iterations }).toEqual({ converged: false, iterations: maxIterations });
    }
  });

  it("falls back to the plain step when the jump lowers the log-likelihood", () => {
    // The log-likelihood falls at the point of an extrapolation beyond the target, so that the cycle goes to x2.
    let jumps = 0;
    let x = 0;
    const iteration: Iteration = {
      step() {
        const next = x + (3 - x) * 0.1;
        const logLikelihood = x > 3 ? Number.NEGATIVE_INFINITY : -((3 - x) ** 2);
        const change = Math.abs(next - x);
        x = next;
        return { logLikelihood, change };
      },
      get: () => Float64Array.of(x),
      set: (state) => {
        jumps++;
        x = state[0] ?? 0;
      },
    };
    const { converged } = squarem(iteration, { tolerance: 1e-6, maxIterations: 500 });
    expect(converged).toBe(true);
    expect(jumps).toBeGreaterThan(0);
    expect(x).toBeCloseTo(3, 4);
  });

  it("stops when an iteration changes nothing", () => {
    const { converged, iterations } = squarem(linear(0, 0.5, 0), { tolerance: 1e-6, maxIterations: 50 });
    expect({ converged, iterations }).toEqual({ converged: true, iterations: 1 });
  });
});

describe("summedScoreTable", () => {
  for (const form of scoring.forms) {
    it(`matches mirt's EAPsum and base R for the ${modelNames(form)} items`, () => {
      const items = fixtureItems(form);
      for (const { prior, estimates } of form.tables) {
        const table = summedScoreTable(items, { prior });
        expect(table.map((row) => row.score)).toEqual(estimates.map((row) => row.score));
        table.forEach((row, s) => {
          const expected = defined(estimates[s]);
          expectClose(row.theta, expected.theta, TABLE_TOLERANCE, `theta of ${s}`);
          expectClose(row.standardError, expected.standardError, TABLE_TOLERANCE, `SE of ${s}`);
          expectClose(row.probability, expected.probability, TABLE_TOLERANCE, `probability of ${s}`);
        });
      }
    });
  }

  it("gives the EAP estimate of the single response of a one-item form", () => {
    const item = defined(ITEMS_2PL[0]);
    const table = summedScoreTable([item]);
    for (const row of table) {
      const estimate = posteriorMeanEstimate([item], [row.score]);
      expectClose(row.theta, estimate.theta, 1e-12);
      expectClose(row.standardError, estimate.standardError, 1e-12);
    }
  });

  for (const form of testletTables.forms) {
    it(`matches base R's enumeration of the patterns and mirt's EAPsum_2.0 with the testlets of ${form.name}`, () => {
      const items = form.items.map(({ id, model, parameters }) => ({ id, model: defined(MODELS[model]), parameters }));
      const reference = defined(testletTables.tables.find((table) => table.form === form.name));
      for (const { prior, estimates } of reference.tables) {
        const table = summedScoreTable(items, { prior, testlets: form.testlets });
        expect(table.map((row) => row.score)).toEqual(estimates.map((row) => row.score));
        table.forEach((row, s) => {
          const expected = defined(estimates[s]);
          expectClose(row.theta, expected.theta, TABLE_TOLERANCE, `theta of ${s}`);
          expectClose(row.standardError, expected.standardError, TABLE_TOLERANCE, `SE of ${s}`);
          expectClose(row.probability, expected.probability, TABLE_TOLERANCE, `probability of ${s}`);
        });
      }
    });
  }

  describe("with testlets", () => {
    const form = defined(testletTables.forms.find(({ name }) => name.startsWith("eight 2PL")));
    const items = form.items.map(({ id, parameters }) => ({ id, model: twoParameterLogistic, parameters }));

    it("gives the table of independent items as the testlets' SDs shrink to 0", () => {
      const independent = summedScoreTable(items);
      const testlets = form.testlets.map(({ items: ids }) => ({ items: ids, sd: 1e-9 }));
      summedScoreTable(items, { testlets }).forEach((row, s) => {
        const expected = defined(independent[s]);
        expectClose(row.theta, expected.theta, 1e-8, `theta of ${s}`);
        expectClose(row.probability, expected.probability, 1e-8, `probability of ${s}`);
      });
      expect(summedScoreTable(items, { testlets: [] })).toEqual(independent);
    });

    it("gives probabilities of the scores that sum to 1", () => {
      const table = summedScoreTable(items, { testlets: form.testlets, prior: { mean: -0.5, sd: 2 } });
      expectClose(
        table.reduce((sum, { probability }) => sum + probability, 0),
        1,
        1e-12,
      );
    });

    it("skips the ids of testlet items that are not on the form", () => {
      const testlets = [{ items: ["t1a", "t1b", "t1c", "elsewhere"], sd: 0.8 }, ...form.testlets.slice(1)];
      expect(summedScoreTable(items, { testlets })).toEqual(summedScoreTable(items, { testlets: form.testlets }));
      const none = [{ items: ["elsewhere"], sd: 2 }];
      expect(summedScoreTable(items, { testlets: none })).toEqual(summedScoreTable(items));
    });

    it("integrates each effect over the quadrature rule of the option", () => {
      const quadrature = { method: "gauss-hermite", points: 41 } as const;
      const coarse = summedScoreTable(items, { testlets: form.testlets, quadrature });
      summedScoreTable(items, { testlets: form.testlets }).forEach((row, s) => {
        expectClose(defined(coarse[s]).theta, row.theta, 1e-5, `theta of ${s}`);
      });
    });

    it("rejects testlets that are invalid or items without ids", () => {
      const sd = 1;
      expect(() => summedScoreTable(ITEMS_2PL, { testlets: [{ items: ["t1a"], sd }] })).toThrow(
        "Item 1 needs a string id, found undefined.",
      );
      expect(() => summedScoreTable(items, { testlets: [{ items: ["t1a"], sd: 0 }] })).toThrow(
        "The sd of testlet 1 must be a positive number, found 0.",
      );
      expect(() => summedScoreTable(items, { testlets: [{ items: ["t1a", "t1a"], sd }] })).toThrow(
        'Item "t1a" is in a testlet twice.',
      );
      expect(() => summedScoreTable(items, { testlets: JSON.parse('{"items":["t1a"],"sd":1}') })).toThrow(
        "The option testlets must be an array of testlets, found an object.",
      );
      expect(() => summedScoreTable(items, { testlets: JSON.parse("null") })).toThrow(
        "The option testlets must be an array of testlets, found null.",
      );
      expect(() => summedScoreTable(items, { testlets: [JSON.parse('"t1a"')] })).toThrow(
        'Testlet 1 must be an object { items, sd }, found "t1a".',
      );
      expect(() => summedScoreTable(items, { testlets: [{ items: [JSON.parse("3")], sd }] })).toThrow(
        "Testlet 1 needs item ids, found 3.",
      );
    });
  });

  it("leaves marginal reliability without testlets, which its information takes as independent", () => {
    const testlets = [{ items: ["item1", "item2"], sd: 1 }];
    // @ts-expect-error: the information of a testlet's patterns does not sum over its items.
    expect(() => marginalReliability(ITEMS_2PL, { testlets })).toThrow('There is no option "testlets"');
  });

  it("rejects a form without items and invalid items", () => {
    expect(() => summedScoreTable([])).toThrow("A summed-score table needs at least one item.");
    expect(() => summedScoreTable([{ model: twoParameterLogistic, parameters: { a: 1, b: 0, c: 0.1, d: 1 } }])).toThrow(
      "Item 1: A 2PL item has c = 0, found c = 0.1.",
    );
    expect(() => summedScoreTable(JSON.parse("{}"))).toThrow(
      "A summed-score table needs an array of items, found an object.",
    );
    expect(() => summedScoreTable(ITEMS_2PL, JSON.parse("null"))).toThrow("The options must be an object, found null.");
    expect(() => summedScoreTable(ITEMS_2PL, JSON.parse('{"verbose":1}'))).toThrow('There is no option "verbose"');
  });
});

describe("testlets", () => {
  for (const reference of testlet.cases) {
    it(`matches ${reference.name}`, () => {
      const matrix = buildResponseMatrix(reference.responses, { itemIds: reference.itemIds });
      const result = calibrate(matrix, twoParameterLogistic, {
        testlets: definedEntries(reference.testlets),
        quadrature: { points: testlet.points },
        tolerance: 1e-8,
        maxIterations: 100_000,
        standardErrors: "hessian",
      });
      expect(result.converged).toBe(true);
      reference.items.forEach((expected, i) => {
        const actual = defined(result.items[i]);
        expectParameters(actual.parameters, expected.parameters, TESTLET_PARAMETER_TOLERANCE, actual.id);
        for (const [name, value] of Object.entries(expected.standardErrors)) {
          const relative = TESTLET_STANDARD_ERROR_TOLERANCE * value;
          expectClose(defined(actual.standardErrors?.[name]), value, relative, `${actual.id} SE of ${name}`);
        }
      });
      result.testlets.forEach((actual, t) => {
        expect(actual.items).toEqual(Object.values(definedEntries(reference.testlets))[t]);
        expectClose(actual.sd, defined(reference.sd[t]), TESTLET_PARAMETER_TOLERANCE, `${actual.id} sd`);
        const error = defined(reference.sdStandardErrors[t]);
        expectClose(actual.standardError, error, TESTLET_STANDARD_ERROR_TOLERANCE * error, `${actual.id} SE`);
      });
      expectClose(result.logLikelihood, reference.logLikelihood, FIXED_LOG_LIKELIHOOD_TOLERANCE, "log-likelihood");
      expectClose(result.aic, -2 * result.logLikelihood + 2 * 27, 1e-9, "AIC");
    });

    it(`matches base R's EAP with the testlet effects integrated out on ${reference.name}`, () => {
      const items = reference.items.map(({ parameters }, j) => ({
        id: defined(reference.itemIds[j]),
        model: twoParameterLogistic,
        parameters,
      }));
      const testlets = Object.values(definedEntries(reference.testlets)).map((ids, t) => ({
        items: ids,
        sd: defined(reference.sd[t]),
      }));
      reference.abilities.forEach((expected, r) => {
        const actual = posteriorMeanEstimate(items, defined(reference.responses[r]), { testlets });
        expectClose(actual.theta, expected.theta, EXACT_ABILITY_TOLERANCE, `theta ${r + 1}`);
        expectClose(actual.standardError, expected.standardError, EXACT_ABILITY_TOLERANCE, `SE ${r + 1}`);
      });
    });
  }

  for (const reference of testletGroups.cases) {
    it(`matches ${reference.name}`, () => {
      const matrix = buildResponseMatrix(reference.responses, { itemIds: reference.itemIds });
      const result = calibrate(matrix, twoParameterLogistic, {
        groups: reference.groups,
        latent: Object.fromEntries(
          Object.entries(reference.latent).map(([group, name]) => [group, latentDistribution(name)]),
        ),
        fixed: definedEntries(reference.fixed),
        testlets: definedEntries(reference.testlets),
        quadrature: { points: testletGroups.points },
        tolerance: 1e-8,
        maxIterations: 100_000,
        standardErrors: "hessian",
      });
      expect(result.converged).toBe(true);
      reference.items.forEach((expected, i) => {
        const actual = defined(result.items[i]);
        expectParameters(actual.parameters, expected.parameters, TESTLET_PARAMETER_TOLERANCE, actual.id);
        if (expected.standardErrors === null) {
          expect(actual.standardErrors, actual.id).toBeNull();
          return;
        }
        for (const [name, value] of Object.entries(expected.standardErrors)) {
          const relative = TESTLET_STANDARD_ERROR_TOLERANCE * value;
          expectClose(defined(actual.standardErrors?.[name]), value, relative, `${actual.id} SE of ${name}`);
        }
      });
      result.testlets.forEach((actual, t) => {
        expect(actual.items).toEqual(Object.values(definedEntries(reference.testlets))[t]);
        expectClose(actual.sd, defined(reference.sd[t]), TESTLET_PARAMETER_TOLERANCE, `${actual.id} sd`);
        const error = defined(reference.sdStandardErrors[t]);
        expectClose(actual.standardError, error, TESTLET_STANDARD_ERROR_TOLERANCE * error, `${actual.id} SE`);
      });
      expect(result.groups.map((group) => group.group)).toEqual([...new Set(reference.groups)]);
      reference.distributions.forEach((expected, g) => {
        const actual = defined(result.groups[g]);
        expectClose(actual.mean, expected.mean, TESTLET_PARAMETER_TOLERANCE, `mean of group ${g + 1}`);
        expectClose(actual.sd, expected.sd, TESTLET_PARAMETER_TOLERANCE, `sd of group ${g + 1}`);
        if (expected.standardErrors === null) {
          expect(actual.standardErrors).toBeNull();
          return;
        }
        const errors = defined(actual.standardErrors);
        expectClose(
          errors.mean,
          expected.standardErrors.mean,
          TESTLET_STANDARD_ERROR_TOLERANCE * errors.mean,
          "SE of mean",
        );
        expectClose(errors.sd, expected.standardErrors.sd, TESTLET_STANDARD_ERROR_TOLERANCE * errors.sd, "SE of sd");
      });
      expectClose(result.logLikelihood, reference.logLikelihood, FIXED_LOG_LIKELIHOOD_TOLERANCE, "log-likelihood");
      expectClose(result.aic, -2 * result.logLikelihood + 2 * reference.parameterCount, 1e-9, "AIC");
      // mirt's bfactor with the same constraints stops at the maximum within its tolerance, never above it.
      if (reference.mirtLogLikelihood !== undefined) {
        expect(result.logLikelihood).toBeGreaterThanOrEqual(reference.mirtLogLikelihood - MIRT_LOG_LIKELIHOOD_SLACK);
      }
    });
  }

  it(`matches ${testletGroups.histogram.name}`, () => {
    const { histogram } = testletGroups;
    const result = calibrate(
      buildResponseMatrix(histogram.responses, { itemIds: histogram.itemIds }),
      twoParameterLogistic,
      {
        groups: histogram.groups,
        latent: { b: "empirical-histogram" },
        fixed: definedEntries(histogram.fixed),
        testlets: definedEntries(histogram.testlets),
        quadrature: { points: testletGroups.points },
        tolerance: histogram.tolerance,
        maxIterations: 100_000,
      },
    );
    expect(result).toMatchObject({ converged: true, iterations: histogram.iterations });
    histogram.weights.forEach((weight, q) => {
      expectClose(defined(result.groups[1]?.weights[q]), weight, HISTOGRAM_TOLERANCE, `weight ${q}`);
    });
    result.testlets.forEach((actual, t) => {
      expectClose(actual.sd, defined(histogram.sd[t]), HISTOGRAM_TOLERANCE, `${actual.id} sd`);
    });
    expectClose(result.logLikelihood, histogram.logLikelihood, 1e-8, "log-likelihood");
  });

  it("estimates the same with the groups renamed and the persons in another order", () => {
    const reference = defined(testletGroups.cases[0]);
    const options = {
      latent: { second: latentDistribution("normal") },
      testlets: definedEntries(reference.testlets),
      quadrature: { points: testletGroups.points },
      tolerance: 1e-8,
      maxIterations: 100_000,
    };
    const persons = reference.responses.length;
    const order = Array.from({ length: persons }, (_, p) => (p * 7 + 3) % persons);
    const first = calibrate(
      buildResponseMatrix(reference.responses, { itemIds: reference.itemIds }),
      twoParameterLogistic,
      {
        ...options,
        groups: reference.groups.map((group) => (group === "a" ? "first" : "second")),
      },
    );
    const shuffled = calibrate(
      buildResponseMatrix(
        order.map((p) => defined(reference.responses[p])),
        { itemIds: reference.itemIds },
      ),
      twoParameterLogistic,
      { ...options, groups: order.map((p) => (defined(reference.groups[p]) === "a" ? "first" : "second")) },
    );
    first.items.forEach((item, i) => {
      expectParameters(defined(shuffled.items[i]).parameters, item.parameters, 1e-6, item.id);
    });
    first.testlets.forEach((testletEstimate, t) => {
      expectClose(defined(shuffled.testlets[t]).sd, testletEstimate.sd, 1e-6, testletEstimate.id);
    });
    expectClose(defined(shuffled.groups[1]).mean, defined(first.groups[1]).mean, 1e-6, "mean");
    expectClose(shuffled.logLikelihood, first.logLikelihood, 1e-6, "log-likelihood");
  });

  it("lets a group with no respondents to a testlet leave that testlet's SD to the other groups", () => {
    const reference = defined(testletGroups.cases[0]);
    const skipped = new Set(defined(Object.values(reference.testlets)[1]));
    const rows = reference.responses.map((row, p) =>
      reference.groups[p] === "b"
        ? row.map((value, j) => (skipped.has(defined(reference.itemIds[j])) ? null : value))
        : row,
    );
    const result = calibrate(buildResponseMatrix(rows, { itemIds: reference.itemIds }), twoParameterLogistic, {
      groups: reference.groups,
      latent: { b: "normal" },
      testlets: definedEntries(reference.testlets),
      quadrature: { points: testletGroups.points },
      tolerance: 1e-6,
      maxIterations: 100_000,
    });
    expect(result.converged).toBe(true);
    for (const { sd } of result.testlets) {
      expect(sd).toBeGreaterThan(0);
      expect(Number.isFinite(sd)).toBe(true);
    }
  });

  it("rejects a testlet of fixed items that no person answered, whose SD would be 0 / 0", () => {
    const rows = simulate(5, 100, ITEMS_2PL).map((row) => [...row, null, null]);
    const fixed = { item5: { a: 1, b: 0, c: 0, d: 1 }, item6: { a: 1.2, b: 0.5, c: 0, d: 1 } };
    expect(() =>
      calibrate(buildResponseMatrix(rows), twoParameterLogistic, { fixed, testlets: { t: ["item5", "item6"] } }),
    ).toThrow('Testlet "t" has no response, so the SD of its effect has nothing to be estimated from.');
  });

  it("keeps a testlet's SD finite when the data leave no room for testlet effects", () => {
    const matrix = buildResponseMatrix([
      [1, 1, 0, 1, 0, 1],
      [0, 0, 1, 1, 1, 0],
      [1, 1, 1, 0, 1, 1],
      [0, 1, 0, 0, 0, 1],
      [1, 0, 1, 1, 1, 1],
      [0, 0, 0, 1, 0, 0],
    ]);
    const result = calibrate(matrix, twoParameterLogistic, {
      testlets: { first: ["item1", "item2", "item3"], second: ["item4", "item5", "item6"] },
      standardErrors: "hessian",
    });
    expect(nonFinite(result)).toEqual([]);
    for (const { sd } of result.testlets) expect(sd).toBeGreaterThan(0);
  });

  const named = ITEMS_2PL.map((item, i) => ({ ...item, id: `item${i + 1}` }));

  it("gives the EAP without testlets as the effects' SD goes to 0", () => {
    const testlets = [{ items: ["item1", "item2"], sd: 1e-9 }];
    const plain = posteriorMeanEstimate(named, [1, 0, 1, 1]);
    const nested = posteriorMeanEstimate(named, [1, 0, 1, 1], { testlets });
    expectClose(nested.theta, plain.theta, 1e-8);
    expectClose(nested.standardError, plain.standardError, 1e-8);
  });

  it("reports no testlets without them, and rejects testlets that do not fit the items or the options", () => {
    const rows = simulate(5, 200, ITEMS_2PL);
    const matrix = buildResponseMatrix(rows);
    expect(calibrate(matrix, twoParameterLogistic).testlets).toEqual([]);
    const options = { maxIterations: 3 };
    expect(() => calibrate(matrix, twoParameterLogistic, { ...options, testlets: { a: ["item1"] } })).toThrow(
      'Testlet "a" needs at least two items.',
    );
    expect(() => calibrate(matrix, twoParameterLogistic, { ...options, testlets: { a: ["item1", "q9"] } })).toThrow(
      'Testlet "a" names "q9", which is not an item id.',
    );
    expect(() =>
      calibrate(matrix, twoParameterLogistic, {
        ...options,
        testlets: { a: ["item1", "item2"], b: ["item2", "item3"] },
      }),
    ).toThrow('Item "item2" is in more than one testlet.');
    const testlets = { a: ["item1", "item2"] };
    expect(() =>
      calibrate(matrix, twoParameterLogistic, {
        ...options,
        testlets,
        quadrature: { method: "gauss-hermite", points: 21 },
      }),
    ).toThrow("Testlets need the rectangular quadrature rule.");
    const groups = rows.map((_, p) => (p < 100 ? "a" : "b"));
    expect(() => calibrate(matrix, twoParameterLogistic, { ...options, testlets, groups, latent: "normal" })).toThrow(
      "Estimating every group's distribution of theta needs fixed items or a standard normal group to fix the scale.",
    );
    expect(() =>
      calibrate(matrix, twoParameterLogistic, {
        ...options,
        testlets,
        groups,
        latent: { b: "normal" },
        quadrature: { method: "gauss-hermite" },
      }),
    ).toThrow("Testlets need the rectangular quadrature rule.");
    expect(() =>
      calibrate(matrix, twoParameterLogistic, { ...options, testlets, standardErrors: "cross-product" }),
    ).toThrow('Testlets need the standard errors "none" or "hessian".');
    const twice = [
      { items: ["item1", "item2"], sd: 1 },
      { items: ["item2", "item3"], sd: 1 },
    ];
    expect(() => posteriorMeanEstimate(named, [1, 0, 1, 1], { testlets: twice })).toThrow(
      'Item "item2" is in a testlet twice.',
    );
    expect(() => posteriorMeanEstimate(ITEMS_2PL, [1, 0, 1, 1], { testlets: [{ items: ["item1"], sd: 1 }] })).toThrow(
      "Item 1 needs a string id, found undefined.",
    );
    expect(() =>
      posteriorMeanEstimate(named, [1, 0, 1, 1], { testlets: [{ items: ["item1", "item2"], sd: 0 }] }),
    ).toThrow("The sd of testlet 1 must be a positive number, found 0.");
    expect(() =>
      posteriorMeanEstimate(named, [1, 0, 1, 1], { testlets: [{ items: JSON.parse("[0, 1]"), sd: 1 }] }),
    ).toThrow("Testlet 1 needs item ids, found 0.");
    expect(() =>
      posteriorMeanEstimate([...named, defined(named[0])], [1, 0, 1, 1, 1], {
        testlets: [{ items: ["item1"], sd: 1 }],
      }),
    ).toThrow('Item ids must be unique; "item1" appears twice.');
  });

  it("scores each person's answered items with the testlets of a calibration, as abilityEstimates gives them", () => {
    // Persons who skip a testlet item are estimated from a subset of the items: testlets name items by id.
    const rows = [
      [null, 1, 1, 0],
      [1, null, 0, 1],
      [1, 1, 1, 1],
    ];
    const testlets = [{ items: ["item1", "item2"], sd: 1.5 }];
    const estimates = abilityEstimates(buildResponseMatrix(rows), named, {
      estimator: (items, responses) => posteriorMeanEstimate(items, responses, { testlets }),
    });
    rows.forEach((row, person) => {
      const expected = posteriorMeanEstimate(named, row, { testlets });
      expect(estimates[person]).toEqual({ id: `person${person + 1}`, ...expected });
    });
  });
});

describe("generalizedGradedUnfolding", () => {
  it("is the partial credit model of 2C + 2 categories with thresholds b + τ, each category joined with its mirror", () => {
    fc.assert(
      fc.property(arbitraryUnfoldingItem, arbitraryTheta, ({ parameters: { a, b, t } }, theta) => {
        const top = 2 * t.length + 1;
        const thresholds = [...t, 0, ...t.map((threshold) => -threshold).reverse()].map((threshold) => b + threshold);
        const subjective = new Float64Array(top + 1);
        generalizedPartialCredit.probabilities({ a, b: thresholds }, theta, subjective);
        const [probabilities = []] = evaluate({ model: generalizedGradedUnfolding, parameters: { a, b, t } }, theta);
        probabilities.forEach((probability, z) => {
          expectClose(probability, defined(subjective[z]) + defined(subjective[top - z]), 1e-12);
        });
      }),
    );
  });

  it("gives the same probabilities when theta and b are reflected", () => {
    fc.assert(
      fc.property(arbitraryUnfoldingItem, arbitraryTheta, (item, theta) => {
        const reflected = { ...item, parameters: { ...item.parameters, b: -item.parameters.b } };
        const [probabilities = []] = evaluate(item, theta);
        const [mirrored = []] = evaluate(reflected, -theta);
        probabilities.forEach((probability, z) => {
          expectClose(probability, defined(mirrored[z]), 1e-14);
        });
      }),
    );
  });

  it("names a and b, bounded as the GGUM package bounds them, and the thresholds", () => {
    expect(generalizedGradedUnfolding.parameters(4)).toEqual([
      { name: "a", lower: 0.1, upper: 10 },
      { name: "b", lower: -10, upper: 10 },
      { name: "t1" },
      { name: "t2" },
      { name: "t3" },
    ]);
    expect(generalizedGradedUnfolding.fromVector(Float64Array.of(1.5, -0.5, -1, -0.4))).toEqual({
      a: 1.5,
      b: -0.5,
      t: [-1, -0.4],
    });
  });

  it("rejects a slope that is not above 0, a location that is not finite and an item without thresholds", () => {
    const message = (found: string): string =>
      `A GGUM item needs a finite a above 0, a finite b and at least one finite threshold t, found ${found}.`;
    expect(() => generalizedGradedUnfolding.toVector({ a: 0, b: 0, t: [-1] })).toThrow(
      message("a = 0, b = 0, t = [-1]"),
    );
    expect(() => generalizedGradedUnfolding.toVector({ a: 1, b: Number.NaN, t: [-1] })).toThrow(
      message("a = 1, b = NaN, t = [-1]"),
    );
    expect(() => generalizedGradedUnfolding.toVector({ a: 1, b: 0, t: [] })).toThrow(message("a = 1, b = 0, t = []"));
    expect(() => generalizedGradedUnfolding.toVector(JSON.parse('{ "a": "1", "b": 0, "t": {} }'))).toThrow(
      message('a = "1", b = 0, t = an object'),
    );
  });
});

describe("calibrateUnfolding", () => {
  const cases = Object.entries(unfolding.cases);
  /** The cross-product calibration of each case, made once for the tests that read it. */
  const fits = new Map<string, UnfoldingCalibration>();
  const fitOf = (name: string, responses: readonly string[]): UnfoldingCalibration => {
    const found = fits.get(name) ?? calibrateUnfolding(unfoldingMatrix(responses), { standardErrors: "cross-product" });
    fits.set(name, found);
    return found;
  };

  it("starts from the GGUM package's starts by Roberts and Laughlin's procedure", () => {
    for (const [name, expected] of Object.entries(unfolding.starts)) {
      const responses = compressResponses(
        unfoldingMatrix(defined(unfolding.cases[name as keyof typeof unfolding.cases]).responses),
      );
      const tallies = categoryTallies(responses);
      const proportions = responses.itemIds.map((id, i) =>
        checkCategories(id, generalizedGradedUnfolding, tallies[i] ?? NO_RESPONSES),
      );
      unfoldingStarts(responses, proportions).forEach((start, i) => {
        expectParameters(start, { a: 1, b: expected.b[i], t: expected.t[i] }, 1e-12, `${name} item ${i + 1}`);
      });
    }
  });

  for (const [name, reference] of cases) {
    it(`matches the GGUM package and the base R cross-product errors on ${name}`, () => {
      const result = fitOf(name, reference.responses);
      expect(result.converged).toBe(true);
      expect(result.persons).toBe(reference.persons);
      expectClose(result.logLikelihood, reference.expected.logLikelihood, UNFOLDING_LOG_LIKELIHOOD_TOLERANCE);
      result.items.forEach((item, i) => {
        const { a, b, t } = defined(reference.expected.items[i]);
        expect(item.model).toBe(generalizedGradedUnfolding);
        expectParameters(item.parameters, { a, b, t }, UNFOLDING_PARAMETER_TOLERANCE, `${name} ${item.id}`);
        expectParameters(
          item.standardErrors,
          reference.expected.crossProductErrors[i],
          UNFOLDING_STANDARD_ERROR_TOLERANCE,
          `${name} ${item.id} errors`,
          Math.abs,
        );
      });
    });
  }

  it("recovers every location within 0.75 and their RMSE within 0.2 in each case, up to the reflection", () => {
    for (const [name, reference] of cases) {
      const { items } = fitOf(name, reference.responses);
      const sign = orientation(items, reference.truth);
      const errors = items.map((item, i) => item.parameters.b - sign * defined(reference.truth[i]).b);
      expect(Math.max(...errors.map(Math.abs)), name).toBeLessThanOrEqual(0.75);
      expect(
        Math.sqrt(errors.reduce((sum, error) => sum + error * error, 0) / errors.length),
        name,
      ).toBeLessThanOrEqual(0.2);
    }
  });

  it("covers the true a and b with 0.92 to 0.98 of its 95% intervals over the 150 items of the seeds, where the GGUM package's errors of b cover fewer", () => {
    const covered = { a: 0, b: 0, ggumB: 0, items: 0 };
    for (const [name, reference] of cases.filter(([name]) => name.startsWith("seed"))) {
      const { items } = fitOf(name, reference.responses);
      const sign = orientation(items, reference.truth);
      items.forEach(({ parameters, standardErrors }, i) => {
        const truth = defined(reference.truth[i]);
        const { b: ggumB, standardErrors: ggumErrors } = defined(reference.expected.items[i]);
        const errors = defined(standardErrors);
        covered.items++;
        if (Math.abs(parameters.a - truth.a) < NORMAL_95 * defined(errors.a)) covered.a++;
        if (Math.abs(parameters.b - sign * truth.b) < NORMAL_95 * defined(errors.b)) covered.b++;
        if (Math.abs(ggumB - sign * truth.b) < NORMAL_95 * ggumErrors.b) covered.ggumB++;
      });
    }
    expect(covered.items).toBe(150);
    for (const share of [covered.a / covered.items, covered.b / covered.items]) {
      expect(share).toBeGreaterThanOrEqual(0.92);
      expect(share).toBeLessThanOrEqual(0.98);
    }
    expect([covered.a, covered.b, covered.ggumB]).toEqual([141, 142, 125]);
  });

  it("gives Hessian errors within 10% of the cross-product errors", () => {
    const [name = "", reference] = defined(cases[0]);
    const hessian = calibrateUnfolding(unfoldingMatrix(defined(reference).responses), { standardErrors: "hessian" });
    fitOf(name, defined(reference).responses).items.forEach((item, i) => {
      expectParameters(
        defined(hessian.items[i]).standardErrors,
        item.standardErrors,
        0.1,
        `${item.id} Hessian errors`,
        Math.abs,
      );
    });
  });

  it("reaches the maximum that the GGUM package stops short of at its default tolerance", () => {
    const { case: name, items, logLikelihood, tightLogLikelihood } = unfolding.defaultTolerance;
    const reference = defined(unfolding.cases[name as keyof typeof unfolding.cases]);
    const result = fitOf(name, reference.responses);
    const gaps = result.items.map((item, i) => Math.abs(item.parameters.b - defined(items[i]).b));
    expect(Math.max(...gaps)).toBeGreaterThan(0.15);
    expect(tightLogLikelihood - logLikelihood).toBeGreaterThan(0.14);
  });

  it("leaves out the persons who disagree with every item they answered", () => {
    const rows = [
      [0, 1, 0],
      [3, 0, 1],
      [0, 3, 2],
      [1, 0, null],
      [1, 2, 0],
      [2, 1, 3],
      [null, null, null],
      [0, 1, 2],
    ];
    const result = calibrateUnfolding(buildResponseMatrix(rows), { maxIterations: 20 });
    expect(result.persons).toBe(5);
    const kept = calibrateUnfolding(buildResponseMatrix(rows.filter((_, person) => ![0, 3, 6].includes(person))), {
      maxIterations: 20,
    });
    expect(result.items.map((item) => item.parameters)).toEqual(kept.items.map((item) => item.parameters));
    expect(result.logLikelihood).toBe(kept.logLikelihood);
    expect(result.bic).toBe(kept.bic);
  });

  it("gives records the calibration of the matrix", () => {
    const rows = defined(unfolding.cases.mixed).responses.slice(0, 300);
    const matrix = unfoldingMatrix(rows);
    const records = buildResponseRecords(
      rows.flatMap((row, person) =>
        Array.from(row, (response, item) => ({
          person: defined(matrix.personIds[person]),
          item: defined(matrix.itemIds[item]),
          response: Number(response),
        })),
      ),
    );
    expect(calibrateUnfolding(records, { maxIterations: 30 })).toEqual(
      calibrateUnfolding(matrix, { maxIterations: 30 }),
    );
  });

  it("takes the quadrature and the stopping rules, and reports the last stage's iterations", () => {
    const reference = defined(unfolding.cases.mixed);
    const matrix = unfoldingMatrix(reference.responses);
    const short = calibrateUnfolding(matrix, { maxIterations: 1 });
    expect(short.converged).toBe(false);
    expect(short.iterations).toBe(1);
    expect(short.items.every((item) => item.standardErrors === null)).toBe(true);
    const hermite = calibrateUnfolding(matrix, { quadrature: { method: "gauss-hermite", points: 41 } });
    expect(hermite.converged).toBe(true);
    defined(reference.expected.items).forEach(({ b }, i) => {
      expectClose(defined(hermite.items[i]).parameters.b, b, 0.02);
    });
  });

  it("calibrates by calibrate from starts by proportions alone, which place every location on one side", () => {
    const matrix = unfoldingMatrix(defined(unfolding.cases.mixed).responses);
    const result = calibrate(matrix, generalizedGradedUnfolding, { maxIterations: 50 });
    expect(nonFinite(result)).toEqual([]);
    expect(result.items.every((item) => item.parameters.a > 0)).toBe(true);
  });

  it("bounds the standard errors of the last stage to 2,000 parameters before any stage runs", () => {
    // 401 items of four categories estimate a, b and three thresholds each: 2,005 parameters.
    const rows = [0, 1, 2, 3].map((k) => Array.from({ length: 401 }, (_, j) => (j + k) % 4));
    expect(() => calibrateUnfolding(buildResponseMatrix(rows), { standardErrors: "cross-product" })).toThrow(
      "Standard errors take at most 2000 estimated parameters, as their information holds every pair of them; found 2005.",
    );
  });

  it("rejects more than 500 items, whose first stage would take minutes from data of a few kilobytes", () => {
    // Four persons answer each of 501 items in each of its four categories: a design of 2,004 records.
    const records = Array.from({ length: 501 }, (_, j) =>
      Array.from({ length: 4 }, (_, k) => ({ person: `p${4 * j + k}`, item: `i${j}`, response: k })),
    ).flat();
    expect(() => calibrateUnfolding(buildResponseRecords(records))).toThrow(
      "Unfolding calibration takes at most 500 items, as its first stage estimates the locations of every item at once; found 501.",
    );
  });

  it("rejects invalid options and data, and an item with responses in one category", () => {
    const matrix = buildResponseMatrix([
      [0, 2],
      [2, 1],
      [1, 0],
    ]);
    expect(() => calibrateUnfolding(matrix, JSON.parse("null"))).toThrow(DataError);
    expect(() => calibrateUnfolding(matrix, JSON.parse('{ "tolerance": "1e-4" }'))).toThrow(DataError);
    expect(() => calibrateUnfolding(matrix, JSON.parse('{ "standardErrors": "sandwich" }'))).toThrow(DataError);
    expect(() => calibrateUnfolding(matrix, JSON.parse('{ "quadrature": { "points": 1 } }'))).toThrow(DataError);
    expect(() => calibrateUnfolding(JSON.parse('{ "responses": {} }'))).toThrow(DataError);
    expect(() =>
      calibrateUnfolding(
        buildResponseMatrix([
          [0, 0],
          [0, 1],
        ]),
      ),
    ).toThrow('Item "item1" has responses in fewer than two categories; calibration needs at least two.');
    expect(() =>
      calibrateUnfolding(
        buildResponseMatrix([
          [0, 1],
          [1, 0],
          [3, 3],
          [2, 2],
        ]),
      ),
    ).toThrow(
      'Item "item1" has no response in category 0 from a person who agrees with an item; unfolding calibration leaves out the persons who disagree with every item they answered.',
    );
  });
});

/** The statistic of the likelihood ratio of two calibrations of the same responses. */
const NESTED_TOLERANCE = 1e-3;

describe("the recipe for the nested-model chi-square", () => {
  /** The recipe of irt.md: the chi-square of a restricted fit against a general one, with its degrees of freedom and p-value. */
  function likelihoodRatio(
    restricted: { aic: number; logLikelihood: number },
    general: { aic: number; logLikelihood: number },
  ) {
    const parameters = (fit: { aic: number; logLikelihood: number }) => (fit.aic + 2 * fit.logLikelihood) / 2;
    const chiSquare = 2 * (general.logLikelihood - restricted.logLikelihood);
    const degreesOfFreedom = parameters(general) - parameters(restricted);
    return { chiSquare, degreesOfFreedom, pValue: chiSquareUpperTail(chiSquare, degreesOfFreedom) };
  }

  it("equals mirt's anova of the 1PL against the 2PL", () => {
    const { slopes } = nestedModels;
    const matrix = buildResponseMatrix(slopes.responses, { itemIds: [...slopes.itemIds] });
    const result = likelihoodRatio(calibrate(matrix, oneParameterLogistic), calibrate(matrix, twoParameterLogistic));
    expectClose(result.chiSquare, slopes.chiSquare, NESTED_TOLERANCE, "chi-square");
    expect(result.degreesOfFreedom).toBe(slopes.degreesOfFreedom);
    expectClose(result.pValue, slopes.pValue, NESTED_TOLERANCE, "p-value");
  });

  it("equals mirt's anova of the partial credit model against the generalized partial credit model", () => {
    const { credit } = nestedModels;
    const matrix = buildResponseMatrix(credit.responses, { itemIds: [...credit.itemIds] });
    const result = likelihoodRatio(calibrate(matrix, partialCredit), calibrate(matrix, generalizedPartialCredit));
    expectClose(result.chiSquare, credit.chiSquare, NESTED_TOLERANCE, "chi-square");
    expect(result.degreesOfFreedom).toBe(credit.degreesOfFreedom);
    expectClose(result.pValue, credit.pValue, NESTED_TOLERANCE, "p-value");
  });

  it("recovers the difference in the numbers of parameters that mirt counts", () => {
    const { slopes, credit } = nestedModels;
    expect(slopes.general.parameters - slopes.restricted.parameters).toBe(slopes.degreesOfFreedom);
    expect(credit.general.parameters - credit.restricted.parameters).toBe(credit.degreesOfFreedom);
  });

  it("gives a statistic of 0 for a model against itself, which has no degrees of freedom", () => {
    const matrix = buildResponseMatrix(nestedModels.slopes.responses, { itemIds: [...nestedModels.slopes.itemIds] });
    const fit = calibrate(matrix, twoParameterLogistic);
    expect(likelihoodRatio(fit, fit)).toMatchObject({ chiSquare: 0, degreesOfFreedom: 0 });
  });
});

/** The shares of the Hessian's standard errors that the bootstrap's may differ by. */
const BOOTSTRAP_SHARE = 0.15;
/** The calibrations of resampled persons that a standard error is computed from. */
const BOOTSTRAP_REPLICATES = 400;

describe("the recipe for bootstrap standard errors", () => {
  it("gives standard errors within 15% of the Hessian's on 1,000 persons and 8 items", () => {
    const random = createRandom(5);
    const items = Array.from({ length: 8 }, (_, j) => ({
      model: twoParameterLogistic,
      parameters: { a: 0.8 + 0.1 * j, b: -1.4 + 0.4 * j, c: 0, d: 1 },
    }));
    const matrix = simulateResponses(items, simulateThetas(1000, random), random);
    const fit = calibrate(matrix, twoParameterLogistic, { standardErrors: "hessian" });

    // The recipe of irt.md: calibrations of persons drawn with replacement.
    const persons = matrix.responses.length;
    const replicates = Array.from({ length: BOOTSTRAP_REPLICATES }, () => {
      const rows = Array.from(
        { length: persons },
        () => matrix.responses[Math.floor(random.uniform() * persons)] ?? [],
      );
      return calibrate(buildResponseMatrix(rows, { itemIds: matrix.itemIds }), twoParameterLogistic);
    });
    const standardDeviation = (values: readonly number[]) => {
      const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
      return Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (values.length - 1));
    };

    expect(replicates.every((replicate) => replicate.converged)).toBe(true);
    fit.items.forEach((item, j) => {
      for (const name of ["a", "b"] as const) {
        const bootstrap = standardDeviation(
          replicates.map((replicate) => defined(replicate.items[j]).parameters[name]),
        );
        const hessian = defined(item.standardErrors?.[name]);
        expect(Math.abs(bootstrap / hessian - 1), `${item.id} ${name}`).toBeLessThan(BOOTSTRAP_SHARE);
      }
    });
  });
});

describe("the recipe for an item with an empty category", () => {
  it("collapses the empty end categories and writes the item back over every point with structural zeros", () => {
    const random = createRandom(14);
    const items = [
      ...[-1, -0.5, 0, 0.3, 0.8, 1.2].map((b) => ({
        model: twoParameterLogistic,
        parameters: { a: 1.2, b, c: 0, d: 1 },
      })),
      {
        model: generalizedPartialCredit,
        parameters: { a: 0.8, b: [-5, -2.5, -1.6, -0.9, -0.3, 0.2, 0.8, 1.5, 2.4, 5] },
      },
    ];
    const rows = simulateResponses(items, simulateThetas(500, random), random).responses;
    const models: IrtModel<LogisticItem | ThresholdItem>[] = [
      ...items.slice(0, 6).map(() => twoParameterLogistic),
      generalizedPartialCredit,
    ];
    expect(() => calibrate(buildResponseMatrix(rows), models)).toThrow('Item "item7" has no response in category 0');
    const scores = rows.flatMap((row) => (row[6] === null || row[6] === undefined ? [] : [row[6]]));
    const lowest = scores.reduce((least, score) => Math.min(least, score));
    const highest = scores.reduce((most, score) => Math.max(most, score));
    expect([lowest, highest]).toEqual([1, 9]);
    const collapsed = rows.map((row) => row.map((score, j) => (j === 6 && score !== null ? score - lowest : score)));
    const fit = calibrate(buildResponseMatrix(collapsed), models);
    const { a, b } = defined(fit.items[6]).parameters as ThresholdItem;
    const reach = 6 + 30 / a;
    const thresholds = Array.from({ length: 10 }, (_, k) =>
      k < lowest ? -reach : k >= highest ? reach : (b[k - lowest] ?? 0),
    );
    const text = { model: generalizedPartialCredit, parameters: { a, b: thresholds } };
    const table = summedScoreTable([...fit.items.slice(0, 6), text]);
    expect(table).toHaveLength(17);
    expectClose(defined(table[0]).probability, 3.9e-17, 0.05e-17, "probability of 0");
    expectClose(defined(defined(table[1]).theta), -1.993, 5e-4, "theta of 1");
    // The table over the points is the collapsed one moved by the lowest point, with structural zeros at the ends.
    const shorter = summedScoreTable(fit.items);
    shorter.forEach((row, score) => {
      expectClose(defined(table[score + lowest]).probability, row.probability, 1e-12, `probability of ${score}`);
      expectClose(defined(table[score + lowest]?.theta), defined(row.theta), 1e-9, `theta of ${score}`);
    });
    expect(defined(table.at(-1)).probability).toBeLessThan(1e-12);
  });
});

describe("properties", () => {
  it("a shift of every threshold leaves the conditional probability of a response pattern unchanged", () => {
    const item = fc.array(fc.double({ min: -3, max: 3, noNaN: true }), { minLength: 1, maxLength: 3 });
    fc.assert(
      fc.property(
        fc.array(item, { minLength: 2, maxLength: 4 }),
        fc.double({ min: -2, max: 2, noNaN: true }),
        fc.array(fc.nat(), { minLength: 4, maxLength: 4 }),
        (thresholds, shift, draws) => {
          const pattern = thresholds.map((tau, i) => (draws[i] ?? 0) % (tau.length + 1));
          const logProbability = (offset: number): number => {
            const eta = thresholds.map((tau) => {
              let sum = 0;
              return tau.map((value) => (sum += value + offset));
            });
            const functions = eta.map((item) => itemFunctions(item)).reduce(combine, NO_ITEMS);
            const total = pattern.reduce((sum, response) => sum + response, 0);
            const weight = pattern.reduce(
              (sum, response, i) => sum - (response === 0 ? 0 : (eta[i]?.[response - 1] ?? 0)),
              0,
            );
            return weight - logFunction(functions, total);
          };
          expectClose(logProbability(shift), logProbability(0), 1e-9);
        },
      ),
    );
  });

  it("conditional maximum likelihood does not depend on the order of persons", () => {
    fc.assert(
      fc.property(fc.integer({ min: 1, max: 1_000_000 }), (seed) => {
        const random = createRandom(seed);
        const items = simulateItems(partialCredit, 4, random, { categoryCount: 3 });
        const { responses } = simulateResponses(items, simulateThetas(150, random), random);
        let forward: ConditionalCalibration;
        try {
          forward = conditionalMaximumLikelihood(buildResponseMatrix(responses));
        } catch (error) {
          if (error instanceof DataError) return;
          throw error;
        }
        const backward = conditionalMaximumLikelihood(buildResponseMatrix([...responses].reverse()));
        expectClose(backward.logLikelihood, forward.logLikelihood, 1e-9 * Math.abs(forward.logLikelihood));
        backward.items.forEach((item, i) => {
          const b = (item.parameters as ThresholdItem).b;
          const expected = (defined(forward.items[i]).parameters as ThresholdItem).b;
          b.forEach((value, k) => {
            expectClose(value, expected[k] ?? null, 1e-9);
          });
        });
      }),
      { numRuns: 20 },
    );
  });

  it("category probabilities lie in [0, 1] and sum to 1", () => {
    fc.assert(
      fc.property(arbitraryItem, arbitraryTheta, (item, theta) => {
        const [probabilities = []] = evaluate(item, theta);
        expect(probabilities.every((p) => p >= 0 && p <= 1)).toBe(true);
        expectClose(
          probabilities.reduce((sum, p) => sum + p, 0),
          1,
          1e-14,
        );
      }),
    );
  });

  it("information is non-negative and the sum over categories of P'² / P", () => {
    fc.assert(
      fc.property(arbitraryItem, arbitraryTheta, (item, theta) => {
        const information = itemInformation(item, theta);
        expect(information).toBeGreaterThanOrEqual(0);
        const [probabilities = [], first = []] = evaluate(item, theta);
        const sum = probabilities.reduce((total, p, k) => (p > 0 ? total + defined(first[k]) ** 2 / p : total), 0);
        expectClose(information, sum, 1e-9 * Math.max(1, sum));
      }),
    );
  });

  it("derivatives match central differences", () => {
    const h = 1e-6;
    fc.assert(
      fc.property(arbitraryItem, fc.double({ min: -4, max: 4, noNaN: true }), (item, theta) => {
        const { model: itemModel, parameters } = item;
        const count = itemModel.categoryCount(parameters);
        const at = (t: number, values = parameters): Float64Array => {
          const out = new Float64Array(count);
          itemModel.probabilities(values, t, out);
          return out;
        };
        const [, first = [], second = []] = evaluate(item, theta);
        const [, above = []] = evaluate(item, theta + h);
        const [, below = []] = evaluate(item, theta - h);
        const vector = itemModel.toVector(parameters);
        const derivatives = new Float64Array(count * vector.length);
        itemModel.parameterDerivatives(parameters, theta, derivatives);
        for (let k = 0; k < count; k++) {
          const slopeDifference = (defined(at(theta + h)[k]) - defined(at(theta - h)[k])) / (2 * h);
          expectClose(defined(first[k]), slopeDifference, 1e-6, `dP(${k})/dθ`);
          const curvatureDifference = (defined(above[k]) - defined(below[k])) / (2 * h);
          expectClose(defined(second[k]), curvatureDifference, 1e-5, `d²P(${k})/dθ²`);
          vector.forEach((value, j) => {
            const shifted = (step: number): number => {
              const changed = Float64Array.from(vector);
              changed[j] = value + step;
              return defined(at(theta, itemModel.fromVector(changed))[k]);
            };
            const expected = (shifted(h) - shifted(-h)) / (2 * h);
            expectClose(defined(derivatives[k * vector.length + j]), expected, 1e-6, `dP(${k})/dξ(${j})`);
          });
        }
      }),
    );
  });

  it("the GPCM with equal slopes equals the PCM, and with thresholds b + τ_k the RSM", () => {
    fc.assert(
      fc.property(arbitrarySlope, arbitraryLocations, arbitraryLocation, arbitraryTheta, (a, b, shift, theta) => {
        expect(evaluate({ model: partialCredit, parameters: { a, b } }, theta)).toEqual(
          evaluate({ model: generalizedPartialCredit, parameters: { a, b } }, theta),
        );
        const rsm = evaluate({ model: ratingScale, parameters: { a, b: shift, steps: b } }, theta);
        const gpcm = evaluate(
          { model: generalizedPartialCredit, parameters: { a, b: b.map((step) => shift + step) } },
          theta,
        );
        rsm.forEach((values, i) => {
          values.forEach((value, k) => {
            expectClose(value, defined(gpcm[i]?.[k]), 1e-10 * Math.max(1, Math.abs(value)));
          });
        });
      }),
    );
  });

  it("ability estimates are finite and inside the range", () => {
    const pools = fc
      .array(arbitraryItem, { minLength: 1, maxLength: 6 })
      .chain((pool) =>
        fc.tuple(
          fc.constant(pool),
          fc.tuple(
            ...pool.map(({ model: itemModel, parameters }) =>
              fc.option(fc.integer({ min: 0, max: itemModel.categoryCount(parameters) - 1 }), { nil: null }),
            ),
          ),
        ),
      );
    fc.assert(
      fc.property(pools, ([pool, responses]) => {
        const answered = responses.some((response) => response !== null);
        const estimates = [
          posteriorModeEstimate(pool, responses),
          posteriorMeanEstimate(pool, responses),
          ...(answered
            ? [maximumLikelihoodEstimate(pool, responses), weightedLikelihoodEstimate(pool, responses)]
            : []),
        ];
        for (const { theta, standardError } of estimates) {
          expect(theta).toBeGreaterThanOrEqual(-6);
          expect(theta).toBeLessThanOrEqual(6);
          expect(Number.isNaN(standardError)).toBe(false);
          expect(standardError).toBeGreaterThan(0);
        }
      }),
    );
  });

  it("summed-score probabilities sum to 1, and marginal reliability lies in [0, 1)", () => {
    fc.assert(
      fc.property(fc.array(arbitraryItem, { minLength: 1, maxLength: 6 }), (items) => {
        const table = summedScoreTable(items);
        expectClose(
          table.reduce((sum, row) => sum + row.probability, 0),
          1,
          1e-12,
        );
        for (const row of table) expect(Math.abs(defined(row.theta))).toBeLessThan(6);
        const reliability = marginalReliability(items);
        expect(reliability).toBeGreaterThanOrEqual(0);
        expect(reliability).toBeLessThan(1);
      }),
    );
  });

  it("summed-score tables with testlets sum to 1 and do not depend on the order of the items", () => {
    const sd = fc.double({ min: 0.1, max: 3, noNaN: true });
    fc.assert(
      fc.property(
        fc.array(fc.tuple(arbitraryItem, fc.integer({ min: -1, max: 1 })), { minLength: 1, maxLength: 6 }),
        fc.tuple(sd, sd),
        (rows, sds) => {
          const items = rows.map(([item], i) => ({ ...item, id: `i${i}` }));
          const testlets = sds.map((value, t) => ({
            items: rows.flatMap(([, testlet], i) => (testlet === t ? [`i${i}`] : [])),
            sd: value,
          }));
          const table = summedScoreTable(items, { testlets });
          expectClose(
            table.reduce((sum, row) => sum + row.probability, 0),
            1,
            1e-12,
          );
          const reversed = summedScoreTable(items.toReversed(), { testlets });
          table.forEach((row, s) => {
            expectClose(defined(reversed[s]).probability, row.probability, 1e-12, `probability of ${s}`);
            expectClose(defined(reversed[s]).theta, row.theta, 1e-9, `theta of ${s}`);
          });
        },
      ),
    );
  });

  it("calibration results contain no NaN or infinities", () => {
    const rows = fc
      .integer({ min: 2, max: 4 })
      .chain((items) =>
        fc.array(
          fc.array(fc.option(fc.constantFrom(0, 1, 2), { freq: 9, nil: null }), { minLength: items, maxLength: items }),
          { minLength: 2, maxLength: 30 },
        ),
      );
    const models = fc.constantFrom<IrtModel>(
      oneParameterLogistic,
      twoParameterLogistic,
      threeParameterLogistic,
      gradedResponse,
      generalizedPartialCredit,
      partialCredit,
      ratingScale,
      nominalResponse,
      generalizedGradedUnfolding,
    );
    fc.assert(
      fc.property(rows, models, (data, m) => {
        try {
          expect(nonFinite(calibrate(buildResponseMatrix(data), m, { maxIterations: 100 }))).toEqual([]);
        } catch (error) {
          if (!(error instanceof DataError)) throw error;
        }
      }),
      { numRuns: 80 },
    );
  });

  it("the log-likelihood of EM never falls below that of its first iteration, and rises over the first two", () => {
    const pools: readonly (readonly IrtItem[])[] = [ITEMS_2PL, ITEMS_GPCM];
    fc.assert(
      fc.property(
        fc.integer({ min: 0, max: 2 ** 31 }),
        fc.constantFrom<[IrtModel, number]>(
          [oneParameterLogistic, 0],
          [twoParameterLogistic, 0],
          [gradedResponse, 1],
          [generalizedPartialCredit, 1],
        ),
        (seed, [m, pool]) => {
          const matrix = buildResponseMatrix(simulate(seed, 150, defined(pools[pool])));
          const [first = 0, second = 0, ...later] = Array.from(
            { length: 12 },
            (_, i) => calibrate(matrix, m, { maxIterations: i + 1, tolerance: 1e-12 }).logLikelihood,
          );
          expect(second).toBeGreaterThanOrEqual(first - 1e-9);
          // The extrapolation of the third iteration may fall below the second iteration, as it may rise far above it.
          for (const logLikelihood of later) expect(logLikelihood).toBeGreaterThanOrEqual(first - 1e-9);
        },
      ),
      { numRuns: 20 },
    );
  });

  it("seeded runs reproduce exactly", () => {
    const pool = [...ITEMS_2PL, ...ITEMS_GPCM];
    const models: IrtModel<LogisticItem | ThresholdItem>[] = [
      ...ITEMS_2PL.map(() => threeParameterLogistic),
      ...ITEMS_GPCM.map(() => generalizedPartialCredit),
    ];
    fc.assert(
      fc.property(fc.integer({ min: 0, max: 2 ** 31 }), (seed) => {
        const run = (): unknown => {
          const matrix = buildResponseMatrix(simulate(seed, 300, pool));
          return [
            calibrate(matrix, models, { priors: { c: betaPrior(5, 17) }, maxIterations: 50 }),
            jointMaximumLikelihood(matrix),
          ];
        };
        expect(run()).toEqual(run());
      }),
      { numRuns: 10 },
    );
  });

  it("person weights all equal give the unweighted calibration", () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 1, max: 1_000_000 }),
        fc.double({ min: 1e-3, max: 1e3, noNaN: true }),
        (seed, weight) => {
          const matrix = buildResponseMatrix(simulate(seed, 100, ITEMS_2PL));
          const options = { standardErrors: "cross-product" } as const;
          const personWeights = matrix.personIds.map(() => weight);
          expect(calibrate(matrix, twoParameterLogistic, { ...options, personWeights })).toEqual(
            calibrate(matrix, twoParameterLogistic, options),
          );
        },
      ),
      { numRuns: 20 },
    );
  });

  it("person weights scaled by a constant rescale alike, and by a power of 2 give the same calibration", () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 1, max: 1_000_000 }),
        fc.array(fc.double({ min: 0.1, max: 10, noNaN: true }), { minLength: 100, maxLength: 100 }),
        fc.double({ min: 1e-3, max: 1e3, noNaN: true }),
        fc.integer({ min: -60, max: 60 }),
        (seed, weights, scale, power) => {
          const matrix = buildResponseMatrix(simulate(seed, 100, ITEMS_2PL));
          const fit = (factor: number) =>
            calibrate(matrix, twoParameterLogistic, {
              standardErrors: "cross-product",
              personWeights: weights.map((weight) => weight * factor),
            });
          expect(fit(2 ** power)).toEqual(fit(1));
          const answered = weights.map(() => true);
          const base = defined(rescaledWeights(weights, answered));
          const scaled = defined(
            rescaledWeights(
              weights.map((weight) => weight * scale),
              answered,
            ),
          );
          scaled.forEach((weight, person) => {
            const expected = defined(base[person]);
            expectClose(weight, expected, RESCALED_WEIGHTS_TOLERANCE * expected, `person ${person + 1}`);
          });
        },
      ),
      { numRuns: 20 },
    );
  });
});
