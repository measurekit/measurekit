import { chiSquareUpperTail, mean } from "@measurekit/core";
import fc from "fast-check";
import { describe, expect, expectTypeOf, it } from "vitest";
import { purified } from "../src/dif/internal.js";
import {
  buildResponseMatrix,
  createRandom,
  DataError,
  type DifGroup,
  type DifPurification,
  differentialFunctioning,
  fourParameterLogistic,
  generalizedPartialCredit,
  gradedResponse,
  type IrtDifOptions,
  type IrtDifTest,
  type IrtModel,
  irtLikelihoodRatioDif,
  type LinkedDifOptions,
  type LogisticItem,
  type LogisticRegressionDif,
  logisticDiscriminantDif,
  logisticRegressionDif,
  lordChiSquare,
  type MantelHaenszel,
  type MantelTest,
  mantelHaenszel,
  mantelTest,
  type OrdinalLogisticDif,
  oneParameterLogistic,
  ordinalLogisticRegressionDif,
  type PurificationOptions,
  type RajuArea,
  type ResponseMatrix,
  rajuAreaDif,
  type Sibtest,
  sibtest,
  simulateItems,
  simulateResponses,
  simulateThetas,
  type TestItem,
  threeParameterLogistic,
  twoParameterLogistic,
} from "../src/index.js";
import { arbitraryItem } from "./arbitraries.js";
import { defined, expectClose } from "./assertions.js";
import { readFixture } from "./fixtures.js";

interface DifFixture {
  cases: {
    name: string;
    responses: number[][];
    focal: number[];
    mantelHaenszel: {
      continuityCorrection: boolean;
      chiSquare: number[];
      significance: number[];
      oddsRatio: number[];
      delta: number[];
      deltaStandardError: number[];
      etsCategory: string[];
    }[];
    logistic: {
      uniform: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
      nonuniform: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
      both: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
    };
    purifiedMantelHaenszel: {
      chiSquare: number[];
      significance: number[];
      oddsRatio: number[];
      delta: number[];
      deltaStandardError: number[];
      etsCategory: string[];
      anchors: number[];
      iterations: number;
      converged: boolean;
    };
    purifiedLogistic: {
      uniform: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
      nonuniform: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
      both: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
      anchors: number[];
      iterations: number;
      converged: boolean;
    };
    sparse: {
      responses: (number | null)[][];
      mantelHaenszel: {
        chiSquare: number[];
        significance: number[];
        oddsRatio: number[];
        delta: number[];
        deltaStandardError: number[];
        etsCategory: string[];
      };
      logistic: {
        uniform: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        nonuniform: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        both: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
      };
      purifiedMantelHaenszel: {
        chiSquare: number[];
        significance: number[];
        oddsRatio: number[];
        delta: number[];
        deltaStandardError: number[];
        etsCategory: string[];
        anchors: number[];
        iterations: number;
        converged: boolean;
      };
      purifiedLogistic: {
        uniform: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        nonuniform: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        both: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        anchors: number[];
        iterations: number;
        converged: boolean;
      };
    };
    lord: {
      chiSquare: number[];
      significance: number[];
      reference: { a: number[]; b: number[] };
      focal: { a: number[]; b: number[] };
      transformation: { slope: number; intercept: number };
    };
  }[];
}
const fixture = readFixture<DifFixture>("dif.json");

// biome-ignore-start lint/style/useNamingConvention: the keys are those of the fixture file.
interface DifIrtFixture {
  tolerance: number;
  likelihoodRatio: {
    "verbal, 2PL": {
      input: { fixture: string; case: string; sparse: boolean };
      model: string;
      default: { chiSquare: number[]; degreesOfFreedom: number[]; significance: number[] };
      logLikelihood: number;
      focal: { mean: number; sd: number };
      purified: {
        chiSquare: number[];
        degreesOfFreedom: number[];
        significance: number[];
        anchors: number[];
        iterations: number;
        converged: boolean;
        logLikelihood: number;
      };
    };
    "bfi, graded": {
      input: { fixture: string; case: string; sparse: boolean };
      model: string;
      default: { chiSquare: number[]; degreesOfFreedom: number[]; significance: number[] };
      logLikelihood: number;
      focal: { mean: number; sd: number };
      purified: {
        chiSquare: number[];
        degreesOfFreedom: number[];
        significance: number[];
        anchors: number[];
        iterations: number;
        converged: boolean;
        logLikelihood: number;
      };
    };
  };
  raju: {
    verbal: {
      input: { fixture: string; case: string; sparse: boolean };
      default: {
        signedArea: number[];
        signedZ: number[];
        unsignedArea: number[];
        unsignedZ: number[];
        significance: number[];
        difrUnsignedArea: number[];
        difrSignificance: number[];
      };
      purified: {
        signedArea: number[];
        signedZ: number[];
        unsignedArea: number[];
        unsignedZ: number[];
        significance: number[];
        difrUnsignedArea: number[];
        difrSignificance: number[];
        anchors: number[];
        iterations: number;
        converged: boolean;
      };
    };
    simulated: {
      input: { fixture: string; case: string; sparse: boolean };
      default: {
        signedArea: number[];
        signedZ: number[];
        unsignedArea: number[];
        unsignedZ: number[];
        significance: number[];
        difrUnsignedArea: number[];
        difrSignificance: number[];
      };
      purified: {
        signedArea: number[];
        signedZ: number[];
        unsignedArea: number[];
        unsignedZ: number[];
        significance: number[];
        difrUnsignedArea: number[];
        difrSignificance: number[];
        anchors: number[];
        iterations: number;
        converged: boolean;
      };
    };
  };
  population: { nodes: number[]; weights: number[] };
  functioning: {
    "2PL": {
      model: string;
      reference: { a: number; b: number; c: number; d: number }[];
      focal: { a: number; b: number; c: number; d: number }[];
      ncdif: number[];
      cdif: number[];
      dtf: number;
    };
    graded: {
      model: string;
      reference: { a: number; b: number[] }[];
      focal: { a: number; b: number[] }[];
      ncdif: number[];
      cdif: number[];
      dtf: number;
    };
  };
}
// biome-ignore-end lint/style/useNamingConvention: the keys are those of the fixture file.
const irtFixture = readFixture<DifIrtFixture>("dif-irt.json");

interface DifLordPurificationFixture {
  cases: {
    name: string;
    chiSquare: number[];
    significance: number[];
    iterations: number;
    converged: boolean;
    anchors: number[];
    transformation: { slope: number; intercept: number };
  }[];
}
const lordPurification = readFixture<DifLordPurificationFixture>("dif-lord-purification.json");

interface DifFixedParametersFixture {
  itemIds: string[];
  responses: number[][];
  focal: number[];
  fixedParameters: Record<string, { c: number }>;
  calibrationTolerance: number;
  lord: {
    chiSquare: number[];
    degreesOfFreedom: number;
    significance: number[];
    reference: { a: number[]; b: number[] };
    focal: { a: number[]; b: number[] };
    transformation: { slope: number; intercept: number };
  };
  raju: RajuFixture;
  likelihoodRatioTolerance: number;
  likelihoodRatio: {
    chiSquare: number[];
    degreesOfFreedom: number[];
    significance: number[];
    logLikelihood: number;
    focal: { mean: number; sd: number };
  };
}
const heldFixture = readFixture<DifFixedParametersFixture>("dif-fixed-parameters.json");
/** The 3PL items of dif-fixed-parameters.json with their lower asymptotes held, as difR's `c` holds them. */
const heldCase = {
  matrix: buildResponseMatrix(heldFixture.responses, { itemIds: heldFixture.itemIds }),
  groups: groups(heldFixture.focal),
  options: {
    model: threeParameterLogistic,
    fixedParameters: heldFixture.fixedParameters,
    tolerance: heldFixture.calibrationTolerance,
    maxIterations: 10_000,
  },
};

interface DifPolytomousFixture {
  cases: {
    name: string;
    responses: number[][];
    focal: number[];
    theta: number[];
    complete: {
      mantel: {
        chiSquare: number[];
        significance: number[];
        generalizedChiSquare: number[];
        generalizedDegreesOfFreedom: number[];
        generalizedSignificance: number[];
        standardizedMeanDifference: number[];
        effectSize: number[];
        etsCategory: string[];
      };
      ordinal: {
        uniform: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        nonuniform: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        both: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        slopeChange: number[];
      };
      ordinalTheta: {
        uniform: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        nonuniform: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        both: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        slopeChange: number[];
      };
      purifiedMantel: {
        chiSquare: number[];
        significance: number[];
        generalizedChiSquare: number[];
        generalizedDegreesOfFreedom: number[];
        generalizedSignificance: number[];
        standardizedMeanDifference: number[];
        effectSize: number[];
        etsCategory: string[];
        anchors: number[];
        iterations: number;
        converged: boolean;
      };
      purifiedOrdinal: {
        uniform: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        nonuniform: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        both: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        slopeChange: number[];
        anchors: number[];
        iterations: number;
        converged: boolean;
      };
      lordif: {
        uniform: { significance: number[]; rSquaredChange: number[] };
        nonuniform: { significance: number[]; rSquaredChange: number[] };
        both: { significance: number[]; rSquaredChange: number[] };
        slopeChange: number[];
      };
      lordifTheta: {
        uniform: { significance: number[]; rSquaredChange: number[] };
        nonuniform: { significance: number[]; rSquaredChange: number[] };
        both: { significance: number[]; rSquaredChange: number[] };
        slopeChange: number[];
      };
    };
    sparse: {
      responses: (number | null)[][];
      theta: number[];
      mantel: {
        chiSquare: number[];
        significance: number[];
        generalizedChiSquare: number[];
        generalizedDegreesOfFreedom: number[];
        generalizedSignificance: number[];
        standardizedMeanDifference: number[];
        effectSize: number[];
        etsCategory: string[];
      };
      ordinal: {
        uniform: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        nonuniform: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        both: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        slopeChange: number[];
      };
      ordinalTheta: {
        uniform: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        nonuniform: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        both: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        slopeChange: number[];
      };
      purifiedMantel: {
        chiSquare: number[];
        significance: number[];
        generalizedChiSquare: number[];
        generalizedDegreesOfFreedom: number[];
        generalizedSignificance: number[];
        standardizedMeanDifference: number[];
        effectSize: number[];
        etsCategory: string[];
        anchors: number[];
        iterations: number;
        converged: boolean;
      };
      purifiedOrdinal: {
        uniform: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        nonuniform: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        both: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        slopeChange: number[];
        anchors: number[];
        iterations: number;
        converged: boolean;
      };
    };
  }[];
}
const polytomous = readFixture<DifPolytomousFixture>("dif-polytomous.json");

interface DifPolytomousTestsFixture {
  monteCarlo: { spread: number; equation: number; difr: number; difrReversed: number };
  cases: {
    verbal: {
      input: { fixture: string; case: string; sparse: boolean };
      liuAgresti: {
        oddsRatio: number[];
        logStandardError: number[];
        z: number[];
        significance: number[];
        difrStandardError: number[];
        difrReversedStandardError: number[];
      };
      discriminant: {
        uniform: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        nonuniform: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        both: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
      };
      purifiedDiscriminant: {
        uniform: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        nonuniform: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        both: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        anchors: number[];
        iterations: number;
        converged: boolean;
      };
    };
    simulated: {
      input: { fixture: string; case: string; sparse: boolean };
      liuAgresti: {
        oddsRatio: number[];
        logStandardError: number[];
        z: number[];
        significance: number[];
        difrStandardError: number[];
        difrReversedStandardError: number[];
      };
      discriminant: {
        uniform: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        nonuniform: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        both: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
      };
      purifiedDiscriminant: {
        uniform: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        nonuniform: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        both: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        anchors: number[];
        iterations: number;
        converged: boolean;
      };
    };
    bfi: {
      input: { fixture: string; case: string; sparse: boolean };
      liuAgresti: {
        oddsRatio: number[];
        logStandardError: number[];
        z: number[];
        significance: number[];
        difrStandardError: number[];
        difrReversedStandardError: number[];
      };
      discriminant: {
        uniform: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        nonuniform: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        both: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
      };
      purifiedDiscriminant: {
        uniform: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        nonuniform: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        both: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        anchors: number[];
        iterations: number;
        converged: boolean;
      };
      purifiedLiuAgresti: {
        oddsRatio: number[];
        logStandardError: number[];
        z: number[];
        significance: number[];
        difrStandardError: number[];
        difrReversedStandardError: number[];
        anchors: number[];
      };
      discriminantTheta: {
        uniform: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        nonuniform: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        both: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
      };
    };
    graded: {
      input: { fixture: string; case: string; sparse: boolean };
      liuAgresti: {
        oddsRatio: number[];
        logStandardError: number[];
        z: number[];
        significance: number[];
        difrStandardError: number[];
        difrReversedStandardError: number[];
      };
      discriminant: {
        uniform: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        nonuniform: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        both: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
      };
      purifiedDiscriminant: {
        uniform: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        nonuniform: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        both: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        anchors: number[];
        iterations: number;
        converged: boolean;
      };
      purifiedLiuAgresti: {
        oddsRatio: number[];
        logStandardError: number[];
        z: number[];
        significance: number[];
        difrStandardError: number[];
        difrReversedStandardError: number[];
        anchors: number[];
      };
      discriminantTheta: {
        uniform: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        nonuniform: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
        both: { chiSquare: number[]; significance: number[]; rSquaredChange: number[] };
      };
    };
  };
}
const polytomousTests = readFixture<DifPolytomousTestsFixture>("dif-polytomous-tests.json");

interface DifSibtestFixture {
  guessing: number;
  cases: {
    verbal: {
      input: { fixture: string; case: string; sparse: boolean };
      default: {
        beta: number[];
        standardError: number[];
        chiSquare: number[];
        significance: number[];
        persons: number[];
      };
      uncorrected: {
        beta: number[];
        standardError: number[];
        chiSquare: number[];
        significance: number[];
        persons: number[];
      };
      guessing: {
        beta: number[];
        standardError: number[];
        chiSquare: number[];
        significance: number[];
        persons: number[];
      };
      smallCells: {
        beta: number[];
        standardError: number[];
        chiSquare: number[];
        significance: number[];
        persons: number[];
      };
      largeCells: { beta: null[]; standardError: null[]; chiSquare: null[]; significance: null[]; persons: number[] };
      purified: {
        beta: number[];
        standardError: number[];
        chiSquare: number[];
        significance: number[];
        persons: number[];
        anchors: number[];
        iterations: number;
        converged: boolean;
      };
      purifiedUncorrected: {
        beta: number[];
        standardError: number[];
        chiSquare: number[];
        significance: number[];
        persons: number[];
        anchors: number[];
        iterations: number;
        converged: boolean;
      };
      bundles: {
        name: string;
        items: number[];
        default: { beta: number; standardError: number; chiSquare: number; significance: number; persons: number };
        purified: { beta: number; standardError: number; chiSquare: number; significance: number; persons: number };
      }[];
    };
    simulated: {
      input: { fixture: string; case: string; sparse: boolean };
      default: {
        beta: number[];
        standardError: number[];
        chiSquare: number[];
        significance: number[];
        persons: number[];
      };
      uncorrected: {
        beta: number[];
        standardError: number[];
        chiSquare: number[];
        significance: number[];
        persons: number[];
      };
      guessing: {
        beta: number[];
        standardError: number[];
        chiSquare: number[];
        significance: number[];
        persons: number[];
      };
      smallCells: {
        beta: number[];
        standardError: number[];
        chiSquare: number[];
        significance: number[];
        persons: number[];
      };
      largeCells: {
        beta: number[];
        standardError: number[];
        chiSquare: number[];
        significance: number[];
        persons: number[];
      };
      purified: {
        beta: number[];
        standardError: number[];
        chiSquare: number[];
        significance: number[];
        persons: number[];
        anchors: number[];
        iterations: number;
        converged: boolean;
      };
      purifiedUncorrected: {
        beta: number[];
        standardError: number[];
        chiSquare: number[];
        significance: number[];
        persons: number[];
        anchors: number[];
        iterations: number;
        converged: boolean;
      };
      bundles: {
        name: string;
        items: number[];
        default: { beta: number; standardError: number; chiSquare: number; significance: number; persons: number };
        purified: { beta: number; standardError: number; chiSquare: number; significance: number; persons: number };
      }[];
    };
    bfi: {
      input: { fixture: string; case: string; sparse: boolean };
      default: {
        beta: number[];
        standardError: number[];
        chiSquare: number[];
        significance: number[];
        persons: number[];
      };
      uncorrected: {
        beta: number[];
        standardError: number[];
        chiSquare: number[];
        significance: number[];
        persons: number[];
      };
      guessing: {
        beta: number[];
        standardError: number[];
        chiSquare: number[];
        significance: number[];
        persons: number[];
      };
      smallCells: {
        beta: number[];
        standardError: number[];
        chiSquare: number[];
        significance: number[];
        persons: number[];
      };
      largeCells: {
        beta: number[];
        standardError: number[];
        chiSquare: number[];
        significance: number[];
        persons: number[];
      };
      purified: {
        beta: number[];
        standardError: number[];
        chiSquare: number[];
        significance: number[];
        persons: number[];
        anchors: number[];
        iterations: number;
        converged: boolean;
      };
      purifiedUncorrected: {
        beta: number[];
        standardError: number[];
        chiSquare: number[];
        significance: number[];
        persons: number[];
        anchors: number[];
        iterations: number;
        converged: boolean;
      };
      bundles: {
        name: string;
        items: number[];
        default: { beta: number; standardError: number; chiSquare: number; significance: number; persons: number };
        purified: { beta: number; standardError: number; chiSquare: number; significance: number; persons: number };
      }[];
    };
    graded: {
      input: { fixture: string; case: string; sparse: boolean };
      default: {
        beta: number[];
        standardError: number[];
        chiSquare: number[];
        significance: number[];
        persons: number[];
      };
      uncorrected: {
        beta: number[];
        standardError: number[];
        chiSquare: number[];
        significance: number[];
        persons: number[];
      };
      guessing: {
        beta: number[];
        standardError: number[];
        chiSquare: number[];
        significance: number[];
        persons: number[];
      };
      smallCells: {
        beta: number[];
        standardError: number[];
        chiSquare: number[];
        significance: number[];
        persons: number[];
      };
      largeCells: {
        beta: number[];
        standardError: number[];
        chiSquare: number[];
        significance: number[];
        persons: number[];
      };
      purified: {
        beta: number[];
        standardError: number[];
        chiSquare: number[];
        significance: number[];
        persons: number[];
        anchors: number[];
        iterations: number;
        converged: boolean;
      };
      purifiedUncorrected: {
        beta: number[];
        standardError: number[];
        chiSquare: number[];
        significance: number[];
        persons: number[];
        anchors: number[];
        iterations: number;
        converged: boolean;
      };
      bundles: {
        name: string;
        items: number[];
        default: { beta: number; standardError: number; chiSquare: number; significance: number; persons: number };
        purified: { beta: number; standardError: number; chiSquare: number; significance: number; persons: number };
      }[];
    };
  };
  recipes: {
    verbal: { standardization: number[]; variance: { standardDeviation: number; shrunken: number[] } };
    simulated: { standardization: number[]; variance: { standardDeviation: number; shrunken: number[] } };
  };
}
const sibtestFixture = readFixture<DifSibtestFixture>("dif-sibtest.json");

// Tolerances against R, explained in docs/validation.md.
/** Mantel-Haenszel statistics against difR and base R. */
const MANTEL_HAENSZEL_TOLERANCE = 1e-12;
/** Logistic regression chi-squares and R² changes against difR's glm fits. */
const LOGISTIC_TOLERANCE = 1e-9;
/** Lord's chi-square, relative, against difR on converged ltm calibrations. */
const LORD_RELATIVE_TOLERANCE = 1e-5;
/** The mean-sigma transformation of the focal scale. */
const LORD_TRANSFORMATION_TOLERANCE = 1e-8;
/** Item parameters and the transformation against mirt's calibrations with g held, both run to the optimum. */
const HELD_PARAMETER_TOLERANCE = 1e-8;
/** Mantel's chi-square, the generalized chi-square and the standardized mean difference against base R. */
const MANTEL_TEST_TOLERANCE = 1e-12;
/** Ordinal logistic regression chi-squares, R² changes and slope changes against MASS::polr. */
const ORDINAL_TOLERANCE = 1e-8;
/** lordif's statistics, which it rounds to 4 decimals. */
const LORDIF_TOLERANCE = 5e-5 + 1e-9;
/** SIBTEST's statistics against mirt, the same sums, apart from rounding. */
const SIBTEST_TOLERANCE = 1e-10;
/** The recipes for standardization and DIF variance against difR's difStd and sirt's dif.variance. */
const RECIPE_TOLERANCE = 1e-10;
/** Raju's areas against difR on converged ltm calibrations: the same estimates and closed forms. */
const RAJU_AREA_TOLERANCE = 1e-8;
/** Raju's z and significance, which carry the standard errors of the observed information. */
const RAJU_Z_TOLERANCE = 1e-5;
/** IRT likelihood-ratio chi-squares, significances and log-likelihoods against mirt, both run to 1e-7. */
const LIKELIHOOD_RATIO_TOLERANCE = 1e-5;
/** The focal group's mean and standard deviation in the constrained model against mirt's, which stop at 1e-7. */
const LATENT_TOLERANCE = 1e-4;
/** NCDIF, CDIF and DTF against DFIT: the same sums over the same nodes. */
const FUNCTIONING_TOLERANCE = 1e-12;
/** The logistic discriminant function's chi-squares, significances and R² changes against glm. */
const DISCRIMINANT_TOLERANCE = 1e-8;
/** Liu and Agresti's odds ratio, relative, its standard error, z and significance against difR and base R. */
const LIU_AGRESTI_TOLERANCE = 1e-10;

/** Item 1 everyone answered correctly; items 2 and 3 vary. */
const CONSTANT_ROWS = [
  [1, 0, 1],
  [1, 1, 0],
  [1, 0, 0],
  [1, 1, 1],
  [1, 1, 0],
  [1, 0, 1],
];
const CONSTANT_GROUPS: DifGroup[] = ["reference", "reference", "reference", "focal", "focal", "focal"];

/** Two persons in each of two score groups: 1 holds responses 0 and 1 to item 1, 5 holds 2 and 3. */
const DISJOINT_ROWS = [
  [0, 1],
  [1, 0],
  [2, 3],
  [3, 2],
];

/** Three items, of highest scores 2, 2 and 1, answered by twelve persons. */
const SCALING_ROWS = [
  [0, 0, 0],
  [1, 0, 1],
  [1, 1, 0],
  [2, 1, 1],
  [2, 2, 1],
  [0, 1, 0],
  [0, 1, 0],
  [1, 1, 1],
  [2, 1, 0],
  [2, 2, 1],
  [1, 2, 1],
  [2, 0, 1],
];
const SCALING_GROUPS: DifGroup[] = SCALING_ROWS.map((_, person) => (person < 6 ? "reference" : "focal"));

/** Two items: item 2 scores 0, 1 and 2, so that its score 1 is the only score between the lowest and the highest. */
const ONE_MIDDLE_SCORE_ROWS = [
  [0, 0],
  [0, 1],
  [1, 1],
  [1, 1],
  [1, 2],
  [0, 0],
  [0, 1],
  [0, 1],
  [1, 1],
  [1, 2],
];
const ONE_MIDDLE_SCORE_GROUPS: DifGroup[] = ONE_MIDDLE_SCORE_ROWS.map((_, person) =>
  person < 5 ? "reference" : "focal",
);
const ONE_MIDDLE_SCORE_OPTIONS = { smallCell: 2, correction: false };
/** The statistics of an item or bundle of the persons of ONE_MIDDLE_SCORE_ROWS without a defined SIBTEST, but its id. */
const UNDEFINED_SIBTEST = { beta: null, standardError: null, chiSquare: null, significance: null, persons: 10 };

function groups(focal: readonly number[]): DifGroup[] {
  return focal.map((flag) => (flag === 1 ? "focal" : "reference"));
}

/** Logistic regression tests of a fixture, by test, then item. */
type LogisticFixture = Readonly<
  Record<
    "uniform" | "nonuniform" | "both",
    {
      readonly chiSquare: readonly (number | null)[];
      readonly significance: readonly (number | null)[];
      readonly rSquaredChange: readonly (number | null)[];
    }
  >
>;

/** Mantel-Haenszel statistics of a fixture, by statistic, then item. */
interface MantelHaenszelFixture {
  readonly chiSquare: readonly (number | null)[];
  readonly significance: readonly (number | null)[];
  readonly oddsRatio: readonly (number | null)[];
  readonly delta: readonly (number | null)[];
  readonly deltaStandardError: readonly (number | null)[];
  readonly etsCategory: readonly (string | null)[];
}

/** A purification of a fixture: R's anchors, 1-based, rounds and convergence. */
interface PurificationFixture {
  readonly anchors: readonly number[];
  readonly iterations: number;
  readonly converged: boolean;
}

function expectLogistic(
  result: readonly (LogisticRegressionDif | OrdinalLogisticDif)[],
  expected: LogisticFixture,
  tolerance = LOGISTIC_TOLERANCE,
): void {
  for (const type of ["uniform", "nonuniform", "both"] as const) {
    const tests = expected[type];
    result.forEach((item, i) => {
      const test = item[type];
      expectClose(test.chiSquare, tests.chiSquare[i] ?? null, tolerance, `${type} ${i + 1}`);
      expectClose(test.significance, tests.significance[i] ?? null, tolerance, `${type} p ${i + 1}`);
      expectClose(test.rSquaredChange, tests.rSquaredChange[i] ?? null, tolerance, `${type} R²`);
      expect(test.degreesOfFreedom).toBe(type === "both" ? 2 : 1);
    });
  }
}

function expectMantelHaenszel(result: readonly MantelHaenszel[], expected: MantelHaenszelFixture): void {
  result.forEach((item, i) => {
    for (const name of ["chiSquare", "significance", "oddsRatio", "delta", "deltaStandardError"] as const) {
      expectClose(item[name], expected[name][i] ?? null, MANTEL_HAENSZEL_TOLERANCE, `${name} ${i + 1}`);
    }
    expect(item.etsCategory).toBe(expected.etsCategory[i]);
  });
}

/** Mantel statistics of a fixture, by statistic, then item. */
interface MantelTestFixture {
  readonly chiSquare: readonly (number | null)[];
  readonly significance: readonly (number | null)[];
  readonly generalizedChiSquare: readonly (number | null)[];
  readonly generalizedDegreesOfFreedom: readonly number[];
  readonly generalizedSignificance: readonly (number | null)[];
  readonly standardizedMeanDifference: readonly (number | null)[];
  readonly effectSize: readonly (number | null)[];
  readonly etsCategory: readonly (string | null)[];
}

/** Ordinal logistic regression tests of a fixture: those of a logistic regression, and the slope change by item. */
type OrdinalFixture = LogisticFixture & { readonly slopeChange: readonly (number | null)[] };

/** lordif's rounded p-values and R² changes by test, then item, and its slope change by item. */
type LordifFixture = Readonly<
  Record<
    "uniform" | "nonuniform" | "both",
    { readonly significance: readonly number[]; readonly rSquaredChange: readonly number[] }
  >
> & { readonly slopeChange: readonly number[] };

function expectMantelTest(result: readonly MantelTest[], expected: MantelTestFixture): void {
  result.forEach((item, i) => {
    const label = `item ${i + 1}`;
    expectClose(item.chiSquare, expected.chiSquare[i] ?? null, MANTEL_TEST_TOLERANCE, `chi-square ${label}`);
    expectClose(item.significance, expected.significance[i] ?? null, MANTEL_TEST_TOLERANCE, `p ${label}`);
    const { generalized } = item;
    const chiSquare = expected.generalizedChiSquare[i] ?? null;
    expectClose(generalized.chiSquare, chiSquare, MANTEL_TEST_TOLERANCE, `generalized chi-square ${label}`);
    expectClose(generalized.significance, expected.generalizedSignificance[i] ?? null, MANTEL_TEST_TOLERANCE, label);
    expect(generalized.degreesOfFreedom, label).toBe(expected.generalizedDegreesOfFreedom[i]);
    const smd = expected.standardizedMeanDifference[i] ?? null;
    expectClose(item.standardizedMeanDifference, smd, MANTEL_TEST_TOLERANCE, `SMD ${label}`);
    expectClose(item.effectSize, expected.effectSize[i] ?? null, MANTEL_TEST_TOLERANCE, `effect size ${label}`);
    expect(item.etsCategory, label).toBe(expected.etsCategory[i]);
  });
}

/** Compares ordinal logistic regression with polr's, and with lordif's statistics, which it rounds to 4 decimals. */
function expectOrdinal(
  result: readonly OrdinalLogisticDif[],
  expected: OrdinalFixture,
  lordif: LordifFixture | null = null,
): void {
  expectLogistic(result, expected, ORDINAL_TOLERANCE);
  result.forEach((item, i) => {
    expectClose(item.slopeChange, expected.slopeChange[i] ?? null, ORDINAL_TOLERANCE, `slope change ${i + 1}`);
    if (lordif === null) return;
    for (const type of ["uniform", "nonuniform", "both"] as const) {
      const { significance, rSquaredChange } = lordif[type];
      expectClose(item[type].significance, significance[i] ?? null, LORDIF_TOLERANCE, `lordif ${type} p ${i + 1}`);
      expectClose(item[type].rSquaredChange, rSquaredChange[i] ?? null, LORDIF_TOLERANCE, `lordif ${type} R² ${i + 1}`);
    }
    expectClose(item.slopeChange, lordif.slopeChange[i] ?? null, LORDIF_TOLERANCE, `lordif slope change ${i + 1}`);
  });
}

/** SIBTEST statistics of a fixture, by item, with the persons of each analysis. */
type SibtestFixture = Readonly<
  Record<"beta" | "standardError" | "chiSquare" | "significance", readonly (number | null)[]> &
    Record<"persons", readonly number[]>
>;

/** SIBTEST statistics of a fixture, of one bundle. */
type BundleFixture = Readonly<Record<"beta" | "standardError" | "chiSquare" | "significance", number | null>> & {
  readonly persons: number;
};

function expectSibtest(result: readonly Sibtest[], expected: SibtestFixture): void {
  expect(result).toHaveLength(expected.beta.length);
  result.forEach((item, i) => {
    const label = `item ${i + 1}`;
    expectClose(item.beta, expected.beta[i] ?? null, SIBTEST_TOLERANCE, `beta ${label}`);
    expectClose(item.standardError, expected.standardError[i] ?? null, SIBTEST_TOLERANCE, `SE ${label}`);
    expectClose(item.chiSquare, expected.chiSquare[i] ?? null, SIBTEST_TOLERANCE, `chi-square ${label}`);
    expectClose(item.significance, expected.significance[i] ?? null, SIBTEST_TOLERANCE, `p ${label}`);
    expect(item.persons, `persons ${label}`).toBe(expected.persons[i]);
  });
}

function expectBundle(result: Sibtest | undefined, expected: BundleFixture): void {
  const { beta, standardError, chiSquare, significance, persons } = expected;
  expectSibtest([defined(result)], {
    beta: [beta],
    standardError: [standardError],
    chiSquare: [chiSquare],
    significance: [significance],
    persons: [persons],
  });
}

/** The responses and groups of a case of dif.json or dif-polytomous.json, complete or sparse, by its name. */
function sibtestInput(input: { readonly fixture: string; readonly case: string; readonly sparse: boolean }): {
  matrix: ResponseMatrix;
  personGroups: DifGroup[];
} {
  const cases = input.fixture === "dif" ? fixture.cases : polytomous.cases;
  const found = defined(cases.find((dataset) => dataset.name === input.case));
  return {
    matrix: buildResponseMatrix(input.sparse ? found.sparse.responses : found.responses),
    personGroups: groups(found.focal),
  };
}

/** The values of the recipes on a case of dif.json, by its name. */
function recipeValues(name: string): (typeof sibtestFixture.recipes)["verbal"] {
  return defined(Object.entries(sibtestFixture.recipes).find(([key]) => key === name)?.[1]);
}

/**
 * The data of the guide's samples: 500 persons per group answer 2PL items from the seed 2026, some items shifted for
 * the focal group, whose theta is lower by `focalShift`.
 */
function guideSample(
  itemCount: number,
  shifts: Readonly<Record<number, number>>,
  focalShift: number,
): { matrix: ResponseMatrix; personGroups: DifGroup[] } {
  const random = createRandom(2026);
  const items = simulateItems(twoParameterLogistic, itemCount, random);
  const shifted = items.map((item, j) => ({
    ...item,
    parameters: { ...item.parameters, b: item.parameters.b + (shifts[j] ?? 0) },
  }));
  const reference = simulateResponses(items, simulateThetas(500, random), random);
  const focalThetas = simulateThetas(500, random).map((theta) => theta - focalShift);
  const focal = simulateResponses(shifted, focalThetas, random);
  const matrix = buildResponseMatrix([...reference.responses, ...focal.responses]);
  return { matrix, personGroups: matrix.personIds.map((_, person) => (person < 500 ? "reference" : "focal")) };
}

/** Rounds to the significant digits a guide shows. */
function shown(value: number | null | undefined, digits: number): number {
  return Number(defined(value).toPrecision(digits));
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

function expectPurification(actual: DifPurification | null, expected: PurificationFixture): void {
  expect(actual).toEqual({
    anchors: expected.anchors.map((index) => `item${index}`),
    iterations: expected.iterations,
    converged: expected.converged,
  });
}

describe("logisticRegressionDif", () => {
  for (const dataset of fixture.cases) {
    it(`matches difR's difLogistic on ${dataset.name} data`, () => {
      const result = logisticRegressionDif(buildResponseMatrix(dataset.responses), groups(dataset.focal));
      expect(result.purification).toBeNull();
      expectLogistic(result.items, dataset.logistic);
    });

    it(`matches difR's purified difLogistic on ${dataset.name} data`, () => {
      const expected = dataset.purifiedLogistic;
      const result = logisticRegressionDif(buildResponseMatrix(dataset.responses), groups(dataset.focal), {
        purify: true,
      });
      expectLogistic(result.items, expected);
      expectPurification(result.purification, expected);
    });

    it(`matches base R's rule for sparse data, purified and not, on ${dataset.name} data`, () => {
      const matrix = buildResponseMatrix(dataset.sparse.responses);
      expectLogistic(logisticRegressionDif(matrix, groups(dataset.focal)).items, dataset.sparse.logistic);
      const purified = logisticRegressionDif(matrix, groups(dataset.focal), { purify: true });
      expectLogistic(purified.items, dataset.sparse.purifiedLogistic);
      expectPurification(purified.purification, dataset.sparse.purifiedLogistic);
    });
  }

  it("gives null tests where every person has the same matching score, which leaves no score to match on", () => {
    // Every person answers one of the two items, so every matching score is 1, while item 1 is easier for the
    // reference group: without the rule, each regression stays at its start and the tests are 0.
    const rows = [
      [1, 0],
      [1, 0],
      [1, 0],
      [0, 1],
      [0, 1],
      [0, 1],
      [0, 1],
      [1, 0],
      [1, 0],
      [0, 1],
    ];
    const rowGroups: DifGroup[] = rows.map((_, person) => (person < 4 ? "reference" : "focal"));
    const { items } = logisticRegressionDif(buildResponseMatrix(rows), rowGroups);
    for (const item of items) {
      expect([item.uniform.chiSquare, item.nonuniform.chiSquare, item.both.chiSquare]).toEqual([null, null, null]);
    }
  });

  it("gives null tests whose model has collinear terms, and the others", () => {
    // Every focal person has the matching score 1, so S G equals G: the interaction cannot be tested.
    const rows = [
      [0, 0, 0],
      [1, 0, 0],
      [1, 1, 0],
      [1, 1, 1],
      [0, 1, 1],
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
      [1, 0, 0],
      [0, 1, 0],
    ];
    const rowGroups: DifGroup[] = rows.map((_, person) => (person < 5 ? "reference" : "focal"));
    const item = defined(logisticRegressionDif(buildResponseMatrix(rows), rowGroups).items[0]);
    expect(item.uniform.chiSquare).not.toBeNull();
    expect([item.nonuniform.chiSquare, item.both.chiSquare]).toEqual([null, null]);
  });

  it("gives null tests for an item every person answered alike or with persons of one group", () => {
    const { items } = logisticRegressionDif(buildResponseMatrix(CONSTANT_ROWS), CONSTANT_GROUPS);
    expect(items[0]?.both).toEqual({ chiSquare: null, degreesOfFreedom: 2, significance: null, rSquaredChange: null });
    expect(items[1]?.uniform.chiSquare).not.toBeNull();
    const rows = CONSTANT_ROWS.map((row, person) => [person < 3 ? (row[0] ?? null) : null, ...row.slice(1)]);
    expect(logisticRegressionDif(buildResponseMatrix(rows), CONSTANT_GROUPS).items[0]?.uniform.chiSquare).toBeNull();
  });
});

/** The calibration of difR's ltm: 21 Gauss-Hermite points, run to convergence. */
const LORD_CALIBRATION = {
  model: twoParameterLogistic,
  quadrature: { method: "gauss-hermite", points: 21 },
  tolerance: 1e-10,
  maxIterations: 10_000,
} as const;

describe("lordChiSquare", () => {
  for (const dataset of fixture.cases) {
    it(`matches difR's difLord on ${dataset.name} data`, () => {
      const result = lordChiSquare(buildResponseMatrix(dataset.responses), groups(dataset.focal), LORD_CALIBRATION);
      const { lord } = dataset;
      expect(result.purification).toBeNull();
      result.items.forEach((item, i) => {
        const expected = lord.chiSquare[i] ?? 0;
        expectClose(item.chiSquare, expected, LORD_RELATIVE_TOLERANCE * expected, `chi-square ${i + 1}`);
        expect(item.degreesOfFreedom).toBe(2);
      });
      expectClose(result.transformation.slope, lord.transformation.slope, LORD_TRANSFORMATION_TOLERANCE);
      expectClose(result.transformation.intercept, lord.transformation.intercept, LORD_TRANSFORMATION_TOLERANCE);
      expect(result.reference.converged && result.focal.converged).toBe(true);
    });
  }

  for (const expected of lordPurification.cases) {
    it(`matches difR's purified difLord on ${expected.name} data`, () => {
      const dataset = defined(fixture.cases.find((reference) => reference.name === expected.name));
      const result = lordChiSquare(buildResponseMatrix(dataset.responses), groups(dataset.focal), {
        ...LORD_CALIBRATION,
        purify: true,
      });
      result.items.forEach((item, i) => {
        const chiSquare = expected.chiSquare[i] ?? 0;
        expectClose(item.chiSquare, chiSquare, LORD_RELATIVE_TOLERANCE * chiSquare, `chi-square ${i + 1}`);
      });
      expectClose(result.transformation.slope, expected.transformation.slope, LORD_TRANSFORMATION_TOLERANCE);
      expectClose(result.transformation.intercept, expected.transformation.intercept, LORD_TRANSFORMATION_TOLERANCE);
      expectPurification(result.purification, expected);
    });
  }

  it("matches difR's difLord with c on 3PL items whose lower asymptotes are held in both groups", () => {
    const result = lordChiSquare(heldCase.matrix, heldCase.groups, heldCase.options);
    const { lord } = heldFixture;
    result.items.forEach((item, i) => {
      const expected = lord.chiSquare[i] ?? 0;
      expectClose(item.chiSquare, expected, LORD_RELATIVE_TOLERANCE * expected, `chi-square ${i + 1}`);
      expectClose(item.significance, lord.significance[i] ?? null, LORD_RELATIVE_TOLERANCE, `significance ${i + 1}`);
      expect(item.degreesOfFreedom).toBe(lord.degreesOfFreedom);
    });
    for (const [calibration, expected] of [
      [result.reference, lord.reference],
      [result.focal, lord.focal],
    ] as const) {
      calibration.items.forEach((item, i) => {
        expectClose(item.parameters.a, expected.a[i] ?? 0, HELD_PARAMETER_TOLERANCE, `a of ${item.id}`);
        expectClose(item.parameters.b, expected.b[i] ?? 0, HELD_PARAMETER_TOLERANCE, `b of ${item.id}`);
        expect(item.parameters.c).toBe(heldFixture.fixedParameters[item.id]?.c);
      });
    }
    expectClose(result.transformation.slope, lord.transformation.slope, HELD_PARAMETER_TOLERANCE);
    expectClose(result.transformation.intercept, lord.transformation.intercept, HELD_PARAMETER_TOLERANCE);
  });

  it("refuses held parameters that the link moves, that a model lacks, or of other ids", () => {
    const matrix = buildResponseMatrix(CONSTANT_ROWS);
    const options = { model: threeParameterLogistic };
    expect(() => lordChiSquare(matrix, CONSTANT_GROUPS, { ...options, fixedParameters: { item2: { a: 1 } } })).toThrow(
      'Lord\'s chi-square holds the asymptotes c and d alone, which the link leaves unchanged; the option fixedParameters holds "a" of item "item2".',
    );
    expect(() =>
      rajuAreaDif(matrix, CONSTANT_GROUPS, { ...options, fixedParameters: { item2: { b: 0, c: 0.2 } } }),
    ).toThrow(
      'Raju\'s area holds the asymptotes c and d alone, which the link leaves unchanged; the option fixedParameters holds "b" of item "item2".',
    );
    expect(() =>
      lordChiSquare(matrix, CONSTANT_GROUPS, { model: twoParameterLogistic, fixedParameters: { item2: { c: 0.2 } } }),
    ).toThrow('The option fixedParameters holds "c" of item "item2", whose 2PL model has a, b.');
    expect(() => lordChiSquare(matrix, CONSTANT_GROUPS, { ...options, fixedParameters: { q1: { c: 0.2 } } })).toThrow(
      'The option fixedParameters holds parameters of "q1", which is not an item id.',
    );
    expect(() => lordChiSquare(matrix, CONSTANT_GROUPS, { ...options, fixedParameters: JSON.parse("null") })).toThrow(
      "The option fixedParameters must be an object of values by item id and parameter name, found null.",
    );
  });

  it("rejects models other than the 2PL, 3PL and 4PL, and a group without persons", () => {
    const matrix = buildResponseMatrix(CONSTANT_ROWS);
    const custom = { ...twoParameterLogistic, name: "custom 2PL" };
    expect(() => lordChiSquare(matrix, CONSTANT_GROUPS, { model: oneParameterLogistic })).toThrow(DataError);
    expect(() => lordChiSquare(matrix, CONSTANT_GROUPS, { model: custom })).toThrow(DataError);
    const references: DifGroup[] = CONSTANT_GROUPS.map(() => "reference");
    expect(() => lordChiSquare(matrix, references, { model: twoParameterLogistic })).toThrow(DataError);
  });

  it("takes no person weights, which align with the whole matrix while each group's calibration takes its own", () => {
    expectTypeOf<LinkedDifOptions>().not.toHaveProperty("personWeights");
  });
});

/** The cases of dif.json and dif-polytomous.json that the fixtures of the other units key by name. */
interface FixtureInput {
  readonly fixture: string;
  readonly case: string;
  readonly sparse: boolean;
}

/** The responses and groups of a case of dif.json or dif-polytomous.json. */
function fixtureCase(input: FixtureInput): { matrix: ResponseMatrix; groups: DifGroup[] } {
  const cases: readonly {
    name: string;
    responses: (number | null)[][];
    focal: number[];
    sparse: { responses: (number | null)[][] };
  }[] = input.fixture === "dif" ? fixture.cases : polytomous.cases;
  const dataset = defined(cases.find((candidate) => candidate.name === input.case));
  return {
    matrix: buildResponseMatrix(input.sparse ? dataset.sparse.responses : dataset.responses),
    groups: groups(dataset.focal),
  };
}

/** Raju's statistics of a fixture, by statistic, then item. */
type RajuFixture = Readonly<
  Record<"signedArea" | "unsignedArea" | "signedZ" | "unsignedZ" | "significance", readonly (number | null)[]>
>;

function expectRaju(actual: readonly RajuArea[], expected: RajuFixture): void {
  actual.forEach((item, i) => {
    expectClose(item.signedArea, expected.signedArea[i] ?? null, RAJU_AREA_TOLERANCE, `signed area ${i + 1}`);
    expectClose(item.unsignedArea, expected.unsignedArea[i] ?? null, RAJU_AREA_TOLERANCE, `unsigned area ${i + 1}`);
    expectClose(item.signedZ, expected.signedZ[i] ?? null, RAJU_Z_TOLERANCE, `signed z ${i + 1}`);
    expectClose(item.unsignedZ, expected.unsignedZ[i] ?? null, RAJU_Z_TOLERANCE, `unsigned z ${i + 1}`);
    expectClose(item.significance, expected.significance[i] ?? null, RAJU_Z_TOLERANCE, `significance ${i + 1}`);
  });
}

describe("rajuAreaDif", () => {
  for (const [name, expected] of Object.entries(irtFixture.raju)) {
    const { matrix, groups: caseGroups } = fixtureCase(expected.input);

    it(`matches difR's difRaju on ${name} data`, () => {
      const result = rajuAreaDif(matrix, caseGroups, LORD_CALIBRATION);
      expect(result.purification).toBeNull();
      expectRaju(result.items, expected.default);
    });

    it(`matches difR's purified difRaju on ${name} data`, () => {
      const result = rajuAreaDif(matrix, caseGroups, { ...LORD_CALIBRATION, purify: true });
      expectRaju(result.items, expected.purified);
      expectPurification(result.purification, expected.purified);
    });
  }

  it("gives the area between the curves where difR's unsigned area is negative, and a significance below 1", () => {
    const expected = irtFixture.raju.verbal.default;
    const below = expected.difrUnsignedArea.flatMap((area, i) => (area < 0 ? [i] : []));
    expect(below.length).toBeGreaterThan(0);
    const { matrix, groups: caseGroups } = fixtureCase(irtFixture.raju.verbal.input);
    const { items } = rajuAreaDif(matrix, caseGroups, LORD_CALIBRATION);
    for (const i of below) {
      expectClose(items[i]?.unsignedArea ?? null, -(expected.difrUnsignedArea[i] ?? 0), RAJU_AREA_TOLERANCE);
      expect(expected.difrSignificance[i]).toBeGreaterThan(1);
      expect(items[i]?.significance).toBeLessThan(1);
    }
  });

  it("matches difR's difRaju with c on 3PL items whose lower asymptotes are held in both groups", () => {
    const result = rajuAreaDif(heldCase.matrix, heldCase.groups, heldCase.options);
    expectRaju(result.items, heldFixture.raju);
  });

  it("gives 4PL items whose asymptotes are held at c and 1 the areas difR gives the 3PL items", () => {
    const fixedParameters = Object.fromEntries(
      Object.entries(heldFixture.fixedParameters).map(([id, { c }]) => [id, { c, d: 1 }]),
    );
    const result = rajuAreaDif(heldCase.matrix, heldCase.groups, {
      ...heldCase.options,
      model: fourParameterLogistic,
      fixedParameters,
    });
    expectRaju(result.items, heldFixture.raju);
  });

  it("rejects models other than the 2PL, 3PL and 4PL, an item with an estimated asymptote, and a group without persons", () => {
    const matrix = buildResponseMatrix(CONSTANT_ROWS);
    for (const model of [oneParameterLogistic, threeParameterLogistic, { ...twoParameterLogistic, name: "copy" }]) {
      expect(() => rajuAreaDif(matrix, CONSTANT_GROUPS, { model })).toThrow(DataError);
    }
    expect(() =>
      rajuAreaDif(matrix, CONSTANT_GROUPS, {
        model: [twoParameterLogistic, threeParameterLogistic, twoParameterLogistic],
      }),
    ).toThrow(
      'Raju\'s area needs items that estimate a and b alone, as the 2PL does, or the 3PL and 4PL with their asymptotes in the option fixedParameters; item "item2" of the 3PL model estimates a, b, c.',
    );
    expect(() =>
      rajuAreaDif(matrix, CONSTANT_GROUPS, { model: fourParameterLogistic, fixedParameters: { item1: { c: 0.2 } } }),
    ).toThrow('item "item1" of the 4PL model estimates a, b, d.');
    expect(() => rajuAreaDif(matrix, CONSTANT_GROUPS, { model: oneParameterLogistic })).toThrow(
      'Raju\'s area needs the 2PL, 3PL or 4PL model, found the "1PL" model.',
    );
    const focal: DifGroup[] = CONSTANT_GROUPS.map(() => "focal");
    expect(() => rajuAreaDif(matrix, focal, { model: twoParameterLogistic })).toThrow(
      "Raju's area needs persons in the reference group.",
    );
  });

  it("rejects options of other types", () => {
    const matrix = buildResponseMatrix(CONSTANT_ROWS);
    expect(() => rajuAreaDif(matrix, CONSTANT_GROUPS, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => rajuAreaDif(matrix, CONSTANT_GROUPS, JSON.parse('{"verbose":1}'))).toThrow(
      'There is no option "verbose"',
    );
    expect(() => rajuAreaDif(matrix, CONSTANT_GROUPS, { model: JSON.parse('"2PL"') })).toThrow(
      'The option model must be the 2PL, 3PL or 4PL model, found "2PL".',
    );
    expect(() =>
      rajuAreaDif(matrix, CONSTANT_GROUPS, { model: twoParameterLogistic, purify: JSON.parse('"yes"') }),
    ).toThrow('The option purify must be a boolean, found "yes".');
    expect(() => rajuAreaDif(matrix, JSON.parse("null"), { model: twoParameterLogistic })).toThrow(
      "The groups must be an array, found null.",
    );
  });
});

/** The models of the likelihood-ratio fixture by its names for them. */
const LR_MODELS: Readonly<Record<string, IrtModel<unknown>>> = {
  "2pl": twoParameterLogistic,
  grm: gradedResponse,
  gpcm: generalizedPartialCredit,
};

/** The likelihood-ratio tests of a fixture, by statistic, then item. */
type LikelihoodRatioFixture = Readonly<
  Record<"chiSquare" | "degreesOfFreedom" | "significance", readonly (number | null)[]>
>;

function expectLikelihoodRatio(actual: readonly IrtDifTest[], expected: LikelihoodRatioFixture): void {
  actual.forEach((item, i) => {
    expectClose(item.chiSquare, expected.chiSquare[i] ?? null, LIKELIHOOD_RATIO_TOLERANCE, `chi-square ${i + 1}`);
    expect(item.degreesOfFreedom).toBe(expected.degreesOfFreedom[i]);
    expectClose(item.significance, expected.significance[i] ?? null, LIKELIHOOD_RATIO_TOLERANCE, `p ${i + 1}`);
  });
}

describe("irtLikelihoodRatioDif", () => {
  const options = { tolerance: irtFixture.tolerance, maxIterations: 20_000 };

  for (const [name, expected] of Object.entries(irtFixture.likelihoodRatio)) {
    it(`matches mirt's multiple-group likelihood ratios on ${name}`, () => {
      const { matrix, groups: caseGroups } = fixtureCase(expected.input);
      const model = defined(LR_MODELS[expected.model]);
      const result = irtLikelihoodRatioDif(matrix, caseGroups, { ...options, model });
      expect(result.purification).toBeNull();
      expectLikelihoodRatio(result.items, expected.default);
      const { constrained } = result;
      expect(constrained.converged).toBe(true);
      expectClose(constrained.logLikelihood, expected.logLikelihood, LIKELIHOOD_RATIO_TOLERANCE);
      const focal = constrained.groups.find((group) => group.group === "focal");
      expectClose(focal?.mean ?? null, expected.focal.mean, LATENT_TOLERANCE);
      expectClose(focal?.sd ?? null, expected.focal.sd, LATENT_TOLERANCE);
      expect(constrained.items.map((item) => item.id)).toEqual(matrix.itemIds);
    });
  }

  for (const name of ["verbal, 2PL", "bfi, graded"] as const) {
    it(`matches the purification of mirt's likelihood ratios on ${name}`, () => {
      const expected = irtFixture.likelihoodRatio[name];
      const { matrix, groups: caseGroups } = fixtureCase(expected.input);
      const model = defined(LR_MODELS[expected.model]);
      const result = irtLikelihoodRatioDif(matrix, caseGroups, { ...options, model, purify: true });
      const pure = defined(expected.purified);
      expectLikelihoodRatio(result.items, pure);
      expectPurification(result.purification, pure);
      expectClose(result.constrained.logLikelihood, pure.logLikelihood, LIKELIHOOD_RATIO_TOLERANCE);
      const free = matrix.itemIds.filter((_, i) => !pure.anchors.includes(i + 1));
      expect(result.constrained.items.map((item) => item.id)).toEqual(
        matrix.itemIds.flatMap((id) => (free.includes(id) ? [`${id}:reference`, `${id}:focal`] : [id])),
      );
    });
  }

  it("matches mirt's multiple-group likelihood ratios with the lower asymptotes of 3PL items held in both groups", () => {
    const { likelihoodRatio } = heldFixture;
    const result = irtLikelihoodRatioDif(heldCase.matrix, heldCase.groups, {
      ...heldCase.options,
      tolerance: heldFixture.likelihoodRatioTolerance,
    });
    expectLikelihoodRatio(result.items, likelihoodRatio);
    const { constrained } = result;
    expect(constrained.converged).toBe(true);
    expectClose(constrained.logLikelihood, likelihoodRatio.logLikelihood, LIKELIHOOD_RATIO_TOLERANCE);
    const focal = constrained.groups.find((group) => group.group === "focal");
    expectClose(focal?.mean ?? null, likelihoodRatio.focal.mean, LATENT_TOLERANCE);
    expectClose(focal?.sd ?? null, likelihoodRatio.focal.sd, LATENT_TOLERANCE);
    for (const item of constrained.items) expect(item.parameters.c).toBe(heldFixture.fixedParameters[item.id]?.c);
  });

  it("leaves a held parameter out of the degrees of freedom, and refuses held parameters of other ids or models", () => {
    const { matrix, groups: caseGroups } = fixtureCase(irtFixture.likelihoodRatio["verbal, 2PL"].input);
    const subset = buildResponseMatrix(matrix.responses.map((row) => row.slice(0, 6)));
    const { items, constrained } = irtLikelihoodRatioDif(subset, caseGroups, {
      model: twoParameterLogistic,
      fixedParameters: { item2: { a: 1 } },
    });
    expect(items.map((item) => item.degreesOfFreedom)).toEqual([2, 1, 2, 2, 2, 2]);
    expect(constrained.items[1]?.parameters.a).toBe(1);
    expect(() =>
      irtLikelihoodRatioDif(subset, caseGroups, {
        model: twoParameterLogistic,
        fixedParameters: { "item1:focal": { a: 1 } },
      }),
    ).toThrow('The option fixedParameters holds parameters of "item1:focal", which is not an item id.');
    expect(() =>
      irtLikelihoodRatioDif(subset, caseGroups, { model: gradedResponse, fixedParameters: { item1: { c: 0.2 } } }),
    ).toThrow('The option fixedParameters holds "c" of item "item1", whose GRM model has a, b1.');
  });

  it("gives the values of the guide's sample, and the differential functioning of its free items", () => {
    const { matrix, personGroups } = guideSample(10, { 2: 1 }, 0);
    const dif = irtLikelihoodRatioDif(matrix, personGroups, { model: twoParameterLogistic, purify: true });
    const item3 = defined(dif.items[2]);
    expect([shown(item3.chiSquare, 3), item3.degreesOfFreedom, shown(item3.significance, 2)]).toEqual([34, 2, 4.2e-8]);
    expect(dif.purification).toEqual({
      anchors: matrix.itemIds.filter((id) => id !== "item3" && id !== "item4"),
      iterations: 1,
      converged: true,
    });
    expect(dif.constrained.items.slice(0, 5).map((item) => item.id)).toEqual([
      "item1",
      "item2",
      "item3:reference",
      "item3:focal",
      "item4:reference",
    ]);
    const freeOf = (group: string): TestItem[] =>
      dif.constrained.items
        .filter((item) => !item.id.includes(":") || item.id.endsWith(`:${group}`))
        .map((item) => ({ ...item, id: item.id.replace(`:${group}`, "") }));
    const population = defined(dif.constrained.groups.find((distribution) => distribution.group === "focal"));
    const functioning = differentialFunctioning(freeOf("reference"), freeOf("focal"), population);
    const functioned = defined(functioning.items[2]);
    expect([shown(functioned.ncdif, 3), shown(functioned.cdif, 3), shown(functioning.dtf, 3)]).toEqual([
      0.0276, 0.0219, 0.0263,
    ]);
    const inPassage = (item: TestItem): boolean => ["item3", "item4"].includes(item.id);
    const passage = differentialFunctioning(
      freeOf("reference").filter(inPassage),
      freeOf("focal").filter(inPassage),
      population,
    );
    expect(shown(passage.dtf, 3)).toBe(0.0263);
  });

  it("gives the same tests for one model and a list of it", () => {
    const { matrix, groups: caseGroups } = fixtureCase(irtFixture.likelihoodRatio["verbal, 2PL"].input);
    const subset = buildResponseMatrix(matrix.responses.map((row) => row.slice(0, 6)));
    const single = irtLikelihoodRatioDif(subset, caseGroups, { model: twoParameterLogistic });
    const list = irtLikelihoodRatioDif(subset, caseGroups, { model: subset.itemIds.map(() => twoParameterLogistic) });
    expect(list).toEqual(single);
  });

  it("gives a null test for an item a group lacks a category of, and goes on with the others", () => {
    const { matrix, groups: caseGroups } = fixtureCase(irtFixture.likelihoodRatio["bfi, graded"].input);
    const highest = Math.max(...matrix.responses.map((row) => row[0] ?? 0));
    const rows = matrix.responses.map((row, person) =>
      caseGroups[person] === "focal" && row[0] === highest ? [highest - 1, ...row.slice(1)] : row,
    );
    const { items } = irtLikelihoodRatioDif(buildResponseMatrix(rows), caseGroups, { model: gradedResponse });
    expect(items[0]).toEqual({ id: "item1", chiSquare: null, degreesOfFreedom: highest + 1, significance: null });
    expect(items.slice(1).every((item) => item.chiSquare !== null)).toBe(true);
  });

  it("gives null tests where the fits do not converge, and reports it in the constrained calibration", () => {
    const { matrix, groups: caseGroups } = fixtureCase(irtFixture.likelihoodRatio["verbal, 2PL"].input);
    const { items, constrained } = irtLikelihoodRatioDif(matrix, caseGroups, {
      model: twoParameterLogistic,
      maxIterations: 1,
    });
    expect(constrained.converged).toBe(false);
    expect(items.every((item) => item.chiSquare === null && item.degreesOfFreedom === 2)).toBe(true);
  });

  it("rejects fewer than two items, the Gauss-Hermite rule and options of other types", () => {
    const matrix = buildResponseMatrix(CONSTANT_ROWS);
    const one = buildResponseMatrix(CONSTANT_ROWS.map((row) => row.slice(1, 2)));
    expect(() => irtLikelihoodRatioDif(one, CONSTANT_GROUPS, { model: twoParameterLogistic })).toThrow(
      "IRT likelihood-ratio DIF needs at least 2 items, found 1.",
    );
    expect(() =>
      irtLikelihoodRatioDif(matrix, CONSTANT_GROUPS, {
        model: twoParameterLogistic,
        quadrature: { method: "gauss-hermite" },
      }),
    ).toThrow(DataError);
    expect(() => irtLikelihoodRatioDif(matrix, CONSTANT_GROUPS, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => irtLikelihoodRatioDif(matrix, CONSTANT_GROUPS, JSON.parse('{"verbose":1}'))).toThrow(
      'There is no option "verbose"',
    );
    expect(() => irtLikelihoodRatioDif(matrix, CONSTANT_GROUPS, { model: JSON.parse('"2PL"') })).toThrow(DataError);
    expect(() =>
      irtLikelihoodRatioDif(matrix, CONSTANT_GROUPS, { model: twoParameterLogistic, purify: JSON.parse("1") }),
    ).toThrow("The option purify must be a boolean, found 1.");
    expect(() => irtLikelihoodRatioDif(matrix, CONSTANT_GROUPS.slice(1), { model: twoParameterLogistic })).toThrow(
      DataError,
    );
  });

  it("takes no person weights, whose likelihood ratio would not be chi-square", () => {
    expectTypeOf<IrtDifOptions<LogisticItem>>().not.toHaveProperty("personWeights");
  });
});

describe("differentialFunctioning", () => {
  const { population } = irtFixture;

  for (const [name, expected] of Object.entries(irtFixture.functioning)) {
    const model: IrtModel<unknown> = expected.model === "2pl" ? twoParameterLogistic : gradedResponse;
    const items = (parameters: readonly unknown[]): TestItem[] =>
      parameters.map((each, i) => ({ id: `item${i + 1}`, model, parameters: each }));

    it(`matches DFIT's Ncdif, Cdif and Dtf on ${name} items`, () => {
      const result = differentialFunctioning(items(expected.reference), items(expected.focal), population);
      result.items.forEach((item, i) => {
        expectClose(item.ncdif, expected.ncdif[i] ?? 0, FUNCTIONING_TOLERANCE, `NCDIF ${i + 1}`);
        expectClose(item.cdif, expected.cdif[i] ?? 0, FUNCTIONING_TOLERANCE, `CDIF ${i + 1}`);
      });
      expectClose(result.dtf, expected.dtf, FUNCTIONING_TOLERANCE);
    });
  }

  it("gives a bundle's functioning over its items alone, its DTF the sum of their CDIF", () => {
    const expected = irtFixture.functioning["2PL"];
    // Items 1 and 3: item 3 is equal in both groups, so the bundle's DBF is item 1's NCDIF.
    const items = (parameters: readonly LogisticItem[]): TestItem<LogisticItem>[] =>
      [0, 2].map((i) => ({ id: `item${i + 1}`, model: twoParameterLogistic, parameters: defined(parameters[i]) }));
    const bundle = differentialFunctioning(items(expected.reference), items(expected.focal), population);
    expectClose(
      bundle.dtf,
      bundle.items.reduce((sum, item) => sum + item.cdif, 0),
      1e-15,
    );
    expectClose(bundle.dtf, expected.ncdif[0] ?? 0, FUNCTIONING_TOLERANCE);
    // In the whole test, item 2's DIF cancels part of item 1's.
    expect(expected.dtf).toBeLessThan(bundle.dtf);
  });

  it("rejects lists that differ in their items or order, and a population that is not a distribution", () => {
    const item = (id: string, b: number): TestItem<LogisticItem> => ({
      id,
      model: twoParameterLogistic,
      parameters: { a: 1, b, c: 0, d: 1 },
    });
    const reference = [item("x", 0), item("y", 1)];
    const population = { nodes: [0], weights: [1] };
    expect(() => differentialFunctioning(reference, [item("x", 0)], population)).toThrow(
      "Found 1 focal items for 2 reference items.",
    );
    expect(() => differentialFunctioning(reference, [item("y", 1), item("x", 0)], population)).toThrow(
      'The focal items must be the reference items in the same order; item 1 is "x" in the reference list and "y" in the focal list.',
    );
    expect(() => differentialFunctioning(reference, reference, { nodes: [0], weights: [0.5] })).toThrow(
      "The weights of the population must sum to 1, found 0.5.",
    );
    expect(() => differentialFunctioning(reference, reference, JSON.parse("null"))).toThrow(DataError);
    expect(() => differentialFunctioning(reference, reference, JSON.parse("{}"))).toThrow(DataError);
    expect(() => differentialFunctioning(JSON.parse('"x"'), reference, population)).toThrow(DataError);
    expect(differentialFunctioning([], [], population)).toEqual({ items: [], dtf: 0 });
  });
});

describe("purified", () => {
  /** Statistics that flag the first `limit` items whatever the anchors are. */
  const flaggedBelow = (limit: number) => (anchors: Uint8Array) =>
    Array.from(anchors, (_, item) => ({ item, flag: item < limit }));

  it("stops before fewer anchors than the statistics accept are left", () => {
    const ids = ["a", "b", "c", "d"];
    const run = (minimumAnchors: number) =>
      purified(ids, true, flaggedBelow(3), (entry) => entry.flag, minimumAnchors).purification;
    expect(run(1)).toEqual({ anchors: ["d"], iterations: 1, converged: true });
    expect(run(2)).toEqual({ anchors: ["a", "b", "c", "d"], iterations: 0, converged: false });
  });
});

describe("mantelHaenszel", () => {
  for (const dataset of fixture.cases) {
    for (const expected of dataset.mantelHaenszel) {
      const correction = expected.continuityCorrection ? "with" : "without";
      it(`matches difR's difMH ${correction} continuity correction on ${dataset.name} data`, () => {
        const result = mantelHaenszel(buildResponseMatrix(dataset.responses), groups(dataset.focal), {
          continuityCorrection: expected.continuityCorrection,
        });
        expectMantelHaenszel(result.items, expected);
      });
    }

    it(`matches difR's purified difMH on ${dataset.name} data`, () => {
      const expected = dataset.purifiedMantelHaenszel;
      const result = mantelHaenszel(buildResponseMatrix(dataset.responses), groups(dataset.focal), { purify: true });
      expectMantelHaenszel(result.items, expected);
      expectPurification(result.purification, expected);
    });

    it(`matches base R's rule for sparse data, purified and not, on ${dataset.name} data`, () => {
      const matrix = buildResponseMatrix(dataset.sparse.responses);
      expectMantelHaenszel(mantelHaenszel(matrix, groups(dataset.focal)).items, dataset.sparse.mantelHaenszel);
      const purified = mantelHaenszel(matrix, groups(dataset.focal), { purify: true });
      expectMantelHaenszel(purified.items, dataset.sparse.purifiedMantelHaenszel);
      expectPurification(purified.purification, dataset.sparse.purifiedMantelHaenszel);
    });
  }

  it("gives null statistics for an item every person answered alike", () => {
    const result = mantelHaenszel(buildResponseMatrix(CONSTANT_ROWS), CONSTANT_GROUPS).items;
    expect(result[0]).toEqual({
      id: "item1",
      chiSquare: null,
      significance: null,
      oddsRatio: null,
      delta: null,
      deltaStandardError: null,
      etsCategory: null,
    });
  });

  it("analyzes each item on the persons who answered it, matched on their scaled score", () => {
    const base = mantelHaenszel(buildResponseMatrix(CONSTANT_ROWS), CONSTANT_GROUPS).items;
    // A person who did not answer item 1, or answered no other item, changes nothing for item 1.
    const rows = [...CONSTANT_ROWS, [null, 1, 0], [1, null, null]];
    const sparse = mantelHaenszel(buildResponseMatrix(rows), [...CONSTANT_GROUPS, "focal", "reference"]).items;
    expect(sparse[0]).toEqual(base[0]);
    // [1, 1, null] answered 1 of 1 other item correct, which scales to 2 of 2: score 3, not 2. Score 2 then holds
    // [0, 1, 1] and [1, 1, 0] alone: A = 0, B = 1, C = 1, D = 0, so Σ (A - E(A)) = -0.5 and Var(A) = 0.25.
    const scaled = [
      [1, 1, null],
      [1, 1, 1],
      [0, 1, 1],
      [1, 1, 0],
    ];
    const result = mantelHaenszel(buildResponseMatrix(scaled), ["reference", "focal", "reference", "focal"], {
      continuityCorrection: false,
    });
    expect(result.items[0]?.chiSquare).toBeCloseTo(1, 15);
  });

  it("purifies the matching criterion until the flags settle, and reports no rounds without flags", () => {
    const { purification } = mantelHaenszel(buildResponseMatrix(CONSTANT_ROWS), CONSTANT_GROUPS, { purify: true });
    expect(purification).toEqual({ anchors: ["item1", "item2", "item3"], iterations: 0, converged: true });
  });

  it("rejects invalid groups, polytomous items and a group without responses", () => {
    const matrix = buildResponseMatrix(CONSTANT_ROWS);
    expect(() => mantelHaenszel(matrix, CONSTANT_GROUPS.slice(1))).toThrow(DataError);
    // Groups from untyped data, as JavaScript callers may pass them.
    const invalid: DifGroup[] = JSON.parse(JSON.stringify(["other", ...CONSTANT_GROUPS.slice(1)]));
    expect(() => mantelHaenszel(matrix, invalid)).toThrow(DataError);
    expect(() =>
      mantelHaenszel(
        buildResponseMatrix([
          [2, 1],
          [0, 1],
        ]),
        ["reference", "focal"],
      ),
    ).toThrow(DataError);
    expect(() =>
      mantelHaenszel(
        buildResponseMatrix([
          [1, 1],
          [null, null],
        ]),
        ["reference", "focal"],
      ),
    ).toThrow("Mantel-Haenszel DIF needs persons with responses in both groups.");
  });
});

/** Liu and Agresti's statistics of a fixture, by statistic, then item. */
type LiuAgrestiFixture = Readonly<
  Record<"oddsRatio" | "logStandardError" | "z" | "significance", readonly (number | null)[]>
>;

function expectLiuAgresti(actual: readonly MantelTest[], expected: LiuAgrestiFixture): void {
  actual.forEach(({ liuAgresti }, i) => {
    const oddsRatio = expected.oddsRatio[i] ?? null;
    expectClose(liuAgresti?.oddsRatio ?? null, oddsRatio, LIU_AGRESTI_TOLERANCE * (oddsRatio ?? 1), `ψ ${i + 1}`);
    expectClose(liuAgresti?.logStandardError ?? null, expected.logStandardError[i] ?? null, LIU_AGRESTI_TOLERANCE);
    expectClose(liuAgresti?.z ?? null, expected.z[i] ?? null, LIU_AGRESTI_TOLERANCE, `z ${i + 1}`);
    expectClose(liuAgresti?.significance ?? null, expected.significance[i] ?? null, LIU_AGRESTI_TOLERANCE);
  });
}

describe("mantelTest", () => {
  for (const dataset of polytomous.cases) {
    it(`matches base R's Mantel and generalized Mantel-Haenszel statistics on ${dataset.name} data`, () => {
      const result = mantelTest(buildResponseMatrix(dataset.responses), groups(dataset.focal));
      expect(result.purification).toBeNull();
      expectMantelTest(result.items, dataset.complete.mantel);
    });

    it(`matches base R's purified statistics on ${dataset.name} data`, () => {
      const expected = dataset.complete.purifiedMantel;
      const result = mantelTest(buildResponseMatrix(dataset.responses), groups(dataset.focal), { purify: true });
      expectMantelTest(result.items, expected);
      expectPurification(result.purification, expected);
    });

    it(`matches base R's rule for sparse data, purified and not, on ${dataset.name} data`, () => {
      const { sparse } = dataset;
      const matrix = buildResponseMatrix(sparse.responses);
      expectMantelTest(mantelTest(matrix, groups(dataset.focal)).items, sparse.mantel);
      const purified = mantelTest(matrix, groups(dataset.focal), { purify: true });
      expectMantelTest(purified.items, sparse.purifiedMantel);
      expectPurification(purified.purification, sparse.purifiedMantel);
    });
  }

  for (const [name, expected] of Object.entries(polytomousTests.cases)) {
    it(`matches difR's Liu-Agresti odds ratio and Liu and Agresti's standard error on ${name} data`, () => {
      const { matrix, groups: caseGroups } = fixtureCase(expected.input);
      expectLiuAgresti(mantelTest(matrix, caseGroups).items, expected.liuAgresti);
      const purifiedLiuAgresti = "purifiedLiuAgresti" in expected ? expected.purifiedLiuAgresti : undefined;
      if (purifiedLiuAgresti === undefined) return;
      const purified = mantelTest(matrix, caseGroups, { purify: true });
      expectLiuAgresti(purified.items, purifiedLiuAgresti);
      expect(purified.purification?.anchors).toEqual(purifiedLiuAgresti.anchors.map((index) => `item${index}`));
    });
  }

  it("gives the Mantel-Haenszel odds ratio as the Liu-Agresti odds ratio of dichotomous items", () => {
    for (const dataset of fixture.cases) {
      const matrix = buildResponseMatrix(dataset.sparse.responses);
      const mantel = mantelTest(matrix, groups(dataset.focal)).items;
      mantelHaenszel(matrix, groups(dataset.focal)).items.forEach((item, i) => {
        expectClose(mantel[i]?.liuAgresti?.oddsRatio ?? null, item.oddsRatio, 1e-12 * (item.oddsRatio ?? 1));
      });
    }
  });

  it("keeps the standard error when the groups swap, where difR's depends on their order", () => {
    const expected = polytomousTests.cases.bfi;
    const { matrix, groups: caseGroups } = fixtureCase(expected.input);
    const swapped = caseGroups.map((group): DifGroup => (group === "focal" ? "reference" : "focal"));
    const forward = mantelTest(matrix, caseGroups).items;
    mantelTest(matrix, swapped).items.forEach((item, i) => {
      const { oddsRatio = 0, logStandardError = null } = forward[i]?.liuAgresti ?? {};
      expectClose(item.liuAgresti?.oddsRatio ?? null, 1 / oddsRatio, 1e-12 / oddsRatio);
      expectClose(item.liuAgresti?.logStandardError ?? null, logStandardError, 1e-12);
      const difr = expected.liuAgresti.difrStandardError[i] ?? 0;
      const reversed = expected.liuAgresti.difrReversedStandardError[i] ?? 0;
      expect(Math.abs(difr - reversed)).toBeGreaterThan(0.01);
    });
    // On 1,000 seeded sets, the standard error of equation 3 is the spread of ln ψ̂ within 5%; difR's is not.
    const { spread, equation, difr, difrReversed } = polytomousTests.monteCarlo;
    expect(Math.abs(equation / spread - 1)).toBeLessThan(0.05);
    expect([difr / spread, difrReversed / spread].map((ratio) => Math.abs(ratio - 1) > 0.5)).toEqual([true, true]);
  });

  it("gives no Liu-Agresti odds ratio without a score group of both groups, nor a test of a ratio of 0", () => {
    const apart = mantelTest(buildResponseMatrix(DISJOINT_ROWS), ["reference", "reference", "focal", "focal"]).items;
    expect(apart[0]?.liuAgresti).toBeNull();
    // In both score groups of DISJOINT_ROWS the focal response to item 1 is above the reference response: ψ̂ = 0.
    const { items } = mantelTest(buildResponseMatrix(DISJOINT_ROWS), ["reference", "focal", "reference", "focal"]);
    expect(items[0]?.liuAgresti).toEqual({ oddsRatio: 0, logStandardError: null, z: null, significance: null });
  });

  it("gives the Mantel-Haenszel chi-square without the continuity correction for dichotomous items", () => {
    const { responses, focal } = fixture.cases[0] ?? { responses: [], focal: [] };
    const matrix = buildResponseMatrix(responses);
    const mantel = mantelTest(matrix, groups(focal)).items;
    const reference = mantelHaenszel(matrix, groups(focal), { continuityCorrection: false }).items;
    reference.forEach((item, i) => {
      expectClose(mantel[i]?.chiSquare ?? null, item.chiSquare, MANTEL_TEST_TOLERANCE, `chi-square ${i + 1}`);
      expectClose(mantel[i]?.generalized.chiSquare ?? null, item.chiSquare, MANTEL_TEST_TOLERANCE, `general ${i + 1}`);
      expect(mantel[i]?.generalized.degreesOfFreedom).toBe(1);
    });
  });

  it("derives the statistics of two score groups by hand", () => {
    const { items } = mantelTest(buildResponseMatrix(DISJOINT_ROWS), ["reference", "focal", "reference", "focal"]);
    // Score group 1 holds [0, 1] and [1, 0], group 5 holds [2, 3] and [3, 2]; in each, F = E(F) + 1/2 and Var(F) = 1/4,
    // so the chi-square is 1 / (1/2) = 2. The standardized mean difference is 1: the focal mean is 1 higher in both.
    // The responses 0, 1, 2 and 3 have a sample variance of 5/3.
    expect(items[0]?.chiSquare).toBeCloseTo(2, 14);
    expect(items[0]?.significance).toBeCloseTo(0.15729920705028516, 14);
    expect(items[0]?.standardizedMeanDifference).toBeCloseTo(1, 14);
    expect(items[0]?.effectSize).toBeCloseTo(Math.sqrt(3 / 5), 14);
    expect(items[0]?.etsCategory).toBe("AA");
    expect(items[1]?.standardizedMeanDifference).toBeCloseTo(-1, 14);
  });

  it("gives a null generalized chi-square where the score groups use disjoint categories", () => {
    const { items } = mantelTest(buildResponseMatrix(DISJOINT_ROWS), ["reference", "focal", "reference", "focal"]);
    expect(items[0]?.generalized).toEqual({ chiSquare: null, degreesOfFreedom: 3, significance: null });
  });

  it("counts only the categories used in score groups of more than one person", () => {
    const rows = [...DISJOINT_ROWS, [4, 3]];
    const { items } = mantelTest(buildResponseMatrix(rows), ["reference", "focal", "reference", "focal", "focal"]);
    expect(items[0]?.generalized.degreesOfFreedom).toBe(3);
    expect(items[0]?.chiSquare).toBeCloseTo(2, 14);
    expect(items[0]?.standardizedMeanDifference).toBeCloseTo(1, 14);
  });

  it("tables only the scores and responses that occur, so responses far apart cost no more than close ones", () => {
    // Scaling every response by c scales F - E(F) by c and Var(F) by c², so the chi-square and the effect size stay.
    const scale = 2 ** 20;
    const scaled = DISJOINT_ROWS.map((row) => row.map((response) => response * scale));
    const { items } = mantelTest(buildResponseMatrix(scaled), ["reference", "focal", "reference", "focal"]);
    expect(items[0]?.chiSquare).toBeCloseTo(2, 12);
    expect(items[0]?.standardizedMeanDifference).toBeCloseTo(scale, 6);
    expect(items[0]?.effectSize).toBeCloseTo(Math.sqrt(3 / 5), 12);
    expect(items[0]?.generalized).toEqual({ chiSquare: null, degreesOfFreedom: 3, significance: null });
  });

  it("gives null statistics for an item every person answered alike", () => {
    const result = mantelTest(buildResponseMatrix(CONSTANT_ROWS), CONSTANT_GROUPS).items[0];
    expect(result).toEqual({
      id: "item1",
      chiSquare: null,
      significance: null,
      generalized: { chiSquare: null, degreesOfFreedom: 0, significance: null },
      standardizedMeanDifference: 0,
      effectSize: null,
      etsCategory: null,
      liuAgresti: null,
    });
  });

  it("scales the matching score of a person who skipped an item to the highest possible score of its items", () => {
    const answered = [...SCALING_ROWS, [2, 1, 1]];
    const skipped = [...SCALING_ROWS, [2, 1, null]];
    const scalingGroups = [...SCALING_GROUPS, "focal" as const];
    // [2, 1, null] answered 1 of 2 on item 2, which scales to 3 / 2 of the other items' 3 points, 2 with halves up: the
    // score of item 1 is 2 + 2, as that of [2, 1, 1].
    const full = mantelTest(buildResponseMatrix(answered), scalingGroups).items[0];
    expect(full?.chiSquare).not.toBeNull();
    expect(mantelTest(buildResponseMatrix(skipped), scalingGroups).items[0]).toEqual(full);
  });

  it("rejects groups without responses and invalid purification", () => {
    const missing = buildResponseMatrix([
      [2, 1],
      [null, null],
    ]);
    expect(() => mantelTest(missing, ["reference", "focal"])).toThrow(
      "Mantel DIF needs persons with responses in both groups.",
    );
  });

  it("refuses an item whose persons use more than 255 categories, whose generalized test holds every pair", () => {
    const rows = Array.from({ length: 512 }, (_, person) => [person % 256, person % 2]);
    const groups: DifGroup[] = rows.map((_, person) => (person < 256 ? "reference" : "focal"));
    expect(() => mantelTest(buildResponseMatrix(rows), groups)).toThrow(
      'Mantel DIF takes items of at most 255 categories, found 256 used for item "item1".',
    );
    const fewer = rows.map(([first = 0, second = 0]) => [Math.min(first, 254), second]);
    expect(mantelTest(buildResponseMatrix(fewer), groups).items[0]?.generalized.degreesOfFreedom).toBe(254);
  });
});

describe("ordinalLogisticRegressionDif", () => {
  for (const dataset of polytomous.cases) {
    it(`matches MASS::polr and lordif's rundif on ${dataset.name} data`, () => {
      const result = ordinalLogisticRegressionDif(buildResponseMatrix(dataset.responses), groups(dataset.focal));
      expect(result.purification).toBeNull();
      expectOrdinal(result.items, dataset.complete.ordinal, dataset.complete.lordif);
    });

    it(`matches polr and rundif on ${dataset.name} data matched on theta`, () => {
      const result = ordinalLogisticRegressionDif(buildResponseMatrix(dataset.responses), groups(dataset.focal), {
        matching: dataset.theta,
      });
      expectOrdinal(result.items, dataset.complete.ordinalTheta, dataset.complete.lordifTheta);
    });

    it(`matches polr's purified statistics on ${dataset.name} data`, () => {
      const expected = dataset.complete.purifiedOrdinal;
      const result = ordinalLogisticRegressionDif(buildResponseMatrix(dataset.responses), groups(dataset.focal), {
        purify: true,
      });
      expectOrdinal(result.items, expected);
      expectPurification(result.purification, expected);
    });

    it(`matches polr's rule for sparse data, purified, not and matched on theta, on ${dataset.name} data`, () => {
      const { sparse } = dataset;
      const matrix = buildResponseMatrix(sparse.responses);
      expectOrdinal(ordinalLogisticRegressionDif(matrix, groups(dataset.focal)).items, sparse.ordinal);
      const options = { matching: sparse.theta };
      expectOrdinal(ordinalLogisticRegressionDif(matrix, groups(dataset.focal), options).items, sparse.ordinalTheta);
      const purified = ordinalLogisticRegressionDif(matrix, groups(dataset.focal), { purify: true });
      expectOrdinal(purified.items, sparse.purifiedOrdinal);
      expectPurification(purified.purification, sparse.purifiedOrdinal);
    });
  }

  it("equals logistic regression for dichotomous items", () => {
    for (const dataset of fixture.cases) {
      const matrix = buildResponseMatrix(dataset.responses);
      const ordinal = ordinalLogisticRegressionDif(matrix, groups(dataset.focal)).items;
      const logistic = logisticRegressionDif(matrix, groups(dataset.focal)).items;
      logistic.forEach((item, i) => {
        for (const type of ["uniform", "nonuniform", "both"] as const) {
          for (const name of ["chiSquare", "significance", "rSquaredChange"] as const) {
            const actual = ordinal[i]?.[type][name] ?? null;
            expectClose(actual, item[type][name], ORDINAL_TOLERANCE, `${dataset.name} ${type} ${name} ${i + 1}`);
          }
        }
      });
    }
  });

  it("uses the categories the persons used, whichever integers the responses are", () => {
    const dataset = polytomous.cases[0];
    const matrix = buildResponseMatrix(dataset?.responses ?? []);
    const options = { matching: dataset?.theta ?? [] };
    const spread = [0, 3, 4, 9, 20, 21];
    const relabeled = buildResponseMatrix(
      matrix.responses.map((row) => row.map((response) => spread[response ?? 0] ?? 0)),
    );
    const expected = ordinalLogisticRegressionDif(matrix, groups(dataset?.focal ?? []), options);
    expect(ordinalLogisticRegressionDif(relabeled, groups(dataset?.focal ?? []), options)).toEqual(expected);
  });

  it("scales the matching score of a person who skipped an item as the Mantel test does", () => {
    const answered = [...SCALING_ROWS, [2, 1, 1]];
    const skipped = [...SCALING_ROWS, [2, 1, null]];
    const scalingGroups = [...SCALING_GROUPS, "focal" as const];
    const full = ordinalLogisticRegressionDif(buildResponseMatrix(answered), scalingGroups).items[0];
    expect(full?.both.chiSquare).not.toBeNull();
    expect(ordinalLogisticRegressionDif(buildResponseMatrix(skipped), scalingGroups).items[0]).toEqual(full);
  });

  it("leaves out the persons without a matching score and does not depend on its scale", () => {
    const dataset = polytomous.cases[0];
    const focal = dataset?.focal.slice(0, 400) ?? [];
    const rows = dataset?.responses.slice(0, 400) ?? [];
    const theta = dataset?.theta.slice(0, 400) ?? [];
    const dropped = theta.map((score, person) => (person % 5 === 0 ? null : score));
    const kept = rows.map((_, person) => person % 5 !== 0);
    const withNulls = ordinalLogisticRegressionDif(buildResponseMatrix(rows), groups(focal), { matching: dropped });
    const without = ordinalLogisticRegressionDif(
      buildResponseMatrix(rows.filter((_, person) => kept[person])),
      groups(focal.filter((_, person) => kept[person])),
      { matching: theta.filter((_, person) => kept[person]) },
    );
    expect(withNulls.items).toEqual(without.items);
    const rescaled = ordinalLogisticRegressionDif(buildResponseMatrix(rows), groups(focal), {
      matching: dropped.map((score) => (score === null ? null : 3 * score + 7)),
    });
    rescaled.items.forEach((item, i) => {
      const reference = without.items[i];
      expectClose(item.both.chiSquare, reference?.both.chiSquare ?? null, 1e-7, `both ${i + 1}`);
      expectClose(item.slopeChange, reference?.slopeChange ?? null, 1e-7, `slope change ${i + 1}`);
    });
  });

  it("gives null tests for an item every person answered alike, persons of one group or an equal matching score", () => {
    const alike = ordinalLogisticRegressionDif(buildResponseMatrix(CONSTANT_ROWS), CONSTANT_GROUPS).items;
    expect(alike[0]?.both).toEqual({ chiSquare: null, degreesOfFreedom: 2, significance: null, rSquaredChange: null });
    expect(alike[0]?.slopeChange).toBeNull();
    expect(alike[1]?.uniform.chiSquare).not.toBeNull();
    const oneGroup = CONSTANT_ROWS.map((row, person) => [person < 3 ? (row[0] ?? null) : null, ...row.slice(1)]);
    const options = { matching: [1, 2, 3, 4, 5, 6] };
    expect(
      ordinalLogisticRegressionDif(buildResponseMatrix(oneGroup), CONSTANT_GROUPS, options).items[0]?.uniform,
    ).toEqual({
      chiSquare: null,
      degreesOfFreedom: 1,
      significance: null,
      rSquaredChange: null,
    });
    const equal = ordinalLogisticRegressionDif(buildResponseMatrix(CONSTANT_ROWS), CONSTANT_GROUPS, {
      matching: [4, 4, 4, 4, 4, 4],
    });
    expect(equal.items[1]?.both.chiSquare).toBeNull();
  });

  it("computes the tests where a group term diverges", () => {
    // Group 1 answers 0 or 1, group 2 only 2: the group term has no finite estimate.
    const rows = [[0], [1], [0], [1], [0], [1], [2], [2], [2], [2], [2], [2]];
    const separated = rows.map((_, person) => (person < 6 ? "reference" : "focal")) as DifGroup[];
    const matching = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6];
    const { uniform } = ordinalLogisticRegressionDif(buildResponseMatrix(rows), separated, { matching }).items[0] ?? {};
    expect(uniform?.chiSquare).toBeGreaterThan(10);
    expect(uniform?.significance).toBeLessThan(0.01);
  });

  it("rejects a matching score that purification would have to change, and an invalid one", () => {
    const matrix = buildResponseMatrix(CONSTANT_ROWS);
    expect(() =>
      ordinalLogisticRegressionDif(matrix, CONSTANT_GROUPS, { matching: [1, 2, 3, 4, 5, 6], purify: true }),
    ).toThrow("The option purify purifies the total score, so it cannot be used with the option matching.");
    expect(() => ordinalLogisticRegressionDif(matrix, CONSTANT_GROUPS, { matching: [1, 2, 3] })).toThrow(
      "Found 3 matching scores for 6 persons.",
    );
    const infinite: (number | null)[] = JSON.parse('[1, 2, 3, 4, 5, "high"]');
    expect(() => ordinalLogisticRegressionDif(matrix, CONSTANT_GROUPS, { matching: infinite })).toThrow(
      'The matching score of person "person6" must be a finite number or null, found "high".',
    );
    expect(() => ordinalLogisticRegressionDif(matrix, CONSTANT_GROUPS, { matching: JSON.parse("null") })).toThrow(
      "The option matching must be an array, found null.",
    );
    expect(() =>
      ordinalLogisticRegressionDif(matrix, CONSTANT_GROUPS, { matching: [1, 2, Number.NaN, 4, 5, 6] }),
    ).toThrow('The matching score of person "person3" must be a finite number or null, found NaN.');
  });

  it("rejects groups without responses", () => {
    const missing = buildResponseMatrix([
      [2, 1],
      [null, null],
    ]);
    expect(() => ordinalLogisticRegressionDif(missing, ["reference", "focal"])).toThrow(
      "Ordinal logistic regression DIF needs persons with responses in both groups.",
    );
  });

  it("refuses an item whose persons use more than 255 categories, whose regression holds every pair", () => {
    const rows = Array.from({ length: 512 }, (_, person) => [person % 256, person % 2]);
    const groups: DifGroup[] = rows.map((_, person) => (person < 256 ? "reference" : "focal"));
    expect(() => ordinalLogisticRegressionDif(buildResponseMatrix(rows), groups)).toThrow(
      'Ordinal logistic regression DIF takes items of at most 255 categories, found 256 used for item "item1".',
    );
  });
});

describe("logisticDiscriminantDif", () => {
  it("gives the values of the guide's polytomous sample, with the Mantel test's Liu-Agresti odds ratio", () => {
    const random = createRandom(2026);
    const items = simulateItems(gradedResponse, 10, random, { categoryCount: 4 });
    const shifted = items.map((item, j) =>
      j === 2 ? { ...item, parameters: { ...item.parameters, b: item.parameters.b.map((b) => b + 1) } } : item,
    );
    const reference = simulateResponses(items, simulateThetas(500, random), random);
    const focal = simulateResponses(shifted, simulateThetas(500, random), random);
    const matrix = buildResponseMatrix([...reference.responses, ...focal.responses]);
    const sampleGroups: DifGroup[] = matrix.personIds.map((_, person) => (person < 500 ? "reference" : "focal"));
    const odds = defined(mantelTest(matrix, sampleGroups).items[2]?.liuAgresti);
    expect([odds.oddsRatio, odds.logStandardError, odds.z].map((value) => shown(value, 3))).toEqual([
      3.89, 0.144, 9.47,
    ]);
    expect(shown(odds.significance, 2)).toBe(2.8e-21);
    const { uniform } = defined(logisticDiscriminantDif(matrix, sampleGroups).items[2]);
    expect([shown(uniform.chiSquare, 3), shown(uniform.significance, 2), shown(uniform.rSquaredChange, 3)]).toEqual([
      98.1, 4e-23, 0.124,
    ]);
  });

  for (const [name, expected] of Object.entries(polytomousTests.cases)) {
    it(`matches base R's glm on ${name} data, purified and not`, () => {
      const { matrix, groups: caseGroups } = fixtureCase(expected.input);
      const result = logisticDiscriminantDif(matrix, caseGroups);
      expect(result.purification).toBeNull();
      expectLogistic(result.items, expected.discriminant, DISCRIMINANT_TOLERANCE);
      const purified = logisticDiscriminantDif(matrix, caseGroups, { purify: true });
      expectLogistic(purified.items, expected.purifiedDiscriminant, DISCRIMINANT_TOLERANCE);
      expectPurification(purified.purification, expected.purifiedDiscriminant);
    });
  }

  for (const [name, expected] of Object.entries(polytomousTests.cases)) {
    if (!("discriminantTheta" in expected)) continue;
    it(`matches base R's glm on ${name} data matched on theta`, () => {
      const dataset = defined(polytomous.cases.find((candidate) => candidate.name === expected.input.case));
      const matching = expected.input.sparse ? dataset.sparse.theta : dataset.theta;
      const { matrix, groups: caseGroups } = fixtureCase(expected.input);
      const { items } = logisticDiscriminantDif(matrix, caseGroups, { matching });
      expectLogistic(items, expected.discriminantTheta, DISCRIMINANT_TOLERANCE);
    });
  }

  it("enters the response as one numeric predictor, so its tests have 1, 1 and 2 degrees of freedom", () => {
    const { matrix, groups: caseGroups } = fixtureCase(polytomousTests.cases.bfi.input);
    const item = defined(logisticDiscriminantDif(matrix, caseGroups).items[0]);
    expect([item.uniform, item.nonuniform, item.both].map((test) => test.degreesOfFreedom)).toEqual([1, 1, 2]);
  });

  it("gives null tests for an item answered alike, by one group, or with one matching score", () => {
    const { items } = logisticDiscriminantDif(buildResponseMatrix(CONSTANT_ROWS), CONSTANT_GROUPS);
    expect(items[0]?.both).toEqual({ chiSquare: null, degreesOfFreedom: 2, significance: null, rSquaredChange: null });
    expect(items[1]?.uniform.chiSquare).not.toBeNull();
    const matching = CONSTANT_ROWS.map(() => 1);
    for (const item of logisticDiscriminantDif(buildResponseMatrix(CONSTANT_ROWS), CONSTANT_GROUPS, { matching })
      .items) {
      expect(item.both.chiSquare).toBeNull();
    }
    const rows = CONSTANT_ROWS.map((row, person) => [
      row[0] ?? null,
      person < 3 ? (row[1] ?? null) : null,
      row[2] ?? null,
    ]);
    expect(logisticDiscriminantDif(buildResponseMatrix(rows), CONSTANT_GROUPS).items[1]?.uniform.chiSquare).toBeNull();
  });

  it("rejects purify with matching, a matching score that is not finite, and options of other types", () => {
    const matrix = buildResponseMatrix(CONSTANT_ROWS);
    const matching = CONSTANT_ROWS.map(() => 0);
    expect(() => logisticDiscriminantDif(matrix, CONSTANT_GROUPS, { matching, purify: true })).toThrow(
      "The option purify purifies the total score, so it cannot be used with the option matching.",
    );
    expect(() =>
      logisticDiscriminantDif(matrix, CONSTANT_GROUPS, { matching: [...matching.slice(1), Number.NaN] }),
    ).toThrow('The matching score of person "person6" must be a finite number or null, found NaN.');
    expect(() => logisticDiscriminantDif(matrix, CONSTANT_GROUPS, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => logisticDiscriminantDif(matrix, CONSTANT_GROUPS, JSON.parse('{"verbose":1}'))).toThrow(
      'There is no option "verbose"',
    );
    expect(() => logisticDiscriminantDif(matrix, CONSTANT_GROUPS, { matching: JSON.parse("{}") })).toThrow(
      "The option matching must be an array",
    );
    expect(() => logisticDiscriminantDif(matrix, JSON.parse('"focal"'))).toThrow(
      'The groups must be an array, found "focal".',
    );
  });
});

describe("sibtest", () => {
  for (const [name, expected] of Object.entries(sibtestFixture.cases)) {
    const { matrix, personGroups } = sibtestInput(expected.input);

    it(`matches mirt's SIBTEST and difR's difSIBTEST on ${name}`, () => {
      const result = sibtest(matrix, personGroups);
      expect(result.purification).toBeNull();
      expect(result.bundles).toEqual([]);
      expectSibtest(result.items, expected.default);
    });

    it(`matches mirt without the correction and with guessing on ${name}`, () => {
      expectSibtest(sibtest(matrix, personGroups, { correction: false }).items, expected.uncorrected);
      expectSibtest(sibtest(matrix, personGroups, { guessing: sibtestFixture.guessing }).items, expected.guessing);
    });

    it(`matches mirt's Jmin as the option smallCell on ${name}`, () => {
      expectSibtest(sibtest(matrix, personGroups, { smallCell: 2 }).items, expected.smallCells);
      expectSibtest(sibtest(matrix, personGroups, { smallCell: 20 }).items, expected.largeCells);
    });

    const bundles = Object.fromEntries(
      expected.bundles.map(({ name: bundle, items }) => [bundle, items.map((item) => defined(matrix.itemIds[item]))]),
    );

    it(`matches mirt's suspect_set for bundles on ${name}`, () => {
      const result = sibtest(matrix, personGroups, { bundles });
      expect(result.bundles.map((bundle) => bundle.id)).toEqual(expected.bundles.map((bundle) => bundle.name));
      expected.bundles.forEach((bundle, i) => {
        expectBundle(result.bundles[i], bundle.default);
      });
      expectSibtest(result.items, expected.default);
    });

    it(`matches difR's purification, and matches bundles on the final anchors, on ${name}`, () => {
      const result = sibtest(matrix, personGroups, { purify: true, bundles });
      expectSibtest(result.items, expected.purified);
      expectPurification(result.purification, expected.purified);
      expected.bundles.forEach((bundle, i) => {
        expectBundle(result.bundles[i], bundle.purified);
      });
    });

    it(`matches the purification without the correction on ${name}`, () => {
      const result = sibtest(matrix, personGroups, { purify: true, correction: false });
      expectSibtest(result.items, expected.purifiedUncorrected);
      expectPurification(result.purification, expected.purifiedUncorrected);
    });
  }

  it("flags a shifted item and holds the false alarms of clean items on 30 sets, as the guide says", () => {
    const sets = 30;
    const shiftedItem = 0;
    const random = createRandom(2026);
    let hits = 0;
    let cleanTests = 0;
    const falseAlarms = { corrected: 0, uncorrected: 0 };
    const flagged = (item: Sibtest | undefined): number => ((item?.significance ?? 1) < 0.05 ? 1 : 0);
    for (let set = 0; set < sets; set++) {
      const items = simulateItems(twoParameterLogistic, 20, random);
      const shifted = items.map((item, j) =>
        j === shiftedItem ? { ...item, parameters: { ...item.parameters, b: item.parameters.b + 0.8 } } : item,
      );
      const reference = simulateResponses(items, simulateThetas(500, random), random);
      const focalThetas = simulateThetas(500, random).map((theta) => theta - 0.5);
      const focal = simulateResponses(shifted, focalThetas, random);
      const matrix = buildResponseMatrix([...reference.responses, ...focal.responses]);
      const personGroups: DifGroup[] = matrix.personIds.map((_, person) => (person < 500 ? "reference" : "focal"));
      const uncorrected = sibtest(matrix, personGroups, { correction: false }).items;
      sibtest(matrix, personGroups).items.forEach((item, j) => {
        if (j === shiftedItem) {
          hits += flagged(item);
          // The item is harder for the focal group, so it favors the reference group.
          expect(item.beta).toBeGreaterThan(0);
        } else {
          falseAlarms.corrected += flagged(item);
          falseAlarms.uncorrected += flagged(uncorrected[j]);
          cleanTests++;
        }
      });
    }
    // The design's thresholds, fixed before the first run.
    expect(hits).toBeGreaterThanOrEqual(28);
    expect(falseAlarms.corrected / cleanTests).toBeLessThanOrEqual(0.08);
    // The guide's numbers: 5.6% with the correction, 9.8% without it.
    expect(shown((100 * falseAlarms.corrected) / cleanTests, 2)).toBe(5.6);
    expect(shown((100 * falseAlarms.uncorrected) / cleanTests, 2)).toBe(9.8);
  });

  it("gives the values of the guide's sample", () => {
    const { matrix, personGroups } = guideSample(20, { 2: 0.5, 3: 0.5 }, 0.5);
    const { items } = sibtest(matrix, personGroups);
    const item3 = defined(items[2]);
    expect([item3.beta, item3.standardError].map((value) => shown(value, 2))).toEqual([0.092, 0.025]);
    expect([shown(item3.chiSquare, 3), shown(item3.significance, 2), item3.persons]).toEqual([13.5, 0.00024, 1000]);
    const item11 = defined(items[10]);
    expect(
      [item11.beta, item11.standardError, item11.chiSquare, item11.significance].map((value) => shown(value, 2)),
    ).toEqual([0.01, 0.027, 0.14, 0.71]);
    const purified = sibtest(matrix, personGroups, { bundles: { passage: ["item3", "item4"] }, purify: true });
    const passage = defined(purified.bundles[0]);
    expect([shown(passage.beta, 3), shown(passage.standardError, 2), shown(passage.chiSquare, 3)]).toEqual([
      0.21, 0.041, 26.9,
    ]);
    expect(shown(passage.significance, 2)).toBe(2.1e-7);
    expect([shown(purified.items[2]?.chiSquare, 3), shown(purified.items[3]?.chiSquare, 3)]).toEqual([16, 14.2]);
  });

  it("counts the persons each analysis uses: 45% leave at 2% of the responses missing over 30 items, as the guide says", () => {
    const random = createRandom(2026);
    const items = simulateItems(twoParameterLogistic, 30, random);
    const { responses } = simulateResponses(items, simulateThetas(2000, random), random);
    const sparse = responses.map((row) => row.map((response) => (random.uniform() < 0.02 ? null : response)));
    const personGroups: DifGroup[] = sparse.map((_, person) => (person % 2 === 0 ? "reference" : "focal"));
    const { items: statistics } = sibtest(buildResponseMatrix(sparse), personGroups);
    const complete = sparse.filter((row) => !row.includes(null)).length;
    for (const item of statistics) expect(item.persons).toBe(complete);
    // 1 - 0.98³⁰ of the persons miss a response: 45%, and the seeded share lies within 2% of it.
    expect(Math.round(100 * (1 - 0.98 ** 30))).toBe(45);
    expect(Math.abs(1 - complete / sparse.length - (1 - 0.98 ** 30))).toBeLessThan(0.02);
  });

  it("derives the statistics of one score group by hand", () => {
    // Only score 1 of item 2 lies between the lowest and the highest. It holds [0, 1, 1] and [0, 0, 1], whose means are
    // 2/3 and 1/3 and whose sample variances are 1/3.
    const { items } = sibtest(
      buildResponseMatrix(ONE_MIDDLE_SCORE_ROWS),
      ONE_MIDDLE_SCORE_GROUPS,
      ONE_MIDDLE_SCORE_OPTIONS,
    );
    expectClose(items[0]?.beta ?? null, 1 / 3, SIBTEST_TOLERANCE);
    expectClose(items[0]?.standardError ?? null, Math.sqrt(2) / 3, SIBTEST_TOLERANCE);
    expectClose(items[0]?.chiSquare ?? null, 1 / 2, SIBTEST_TOLERANCE);
    expectClose(items[0]?.significance ?? null, chiSquareUpperTail(1 / 2, 1), SIBTEST_TOLERANCE);
    // Matched on item 1, whose two scores are the lowest and the highest, nothing is left to compare.
    expect(items[1]).toEqual({ ...UNDEFINED_SIBTEST, id: "item2" });
  });

  it("uses a score group only above smallCell persons of each group, as mirt's Jmin", () => {
    // The score group holds 3 persons of each group: more than 2, but not more than 3.
    const matrix = buildResponseMatrix(ONE_MIDDLE_SCORE_ROWS);
    expect(sibtest(matrix, ONE_MIDDLE_SCORE_GROUPS, ONE_MIDDLE_SCORE_OPTIONS).items[0]?.beta).not.toBeNull();
    const options = { ...ONE_MIDDLE_SCORE_OPTIONS, smallCell: 3 };
    expect(sibtest(matrix, ONE_MIDDLE_SCORE_GROUPS, options).items[0]).toEqual({ ...UNDEFINED_SIBTEST, id: "item1" });
  });

  it("is undefined with the correction on one matching item, whose coefficient alpha is undefined", () => {
    const matrix = buildResponseMatrix(ONE_MIDDLE_SCORE_ROWS);
    expect(sibtest(matrix, ONE_MIDDLE_SCORE_GROUPS, { smallCell: 2 }).items[0]).toEqual({
      ...UNDEFINED_SIBTEST,
      id: "item1",
    });
  });

  it("leaves out a person with a missing response to a matching or a studied item", () => {
    const complete = sibtest(
      buildResponseMatrix(ONE_MIDDLE_SCORE_ROWS),
      ONE_MIDDLE_SCORE_GROUPS,
      ONE_MIDDLE_SCORE_OPTIONS,
    );
    const rows = [...ONE_MIDDLE_SCORE_ROWS, [1, null], [null, 1]];
    const personGroups: DifGroup[] = [...ONE_MIDDLE_SCORE_GROUPS, "focal", "reference"];
    const sparse = sibtest(buildResponseMatrix(rows), personGroups, ONE_MIDDLE_SCORE_OPTIONS);
    expect(sparse.items).toEqual(complete.items);
  });

  it("gives null statistics when no person answered the items of an analysis", () => {
    const rows = ONE_MIDDLE_SCORE_ROWS.map(([studied, matching]) => [studied ?? null, matching ?? null, null]);
    const { items } = sibtest(buildResponseMatrix(rows), ONE_MIDDLE_SCORE_GROUPS, ONE_MIDDLE_SCORE_OPTIONS);
    for (const item of items) expect(item).toEqual({ ...UNDEFINED_SIBTEST, id: item.id, persons: 0 });
  });

  it("gives null statistics for an item every person answered alike, which leaves no score group a variance", () => {
    const rows = ONE_MIDDLE_SCORE_ROWS.map(([, matching]) => [1, matching ?? 0]);
    const { items } = sibtest(buildResponseMatrix(rows), ONE_MIDDLE_SCORE_GROUPS, ONE_MIDDLE_SCORE_OPTIONS);
    expect(items[0]).toEqual({ ...UNDEFINED_SIBTEST, id: "item1" });
  });

  it("gives null statistics for a bundle of every item, which leaves nothing to match on", () => {
    const options = { ...ONE_MIDDLE_SCORE_OPTIONS, bundles: { all: ["item1", "item2"] } };
    const result = sibtest(buildResponseMatrix(ONE_MIDDLE_SCORE_ROWS), ONE_MIDDLE_SCORE_GROUPS, options);
    expect(result.bundles).toEqual([{ ...UNDEFINED_SIBTEST, id: "all" }]);
  });

  it("tables only the matching scores that occur, so scores far apart cost no more than close ones", () => {
    const { responses, focal } = defined(polytomous.cases.find((dataset) => dataset.name === "bfi"));
    const scale = 2 ** 16;
    const scaled = responses.map((row) => row.map((response) => response * scale));
    const close = sibtest(buildResponseMatrix(responses), groups(focal)).items;
    const far = sibtest(buildResponseMatrix(scaled), groups(focal)).items;
    far.forEach((item, i) => {
      expectClose(item.chiSquare, close[i]?.chiSquare ?? null, 1e-9, `chi-square ${i + 1}`);
      expectClose(item.beta, (close[i]?.beta ?? 0) * scale, 1e-6, `beta ${i + 1}`);
    });
  });

  it("takes a bundle by its own name, whatever the name", () => {
    const bundles = JSON.parse('{ "__proto__": ["item1"], "constructor": ["item2"] }');
    const result = sibtest(buildResponseMatrix(ONE_MIDDLE_SCORE_ROWS), ONE_MIDDLE_SCORE_GROUPS, {
      ...ONE_MIDDLE_SCORE_OPTIONS,
      bundles,
    });
    expect(result.bundles.map((bundle) => bundle.id)).toEqual(["__proto__", "constructor"]);
  });

  it("stops purifying, not converged, before the anchors are too few to test each on the others", () => {
    // With the correction, 3 anchors: bfi's first round flags 3 of its 5 items, and 2 anchors would leave each anchor
    // one matching item, whose coefficient alpha is undefined.
    const bfi = sibtestInput({ fixture: "dif-polytomous", case: "bfi", sparse: false });
    const first = sibtest(bfi.matrix, bfi.personGroups, { purify: true });
    expect(first.items.filter((item) => (item.significance ?? 1) < 0.05)).toHaveLength(3);
    expect(first.purification).toEqual({ anchors: bfi.matrix.itemIds, iterations: 0, converged: false });
    // Without it, 3 for dichotomous items, which two give a score between the lowest and the highest: after 4 rounds the
    // simulated items' last round leaves 2.
    const simulated = sibtestInput({ fixture: "dif", case: "simulated", sparse: false });
    const uncorrected = sibtest(simulated.matrix, simulated.personGroups, { purify: true, correction: false });
    expect(uncorrected.purification).toMatchObject({ iterations: 4, converged: false });
    expect(uncorrected.purification?.anchors).toHaveLength(4);
    expect(uncorrected.items.filter((item) => (item.significance ?? 1) >= 0.05)).toHaveLength(2);
    for (const anchor of defined(uncorrected.purification).anchors) {
      expect(uncorrected.items.find((item) => item.id === anchor)?.beta).not.toBeNull();
    }
    // And 2 for items of three categories or more, which one gives such a score.
    const graded = sibtestInput({ fixture: "dif-polytomous", case: "simulated", sparse: true });
    const polytomousRounds = sibtest(graded.matrix, graded.personGroups, { purify: true, correction: false });
    expect(polytomousRounds.purification).toMatchObject({ iterations: 4, converged: false });
    expect(polytomousRounds.purification?.anchors).toHaveLength(2);
  });

  describe("errors", () => {
    const matrix = buildResponseMatrix(ONE_MIDDLE_SCORE_ROWS);
    const analyze = (options: unknown) => sibtest(matrix, ONE_MIDDLE_SCORE_GROUPS, JSON.parse(JSON.stringify(options)));

    it("rejects options, a matrix and groups of other types", () => {
      expect(() => sibtest(matrix, ONE_MIDDLE_SCORE_GROUPS, JSON.parse("null"))).toThrow(
        "The options must be an object, found null.",
      );
      expect(() => sibtest(matrix, ONE_MIDDLE_SCORE_GROUPS, JSON.parse('{"verbose":1}'))).toThrow(
        'There is no option "verbose"',
      );
      expect(() => sibtest(JSON.parse("null"), ONE_MIDDLE_SCORE_GROUPS)).toThrow(DataError);
      expect(() => sibtest(matrix, JSON.parse("null"))).toThrow("The groups must be an array, found null.");
      expect(() => sibtest(matrix, JSON.parse('"focal"'))).toThrow('The groups must be an array, found "focal".');
      expect(() => sibtest(matrix, ONE_MIDDLE_SCORE_GROUPS.slice(1))).toThrow(DataError);
      expect(() =>
        sibtest(
          matrix,
          ONE_MIDDLE_SCORE_GROUPS.map((): DifGroup => "focal"),
        ),
      ).toThrow("SIBTEST needs persons with responses in both groups.");
    });

    it("rejects a smallCell that is not an integer from 2", () => {
      for (const [smallCell, found] of [
        [1, "1"],
        [2.5, "2.5"],
        ["5", '"5"'],
        [null, "null"],
      ]) {
        expect(() => analyze({ smallCell })).toThrow(`The option smallCell must be an integer from 2, found ${found}.`);
      }
    });

    it("rejects a correction or purify option that is not a boolean", () => {
      expect(() => analyze({ correction: "yes" })).toThrow('The option correction must be a boolean, found "yes".');
      expect(() => analyze({ correction: null })).toThrow("The option correction must be a boolean, found null.");
      expect(() => analyze({ purify: "yes" })).toThrow('The option purify must be a boolean, found "yes".');
    });

    it("rejects a guessing option that is not a number from 0 to below 1", () => {
      expect(() => analyze({ guessing: -0.1 })).toThrow(
        "The option guessing must be a non-negative number, found -0.1.",
      );
      expect(() => analyze({ guessing: "0.2" })).toThrow(
        'The option guessing must be a non-negative number, found "0.2".',
      );
      expect(() => analyze({ guessing: null })).toThrow(
        "The option guessing must be a non-negative number, found null.",
      );
      expect(() => analyze({ guessing: 1 })).toThrow("The option guessing must be below 1, found 1.");
    });

    it("rejects bundles that are not an object of lists of known items without repeats", () => {
      expect(() => analyze({ bundles: null })).toThrow(
        "The option bundles must be an object of item ids by bundle, found null.",
      );
      expect(() => analyze({ bundles: ["item1"] })).toThrow(
        "The option bundles must be an object of item ids by bundle, found an array.",
      );
      expect(() => analyze({ bundles: { first: "item1" } })).toThrow(
        'The bundle "first" must be an array of item ids, found "item1".',
      );
      expect(() => analyze({ bundles: { first: [] } })).toThrow('The bundle "first" needs at least one item.');
      expect(() => analyze({ bundles: { first: ["item1", "nope"] } })).toThrow(
        'The bundle "first" names the unknown item "nope".',
      );
      expect(() => analyze({ bundles: { first: [1] } })).toThrow('The bundle "first" names the unknown item 1.');
      expect(() => analyze({ bundles: { first: ["item1", "item1"] } })).toThrow(
        'The bundle "first" names the item "item1" twice.',
      );
    });
  });
});

describe("options and groups", () => {
  const matrix = buildResponseMatrix(CONSTANT_ROWS);

  it("reject a purify or continuity correction option that is not a boolean", () => {
    const purify: boolean = JSON.parse('"yes"');
    expect(() => mantelHaenszel(matrix, CONSTANT_GROUPS, { purify })).toThrow(
      'The option purify must be a boolean, found "yes".',
    );
    expect(() => logisticRegressionDif(matrix, CONSTANT_GROUPS, { purify })).toThrow(
      'The option purify must be a boolean, found "yes".',
    );
    expect(() => mantelHaenszel(matrix, CONSTANT_GROUPS, { continuityCorrection: JSON.parse("0") })).toThrow(
      "The option continuityCorrection must be a boolean, found 0.",
    );
  });

  it("reject groups and options of other types in the polytomous statistics", () => {
    const purify: boolean = JSON.parse('"yes"');
    const analyzes: ((options: PurificationOptions) => unknown)[] = [
      (options) => mantelTest(matrix, CONSTANT_GROUPS, options),
      (options) => ordinalLogisticRegressionDif(matrix, CONSTANT_GROUPS, options),
    ];
    for (const analyze of analyzes) {
      expect(() => analyze({ purify })).toThrow('The option purify must be a boolean, found "yes".');
      expect(() => analyze(JSON.parse("null"))).toThrow("The options must be an object, found null.");
      expect(() => analyze(JSON.parse('{"verbose":1}'))).toThrow('There is no option "verbose"');
    }
    expect(() => mantelTest(matrix, JSON.parse("null"))).toThrow("The groups must be an array, found null.");
    expect(() => ordinalLogisticRegressionDif(matrix, JSON.parse('"focal"'))).toThrow(
      'The groups must be an array, found "focal".',
    );
    expect(() => mantelTest(matrix, CONSTANT_GROUPS.slice(1))).toThrow(DataError);
  });

  it("reject a group other than reference or focal, and groups, options and models of other types", () => {
    const groups: DifGroup[] = JSON.parse(JSON.stringify(CONSTANT_GROUPS.map((_, i) => (i === 0 ? "both" : "focal"))));
    expect(() => mantelHaenszel(matrix, groups)).toThrow(
      'The group of person "person1" must be "reference" or "focal", found "both".',
    );
    expect(() => lordChiSquare(matrix, CONSTANT_GROUPS, { model: JSON.parse("null") })).toThrow(
      "The option model must be the 2PL, 3PL or 4PL model, found null.",
    );
    expect(() => mantelHaenszel(matrix, JSON.parse("null"))).toThrow("The groups must be an array, found null.");
    expect(() => lordChiSquare(matrix, CONSTANT_GROUPS, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => lordChiSquare(matrix, CONSTANT_GROUPS, JSON.parse('{"verbose":1}'))).toThrow(
      'There is no option "verbose"',
    );
    expect(() => logisticRegressionDif(matrix, CONSTANT_GROUPS, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => logisticRegressionDif(matrix, CONSTANT_GROUPS, JSON.parse('{"verbose":1}'))).toThrow(
      'There is no option "verbose"',
    );
    expect(() => logisticRegressionDif(matrix, CONSTANT_GROUPS, { purify: JSON.parse("null") })).toThrow(
      "The option purify must be a boolean, found null.",
    );
    expect(() =>
      lordChiSquare(matrix, CONSTANT_GROUPS, { model: twoParameterLogistic, purify: JSON.parse('"yes"') }),
    ).toThrow('The option purify must be a boolean, found "yes".');
  });
});

describe("the recipe for several focal groups", () => {
  const random = createRandom(2026);
  const items = simulateItems(twoParameterLogistic, 10, random);
  const harder = (shifts: Readonly<Record<number, number>>) =>
    items.map((item, j) => ({ ...item, parameters: { ...item.parameters, b: item.parameters.b + (shifts[j] ?? 0) } }));
  // Item 5 is one point harder for both focal groups, and item 8 half a point harder for the second.
  const populations = { reference: items, first: harder({ 4: 1 }), second: harder({ 4: 1, 7: 0.5 }) };
  const responses = Object.fromEntries(
    Object.entries(populations).map(([name, population]) => [
      name,
      simulateResponses(population, simulateThetas(500, random), random).responses,
    ]),
  );

  /** The ids of the items that Mantel-Haenszel flags for a focal group against the reference group, Bonferroni over the 20 tests. */
  const flagged = (focal: "first" | "second"): string[] => {
    const matrix = buildResponseMatrix([...(responses.reference ?? []), ...(responses[focal] ?? [])]);
    const groups: DifGroup[] = matrix.personIds.map((_, person) => (person < 500 ? "reference" : "focal"));
    return mantelHaenszel(matrix, groups)
      .items.filter((item) => (item.significance ?? 1) < 0.05 / 20)
      .map((item) => item.id);
  };

  it("flags the item that is harder for both focal groups against the reference group in each comparison", () => {
    expect(flagged("first")).toContain("item5");
    expect(flagged("second")).toContain("item5");
  });

  it("flags the item that is harder for the second focal group alone in its comparison", () => {
    expect(flagged("second")).toContain("item8");
    expect(flagged("first")).not.toContain("item8");
  });
});

describe("the recipe for standardization", () => {
  for (const dataset of fixture.cases) {
    it(`equals difR's difStd with the focal group's weights on ${dataset.name} data`, () => {
      const expected = recipeValues(dataset.name).standardization;
      const { items } = mantelTest(buildResponseMatrix(dataset.responses), groups(dataset.focal));
      items.forEach((item, i) => {
        expectClose(item.standardizedMeanDifference, expected[i] ?? null, RECIPE_TOLERANCE, `item ${i + 1}`);
      });
    });
  }

  it("gives the values of the guide's sample", () => {
    const { matrix, personGroups } = guideSample(15, { 2: 1 }, 0);
    const differences = mantelTest(matrix, personGroups).items.slice(0, 4);
    expect(differences.map((item) => Number(defined(item.standardizedMeanDifference).toFixed(3)))).toEqual([
      0.009, -0.009, -0.111, -0.002,
    ]);
  });
});

describe("the recipe for DIF variance", () => {
  /** The DIF variance of the items' MH D-DIF, and each estimate shrunken towards their mean, as the guide writes them. */
  const difVariance = (matrix: ResponseMatrix, personGroups: DifGroup[]) => {
    const estimates = mantelHaenszel(matrix, personGroups).items.flatMap(({ delta, deltaStandardError }) =>
      delta === null || deltaStandardError === null ? [] : [{ delta, standardError: deltaStandardError }],
    );
    const center = mean(estimates.map(({ delta }) => delta));
    const weights = estimates.map(({ standardError }) => standardError ** -2);
    const excess = sum(estimates.map(({ delta }, i) => (weights[i] ?? 0) ** 2 * (delta - center) ** 2)) - sum(weights);
    const variance = Math.max(excess / sum(weights.map((weight) => weight ** 2)), 0);
    const shrunken = estimates.map(({ delta, standardError }) => {
      const shrinkage = variance / (variance + standardError ** 2);
      return shrinkage * delta + (1 - shrinkage) * center;
    });
    return { estimates, standardDeviation: Math.sqrt(variance), shrunken };
  };

  for (const dataset of fixture.cases) {
    it(`equals sirt's dif.variance on ${dataset.name} data`, () => {
      const expected = recipeValues(dataset.name).variance;
      const result = difVariance(buildResponseMatrix(dataset.responses), groups(dataset.focal));
      expectClose(result.standardDeviation, expected.standardDeviation, RECIPE_TOLERANCE, "DIF standard deviation");
      result.shrunken.forEach((shrunken, i) => {
        expectClose(shrunken, expected.shrunken[i] ?? null, RECIPE_TOLERANCE, `item ${i + 1}`);
      });
    });
  }

  it("gives the values of the guide's sample", () => {
    const { matrix, personGroups } = guideSample(15, { 2: 1 }, 0);
    const result = difVariance(matrix, personGroups);
    expect(shown(result.standardDeviation, 3)).toBe(0.361);
    expect([shown(result.shrunken[2], 2), shown(result.estimates[2]?.delta, 3)]).toEqual([-0.86, -2.29]);
  });
});

describe("the recipe for power by focal group size", () => {
  it("finds an item of size C more often in a larger focal group, and one of size B near its bound about half the time", () => {
    const random = createRandom(2026);
    const items = simulateItems(twoParameterLogistic, 20, random);
    const focalItems = items.map((item, j) => {
      const shift = j === 0 ? 1 : j === 1 ? 0.5 : 0;
      return { ...item, parameters: { ...item.parameters, b: item.parameters.b + shift } };
    });
    const found = (focal: number) => {
      const flagged = [0, 0];
      for (let replication = 0; replication < 100; replication++) {
        const rows = [
          ...simulateResponses(items, simulateThetas(700, random), random).responses,
          ...simulateResponses(focalItems, simulateThetas(focal, random), random).responses,
        ];
        const groups: DifGroup[] = rows.map((_, person) => (person < 700 ? "reference" : "focal"));
        const result = mantelHaenszel(buildResponseMatrix(rows), groups).items;
        for (const j of [0, 1]) if (result[j]?.etsCategory !== "A") flagged[j] = (flagged[j] ?? 0) + 1;
      }
      return flagged.map((count) => count / 100);
    };
    expect(found(100)).toEqual([0.9, 0.44]);
    expect(found(200)).toEqual([0.99, 0.58]);
    expect(found(500)).toEqual([1, 0.58]);
    // The sizes the comment of the guide states, from 20,000 persons a group.
    const large = createRandom(1);
    const rows = [
      ...simulateResponses(items, simulateThetas(20_000, large), large).responses,
      ...simulateResponses(focalItems, simulateThetas(20_000, large), large).responses,
    ];
    const groups: DifGroup[] = rows.map((_, person) => (person < 20_000 ? "reference" : "focal"));
    const [first, second] = mantelHaenszel(buildResponseMatrix(rows), groups).items;
    expect([first?.etsCategory, second?.etsCategory]).toEqual(["C", "B"]);
    expectClose(defined(first?.delta), -2.1, 0.05, "MH D-DIF of item 1");
    expectClose(defined(second?.delta), -1.1, 0.05, "MH D-DIF of item 2");
  });
});

describe("properties", () => {
  it("IRT likelihood-ratio chi-squares do not change when the groups swap", () => {
    const { matrix, groups: caseGroups } = fixtureCase(irtFixture.likelihoodRatio["verbal, 2PL"].input);
    const subset = buildResponseMatrix(matrix.responses.map((row) => row.slice(0, 6)));
    const swapped = caseGroups.map((group): DifGroup => (group === "focal" ? "reference" : "focal"));
    const options = { model: twoParameterLogistic, tolerance: 1e-8, maxIterations: 20_000 };
    const forward = irtLikelihoodRatioDif(subset, caseGroups, options).items;
    irtLikelihoodRatioDif(subset, swapped, options).items.forEach((item, i) => {
      expectClose(item.chiSquare, forward[i]?.chiSquare ?? null, LIKELIHOOD_RATIO_TOLERANCE);
    });
  });

  it("swapping the groups inverts the Liu-Agresti odds ratio, and so does reversing the categories", () => {
    const rows = fc.tuple(fc.integer({ min: 4, max: 30 }), fc.integer({ min: 1, max: 4 })).chain(([persons, items]) =>
      fc.array(fc.array(fc.integer({ min: 0, max: 3 }), { minLength: items, maxLength: items }), {
        minLength: persons,
        maxLength: persons,
      }),
    );
    fc.assert(
      fc.property(rows, fc.array(fc.boolean(), { minLength: 30, maxLength: 30 }), (matrixRows, focal) => {
        const rowGroups = matrixRows.map((_, i): DifGroup => (focal[i] === true ? "focal" : "reference"));
        const swapped = rowGroups.map((group): DifGroup => (group === "focal" ? "reference" : "focal"));
        const reversed = matrixRows.map((row) => row.map((response) => 3 - response));
        const forward = mantelTest(buildResponseMatrix(matrixRows), rowGroups).items;
        const others = [
          mantelTest(buildResponseMatrix(matrixRows), swapped).items,
          mantelTest(buildResponseMatrix(reversed), rowGroups).items,
        ];
        forward.forEach(({ liuAgresti }, i) => {
          for (const other of others) {
            const inverse = other[i]?.liuAgresti;
            if (!liuAgresti || !inverse || liuAgresti.oddsRatio === 0 || inverse.oddsRatio === 0) continue;
            expect(liuAgresti.oddsRatio * inverse.oddsRatio).toBeCloseTo(1, 10);
          }
        });
      }),
    );
  });

  it("identical items have no differential functioning", () => {
    const population = fc
      .array(fc.tuple(fc.double({ min: -4, max: 4, noNaN: true }), fc.double({ min: 0.01, max: 1, noNaN: true })), {
        minLength: 1,
        maxLength: 8,
      })
      .map((pairs) => {
        const total = pairs.reduce((sum, [, weight]) => sum + weight, 0);
        return { nodes: pairs.map(([node]) => node), weights: pairs.map(([, weight]) => weight / total) };
      });
    fc.assert(
      fc.property(fc.array(arbitraryItem, { minLength: 1, maxLength: 5 }), population, (items, distribution) => {
        const tested = items.map((item, i) => ({ ...item, id: `item${i + 1}` }));
        const result = differentialFunctioning(tested, tested, distribution);
        expect(result.dtf).toBe(0);
        for (const item of result.items) expect([item.ncdif, item.cdif]).toEqual([0, 0]);
      }),
    );
  });

  /** Response matrices of 0 and 1: 2 to 30 persons, 1 to 6 items. */
  const binaryRows = fc
    .tuple(fc.integer({ min: 2, max: 30 }), fc.integer({ min: 1, max: 6 }))
    .chain(([persons, items]) =>
      fc.array(fc.array(fc.integer({ min: 0, max: 1 }), { minLength: items, maxLength: items }), {
        minLength: persons,
        maxLength: persons,
      }),
    );

  it("finds no DIF between identical groups", () => {
    fc.assert(
      fc.property(binaryRows, (rows) => {
        const matrix = buildResponseMatrix([...rows, ...rows]);
        const groups: DifGroup[] = [...rows.map(() => "reference" as const), ...rows.map(() => "focal" as const)];
        for (const item of mantelHaenszel(matrix, groups).items) {
          if (item.chiSquare === null) continue;
          expect(item.chiSquare).toBe(0);
          expect(item.oddsRatio).toBe(1);
          expect(item.etsCategory).toBe("A");
        }
        for (const item of logisticRegressionDif(matrix, groups).items) {
          if (item.both.chiSquare === null) continue;
          expect(item.both.chiSquare).toBeLessThan(1e-9);
        }
      }),
    );
  });

  it("Mantel-Haenszel statistics do not depend on the order of persons", () => {
    fc.assert(
      fc.property(binaryRows, fc.array(fc.boolean(), { minLength: 30, maxLength: 30 }), (rows, focal) => {
        const groups: DifGroup[] = rows.map((_, i) => (focal[i] === true || i === 0 ? "focal" : "reference"));
        groups[1] = "reference";
        const forward = mantelHaenszel(buildResponseMatrix(rows), groups).items;
        const backward = mantelHaenszel(buildResponseMatrix([...rows].reverse()), [...groups].reverse()).items;
        forward.forEach((item, i) => {
          expectClose(backward[i]?.chiSquare ?? null, item.chiSquare, 1e-12 * Math.max(1, item.chiSquare ?? 0));
          expect(backward[i]?.etsCategory).toBe(item.etsCategory);
        });
      }),
    );
  });

  /** Response matrices of 0 to 3: 2 to 40 persons, 1 to 5 items. */
  const gradedRows = fc
    .tuple(fc.integer({ min: 2, max: 40 }), fc.integer({ min: 1, max: 5 }))
    .chain(([persons, items]) =>
      fc.array(fc.array(fc.integer({ min: 0, max: 3 }), { minLength: items, maxLength: items }), {
        minLength: persons,
        maxLength: persons,
      }),
    );

  it("finds no polytomous DIF between identical groups", () => {
    fc.assert(
      fc.property(gradedRows, (rows) => {
        const matrix = buildResponseMatrix([...rows, ...rows]);
        const groups: DifGroup[] = [...rows.map(() => "reference" as const), ...rows.map(() => "focal" as const)];
        for (const item of mantelTest(matrix, groups).items) {
          if (item.chiSquare !== null) expect(item.chiSquare).toBe(0);
          if (item.generalized.chiSquare !== null) expect(item.generalized.chiSquare).toBeLessThan(1e-9);
          if (item.standardizedMeanDifference !== null) expect(item.standardizedMeanDifference).toBe(0);
          if (item.etsCategory !== null) expect(item.etsCategory).toBe("AA");
        }
        for (const item of ordinalLogisticRegressionDif(matrix, groups).items) {
          if (item.both.chiSquare !== null) expect(item.both.chiSquare).toBeLessThan(1e-6);
        }
      }),
    );
  });

  it("gives chi-squares from 0 and significances from 0 to 1 for any responses", () => {
    fc.assert(
      fc.property(gradedRows, (rows) => {
        const groups: DifGroup[] = rows.map((_, person) => (person % 2 === 0 ? "focal" : "reference"));
        const matrix = buildResponseMatrix(rows);
        const probabilities: (number | null)[] = [];
        for (const item of mantelTest(matrix, groups).items) {
          expect(item.chiSquare ?? 0).toBeGreaterThanOrEqual(0);
          expect(item.generalized.chiSquare ?? 0).toBeGreaterThanOrEqual(0);
          probabilities.push(item.significance, item.generalized.significance);
        }
        for (const item of ordinalLogisticRegressionDif(matrix, groups).items) {
          for (const test of [item.uniform, item.nonuniform, item.both]) {
            expect(test.chiSquare ?? 0).toBeGreaterThanOrEqual(0);
            probabilities.push(test.significance);
          }
        }
        for (const probability of probabilities) {
          expect(probability ?? 0.5).toBeGreaterThanOrEqual(0);
          expect(probability ?? 0.5).toBeLessThanOrEqual(1);
        }
      }),
    );
  });

  it("Mantel statistics do not depend on the order of persons", () => {
    fc.assert(
      fc.property(gradedRows, fc.array(fc.boolean(), { minLength: 40, maxLength: 40 }), (rows, focal) => {
        const groups: DifGroup[] = rows.map((_, i) => (focal[i] === true || i === 0 ? "focal" : "reference"));
        groups[1] = "reference";
        const forward = mantelTest(buildResponseMatrix(rows), groups).items;
        const backward = mantelTest(buildResponseMatrix([...rows].reverse()), [...groups].reverse()).items;
        forward.forEach((item, i) => {
          const other = backward[i];
          expectClose(other?.chiSquare ?? null, item.chiSquare, 1e-12 * Math.max(1, item.chiSquare ?? 0));
          expectClose(
            other?.generalized.chiSquare ?? null,
            item.generalized.chiSquare,
            1e-9 * Math.max(1, item.generalized.chiSquare ?? 0),
          );
          expectClose(other?.standardizedMeanDifference ?? null, item.standardizedMeanDifference, 1e-12);
          expect(other?.etsCategory).toBe(item.etsCategory);
        });
      }),
    );
  });

  /** Responses of 0 to 2 of enough persons that the middle score groups are used: 40 to 80 persons, 3 to 6 items. */
  const crowdedRows = fc
    .tuple(fc.integer({ min: 40, max: 80 }), fc.integer({ min: 3, max: 6 }))
    .chain(([persons, items]) =>
      fc.array(fc.array(fc.integer({ min: 0, max: 2 }), { minLength: items, maxLength: items }), {
        minLength: persons,
        maxLength: persons,
      }),
    );
  const alternate = (rows: readonly unknown[]): DifGroup[] =>
    rows.map((_, person) => (person % 2 === 0 ? "focal" : "reference"));

  it("negates SIBTEST's beta and keeps its other statistics when the groups are swapped", () => {
    let computed = 0;
    fc.assert(
      fc.property(crowdedRows, (rows) => {
        const matrix = buildResponseMatrix(rows);
        const personGroups = alternate(rows);
        const swapped = personGroups.map((group): DifGroup => (group === "focal" ? "reference" : "focal"));
        const other = sibtest(matrix, swapped, { smallCell: 2 }).items;
        sibtest(matrix, personGroups, { smallCell: 2 }).items.forEach((item, i) => {
          if (item.beta !== null) computed++;
          expectClose(other[i]?.beta ?? null, item.beta === null ? null : -item.beta, 1e-9);
          expectClose(other[i]?.standardError ?? null, item.standardError, 1e-9);
          expectClose(other[i]?.chiSquare ?? null, item.chiSquare, 1e-9 * Math.max(1, item.chiSquare ?? 0));
        });
      }),
    );
    expect(computed).toBeGreaterThan(0);
  });

  it("gives a bundle of one item the statistics of that item", () => {
    fc.assert(
      fc.property(crowdedRows, fc.boolean(), (rows, purify) => {
        const matrix = buildResponseMatrix(rows);
        const bundles = Object.fromEntries(matrix.itemIds.map((id) => [`bundle of ${id}`, [id]]));
        const result = sibtest(matrix, alternate(rows), { smallCell: 2, bundles, purify });
        result.items.forEach((item, i) => {
          expect(result.bundles[i]).toEqual({ ...item, id: `bundle of ${item.id}` });
        });
      }),
    );
  });

  it("SIBTEST statistics do not depend on the order of persons", () => {
    fc.assert(
      fc.property(crowdedRows, (rows) => {
        const personGroups = alternate(rows);
        const forward = sibtest(buildResponseMatrix(rows), personGroups, { smallCell: 2 }).items;
        const backward = sibtest(buildResponseMatrix([...rows].reverse()), [...personGroups].reverse(), {
          smallCell: 2,
        }).items;
        forward.forEach((item, i) => {
          expectClose(backward[i]?.beta ?? null, item.beta, 1e-9);
          expectClose(backward[i]?.chiSquare ?? null, item.chiSquare, 1e-9 * Math.max(1, item.chiSquare ?? 0));
        });
      }),
    );
  });

  it("finds no SIBTEST DIF between identical groups", () => {
    fc.assert(
      fc.property(crowdedRows, (rows) => {
        const matrix = buildResponseMatrix([...rows, ...rows]);
        const personGroups: DifGroup[] = [...rows.map(() => "reference" as const), ...rows.map(() => "focal" as const)];
        for (const item of sibtest(matrix, personGroups, { smallCell: 2 }).items) {
          if (item.beta !== null) expect(Math.abs(item.beta)).toBeLessThan(1e-12);
          if (item.significance !== null) expect(item.significance).toBeCloseTo(1, 9);
        }
      }),
    );
  });

  it("gives SIBTEST's chi-squares from 0 and significances from 0 to 1 for any responses", () => {
    fc.assert(
      fc.property(crowdedRows, fc.boolean(), (rows, correction) => {
        const { items } = sibtest(buildResponseMatrix(rows), alternate(rows), { smallCell: 2, correction });
        for (const item of items) {
          expect(item.chiSquare ?? 0).toBeGreaterThanOrEqual(0);
          expect(item.standardError ?? 0).toBeGreaterThanOrEqual(0);
          expect(item.significance ?? 0.5).toBeGreaterThanOrEqual(0);
          expect(item.significance ?? 0.5).toBeLessThanOrEqual(1);
        }
      }),
    );
  });
});
