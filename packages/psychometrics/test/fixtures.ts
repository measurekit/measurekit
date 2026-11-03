import { readFileSync } from "node:fs";

const FIXTURES = new URL("./fixtures/", import.meta.url);

/**
 * Reads a JSON file of `test/fixtures/` as the type `T`, which the test declares for the fields it reads. A test never
 * imports a fixture: the type checker builds a type for every element of an imported JSON file, and the fixtures of a
 * package add up to hundreds of thousands of types.
 */
export function readFixture<T>(name: string): T {
  return JSON.parse(readFileSync(new URL(name, FIXTURES), "utf8"));
}

export interface BenchCdmFixture {
  itemIds: string[];
  responses: number[][];
  qMatrix: Record<string, string[]>;
}

export interface Cat3plFixture {
  source: string;
  items: { id: string; a: number; b: number; c: number; d: number }[];
  peaks: number[];
  runs: {
    name: string;
    selection: string;
    estimators: string[];
    steps: {
      method: string | null;
      theta: number;
      standardError: number | null;
      nextItem: string;
      response: number;
      scores?: (number | null)[];
    }[];
    final: { method: string; theta: number; standardError: number };
  }[];
  stoppingRules: { rule: string; threshold: number }[];
  stopping: { administered: number; theta: number; standardError: number; decisions: boolean[]; any: boolean }[];
}

export interface DimensionalityFactorsFixture {
  source: string;
  truth: { loadings: number[]; correlation: number };
  simulated: { itemIds: string[]; responses: number[][] };
  cases: {
    name: string;
    dataset: string;
    correlation: string;
    factors: number;
    rotation: string;
    matrix: number[][];
    base: { communalities: number[]; unrotated: number[][]; loadings: number[][]; phi: number[][] };
    psych: { loadings: number[][]; phi: number[][]; communalities: number[] };
    map: { psych: number[]; base: number[] } | null;
  }[];
  smoothing: { name: string; raw: number[][]; smoothed: number[][] }[];
}

export interface BenchIrtFixture {
  binary: { itemIds: string[]; responses: number[][] };
  three: { itemIds: string[]; responses: number[][] };
  graded: { itemIds: string[]; responses: number[][] };
  testlet: {
    itemIds: string[];
    responses: number[][];
    testlets: { t1: string[]; t2: string[]; t3: string[]; t4: string[]; t5: string[]; t6: string[] };
    points: number;
  };
  items: { a: number[]; b: number[] };
}

export interface BenchIrtConditionalFixture {
  rasch: { itemIds: string[]; responses: number[][] };
  partialCredit: { itemIds: string[]; responses: number[][] };
}

export interface FitResidualsFixture {
  cases: {
    name: string;
    responses: (number | null)[][];
    items: { model: string; parameters: { a: number; b: number | number[]; c?: number; d?: number } }[];
    thetas: number[];
    standardizedLogLikelihood: number[];
    q3: number[][];
  }[];
}

// biome-ignore-start lint/style/useNamingConvention: the keys are those of the fixture file.
export interface RaterAutomatedScoringFixture {
  cases: {
    "PERSUADE 2.0": {
      scale: number[];
      human: number[];
      machine: number[];
      groups: string[];
      overall: {
        kappa: number;
        quadraticKappa: number;
        exactPercent: number;
        adjacentPercent: number;
        smd: number;
        correlation: number;
        rSquared: number;
        rmse: number;
        machineSd: number;
        humanSd: number;
      };
      byGroup: { group: string; subjects: number; smd: number }[];
    };
    ELLIPSE: {
      scale: number[];
      first: number[];
      second: number[];
      machine: number[];
      humanSystem: {
        kappa: number;
        quadraticKappa: number;
        exactPercent: number;
        adjacentPercent: number;
        smd: number;
        correlation: number;
        rSquared: number;
        rmse: number;
        machineSd: number;
        humanSd: number;
      };
      humanHuman: {
        kappa: number;
        quadraticKappa: number;
        exactPercent: number;
        adjacentPercent: number;
        smd: number;
        correlation: number;
        rSquared: number;
        rmse: number;
        machineSd: number;
        humanSd: number;
      };
      degradation: {
        correlation: number;
        kappa: number;
        quadraticKappa: number;
        exactPercent: number;
        adjacentPercent: number;
      };
      disattenuated: number;
    };
  };
}
// biome-ignore-end lint/style/useNamingConvention: the keys are those of the fixture file.

export interface IrtFixedParametersFixture {
  source: string;
  itemIds: string[];
  responses: number[][];
  models: string[];
  fixedParameters: {
    item1: { c: number };
    item2: { c: number };
    item3: { c: number };
    item4: { c: number };
    item5: { c: number };
    item6: { c: number };
    item12: { a: number };
  };
  calibrations: {
    standardErrors: string;
    items: { parameters: { a: number; b: number; c: number; d: number }; standardErrors: { a?: number; b: number } }[];
    logLikelihood: number;
    aic: number;
    bic: number;
    estimated: number;
  }[];
  summedScore: { chiSquare: number[]; degreesOfFreedom: number[]; significance: number[] };
}
