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

export interface BenchBktFixture {
  learner: string[];
  skill: string[];
  item: string[];
  time: number[];
  outcome: number[];
}

export interface BenchEloFixture {
  learner: string[];
  item: string[];
  time: number[];
  outcome: number[];
}

export interface EloHoldoutFixture {
  events: { learner: string; item: string; skills: string[]; time: number; outcome: number }[];
  timeBounds: number[];
  folds: { name: string; train: number[]; test: number[]; predictions: { basic: number[]; skills: number[] } }[];
}

// biome-ignore-start lint/style/useNamingConvention: the keys are those of the fixture file.
export interface StoredStateFixture {
  log: { learner: string; item: string; skills: string[]; time: number; outcome: number | null }[];
  models: {
    bkt: {
      options: { forgetting: boolean; resourceLearning: boolean };
      parameters: {
        skills: {
          fractions: {
            prior: number;
            learn: number;
            forget: number;
            guess: number;
            slip: number;
            resourceLearning: { video: { learn: number; forget: number } };
          };
          decimals: {
            prior: number;
            learn: number;
            forget: number;
            guess: number;
            slip: number;
            resourceLearning: { video: { learn: number; forget: number } };
          };
        };
      };
      states: Record<string, { learner: string; skills: { fractions: number; decimals: number } }>;
      attempt: { learner: string; item: string; skills: string[]; time: number };
      prediction: number;
      answered: {
        event: { learner: string; item: string; skills: string[]; time: number; outcome: number };
        state: { learner: string; skills: { fractions: number; decimals: number } };
      };
      lesson: {
        event: { learner: string; item: string; skills: string[]; time: number; outcome: null };
        state: { learner: string; skills: { fractions: number; decimals: number } };
      };
    };
    elo: {
      options: { skills: boolean };
      parameters: {
        items: {
          q2: { difficulty: number; count: number };
          q3: { difficulty: number; count: number };
          q4: { difficulty: number; count: number };
          q1: { difficulty: number; count: number };
        };
      };
      states: Record<
        string,
        {
          learner: string;
          ability: number;
          count: number;
          skills: { decimals: { ability: number; count: number }; fractions: { ability: number; count: number } };
        }
      >;
      attempt: { learner: string; item: string; skills: string[]; time: number };
      prediction: number;
      answered: {
        event: { learner: string; item: string; skills: string[]; time: number; outcome: number };
        state: {
          learner: string;
          ability: number;
          count: number;
          skills: { decimals: { ability: number; count: number }; fractions: { ability: number; count: number } };
        };
      };
      lesson: {
        event: { learner: string; item: string; skills: string[]; time: number; outcome: null };
        state: {
          learner: string;
          ability: number;
          count: number;
          skills: { decimals: { ability: number; count: number }; fractions: { ability: number; count: number } };
        };
      };
    };
    irt: {
      options: {
        parameters: {
          items: {
            q1: { a: number; b: number };
            q2: { a: number; b: number };
            q3: { a: number; b: number };
            q4: { a: number; b: number };
          };
        };
      };
      parameters: {
        items: {
          q1: { a: number; b: number };
          q2: { a: number; b: number };
          q3: { a: number; b: number };
          q4: { a: number; b: number };
        };
      };
      states: Record<string, { learner: string; logDensity: number[]; slope: number[]; curvature: number[] }>;
      attempt: { learner: string; item: string; skills: string[]; time: number };
      prediction: number;
      answered: {
        event: { learner: string; item: string; skills: string[]; time: number; outcome: number };
        state: { learner: string; logDensity: number[]; slope: number[]; curvature: number[] };
      };
      lesson: {
        event: { learner: string; item: string; skills: string[]; time: number; outcome: null };
        state: { learner: string; logDensity: number[]; slope: number[]; curvature: number[] };
      };
    };
    kst: {
      options: { structure: { domain: string[]; states: string[][] } };
      parameters: {
        probabilities: number[];
        items: {
          q2: { guess: number; slip: number };
          q3: { guess: number; slip: number };
          q4: { guess: number; slip: number };
          q1: { guess: number; slip: number };
        };
      };
      states: Record<string, { learner: string; probabilities: number[] }>;
      attempt: { learner: string; item: string; skills: string[]; time: number };
      prediction: number;
      answered: {
        event: { learner: string; item: string; skills: string[]; time: number; outcome: number };
        state: { learner: string; probabilities: number[] };
      };
      lesson: {
        event: { learner: string; item: string; skills: string[]; time: number; outcome: null };
        state: { learner: string; probabilities: number[] };
      };
    };
    logistic: {
      options: { features: (string | { decayedExposures?: number; recencyExposure?: number })[] };
      parameters: {
        skills: {
          decimals: {
            intercept: number;
            coefficients: {
              opportunities: number;
              successes: number;
              exposures: number;
              "decayedExposures(0.7)": number;
              "recencyExposure(0.5)": number;
            };
          };
          fractions: {
            intercept: number;
            coefficients: {
              opportunities: number;
              successes: number;
              exposures: number;
              "decayedExposures(0.7)": number;
              "recencyExposure(0.5)": number;
            };
          };
        };
      };
      states: Record<
        string,
        {
          learner: string;
          skills: {
            decimals: { successes: number; failures: number; last: number; decayed: Record<string, never> };
            fractions: { successes: number; failures: number; last: number; decayed: Record<string, never> };
          };
          exposures: {
            fractions: { count: number; last: number; decayed: { "0.7": number } };
            decimals: { count: number; last: number; decayed: { "0.7": number } };
          };
        }
      >;
      attempt: { learner: string; item: string; skills: string[]; time: number };
      prediction: number;
      answered: {
        event: { learner: string; item: string; skills: string[]; time: number; outcome: number };
        state: {
          learner: string;
          skills: {
            decimals: { successes: number; failures: number; last: number; decayed: Record<string, never> };
            fractions: { successes: number; failures: number; last: number; decayed: Record<string, never> };
          };
          exposures: {
            fractions: { count: number; last: number; decayed: { "0.7": number } };
            decimals: { count: number; last: number; decayed: { "0.7": number } };
          };
        };
      };
      lesson: {
        event: { learner: string; item: string; skills: string[]; time: number; outcome: null };
        state: {
          learner: string;
          skills: {
            decimals: { successes: number; failures: number; last: number; decayed: Record<string, never> };
            fractions: { successes: number; failures: number; last: number; decayed: Record<string, never> };
          };
          exposures: {
            fractions: { count: number; last: number; decayed: { "0.7": number } };
            decimals: { count: number; last: number; decayed: { "0.7": number } };
          };
        };
      };
    };
  };
}
// biome-ignore-end lint/style/useNamingConvention: the keys are those of the fixture file.

export interface LogisticLktFeaturesFixture {
  events: { learner: string; item: string; skills: string[]; time: number; outcome: number; duration: number }[];
  features: {
    name: string;
    feature:
      | string
      | {
          powerOpportunities?: number;
          logit?: number;
          decayedProportionFailed?: number;
          recencySuccess?: number;
          recencyFailure?: number;
          base?: number;
          baseSuccess?: number;
          baseFailure?: number;
          basePractice?: { decay: number; scale: number };
          dash?: number;
          dashSuccess?: number;
          ppe?: { power: number; decay: number; spacing: number; weight: number };
          windowOpportunities?: number;
          windowSuccesses?: number;
          windowFailures?: number;
        };
    lkt: string | null;
    values: number[];
  }[];
}

export interface LogisticRatesFixture {
  bounds: { lower: number[]; upper: number[]; start: number[] };
  events: { learner: string; item: string; skills: string[]; time: number; outcome: number }[];
  rates: number[];
  logLikelihood: number;
  gradient: number[];
  recovery: { logs: number; mean: number[]; sd: number[] };
}
