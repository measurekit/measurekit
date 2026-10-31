import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  calibrate,
  calibrateMultidimensional,
  createRandom,
  DataError,
  type LoadingPattern,
  logNormalPrior,
  type MultidimensionalCalibration,
  type MultidimensionalGradedParameters,
  type MultidimensionalItem,
  type MultidimensionalLogisticParameters,
  type MultidimensionalModel,
  type MultidimensionalTestItem,
  multidimensionalEstimates,
  multidimensionalGradedResponse,
  multidimensionalInformation,
  multidimensionalPosteriorMeanEstimate,
  multidimensionalPosteriorModeEstimate,
  multidimensionalThreeParameterLogistic,
  multidimensionalTwoParameterLogistic,
  normalPrior,
  type ParameterPrior,
  posteriorMeanEstimate,
  posteriorModeEstimate,
  type RotationOptions,
  responseMatrix,
  responseRecords,
  rotateLoadings,
  twoParameterLogistic,
} from "../src/index.js";
import { compressResponses } from "../src/irt/responses.js";
import { setUp } from "../src/multidimensional/calibration-setup.js";
import { latentDistribution } from "../src/multidimensional/correlation.js";
import { designOf } from "../src/multidimensional/design.js";
import { productGrid } from "../src/multidimensional/grid.js";
import { gridLikelihood } from "../src/multidimensional/grid-likelihood.js";
import { createChain, responsesByItem, responsesByPerson } from "../src/multidimensional/mhrm-chain.js";
import {
  createCorrelations,
  latentObservedInformation,
  latentScore,
  personLatentScores,
  secondMoments,
} from "../src/multidimensional/mhrm-latent.js";
import { createScratch, itemMoments, observedItemInformation } from "../src/multidimensional/mhrm-moments.js";
import exploratory from "./fixtures/multidimensional-exploratory.json" with { type: "json" };
import graded from "./fixtures/multidimensional-graded.json" with { type: "json" };
import graded3 from "./fixtures/multidimensional-graded3.json" with { type: "json" };
import information from "./fixtures/multidimensional-information.json" with { type: "json" };
import m2pl from "./fixtures/multidimensional-m2pl.json" with { type: "json" };
import m3pl from "./fixtures/multidimensional-m3pl.json" with { type: "json" };
import mhrm from "./fixtures/multidimensional-mhrm.json" with { type: "json" };
import missing from "./fixtures/multidimensional-missing.json" with { type: "json" };
import rotation from "./fixtures/multidimensional-rotation.json" with { type: "json" };
import { defined, expectClose } from "./helpers.js";

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
  fixture: Fixture,
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
  return calibrateMultidimensional(responseMatrix(fixture.responses, { itemIds: [...fixture.itemIds] }), {
    model: modelOf(fixture.model),
    loadings: fixture.loadings,
    points: fixture.points,
    standardErrors,
    maxIterations: 2000,
    ...extra,
  });
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
        const fit = calibrateMultidimensional(responseMatrix(fixture.responses, { itemIds: fixture.itemIds }), {
          model: multidimensionalTwoParameterLogistic,
          dimensions: 2,
          points: fixture.points,
          tolerance: 1e-8,
          maxIterations: 20000,
        });
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
      responseMatrix(m2pl.responses.slice(0, 400), { itemIds: [...m2pl.itemIds] }),
      {
        model: multidimensionalTwoParameterLogistic,
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
    const fit = calibrateMultidimensional(responseMatrix(m2pl.responses, { itemIds: [...m2pl.itemIds] }), {
      model: multidimensionalTwoParameterLogistic,
      loadings: m2pl.loadings,
      correlated: false,
      points: 31,
      tolerance: 1e-6,
    });
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
    const data = responseMatrix(
      m2pl.responses.map((row) => row.slice(0, 6)),
      { itemIds: [...columns] },
    );
    const one = calibrateMultidimensional(data, {
      model: multidimensionalTwoParameterLogistic,
      dimensions: 1,
      standardErrors: "hessian",
      tolerance: 1e-10,
    });
    const reference = calibrate(data, { model: twoParameterLogistic, standardErrors: "hessian", tolerance: 1e-10 });
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
    const sparse = calibrateMultidimensional(responseRecords(records), {
      model: multidimensionalTwoParameterLogistic,
      loadings: missing.loadings,
      points: 15,
      tolerance: 1e-8,
    });
    const dense = calibrateMultidimensional(responseMatrix(missing.responses, { itemIds: [...missing.itemIds] }), {
      model: multidimensionalTwoParameterLogistic,
      loadings: missing.loadings,
      points: 15,
      tolerance: 1e-8,
    });
    expectClose(sparse.logLikelihood, dense.logLikelihood, 1e-8);
    dense.items.forEach((item, i) => {
      expectClose(defined(sparse.items[i]?.parameters.a[0]), defined(item.parameters.a[0]), 1e-8);
    });
  });

  it("ignores persons without responses", () => {
    const rows = [...missing.responses.slice(0, 300)];
    const base = calibrateMultidimensional(responseMatrix(rows, { itemIds: [...missing.itemIds] }), {
      model: multidimensionalTwoParameterLogistic,
      loadings: missing.loadings,
      points: 15,
      tolerance: 1e-9,
    });
    const padded = calibrateMultidimensional(
      responseMatrix([...rows, new Array<number | null>(12).fill(null)], { itemIds: [...missing.itemIds] }),
      { model: multidimensionalTwoParameterLogistic, loadings: missing.loadings, points: 15, tolerance: 1e-9 },
    );
    expect(padded.logLikelihood).toBe(base.logLikelihood);
    expect(padded.bic).toBe(base.bic);
  });

  it("estimates posterior modes with priors", () => {
    const columns = [...m2pl.itemIds];
    const data = responseMatrix(m2pl.responses.slice(0, 200), { itemIds: columns });
    const options = {
      model: multidimensionalTwoParameterLogistic,
      loadings: m2pl.loadings,
      points: 15,
      tolerance: 1e-8,
    };
    const plain = calibrateMultidimensional(data, options);
    const prior = calibrateMultidimensional(data, {
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
    const lognormal = calibrateMultidimensional(data, { ...options, priors: { a: logNormalPrior(0, 0.3) } });
    expect(lognormal.items.every(({ parameters }) => (parameters.a[0] ?? 0) >= 0)).toBe(true);
  });

  it("starts exploratory models from the principal components and uses the dimensions of each item once", () => {
    const fit = calibrateMultidimensional(
      responseMatrix(m2pl.responses.slice(0, 300), { itemIds: [...m2pl.itemIds] }),
      {
        model: multidimensionalTwoParameterLogistic,
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

  it("names the dimensions of a confirmatory model in order of first appearance among the items", () => {
    const data = responseMatrix(m2pl.responses.slice(0, 150), { itemIds: [...m2pl.itemIds] });
    const loadings: Record<string, string[]> = {};
    m2pl.itemIds.forEach((id, i) => {
      loadings[id] = i < 6 ? ["verbal"] : ["quantitative"];
    });
    loadings[defined(m2pl.itemIds[5])] = ["verbal", "quantitative"];
    const fit = calibrateMultidimensional(data, {
      model: multidimensionalTwoParameterLogistic,
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
      responseMatrix(graded.responses.slice(0, 300), { itemIds: [...graded.itemIds] }),
      {
        model: multidimensionalGradedResponse,
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
      responseMatrix(m2pl.responses.slice(0, 200), { itemIds: [...m2pl.itemIds] }),
      {
        model: multidimensionalTwoParameterLogistic,
        loadings: m2pl.loadings,
        points: 11,
        maxIterations: 3,
      },
    );
    expect(short.converged).toBe(false);
    expect(short.iterations).toBe(3);
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
    return calibrateMultidimensional(responseMatrix(fixture.responses, { itemIds: [...fixture.itemIds] }), {
      model: modelOf(fixture.model),
      loadings: fixture.loadings,
      method: "mhrm",
      ...extra,
    });
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
    const data = responseMatrix(rows, { itemIds: [...m2pl.itemIds] });
    const options = { model: multidimensionalTwoParameterLogistic, loadings: m2pl.loadings, method: "mhrm" } as const;

    it("gives the same calibration for the same seed and another for another seed", () => {
      const first = calibrateMultidimensional(data, { ...options, seed: 5, standardErrors: "hessian" });
      expect(calibrateMultidimensional(data, { ...options, seed: 5, standardErrors: "hessian" })).toEqual(first);
      const other = calibrateMultidimensional(data, { ...options, seed: 6 });
      expect(other.items[0]?.parameters).not.toEqual(first.items[0]?.parameters);
    });

    it("reports the cycles of its last stage and whether the tolerance was met", () => {
      const short = calibrateMultidimensional(data, { ...options, maxIterations: 3 });
      expect(short.converged).toBe(false);
      expect(short.iterations).toBe(3);
      const long = calibrateMultidimensional(data, { ...options, tolerance: 0.05 });
      expect(long.converged).toBe(true);
      expect(long.iterations).toBeGreaterThanOrEqual(3);
      expect(long.iterations).toBeLessThan(2000);
    });

    it("reports AIC and BIC from the same number of parameters as EM", () => {
      const fit = calibrateMultidimensional(data, { ...options, maxIterations: 5 });
      expectClose(fit.aic, -2 * fit.logLikelihood + 2 * 25, 1e-9);
      expectClose(fit.bic, -2 * fit.logLikelihood + 25 * Math.log(250), 1e-9);
    });

    it("fixes the correlations at 0 when they are not estimated and has none in one dimension", () => {
      const uncorrelated = calibrateMultidimensional(data, { ...options, correlated: false, maxIterations: 5 });
      expect(uncorrelated.correlations).toEqual([
        [1, 0],
        [0, 1],
      ]);
      expect(uncorrelated.correlationStandardErrors).toBeNull();
      const one = calibrateMultidimensional(
        responseMatrix(
          rows.map((row) => row.slice(0, 6)),
          { itemIds: m2pl.itemIds.slice(0, 6) },
        ),
        { model: multidimensionalTwoParameterLogistic, dimensions: 1, method: "mhrm", standardErrors: "hessian" },
      );
      expect(one.correlations).toEqual([[1]]);
      expect(one.items.every(({ standardErrors }) => standardErrors !== null)).toBe(true);
    });

    it("estimates exploratory models with the first items' later slopes fixed at 0", () => {
      const fit = calibrateMultidimensional(data, {
        model: multidimensionalTwoParameterLogistic,
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
      const plain = calibrateMultidimensional(data, { ...options, seed: 2 });
      const shrunk = calibrateMultidimensional(data, {
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
      const fit = calibrateMultidimensional(responseRecords(records), { ...options, maxIterations: 5 });
      expect(fit.items).toHaveLength(12);
      expect(fit.logLikelihood).toBeLessThan(0);
    });

    it("chooses EM up to three dimensions and MH-RM above", () => {
      // Only EM has a grid: the option points is accepted by the default method up to three dimensions and rejected above.
      expect(() =>
        calibrateMultidimensional(data, {
          model: multidimensionalTwoParameterLogistic,
          loadings: m2pl.loadings,
          points: 5,
          maxIterations: 1,
        }),
      ).not.toThrow();
      const four = Object.fromEntries(m2pl.itemIds.map((id, i) => [id, [`D${(i % 4) + 1}`]]));
      expect(() =>
        calibrateMultidimensional(data, { model: multidimensionalTwoParameterLogistic, loadings: four, points: 5 }),
      ).toThrow('The option points sets the grid of EM, which the method "mhrm" does not use.');
      const fit = calibrateMultidimensional(data, {
        model: multidimensionalTwoParameterLogistic,
        loadings: four,
        maxIterations: 3,
      });
      expect(fit.dimensions).toEqual(["D1", "D2", "D3", "D4"]);
      expect(fit.correlations).toHaveLength(4);
    });
  });

  it("gives the same calibration for the same seed, whatever the seed", () => {
    const data = responseMatrix(
      m2pl.responses.slice(0, 40).map((row) => row.slice(0, 6)),
      { itemIds: m2pl.itemIds.slice(0, 6) },
    );
    fc.assert(
      fc.property(fc.integer({ min: 0, max: 2 ** 31 }), (seed) => {
        const options = {
          model: multidimensionalTwoParameterLogistic,
          dimensions: 1,
          method: "mhrm",
          seed,
          maxIterations: 2,
        } as const;
        expect(calibrateMultidimensional(data, options)).toEqual(calibrateMultidimensional(data, options));
      }),
      { numRuns: 4 },
    );
  }, 120_000);
});

describe("MH-RM internals", () => {
  const items5 = mirtMhrm.itemIds;
  const responses = compressResponses(responseMatrix(mirtMhrm.responses, { itemIds: [...items5] }));

  it("computes the complete-data score and information of mirt's Deriv", () => {
    const design = designOf({ loadings: mirtMhrm.loadings }, items5);
    const states = setUp(responses, { model: multidimensionalGradedResponse, loadings: mirtMhrm.loadings }, design, 0);
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
          const compressed = compressResponses(responseMatrix(rows));
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
      responseMatrix(
        m2pl.responses.slice(0, 250).map((row) => row.slice(0, 6)),
        { itemIds },
      ),
    );
    const states = setUp(
      calibrated,
      { model: multidimensionalTwoParameterLogistic, loadings },
      designOf({ loadings }, itemIds),
      0,
    );
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
      responsesByItem(compressResponses(responseMatrix([pattern], { itemIds }))),
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
    const data = responseMatrix(
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
      const results = multidimensionalEstimates(data, items, multidimensionalPosteriorMeanEstimate);
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
      const dense = multidimensionalEstimates(data, items, estimator);
      const records = responseRecords([
        { person: "a", item: defined(m2pl.itemIds[0]), response: 1 },
        { person: "a", item: defined(m2pl.itemIds[7]), response: 0 },
        { person: "b", item: defined(m2pl.itemIds[3]), response: 1 },
      ]);
      const sparse = multidimensionalEstimates(records, items, estimator);
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
      expect(() => multidimensionalEstimates(data, items, JSON.parse("null"))).toThrow(DataError);
      expect(() => multidimensionalEstimates(data, [...items, defined(items[0])])).toThrow(
        `Item ids must be unique; "${m2pl.itemIds[0]}" appears twice.`,
      );
      expect(() => multidimensionalEstimates(data, JSON.parse("null"))).toThrow(
        "The items must be an array, found null.",
      );
    });
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

describe("errors", () => {
  const data = responseMatrix(m2pl.responses.slice(0, 40), { itemIds: [...m2pl.itemIds] });
  const loadings: LoadingPattern = m2pl.loadings;
  const model = multidimensionalTwoParameterLogistic;

  it("needs exactly one of dimensions and loadings", () => {
    expect(() => calibrateMultidimensional(data, { model })).toThrow(
      "Calibration needs the option dimensions, for an exploratory model, or loadings, for a confirmatory one.",
    );
    expect(() => calibrateMultidimensional(data, { model, dimensions: 2, loadings })).toThrow(
      "Calibration takes the option dimensions or the option loadings, not both.",
    );
  });

  it("limits EM to three dimensions and needs at least as many items as dimensions", () => {
    expect(() => calibrateMultidimensional(data, { model, method: "em", dimensions: 4 })).toThrow(
      "EM calibrates up to 3 dimensions, found 4.",
    );
    const names = Object.fromEntries(m2pl.itemIds.map((id, i) => [id, [`d${i % 4}`]]));
    expect(() => calibrateMultidimensional(data, { model, method: "em", loadings: names })).toThrow(
      "EM calibrates up to 3 dimensions, found 4.",
    );
    expect(() =>
      calibrateMultidimensional(
        responseMatrix([
          [0, 1],
          [1, 0],
          [1, 1],
        ]),
        { model, dimensions: 3 },
      ),
    ).toThrow("An exploratory model of 3 dimensions needs at least 3 items, found 2.");
  });

  it("rejects the method, the seed and the options that a method does not take", () => {
    const options = { model, loadings };
    expect(() => calibrateMultidimensional(data, { ...options, method: "mcmc" as never })).toThrow(
      'The option method must be "em" or "mhrm", found "mcmc".',
    );
    expect(() => calibrateMultidimensional(data, { ...options, method: JSON.parse("null") })).toThrow(DataError);
    expect(() => calibrateMultidimensional(data, { ...options, seed: 1.5 })).toThrow(
      "A seed must be a safe integer, found 1.5.",
    );
    expect(() => calibrateMultidimensional(data, { ...options, seed: JSON.parse('"7"') })).toThrow(DataError);
    expect(() => calibrateMultidimensional(data, { ...options, method: "mhrm", points: 15 })).toThrow(
      'The option points sets the grid of EM, which the method "mhrm" does not use.',
    );
    expect(() =>
      calibrateMultidimensional(data, { ...options, method: "mhrm", standardErrors: "cross-product" }),
    ).toThrow(
      'MH-RM has no posterior over a grid to cross-multiply: the option standardErrors needs "none" or "hessian" with the method "mhrm".',
    );
  });

  it("checks the pattern of loadings", () => {
    const first = defined(m2pl.itemIds[0]);
    expect(() => calibrateMultidimensional(data, { model, loadings: { ...loadings, nowhere: ["F1"] } })).toThrow(
      'Loadings are given for "nowhere", which is not an item id.',
    );
    const { [first]: _, ...without } = loadings;
    expect(() => calibrateMultidimensional(data, { model, loadings: without })).toThrow(
      `The loadings of item "${first}" must be an array of at least one dimension name, found undefined.`,
    );
    expect(() => calibrateMultidimensional(data, { model, loadings: { ...loadings, [first]: [] } })).toThrow(
      `The loadings of item "${first}" must be an array of at least one dimension name, found an array.`,
    );
    expect(() => calibrateMultidimensional(data, { model, loadings: { ...loadings, [first]: ["F1", "F1"] } })).toThrow(
      `Item "${first}" loads on a dimension twice: F1, F1.`,
    );
    expect(() => calibrateMultidimensional(data, { model, loadings: JSON.parse("null") })).toThrow(
      "The option loadings must be an object of dimension names by item id, found null.",
    );
    expect(() =>
      calibrateMultidimensional(data, { model, loadings: { ...loadings, [first]: JSON.parse('"F1"') } }),
    ).toThrow(DataError);
    expect(() =>
      calibrateMultidimensional(data, { model, loadings: { ...loadings, [first]: [JSON.parse("1")] } }),
    ).toThrow(DataError);
  });

  it("checks the other options", () => {
    const options = { model, loadings };
    expect(() => calibrateMultidimensional(data, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => calibrateMultidimensional(JSON.parse("null"), options)).toThrow(DataError);
    expect(() => calibrateMultidimensional(data, { ...options, dimensions: undefined, loadings: undefined })).toThrow(
      DataError,
    );
    expect(() => calibrateMultidimensional(data, { model, dimensions: 1.5 })).toThrow(DataError);
    expect(() => calibrateMultidimensional(data, { model, dimensions: JSON.parse("null") })).toThrow(DataError);
    expect(() => calibrateMultidimensional(data, { ...options, correlated: JSON.parse('"yes"') })).toThrow(
      'The option correlated must be a boolean, found "yes".',
    );
    expect(() => calibrateMultidimensional(data, { ...options, points: 1 })).toThrow(
      "The option points must be an integer from 2, found 1.",
    );
    expect(() => calibrateMultidimensional(data, { ...options, points: JSON.parse('"31"') })).toThrow(DataError);
    expect(() => calibrateMultidimensional(data, { ...options, points: 2000 })).toThrow(
      "A grid of 2000 points in 2 dimensions has more than 1000000 nodes; use fewer points.",
    );
    expect(() => calibrateMultidimensional(data, { ...options, standardErrors: "bootstrap" as never })).toThrow(
      'The option standardErrors must be "none", "hessian" or "cross-product", found "bootstrap".',
    );
    expect(() => calibrateMultidimensional(data, { ...options, tolerance: -1 })).toThrow(DataError);
    expect(() => calibrateMultidimensional(data, { ...options, maxIterations: 0 })).toThrow(DataError);
    expect(() => calibrateMultidimensional(data, { ...options, priors: JSON.parse("null") })).toThrow(
      "The option priors must be an object of priors by parameter name, found null.",
    );
    expect(() => calibrateMultidimensional(data, { ...options, priors: { a: JSON.parse("{}") } })).toThrow(
      'The prior of "a" must be a prior such as normalPrior(0, 1), found an object.',
    );
    expect(() => calibrateMultidimensional(data, { ...options, priors: { d: normalPrior(1, 1) } })).toThrow(
      'A prior is given for "d", which no estimated item has.',
    );
  });

  it("checks the models", () => {
    const options = { loadings };
    expect(() => calibrateMultidimensional(data, { ...options, model: JSON.parse("null") })).toThrow(
      "The option model must be a multidimensional model such as multidimensionalTwoParameterLogistic, found null.",
    );
    expect(() => calibrateMultidimensional(data, { ...options, model: [model, model] })).toThrow(
      "Found 2 models for 12 items.",
    );
    expect(() => calibrateMultidimensional(data, { ...options, model: JSON.parse("{}") })).toThrow(DataError);
    // Binary data cannot be a graded item with a single category, and graded data need every category up to the highest.
    const polytomous = responseMatrix([
      [0, 1],
      [1, 0],
      [2, 1],
      [2, 0],
    ]);
    expect(() => calibrateMultidimensional(polytomous, { dimensions: 1, model })).toThrow(
      'Item "item1" has a response of 2, but the M2PL model allows responses from 0 to 1.',
    );
    const gap = responseMatrix([
      [0, 1],
      [3, 0],
      [3, 1],
      [0, 0],
    ]);
    expect(() => calibrateMultidimensional(gap, { dimensions: 1, model: multidimensionalGradedResponse })).toThrow(
      'Item "item1" has no response in category 1',
    );
    const constant = responseMatrix([
      [0, 0],
      [0, 1],
      [0, 0],
    ]);
    expect(() => calibrateMultidimensional(constant, { dimensions: 1, model })).toThrow(
      'Item "item1" has responses in fewer than two categories',
    );
  });

  it("needs at least one response", () => {
    expect(() =>
      calibrateMultidimensional(
        responseMatrix([
          [null, null],
          [null, null],
        ]),
        { dimensions: 1, model },
      ),
    ).toThrow("Calibration needs at least one response.");
  });
});

describe("properties", () => {
  /** Small simulated data: a matrix of persons whose order the properties permute. */
  const rows = m2pl.responses.slice(0, 120);
  const columns = [...m2pl.itemIds];

  it("gives estimates that do not depend on the order of the persons", () => {
    const options = {
      model: multidimensionalTwoParameterLogistic,
      loadings: m2pl.loadings,
      points: 11,
      tolerance: 1e-10,
    };
    const fit = calibrateMultidimensional(responseMatrix(rows, { itemIds: columns }), options);
    fc.assert(
      fc.property(fc.shuffledSubarray(rows, { minLength: rows.length, maxLength: rows.length }), (shuffled) => {
        const other = calibrateMultidimensional(responseMatrix(shuffled, { itemIds: columns }), options);
        expectClose(other.logLikelihood, fit.logLikelihood, 1e-8);
        other.items.forEach(({ parameters }, i) => {
          expectClose(defined(parameters.a[i < 6 ? 0 : 1]), defined(fit.items[i]?.parameters.a[i < 6 ? 0 : 1]), 1e-6);
        });
        expectClose(defined(other.correlations[0]?.[1]), defined(fit.correlations[0]?.[1]), 1e-6);
      }),
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

describe("input of other types", () => {
  const item = defined(itemsOf(m2pl)[0]);
  const items = itemsOf(m2pl);

  it("rejects options, items, responses and priors that are not what they should be", () => {
    expect(() => multidimensionalPosteriorMeanEstimate(items, [], JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => multidimensionalPosteriorModeEstimate(items, [], JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
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
