import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { correlationEigenvalues } from "../src/dimensionality/eigenvalues.js";
import {
  createRandom,
  DataError,
  mcdonaldOmega,
  type ParallelAnalysis,
  type ParallelAnalysisOptions,
  parallelAnalysis,
  polychoricCorrelations,
  responseMatrix,
  simulateItems,
  simulateResponses,
  simulateThetas,
  twoParameterLogistic,
} from "../src/index.js";
import { bivariateNormalCdf, bivariateNormalDensity } from "../src/math/bivariate-normal.js";
import { minresLoadings } from "../src/math/minres.js";
import { normalCdf } from "../src/math/normal.js";
import bivariate from "./fixtures/dimensionality-bivariate-normal.json" with { type: "json" };
import parallel from "./fixtures/dimensionality-parallel.json" with { type: "json" };
import polychoric from "./fixtures/dimensionality-polychoric.json" with { type: "json" };
import { defined, expectClose } from "./helpers.js";

// Tolerances against R, explained in docs/validation.md.
/** The bivariate normal distribution function against mvtnorm's Miwa algorithm. */
const DISTRIBUTION_TOLERANCE = 1e-14;
/** Polychoric correlations against the base R likelihood maximized to 1e-12. */
const CORRELATION_TOLERANCE = 1e-8;
/** Thresholds against qnorm. */
const THRESHOLD_TOLERANCE = 1e-12;
/** Correlations against psych and polycor, whose optimize stops at about 1e-4 of the maximum. */
const PSYCH_CORRELATION_TOLERANCE = 1e-4;
/** Observed eigenvalues against base R's eigen and L-BFGS-B minres solution. */
const EIGENVALUE_TOLERANCE = 1e-8;
/** The factor eigenvalues against psych's, whose L-BFGS-B stops short of the minimum of its residuals. */
const PSYCH_EIGENVALUE_TOLERANCE = 1e-4;
/** Means of random eigenvalues against a reference of its own random data: this many Monte Carlo standard errors. */
const MONTE_CARLO_SIGMAS = 4;
/**
 * Quantiles of random eigenvalues: more standard errors of a normal variable's quantile, as the first factor eigenvalue
 * of random data has a heavy upper tail, from the data sets whose loading is at its bound.
 */
const QUANTILE_SIGMAS = 6;

function responses(rows: readonly (readonly (number | null)[])[]): ReturnType<typeof responseMatrix> {
  return responseMatrix(rows);
}

/** The 1,000 persons' responses of a seeded 2PL test of `items` items, as a response matrix. */
function simulated(items: number, persons: number, seed: number): ReturnType<typeof responseMatrix> {
  const random = createRandom(seed);
  const pool = simulateItems(twoParameterLogistic, items, random);
  return simulateResponses(pool, simulateThetas(persons, random), random);
}

describe("bivariateNormalCdf", () => {
  it("matches mvtnorm's distribution function at random points and at the edges of the algorithm", () => {
    for (const { h, k, rho, p } of bivariate.cases) {
      expectClose(bivariateNormalCdf(h, k, rho), p, DISTRIBUTION_TOLERANCE, `h ${h}, k ${k}, rho ${rho}`);
    }
  });

  it("is the product of the margins for independent variables and the margin for an infinite threshold", () => {
    expectClose(bivariateNormalCdf(0.3, -1.2, 0), normalCdf(0.3) * normalCdf(-1.2), 1e-16);
    expect(bivariateNormalCdf(Number.POSITIVE_INFINITY, 0.7, 0.4)).toBe(normalCdf(0.7));
    expect(bivariateNormalCdf(0.7, Number.POSITIVE_INFINITY, -0.4)).toBe(normalCdf(0.7));
    expect(bivariateNormalCdf(Number.NEGATIVE_INFINITY, 0.7, 0.4)).toBe(0);
    expect(bivariateNormalCdf(0.7, Number.NEGATIVE_INFINITY, 0.4)).toBe(0);
  });

  it("is concentrated on a line at a correlation of 1 or -1", () => {
    expect(bivariateNormalCdf(0.5, -0.2, 1)).toBe(normalCdf(-0.2));
    expectClose(bivariateNormalCdf(0.5, 0.2, -1), normalCdf(0.5) + normalCdf(0.2) - 1, 1e-16);
    expect(bivariateNormalCdf(-0.5, 0.2, -1)).toBe(0);
  });

  it("gives the quadrant of equal thresholds at zero as 1/4 + asin(rho) / (2 pi)", () => {
    for (const rho of [-0.99, -0.5, 0.2, 0.9, 0.999]) {
      expectClose(bivariateNormalCdf(0, 0, rho), 0.25 + Math.asin(rho) / (2 * Math.PI), 1e-15, `rho ${rho}`);
    }
  });

  it("has the density as its derivative with respect to rho", () => {
    for (const [h, k, rho] of [
      [0.3, -0.8, 0.4],
      [-1.5, 1.1, -0.7],
      [2, 2, 0.95],
    ] as const) {
      const step = 1e-6;
      const slope = (bivariateNormalCdf(h, k, rho + step) - bivariateNormalCdf(h, k, rho - step)) / (2 * step);
      expectClose(slope, bivariateNormalDensity(h, k, rho), 1e-8, `h ${h}, k ${k}, rho ${rho}`);
    }
  });
});

describe("polychoricCorrelations", () => {
  for (const reference of polychoric.cases) {
    const matrix = responseMatrix(reference.responses, { itemIds: reference.itemIds });
    const result = polychoricCorrelations(matrix, { correction: reference.correction });

    it(`matches the maximum likelihood correlations and thresholds in base R on ${reference.name}`, () => {
      expect(result.itemIds).toEqual(reference.itemIds);
      for (const i of reference.used) {
        defined(result.thresholds[i]).forEach((threshold, k) => {
          expectClose(
            threshold,
            defined(reference.thresholds[i]?.[k]),
            THRESHOLD_TOLERANCE,
            `item ${i + 1}, threshold ${k + 1}`,
          );
        });
        for (const j of reference.used) {
          expectClose(
            defined(result.correlations[i])[j] ?? null,
            defined(reference.baseR[i])[j] ?? null,
            CORRELATION_TOLERANCE,
            `items ${i + 1} and ${j + 1}`,
          );
        }
      }
    });

    it(`matches psych's polychoric and tetrachoric correlations within its accuracy on ${reference.name}`, () => {
      result.correlations.forEach((row, i) => {
        row.forEach((value, j) => {
          expectClose(
            value,
            defined(reference.psych[i])[j] ?? null,
            PSYCH_CORRELATION_TOLERANCE,
            `items ${i + 1} and ${j + 1}`,
          );
        });
      });
    });

    if (reference.polycor !== null && reference.polycor !== undefined) {
      const { polycor } = reference;
      it(`matches polycor's two-step polychor on ${reference.name}`, () => {
        for (const i of reference.used) {
          for (const j of reference.used) {
            expectClose(
              defined(result.correlations[i])[j] ?? null,
              defined(polycor[i])[j] ?? null,
              PSYCH_CORRELATION_TOLERANCE,
              `items ${i + 1} and ${j + 1}`,
            );
          }
        }
      });
    }
  }

  it("gives sin(pi (a - b) / (2 (a + b))) for a 2 by 2 table with equal margins and diagonals a and b", () => {
    // The thresholds are 0, so P11 = 1/4 + asin(rho) / (2 pi), which the likelihood sets to a / (2 (a + b)).
    const rows = [
      ...Array.from({ length: 30 }, () => [0, 0]),
      ...Array.from({ length: 30 }, () => [1, 1]),
      ...Array.from({ length: 10 }, () => [0, 1]),
      ...Array.from({ length: 10 }, () => [1, 0]),
    ];
    const { correlations, thresholds } = polychoricCorrelations(responses(rows));
    expectClose(defined(correlations[0])[1] ?? null, Math.sin(Math.PI / 4), 1e-12);
    expect(thresholds).toEqual([[0], [0]]);
  });

  it("gives 0 for a table of independent variables and a negative correlation for a reversed one", () => {
    const independent = [
      ...Array.from({ length: 25 }, () => [0, 0]),
      ...Array.from({ length: 25 }, () => [0, 1]),
      ...Array.from({ length: 25 }, () => [1, 0]),
      ...Array.from({ length: 25 }, () => [1, 1]),
    ];
    expectClose(defined(polychoricCorrelations(responses(independent)).correlations[0])[1] ?? null, 0, 1e-12);
    const reversed = independent.map(([x, y]) => [x ?? 0, y === 0 ? 1 : 0]);
    expect(
      defined(polychoricCorrelations(responses(reversed.concat(reversed.slice(0, 20)))).correlations[0])[1],
    ).toBeLessThan(0);
  });

  it("adds the correction to an empty cell of two items with two categories, and gives 1 or -1 without it", () => {
    const identical = [...Array.from({ length: 20 }, () => [0, 0]), ...Array.from({ length: 20 }, () => [1, 1])];
    const corrected = defined(polychoricCorrelations(responses(identical)).correlations[0])[1] ?? Number.NaN;
    expect(corrected).toBeGreaterThan(0.9);
    expect(corrected).toBeLessThan(1);
    expect(defined(polychoricCorrelations(responses(identical), { correction: 0 }).correlations[0])[1]).toBe(1);
    const opposed = identical.map(([x, y]) => [x ?? 0, y === 0 ? 1 : 0]);
    expect(defined(polychoricCorrelations(responses(opposed), { correction: 0 }).correlations[0])[1]).toBe(-1);
    expect(defined(polychoricCorrelations(responses(opposed)).correlations[0])[1]).toBeGreaterThan(-1);
  });

  it("does not correct the empty cells of items with more than two categories", () => {
    const rows = [
      ...Array.from({ length: 12 }, () => [0, 0]),
      ...Array.from({ length: 10 }, () => [1, 1]),
      ...Array.from({ length: 12 }, () => [2, 2]),
      [0, 1],
      [1, 0],
    ];
    const matrix = responses(rows);
    expect(polychoricCorrelations(matrix, { correction: 0 })).toEqual(
      polychoricCorrelations(matrix, { correction: 5 }),
    );
  });

  it("uses the observed categories in order, whatever their numbers", () => {
    const rows: [number, number][] = [
      [0, 0],
      [0, 1],
      [1, 1],
      [1, 2],
      [2, 2],
      [2, 2],
      [1, 0],
      [0, 0],
      [2, 1],
      [1, 1],
    ];
    const relabeled = rows.map(([x, y]) => [x * 3 + 4, y * 5]);
    expect(polychoricCorrelations(responses(relabeled))).toEqual(polychoricCorrelations(responses(rows)));
  });

  it("skips categories nobody chose when it cuts the thresholds", () => {
    const rows = [
      [0, 0],
      [0, 2],
      [2, 0],
      [2, 2],
      [0, 0],
      [2, 2],
      [0, 2],
      [2, 0],
      [0, 0],
    ];
    const { thresholds } = polychoricCorrelations(responses(rows));
    expect(thresholds.map((list) => list.length)).toEqual([1, 1]);
  });

  it("gives null for an item with fewer than two observed categories, in its row, column and diagonal", () => {
    const { correlations, thresholds } = polychoricCorrelations(
      responses([
        [1, 0, 1, null],
        [1, 1, 0, null],
        [1, 0, 1, null],
        [1, 1, 0, null],
        [1, 1, 1, null],
      ]),
    );
    expect(correlations[0]).toEqual([null, null, null, null]);
    expect(correlations[3]).toEqual([null, null, null, null]);
    expect(correlations.map((row) => row[0])).toEqual([null, null, null, null]);
    expect(correlations.map((row) => row[3])).toEqual([null, null, null, null]);
    expect(correlations[1]?.[1]).toBe(1);
    expect(thresholds[0]).toEqual([]);
    expect(thresholds[3]).toEqual([]);
    expect(thresholds[1]).toEqual([defined(thresholds[1]?.[0])]);
  });

  it("gives null for a pair of items that no person answered both", () => {
    const { correlations } = polychoricCorrelations(
      responses([
        [0, null],
        [1, null],
        [null, 0],
        [null, 1],
      ]),
    );
    expect(correlations).toEqual([
      [1, null],
      [null, 1],
    ]);
  });

  it("gives the matrix of one item, 1 for an item with two categories", () => {
    const { correlations, thresholds, itemIds } = polychoricCorrelations(responses([[0], [1], [1]]));
    expect(correlations).toEqual([[1]]);
    expect(itemIds).toEqual(["item1"]);
    expect(thresholds).toHaveLength(1);
  });

  it("gives correlations of two persons: 1 for agreeing ones without the correction, and less with it", () => {
    const rows = [
      [0, 0],
      [1, 1],
    ];
    expect(defined(polychoricCorrelations(responses(rows), { correction: 0 }).correlations[0])[1]).toBe(1);
    const corrected = defined(polychoricCorrelations(responses(rows)).correlations[0])[1] ?? Number.NaN;
    expect(corrected).toBeGreaterThan(0);
    expect(corrected).toBeLessThan(1);
  });

  it("gives correlations from one person who answered both items, where the table has a single count", () => {
    const { correlations } = polychoricCorrelations(
      responses([
        [0, 0],
        [1, null],
        [null, 1],
        [0, null],
        [null, 0],
      ]),
    );
    const value = defined(correlations[0])[1] ?? 0;
    expect(Math.abs(value)).toBeLessThanOrEqual(1);
  });

  it("keeps the correlation of items that agree in the same direction near 1 with many categories", () => {
    const rows = Array.from({ length: 200 }, (_, i) => [i % 5, i % 5]);
    const value = defined(polychoricCorrelations(responses(rows)).correlations[0])[1] ?? 0;
    expect(value).toBeGreaterThan(0.99);
  });

  it("rejects options and matrices of other types and invalid options", () => {
    const matrix = responses([
      [0, 1],
      [1, 0],
      [1, 1],
    ]);
    expect(() => polychoricCorrelations(matrix, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => polychoricCorrelations(matrix, { correction: JSON.parse('"0.5"') })).toThrow(
      'The option correction must be a non-negative number, found "0.5".',
    );
    expect(() => polychoricCorrelations(matrix, { correction: -1 })).toThrow(
      "The option correction must be a non-negative number, found -1.",
    );
    expect(() => polychoricCorrelations(matrix, { correction: JSON.parse("null") })).toThrow(DataError);
    expect(() => polychoricCorrelations(JSON.parse("null"))).toThrow(
      "A response matrix needs the arrays personIds, itemIds and responses.",
    );
    expect(() => polychoricCorrelations(JSON.parse('{"personIds":[],"itemIds":[],"responses":"x"}'))).toThrow(
      DataError,
    );
  });
});

describe("minresLoadings", () => {
  /** A correlation matrix by rows with `value` off the diagonal. */
  function equicorrelation(size: number, value: number): Float64Array {
    return Float64Array.from({ length: size * size }, (_, index) =>
      Math.floor(index / size) === index % size ? 1 : value,
    );
  }

  it("gives the loadings sqrt(r) for equal correlations r, which one factor fits exactly", () => {
    const { loadings, converged } = minresLoadings(equicorrelation(4, 0.36), 4);
    expect(converged).toBe(true);
    for (const loading of loadings) expectClose(loading, 0.6, 1e-10);
  });

  it("fits three correlations exactly, also where a loading is above 1", () => {
    // λ1 λ2 = λ1 λ3 = 0.9 and λ2 λ3 = 0.1 give λ1² = 0.9 · 0.9 / 0.1: a Heywood case, which the bound of 1 cuts off.
    const r = Float64Array.of(1, 0.9, 0.9, 0.9, 1, 0.1, 0.9, 0.1, 1);
    const free = minresLoadings(r, 3);
    expectClose(free.loadings[0] ?? null, Math.sqrt(8.1), 1e-8);
    expectClose(free.loadings[1] ?? null, 0.9 / Math.sqrt(8.1), 1e-8);
    const bounded = minresLoadings(r, 3, 1);
    expect(bounded.converged).toBe(true);
    expect(bounded.loadings[0]).toBe(1);
    // The free loadings meet the first-order conditions of the residuals: Σ_(j≠i) (r_ij - λ_i λ_j) λ_j = 0.
    const [l1, l2, l3] = bounded.loadings;
    expectClose(
      (0.9 - defined(l1) * defined(l2)) * defined(l1) + (0.1 - defined(l2) * defined(l3)) * defined(l3),
      0,
      1e-8,
    );
  });

  it("gives loadings with a sum that is not negative", () => {
    const r = equicorrelation(3, 0.5);
    expect(minresLoadings(r, 3).loadings.reduce((sum, value) => sum + value, 0)).toBeGreaterThan(0);
  });

  it("starts from the leading eigenvector when the correlations sum to a value that is not positive", () => {
    const r = equicorrelation(3, -0.5);
    const { loadings, converged } = minresLoadings(r, 3, 0.99);
    expect(converged).toBe(true);
    expect(loadings.every((value) => Math.abs(value) <= 0.99)).toBe(true);
  });

  it("serves omega, whose loadings it reports", () => {
    const result = mcdonaldOmega(simulated(6, 300, 4));
    expect(result.converged).toBe(true);
    expect(result.loadings).toHaveLength(6);
  });
});

describe("correlationEigenvalues", () => {
  for (const [index, reference] of parallel.randomMatrices.entries()) {
    const size = reference.matrix.length;
    const r = Float64Array.from(reference.matrix.flat());
    it(`matches base R's eigenvalues and bounded minres factors of permuted bfi correlations ${index + 1}`, () => {
      const { components, factors } = correlationEigenvalues(r, size);
      reference.components.forEach((value, k) => {
        expectClose(components[k] ?? null, value, EIGENVALUE_TOLERANCE, `component ${k + 1}`);
      });
      reference.factors.forEach((value, k) => {
        expectClose(factors[k] ?? null, value, EIGENVALUE_TOLERANCE, `factor ${k + 1}`);
      });
    });
  }

  it("reports the negative eigenvalues of a matrix that is not positive definite", () => {
    const r = Float64Array.of(1, 0.9, -0.9, 0.9, 1, 0.9, -0.9, 0.9, 1);
    const { components } = correlationEigenvalues(r, 3);
    expect(components[2]).toBeLessThan(0);
    expect(components.reduce((sum, value) => sum + value, 0)).toBeCloseTo(3, 12);
  });
});

/** Expects the mean of a replicated eigenvalue within Monte Carlo error of a reference with its own replications. */
function expectMeans(
  actual: readonly number[],
  reference: { readonly mean: readonly number[]; readonly sd: readonly number[] },
  referenceReplications: number,
  replications: number,
  label: string,
): void {
  reference.mean.forEach((mean, k) => {
    const allowed =
      MONTE_CARLO_SIGMAS * defined(reference.sd[k]) * Math.sqrt(1 / referenceReplications + 1 / replications);
    expectClose(actual[k] ?? null, mean, allowed, `${label} ${k + 1}`);
  });
}

/** Expects a quantile of replicated eigenvalues within what a quantile of both sets of replications varies by. */
function expectQuantiles(
  actual: readonly number[],
  reference: { readonly quantile: readonly number[]; readonly sd: readonly number[] },
  referenceReplications: number,
  replications: number,
  label: string,
): void {
  // The standard error of the 0.95 quantile of a normal variable over n draws is sd · sqrt(0.95 · 0.05 / n) / φ(1.645).
  const error = (n: number): number => Math.sqrt((0.95 * 0.05) / n) / 0.10314;
  reference.quantile.forEach((quantile, k) => {
    const allowed =
      QUANTILE_SIGMAS *
      defined(reference.sd[k]) *
      Math.sqrt(error(referenceReplications) ** 2 + error(replications) ** 2);
    expectClose(actual[k] ?? null, quantile, allowed, `${label} ${k + 1}`);
  });
}

describe("parallelAnalysis", () => {
  for (const reference of parallel.cases) {
    const matrix = responseMatrix(reference.responses, { itemIds: reference.itemIds });
    const options: ParallelAnalysisOptions = {
      correlation: reference.correlation === "pearson" ? "pearson" : "polychoric",
    };
    const pearson = options.correlation === "pearson";

    it(`matches the observed eigenvalues of base R and psych on ${reference.name}`, () => {
      const { components, factors } = parallelAnalysis(matrix, createRandom(1), { ...options, replications: 1 });
      reference.observed.components.forEach((value, k) => {
        expectClose(components.observed[k] ?? null, value, EIGENVALUE_TOLERANCE, `component ${k + 1}`);
        // psych's polychoric correlations are within 1e-4 of the maximum likelihood ones, and so are its eigenvalues.
        expectClose(
          components.observed[k] ?? null,
          defined(reference.psych.components[k]),
          pearson ? EIGENVALUE_TOLERANCE : PSYCH_EIGENVALUE_TOLERANCE,
          `psych component ${k + 1}`,
        );
      });
      reference.observed.factors.forEach((value, k) => {
        expectClose(factors.observed[k] ?? null, value, EIGENVALUE_TOLERANCE, `factor ${k + 1}`);
        expectClose(
          factors.observed[k] ?? null,
          defined(reference.observed.psychFactors[k]),
          PSYCH_EIGENVALUE_TOLERANCE,
          `psych factor ${k + 1}`,
        );
      });
    });

    describe(`on ${reference.name}`, () => {
      const replications = pearson ? 1000 : 200;
      const result: ParallelAnalysis = parallelAnalysis(matrix, createRandom(2026), { ...options, replications });

      it("has random eigenvalues with the means of psych's resampled data, within Monte Carlo error", () => {
        const { psych } = reference;
        expectMeans(
          result.components.randomMean,
          psych.resampledComponents,
          psych.iterations,
          replications,
          "component",
        );
        expectMeans(result.factors.randomMean, psych.resampledFactors, psych.iterations, replications, "factor");
      });

      it("has random quantiles near those of psych's resampled data, within Monte Carlo error", () => {
        const { psych } = reference;
        expectQuantiles(
          result.components.randomQuantile,
          psych.resampledComponents,
          psych.iterations,
          replications,
          "component",
        );
        expectQuantiles(
          result.factors.randomQuantile,
          psych.resampledFactors,
          psych.iterations,
          replications,
          "factor",
        );
      });

      it("suggests the numbers of components and factors that psych does from its resampled quantiles", () => {
        expect(result.components.suggested).toBe(reference.psych.suggested.resampledComponents);
        expect(result.factors.suggested).toBe(reference.psych.suggested.resampledFactors);
      });
    });
  }

  const bfi = defined(parallel.cases[0]);
  const bfiMatrix = responseMatrix(bfi.responses, { itemIds: bfi.itemIds });

  it("has random eigenvalues with the means and quantiles of permuted data in base R", () => {
    const reference = defined(bfi.permutation);
    const replications = 1000;
    const { components, factors } = parallelAnalysis(bfiMatrix, createRandom(11), {
      correlation: "pearson",
      replications,
    });
    expectMeans(components.randomMean, reference.components, reference.iterations, replications, "component");
    expectMeans(factors.randomMean, reference.factors, reference.iterations, replications, "factor");
    expectQuantiles(components.randomQuantile, reference.components, reference.iterations, replications, "component");
    expectQuantiles(factors.randomQuantile, reference.factors, reference.iterations, replications, "factor");
  });

  it("compares with normal data as psych's simulated data, and suggests the numbers psych does", () => {
    const { psych } = bfi;
    const replications = 300;
    const { components, factors } = parallelAnalysis(bfiMatrix, createRandom(12), {
      correlation: "pearson",
      comparison: "normal",
      replications,
    });
    expectMeans(components.randomMean, psych.normalComponents, psych.iterations, replications, "component");
    expectMeans(factors.randomMean, psych.normalFactors, psych.iterations, replications, "factor");
    expect(components.suggested).toBe(psych.suggested.normalComponents);
    expect(factors.suggested).toBe(psych.suggested.normalFactors);
    expect(components.suggested).toBe(psych.psychSuggested.components);
    expect(factors.suggested).toBe(psych.psychSuggested.factors);
  });

  it("reproduces its result from the seed", () => {
    const matrix = simulated(8, 120, 3);
    const run = (seed: number) => parallelAnalysis(matrix, createRandom(seed), { replications: 20 });
    expect(run(5)).toEqual(run(5));
    expect(run(5)).not.toEqual(run(6));
  });

  it("suggests one dimension for the items of a unidimensional test and none for independent items", () => {
    const test = parallelAnalysis(simulated(10, 600, 8), createRandom(1), { replications: 50 });
    expect(test.components.suggested).toBe(1);
    expect(test.factors.suggested).toBe(1);
    const random = createRandom(9);
    const independent = responseMatrix(
      Array.from({ length: 400 }, () => Array.from({ length: 8 }, () => (random.uniform() < 0.5 ? 1 : 0))),
    );
    const none = parallelAnalysis(independent, createRandom(2), { replications: 50 });
    expect(none.components.suggested).toBe(0);
    expect(none.factors.suggested).toBe(0);
  });

  it("suggests two dimensions for items of two independent traits", () => {
    const random = createRandom(31);
    const rows = Array.from({ length: 800 }, () => {
      const first = random.normal();
      const second = random.normal();
      return Array.from({ length: 12 }, (_, j) => {
        const trait = j < 6 ? first : second;
        return random.uniform() < 1 / (1 + Math.exp(-1.7 * trait)) ? 1 : 0;
      });
    });
    const result = parallelAnalysis(responses(rows), createRandom(4), { replications: 50 });
    expect(result.components.suggested).toBe(2);
    expect(result.factors.suggested).toBe(2);
  });

  it("reports the eigenvalues of Pearson correlations of scores treated as continuous", () => {
    const matrix = responses(
      Array.from({ length: 60 }, (_, i) => [i % 7, (i * 3) % 7, (i + (i % 3)) % 7, (i * 5 + 1) % 7]),
    );
    const { components } = parallelAnalysis(matrix, createRandom(1), { correlation: "pearson", replications: 5 });
    expect(components.observed.reduce((sum, value) => sum + value, 0)).toBeCloseTo(4, 12);
    expect(components.observed).toHaveLength(4);
  });

  it("uses the quantile of the random eigenvalues to count the leading observed eigenvalues above it", () => {
    const matrix = simulated(8, 300, 14);
    const low = parallelAnalysis(matrix, createRandom(1), { replications: 30, quantile: 0 });
    const high = parallelAnalysis(matrix, createRandom(1), { replications: 30, quantile: 1 });
    low.components.randomQuantile.forEach((value, k) => {
      expect(value).toBeLessThanOrEqual(defined(low.components.randomMean[k]) + 1e-12);
      expect(defined(high.components.randomQuantile[k])).toBeGreaterThanOrEqual(value);
    });
    expect(low.components.suggested).toBeGreaterThanOrEqual(high.components.suggested);
  });

  it("works with one replication, whose mean and quantile are its eigenvalues", () => {
    const { components } = parallelAnalysis(simulated(6, 100, 2), createRandom(1), { replications: 1 });
    expect(components.randomMean).toEqual(components.randomQuantile);
  });

  it("rejects options of other types and invalid ones", () => {
    const matrix = simulated(5, 60, 1);
    const random = createRandom(1);
    const options = (json: string): ParallelAnalysisOptions => JSON.parse(json);
    expect(() => parallelAnalysis(matrix, random, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => parallelAnalysis(matrix, random, options('{"correlation":"spearman"}'))).toThrow(
      'The option correlation must be "pearson" or "polychoric", found "spearman".',
    );
    expect(() => parallelAnalysis(matrix, random, options('{"comparison":"permuted"}'))).toThrow(
      'The option comparison must be "resampled" or "normal", found "permuted".',
    );
    expect(() => parallelAnalysis(matrix, random, options('{"replications":0}'))).toThrow(
      "The option replications must be a positive integer, found 0.",
    );
    expect(() => parallelAnalysis(matrix, random, options('{"replications":"20"}'))).toThrow(
      'The option replications must be a positive integer, found "20".',
    );
    expect(() => parallelAnalysis(matrix, random, options('{"quantile":1.5}'))).toThrow(
      "The option quantile must be a number from 0 to 1, found 1.5.",
    );
    expect(() => parallelAnalysis(matrix, random, options('{"correction":-1}'))).toThrow(
      "The option correction must be a non-negative number, found -1.",
    );
    expect(() => parallelAnalysis(matrix, random, { comparison: "normal" })).toThrow(
      'The option comparison "normal" needs the option correlation "pearson", found "polychoric".',
    );
    expect(() => parallelAnalysis(matrix, JSON.parse("null"))).toThrow(DataError);
    expect(() => parallelAnalysis(JSON.parse("null"), random)).toThrow(
      "A response matrix needs the arrays personIds, itemIds and responses.",
    );
  });

  it("rejects matrices with fewer than three items, an item of one category, or a pair without persons", () => {
    const random = createRandom(1);
    expect(() =>
      parallelAnalysis(
        responses([
          [0, 1],
          [1, 0],
          [1, 1],
        ]),
        random,
      ),
    ).toThrow("Parallel analysis needs at least 3 items, as one factor is not identified by two, found 2.");
    expect(() =>
      parallelAnalysis(
        responses([
          [0, 1, 1],
          [1, 0, 1],
          [1, 1, 1],
        ]),
        random,
      ),
    ).toThrow('Parallel analysis needs items with at least two observed categories; item "item3" has 1.');
    const disjoint = responses([
      [0, 0, null],
      [1, 1, null],
      [0, 1, null],
      [1, 0, null],
      [null, 0, 0],
      [null, 1, 1],
      [null, 1, 0],
      [null, 0, 1],
    ]);
    expect(() => parallelAnalysis(disjoint, random, { replications: 2 })).toThrow(
      'The observed data gives no correlation of items "item1" and "item3": too few persons answered both.',
    );
    expect(() => parallelAnalysis(disjoint, random, { correlation: "pearson", replications: 2 })).toThrow(
      "gives no correlation of items",
    );
  });
});

describe("properties", () => {
  it("the bivariate normal distribution function is a probability, symmetric, and increasing in rho", () => {
    const point = fc.record({
      h: fc.double({ min: -8, max: 8, noNaN: true }),
      k: fc.double({ min: -8, max: 8, noNaN: true }),
      rho: fc.double({ min: -0.9999, max: 0.9999, noNaN: true }),
    });
    fc.assert(
      fc.property(point, ({ h, k, rho }) => {
        const p = bivariateNormalCdf(h, k, rho);
        expect(p).toBeGreaterThanOrEqual(0);
        expect(p).toBeLessThanOrEqual(1);
        expect(bivariateNormalCdf(k, h, rho)).toBeCloseTo(p, 14);
        expect(bivariateNormalCdf(h, k, Math.min(rho + 1e-3, 0.9999))).toBeGreaterThanOrEqual(p - 1e-14);
        // By inclusion and exclusion, P(X <= h, Y <= k) = P(X <= h) + P(Y <= k) - 1 + P(X > h, Y > k), and by symmetry
        // P(X > h, Y > k) is the distribution function at (-h, -k).
        const upper = normalCdf(h) + normalCdf(k) - 1 + bivariateNormalCdf(-h, -k, rho);
        expect(Math.abs(p - upper)).toBeLessThan(1e-14);
      }),
      { numRuns: 300 },
    );
  });

  const arbitraryResponses = fc
    .record({ items: fc.integer({ min: 2, max: 5 }), persons: fc.integer({ min: 4, max: 40 }) })
    .chain(({ items, persons }) =>
      fc.array(
        fc.array(fc.option(fc.integer({ min: 0, max: 3 }), { nil: null, freq: 8 }), {
          minLength: items,
          maxLength: items,
        }),
        { minLength: persons, maxLength: persons },
      ),
    );

  it("polychoric correlations are symmetric with 1 or null on the diagonal and within -1 to 1", () => {
    fc.assert(
      fc.property(arbitraryResponses, (rows) => {
        const { correlations } = polychoricCorrelations(responses(rows));
        correlations.forEach((row, i) => {
          row.forEach((value, j) => {
            expect(value).toBe(defined(correlations[j])[i]);
            if (i === j) expect(value === 1 || value === null).toBe(true);
            else if (value !== null) expect(Math.abs(value)).toBeLessThanOrEqual(1);
          });
        });
      }),
      { numRuns: 100 },
    );
  });

  it("polychoric correlations do not depend on the order of the persons", () => {
    fc.assert(
      fc.property(arbitraryResponses, fc.integer(), (rows, seed) => {
        const random = createRandom(seed);
        const shuffled = [...rows];
        for (let i = shuffled.length - 1; i > 0; i--) {
          const j = Math.floor(random.uniform() * (i + 1));
          [shuffled[i], shuffled[j]] = [defined(shuffled[j]), defined(shuffled[i])];
        }
        expect(polychoricCorrelations(responses(shuffled))).toEqual(polychoricCorrelations(responses(rows)));
      }),
      { numRuns: 60 },
    );
  });

  it("polychoric correlations do not change when an item's categories are renumbered in order", () => {
    fc.assert(
      fc.property(arbitraryResponses, (rows) => {
        const renumbered = rows.map((row) => row.map((value, j) => (value === null || j > 0 ? value : value * 2 + 1)));
        expect(polychoricCorrelations(responses(renumbered))).toEqual(polychoricCorrelations(responses(rows)));
      }),
      { numRuns: 60 },
    );
  });

  it("minres loadings stay within their bound", () => {
    fc.assert(
      fc.property(
        fc.array(fc.double({ min: -0.6, max: 0.95, noNaN: true }), { minLength: 6, maxLength: 6 }),
        (values) => {
          const [a = 0, b = 0, c = 0, d = 0, e = 0, f = 0] = values;
          const r = Float64Array.of(1, a, b, c, a, 1, d, e, b, d, 1, f, c, e, f, 1);
          const { loadings } = minresLoadings(r, 4, Math.sqrt(0.995));
          for (const loading of loadings) expect(Math.abs(loading)).toBeLessThanOrEqual(Math.sqrt(0.995) + 1e-15);
        },
      ),
      { numRuns: 100 },
    );
  });

  const arbitraryTest = fc
    .record({ items: fc.integer({ min: 3, max: 5 }), persons: fc.integer({ min: 20, max: 60 }) })
    .chain(({ items, persons }) =>
      fc.array(fc.array(fc.integer({ min: 0, max: 2 }), { minLength: items, maxLength: items }), {
        minLength: persons,
        maxLength: persons,
      }),
    );

  it("parallel analysis has decreasing eigenvalues that sum to the items, and a suggestion in range", () => {
    fc.assert(
      fc.property(
        arbitraryTest,
        fc.integer(),
        fc.constantFrom<"pearson" | "polychoric">("pearson", "polychoric"),
        (rows, seed, correlation) => {
          const matrix = responses(rows);
          let result: ParallelAnalysis;
          try {
            result = parallelAnalysis(matrix, createRandom(seed), { correlation, replications: 4 });
          } catch (error) {
            // Items without two observed categories, or pairs without persons, are rejected.
            expect(error).toBeInstanceOf(DataError);
            return;
          }
          const size = matrix.itemIds.length;
          for (const comparison of [result.components, result.factors]) {
            comparison.observed.forEach((value, k) => {
              if (k > 0) expect(value).toBeLessThanOrEqual(defined(comparison.observed[k - 1]) + 1e-12);
            });
            expect(comparison.suggested).toBeGreaterThanOrEqual(0);
            expect(comparison.suggested).toBeLessThanOrEqual(size);
            expect(comparison.observed).toHaveLength(size);
            expect(comparison.randomMean).toHaveLength(size);
            expect(comparison.randomQuantile).toHaveLength(size);
          }
          expect(result.components.observed.reduce((sum, value) => sum + value, 0)).toBeCloseTo(size, 8);
          expect(result.components.randomMean.reduce((sum, value) => sum + value, 0)).toBeCloseTo(size, 8);
        },
      ),
      { numRuns: 40 },
    );
  });

  it("parallel analysis does not depend on the order of the persons for the observed eigenvalues", () => {
    fc.assert(
      fc.property(arbitraryTest, (rows) => {
        const options = { correlation: "pearson", replications: 1 } as const;
        try {
          const forward = parallelAnalysis(responses(rows), createRandom(1), options).components.observed;
          const reversed = parallelAnalysis(responses([...rows].reverse()), createRandom(1), options).components
            .observed;
          forward.forEach((value, k) => {
            expectClose(value, defined(reversed[k]), 1e-9);
          });
        } catch (error) {
          expect(error).toBeInstanceOf(DataError);
        }
      }),
      { numRuns: 30 },
    );
  });
});
