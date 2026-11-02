import { createRandom, type Random } from "@measurekit/core";
import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  automatedItemSelection,
  conditionalAssociation,
  DataError,
  guttmanErrors,
  type ItemCheck,
  invariantItemOrdering,
  molenaarSijtsmaReliability,
  monotonicity,
  type ResponseMatrix,
  responseMatrix,
  scalability,
  twoWayImputation,
} from "../src/index.js";
import { defined, expectClose } from "./assertions.js";
import fixture from "./fixtures/mokken.json" with { type: "json" };
import imputation from "./fixtures/mokken-imputation.json" with { type: "json" };
import reliability from "./fixtures/mokken-reliability.json" with { type: "json" };
import steps from "./fixtures/mokken-steps.json" with { type: "json" };

// Tolerances against R, explained in docs/validation.md.
/** Coefficients and their standard errors against coefH, and H^T against coefHT. */
const COEFFICIENT_TOLERANCE = 1e-10;
/** Violation sizes, sums and test statistics against the checks' results. */
const STATISTIC_TOLERANCE = 1e-8;
/** MS against check.reliability: sums of the same proportions in another order. */
const RELIABILITY_TOLERANCE = 1e-10;
/** MS against mokken's own value where no tie of item steps is split by the rounding of its proportions. */
const ROUNDING_TOLERANCE = 1e-12;
/** The largest gap between MS and mokken's own value where the rounding of its proportions splits ties: 0.021 in the fixture. */
const SPLIT_TIES_GAP = 0.03;
/** How far the mean of MS may lie from the mean reliability of parallel forms in the study: it was 0.014 in the fixture. */
const STUDY_BIAS = 0.03;
/** The indices of conditional association against check.ca: sums of normal probabilities of the same correlations. */
const ASSOCIATION_TOLERANCE = 1e-10;
/** The mean over items of the imputed data's item variance less the complete data's, at 10% missing. */
const IMPUTED_VARIANCE_TOLERANCE = 0.02;
/** G+ against check.errors where steps tie, as it rounds a mean over 1,000 random orders of them. */
const RANDOM_ORDER_GAP = 0.5;

interface ReferenceCheck {
  readonly id: string;
  readonly activeComparisons: number;
  readonly violations: number;
  readonly maximumViolation: number;
  readonly violationSum: number;
  readonly maximumZ: number;
  readonly significantViolations: number;
  readonly criterion: number;
  /** What mokken reports, present where its rounding changes the count or the criterion. */
  readonly mokkenRounded?: { readonly significantViolations?: number; readonly criterion: number };
}

interface ReferenceCase {
  readonly name: string;
  readonly itemIds: string[];
  readonly responses: (number | null)[][];
  readonly scalability: {
    readonly pairs: number[][];
    readonly pairErrors: number[][];
    readonly items: number[];
    readonly itemErrors: number[];
    readonly scale: number;
    readonly scaleError: number;
  };
  readonly selections: { readonly lowerBound: number; readonly alpha: number; readonly scales: number[] }[];
  readonly monotonicity: {
    readonly minimumViolation: number;
    readonly minimumGroupSize: number | null;
    readonly items: ReferenceCheck[];
  }[];
  readonly invariantOrdering: {
    readonly alpha: number;
    readonly selection: boolean;
    readonly minimumGroupSize: number | null;
    readonly items: ReferenceCheck[];
    readonly removed: string[];
    readonly coefficient: number;
  }[];
}

// TypeScript types each case of the JSON with the keys of the others, so the cases are read back as one type.
const cases: readonly ReferenceCase[] = JSON.parse(JSON.stringify(fixture.cases));

function matrixOf(reference: ReferenceCase): ResponseMatrix {
  return responseMatrix(reference.responses, { itemIds: reference.itemIds });
}

/** Returns the option for a minimum group size that the reference may leave at its default. */
function groupSize(size: number | null): { minimumGroupSize?: number } {
  return size === null ? {} : { minimumGroupSize: size };
}

/** Expects checks to match the references' counts exactly and their sizes and statistics within the tolerance. */
function expectChecks(actual: readonly ItemCheck[], expected: readonly ReferenceCheck[], label: string): void {
  expect(actual, label).toHaveLength(expected.length);
  for (const check of actual) {
    const reference = defined(expected.find(({ id }) => id === check.id));
    const where = `${label} ${check.id}`;
    expect(check.activeComparisons, where).toBe(reference.activeComparisons);
    expect(check.violations, where).toBe(reference.violations);
    expect(check.significantViolations, where).toBe(reference.significantViolations);
    expect(check.criterion, where).toBe(reference.criterion);
    expectClose(check.maximumViolation, reference.maximumViolation, STATISTIC_TOLERANCE, `${where} maximum`);
    expectClose(check.violationSum, reference.violationSum, STATISTIC_TOLERANCE, `${where} sum`);
    // mokken reports 0 for the statistic of an item without violations, which has none.
    expectClose(check.maximumZ ?? 0, reference.maximumZ, STATISTIC_TOLERANCE, `${where} z`);
    expect(check.maximumZ === null, where).toBe(reference.violations === 0);
  }
}

/** Lists the items whose reference differs from mokken's own values, which its rounding changes. */
function roundedByMokken(
  runsOf: (reference: ReferenceCase) => readonly { readonly items: readonly ReferenceCheck[] }[],
): (ReferenceCheck & { readonly name: string })[] {
  return cases.flatMap((reference) =>
    runsOf(reference).flatMap(({ items }) =>
      items
        .filter(({ mokkenRounded }) => mokkenRounded !== undefined)
        .map((item) => ({ name: reference.name, ...item })),
    ),
  );
}

/** Dichotomous scores of persons who pass the first k of the items, in order of difficulty: a perfect Guttman scale. */
function guttman(items: number, personsPerScore: number): number[][] {
  return Array.from({ length: (items + 1) * personsPerScore }, (_, person) => {
    const passed = Math.floor(person / personsPerScore);
    return Array.from({ length: items }, (__, item) => (item < passed ? 1 : 0));
  });
}

describe("twoWayImputation", () => {
  /** A generator whose normal draws are all `value`, so that the imputation's error is `value` standard deviations. */
  const constant = (value: number): Random => ({ uniform: () => 0.5, normal: () => value });

  for (const reference of imputation.cases) {
    const options = {
      ...(reference.minimum === null ? {} : { minimum: reference.minimum }),
      ...(reference.maximum === null ? {} : { maximum: reference.maximum }),
    };
    const matrix = responseMatrix(reference.responses);

    it(`imputes as twoway without its error term on ${reference.name}`, () => {
      expect(twoWayImputation(matrix, constant(0), options).responses).toEqual(reference.imputed);
    });

    it(`adds an error of the residuals' standard deviation, not their variance, on ${reference.name}`, () => {
      // One standard deviation, √v, above the two-way value of each missing cell, rounded and bounded.
      const deviation = Math.sqrt(reference.residualVariance);
      const low = reference.minimum ?? Math.min(...reference.responses.flat().filter((value) => value !== null));
      const high = reference.maximum ?? Math.max(...reference.responses.flat().filter((value) => value !== null));
      const imputed = twoWayImputation(matrix, constant(1), options).responses;
      reference.responses.forEach((row, person) => {
        row.forEach((response, item) => {
          const twoWay = defined(reference.twoWay[person]?.[item]);
          const expected = response ?? Math.min(high, Math.max(low, Math.round(twoWay + deviation)));
          expect(imputed[person]?.[item], `${person + 1}, ${item + 1}`).toBe(expected);
        });
      });
    });
  }

  it("keeps the variance of the items within 0.02 on average when 10% of the responses are missing", () => {
    const reference = defined(imputation.cases.find(({ name }) => name === "acl, ten adjectives, 10% missing"));
    const varianceOf = (values: readonly number[]) => {
      const center = values.reduce((sum, value) => sum + value, 0) / values.length;
      return values.reduce((sum, value) => sum + (value - center) ** 2, 0) / (values.length - 1);
    };
    const itemVariances = (rows: readonly (readonly (number | null)[])[]) =>
      defined(rows[0]).map((_, item) => varianceOf(rows.map((row) => defined(row[item]))));
    const complete = itemVariances(imputation.complete);
    const random = createRandom(20261102);
    const gaps = Array.from({ length: 20 }, () => {
      const imputed = itemVariances(twoWayImputation(responseMatrix(reference.responses), random).responses);
      return imputed.reduce((sum, value, item) => sum + value - defined(complete[item]), 0) / imputed.length;
    });
    expect(Math.abs(gaps.reduce((sum, gap) => sum + gap, 0) / gaps.length)).toBeLessThan(IMPUTED_VARIANCE_TOLERANCE);
  });

  it("rounds a two-way value of a half to the even integer, as R does", () => {
    const reference = defined(imputation.cases.find(({ name }) => name === "a two-way value of a half"));
    expect(reference.twoWay[0]?.[2]).toBe(2.5);
    expect(twoWayImputation(responseMatrix(reference.responses), constant(0)).responses[0]?.[2]).toBe(2);
  });

  it("keeps the responses, the ids and the times, and returns a copy of a matrix without missing responses", () => {
    const matrix = responseMatrix(
      [
        [1, null, 2],
        [0, 1, 1],
        [2, 2, null],
      ],
      {
        personIds: ["a", "b", "c"],
        itemIds: ["x", "y", "z"],
        times: [
          [3, null, 4],
          [5, 6, 7],
          [8, 9, null],
        ],
      },
    );
    const imputed = twoWayImputation(matrix, createRandom(1));
    expect(imputed.personIds).toEqual(["a", "b", "c"]);
    expect(imputed.itemIds).toEqual(["x", "y", "z"]);
    expect(imputed.times).toEqual(matrix.times);
    expect(imputed.responses.flat().every((value) => value !== null)).toBe(true);
    const complete = responseMatrix([
      [1, 0],
      [0, 1],
    ]);
    const copy = twoWayImputation(complete, createRandom(1));
    expect(copy).toEqual(complete);
    expect(copy.responses).not.toBe(complete.responses);
  });

  it("rejects a person or an item without responses, bounds inside the responses and arguments of other types", () => {
    const matrix = responseMatrix([
      [1, null],
      [0, 1],
      [2, 2],
    ]);
    expect(() =>
      twoWayImputation(
        responseMatrix([
          [1, 0],
          [null, null],
        ]),
        createRandom(1),
      ),
    ).toThrow('Two-way imputation needs a response of every person; "person2" has none.');
    expect(() =>
      twoWayImputation(
        responseMatrix([
          [1, null],
          [0, null],
        ]),
        createRandom(1),
      ),
    ).toThrow('Two-way imputation needs a response to every item; "item2" has none.');
    expect(() => twoWayImputation(matrix, createRandom(1), { minimum: 1 })).toThrow(
      "The option minimum must be at most the lowest response, 0, found 1.",
    );
    expect(() => twoWayImputation(matrix, createRandom(1), { maximum: 1 })).toThrow(
      "The option maximum must be at least the highest response, 2, found 1.",
    );
    expect(() => twoWayImputation(matrix, createRandom(1), { minimum: 0.5 })).toThrow("The option minimum");
    expect(() => twoWayImputation(matrix, createRandom(1), JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => twoWayImputation(matrix, JSON.parse("{}"))).toThrow("A random number generator must be one");
    expect(() => twoWayImputation(JSON.parse("null"), createRandom(1))).toThrow(DataError);
  });
});

describe("guttmanErrors", () => {
  for (const reference of steps.errors) {
    it(`gives G+ and Tukey's fence as the mean over the orders of tied steps for ${reference.name}`, () => {
      const result = guttmanErrors(responseMatrix(reference.responses, { itemIds: reference.itemIds }));
      expect(result.errors).toEqual(reference.errors);
      expectClose(result.upperFence, reference.upperFence, 1e-12, "fence");
      expect(result.outliers).toEqual(reference.outliers);
    });
  }

  it("equals check.errors where no steps tie, and stays within 1/2 of its rounded mean of random orders where they do", () => {
    for (const reference of steps.errors) {
      const counted = guttmanErrors(responseMatrix(reference.responses)).errors.filter((count) => count !== null);
      counted.forEach((count, person) => {
        const gap = Math.abs(count - defined(reference.mokken.errors[person]));
        expect(gap, `${reference.name} ${person}`).toBeLessThanOrEqual(reference.mokken.ties ? RANDOM_ORDER_GAP : 0);
      });
      if (!reference.mokken.ties) expect(reference.upperFence, reference.name).toBe(reference.mokken.upperFence);
    }
  });

  it("counts G of dichotomous items, the easier items failed before the harder passed, as derived by hand", () => {
    // Items passed by 5, 3 and 2 persons: the last person fails the first two and passes the third, two errors. The
    // quartiles of six 0s and a 2 are 0, so the fence is 0 and that person is the outlier.
    const rows = [
      [1, 1, 1],
      [1, 1, 0],
      [1, 1, 0],
      [1, 0, 0],
      [1, 0, 0],
      [0, 0, 0],
      [0, 0, 1],
    ];
    expect(guttmanErrors(responseMatrix(rows))).toEqual({
      errors: [0, 0, 0, 0, 0, 0, 2],
      upperFence: 0,
      outliers: [6],
    });
  });

  it("counts half an error for a step failed that as many persons pass as the step passed", () => {
    const rows = [
      [1, 0],
      [0, 1],
      [1, 1],
    ];
    expect(guttmanErrors(responseMatrix(rows)).errors).toEqual([0.5, 0.5, 0]);
  });

  it("counts the errors over the steps of polytomous items", () => {
    // Steps by persons passing: item 2 at 1 (3), item 1 at 1 (2), item 1 at 2 (1), item 2 at 2 (0). Scores 2 and 0 pass
    // the second and third steps and fail the first: two errors.
    const rows = [
      [2, 0],
      [0, 1],
      [1, 1],
      [0, 1],
    ];
    expect(guttmanErrors(responseMatrix(rows)).errors).toEqual([2, 0, 0, 0]);
  });

  it("gives no count to a person with a missing response, and indexes outliers by the matrix's rows", () => {
    const reference = defined(steps.errors.find(({ name }) => name === "transitive reasoning, missing"));
    const result = guttmanErrors(responseMatrix(reference.responses));
    reference.responses.forEach((row, person) => {
      expect(result.errors[person] === null, `person ${person}`).toBe(row.includes(null));
    });
    expect(result.outliers.length).toBeGreaterThan(0);
    for (const person of result.outliers) expect(defined(result.errors[person])).toBeGreaterThan(result.upperFence);
  });

  it("is the same for scores shifted to start above 0", () => {
    const rows = [
      [0, 1, 2],
      [1, 2, 0],
      [2, 2, 1],
      [0, 0, 1],
      [1, 1, 1],
    ];
    expect(guttmanErrors(responseMatrix(rows.map((row) => row.map((score) => score + 3))))).toEqual(
      guttmanErrors(responseMatrix(rows)),
    );
  });

  it("rejects matrices that cannot be analyzed", () => {
    expect(() => guttmanErrors(responseMatrix([[1], [0]]))).toThrow("Guttman errors needs at least 2 items, found 1.");
    expect(() =>
      guttmanErrors(
        responseMatrix([
          [1, 0],
          [null, 1],
        ]),
      ),
    ).toThrow("Guttman errors needs at least 2 persons who answered every item, found 1.");
    expect(() =>
      guttmanErrors(
        responseMatrix([
          [0, 0],
          [10, 0],
        ]),
      ),
    ).toThrow("needs scores from at most 10 categories");
    expect(() => guttmanErrors(JSON.parse("null"))).toThrow(DataError);
    expect(() => guttmanErrors(JSON.parse('{"itemIds": ["a"], "responses": {}}'))).toThrow(DataError);
  });
});

describe("scalability", () => {
  for (const reference of cases) {
    it(`matches the coefficients and standard errors of coefH for ${reference.name}`, () => {
      const result = scalability(matrixOf(reference));
      const expected = reference.scalability;
      result.items.forEach((item, i) => {
        expect(item.id).toBe(reference.itemIds[i]);
        expectClose(item.coefficient, defined(expected.items[i]), COEFFICIENT_TOLERANCE, `H_${i + 1}`);
        expectClose(item.standardError, defined(expected.itemErrors[i]), COEFFICIENT_TOLERANCE, `se H_${i + 1}`);
      });
      for (const [i, row] of result.pairs.entries()) {
        for (const [j, pair] of row.entries()) {
          if (i === j) {
            expect(pair).toBeNull();
            continue;
          }
          expectClose(
            pair?.coefficient ?? null,
            defined(expected.pairs[i]?.[j]),
            COEFFICIENT_TOLERANCE,
            `H_${i + 1}${j + 1}`,
          );
          expectClose(
            pair?.standardError ?? null,
            defined(expected.pairErrors[i]?.[j]),
            COEFFICIENT_TOLERANCE,
            `se H_${i + 1}${j + 1}`,
          );
        }
      }
      expectClose(result.scale.coefficient, expected.scale, COEFFICIENT_TOLERANCE, "H");
      expectClose(result.scale.standardError, expected.scaleError, COEFFICIENT_TOLERANCE, "se H");
    });
  }

  it("gives 1 with no standard error for a perfect Guttman scale", () => {
    const result = scalability(responseMatrix(guttman(4, 20)));
    expect(result.scale.coefficient).toBeCloseTo(1, 12);
    expect(result.scale.standardError).toBeCloseTo(0, 12);
    for (const item of result.items) expect(item.coefficient).toBeCloseTo(1, 12);
    for (const pair of result.pairs.flat()) if (pair !== null) expect(pair.coefficient).toBeCloseTo(1, 12);
  });

  it("gives 0 for two dichotomous items that are independent in the sample", () => {
    const result = scalability(
      responseMatrix([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
      ]),
    );
    expect(result.scale.coefficient).toBeCloseTo(0, 12);
    expect(result.pairs[0]?.[1]?.coefficient).toBeCloseTo(0, 12);
  });

  it("uses the persons who answered every item", () => {
    const reference = defined(cases.find(({ name }) => name === "DS14"));
    const complete = reference.responses.filter((row) => !row.includes(null));
    expect(complete.length).toBeLessThan(reference.responses.length);
    expect(scalability(matrixOf(reference))).toEqual(
      scalability(responseMatrix(complete, { itemIds: reference.itemIds })),
    );
  });

  it("is the same for scores shifted to start above 0", () => {
    const reference = defined(cases.find(({ name }) => name === "cavalini"));
    const shifted = reference.responses.map((row) => row.map((score) => (score ?? 0) + 2));
    expect(scalability(responseMatrix(shifted, { itemIds: reference.itemIds }))).toEqual(
      scalability(matrixOf(reference)),
    );
  });

  it("leaves out an item without variance, which has no coefficients", () => {
    const base = guttman(3, 12).map((row, person) => [...row.slice(0, 2), person % 5 < 2 ? 1 : 0, row[2] ?? 0]);
    const plain = scalability(responseMatrix(base));
    const withConstant = scalability(
      responseMatrix(
        base.map((row) => [...row, 1]),
        { itemIds: ["a", "b", "c", "d", "constant"] },
      ),
    );
    expect(withConstant.items[4]).toEqual({ id: "constant", coefficient: null, standardError: null });
    expect(withConstant.pairs[4]?.[0]).toEqual({ coefficient: null, standardError: null });
    expect(withConstant.pairs[0]?.[4]).toEqual({ coefficient: null, standardError: null });
    expect(withConstant.scale).toEqual(plain.scale);
    withConstant.items.slice(0, 4).forEach((item, i) => {
      expect(item.coefficient).toBe(plain.items[i]?.coefficient);
      expect(item.standardError).toBe(plain.items[i]?.standardError);
    });
  });

  it("has no coefficients when every item is constant", () => {
    const result = scalability(
      responseMatrix([
        [1, 1],
        [1, 1],
        [1, 1],
      ]),
    );
    expect(result.scale).toEqual({ coefficient: null, standardError: null });
    expect(result.items.map(({ coefficient }) => coefficient)).toEqual([null, null]);
  });

  it("rejects matrices that cannot be analyzed", () => {
    expect(() => scalability(JSON.parse("null"))).toThrow(DataError);
    expect(() => scalability(responseMatrix([[0], [1], [1]]))).toThrow("Scalability needs at least 2 items, found 1.");
    expect(() =>
      scalability(
        responseMatrix([
          [0, 1],
          [1, null],
          [null, 1],
        ]),
      ),
    ).toThrow("Scalability needs at least 2 persons who answered every item, found 1.");
    expect(() =>
      scalability(
        responseMatrix([
          [0, 10],
          [1, 2],
          [2, 3],
        ]),
      ),
    ).toThrow("Scalability needs scores from at most 10 categories, as the mokken package does, found 11.");
    expect(() => scalability(responseMatrix([[0, 1, 1]]))).toThrow(DataError);
  });
});

describe("automatedItemSelection", () => {
  for (const reference of cases) {
    it(`matches the scales of aisp for ${reference.name}`, () => {
      const matrix = matrixOf(reference);
      for (const { lowerBound, alpha, scales } of reference.selections) {
        const result = automatedItemSelection(matrix, { lowerBound, alpha });
        const numbers = [...new Set(scales.filter((scale) => scale > 0))].sort((a, b) => a - b);
        const expected = numbers.map((scale) => reference.itemIds.filter((_, i) => scales[i] === scale));
        expect(result.scales, `lower bound ${lowerBound}, alpha ${alpha}`).toEqual(expected);
        expect(result.unscalable, `lower bound ${lowerBound}, alpha ${alpha}`).toEqual(
          reference.itemIds.filter((_, i) => scales[i] === 0),
        );
      }
    });
  }

  const reference = defined(cases.find(({ name }) => name === "two dimensions"));
  const matrix = matrixOf(reference);

  it("takes 0.3 and 0.05 by default", () => {
    expect(automatedItemSelection(matrix)).toEqual(automatedItemSelection(matrix, { lowerBound: 0.3, alpha: 0.05 }));
  });

  it("puts two dimensions of items in two scales and the noise items in none", () => {
    const result = automatedItemSelection(matrix);
    // The scale of the pair with the largest H_ij is formed first.
    expect(result.scales).toEqual([
      ["i7", "i8", "i9", "i10", "i11", "i12"],
      ["i1", "i2", "i3", "i4", "i5", "i6"],
    ]);
    expect(result.unscalable).toEqual(["i13", "i14"]);
  });

  it("leaves every item in exactly one scale or none", () => {
    for (const { lowerBound } of reference.selections) {
      const result = automatedItemSelection(matrix, { lowerBound });
      const placed = [...result.scales.flat(), ...result.unscalable];
      expect([...placed].sort()).toEqual([...reference.itemIds].sort());
    }
  });

  it("leaves no scale when the lower bound is above every coefficient of a pair", () => {
    expect(automatedItemSelection(matrix, { lowerBound: 0.99 })).toEqual({ scales: [], unscalable: reference.itemIds });
  });

  it("leaves out items without variance and does not let them change the scales", () => {
    const withConstant = responseMatrix(
      reference.responses.map((row) => [...row, 1]),
      { itemIds: [...reference.itemIds, "constant"] },
    );
    const result = automatedItemSelection(withConstant);
    expect(result.scales).toEqual(automatedItemSelection(matrix).scales);
    expect(result.unscalable).toEqual([...automatedItemSelection(matrix).unscalable, "constant"]);
  });

  it("lists the items of a scale in the order of the matrix", () => {
    const reversed = responseMatrix(
      reference.responses.map((row) => [...row].reverse()),
      { itemIds: [...reference.itemIds].reverse() },
    );
    for (const scale of automatedItemSelection(reversed).scales) {
      const positions = scale.map((id) => defined(reversed.itemIds.indexOf(id)));
      expect(positions).toEqual([...positions].sort((a, b) => a - b));
    }
  });

  it("rejects options of other types and matrices that cannot be analyzed", () => {
    expect(() => automatedItemSelection(matrix, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => automatedItemSelection(matrix, { lowerBound: JSON.parse('"0.3"') })).toThrow(
      'The option lowerBound must be a number from 0 to 1, found "0.3".',
    );
    expect(() => automatedItemSelection(matrix, { lowerBound: 1.5 })).toThrow(DataError);
    expect(() => automatedItemSelection(matrix, { alpha: 0 })).toThrow(
      "The option alpha must be a positive number, found 0.",
    );
    expect(() => automatedItemSelection(matrix, { alpha: 1 })).toThrow(
      "The option alpha must be a number above 0 and below 1, found 1.",
    );
    expect(() => automatedItemSelection(matrix, { alpha: JSON.parse("null") })).toThrow(DataError);
    expect(() => automatedItemSelection(responseMatrix([[0], [1]]))).toThrow(DataError);
  });
});

describe("conditionalAssociation", () => {
  for (const reference of steps.association) {
    it(`matches W1, W2, W3 and the flags of the first pass of check.ca for ${reference.name}`, () => {
      const matrix = responseMatrix(reference.responses, { itemIds: reference.itemIds });
      const result = conditionalAssociation(matrix, { minimumSize: reference.minimumSize });
      expect(result.items).toEqual(reference.itemIds);
      const expectIndex = (
        actual: readonly (readonly (number | null)[])[],
        expected: readonly (number | null)[][],
        name: string,
      ) => {
        expected.forEach((row, a) => {
          row.forEach((value, b) => {
            expectClose(actual[a]?.[b] ?? null, value, ASSOCIATION_TOLERANCE, `${name} ${a + 1},${b + 1}`);
          });
        });
      };
      expectIndex(result.w1, reference.w1, "W1");
      expectIndex(result.w3, reference.w3, "W3");
      reference.w2.forEach((value, a) => {
        expectClose(defined(result.w2[a]), value, ASSOCIATION_TOLERANCE, `W2 ${a + 1}`);
      });
      expect(result.flagged).toEqual(reference.flagged);
    });
  }

  const dependent = defined(steps.association.find(({ name }) => name === "a dependent pair"));
  const matrix = responseMatrix(dependent.responses, { itemIds: dependent.itemIds });

  it("flags the planted pair of positively dependent items by W1, and one of them by W2", () => {
    const { flagged } = conditionalAssociation(matrix);
    expect(flagged.w1).toContainEqual(["d9", "d10"]);
    expect(flagged.w2).toContain("d9");
  });

  it("gives W2 as the sums of the rows of W3, which is symmetric", () => {
    const result = conditionalAssociation(matrix);
    result.w3.forEach((row, a) => {
      row.forEach((value, b) => {
        expect(value).toBe(result.w3[b]?.[a]);
      });
      const sum = row.reduce<number>((total, value) => total + (value ?? 0), 0);
      expectClose(defined(result.w2[a]), sum, 1e-12, `W2 ${a + 1}`);
    });
  });

  it("takes rest score groups of at least 4 persons by default, and larger groups when asked", () => {
    expect(conditionalAssociation(matrix, { minimumSize: 4 })).toEqual(conditionalAssociation(matrix));
    expect(conditionalAssociation(matrix, { minimumSize: 30 }).w3).not.toEqual(conditionalAssociation(matrix).w3);
  });

  it("gives 0 to every index of copies of one item, which are constant within every group, and flags none", () => {
    const column = [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1];
    const result = conditionalAssociation(responseMatrix(column.map((score) => [score, score, score])));
    expect(result.w1).toEqual([
      [null, 0, 0],
      [0, null, 0],
      [0, 0, null],
    ]);
    expect(result.w2).toEqual([0, 0, 0]);
    expect(result.flagged).toEqual({ w1: [], w2: [], w3: [] });
  });

  it("gives a weight of 1/2 to each group in which two items are uncorrelated", () => {
    // Given item 3, each of its two groups holds 4 persons whose first two items are uncorrelated, a weight of 1/2:
    // W1 of item 1 given item 3 is 1/2 + 1/2.
    const rows = [
      [0, 0, 0],
      [0, 1, 0],
      [1, 0, 0],
      [1, 1, 0],
      [0, 0, 1],
      [0, 1, 1],
      [1, 0, 1],
      [1, 1, 1],
    ];
    expect(conditionalAssociation(responseMatrix(rows)).w1[0]?.[2]).toBe(1);
  });

  it("rejects options of other types, fewer than three items and groups larger than the persons", () => {
    expect(() => conditionalAssociation(matrix, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => conditionalAssociation(matrix, { minimumSize: JSON.parse('"4"') })).toThrow(
      'The option minimumSize must be a positive integer, found "4".',
    );
    expect(() => conditionalAssociation(matrix, { minimumSize: 0 })).toThrow("The option minimumSize");
    expect(() => conditionalAssociation(matrix, { minimumSize: 601 })).toThrow(
      "The option minimumSize must be at most the 600 persons who answered every item, found 601.",
    );
    expect(() =>
      conditionalAssociation(
        responseMatrix([
          [0, 1],
          [1, 0],
        ]),
      ),
    ).toThrow("Conditional association needs at least 3 items, found 2.");
    expect(() => conditionalAssociation(JSON.parse("null"))).toThrow(DataError);
  });
});

describe("monotonicity", () => {
  for (const reference of cases) {
    it(`matches the counts, sums and criterion of check.monotonicity for ${reference.name}`, () => {
      const matrix = matrixOf(reference);
      for (const { minimumViolation, minimumGroupSize, items } of reference.monotonicity) {
        const result = monotonicity(matrix, { minimumViolation, ...groupSize(minimumGroupSize) });
        expectChecks(result, items, `minimum violation ${minimumViolation}`);
      }
    });
  }

  it("counts one violation more than check.monotonicity where a z lies between its cutoff of 1.6449 and the exact quantile", () => {
    const differences = roundedByMokken(({ monotonicity: runs }) => runs).map(
      ({ name, id, significantViolations, mokkenRounded }) => [
        name,
        id,
        significantViolations - (mokkenRounded?.significantViolations ?? 0),
      ],
    );
    expect(differences).toEqual([["cavalini", "Item17", 1]]);
  });

  /** Persons in three rest score groups of 50; the item passed by 40, 30 and 10 of them, by rest score 0, 1 and 2. */
  const decreasing = (() => {
    const rows: number[][] = [];
    const passed = [40, 30, 10];
    for (let group = 0; group < 3; group++) {
      for (let person = 0; person < 50; person++) {
        rows.push([person < (passed[group] ?? 0) ? 1 : 0, group >= 1 ? 1 : 0, group >= 2 ? 1 : 0]);
      }
    }
    return responseMatrix(rows, { itemIds: ["decreasing", "first", "second"] });
  })();

  it("counts the violations of an item whose proportions fall with the rest score, as derived by hand", () => {
    const check = defined(monotonicity(decreasing).find(({ id }) => id === "decreasing"));
    // Proportions 0.8, 0.6 and 0.2: three pairs of groups, each a decrease above 0.03, of 0.2, 0.4 and 0.6.
    expect(check.activeComparisons).toBe(3);
    expect(check.violations).toBe(3);
    expect(check.maximumViolation).toBeCloseTo(0.6, 12);
    expect(check.violationSum).toBeCloseTo(1.2, 12);
    // The largest z, of the groups 0 and 2: 2 |√(11 · 11) - √(40 · 40)| / √99 = 58 / √99.
    expect(check.maximumZ).toBeCloseTo(58 / Math.sqrt(99), 10);
    expect(check.significantViolations).toBe(3);
    expect(check.criterion).toBeGreaterThan(700);
  });

  it("finds no violation in a perfect Guttman scale", () => {
    for (const check of monotonicity(responseMatrix(guttman(4, 25)))) {
      expect(check.violations).toBe(0);
      expect(check.maximumViolation).toBe(0);
      expect(check.maximumZ).toBeNull();
      expect(check.criterion).toBe(0);
    }
  });

  it("uses a minimum group size of a tenth of the persons from 500 and a fifth below", () => {
    const reference = defined(cases.find(({ name }) => name === "two dimensions"));
    expect(reference.responses).toHaveLength(500);
    const matrix = matrixOf(reference);
    expect(monotonicity(matrix)).toEqual(monotonicity(matrix, { minimumGroupSize: 50 }));
    expect(monotonicity(matrix)).not.toEqual(monotonicity(matrix, { minimumGroupSize: 100 }));
    const fewer = responseMatrix(reference.responses.slice(0, 499), { itemIds: reference.itemIds });
    expect(monotonicity(fewer)).toEqual(monotonicity(fewer, { minimumGroupSize: 99 }));
  });

  it("counts a decrease only above the minimum violation", () => {
    const strict = monotonicity(decreasing, { minimumViolation: 0.3 });
    const check = defined(strict.find(({ id }) => id === "decreasing"));
    expect(check.violations).toBe(2);
    expect(check.violationSum).toBeCloseTo(1, 12);
    expect(monotonicity(decreasing, { minimumViolation: 0.7 }).every(({ violations }) => violations === 0)).toBe(true);
  });

  it("finds fewer significant violations at a lower level", () => {
    const loose = defined(monotonicity(decreasing, { alpha: 0.05 }).find(({ id }) => id === "decreasing"));
    const strict = defined(monotonicity(decreasing, { alpha: 0.001 }).find(({ id }) => id === "decreasing"));
    expect(loose.significantViolations).toBe(3);
    expect(strict.significantViolations).toBe(2);
  });

  it("rejects options of other types and too few persons", () => {
    expect(() => monotonicity(decreasing, JSON.parse("null"))).toThrow("The options must be an object, found null.");
    expect(() => monotonicity(decreasing, { minimumViolation: -0.1 })).toThrow(
      "The option minimumViolation must be a non-negative number, found -0.1.",
    );
    expect(() => monotonicity(decreasing, { minimumViolation: JSON.parse("null") })).toThrow(DataError);
    expect(() => monotonicity(decreasing, { minimumGroupSize: 0 })).toThrow(
      "The option minimumGroupSize must be a positive integer, found 0.",
    );
    expect(() => monotonicity(decreasing, { minimumGroupSize: JSON.parse('"50"') })).toThrow(DataError);
    expect(() => monotonicity(decreasing, { alpha: 1 })).toThrow(
      "The option alpha must be a number above 0 and below 1, found 1.",
    );
    expect(() => monotonicity(decreasing, { minimumGroupSize: 100 })).toThrow(
      "The minimum group size 100 is more than half of the 150 persons who answered every item.",
    );
    expect(() => monotonicity(responseMatrix(guttman(3, 5)))).toThrow(
      "The monotonicity check needs at least 50 persons, the minimum group size, who answered every item, found 20.",
    );
    expect(() => monotonicity(responseMatrix(guttman(3, 5)), { minimumGroupSize: 5 })).not.toThrow();
  });
});

describe("invariantItemOrdering", () => {
  for (const reference of cases) {
    it(`matches the counts, criterion, removed items and H^T of check.iio for ${reference.name}`, () => {
      const matrix = matrixOf(reference);
      for (const { alpha, selection, minimumGroupSize, items, removed, coefficient } of reference.invariantOrdering) {
        const label = `alpha ${alpha}, selection ${selection}`;
        const result = invariantItemOrdering(matrix, { alpha, selection, ...groupSize(minimumGroupSize) });
        expectChecks(result.items, items, label);
        expect(result.removed, label).toEqual(removed);
        expectClose(result.coefficient, coefficient, COEFFICIENT_TOLERANCE, `${label} H^T`);
      }
    });
  }

  it("has a criterion one above the summary of check.iio where that summary rounds H_i across a whole number", () => {
    const differences = roundedByMokken(({ invariantOrdering: runs }) => runs).map(
      ({ name, id, criterion, mokkenRounded }) => [name, id, criterion - (mokkenRounded?.criterion ?? 0)],
    );
    expect(differences).toEqual([
      ["acl four scales", "withdrawn*", 1],
      ["acl four scales", "withdrawn*", 1],
    ]);
  });

  it("finds no violation in a perfect Guttman scale, whose H^T is 1", () => {
    const result = invariantItemOrdering(responseMatrix(guttman(4, 25)), { minimumGroupSize: 10 });
    expect(result.removed).toEqual([]);
    expect(result.coefficient).toBeCloseTo(1, 12);
    for (const check of result.items) {
      expect(check.violations).toBe(0);
      expect(check.criterion).toBe(0);
    }
  });

  const noisy = defined(cases.find(({ name }) => name === "two dimensions"));
  const noisyMatrix = matrixOf(noisy);

  it("removes the items that violate the ordering with the most others, noise items first", () => {
    expect(invariantItemOrdering(noisyMatrix).removed).toEqual(["i13", "i14"]);
  });

  it("keeps every item without selection and has a lower H^T than with it", () => {
    const kept = invariantItemOrdering(noisyMatrix, { selection: false });
    expect(kept.removed).toEqual([]);
    expect(kept.coefficient).toBeLessThan(defined(invariantItemOrdering(noisyMatrix).coefficient));
  });

  it("takes 0.03 per step of the highest score as the minimum violation", () => {
    const reference = defined(cases.find(({ name }) => name === "cavalini"));
    const matrix = matrixOf(reference);
    expect(invariantItemOrdering(matrix)).toEqual(invariantItemOrdering(matrix, { minimumViolation: 0.09 }));
  });

  it("counts a group in which every person scores the same amount higher on the harder item as significant", () => {
    // Items a and b are scored 0 to 2 and c separates three groups of 20 by its score. In the group with c = 0, b is a
    // plus 1 for every person, so the paired differences have no variance; a is easier overall.
    const rows: number[][] = [];
    for (let person = 0; person < 20; person++) rows.push([person % 2, (person % 2) + 1, 0]);
    for (let person = 0; person < 20; person++) rows.push([2, 0, 1]);
    for (let person = 0; person < 20; person++) rows.push([2, 1, 2]);
    const result = invariantItemOrdering(responseMatrix(rows, { itemIds: ["a", "b", "c"] }), { minimumGroupSize: 20 });
    const [first, second] = [defined(result.items[0]), defined(result.items[1])];
    expect(first.significantViolations).toBeGreaterThan(0);
    expect(second.significantViolations).toBeGreaterThan(0);
  });

  it("rejects options of other types, fewer than three items and too few persons", () => {
    expect(() => invariantItemOrdering(noisyMatrix, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => invariantItemOrdering(noisyMatrix, { selection: JSON.parse('"yes"') })).toThrow(
      "The option selection must be true or false, found yes.",
    );
    expect(() => invariantItemOrdering(noisyMatrix, { minimumGroupSize: 1 })).toThrow(DataError);
    expect(() => invariantItemOrdering(noisyMatrix, { alpha: 0 })).toThrow(DataError);
    expect(() => invariantItemOrdering(responseMatrix(guttman(2, 30)))).toThrow(
      "The invariant item ordering check needs at least 3 items, found 2.",
    );
    expect(() => invariantItemOrdering(responseMatrix(guttman(3, 10)))).toThrow(
      "The invariant item ordering check needs a minimum group size of at least 2, and at least that many persons who answered every item; found 50 and 40.",
    );
    expect(() => invariantItemOrdering(JSON.parse("null"))).toThrow(DataError);
  });
});

describe("molenaarSijtsmaReliability", () => {
  for (const reference of reliability.cases) {
    it(`matches check.reliability of the mokken package on ${reference.name}`, () => {
      const matrix = responseMatrix(reference.responses, { itemIds: reference.itemIds });
      expectClose(molenaarSijtsmaReliability(matrix), reference.ms, RELIABILITY_TOLERANCE, reference.name);
    });
  }

  it("equals mokken's own value where its rounding does not split tied steps, and stays within 0.03 where it does", () => {
    for (const reference of [...reliability.cases, ...reliability.study]) {
      const gap = Math.abs(reference.ms - reference.mokkenMs);
      expect(gap).toBeLessThanOrEqual(reference.splitTies === 0 ? ROUNDING_TOLERANCE : SPLIT_TIES_GAP);
    }
  });

  it("estimates the reliability of parallel forms in a study of the graded response model", () => {
    const estimates = reliability.study.map((set) => molenaarSijtsmaReliability(responseMatrix(set.responses)));
    estimates.forEach((estimate, i) => {
      expectClose(estimate, defined(reliability.study[i]).ms, RELIABILITY_TOLERANCE, `set ${i}`);
    });
    const average = (values: readonly number[]): number =>
      values.reduce((sum, value) => sum + value, 0) / values.length;
    const parallel = average(reliability.study.map(({ parallelForms }) => parallelForms));
    expect(Math.abs(average(estimates) - parallel)).toBeLessThan(STUDY_BIAS);
    // Coefficient alpha is a lower bound of the reliability, and MS is the closer estimate.
    const alpha = average(reliability.study.map((set) => set.alpha));
    expect(alpha).toBeLessThan(average(estimates));
  });

  it("derives the coefficient of three response patterns by hand", () => {
    // One person with no item passed, one passing the first, two passing both. The steps pass 3/4 (item 1) and 1/2
    // (item 2), observed together by 1/2. The step of item 2 pairs with itself 1/3 and 1/2 from its neighbors, 5/12 on
    // average; that of item 1, 3/4 and 5/8, 11/16. Together the joint proportions exceed the products by 5/12 +
    // 2 (1/2 - 3/8) + 11/16 - 9/16 = 13/24 and the total's variance is 11/16, so MS = 26/33.
    const rows = [
      [0, 0],
      [1, 0],
      [1, 1],
      [1, 1],
    ];
    expect(molenaarSijtsmaReliability(responseMatrix(rows))).toBeCloseTo(26 / 33, 14);
  });

  it("is 1 minus the error variance share, so a Guttman scale is more reliable than independent items", () => {
    const scale = molenaarSijtsmaReliability(responseMatrix(guttman(6, 20)));
    expect(scale).toBeGreaterThan(0.8);
  });

  it("takes the lowest score of all items as 0", () => {
    const rows = [
      [0, 1, 2, 1],
      [1, 1, 2, 0],
      [2, 0, 1, 2],
      [3, 2, 3, 2],
      [0, 0, 0, 1],
      [2, 2, 1, 1],
    ];
    expect(molenaarSijtsmaReliability(responseMatrix(rows.map((row) => row.map((score) => score + 2))))).toBe(
      molenaarSijtsmaReliability(responseMatrix(rows)),
    );
  });

  it("uses the persons who answered every item", () => {
    const rows = [
      [0, 0, 1],
      [1, 0, 0],
      [1, 1, 1],
      [1, 1, 0],
      [0, 1, 1],
      [1, 0, 1],
    ];
    const withMissing = [...rows, [null, 1, 1], [1, null, 0]];
    expect(molenaarSijtsmaReliability(responseMatrix(withMissing))).toBe(
      molenaarSijtsmaReliability(responseMatrix(rows)),
    );
  });

  it("rejects a matrix of the wrong shape or of scores without variance", () => {
    expect(() => molenaarSijtsmaReliability(responseMatrix([[1], [0], [1]]))).toThrow(
      "Molenaar-Sijtsma reliability needs at least 2 items, found 1.",
    );
    expect(() => molenaarSijtsmaReliability(responseMatrix([[1, 0]]))).toThrow(
      "Molenaar-Sijtsma reliability needs at least 2 persons who answered every item, found 1.",
    );
    expect(() =>
      molenaarSijtsmaReliability(
        responseMatrix([
          [1, 0],
          [0, 1],
        ]),
      ),
    ).toThrow("Molenaar-Sijtsma reliability needs total scores with variance.");
    expect(() =>
      molenaarSijtsmaReliability(
        responseMatrix([
          [0, 0],
          [10, 0],
        ]),
      ),
    ).toThrow("needs scores from at most 10 categories, as the mokken package does, found 11.");
    expect(() => molenaarSijtsmaReliability(JSON.parse("null"))).toThrow(DataError);
  });
});

describe("properties", () => {
  /** Dichotomous scores of 60 to 120 persons on 4 to 6 items, with every item varying. */
  const matrix = fc
    .tuple(fc.integer({ min: 60, max: 120 }), fc.integer({ min: 4, max: 6 }))
    .chain(([persons, items]) =>
      fc.array(fc.array(fc.integer({ min: 0, max: 1 }), { minLength: items, maxLength: items }), {
        minLength: persons,
        maxLength: persons,
      }),
    )
    .filter((rows) => (rows[0] ?? []).every((_, item) => new Set(rows.map((row) => row[item])).size === 2));

  it("imputation leaves the observed responses as they are and imputes within the bounds", () => {
    const withMissing = matrix.chain((rows) =>
      fc
        .array(fc.boolean(), { minLength: rows.length, maxLength: rows.length })
        .map((drops) =>
          rows.map((row, person) =>
            row.map((value, item) => (drops[person] && item === person % row.length ? null : value)),
          ),
        ),
    );
    fc.assert(
      fc.property(withMissing, fc.integer({ min: 0, max: 1000 }), (rows, seed) => {
        const imputed = twoWayImputation(responseMatrix(rows), createRandom(seed)).responses;
        rows.forEach((row, person) => {
          row.forEach((value, item) => {
            const result = defined(imputed[person]?.[item]);
            if (value !== null) expect(result).toBe(value);
            else expect(result === 0 || result === 1).toBe(true);
          });
        });
      }),
      { numRuns: 30 },
    );
  });

  it("G+ is 0 for every person of a perfect Guttman scale, and is a multiple of 1/2", () => {
    expect(guttmanErrors(responseMatrix(guttman(5, 3))).errors.every((count) => count === 0)).toBe(true);
    fc.assert(
      fc.property(matrix, (rows) => {
        for (const count of guttmanErrors(responseMatrix(rows)).errors) {
          expect(Number.isInteger(2 * defined(count))).toBe(true);
          expect(defined(count)).toBeGreaterThanOrEqual(0);
        }
      }),
      { numRuns: 30 },
    );
  });

  it("the indices of conditional association and their flags do not depend on the order of the persons", () => {
    fc.assert(
      fc.property(matrix, (rows) => {
        const forward = conditionalAssociation(responseMatrix(rows));
        const backward = conditionalAssociation(responseMatrix([...rows].reverse()));
        expect(backward.flagged).toEqual(forward.flagged);
        forward.w1.flat().forEach((value, entry) => {
          expect(backward.w1.flat()[entry] ?? null).toBeCloseTo(value ?? 0, 12);
        });
        forward.w2.forEach((value, item) => {
          expect(backward.w2[item]).toBeCloseTo(value, 12);
        });
      }),
      { numRuns: 20 },
    );
  });

  it("pair coefficients are at most 1 and standard errors are not negative", () => {
    fc.assert(
      fc.property(matrix, (rows) => {
        const result = scalability(responseMatrix(rows));
        for (const pair of result.pairs.flat()) {
          if (pair === null) continue;
          expect(pair.coefficient ?? 0).toBeLessThanOrEqual(1 + 1e-12);
          expect(pair.standardError ?? 0).toBeGreaterThanOrEqual(0);
        }
        expect(result.scale.coefficient ?? 0).toBeLessThanOrEqual(1 + 1e-12);
      }),
      { numRuns: 30 },
    );
  });

  it("results do not depend on the order of the persons", () => {
    fc.assert(
      fc.property(matrix, (rows) => {
        const reversed = [...rows].reverse();
        expect(scalability(responseMatrix(reversed)).scale.coefficient).toBeCloseTo(
          scalability(responseMatrix(rows)).scale.coefficient ?? 0,
          12,
        );
        expect(automatedItemSelection(responseMatrix(reversed)).scales).toEqual(
          automatedItemSelection(responseMatrix(rows)).scales,
        );
        expect(monotonicity(responseMatrix(reversed), { minimumGroupSize: 20 })).toEqual(
          monotonicity(responseMatrix(rows), { minimumGroupSize: 20 }),
        );
      }),
      { numRuns: 20 },
    );
  });

  it("violations never outnumber the active comparisons of an item, and criteria are not negative", () => {
    fc.assert(
      fc.property(matrix, (rows) => {
        for (const check of monotonicity(responseMatrix(rows), { minimumGroupSize: 20 })) {
          expect(check.violations).toBeLessThanOrEqual(check.activeComparisons);
          expect(check.criterion).toBeGreaterThanOrEqual(0);
          expect(check.significantViolations).toBeLessThanOrEqual(check.violations);
        }
      }),
      { numRuns: 30 },
    );
  });

  it("an item permutation permutes the coefficients and keeps the scale's", () => {
    fc.assert(
      fc.property(matrix, (rows) => {
        const reversed = rows.map((row) => [...row].reverse());
        const ids = (rows[0] ?? []).map((_, item) => `item${item + 1}`);
        const original = scalability(responseMatrix(rows, { itemIds: ids }));
        const permuted = scalability(responseMatrix(reversed, { itemIds: [...ids].reverse() }));
        expect(permuted.scale.coefficient).toBeCloseTo(original.scale.coefficient ?? 0, 12);
        expect(permuted.scale.standardError).toBeCloseTo(original.scale.standardError ?? 0, 10);
        [...permuted.items].reverse().forEach((item, i) => {
          expect(item.coefficient).toBeCloseTo(original.items[i]?.coefficient ?? 0, 12);
        });
      }),
      { numRuns: 20 },
    );
  });

  it("the reliability of the total score does not depend on the order of persons or items", () => {
    fc.assert(
      fc.property(matrix, (rows) => {
        const forward = molenaarSijtsmaReliability(responseMatrix(rows));
        expect(molenaarSijtsmaReliability(responseMatrix([...rows].reverse()))).toBeCloseTo(forward, 12);
        expect(molenaarSijtsmaReliability(responseMatrix(rows.map((row) => [...row].reverse())))).toBeCloseTo(
          forward,
          12,
        );
      }),
      { numRuns: 30 },
    );
  });

  it("the reliability of the total score is finite and is the same after a JSON round trip", () => {
    for (const reference of reliability.cases) {
      const value = molenaarSijtsmaReliability(responseMatrix(reference.responses, { itemIds: reference.itemIds }));
      expect(Number.isFinite(value), reference.name).toBe(true);
      expect(
        molenaarSijtsmaReliability(
          JSON.parse(JSON.stringify(responseMatrix(reference.responses, { itemIds: reference.itemIds }))),
        ),
      ).toBe(value);
    }
  });
});
