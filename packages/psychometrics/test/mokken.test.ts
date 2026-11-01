import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  automatedItemSelection,
  DataError,
  type ItemCheck,
  invariantItemOrdering,
  molenaarSijtsmaReliability,
  monotonicity,
  type ResponseMatrix,
  responseMatrix,
  scalability,
} from "../src/index.js";
import { defined, expectClose } from "./assertions.js";
import fixture from "./fixtures/mokken.json" with { type: "json" };
import reliability from "./fixtures/mokken-reliability.json" with { type: "json" };

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
