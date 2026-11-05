import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { buildResponseMatrix, type ResponseMatrix } from "../src/data/index.js";
import { DataError } from "../src/index.js";
import { type IrtItem, nominalResponse, threeParameterLogistic, twoParameterLogistic } from "../src/irt/index.js";
import {
  generalizedBinomialTest,
  kIndex,
  m4Index,
  omegaIndex,
  type PersonPair,
  selectPairs,
} from "../src/similarity/index.js";
import { defined } from "./assertions.js";
import { readFixture } from "./fixtures.js";

interface SimilarityCopyingFixture {
  personIds: string[];
  itemIds: string[];
  key: number[];
  nominal: { responses: number[][]; items: { a: number[]; c: number[] }[]; thetas: number[] };
  scores: { responses: number[][]; items: { a: number; b: number; c: number; d: number }[]; thetas: number[] };
  rooms: string[];
  roomPairs: [number, number][];
  planted: [number, number][];
  pairs: [number, number][];
  expected: Record<"nominal" | "scores", CopyDetectIndices[]>;
}

interface CopyDetectIndices {
  omega: number;
  omegaSignificance: number;
  expectedMatches: number;
  generalizedBinomial: number;
  m4: number;
  k: number | null;
  k1: number | null;
  k2: number | null;
  s1: number | null;
  s2: number | null;
}

const fixture = readFixture<SimilarityCopyingFixture>("similarity-copying.json");
const idOf = (person: number): string => defined(fixture.personIds[person]);
const pairs = fixture.pairs.map(([a, b]): PersonPair => [idOf(a), idOf(b)]);

const FORMS = {
  nominal: {
    matrix: buildResponseMatrix(fixture.nominal.responses, { personIds: fixture.personIds, itemIds: fixture.itemIds }),
    items: fixture.nominal.items.map((parameters): IrtItem => ({ model: nominalResponse, parameters })),
    thetas: fixture.nominal.thetas,
    key: fixture.key,
  },
  scores: {
    matrix: buildResponseMatrix(fixture.scores.responses, { personIds: fixture.personIds, itemIds: fixture.itemIds }),
    items: fixture.scores.items.map((parameters): IrtItem => ({ model: threeParameterLogistic, parameters })),
    thetas: fixture.scores.thetas,
    key: fixture.itemIds.map(() => 1),
  },
};

/** The relative tolerances of docs/psychometrics/validation.md. */
const TOLERANCE = { omega: 1e-10, distribution: 1e-12, binomial: 1e-10, poisson: 1e-8 };

/** Expects each value within `tolerance` times the reference's of it, and `null` where the reference has none. */
function expectRelative(actual: readonly (number | null)[], expected: readonly (number | null)[], tolerance: number) {
  expect(actual).toHaveLength(expected.length);
  expected.forEach((value, k) => {
    const found = actual[k] ?? null;
    if (value === null) {
      expect(found, `pair ${k + 1}`).toBeNull();
      return;
    }
    expect(Math.abs(defined(found) - value), `pair ${k + 1}`).toBeLessThanOrEqual(tolerance * Math.abs(value));
  });
}

const DICHOTOMOUS: IrtItem[] = [-1, -0.5, 0, 0.5, 1, 1.5].map((b) => ({
  model: twoParameterLogistic,
  parameters: { a: 1.2, b, c: 0, d: 1 },
}));
const NOMINAL: IrtItem = { model: nominalResponse, parameters: { a: [-1, 0, 1], c: [0, 0.5, 0] } };

/** Two persons on `count` items, the first answering 0 and 1 in turn and the second the same. */
function twins(count: number): ResponseMatrix {
  const row = Array.from({ length: count }, (_, i) => i % 2);
  return buildResponseMatrix([row, row.slice()]);
}

describe("selectPairs", () => {
  it("forms every pair of the persons of a room once, as CopyDetect's centers forms them both ways", () => {
    const { pairs: selected, skipped } = selectPairs(fixture.personIds, { room: fixture.rooms });
    const both = selected.flatMap(([a, b]) => [`${a} ${b}`, `${b} ${a}`]).sort();
    const centers = fixture.roomPairs.map(([a, b]) => `${idOf(a)} ${idOf(b)}`).sort();
    expect(both).toEqual(centers);
    expect(skipped).toEqual([]);
  });

  it("keeps a pair that two contexts form once, and orders the pairs by their persons", () => {
    const selection = selectPairs(["a", "b", "c", "d"], {
      room: ["r2", "r1", "r2", "r1"],
      network: ["n1", "n1", "n1", null],
    });
    expect(selection.pairs).toEqual([
      ["a", "b"],
      ["a", "c"],
      ["b", "c"],
      ["b", "d"],
    ]);
  });

  it("forms no pairs of a value more than maxGroup persons share, and names it", () => {
    const selection = selectPairs(
      ["a", "b", "c", "d", "e"],
      { school: ["s1", "s1", "s1", "s2", "s2"], room: ["r1", "r1", null, null, "r2"] },
      { maxGroup: 2 },
    );
    expect(selection.pairs).toEqual([
      ["a", "b"],
      ["d", "e"],
    ]);
    expect(selection.skipped).toEqual([{ context: "school", value: "s1", persons: 3 }]);
  });

  it("forms nothing without contexts, from persons alone in their values or without values", () => {
    expect(selectPairs(["a", "b"], {})).toEqual({ pairs: [], skipped: [] });
    expect(selectPairs(["a", "b"], { room: ["r1", "r2"], network: [null, null] }).pairs).toEqual([]);
  });

  it("removes repeated pairs as it goes, so that contexts forming the same pairs stay within maxPairs", () => {
    const persons = ["a", "b", "c"];
    const contexts = Object.fromEntries(Array.from({ length: 10 }, (_, k) => [`c${k}`, ["v", "v", "v"]]));
    expect(selectPairs(persons, contexts, { maxPairs: 3 }).pairs).toHaveLength(3);
  });

  it("refuses more pairs than maxPairs", () => {
    expect(() => selectPairs(["a", "b", "c"], { room: ["r", "r", "r"] }, { maxPairs: 2 })).toThrow(
      "The contexts form more than 2 pairs, the option maxPairs: pass contexts of fewer persons, a lower maxGroup or a higher maxPairs.",
    );
    const many = Array.from({ length: 60 }, (_, k) => `p${k}`);
    expect(() => selectPairs(many, { room: many.map(() => "r") }, { maxPairs: 100 })).toThrow(DataError);
  });

  it("refuses invalid ids, contexts and options", () => {
    expect(() => selectPairs(["a", "a"], { room: ["r", "r"] })).toThrow('The person id "a" appears more than once.');
    expect(() => selectPairs(["a", "b"], { room: ["r"] })).toThrow('The context "room" has 1 values for 2 persons.');
    expect(() => selectPairs(["a", "b"], JSON.parse('{ "room": ["r", 3] }'))).toThrow(
      'The value of the context "room" of person "b" must be a string or null, found 3.',
    );
    expect(() => selectPairs(["a", "b"], JSON.parse('{ "room": "r" }'))).toThrow(
      'The values of the context "room" must be an array, found "r".',
    );
    expect(() => selectPairs(["a", "b"], {}, { maxGroup: 1 })).toThrow(
      "The option maxGroup must be an integer from 2, found 1.",
    );
    expect(() => selectPairs(["a", "b"], {}, { maxPairs: 0.5 })).toThrow(
      "The option maxPairs must be an integer from 1, found 0.5.",
    );
    expect(() => selectPairs(["a", "b"], {}, JSON.parse('{ "maxPair": 3 }'))).toThrow('There is no option "maxPair"');
  });

  it("refuses arguments of other types", () => {
    expect(() => selectPairs(JSON.parse("null"), {})).toThrow("The person ids must be an array, found null.");
    expect(() => selectPairs(["a"], JSON.parse("null"))).toThrow(
      "The contexts must be an object of each context's values by person, found null.",
    );
    expect(() => selectPairs(["a"], {}, JSON.parse("null"))).toThrow("The options must be an object, found null.");
    expect(() => selectPairs(["a"], {}, JSON.parse('{ "maxGroup": "3" }'))).toThrow(DataError);
  });
});

describe("omegaIndex", () => {
  for (const name of ["nominal", "scores"] as const) {
    it(`matches CopyDetect's omega of ${name} responses`, () => {
      const { matrix, items, thetas } = FORMS[name];
      const expected = fixture.expected[name];
      const result = omegaIndex(matrix, items, thetas, pairs);
      expectRelative(
        result.map((pair) => pair.omega),
        expected.map((pair) => pair.omega),
        TOLERANCE.omega,
      );
      expectRelative(
        result.map((pair) => pair.significance),
        expected.map((pair) => pair.omegaSignificance),
        TOLERANCE.omega,
      );
      expectRelative(
        result.map((pair) => pair.expectedMatches),
        expected.map((pair) => pair.expectedMatches),
        TOLERANCE.omega,
      );
    });
  }

  it("counts the matches of the items both answered, from the copier's side", () => {
    const matrix = buildResponseMatrix(
      [
        [0, 1, 2, null, 1, 0],
        [0, 1, 1, 2, null, 0],
      ],
      { personIds: ["c", "s"] },
    );
    const [forward, backward] = omegaIndex(
      matrix,
      Array(6).fill(NOMINAL),
      [1, -1],
      [
        ["c", "s"],
        ["s", "c"],
      ],
    );
    expect(defined(forward).persons).toEqual(["c", "s"]);
    expect(defined(forward).compared).toBe(4);
    expect(defined(forward).matches).toBe(3);
    expect(defined(forward).omega).not.toBeCloseTo(defined(defined(backward).omega), 3);
  });

  it("gives null when the pair compares no item", () => {
    const matrix = buildResponseMatrix([
      [0, null],
      [null, 1],
    ]);
    expect(omegaIndex(matrix, DICHOTOMOUS.slice(0, 2), [0, 0], [["person1", "person2"]])).toEqual([
      { persons: ["person1", "person2"], compared: 0, matches: 0, expectedMatches: 0, omega: null, significance: null },
    ]);
  });

  it("refuses invalid thetas, pairs and responses", () => {
    const matrix = twins(6);
    expect(() => omegaIndex(matrix, DICHOTOMOUS, [0], [["person1", "person2"]])).toThrow(
      "Found 1 thetas for 2 persons.",
    );
    expect(() => omegaIndex(matrix, DICHOTOMOUS, [0, Number.NaN], [["person1", "person2"]])).toThrow(
      'The theta of person "person2" must be a finite number, found NaN.',
    );
    expect(() => omegaIndex(matrix, DICHOTOMOUS, [0, 0], [["person1", "person3"]])).toThrow(
      'Pair 1 names "person3", which is not a person id of the matrix.',
    );
    expect(() => omegaIndex(matrix, DICHOTOMOUS, [0, 0], [["person1", "person1"]])).toThrow(
      'Pair 1 names the person "person1" twice.',
    );
    expect(() => omegaIndex(matrix, DICHOTOMOUS, [0, 0], JSON.parse('[["person1"]]'))).toThrow(
      "Pair 1 must be an array of two person ids, found an array of length 1.",
    );
    expect(() => omegaIndex(buildResponseMatrix([[2], [0]]), [defined(DICHOTOMOUS[0])], [0, 0], [])).toThrow(
      'The response of person "person1" to item "item1" is 2, but the item has categories 0 to 1.',
    );
  });

  it("refuses arguments of other types", () => {
    const matrix = twins(6);
    expect(() => omegaIndex(JSON.parse("null"), DICHOTOMOUS, [0, 0], [])).toThrow(DataError);
    expect(() => omegaIndex(matrix, JSON.parse("{}"), [0, 0], [])).toThrow(
      "The items must be an array, found an object.",
    );
    expect(() => omegaIndex(matrix, DICHOTOMOUS, JSON.parse('["0", 0]'), [])).toThrow(DataError);
    expect(() => omegaIndex(matrix, DICHOTOMOUS, [0, 0], JSON.parse("null"))).toThrow(
      "The pairs must be an array of pairs of person ids, found null.",
    );
  });
});

describe("kIndex", () => {
  for (const name of ["nominal", "scores"] as const) {
    it(`matches CopyDetect's K, K1, K2, S1 and S2 of ${name} responses`, () => {
      const { matrix, key } = FORMS[name];
      const expected = fixture.expected[name];
      const result = kIndex(matrix, pairs, { key });
      for (const [index, tolerance] of [
        ["k", TOLERANCE.binomial],
        ["k1", TOLERANCE.binomial],
        ["k2", TOLERANCE.binomial],
        ["s1", TOLERANCE.poisson],
        ["s2", TOLERANCE.poisson],
      ] as const) {
        expectRelative(
          result.map((pair) => pair[index]),
          expected.map((pair) => pair[index]),
          tolerance,
        );
      }
    });
  }

  it("gives null indices when the source has no incorrect response", () => {
    const matrix = buildResponseMatrix([
      [1, 0, 1],
      [1, 1, 1],
      [0, 0, 1],
    ]);
    expect(kIndex(matrix, [["person1", "person2"]])).toEqual([
      {
        persons: ["person1", "person2"],
        sourceIncorrect: 0,
        identicalIncorrect: 0,
        k: null,
        k1: null,
        k2: null,
        s1: null,
        s2: null,
      },
    ]);
  });

  it("counts identical incorrect responses as the same category, and a missing response as neither", () => {
    // p1 copies p2's two errors; p3 and p4 make the copier's group, p5 another.
    const matrix = buildResponseMatrix([
      [2, 1, 0, null],
      [2, 1, 0, 0],
      [1, 0, 0, 3],
      [3, 2, 0, 0],
      [0, 0, 0, 0],
    ]);
    const [pair] = kIndex(matrix, [["person1", "person2"]], { key: [0, 0, 0, 0] });
    expect(pair).toMatchObject({ sourceIncorrect: 2, identicalIncorrect: 2 });
    // The copier's group, of two incorrect responses without the source: p1 shares 2, p3 and p4 share 0, so the rate
    // is 2 / 3 over 2 responses, and K the probability of 2 matches of 2 at 1 / 3.
    expect(defined(pair).k).toBeCloseTo(1 / 9, 14);
  });

  it("needs the key when a response is above 1, and refuses invalid options", () => {
    const matrix = buildResponseMatrix([
      [0, 2],
      [1, 2],
    ]);
    expect(() => kIndex(matrix, [["person1", "person2"]])).toThrow(
      "The K index needs the option key when a response is above 1.",
    );
    expect(() => kIndex(matrix, [["person1", "person2"]], { key: [0, -1] })).toThrow(
      'The key of item "item2" must be a category, an integer from 0, found -1.',
    );
    expect(() => kIndex(matrix, [["person1", "person2"]], { key: [0, 0], guessing: 1 })).toThrow(
      "The option guessing must be a number between 0 and 1, exclusive, found 1.",
    );
  });

  it("refuses counts of more than 2^24 cells, which number the categories that occur", () => {
    // Person p gives category p to the items before the pth, and the key to the others: 330 groups, and on item i the
    // key and 329 - i other categories.
    const rows = Array.from({ length: 330 }, (_, person) =>
      Array.from({ length: 329 }, (_, item) => (item < person ? person : 0)),
    );
    expect(() => kIndex(buildResponseMatrix(rows), [["person1", "person2"]], { key: Array(329).fill(0) })).toThrow(
      "The K index counts each category of each item in each group of persons with as many incorrect responses, at most 16777216 cells, found 330 groups of 54614 categories.",
    );
    const large = buildResponseMatrix([[2 ** 31 - 1], [0], [1]]);
    expect(kIndex(large, [["person1", "person3"]], { key: [0] })[0]).toMatchObject({ identicalIncorrect: 0, k: 1 });
  });

  it("refuses arguments of other types", () => {
    expect(() => kIndex(JSON.parse("null"), [])).toThrow(DataError);
    expect(() => kIndex(twins(2), JSON.parse("{}"))).toThrow(
      "The pairs must be an array of pairs of person ids, found an object.",
    );
    expect(() => kIndex(twins(2), [], JSON.parse("null"))).toThrow("The options must be an object, found null.");
    expect(() => kIndex(twins(2), [], JSON.parse('{ "guessing": "0.2" }'))).toThrow(DataError);
  });
});

describe("generalizedBinomialTest", () => {
  for (const name of ["nominal", "scores"] as const) {
    it(`matches CopyDetect's generalized binomial test of ${name} responses`, () => {
      const { matrix, items, thetas } = FORMS[name];
      expectRelative(
        generalizedBinomialTest(matrix, items, thetas, pairs).map((pair) => pair.significance),
        fixture.expected[name].map((pair) => pair.generalizedBinomial),
        TOLERANCE.distribution,
      );
    });
  }

  it("is symmetric in the persons, and 1 at no match or no item", () => {
    const { matrix, items, thetas } = FORMS.nominal;
    const [one, other] = generalizedBinomialTest(matrix, items, thetas, [
      ["p1", "p2"],
      ["p2", "p1"],
    ]);
    expect(defined(other).significance).toBe(defined(one).significance);
    expect(defined(other).expectedMatches).toBeCloseTo(defined(one).expectedMatches, 12);
    const apart = buildResponseMatrix([
      [1, 0, null],
      [0, 1, null],
    ]);
    expect(
      generalizedBinomialTest(apart, DICHOTOMOUS.slice(0, 3), [0, 0], [["person1", "person2"]])[0]?.significance,
    ).toBe(1);
    const disjoint = buildResponseMatrix([
      [1, null],
      [null, 1],
    ]);
    expect(generalizedBinomialTest(disjoint, DICHOTOMOUS.slice(0, 2), [0, 0], [["person1", "person2"]])).toEqual([
      { persons: ["person1", "person2"], compared: 0, matches: 0, expectedMatches: 0, significance: 1 },
    ]);
  });

  it("refuses a pair that compares more than 1,000 items", () => {
    const items = Array.from(
      { length: 1001 },
      (): IrtItem => ({ model: twoParameterLogistic, parameters: { a: 1, b: 0, c: 0, d: 1 } }),
    );
    expect(() => generalizedBinomialTest(twins(1001), items, [0, 0], [["person1", "person2"]])).toThrow(
      "The generalized binomial test compares at most 1000 items in a pair, found a pair that answered 1001 items in common.",
    );
  });
});

describe("m4Index", () => {
  for (const name of ["nominal", "scores"] as const) {
    it(`matches CopyDetect's M4 of ${name} responses`, () => {
      const { matrix, items, thetas, key } = FORMS[name];
      expectRelative(
        m4Index(matrix, items, thetas, pairs, { key }).map((pair) => pair.significance),
        fixture.expected[name].map((pair) => pair.m4),
        TOLERANCE.distribution,
      );
    });
  }

  it("counts identical correct and incorrect responses, symmetrically, with category 1 keyed by default", () => {
    const matrix = buildResponseMatrix([
      [1, 0, 0, 1, null, 1],
      [1, 0, 1, 1, 0, 0],
    ]);
    const [one, other] = m4Index(
      matrix,
      DICHOTOMOUS,
      [0.3, -0.2],
      [
        ["person1", "person2"],
        ["person2", "person1"],
      ],
    );
    expect(one).toMatchObject({ compared: 5, identicalCorrect: 2, identicalIncorrect: 1 });
    expect(defined(other).significance).toBeCloseTo(defined(one).significance, 14);
    const keyed = m4Index(matrix, DICHOTOMOUS, [0.3, -0.2], [["person1", "person2"]], { key: [0, 0, 0, 0, 0, 0] });
    expect(keyed[0]).toMatchObject({ identicalCorrect: 1, identicalIncorrect: 2 });
  });

  it("is 1 at no identical response", () => {
    const matrix = buildResponseMatrix([
      [1, 0, 1],
      [0, 1, 0],
    ]);
    expect(m4Index(matrix, DICHOTOMOUS.slice(0, 3), [0, 0], [["person1", "person2"]])[0]?.significance).toBeCloseTo(
      1,
      14,
    );
  });

  it("needs the key of items of more than two categories, and refuses keys that are not categories", () => {
    const matrix = buildResponseMatrix([
      [0, 2],
      [0, 2],
    ]);
    expect(() => m4Index(matrix, [NOMINAL, NOMINAL], [0, 0], [["person1", "person2"]])).toThrow(
      "M4 needs the option key when an item has more than two categories.",
    );
    expect(() => m4Index(matrix, [NOMINAL, NOMINAL], [0, 0], [["person1", "person2"]], { key: [0, 3] })).toThrow(
      'The key of item "item2" must be a category, an integer from 0 to 2, found 3.',
    );
    expect(() => m4Index(matrix, [NOMINAL, NOMINAL], [0, 0], [["person1", "person2"]], { key: [0] })).toThrow(
      "The option key has 1 categories for 2 items.",
    );
    expect(() => m4Index(matrix, [NOMINAL, NOMINAL], [0, 0], [], JSON.parse('{ "key": null }'))).toThrow(
      "The option key must be an array of the keyed category of each item, found null.",
    );
    expect(() => m4Index(matrix, [NOMINAL, NOMINAL], [0, 0], [], JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
  });

  it("refuses a pair that compares more than 1,000 items", () => {
    const items = Array.from(
      { length: 1001 },
      (): IrtItem => ({ model: twoParameterLogistic, parameters: { a: 1, b: 0, c: 0, d: 1 } }),
    );
    expect(() => m4Index(twins(1001), items, [0, 0], [["person1", "person2"]])).toThrow(
      "M4 compares at most 1000 items in a pair, found a pair that answered 1001 items in common.",
    );
  });
});

describe("properties", () => {
  /** Persons' choices of one of three options on six items, and their thetas. */
  const arbitraryData = fc.record({
    rows: fc.array(fc.array(fc.option(fc.integer({ min: 0, max: 2 }), { freq: 6 }), { minLength: 6, maxLength: 6 }), {
      minLength: 3,
      maxLength: 12,
    }),
    thetas: fc.array(fc.double({ min: -3, max: 3, noNaN: true }), { minLength: 12, maxLength: 12 }),
  });
  const items = Array(6).fill(NOMINAL);
  const key = [0, 1, 2, 0, 1, 2];

  it("gives significances from 0 to 1 and no NaN", () => {
    fc.assert(
      fc.property(arbitraryData, ({ rows, thetas }) => {
        const matrix = buildResponseMatrix(rows);
        const persons = thetas.slice(0, rows.length);
        const all = selectPairs(matrix.personIds, { room: matrix.personIds.map(() => "r") }).pairs;
        const values = [
          ...omegaIndex(matrix, items, persons, all).map((pair) => pair.significance ?? 0.5),
          ...generalizedBinomialTest(matrix, items, persons, all).map((pair) => pair.significance),
          ...m4Index(matrix, items, persons, all, { key }).map((pair) => pair.significance),
          ...kIndex(matrix, all, { key }).flatMap((pair) => [pair.k, pair.k1, pair.k2, pair.s1, pair.s2]),
        ];
        for (const value of values) if (value !== null) expect(value >= 0 && value <= 1).toBe(true);
      }),
    );
  });

  it("leaves out of a pair an item either person did not answer, as if it were not given", () => {
    fc.assert(
      fc.property(arbitraryData, fc.integer({ min: 0, max: 5 }), ({ rows, thetas }, dropped) => {
        const [first, second] = rows;
        if (first === undefined || second === undefined) return;
        const missing = [first.map((response, i) => (i === dropped ? null : response)), second];
        const kept = [first.filter((_, i) => i !== dropped), second.filter((_, i) => i !== dropped)];
        const keptItems = items.slice(0, 5);
        const keptKey = key.filter((_, i) => i !== dropped);
        const both: PersonPair[] = [["person1", "person2"]];
        const one = buildResponseMatrix(missing);
        const other = buildResponseMatrix(kept);
        const at = thetas.slice(0, 2);
        expect(omegaIndex(one, items, at, both)).toEqual(omegaIndex(other, keptItems, at, both));
        expect(generalizedBinomialTest(one, items, at, both)).toEqual(
          generalizedBinomialTest(other, keptItems, at, both),
        );
        expect(m4Index(one, items, at, both, { key })).toEqual(m4Index(other, keptItems, at, both, { key: keptKey }));
      }),
    );
  });

  it("does not depend on the order of the persons", () => {
    fc.assert(
      fc.property(arbitraryData, ({ rows }) => {
        const ids = rows.map((_, k) => `person${k + 1}`);
        const matrix = buildResponseMatrix(rows, { personIds: ids });
        const reversed = buildResponseMatrix(rows.slice().reverse(), { personIds: ids.slice().reverse() });
        const all = selectPairs(ids, { room: ids.map(() => "r") }).pairs;
        expect(kIndex(reversed, all, { key })).toEqual(kIndex(matrix, all, { key }));
      }),
    );
  });

  it("selects the pairs of the persons who share a value, each once", () => {
    fc.assert(
      fc.property(
        fc.array(fc.tuple(fc.option(fc.constantFrom("a", "b", "c")), fc.option(fc.constantFrom("x", "y"))), {
          minLength: 1,
          maxLength: 15,
        }),
        (values) => {
          const ids = values.map((_, k) => `p${k}`);
          const rooms = values.map(([room]) => room);
          const networks = values.map(([, network]) => network);
          const { pairs: selected } = selectPairs(ids, { room: rooms, network: networks });
          const expected: string[] = [];
          for (let a = 0; a < ids.length; a++) {
            for (let b = a + 1; b < ids.length; b++) {
              const shares =
                (rooms[a] !== null && rooms[a] === rooms[b]) || (networks[a] !== null && networks[a] === networks[b]);
              if (shares) expected.push(`${ids[a]} ${ids[b]}`);
            }
          }
          expect(selected.map(([a, b]) => `${a} ${b}`)).toEqual(expected);
        },
      ),
    );
  });
});
