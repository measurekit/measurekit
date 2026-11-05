import { describe, expect, it } from "vitest";
import {
  buildResponseMatrix,
  calibrate,
  calibrateMultidimensional,
  createAdaptiveTest,
  decisionStudy,
  expectedRating,
  type FacetedScore,
  generalizabilityStudy,
  itemFeatureRegression,
  linearChange,
  manyFacetRasch,
  multidimensionalTwoParameterLogistic,
  qMatrixIdentifiability,
  responseTimeEffort,
  selectPairs,
  subscoreValue,
  testLength,
  transformItem,
  twoParameterLogistic,
} from "../src/index.js";
import { designOf, structureOf } from "../src/multidimensional/internal.js";
import { defined } from "./assertions.js";

/** The names that reach a member of every object instead of an entry. */
const RESERVED = ["__proto__", "prototype", "constructor"] as const;

/** Returns the record of JSON with the entries, as an own `__proto__` comes only from JSON or a computed key. */
function record<T>(entries: readonly (readonly [string, T])[]): Readonly<Record<string, T>> {
  return Object.fromEntries(entries);
}

/** Returns responses that a two-parameter calibration of four items converges on, without a seed. */
function fourItemRows(): number[][] {
  return Array.from({ length: 200 }, (_, person) =>
    [0, 1, 2, 3].map((item) => ((person * (item + 3) + (person >> 2)) % 5 < (person % 7 > 2 ? 4 : 1) ? 1 : 0)),
  );
}

const STATIC_RATER_SCORES: FacetedScore[] = ["p1", "p2", "p3", "p4"].flatMap((person, p) =>
  ["r1", "r2"].flatMap((rater, r) =>
    ["i1", "i2"].map((item, i) => ({ levels: { person, rater, item }, score: (p + r + i) % 4 })),
  ),
);

/** Returns the scores with the facet `rater` under another name. */
function withRaterNamed(name: string): FacetedScore[] {
  return STATIC_RATER_SCORES.map(({ levels: { rater = "", ...levels }, score }) => ({
    levels: { ...levels, ...record([[name, rater]]) },
    score,
  }));
}

describe("a record whose keys are parameter names", () => {
  const item = { id: "q1", model: twoParameterLogistic, parameters: { a: 1, b: 0, c: 0, d: 1 } };

  for (const key of RESERVED) {
    it(`refuses the key ${key} in an item's standard errors, which plain assignment would drop unseen`, () => {
      const standardErrors = JSON.parse(`{ "a": 0.1, "${key}": 0.2 }`);
      expect(() => transformItem({ ...item, standardErrors }, { slope: 1, intercept: 0 })).toThrow(
        `An item's standard errors must be an object of numbers by parameter name or null, found the reserved key "${key}".`,
      );
    });

    it(`refuses the key ${key} in the priors and the held parameters of a calibration`, () => {
      const matrix = buildResponseMatrix(fourItemRows());
      const prior = { mean: 0, sd: 1 };
      expect(() =>
        calibrate(matrix, twoParameterLogistic, {
          priors: JSON.parse(`{ "${key}": ${JSON.stringify(prior)} }`),
        }),
      ).toThrow(`found the reserved key "${key}".`);
      const fixedParameters = record([[defined(matrix.itemIds[0]), JSON.parse(`{ "${key}": 1 }`)]]);
      expect(() => calibrate(matrix, twoParameterLogistic, { fixedParameters })).toThrow(
        `found the reserved key "${key}".`,
      );
    });
  }
});

describe("a facet named as a member of an object", () => {
  for (const name of RESERVED) {
    it(`needs its size, and takes it, in a decision study of a study that is not balanced, as ${name}`, () => {
      const scores = withRaterNamed(name).slice(1);
      const study = generalizabilityStudy(scores, { object: "person", facets: [name, "item"], unbalanced: true });
      expect(study.balanced).toBe(false);
      expect(() => decisionStudy(study)).toThrow(`the option sampleSizes must give one for "${name}".`);
      const plain = generalizabilityStudy(STATIC_RATER_SCORES.slice(1), {
        object: "person",
        facets: ["rater", "item"],
        unbalanced: true,
      });
      const sampleSizes = record([
        [name, 3],
        ["item", 5],
      ]);
      expect(decisionStudy(study, { sampleSizes }).generalizability).toBe(
        decisionStudy(plain, { sampleSizes: { rater: 3, item: 5 } }).generalizability,
      );
    });

    it(`takes the sizes of a balanced study and a nesting, as ${name}`, () => {
      const study = generalizabilityStudy(withRaterNamed(name), { object: "person", facets: [name, "item"] });
      expect(Object.keys(study.sampleSizes)).toContain(name);
      const plain = generalizabilityStudy(STATIC_RATER_SCORES, { object: "person", facets: ["rater", "item"] });
      expect(decisionStudy(study).generalizability).toBe(decisionStudy(plain).generalizability);
      const nested = record([["item", [name]]]);
      expect(decisionStudy(study, { nested, sampleSizes: record([[name, 2]]) }).components.length).toBeGreaterThan(0);
    });

    it(`averages over its elements in an expected rating, and takes a level of it, as ${name}`, () => {
      const scores = withRaterNamed(name);
      const model = manyFacetRasch(scores, ["item", name]);
      const plain = manyFacetRasch(STATIC_RATER_SCORES, ["item", "rater"]);
      expect(expectedRating(model, 0)).toBe(expectedRating(plain, 0));
      expect(expectedRating(model, 0, { levels: record([[name, "r1"]]) })).toBe(
        expectedRating(plain, 0, { levels: { rater: "r1" } }),
      );
    });
  }
});

describe("a record whose keys are the user's identifiers", () => {
  const ids = ["__proto__", "constructor", "prototype", "item4"];

  it("holds them in the contexts of pairs, by name of the context", () => {
    const contexts = record([
      ["constructor", ["r1", "r1"]],
      ["__proto__", ["n1", "n2"]],
    ]);
    expect(selectPairs(["a", "b"], contexts).pairs).toEqual([["a", "b"]]);
  });

  it("holds them in a Q-matrix", () => {
    const qMatrix = record([
      ["constructor", ["a"]],
      ["__proto__", ["a"]],
      ["prototype", ["a"]],
    ]);
    expect(qMatrixIdentifiability(qMatrix).singleItems).toEqual({ a: true });
  });

  it("holds them in the specific and primary dimensions, and in the loadings, of a design", () => {
    const specific = record(ids.map((id, index) => [id, index < 2 ? "s" : null]));
    const primary = record(ids.map((id) => [id, ["g"]]));
    expect(structureOf(specific, ids, primary).design.dimensionNames).toEqual(["g", "s"]);
    const loadings = record(ids.map((id) => [id, ["g"]]));
    expect(designOf({ loadings }, ids).dimensionNames).toEqual(["g"]);
  });

  it("holds them in the thresholds of response time effort", () => {
    const data = {
      ...buildResponseMatrix(
        [
          [1, 1],
          [0, 1],
        ],
        { itemIds: ["item1", "constructor"] },
      ),
      times: [
        [2, 5],
        [1, 8],
      ],
    };
    const thresholds = JSON.parse('{ "item1": 2, "constructor": 6 }');
    expect(responseTimeEffort(data, thresholds).map(({ rapidGuesses }) => rapidGuesses)).toEqual([1, 1]);
  });

  it("holds them in the item groups of a linear change", () => {
    const occasion = (flip: number) =>
      buildResponseMatrix(
        Array.from({ length: 40 }, (_, person) =>
          [0, 1, 2, 3].map((item) => ((person + item * 3 + flip) % 5 < 2 + (person % 3) ? 1 : 0)),
        ),
        { itemIds: ids },
      );
    const itemGroups = record(ids.map((id, index) => [id, index < 2 ? "constructor" : "__proto__"]));
    const { effects } = linearChange([occasion(0), occasion(1)], { itemGroups });
    expect(effects.map(({ itemGroup }) => itemGroup)).toEqual(["constructor", "__proto__"]);
  });

  it("holds them as section names", () => {
    const matrix = buildResponseMatrix(fourItemRows(), { itemIds: ids });
    const sections = record([
      ["constructor", ["__proto__", "constructor"]],
      ["__proto__", ["prototype", "item4"]],
    ]);
    expect(subscoreValue(matrix, sections).map(({ section }) => section)).toEqual(["constructor", "__proto__"]);
  });

  it("holds them as ids of an item's features", () => {
    const items = ids.map((id, index) => ({
      id,
      model: twoParameterLogistic,
      parameters: { a: 1, b: index / 2, c: 0, d: 1 },
      estimated: ["b"],
      standardErrors: { b: 0.2 },
    }));
    const features = record(ids.map((id, index) => [id, { x: index * index }]));
    expect(itemFeatureRegression(items, features).effects).toHaveLength(2);
  });

  it("holds them as the items that a calibration fixes, and as testlets and covariates", () => {
    const matrix = buildResponseMatrix(fourItemRows(), { itemIds: ids });
    const fixed = record([["constructor", { a: 1, b: 0, c: 0, d: 1 }]]);
    const held = calibrate(matrix, twoParameterLogistic, { fixed, maxIterations: 20 });
    expect(held.items.find(({ id }) => id === "constructor")?.parameters).toMatchObject({ a: 1, b: 0 });
    const testlets = record([
      ["constructor", ["__proto__", "constructor"]],
      ["__proto__", ["prototype", "item4"]],
    ]);
    const quadrature = { method: "rectangular", points: 11 } as const;
    expect(calibrate(matrix, twoParameterLogistic, { testlets, quadrature, maxIterations: 2 }).items).toHaveLength(4);
    const covariates = record([["constructor", matrix.personIds.map((_, person) => person % 2)]]);
    expect(calibrate(matrix, twoParameterLogistic, { covariates, maxIterations: 2 }).regression).toBeDefined();
  });

  it("holds them as the items that a multidimensional calibration fixes", () => {
    const matrix = buildResponseMatrix(fourItemRows(), { itemIds: ids });
    const fixed = record([["constructor", { a: [1], intercept: 0, c: 0, d: 1 }]]);
    const { items } = calibrateMultidimensional(matrix, multidimensionalTwoParameterLogistic, {
      dimensions: 1,
      fixed,
      maxIterations: 2,
    });
    expect(items.find(({ id }) => id === "constructor")?.parameters).toMatchObject({ intercept: 0 });
  });

  it("holds them as the content areas of an adaptive test", () => {
    const model = twoParameterLogistic;
    const pool = ["constructor", "__proto__"].flatMap((area) =>
      [0, 1].map((i) => ({ id: `${area}${i}`, model, parameters: { a: 1, b: i, c: 0, d: 1 }, contentAreas: [area] })),
    );
    const contentTargets = record([
      ["constructor", 1],
      ["__proto__", 1],
    ]);
    expect(createAdaptiveTest(pool, testLength(2), { contentTargets })).toBeDefined();
  });
});
