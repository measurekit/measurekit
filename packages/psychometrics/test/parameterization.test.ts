import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  DataError,
  fourParameterLogistic,
  fromMirtIrt,
  fromMirtMultidimensional,
  fromMirtSlopeIntercept,
  generalizedGradedUnfolding,
  generalizedPartialCredit,
  gradedResponse,
  type IrtItem,
  type IrtModel,
  type LogisticItem,
  type MultidimensionalItem,
  type MultidimensionalModel,
  multidimensionalGradedResponse,
  multidimensionalThreeParameterLogistic,
  multidimensionalTwoParameterLogistic,
  nominalResponse,
  oneParameterLogistic,
  threeParameterLogistic,
  toMirtIrt,
  toMirtMultidimensional,
  toMirtSlopeIntercept,
  twoParameterLogistic,
} from "../src/index.js";
import { arbitraryItem } from "./arbitraries.js";
import { defined, expectClose, expectParameters } from "./assertions.js";
import { MODELS } from "./fixture-items.js";
import mirt from "./fixtures/parameterization-mirt.json" with { type: "json" };
import multidimensional from "./fixtures/parameterization-multidimensional.json" with { type: "json" };

/** Conversions against mirt's coefficients: a few multiplications, divisions and sums. */
const TOLERANCE = 1e-15;
/** Category probabilities against mirt's probtrace. */
const PROBABILITY_TOLERANCE = 1e-15;

/** A relative tolerance: `TOLERANCE` times the magnitude of the value, at least 1. */
const relative = (value: number): number => Math.max(1, Math.abs(value));

/** Expects parameters or coefficients with the names of `expected`, in order, within the relative tolerance. */
function expectRelative(actual: unknown, expected: unknown, label: string): void {
  expectParameters(actual, expected, TOLERANCE, label, relative);
}

describe("mirt", () => {
  mirt.items.forEach((row, i) => {
    const itemModel = defined(MODELS[row.model]);
    it(`converts mirt's ${row.itemtype} coefficients of item ${i + 1} to item parameters and back`, () => {
      const item = fromMirtIrt(row.irt, itemModel);
      expect(item.model).toBe(itemModel);
      expectRelative(item.parameters, row.parameters, "from the IRT form");
      expectRelative(fromMirtSlopeIntercept(row.slopeIntercept, itemModel).parameters, row.parameters, "from SI");
      mirt.theta.forEach((theta, j) => {
        const probabilities = new Float64Array(itemModel.categoryCount(item.parameters));
        itemModel.probabilities(item.parameters, theta, probabilities);
        probabilities.forEach((p, k) => {
          expectClose(p, defined(row.probabilities[j]?.[k]), PROBABILITY_TOLERANCE, `P(${k}) at ${theta}`);
        });
      });
      if (row.model === "rsm") {
        // mirt's rsm form is not unique per item: the conversions write the steps as b1 to b(K-1) and c = -a b.
        for (const to of [toMirtIrt, toMirtSlopeIntercept]) {
          expectRelative(fromMirtIrt(to(item), itemModel).parameters, row.parameters, "round trip");
        }
      } else {
        expectRelative(toMirtIrt(item), row.irt, "to the IRT form");
        expectRelative(toMirtSlopeIntercept(item), row.slopeIntercept, "to the slope-intercept form");
      }
    });
  });

  it("names the asymptotes c and d, which mirt names g and u", () => {
    expect(fromMirtIrt({ a: 1.1, b: 0.4, g: 0.2, u: 0.9 }, fourParameterLogistic).parameters).toEqual({
      a: 1.1,
      b: 0.4,
      c: 0.2,
      d: 0.9,
    });
    const item = { model: threeParameterLogistic, parameters: { a: 2, b: 0.5, c: 0.1, d: 1 } };
    expect(toMirtSlopeIntercept(item)).toEqual({ a1: 2, d: -1, g: 0.1, u: 1 });
  });

  it("rejects names a form does not use, missing names, a zero slope and invalid parameters", () => {
    expect(() => fromMirtIrt({ a: 1, b: 0, g: 0, u: 1, d1: 2 }, twoParameterLogistic)).toThrow(
      "The mirt coefficients of a 2PL item have names its IRT form does not use: d1.",
    );
    expect(() => fromMirtIrt({ a: 1, b1: 0 }, twoParameterLogistic)).toThrow(
      "The mirt coefficients of a 2PL item need a finite b, found undefined.",
    );
    expect(() => fromMirtSlopeIntercept({ a1: 0, d: 1, g: 0, u: 1 }, twoParameterLogistic)).toThrow(
      "An item with slope a1 = 0 has no IRT parameters, which divide by a1.",
    );
    expect(() => fromMirtIrt({ a: 1, b: 0, g: 1.2, u: 1 }, fourParameterLogistic)).toThrow(DataError);
    expect(() => fromMirtIrt({ a: 1, b1: 1, b2: 0 }, gradedResponse)).toThrow(DataError);
    expect(() => fromMirtSlopeIntercept({ a1: 1, ak0: 0, ak1: 2, d0: 0, d1: 1 }, generalizedPartialCredit)).toThrow(
      "The (generalized) partial credit model has mirt's scores ak0, ak1, ... equal to 0, 1, ... and d0 = 0, one d per score, found ak = [0, 2], d = [0, 1].",
    );
    const flat = { model: nominalResponse, parameters: { a: [1, 0, 1], c: [0, 1, 0] } };
    expect(() => toMirtSlopeIntercept(flat)).toThrow(
      "An NRM item whose first and last categories have equal slopes has no slope-intercept form in mirt, which fixes ak0 = 0 and ak(K-1) = K - 1.",
    );
    expect(() => toMirtIrt({ model: twoParameterLogistic, parameters: { a: Number.NaN, b: 0, c: 0, d: 1 } })).toThrow(
      DataError,
    );
    expect(() => toMirtIrt(JSON.parse("null"))).toThrow(
      "The item must be an item with a model and parameters, found null.",
    );
    expect(() => toMirtSlopeIntercept({ model: twoParameterLogistic, parameters: JSON.parse("null") })).toThrow(
      "The item: Logistic item parameters need finite a and b",
    );
    expect(() => fromMirtSlopeIntercept({ a1: 1, d: 0, g: 0, u: 1 }, JSON.parse("{}"))).toThrow(
      "The model must be one of the built-in models, such as twoParameterLogistic, found an object.",
    );
    expect(() => fromMirtIrt(JSON.parse("null"), twoParameterLogistic)).toThrow(
      "The mirt coefficients must be an object of numbers by name, found null.",
    );
    expect(() => fromMirtIrt({ a: 1, b: 0, g: 0, u: 1 }, JSON.parse("null"))).toThrow(
      "The model must be one of the built-in models, such as twoParameterLogistic, found null.",
    );
    expect(() => fromMirtIrt(JSON.parse('{ "a": "1", "b": 0, "g": 0, "u": 1 }'), twoParameterLogistic)).toThrow(
      'The mirt coefficients of a 2PL item need a finite a, found "1".',
    );
  });

  it("converts only the built-in models", () => {
    const custom: IrtModel<LogisticItem> = { ...oneParameterLogistic, name: "Rasch" };
    expect(() => toMirtIrt({ model: custom, parameters: { a: 1, b: 0, c: 0, d: 1 } })).toThrow(
      `Only the built-in models other than the GGUM convert to and from mirt, found the "Rasch" model.`,
    );
    expect(() => toMirtIrt({ model: generalizedGradedUnfolding, parameters: { a: 1, b: 0, t: [-1] } })).toThrow(
      `Only the built-in models other than the GGUM convert to and from mirt, found the "GGUM" model.`,
    );
    expect(() => fromMirtIrt({ a: 1, b: 0, g: 0, u: 1 }, custom)).toThrow(DataError);
  });

  it("round-trips the parameters of every built-in model through both forms", () => {
    // The GGUM does not convert; the slope-intercept forms divide by the slope, and mirt fixes the nominal model's first
    // and last slopes apart.
    const convertible = arbitraryItem.filter(({ model, parameters }) => {
      if (model === generalizedGradedUnfolding) return false;
      const vector = model.toVector(parameters);
      if (model !== nominalResponse) return Math.abs(vector[0] ?? 0) > 1e-3;
      // The vector holds every slope but the last, which is minus their sum.
      const slopes = vector.subarray(0, vector.length / 2);
      return Math.abs((slopes[0] ?? 0) + slopes.reduce((sum, value) => sum + value, 0)) > 1e-3;
    });
    fc.assert(
      fc.property(convertible, (item: IrtItem) => {
        const canonical = item.model.fromVector(item.model.toVector(item.parameters));
        expectParameters(fromMirtIrt(toMirtIrt(item), item.model).parameters, canonical, 1e-11, "IRT", relative);
        const back = fromMirtSlopeIntercept(toMirtSlopeIntercept(item), item.model);
        expectParameters(back.parameters, canonical, 1e-11, "slope-intercept", relative);
      }),
    );
  });
});

describe("mirt, multidimensional", () => {
  const ModelsByName = {
    m2pl: multidimensionalTwoParameterLogistic,
    m3pl: multidimensionalThreeParameterLogistic,
    mgrm: multidimensionalGradedResponse,
  } as const;

  /** An item's parameters as the fixture gives them, the logistic and graded names whichever the model has. */
  interface FixtureParameters {
    readonly a: readonly number[];
    readonly intercept?: number;
    readonly intercepts?: readonly number[];
    readonly c?: number;
    readonly d?: number;
  }

  multidimensional.items.forEach((row, i) => {
    const model: MultidimensionalModel = defined(ModelsByName[row.model as keyof typeof ModelsByName]);
    it(`converts mirt's ${row.itemtype} coefficients of item ${i + 1} on ${row.dimensions} dimensions to item parameters and back`, () => {
      const item = fromMirtMultidimensional(row.slopeIntercept, model);
      expect(item.model).toBe(model);
      const parameters: FixtureParameters = row.parameters;
      expectParameters(item.parameters, parameters, TOLERANCE, "from mirt", relative);
      expectParameters(toMirtMultidimensional(item), row.slopeIntercept, TOLERANCE, "to mirt", relative);
      row.theta.forEach((theta, j) => {
        const probabilities = new Float64Array(model.categoryCount(item.parameters));
        model.probabilities(item.parameters, Float64Array.from(theta), probabilities);
        probabilities.forEach((p, k) => {
          expectClose(p, defined(row.probabilities[j]?.[k]), PROBABILITY_TOLERANCE, `P(${k}) at ${theta.join(", ")}`);
        });
      });
    });
  });

  it("names the asymptotes c and d, which mirt names g and u, and the intercept d", () => {
    const coefficients = { a1: 1.1, a2: 0.4, d: 0.7, g: 0.2, u: 1 };
    const item = fromMirtMultidimensional(coefficients, multidimensionalThreeParameterLogistic);
    expect(item.parameters).toEqual({ a: [1.1, 0.4], intercept: 0.7, c: 0.2, d: 1 });
    expect(
      toMirtMultidimensional({
        model: multidimensionalTwoParameterLogistic,
        parameters: { a: [2], intercept: -1, c: 0, d: 1 },
      }),
    ).toEqual({
      a1: 2,
      d: -1,
      g: 0,
      u: 1,
    });
    expect(
      toMirtMultidimensional({
        model: multidimensionalGradedResponse,
        parameters: { a: [1, 2, 3], intercepts: [1, 0] },
      }),
    ).toEqual({
      a1: 1,
      a2: 2,
      a3: 3,
      d1: 1,
      d2: 0,
    });
  });

  it("rejects names a form does not use, missing names and invalid parameters", () => {
    expect(() =>
      fromMirtMultidimensional({ a1: 1, d: 0, g: 0, u: 1, d1: 2 }, multidimensionalTwoParameterLogistic),
    ).toThrow("The mirt coefficients of a multidimensional M2PL item have names its form does not use: d1.");
    expect(() => fromMirtMultidimensional({ d: 0, g: 0, u: 1 }, multidimensionalTwoParameterLogistic)).toThrow(
      "The mirt coefficients of a multidimensional M2PL item need a finite a1, found undefined.",
    );
    expect(() => fromMirtMultidimensional({ a1: 1, a2: 1 }, multidimensionalGradedResponse)).toThrow(
      "The mirt coefficients of a multidimensional MGRM item need a finite d1, found undefined.",
    );
    expect(() => fromMirtMultidimensional({ a1: 1, d1: 0, d2: 1 }, multidimensionalGradedResponse)).toThrow(DataError);
    expect(() => fromMirtMultidimensional({ a1: 1, d: 0, g: 0.3, u: 1 }, multidimensionalTwoParameterLogistic)).toThrow(
      "The multidimensional M2PL item of the mirt coefficients: An M2PL item has c = 0, found c = 0.3.",
    );
    expect(() =>
      fromMirtMultidimensional(
        JSON.parse('{ "a1": "1", "d": 0, "g": 0, "u": 1 }'),
        multidimensionalTwoParameterLogistic,
      ),
    ).toThrow('need a finite a1, found "1".');
  });

  it("converts only the built-in multidimensional models, and rejects input of other types", () => {
    const custom = { ...multidimensionalTwoParameterLogistic, name: "Custom" };
    expect(() => toMirtMultidimensional({ model: custom, parameters: { a: [1], intercept: 0, c: 0, d: 1 } })).toThrow(
      'Only the built-in models convert to and from mirt, found the "Custom" model.',
    );
    expect(() => fromMirtMultidimensional({ a1: 1, d: 0, g: 0, u: 1 }, custom)).toThrow(DataError);
    expect(() => toMirtMultidimensional(JSON.parse("null"))).toThrow(
      "The item must be an item with a model and parameters, found null.",
    );
    expect(() =>
      toMirtMultidimensional({ model: multidimensionalGradedResponse, parameters: JSON.parse("null") }),
    ).toThrow("The item: An MGRM item needs finite slopes a");
    expect(() => fromMirtMultidimensional(JSON.parse("null"), multidimensionalTwoParameterLogistic)).toThrow(
      "The mirt coefficients must be an object of numbers by name, found null.",
    );
    expect(() => fromMirtMultidimensional({ a1: 1, d: 0, g: 0, u: 1 }, JSON.parse("null"))).toThrow(
      "The model must be one of the built-in multidimensional models, such as multidimensionalTwoParameterLogistic, found null.",
    );
  });

  it("round-trips the parameters of every built-in multidimensional model", () => {
    const slope = fc.double({ min: -3, max: 3, noNaN: true });
    const location = fc.double({ min: -4, max: 4, noNaN: true });
    const items: fc.Arbitrary<MultidimensionalItem> = fc.oneof(
      fc
        .record({
          a: fc.array(slope, { minLength: 1, maxLength: 4 }),
          intercept: location,
          c: fc.double({ min: 0, max: 0.4, noNaN: true }),
        })
        .map(({ a, intercept, c }) => ({
          model: multidimensionalThreeParameterLogistic,
          parameters: { a, intercept, c, d: 1 },
        })),
      fc
        .record({
          a: fc.array(slope, { minLength: 1, maxLength: 4 }),
          first: location,
          gaps: fc.array(fc.double({ min: 0.05, max: 2, noNaN: true }), { minLength: 1, maxLength: 4 }),
        })
        .map(({ a, first, gaps }) => {
          const intercepts = [first];
          for (const gap of gaps) intercepts.push((intercepts.at(-1) ?? 0) - gap);
          return { model: multidimensionalGradedResponse, parameters: { a, intercepts } };
        }),
    );
    fc.assert(
      fc.property(items, (item) => {
        const back = fromMirtMultidimensional(toMirtMultidimensional(item), item.model);
        expect(back.parameters).toEqual(item.parameters);
      }),
    );
  });
});
