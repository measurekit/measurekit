import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  DataError,
  equipercentileEquating,
  fourParameterLogistic,
  generalizedPartialCredit,
  gradedResponse,
  haebaraLinking,
  type IrtItem,
  linearEquating,
  logLinearPresmoothing,
  meanMeanLinking,
  meanSigmaLinking,
  nominalResponse,
  oneParameterLogistic,
  ratingScale,
  type ScaleTransformation,
  stockingLordLinking,
  threeParameterLogistic,
  transformItem,
  transformTheta,
  trueScoreEquating,
  twoParameterLogistic,
} from "../src/index.js";
import linking from "./fixtures/equating-linking.json" with { type: "json" };
import observed from "./fixtures/equating-observed-score.json" with { type: "json" };
import presmoothing from "./fixtures/equating-presmoothing.json" with { type: "json" };
import trueScore from "./fixtures/equating-true-score.json" with { type: "json" };
import { arbitraryItem, arbitraryLocation, arbitraryLocations, defined, expectClose, fixtureItems } from "./helpers.js";

// Tolerances against R, explained in docs/validation.md.
/** Moment methods: closed forms. */
const MOMENT_TOLERANCE = 1e-12;
/** Haebara and Stocking-Lord against equateIRT's and base R's nlminb, which stop near 1e-8. */
const CRITERION_TOLERANCE = 1e-7;
/** Equated scores. */
const SCORE_TOLERANCE = 1e-12;
/** Equipercentile equivalents against equate, which rounds percentile ranks to 10 decimals. */
const ROUNDED_TOLERANCE = 1e-6;
/** Presmoothed counts, relative to max(1, count), against a glm iterated to a deviance change of 1e-15. */
const PRESMOOTHING_TOLERANCE = 1e-11;

function expectTransformation(actual: ScaleTransformation, expected: ScaleTransformation, tolerance: number): void {
  expectClose(actual.slope, expected.slope, tolerance, "slope");
  expectClose(actual.intercept, expected.intercept, tolerance, "intercept");
}

describe("characteristic curve linking", () => {
  for (const fixture of linking.cases) {
    const quadrature = fixture.name === "mixed format" ? undefined : { method: "gauss-hermite" as const, points: 30 };
    it(`matches the Haebara and Stocking-Lord transformations of the ${fixture.name} items`, () => {
      const from = fixtureItems(fixture.from);
      const to = fixtureItems(fixture.to);
      const haebara = haebaraLinking(from, to, { quadrature });
      const stockingLord = stockingLordLinking(from, to, { quadrature });
      expect(haebara.converged && stockingLord.converged).toBe(true);
      expectTransformation(haebara, fixture.transformations.haebara, CRITERION_TOLERANCE);
      expectTransformation(stockingLord, fixture.transformations.stockingLord, CRITERION_TOLERANCE);
    });
  }

  it("converges where rounding of the criterion stops the descent short of the gradient tolerance", () => {
    const item = (a: number, b: number, c: number): IrtItem => ({
      model: threeParameterLogistic,
      parameters: { a, b, c, d: 1 },
    });
    const from = [item(1.56, -0.343, 0.146), item(0.858, -0.43, 0.22), item(1.02, -1.26, 0.198)];
    const to = [item(1.42, -0.297, 0.146), item(0.848, 0.337, 0.22), item(1.23, -1.32, 0.198)];
    expect(haebaraLinking(from, to).converged).toBe(true);
    expect(stockingLordLinking(from, to).converged).toBe(true);
  });

  it("links items of a custom model and rejects lists that do not pair up", () => {
    const custom = { ...twoParameterLogistic, name: "custom 2PL" };
    const from = fixtureItems(linking.cases[0]?.from ?? []).map((item) => ({ ...item, model: custom }));
    const to = fixtureItems(linking.cases[0]?.to ?? []).map((item) => ({ ...item, model: custom }));
    expect(haebaraLinking(from, to).converged).toBe(true);
    expect(() => haebaraLinking(from, to.slice(1))).toThrow(DataError);
    expect(() => stockingLordLinking([], [])).toThrow(DataError);
    expect(() => haebaraLinking(from, fixtureItems(linking.cases[0]?.to ?? []))).toThrow(DataError);
    expect(() => haebaraLinking(from, JSON.parse("null"))).toThrow(
      "The common items on the to scale must be an array, found null.",
    );
    expect(() => stockingLordLinking(from, [JSON.parse("null"), ...to.slice(1)])).toThrow(
      "Common item 1 on the to scale must be an item with a model and parameters, found null.",
    );
    expect(() => stockingLordLinking(from, to, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
  });
});

describe("moment linking", () => {
  for (const fixture of linking.cases) {
    it(`matches the mean-mean and mean-sigma transformations of the ${fixture.name} items`, () => {
      const from = fixtureItems(fixture.from);
      const to = fixtureItems(fixture.to);
      expectTransformation(meanMeanLinking(from, to), fixture.transformations.meanMean, MOMENT_TOLERANCE);
      expectTransformation(meanSigmaLinking(from, to), fixture.transformations.meanSigma, MOMENT_TOLERANCE);
    });
  }

  it("gives A = 1 for Rasch items and rejects NRM items and locations without variance", () => {
    const rasch = (b: number): IrtItem => ({ model: oneParameterLogistic, parameters: { a: 1, b, c: 0, d: 1 } });
    expect(meanMeanLinking([rasch(-1), rasch(0.5)], [rasch(-0.6), rasch(0.9)])).toEqual({ slope: 1, intercept: 0.4 });
    const nominal: IrtItem = { model: nominalResponse, parameters: { a: [-1, 0, 1], c: [0, 0.5, -0.5] } };
    expect(() => meanMeanLinking([nominal], [nominal])).toThrow(DataError);
    expect(() => meanSigmaLinking([rasch(0)], [rasch(1)])).toThrow(DataError);
    expect(() => transformItem({ ...nominal, model: { ...nominalResponse } }, { slope: 1, intercept: 0 })).toThrow(
      DataError,
    );
  });
});

describe("transformItem and transformTheta", () => {
  it("move an item and theta together: the item gives A θ + B the probabilities it gave θ, for every model", () => {
    fc.assert(
      fc.property(
        arbitraryItem,
        fc.double({ min: 0.2, max: 5, noNaN: true }),
        fc.double({ min: -3, max: 3, noNaN: true }),
        fc.double({ min: -4, max: 4, noNaN: true }),
        (item, slope, intercept, theta) => {
          const transformation = { slope, intercept };
          const moved = transformItem(item, transformation);
          const count = item.model.categoryCount(item.parameters);
          const before = new Float64Array(count);
          const after = new Float64Array(count);
          item.model.probabilities(item.parameters, theta, before);
          moved.model.probabilities(moved.parameters, transformTheta(theta, transformation), after);
          before.forEach((p, k) => {
            expectClose(after[k] ?? null, p, 1e-12, `category ${k}`);
          });
        },
      ),
    );
  });

  it("keeps an item's other fields and scales its standard errors with their parameters", () => {
    const transformation = { slope: 2, intercept: 0.5 };
    const multipleChoice = {
      id: "q1",
      contentArea: "algebra",
      model: threeParameterLogistic,
      parameters: { a: 1.2, b: 0.3, c: 0.2, d: 1 },
      standardErrors: { a: 0.1, b: 0.2, c: 0.05 },
    };
    const moved = transformItem(multipleChoice, transformation);
    expect(moved).toMatchObject({ id: "q1", contentArea: "algebra", standardErrors: { a: 0.05, b: 0.4, c: 0.05 } });
    const essay = {
      id: "q2",
      model: ratingScale,
      parameters: { a: 1, b: 0.2, steps: [-0.5, 0.5] },
      standardErrors: { a: 0.1, b: 0.3, step1: 0.2 },
    };
    expect(transformItem(essay, transformation).standardErrors).toEqual({ a: 0.05, b: 0.6, step1: 0.4 });
    const nominal = {
      model: nominalResponse,
      parameters: { a: [-1, 0, 1], c: [0.5, 0, -0.5] },
      standardErrors: { a1: 0.2, a2: 0.1, c1: 0.3, c2: 0.2 },
    };
    expect(transformItem(nominal, transformation).standardErrors).toBeNull();
    expect(transformItem(nominal, { slope: 2, intercept: 0 }).standardErrors).toEqual({
      a1: 0.1,
      a2: 0.05,
      c1: 0.3,
      c2: 0.2,
    });
    expect(transformItem({ ...multipleChoice, standardErrors: null }, transformation).standardErrors).toBeNull();
    expect(() => transformItem({ ...multipleChoice, standardErrors: { a: -1 } }, transformation)).toThrow(
      'The standard error of "a" must be a non-negative number, found -1.',
    );
  });

  it("rejects an invalid transformation or theta", () => {
    const item: IrtItem = { model: twoParameterLogistic, parameters: { a: 1, b: 0, c: 0, d: 1 } };
    expect(() => transformItem(item, { slope: 0, intercept: 0 })).toThrow(DataError);
    expect(() => transformTheta(Number.NaN, { slope: 1, intercept: 0 })).toThrow(DataError);
    expect(() => transformTheta(0, { slope: 1, intercept: Number.POSITIVE_INFINITY })).toThrow(DataError);
    expect(() => meanMeanLinking(JSON.parse("null"), [])).toThrow(
      "The common items on the from scale must be an array, found null.",
    );
    expect(() => meanSigmaLinking([], JSON.parse("{}"))).toThrow(
      "The common items on the to scale must be an array, found an object.",
    );
    expect(() => transformTheta(0, JSON.parse("null"))).toThrow(
      "A scale transformation must be an object { slope, intercept }, found null.",
    );
    expect(() => transformItem(JSON.parse("null"), { slope: 1, intercept: 0 })).toThrow(
      "The item must be an item with a model and parameters, found null.",
    );
  });
});

describe("observed-score equating", () => {
  for (const fixture of observed.cases) {
    it(`matches equate's linear and equipercentile equating of ${fixture.name}`, () => {
      linearEquating(fixture.from, fixture.to).forEach((score, x) => {
        expectClose(score, defined(fixture.linear[x]), SCORE_TOLERANCE, `linear ${x}`);
      });
      equipercentileEquating(fixture.from, fixture.to).forEach((score, x) => {
        expectClose(score, defined(fixture.equipercentileUnrounded[x]), SCORE_TOLERANCE, `equipercentile ${x}`);
        expectClose(score, defined(fixture.equipercentile[x]), ROUNDED_TOLERANCE, `equate ${x}`);
      });
    });
  }

  it("maps scores without persons below and above the distribution to the ends of the scale", () => {
    expect(equipercentileEquating([0, 2, 2, 0], [1, 1, 1, 1])).toEqual([-0.5, 0.5, 2.5, 3.5]);
  });

  it("averages the two inverses where the target has scores without persons", () => {
    // Score 1 has rank 0.5, the cumulative share of `to` at 1 and 2: the inverse from above is 2.5, from below 1.5.
    expect(equipercentileEquating([1, 0, 1], [1, 1, 0, 2])).toEqual([0.5, 2, 3]);
  });

  it("rejects invalid distributions", () => {
    expect(() => linearEquating([3], [1, 2])).toThrow(DataError);
    expect(() => linearEquating([0, 1], [1, 2])).toThrow(DataError);
    expect(() => linearEquating([2, 0], [1, 2])).toThrow(DataError);
    expect(() => equipercentileEquating([1, -1], [1, 2])).toThrow(DataError);
    expect(() => equipercentileEquating([0, 0], [1, 2])).toThrow(DataError);
    expect(() => equipercentileEquating([1, Number.NaN], [1, 2])).toThrow(DataError);
    expect(() => linearEquating(JSON.parse("null"), [1, 2])).toThrow(
      "The from score distribution must be an array of counts, found null.",
    );
    expect(() => logLinearPresmoothing([1, 2, 3], JSON.parse('"1"'))).toThrow(
      'A log-linear degree must be an integer from 1 to 2, one less than the number of scores, found "1".',
    );
  });
});

describe("logLinearPresmoothing", () => {
  for (const fixture of presmoothing.cases) {
    it(`matches a Poisson glm and equate's presmoothing on ${fixture.name}`, () => {
      for (const { degree, distribution } of fixture.fits) {
        const result = logLinearPresmoothing(fixture.distribution, degree);
        expect(result.converged).toBe(true);
        result.distribution.forEach((count, x) => {
          const expected = defined(distribution[x]);
          expectClose(count, expected, PRESMOOTHING_TOLERANCE * Math.max(1, expected), `degree ${degree}, score ${x}`);
        });
      }
    });
  }

  it("keeps the number of persons and the first C moments of the scores", () => {
    const counts = fc.array(fc.integer({ min: 0, max: 50 }), { minLength: 5, maxLength: 30 });
    fc.assert(
      fc.property(counts, fc.integer({ min: 1, max: 4 }), (distribution, degree) => {
        fc.pre(distribution.reduce((sum, count) => sum + count, 0) > 0);
        const { distribution: fitted } = logLinearPresmoothing(distribution, degree);
        for (let power = 0; power <= degree; power++) {
          const moment = (values: readonly number[]): number =>
            values.reduce((sum, count, x) => sum + count * (x / distribution.length) ** power, 0);
          expectClose(
            moment(fitted),
            moment(distribution),
            1e-7 * Math.max(1, moment(distribution)),
            `moment ${power}`,
          );
        }
      }),
    );
  });

  it("rejects a degree outside 1 to the number of scores less 1", () => {
    expect(() => logLinearPresmoothing([1, 2, 3], 3)).toThrow(
      "A log-linear degree must be an integer from 1 to 2, one less than the number of scores, found 3.",
    );
    expect(() => logLinearPresmoothing([1, 2, 3], 0.5)).toThrow(DataError);
    expect(() => logLinearPresmoothing([0, 0, 0], 1)).toThrow(DataError);
  });
});

describe("trueScoreEquating", () => {
  for (const fixture of trueScore.cases) {
    it(`matches base R's true-score equating of ${fixture.name} forms`, () => {
      trueScoreEquating(fixtureItems(fixture.from), fixtureItems(fixture.to)).forEach((score, x) => {
        expectClose(score, defined(fixture.equated[x]), SCORE_TOLERANCE, `score ${x}`);
      });
    });
  }

  it("maps scores above the upper asymptotes linearly to the highest score", () => {
    const item = (d: number): IrtItem => ({ model: fourParameterLogistic, parameters: { a: 1, b: 0, c: 0, d } });
    expect(trueScoreEquating([item(0.5), item(0.5)], [item(0.8), item(0.8)])).toEqual([0, 1.6, 2]);
  });

  it("rejects NRM items, slopes that are not positive and empty forms", () => {
    const nominal: IrtItem = { model: nominalResponse, parameters: { a: [-1, 0, 1], c: [0, 0.5, -0.5] } };
    const falling: IrtItem = { model: twoParameterLogistic, parameters: { a: -1, b: 0, c: 0, d: 1 } };
    const rising: IrtItem = { model: twoParameterLogistic, parameters: { a: 1, b: 0, c: 0, d: 1 } };
    expect(() => trueScoreEquating([nominal], [rising])).toThrow(DataError);
    expect(() => trueScoreEquating([rising], [falling])).toThrow(DataError);
    expect(() => trueScoreEquating([], [rising])).toThrow(DataError);
    expect(() => trueScoreEquating([rising], JSON.parse("null"))).toThrow(
      "The items of the to form must be an array, found null.",
    );
  });
});

describe("properties", () => {
  /** Items whose expected score increases with theta: the 3PL, GRM and GPCM with positive slopes. */
  const risingItem: fc.Arbitrary<IrtItem> = fc.oneof(
    fc
      .record({
        a: fc.double({ min: 0.3, max: 2.5, noNaN: true }),
        b: fc.double({ min: -2.5, max: 2.5, noNaN: true }),
        c: fc.double({ min: 0, max: 0.3, noNaN: true }),
      })
      .map(({ a, b, c }) => ({ model: threeParameterLogistic, parameters: { a, b, c, d: 1 } })),
    fc
      .record({
        a: fc.double({ min: 0.3, max: 2.5, noNaN: true }),
        first: arbitraryLocation,
        gaps: fc.array(fc.double({ min: 0.2, max: 1.5, noNaN: true }), { maxLength: 3 }),
      })
      .map(({ a, first, gaps }) => {
        const b = [first];
        for (const gap of gaps) b.push((b.at(-1) ?? 0) + gap);
        return { model: gradedResponse, parameters: { a, b } };
      }),
    fc
      .record({ a: fc.double({ min: 0.3, max: 2.5, noNaN: true }), b: arbitraryLocations })
      .map(({ a, b }) => ({ model: generalizedPartialCredit, parameters: { a, b } })),
  );
  const form = fc.array(risingItem, { minLength: 2, maxLength: 6 });
  const distribution = fc.array(fc.integer({ min: 0, max: 50 }), { minLength: 2, maxLength: 20 });

  it("links a form to itself with slope 1 and intercept 0", () => {
    fc.assert(
      fc.property(form, (items) => {
        expectTransformation(meanMeanLinking(items, items), { slope: 1, intercept: 0 }, 1e-12);
        expectTransformation(haebaraLinking(items, items), { slope: 1, intercept: 0 }, 1e-7);
        expectTransformation(stockingLordLinking(items, items), { slope: 1, intercept: 0 }, 1e-7);
      }),
      { numRuns: 30 },
    );
  });

  it("recovers the transformation that moved the items", () => {
    fc.assert(
      fc.property(
        form,
        fc.double({ min: 0.6, max: 1.6, noNaN: true }),
        fc.double({ min: -1, max: 1, noNaN: true }),
        (items, slope, intercept) => {
          const transformation = { slope, intercept };
          const moved = items.map((item) => transformItem(item, transformation));
          expectTransformation(meanMeanLinking(items, moved), transformation, 1e-12);
          expectTransformation(haebaraLinking(items, moved), transformation, 1e-6);
          expectTransformation(stockingLordLinking(items, moved), transformation, 1e-6);
        },
      ),
      { numRuns: 30 },
    );
  });

  it("observed-score equating is monotone, and the identity between equal distributions without empty scores", () => {
    fc.assert(
      fc.property(distribution, distribution, (from, to) => {
        fc.pre(from.some((count) => count > 0) && to.some((count) => count > 0));
        const equipercentile = equipercentileEquating(from, to);
        equipercentile.slice(1).forEach((score, x) => {
          expect(score).toBeGreaterThanOrEqual(equipercentile[x] ?? Number.NaN);
        });
        const positive = to.map((count) => count + 1);
        equipercentileEquating(positive, positive).forEach((score, x) => {
          expectClose(score, x, 1e-12);
        });
        linearEquating(positive, positive).forEach((score, x) => {
          expectClose(score, x, 1e-12);
        });
      }),
    );
  });

  it("true-score equating is monotone, and the identity from a form to itself", () => {
    fc.assert(
      fc.property(form, form, (from, to) => {
        const equated = trueScoreEquating(from, to);
        equated.slice(1).forEach((score, x) => {
          expect(score).toBeGreaterThanOrEqual(equated[x] ?? Number.NaN);
        });
        trueScoreEquating(from, from).forEach((score, x) => {
          expectClose(score, x, 1e-9);
        });
      }),
      { numRuns: 50 },
    );
  });
});
