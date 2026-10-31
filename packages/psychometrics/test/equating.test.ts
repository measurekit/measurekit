import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  type BivariateDistribution,
  bivariateLogLinearPresmoothing,
  bootstrapEquating,
  chainedEquipercentileEquating,
  chainedLinearEquating,
  createRandom,
  DataError,
  equipercentileEquating,
  fourParameterLogistic,
  frequencyEstimationEquating,
  generalizedPartialCredit,
  gradedResponse,
  haebaraLinking,
  type IrtItem,
  levineEquating,
  linearEquating,
  logLinearPresmoothing,
  meanMeanLinking,
  meanSigmaLinking,
  nominalResponse,
  oneParameterLogistic,
  ratingScale,
  robustZ,
  type ScaleTransformation,
  stockingLordLinking,
  threeParameterLogistic,
  transformItem,
  transformTheta,
  trueScoreEquating,
  tuckerEquating,
  twoParameterLogistic,
} from "../src/index.js";
import bivariatePresmoothing from "./fixtures/equating-bivariate-presmoothing.json" with { type: "json" };
import bootstrap from "./fixtures/equating-bootstrap.json" with { type: "json" };
import linking from "./fixtures/equating-linking.json" with { type: "json" };
import neat from "./fixtures/equating-neat.json" with { type: "json" };
import observed from "./fixtures/equating-observed-score.json" with { type: "json" };
import presmoothing from "./fixtures/equating-presmoothing.json" with { type: "json" };
import robust from "./fixtures/equating-robust-z.json" with { type: "json" };
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
/** Bootstrap standard errors above BOOTSTRAP_FLOOR against equate's, which have a Monte Carlo error of their own. */
const BOOTSTRAP_TOLERANCE = 0.1;
const BOOTSTRAP_FLOOR = 0.05;

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

const SEED = 20261029;

/** Returns the persons in a bivariate distribution. */
function personsOf(distribution: BivariateDistribution): number {
  return distribution.reduce((sum, row) => sum + row.reduce((rowSum, count) => rowSum + count, 0), 0);
}

/** Returns the persons at each total score of a bivariate distribution, and at each anchor score. */
function marginsOf(distribution: BivariateDistribution): [number[], number[]] {
  const columns = (distribution[0] ?? []).map((_, v) => distribution.reduce((sum, row) => sum + (row[v] ?? 0), 0));
  return [distribution.map((row) => row.reduce((sum, count) => sum + count, 0)), columns];
}

describe("linear equating through an anchor", () => {
  for (const fixture of neat.cases) {
    it(`matches equate's Tucker, Levine and chained linear equating of ${fixture.name}`, () => {
      const anchor = fixture.anchor === "external" ? "external" : "internal";
      for (const { method, weight, trueScore: lts, equated } of fixture.linear) {
        const weights = weight === null ? {} : { weight };
        const actual =
          method === "tucker"
            ? tuckerEquating(fixture.from, fixture.to, weights)
            : method === "levine"
              ? levineEquating(fixture.from, fixture.to, { ...weights, trueScore: lts, anchor })
              : chainedLinearEquating(fixture.from, fixture.to);
        expect(actual).toHaveLength(equated.length);
        actual.forEach((score, x) => {
          expectClose(score, defined(equated[x]), SCORE_TOLERANCE, `${method} ${weight} ${lts} ${x}`);
        });
      }
    });

    it(`weighs the groups by their numbers of persons unless told otherwise in ${fixture.name}`, () => {
      const given = tuckerEquating(fixture.from, fixture.to, { weight: fixture.share });
      tuckerEquating(fixture.from, fixture.to).forEach((score, x) => {
        expectClose(score, defined(given[x]), SCORE_TOLERANCE, `score ${x}`);
      });
    });
  }

  it("gives the Levine true-score equating whatever the weight", () => {
    const { from, to } = defined(neat.cases[0]);
    const light = levineEquating(from, to, { trueScore: true, weight: 0.1 });
    levineEquating(from, to, { trueScore: true, weight: 0.9 }).forEach((score, x) => {
      expectClose(score, defined(light[x]), SCORE_TOLERANCE, `score ${x}`);
    });
  });

  it("equates the scores of identical groups to themselves", () => {
    const groups = [
      [8, 3, 1],
      [4, 9, 4],
      [1, 3, 8],
    ];
    for (const equate of [tuckerEquating, levineEquating, chainedLinearEquating]) {
      equate(groups, groups).forEach((score, x) => {
        expectClose(score, x, 1e-12, `score ${x}`);
      });
    }
  });

  it("rejects an anchor and total scores that give no equating", () => {
    const groups = [
      [8, 3, 1],
      [4, 9, 4],
      [1, 3, 8],
    ];
    expect(() =>
      tuckerEquating(
        [
          [2, 0],
          [3, 0],
        ],
        [
          [2, 1],
          [1, 2],
        ],
      ),
    ).toThrow("Equating through an anchor needs total and anchor scores with variance in the from distribution.");
    expect(() =>
      chainedLinearEquating(groups, [
        [0, 0, 0],
        [0, 0, 1],
      ]),
    ).toThrow("Equating through an anchor needs at least two persons in the to distribution.");
    expect(() =>
      levineEquating(
        [
          [1, 4],
          [4, 1],
        ],
        [
          [4, 1],
          [1, 4],
        ],
      ),
    ).toThrow("Levine equating needs total and anchor scores with a positive covariance in the from distribution.");
    expect(
      tuckerEquating(
        [
          [1, 4],
          [4, 1],
        ],
        [
          [4, 1],
          [1, 4],
        ],
      ),
    ).toHaveLength(2);
    const from = [
      [10, 2, 1],
      [3, 8, 4],
      [1, 3, 9],
    ];
    const to = [
      [100, 1, 0],
      [100, 1, 0],
      [100, 1, 1],
    ];
    expect(() => levineEquating(from, to, { weight: 0 })).toThrow(
      "The synthetic population has total scores without variance; use another weight or method.",
    );
    expect(levineEquating(from, to, { weight: 0.5 })).toHaveLength(3);
  });

  it("rejects invalid distributions", () => {
    const groups = [
      [8, 3, 1],
      [4, 9, 4],
    ];
    expect(() => tuckerEquating(JSON.parse("null"), groups)).toThrow(
      "The from distribution must be an array with a row of counts for each total score, found null.",
    );
    expect(() => tuckerEquating(groups, JSON.parse('[[1, 2], "row"]'))).toThrow(
      'Row 2 of the to distribution must be an array of counts for each anchor score, found "row".',
    );
    expect(() =>
      tuckerEquating(groups, [
        [1, 2, 3],
        [4, 5],
      ]),
    ).toThrow(
      "The to distribution needs the same number of anchor scores in every row, found 3 in row 1 and 2 in row 2.",
    );
    expect(() => tuckerEquating(groups, [[1, 2, 3]])).toThrow(
      "The to distribution needs at least the total scores 0 and 1 and the anchor scores 0 and 1, found 1 and 3 scores.",
    );
    expect(() => tuckerEquating(groups, [[1], [2]])).toThrow(DataError);
    expect(() =>
      tuckerEquating(
        [
          [1, -1],
          [2, 2],
        ],
        groups,
      ),
    ).toThrow(
      "The from distribution needs a non-negative count at each pair of scores, found -1 at the total score 0 and the anchor score 1.",
    );
    expect(() =>
      tuckerEquating(
        [
          [1, Number.NaN],
          [2, 2],
        ],
        groups,
      ),
    ).toThrow(DataError);
    expect(() => tuckerEquating(JSON.parse('[[1, "2"], [2, 2]]'), groups)).toThrow(DataError);
    expect(() =>
      tuckerEquating(
        [
          [0, 0],
          [0, 0],
        ],
        groups,
      ),
    ).toThrow("The from distribution needs at least one person.");
    expect(() =>
      tuckerEquating(groups, [
        [1, 2],
        [3, 4],
      ]),
    ).toThrow("The from and to distributions need the same anchor scores, found 3 and 2.");
  });

  it("rejects invalid options", () => {
    const groups = [
      [8, 3, 1],
      [4, 9, 4],
    ];
    expect(() => tuckerEquating(groups, groups, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => tuckerEquating(groups, groups, { weight: JSON.parse('"1"') })).toThrow(
      'The option weight must be a number from 0 to 1, found "1".',
    );
    expect(() => tuckerEquating(groups, groups, { weight: JSON.parse("null") })).toThrow(DataError);
    expect(() => levineEquating(groups, groups, { weight: 1.5 })).toThrow(DataError);
    expect(() => levineEquating(groups, groups, { anchor: JSON.parse('"side"') })).toThrow(
      'The option anchor must be "internal" or "external", found "side".',
    );
    expect(() => levineEquating(groups, groups, { trueScore: JSON.parse('"yes"') })).toThrow(
      'The option trueScore must be a boolean, found "yes".',
    );
    expect(() => levineEquating(groups, groups, JSON.parse("[]"))).toThrow(DataError);
  });
});

describe("equipercentile equating through an anchor", () => {
  for (const fixture of neat.cases) {
    it(`matches the definition and equate's chained and frequency estimation equating of ${fixture.name}`, () => {
      for (const { method, weight, equate, exact } of fixture.equipercentile) {
        const actual =
          method === "chained"
            ? chainedEquipercentileEquating(fixture.from, fixture.to)
            : frequencyEstimationEquating(fixture.from, fixture.to, { weight: defined(weight) });
        expect(actual).toHaveLength(exact.length);
        actual.forEach((score, x) => {
          expectClose(score, defined(exact[x]), SCORE_TOLERANCE, `${method} ${weight} ${x}`);
          expectClose(score, defined(equate[x]), ROUNDED_TOLERANCE, `equate ${method} ${weight} ${x}`);
        });
      }
    });
  }

  it("weighs the groups by their numbers of persons in frequency estimation unless told otherwise", () => {
    for (const fixture of neat.cases) {
      const given = frequencyEstimationEquating(fixture.from, fixture.to, { weight: fixture.share });
      frequencyEstimationEquating(fixture.from, fixture.to).forEach((score, x) => {
        expectClose(score, defined(given[x]), SCORE_TOLERANCE, `${fixture.name} ${x}`);
      });
    }
  });

  it("maps the scores of a form without persons below and above its scale to the ends of the scale", () => {
    const from = [
      [0, 0],
      [3, 1],
      [1, 3],
      [0, 0],
    ];
    const to = [
      [2, 2],
      [1, 1],
      [2, 3],
    ];
    const chained = chainedEquipercentileEquating(from, to);
    const estimated = frequencyEstimationEquating(from, to);
    for (const equated of [chained, estimated]) {
      expect(equated[0]).toBe(-0.5);
      expect(equated[3]).toBe(2.5);
    }
  });

  it("estimates frequencies where no group has persons at an anchor score", () => {
    const from = [
      [3, 0, 1],
      [2, 0, 4],
      [1, 0, 3],
    ];
    const to = [
      [4, 0, 1],
      [2, 0, 3],
      [1, 0, 5],
    ];
    expect(frequencyEstimationEquating(from, to).every(Number.isFinite)).toBe(true);
    expect(chainedEquipercentileEquating(from, to).every(Number.isFinite)).toBe(true);
  });

  it("rejects an anchor score that persons in only one group have, which presmoothing resolves", () => {
    const from = [
      [3, 1, 1],
      [2, 2, 4],
      [1, 3, 3],
    ];
    const to = [
      [4, 0, 1],
      [2, 0, 3],
      [1, 0, 5],
    ];
    expect(() => frequencyEstimationEquating(from, to)).toThrow(
      "Frequency estimation needs persons with the anchor score 1 in both distributions or neither, but the to distribution has none. Smooth the distributions with bivariateLogLinearPresmoothing.",
    );
    expect(() => frequencyEstimationEquating(to, from)).toThrow(
      "Frequency estimation needs persons with the anchor score 1 in both distributions or neither, but the from distribution has none.",
    );
    const smoothed = bivariateLogLinearPresmoothing(to, { degrees: [2, 2] }).distribution;
    expect(frequencyEstimationEquating(from, smoothed).every(Number.isFinite)).toBe(true);
  });

  it("rejects invalid input", () => {
    const groups = [
      [8, 3, 1],
      [4, 9, 4],
    ];
    expect(() => chainedEquipercentileEquating(JSON.parse("{}"), groups)).toThrow(DataError);
    expect(() =>
      chainedEquipercentileEquating(groups, [
        [1, 2],
        [3, 4],
      ]),
    ).toThrow(DataError);
    expect(() => frequencyEstimationEquating(groups, groups, JSON.parse("null"))).toThrow(DataError);
    expect(() => frequencyEstimationEquating(groups, groups, { weight: -0.1 })).toThrow(
      "The option weight must be a number from 0 to 1, found -0.1.",
    );
  });
});

describe("bivariateLogLinearPresmoothing", () => {
  for (const fixture of bivariatePresmoothing.cases) {
    it(`matches a Poisson glm on ${fixture.name}`, () => {
      for (const { degrees, crossDegree, distribution } of fixture.fits) {
        const [total = 0, anchor = 0] = degrees;
        const result = bivariateLogLinearPresmoothing(fixture.distribution, {
          degrees: [total, anchor],
          crossDegree,
        });
        expect(result.converged).toBe(true);
        result.distribution.forEach((row, x) => {
          row.forEach((count, v) => {
            const expected = defined(distribution[x]?.[v]);
            expectClose(
              count,
              expected,
              PRESMOOTHING_TOLERANCE * Math.max(1, expected),
              `degrees ${degrees}, cross ${crossDegree}, cell ${x} ${v}`,
            );
          });
        });
      }
    });
  }

  it("keeps the number of persons, the moments of the scores and the cross moments", () => {
    const counts = fc.array(fc.array(fc.integer({ min: 1, max: 30 }), { minLength: 5, maxLength: 5 }), {
      minLength: 6,
      maxLength: 6,
    });
    fc.assert(
      fc.property(counts, fc.integer({ min: 0, max: 2 }), (distribution, crossDegree) => {
        const { distribution: fitted } = bivariateLogLinearPresmoothing(distribution, {
          degrees: [3, 2],
          crossDegree,
        });
        const moment = (cells: BivariateDistribution, i: number, j: number): number =>
          cells.reduce(
            (sum, row, x) => sum + row.reduce((rowSum, count, v) => rowSum + count * (x / 6) ** i * (v / 5) ** j, 0),
            0,
          );
        const powers: [number, number][] = [[0, 0]];
        for (let i = 1; i <= 3; i++) powers.push([i, 0]);
        for (let j = 1; j <= 2; j++) powers.push([0, j]);
        for (let i = 1; i <= crossDegree; i++) for (let j = 1; j <= crossDegree; j++) powers.push([i, j]);
        for (const [i, j] of powers) {
          expectClose(moment(fitted, i, j), moment(distribution, i, j), 1e-7 * moment(distribution, i, j), `${i} ${j}`);
        }
      }),
      { numRuns: 30 },
    );
  });

  it("gives every cell a positive count, including those without persons", () => {
    const { distribution } = bivariateLogLinearPresmoothing(
      [
        [4, 1, 0, 0],
        [2, 5, 1, 0],
        [0, 2, 6, 2],
        [0, 0, 3, 7],
      ],
      { degrees: [2, 2] },
    );
    expect(distribution.every((row) => row.every((count) => count > 0))).toBe(true);
  });

  it("rejects degrees outside their ranges", () => {
    const groups = [
      [8, 3, 1],
      [4, 9, 4],
      [1, 3, 8],
      [2, 2, 2],
    ];
    expect(() => bivariateLogLinearPresmoothing(groups, { degrees: [4, 2] })).toThrow(
      "The degree of the total scores must be an integer from 1 to 3, one less than the number of scores, found 4.",
    );
    expect(() => bivariateLogLinearPresmoothing(groups, { degrees: [2, 3] })).toThrow(
      "The degree of the anchor scores must be an integer from 1 to 2, one less than the number of scores, found 3.",
    );
    expect(() => bivariateLogLinearPresmoothing(groups, { degrees: [2, 2], crossDegree: 3 })).toThrow(
      "The option crossDegree must be an integer from 0 to 2, the smaller of the degrees, found 3.",
    );
    expect(() => bivariateLogLinearPresmoothing(groups, { degrees: [2, 2], crossDegree: JSON.parse('"1"') })).toThrow(
      DataError,
    );
    expect(() => bivariateLogLinearPresmoothing(groups, { degrees: JSON.parse("[2]") })).toThrow(
      "The option degrees must have the degrees of the total and the anchor scores, found 1.",
    );
    expect(() => bivariateLogLinearPresmoothing(groups, { degrees: JSON.parse('"22"') })).toThrow(DataError);
    expect(() => bivariateLogLinearPresmoothing(groups, { degrees: JSON.parse('[2, "2"]') })).toThrow(DataError);
    expect(() => bivariateLogLinearPresmoothing(groups, JSON.parse("null"))).toThrow(DataError);
    expect(() => bivariateLogLinearPresmoothing(JSON.parse("null"), { degrees: [1, 1] })).toThrow(DataError);
  });
});

describe("bootstrapEquating", () => {
  const smooth = (distribution: BivariateDistribution): BivariateDistribution =>
    bivariateLogLinearPresmoothing(distribution, { degrees: [4, 3] }).distribution;

  /** Expects standard errors within BOOTSTRAP_TOLERANCE of equate's where equate's are above BOOTSTRAP_FLOOR. */
  function expectErrors(actual: readonly number[], expected: readonly number[], label: string): void {
    expect(actual, label).toHaveLength(expected.length);
    expected.forEach((reference, x) => {
      if (reference > BOOTSTRAP_FLOOR) {
        expectClose(defined(actual[x]), reference, BOOTSTRAP_TOLERANCE * reference, `${label} ${x}`);
      }
    });
  }

  const options = { replications: bootstrap.replications };

  it("matches the standard errors of equate's bootstrap of equating through an anchor", () => {
    const sample = { design: "neat" as const, from: bootstrap.neat.from, to: bootstrap.neat.to };
    const { errors } = bootstrap.neat;
    const methods = {
      tucker: (s: typeof sample) => tuckerEquating(s.from, s.to),
      levine: (s: typeof sample) => levineEquating(s.from, s.to),
      chainedLinear: (s: typeof sample) => chainedLinearEquating(s.from, s.to),
      chainedEquipercentile: (s: typeof sample) => chainedEquipercentileEquating(s.from, s.to),
    };
    for (const [name, equate] of Object.entries(methods)) {
      const reference = defined(errors[name as keyof typeof errors]);
      expectErrors(bootstrapEquating(sample, equate, createRandom(SEED), options), reference, name);
    }
  });

  it("matches the standard errors of equate's bootstrap of frequency estimation, which smooths each resample", () => {
    const sample = { design: "neat" as const, from: bootstrap.neat.from, to: bootstrap.neat.to };
    // Smoothing every resample takes seconds, so a quarter of equate's replications: the Monte Carlo error of the
    // standard errors rises from 1.6% to 3.2%, within the tolerance.
    const standardErrors = bootstrapEquating(
      sample,
      (s) => frequencyEstimationEquating(smooth(s.from), smooth(s.to)),
      createRandom(SEED),
      { replications: bootstrap.replications / 4 },
    );
    expectErrors(standardErrors, bootstrap.neat.errors.frequencyEstimation, "frequency estimation");
  });

  it("matches the standard errors of equate's bootstrap of random groups", () => {
    const sample = {
      design: "random-groups" as const,
      from: bootstrap.randomGroups.from,
      to: bootstrap.randomGroups.to,
    };
    const { errors } = bootstrap.randomGroups;
    expectErrors(
      bootstrapEquating(sample, (s) => linearEquating(s.from, s.to), createRandom(SEED), options),
      errors.linear,
      "linear",
    );
    expectErrors(
      bootstrapEquating(sample, (s) => equipercentileEquating(s.from, s.to), createRandom(SEED), options),
      errors.equipercentile,
      "equipercentile",
    );
  });

  it("matches the standard errors of equate's bootstrap of a single group, which resamples pairs of scores", () => {
    const sample = { design: "single-group" as const, joint: bootstrap.singleGroup.joint };
    const { errors } = bootstrap.singleGroup;
    const margins = (s: typeof sample): [number[], number[]] => marginsOf(s.joint);
    expectErrors(
      bootstrapEquating(sample, (s) => linearEquating(...margins(s)), createRandom(SEED), options),
      errors.linear,
      "linear",
    );
    expectErrors(
      bootstrapEquating(sample, (s) => equipercentileEquating(...margins(s)), createRandom(SEED), options),
      errors.equipercentile,
      "equipercentile",
    );
  });

  it("resamples as many persons as each group has, with whole numbers in every cell", () => {
    const sample = { design: "neat" as const, from: bootstrap.neat.from, to: bootstrap.neat.to };
    const totals: [number, number][] = [];
    bootstrapEquating(
      sample,
      (s) => {
        totals.push([personsOf(s.from), personsOf(s.to)]);
        expect(s.from.flat().every(Number.isInteger)).toBe(true);
        return tuckerEquating(s.from, s.to);
      },
      createRandom(SEED),
      { replications: 5 },
    );
    expect(totals).toHaveLength(6);
    for (const total of totals) expect(total).toEqual([personsOf(sample.from), personsOf(sample.to)]);
  });

  it("gives the same standard errors for the same seed and other ones for another", () => {
    const sample = {
      design: "random-groups" as const,
      from: bootstrap.randomGroups.from,
      to: bootstrap.randomGroups.to,
    };
    const equate = (s: typeof sample): number[] => linearEquating(s.from, s.to);
    const first = bootstrapEquating(sample, equate, createRandom(1), { replications: 50 });
    expect(bootstrapEquating(sample, equate, createRandom(1), { replications: 50 })).toEqual(first);
    expect(bootstrapEquating(sample, equate, createRandom(2), { replications: 50 })).not.toEqual(first);
  });

  it("halves the standard errors when the numbers of persons quadruple", () => {
    const sample = {
      design: "random-groups" as const,
      from: bootstrap.randomGroups.from,
      to: bootstrap.randomGroups.to,
    };
    const larger = {
      design: "random-groups" as const,
      from: sample.from.map((n) => 4 * n),
      to: sample.to.map((n) => 4 * n),
    };
    const equate = (s: typeof sample): number[] => linearEquating(s.from, s.to);
    const small = bootstrapEquating(sample, equate, createRandom(SEED), { replications: 1000 });
    const large = bootstrapEquating(larger, equate, createRandom(SEED), { replications: 1000 });
    const rms = (errors: readonly number[]): number =>
      Math.sqrt(errors.reduce((sum, error) => sum + error ** 2, 0) / errors.length);
    expectClose(rms(large) / rms(small), 0.5, 0.03);
  });

  it("approaches the normal-theory standard error of linear equating from random groups", () => {
    // Discretized normal distributions of 20,000 persons each, to which the formula of Kolen and Brennan applies:
    // sd(Y) sqrt((1 + z² / 2)(1 / N_X + 1 / N_Y)) at the standardized score z of x.
    const persons = 20_000;
    const normal = (mean: number, sd: number): number[] =>
      Array.from({ length: 101 }, (_, x) =>
        Math.round((persons * Math.exp(-0.5 * ((x - mean) / sd) ** 2)) / (sd * Math.sqrt(2 * Math.PI))),
      );
    const sample = { design: "random-groups" as const, from: normal(48, 10), to: normal(52, 12) };
    const moments = (counts: readonly number[]): [number, number, number] => {
      const n = counts.reduce((sum, count) => sum + count, 0);
      const mean = counts.reduce((sum, count, x) => sum + count * x, 0) / n;
      const variance = counts.reduce((sum, count, x) => sum + count * (x - mean) ** 2, 0) / n;
      return [n, mean, Math.sqrt(variance)];
    };
    const [fromPersons, fromMean, fromSd] = moments(sample.from);
    const [toPersons, , toSd] = moments(sample.to);
    const errors = bootstrapEquating(sample, (s) => linearEquating(s.from, s.to), createRandom(SEED), {
      replications: 1000,
    });
    for (const x of [30, 48, 70]) {
      const z = (x - fromMean) / fromSd;
      const formula = toSd * Math.sqrt((1 + z ** 2 / 2) * (1 / fromPersons + 1 / toPersons));
      expectClose(defined(errors[x]), formula, 0.1 * formula, `score ${x}`);
    }
  });

  it("stops with the error of an equating that throws on a resample", () => {
    const groups = [
      [40, 1],
      [40, 0],
    ];
    const sample = { design: "neat" as const, from: groups, to: groups };
    expect(() =>
      bootstrapEquating(sample, (s) => frequencyEstimationEquating(s.from, s.to), createRandom(SEED), {
        replications: 200,
      }),
    ).toThrow("Frequency estimation needs persons with the anchor score");
  });

  it("rejects invalid arguments", () => {
    const sample = { design: "random-groups" as const, from: [1, 2, 3], to: [3, 2, 1] };
    const equate = (s: typeof sample): number[] => linearEquating(s.from, s.to);
    const random = createRandom(SEED);
    expect(() => bootstrapEquating(sample, equate, random, { replications: 1 })).toThrow(
      "The option replications must be an integer from 2, found 1.",
    );
    expect(() => bootstrapEquating(sample, equate, random, { replications: 2.5 })).toThrow(
      "The option replications must be a positive integer, found 2.5.",
    );
    expect(() => bootstrapEquating(sample, equate, random, { replications: JSON.parse('"5"') })).toThrow(DataError);
    expect(() => bootstrapEquating(sample, equate, random, JSON.parse("null"))).toThrow(DataError);
    expect(() => bootstrapEquating(sample, JSON.parse("null"), random)).toThrow(
      "The equate argument must be a function of a sample, found null.",
    );
    expect(() => bootstrapEquating(sample, equate, JSON.parse("{}"))).toThrow(DataError);
    expect(() => bootstrapEquating(JSON.parse("null"), equate, random)).toThrow(
      "The sample must be an object with a design, found null.",
    );
    expect(() => bootstrapEquating(JSON.parse('{"design": "sg"}'), equate, random)).toThrow(
      'The design of the sample must be "random-groups", "single-group" or "neat", found "sg".',
    );
    expect(() => bootstrapEquating({ ...sample, from: [1, 2.5, 3] }, equate, random)).toThrow(
      "The from distribution needs whole numbers of persons to resample, found 2.5 in cell 1.",
    );
    expect(() => bootstrapEquating({ ...sample, to: [0, 0, 0] }, equate, random)).toThrow(DataError);
    expect(() =>
      bootstrapEquating(
        {
          design: "single-group" as const,
          joint: [
            [1, 0.5],
            [2, 2],
          ],
        },
        () => [0],
        random,
      ),
    ).toThrow("The joint distribution needs whole numbers of persons to resample, found 0.5 in cell 1.");
    expect(() => bootstrapEquating(sample, () => JSON.parse('"scores"'), random)).toThrow(
      'The equate function must return an array of equated scores, found "scores".',
    );
    expect(() => bootstrapEquating(sample, () => [1, Number.NaN], random)).toThrow(
      "The equate function must return finite equated scores, found NaN at 1.",
    );
    let calls = 0;
    expect(() => bootstrapEquating(sample, () => (calls++ === 0 ? [1, 2] : [1, 2, 3]), random)).toThrow(
      "The equate function must return 2 equated scores every time, found 3.",
    );
  });
});

describe("robustZ", () => {
  for (const fixture of robust.cases) {
    it(`matches the robust z of base R for ${fixture.name} items`, () => {
      const drift = robustZ(fixtureItems(fixture.from), fixtureItems(fixture.to));
      expect(drift).toHaveLength(fixture.z.length);
      drift.forEach(({ difference, z, flagged }, j) => {
        expectClose(difference, defined(fixture.differences[j]), 1e-12, `difference ${j}`);
        expectClose(z, defined(fixture.z[j]), 1e-12, `z ${j}`);
        expect(flagged, `flagged ${j}`).toBe(fixture.flagged[j]);
      });
    });
  }

  it("flags the items that drifted among items that moved by a shift", () => {
    const rasch = defined(robust.cases[0]);
    const flagged = robustZ(fixtureItems(rasch.from), fixtureItems(rasch.to)).flatMap(({ flagged: drifted }, j) =>
      drifted ? [j + 1] : [],
    );
    expect(flagged).toEqual([4, 11]);
  });

  it("takes a polytomous item's location as the mean of its thresholds", () => {
    const item = (b: readonly number[]): IrtItem => ({ model: generalizedPartialCredit, parameters: { a: 1, b } });
    const from = [item([-1, 1]), item([0, 1]), item([-0.5, 0.5, 1.5]), item([1])];
    const to = [item([-0.5, 1.5]), item([0.5, 1.5]), item([0.5, 1.5, 2.5]), item([1.5])];
    // Every location moved by 0.5 except the last item's, which moved by 1.
    expect(robustZ(from, to).map(({ difference }) => difference)).toEqual([0.5, 0.5, 1, 0.5]);
  });

  it("rejects common items that do not pair up, NRM items and differences without spread", () => {
    const item = (b: number): IrtItem => ({ model: twoParameterLogistic, parameters: { a: 1, b, c: 0, d: 1 } });
    const from = [item(0), item(1), item(2)];
    expect(() => robustZ(from, [item(0)])).toThrow(DataError);
    expect(() => robustZ([], [])).toThrow(DataError);
    expect(() => robustZ(JSON.parse("null"), from)).toThrow(DataError);
    expect(() => robustZ(from, from)).toThrow(
      "Robust z needs common items whose differences have an interquartile range above 0.",
    );
    expect(() => robustZ([item(0)], [item(1)])).toThrow(DataError);
    const nominal: IrtItem = { model: nominalResponse, parameters: { a: [1, 2], c: [0, 1] } };
    expect(() => robustZ([nominal, nominal], [nominal, nominal])).toThrow(DataError);
    expect(() => robustZ(from, [item(0), { ...item(1), model: oneParameterLogistic }, item(2)])).toThrow(DataError);
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

  /** Groups of 40 to 400 persons over 4 total scores and 3 anchor scores, with every cell occupied. */
  const group = fc.array(fc.array(fc.integer({ min: 1, max: 100 }), { minLength: 3, maxLength: 3 }), {
    minLength: 4,
    maxLength: 4,
  });

  it("equating through an anchor by the linear methods is an increasing line", () => {
    fc.assert(
      fc.property(group, group, (from, to) => {
        for (const equated of [tuckerEquating(from, to), chainedLinearEquating(from, to)]) {
          const slope = defined(equated[1]) - defined(equated[0]);
          expect(slope).toBeGreaterThan(0);
          equated.slice(1).forEach((score, x) => {
            expectClose(score - defined(equated[x]), slope, 1e-9 * Math.max(1, slope), `step ${x}`);
          });
        }
      }),
    );
  });

  it("equating through an anchor by the equipercentile methods is monotone and stays within the scale", () => {
    fc.assert(
      fc.property(group, group, (from, to) => {
        for (const equated of [chainedEquipercentileEquating(from, to), frequencyEstimationEquating(from, to)]) {
          equated.forEach((score, x) => {
            expect(score).toBeGreaterThanOrEqual(-0.5);
            expect(score).toBeLessThanOrEqual(3.5);
            expect(score).toBeGreaterThanOrEqual(equated[x - 1] ?? Number.NEGATIVE_INFINITY);
          });
        }
      }),
    );
  });

  it("equating through an anchor by the equipercentile methods depends on the relative frequencies only", () => {
    const doubled = (counts: number[][]): number[][] => counts.map((row) => row.map((count) => 2 * count));
    fc.assert(
      fc.property(group, group, (from, to) => {
        for (const equate of [chainedEquipercentileEquating, frequencyEstimationEquating]) {
          const equated = equate(from, to);
          equate(doubled(from), doubled(to)).forEach((score, x) => {
            expectClose(score, defined(equated[x]), 1e-9, `score ${x}`);
          });
        }
      }),
      { numRuns: 30 },
    );
  });

  it("robust z does not change when every location on one scale moves by a constant, or with the order of the items", () => {
    // Locations in hundredths, so that differences are either equal or at least 0.01 apart.
    const location = fc.integer({ min: -300, max: 300 }).map((hundredths) => hundredths / 100);
    const item = (b: number): IrtItem => ({ model: twoParameterLogistic, parameters: { a: 1, b, c: 0, d: 1 } });
    fc.assert(
      fc.property(fc.array(fc.tuple(location, location), { minLength: 4, maxLength: 12 }), location, (pairs, shift) => {
        const from = pairs.map(([b]) => item(b));
        const to = pairs.map(([, b]) => item(b));
        const spread = new Set(pairs.map(([before, after]) => Math.round((after - before) * 100)));
        fc.pre(spread.size >= pairs.length - 1 && spread.size > 3);
        const z = robustZ(from, to).map((entry) => entry.z);
        const moved = robustZ(
          from,
          pairs.map(([, b]) => item(b + shift)),
        ).map((entry) => entry.z);
        z.forEach((value, j) => {
          expectClose(defined(moved[j]), value, 1e-9, `item ${j}`);
        });
        const reversed = robustZ([...from].reverse(), [...to].reverse()).map((entry) => entry.z);
        z.forEach((value, j) => {
          expectClose(defined(reversed[z.length - 1 - j]), value, 1e-12, `item ${j}`);
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
