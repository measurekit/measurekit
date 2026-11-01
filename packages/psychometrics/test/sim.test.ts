import { cholesky, mean } from "@measurekit/core";
import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  calibrate,
  calibrateMultidimensional,
  createRandom,
  DataError,
  type Distribution,
  fourParameterLogistic,
  generalizedPartialCredit,
  gradedResponse,
  type IrtItem,
  type IrtModel,
  type LogisticItem,
  logNormalDistribution,
  type MultidimensionalItem,
  type MultidimensionalLogisticParameters,
  multidimensionalGradedResponse,
  multidimensionalThreeParameterLogistic,
  multidimensionalTwoParameterLogistic,
  type NominalParameters,
  nominalResponse,
  normalDistribution,
  oneParameterLogistic,
  partialCredit,
  type RatingScaleParameters,
  ratingScale,
  simulateCdmResponses,
  simulateItems,
  simulateMultidimensionalResponses,
  simulateResponses,
  simulateThetas,
  type ThresholdItem,
  threeParameterLogistic,
  twoParameterLogistic,
  uniformDistribution,
} from "../src/index.js";
import { defined } from "./assertions.js";
import { MODELS } from "./fixture-items.js";

// Statistical checks use fixed seeds and bounds of four standard errors, which a correct simulation exceeds with a
// probability below 1e-4 per check: a failure is a defect, not chance.

/** Large samples for the moments of distributions. */
const DRAWS = 100_000;

/** catR's default slope of polytomous items, logN(0, 0.1225²): its mean and standard deviation. */
const LOG_NORMAL_MEAN = Math.exp(0.1225 ** 2 / 2);
const LOG_NORMAL_SD = LOG_NORMAL_MEAN * Math.sqrt(Math.expm1(0.1225 ** 2));

/**
 * Expects the sample mean within four standard errors of `expectedMean`, and the sample standard deviation within four
 * standard errors of `expectedSd`, the latter's standard error √((m4 - s⁴) / n) / (2s) from the fourth central moment.
 */
function expectMoments(values: readonly number[], expectedMean: number, expectedSd: number, label: string): void {
  const n = values.length;
  const center = mean(values);
  const squares = values.map((value) => (value - center) ** 2);
  const sd = Math.sqrt((mean(squares) * n) / (n - 1));
  const fourth = mean(squares.map((square) => square * square));
  expect(Math.abs(center - expectedMean), `${label} mean`).toBeLessThan((4 * expectedSd) / Math.sqrt(n));
  expect(Math.abs(sd - expectedSd), `${label} sd`).toBeLessThan((4 * Math.sqrt((fourth - sd ** 4) / n)) / (2 * sd));
}

/** Expects the share of `count` in `n` within four binomial standard errors of `p`. */
function expectShare(count: number, n: number, p: number, label: string): void {
  expect(Math.abs(count / n - p), label).toBeLessThanOrEqual(4 * Math.sqrt((p * (1 - p)) / n));
}

function sample(distribution: Distribution, seed: number): number[] {
  const random = createRandom(seed);
  return Array.from({ length: DRAWS }, () => distribution(random));
}

describe("simulateCdmResponses", () => {
  const model = {
    attributes: ["a", "b"],
    items: [
      { id: "q1", attributes: ["a"], probabilities: [0.2, 0.9] },
      { id: "q2", attributes: ["a", "b"], probabilities: [0.1, 0.3, 0.6, 0.95] },
    ],
  };

  it("answers each item correctly with its latent group's probability, reproducibly", () => {
    const count = 20_000;
    const profiles = Array.from({ length: count }, () => [0, 1]);
    const matrix = simulateCdmResponses(model, profiles, createRandom(5));
    const shares = [0, 1].map((j) => mean(matrix.responses.map((row) => defined(row[j]))));
    // Profile 01 is in group 0 of q1 and group 01, the third, of q2.
    for (const [share, p] of [
      [defined(shares[0]), 0.2],
      [defined(shares[1]), 0.6],
    ] as const) {
      expect(Math.abs(share - p)).toBeLessThan(4 * Math.sqrt((p * (1 - p)) / count));
    }
    expect(simulateCdmResponses(model, profiles.slice(0, 50), createRandom(9))).toEqual(
      simulateCdmResponses(model, profiles.slice(0, 50), createRandom(9)),
    );
    expect(matrix.itemIds).toEqual(["q1", "q2"]);
  });

  it("rejects profiles and items that do not fit the attributes", () => {
    const random = createRandom(1);
    expect(() => simulateCdmResponses(model, [[1]], random)).toThrow(
      "Profile 1 needs a 0 or 1 for each of the 2 attributes.",
    );
    const wrong = { ...model, items: [{ id: "q1", attributes: ["c"], probabilities: [0.2, 0.9] }] };
    expect(() => simulateCdmResponses(wrong, [[1, 0]], random)).toThrow(
      'Item "q1" has attribute "c", which is not an attribute.',
    );
    const short = { ...model, items: [{ id: "q1", attributes: ["a", "b"], probabilities: [0.2, 0.9] }] };
    expect(() => simulateCdmResponses(short, [[1, 0]], random)).toThrow(
      'Item "q1" needs 4 probabilities from 0 to 1, one per latent group.',
    );
    const item = { id: "q1", attributes: ["a"], probabilities: [0.2, 0.9] };
    expect(() => simulateCdmResponses({ attributes: ["a", "a"], items: [item] }, [[1, 0]], random)).toThrow(
      'The attribute id "a" appears more than once.',
    );
    const none = { ...model, items: [{ ...item, attributes: [], probabilities: [0.5] }] };
    expect(() => simulateCdmResponses(none, [[1, 0]], random)).toThrow('Item "q1" needs at least one attribute.');
    const twice = { ...model, items: [{ ...item, attributes: ["a", "a"], probabilities: [0.2, 0.4, 0.6, 0.9] }] };
    expect(() => simulateCdmResponses(twice, [[1, 0]], random)).toThrow('Item "q1" lists an attribute twice.');
    expect(() => simulateCdmResponses({ ...model, items: [item, item] }, [[1, 0]], random)).toThrow(
      'The item id "q1" appears more than once.',
    );
    expect(() => simulateCdmResponses(model, [], random)).toThrow(
      "A response matrix needs at least one person and one item.",
    );
  });

  it("rejects models, profiles and random number generators of other types from JavaScript", () => {
    const random = createRandom(1);
    expect(() => simulateCdmResponses(JSON.parse("null"), [[1, 0]], random)).toThrow(
      "A cognitive diagnosis model must be an object { attributes, items }, found null.",
    );
    expect(() => simulateCdmResponses({ ...model, items: JSON.parse("[null]") }, [[1, 0]], random)).toThrow(
      "Item 1 must be an object { id, attributes, probabilities }, found null.",
    );
    expect(() => simulateCdmResponses(model, JSON.parse("[null]"), random)).toThrow(
      "Profile 1 needs a 0 or 1 for each of the 2 attributes.",
    );
    expect(() => simulateCdmResponses(model, [[1, 0]], JSON.parse("null"))).toThrow(
      "A random number generator must be one from createRandom, found null.",
    );
  });
});

describe("distributions", () => {
  it("draw with the mean and standard deviation of the distribution", () => {
    expectMoments(sample(normalDistribution(0.5, 2), 1), 0.5, 2, "normal");
    const meanLog = -0.2;
    const sdLog = 0.5;
    const logMean = Math.exp(meanLog + sdLog ** 2 / 2);
    expectMoments(
      sample(logNormalDistribution(meanLog, sdLog), 2),
      logMean,
      logMean * Math.sqrt(Math.expm1(sdLog ** 2)),
      "log-normal",
    );
    const uniform = sample(uniformDistribution(-1, 3), 3);
    expectMoments(uniform, 1, 4 / Math.sqrt(12), "uniform");
    expect(uniform.every((value) => value >= -1 && value <= 3)).toBe(true);
  });

  it("reject invalid parameters", () => {
    expect(() => normalDistribution(0, 0)).toThrow(
      "The sd of a normal distribution must be a positive number, found 0.",
    );
    expect(() => normalDistribution(Number.NaN, 1)).toThrow(DataError);
    expect(() => logNormalDistribution(0, Number.POSITIVE_INFINITY)).toThrow(
      "The sdLog of a log-normal distribution must be a positive number, found Infinity.",
    );
    expect(() => uniformDistribution(1, 1)).toThrow(
      "A uniform distribution needs the lower bound below the upper, found [1, 1].",
    );
    expect(() => normalDistribution(0, JSON.parse("null"))).toThrow(
      "The sd of a normal distribution must be a positive number, found null.",
    );
    expect(() => logNormalDistribution(JSON.parse("[]"), 1)).toThrow(
      "The meanLog of a log-normal distribution must be a finite number, found an array.",
    );
    expect(() => uniformDistribution(JSON.parse('"0"'), 1)).toThrow(
      'The lower bound of a uniform distribution must be a finite number, found "0".',
    );
  });
});

describe("simulateItems", () => {
  const items = <TParameters>(model: IrtModel<TParameters>, seed: number, categoryCount?: number): TParameters[] =>
    simulateItems(model, 20_000, createRandom(seed), { categoryCount }).map((item) => item.parameters);

  it("draws logistic items from catR's genDichoMatrix defaults", () => {
    const parameters: LogisticItem[] = items(fourParameterLogistic, 4);
    for (const [name, expectedMean, expectedSd] of [
      ["a", 1, 0.2],
      ["b", 0, 1],
      ["c", 0.125, 0.25 / Math.sqrt(12)],
      ["d", 0.875, 0.25 / Math.sqrt(12)],
    ] as const) {
      const values = parameters.map((item) => item[name]);
      expectMoments(values, expectedMean, expectedSd, name);
    }
    expect(items(twoParameterLogistic, 5).every(({ c, d }) => c === 0 && d === 1)).toBe(true);
    expect(items(threeParameterLogistic, 6).every(({ c, d }) => c < 0.25 && d === 1)).toBe(true);
  });

  it("draws GRM and GPCM items from catR's genPolyMatrix defaults, the GRM's thresholds increasing", () => {
    const graded: ThresholdItem[] = items(gradedResponse, 7, 4);
    const partial: ThresholdItem[] = items(generalizedPartialCredit, 8, 4);
    for (const [label, parameters] of [
      ["GRM", graded],
      ["GPCM", partial],
    ] as const) {
      const slopes = parameters.map(({ a }) => a);
      expectMoments(slopes, LOG_NORMAL_MEAN, LOG_NORMAL_SD, `${label} a`);
      expect(parameters.every(({ b }) => b.length === 3)).toBe(true);
    }
    expect(graded.every(({ b }) => b.every((threshold, k) => k === 0 || threshold > defined(b[k - 1])))).toBe(true);
    // Each GPCM threshold is one N(0, 1) draw; the GRM's are the order statistics of three, the middle one with
    // variance 1 - √3 / π.
    const second = (parameters: readonly ThresholdItem[]): number[] => parameters.map(({ b }) => defined(b[1]));
    expectMoments(second(partial), 0, 1, "GPCM b2");
    expectMoments(second(graded), 0, Math.sqrt(1 - Math.sqrt(3) / Math.PI), "GRM b2");
  });

  it("draws the NRM's slopes and intercepts of categories from 1, category 0's at 0, and centers them", () => {
    const parameters: NominalParameters[] = items(nominalResponse, 9);
    for (const { a, c } of parameters) {
      expect(Math.abs(a.reduce((sum, value) => sum + value, 0))).toBeLessThan(1e-12);
      expect(Math.abs(c.reduce((sum, value) => sum + value, 0))).toBeLessThan(1e-12);
    }
    // Category k's slope and intercept less category 0's are the draws.
    const slopes = parameters.map(({ a }) => defined(a[2]) - defined(a[0]));
    const intercepts = parameters.map(({ c }) => defined(c[1]) - defined(c[0]));
    expectMoments(slopes, LOG_NORMAL_MEAN, LOG_NORMAL_SD, "a");
    expectMoments(intercepts, 0, 1, "c");
  });

  it("draws parameters the model's items share once, with a = 1 by default", () => {
    const random = createRandom(10);
    for (const model of [oneParameterLogistic, partialCredit]) {
      expect(new Set(simulateItems<{ a: number }>(model, 50, random).map(({ parameters }) => parameters.a))).toEqual(
        new Set([1]),
      );
    }
    const given = simulateItems(oneParameterLogistic, 50, random, { distributions: { a: normalDistribution(1, 0.2) } });
    expect(new Set(given.map(({ parameters }) => parameters.a)).size).toBe(1);
    const rsm = simulateItems(ratingScale, 50, random, { categoryCount: 5 });
    expect(new Set(rsm.map(({ parameters }) => JSON.stringify([parameters.a, parameters.steps]))).size).toBe(1);
    expect(new Set(rsm.map(({ parameters }) => parameters.b)).size).toBe(50);
  });

  it("keeps the RSM's thresholds as drawn, moving the mean of the steps into the locations", () => {
    const parameters: RatingScaleParameters[] = simulateItems(ratingScale, 3, createRandom(11), {
      categoryCount: 4,
    }).map((item) => item.parameters);
    // a = 1 takes no draw; the three steps come first, then each item's location.
    const random = createRandom(11);
    const steps = [random.normal(), random.normal(), random.normal()];
    parameters.forEach(({ b, steps: centered }) => {
      const location = random.normal();
      expect(Math.abs(centered.reduce((sum, step) => sum + step, 0))).toBeLessThan(1e-12);
      centered.forEach((step, k) => {
        expect(Math.abs(b + step - (location + defined(steps[k])))).toBeLessThan(1e-12);
      });
    });
  });

  it("takes given distributions in place of the defaults", () => {
    const random = createRandom(12);
    const logistic = simulateItems(threeParameterLogistic, 100, random, {
      distributions: { a: () => 1.5, c: uniformDistribution(0.1, 0.2) },
    });
    expect(logistic.every(({ parameters: { a, c } }) => a === 1.5 && c >= 0.1 && c <= 0.2)).toBe(true);
    const decreasing = simulateItems(gradedResponse, 100, random, { distributions: { a: () => -1 } });
    expect(decreasing.every(({ parameters: { b } }) => defined(b[0]) > defined(b[1]))).toBe(true);
  });

  it("rejects other models, counts, categories, parameter names and invalid draws", () => {
    const random = createRandom(13);
    const custom: IrtModel = { ...twoParameterLogistic, name: "custom" };
    expect(() => simulateItems(custom, 1, random)).toThrow(
      `Simulation supports the built-in models only, found the "custom" model.`,
    );
    expect(() => simulateItems(JSON.parse("null"), 1, random)).toThrow(
      "The model must be one of the built-in models, such as twoParameterLogistic, found null.",
    );
    expect(() => simulateItems(twoParameterLogistic, 0, random)).toThrow(
      "The number of items must be a positive integer, found 0.",
    );
    expect(() => simulateItems(twoParameterLogistic, 1, random, { categoryCount: 3 })).toThrow(
      "2PL items have 2 categories, found 3.",
    );
    expect(() => simulateItems(gradedResponse, 1, random, { categoryCount: 1 })).toThrow(
      "The option categoryCount must be an integer from 2, found 1.",
    );
    expect(() => simulateItems(twoParameterLogistic, 1, random, { distributions: { c: () => 0.2 } })).toThrow(
      'The 2PL model draws a, b; found a distribution for "c".',
    );
    expect(() => simulateItems(twoParameterLogistic, 1, random, { distributions: { a: JSON.parse("1") } })).toThrow(
      "The distribution of a must be a function of a random number generator, found 1.",
    );
    expect(() => simulateItems(threeParameterLogistic, 2, random, { distributions: { c: () => 1 } })).toThrow(
      "Simulated item 1: Logistic item parameters need finite a and b and 0 ≤ c < d ≤ 1, found a = ",
    );
    expect(() => simulateItems(twoParameterLogistic, 1, JSON.parse("null"))).toThrow(
      "A random number generator must be one from createRandom, found null.",
    );
    expect(() => simulateItems(twoParameterLogistic, 1, random, { distributions: JSON.parse("null") })).toThrow(
      "The option distributions must be an object of distributions by parameter name, found null.",
    );
  });
});

describe("simulateResponses", () => {
  const Items: readonly IrtItem[] = [
    { model: threeParameterLogistic, parameters: { a: 1.4, b: 0.2, c: 0.2, d: 1 } },
    { model: gradedResponse, parameters: { a: 1.1, b: [-1, 0.3, 1.2] } },
    { model: generalizedPartialCredit, parameters: { a: 0.8, b: [0.5, -0.6, 1.5] } },
    { model: ratingScale, parameters: { a: 1.2, b: 0.1, steps: [-0.8, 0.8] } },
    { model: nominalResponse, parameters: { a: [-1, 0, 1], c: [0.5, 0.2, -0.7] } },
  ];

  it("draws each category with its probability at the person's theta", () => {
    const theta = 0.4;
    const persons = 20_000;
    const matrix = simulateResponses(Items, Array<number>(persons).fill(theta), createRandom(14));
    Items.forEach((item, j) => {
      const count = item.model.categoryCount(item.parameters);
      const probabilities = new Float64Array(count);
      item.model.probabilities(item.parameters, theta, probabilities);
      probabilities.forEach((p, k) => {
        const observed = matrix.responses.filter((row) => row[j] === k).length;
        expectShare(observed, persons, p, `item ${j + 1}, category ${k}`);
      });
    });
  });

  it("makes responses missing completely at random, keeping the others", () => {
    const thetas = simulateThetas(400, createRandom(15));
    const complete = simulateResponses(Items, thetas, createRandom(16));
    const missing = simulateResponses(Items, thetas, createRandom(16), { missingProbability: 0.2 });
    const cells = missing.responses.flat();
    expectShare(cells.filter((response) => response === null).length, cells.length, 0.2, "missing");
    missing.responses.forEach((row, i) => {
      row.forEach((response, j) => {
        if (response !== null) expect(response).toBe(complete.responses[i]?.[j]);
      });
    });
    const none = simulateResponses(Items, thetas, createRandom(16), { missingProbability: 1 });
    expect(none.responses.flat().every((response) => response === null)).toBe(true);
  });

  it("returns a matrix with a row per theta, a column per item and the default ids", () => {
    const matrix = simulateResponses(Items.slice(0, 2), [0, 1, -1], createRandom(17));
    expect(matrix.personIds).toEqual(["person1", "person2", "person3"]);
    expect(matrix.itemIds).toEqual(["item1", "item2"]);
  });

  it("rejects missing items and thetas, invalid items, thetas and missing probabilities", () => {
    const random = createRandom(18);
    expect(() => simulateResponses([], [0], random)).toThrow("A simulation needs at least one item.");
    expect(() => simulateResponses(Items, [], random)).toThrow("A simulation needs at least one theta.");
    expect(() => simulateResponses(Items, [0, Number.NaN], random)).toThrow(
      "Theta 2 must be a finite number, found NaN.",
    );
    expect(() => simulateResponses([{ model: gradedResponse, parameters: { a: 1, b: [1, 0] } }], [0], random)).toThrow(
      "Item 1: A GRM item",
    );
    expect(() => simulateResponses(Items, [0], random, { missingProbability: 1.5 })).toThrow(
      "The option missingProbability must be a number from 0 to 1, found 1.5.",
    );
    expect(() => simulateResponses(Items, JSON.parse("null"), random)).toThrow(
      "A simulation needs an array of thetas, found null.",
    );
    expect(() => simulateResponses(Items, [0], JSON.parse("null"))).toThrow(
      "A random number generator must be one from createRandom, found null.",
    );
    expect(() => simulateResponses(Items, [0], random, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
  });
});

describe("simulateMultidimensionalResponses", () => {
  const Items: readonly MultidimensionalItem[] = [
    { model: multidimensionalTwoParameterLogistic, parameters: { a: [1.2, 0.5], intercept: -0.3, c: 0, d: 1 } },
    { model: multidimensionalThreeParameterLogistic, parameters: { a: [0.4, 1.4], intercept: 0.2, c: 0.2, d: 1 } },
    { model: multidimensionalGradedResponse, parameters: { a: [0.9, -0.7], intercepts: [1.2, -0.1, -1.4] } },
  ];

  it("draws each category with its probability at the person's thetas", () => {
    const theta = [0.4, -0.6];
    const persons = 20_000;
    const matrix = simulateMultidimensionalResponses(
      Items,
      Array.from({ length: persons }, () => theta),
      createRandom(31),
    );
    Items.forEach((item, j) => {
      const probabilities = new Float64Array(item.model.categoryCount(item.parameters));
      item.model.probabilities(item.parameters, Float64Array.from(theta), probabilities);
      probabilities.forEach((p, k) => {
        const observed = matrix.responses.filter((row) => row[j] === k).length;
        expectShare(observed, persons, p, `item ${j + 1}, category ${k}`);
      });
    });
  });

  it("draws from items on three dimensions", () => {
    const items: readonly MultidimensionalItem[] = [
      { model: multidimensionalGradedResponse, parameters: { a: [1, 0.3, -0.5], intercepts: [0.6, -0.8] } },
    ];
    const theta = [-0.2, 0.5, 1];
    const persons = 20_000;
    const matrix = simulateMultidimensionalResponses(
      items,
      Array.from({ length: persons }, () => theta),
      createRandom(32),
    );
    const probabilities = new Float64Array(3);
    defined(items[0]).model.probabilities(defined(items[0]).parameters, Float64Array.from(theta), probabilities);
    probabilities.forEach((p, k) => {
      expectShare(matrix.responses.filter((row) => row[0] === k).length, persons, p, `category ${k}`);
    });
  });

  it("makes responses missing completely at random, keeping the others", () => {
    const random = createRandom(33);
    const thetas = Array.from({ length: 400 }, () => [random.normal(), random.normal()]);
    const complete = simulateMultidimensionalResponses(Items, thetas, createRandom(34));
    const missing = simulateMultidimensionalResponses(Items, thetas, createRandom(34), { missingProbability: 0.2 });
    const cells = missing.responses.flat();
    expectShare(cells.filter((response) => response === null).length, cells.length, 0.2, "missing");
    missing.responses.forEach((row, i) => {
      row.forEach((response, j) => {
        if (response !== null) expect(response).toBe(complete.responses[i]?.[j]);
      });
    });
  });

  it("returns a matrix with a row per person, a column per item and the default ids", () => {
    const matrix = simulateMultidimensionalResponses(
      Items.slice(0, 2),
      [
        [0, 1],
        [1, -1],
        [0, 0],
      ],
      createRandom(35),
    );
    expect(matrix.personIds).toEqual(["person1", "person2", "person3"]);
    expect(matrix.itemIds).toEqual(["item1", "item2"]);
  });

  it("rejects missing items and thetas, invalid items, thetas and missing probabilities", () => {
    const random = createRandom(36);
    expect(() => simulateMultidimensionalResponses([], [[0, 0]], random)).toThrow(
      "A simulation needs at least one item.",
    );
    expect(() => simulateMultidimensionalResponses(Items, [], random)).toThrow(
      "A simulation needs at least one person's thetas.",
    );
    expect(() => simulateMultidimensionalResponses(Items, [[0]], random)).toThrow(
      "Person 1 needs 2 thetas for items on 2 dimensions, found 1.",
    );
    expect(() =>
      simulateMultidimensionalResponses(
        Items,
        [
          [0, 0],
          [1, Number.NaN],
        ],
        random,
      ),
    ).toThrow("Theta 2 of person 2 must be a finite number, found NaN.");
    expect(() =>
      simulateMultidimensionalResponses(
        [{ model: multidimensionalGradedResponse, parameters: { a: [1], intercepts: [1, 2] } }],
        [[0]],
        random,
      ),
    ).toThrow("Item 1: An MGRM item needs strictly decreasing intercepts");
    expect(() => simulateMultidimensionalResponses(Items, [[0, 0]], random, { missingProbability: 1.5 })).toThrow(
      "The option missingProbability must be a number from 0 to 1, found 1.5.",
    );
    expect(() => simulateMultidimensionalResponses(Items, JSON.parse("null"), random)).toThrow(
      "A simulation needs an array of thetas by person, found null.",
    );
    expect(() => simulateMultidimensionalResponses(Items, [JSON.parse("null")], random)).toThrow(
      "The thetas of person 1 must be an array, found null.",
    );
    expect(() => simulateMultidimensionalResponses(JSON.parse("null"), [[0, 0]], random)).toThrow(DataError);
    expect(() => simulateMultidimensionalResponses(Items, [[0, 0]], JSON.parse("null"))).toThrow(
      "A random number generator must be one from createRandom, found null.",
    );
    expect(() => simulateMultidimensionalResponses(Items, [[0, 0]], random, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
  });
});

describe("simulateThetas", () => {
  it("draws N(0, 1) by default, or from the distribution given", () => {
    expectMoments(simulateThetas(DRAWS, createRandom(19)), 0, 1, "default");
    expectMoments(simulateThetas(DRAWS, createRandom(20), normalDistribution(-0.5, 1.5)), -0.5, 1.5, "given");
  });

  it("rejects counts and draws that are not finite", () => {
    const random = createRandom(21);
    expect(() => simulateThetas(1.5, random)).toThrow("The number of thetas must be a positive integer, found 1.5.");
    expect(() => simulateThetas(2, random, () => Number.POSITIVE_INFINITY)).toThrow(
      "Theta 1 must be a finite number, found Infinity.",
    );
    expect(() => simulateThetas(2, random, JSON.parse("null"))).toThrow(
      "The distribution must be a function of a random number generator, found null.",
    );
    expect(() => simulateThetas(2, JSON.parse("{}"))).toThrow(
      "A random number generator must be one from createRandom, found an object.",
    );
  });
});

describe("recovery", () => {
  it("recovers the parameters of simulated 2PL items by calibration", () => {
    const random = createRandom(20261029);
    const items = simulateItems(twoParameterLogistic, 20, random);
    const matrix = simulateResponses(items, simulateThetas(2000, random), random);
    const result = calibrate(matrix, { model: twoParameterLogistic, standardErrors: "hessian" });
    expect(result.converged).toBe(true);
    const errors: Record<"a" | "b", number[]> = { a: [], b: [] };
    result.items.forEach((estimate, j) => {
      const truth = defined(items[j]).parameters;
      const standardErrors = defined(estimate.standardErrors);
      for (const name of ["a", "b"] as const) {
        const error = estimate.parameters[name] - truth[name];
        // Every estimate within four of its standard errors of the true value.
        expect(Math.abs(error), `${estimate.id} ${name}`).toBeLessThan(4 * defined(standardErrors[name]));
        errors[name].push(error);
      }
    });
    // Root mean square errors over the items below 0.15 for a and 0.2 for b: under twice the root mean square standard
    // errors of 2,000 persons, 0.08 and 0.12.
    const rootMeanSquare = (values: readonly number[]): number => Math.sqrt(mean(values.map((value) => value * value)));
    expect(rootMeanSquare(errors.a)).toBeLessThan(0.15);
    expect(rootMeanSquare(errors.b)).toBeLessThan(0.2);
  });

  it("recovers the parameters of simulated multidimensional items by calibration", () => {
    // Three correlated dimensions with eight items each, each item loading on one dimension. Thresholds fixed before
    // the first run: every estimate within four standard errors of the true value, and a root mean square error of the
    // slopes below 0.2.
    const random = createRandom(20261029);
    const correlations = [
      [1, 0.3, 0.5],
      [0.3, 1, 0.4],
      [0.5, 0.4, 1],
    ];
    const factor = defined(cholesky(Float64Array.from(correlations.flat()), 3));
    const items: MultidimensionalItem<MultidimensionalLogisticParameters>[] = Array.from({ length: 24 }, (_, i) => ({
      model: multidimensionalTwoParameterLogistic,
      parameters: {
        a: [0, 1, 2].map((d) => (d === i % 3 ? 0.8 + 0.1 * Math.floor(i / 3) : 0)),
        intercept: ((i % 5) - 2) * 0.5,
        c: 0,
        d: 1,
      },
    }));
    const thetas = Array.from({ length: 1500 }, () => {
      const z = [random.normal(), random.normal(), random.normal()];
      return [0, 1, 2].map((d) => [0, 1, 2].reduce((sum, k) => sum + (factor[d * 3 + k] ?? 0) * defined(z[k]), 0));
    });
    const matrix = simulateMultidimensionalResponses(items, thetas, random);
    const loadings = Object.fromEntries(items.map((_, i) => [`item${i + 1}`, [`D${(i % 3) + 1}`]]));
    const result = calibrateMultidimensional(matrix, {
      model: multidimensionalTwoParameterLogistic,
      loadings,
      points: 11,
      standardErrors: "cross-product",
      tolerance: 1e-5,
    });
    expect(result.converged).toBe(true);
    const slopeErrors: number[] = [];
    result.items.forEach((estimate, j) => {
      const truth = defined(items[j]).parameters;
      const standardErrors = defined(estimate.standardErrors);
      const d = j % 3;
      const slopeError = defined(estimate.parameters.a[d]) - defined(truth.a[d]);
      expect(Math.abs(slopeError), `${estimate.id} slope`).toBeLessThan(4 * defined(standardErrors[`a${d + 1}`]));
      expect(Math.abs(estimate.parameters.intercept - truth.intercept), `${estimate.id} intercept`).toBeLessThan(
        4 * defined(standardErrors.intercept),
      );
      slopeErrors.push(slopeError);
    });
    for (const [i, j] of [
      [1, 0],
      [2, 0],
      [2, 1],
    ] as const) {
      const error = defined(result.correlations[i]?.[j]) - defined(correlations[i]?.[j]);
      expect(Math.abs(error), `correlation ${i},${j}`).toBeLessThan(
        4 * defined(result.correlationStandardErrors?.[i]?.[j]),
      );
    }
    expect(Math.sqrt(mean(slopeErrors.map((error) => error * error)))).toBeLessThan(0.2);
  });

  it("recovers the parameters of simulated five-dimensional items by MH-RM", () => {
    // Five dimensions with correlations 0.4 to the power of their distance, six items each, each item loading on one
    // dimension. Thresholds fixed before the first run: every estimate within four standard errors of the true value,
    // and a root mean square error of the slopes below 0.2.
    const random = createRandom(20261029);
    const dimensions = 5;
    const correlations = Array.from({ length: dimensions }, (_, i) =>
      Array.from({ length: dimensions }, (_, j) => 0.4 ** Math.abs(i - j)),
    );
    const factor = defined(cholesky(Float64Array.from(correlations.flat()), dimensions));
    const items: MultidimensionalItem<MultidimensionalLogisticParameters>[] = Array.from({ length: 30 }, (_, i) => ({
      model: multidimensionalTwoParameterLogistic,
      parameters: {
        a: Array.from({ length: dimensions }, (_, d) =>
          d === i % dimensions ? 0.8 + 0.1 * Math.floor(i / dimensions) : 0,
        ),
        intercept: ((i % 7) - 3) * 0.4,
        c: 0,
        d: 1,
      },
    }));
    const thetas = Array.from({ length: 2000 }, () => {
      const z = Array.from({ length: dimensions }, () => random.normal());
      return Array.from({ length: dimensions }, (_, d) =>
        z.reduce((sum, value, k) => sum + (factor[d * dimensions + k] ?? 0) * value, 0),
      );
    });
    const matrix = simulateMultidimensionalResponses(items, thetas, random);
    const loadings = Object.fromEntries(items.map((_, i) => [`item${i + 1}`, [`D${(i % dimensions) + 1}`]]));
    const result = calibrateMultidimensional(matrix, {
      model: multidimensionalTwoParameterLogistic,
      loadings,
      method: "mhrm",
      seed: 20261029,
      standardErrors: "hessian",
    });
    expect(result.converged).toBe(true);
    const slopeErrors: number[] = [];
    result.items.forEach((estimate, j) => {
      const truth = defined(items[j]).parameters;
      const standardErrors = defined(estimate.standardErrors);
      const d = j % dimensions;
      const slopeError = defined(estimate.parameters.a[d]) - defined(truth.a[d]);
      expect(Math.abs(slopeError), `${estimate.id} slope`).toBeLessThan(4 * defined(standardErrors[`a${d + 1}`]));
      expect(Math.abs(estimate.parameters.intercept - truth.intercept), `${estimate.id} intercept`).toBeLessThan(
        4 * defined(standardErrors.intercept),
      );
      slopeErrors.push(slopeError);
    });
    for (let i = 1; i < dimensions; i++) {
      for (let j = 0; j < i; j++) {
        const error = defined(result.correlations[i]?.[j]) - defined(correlations[i]?.[j]);
        expect(Math.abs(error), `correlation ${i},${j}`).toBeLessThan(
          4 * defined(result.correlationStandardErrors?.[i]?.[j]),
        );
      }
    }
    expect(Math.sqrt(mean(slopeErrors.map((error) => error * error)))).toBeLessThan(0.2);
  }, 300_000);
});

describe("properties", () => {
  const seeds = fc.integer({ min: 0, max: 2 ** 31 });
  const models = fc.constantFrom(...Object.values(MODELS));

  it("seeded simulations reproduce exactly", () => {
    fc.assert(
      fc.property(seeds, models, fc.double({ min: 0, max: 1, noNaN: true }), (seed, model, missingProbability) => {
        const run = (): unknown => {
          const random = createRandom(seed);
          const items = simulateItems(model, 5, random);
          return simulateResponses(items, simulateThetas(20, random), random, { missingProbability });
        };
        expect(run()).toEqual(run());
      }),
      { numRuns: 50 },
    );
  });

  it("simulated items are valid for their model, and responses are their categories", () => {
    fc.assert(
      fc.property(seeds, models, fc.integer({ min: 2, max: 6 }), (seed, model, categories) => {
        const random = createRandom(seed);
        const categoryCount = model.maxCategories === 2 ? 2 : categories;
        const items = simulateItems(model, 4, random, { categoryCount });
        for (const item of items) {
          expect(() => item.model.toVector(item.parameters)).not.toThrow();
          expect(item.model.categoryCount(item.parameters)).toBe(categoryCount);
        }
        const { responses } = simulateResponses(items, simulateThetas(10, random), random);
        const valid = responses
          .flat()
          .every((response) => response !== null && response >= 0 && response < categoryCount);
        expect(valid).toBe(true);
      }),
      { numRuns: 100 },
    );
  });

  it("seeded multidimensional simulations reproduce exactly", () => {
    fc.assert(
      fc.property(seeds, fc.double({ min: 0, max: 1, noNaN: true }), (seed, missingProbability) => {
        const run = (): unknown => {
          const random = createRandom(seed);
          const thetas = Array.from({ length: 20 }, () => [random.normal(), random.normal()]);
          return simulateMultidimensionalResponses(
            [
              {
                model: multidimensionalThreeParameterLogistic,
                parameters: { a: [1, 0.5], intercept: 0, c: 0.1, d: 1 },
              },
              { model: multidimensionalGradedResponse, parameters: { a: [0.3, 1], intercepts: [1, -1] } },
            ],
            thetas,
            random,
            { missingProbability },
          );
        };
        expect(run()).toEqual(run());
      }),
      { numRuns: 30 },
    );
  });
});
