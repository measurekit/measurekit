import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  createRandom,
  DataError,
  type DecisionStudy,
  type DecisionStudyOptions,
  decisionStudy,
  type FacetedScore,
  type GeneralizabilityDesign,
  type GeneralizabilityStudy,
  generalizabilityStudy,
} from "../src/index.js";
import { defined, expectClose } from "./assertions.js";
import { readFixture } from "./fixtures.js";

interface GeneralizabilityFixture {
  cases: {
    name: string;
    design: { object: string; facets: string[]; nested: { rater?: string[]; item?: string[] } };
    scores: { levels: { person: string; item: string; rater?: string }; score: number }[];
    components: {
      facets: string[];
      nestedIn: string[];
      degreesOfFreedom: number;
      sumOfSquares: number;
      meanSquare: number;
      variance: number;
      reml?: number;
    }[];
    studies: {
      options: { sampleSizes?: { item: number; rater?: number }; fixed?: string[]; nested?: { rater: string[] } };
      universeScoreVariance: number;
      relativeErrorVariance: number;
      absoluteErrorVariance: number;
      generalizability: number;
      dependability: number;
    }[];
  }[];
}
const fixture = readFixture<GeneralizabilityFixture>("generalizability.json");

interface GeneralizabilityUnbalancedFixture {
  cases: {
    name: string;
    design: { object: string; facets: string[]; nested: { rater?: string[]; item?: string[] }; unbalanced: boolean };
    scores: { levels: { person: string; rater?: string; item?: string }; score: number }[];
    components: {
      facets: string[];
      nestedIn: string[];
      degreesOfFreedom: number;
      sumOfSquares: number;
      meanSquare: number;
      variance: number;
    }[];
    studies: {
      options: { sampleSizes: { rater: number; item?: number } };
      universeScoreVariance: number;
      relativeErrorVariance: number;
      absoluteErrorVariance: number;
      generalizability: number;
      dependability: number;
    }[];
  }[];
}
const unbalancedFixture = readFixture<GeneralizabilityUnbalancedFixture>("generalizability-unbalanced.json");

// Tolerances against R, explained in docs/validation.md.
/** Sums of squares, mean squares and variance components against aov and the expected mean squares, relative. */
const ANOVA_TOLERANCE = 1e-10;
/** Variance components against lme4's REML estimates, absolute: the optimizer stops where the likelihood is flat. */
const REML_TOLERANCE = 1e-5;
/** Variances and coefficients of decision studies against the textbook formulas, relative. */
const DECISION_TOLERANCE = 1e-10;
/** Method 1 against base R's computation of it: both solve the same linear system. */
const UNBALANCED_TOLERANCE = 1e-8;
/** The unbalanced estimates of balanced data against the balanced analysis. */
const BALANCED_TOLERANCE = 1e-10;

interface ReferenceComponent {
  readonly facets: string[];
  readonly nestedIn: string[];
  readonly degreesOfFreedom: number;
  readonly sumOfSquares: number;
  readonly meanSquare: number;
  readonly variance: number;
  readonly reml?: number;
}

interface ReferenceStudy {
  readonly options: DecisionStudyOptions;
  readonly universeScoreVariance: number;
  readonly relativeErrorVariance: number;
  readonly absoluteErrorVariance: number;
  readonly generalizability: number;
  readonly dependability: number;
}

interface ReferenceCase {
  readonly name: string;
  readonly design: GeneralizabilityDesign;
  readonly scores: FacetedScore[];
  readonly components: ReferenceComponent[];
  readonly studies: ReferenceStudy[];
}

// TypeScript types each case of the JSON with the keys of the others, so the cases are read back as one type.
const cases: readonly ReferenceCase[] = JSON.parse(JSON.stringify(fixture.cases));
const unbalancedCases: readonly ReferenceCase[] = JSON.parse(JSON.stringify(unbalancedFixture.cases));

/** Returns the case with the name. */
function caseNamed(name: string): ReferenceCase {
  return defined(cases.find((reference) => reference.name === name));
}

/** Returns the unbalanced case with the name. */
function unbalancedCase(name: string): ReferenceCase {
  return defined(unbalancedCases.find((reference) => reference.name === name));
}

/** Expects `actual` within `tolerance` times the larger of 1 and the expected value's size. */
function expectRelative(actual: number, expected: number, tolerance: number, label: string): void {
  expectClose(actual, expected, tolerance * Math.max(1, Math.abs(expected)), label);
}

/** Returns the scores of a persons-by-items matrix, in long format. */
function crossed(matrix: readonly (readonly number[])[]): FacetedScore[] {
  return matrix.flatMap((row, p) => row.map((score, i) => ({ levels: { person: `p${p}`, item: `i${i}` }, score })));
}

const PERSON_BY_ITEM: GeneralizabilityDesign = { facets: ["item"] };

describe("generalizabilityStudy", () => {
  for (const reference of cases) {
    it(`matches aov and the expected mean squares for ${reference.name}`, () => {
      const study = generalizabilityStudy(reference.scores, reference.design);
      expect(study.components).toHaveLength(reference.components.length);
      study.components.forEach((component, k) => {
        const expected = defined(reference.components[k]);
        expect(component.facets, `facets ${k}`).toEqual(expected.facets);
        expect(component.nestedIn, `nestedIn ${k}`).toEqual(expected.nestedIn);
        expect(component.degreesOfFreedom, `df ${k}`).toBe(expected.degreesOfFreedom);
        expectRelative(component.sumOfSquares, expected.sumOfSquares, ANOVA_TOLERANCE, `sum of squares ${k}`);
        expectRelative(component.meanSquare, expected.meanSquare, ANOVA_TOLERANCE, `mean square ${k}`);
        expectRelative(component.variance, expected.variance, ANOVA_TOLERANCE, `variance ${k}`);
      });
    });

    if (reference.components.every((component) => component.reml !== undefined)) {
      it(`matches the REML estimates of lme4 for ${reference.name}`, () => {
        const study = generalizabilityStudy(reference.scores, reference.design);
        study.components.forEach((component, k) => {
          expectClose(
            component.variance,
            defined(defined(reference.components[k]).reml),
            REML_TOLERANCE,
            `variance ${k}`,
          );
        });
      });
    }
  }

  for (const reference of unbalancedCases) {
    it(`matches Method 1 in base R for ${reference.name} with empty cells`, () => {
      const study = generalizabilityStudy(reference.scores, reference.design);
      expect(study.components).toHaveLength(reference.components.length);
      study.components.forEach((component, k) => {
        const expected = defined(reference.components[k]);
        expect(component.facets, `facets ${k}`).toEqual(expected.facets);
        expect(component.nestedIn, `nestedIn ${k}`).toEqual(expected.nestedIn);
        expect(component.degreesOfFreedom, `df ${k}`).toBe(expected.degreesOfFreedom);
        expectRelative(component.sumOfSquares, expected.sumOfSquares, UNBALANCED_TOLERANCE, `sum of squares ${k}`);
        expectRelative(component.meanSquare, expected.meanSquare, UNBALANCED_TOLERANCE, `mean square ${k}`);
        expectRelative(component.variance, expected.variance, UNBALANCED_TOLERANCE, `variance ${k}`);
      });
    });
  }

  it("gives the estimates of the balanced analysis for balanced data", () => {
    for (const reference of cases) {
      const balanced = generalizabilityStudy(reference.scores, reference.design);
      const study = generalizabilityStudy(reference.scores, { ...reference.design, unbalanced: true });
      expect(study.balanced, reference.name).toBe(true);
      expect(study.sampleSizes, reference.name).toEqual(balanced.sampleSizes);
      study.components.forEach((component, k) => {
        const expected = defined(balanced.components[k]);
        expect(component.facets, `${reference.name} facets ${k}`).toEqual(expected.facets);
        expect(component.degreesOfFreedom, `${reference.name} df ${k}`).toBe(expected.degreesOfFreedom);
        expectRelative(component.sumOfSquares, expected.sumOfSquares, BALANCED_TOLERANCE, `${reference.name} SS ${k}`);
        expectRelative(component.meanSquare, expected.meanSquare, BALANCED_TOLERANCE, `${reference.name} MS ${k}`);
        expectRelative(component.variance, expected.variance, BALANCED_TOLERANCE, `${reference.name} variance ${k}`);
      });
    }
  });

  it("reports the most elements of each facet in an unbalanced study, and that it is not balanced", () => {
    const { scores, design } = unbalancedCase("r:p");
    const study = generalizabilityStudy(scores, design);
    expect(study.balanced).toBe(false);
    expect(study.sampleSizes).toEqual({ person: 200, rater: 4 });
    expect(study.nested).toEqual({ rater: ["person"] });
    expect(generalizabilityStudy(unbalancedCase("i:p").scores, unbalancedCase("i:p").design).sampleSizes).toEqual({
      person: 250,
      item: 12,
    });
  });

  it("does not depend on the order of the scores of an unbalanced design", () => {
    const { scores, design } = unbalancedCase("(r:p) x i");
    const study = generalizabilityStudy(scores, design);
    const reversed = generalizabilityStudy([...scores].reverse(), design);
    reversed.components.forEach((component, k) => {
      const expected = defined(study.components[k]);
      expect(component.degreesOfFreedom).toBe(expected.degreesOfFreedom);
      expectRelative(component.variance, expected.variance, 1e-9, `variance ${k}`);
    });
  });

  it("reports a negative component of an unbalanced study as estimated", () => {
    const study = generalizabilityStudy(unbalancedCase("p x r x i").scores, unbalancedCase("p x r x i").design);
    expect(defined(study.components[1]).variance).toBeLessThan(0);
  });

  it("reports the design, the sizes of the facets and the effects in order", () => {
    const study = generalizabilityStudy(caseNamed("(r:p) x i").scores, caseNamed("(r:p) x i").design);
    expect(study.object).toBe("person");
    expect(study.facets).toEqual(["rater", "item"]);
    expect(study.nested).toEqual({ rater: ["person"] });
    expect(study.sampleSizes).toEqual({ person: 25, rater: 3, item: 4 });
    expect(study.components.map(({ facets, nestedIn }) => [facets.join(" x "), nestedIn.join(" x ")])).toEqual([
      ["person", ""],
      ["item", ""],
      ["rater", "person"],
      ["person x item", ""],
      ["rater x item", "person"],
    ]);
  });

  it("gives the exact components of a small design that can be solved by hand", () => {
    // Persons 1 and 5 on item 1, 3 and 7 on item 2: person means 2 and 6, item means 3 and 5, grand mean 4. The sums of
    // squares are 16 and 4, and every score is the sum of its person and item effects, so the residual is 0.
    const study = generalizabilityStudy(
      crossed([
        [1, 3],
        [5, 7],
      ]),
      PERSON_BY_ITEM,
    );
    expect(study.components.map(({ sumOfSquares }) => sumOfSquares)).toEqual([16, 4, 0]);
    expect(study.components.map(({ degreesOfFreedom }) => degreesOfFreedom)).toEqual([1, 1, 1]);
    expect(study.components.map(({ variance }) => variance)).toEqual([8, 2, 0]);
  });

  it("keeps its sums of squares for scores far from 0", () => {
    const base = crossed([
      [1, 3, 2],
      [5, 7, 8],
      [2, 2, 6],
    ]);
    const shifted = base.map(({ levels, score }) => ({ levels, score: score + 1e7 }));
    const study = generalizabilityStudy(base, PERSON_BY_ITEM);
    generalizabilityStudy(shifted, PERSON_BY_ITEM).components.forEach((component, k) => {
      expectRelative(component.sumOfSquares, defined(study.components[k]).sumOfSquares, 1e-8, `sum of squares ${k}`);
    });
  });

  it("identifies nested elements within their parents, whether labels repeat or are unique", () => {
    const { scores, design } = caseNamed("(r:p) x i");
    const repeated = generalizabilityStudy(scores, design);
    const unique = scores.map(({ levels, score }) => ({
      levels: { ...levels, rater: `${levels.person}/${levels.rater}` },
      score,
    }));
    expect(generalizabilityStudy(unique, design)).toEqual(repeated);
    // Raters with arbitrary labels: a different order of labels within each person.
    const shuffled = scores.map(({ levels, score }) => ({
      levels: { ...levels, rater: `${(Number(levels.rater) + Number(levels.person)) % 3}` },
      score,
    }));
    expect(generalizabilityStudy(shuffled, design).components.map(({ degreesOfFreedom }) => degreesOfFreedom)).toEqual(
      repeated.components.map(({ degreesOfFreedom }) => degreesOfFreedom),
    );
  });

  it("does not depend on the order of the scores", () => {
    const { scores, design } = caseNamed("p x r x i");
    const study = generalizabilityStudy(scores, design);
    for (const reordered of [
      [...scores].reverse(),
      scores.filter((_, k) => k % 2 === 0).concat(scores.filter((_, k) => k % 2 === 1)),
    ]) {
      generalizabilityStudy(reordered, design).components.forEach((component, k) => {
        const expected = defined(study.components[k]);
        expect(component.facets).toEqual(expected.facets);
        expectRelative(component.variance, expected.variance, 1e-10, `variance ${k}`);
      });
    }
  });

  it("ignores other entries of the levels of a score", () => {
    const { scores, design } = caseNamed("p x i");
    const extended = scores.map(({ levels, score }) => ({ levels: { ...levels, occasion: "1" }, score }));
    expect(generalizabilityStudy(extended, design)).toEqual(generalizabilityStudy(scores, design));
  });

  it("takes the object of measurement from the design", () => {
    const { scores } = caseNamed("p x i");
    const students = scores.map(({ levels, score }) => ({
      levels: { student: levels.person ?? "", item: levels.item ?? "" },
      score,
    }));
    const study = generalizabilityStudy(students, { object: "student", facets: ["item"] });
    expect(study.object).toBe("student");
    expect(study.components[0]?.facets).toEqual(["student"]);
    expect(study.components.map(({ variance }) => variance)).toEqual(
      generalizabilityStudy(scores, PERSON_BY_ITEM).components.map(({ variance }) => variance),
    );
  });

  it("estimates a negative component as it is and reports the others", () => {
    const { scores, design } = caseNamed("p x i with a negative component");
    const [person, item, residual] = generalizabilityStudy(scores, design).components;
    expect(defined(item).variance).toBeLessThan(0);
    expect(defined(person).variance).toBeGreaterThan(0);
    expect(defined(residual).variance).toBeGreaterThan(0);
  });

  it("gives zero components for scores that do not vary", () => {
    const study = generalizabilityStudy(
      crossed([
        [4, 4, 4],
        [4, 4, 4],
      ]),
      PERSON_BY_ITEM,
    );
    expect(study.components.map(({ variance }) => Math.abs(variance))).toEqual([0, 0, 0]);
  });

  it("names the first combination that has two scores", () => {
    const scores = crossed([
      [1, 2],
      [3, 4],
    ]);
    expect(() => generalizabilityStudy([...scores.slice(0, 3), defined(scores[1])], PERSON_BY_ITEM)).toThrow(
      'The design is not balanced: there is more than one score for person "p0", item "i1".',
    );
    expect(() =>
      generalizabilityStudy([...scores, { levels: { person: "p1", item: "i1" }, score: 5 }], PERSON_BY_ITEM),
    ).toThrow('The design is not balanced: there is more than one score for person "p1", item "i1".');
  });

  it("names the first combination that has no score", () => {
    const scores = crossed([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ]);
    const without = scores.filter((_, k) => k !== 4);
    expect(() => generalizabilityStudy(without, PERSON_BY_ITEM)).toThrow(
      'The design is not balanced: there is no score for person "p1", item "i1".',
    );
    const unequal = scores.filter((_, k) => k !== 5 && k !== 8);
    expect(() => generalizabilityStudy(unequal, PERSON_BY_ITEM)).toThrow(
      'The design is not balanced: there is no score for person "p1", item "i2".',
    );
  });

  it("names a missing combination of a nested facet by its number within its parent", () => {
    const { scores, design } = caseNamed("i:p");
    const [first] = scores;
    const fewer = scores.filter(
      (entry) => !(entry.levels.person === first?.levels.person && entry.levels.item === "6"),
    );
    expect(() => generalizabilityStudy(fewer, design)).toThrow(
      `The design is not balanced: there is no score for person "${first?.levels.person}", item number 6.`,
    );
  });

  it("rejects a facet with fewer than 2 elements and too few scores", () => {
    expect(() => generalizabilityStudy(crossed([[1], [2], [3], [4]]), PERSON_BY_ITEM)).toThrow(
      'The facet "item" needs at least 2 elements, found 1.',
    );
    expect(() => generalizabilityStudy(crossed([[1, 2]]).concat([]), PERSON_BY_ITEM)).toThrow(
      "A design of 2 facets needs at least 4 scores, two elements of every facet, found 2.",
    );
    const { scores } = caseNamed("i:p");
    const single = scores.filter((entry) => entry.levels.item === "1");
    expect(() => generalizabilityStudy(single, { facets: ["item"], nested: { item: ["person"] } })).toThrow(
      'The facet "item" needs at least 2 elements in each combination of the elements it is nested in, found 1.',
    );
  });

  it("rejects scores of other types", () => {
    const scores = crossed([
      [1, 2],
      [3, 4],
    ]);
    expect(() => generalizabilityStudy(JSON.parse("null"), PERSON_BY_ITEM)).toThrow(
      "The scores must be an array of objects { levels, score }, found null.",
    );
    expect(() =>
      generalizabilityStudy(JSON.parse('[{"levels": {"person": "a", "item": "b"}, "score": "3"}]'), PERSON_BY_ITEM),
    ).toThrow('Score 1 needs a finite score, found "3".');
    expect(() => generalizabilityStudy([{ levels: { person: "a" }, score: 1 }, ...scores], PERSON_BY_ITEM)).toThrow(
      'Score 1 needs a string element for the facet "item", found undefined.',
    );
    expect(() =>
      generalizabilityStudy([{ levels: { person: "a", item: "b" }, score: Number.NaN }], PERSON_BY_ITEM),
    ).toThrow(DataError);
    expect(() => generalizabilityStudy([], PERSON_BY_ITEM)).toThrow(DataError);
  });

  it("rejects designs of other types and invalid designs", () => {
    const scores = crossed([
      [1, 2],
      [3, 4],
    ]);
    const study = (design: unknown): GeneralizabilityStudy =>
      generalizabilityStudy(scores, JSON.parse(JSON.stringify(design)));
    expect(() => generalizabilityStudy(scores, JSON.parse("null"))).toThrow(
      "The design must be an object { object, facets, nested }, found null.",
    );
    expect(() => study({ facets: "item" })).toThrow(
      'The facets of the design must be an array of facet names, found "item".',
    );
    expect(() => study({ facets: [] })).toThrow(
      "The design needs at least one facet besides the object of measurement.",
    );
    expect(() => study({ object: "", facets: ["item"] })).toThrow(
      'The object of measurement must be a non-empty string, found "".',
    );
    expect(() => study({ object: 3, facets: ["item"] })).toThrow(DataError);
    expect(() => study({ facets: ["item", 7] })).toThrow("A facet name must be a non-empty string, found 7.");
    expect(() => study({ facets: ["item", "item"] })).toThrow('Facet names must be distinct; "item" appears twice.');
    expect(() => study({ facets: ["person"] })).toThrow('Facet names must be distinct; "person" appears twice.');
    expect(() => study({ facets: ["item"], nested: null })).toThrow(
      "The option nested of the design must be an object of the facets each facet is nested in, found null.",
    );
    expect(() => study({ facets: ["item"], nested: { rater: ["person"] } })).toThrow(
      'The option nested of the design names "rater", which is not a facet of the design.',
    );
    expect(() => study({ facets: ["item"], nested: { person: ["item"] } })).toThrow(
      'The object of measurement "person" cannot be nested in another facet.',
    );
    expect(() => study({ facets: ["item"], nested: { item: "person" } })).toThrow(
      'The option nested of the design must give the facets that "item" is nested in as an array, found "person".',
    );
    expect(() => study({ facets: ["item"], nested: { item: ["rater"] } })).toThrow(
      'Facet "item" is nested in "rater", which is not a facet of the design.',
    );
    expect(() => study({ facets: ["item"], nested: { item: ["item"] } })).toThrow(
      'Facet "item" cannot be nested in itself.',
    );
    expect(() => study({ facets: ["rater", "item"], nested: { rater: ["item"], item: ["rater"] } })).toThrow(
      'The nesting of the facets is circular at "rater".',
    );
    const many = Array.from({ length: 20 }, (_, k) => `facet${k}`);
    expect(() => generalizabilityStudy(scores, { facets: many })).toThrow(
      "A design has at most 6 facets with the object of measurement, found 21.",
    );
    expect(() => study({ facets: ["item"], unbalanced: "yes" })).toThrow(
      'The option unbalanced of the design must be a boolean, found "yes".',
    );
    expect(() => study({ facets: ["item"], unbalanced: null })).toThrow(DataError);
  });

  describe("without the option unbalanced", () => {
    const missing = crossed([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ]).filter((_, k) => k !== 4);

    it("still throws for a missing score, which the option accepts", () => {
      expect(() => generalizabilityStudy(missing, PERSON_BY_ITEM)).toThrow(
        'The design is not balanced: there is no score for person "p1", item "i1".',
      );
      const study = generalizabilityStudy(missing, { ...PERSON_BY_ITEM, unbalanced: true });
      expect(study.balanced).toBe(false);
      expect(study.sampleSizes).toEqual({ person: 3, item: 3 });
      expect(study.components.map(({ degreesOfFreedom }) => degreesOfFreedom)).toEqual([2, 2, 3]);
    });
  });

  it("names the first combination that has two scores, in an unbalanced design", () => {
    const scores = crossed([
      [1, 2],
      [3, 4],
    ]);
    const design = { ...PERSON_BY_ITEM, unbalanced: true };
    expect(() =>
      generalizabilityStudy([...scores, { levels: { person: "p1", item: "i1" }, score: 5 }], design),
    ).toThrow(
      'There is more than one score for person "p1", item "i1"; an unbalanced design allows at most one for each combination of elements.',
    );
  });

  it("names an effect whose variance cannot be told apart from the rest", () => {
    // Persons p0 to p2 and raters r0 to r2 linked in a chain: 5 scores for 3 + 3 - 1 effects of the additive model, so
    // the person by rater effect has no degrees of freedom and is confounded with the error.
    const chain = [
      ["p0", "r0"],
      ["p0", "r1"],
      ["p1", "r1"],
      ["p1", "r2"],
      ["p2", "r2"],
    ].map(([person, rater], k) => ({ levels: { person: person ?? "", rater: rater ?? "" }, score: k * k }));
    expect(() => generalizabilityStudy(chain, { facets: ["rater"], unbalanced: true })).toThrow(
      'The variance component of "person x rater" cannot be estimated from these scores: it is confounded with those of other effects, as when a facet has one element or two facets always occur together.',
    );
    const nested = [
      ["p0", "a"],
      ["p0", "b"],
      ["p1", "a"],
      ["p2", "a"],
    ].map(([person, rater], k) => ({ levels: { person: person ?? "", rater: rater ?? "" }, score: k }));
    const rated = { facets: ["rater"], nested: { rater: ["person"] }, unbalanced: true };
    expect(generalizabilityStudy(nested, rated).components.map(({ degreesOfFreedom }) => degreesOfFreedom)).toEqual([
      2, 1,
    ]);
    expect(() =>
      generalizabilityStudy(
        nested.filter((entry) => entry.levels.rater === "a"),
        rated,
      ),
    ).toThrow(
      'The facet "rater" needs at least 2 elements in each combination of the elements it is nested in, found 1.',
    );
  });

  it("rejects a design of more than 6 facets, balanced or not, as GENOVA and urGENOVA", () => {
    /** Every combination of two elements of each facet, one score each: a balanced design of `count` facets. */
    const crossed = (
      count: number,
    ): { facets: string[]; scores: { levels: Record<string, string>; score: number }[] } => {
      const facets = Array.from({ length: count - 1 }, (_, k) => `facet${k}`);
      const names = ["person", ...facets];
      const scores = Array.from({ length: 2 ** count }, (_, cell) => ({
        levels: Object.fromEntries(names.map((name, f) => [name, String((cell >> f) & 1)])),
        score: (cell * 7) % 5,
      }));
      return { facets, scores };
    };
    const six = crossed(6);
    expect(generalizabilityStudy(six.scores, { facets: six.facets }).components).toHaveLength(63);
    // A balanced design of 16 facets of two elements takes about 20 seconds, and of 20 facets tens of gigabytes.
    const seven = crossed(7);
    const message = "A design has at most 6 facets with the object of measurement, found 7.";
    expect(() => generalizabilityStudy(seven.scores, { facets: seven.facets })).toThrow(message);
    expect(() => generalizabilityStudy(seven.scores.slice(1), { facets: seven.facets, unbalanced: true })).toThrow(
      message,
    );
  });
});

describe("decisionStudy", () => {
  for (const reference of [...cases, ...unbalancedCases].filter(({ studies }) => studies.length > 0)) {
    const unbalanced = reference.design.unbalanced === true;
    it(`matches the textbook formulas for ${reference.name}${unbalanced ? " with empty cells" : ""}`, () => {
      const study = generalizabilityStudy(reference.scores, reference.design);
      for (const expected of reference.studies) {
        const label = JSON.stringify(expected.options);
        const actual = decisionStudy(study, expected.options);
        expectRelative(
          actual.universeScoreVariance,
          expected.universeScoreVariance,
          DECISION_TOLERANCE,
          `universe ${label}`,
        );
        expectRelative(
          actual.relativeErrorVariance,
          expected.relativeErrorVariance,
          DECISION_TOLERANCE,
          `relative ${label}`,
        );
        expectRelative(
          actual.absoluteErrorVariance,
          expected.absoluteErrorVariance,
          DECISION_TOLERANCE,
          `absolute ${label}`,
        );
        expectRelative(
          actual.generalizability,
          expected.generalizability,
          DECISION_TOLERANCE,
          `generalizability ${label}`,
        );
        expectRelative(actual.dependability, expected.dependability, DECISION_TOLERANCE, `Phi ${label}`);
      }
    });
  }

  const raters = generalizabilityStudy(caseNamed("p x r x i").scores, caseNamed("p x r x i").design);
  const label = (component: { facets: readonly string[]; nestedIn: readonly string[] }): string =>
    `${component.facets.join(" x ")}${component.nestedIn.length > 0 ? ` in ${component.nestedIn.join(" x ")}` : ""}`;

  it("takes the sizes of the study unless told otherwise", () => {
    const defaults = decisionStudy(raters);
    const explicit = decisionStudy(raters, { sampleSizes: { rater: 4, item: 6 } });
    expect(defaults).toEqual(explicit);
    expect(defaults.components.find((component) => label(component) === "person x rater x item")?.divisor).toBe(24);
  });

  it("divides each component by the sizes of its facets other than the object", () => {
    const { components } = decisionStudy(raters, { sampleSizes: { rater: 2, item: 10 } });
    const divisors = Object.fromEntries(components.map((component) => [label(component), component.divisor]));
    expect(divisors).toEqual({
      person: 1,
      rater: 2,
      item: 10,
      "person x rater": 2,
      "person x item": 10,
      "rater x item": 20,
      "person x rater x item": 20,
    });
  });

  it("puts each effect in the universe score or the relative or absolute error", () => {
    const flags = (study: GeneralizabilityStudy, options: DecisionStudyOptions): Record<string, string> =>
      Object.fromEntries(
        decisionStudy(study, options).components.map((component) => [
          label(component),
          [component.universe && "universe", component.relative && "relative", component.absolute && "absolute"]
            .filter(Boolean)
            .join(" "),
        ]),
      );
    expect(flags(raters, {})).toEqual({
      person: "universe",
      rater: "absolute",
      item: "absolute",
      "person x rater": "relative absolute",
      "person x item": "relative absolute",
      "rater x item": "absolute",
      "person x rater x item": "relative absolute",
    });
    expect(flags(raters, { fixed: ["item"] })).toEqual({
      person: "universe",
      rater: "absolute",
      item: "",
      "person x rater": "relative absolute",
      "person x item": "universe",
      "rater x item": "absolute",
      "person x rater x item": "relative absolute",
    });
  });

  it("counts a negative component as 0 and reports the variance it adds", () => {
    const { scores, design } = caseNamed("p x i with a negative component");
    const study = generalizabilityStudy(scores, design);
    const { components, absoluteErrorVariance, relativeErrorVariance } = decisionStudy(study);
    expect(components.find((component) => component.facets.join("") === "item")?.variance).toBe(0);
    expect(absoluteErrorVariance).toBe(relativeErrorVariance);
  });

  it("pools the components of a crossed facet that the decision study nests", () => {
    const pooled = decisionStudy(raters, { nested: { rater: ["person"] } });
    const byLabel = Object.fromEntries(pooled.components.map((component) => [label(component), component.variance]));
    const variance = (...facets: string[]): number =>
      defined(raters.components.find((component) => component.facets.join("") === facets.join("")))?.variance;
    expect(Object.keys(byLabel)).toEqual([
      "person",
      "item",
      "rater in person",
      "person x item",
      "rater x item in person",
    ]);
    expectRelative(defined(byLabel["rater in person"]), variance("rater") + variance("person", "rater"), 1e-12, "r:p");
    expectRelative(
      defined(byLabel["rater x item in person"]),
      variance("rater", "item") + variance("person", "rater", "item"),
      1e-12,
      "ri:p",
    );
  });

  it("accepts a nesting the study already has", () => {
    const { scores, design } = caseNamed("(r:p) x i");
    const study = generalizabilityStudy(scores, design);
    const sizes = { sampleSizes: { rater: 2, item: 6 } };
    expect(decisionStudy(study, { ...sizes, nested: { rater: ["person"] } })).toEqual(decisionStudy(study, sizes));
  });

  it("treats a decision study with every facet fixed as having no error", () => {
    const study = generalizabilityStudy(caseNamed("p x i").scores, caseNamed("p x i").design);
    const result = decisionStudy(study, { fixed: ["item"] });
    expect(result.relativeErrorVariance).toBe(0);
    expect(result.absoluteErrorVariance).toBe(0);
    expect(result.generalizability).toBe(1);
    expect(result.dependability).toBe(1);
  });

  it("needs the sizes of every facet of a study that is not balanced", () => {
    const study = generalizabilityStudy(unbalancedCase("p x r x i").scores, unbalancedCase("p x r x i").design);
    expect(() => decisionStudy(study)).toThrow(
      'The study is not balanced, so its facets have no one number of elements: the option sampleSizes must give one for "rater".',
    );
    expect(() => decisionStudy(study, { sampleSizes: { rater: 3 } })).toThrow(
      'The study is not balanced, so its facets have no one number of elements: the option sampleSizes must give one for "item".',
    );
    expect(decisionStudy(study, { sampleSizes: { rater: 3, item: 5 } }).generalizability).toBeGreaterThan(0);
  });

  it("takes the sizes of balanced data that went through the unbalanced analysis", () => {
    const { scores, design } = caseNamed("p x r x i");
    const balanced = generalizabilityStudy(scores, design);
    const unbalanced = generalizabilityStudy(scores, { ...design, unbalanced: true });
    const [expected, actual] = [decisionStudy(balanced), decisionStudy(unbalanced)];
    expectRelative(actual.generalizability, expected.generalizability, BALANCED_TOLERANCE, "generalizability");
    expectRelative(actual.dependability, expected.dependability, BALANCED_TOLERANCE, "dependability");
  });

  it("works from a study that has been through JSON", () => {
    const study = generalizabilityStudy(caseNamed("(i:p) x (r:p)").scores, caseNamed("(i:p) x (r:p)").design);
    expect(decisionStudy(JSON.parse(JSON.stringify(study)), { sampleSizes: { item: 5 } })).toEqual(
      decisionStudy(study, { sampleSizes: { item: 5 } }),
    );
  });

  it("rejects options of other types and invalid options", () => {
    expect(() => decisionStudy(raters, JSON.parse("null"))).toThrow("The options must be an object, found null.");
    expect(() => decisionStudy(raters, { sampleSizes: JSON.parse('"3"') })).toThrow(
      'The option sampleSizes must be an object of numbers of elements by facet, found "3".',
    );
    expect(() => decisionStudy(raters, { sampleSizes: { item: 0 } })).toThrow(
      'The option sampleSizes for "item" must be a positive integer, found 0.',
    );
    expect(() => decisionStudy(raters, { sampleSizes: { item: 2.5 } })).toThrow(DataError);
    expect(() => decisionStudy(raters, { sampleSizes: { item: JSON.parse('"2"') } })).toThrow(DataError);
    expect(() => decisionStudy(raters, { sampleSizes: { occasion: 2 } })).toThrow(
      'The option sampleSizes names "occasion", which is not a facet of the study; its facets are "rater", "item".',
    );
    expect(() => decisionStudy(raters, { sampleSizes: { person: 2 } })).toThrow(DataError);
    expect(() => decisionStudy(raters, { fixed: JSON.parse('"item"') })).toThrow(
      'The option fixed must be an array of facet names, found "item".',
    );
    expect(() => decisionStudy(raters, { fixed: ["occasion"] })).toThrow(
      'The option fixed names "occasion", which is not a facet of the study; its facets are "rater", "item".',
    );
    expect(() => decisionStudy(raters, { nested: JSON.parse("null") })).toThrow(
      "The option nested must be an object of the facets each facet is nested in, found null.",
    );
    expect(() => decisionStudy(raters, { nested: { person: ["item"] } })).toThrow(
      'The object of measurement "person" cannot be nested in another facet.',
    );
    expect(() => decisionStudy(raters, { nested: { rater: ["item"], item: ["rater"] } })).toThrow(
      'The nesting of the facets is circular at "rater".',
    );
    const { scores, design } = caseNamed("i:r:p");
    const chain = generalizabilityStudy(scores, design);
    expect(() => decisionStudy(chain, { nested: { rater: ["item"] } })).toThrow(DataError);
  });

  it("rejects a study that is not the result of generalizabilityStudy", () => {
    const parsed = (changes: object): GeneralizabilityStudy => JSON.parse(JSON.stringify({ ...raters, ...changes }));
    expect(() => decisionStudy(JSON.parse("null"))).toThrow(
      "The study must be an object from generalizabilityStudy, found null.",
    );
    expect(() => decisionStudy(parsed({ sampleSizes: 4 }))).toThrow(DataError);
    expect(() => decisionStudy(parsed({ facets: "rater" }))).toThrow(DataError);
    expect(() => decisionStudy(parsed({ components: {} }))).toThrow(DataError);
    expect(() => decisionStudy(parsed({ object: "" }))).toThrow(DataError);
    expect(() => decisionStudy(parsed({ sampleSizes: { person: 40, rater: 4 } }))).toThrow(
      'The size of "item" in the study must be a positive integer, found undefined.',
    );
    expect(() => decisionStudy(parsed({ components: raters.components.slice(1) }))).toThrow(
      "The study needs one component for each of the 7 effects of its design, found 6.",
    );
    const duplicate = [...raters.components.slice(0, 6), defined(raters.components[0])];
    expect(() => decisionStudy(parsed({ components: duplicate }))).toThrow(
      "Component 7 of the study does not name a distinct effect of the study's design; the effects are those of the study from generalizabilityStudy.",
    );
    const renamed = raters.components.map((component, k) =>
      k === 0 ? { ...component, facets: ["occasion"] } : component,
    );
    expect(() => decisionStudy(parsed({ components: renamed }))).toThrow(
      'Component 1 of the study names "occasion", which is not a facet of the study.',
    );
    const infinite = raters.components.map((component, k) =>
      k === 0 ? { ...component, variance: Number.NaN } : component,
    );
    expect(() => decisionStudy(parsed({ components: infinite }))).toThrow(DataError);
    expect(() => decisionStudy(parsed({ balanced: "yes" }))).toThrow(
      'The field balanced of the study must be a boolean, found "yes".',
    );
    const notRecord = JSON.parse(JSON.stringify({ ...raters, components: [...raters.components.slice(1), 5] }));
    expect(() => decisionStudy(notRecord)).toThrow(DataError);
  });

  it("rejects a study whose universe score and relative error variances are both 0", () => {
    const flat = generalizabilityStudy(
      crossed([
        [4, 4, 4],
        [4, 4, 4],
      ]),
      PERSON_BY_ITEM,
    );
    expect(() => decisionStudy(flat)).toThrow(
      "The coefficients are undefined: the universe score variance and the relative error variance are both 0.",
    );
  });
});

describe("the recipe for a multivariate G study", () => {
  const persons = 300;
  const items = 6;
  const random = createRandom(2026);
  const reading = Array.from({ length: persons }, () => random.normal());
  const listening = reading.map((score) => 0.6 * score + 0.8 * random.normal());
  const readingItems = Array.from({ length: items }, () => 0.3 * random.normal());
  const listeningItems = Array.from({ length: items }, () => 0.3 * random.normal());
  const noise = () => random.normal();
  const x = reading.map((truth) => readingItems.map((effect) => truth + effect + noise()));
  const y = listening.map((truth) => listeningItems.map((effect) => truth + effect + noise()));

  /** The variance components of p × i for a score at each cell. */
  const components = (cell: (person: number, item: number) => number): number[] =>
    generalizabilityStudy(
      Array.from({ length: persons * items }, (_, k) => {
        const person = Math.floor(k / items);
        const item = k % items;
        return { levels: { person: `p${person}`, item: `i${item}` }, score: cell(person, item) };
      }),
      { facets: ["item"] },
    ).components.map(({ variance }) => variance);

  const varianceX = components((p, i) => defined(x[p])[i] ?? 0);
  const varianceY = components((p, i) => defined(y[p])[i] ?? 0);
  const varianceSum = components((p, i) => (defined(x[p])[i] ?? 0) + (defined(y[p])[i] ?? 0));
  const covariances = varianceSum.map((total, k) => (total - (varianceX[k] ?? 0) - (varianceY[k] ?? 0)) / 2);

  /** The covariance components from the mean products of the two scores, the way the mean squares are made. */
  const direct = (): number[] => {
    const mean = (values: number[]): number => values.reduce((sum, value) => sum + value, 0) / values.length;
    const cell = (matrix: number[][], p: number, i: number): number => defined(matrix[p])[i] ?? 0;
    const personMeans = (matrix: number[][]) => matrix.map(mean);
    const itemMeans = (matrix: number[][]) =>
      Array.from({ length: items }, (_, i) => mean(matrix.map((row) => row[i] ?? 0)));
    const grand = (matrix: number[][]) => mean(matrix.flat());
    const [px, py, ix, iy, gx, gy] = [personMeans(x), personMeans(y), itemMeans(x), itemMeans(y), grand(x), grand(y)];
    let personProducts = 0;
    for (let p = 0; p < persons; p++) personProducts += ((px[p] ?? 0) - gx) * ((py[p] ?? 0) - gy);
    let itemProducts = 0;
    for (let i = 0; i < items; i++) itemProducts += ((ix[i] ?? 0) - gx) * ((iy[i] ?? 0) - gy);
    let cellProducts = 0;
    for (let p = 0; p < persons; p++) {
      for (let i = 0; i < items; i++) {
        cellProducts +=
          (cell(x, p, i) - (px[p] ?? 0) - (ix[i] ?? 0) + gx) * (cell(y, p, i) - (py[p] ?? 0) - (iy[i] ?? 0) + gy);
      }
    }
    const productPerson = (items * personProducts) / (persons - 1);
    const productItem = (persons * itemProducts) / (items - 1);
    const productCell = cellProducts / ((persons - 1) * (items - 1));
    return [(productPerson - productCell) / items, (productItem - productCell) / persons, productCell];
  };

  it("takes the covariance components of two scores from studies of each and of their sum, as the direct estimate does", () => {
    const expected = direct();
    covariances.forEach((covariance, k) => {
      expect(covariance).toBeCloseTo(defined(expected[k]), 9);
    });
  });
});

describe("properties", () => {
  /** Scores of persons by items from 0 to 20 in a matrix of 2 to 6 persons and 2 to 5 items. */
  const matrix = fc.tuple(fc.integer({ min: 2, max: 6 }), fc.integer({ min: 2, max: 5 })).chain(([persons, items]) =>
    fc.array(fc.array(fc.integer({ min: 0, max: 20 }), { minLength: items, maxLength: items }), {
      minLength: persons,
      maxLength: persons,
    }),
  );

  it("degrees of freedom and sums of squares add up to those of the scores, for every design", () => {
    const total = (scores: readonly FacetedScore[]): number => {
      const mean = scores.reduce((sum, { score }) => sum + score, 0) / scores.length;
      return scores.reduce((sum, { score }) => sum + (score - mean) ** 2, 0);
    };
    for (const reference of cases) {
      const { components } = generalizabilityStudy(reference.scores, reference.design);
      expect(
        components.reduce((sum, { degreesOfFreedom }) => sum + degreesOfFreedom, 0),
        reference.name,
      ).toBe(reference.scores.length - 1);
      expectRelative(
        components.reduce((sum, { sumOfSquares }) => sum + sumOfSquares, 0),
        total(reference.scores),
        1e-10,
        reference.name,
      );
    }
    fc.assert(
      fc.property(matrix, (values) => {
        const scores = crossed(values);
        const { components } = generalizabilityStudy(scores, PERSON_BY_ITEM);
        expect(components.reduce((sum, { degreesOfFreedom }) => sum + degreesOfFreedom, 0)).toBe(scores.length - 1);
        expectRelative(
          components.reduce((sum, { sumOfSquares }) => sum + sumOfSquares, 0),
          total(scores),
          1e-9,
          "sum of squares",
        );
      }),
    );
  });

  it("components scale with the square of a scale and do not change with a shift of the scores", () => {
    fc.assert(
      fc.property(matrix, fc.integer({ min: 1, max: 5 }), fc.integer({ min: -50, max: 50 }), (values, scale, shift) => {
        const study = generalizabilityStudy(crossed(values), PERSON_BY_ITEM);
        const moved = generalizabilityStudy(
          crossed(values.map((row) => row.map((score) => scale * score + shift))),
          PERSON_BY_ITEM,
        );
        moved.components.forEach((component, k) => {
          const expected = defined(study.components[k]);
          expect(component.degreesOfFreedom).toBe(expected.degreesOfFreedom);
          expectRelative(component.variance, scale ** 2 * expected.variance, 1e-9, `variance ${k}`);
          expectRelative(component.meanSquare, scale ** 2 * expected.meanSquare, 1e-9, `mean square ${k}`);
        });
      }),
    );
  });

  /** Persons by items as a matrix with at most one empty cell for each person, as the item to leave out or -1. */
  const sparse = fc.tuple(fc.integer({ min: 3, max: 6 }), fc.integer({ min: 3, max: 5 })).chain(([persons, items]) =>
    fc.tuple(
      fc.array(fc.array(fc.integer({ min: 0, max: 20 }), { minLength: items, maxLength: items }), {
        minLength: persons,
        maxLength: persons,
      }),
      fc.array(fc.integer({ min: -1, max: items - 1 }), { minLength: persons, maxLength: persons }),
    ),
  );
  const withoutCells = (values: readonly (readonly number[])[], left: readonly number[], map = (x: number) => x) =>
    crossed(values.map((row) => row.map(map))).filter(
      (_, k) => left[Math.floor(k / (values[0]?.length ?? 1))] !== k % (values[0]?.length ?? 1),
    );
  const Unbalanced: GeneralizabilityDesign = { ...PERSON_BY_ITEM, unbalanced: true };

  it("unbalanced components scale with the square of a scale and do not change with a shift", () => {
    fc.assert(
      fc.property(
        sparse,
        fc.integer({ min: 1, max: 5 }),
        fc.integer({ min: -50, max: 50 }),
        ([values, left], scale, shift) => {
          const study = generalizabilityStudy(withoutCells(values, left), Unbalanced);
          const moved = generalizabilityStudy(
            withoutCells(values, left, (x) => scale * x + shift),
            Unbalanced,
          );
          moved.components.forEach((component, k) => {
            const expected = defined(study.components[k]);
            expectRelative(component.variance, scale ** 2 * expected.variance, 1e-8, `variance ${k}`);
            expectRelative(component.sumOfSquares, scale ** 2 * expected.sumOfSquares, 1e-8, `sum of squares ${k}`);
          });
        },
      ),
    );
  });

  it("unbalanced degrees of freedom add up to those of the scores", () => {
    fc.assert(
      fc.property(sparse, ([values, left]) => {
        const scores = withoutCells(values, left);
        const { components } = generalizabilityStudy(scores, Unbalanced);
        expect(components.reduce((sum, { degreesOfFreedom }) => sum + degreesOfFreedom, 0)).toBe(scores.length - 1);
      }),
    );
  });

  it("recovers the variance components from persons by raters with a quarter of the scores missing", () => {
    // 200 seeded samples of 60 persons and 4 raters with components 1, 0.2 and 0.5. The thresholds were fixed before the
    // first run: the mean of each estimate within 0.1 of its component, which Method 1, being unbiased, puts at about 5
    // standard errors of the mean.
    const random = createRandom(20261029);
    const totals = [0, 0, 0];
    const samples = 200;
    for (let sample = 0; sample < samples; sample++) {
      const persons = Array.from({ length: 60 }, () => random.normal());
      const raters = Array.from({ length: 4 }, () => Math.sqrt(0.2) * random.normal());
      const scores: FacetedScore[] = [];
      for (const [p, person] of persons.entries()) {
        for (const [r, rater] of raters.entries()) {
          const score = 5 + person + rater + Math.sqrt(0.5) * random.normal();
          if (random.uniform() < 0.75) scores.push({ levels: { person: `p${p}`, rater: `r${r}` }, score });
        }
      }
      const { components } = generalizabilityStudy(scores, { facets: ["rater"], unbalanced: true });
      components.forEach((component, k) => {
        totals[k] = (totals[k] ?? 0) + component.variance / samples;
      });
    }
    for (const [k, truth] of [1, 0.2, 0.5].entries()) {
      expect(Math.abs((totals[k] ?? 0) - truth), `component ${k}`).toBeLessThan(0.1);
    }
  });

  it("components do not depend on the order of persons and items", () => {
    fc.assert(
      fc.property(matrix, (values) => {
        const study = generalizabilityStudy(crossed(values), PERSON_BY_ITEM);
        const reversed = generalizabilityStudy(
          crossed([...values].reverse().map((row) => [...row].reverse())),
          PERSON_BY_ITEM,
        );
        reversed.components.forEach((component, k) => {
          expectRelative(component.variance, defined(study.components[k]).variance, 1e-9, `variance ${k}`);
        });
      }),
    );
  });

  it("the coefficients are from 0 to 1, dependability is at most generalizability, and both rise with more items", () => {
    fc.assert(
      fc.property(matrix, fc.integer({ min: 1, max: 10 }), (values, items) => {
        const study = generalizabilityStudy(crossed(values), PERSON_BY_ITEM);
        // Scores with no person variance and no error variance leave the coefficients undefined.
        const decide = (size: number): DecisionStudy | null => {
          try {
            return decisionStudy(study, { sampleSizes: { item: size } });
          } catch (error) {
            expect(error).toBeInstanceOf(DataError);
            return null;
          }
        };
        const result = decide(items);
        const more = decide(items + 1);
        if (result === null || more === null) return;
        for (const step of [result, more]) {
          expect(step.generalizability).toBeGreaterThanOrEqual(0);
          expect(step.generalizability).toBeLessThanOrEqual(1);
          expect(step.dependability).toBeGreaterThanOrEqual(0);
          expect(step.dependability).toBeLessThanOrEqual(step.generalizability + 1e-12);
        }
        expect(more.generalizability).toBeGreaterThanOrEqual(result.generalizability - 1e-12);
        expect(more.dependability).toBeGreaterThanOrEqual(result.dependability - 1e-12);
      }),
    );
  });

  it("a study that has been through JSON gives the same decision study", () => {
    fc.assert(
      fc.property(matrix, (values) => {
        const study = generalizabilityStudy(crossed(values), PERSON_BY_ITEM);
        expect(JSON.parse(JSON.stringify(study))).toEqual(study);
      }),
    );
  });
});
