import { checkRecord } from "@measurekit/core";
import { checkFacetedScores, type FacetedScore } from "../data/faceted-scores.js";
import { type BalancedScores, balancedScores } from "./balanced-scores.js";
import { buildStructure, type Effect, effectsOf, facetsOf, namesOf, type Structure } from "./structure.js";

/**
 * The design of a generalizability study: the facet of the objects of measurement, the other facets and which are
 * nested in which.
 */
export interface GeneralizabilityDesign {
  /**
   * The facet of the objects of measurement, whose scores the study separates from error.
   *
   * @defaultValue `"person"`
   */
  readonly object?: string;
  /** The other facets, distinct names that differ from the object's, such as `["rater", "item"]`. */
  readonly facets: readonly string[];
  /**
   * The facets each facet is nested in, by facet name: `{ rater: ["person"] }` is the design (r:p) x i, in which each
   * person has raters of their own. The object of measurement is not nested. Nested elements are identified within
   * the elements they are nested in, so labels may repeat from one to the next.
   *
   * @defaultValue every facet crossed with every other
   */
  readonly nested?: Readonly<Record<string, readonly string[]>>;
}

/** One effect of a study: its sums of squares, mean square and the estimate of its variance component. */
export interface VarianceComponent {
  /** The effect's own facets, in the design's order: the rater and the item of the effect of ri:p. */
  readonly facets: readonly string[];
  /** The facets the effect is nested in, such as the person of r:p; none for a crossed effect. */
  readonly nestedIn: readonly string[];
  /** The degrees of freedom: the product of the sizes less 1 of its own facets and the sizes of those it is nested in. */
  readonly degreesOfFreedom: number;
  /** The sum of squares. */
  readonly sumOfSquares: number;
  /** The sum of squares over the degrees of freedom. */
  readonly meanSquare: number;
  /** The analysis of variance estimate of the variance component, which is negative when the mean squares say so. */
  readonly variance: number;
}

/** The variance components of a balanced design, estimated by the analysis of variance. */
export interface GeneralizabilityStudy {
  /** The facet of the objects of measurement. */
  readonly object: string;
  /** The other facets, in the order of the design. */
  readonly facets: readonly string[];
  /** The facets each nested facet is nested in, by facet name; crossed facets are absent. */
  readonly nested: Readonly<Record<string, readonly string[]>>;
  /** The number of elements of each facet, the object's included: for a nested facet, within each element it is nested in. */
  readonly sampleSizes: Readonly<Record<string, number>>;
  /**
   * The effects of the design, one component each, ordered by their number of facets and then by the design's order of
   * facets: for p x i, the person, the item, then the person by item effect.
   */
  readonly components: readonly VarianceComponent[];
}

/**
 * Estimates the variance components of a balanced crossed or nested design by the analysis of variance.
 *
 * @remarks
 * Every combination of the facets' elements has one score, and a nested facet has the same number of elements in each
 * element it is nested in; `scores` are in long format, one `FacetedScore` for each, with a string element of the
 * object and of every facet in `levels`, other entries of `levels` being ignored. Each facet needs at least 2 elements,
 * so a design of k facets has at least 2^k scores. Sums of squares come from the marginal means of each effect, and the
 * variance components from the expected mean squares of a random-effects model, solved from the highest-order effect
 * down: each mean square is a sum of the variance components of the effects that include its facets, each multiplied by
 * the product of the sizes of the facets it does not include. The highest-order effect is confounded with error. A
 * component is negative when its mean square is below that of the effects it is confounded with; it is reported as
 * estimated, and a decision study treats it as 0. Designs that are not balanced, such as incomplete rating designs, are
 * outside the study; the many-facet Rasch model of the rater module estimates them.
 *
 * @throws `DataError` when the design is not an object with distinct non-empty facet names and a nesting of the facets
 * that is acyclic and leaves the object crossed, the scores are not an array of objects with a finite `score` and the
 * elements of the design's facets, a facet has fewer than 2 elements, or the design is not balanced, naming the first
 * combination of elements with more than one score or with none.
 * @see {@link https://doi.org/10.1007/978-1-4757-3456-0 | Brennan (2001)}
 */
export function generalizabilityStudy(
  scores: readonly FacetedScore[],
  design: GeneralizabilityDesign,
): GeneralizabilityStudy {
  checkRecord(design, "The design must be an object { object, facets, nested }");
  const { object = "person", facets, nested = {} } = design;
  const structure = buildStructure(object, facets, nested, "The option nested of the design");
  checkFacetedScores(scores, structure.names);
  const table = balancedScores(scores, structure);
  return {
    object,
    facets: [...facets],
    nested: Object.fromEntries(Object.entries(nested).map(([facet, parents]) => [facet, [...parents]])),
    sampleSizes: Object.fromEntries(structure.names.map((name, facet) => [name, table.sizes[facet] ?? 0])),
    components: components(structure, table),
  };
}

/** Returns the components of the design from its table of scores. */
function components(structure: Structure, table: BalancedScores): VarianceComponent[] {
  const effects = effectsOf(structure);
  const total = table.cells.length;
  const sums = marginalSums(table, structure);
  const size = (mask: number): number =>
    facetsOf(mask).reduce((product, facet) => product * (table.sizes[facet] ?? 1), 1);
  const uncorrected = (mask: number): number => {
    const marginal = sums(mask);
    let squares = 0;
    for (const value of marginal) squares += value * value;
    return squares / (total / marginal.length);
  };
  const rows = effects.map((effect) => {
    const degreesOfFreedom = facetsOf(effect.own).reduce(
      (product, facet) => product * ((table.sizes[facet] ?? 1) - 1),
      size(effect.nestedIn),
    );
    const sumOfSquares = alternatingSum(effect, uncorrected);
    return { effect, degreesOfFreedom, sumOfSquares, meanSquare: sumOfSquares / degreesOfFreedom };
  });
  const variances = new Array<number>(rows.length).fill(0);
  // Each mean square is the sum, over the effects whose facets include its own, of their variance components times the
  // product of the sizes of the facets they lack, so the effects with the most facets are solved first.
  for (let row = rows.length - 1; row >= 0; row--) {
    const mask = rows[row]?.effect.mask ?? 0;
    let expected = rows[row]?.meanSquare ?? 0;
    for (let other = row + 1; other < rows.length; other++) {
      const included = rows[other]?.effect.mask ?? 0;
      if ((included & mask) === mask) expected -= (total / size(included)) * (variances[other] ?? 0);
    }
    variances[row] = expected / (total / size(mask));
  }
  return rows.map(({ effect, degreesOfFreedom, sumOfSquares, meanSquare }, row) => ({
    facets: namesOf(structure, effect.own),
    nestedIn: namesOf(structure, effect.nestedIn),
    degreesOfFreedom,
    sumOfSquares,
    meanSquare,
    variance: variances[row] ?? 0,
  }));
}

/**
 * Returns the sum of squares of an effect: the uncorrected sums of squares of the facets it is nested in, plus each
 * subset of its own facets, added and subtracted in turn by the number of own facets left out.
 */
function alternatingSum(effect: Effect, uncorrected: (mask: number) => number): number {
  let sum = 0;
  const own = facetsOf(effect.own).length;
  for (let subset = effect.own; ; subset = (subset - 1) & effect.own) {
    const left = own - facetsOf(subset).length;
    sum += (left % 2 === 0 ? 1 : -1) * uncorrected(subset | effect.nestedIn);
    if (subset === 0) break;
  }
  return sum;
}

/**
 * Returns a function of the facets of an effect, as a mask, for the sums of the table's scores over the combinations
 * of the others' elements, by the combinations of the effect's own: computed once for each mask, each from a mask with
 * one more facet.
 */
function marginalSums(table: BalancedScores, structure: Structure): (mask: number) => Float64Array {
  const full = 2 ** structure.names.length - 1;
  const memo = new Map<number, Float64Array>([[full, table.cells]]);
  const sumsOf = (mask: number): Float64Array => {
    const known = memo.get(mask);
    if (known !== undefined) return known;
    // A facet that is missing and whose ancestors are all present: adding it leaves a set that holds its ancestors.
    const facet = structure.names.findIndex(
      (_, f) => (mask >> f) % 2 === 0 && ((structure.ancestors[f] ?? 0) & ~mask) === 0,
    );
    const wider = mask | (1 << facet);
    const dimensions = facetsOf(wider).map((f) => table.sizes[f] ?? 1);
    const sums = sumOverAxis(sumsOf(wider), dimensions, facetsOf(wider).indexOf(facet));
    memo.set(mask, sums);
    return sums;
  };
  return sumsOf;
}

/** Returns the sums of an array of the given dimensions over one of its axes. */
function sumOverAxis(values: Float64Array, dimensions: readonly number[], axis: number): Float64Array {
  const length = dimensions[axis] ?? 1;
  let inner = 1;
  for (let k = axis + 1; k < dimensions.length; k++) inner *= dimensions[k] ?? 1;
  const outer = values.length / (length * inner);
  const sums = new Float64Array(outer * inner);
  for (let o = 0; o < outer; o++) {
    for (let level = 0; level < length; level++) {
      const from = (o * length + level) * inner;
      for (let j = 0; j < inner; j++) sums[o * inner + j] = (sums[o * inner + j] ?? 0) + (values[from + j] ?? 0);
    }
  }
  return sums;
}
