import { checkBoolean, checkRecord } from "@measurekit/core";
import { checkFacetedScores, type FacetedScore } from "../data/faceted-scores.js";
import { type BalancedScores, balancedScores } from "./balanced-scores.js";
import { describeElements, facetSizes, numberScores } from "./numbering.js";
import { buildStructure, type Effect, effectsOf, facetsOf, namesOf, type Structure } from "./structure.js";
import { type EffectEstimate, unbalancedEstimates } from "./unbalanced.js";

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
  /**
   * Accepts empty cells and unequal numbers of nested elements, and estimates the components by the analogous analysis
   * of variance. An option, so that a missing score in a design meant to be balanced still throws.
   *
   * @defaultValue `false`
   */
  readonly unbalanced?: boolean;
}

/** One effect of a study: its sums of squares, mean square and the estimate of its variance component. */
export interface VarianceComponent {
  /** The effect's own facets, in the design's order: the rater and the item of the effect of ri:p. */
  readonly facets: readonly string[];
  /** The facets the effect is nested in, such as the person of r:p; none for a crossed effect. */
  readonly nestedIn: readonly string[];
  /**
   * The degrees of freedom: in a balanced study, the product of the sizes less 1 of its own facets and the sizes of those
   * it is nested in; in an unbalanced one, the number of the effect's cells less 1 and less those of the effects within it.
   */
  readonly degreesOfFreedom: number;
  /** The sum of squares, the analogous one in an unbalanced study. */
  readonly sumOfSquares: number;
  /** The sum of squares over the degrees of freedom. */
  readonly meanSquare: number;
  /** The analysis of variance estimate of the variance component, which is negative when the mean squares say so. */
  readonly variance: number;
}

/** The variance components of a design, estimated by the analysis of variance. */
export interface GeneralizabilityStudy {
  /** The facet of the objects of measurement. */
  readonly object: string;
  /** The other facets, in the order of the design. */
  readonly facets: readonly string[];
  /** The facets each nested facet is nested in, by facet name; crossed facets are absent. */
  readonly nested: Readonly<Record<string, readonly string[]>>;
  /**
   * The number of elements of each facet, the object's included: for a nested facet, within each element it is nested
   * in, and in an unbalanced study the most in any.
   */
  readonly sampleSizes: Readonly<Record<string, number>>;
  /**
   * Whether every combination of the facets' elements has a score, so that each facet has one number of elements. A
   * decision study of a study that is not balanced needs its sizes given.
   */
  readonly balanced: boolean;
  /**
   * The effects of the design, one component each, ordered by their number of facets and then by the design's order of
   * facets: for p x i, the person, the item, then the person by item effect.
   */
  readonly components: readonly VarianceComponent[];
}

/**
 * Estimates the variance components of a crossed or nested design by the analysis of variance.
 *
 * @remarks
 * By default the design is balanced: every combination of the facets' elements has one score, and a nested facet has
 * the same number of elements in each element it is nested in. `scores` are in long format, one `FacetedScore` for each,
 * with a string element of the object and of every facet in `levels`, other entries of `levels` being ignored. Each
 * facet needs at least 2 elements, so a design of k facets has at least 2^k scores. Sums of squares come from the
 * marginal means of each effect, and the variance components from the expected mean squares of a random-effects model,
 * solved from the highest-order effect down: each mean square is a sum of the variance components of the effects that
 * include its facets, each multiplied by the product of the sizes of the facets it does not include. The highest-order
 * effect is confounded with error. A component is negative when its mean square is below that of the effects it is
 * confounded with; it is reported as estimated, and a decision study treats it as 0.
 *
 * With `unbalanced` the design may have empty cells and unequal numbers of nested elements, up to 6 facets with the
 * object. The components are Henderson's Method 1, the analogous analysis of variance of Brennan (2001, chapters 7 and
 * 8) and the estimator of urGENOVA: each effect's T, the sum over its cells of the squared sum of the cell's scores over
 * their count, has an expectation that is linear in the variance components with coefficients from the counts alone, and
 * the system of one equation for each effect is solved. Data that are balanced give the estimates of the balanced
 * analysis. The sums of squares and mean squares are the analogous ones, and a decision study needs the sizes given.
 * Measures of raters from incomplete designs are the many-facet Rasch model's, of the rater module.
 *
 * @throws `DataError` when the design is not an object with distinct non-empty facet names and a nesting of the facets
 * that is acyclic and leaves the object crossed, `unbalanced` is not a boolean, the scores are not an array of objects
 * with a finite `score` and the elements of the design's facets, or a facet has fewer than 2 elements. Without
 * `unbalanced` also when the design is not balanced, naming the first combination of elements with more than one score
 * or with none. With it when the design has more than 6 facets, a combination has more than one score, or an effect has
 * no degrees of freedom or a component cannot be told apart from the others, naming the effect.
 * @see {@link https://doi.org/10.1007/978-1-4757-3456-0 | Brennan (2001)}
 * @see {@link https://doi.org/10.2307/3001853 | Henderson (1953)}
 */
export function generalizabilityStudy(
  scores: readonly FacetedScore[],
  design: GeneralizabilityDesign,
): GeneralizabilityStudy {
  checkRecord(design, "The design must be an object { object, facets, nested }");
  const { object = "person", facets, nested = {}, unbalanced = false } = design;
  checkBoolean(unbalanced, "The option unbalanced of the design");
  const structure = buildStructure(object, facets, nested, "The option nested of the design");
  checkFacetedScores(scores, structure.names);
  const { estimates, sizes } = unbalanced ? unbalancedStudy(scores, structure) : balancedStudy(scores, structure);
  return {
    object,
    facets: [...facets],
    nested: Object.fromEntries(Object.entries(nested).map(([facet, parents]) => [facet, [...parents]])),
    sampleSizes: Object.fromEntries(structure.names.map((name, facet) => [name, sizes[facet] ?? 0])),
    balanced: sizes.reduce((product, size) => product * size, 1) === scores.length,
    components: estimates.map(({ effect, ...rest }) => ({
      facets: namesOf(structure, effect.own),
      nestedIn: namesOf(structure, effect.nestedIn),
      ...rest,
    })),
  };
}

/** Returns the estimates of a balanced design, and the number of elements of each facet. */
function balancedStudy(
  scores: readonly FacetedScore[],
  structure: Structure,
): { estimates: EffectEstimate[]; sizes: readonly number[] } {
  const table = balancedScores(scores, structure);
  return { estimates: balancedEstimates(structure, table), sizes: table.sizes };
}

/** Returns the estimates of a design with empty cells, and the most elements each facet has. */
function unbalancedStudy(
  scores: readonly FacetedScore[],
  structure: Structure,
): { estimates: EffectEstimate[]; sizes: readonly number[] } {
  const { coordinates, numbering } = numberScores(scores, structure);
  const sizes = facetSizes(numbering, structure);
  const estimates = unbalancedEstimates(
    scores.map(({ score }) => score),
    coordinates,
    sizes,
    structure,
    (cell) => describeElements(cell, structure, numbering),
  );
  return { estimates, sizes };
}

/** Returns the estimates of the effects of the design from its table of scores. */
function balancedEstimates(structure: Structure, table: BalancedScores): EffectEstimate[] {
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
  return rows.map((row, k) => ({ ...row, variance: variances[k] ?? 0 }));
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
