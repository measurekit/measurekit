import {
  checkArray,
  checkBoolean,
  checkFinite,
  checkOptions,
  checkPositiveInteger,
  checkRecord,
  DataError,
  describe,
  isNonEmptyString,
} from "@measurekit/core";
import { buildStructure, closeOver, type Effect, effectsOf, facetsOf, namesOf, type Structure } from "./structure.js";
import type { GeneralizabilityStudy } from "./study.js";

/** Options of {@link decisionStudy}. */
export interface DecisionStudyOptions {
  /**
   * The number of elements of each facet in the decision study, by facet name: positive integers; for a nested facet,
   * within each element it is nested in. A fixed facet's is the number of conditions its universe has. A study that is
   * not balanced has no one size for a facet, so every facet other than the object needs one.
   *
   * @defaultValue the study's sizes
   */
  readonly sampleSizes?: Readonly<Record<string, number>>;
  /**
   * The facets fixed in the universe of generalization: the decision study's conditions of each are all the universe
   * has, so its differences are not error.
   *
   * @defaultValue `[]`
   */
  readonly fixed?: readonly string[];
  /**
   * The facets the decision study nests that the study crossed, by facet name, such as `{ rater: ["person"] }` for
   * raters of each person's own. They are added to the nesting of the study, which a decision study cannot undo.
   *
   * @defaultValue none
   */
  readonly nested?: Readonly<Record<string, readonly string[]>>;
}

/** One effect of a decision study, with the share it contributes to the universe score and error variances. */
export interface DecisionComponent {
  /** The effect's own facets, in the design's order. */
  readonly facets: readonly string[];
  /** The facets the effect is nested in. */
  readonly nestedIn: readonly string[];
  /**
   * The variance component of the effect in the decision study's design: the study's components pooled into it, those
   * below 0 counting as 0.
   */
  readonly variance: number;
  /** The product of the decision study's sizes of the effect's facets other than the object of measurement. */
  readonly divisor: number;
  /** Whether the effect is in the universe score variance: it has the object and only fixed facets besides. */
  readonly universe: boolean;
  /** Whether the effect is in the relative error variance: it has the object and a random facet. */
  readonly relative: boolean;
  /** Whether the effect is in the absolute error variance: it has a random facet, with the object or without. */
  readonly absolute: boolean;
}

/** The variances and coefficients of a decision study. */
export interface DecisionStudy {
  /** The effects of the decision study's design, ordered as the study's. */
  readonly components: readonly DecisionComponent[];
  /** The universe score variance: the object's component and those it forms with fixed facets, over their divisors. */
  readonly universeScoreVariance: number;
  /** The relative error variance σ²(δ), for decisions that rank the objects. */
  readonly relativeErrorVariance: number;
  /** The absolute error variance σ²(Δ), for decisions about a score's level. */
  readonly absoluteErrorVariance: number;
  /** The generalizability coefficient Eρ²: the universe score variance over it plus the relative error variance. */
  readonly generalizability: number;
  /** The dependability coefficient Φ: the universe score variance over it plus the absolute error variance. */
  readonly dependability: number;
}

/**
 * Computes the variances and the generalizability and dependability coefficients of a measurement procedure from a
 * study's variance components.
 *
 * @remarks
 * The decision study averages over the procedure's conditions: `sampleSizes` elements of each facet. Each component of
 * the study joins the effect of the decision study's design that holds its facets and those that effect is nested in,
 * so that nesting a crossed facet pools components: σ²(r:p) = σ²(r) + σ²(pr). A pooled component below 0 counts as 0.
 * The component of an effect with the object is divided by the sizes of its other facets; so is that of an effect
 * without it. Effects with the object and a random facet make the relative error variance, and with those without the
 * object that have a random facet, the absolute error variance. Effects with the object and only fixed facets join the
 * universe score variance, which the object's own component starts; effects of fixed facets alone belong to neither.
 * The coefficients are Eρ² = σ²(τ) / (σ²(τ) + σ²(δ)) and Φ = σ²(τ) / (σ²(τ) + σ²(Δ)).
 *
 * @throws `DataError` when the study is not the result of `generalizabilityStudy`, with its design and one finite
 * component for each effect of it; when an option is invalid, a size is not a positive integer, a facet named is not one
 * of the study's other than the object, or the added nesting is circular or nests the object; when the study is not
 * balanced and `sampleSizes` lacks a facet other than the object; or when the universe score and relative error
 * variances are both 0, so that the coefficients are undefined.
 * @see {@link https://doi.org/10.1007/978-1-4757-3456-0 | Brennan (2001)}
 */
export function decisionStudy(study: GeneralizabilityStudy, options: DecisionStudyOptions = {}): DecisionStudy {
  checkOptions(options);
  const { sampleSizes = {}, fixed = [], nested = {} } = options;
  const { structure, variances } = checkStudy(study);
  const [object = "", ...facets] = structure.names;
  checkRecord(sampleSizes, "The option sampleSizes must be an object of numbers of elements by facet");
  const sizes = new Map(structure.names.map((name, facet) => [facet, study.sampleSizes[name] ?? 1]));
  for (const [name, size] of Object.entries(sampleSizes)) {
    sizes.set(facetIndex(structure, name, "sampleSizes"), checkedSize(size, `The option sampleSizes for "${name}"`));
  }
  const unsized = facets.find((name) => sampleSizes[name] === undefined);
  if (!study.balanced && unsized !== undefined) {
    throw new DataError(
      `The study is not balanced, so its facets have no one number of elements: the option sampleSizes must give one for "${unsized}".`,
    );
  }
  checkArray(fixed, "The option fixed must be an array of facet names");
  const fixedMask = fixed.reduce((mask, name) => mask | (1 << facetIndex(structure, name, "fixed")), 0);
  // The added nesting is checked alone first, so that its errors name the option; merged with the study's it can still
  // be circular.
  buildStructure(object, facets, nested, "The option nested");
  const merged = buildStructure(object, facets, mergeNesting(study.nested, nested), "The option nested");
  const pooled = pool(merged, variances);
  const rows = effectsOf(merged).map((effect) => decisionComponent(merged, effect, pooled, sizes, fixedMask));
  const sum = (pick: (row: DecisionComponent) => boolean): number =>
    rows.reduce((total, row) => (pick(row) ? total + row.variance / row.divisor : total), 0);
  const universeScoreVariance = sum((row) => row.universe);
  const relativeErrorVariance = sum((row) => row.relative);
  const absoluteErrorVariance = sum((row) => row.absolute);
  if (!(universeScoreVariance + relativeErrorVariance > 0)) {
    throw new DataError(
      "The coefficients are undefined: the universe score variance and the relative error variance are both 0.",
    );
  }
  return {
    components: rows,
    universeScoreVariance,
    relativeErrorVariance,
    absoluteErrorVariance,
    generalizability: universeScoreVariance / (universeScoreVariance + relativeErrorVariance),
    dependability: universeScoreVariance / (universeScoreVariance + absoluteErrorVariance),
  };
}

/** Checks a number of elements and returns it. */
function checkedSize(size: unknown, label = "A number of elements in the study"): number {
  checkPositiveInteger(size, label);
  return size;
}

/** Returns the index of a facet of the study other than the object. */
function facetIndex(structure: Structure, name: string, option: string): number {
  const index = structure.names.indexOf(name);
  if (index < 1) {
    throw new DataError(
      `The option ${option} names ${describe(name)}, which is not a facet of the study; its facets are ${structure.names
        .slice(1)
        .map((facet) => `"${facet}"`)
        .join(", ")}.`,
    );
  }
  return index;
}

/** Returns the nesting of the study with the facets the decision study nests added. */
function mergeNesting(
  study: Readonly<Record<string, readonly string[]>>,
  added: Readonly<Record<string, readonly string[]>>,
): Record<string, string[]> {
  const merged = new Map<string, Set<string>>();
  for (const source of [study, added]) {
    for (const [facet, parents] of Object.entries(source)) {
      const known = merged.get(facet) ?? new Set<string>();
      for (const parent of parents) known.add(parent);
      merged.set(facet, known);
    }
  }
  return Object.fromEntries([...merged].map(([facet, parents]) => [facet, [...parents]]));
}

/** Returns the study's components, those below 0 as 0, summed into the effect of the decision study that holds them. */
function pool(structure: Structure, variances: ReadonlyMap<number, number>): Map<number, number> {
  const pooled = new Map<number, number>();
  for (const [mask, variance] of variances) {
    const joined = closeOver(structure, mask);
    pooled.set(joined, (pooled.get(joined) ?? 0) + Math.max(0, variance));
  }
  return pooled;
}

/** Returns an effect of the decision study with its variance, divisor and the variances it belongs to. */
function decisionComponent(
  structure: Structure,
  effect: Effect,
  pooled: ReadonlyMap<number, number>,
  sizes: ReadonlyMap<number, number>,
  fixedMask: number,
): DecisionComponent {
  const others = facetsOf(effect.mask).filter((facet) => facet !== 0);
  const divisor = others.reduce((product, facet) => product * (sizes.get(facet) ?? 1), 1);
  const random = others.some((facet) => ((fixedMask >> facet) & 1) === 0);
  const withObject = (effect.mask & 1) === 1;
  return {
    facets: namesOf(structure, effect.own),
    nestedIn: namesOf(structure, effect.nestedIn),
    variance: pooled.get(effect.mask) ?? 0,
    divisor,
    universe: withObject && !random,
    relative: withObject && random,
    absolute: random,
  };
}

/**
 * Checks a study, which may come from storage, and returns the structure of its design and its variance components by
 * the mask of the effect each belongs to.
 *
 * @throws `DataError` when it is not a study of a valid design with a size for each facet and one finite component for
 * each effect.
 */
function checkStudy(study: GeneralizabilityStudy): { structure: Structure; variances: Map<number, number> } {
  checkRecord(study, "The study must be an object from generalizabilityStudy");
  checkRecord(study.sampleSizes, "The sampleSizes of the study must be an object of numbers by facet");
  checkArray(study.facets, "The facets of the study must be an array of facet names");
  checkArray(study.components, "The components of the study must be an array");
  checkBoolean(study.balanced, "The field balanced of the study");
  const structure = buildStructure(study.object, study.facets, study.nested, "The nested facets of the study");
  structure.names.forEach((name) => {
    checkPositiveInteger(study.sampleSizes[name], `The size of "${name}" in the study`);
  });
  const effects = new Map(effectsOf(structure).map((effect) => [`${effect.own}:${effect.nestedIn}`, effect.mask]));
  const variances = new Map<number, number>();
  study.components.forEach((component, index) => {
    const label = `Component ${index + 1} of the study`;
    checkRecord(component, `${label} must be an object`);
    checkArray(component.facets, `${label} needs facets, an array of facet names`);
    checkArray(component.nestedIn, `${label} needs nestedIn, an array of facet names`);
    checkFinite(component.variance, `The variance of component ${index + 1} of the study`);
    const mask = effects.get(
      `${maskOf(structure, component.facets, label)}:${maskOf(structure, component.nestedIn, label)}`,
    );
    if (mask === undefined || variances.has(mask)) {
      throw new DataError(
        `${label} does not name a distinct effect of the study's design; the effects are those of the study from generalizabilityStudy.`,
      );
    }
    variances.set(mask, component.variance);
  });
  if (variances.size !== effects.size) {
    throw new DataError(
      `The study needs one component for each of the ${effects.size} effects of its design, found ${study.components.length}.`,
    );
  }
  return { structure, variances };
}

/** Returns the mask of the named facets of a component. */
function maskOf(structure: Structure, names: readonly string[], label: string): number {
  return names.reduce((mask, name) => {
    const index = structure.names.indexOf(name);
    if (!isNonEmptyString(name) || index < 0) {
      throw new DataError(`${label} names ${describe(name)}, which is not a facet of the study.`);
    }
    return mask | (1 << index);
  }, 0);
}
