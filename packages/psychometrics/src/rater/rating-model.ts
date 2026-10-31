import { checkArray, checkRecord, DataError, describe, isNonEmptyString } from "@measurekit/core";
import type { FacetedScore } from "../data/faceted-scores.js";
import type { ThetaPrior } from "../irt/ability.js";
import type { IrtItem } from "../irt/model.js";
import { partialCredit, partialCreditProbabilities } from "../irt/partial-credit.js";
import type { ThresholdParameters } from "../irt/thresholds.js";

/** What the functions that use an estimated many-facet model read of it. */
export interface RatingModelData {
  /** The facet that names the persons. */
  readonly person: string;
  /** The facet whose elements each have thresholds, or `null` for one shared scale. */
  readonly thresholdFacet: string | null;
  /** The facets other than the person facet, with the measure of each element. */
  readonly facets: readonly {
    readonly facet: string;
    readonly elements: readonly { readonly id: string; readonly measure: number }[];
  }[];
  /** The thresholds of each set. */
  readonly thresholds: readonly { readonly element: string | null; readonly thresholds: readonly number[] }[];
  /** The distribution of theta. */
  readonly population: ThetaPrior;
}

/** A model with its elements looked up by id. */
export interface RatingStructure {
  /** The facet names. */
  readonly facets: readonly string[];
  /** The measure of each element, by facet. */
  readonly measures: readonly ReadonlyMap<string, number>[];
  /** The index of the threshold facet in `facets`, -1 for the shared scale. */
  readonly thresholdFacet: number;
  /** The thresholds of each set, by the id of the element that owns it, `null` for the shared set. */
  readonly sets: ReadonlyMap<string | null, readonly number[]>;
}

/**
 * Checks that a value is an estimated many-facet model: facets with elements and measures, thresholds for the shared
 * scale or for every element of the threshold facet, and a normal distribution of theta.
 *
 * @throws `DataError` naming the first part that is missing or invalid.
 */
export function checkManyFacetRasch(model: RatingModelData): void {
  checkRecord(model, "The model must be an estimated many-facet Rasch model");
  if (!isNonEmptyString(model.person)) {
    throw new DataError(`The model needs a person facet name, found ${describe(model.person)}.`);
  }
  checkArray(model.facets, "The model needs facets, an array of { facet, elements }");
  model.facets.forEach(checkFacet);
  const { thresholdFacet } = model;
  if (thresholdFacet !== null && !model.facets.some(({ facet }) => facet === thresholdFacet)) {
    throw new DataError(
      `The model's thresholdFacet must be null or a facet of the model, found ${describe(thresholdFacet)}.`,
    );
  }
  checkArray(model.thresholds, "The model needs thresholds, an array of { element, thresholds }");
  model.thresholds.forEach(checkSet);
  checkRecord(model.population, "The model needs a population { mean, sd }");
  const { mean, sd } = model.population;
  if (!(Number.isFinite(mean) && Number.isFinite(sd) && sd > 0)) {
    throw new DataError("The model's population needs a finite mean and a positive finite sd.");
  }
  checkSetsMatch(model);
}

function checkFacet(facet: RatingModelData["facets"][number], index: number): void {
  const label = `Facet ${index + 1} of the model`;
  checkRecord(facet, `${label} must be an object { facet, elements }`);
  if (!isNonEmptyString(facet.facet)) throw new DataError(`${label} needs a name, found ${describe(facet.facet)}.`);
  checkArray(facet.elements, `${label} needs elements, an array of { id, measure }`);
  facet.elements.forEach((element, e) => {
    checkRecord(element, `Element ${e + 1} of ${label} must be an object { id, measure }`);
    if (typeof element.id !== "string" || !Number.isFinite(element.measure)) {
      throw new DataError(`Element ${e + 1} of ${label} needs a string id and a finite measure.`);
    }
  });
}

function checkSet(set: RatingModelData["thresholds"][number], index: number): void {
  const label = `Thresholds ${index + 1} of the model`;
  checkRecord(set, `${label} must be an object { element, thresholds }`);
  checkArray(set.thresholds, `${label} needs thresholds, an array of numbers`);
  if (!(set.thresholds.length > 0 && set.thresholds.every((value) => Number.isFinite(value)))) {
    throw new DataError(`${label} needs at least one finite threshold.`);
  }
  if (set.element !== null && typeof set.element !== "string") {
    throw new DataError(`${label} needs an element that is a string, or null for the shared scale.`);
  }
}

/** Throws unless the model has one shared set, or a set for each element of its threshold facet. */
function checkSetsMatch(model: RatingModelData): void {
  const owner = model.facets.find(({ facet }) => facet === model.thresholdFacet);
  const expected = owner === undefined ? [null] : owner.elements.map(({ id }) => id);
  const found = model.thresholds.map(({ element }) => element);
  if (expected.length !== found.length || !expected.every((id) => found.includes(id))) {
    throw new DataError(
      owner === undefined
        ? "The model needs one set of thresholds, with the element null, for the shared scale."
        : `The model needs a set of thresholds for each element of the facet "${owner.facet}".`,
    );
  }
}

/** Looks up the elements of a checked model. */
export function ratingStructure(model: RatingModelData): RatingStructure {
  const facets = model.facets.map(({ facet }) => facet);
  return {
    facets,
    measures: model.facets.map(({ elements }) => new Map(elements.map(({ id, measure }) => [id, measure]))),
    thresholdFacet: model.thresholdFacet === null ? -1 : facets.indexOf(model.thresholdFacet),
    sets: new Map(model.thresholds.map(({ element, thresholds }) => [element, thresholds])),
  };
}

/** Returns the measure of an element of the facet at index `facet`, or throws when the model has no such element. */
function measureOf(structure: RatingStructure, facet: number, id: string): number {
  const measure = structure.measures[facet]?.get(id);
  if (measure === undefined) {
    throw new DataError(`The facet "${structure.facets[facet]}" of the model has no element ${describe(id)}.`);
  }
  return measure;
}

/** Returns the thresholds owned by an element of the threshold facet, or the shared ones for `null`. */
function setOf(structure: RatingStructure, id: string | null): readonly number[] {
  const thresholds = structure.sets.get(id);
  if (thresholds === undefined) {
    throw new DataError(
      `The ${structure.facets[structure.thresholdFacet]} ${describe(id)} has no thresholds in the model.`,
    );
  }
  return thresholds;
}

/**
 * Returns the item that a combination of elements, one for each facet of the model in order, is for a person: a
 * partial credit item with a = 1 whose steps are the sum of the measures plus the thresholds of its set.
 *
 * @throws `DataError` when the model lacks an element.
 */
function ratingItem(structure: RatingStructure, elements: readonly string[]): IrtItem<ThresholdParameters> {
  const location = elements.reduce((sum, id, f) => sum + measureOf(structure, f, id), 0);
  const owner = structure.thresholdFacet < 0 ? null : (elements[structure.thresholdFacet] ?? "");
  return {
    model: partialCredit,
    parameters: { a: 1, b: setOf(structure, owner).map((threshold) => location + threshold) },
  };
}

/**
 * Returns the expected rating of a person at `theta` on a combination of elements, one for each facet in order. A
 * `null` stands for the facet's mean, a measure of 0; with the threshold facet at `null`, the expected ratings under the
 * thresholds of each of its elements are averaged.
 *
 * @throws `DataError` when the model lacks an element.
 */
export function expectedRatingAt(
  structure: RatingStructure,
  theta: number,
  elements: readonly (string | null)[],
): number {
  const location = elements.reduce((sum: number, id, f) => sum + (id === null ? 0 : measureOf(structure, f, id)), 0);
  const owner = structure.thresholdFacet < 0 ? null : (elements[structure.thresholdFacet] ?? null);
  const sets =
    structure.thresholdFacet >= 0 && owner === null ? [...structure.sets.values()] : [setOf(structure, owner)];
  const totals = sets.map((thresholds) => {
    const probabilities = new Float64Array(thresholds.length + 1);
    partialCreditProbabilities(1, location, thresholds, theta, probabilities);
    return probabilities.reduce((sum, probability, k) => sum + k * probability, 0);
  });
  return totals.reduce((sum, value) => sum + value, 0) / totals.length;
}

/**
 * Returns the item each score was given on, aligned with the scores: for the elements of the score in the model's
 * facets, in order. Items of equal combinations are shared.
 *
 * @throws `DataError` when the model lacks an element of a score, or a score is not a category of its item.
 */
export function scoredItems(
  scores: readonly FacetedScore[],
  structure: RatingStructure,
): IrtItem<ThresholdParameters>[] {
  const cache = new Map<string, IrtItem<ThresholdParameters>>();
  return scores.map(({ levels, score }, index) => {
    const elements = structure.facets.map((facet) => levels[facet] ?? "");
    const key = JSON.stringify(elements);
    let item = cache.get(key);
    if (item === undefined) {
      item = ratingItem(structure, elements);
      cache.set(key, item);
    }
    const categories = item.parameters.b.length + 1;
    if (!(Number.isSafeInteger(score) && score >= 0 && score < categories)) {
      throw new DataError(
        `Score ${index + 1} is ${describe(score)}, but its thresholds give categories 0 to ${categories - 1}.`,
      );
    }
    return item;
  });
}
