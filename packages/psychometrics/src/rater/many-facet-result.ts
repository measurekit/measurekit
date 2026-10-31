import type { ThetaPrior } from "../irt/ability.js";
import type { LinearRaschFit } from "../irt/linear-rasch.js";
import type { DesignFacet, FacetDesign } from "./facet-design.js";
import { type FacetElement, type FacetSummary, summarizeFacet } from "./facet-summary.js";
import { expectedRatingAt, ratingStructure } from "./rating-model.js";

/** The thresholds of one set of a many-facet model. */
export interface RatingThresholds {
  /** The element of the threshold facet that owns the thresholds; `null` for the scale all elements share. */
  readonly element: string | null;
  /** The thresholds τ_1 to τ_(K-1) between neighboring categories, in logits. They sum to 0. */
  readonly thresholds: readonly number[];
  /**
   * Their standard errors, 0 for a threshold the constraint fixes, as the one threshold of a two-category scale;
   * `null` without standard errors.
   */
  readonly standardErrors: readonly number[] | null;
}

/** Returns the variance of Σ coefficients[a] * parameter[indices[a]] under the covariance matrix `covariance`. */
function contrastVariance(
  covariance: Float64Array,
  size: number,
  indices: readonly number[],
  coefficients: readonly number[],
): number {
  let variance = 0;
  indices.forEach((i, a) => {
    indices.forEach((j, b) => {
      variance += (coefficients[a] ?? 0) * (coefficients[b] ?? 0) * (covariance[i * size + j] ?? 0);
    });
  });
  return variance;
}

/**
 * Returns the measures of a facet's elements from its free parameters, the last being minus the sum of the others, and
 * their standard errors by the delta method; the errors are `null` without a covariance matrix.
 */
function facetMeasures(
  parameters: Float64Array,
  covariance: Float64Array | null,
  size: number,
  facet: DesignFacet,
): { measures: number[]; errors: (number | null)[] } {
  const free = Math.max(facet.ids.length - 1, 0);
  const indices = Array.from({ length: free }, (_, a) => facet.offset + a);
  const measures = indices.map((index) => parameters[index] ?? 0);
  measures.push(0 - measures.reduce((sum, measure) => sum + measure, 0));
  const errors = measures.map((_, e) => {
    if (covariance === null) return null;
    const own = e < free;
    const coefficients = own ? [1] : indices.map(() => -1);
    return Math.sqrt(contrastVariance(covariance, size, own ? [facet.offset + e] : indices, coefficients));
  });
  return { measures, errors };
}

/** Returns the thresholds of set `set` with their standard errors by the delta method. */
function setThresholds(fit: LinearRaschFit, set: number, element: string | null): RatingThresholds {
  const { layout, covariance } = fit;
  const thresholds = Array.from(fit.thresholds[set] ?? []);
  const start = layout.thresholdStarts[set] ?? 0;
  const free = thresholds.length - 1;
  const indices = Array.from({ length: free }, (_, a) => start + a);
  const standardErrors =
    covariance === null
      ? null
      : thresholds.map((_, j) => {
          const own = j < free;
          const coefficients = own ? [1] : indices.map(() => -1);
          return Math.sqrt(contrastVariance(covariance, layout.size, own ? [start + j] : indices, coefficients));
        });
  return { element, thresholds, standardErrors };
}

/** The facets, thresholds and population of an estimated many-facet model. */
export interface ManyFacetEstimates {
  readonly facets: readonly FacetSummary[];
  readonly thresholds: readonly RatingThresholds[];
  readonly population: ThetaPrior;
}

/**
 * Returns the facet summaries, thresholds and population of an estimated design: measures and thresholds from the
 * engine's parameters, standard errors by the delta method, and fair averages from the model they make.
 */
export function manyFacetParts(
  design: FacetDesign,
  fit: LinearRaschFit,
  person: string,
  thresholdFacet: string | null,
): ManyFacetEstimates {
  const population: ThetaPrior = { mean: fit.mean, sd: fit.sd };
  const measured = design.facets.map((facet) => facetMeasures(fit.parameters, fit.covariance, fit.layout.size, facet));
  const thresholds = design.sets.map((element, set) => setThresholds(fit, set, element));
  const structure = ratingStructure({
    person,
    thresholdFacet,
    facets: design.facets.map(({ name, ids }, f) => ({
      facet: name,
      elements: ids.map((id, e) => ({ id, measure: measured[f]?.measures[e] ?? 0 })),
    })),
    thresholds,
    population,
  });
  const facets = design.facets.map((facet, f) => {
    const elements: FacetElement[] = facet.ids.map((id, e) => {
      const levels = design.facets.map((_, other) => (other === f ? id : null));
      const count = facet.counts[e] ?? 0;
      return {
        id,
        measure: measured[f]?.measures[e] ?? 0,
        standardError: measured[f]?.errors[e] ?? null,
        count,
        observedAverage: (facet.sums[e] ?? 0) / count,
        fairAverage: expectedRatingAt(structure, population.mean, levels),
      };
    });
    return summarizeFacet(facet.name, elements);
  });
  return { facets, thresholds, population };
}
