import {
  checkArray,
  checkChoice,
  checkOptions,
  checkStopping,
  DataError,
  describe,
  isNonEmptyString,
} from "@measurekit/core";
import type { FacetedScore } from "../data/faceted-scores.js";
import type { ThetaPrior } from "../irt/ability.js";
import { fitLinearRasch } from "../irt/linear-rasch.js";
import { type QuadratureOptions, quadratureRule } from "../irt/quadrature.js";
import { facetDesign } from "./facet-design.js";
import type { FacetSummary } from "./facet-summary.js";
import { manyFacetParts, type RatingThresholds } from "./many-facet-result.js";

/** How the standard errors of a many-facet model are computed. */
export type FacetStandardErrors = "none" | "hessian";

/** Options of {@link manyFacetRasch}. */
export interface ManyFacetRaschOptions {
  /**
   * The facet that names the persons.
   *
   * @defaultValue `"person"`
   */
  readonly person?: string;
  /**
   * The other facets, such as `["item", "rater"]`: each score has an element of each. Facets beyond the first, such as
   * task or occasion, add terms to the model. At least one, none repeated, none the person facet.
   */
  readonly facets: readonly string[];
  /**
   * A facet whose elements each get their own thresholds, as the partial credit model: one of `facets`.
   *
   * @defaultValue one rating scale shared by all elements
   */
  readonly thresholdFacet?: string;
  /**
   * The quadrature rule over theta.
   *
   * @defaultValue 61 rectangular points
   */
  readonly quadrature?: QuadratureOptions;
  /**
   * `"hessian"` computes the standard errors from the Hessian of the marginal log-likelihood; `"none"` leaves them
   * out.
   *
   * @defaultValue `"hessian"`
   */
  readonly standardErrors?: FacetStandardErrors;
  /**
   * Estimation has converged when no parameter changes by more than this in an EM iteration.
   *
   * @defaultValue `1e-4`
   */
  readonly tolerance?: number;
  /**
   * The most EM iterations to run.
   *
   * @defaultValue `500`
   */
  readonly maxIterations?: number;
}

/** A many-facet Rasch model estimated by marginal maximum likelihood. */
export interface ManyFacetRasch {
  /** The facet that names the persons. */
  readonly person: string;
  /** The facet whose elements each have thresholds; `null` for one rating scale shared by all. */
  readonly thresholdFacet: string | null;
  /** The facets other than the person facet, in the order of the option, with their elements. */
  readonly facets: readonly FacetSummary[];
  /** The thresholds of each set: one for the shared scale, or one for each element of the threshold facet. */
  readonly thresholds: readonly RatingThresholds[];
  /** The mean and the standard deviation of the normal distribution of theta. */
  readonly population: ThetaPrior;
  /** The marginal log-likelihood at the estimates. */
  readonly logLikelihood: number;
  /** Akaike's information criterion: -2 logLikelihood + 2p, with p the estimated parameters, μ and σ included. */
  readonly aic: number;
  /** The Bayesian information criterion: -2 logLikelihood + p ln n, with n the number of persons. */
  readonly bic: number;
  /** Whether EM reached `tolerance` within `maxIterations`. */
  readonly converged: boolean;
  /** EM iterations run. */
  readonly iterations: number;
}

const DEFAULT_TOLERANCE = 1e-4;
const DEFAULT_MAX_ITERATIONS = 500;
const STANDARD_ERRORS: readonly FacetStandardErrors[] = ["none", "hessian"];

/**
 * Estimates the many-facet Rasch model of faceted scores by marginal maximum likelihood: persons, items, raters and
 * further facets on one logit scale.
 *
 * @remarks
 * The model gives a score in category k rather than k - 1 the log odds θ - Σ_f δ_(f, element) - τ_(s, k), with θ the
 * person's ability, δ the measure of the score's element in each facet f and τ the thresholds of the set s: all scores
 * share one set, or the set of the score's element of `thresholdFacet`. Theta is normal with a mean and a standard
 * deviation that are estimated. Each facet's measures and each set's thresholds sum to 0, so that positive measures are
 * harder items and more severe raters. Each distinct combination of the elements of the other facets is a virtual item
 * that every person it rated answered, and a person's scores are independent given theta. Estimation is the EM
 * algorithm of `calibrate`, with Newton's method for the measures, thresholds, mean and standard deviation together.
 *
 * A person may have several scores for one combination of elements, which are independent observations. The scores
 * must connect all persons and elements: each score links its person to its elements, and several subsets that share
 * none cannot be put on one scale. Each set's scores must use the categories from 0 to its highest, at least two.
 *
 * @throws `DataError` when the options are invalid, the scores are not objects { levels, score } with an element of
 * each facet and an integer score from 0, fewer than two persons were scored, the design is not connected, or a
 * threshold set uses fewer than two categories or skips one.
 * @see {@link https://doi.org/10.1007/BF02293801 | Bock and Aitkin (1981)}
 */
export function manyFacetRasch(scores: readonly FacetedScore[], options: ManyFacetRaschOptions): ManyFacetRasch {
  checkOptions(options);
  const {
    person = "person",
    facets,
    thresholdFacet,
    quadrature = {},
    standardErrors = "hessian",
    tolerance = DEFAULT_TOLERANCE,
    maxIterations = DEFAULT_MAX_ITERATIONS,
  } = options;
  checkFacetNames(person, facets);
  if (thresholdFacet !== undefined) checkChoice(thresholdFacet, "The option thresholdFacet", facets);
  checkChoice(standardErrors, "The option standardErrors", STANDARD_ERRORS);
  checkStopping(tolerance, maxIterations);
  const rule = quadratureRule(quadrature);
  const design = facetDesign(scores, person, facets, thresholdFacet);
  const fit = fitLinearRasch(design.model, {
    rule,
    tolerance,
    maxIterations,
    standardErrors: standardErrors === "hessian",
  });
  const persons = design.personIds.length;
  return {
    person,
    thresholdFacet: thresholdFacet ?? null,
    ...manyFacetParts(design, fit, person, thresholdFacet ?? null),
    logLikelihood: fit.logLikelihood,
    aic: -2 * fit.logLikelihood + 2 * fit.layout.size,
    bic: -2 * fit.logLikelihood + fit.layout.size * Math.log(persons),
    converged: fit.converged,
    iterations: fit.iterations,
  };
}

function checkFacetNames(person: string, facets: readonly string[]): void {
  if (!isNonEmptyString(person)) {
    throw new DataError(`The option person must be a non-empty string, found ${describe(person)}.`);
  }
  checkArray(facets, 'The option facets must be an array of facet names, such as ["item", "rater"]');
  if (facets.length === 0) throw new DataError("The option facets needs at least one facet.");
  facets.forEach((facet, index) => {
    if (!isNonEmptyString(facet)) {
      throw new DataError(
        `Facet ${index + 1} of the option facets must be a non-empty string, found ${describe(facet)}.`,
      );
    }
  });
  const names = [person, ...facets];
  const repeated = names.find((name, index) => names.indexOf(name) !== index);
  if (repeated !== undefined) {
    throw new DataError(`The person facet and the option facets need different names, found "${repeated}" twice.`);
  }
}
