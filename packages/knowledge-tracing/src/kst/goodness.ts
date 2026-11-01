import { checkRecord, chiSquareUpperTail, DataError, describe } from "@measurekit/core";
import type { Constraints, Group, RateConstraints } from "./estimation.js";

/**
 * Up to this many items pks counts every one of the 2 ^ items patterns of responses in the degrees of freedom of the fit,
 * as its `zeropad` does; beyond, only the patterns that occur.
 */
const ZERO_PAD_ITEMS = 16;

/** The fit of a model to the learners' response patterns, against the model that gives each pattern its own probability. */
export interface GoodnessOfFit {
  /** G², twice the log-likelihood ratio of the saturated model to the fitted one. */
  readonly g2: number;
  /** The smaller of the patterns of responses and the learners, less the parameters: pks's degrees of freedom. */
  readonly degreesOfFreedom: number;
  /** The probability of a G² at least as large under a chi-square distribution; `null` unless the degrees are above 0. */
  readonly pValue: number | null;
}

/** The result of {@link compareFits}: the test of a restricted model against a more general one. */
export interface FitComparison {
  /** The restricted model's G² less the general one's. */
  readonly g2: number;
  /** The restricted model's degrees of freedom less the general one's. */
  readonly degreesOfFreedom: number;
  /** The probability of a difference at least as large under a chi-square distribution. */
  readonly pValue: number;
}

/** Returns the number of parameters of a BLIM: states less 1, and a rate for each group of free items, per kind. */
export function parameterCount(states: number, constraints: Constraints): number {
  const free = ({ held, group }: RateConstraints): number =>
    new Set(Array.from(group).filter((_, i) => held[i] !== 1)).size;
  return states - 1 + free(constraints.slip) + free(constraints.guess);
}

/**
 * Returns G² of a fit and its degrees of freedom, for groups that are the response patterns of the learners: G² is twice
 * the sum, over patterns, of their count times ln(count / (learners × probability)), written with the log-likelihood.
 */
export function goodnessOfFit(
  groups: readonly Group[],
  logLikelihood: number,
  parameters: number,
  items: number,
): GoodnessOfFit {
  const learners = groups.reduce((sum, group) => sum + group.weight, 0);
  const saturated = groups.reduce((sum, group) => sum + group.weight * Math.log(group.weight / learners), 0);
  // The saturated log-likelihood is the larger, but a fit that reproduces the counts can round just above it.
  const g2 = Math.max(0, 2 * (saturated - logLikelihood));
  const patterns = items <= ZERO_PAD_ITEMS ? 2 ** items - 1 : groups.length;
  const degreesOfFreedom = Math.min(patterns, learners) - parameters;
  return { g2, degreesOfFreedom, pValue: degreesOfFreedom > 0 ? chiSquareUpperTail(g2, degreesOfFreedom) : null };
}

/**
 * Tests a restricted model against a more general one fitted to the same learners: the difference of their G² and of
 * their degrees of freedom, as `anova` of the pks package.
 *
 * @remarks
 * It is `null` when either fit has no `goodnessOfFit`, when the general model has no fewer parameters, or when the
 * restricted model fits better, which a nested model cannot do by maximum likelihood.
 *
 * @param restricted - The fit with fewer parameters, such as fixed or equal rates.
 * @param general - The fit it is nested in.
 * @throws `DataError` when an argument is not a fit with a `goodnessOfFit` of `null` or of numbers.
 */
export function compareFits(
  restricted: { readonly goodnessOfFit: GoodnessOfFit | null },
  general: { readonly goodnessOfFit: GoodnessOfFit | null },
): FitComparison | null {
  const first = goodnessOf(restricted, "The restricted fit");
  const second = goodnessOf(general, "The general fit");
  if (first === null || second === null) return null;
  const g2 = first.g2 - second.g2;
  const degreesOfFreedom = first.degreesOfFreedom - second.degreesOfFreedom;
  if (!(degreesOfFreedom > 0 && g2 >= 0)) return null;
  return { g2, degreesOfFreedom, pValue: chiSquareUpperTail(g2, degreesOfFreedom) };
}

/** Returns the goodness of fit of a fit, checking that it is one. */
function goodnessOf(fit: { readonly goodnessOfFit: GoodnessOfFit | null }, label: string): GoodnessOfFit | null {
  checkRecord(fit, `${label} must be a fit of a KST model`);
  const found = fit.goodnessOfFit;
  if (found === null) return null;
  checkRecord(found, `${label} needs a goodnessOfFit, an object or null`);
  if (!(Number.isFinite(found.g2) && Number.isFinite(found.degreesOfFreedom))) {
    throw new DataError(`${label} needs a goodnessOfFit of numbers, found ${describe(found)}.`);
  }
  return found;
}
