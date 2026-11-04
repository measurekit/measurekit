import { checkChoice } from "@measurekit/core";
import type { CheckedThetaDistribution, IrtItem } from "../irt/internal.js";
import { type ObservedScoreSetup, observedScoreEquivalents, observedScoreSetup } from "./irt-observed-score.js";
import {
  checkedVariance,
  differenceGradient,
  type IrtEquatingErrorOptions,
  parameterTargets,
} from "./irt-standard-errors.js";
import {
  type CalibratedForm,
  type CommonPositions,
  checkForm,
  commonPositions,
  type Link,
  type LinkErrorOptions,
  linker,
  linkOf,
  movedForm,
  nearbyLink,
} from "./link-covariance.js";
import { type ScaleTransformation, transformItem } from "./linking.js";
import { trueScoreThetas } from "./true-score.js";

/** Options of {@link linkedEquatingErrors}. */
export interface LinkedEquatingOptions extends IrtEquatingErrorOptions, LinkErrorOptions {}

const METHODS = ["true-score", "observed-score"];

/**
 * Returns the standard error of each equated score when the two forms were calibrated apart and linked, the error of
 * the link included.
 *
 * @remarks
 * The common items, matched by id, link the `from` form's scale to the `to` form's by the `linking` method; the `from`
 * items move to that scale, and the scores of `from` are equated by {@link trueScoreEquating} or, with `method:
 * "observed-score"`, {@link irtObservedScoreEquating}. An equated score is then a function of the parameters of both
 * forms, a common item having an estimate in each, and its variance is the sum over the two forms of gᵀ Σ g, g its
 * gradient by central differences in the form's parameters and Σ their covariance. For the observed-score method,
 * `populations.from` lies on the `from` form's own scale and moves with the link, as the group's N(0, 1) of its
 * calibration; `populations.to` lies on the `to` scale. Parameters that a covariance lacks are known.
 *
 * @param from - The form to equate, with its items and their covariance.
 * @param to - The form to equate to, calibrated apart.
 * @param options - The equating method, its populations and weight, and the linking method with its quadrature.
 * @returns The standard error at each score of the form to equate, from 0.
 * @throws `DataError` in the cases of {@link linkCovariance} and of the equating.
 * @see {@link https://doi.org/10.1177/01466216010251004 | Ogasawara (2001)}
 * @see {@link https://doi.org/10.18637/jss.v068.i07 | Battauz (2015)}
 */
export function linkedEquatingErrors(
  from: CalibratedForm,
  to: CalibratedForm,
  options: LinkedEquatingOptions = {},
): readonly number[] {
  const link = linker(options);
  const { method = "true-score" } = options;
  checkChoice(method, "The option method", METHODS);
  const source = checkForm(from, "The from form");
  const target = checkForm(to, "The to form");
  const forms = [source, target];
  const common = commonPositions(source, target, "from and to forms");
  const setup = method === "observed-score" ? observedScoreSetup(options) : null;
  const items = forms.map((form) => form.items);
  const base = linkOf(source.items, target.items, common, link);
  const { scores, equate } = equator(base, source.items, target.items, common, setup);
  const variances = new Float64Array(scores);
  forms.forEach((form, f) => {
    const targets = parameterTargets([form.items], [form.indices], form.labels);
    const used = targets.flatMap((found, k) => (found.length > 0 ? [k] : []));
    const gradient = differenceGradient([form.items], targets, used, scores, (found, step) =>
      equate(movedForm(items, f, found, step)),
    );
    for (let x = 0; x < scores; x++) {
      const row = gradient.subarray(x * used.length, (x + 1) * used.length);
      variances[x] = (variances[x] ?? 0) + checkedVariance(row, used, form.matrix, x);
    }
  });
  return Array.from(variances, Math.sqrt);
}

/** The number of equated scores, and the equating of the forms with a parameter moved by a small step. */
interface Equator {
  readonly scores: number;
  equate(moved: readonly (readonly IrtItem[])[]): readonly number[];
}

/**
 * Returns the equating at a moved parameter: the link a Newton step from the base link and, for true scores, the theta
 * of each score a Newton step from its theta in the base equating, so that no step minimizes or solves again.
 */
function equator(
  base: Link,
  fromItems: readonly IrtItem[],
  toItems: readonly IrtItem[],
  common: CommonPositions,
  setup: ObservedScoreSetup | null,
): Equator {
  const linked = (items: readonly IrtItem[], transformation: ScaleTransformation): IrtItem[] =>
    items.map((item) => transformItem(item, transformation));
  if (setup === null) {
    const trueScores = trueScoreThetas(linked(fromItems, base.transformation), toItems);
    return {
      scores: trueScores.equated.length,
      equate: ([from = [], to = []]) => trueScores.nearby(linked(from, nearbyLink(base, from, to, common)), to),
    };
  }
  const observed = (
    transformation: ScaleTransformation,
    from: readonly IrtItem[],
    to: readonly IrtItem[],
  ): readonly number[] =>
    observedScoreEquivalents(linked(from, transformation), to, {
      ...setup,
      from: movedPopulation(setup.from, transformation.slope, transformation.intercept),
    });
  return {
    scores: observed(base.transformation, fromItems, toItems).length,
    equate: ([from = [], to = []]) => observed(nearbyLink(base, from, to, common), from, to),
  };
}

/** Returns a population with each node θ at slope θ + intercept, its weights kept. */
function movedPopulation(
  population: CheckedThetaDistribution,
  slope: number,
  intercept: number,
): CheckedThetaDistribution {
  return { ...population, nodes: population.nodes.map((theta) => slope * theta + intercept) };
}
