import { checkChoice, checkFields, checkIntegerFrom, checkRecord, DataError, type FieldNames } from "@measurekit/core";
import { gaussHermite, type Quadrature, rectangular } from "../math/internal.js";

/** How to place quadrature nodes over theta. */
export type QuadratureMethod = "rectangular" | "gauss-hermite";

/** The quadrature rule that integrates over theta, in calibration and EAP estimation. */
export interface QuadratureOptions {
  /**
   * `rectangular`: equally spaced nodes from -6 to 6 standard deviations, weighted by the normal density, as mirt;
   * `gauss-hermite`: the Gauss-Hermite rule for the normal density.
   *
   * @defaultValue `"rectangular"`
   */
  readonly method?: QuadratureMethod;
  /**
   * The number of nodes, from 2 to 1,000.
   *
   * @defaultValue `61`
   */
  readonly points?: number;
}

const QUADRATURE_OPTIONS: FieldNames<QuadratureOptions> = { method: true, points: true };

const METHODS: readonly QuadratureMethod[] = ["rectangular", "gauss-hermite"];

/**
 * The most nodes of a rule: a spacing of 0.012 between -6 and 6, far finer than any integral over theta needs, and a size
 * that keeps an option from allocating gigabytes for its nodes and for every item's table over them.
 */
const MAX_POINTS = 1_000;

/**
 * The Gauss-Hermite rule asked for last: it takes a third of a millisecond at 61 nodes, and EAP estimation asks for one per
 * person. A rectangular rule takes a microsecond.
 */
let lastGaussHermite: Quadrature | undefined;

function gaussHermiteRule(points: number): Quadrature {
  if (lastGaussHermite?.nodes.length !== points) lastGaussHermite = gaussHermite(points);
  return lastGaussHermite;
}

/**
 * Returns the rule of `options` for the normal distribution N(mean, sd²): nodes of the standard rule scaled to it,
 * with weights summing to 1.
 *
 * @throws `DataError` when the options are not an object, the method is unknown or the number of points is not an
 * integer from 2 to 1,000.
 */
export function quadratureRule(options: QuadratureOptions = {}, mean = 0, sd = 1): Quadrature {
  checkRecord(options, "The option quadrature must be an object { method, points }");
  checkFields(options, QUADRATURE_OPTIONS, "The option quadrature");
  const { method = "rectangular", points = 61 } = options;
  checkIntegerFrom(points, "The quadrature points", 2);
  if (points > MAX_POINTS) {
    throw new DataError(`The quadrature points must be an integer from 2 to ${MAX_POINTS}, found ${points}.`);
  }
  checkChoice(method, "The quadrature method", METHODS);
  const rule = method === "rectangular" ? rectangular(points) : gaussHermiteRule(points);
  return { nodes: rule.nodes.map((z) => mean + sd * z), weights: rule.weights.slice() };
}
