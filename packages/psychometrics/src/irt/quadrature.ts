import { checkChoice, checkIntegerFrom, checkRecord } from "@measurekit/core";
import { gaussHermite, type Quadrature, rectangular } from "../math/quadrature.js";

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
   * The number of nodes, at least 2.
   *
   * @defaultValue `61`
   */
  readonly points?: number;
}

const METHODS: readonly QuadratureMethod[] = ["rectangular", "gauss-hermite"];

/** The standard rules computed so far, by method and number of points: EAP estimation asks for one per person. */
const STANDARD_RULES = new Map<string, Quadrature>();

/**
 * Returns the rule of `options` for the normal distribution N(mean, sd²): nodes of the standard rule scaled to it,
 * with weights summing to 1.
 *
 * @throws `DataError` when the options are not an object, the method is unknown or the number of points is not an
 * integer from 2.
 */
export function quadratureRule(options: QuadratureOptions = {}, mean = 0, sd = 1): Quadrature {
  checkRecord(options, "The option quadrature must be an object { method, points }");
  const { method = "rectangular", points = 61 } = options;
  checkIntegerFrom(points, "The quadrature points", 2);
  checkChoice(method, "The quadrature method", METHODS);
  const key = `${method} ${points}`;
  let rule = STANDARD_RULES.get(key);
  if (rule === undefined) {
    rule = method === "rectangular" ? rectangular(points) : gaussHermite(points);
    STANDARD_RULES.set(key, rule);
  }
  return { nodes: rule.nodes.map((z) => mean + sd * z), weights: rule.weights.slice() };
}
