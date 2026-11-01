import { bfgs, checkOptions, type OptimizerOptions } from "@measurekit/core";
import { type IrtItem, largestCategoryCount, type QuadratureOptions, quadratureRule } from "../irt/internal.js";

import type { Quadrature } from "../math/quadrature.js";
import { checkCommonItems, type ScaleTransformation } from "./linking.js";

/** A scale transformation that minimizes a characteristic curve criterion, with the minimization's convergence. */
export interface CharacteristicCurveLinking extends ScaleTransformation {
  /**
   * Whether the minimization converged: the gradient reached its tolerance, or rounding of the criterion left no step
   * that makes progress, at the minimum to the criterion's precision.
   */
  readonly converged: boolean;
  /** Iterations of the minimization. */
  readonly iterations: number;
}

/** Options of {@link haebaraLinking} and {@link stockingLordLinking}. */
export interface LinkingOptions {
  /**
   * The quadrature rule over theta ~ N(0, 1) on the `to` scale.
   *
   * @defaultValue 61 rectangular points
   */
  readonly quadrature?: QuadratureOptions;
}

/** BFGS on ln A and B, from A = 1 and B = 0. */
const OPTIMIZER: OptimizerOptions = { tolerance: 1e-10, maxIterations: 200 };

/**
 * Returns the transformation that minimizes Haebara's criterion.
 *
 * @remarks
 * The criterion is the sum over the quadrature nodes θ of the `to` scale, weighted by their N(0, 1) weights, of the
 * squared differences between each common item's category probabilities on the `to` scale and on the `from` scale after
 * the transformation. A dichotomous item adds both of its categories, twice the usual term, which leaves the minimum of
 * a dichotomous test unchanged; polytomous items follow Kim and Lee. Items of any model, a custom one included.
 *
 * @param from - The common items on the scale to move.
 * @param to - The same items on the scale to move to.
 * @param options - The quadrature.
 * @returns The slope and intercept of the transformation, whether the minimization converged and the iterations run.
 * @throws `DataError` when the item lists differ in length or are empty, a pair has different models or numbers of
 * categories, an item is invalid, or the quadrature is invalid.
 * @see {@link https://doi.org/10.4992/psycholres1954.22.144 | Haebara (1980)}
 * @see {@link https://doi.org/10.1111/j.1745-3984.2006.00004.x | Kim and Lee (2006)}
 */
export function haebaraLinking(
  from: readonly IrtItem[],
  to: readonly IrtItem[],
  options: LinkingOptions = {},
): CharacteristicCurveLinking {
  return minimize(from, to, options, "haebara");
}

/**
 * Returns the transformation that minimizes the Stocking-Lord criterion.
 *
 * @remarks
 * The criterion is the sum over the quadrature nodes θ of the `to` scale, weighted by their N(0, 1) weights, of the
 * squared difference between the common items' test characteristic curves, their total expected score, on the `to`
 * scale and on the `from` scale after the transformation. Items of any model, a custom one included.
 *
 * @param from - The common items on the scale to move.
 * @param to - The same items on the scale to move to.
 * @param options - The quadrature.
 * @returns The slope and intercept of the transformation, whether the minimization converged and the iterations run.
 * @throws `DataError` in the cases of {@link haebaraLinking}.
 * @see {@link https://doi.org/10.1177/014662168300700208 | Stocking and Lord (1983)}
 */
export function stockingLordLinking(
  from: readonly IrtItem[],
  to: readonly IrtItem[],
  options: LinkingOptions = {},
): CharacteristicCurveLinking {
  return minimize(from, to, options, "stocking-lord");
}

function minimize(
  from: readonly IrtItem[],
  to: readonly IrtItem[],
  options: LinkingOptions,
  criterion: "haebara" | "stocking-lord",
): CharacteristicCurveLinking {
  checkCommonItems(from, to);
  checkOptions(options);
  const curves = characteristicCurves(from, to, quadratureRule(options.quadrature));
  const term = criterion === "haebara" ? haebaraTerm : stockingLordTerm;
  const result = bfgs(
    (x, gradient) => {
      const slope = Math.exp(x[0] ?? 0);
      const intercept = x[1] ?? 0;
      gradient.fill(0);
      let value = 0;
      for (let q = 0; q < curves.nodes.length; q++) value += term(curves, q, slope, intercept, gradient);
      return value;
    },
    [0, 0],
    OPTIMIZER,
  );
  return {
    slope: Math.exp(result.x[0] ?? 0),
    intercept: result.x[1] ?? 0,
    converged: result.converged,
    iterations: result.iterations,
  };
}

/** The common items with their probabilities on the `to` scale at the nodes, and scratch space for the `from` items. */
interface CharacteristicCurves extends Quadrature {
  readonly from: readonly IrtItem[];
  /** Each item's number of categories. */
  readonly counts: readonly number[];
  /** Each `to` item's probabilities, by node, then category. */
  readonly targets: readonly Float64Array[];
  readonly probabilities: Float64Array;
  readonly first: Float64Array;
  readonly second: Float64Array;
}

function characteristicCurves(
  from: readonly IrtItem[],
  to: readonly IrtItem[],
  rule: Quadrature,
): CharacteristicCurves {
  const counts = to.map((item) => item.model.categoryCount(item.parameters));
  const targets = to.map((item, j) => {
    const count = counts[j] ?? 0;
    const table = new Float64Array(rule.nodes.length * count);
    for (const [q, theta] of rule.nodes.entries()) {
      item.model.probabilities(item.parameters, theta, table.subarray(q * count));
    }
    return table;
  });
  const size = largestCategoryCount(from);
  return {
    ...rule,
    from,
    counts,
    targets,
    probabilities: new Float64Array(size),
    first: new Float64Array(size),
    second: new Float64Array(size),
  };
}

/**
 * Writes item j's `from` probabilities at u and their theta derivatives into the scratch space and returns its `to`
 * probabilities, or `undefined` for a missing item.
 */
function fromItemAt(curves: CharacteristicCurves, j: number, u: number): Float64Array | undefined {
  const item = curves.from[j];
  if (item === undefined) return undefined;
  item.model.probabilities(item.parameters, u, curves.probabilities);
  item.model.thetaDerivatives(item.parameters, u, curves.first, curves.second);
  return curves.targets[j];
}

// Each term returns its node's term of the criterion and adds its gradient with respect to ln A and B. With
// u = (θ - B) / A, a `from` probability P(u) has derivatives -P'(u) u with respect to ln A and -P'(u) / A with respect
// to B, so a difference D = P_to - P(u) has P'(u) u and P'(u) / A.

/** Node q's term of Haebara's criterion: w Σ_j Σ_k (P_to - P(u))². */
function haebaraTerm(
  curves: CharacteristicCurves,
  q: number,
  slope: number,
  intercept: number,
  gradient: Float64Array,
): number {
  const weight = curves.weights[q] ?? 0;
  const u = ((curves.nodes[q] ?? 0) - intercept) / slope;
  let value = 0;
  for (let j = 0; j < curves.from.length; j++) {
    const target = fromItemAt(curves, j, u);
    const count = curves.counts[j] ?? 0;
    for (let k = 0; k < count && target !== undefined; k++) {
      const difference = (target[q * count + k] ?? 0) - (curves.probabilities[k] ?? 0);
      value += weight * difference * difference;
      addGradient(gradient, 2 * weight * difference * (curves.first[k] ?? 0), u, slope);
    }
  }
  return value;
}

/** Node q's term of Stocking and Lord's criterion: w (Σ_j Σ_k k (P_to - P(u)))². */
function stockingLordTerm(
  curves: CharacteristicCurves,
  q: number,
  slope: number,
  intercept: number,
  gradient: Float64Array,
): number {
  const weight = curves.weights[q] ?? 0;
  const u = ((curves.nodes[q] ?? 0) - intercept) / slope;
  let difference = 0;
  let derivative = 0;
  for (let j = 0; j < curves.from.length; j++) {
    const target = fromItemAt(curves, j, u);
    const count = curves.counts[j] ?? 0;
    for (let k = 1; k < count && target !== undefined; k++) {
      difference += k * ((target[q * count + k] ?? 0) - (curves.probabilities[k] ?? 0));
      derivative += k * (curves.first[k] ?? 0);
    }
  }
  addGradient(gradient, 2 * weight * difference * derivative, u, slope);
  return weight * difference * difference;
}

/** Adds `factor` times the derivatives of P(u) with respect to ln A and B, u and 1 / A, to the gradient. */
function addGradient(gradient: Float64Array, factor: number, u: number, slope: number): void {
  gradient[0] = (gradient[0] ?? 0) + factor * u;
  gradient[1] = (gradient[1] ?? 0) + factor / slope;
}
