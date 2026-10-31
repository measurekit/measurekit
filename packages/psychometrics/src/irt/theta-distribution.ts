import { checkArray, checkRecord, DataError, describe, type ThetaDistribution } from "@measurekit/core";

/** The tolerance of the sum of a distribution's weights from 1. */
const WEIGHT_SUM_TOLERANCE = 1e-6;

/** A distribution of theta checked: its nodes and weights. */
export interface CheckedThetaDistribution {
  readonly nodes: Float64Array;
  readonly weights: Float64Array;
}

/**
 * Checks a distribution of theta and returns its nodes and weights.
 *
 * @param subject - Names the distribution in error messages, such as `population of the from group`.
 * @throws `DataError` when it is not an object with arrays of finite nodes and as many non-negative weights that sum
 * to 1 within 1e-6.
 */
export function checkThetaDistribution(distribution: ThetaDistribution, subject: string): CheckedThetaDistribution {
  checkRecord(distribution, `The ${subject} must be an object { nodes, weights }`);
  const { nodes, weights } = distribution;
  checkArray(nodes, `The nodes of the ${subject} must be an array`);
  checkArray(weights, `The weights of the ${subject} must be an array`);
  if (nodes.length === 0 || nodes.length !== weights.length) {
    throw new DataError(
      `The ${subject} needs at least one node and as many weights as nodes, found ${nodes.length} and ${weights.length}.`,
    );
  }
  let sum = 0;
  nodes.forEach((node, q) => {
    const weight: unknown = weights[q];
    if (!Number.isFinite(node)) {
      throw new DataError(`The nodes of the ${subject} must be finite numbers, found ${describe(node)} at ${q}.`);
    }
    if (!(typeof weight === "number" && Number.isFinite(weight) && weight >= 0)) {
      throw new DataError(
        `The weights of the ${subject} must be non-negative numbers, found ${describe(weight)} at ${q}.`,
      );
    }
    sum += weight;
  });
  if (!(Math.abs(sum - 1) <= WEIGHT_SUM_TOLERANCE)) {
    throw new DataError(`The weights of the ${subject} must sum to 1, found ${sum}.`);
  }
  return { nodes: Float64Array.from(nodes), weights: Float64Array.from(weights) };
}
