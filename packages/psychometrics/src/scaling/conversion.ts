import { checkArray, DataError, describe } from "@measurekit/core";
import type { IrtItem } from "../irt/internal.js";

/** Returns the highest total score of the items: the sum of their categories less 1. */
export function highestScore(items: readonly IrtItem[]): number {
  let highest = 0;
  for (const item of items) highest += item.model.categoryCount(item.parameters) - 1;
  return highest;
}

/**
 * Checks a conversion of the total scores of a form to scale scores, one for each total score from 0 to the highest.
 *
 * @param label - Names the conversion in error messages.
 * @throws `DataError` when it is not an array of the form's highest total score plus 1 finite numbers.
 */
export function checkConversion(conversion: readonly number[], items: readonly IrtItem[], label: string): void {
  checkArray(conversion, `${label} must be an array of scale scores`);
  const expected = highestScore(items) + 1;
  if (conversion.length !== expected) {
    throw new DataError(
      `${label} needs ${expected} scale scores, one for each total score from 0 to ${expected - 1}, found ${conversion.length}.`,
    );
  }
  conversion.forEach((score, total) => {
    if (!Number.isFinite(score)) {
      throw new DataError(`${label} needs finite scale scores, found ${describe(score)} for the total score ${total}.`);
    }
  });
}

/**
 * Returns the standard deviation of the scale scores at one node, given the probability of each total score there:
 * the square root of Σ P(x) (c(x) - m)², m = Σ P(x) c(x).
 *
 * @param likelihoods - The probability of each total score at each node, by score, then node.
 * @param nodeCount - The number of nodes.
 * @param node - The index of the node.
 */
export function conditionalDeviation(
  likelihoods: Float64Array,
  nodeCount: number,
  node: number,
  conversion: ArrayLike<number>,
): number {
  let mean = 0;
  for (let score = 0; score < conversion.length; score++) {
    mean += (likelihoods[score * nodeCount + node] ?? 0) * (conversion[score] ?? 0);
  }
  let variance = 0;
  for (let score = 0; score < conversion.length; score++) {
    variance += (likelihoods[score * nodeCount + node] ?? 0) * ((conversion[score] ?? 0) - mean) ** 2;
  }
  return Math.sqrt(variance);
}
