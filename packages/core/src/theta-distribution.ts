/**
 * A discrete distribution of theta: nodes with weights that sum to 1, such as a posterior or a population. Both
 * libraries take this shape, so a posterior moves between them as data.
 */
export interface ThetaDistribution {
  /** The values of theta, in ascending order. */
  readonly nodes: readonly number[];
  /** The probability of each node, summing to 1. */
  readonly weights: readonly number[];
}
