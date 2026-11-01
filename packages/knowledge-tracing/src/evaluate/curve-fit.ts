import { checkCurvePoints, type ErrorRatePoint } from "./curve-points.js";

/** A power law of practice fitted to a learning curve: error rate = `scale` × opportunity^(-`exponent`). */
export interface PowerLaw {
  /** The error rate at the first opportunity. */
  readonly scale: number;
  /** The rate of learning: above 0 where the error rate falls with practice. */
  readonly exponent: number;
  /** The R² of the fit on the logarithms, weighted by the counts; `null` when every error rate is equal. */
  readonly rSquared: number | null;
}

/**
 * Fits the power law of practice, e(n) = a n^(-b), to the error rates of a learning curve by least squares on the
 * logarithms, ln e = ln a - b ln n, with each point weighted by its `count`. Returns `null` unless two points remain.
 *
 * @remarks
 * A point with an error rate of 0 has no logarithm and is left out. The fit describes a curve: it tests no hypothesis,
 * and a curve of few learners per point is noisy at its tail, which the weights count. Pass the points of one skill,
 * such as `learningCurve(log).skills.fractions`, or its `overall` curve.
 *
 * @param points - The error rate of each opportunity.
 * @returns The coefficient and the exponent of the law with their fit; `null` where the points cannot be fitted.
 * @throws `DataError` when the points are malformed.
 */
export function powerLawFit(points: readonly ErrorRatePoint[]): PowerLaw | null {
  checkCurvePoints(points);
  const positive = points.filter((point) => point.errorRate > 0);
  if (positive.length < 2) return null;
  const x = positive.map((point) => Math.log(point.opportunity));
  const y = positive.map((point) => Math.log(point.errorRate));
  const weights = positive.map((point) => point.count);
  const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);
  const meanX = weightedMean(x, weights, totalWeight);
  const meanY = weightedMean(y, weights, totalWeight);
  let xx = 0;
  let xy = 0;
  let yy = 0;
  weights.forEach((weight, index) => {
    const dx = (x[index] ?? 0) - meanX;
    const dy = (y[index] ?? 0) - meanY;
    xx += weight * dx * dx;
    xy += weight * dx * dy;
    yy += weight * dy * dy;
  });
  const slope = xy / xx;
  return {
    scale: Math.exp(meanY - slope * meanX),
    exponent: -slope,
    // Compared as numbers, not through the means, which round: equal error rates have no variance to explain.
    rSquared: y.every((value) => value === y[0]) ? null : (xy * xy) / (xx * yy),
  };
}

function weightedMean(values: readonly number[], weights: readonly number[], totalWeight: number): number {
  return values.reduce((sum, value, index) => sum + value * (weights[index] ?? 0), 0) / totalWeight;
}
