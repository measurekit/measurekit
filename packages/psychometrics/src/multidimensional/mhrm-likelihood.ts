import { cholesky, invertPositiveDefinite, normalizeLogs, type Random } from "@measurekit/core";
import { type CompressedResponses, logProbability } from "../irt/internal.js";
import type { ItemState } from "./calibration-setup.js";
import type { Correlations } from "./mhrm-latent.js";
import { answeredPattern } from "./pattern.js";
import { posteriorMode } from "./posterior-mode.js";

/** The draws per person of the importance sampling of the log-likelihood: mirt's default `draws` of `logLik`. */
const LOG_LIKELIHOOD_DRAWS = 5000;

/**
 * Returns the marginal log-likelihood of the responses, by importance sampling of each person's integral over theta.
 *
 * @remarks
 * The proposal for a person is the normal approximation of the posterior at its mode: the mean is the posterior mode
 * and the covariance the inverse of the test information plus the precision of the prior. The estimate for a person is
 * ln of the mean of f(x, θ_r) / q(θ_r) over 5000 draws θ_r from it, with f the density of the responses and theta, and
 * the log-likelihood is the sum over persons. All persons use the same standard normal draws, which the proposal's
 * Cholesky factor moves and scales.
 */
export function importanceLogLikelihood<TParameters>(
  items: readonly ItemState<TParameters>[],
  data: CompressedResponses,
  correlations: Correlations,
  random: Random,
): number {
  const { dimensions, precision, matrix } = correlations;
  const prior = { mean: new Float64Array(dimensions), covariance: matrix, precision };
  const work: Workspace = {
    standard: Float64Array.from({ length: LOG_LIKELIHOOD_DRAWS * dimensions }, () => random.normal()),
    draws: new Float64Array(LOG_LIKELIHOOD_DRAWS * dimensions),
    logWeights: new Float64Array(LOG_LIKELIHOOD_DRAWS),
    point: new Float64Array(dimensions),
    probabilities: new Float64Array(items.reduce((most, item) => Math.max(most, item.categoryCount), 0)),
  };
  let total = 0;
  for (let person = 0; person < data.offsets.length - 1; person++) {
    const start = data.offsets[person] ?? 0;
    const end = data.offsets[person + 1] ?? 0;
    if (end === start) continue;
    const answered = Array.from(data.items.subarray(start, end)).flatMap((item) => items[item] ?? []);
    const codes = Array.from(data.codes.subarray(start, end));
    const { theta, information } = posteriorMode(answeredPattern(answered, codes, dimensions), prior, dimensions);
    drawProposals(theta, proposalFactor(information, correlations), correlations, work);
    addResponseLogs(answered, codes, work, dimensions);
    total += normalizeLogs(work.logWeights) - Math.log(LOG_LIKELIHOOD_DRAWS);
  }
  return total;
}

/** The standard normal draws shared by the persons, and the arrays one person's draws and weights are built in. */
interface Workspace {
  readonly standard: Float64Array;
  readonly draws: Float64Array;
  readonly logWeights: Float64Array;
  readonly point: Float64Array;
  readonly probabilities: Float64Array;
}

/** Moves the standard normal draws to the proposal with mean `theta` and Cholesky factor `factor`, and starts their log weights at the prior over the proposal. */
function drawProposals(theta: Float64Array, factor: Float64Array, correlations: Correlations, work: Workspace): void {
  const { dimensions } = correlations;
  const { standard, draws, logWeights } = work;
  for (let r = 0; r < LOG_LIKELIHOOD_DRAWS; r++) {
    let squares = 0;
    for (let d = 0; d < dimensions; d++) {
      let sum = theta[d] ?? 0;
      for (let k = 0; k <= d; k++) sum += (factor[d * dimensions + k] ?? 0) * (standard[r * dimensions + k] ?? 0);
      draws[r * dimensions + d] = sum;
      squares += (standard[r * dimensions + d] ?? 0) ** 2;
    }
    logWeights[r] = logRatio(draws, r, squares, factor, correlations);
  }
}

/** Adds ln P(x | θ_r) of the person's responses to the log weight of every draw r. */
function addResponseLogs<TParameters>(
  answered: readonly ItemState<TParameters>[],
  codes: readonly number[],
  { draws, logWeights, point, probabilities }: Workspace,
  dimensions: number,
): void {
  answered.forEach((item, k) => {
    for (let r = 0; r < LOG_LIKELIHOOD_DRAWS; r++) {
      for (let d = 0; d < dimensions; d++) point[d] = draws[r * dimensions + d] ?? 0;
      item.model.probabilities(item.parameters, point, probabilities);
      logWeights[r] = (logWeights[r] ?? 0) + logProbability(probabilities[codes[k] ?? 0] ?? 0);
    }
  });
}

/**
 * Returns the lower triangular Cholesky factor of the proposal's covariance, the inverse of the information; the
 * correlations' own factor where the information cannot be inverted.
 */
function proposalFactor(information: Float64Array, { dimensions, matrix }: Correlations): Float64Array {
  const covariance = invertPositiveDefinite(information, dimensions);
  const factor = covariance === null ? null : cholesky(covariance, dimensions);
  return factor ?? cholesky(matrix, dimensions) ?? new Float64Array(dimensions * dimensions);
}

/**
 * Returns ln [N(θ; 0, Φ) / q(θ)] of draw r, the proposal q being the normal with Cholesky factor L and `squares` the
 * sum of squares of the standard normals that made the draw: -θᵀ Ω θ / 2 - ln |Φ| / 2 + z·z / 2 + ln |L|.
 */
function logRatio(
  draws: Float64Array,
  r: number,
  squares: number,
  factor: Float64Array,
  correlations: Correlations,
): number {
  const { dimensions, precision } = correlations;
  let quadratic = 0;
  for (let i = 0; i < dimensions; i++) {
    for (let j = 0; j < dimensions; j++) {
      quadratic +=
        (draws[r * dimensions + i] ?? 0) * (precision[i * dimensions + j] ?? 0) * (draws[r * dimensions + j] ?? 0);
    }
  }
  let logFactor = 0;
  for (let d = 0; d < dimensions; d++) logFactor += Math.log(factor[d * dimensions + d] ?? 1);
  return -quadratic / 2 - correlations.logDeterminant() / 2 + squares / 2 + logFactor;
}
