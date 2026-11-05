import { checkOptions, DataError, type FieldNames } from "@measurekit/core";
import { type AutomatedScore, checkAutomatedScores, type ScoreRange } from "./automated-scores.js";

/** Options of {@link routingCurve}. */
export interface RoutingCurveOptions {
  /**
   * The score scale, as `automatedScoreAgreement` takes it.
   *
   * @defaultValue none: machine scores are taken as given, and must be categories
   */
  readonly scale?: ScoreRange;
}

const ROUTING_CURVE_OPTIONS: FieldNames<RoutingCurveOptions> = { scale: true };

/** The agreement of the machine with people on the subjects whose confidence reaches a threshold. */
export interface RoutingPoint {
  /** The threshold: the subjects at or above it keep the machine's score, the others go to a person. */
  readonly minConfidence: number;
  /** The share of the subjects at or above the threshold, from 0 to 1. */
  readonly coverage: number;
  /** Their number. */
  readonly subjects: number;
  /** Their exact agreement, a share. */
  readonly exactAgreement: number;
  /** Their adjacent agreement, within 1 point. */
  readonly adjacentAgreement: number;
  /** Their quadratic weighted kappa; `null` where the chance agreement is 1. */
  readonly quadraticKappa: number | null;
  /** Their correlation; `null` for fewer than two subjects or a constant scoring. */
  readonly correlation: number | null;
  /** Their standardized mean difference, the machine's mean less people's over the pooled SD; `null` where undefined. */
  readonly smd: number | null;
  /** Their root mean squared difference. */
  readonly rmse: number;
}

/**
 * Returns the routing curve of automated scores: for each confidence of the scorer, the share of the subjects at or
 * above it and the machine's agreement with people on them, from which a host sets the least confidence at which a
 * machine score stands alone.
 *
 * @remarks
 * The curve of coverage against agreement of selective prediction (El-Yaniv and Wiener), with the statistics of
 * Williamson, Xi and Breyer. A point for each distinct confidence, from the highest: the subjects at or above it are
 * those the machine scores alone, and their exact and adjacent agreement, quadratic weighted kappa, correlation, SMD and
 * RMSE are those of `automatedScoreAgreement` on them. The sums run once over the subjects sorted by confidence, so n
 * subjects take O(n log n). The quadratic weighted kappa is 2 cov / (var_m + var_h + (mean_m - mean_h)²), with moments
 * on n, which is Cohen's with quadratic weights. Choose the threshold from data other than those it is judged on, as the
 * point that agrees best on a sample agrees less on the next.
 *
 * @param scores - The subjects, each with its `confidence`.
 * @param options - The score scale.
 * @returns A point for each distinct confidence, from the highest; the last covers every subject.
 * @throws `DataError` when there is no subject, a subject has no finite confidence, or a score is invalid as
 * `automatedScoreAgreement` says.
 * @see {@link https://www.jmlr.org/papers/v11/el-yaniv10a.html | El-Yaniv and Wiener (2010)}
 * @see {@link https://doi.org/10.1111/j.1745-3992.2011.00223.x | Williamson, Xi and Breyer (2012)}
 */
export function routingCurve(
  scores: readonly AutomatedScore[],
  options: RoutingCurveOptions = {},
): readonly RoutingPoint[] {
  checkOptions(options, ROUTING_CURVE_OPTIONS);
  const { scale } = options;
  const { machine, human, confidence } = checkAutomatedScores(scores, {
    categories: true,
    minSubjects: 1,
    ...(scale ? { scale } : {}),
  });
  confidence.forEach((value, subject) => {
    if (Number.isNaN(value)) throw new DataError(`Subject ${subject + 1} needs a confidence for the routing curve.`);
  });
  const order = Uint32Array.from(confidence.keys()).sort((x, y) => (confidence[y] ?? 0) - (confidence[x] ?? 0));
  const sums: RunningSums = {
    n: 0,
    machine: 0,
    human: 0,
    machineSquares: 0,
    humanSquares: 0,
    products: 0,
    exact: 0,
    adjacent: 0,
  };
  // Every statistic is the same for scores less one value, and the sums of scores less a score stay small, so that
  // scores near 10^8 keep the differences of sums exact.
  const shift = human[order[0] ?? 0] ?? 0;
  const points: RoutingPoint[] = [];
  for (let k = 0; k < order.length; k++) {
    const subject = order[k] ?? 0;
    addPair(sums, (machine[subject] ?? 0) - shift, (human[subject] ?? 0) - shift);
    const next = order[k + 1];
    if (next !== undefined && confidence[next] === confidence[subject]) continue;
    points.push({ minConfidence: confidence[subject] ?? 0, coverage: sums.n / order.length, ...statistics(sums) });
  }
  return points;
}

/** The sums of the pairs of scores added so far, from which their agreement follows. */
interface RunningSums {
  n: number;
  machine: number;
  human: number;
  machineSquares: number;
  humanSquares: number;
  products: number;
  exact: number;
  adjacent: number;
}

/** Adds a pair of scores to the sums. */
function addPair(sums: RunningSums, machine: number, human: number): void {
  sums.n++;
  sums.machine += machine;
  sums.human += human;
  sums.machineSquares += machine * machine;
  sums.humanSquares += human * human;
  sums.products += machine * human;
  const difference = Math.abs(machine - human);
  if (difference === 0) sums.exact++;
  if (difference <= 1) sums.adjacent++;
}

/**
 * Returns the statistics of the pairs. The sums of squares and products about the means, times n, are differences of
 * sums: exact for integer scores while n times the sum of squares stays below 2⁵³.
 */
function statistics(sums: RunningSums): Omit<RoutingPoint, "minConfidence" | "coverage"> {
  const { n } = sums;
  const machineSpread = n * sums.machineSquares - sums.machine * sums.machine;
  const humanSpread = n * sums.humanSquares - sums.human * sums.human;
  const covariance = n * sums.products - sums.machine * sums.human;
  const shift = (sums.machine - sums.human) ** 2;
  const kappaDenominator = machineSpread + humanSpread + shift;
  const squares = sums.machineSquares - 2 * sums.products + sums.humanSquares;
  const pooled = (machineSpread + humanSpread) / (2 * n * (n - 1));
  return {
    subjects: n,
    exactAgreement: sums.exact / n,
    adjacentAgreement: sums.adjacent / n,
    quadraticKappa: kappaDenominator > 0 ? (2 * covariance) / kappaDenominator : null,
    correlation:
      n > 1 && machineSpread > 0 && humanSpread > 0
        ? Math.min(1, Math.max(-1, covariance / Math.sqrt(machineSpread * humanSpread)))
        : null,
    smd: n > 1 && pooled > 0 ? (sums.machine - sums.human) / n / Math.sqrt(pooled) : null,
    rmse: Math.sqrt(Math.max(squares, 0) / n),
  };
}
