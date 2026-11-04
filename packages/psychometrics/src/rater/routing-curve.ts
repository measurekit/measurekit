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
  const sums = new RunningSums();
  const points: RoutingPoint[] = [];
  for (let k = 0; k < order.length; k++) {
    const subject = order[k] ?? 0;
    sums.add(machine[subject] ?? 0, human[subject] ?? 0);
    const next = order[k + 1];
    if (next !== undefined && confidence[next] === confidence[subject]) continue;
    points.push({ minConfidence: confidence[subject] ?? 0, coverage: sums.n / order.length, ...sums.statistics() });
  }
  return points;
}

/** The sums of the pairs of scores added so far, from which their agreement follows. */
class RunningSums {
  n = 0;
  private machine = 0;
  private human = 0;
  private machineSquares = 0;
  private humanSquares = 0;
  private products = 0;
  private exact = 0;
  private adjacent = 0;

  add(machine: number, human: number): void {
    this.n++;
    this.machine += machine;
    this.human += human;
    this.machineSquares += machine * machine;
    this.humanSquares += human * human;
    this.products += machine * human;
    const difference = Math.abs(machine - human);
    if (difference === 0) this.exact++;
    if (difference <= 1) this.adjacent++;
  }

  /**
   * Returns the statistics of the pairs. The sums of squares and products about the means, times n, are differences of
   * sums: exact for integer scores.
   */
  statistics(): Omit<RoutingPoint, "minConfidence" | "coverage"> {
    const { n } = this;
    const machineSpread = n * this.machineSquares - this.machine * this.machine;
    const humanSpread = n * this.humanSquares - this.human * this.human;
    const covariance = n * this.products - this.machine * this.human;
    const shift = (this.machine - this.human) ** 2;
    const kappaDenominator = machineSpread + humanSpread + shift;
    const squares = this.machineSquares - 2 * this.products + this.humanSquares;
    const pooled = (machineSpread + humanSpread) / (2 * n * (n - 1));
    return {
      subjects: n,
      exactAgreement: this.exact / n,
      adjacentAgreement: this.adjacent / n,
      quadraticKappa: kappaDenominator > 0 ? (2 * covariance) / kappaDenominator : null,
      correlation:
        n > 1 && machineSpread > 0 && humanSpread > 0
          ? Math.min(1, Math.max(-1, covariance / Math.sqrt(machineSpread * humanSpread)))
          : null,
      smd: n > 1 && pooled > 0 ? (this.machine - this.human) / n / Math.sqrt(pooled) : null,
      rmse: Math.sqrt(Math.max(squares, 0) / n),
    };
  }
}
