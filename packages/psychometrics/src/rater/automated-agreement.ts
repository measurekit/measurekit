import { checkOptions, type FieldNames } from "@measurekit/core";
import {
  type AutomatedScore,
  checkAutomatedScores,
  type ScoreAgreement,
  type ScoreRange,
  scoreAgreement,
} from "./automated-scores.js";

/** Options of {@link automatedScoreAgreement}. */
export interface AutomatedScoreAgreementOptions {
  /**
   * The score scale: machine scores are trimmed to it and rounded to the nearest point, a half up, before every
   * statistic, as RSMTool's trimmed and rounded scores.
   *
   * @defaultValue none: machine scores are taken as given, and must be categories, integers from 0
   */
  readonly scale?: ScoreRange;
}

const AUTOMATED_SCORE_AGREEMENT_OPTIONS: FieldNames<AutomatedScoreAgreementOptions> = { scale: true };

/** The machine-human statistics less the human-human ones that RSMTool reports as the degradation. */
export interface AgreementDegradation {
  /** Of the correlation. */
  readonly correlation: number | null;
  /** Of Cohen's kappa. */
  readonly kappa: number | null;
  /** Of the quadratic weighted kappa. */
  readonly quadraticKappa: number | null;
  /** Of the exact agreement. */
  readonly exactAgreement: number;
  /** Of the adjacent agreement. */
  readonly adjacentAgreement: number;
}

/** The agreement of the machine with people within one subgroup, its SMD on the population's standard deviations. */
export interface GroupAgreement extends ScoreAgreement {
  /** The subgroup. */
  readonly group: string;
}

/** The evaluation of automated scores against human scores, as RSMTool computes it. */
export interface AutomatedScoreAgreement {
  /** The machine scores, first, against the human scores, over every subject. */
  readonly machineHuman: ScoreAgreement;
  /** The second human scores, first, against the human scores, over the subjects with a second score; `null` without. */
  readonly humanHuman: ScoreAgreement | null;
  /** `machineHuman` less `humanHuman`: negative where the machine agrees less than a second person; `null` without. */
  readonly degradation: AgreementDegradation | null;
  /**
   * The machine-human correlation over the square root of the human-human correlation; `null` without a positive
   * human-human correlation or a machine-human one.
   */
  readonly disattenuatedCorrelation: number | null;
  /**
   * The machine-human agreement within each subgroup, in order of first appearance, its SMD over √((m² + h²) / 2) with
   * m and h the standard deviations of the machine and human scores of every subject.
   */
  readonly groups: readonly GroupAgreement[];
}

/**
 * Returns the agreement of automated scores with human scores: exact and adjacent agreement, kappa, quadratic weighted
 * kappa, correlation, standardized mean difference, RMSE and R², against a second person's agreement and within each
 * subgroup, as RSMTool's evaluation computes them.
 *
 * @remarks
 * The statistics of Williamson, Xi and Breyer, with RSMTool's names in the guide. The machine-human statistics use every
 * subject, and the human-human ones the subjects with a `second` score, as RSMTool does by default; the degradation is
 * their difference and the disattenuated correlation r_mh / √r_hh. Each subgroup's SMD is on the population's standard
 * deviations, Williamson et al.'s and RSMTool's `"williamson"`, so the subgroups share one scale. With `scale`, machine
 * scores are trimmed to it and rounded, a half up, before every statistic; without, they must be categories. Human
 * scores are categories, integers from 0. Pass the result to `validateAutomatedScores` to judge it against thresholds.
 *
 * @param scores - The subjects: each one's machine score, human score, and second score and group where it has them.
 * @param options - The score scale.
 * @returns The machine-human, human-human and subgroup statistics, the degradation and the disattenuated correlation.
 * @throws `DataError` when there are fewer than two subjects, a score is not a finite number, a human or second score is
 * not a category, a machine score is not one without `scale`, a group is not a string or `null`, or the scale is not
 * integers from 0 with `lowest` below `highest`.
 * @see {@link https://doi.org/10.1111/j.1745-3992.2011.00223.x | Williamson, Xi and Breyer (2012)}
 *
 * @example
 * ```ts
 * import { automatedScoreAgreement } from "@measurekit/psychometrics/rater";
 *
 * // Ten essays on a scale of 1 to 6: an engine's unrounded scores, a person's, a second person's and the writers' groups.
 * const machine = [3.4, 3.6, 2.6, 4.4, 3.3, 4.6, 1.8, 3.7, 5.3, 4.2];
 * const human = [3, 4, 2, 5, 3, 4, 1, 3, 6, 4];
 * const second = [3, 4, 3, 5, 2, 4, 2, 3, 5, 4];
 * const scores = machine.map((score, i) => ({
 *   machine: score,
 *   human: human[i] ?? 0,
 *   second: second[i] ?? null,
 *   group: i < 5 ? "a" : "b",
 * }));
 * const agreement = automatedScoreAgreement(scores, { scale: { lowest: 1, highest: 6 } });
 * agreement.machineHuman.quadraticKappa; // 0.78
 * agreement.degradation?.quadraticKappa; // -0.084, the machine's QWK less two people's
 * agreement.groups.map(({ smd }) => smd); // [0, 0.33]
 * ```
 */
export function automatedScoreAgreement(
  scores: readonly AutomatedScore[],
  options: AutomatedScoreAgreementOptions = {},
): AutomatedScoreAgreement {
  checkOptions(options, AUTOMATED_SCORE_AGREEMENT_OPTIONS);
  const { scale } = options;
  const checked = checkAutomatedScores(scores, { categories: true, minSubjects: 2, ...(scale ? { scale } : {}) });
  const { machine, human, second, groups } = checked;
  const machineHuman = scoreAgreement(machine, human);
  const doubled: number[] = [];
  second.forEach((score, subject) => {
    if (!Number.isNaN(score)) doubled.push(subject);
  });
  const humanHuman =
    doubled.length === 0
      ? null
      : scoreAgreement(
          Float64Array.from(doubled, (subject) => second[subject] ?? 0),
          Float64Array.from(doubled, (subject) => human[subject] ?? 0),
        );
  const population = { first: machineHuman.firstSd ?? 0, second: machineHuman.secondSd ?? 0 };
  return {
    machineHuman,
    humanHuman,
    degradation: humanHuman === null ? null : degradationOf(machineHuman, humanHuman),
    disattenuatedCorrelation: disattenuated(machineHuman.correlation, humanHuman?.correlation ?? null),
    groups: subgroups(groups).map(([group, members]) => ({
      group,
      ...scoreAgreement(
        Float64Array.from(members, (subject) => machine[subject] ?? 0),
        Float64Array.from(members, (subject) => human[subject] ?? 0),
        population,
      ),
    })),
  };
}

/** Returns the subjects of each group, in order of the group's first appearance; subjects without a group are left out. */
function subgroups(groups: readonly (string | null)[]): [string, number[]][] {
  const members = new Map<string, number[]>();
  groups.forEach((group, subject) => {
    if (group === null) return;
    const list = members.get(group) ?? [];
    list.push(subject);
    members.set(group, list);
  });
  return [...members.entries()];
}

/** Returns a difference of two statistics, `null` where either is. */
function lessOf(first: number | null, second: number | null): number | null {
  return first === null || second === null ? null : first - second;
}

function degradationOf(machineHuman: ScoreAgreement, humanHuman: ScoreAgreement): AgreementDegradation {
  return {
    correlation: lessOf(machineHuman.correlation, humanHuman.correlation),
    kappa: lessOf(machineHuman.kappa, humanHuman.kappa),
    quadraticKappa: lessOf(machineHuman.quadraticKappa, humanHuman.quadraticKappa),
    exactAgreement: machineHuman.exactAgreement - humanHuman.exactAgreement,
    adjacentAgreement: machineHuman.adjacentAgreement - humanHuman.adjacentAgreement,
  };
}

/** Returns r_mh / √r_hh, or `null` without both or with r_hh not above 0. */
function disattenuated(machineHuman: number | null, humanHuman: number | null): number | null {
  if (machineHuman === null || humanHuman === null || !(humanHuman > 0)) return null;
  return machineHuman / Math.sqrt(humanHuman);
}
