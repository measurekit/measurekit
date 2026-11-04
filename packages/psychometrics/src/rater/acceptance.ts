import {
  checkArray,
  checkFields,
  checkOptions,
  checkRecord,
  DataError,
  describe,
  type FieldNames,
  isRecord,
} from "@measurekit/core";
import type { AutomatedScoreAgreement } from "./automated-agreement.js";

/** The thresholds that automated scores must meet to be accepted; `null` turns a rule off. */
export interface AcceptanceThresholds {
  /** The least quadratic weighted kappa of the machine with people. */
  readonly quadraticKappa: number | null;
  /** The least correlation of the machine with people. */
  readonly correlation: number | null;
  /** The most by which the machine's quadratic weighted kappa may fall below two people's. */
  readonly quadraticKappaDegradation: number | null;
  /** The largest standardized mean difference of the machine from people, either way. */
  readonly smd: number | null;
  /** The largest standardized mean difference in any subgroup, on the population's standard deviations, either way. */
  readonly groupSmd: number | null;
  /** The least and the most of the human SD over the machine SD. */
  readonly sdRatio: readonly [number, number] | null;
  /** The most by which the machine's exact agreement, a share, may fall below two people's. */
  readonly exactAgreementDegradation: number | null;
}

/**
 * The default thresholds of acceptance, those of Williamson, Xi and Breyer: a quadratic weighted kappa and a correlation
 * of at least 0.70, a fall of the kappa from two people's of at most 0.10, an SMD of at most 0.15 and, in a subgroup, of
 * at most 0.10. The SD ratio, from 2/3 to 1.5 in Wang and von Davier, and the fall of the exact agreement, at most 5.125
 * percentage points in other programs' rules, are off.
 *
 * @see {@link https://doi.org/10.1111/j.1745-3992.2011.00223.x | Williamson, Xi and Breyer (2012)}
 */
export const ACCEPTANCE_THRESHOLDS: AcceptanceThresholds = {
  quadraticKappa: 0.7,
  correlation: 0.7,
  quadraticKappaDegradation: 0.1,
  smd: 0.15,
  groupSmd: 0.1,
  sdRatio: null,
  exactAgreementDegradation: null,
};

const THRESHOLD_FIELDS: FieldNames<AcceptanceThresholds> = {
  quadraticKappa: true,
  correlation: true,
  quadraticKappaDegradation: true,
  smd: true,
  groupSmd: true,
  sdRatio: true,
  exactAgreementDegradation: true,
};

/** Options of {@link validateAutomatedScores}. */
export interface ValidateAutomatedScoresOptions {
  /**
   * The thresholds that differ from the defaults, each a number or `null` to turn its rule off.
   *
   * @defaultValue {@link ACCEPTANCE_THRESHOLDS}
   */
  readonly thresholds?: Partial<AcceptanceThresholds>;
}

const VALIDATE_AUTOMATED_SCORES_OPTIONS: FieldNames<ValidateAutomatedScoresOptions> = { thresholds: true };

/** A rule of acceptance. */
export type AcceptanceRule =
  | "quadratic-kappa"
  | "correlation"
  | "quadratic-kappa-degradation"
  | "smd"
  | "group-smd"
  | "sd-ratio"
  | "exact-agreement-degradation";

/** A rule of acceptance that automated scores break. */
export interface AcceptanceFailure {
  /** The rule. */
  readonly rule: AcceptanceRule;
  /**
   * The value it judges: the statistic, for a degradation its fall below two people's, for the SD ratio the human SD
   * over the machine SD; `null` where the data do not define it, which fails the rule.
   */
  readonly value: number | null;
  /** The threshold the value breaks: for the SD ratio, the end it passes. */
  readonly threshold: number;
  /** The subgroup, for `"group-smd"`; `null` otherwise. */
  readonly group: string | null;
}

/** The judgment of automated scores against thresholds of acceptance. */
export interface AutomatedScoreValidation {
  /** Whether every rule on is met. */
  readonly accepted: boolean;
  /** The rules broken, in the order of {@link AcceptanceThresholds}, subgroups in their order. */
  readonly failures: readonly AcceptanceFailure[];
}

/**
 * Judges the agreement of automated scores with human scores against thresholds of acceptance, and returns each rule
 * that it breaks.
 *
 * @remarks
 * The rules, each with a threshold of the option `thresholds` whose defaults and sources are those of
 * {@link ACCEPTANCE_THRESHOLDS}: the machine-human quadratic weighted kappa and correlation at least their thresholds;
 * the fall of the kappa from the human-human one at most its threshold; the absolute SMD at most its threshold, overall
 * and in each subgroup; the human SD over the machine SD within its two ends; the fall of the exact agreement at most its
 * threshold. A statistic that the data leave undefined, such as a fall without second human scores, breaks its rule with
 * a `value` of `null`: turn the rule off with `null` where the data cannot have it. The thresholds are a program's
 * policy, which differs between programs; the guide tables those of others.
 *
 * @param agreement - The agreement, as `automatedScoreAgreement` returns it.
 * @param options - The thresholds that differ from the defaults.
 * @returns Whether every rule is met, and the rules broken.
 * @throws `DataError` when the agreement is not one that `automatedScoreAgreement` returns, or a threshold is not a
 * finite number or `null`, or the SD ratio's ends are not two positive numbers in order.
 * @see {@link https://doi.org/10.1111/j.1745-3992.2011.00223.x | Williamson, Xi and Breyer (2012)}
 */
export function validateAutomatedScores(
  agreement: AutomatedScoreAgreement,
  options: ValidateAutomatedScoresOptions = {},
): AutomatedScoreValidation {
  checkOptions(options, VALIDATE_AUTOMATED_SCORES_OPTIONS);
  const thresholds = thresholdsOf(options.thresholds);
  checkAgreement(agreement);
  const { machineHuman, humanHuman, groups } = agreement;
  const failures: AcceptanceFailure[] = [];
  const fail = (rule: AcceptanceRule, value: number | null, threshold: number, group: string | null = null) => {
    failures.push({ rule, value, threshold, group });
  };
  const least = (rule: AcceptanceRule, value: number | null, threshold: number | null) => {
    if (threshold !== null && !(value !== null && value >= threshold)) fail(rule, value, threshold);
  };
  const most = (rule: AcceptanceRule, value: number | null, threshold: number | null) => {
    if (threshold !== null && !(value !== null && value <= threshold)) fail(rule, value, threshold);
  };
  const within = (
    rule: AcceptanceRule,
    value: number | null,
    threshold: number | null,
    group: string | null = null,
  ) => {
    if (threshold !== null && !(value !== null && Math.abs(value) <= threshold)) fail(rule, value, threshold, group);
  };
  least("quadratic-kappa", machineHuman.quadraticKappa, thresholds.quadraticKappa);
  least("correlation", machineHuman.correlation, thresholds.correlation);
  most(
    "quadratic-kappa-degradation",
    fall(humanHuman?.quadraticKappa, machineHuman.quadraticKappa),
    thresholds.quadraticKappaDegradation,
  );
  within("smd", machineHuman.smd, thresholds.smd);
  for (const { group, smd } of groups) within("group-smd", smd, thresholds.groupSmd, group);
  if (thresholds.sdRatio !== null) {
    const [lowest, highest] = thresholds.sdRatio;
    const { firstSd: machineSd, secondSd: humanSd } = machineHuman;
    const ratio = humanSd !== null && machineSd !== null && machineSd > 0 ? humanSd / machineSd : null;
    if (ratio === null || ratio < lowest) fail("sd-ratio", ratio, lowest);
    else if (ratio > highest) fail("sd-ratio", ratio, highest);
  }
  most(
    "exact-agreement-degradation",
    fall(humanHuman?.exactAgreement, machineHuman.exactAgreement),
    thresholds.exactAgreementDegradation,
  );
  return { accepted: failures.length === 0, failures };
}

/** Returns how far the machine's statistic falls below people's, or `null` without both. */
function fall(people: number | null | undefined, machine: number | null): number | null {
  return people === null || people === undefined || machine === null ? null : people - machine;
}

/**
 * Returns the thresholds, the defaults with the given ones in their place.
 *
 * @throws `DataError` when they are not a record of the thresholds' names, each a finite number or `null`, the SD
 * ratio two positive numbers in order.
 */
function thresholdsOf(given: Partial<AcceptanceThresholds> | undefined): AcceptanceThresholds {
  if (given === undefined) return ACCEPTANCE_THRESHOLDS;
  checkRecord(given, "The option thresholds must be an object of thresholds by rule");
  checkFields(given, THRESHOLD_FIELDS, "The option thresholds");
  for (const [name, value] of Object.entries(given)) {
    if (name === "sdRatio") continue;
    if (!(value === null || (typeof value === "number" && Number.isFinite(value)))) {
      throw new DataError(`The threshold ${name} must be a finite number or null, found ${describe(value)}.`);
    }
  }
  const { sdRatio } = given;
  if (sdRatio !== undefined && sdRatio !== null) {
    const valid =
      Array.isArray(sdRatio) &&
      sdRatio.length === 2 &&
      sdRatio.every((end) => typeof end === "number" && Number.isFinite(end) && end > 0) &&
      sdRatio[0] <= sdRatio[1];
    if (!valid) {
      throw new DataError(
        `The threshold sdRatio must be two positive numbers in order, or null, found ${describe(sdRatio)}.`,
      );
    }
  }
  return { ...ACCEPTANCE_THRESHOLDS, ...given };
}

/** Returns whether a value is a finite number or `null`. */
function isStatistic(value: unknown): boolean {
  return value === null || (typeof value === "number" && Number.isFinite(value));
}

/**
 * Checks the fields of an agreement that the rules read.
 *
 * @throws `DataError` when it is not an agreement as `automatedScoreAgreement` returns it.
 */
function checkAgreement(agreement: AutomatedScoreAgreement): void {
  const message = "The agreement must be one that automatedScoreAgreement returns";
  checkRecord(agreement, message);
  const { machineHuman, humanHuman, groups } = agreement;
  const statistics = (record: unknown, names: readonly string[]): boolean =>
    isRecord(record) && names.every((name) => isStatistic(record[name]));
  const fields = ["quadraticKappa", "correlation", "smd", "firstSd", "secondSd", "exactAgreement"];
  const valid =
    statistics(machineHuman, fields) &&
    (humanHuman === null || statistics(humanHuman, fields)) &&
    Array.isArray(groups) &&
    groups.every((group) => statistics(group, ["smd"]) && typeof group.group === "string");
  if (!valid) throw new DataError(`${message}, found ${describe(agreement)} with missing or invalid statistics.`);
  checkArray(groups, message);
}
