import { checkFinite, checkPositive, checkRecord } from "@measurekit/core";

/** How a logistic model treats learners' proficiency θ: not at all, as a fixed effect, or as a random intercept. */
export type Proficiency = "none" | "fixed" | "random";

/** The values of {@link Proficiency}. */
export const PROFICIENCIES: readonly Proficiency[] = ["none", "fixed", "random"];

/** The parameters of a logistic model: plain data, which JSON serializes. */
export interface LogisticParameters {
  /** The coefficients of each skill, by skill id. */
  readonly skills: Readonly<Record<string, LogisticSkill>>;
  /**
   * Each fitted learner's proficiency θ, by learner id: fixed effects, or the conditional modes of a random intercept;
   * absent without proficiency.
   */
  readonly learners?: Readonly<Record<string, number>>;
  /** σ of the random intercept; absent without one. */
  readonly learnerSd?: number;
}

/** The coefficients of one skill. */
export interface LogisticSkill {
  /** β, the skill's intercept: its easiness. */
  readonly intercept: number;
  /** The coefficient of each feature, by the feature's label, such as `opportunities` or `decayedLogit(0.8)`. */
  readonly coefficients: Readonly<Record<string, number>>;
}

/** Checks parameters, which may come from storage, against the labels of the model's features. */
export function checkParameters(parameters: LogisticParameters, labels: readonly string[]): void {
  checkRecord(parameters, "Logistic parameters must be an object");
  checkRecord(parameters.skills, "Logistic parameters need skills, an object of coefficients by skill id");
  for (const [id, skill] of Object.entries(parameters.skills)) {
    const label = `the logistic parameters of skill "${id}"`;
    checkRecord(skill, `The ${label} must be an object`);
    checkFinite(skill.intercept, `The intercept of ${label}`);
    checkRecord(skill.coefficients, `The ${label} need coefficients, an object of coefficients by feature`);
    for (const feature of labels) {
      checkFinite(skill.coefficients[feature], `The coefficient of ${feature} in ${label}`);
    }
  }
  if (parameters.learners !== undefined) {
    checkRecord(parameters.learners, "Logistic parameters need learners, an object of proficiencies by learner id");
    for (const [learner, theta] of Object.entries(parameters.learners)) {
      checkFinite(theta, `The proficiency of learner "${learner}"`);
    }
  }
  if (parameters.learnerSd !== undefined) checkPositive(parameters.learnerSd, "The learnerSd of logistic parameters");
}
