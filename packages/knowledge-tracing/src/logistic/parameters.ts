import { checkFinite, checkPositive, checkRecord, checkRecordById } from "@measurekit/core";
import { entry } from "../data/internal.js";

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
  /**
   * The intercept and the coefficients of each item the fit saw, by item id; absent unless the model has item
   * intercepts or terms with a coefficient for each item. An item it did not see has 0 for all of them.
   */
  readonly items?: Readonly<Record<string, LogisticSkill>>;
  /**
   * The coefficients shared across skills, items or learners, by label, such as `learner:logSuccesses`, and the
   * covariates' by `covariate:<name>`; absent unless the model has any.
   */
  readonly shared?: Readonly<Record<string, number>>;
}

/** The coefficients of one skill or item. */
export interface LogisticSkill {
  /** β, the intercept: the easiness of the skill or item, 0 where the model has no intercept for them. */
  readonly intercept: number;
  /** The coefficient of each feature, by the feature's label, such as `opportunities` or `decayedLogit(0.8)`. */
  readonly coefficients: Readonly<Record<string, number>>;
}

/** The labels of the coefficients that a model's parameters must have. */
export interface ParameterLabels {
  /** The features with a coefficient for each skill. */
  readonly skills: readonly string[];
  /** The features with a coefficient for each item. */
  readonly items: readonly string[];
  /** The coefficients shared across levels, then the covariates'. */
  readonly shared: readonly string[];
}

/** Checks parameters, which may come from storage, against the labels of the model's coefficients. */
export function checkParameters(parameters: LogisticParameters, labels: ParameterLabels): void {
  checkRecord(parameters, "Logistic parameters must be an object");
  checkRecordById(parameters.skills, "Logistic parameters need skills, an object of coefficients by skill id");
  checkCoefficients(parameters.skills, "skill", labels.skills);
  if (parameters.items !== undefined) {
    checkRecordById(parameters.items, "Logistic parameters need items, an object of coefficients by item id");
    checkCoefficients(parameters.items, "item", labels.items);
  }
  if (parameters.shared !== undefined) {
    checkRecordById(parameters.shared, "Logistic parameters need shared, an object of coefficients by label");
    for (const label of labels.shared) {
      checkFinite(entry(parameters.shared, label), `The shared coefficient ${label} of the logistic parameters`);
    }
    for (const [label, value] of Object.entries(parameters.shared)) {
      checkFinite(value, `The shared coefficient ${label} of the logistic parameters`);
    }
  }
  if (parameters.learners !== undefined) {
    checkRecordById(parameters.learners, "Logistic parameters need learners, an object of proficiencies by learner id");
    for (const [learner, theta] of Object.entries(parameters.learners)) {
      checkFinite(theta, `The proficiency of learner "${learner}"`);
    }
  }
  if (parameters.learnerSd !== undefined) checkPositive(parameters.learnerSd, "The learnerSd of logistic parameters");
}

/** Checks the intercept and the coefficients of each skill or item against the labels of the features. */
function checkCoefficients(
  record: Readonly<Record<string, LogisticSkill>>,
  kind: "skill" | "item",
  labels: readonly string[],
): void {
  for (const [id, found] of Object.entries(record)) {
    const owner = `the logistic parameters of ${kind} "${id}"`;
    checkRecord(found, `The logistic parameters of ${kind} "${id}" must be an object`);
    checkFinite(found.intercept, `The intercept of ${owner}`);
    checkRecordById(
      found.coefficients,
      `The logistic parameters of ${kind} "${id}" need coefficients, an object of coefficients by feature`,
    );
    for (const feature of labels) {
      checkFinite(found.coefficients[feature], `The coefficient of ${feature} in ${owner}`);
    }
  }
}
