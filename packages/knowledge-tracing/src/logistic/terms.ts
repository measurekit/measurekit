import { checkArray, checkChoice, DataError, describe, isNonEmptyString, isRecord } from "@measurekit/core";
import { type Attempt, entry } from "../data/internal.js";
import {
  checkFeature,
  type FeatureSpec,
  featureValue,
  type LogisticFeature,
  needsOf,
  readsExposures,
} from "./features.js";
import type { ParameterLabels } from "./parameters.js";
import type { LogisticState, SkillHistory, StateNeeds } from "./state.js";

/** What a term's history counts over: the learner's attempts at each skill, at the item, or all of them. */
export type TermScope = "skill" | "item" | "learner";

/** The values of {@link TermScope}. */
const SCOPES: readonly TermScope[] = ["skill", "item", "learner"];

/** Whether a term has a coefficient for each skill or item, or one for all of them. */
export type TermCoefficient = "each" | "shared";

/** The values of {@link TermCoefficient}. */
const COEFFICIENTS: readonly TermCoefficient[] = ["each", "shared"];

/** The intercepts a logistic model can have: one for each skill, one for each item. */
export type InterceptKind = "skill" | "item";

/** The values of {@link InterceptKind}. */
const INTERCEPT_KINDS: readonly InterceptKind[] = ["skill", "item"];

/** A feature with the history it counts over and the coefficients it gets. */
export interface LogisticTerm {
  /** The feature of the history. */
  readonly feature: LogisticFeature;
  /**
   * What the history counts over: the learner's earlier attempts at the skill (each of the attempt's skills), at the
   * item, or at any item. The exposure features count over skills only.
   *
   * @defaultValue `"skill"`
   */
  readonly over?: TermScope;
  /**
   * `"each"` gives a coefficient to each skill or item; `"shared"` one to all of them. Over the learner a coefficient
   * for each would be a slope for each learner, a random-slope model.
   *
   * @defaultValue `"each"` over skills and items, and `"shared"` over the learner
   */
  readonly coefficient?: TermCoefficient;
}

/** A checked term. */
export interface TermSpec {
  readonly feature: FeatureSpec;
  readonly over: TermScope;
  readonly coefficient: TermCoefficient;
  /**
   * The key of the term's coefficient: the feature's label for a coefficient of each skill or item, and for one shared
   * `<over>:<feature label>`, such as `learner:logSuccesses`.
   */
  readonly label: string;
}

/** The terms of a model, sorted by where their coefficients go, with what a state tracks for them. */
export interface Terms {
  /** Whether the model has an intercept for each skill, and for each item. */
  readonly intercepts: Readonly<Record<InterceptKind, boolean>>;
  /** The terms with a coefficient for each skill, then for each item. */
  readonly skills: readonly TermSpec[];
  readonly items: readonly TermSpec[];
  /** The terms with one coefficient. */
  readonly shared: readonly TermSpec[];
  /** The names of the covariates, each with one coefficient. */
  readonly covariates: readonly string[];
  /** What a state tracks for the terms' histories. */
  readonly needs: StateNeeds;
  /** The labels of every coefficient the model has besides the intercepts. */
  readonly labels: ParameterLabels;
  /** The work arrays of {@link visitRow}: the values of a skill's block, an item's block, and the shared ones. */
  readonly scratch: { readonly skill: Float64Array; readonly item: Float64Array; readonly shared: Float64Array };
}

/** The prefix of a covariate's label among the shared coefficients. */
const COVARIATE_PREFIX = "covariate:";

/**
 * Returns the checked terms of the `features` option, rejecting unknown features, a coefficient for each learner,
 * exposure features over anything but skills, and a term listed twice. A plain feature is a term over skills.
 */
export function checkTerms(features: readonly (LogisticFeature | LogisticTerm)[]): TermSpec[] {
  checkArray(features, "The option features must be an array");
  const seen = new Set<string>();
  return features.map((entryOfFeatures) => {
    const term = checkTerm(entryOfFeatures);
    const key = `${term.over}/${term.coefficient}/${term.feature.label}`;
    if (seen.has(key)) throw new DataError(listedTwice(term));
    seen.add(key);
    return term;
  });
}

/** Returns the message for a term listed twice: the plain message for a plain feature. */
function listedTwice({ feature, over, coefficient }: TermSpec): string {
  return over === "skill" && coefficient === "each"
    ? `The feature ${feature.label} is listed twice.`
    : `The term ${feature.label} over ${over} with a ${coefficient} coefficient is listed twice.`;
}

/** Returns the checked term of an entry of the option features. */
function checkTerm(entryOfFeatures: LogisticFeature | LogisticTerm): TermSpec {
  const {
    feature,
    over = "skill",
    coefficient = over === "learner" ? "shared" : "each",
  } = isTerm(entryOfFeatures) ? fieldsOf(entryOfFeatures) : { feature: entryOfFeatures };
  const spec = checkFeature(feature);
  const { label } = spec;
  if (over === "learner" && coefficient === "each") {
    throw new DataError(
      `The term ${label} over the learner needs one shared coefficient: a coefficient for each learner is a random slope, which logistic models do not have.`,
    );
  }
  if (over !== "skill" && readsExposures(spec)) {
    throw new DataError(`The feature ${label} counts the exposures to a skill, so its term can only be over skills.`);
  }
  return { feature: spec, over, coefficient, label: coefficient === "each" ? label : `${over}:${label}` };
}

/** The fields of a term as given, and of a plain feature. */
interface TermFields {
  readonly feature: LogisticFeature;
  readonly over?: TermScope;
  readonly coefficient?: TermCoefficient;
}

/** Returns whether an entry of the option features is a term object, which has the field `feature`. */
function isTerm(entryOfFeatures: LogisticFeature | LogisticTerm): entryOfFeatures is LogisticTerm {
  return isRecord(entryOfFeatures) && "feature" in entryOfFeatures;
}

/** Returns the fields of a term object, rejecting those it does not have. */
function fieldsOf(term: LogisticTerm): TermFields {
  for (const key of Object.keys(term)) {
    if (!(key === "feature" || key === "over" || key === "coefficient")) {
      throw new DataError(`A term has no field ${JSON.stringify(key)}; its fields are feature, over and coefficient.`);
    }
  }
  const { feature, over, coefficient } = term;
  if ("over" in term) checkChoice(over, "The field over of a term", SCOPES);
  if ("coefficient" in term) checkChoice(coefficient, "The field coefficient of a term", COEFFICIENTS);
  return { feature, ...(over !== undefined && { over }), ...(coefficient !== undefined && { coefficient }) };
}

/** Returns the checked intercepts: each of `"skill"` and `"item"` at most once. */
export function checkIntercepts(intercepts: readonly InterceptKind[]): Readonly<Record<InterceptKind, boolean>> {
  checkArray(intercepts, "The option intercepts must be an array");
  for (const kind of intercepts) checkChoice(kind, "An intercept of the option intercepts", INTERCEPT_KINDS);
  const repeated = intercepts.find((kind, i) => intercepts.indexOf(kind) !== i);
  if (repeated !== undefined) throw new DataError(`The option intercepts lists ${JSON.stringify(repeated)} twice.`);
  return { skill: intercepts.includes("skill"), item: intercepts.includes("item") };
}

/** Returns the checked names of the covariates: distinct, non-empty strings. */
export function checkCovariateNames(covariates: readonly string[]): readonly string[] {
  checkArray(covariates, "The option covariates must be an array");
  for (const name of covariates) {
    if (!isNonEmptyString(name)) {
      throw new DataError(`The option covariates needs names, non-empty strings, found ${describe(name)}.`);
    }
  }
  const repeated = covariates.find((name, i) => covariates.indexOf(name) !== i);
  if (repeated !== undefined) throw new DataError(`The option covariates lists ${JSON.stringify(repeated)} twice.`);
  return covariates;
}

/** Sorts the checked terms by where their coefficients go, and sets up what a state tracks for them. */
export function compileTerms(
  specs: readonly TermSpec[],
  intercepts: Readonly<Record<InterceptKind, boolean>>,
  covariates: readonly string[],
): Terms {
  const each = (over: TermScope): TermSpec[] =>
    specs.filter((spec) => spec.over === over && spec.coefficient === "each");
  const featuresOver = (over: TermScope): FeatureSpec[] =>
    specs.filter((spec) => spec.over === over).map((spec) => spec.feature);
  const skills = each("skill");
  const items = each("item");
  const shared = specs.filter((spec) => spec.coefficient === "shared");
  const sharedLabels = [...shared.map((spec) => spec.label), ...covariates.map((name) => COVARIATE_PREFIX + name)];
  const widths = { skill: width(intercepts.skill, skills), item: width(intercepts.item, items) };
  return {
    intercepts,
    skills,
    items,
    shared,
    covariates,
    needs: {
      skills: needsOf(featuresOver("skill")),
      items: specs.some((spec) => spec.over === "item") ? needsOf(featuresOver("item")) : null,
      overall: specs.some((spec) => spec.over === "learner") ? needsOf(featuresOver("learner")) : null,
    },
    labels: { skills: skills.map((spec) => spec.label), items: items.map((spec) => spec.label), shared: sharedLabels },
    scratch: {
      skill: new Float64Array(widths.skill),
      item: new Float64Array(widths.item),
      shared: new Float64Array(sharedLabels.length),
    },
  };
}

/** Returns the number of values of a skill's or an item's block: its intercept, if any, then its terms. */
function width(intercept: boolean, terms: readonly TermSpec[]): number {
  return (intercept ? 1 : 0) + terms.length;
}

/** Returns the number of columns a block of skills or items has in a design. */
export function blockWidth(terms: Terms, kind: InterceptKind): number {
  return width(terms.intercepts[kind], kind === "skill" ? terms.skills : terms.items);
}

/**
 * Receives the values that multiply an attempt's coefficients. A block holds the intercept's 1, if the model has the
 * intercept, then the value of each term with a coefficient for each skill or item, in order. The arrays are the model's
 * work arrays: they are valid until the call returns.
 */
export interface RowVisitor {
  /** Called for each of the attempt's skills. */
  skill(skill: string, values: Float64Array): void;
  /** Called once, for the attempt's item, when the model has item coefficients. */
  item(item: string, values: Float64Array): void;
  /** Called once, when the model has shared coefficients, with their values: the terms, then the covariates. */
  shared(values: Float64Array): void;
}

/**
 * Passes `visitor` the values of an attempt's terms from the learner's state before it: what a prediction multiplies by
 * coefficients, and a fit puts in a row of its design. The attempt, its state and its covariates are checked.
 */
export function visitRow(terms: Terms, state: LogisticState, attempt: Attempt, visitor: RowVisitor): void {
  const { scratch, covariates } = terms;
  scratch.shared.fill(0);
  visitSkills(terms, state, attempt, visitor);
  const itemHistory = entry(state.items, attempt.item);
  visitItem(terms, state, attempt, itemHistory, visitor);
  let at = 0;
  for (const term of terms.shared) {
    if (term.over !== "skill") {
      const history = term.over === "item" ? itemHistory : state.overall;
      scratch.shared[at] = featureValue(term.feature, history, undefined, attempt.time, state.practice ?? 0);
    }
    at++;
  }
  for (const name of covariates) scratch.shared[at++] = entry(attempt.covariates, name) ?? 0;
  if (scratch.shared.length > 0) visitor.shared(scratch.shared);
}

/** Visits the block of each of the attempt's skills, and adds its terms with shared coefficients to theirs. */
function visitSkills(terms: Terms, state: LogisticState, attempt: Attempt, visitor: RowVisitor): void {
  const { scratch, skills, shared } = terms;
  const { time } = attempt;
  const practice = state.practice ?? 0;
  const first = terms.intercepts.skill ? 1 : 0;
  for (const skill of attempt.skills) {
    const history = entry(state.skills, skill);
    const exposure = entry(state.exposures, skill);
    if (first === 1) scratch.skill[0] = 1;
    let at = first;
    for (const term of skills) scratch.skill[at++] = featureValue(term.feature, history, exposure, time, practice);
    at = 0;
    for (const term of shared) {
      if (term.over === "skill") {
        scratch.shared[at] = (scratch.shared[at] ?? 0) + featureValue(term.feature, history, exposure, time, practice);
      }
      at++;
    }
    visitor.skill(skill, scratch.skill);
  }
}

/** Visits the block of the attempt's item, when the model has coefficients for items. */
function visitItem(
  terms: Terms,
  state: LogisticState,
  attempt: Attempt,
  history: SkillHistory | undefined,
  visitor: RowVisitor,
): void {
  const { scratch, items } = terms;
  if (scratch.item.length === 0) return;
  const first = terms.intercepts.item ? 1 : 0;
  if (first === 1) scratch.item[0] = 1;
  let at = first;
  for (const term of items) {
    scratch.item[at++] = featureValue(term.feature, history, undefined, attempt.time, state.practice ?? 0);
  }
  visitor.item(attempt.item, scratch.item);
}

/**
 * Checks that an attempt has every covariate that the model reads.
 *
 * @param label - Names the attempt in the error message, such as `Event 3`.
 * @throws `DataError` naming the first covariate missing.
 */
export function checkCovariates(attempt: Attempt, label: string, names: readonly string[]): void {
  for (const name of names) {
    if (entry(attempt.covariates, name) === undefined) {
      throw new DataError(`${label} needs the covariate "${name}", which the model reads.`);
    }
  }
}
