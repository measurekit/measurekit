import { backtrack, logistic, logLogistic } from "@measurekit/core";
import type { ResponseMatrix } from "../data/internal.js";
import { invert } from "../math/internal.js";
import type { DifGroup } from "./groups.js";
import { type LikelihoodRatioTest, likelihoodRatioTest, undefinedTest } from "./likelihood-ratio.js";
import { type LogisticRegressionDif, MAX_ITERATIONS, STEP_TOLERANCE } from "./logistic-regression.js";
import { type MatchedCases, type MatchingOptions, matchedAnalysis } from "./matching.js";
import { type DifAnalysis, significant } from "./purification.js";

/** Ordinal logistic regression DIF tests of one polytomous item. */
export interface OrdinalLogisticDif extends LogisticRegressionDif {
  /** The relative change of the score coefficient when the group joins the model: |β1(1) - β1(0)| / |β1(0)|. */
  readonly slopeChange: number | null;
}

/** The most times a step that leaves the thresholds unordered or raises the deviance is halved. */
const MAX_HALVES = 30;
/** The relative increase of the deviance a step may show: the rounding error of a sum over all persons. */
const DEVIANCE_SLACK = 1e-12;

/**
 * Returns the ordinal logistic regression DIF tests of every polytomous item, as lordif's `rundif` computes them for
 * one matching criterion.
 *
 * @remarks
 * Three cumulative logit models of an item's responses, with the matching score S and G = 1 in the focal group and 0 in
 * the reference group: P(Y ≥ k) = logistic(α_k + β1 S + β2 G + β3 S G), the thresholds α_1 > ... > α_(K-1) ordered.
 * Model 0 has β1 only, model 1 adds β2 and model 2 adds β3. K is the number of categories the item's persons used,
 * whichever categories the matrix could hold. Each model is fitted by Newton's method from the thresholds of the null
 * model, halving a step that leaves the thresholds unordered or raises the deviance. The tests are likelihood-ratio
 * tests; the R² change is Nagelkerke's, where lordif reports McFadden's by default.
 *
 * The matching score is the total score over the matching items, all items or, with `purify`, the anchors, and the
 * studied item. Each item then uses the persons who answered it and another matching item, their score scaled from the
 * matching items they answered (see the guide). With `matching`, each item uses the persons who answered it and have a
 * score. A test is `null` for an item these persons answered alike, that has persons of one group only or for which
 * the matching scores are all equal, and where a model's Hessian is singular. Purification flags items whose test of
 * both terms is significant at 0.05.
 *
 * @param matrix - The responses of the persons.
 * @param groups - The group of each person, in the order of the matrix.
 * @param options - A matching score for each person instead of the total score, and whether to purify it.
 * @returns The tests of every item, and the purification when it was asked for.
 * @throws `DataError` when the matrix or the groups are invalid, either group has no person with a response, `purify`
 * is not a boolean, `matching` is not an array of finite numbers and `null` with one entry per person, or `purify` is
 * `true` with `matching`.
 * @see {@link https://doi.org/10.1097/01.mlr.0000245183.28384.ed | Crane, Gibbons, Jolley and van Belle (2006)}
 */
export function ordinalLogisticRegressionDif(
  matrix: ResponseMatrix,
  groups: readonly DifGroup[],
  options: MatchingOptions = {},
): DifAnalysis<OrdinalLogisticDif> {
  return matchedAnalysis(matrix, groups, options, "Ordinal logistic regression DIF", itemTests, (item) =>
    significant(item.both.significance),
  );
}

/** The categories an item's persons used, with the persons' responses as ranks among them. */
interface UsedCategories {
  /** Each person's response as its rank among the used categories, from 0. */
  readonly ranks: Int32Array;
  /** The number of persons in each used category. */
  readonly counts: Float64Array;
}

function usedCategories(responses: Int32Array): UsedCategories {
  const used = [...new Set(responses)].sort((a, b) => a - b);
  const rankOf = new Map(used.map((response, rank) => [response, rank]));
  const ranks = Int32Array.from(responses, (response) => rankOf.get(response) ?? 0);
  const counts = new Float64Array(used.length);
  for (const rank of ranks) counts[rank] = (counts[rank] ?? 0) + 1;
  return { ranks, counts };
}

/** Returns the three likelihood-ratio tests of an item, `null` where the item's persons cannot show DIF. */
function itemTests(cases: MatchedCases): Omit<OrdinalLogisticDif, "id"> {
  const { ranks, counts } = usedCategories(cases.responses);
  const count = ranks.length;
  const focal = cases.focal.reduce((sum, flag) => sum + flag, 0);
  const center = cases.scores.reduce((sum, score) => sum + score, 0) / count;
  const varies = cases.scores.some((score) => score !== cases.scores[0]);
  if (counts.length < 2 || focal === 0 || focal === count || !varies) {
    return { uniform: undefinedTest(1), nonuniform: undefinedTest(1), both: undefinedTest(2), slopeChange: null };
  }
  // The deviance of the thresholds alone, -2 Σ n_k ln(n_k / n).
  const nullDeviance = -2 * counts.reduce((sum, persons) => sum + persons * Math.log(persons / count), 0);
  const [score, group, interaction] = [1, 2, 3].map((terms) =>
    fitModel({ ranks, scores: cases.scores, focal: cases.focal }, counts, center, terms),
  );
  const test = (
    smaller: OrdinalFit | null | undefined,
    larger: OrdinalFit | null | undefined,
    degreesOfFreedom: number,
  ): LikelihoodRatioTest =>
    smaller && larger
      ? likelihoodRatioTest(smaller.deviance, larger.deviance, degreesOfFreedom, nullDeviance, count)
      : undefinedTest(degreesOfFreedom);
  return {
    uniform: test(score, group, 1),
    nonuniform: test(group, interaction, 1),
    both: test(score, interaction, 2),
    slopeChange: slopeChange(score, group),
  };
}

function slopeChange(score: OrdinalFit | null | undefined, group: OrdinalFit | null | undefined): number | null {
  if (!(score && group && score.slope !== 0)) return null;
  const change = Math.abs((group.slope - score.slope) / score.slope);
  return Number.isFinite(change) ? change : null;
}

/** The persons of a regression, with each response as its rank among the categories used. */
interface RankedCases {
  readonly ranks: Int32Array;
  readonly scores: Float64Array;
  readonly focal: Uint8Array;
}

/** A fitted model: its deviance, -2 ln L, and the coefficient of the matching score. */
interface OrdinalFit {
  readonly deviance: number;
  readonly slope: number;
}

/** A cumulative logit regression during fitting, with scratch space for the covariates and derivatives of one person. */
interface Regression {
  /** The number of thresholds: one less than the categories. */
  readonly thresholds: number;
  /** The number of covariates among S, G and S G, taken in that order. */
  readonly terms: number;
  /** The thresholds α_1 to α_(K-1), then the coefficients of the covariates. */
  readonly parameters: Float64Array;
  /** The parameters at the start of a step. */
  readonly saved: Float64Array;
  readonly gradient: Float64Array;
  readonly hessian: Float64Array;
  /** ln(1 - e^-(α_k - α_(k+1))) at `k - 1`: the log of the gap between neighboring cumulative probabilities. */
  readonly gaps: Float64Array;
  readonly covariates: Float64Array;
  /** The parameter, base variable (upper threshold, lower threshold, linear predictor) and factor of each term. */
  readonly entries: { index: Int32Array; variable: Uint8Array; factor: Float64Array };
  /** The first and second derivatives of a person's log probability in the three base variables. */
  readonly base: { gradient: Float64Array; hessian: Float64Array };
}

/**
 * Fits the cumulative logit model with the first `terms` of S, G and S G by Newton's method, from the thresholds of the
 * null model and zero coefficients, and returns its deviance and score coefficient, or `null` for a singular Hessian
 * at the start. The score is centered, which leaves the deviance and the score coefficient unchanged.
 */
function fitModel(cases: RankedCases, counts: Float64Array, center: number, terms: number): OrdinalFit | null {
  const thresholds = counts.length - 1;
  const size = thresholds + terms;
  const regression: Regression = {
    thresholds,
    terms,
    parameters: new Float64Array(size),
    saved: new Float64Array(size),
    gradient: new Float64Array(size),
    hessian: new Float64Array(size * size),
    gaps: new Float64Array(Math.max(thresholds - 1, 0)),
    covariates: new Float64Array(terms),
    entries: {
      index: new Int32Array(terms + 2),
      variable: new Uint8Array(terms + 2),
      factor: new Float64Array(terms + 2),
    },
    base: { gradient: new Float64Array(3), hessian: new Float64Array(9) },
  };
  // α_k is the log odds of a response in category k or above.
  let above = cases.ranks.length;
  for (let k = 0; k < thresholds; k++) {
    above -= counts[k] ?? 0;
    regression.parameters[k] = Math.log(above / (cases.ranks.length - above));
  }
  let value = evaluate(cases, center, regression);
  for (let iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
    const direction = newtonDirection(regression);
    if (direction === null) {
      if (iteration === 0) return null;
      break;
    }
    const step = halvedStep(cases, center, regression, direction, value);
    if (step === null) break;
    value = step.value;
    if (step.size * maxAbs(direction) <= STEP_TOLERANCE) break;
  }
  return { deviance: 2 * value, slope: regression.parameters[thresholds] ?? 0 };
}

/**
 * Moves the parameters along the Newton direction, halving the step until the thresholds stay ordered and -ln L does
 * not rise, and returns the step and -ln L there; or restores the parameters and returns `null` when no step does.
 */
function halvedStep(
  cases: RankedCases,
  center: number,
  regression: Regression,
  direction: Float64Array,
  value: number,
): { size: number; value: number } | null {
  const { parameters, saved } = regression;
  saved.set(parameters);
  let trial = value;
  const size = backtrack(
    (length) => {
      for (let i = 0; i < parameters.length; i++) parameters[i] = (saved[i] ?? 0) - length * (direction[i] ?? 0);
      trial = ordered(regression) ? evaluate(cases, center, regression) : Number.POSITIVE_INFINITY;
      return trial;
    },
    value,
    0,
    { decrease: 0, slack: DEVIANCE_SLACK * Math.max(1, value), maxHalvings: MAX_HALVES },
  );
  if (size === null) {
    parameters.set(saved);
    return null;
  }
  return { size, value: trial };
}

function maxAbs(values: Float64Array): number {
  return values.reduce((largest, value) => Math.max(largest, Math.abs(value)), 0);
}

/** Returns whether the thresholds are finite and strictly decreasing, as P(Y ≥ k) must be in k. */
function ordered({ thresholds, parameters }: Regression): boolean {
  for (let k = 0; k < thresholds; k++) {
    const threshold = parameters[k] ?? Number.NaN;
    if (!Number.isFinite(threshold) || (k > 0 && threshold >= (parameters[k - 1] ?? 0))) return false;
  }
  return true;
}

/** Returns H⁻¹ g at the regression's parameters, or `null` for a singular Hessian. */
function newtonDirection({ gradient, hessian, parameters }: Regression): Float64Array | null {
  const size = parameters.length;
  const inverse = invert(hessian, size);
  if (inverse === null) return null;
  const direction = new Float64Array(size);
  for (let j = 0; j < size; j++) {
    for (let k = 0; k < size; k++)
      direction[j] = (direction[j] ?? 0) + (inverse[j * size + k] ?? 0) * (gradient[k] ?? 0);
  }
  return direction;
}

/** Writes the gradient and Hessian of -ln L at the regression's parameters and returns -ln L. */
function evaluate(cases: RankedCases, center: number, regression: Regression): number {
  const { terms, parameters, gaps, covariates } = regression;
  for (let k = 0; k < gaps.length; k++) gaps[k] = logOneMinusExp((parameters[k] ?? 0) - (parameters[k + 1] ?? 0));
  regression.gradient.fill(0);
  regression.hessian.fill(0);
  let value = 0;
  for (let person = 0; person < cases.ranks.length; person++) {
    const score = (cases.scores[person] ?? 0) - center;
    const group = cases.focal[person] ?? 0;
    covariates[0] = score;
    if (terms > 1) covariates[1] = group;
    if (terms > 2) covariates[2] = score * group;
    value += addPerson(regression, cases.ranks[person] ?? 0);
  }
  return value;
}

/**
 * Returns ln(1 - e^-gap) for gap > 0, accurately: by expm1 for a small gap and log1p for a large one.
 *
 * @see {@link https://cran.r-project.org/package=Rmpfr/vignettes/log1mexp-note.pdf | Mächler (2012)}
 */
function logOneMinusExp(gap: number): number {
  return gap < Math.LN2 ? Math.log(-Math.expm1(-gap)) : Math.log1p(-Math.exp(-gap));
}

/**
 * Adds a person in category `rank` with the regression's covariates to the gradient and Hessian of -ln L and returns
 * the person's term of -ln L.
 *
 * @remarks
 * P(Y = k) = F_k - F_(k+1) with F_k = logistic(α_k + η), η the linear predictor, F_0 = 1 and F_K = 0. Its log is
 * computed as ln F_k + ln (1 - F_(k+1)) + ln (1 - e^-(α_k - α_(k+1))), which does not cancel in the tails. ln P depends
 * on the parameters through three base variables, the upper threshold α_k, the lower α_(k+1) and η, so the gradient and
 * Hessian of ln P are those of the base variables, carried to the parameters by their coefficients.
 */
function addPerson(regression: Regression, rank: number): number {
  const { thresholds, terms, parameters, covariates, gaps } = regression;
  let predictor = 0;
  for (let j = 0; j < terms; j++) predictor += (parameters[thresholds + j] ?? 0) * (covariates[j] ?? 0);
  // The thresholds above and below the category, absent for the lowest and highest.
  const upper = rank >= 1 ? (parameters[rank - 1] ?? 0) + predictor : null;
  const lower = rank < thresholds ? (parameters[rank] ?? 0) + predictor : null;
  const logProbability =
    (upper === null ? 0 : logLogistic(upper)) +
    (lower === null ? 0 : logLogistic(-lower)) +
    (upper !== null && lower !== null ? (gaps[rank - 1] ?? 0) : 0);
  setBase(regression.base, upper, lower, logProbability);
  addToDerivatives(regression, rank, upper !== null, lower !== null);
  return -logProbability;
}

/**
 * Returns f(x) / P, or f'(x) / P with `derivative`, for the logistic density f, with 0 where the person's category has
 * no threshold on that side.
 */
function densityRatio(x: number | null, logProbability: number, derivative: boolean): number {
  if (x === null) return 0;
  const ratio = Math.exp(logLogistic(x) + logLogistic(-x) - logProbability);
  return derivative ? ratio * (1 - 2 * logistic(x)) : ratio;
}

/** Writes the gradient and Hessian of ln P in the base variables: the upper threshold, the lower, and η. */
function setBase(
  { gradient, hessian }: Regression["base"],
  upper: number | null,
  lower: number | null,
  logProbability: number,
): void {
  const upperRatio = densityRatio(upper, logProbability, false);
  const lowerRatio = densityRatio(lower, logProbability, false);
  const upperSlope = densityRatio(upper, logProbability, true);
  const lowerSlope = densityRatio(lower, logProbability, true);
  const total = upperRatio - lowerRatio;
  gradient.set([upperRatio, -lowerRatio, total]);
  const cross = [
    upperSlope - upperRatio * upperRatio,
    upperRatio * lowerRatio,
    upperSlope - upperRatio * total,
    upperRatio * lowerRatio,
    -lowerSlope - lowerRatio * lowerRatio,
    -lowerSlope + lowerRatio * total,
    upperSlope - upperRatio * total,
    -lowerSlope + lowerRatio * total,
    upperSlope - lowerSlope - total * total,
  ];
  hessian.set(cross);
}

/** Adds the person's gradient and Hessian of ln P, with the opposite sign, to those of -ln L. */
function addToDerivatives(regression: Regression, rank: number, hasUpper: boolean, hasLower: boolean): void {
  const { thresholds, terms, covariates, entries } = regression;
  let count = 0;
  if (hasUpper) setEntry(entries, count++, rank - 1, 0, 1);
  if (hasLower) setEntry(entries, count++, rank, 1, 1);
  for (let j = 0; j < terms; j++) setEntry(entries, count++, thresholds + j, 2, covariates[j] ?? 0);
  for (let e = 0; e < count; e++) subtractEntry(regression, e, count);
}

/** Subtracts the derivatives of ln P in the parameter of entry `e` from the gradient and Hessian row of -ln L. */
function subtractEntry({ entries, base, gradient, hessian, parameters }: Regression, e: number, count: number): void {
  const row = entries.index[e] ?? 0;
  const variable = entries.variable[e] ?? 0;
  const factor = entries.factor[e] ?? 0;
  gradient[row] = (gradient[row] ?? 0) - factor * (base.gradient[variable] ?? 0);
  for (let f = 0; f < count; f++) {
    const at = row * parameters.length + (entries.index[f] ?? 0);
    const second = factor * (entries.factor[f] ?? 0) * (base.hessian[3 * variable + (entries.variable[f] ?? 0)] ?? 0);
    hessian[at] = (hessian[at] ?? 0) - second;
  }
}

function setEntry(entries: Regression["entries"], at: number, index: number, variable: number, factor: number): void {
  entries.index[at] = index;
  entries.variable[at] = variable;
  entries.factor[at] = factor;
}
