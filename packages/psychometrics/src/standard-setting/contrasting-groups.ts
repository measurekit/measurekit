import { checkArray, checkOptions, DataError, describe, type FieldNames } from "@measurekit/core";
import { checkOpenUnitInterval } from "../checks.js";
import { newton } from "../math/internal.js";

/** Options of {@link contrastingGroupsCutScores}. */
export interface ContrastingGroupsOptions {
  /**
   * The probability of the higher category at a cut, between 0 and 1, exclusive.
   *
   * @defaultValue `0.5`
   */
  readonly probability?: number;
}

const CONTRASTING_GROUPS_OPTIONS: FieldNames<ContrastingGroupsOptions> = { probability: true };

/** The cut scores between adjacent categories. */
export interface ContrastingGroupsCutScores {
  /**
   * One per boundary between adjacent categories, from the boundary of categories 0 and 1 up; `null` where the groups
   * separate perfectly, so that no logistic regression has a maximum.
   */
  readonly cutScores: readonly (number | null)[];
  /** Whether every boundary's logistic regression converged: `false` when one separates. */
  readonly converged: boolean;
}

const DEFAULT_PROBABILITY = 0.5;

/**
 * Returns a cut score for each boundary between adjacent categories: the score where a logistic regression of "at or
 * above the boundary" on the score reaches `probability`.
 *
 * @remarks
 * For the boundary k, each person's outcome is 1 at a category of k or above and 0 below, and the maximum likelihood fit
 * of logit P(1 | x) = β₀ + β₁ x gives the cut (logit(probability) - β₀) / β₁, as R's `glm(family = binomial)`. Two
 * categories, masters and non-masters, give the contrasting-groups method (Livingston and Zieky, 1982); more give the
 * Body of Work method, its categories a panel's judgments of collections of work and the scores theirs (Kingston et
 * al., 2001). Where the scores of the two sides of a boundary do not overlap, the likelihood has no maximum: that cut
 * is `null` and `converged` is `false`.
 *
 * @param scores - Each person's total score, or a work's score.
 * @param categories - Each person's category, an integer from 0, in the order of `scores`.
 * @param options - The probability of the higher category at a cut.
 * @returns The cut score of each boundary and whether every fit converged.
 * @throws `DataError` when the scores are not finite numbers, the categories not integers from 0, the two lists differ
 * in length, fewer than two categories occur, a category from 0 to the highest has no person, or `probability` is not
 * between 0 and 1.
 * @see {@link https://doi.org/10.4324/9781410600264 | Kingston, Kahl, Sweeney and Bay (2001)}
 */
export function contrastingGroupsCutScores(
  scores: readonly number[],
  categories: readonly number[],
  options: ContrastingGroupsOptions = {},
): ContrastingGroupsCutScores {
  checkOptions(options, CONTRASTING_GROUPS_OPTIONS);
  const { probability = DEFAULT_PROBABILITY } = options;
  checkOpenUnitInterval(probability, "The option probability");
  const highest = checkGroups(scores, categories);
  const center = scores.reduce((sum, score) => sum + score, 0) / scores.length;
  const centered = Float64Array.from(scores, (score) => score - center);
  const target = Math.log(probability / (1 - probability));
  let converged = true;
  const cutScores = Array.from({ length: highest }, (_, k) => {
    const outcomes = Uint8Array.from(categories, (category) => (category > k ? 1 : 0));
    if (separates(centered, outcomes)) {
      converged = false;
      return null;
    }
    const fit = logisticFit(centered, outcomes);
    converged &&= fit.converged;
    return center + (target - fit.intercept) / fit.slope;
  });
  return { cutScores, converged };
}

/**
 * Checks the scores and categories and returns the highest category.
 *
 * @throws `DataError` when they are invalid, or a category from 0 to the highest has no person.
 */
function checkGroups(scores: readonly number[], categories: readonly number[]): number {
  checkArray(scores, "The scores must be an array of numbers");
  checkArray(categories, "The categories must be an array of integers from 0");
  if (scores.length !== categories.length) {
    throw new DataError(
      `The scores and the categories need one entry for each person, found ${scores.length} and ${categories.length}.`,
    );
  }
  scores.forEach((score, person) => {
    if (!Number.isFinite(score)) {
      throw new DataError(`Person ${person + 1} needs a finite score, found ${describe(score)}.`);
    }
  });
  const present = new Set<number>();
  let highest = 0;
  categories.forEach((category, person) => {
    if (!(Number.isSafeInteger(category) && category >= 0)) {
      throw new DataError(
        `Person ${person + 1} needs a category that is an integer from 0, found ${describe(category)}.`,
      );
    }
    present.add(category);
    highest = Math.max(highest, category);
  });
  if (highest === 0) throw new DataError("Cut scores need persons in at least two categories.");
  // A category nobody holds would give its two boundaries one cut. Requiring every category up to the highest also
  // keeps the boundaries, and their work, as many as the categories that occur, whatever the highest number.
  if (present.size <= highest) {
    let empty = 0;
    while (present.has(empty)) empty++;
    throw new DataError(
      `Cut scores need persons in every category from 0 to the highest, ${highest}; category ${empty} has none.`,
    );
  }
  return highest;
}

/** Returns whether the scores of the outcomes 0 and 1 do not overlap, in either order. */
function separates(scores: Float64Array, outcomes: Uint8Array): boolean {
  const range = [Infinity, -Infinity, Infinity, -Infinity];
  scores.forEach((score, person) => {
    const offset = (outcomes[person] ?? 0) * 2;
    range[offset] = Math.min(range[offset] ?? Infinity, score);
    range[offset + 1] = Math.max(range[offset + 1] ?? -Infinity, score);
  });
  const [lowestZero = 0, highestZero = 0, lowestOne = 0, highestOne = 0] = range;
  return highestZero <= lowestOne || highestOne <= lowestZero;
}

/** A logistic regression on one predictor. */
interface LogisticFit {
  readonly intercept: number;
  readonly slope: number;
  readonly converged: boolean;
}

/**
 * Returns the maximum likelihood fit of logit P(1 | x) = β₀ + β₁ x by Newton's method, from the logit of the mean
 * outcome and a slope of 0.
 */
function logisticFit(scores: Float64Array, outcomes: Uint8Array): LogisticFit {
  const share = outcomes.reduce((sum, outcome) => sum + outcome, 0) / outcomes.length;
  const result = newton(
    (beta, gradient, hessian) => {
      const [intercept = 0, slope = 0] = beta;
      gradient.fill(0);
      hessian.fill(0);
      let value = 0;
      scores.forEach((score, person) => {
        const eta = intercept + slope * score;
        const p = 1 / (1 + Math.exp(-eta));
        // log(1 + e^η) - y η, without overflow at large |η|.
        value += Math.max(eta, 0) + Math.log1p(Math.exp(-Math.abs(eta))) - (outcomes[person] ?? 0) * eta;
        const residual = p - (outcomes[person] ?? 0);
        const weight = p * (1 - p);
        gradient[0] = (gradient[0] ?? 0) + residual;
        gradient[1] = (gradient[1] ?? 0) + residual * score;
        hessian[0] = (hessian[0] ?? 0) + weight;
        hessian[1] = (hessian[1] ?? 0) + weight * score;
        hessian[3] = (hessian[3] ?? 0) + weight * score * score;
      });
      hessian[2] = hessian[1] ?? 0;
      return value;
    },
    [Math.log(share / (1 - share)), 0],
  );
  return { intercept: result.x[0] ?? 0, slope: result.x[1] ?? 0, converged: result.converged };
}
