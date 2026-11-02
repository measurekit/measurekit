import type { ResponseMatrix } from "../data/internal.js";
import type { DifGroup } from "./groups.js";
import type { LikelihoodRatioTest } from "./likelihood-ratio.js";
import { logisticTests } from "./logistic-regression.js";
import { type MatchingOptions, matchedAnalysis } from "./matching.js";
import { type DifAnalysis, significant } from "./purification.js";

/** Logistic discriminant function tests of one item. */
export interface DiscriminantDif {
  /** The item id. */
  readonly id: string;
  /** Uniform DIF: the item's response joins the regression of the group on the score, model 1 against model 0. */
  readonly uniform: LikelihoodRatioTest;
  /** Nonuniform DIF: the product of score and response, model 2 against model 1. */
  readonly nonuniform: LikelihoodRatioTest;
  /** Both: the response and the product, model 2 against model 0. */
  readonly both: LikelihoodRatioTest;
}

/**
 * Returns the logistic discriminant function DIF tests of every item, for items of any number of categories.
 *
 * @remarks
 * Three logistic regressions of the group, G = 1 in the focal group and 0 in the reference group, on the matching score
 * S and the item's response Y, its integer score: model 0, β0 + β1 S; model 1, adding β2 Y; model 2, adding β3 S Y.
 * The tests are likelihood-ratio tests with 1, 1 and 2 degrees of freedom whatever the number of categories, and the R²
 * change is Nagelkerke's, with the deviance of the intercept alone as D0. They are fitted by Newton's method, as
 * `logisticRegressionDif`'s regressions, with the outcome and a predictor exchanged.
 *
 * The matching score and the persons are those of `ordinalLogisticRegressionDif`: the total score over the matching
 * items, all items or, with `purify`, the anchors, and the studied item, scaled from the matching items each person
 * answered (see the guide); or with `matching`, the given score of the persons who answered the item and have one. A
 * test is `null` for an item these persons answered alike, that has persons of one group only or whose matching scores
 * are all equal, and where a model's terms are collinear. Purification flags items whose test of both terms is
 * significant at 0.05.
 *
 * @param matrix - The responses of the persons.
 * @param groups - The group of each person, in the order of the matrix.
 * @param options - A matching score for each person instead of the total score, and whether to purify it.
 * @returns The tests of every item, and the purification when it was asked for.
 * @throws `DataError` when the matrix or the groups are invalid, either group has no person with a response, `purify`
 * is not a boolean, `matching` is not an array of finite numbers and `null` with one entry per person, or `purify` is
 * `true` with `matching`.
 * @see {@link https://doi.org/10.1111/j.1745-3984.1993.tb01069.x | Miller and Spray (1993)}
 */
export function logisticDiscriminantDif(
  matrix: ResponseMatrix,
  groups: readonly DifGroup[],
  options: MatchingOptions = {},
): DifAnalysis<DiscriminantDif> {
  return matchedAnalysis(
    matrix,
    groups,
    options,
    "Logistic discriminant DIF",
    ({ responses, scores, focal }) => logisticTests({ outcomes: focal, scores, predictors: responses }),
    (item) => significant(item.both.significance),
  );
}
