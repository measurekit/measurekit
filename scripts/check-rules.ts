import { spawnSync } from "node:child_process";
import {
  cpSync,
  existsSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";

/**
 * A rule of the code that a test must guard. Replacing `find` with `replace` in `file` removes the rule and leaves valid
 * code, and the tests that `titles` selects in the file `test` must then fail: a test that still passes does not guard
 * the rule.
 */
export interface Rule {
  /** What the rule says, as a sentence. */
  readonly rule: string;
  /** The source file that holds the rule, relative to the repository root. */
  readonly file: string;
  /** The text of the file that implements the rule. It occurs once. */
  readonly find: string;
  /** The text that stands in its place without the rule, and does the rest. */
  readonly replace: string;
  /** The test file that guards the rule, relative to the repository root. */
  readonly test: string;
  /** A pattern of the titles of the tests that guard the rule. */
  readonly titles: string;
}

/** The rules that keep untrusted input from being misread or from costing what its size does not, each with its test. */
export const RULES: readonly Rule[] = [
  {
    rule: "An option a function does not take, or a field a record of settings does not have, is refused, as a misspelled name would leave a default in place.",
    file: "packages/core/src/checks.ts",
    find: "!Object.hasOwn(names, name)",
    replace: "false",
    test: "packages/core/test/checks.test.ts",
    titles: "refuse a name that is not an option",
  },
  {
    rule: "A response above 2³¹ - 1 is refused, as the 32-bit arrays of the analyses would wrap it into another category.",
    file: "packages/psychometrics/src/data/response.ts",
    find: "value <= MAX_RESPONSE",
    replace: "true",
    test: "packages/psychometrics/test/data.test.ts",
    titles: "2³¹|invalid responses",
  },
  {
    rule: "A person with a missing response to a studied or matching item leaves a SIBTEST analysis, and -1 never counts as a score.",
    file: "packages/psychometrics/src/dif/sibtest.ts",
    find: "if (response < 0) return null;",
    replace: "",
    test: "packages/psychometrics/test/dif.test.ts",
    titles: "missing response to a matching|on verbal, sparse",
  },
  {
    rule: "A bundle tests the items of the matrix alone, and a bundle's item is not read as another.",
    file: "packages/psychometrics/src/dif/sibtest.ts",
    find: "if (!positions.has(item))",
    replace: "if (false)",
    test: "packages/psychometrics/test/dif.test.ts",
    titles: "rejects bundles",
  },
  {
    rule: "A bundle counts an item once, as a repeat would weigh it twice in the studied score.",
    file: "packages/psychometrics/src/dif/sibtest.ts",
    find: "if (seen.has(item))",
    replace: "if (false)",
    test: "packages/psychometrics/test/dif.test.ts",
    titles: "rejects bundles",
  },
  {
    rule: "The DIF statistics table the matching scores that occur, so one far response costs no table up to it.",
    file: "packages/psychometrics/src/dif/groups.ts",
    find: "const groupOf = ranksOf(scores);",
    replace:
      "const groupOf = new Map(Array.from({ length: Math.max(...Array.from(scores)) + 1 }, (_, score) => [score, score]));",
    test: "packages/psychometrics/test/dif.test.ts",
    titles: "tables only the",
  },
  {
    rule: "An item a group lacks a category of has a null likelihood-ratio test, as its parameters in that group are not identified.",
    file: "packages/psychometrics/src/dif/irt-likelihood-ratio.ts",
    find: "if (!testable[item]) return undefinedTest;",
    replace: "",
    test: "packages/psychometrics/test/dif.test.ts",
    titles: "lacks a category",
  },
  {
    rule: "Raju's areas take items that estimate a and b alone, as curves with unequal asymptotes enclose an infinite area.",
    file: "packages/psychometrics/src/dif/raju-area.ts",
    find: "if (names.join() !== AREA_PARAMETERS) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/dif.test.ts",
    titles: "an item with an estimated asymptote",
  },
  {
    rule: "Lord's and Raju's statistics hold the asymptotes alone, as the link moves a and b away from their held values.",
    file: "packages/psychometrics/src/dif/linked-calibrations.ts",
    find: "if (!UNLINKED_PARAMETERS.has(name)) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/dif.test.ts",
    titles: "refuses held parameters that the link moves",
  },
  {
    rule: "Differential functioning compares each item with itself: the focal list holds the reference items in their order.",
    file: "packages/psychometrics/src/dif/differential-functioning.ts",
    find: "if (item.id !== reference[i]?.id) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/dif.test.ts",
    titles: "differ in their items or order",
  },
  {
    rule: "A score group with persons of one group adds nothing to the standard error of Liu and Agresti, which divides by each group's persons.",
    file: "packages/psychometrics/src/dif/liu-agresti.ts",
    find: "if (reference === 0 || focal === 0) continue;",
    replace: "",
    test: "packages/psychometrics/test/dif.test.ts",
    titles: "Liu and Agresti's standard error",
  },
  {
    rule: "Conditional maximum likelihood needs a response from the persons it uses in every category of each item, as an empty category has an infinite threshold.",
    file: "packages/psychometrics/src/irt/conditional-likelihood.ts",
    find: "checkItems(matrix.itemIds, highest, model, tallies);",
    replace: "",
    test: "packages/psychometrics/test/irt.test.ts",
    titles: "rejects an item every used person answered alike",
  },
  {
    rule: "A first row that is not an array sizes no item ids, as a length it claims would allocate that many.",
    file: "packages/psychometrics/src/data/response-matrix.ts",
    find: "Array.isArray(first) ? first.length : 0",
    replace: "first?.length ?? 0",
    test: "packages/psychometrics/test/data.test.ts",
    titles: "before its length sizes",
  },
  {
    rule: "The labels of groups are indexed by a map, so that many groups cost time in proportion to the persons.",
    file: "packages/psychometrics/src/irt/responses.ts",
    find: "const known = indices.get(label);",
    replace: "const known = [...indices.keys()].includes(label) ? indices.get(label) : undefined;",
    test: "packages/psychometrics/test/irt.test.ts",
    titles: "many groups",
  },
  {
    rule: "An unfolding calibration takes at most 1,000 items, as its starts hold sums and correlations of every pair of items and decompose them.",
    file: "packages/psychometrics/src/irt/unfolding-calibration.ts",
    find: 'checkMatrixItems(all.itemIds.length, "Unfolding calibration");',
    replace: "",
    test: "packages/psychometrics/test/irt.test.ts",
    titles: "more than 1,000 items, whose starts",
  },
  {
    rule: "A quadrature rule has at most 1,000 points, as an option of a few bytes would otherwise size every item's table over them.",
    file: "packages/psychometrics/src/irt/quadrature.ts",
    find: "if (points > MAX_POINTS) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/irt.test.ts",
    titles: "1,000 quadrature points",
  },
  {
    rule: "Standard errors take at most 2,000 estimated parameters, as their information is a dense matrix of every pair and its inverse takes their cube.",
    file: "packages/psychometrics/src/irt/standard-error-size.ts",
    find: "if (!withinStandardErrorSize(parameters)) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/irt.test.ts",
    titles: "bounds standard errors to 2,000",
  },
  {
    rule: "A range of theta is at most 10,000 wide, as the scan of modal estimation walks it in steps of 0.25.",
    file: "packages/psychometrics/src/irt/ability.ts",
    find: "if (!(upper - lower <= MAX_RANGE_WIDTH)) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/irt.test.ts",
    titles: "wider than 10,000",
  },
  {
    rule: "An item of a testlet enters a summed-score table once, inside its testlet, and is not also counted as an item in none.",
    file: "packages/psychometrics/src/irt/summed-score.ts",
    find: "if ((testletOf[i] ?? -1) >= 0) return;",
    replace: "",
    test: "packages/psychometrics/test/irt.test.ts",
    titles: "enumeration of the patterns and mirt's EAPsum_2.0",
  },
  {
    rule: "A true-score conversion table refuses testlets, which its curve and information would leave out unseen.",
    file: "packages/psychometrics/src/scaling/conversion-table.ts",
    find: 'if (method === "true-score" && testlets !== null) {',
    replace: "if (false) {",
    test: "packages/psychometrics/test/scaling.test.ts",
    titles: "refuses testlets for the true score",
  },
  {
    rule: "Conditional maximum likelihood checks the categories that occur before an item's highest response sizes its parameters, which a far response would overflow.",
    file: "packages/psychometrics/src/irt/conditional-likelihood.ts",
    find: "checkItems(matrix.itemIds, highest, model, everyPerson);",
    replace: "",
    test: "packages/psychometrics/test/irt.test.ts",
    titles: "far above the others from the categories",
  },
  {
    rule: "Conditional maximum likelihood takes at most 2^29 products of its functions with two items left out in a Newton step, as they grow with the patterns of answered items times about the fourth power of their items.",
    file: "packages/psychometrics/src/irt/conditional-likelihood.ts",
    find: "if (products > MAX_PAIR_PRODUCTS) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/irt.test.ts",
    titles: "products of the functions with two items left out",
  },
  {
    rule: "Every category below an item's highest response has a response.",
    file: "packages/psychometrics/src/irt/responses.ts",
    find: "if (counts.size <= highest) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/irt.test.ts",
    titles: "rejects an item without responses",
  },
  {
    rule: "A GGUM item needs a slope above 0, as a slope of 0 makes every category equally likely and a negative one reverses the item.",
    file: "packages/psychometrics/src/irt/unfolding.ts",
    find: "Number.isFinite(a) &&\n    a > 0 &&",
    replace: "Number.isFinite(a) &&",
    test: "packages/psychometrics/test/irt.test.ts",
    titles: "rejects a slope that is not above 0",
  },
  {
    rule: "Each person has one weight, as a list of another length would weigh persons by the weights of others.",
    file: "packages/psychometrics/src/irt/responses.ts",
    find: "if (weights.length !== persons)",
    replace: "if (false)",
    test: "packages/psychometrics/test/irt.test.ts",
    titles: "rejects weights that are not a positive number",
  },
  {
    rule: "A person's weight is a finite number above 0, as 0 or less would drop or subtract the person and a weight that is not finite would void every sum.",
    file: "packages/psychometrics/src/irt/responses.ts",
    find: "checkPositive(weights[person], `The weight of person",
    replace: "(weights[person], `The weight of person",
    test: "packages/psychometrics/test/irt.test.ts",
    titles: "rejects weights that are not a positive number",
  },
  {
    rule: "Weights are divided by the largest before they are summed, so that weights near the largest double do not overflow to a scale of 0.",
    file: "packages/psychometrics/src/irt/responses.ts",
    find: "largest = Math.max(largest, weights[person] ?? 0);",
    replace: "largest = 1;",
    test: "packages/psychometrics/test/irt.test.ts",
    titles: "rescales weights of any size",
  },
  {
    rule: "An item is fixed by an entry of its own, and not by a property of every object.",
    file: "packages/psychometrics/src/irt/calibration-setup.ts",
    find: "Object.hasOwn(fixed, id) ? fixed[id] : undefined",
    replace: "fixed[id]",
    test: "packages/psychometrics/test/irt.test.ts",
    titles: "names of the properties",
  },
  {
    rule: "A held parameter is one its item's model has, so that a misspelled name does not leave the parameter estimated unseen.",
    file: "packages/psychometrics/src/irt/fixed-parameters.ts",
    find: "if (specification === undefined) {",
    replace: "if (specification === undefined) continue;\n    if (false) {",
    test: "packages/psychometrics/test/irt.test.ts",
    titles: "reject unknown and fixed items, parameters the model lacks",
  },
  {
    rule: "A parameter that the items of a model share is not held for one of them, which would split it into two.",
    file: "packages/psychometrics/src/irt/fixed-parameters.ts",
    find: "if (specification.shared === true) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/irt.test.ts",
    titles: "reject unknown and fixed items, parameters the model lacks",
  },
  {
    rule: "A held threshold moves the starts of the other ordered parameters beyond it, so that a valid value is not refused for the starts around it.",
    file: "packages/psychometrics/src/irt/fixed-parameters.ts",
    find: "start[p] = (heldAt(nearest) ?? 0) + (before[p] ?? 0) - (before[nearest] ?? 0);",
    replace: "start[p] = heldAt(p) ?? before[p] ?? 0;",
    test: "packages/psychometrics/test/irt.test.ts",
    titles: "move the starts of the other ordered parameters",
  },
  {
    rule: "An ordered parameter between two held ones starts between them, so that a valid pair of values is not refused for the starts around them.",
    file: "packages/psychometrics/src/irt/fixed-parameters.ts",
    find: "if (lower !== undefined && upper !== undefined && lower < upper) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/irt.test.ts",
    titles: "move the starts of the other ordered parameters",
  },
  {
    rule: "A slope the pattern of loadings fixes at 0 is not held, as it would load the item on a dimension the pattern leaves out.",
    file: "packages/psychometrics/src/multidimensional/calibration-setup.ts",
    find: "if (held.has(name) && loads[d] !== true) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/multidimensional.test.ts",
    titles: "reject unknown and fixed items, parameters the item lacks",
  },
  {
    rule: "An item keeps a parameter free among those it estimates, as a slope the pattern fixes at 0 is not one: an item with none belongs in the option fixed.",
    file: "packages/psychometrics/src/multidimensional/calibration-setup.ts",
    find: "checkHeldParameters(id, model.name, estimable, held);",
    replace: "checkHeldParameters(id, model.name, specifications, held);",
    test: "packages/psychometrics/test/multidimensional.test.ts",
    titles: "reject unknown and fixed items, parameters the item lacks",
  },
  {
    rule: "A held value leaves a valid item, so that c = 1 is refused and not estimated around.",
    file: "packages/psychometrics/src/multidimensional/calibration-setup.ts",
    find: "if (held !== undefined) checkHeldItem(id, model, proportions, dimensions, design.loads[column] ?? [], held);",
    replace: "",
    test: "packages/psychometrics/test/multidimensional.test.ts",
    titles: "reject unknown and fixed items, parameters the item lacks",
  },
  {
    rule: "A held parameter is not estimated: it stays at its value, and leaves the standard errors and the criteria.",
    file: "packages/psychometrics/src/multidimensional/calibration-setup.ts",
    find: '&& held?.has(specifications[j]?.name ?? "") !== true',
    replace: "",
    test: "packages/psychometrics/test/multidimensional.test.ts",
    titles: "keep each held value exactly",
  },
  {
    rule: "The scores of an item with a held slope are taken over that slope's dimension, as its probabilities depend on it.",
    file: "packages/psychometrics/src/multidimensional/posterior-scores.ts",
    find: "if (free.length > 0) for (let d = 0; d < dimensions; d++) if ((vector[d] ?? 0) !== 0) set |= 1 << d;",
    replace: "",
    test: "packages/psychometrics/test/multidimensional.test.ts",
    titles: "match mirt with est = FALSE",
  },
  {
    rule: "A dimension whose slopes sum below 0 is turned with its loadings, correlations and groups, so that an estimator's mirror image at the same likelihood gives one calibration.",
    file: "packages/psychometrics/src/multidimensional/calibration.ts",
    find: "return sum < 0 ? -1 : 1;",
    replace: "return 1;",
    test: "packages/psychometrics/test/multidimensional.test.ts",
    titles: "turns a dimension whose slopes sum below zero",
  },
  {
    rule: "A dimension that a fixed item or a held slope orients is not turned, as the turn would change what was given.",
    file: "packages/psychometrics/src/multidimensional/calibration.ts",
    find: "if (oriented) turnable[d] = false;",
    replace: "",
    test: "packages/psychometrics/test/multidimensional.test.ts",
    titles: "keeps the orientation that fixed items give",
  },
  {
    rule: "A bifactor or two-tier calibration turns a dimension whose slopes sum below 0, as a multidimensional one does.",
    file: "packages/psychometrics/src/multidimensional/bifactor.ts",
    find: "return orientDimensions(calibration, turnableDimensions(solution.items, structure.design));",
    replace: "return calibration;",
    test: "packages/psychometrics/test/bifactor.test.ts",
    titles: "turns a dimension whose slopes sum below zero",
  },
  {
    rule: "A fixed items' covariance labels fixed items only, as an earlier covariance says nothing of an estimate made here.",
    file: "packages/psychometrics/src/irt/fixed-error.ts",
    find: "if (item.estimated) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/irt.test.ts",
    titles: "refuses a covariance without standard errors",
  },
  {
    rule: "A fixed items' covariance lists a parameter once, as twice would count its error twice.",
    file: "packages/psychometrics/src/irt/fixed-error.ts",
    find: "if (listed.has(key)) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/irt.test.ts",
    titles: "refuses a covariance without standard errors",
  },
  {
    rule: "A fixed items' covariance that adds a negative variance is refused, not taken as a smaller standard error.",
    file: "packages/psychometrics/src/irt/fixed-error.ts",
    find: "if (value < -VARIANCE_ROUNDING * terms) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/irt.test.ts",
    titles: "refuses a covariance without standard errors",
  },
  {
    rule: "An item with its held parameters is a valid item of its model, so that a lower asymptote at 1 is not calibrated.",
    file: "packages/psychometrics/src/irt/calibration-setup.ts",
    find: "checkItem({ model, parameters: model.fromVector(vector) },",
    replace: "void ({ model, parameters: model.fromVector(vector) },",
    test: "packages/psychometrics/test/irt.test.ts",
    titles: "reject unknown and fixed items, parameters the model lacks",
  },
  {
    rule: "An item has a threshold by an entry of its own, and not by a property of every object.",
    file: "packages/psychometrics/src/response-time/rapid-guessing.ts",
    find: "(Object.hasOwn(thresholds, id) ? (thresholds[id] ?? Number.NaN) : Number.NaN)",
    replace: "(thresholds[id] ?? Number.NaN)",
    test: "packages/psychometrics/test/response-time.test.ts",
    titles: "counts a response at its threshold",
  },
  {
    rule: "A feature of an item is an entry of its own, and not a property of every object.",
    file: "packages/psychometrics/src/explanatory/predict-item-parameter.ts",
    find: "Object.hasOwn(features, effect.name) ? features[effect.name] : undefined",
    replace: "features[effect.name]",
    test: "packages/psychometrics/test/explanatory.test.ts",
    titles: "rejects a regression, features or values",
  },
  {
    rule: "Kernel curves need a response in every category up to an item's highest, so that one far response costs no curves up to it.",
    file: "packages/psychometrics/src/fit/kernel-curves.ts",
    find: "if (counts.size <= highest) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/fit.test.ts",
    titles: "rejects an item without responses, an empty category below the highest, and fewer than two persons",
  },
  {
    rule: "A person without responses has no rank in kernel curves, so that an empty row moves no other person's normal score.",
    file: "packages/psychometrics/src/fit/kernel-curves.ts",
    find: "if (end === start) continue;",
    replace: "",
    test: "packages/psychometrics/test/fit.test.ts",
    titles: "leaves out a missing response and ranks no person without any",
  },
  {
    rule: "Classification cuts increase without overlapping, so that a category counts the cuts below the estimate.",
    file: "packages/psychometrics/src/cat/classification-stopping.ts",
    find: "if (previous !== undefined && !(current[0] >= previous[1] && current[1] > previous[1])) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/cat.test.ts",
    titles: "reject cuts that do not increase, and invalid levels and rates",
  },
  {
    rule: "The SPRT's error rates sum below 1, so that its lower bound is below 0 and its upper above, and no test is decided before a response.",
    file: "packages/psychometrics/src/cat/classification-stopping.ts",
    find: "if (!(alpha + beta < 1)) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/cat.test.ts",
    titles: "reject cuts that do not increase, and invalid levels and rates",
  },
  {
    rule: "lz* takes dichotomous items alone, so that a third category is not read as a wrong response.",
    file: "packages/psychometrics/src/fit/corrected-person-fit.ts",
    find: "if (count !== 2) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/fit.test.ts",
    titles: "rejects polytomous items, unknown estimators and invalid priors, and arguments of other types",
  },
  {
    rule: "The person fit indices of Mokken take 0/1 responses alone, so that a score of 2 is not read as a pass.",
    file: "packages/psychometrics/src/mokken/person-scalability.ts",
    find: 'checkDichotomous(matrix, "Person scalability");',
    replace: "",
    test: "packages/psychometrics/test/mokken.test.ts",
    titles: "rejects responses other than 0 and 1, and arguments of other types",
  },
  {
    rule: "A machine score that is not a category is refused without a scale, so that kappa never reads 2.5 as a category.",
    file: "packages/psychometrics/src/rater/automated-scores.ts",
    find: "if (categories && !isCategory(value)) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/rater.test.ts",
    titles: "rejects machine scores that are not categories without a scale",
  },
  {
    rule: "A routing point stands only after the last subject of its confidence, so tied subjects are never split by a threshold.",
    file: "packages/psychometrics/src/rater/routing-curve.ts",
    find: "if (next !== undefined && confidence[next] === confidence[subject]) continue;",
    replace: "",
    test: "packages/psychometrics/test/rater.test.ts",
    titles: "at or above each confidence",
  },
  {
    rule: "The knots of a stored isotonic mapping ascend and do not fall, so that a mapping out of order is not read.",
    file: "packages/psychometrics/src/rater/score-mapping.ts",
    find: "(previous === undefined || (knot.machine > previous.machine && knot.human >= previous.human))",
    replace: "true",
    test: "packages/psychometrics/test/rater.test.ts",
    titles: "mappings that are not valid",
  },
  {
    rule: "A population's standard deviations are non-negative numbers, so that a negative or a string is not read as a scale.",
    file: "packages/psychometrics/src/rater/mean-difference.ts",
    find: 'checkNonNegative(population.second, "The population\'s standard deviation of the second rater");',
    replace: "",
    test: "packages/psychometrics/test/rater.test.ts",
    titles: "rejects population deviations that are not non-negative numbers, and options of other types",
  },
  {
    rule: "A limited-information statistic stops at 60 items, before the moments, whose matrix grows with their fourth power.",
    file: "packages/psychometrics/src/fit/limited-information.ts",
    find: "if (items.length > MAX_MOMENT_ITEMS) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/fit.test.ts",
    titles: "refuses more than 60 items",
  },
  {
    rule: "A limited-information statistic refuses a calibration whose items do not record their estimated parameters, so that an anchor is not read as an estimated item.",
    file: "packages/psychometrics/src/fit/limited-information.ts",
    find: "if (!Array.isArray(estimated)) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/fit.test.ts",
    titles: "reads the estimated parameters from the items",
  },
  {
    rule: "The estimated parameters of an item name parameters of its model, so that a misspelled name does not count as an anchor.",
    file: "packages/psychometrics/src/fit/inputs.ts",
    find: "if (!known.has(name)) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/fit.test.ts",
    titles: "rejects an estimated list",
  },
  {
    rule: "A limited-information statistic leaves out a person with a missing response, who is not read as a score of 0.",
    file: "packages/psychometrics/src/fit/limited-information.ts",
    find: "matrix.responses.filter((row) => !row.includes(null))",
    replace: "matrix.responses",
    test: "packages/psychometrics/test/fit.test.ts",
    titles: "uses the persons who answered every item",
  },
  {
    rule: "A split of the persons of a Rasch test gives each person a label, so that no person is left out of the groups unseen.",
    file: "packages/psychometrics/src/fit/rasch-tests.ts",
    find: "if (split.length !== matrix.personIds.length) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/fit.test.ts",
    titles: "andersenTest.*refuses arguments of other types",
  },
  {
    rule: "A set of the Martin-Löf test names an item once, so that no item counts in two sets.",
    file: "packages/psychometrics/src/fit/rasch-tests.ts",
    find: 'typeof id === "string" && !placed.has(id)',
    replace: 'typeof id === "string"',
    test: "packages/psychometrics/test/fit.test.ts",
    titles: "refuses sets that do not hold every item once",
  },
  {
    rule: "The Martin-Löf test leaves out a person with a missing response, who is not read as a score of 0.",
    file: "packages/psychometrics/src/fit/rasch-tests.ts",
    find: ".filter((row) => !matrix.responses[row]?.includes(null))",
    replace: "",
    test: "packages/psychometrics/test/fit.test.ts",
    titles: "martinLofTest.*uses the persons who answered every item",
  },
  {
    rule: "A local-dependence table counts the persons who answered both items, and a missing response is not read as category 0.",
    file: "packages/psychometrics/src/fit/local-dependence.ts",
    find: "if (k === null || l === null) continue;",
    replace: "",
    test: "packages/psychometrics/test/fit.test.ts",
    titles: "localDependenceChiSquare.*uses the persons who answered both items",
  },
  {
    rule: "The fit of a diagnosis model stops at 60 items, before the moments, whose matrix grows with their fourth power.",
    file: "packages/psychometrics/src/cdm/model-fit.ts",
    find: "if (models.length > MAX_MOMENT_ITEMS) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/cdm.test.ts",
    titles: "refuses more than 60 items",
  },
  {
    rule: "The fit of a diagnosis model leaves out a person with a missing response, who is not read as a score of 0.",
    file: "packages/psychometrics/src/cdm/model-fit.ts",
    find: "matrix.responses.filter((row) => !row.includes(null))",
    replace: "matrix.responses",
    test: "packages/psychometrics/test/cdm.test.ts",
    titles: "uses the persons who answered every item",
  },
  {
    rule: "The two tables of a counterbalanced kernel equating have the same scores, so that no score of one group is read as another's.",
    file: "packages/psychometrics/src/equating/kernel.ts",
    find: "if (tables[0].probabilities.length !== tables[1].probabilities.length || tables[0].columns !== tables[1].columns) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/equating.test.ts",
    titles: "rejects invalid data",
  },
  {
    rule: "The two tables of a chained or post-stratification kernel equating have the same anchor scores.",
    file: "packages/psychometrics/src/equating/kernel.ts",
    find: "  neatTables(data.from, data.to);\n",
    replace: "",
    test: "packages/psychometrics/test/equating.test.ts",
    titles: "rejects invalid data",
  },
  {
    rule: "The standard error of a difference of kernel equatings pairs error vectors of one length, from one design and data.",
    file: "packages/psychometrics/src/equating/kernel.ts",
    find: "if (vector.length !== other?.length) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/equating.test.ts",
    titles: "rejects equatings of different designs",
  },
  {
    rule: "A link pairs the items of two forms by id, so that no item of one form is read as another's.",
    file: "packages/psychometrics/src/equating/link-covariance.ts",
    find: "    if (k === undefined) continue;\n",
    replace: "",
    test: "packages/psychometrics/test/equating.test.ts",
    titles: "rejects too few forms",
  },
  {
    rule: "The errors of a link need covariances that name the parameters of its common items.",
    file: "packages/psychometrics/src/equating/link-covariance.ts",
    find: "if (parameters === 0) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/equating.test.ts",
    titles: "rejects too few forms",
  },
  {
    rule: "Every category up to the highest has persons, so that a category nobody holds is not read as a cut and a large one does not cost a boundary for each number below it.",
    file: "packages/psychometrics/src/standard-setting/contrasting-groups.ts",
    find: "if (present.size <= highest) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/standard-setting.test.ts",
    titles: "rejects invalid scores, categories and options|category that nobody holds",
  },
  {
    rule: "A category is an integer from 0, so that no fraction or negative number is read as a level.",
    file: "packages/psychometrics/src/standard-setting/contrasting-groups.ts",
    find: "if (!(Number.isSafeInteger(category) && category >= 0)) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/standard-setting.test.ts",
    titles: "rejects invalid scores, categories and options",
  },
  {
    rule: "Each person has a mark and a rating at every station, so that a short row is not read as fewer stations and a long one is not cut.",
    file: "packages/psychometrics/src/standard-setting/stations.ts",
    find: "if (row.length !== stations) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/standard-setting.test.ts",
    titles: "rejects marks, ratings and grades",
  },
  {
    rule: "A mark or a rating is a finite number, so that a string or null is not converted into one.",
    file: "packages/psychometrics/src/standard-setting/stations.ts",
    find: "if (!Number.isFinite(value)) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/standard-setting.test.ts",
    titles: "rejects marks, ratings and grades",
  },
  {
    rule: "Borderline regression takes at most 2^28 persons × stations², so that the test's standard error, a sum over the persons for each pair of stations, does not cost minutes.",
    file: "packages/psychometrics/src/standard-setting/borderline-regression.ts",
    find: "if (persons * count * count > MAX_PAIR_PRODUCTS) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/standard-setting.test.ts",
    titles: "refuses more than",
  },
  {
    rule: "A score outside its range is refused, so that no proportion outside [0, 1] enters the true-score moments.",
    file: "packages/psychometrics/src/standard-setting/livingston-lewis.ts",
    find: "score >= minimum && score <= maximum",
    replace: "true",
    test: "packages/psychometrics/test/standard-setting.test.ts",
    titles: "rejects invalid scores, cuts and options",
  },
  {
    rule: "The rounded effective length is at most 10,000 trials, so that a reliability near 1 does not cost minutes.",
    file: "packages/psychometrics/src/standard-setting/livingston-lewis.ts",
    find: "trials >= 1 && trials <= MAX_EFFECTIVE_LENGTH",
    replace: "trials >= 1",
    test: "packages/psychometrics/test/standard-setting.test.ts",
    titles: "effective length outside its limits",
  },
  {
    rule: "A starting item the pool does not have is refused, so that a misspelled id is not skipped as an ineligible one.",
    file: "packages/psychometrics/src/cat/starting-items.ts",
    find: "if (index === undefined) throw new DataError(",
    replace: "if (index === undefined) return -1;\n    if (index < 0) throw new DataError(",
    test: "packages/psychometrics/test/cat.test.ts",
    titles: "starting item the pool does not have",
  },
  {
    rule: "Selection by the closest difficulty refuses an item without a b, so that no score is NaN.",
    file: "packages/psychometrics/src/cat/closest-difficulty.ts",
    find: "if (!Number.isFinite(difficulty)) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/cat.test.ts",
    titles: "item without a difficulty",
  },
  {
    rule: "The discrimination index takes items of two categories, so that a polytomous item is not read as dichotomous.",
    file: "packages/psychometrics/src/cat/weighted-rules.ts",
    find: "if (count !== 2) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/cat.test.ts",
    titles: "discrimination index of an item that does not have two categories",
  },
  {
    rule: "A cognitive diagnosis design stops at 15 attributes, 2^15 profiles, before a fit holds them.",
    file: "packages/psychometrics/src/cdm/q-matrix.ts",
    find: "if (attributes.length > MAX_ATTRIBUTES) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/cdm.test.ts",
    titles: "refuses more than 15 attributes",
  },
  {
    rule: "A cognitive diagnosis design stops at 2^22 cells of items by profiles, which a fit holds as tables of latent groups.",
    file: "packages/psychometrics/src/cdm/q-matrix.ts",
    find: "if (cells > MAX_PROFILE_CELLS) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/cdm.test.ts",
    titles: "refuses more than 15 attributes",
  },
  {
    rule: "A cognitive diagnosis design stops at 2^18 latent groups over all items, the success probabilities a fit holds.",
    file: "packages/psychometrics/src/cdm/q-matrix.ts",
    find: "if (groups > MAX_LATENT_GROUPS) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/cdm.test.ts",
    titles: "refuses more than 15 attributes",
  },
  {
    rule: "The polytomous DIF statistics stop at 255 categories of an item, as the generalized test holds every pair of them.",
    file: "packages/psychometrics/src/dif/groups.ts",
    find: "if (categories > MAX_CATEGORIES) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/dif.test.ts",
    titles: "more than 255 categories",
  },
  {
    rule: "A lognormal calibration of more than 2,000 items gives no standard errors, whose information holds every pair.",
    file: "packages/psychometrics/src/response-time/lognormal.ts",
    find: "withinStandardErrorSize(itemCount) ? intensityErrors(timed, state) : null",
    replace: "intensityErrors(timed, state)",
    test: "packages/psychometrics/test/response-time.test.ts",
    titles: "standard errors of more than 2,000 items",
  },
  {
    rule: "The joint calibration of speed and accuracy takes Hessian standard errors of at most 2,000 estimated parameters, by the bound of every calibration.",
    file: "packages/psychometrics/src/response-time/speed-accuracy.ts",
    find: 'if (standardErrors === "hessian") checkStandardErrorSize(estimated);',
    replace: "",
    test: "packages/psychometrics/test/response-time.test.ts",
    titles: "refuses standard errors of more than 2,000 estimated parameters",
  },
  {
    rule: "Linear change stops at 1,000 effects, before its information holds a matrix of every pair of them.",
    file: "packages/psychometrics/src/explanatory/linear-change.ts",
    find: "if (layoutSize(layout) > MAX_EFFECTS) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/explanatory.test.ts",
    titles: "refuses more than 1,000 effects",
  },
  {
    rule: "A many-facet model stops at 1,000 parameters, before its Newton steps hold a matrix of every pair of them.",
    file: "packages/psychometrics/src/irt/linear-rasch.ts",
    find: "if (parameters > MAX_PARAMETERS) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/rater.test.ts",
    titles: "more than 1,000 parameters|more biases than",
  },
  {
    rule: "A many-facet model's rating scale stops at 255 categories, FACETS's limit.",
    file: "packages/psychometrics/src/irt/linear-rasch.ts",
    find: "if (categories > MAX_CATEGORIES) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/rater.test.ts",
    titles: "more than 1,000 parameters",
  },
  {
    rule: "A many-facet model stops at 2^24 expected counts of its virtual items, quadrature points and categories.",
    file: "packages/psychometrics/src/irt/linear-rasch.ts",
    find: "if (counts > MAX_EXPECTED_COUNTS) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/rater.test.ts",
    titles: "2\\^24 expected counts at once",
  },
  {
    rule: "An LLTM stops at the size of its Rasch comparison before it reads the features of every item.",
    file: "packages/psychometrics/src/explanatory/lltm.ts",
    find: "checkLinearRaschSize(raschModel, rule.nodes.length, RASCH_MODEL_NAME);",
    replace: "",
    test: "packages/psychometrics/test/explanatory.test.ts",
    titles: "Rasch comparison of more than 1,000 parameters",
  },
  {
    rule: "A diagnostic adaptive test stops at 2^22 cells of items by profiles and 2^18 latent groups, before it holds its profiles.",
    file: "packages/psychometrics/src/cat/diagnostic-items.ts",
    find: "  checkDesignSize(\n    attributes.length,\n    items.map((checked) => checked.positions.length),\n  );\n",
    replace: "",
    test: "packages/psychometrics/test/cat.test.ts",
    titles: "refuses a pool of more than",
  },
  {
    rule: "The incomplete standard errors stop at their limit of success probabilities, before the calibration.",
    file: "packages/psychometrics/src/cdm/standard-errors.ts",
    find: "if (probabilities > MAX_INCOMPLETE_PROBABILITIES) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/cdm.test.ts",
    titles: "refuses incomplete standard errors",
  },
  {
    rule: "An EM calibration of several dimensions stops at 2^24 values of its items at the nodes of its grids, before it allocates them.",
    file: "packages/psychometrics/src/multidimensional/grid.ts",
    find: "if (entries > MAX_TABLE_ENTRIES) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/multidimensional.test.ts",
    titles: "limits the probabilities of the items' categories that EM keeps",
  },
  {
    rule: "The tables of pairs of items that Louis' standard errors of a bifactor model take count toward the limit of the tables.",
    file: "packages/psychometrics/src/multidimensional/bifactor-fit.ts",
    find: "    if (pairs) {",
    replace: "    if (false) {",
    test: "packages/psychometrics/test/bifactor.test.ts",
    titles: "counts toward the limit the tables of pairs",
  },
  {
    rule: "The scan for the start of a posterior mode of many dimensions grows with them, not as 3 to their power.",
    file: "packages/psychometrics/src/multidimensional/posterior-mode.ts",
    find: "if (dimensions <= MAX_GRID_SCAN_DIMENSIONS) {",
    replace: "if (true) {",
    test: "packages/psychometrics/test/multidimensional.test.ts",
    titles: "finds the posterior mode in many dimensions",
  },
  {
    rule: "The weighted likelihood estimate takes Newton's step where Fisher scoring's does not halve the weighted score, which scoring alone leaves short of the root.",
    file: "packages/psychometrics/src/multidimensional/likelihood-mode.ts",
    find: "else moved = newton(theta, values, free, direction) && backtrack(moveTo, current, 0, halving) !== null;",
    replace: "else moved = false;",
    test: "packages/psychometrics/test/multidimensional.test.ts",
    titles: "matches the base R MLE and WLE",
  },
  {
    rule: "A group whose correlations are estimated needs more persons than its correlated dimensions, whose correlations would otherwise run to ±1.",
    file: "packages/psychometrics/src/multidimensional/groups.ts",
    find: "if (persons > dimensions) return;",
    replace: "return;",
    test: "packages/psychometrics/test/multidimensional.test.ts",
    titles: "needs more persons than correlated dimensions",
  },
  {
    rule: "The Gauss-Hermite rule kept for reuse is replaced when another number of points is asked for, so that no call integrates over the nodes of another.",
    file: "packages/psychometrics/src/irt/quadrature.ts",
    find: "if (lastGaussHermite?.nodes.length !== points) lastGaussHermite = gaussHermite(points);",
    replace: "lastGaussHermite ??= gaussHermite(points);",
    test: "packages/psychometrics/test/irt.test.ts",
    titles: "gives each Gauss-Hermite rule its own nodes",
  },
  {
    rule: "Calibration, scoring, information and adaptive tests take at most 100 dimensions, one shared limit, so that every calibration can be scored and scoring's time, which grows as the cube of the dimensions, stays bounded.",
    file: "packages/psychometrics/src/multidimensional/items.ts",
    find: "if (dimensions > MAX_DIMENSIONS) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/multidimensional.test.ts",
    titles: "takes at most 100 dimensions",
  },
  {
    rule: "A person with a missing response has no count of Guttman errors, and the counts of the others stay with their rows.",
    file: "packages/psychometrics/src/mokken/guttman-errors.ts",
    find: "(row.includes(null) ? null : (totals[complete++] ?? 0))",
    replace: "(totals[complete++] ?? 0)",
    test: "packages/psychometrics/test/mokken.test.ts",
    titles: "gives no count to a person with a missing response",
  },
  {
    rule: "Two-way imputation refuses a person without responses, whose mean would make every imputed response NaN.",
    file: "packages/psychometrics/src/mokken/two-way-imputation.ts",
    find: "if (answered === 0) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/mokken.test.ts",
    titles: "rejects a person or an item without responses",
  },
  {
    rule: "The groups of a rest score are at most as many persons as answered every item, so that no group is read beyond them.",
    file: "packages/psychometrics/src/mokken/conditional-association.ts",
    find: "if (minimumSize > data.persons) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/mokken.test.ts",
    titles: "groups larger than the persons",
  },
  {
    rule: "An extraction other than minres and maximum likelihood is refused, and never read as maximum likelihood.",
    file: "packages/psychometrics/src/dimensionality/exploratory-factors.ts",
    find: 'checkChoice(extraction, "The option extraction", EXTRACTIONS);',
    replace: "",
    test: "packages/psychometrics/test/dimensionality.test.ts",
    titles: "rejects options of other types",
  },
  {
    rule: "Every row of a factor solution's loadings has as many factors as the first, so that no loading is read as missing.",
    file: "packages/psychometrics/src/dimensionality/schmid-leiman.ts",
    find: "row.length !== width || ",
    replace: "",
    test: "packages/psychometrics/test/dimensionality.test.ts",
    titles: "rejects solutions of fewer than three factors",
  },
  {
    rule: "A norm sample takes finite scores alone, and a score of another type is refused.",
    file: "packages/psychometrics/src/scaling/norm-statistics.ts",
    find: "checkFinite(score, ",
    replace: "void (score, ",
    test: "packages/psychometrics/test/scaling.test.ts",
    titles: "rejects fewer than two scores",
  },
  {
    rule: "Person weights are divided by the power of two at or below the largest, so that their sums and the sums of their squares neither overflow nor underflow, and no rank or standard error is NaN, infinite or 0 for want of range.",
    file: "packages/psychometrics/src/scaling/percentile-ranks.ts",
    find: "2 ** Math.floor(Math.log2(largest))",
    replace: "1",
    test: "packages/psychometrics/test/scaling.test.ts",
    titles: "even those whose sum overflows|weights of any magnitude",
  },
  {
    rule: "A component that leaves an item no variance ends the test, and gives no NaN.",
    file: "packages/psychometrics/src/dimensionality/minimum-average-partial.ts",
    find: "for (let i = 0; i < size; i++) if (!((covariance[i * size + i] ?? 0) > 0)) return null;",
    replace: "",
    test: "packages/psychometrics/test/dimensionality.test.ts",
    titles: "refuses items whose correlations are all 0",
  },
  {
    rule: "An item of a polychoric correlation has at most 8 observed categories, so that items of as many categories as persons do not cost each pair the square of the persons.",
    file: "packages/psychometrics/src/dimensionality/polychoric.ts",
    find: "if (many !== undefined) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/dimensionality.test.ts",
    titles: "refuses an item of more",
  },
  {
    rule: "An analysis of the pairs of items takes at most 1,000 items, so that a few persons answering many items do not cost a matrix of their square and its decompositions.",
    file: "packages/psychometrics/src/dimensionality/coded-items.ts",
    find: "  checkMatrixItems(matrix.itemIds.length, analysis);\n",
    replace: "",
    test: "packages/psychometrics/test/dimensionality.test.ts",
    titles: "takes at most 1,000 items",
  },
  {
    rule: "A covariance sums the values less the first of each, so that scores far from 0, such as near 1e8, are not cancelled to noise.",
    file: "packages/psychometrics/src/math/statistics.ts",
    find: "const firstX = x[0] ?? 0;\n  const firstY = y[0] ?? 0;",
    replace: "const firstX = 0;\n  const firstY = 0;",
    test: "packages/psychometrics/test/math.test.ts",
    titles: "scores far from 0",
  },
  {
    rule: "A matrix with an entry that is not finite is not decomposed, as its QL iterations would each run to their limit, some 15 times the work of a finite matrix.",
    file: "packages/psychometrics/src/math/eigen.ts",
    find: "if (!work.every(Number.isFinite)) return { ...notDecomposed(size)",
    replace: "if (false) return { ...notDecomposed(size)",
    test: "packages/psychometrics/test/math.test.ts",
    titles: "decomposes no matrix with an entry that is not finite",
  },
  {
    rule: "A matrix with an entry that is not finite has no eigenvalues alone either, as its QL iterations would each run to their limit.",
    file: "packages/psychometrics/src/math/eigen.ts",
    find: "if (!work.every(Number.isFinite)) return notDecomposed(size);",
    replace: "",
    test: "packages/psychometrics/test/math.test.ts",
    titles: "eigenvalues alone as the decomposition gives them",
  },
  {
    rule: "The QL iterations of one eigenvalue stop at their limit, so that finite entries whose arithmetic overflows end without converging.",
    file: "packages/psychometrics/src/math/eigen.ts",
    find: "if (count === MAX_ITERATIONS) return",
    replace: "if (count === 2 * MAX_ITERATIONS) return",
    test: "packages/psychometrics/test/math.test.ts",
    titles: "stops at the limit of QL iterations",
  },
  {
    rule: "Smoothing takes at most 1,000 rows, as its eigendecomposition takes their cube, from a matrix that the caller gives.",
    file: "packages/psychometrics/src/dimensionality/smoothing.ts",
    find: 'checkMatrixItems(size, "Smoothing");',
    replace: "",
    test: "packages/psychometrics/test/dimensionality.test.ts",
    titles: "smooths at most 1,000 rows",
  },
  {
    rule: "Omega takes at most 1,000 items, so that a few persons answering many items do not cost a matrix of their square.",
    file: "packages/psychometrics/src/ctt/omega.ts",
    find: '  checkMatrixItems(k, "Omega");\n',
    replace: "",
    test: "packages/psychometrics/test/ctt.test.ts",
    titles: "takes at most 1,000 items for omega",
  },
  {
    rule: "A G study has at most 6 facets with the object, so that a design of many facets of two elements does not cost 3^k marginal sums.",
    file: "packages/psychometrics/src/generalizability/structure.ts",
    find: "if (names.length > MAX_FACETS) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/generalizability.test.ts",
    titles: "more than 6 facets",
  },
  {
    rule: "A predicted theta score reports at most 10,000 scores, so that a scale of a few bytes does not ask for billions of boundaries.",
    file: "packages/psychometrics/src/scaling/predict-theta-score.ts",
    find: "if (!(length <= MAX_REPORTABLE_SCORES)) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/scaling.test.ts",
    titles: "reports at most 10,000 scores",
  },
  {
    rule: "A log-linear presmoothing has a degree of at most 20, so that a degree near a long distribution's length does not cost the cube of its terms.",
    file: "packages/psychometrics/src/equating/presmoothing.ts",
    find: "const highest = Math.min(distribution.length - 1, MAX_LOG_LINEAR_DEGREE);",
    replace: "const highest = distribution.length - 1;",
    test: "packages/psychometrics/test/equating.test.ts",
    titles: "fits degrees up to 20",
  },
  {
    rule: "An array with a hole is refused, as forEach, map and every skip a hole that a check would read.",
    file: "packages/core/src/checks.ts",
    find: "if (hole >= 0) throw",
    replace: "if (false) throw",
    test: "packages/core/test/checks.test.ts",
    titles: "with a hole",
  },
  {
    rule: "A record of names refuses a reserved key, which an assignment or a read by that name would lose unseen.",
    file: "packages/core/src/checks.ts",
    find: "if (Object.hasOwn(value, key)) throw",
    replace: "if (false) throw",
    test: "packages/core/test/checks.test.ts",
    titles: "refuse the own key",
  },
  {
    rule: "An item's standard errors refuse a reserved key, which their copy by assignment would drop unseen.",
    file: "packages/psychometrics/src/equating/linking.ts",
    find: 'checkRecord(errors, "An item\'s standard errors must be an object of numbers by parameter name or null");',
    replace: "",
    test: "packages/psychometrics/test/reserved-keys.test.ts",
    titles: "in an item's standard errors",
  },
  {
    rule: "A facet's size in a decision study is an own entry, as an inherited member would pass for a size that a study not balanced must be given.",
    file: "packages/psychometrics/src/generalizability/decision-study.ts",
    find: "return Object.hasOwn(sizes, name) ? sizes[name] : undefined;",
    replace: "return sizes[name];",
    test: "packages/psychometrics/test/reserved-keys.test.ts",
    titles: "needs its size",
  },
  {
    rule: "A facet's level in an expected rating is an own entry, as an inherited member would be taken for an element.",
    file: "packages/psychometrics/src/rater/expected-rating.ts",
    find: "(Object.hasOwn(levels, facet) ? (levels[facet] ?? null) : null)",
    replace: "(levels[facet] ?? null)",
    test: "packages/psychometrics/test/reserved-keys.test.ts",
    titles: "averages over its elements",
  },
  {
    rule: "The rates of pairs of resources are built as own entries, as an assignment to __proto__ would lose a resource's rates unseen.",
    file: "packages/knowledge-tracing/src/bkt/estimation.ts",
    find: "Object.fromEntries(pairs)",
    replace:
      "[...pairs].reduce((all: Record<string, Record<string, BktLearning>>, [previous, rates]) => { all[previous] = rates; return all; }, {})",
    test: "packages/knowledge-tracing/test/bkt.test.ts",
    titles: "named as members of an object",
  },
  {
    rule: "Kappa takes a table of the range only where the range is no wider than the pairs.",
    file: "packages/core/src/kappa.ts",
    find: "highest - lowest < n",
    replace: "true",
    test: "packages/core/test/kappa.test.ts",
    titles: "however far apart",
  },
  {
    rule: "A line search asks a function given apart for its gradient at the length it accepts alone, as a gradient by finite differences costs a fit for each of its points.",
    file: "packages/core/src/descent.ts",
    find: "whole ? f(trial.x, trial.gradient) : f.value(trial.x);",
    replace: "whole ? f(trial.x, trial.gradient) : f.value(trial.x); if (!whole) f.gradient(trial.x, trial.gradient);",
    test: "packages/core/test/numerics.test.ts",
    titles: "for its gradient at the accepted one alone",
  },
  {
    rule: "A minimization never reports convergence at a point that is not finite.",
    file: "packages/core/src/bfgs.ts",
    find: "converged: isFinitePoint(point)",
    replace: "converged: true",
    test: "packages/core/test/optimize.test.ts",
    titles: "not finite",
  },
  {
    rule: "An id sets an entry of a record, and never its prototype.",
    file: "packages/knowledge-tracing/src/data/record.ts",
    find: 'if (key === "__proto__")',
    replace: "if (false)",
    test: "packages/knowledge-tracing/test/logistic.test.ts",
    titles: "logistic fits replayed",
  },
  {
    rule: "A nominal item has as many intercepts as slopes, so that a category without an intercept is never read as 0.",
    file: "packages/knowledge-tracing/src/irt/items.ts",
    find: "a.length >= 2 && a.length === c.length",
    replace: "a.length >= 2",
    test: "packages/knowledge-tracing/test/irt.test.ts",
    titles: "rejects invalid items",
  },
  {
    rule: "A nominal item has at least two categories, as one category makes every outcome certain.",
    file: "packages/knowledge-tracing/src/irt/items.ts",
    find: "a.length >= 2 && a.length === c.length",
    replace: "a.length === c.length",
    test: "packages/knowledge-tracing/test/irt.test.ts",
    titles: "rejects invalid items",
  },
  {
    rule: "A state serves the learner it belongs to.",
    file: "packages/knowledge-tracing/src/data/learner.ts",
    find: "if (attempt.learner !== learner) {",
    replace: "if (false) {",
    test: "packages/knowledge-tracing/test/elo.test.ts",
    titles: "the state of another learner",
  },
  {
    rule: "A stored BKT state holds a probability at each skill a call reads, so that no value outside 0 to 1 enters a prediction or an update.",
    file: "packages/knowledge-tracing/src/bkt/state.ts",
    find: "if (Object.hasOwn(state.skills, skill) && !isUnitInterval(state.skills[skill])) {",
    replace: "if (false) {",
    test: "packages/knowledge-tracing/test/bkt.test.ts",
    titles: "rejects malformed states, as from storage",
  },
  {
    rule: "A stored Elo state holds a rating and a count at each skill a call reads.",
    file: "packages/knowledge-tracing/src/elo/state.ts",
    find: "if (!Object.hasOwn(state.skills, skill)) continue;",
    replace: "continue;",
    test: "packages/knowledge-tracing/test/elo.test.ts",
    titles: "rejects malformed states, as from storage",
  },
  {
    rule: "A stored logistic state holds a well-formed history at each skill and item a call reads.",
    file: "packages/knowledge-tracing/src/logistic/state.ts",
    find: "if (Object.hasOwn(histories, id)) checkHistory(",
    replace: "if (false) checkHistory(",
    test: "packages/knowledge-tracing/test/logistic.test.ts",
    titles: "reject malformed states, as from storage",
  },
  {
    rule: "Ids joined into one key stay apart, whatever characters they hold, so that two lists of ids never share an entry.",
    file: "packages/knowledge-tracing/src/data/composite-key.ts",
    find: "JSON.stringify(ids)",
    replace: 'ids.join("\\u0000")',
    test: "packages/knowledge-tracing/test/data.test.ts",
    titles: "whatever characters their ids hold",
  },
  {
    rule: "The analyses of the relations between elements take at most 1,000 elements, as their tables grow with the square of their number.",
    file: "packages/knowledge-tracing/src/kst/element-limit.ts",
    find: "if (count <= MAX_ELEMENTS) return;",
    replace: "return;",
    test: "packages/knowledge-tracing/test/kst.test.ts",
    titles: "up to 1,000 items|up to the 1,000 elements",
  },
  {
    rule: "A pair of items that no learner answered both of leaves the difference test, and is not read as a pair without counterexamples.",
    file: "packages/knowledge-tracing/src/kst/iita.ts",
    find: "return observed === null || observed[i * m + j] === 1;",
    replace: "return true;",
    test: "packages/knowledge-tracing/test/kst.test.ts",
    titles: "two booklets",
  },
  {
    rule: "Prerequisites whose implications no learner answered both items of are refused, as the responses cannot test them.",
    file: "packages/knowledge-tracing/src/kst/diff-test.ts",
    find: "if (!hasObservedImplication(relation, data)) {",
    replace: "if (false) {",
    test: "packages/knowledge-tracing/test/kst.test.ts",
    titles: "rejects what cannot be tested",
  },
  {
    rule: "The candidates of IITA hold at most 24 million cells, as each lists the prerequisites of every item.",
    file: "packages/knowledge-tracing/src/kst/iita.ts",
    find: "if (cells > MAX_CANDIDATE_CELLS) {",
    replace: "if (false) {",
    test: "packages/knowledge-tracing/test/kst.test.ts",
    titles: "24 million cells",
  },
  {
    rule: "The masks of a KST model hold at most 2^28 cells, a byte for each state and set of skills its attempts name.",
    file: "packages/knowledge-tracing/src/kst/kst.ts",
    find: "if ((masks.size + 1) * compiled.count > MAX_MASK_CELLS) {",
    replace: "if (false) {",
    test: "packages/knowledge-tracing/test/kst.test.ts",
    titles: "past 2\\^28 cells",
  },
  {
    rule: "Local dependence takes at most 1,000 items, so that a few persons answering many items do not cost a statistic of each of their pairs.",
    file: "packages/psychometrics/src/fit/local-dependence.ts",
    find: '  checkMatrixItems(items.length, "Local dependence");\n',
    replace: "",
    test: "packages/psychometrics/test/fit.test.ts",
    titles: "takes at most 1,000 items, as it and Yen's Q3",
  },
  {
    rule: "Yen's Q3 takes at most 1,000 items, so that a few persons answering many items do not cost a matrix of their square.",
    file: "packages/psychometrics/src/fit/local-dependence.ts",
    find: '  checkMatrixItems(items.length, "Yen\'s Q3");\n',
    replace: "",
    test: "packages/psychometrics/test/fit.test.ts",
    titles: "takes at most 1,000 items, as it and Yen's Q3",
  },
  {
    rule: "A Mokken scale analysis takes at most 1,000 items, so that a few persons answering many items do not cost every pair of them.",
    file: "packages/psychometrics/src/mokken/scores.ts",
    find: "if (items > MAX_MATRIX_ITEMS) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/mokken.test.ts",
    titles: "every analysis of their pairs",
  },
  {
    rule: "Molenaar-Sijtsma reliability takes at most 1,000 item steps, so that items of many categories do not cost a matrix of the square of their steps.",
    file: "packages/psychometrics/src/mokken/reliability.ts",
    find: "if (size > MAX_MATRIX_ITEMS) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/mokken.test.ts",
    titles: "1,000 item steps",
  },
  {
    rule: "The Q3 bootstrap checks the calibration's options against the matrix before any draw, as every draw can be drawn again before one is calibrated.",
    file: "packages/psychometrics/src/fit/q3-critical-value.ts",
    find: "checkCalibrationOptions(\n    matrix,",
    replace: "((..._: unknown[]) => undefined)(\n    matrix,",
    test: "packages/psychometrics/test/fit.test.ts",
    titles: "before any draw",
  },
  {
    rule: "A replication of the Q3 bootstrap misses the responses that the data miss, as the calibration's persons did.",
    file: "packages/psychometrics/src/fit/q3-critical-value.ts",
    find: "((matrix.responses[person]?.[column] ?? null) === null ? null : response)",
    replace: "response",
    test: "packages/psychometrics/test/fit.test.ts",
    titles: "yenQ3CriticalValue.*gives the replications of the recipe",
  },
  {
    rule: "A replication of the Q3 bootstrap whose estimated item lacks a category is drawn again, never calibrated as another item.",
    file: "packages/psychometrics/src/fit/q3-critical-value.ts",
    find: "hasEveryCategory(simulated, items, estimated) ? excessesOf(simulated) : null",
    replace: "excessesOf(simulated)",
    test: "packages/psychometrics/test/fit.test.ts",
    titles: "yenQ3CriticalValue.*draws again a replication",
  },
  {
    rule: "The Q3 bootstrap leaves out the pairs of a near-deterministic item, whose Q3 rests on a few persons.",
    file: "packages/psychometrics/src/fit/q3-critical-value.ts",
    find: "if (value === null || tested[i] !== true || tested[j] !== true) continue;",
    replace: "if (value === null) continue;",
    test: "packages/psychometrics/test/fit.test.ts",
    titles: "yenQ3CriticalValue.*leaves out the pairs",
  },
  {
    rule: "The tables of local dependence hold at most 2^24 cells, so that items of many categories do not cost the square of them.",
    file: "packages/psychometrics/src/fit/local-dependence.ts",
    find: "if (cells > MAX_TABLE_CELLS) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/fit.test.ts",
    titles: "more than 2\\^24 cells in all",
  },
  {
    rule: "C2 has at most 1,830 moments, so that items of many categories do not grow its covariance matrix past what 60 items allow.",
    file: "packages/psychometrics/src/fit/limited-information.ts",
    find: "if (moments > MAX_MOMENTS) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/fit.test.ts",
    titles: "refuses C2 of more than 1,830 moments",
  },
  {
    rule: "A Rasch test leaves out an item whose highest score is beyond the persons of a group before its categories take room, so that one far response costs no table of its categories.",
    file: "packages/psychometrics/src/fit/rasch-tests.ts",
    find: "return top >= 1 && top < smallest;",
    replace: "return top >= 1;",
    test: "packages/psychometrics/test/fit.test.ts",
    titles: "beyond the persons of a group",
  },
  {
    rule: "Form assembly takes one constraint for each content area, so that a second is not read with no item counted for the first.",
    file: "packages/psychometrics/src/cat/form-assembly.ts",
    find: "const constraints = readConstraints(content);",
    replace: "const constraints = content.flatMap((constraint) => readConstraints([constraint]));",
    test: "packages/psychometrics/test/cat.test.ts",
    titles: "assembleForms.*rejects options of other types and invalid ones",
  },
  {
    rule: "S-X² takes items of a highest total score of at most 300, so that many items do not cost their cube.",
    file: "packages/psychometrics/src/fit/summed-score-fit.ts",
    find: "if (maxScore > MAX_TOTAL_SCORE) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/fit.test.ts",
    titles: "highest total score of at most 300",
  },
  {
    rule: "multistageCuts searches a range at most 1,000 wide, so that an absurd range costs no grid of gigabytes.",
    file: "packages/psychometrics/src/cat/multistage-cuts.ts",
    find: "if (!(upper - lower <= MAX_RANGE_WIDTH)) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/cat.test.ts",
    titles: "refuses a range that is not an interval",
  },
  {
    rule: "Form assembly holds at most 2^26 counts of pairs of forms and of forms and pool items, so that many forms cost no table of gigabytes.",
    file: "packages/psychometrics/src/cat/form-assembly.ts",
    find: "if (cells > MAX_ASSEMBLY_CELLS) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/cat.test.ts",
    titles: "assembleForms.*rejects options of other types and invalid ones",
  },
  {
    rule: "A value of a context that more than maxGroup persons share forms no pairs, so that a hall or a school's network does not cost the square of its persons.",
    file: "packages/psychometrics/src/similarity/pairs.ts",
    find: "if (members.length > maxGroup) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/similarity.test.ts",
    titles: "forms no pairs of a value more than maxGroup",
  },
  {
    rule: "A selection of more pairs than maxPairs is refused, so that wide contexts cost no list of gigabytes.",
    file: "packages/psychometrics/src/similarity/pairs.ts",
    find: "if (size > maxPairs) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/similarity.test.ts",
    titles: "refuses more pairs than maxPairs",
  },
  {
    rule: "A context has one value per person, so that a short list is not read as persons without a value.",
    file: "packages/psychometrics/src/similarity/pairs.ts",
    find: "if (list.length !== personIds.length) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/similarity.test.ts",
    titles: "refuses invalid ids, contexts and options",
  },
  {
    rule: "A value of a context is a string or null, so that a number is not read as a value of its own.",
    file: "packages/psychometrics/src/similarity/pairs.ts",
    find: 'if (typeof value !== "string") {',
    replace: "if (false) {",
    test: "packages/psychometrics/test/similarity.test.ts",
    titles: "refuses invalid ids, contexts and options",
  },
  {
    rule: "A pair names two persons of the matrix, so that an unknown id is not read as its first person.",
    file: "packages/psychometrics/src/similarity/pairs.ts",
    find: "if (index === undefined) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/similarity.test.ts",
    titles: "refuses invalid thetas, pairs and responses",
  },
  {
    rule: "A pair names two different persons, so that a person is not compared with itself.",
    file: "packages/psychometrics/src/similarity/pairs.ts",
    find: "if (read[2 * k] === read[2 * k + 1])",
    replace: "if (false)",
    test: "packages/psychometrics/test/similarity.test.ts",
    titles: "refuses invalid thetas, pairs and responses",
  },
  {
    rule: "A keyed category is a category of its item, so that no probability is read beyond the item's.",
    file: "packages/psychometrics/src/similarity/inputs.ts",
    find: "if (!(Number.isSafeInteger(category) && category >= 0 && category <= highest)) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/similarity.test.ts",
    titles: "refuses keys that are not categories|needs the key when a response is above 1",
  },
  {
    rule: "A key of the K index is at most the largest response, so that no key wraps in 32 bits onto a category that responses give.",
    file: "packages/psychometrics/src/similarity/inputs.ts",
    find: "?? MAX_RESPONSE + 1",
    replace: "?? Number.POSITIVE_INFINITY",
    test: "packages/psychometrics/test/similarity.test.ts",
    titles: "needs the key when a response is above 1",
  },
  {
    rule: "An index that counts identical correct responses needs the key of items of more than two categories, so that category 1 is not taken for it.",
    file: "packages/psychometrics/src/similarity/inputs.ts",
    find: "if (absent !== null) throw new DataError(absent);",
    replace: "",
    test: "packages/psychometrics/test/similarity.test.ts",
    titles: "needs the key",
  },
  {
    rule: "The generalized binomial test and M4 compare at most 1,000 items in a pair, so that a long row costs no square or cube of seconds.",
    file: "packages/psychometrics/src/similarity/inputs.ts",
    find: "if (compared > MAX_COMPARED_ITEMS) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/similarity.test.ts",
    titles: "compares more than 1,000 items",
  },
  {
    rule: "The K index counts at most 2^24 cells of categories by group, so that a large category costs no table of gigabytes.",
    file: "packages/psychometrics/src/similarity/k-index.ts",
    find: "if (groupCount * width > MAX_COUNT_CELLS) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/similarity.test.ts",
    titles: "more than 2\\^24 cells",
  },
];

/** The folders that a scratch copy leaves out: generated output and installed packages, which it links instead. */
const SKIPPED = new Set(["node_modules", ".git", "dist", "coverage", "api"]);
/** The longest a test of a copy may take, in milliseconds: a removed bound can make a test wait for ever. */
const TEST_TIMEOUT_MS = 120_000;
/**
 * The longest a run of a copy may take in all, in milliseconds, before it is stopped. It is well above the time of a test,
 * because a test that is busy in synchronous work reports that it ran out of time only when the work ends: the slowest
 * bound that a removed rule leaves to a test to find takes 200 s, and a run stopped before that says nothing of the rule.
 */
const RUN_TIMEOUT_MS = 300_000;
/**
 * The most that a run may print, in bytes. Node.js stops a run at 1 MiB of output by default and cuts the summary off, and a
 * copy without a rule can fail every test of a file with a long message each.
 */
const REPORT_MAX_BYTES = 64 * 1024 * 1024;

/**
 * What checking one rule found: the rule is guarded, it is not, or a run took longer than its limit and nothing is known.
 * A run that is only slow, on a host with other work to do, says nothing about the rule.
 */
export interface Verdict {
  readonly outcome: "guarded" | "unguarded" | "timed-out";
  /** A sentence that says what happened and, when the rule is not guarded, what to do. */
  readonly message: string;
}

/**
 * Removes a rule from the text of its file.
 *
 * @throws `Error` when the text to replace does not occur exactly once.
 */
export function removeRule(source: string, rule: Rule): string {
  const count = source.split(rule.find).length - 1;
  if (count !== 1) {
    throw new Error(`${rule.file} holds ${JSON.stringify(rule.find)} ${count} times; the rule needs it exactly once.`);
  }
  return source.replace(rule.find, () => rule.replace);
}

/**
 * Returns the number of tests of a Vitest run that failed, or `undefined` when the run did not run tests.
 *
 * @remarks
 * A test that did not finish counts as failed, and so does an error of the run. A removed rule can make the worker crash, and
 * the summary then says "29 passed (80)" without a word about the 51 tests that never ran.
 */
export function failedTests(report: string): number | undefined {
  const summary = /^\s*Tests\s+(.*)$/m.exec(report)?.[1];
  if (summary === undefined || !/\d+ (?:failed|passed)/.test(summary)) return undefined;
  const count = (state: string): number => Number(new RegExp(`(\\d+) ${state}`).exec(summary)?.[1] ?? 0);
  const total = Number(/\((\d+)\)\s*$/.exec(summary)?.[1] ?? 0);
  const finished = count("failed") + count("passed") + count("skipped") + count("todo");
  const errors = Number(/^\s*Errors\s+(\d+)/m.exec(report)?.[1] ?? 0);
  return count("failed") + Math.max(0, total - finished) + errors;
}

/**
 * Returns how many of the failed tests of a report of a Vitest run ran out of time. A failed test has one block, which starts
 * with `FAIL`, and the message may be printed again outside it (the reporter of GitHub Actions annotates it), so the blocks
 * are counted and not the messages.
 */
export function timedOutTests(report: string): number {
  return report
    .split(/^ FAIL /m)
    .slice(1)
    .filter((block) => /(?:Test|Hook) timed out in \d+ms/.test(block)).length;
}

/** The folders directly below a folder of the repository, if it exists. */
function foldersIn(root: string, folder: string): string[] {
  const parent = join(root, folder);
  if (!existsSync(parent)) return [];
  return readdirSync(parent, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => join(parent, entry.name));
}

/** Copies the repository into a scratch directory without generated output, and links the installed packages. */
function copyRepository(root: string, scratch: string): void {
  cpSync(root, scratch, { recursive: true, filter: (path) => !SKIPPED.has(basename(path)) });
  for (const directory of [root, ...foldersIn(root, "packages"), join(root, "examples")]) {
    const modules = join(directory, "node_modules");
    if (existsSync(modules)) symlinkSync(modules, join(scratch, directory.slice(root.length), "node_modules"));
  }
}

/** What a run of the tests of a rule reported. */
interface TestRun {
  /** The number of failed tests, or `undefined` when no tests ran. */
  readonly failed: number | undefined;
  readonly report: string;
  /** Whether the run was stopped at its limit, with tests that had not finished. */
  readonly stopped: boolean;
  /** Whether every failed test ran out of time. */
  readonly onlyTimeouts: boolean;
}

/** Runs the tests of a rule in a copy with Vitest and returns what they reported. */
function runTests(root: string, scratch: string, rule: Rule, limit: number): TestRun {
  const vitest = join(root, "node_modules/vitest/vitest.mjs");
  // Threads, so that a run that is stopped leaves no worker behind.
  const options = ["--pool=threads", "--maxWorkers=2", `--testTimeout=${TEST_TIMEOUT_MS}`, "-t", rule.titles];
  const run = spawnSync(process.execPath, [vitest, "run", ...options, rule.test], {
    cwd: scratch,
    encoding: "utf8",
    // biome-ignore lint/style/useNamingConvention: these are the names of the variables.
    env: { ...process.env, NO_COLOR: "1", FORCE_COLOR: "0" },
    killSignal: "SIGKILL",
    timeout: limit,
    maxBuffer: REPORT_MAX_BYTES,
  });
  const report = `${run.stdout}${run.stderr}`;
  const failed = failedTests(report);
  return {
    failed,
    report,
    stopped: run.error !== undefined && "code" in run.error && run.error.code === "ETIMEDOUT",
    onlyTimeouts: failed !== undefined && failed > 0 && timedOutTests(report) >= failed,
  };
}

/** Says that a run took longer than its limit and that this is not a verdict on the rule. */
function timedOut(test: string, where: string, run: TestRun, limit: number): Verdict {
  const slow = timedOutTests(run.report);
  const what = run.stopped
    ? `was stopped after ${limit / 1000} s`
    : `has ${slow} ${slow === 1 ? "test" : "tests"} that ran out of time`;
  return {
    outcome: "timed-out",
    message: `${test} ${what} ${where}, so nothing is known of the rule; run it again on a host with less work.`,
  };
}

/**
 * Checks that the tests of a rule pass on the code, and fail on a copy of the code without the rule. A run that takes longer
 * than its limit on the unchanged code, or is stopped on the copy, is a timeout and not a verdict: the host may be busy.
 *
 * @param root - The repository.
 * @param rule - The rule.
 * @param runLimit - The longest a run of the tests may take in all, in milliseconds.
 */
export function checkRule(root: string, rule: Rule, runLimit: number = RUN_TIMEOUT_MS): Verdict {
  const scratch = mkdtempSync(join(tmpdir(), "measurekit-rule-"));
  try {
    copyRepository(root, scratch);
    const baseline = runTests(root, scratch, rule, runLimit);
    if (baseline.stopped || baseline.onlyTimeouts) {
      return timedOut(rule.test, "on the unchanged code", baseline, runLimit);
    }
    if (baseline.failed !== 0) {
      return {
        outcome: "unguarded",
        message: `${rule.test} does not pass on the unchanged code; fix it first.\n${baseline.report}`,
      };
    }
    const path = join(scratch, rule.file);
    writeFileSync(path, removeRule(readFileSync(path, "utf8"), rule));
    const mutated = runTests(root, scratch, rule, runLimit);
    if (mutated.stopped) return timedOut(rule.test, "on the copy without the rule", mutated, runLimit);
    if (mutated.failed === undefined) {
      return {
        outcome: "unguarded",
        message: `${rule.test} did not run on the copy without the rule; the replacement may be invalid code.\n${mutated.report}`,
      };
    }
    if (mutated.failed === 0) {
      return {
        outcome: "unguarded",
        message: `${rule.test} passes without the rule: add a test that fails when "${rule.rule}" is broken.`,
      };
    }
    return {
      outcome: "guarded",
      message: `${mutated.failed} ${mutated.failed === 1 ? "test fails" : "tests fail"} without the rule.`,
    };
  } finally {
    rmSync(scratch, { recursive: true, force: true });
  }
}

/** The labels of the outcomes in the report, of one width. */
const LABELS = { guarded: "guarded   ", unguarded: "UNGUARDED ", "timed-out": "TIMED OUT " } as const;

/** Checks every rule, or the rules of the given files, and returns the exit code: 1 when a rule is not guarded, 2 when none is not but a run timed out. */
function main(paths: readonly string[]): number {
  const root = resolve(dirname(import.meta.dirname));
  const rules = RULES.filter(({ file }) => paths.length === 0 || paths.some((path) => file.startsWith(path)));
  const counts = { guarded: 0, unguarded: 0, "timed-out": 0 };
  for (const rule of rules) {
    const { outcome, message } = checkRule(root, rule);
    counts[outcome] += 1;
    console.log(`${LABELS[outcome]}${rule.rule} ${message}`);
  }
  console.log(
    counts.guarded === rules.length
      ? `Every one of ${rules.length} rules is guarded.`
      : `${counts.guarded} of ${rules.length} rules are guarded, ${counts.unguarded} are not, and ${counts["timed-out"]} could not be judged because a run timed out.`,
  );
  if (counts.unguarded > 0) return 1;
  return counts["timed-out"] > 0 ? 2 : 0;
}

if (import.meta.main) process.exitCode = main(process.argv.slice(2));
