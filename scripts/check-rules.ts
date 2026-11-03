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
    rule: "Raju's areas take the 2PL alone, as a 3PL with unequal lower asymptotes has an infinite unsigned area.",
    file: "packages/psychometrics/src/dif/raju-area.ts",
    find: "if (model !== twoParameterLogistic) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/dif.test.ts",
    titles: "rejects models other than the 2PL",
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
    rule: "A range of theta is at most 10,000 wide, as the scan of modal estimation walks it in steps of 0.25.",
    file: "packages/psychometrics/src/irt/ability.ts",
    find: "if (!(upper - lower <= MAX_RANGE_WIDTH)) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/irt.test.ts",
    titles: "wider than 10,000",
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
    find: "if (weights.length !== answered.length) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/irt.test.ts",
    titles: "rejects weights that are not a positive number",
  },
  {
    rule: "A person's weight is a finite number above 0, as 0 or less would drop or subtract the person and a weight that is not finite would void every sum.",
    file: "packages/psychometrics/src/irt/responses.ts",
    find: "checkPositive(weight, `The weight of person",
    replace: "(weight, `The weight of person",
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
    rule: "A limited-information statistic counts the parameters the calibration estimated, so that a missing anchor is not read as an estimated item.",
    file: "packages/psychometrics/src/fit/limited-information.ts",
    find: "if (counted !== count) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/fit.test.ts",
    titles: "asks for the anchors",
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
    find: "itemCount > MAX_ERROR_ITEMS ? null : intensityErrors(timed, state)",
    replace: "intensityErrors(timed, state)",
    test: "packages/psychometrics/test/response-time.test.ts",
    titles: "standard errors of more than 2,000 items",
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
    titles: "more than 1,000 parameters",
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
    rule: "The grids that the posterior mean keeps for reuse are the 8 asked for last, within 2^23 coordinates, so that grids of many sizes do not accumulate.",
    file: "packages/psychometrics/src/multidimensional/scoring.ts",
    find: "STANDARD_GRIDS.delete(oldest);",
    replace: "continue;",
    test: "packages/psychometrics/test/multidimensional.test.ts",
    titles: "keeps the 8 grids asked for last",
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
    rule: "Kappa takes a table of the range only where the range is no wider than the pairs.",
    file: "packages/core/src/kappa.ts",
    find: "highest - lowest < n",
    replace: "true",
    test: "packages/core/test/kappa.test.ts",
    titles: "however far apart",
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
    rule: "A state serves the learner it belongs to.",
    file: "packages/knowledge-tracing/src/data/learner.ts",
    find: "if (attempt.learner !== learner) {",
    replace: "if (false) {",
    test: "packages/knowledge-tracing/test/elo.test.ts",
    titles: "the state of another learner",
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
];

/** The folders that a scratch copy leaves out: generated output and installed packages, which it links instead. */
const SKIPPED = new Set(["node_modules", ".git", "dist", "coverage", "api"]);
/** The longest a test of a copy may take, in milliseconds. */
const TEST_TIMEOUT_MS = 120_000;
/** The longest a run of a copy may take, in milliseconds: a rule that bounds work leaves a run that does not end. */
const RUN_TIMEOUT_MS = 150_000;

/** What checking one rule found. */
export interface Verdict {
  readonly guarded: boolean;
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

/** Returns the number of failed tests in the report of a Vitest run, or `undefined` when the run did not run tests. */
export function failedTests(report: string): number | undefined {
  const summary = /^\s*Tests\s+(.*)$/m.exec(report)?.[1];
  if (summary === undefined || !/\d+ (?:failed|passed)/.test(summary)) return undefined;
  return Number(/(\d+) failed/.exec(summary)?.[1] ?? 0);
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

/** Runs the tests of a rule in a copy and returns what they reported; a run that does not end counts as a failure. */
function runTests(root: string, scratch: string, rule: Rule): { failed: number | undefined; report: string } {
  const vitest = join(root, "node_modules/vitest/vitest.mjs");
  // Threads, so that a run that is stopped leaves no worker behind.
  const options = ["--pool=threads", "--maxWorkers=2", `--testTimeout=${TEST_TIMEOUT_MS}`, "-t", rule.titles];
  const run = spawnSync(process.execPath, [vitest, "run", ...options, rule.test], {
    cwd: scratch,
    encoding: "utf8",
    // biome-ignore lint/style/useNamingConvention: these are the names of the variables.
    env: { ...process.env, NO_COLOR: "1", FORCE_COLOR: "0" },
    killSignal: "SIGKILL",
    timeout: RUN_TIMEOUT_MS,
  });
  const report = `${run.stdout}${run.stderr}`;
  return { failed: run.error === undefined ? failedTests(report) : 1, report };
}

/**
 * Checks that the tests of a rule pass on the code, and fail on a copy of the code without the rule.
 *
 * @param root - The repository.
 * @param rule - The rule.
 */
export function checkRule(root: string, rule: Rule): Verdict {
  const scratch = mkdtempSync(join(tmpdir(), "measurekit-rule-"));
  try {
    copyRepository(root, scratch);
    const baseline = runTests(root, scratch, rule);
    if (baseline.failed !== 0) {
      return {
        guarded: false,
        message: `${rule.test} does not pass on the unchanged code; fix it first.\n${baseline.report}`,
      };
    }
    const path = join(scratch, rule.file);
    writeFileSync(path, removeRule(readFileSync(path, "utf8"), rule));
    const mutated = runTests(root, scratch, rule);
    if (mutated.failed === undefined) {
      return {
        guarded: false,
        message: `${rule.test} did not run on the copy without the rule; the replacement may be invalid code.\n${mutated.report}`,
      };
    }
    if (mutated.failed === 0) {
      return {
        guarded: false,
        message: `${rule.test} passes without the rule: add a test that fails when "${rule.rule}" is broken.`,
      };
    }
    return {
      guarded: true,
      message: `${mutated.failed} ${mutated.failed === 1 ? "test fails" : "tests fail"} without the rule.`,
    };
  } finally {
    rmSync(scratch, { recursive: true, force: true });
  }
}

/** Checks every rule, or the rules of the given files, and returns the exit code. */
function main(paths: readonly string[]): number {
  const root = resolve(dirname(import.meta.dirname));
  const rules = RULES.filter(({ file }) => paths.length === 0 || paths.some((path) => file.startsWith(path)));
  let unguarded = 0;
  for (const rule of rules) {
    const { guarded, message } = checkRule(root, rule);
    if (!guarded) unguarded += 1;
    console.log(`${guarded ? "guarded  " : "UNGUARDED"} ${rule.rule} ${message}`);
  }
  console.log(
    unguarded === 0 ? `Every one of ${rules.length} rules is guarded.` : `${unguarded} rules are not guarded.`,
  );
  return unguarded === 0 ? 0 : 1;
}

if (import.meta.main) process.exitCode = main(process.argv.slice(2));
