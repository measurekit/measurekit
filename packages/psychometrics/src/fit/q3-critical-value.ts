import {
  checkIntegerFrom,
  checkOptions,
  checkPositive,
  checkPositiveInteger,
  checkRandom,
  checkUnitInterval,
  DataError,
  describe,
  type FieldNames,
  type Random,
} from "@measurekit/core";
import { checkFunction } from "../checks.js";
import { buildResponseMatrix, type ResponseMatrix } from "../data/internal.js";
import {
  type AbilityEstimator,
  abilityEstimates,
  type CalibrationOptions,
  calibrate,
  checkCalibrationOptions,
  checkItemsOfMatrix,
  type IrtItem,
  posteriorMeanEstimate,
  type TestItem,
} from "../irt/internal.js";
import { sampleQuantile } from "../math/internal.js";
import { simulateResponses, simulateThetas } from "../sim/internal.js";
import { yenQ3 } from "./local-dependence.js";

/**
 * Options of {@link yenQ3CriticalValue}: those of the calibration that estimated the items, which each replication's
 * calibration repeats, and those of the bootstrap.
 */
export interface Q3CriticalValueOptions<TParameters>
  extends Pick<
    CalibrationOptions<TParameters>,
    "fixed" | "fixedParameters" | "priors" | "itemPriors" | "quadrature" | "tolerance" | "maxIterations"
  > {
  /**
   * The number of replications, each a calibration of responses simulated from the items.
   *
   * @defaultValue `40`
   */
  readonly replications?: number;
  /**
   * The quantile of the replications' largest pair excesses, Q3*, that is the critical excess, from 0 to 1.
   *
   * @defaultValue `0.95`
   */
  readonly quantile?: number;
  /**
   * The most draws thrown away because an estimated item lacks a response in one of its categories, which its
   * calibration could not estimate; past them the bootstrap stops with the replications it has.
   *
   * @defaultValue `40`
   */
  readonly maxRedraws?: number;
  /**
   * The largest slope `a` of an item that is tested: an item above it answers almost by theta alone, its residuals are
   * near 0 for most persons, and its pairs' Q3 rests on a few; its pairs are left out, and it is listed.
   *
   * @defaultValue `4`
   */
  readonly slopeLimit?: number;
  /**
   * How each person's theta is estimated from the items, for the residuals.
   *
   * @defaultValue `posteriorMeanEstimate`
   */
  readonly estimator?: AbilityEstimator;
}

const Q3_CRITICAL_VALUE_OPTIONS: FieldNames<Q3CriticalValueOptions<unknown>> = {
  fixed: true,
  fixedParameters: true,
  priors: true,
  itemPriors: true,
  quadrature: true,
  tolerance: true,
  maxIterations: true,
  replications: true,
  quantile: true,
  maxRedraws: true,
  slopeLimit: true,
  estimator: true,
};

/** A pair of items whose excess is above the critical excess. */
export interface Q3Pair {
  /** The ids of the two items, in item order. */
  readonly items: readonly [first: string, second: string];
  /** Yen's Q3 of the pair. */
  readonly q3: number;
  /** Q3 less the mean Q3 of the pairs tested. */
  readonly excess: number;
}

/** The critical value of Yen's Q3 by parametric bootstrap, with the observed pairs above it. */
export interface Q3CriticalValue {
  /**
   * The critical excess: the `quantile` of the replications' largest excesses; `null` without a replication or without
   * two items tested.
   */
  readonly criticalExcess: number | null;
  /** The mean Q3 of the observed pairs tested, which a pair's excess is taken from; `null` without such a pair. */
  readonly meanQ3: number | null;
  /** The observed pairs whose excess is above the critical excess, largest first. */
  readonly flagged: readonly Q3Pair[];
  /** Each replication's largest pair excess, in the order drawn. */
  readonly largestExcesses: readonly number[];
  /** The replications the critical excess comes from: `replications` unless the redraws ran out. */
  readonly replications: number;
  /** The draws thrown away because an estimated item lacked a category. */
  readonly redrawn: number;
  /** The replications whose calibration stopped at `maxIterations` before it converged. */
  readonly notConverged: number;
  /** The ids of the items whose slope is above `slopeLimit`, whose pairs are not tested. */
  readonly nearDeterministic: readonly string[];
}

/** Forty replications, a labeled default: each costs a calibration. */
const DEFAULT_REPLICATIONS = 40;
/** The 95% quantile of Q3*, which holds the chance of any false flag near 0.05 (Christensen, Makransky and Horton). */
const DEFAULT_QUANTILE = 0.95;
/** As many redraws as replications, a labeled default. */
const DEFAULT_MAX_REDRAWS = 40;
/**
 * A labeled default: the calibrations of SAT12, LSAT7 and Science in the fixtures reach slopes of 2.3, and two parts of
 * a task that one action decides reach 175 (`docs/psychometrics/fit.md`).
 */
const DEFAULT_SLOPE_LIMIT = 4;

/**
 * Returns the critical value of Yen's Q3 for a calibration by parametric bootstrap, with the pairs of items above it.
 *
 * @remarks
 * A pair's excess is its Q3 less the mean Q3 of the pairs, at each person's estimate of theta from the items. Each
 * replication draws one standard normal theta per person, `simulateThetas`, and the responses to the items,
 * `simulateResponses`, from `random` in that order; sets missing the responses that `matrix` lacks; calibrates them
 * with the items' models and the calibration's options; and gives its largest pair excess. The critical excess is the
 * `quantile` of the replications' largest excesses, Q3*, by R's default quantile (type 7), which holds the chance of
 * any false flag among the pairs near 1 - `quantile`; the observed pairs whose excess is above it are flagged
 * (Christensen, Makransky and Horton). A draw in which an item that the calibration
 * estimates, not `fixed`, lacks a response in one of its categories is drawn again, at most `maxRedraws` times.
 *
 * An item whose slope `a` is above `slopeLimit` is near-deterministic: its residuals are near 0 for most persons, so
 * its pairs' Q3 rests on a few and the replications' largest excess grows with the slope until no pair is flagged. Its
 * pairs are left out of every excess, observed and replicated, and the item is listed; parts of a task that one action
 * decides are better declared a testlet. Items of models without a parameter `a`, such as the NRM, are always tested.
 *
 * Each replication costs a calibration, an estimate of each person's theta and Q3, which takes at most 1,000 items.
 *
 * @param matrix - The responses the items were calibrated on.
 * @param items - The calibration's items, in the order of the matrix's items, with their ids.
 * @param random - The generator of the replications' thetas and responses.
 * @param options - The calibration's anchors, held parameters, priors, quadrature and stopping rules, and the
 *   replications, the quantile, the redraws, the slope limit and the estimator of theta.
 * @returns The critical excess, the observed mean Q3 and the pairs above the critical excess, each replication's
 *   largest excess, and the counts of replications, redraws and calibrations that did not converge.
 * @throws `DataError` when the matrix, the items or `random` are invalid, the items' ids are not the matrix's, an
 * option is invalid, as a calibration option that `calibrate` refuses for the matrix and the items' models, which is
 * checked before any draw, there are more than 1,000 items, or a replication's calibration throws it.
 * @see {@link https://doi.org/10.1177/0146621616677520 | Christensen, Makransky and Horton (2017)}
 * @see {@link https://doi.org/10.1177/014662168400800201 | Yen (1984)}
 *
 * @example
 * ```ts
 * import { createRandom } from "@measurekit/psychometrics";
 * import { buildResponseMatrix } from "@measurekit/psychometrics/data";
 * import { yenQ3CriticalValue } from "@measurekit/psychometrics/fit";
 * import { calibrate, twoParameterLogistic } from "@measurekit/psychometrics/irt";
 * import { simulateItems, simulateResponses, simulateThetas } from "@measurekit/psychometrics/sim";
 *
 * // 500 persons answer 8 items; items 1 and 2 share a passage, which moves each person's theta for them.
 * const random = createRandom(7);
 * const items = simulateItems(twoParameterLogistic, 8, random);
 * const thetas = simulateThetas(500, random);
 * const passage = simulateResponses(items.slice(0, 2), thetas.map((theta) => theta + 1.5 * random.normal()), random);
 * const others = simulateResponses(items.slice(2), thetas, random);
 * const matrix = buildResponseMatrix(others.responses.map((row, person) => [...(passage.responses[person] ?? []), ...row]));
 * const fit = calibrate(matrix, twoParameterLogistic);
 * const q3 = yenQ3CriticalValue(matrix, fit.items, createRandom(1), { replications: 20 });
 * q3.flagged.map((pair) => pair.items); // => [["item1", "item2"]]
 * ```
 */
export function yenQ3CriticalValue<TParameters>(
  matrix: ResponseMatrix,
  items: readonly TestItem<TParameters>[],
  random: Random,
  options: Q3CriticalValueOptions<TParameters> = {},
): Q3CriticalValue {
  checkItemsOfMatrix(matrix, items);
  checkItemIds(matrix, items);
  checkRandom(random);
  const settings = bootstrapSettings(options);
  // Every draw can be drawn again before any is calibrated, so the calibration's options are checked here.
  checkCalibrationOptions(
    matrix,
    items.map(({ model }) => model),
    settings.calibration,
  );
  const { slopeLimit, estimator } = settings;
  const tested = items.map((item) => !((slopeOf(item) ?? 0) > slopeLimit));
  const nearDeterministic = items.flatMap((item, j) => (tested[j] === true ? [] : [item.id]));
  const observed = pairExcesses(matrix, items, tested, estimator);
  if (observed === null) {
    return {
      criticalExcess: null,
      meanQ3: null,
      flagged: [],
      largestExcesses: [],
      replications: 0,
      redrawn: 0,
      notConverged: 0,
      nearDeterministic,
    };
  }
  const bootstrap = runBootstrap(matrix, items, random, tested, settings);
  const statistics = Float64Array.from(bootstrap.largestExcesses).sort();
  const criticalExcess = statistics.length === 0 ? null : sampleQuantile(statistics, settings.quantile);
  const flagged =
    criticalExcess === null
      ? []
      : observed.pairs.filter(({ excess }) => excess > criticalExcess).sort((x, y) => y.excess - x.excess);
  return {
    criticalExcess,
    meanQ3: observed.mean,
    flagged,
    largestExcesses: bootstrap.largestExcesses,
    replications: bootstrap.largestExcesses.length,
    redrawn: bootstrap.redrawn,
    notConverged: bootstrap.notConverged,
    nearDeterministic,
  };
}

/** The checked options of the bootstrap, with the calibration's options apart. */
interface BootstrapSettings<TParameters> {
  readonly replications: number;
  readonly quantile: number;
  readonly maxRedraws: number;
  readonly slopeLimit: number;
  readonly estimator: AbilityEstimator;
  readonly calibration: Omit<CalibrationOptions<TParameters>, "model">;
}

/**
 * Checks the options of {@link yenQ3CriticalValue} and returns them with their defaults; the calibration's options are
 * checked against the matrix apart.
 *
 * @throws `DataError` when an option of the bootstrap is invalid or an option is unknown.
 */
function bootstrapSettings<TParameters>(options: Q3CriticalValueOptions<TParameters>): BootstrapSettings<TParameters> {
  checkOptions(options, Q3_CRITICAL_VALUE_OPTIONS);
  const {
    replications = DEFAULT_REPLICATIONS,
    quantile = DEFAULT_QUANTILE,
    maxRedraws = DEFAULT_MAX_REDRAWS,
    slopeLimit = DEFAULT_SLOPE_LIMIT,
    estimator = posteriorMeanEstimate,
    ...calibration
  } = options;
  checkPositiveInteger(replications, "The option replications");
  checkUnitInterval(quantile, "The option quantile");
  checkIntegerFrom(maxRedraws, "The option maxRedraws", 0);
  checkPositive(slopeLimit, "The option slopeLimit");
  checkFunction(
    estimator,
    "The option estimator must be a function of items and responses, such as posteriorMeanEstimate",
  );
  return { replications, quantile, maxRedraws, slopeLimit, estimator, calibration };
}

/** The replications of the bootstrap: their largest excesses, and the counts. */
interface Bootstrap {
  readonly largestExcesses: readonly number[];
  readonly redrawn: number;
  readonly notConverged: number;
}

/**
 * Draws and calibrates the replications, drawing again a replication in which an estimated item lacks a category or
 * no pair has a Q3, until there are `replications` of them or the redraws run out.
 */
function runBootstrap<TParameters>(
  matrix: ResponseMatrix,
  items: readonly TestItem<TParameters>[],
  random: Random,
  tested: readonly boolean[],
  { replications, maxRedraws, estimator, calibration }: BootstrapSettings<TParameters>,
): Bootstrap {
  const fixed = calibration.fixed ?? {};
  const estimated = items.map(({ id }) => !Object.hasOwn(fixed, id));
  const largestExcesses: number[] = [];
  const models = items.map(({ model }) => model);
  let redrawn = 0;
  let notConverged = 0;
  const excessesOf = (simulated: ResponseMatrix): PairExcesses | null => {
    const fit = calibrate(simulated, models, calibration);
    if (!fit.converged) notConverged += 1;
    return pairExcesses(simulated, fit.items, tested, estimator);
  };
  while (largestExcesses.length < replications) {
    const simulated = replicate(matrix, items, random);
    const replicated = hasEveryCategory(simulated, items, estimated) ? excessesOf(simulated) : null;
    if (replicated === null) {
      if (redrawn === maxRedraws) break;
      redrawn += 1;
      continue;
    }
    largestExcesses.push(replicated.largest);
  }
  return { largestExcesses, redrawn, notConverged };
}

/**
 * Checks that the items carry the ids of the matrix's items, in its order.
 *
 * @throws `DataError` naming the first item that does not.
 */
function checkItemIds(matrix: ResponseMatrix, items: readonly TestItem[]): void {
  items.forEach(({ id }, column) => {
    const expected = matrix.itemIds[column];
    if (id !== expected) {
      throw new DataError(
        `Item ${column + 1} needs the id "${expected}" of the matrix's item ${column + 1}, found ${describe(id)}: give the items of a calibration of the matrix, in its order.`,
      );
    }
  });
}

/** Returns an item's slope, the entry `a` of its parameter vector, or `null` for a model without one. */
function slopeOf({ model, parameters }: IrtItem): number | null {
  const index = model.parameters(model.categoryCount(parameters)).findIndex(({ name }) => name === "a");
  return index < 0 ? null : Math.abs(model.toVector(parameters)[index] ?? 0);
}

/** The pair excesses of a matrix: each tested pair's Q3 less their mean, the largest of them, and the pairs. */
interface PairExcesses {
  readonly mean: number;
  readonly largest: number;
  readonly pairs: readonly Q3Pair[];
}

/**
 * Returns the excesses of the pairs of tested items whose Q3 is defined, at each person's estimate of theta, or `null`
 * without such a pair. A person without responses has no residual, so the theta it is given is not read.
 */
function pairExcesses(
  matrix: ResponseMatrix,
  items: readonly TestItem[],
  tested: readonly boolean[],
  estimator: AbilityEstimator,
): PairExcesses | null {
  const thetas = abilityEstimates(matrix, items, { estimator }).map(({ theta }) => theta ?? 0);
  const q3 = yenQ3(matrix, items, thetas);
  const pairs: { items: readonly [string, string]; q3: number }[] = [];
  for (let i = 0; i < items.length; i++) {
    for (let j = i + 1; j < items.length; j++) {
      const value = q3[i]?.[j] ?? null;
      if (value === null || tested[i] !== true || tested[j] !== true) continue;
      pairs.push({ items: [matrix.itemIds[i] ?? "", matrix.itemIds[j] ?? ""], q3: value });
    }
  }
  if (pairs.length === 0) return null;
  let sum = 0;
  for (const pair of pairs) sum += pair.q3;
  const mean = sum / pairs.length;
  const excesses = pairs.map((pair) => pair.q3 - mean);
  let largest = Number.NEGATIVE_INFINITY;
  for (const excess of excesses) largest = Math.max(largest, excess);
  return { mean, largest, pairs: pairs.map((pair, k) => ({ ...pair, excess: excesses[k] ?? 0 })) };
}

/**
 * Draws one replication: a standard normal theta for each person and the responses to the items, missing where the
 * matrix's are.
 */
function replicate(matrix: ResponseMatrix, items: readonly TestItem[], random: Random): ResponseMatrix {
  const thetas = simulateThetas(matrix.responses.length, random);
  const simulated = simulateResponses(items, thetas, random);
  if (!matrix.responses.some((row) => row.includes(null))) return simulated;
  const rows = simulated.responses.map((row, person) =>
    row.map((response, column) => ((matrix.responses[person]?.[column] ?? null) === null ? null : response)),
  );
  return buildResponseMatrix(rows, { itemIds: matrix.itemIds });
}

/** Returns whether each estimated item has a response in each of its categories. */
function hasEveryCategory(matrix: ResponseMatrix, items: readonly IrtItem[], estimated: readonly boolean[]): boolean {
  return items.every((item, column) => {
    if (estimated[column] !== true) return true;
    const seen = new Uint8Array(item.model.categoryCount(item.parameters));
    let missing = seen.length;
    for (const row of matrix.responses) {
      const response = row[column] ?? null;
      if (response === null || seen[response] === 1) continue;
      seen[response] = 1;
      missing -= 1;
      if (missing === 0) return true;
    }
    return false;
  });
}
