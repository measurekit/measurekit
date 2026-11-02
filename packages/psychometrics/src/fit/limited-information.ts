import {
  checkArray,
  checkChoice,
  checkFinite,
  checkOptions,
  checkPositive,
  checkRecord,
  chiSquareUpperTail,
  DataError,
  describe,
} from "@measurekit/core";
import { checkOpenUnitInterval } from "../checks.js";
import type { ResponseMatrix } from "../data/internal.js";
import {
  type Calibration,
  checkThetaDistribution,
  type IrtItem,
  type IrtModel,
  itemTrace,
  nominalResponse,
} from "../irt/internal.js";
import {
  addPairDerivatives,
  addWeightDerivatives,
  type FirstOrderRows,
  independenceStatistic,
  limitedInformationStatistic,
  MAX_MOMENT_ITEMS,
  type MomentTables,
  modelMoments,
  pairRow,
  residualCorrelationRoot,
  rmseaInterval,
  rowCount,
  type SampleMoments,
  sampleMoments,
  tableMoments,
} from "../math/internal.js";
import { checkItems } from "./inputs.js";

/** Options of {@link limitedInformationFit}. */
export interface LimitedInformationOptions {
  /**
   * The statistic: `"m2"`, on the first- and second-order margins of dichotomous items; `"m2-star"`, on the means and
   * the mean products of the scores; `"c2"`, on the categories' shares and the mean products. The last two also take
   * ordered polytomous items.
   *
   * @defaultValue `"m2-star"`
   */
  readonly statistic?: "m2" | "m2-star" | "c2";
  /**
   * The ids of the calibration's anchors, which add no parameters. Checked against the calibration's AIC.
   *
   * @defaultValue none
   */
  readonly fixed?: readonly string[];
  /**
   * The level of the RMSEA interval.
   *
   * @defaultValue `0.9`
   */
  readonly level?: number;
}

/** The fit of a calibrated model to the first- and second-order margins of its test. */
export interface LimitedInformationFit {
  /** The statistic; `null` without degrees of freedom or when its covariance matrix is singular. */
  readonly statistic: number | null;
  /**
   * The moments less the rank of Δ: the estimated parameters, unless the first- and second-order margins do not
   * identify them all.
   */
  readonly degreesOfFreedom: number;
  /** The probability of a chi-square at least as large when the model fits; `null` with `statistic`. */
  readonly significance: number | null;
  /** The RMSEA, √(max(0, χ² / df - 1) / (N - 1)), as mirt's; `null` with `statistic`. */
  readonly rmsea: number | null;
  /** The interval of the RMSEA at `level`, as mirt's; `null` with `statistic`. */
  readonly rmseaInterval: readonly [lower: number, upper: number] | null;
  /** The root mean square of the residual correlations over the pairs of items; `null` for an item without variance. */
  readonly srmsr: number | null;
  /**
   * The Tucker-Lewis index against the independence model, which can pass 1; `null` when the independence model's
   * statistic is not above the model's, as mirt leaves it.
   */
  readonly tli: number | null;
  /** The comparative fit index against the independence model, held to [0, 1]; `null` with `tli`. */
  readonly cfi: number | null;
}

type Statistic = NonNullable<LimitedInformationOptions["statistic"]>;

const STATISTICS: readonly Statistic[] = ["m2", "m2-star", "c2"];
const DEFAULT_LEVEL = 0.9;

/**
 * Returns the limited-information fit of a calibration: M2, M2* or C2, with RMSEA, SRMSR, TLI and CFI, as mirt's `M2`.
 *
 * @remarks
 * The statistic is N e′ (Ξ⁻¹ - Ξ⁻¹ Δ (Δ′ Ξ⁻¹ Δ)⁻¹ Δ′ Ξ⁻¹) e, for e the first- and second-order residuals, Ξ their
 * covariance matrix and Δ their Jacobian. It integrates over the nodes and weights of the calibration's group, so it
 * takes one group of a normal distribution, and no covariates or testlets. Δ has a column per estimated parameter: the
 * parameter a model shares among items has one column, an anchor has none, and an estimated normal distribution has the
 * columns of its mean and standard deviation. Those parameters must equal the count in the calibration's AIC. The
 * degrees of freedom are the moments less the rank of Δ, which is the number of parameters unless the margins leave
 * some unidentified; the statistic then uses the whole complement of Δ's span.
 *
 * A `Calibration` does not record its anchors, and the check of the AIC catches a list of the wrong size, not a list of
 * the right size that swaps anchors for estimated items of the same model, which gives a wrong Δ.
 *
 * The independence model holds each item at its observed shares, without a trait. Its statistic is N Σ_(i<j) r_ij², on
 * J (J - 1) / 2 degrees of freedom. TLI = (χ²₀/df₀ - χ²/df) / (χ²₀/df₀ - 1) and CFI = 1 - (χ² - df) / (χ²₀ - df₀).
 * Uses the persons who answered every item; each counts once, also for a calibration with `personWeights`, whose
 * weights the moments do not read.
 *
 * @param matrix - The responses of the persons.
 * @param calibration - The calibration of the matrix's items, in its order, on one group.
 * @param options - The statistic, the anchors and the level of the interval.
 * @returns The statistic, its degrees of freedom and significance, and the fit indices, each `null` where undefined.
 * @throws `DataError` when the matrix, the calibration or an option is invalid, a response is not a category of its item,
 * fewer than two persons answered every item, there are more than 60 items, the calibration has several groups,
 * covariates, testlets or an empirical histogram, `"m2"` meets a polytomous item, an item is nominal, or the estimated
 * parameters differ from the count in the calibration's AIC.
 * @see {@link https://doi.org/10.1007/s11336-005-1295-9 | Maydeu-Olivares and Joe (2006)}
 * @see {@link https://doi.org/10.1111/j.2044-8317.2012.02050.x | Cai and Hansen (2013)}
 */
export function limitedInformationFit<TParameters>(
  matrix: ResponseMatrix,
  calibration: Calibration<TParameters>,
  options: LimitedInformationOptions = {},
): LimitedInformationFit {
  checkOptions(options);
  const { statistic = "m2-star", fixed = [], level = DEFAULT_LEVEL } = options;
  checkChoice(statistic, "The option statistic", STATISTICS);
  checkOpenUnitInterval(level, "The option level");
  const { items, group } = checkCalibration(matrix, calibration);
  checkStatistic(items, statistic);
  const columns = parameterColumns(items, checkFixed(fixed, matrix.itemIds), group.normal);
  checkParameterCount(calibration, columns.count);
  const complete = matrix.responses.filter((row) => !row.includes(null));
  if (complete.length < 2) {
    throw new DataError(
      `A limited-information statistic needs at least two persons who answered every item, found ${complete.length}.`,
    );
  }
  const tables = momentTables(items, group, statistic);
  const moments = modelMoments(tables);
  const sample = sampleMoments(complete, items.length);
  const rows = rowCount(tables);
  const srmsr = residualCorrelationRoot(sample, tableMoments(tables));
  const { statistic: chiSquare, rank } =
    rows > columns.count
      ? limitedInformationStatistic(
          residuals(complete, items, tables, sample, moments.expected, statistic),
          moments.covariance,
          jacobian(items, group, tables, moments.conditional, columns, statistic),
          columns.count,
          sample.persons,
        )
      : { statistic: null, rank: columns.count };
  const degreesOfFreedom = rows - rank;
  return chiSquare === null
    ? { statistic: null, degreesOfFreedom, ...NO_INDICES, srmsr }
    : {
        statistic: chiSquare,
        degreesOfFreedom,
        significance: chiSquareUpperTail(chiSquare, degreesOfFreedom),
        rmsea: Math.sqrt(Math.max(0, chiSquare / degreesOfFreedom - 1) / (sample.persons - 1)),
        rmseaInterval: rmseaInterval(chiSquare, degreesOfFreedom, sample.persons, level),
        srmsr,
        ...comparativeIndices(chiSquare, degreesOfFreedom, sample, items.length),
      };
}

const NO_INDICES = { significance: null, rmsea: null, rmseaInterval: null, tli: null, cfi: null } as const;

/** The distribution of theta of a calibration, as the statistic reads it. */
interface Population {
  readonly nodes: Float64Array;
  readonly weights: Float64Array;
  /** Whether the distribution's mean and standard deviation are estimated. */
  readonly normal: boolean;
  readonly mean: number;
  readonly sd: number;
}

/**
 * Checks a calibration against the matrix and returns its items and its population.
 *
 * @throws `DataError` when it is not a calibration of the matrix's items, or its population is not one group of a normal
 * distribution without covariates or testlets.
 */
function checkCalibration<TParameters>(
  matrix: ResponseMatrix,
  calibration: Calibration<TParameters>,
): { readonly items: readonly IrtItem[]; readonly group: Population } {
  checkRecord(calibration, "The calibration must be the result of calibrate");
  const { items, groups, testlets, regression } = calibration;
  checkItems(matrix, items);
  items.forEach(({ id }, column) => {
    const expected = matrix.itemIds[column];
    if (id !== expected) {
      throw new DataError(`Column ${column + 1} is item "${expected}", but the calibration has ${describe(id)} there.`);
    }
  });
  if (items.length > MAX_MOMENT_ITEMS) {
    throw new DataError(
      `A limited-information statistic takes at most ${MAX_MOMENT_ITEMS} items, found ${items.length}.`,
    );
  }
  checkArray(groups, "The groups of the calibration must be an array");
  checkArray(testlets, "The testlets of the calibration must be an array");
  if (regression !== undefined) throw new DataError("A limited-information statistic does not take covariates.");
  if (testlets.length > 0) throw new DataError("A limited-information statistic does not take testlets.");
  if (groups.length !== 1) {
    throw new DataError(`A limited-information statistic takes one group, found ${groups.length}.`);
  }
  const [group] = groups;
  if (group === undefined) throw new DataError("A limited-information statistic takes one group, found 0.");
  if (group.distribution === "empirical-histogram") {
    throw new DataError("A limited-information statistic does not take an empirical histogram.");
  }
  const { nodes, weights } = checkThetaDistribution(group, "population of the calibration");
  const normal = group.distribution === "normal";
  if (normal) {
    checkFinite(group.mean, "The mean of the calibration's group");
    checkPositive(group.sd, "The standard deviation of the calibration's group");
  }
  return { items, group: { nodes, weights, normal, mean: group.mean, sd: group.sd } };
}

/** Throws when the statistic does not take an item: only M2 is limited to two categories, and a nominal item has no order. */
function checkStatistic(items: readonly IrtItem[], statistic: Statistic): void {
  items.forEach((item, index) => {
    const count = item.model.categoryCount(item.parameters);
    if (count === 2) return;
    if (statistic === "m2") {
      throw new DataError(`The statistic "m2" needs dichotomous items, but item ${index + 1} has ${count} categories.`);
    }
    if (item.model === nominalResponse) {
      throw new DataError(
        `Item ${index + 1} is nominal, and a limited-information statistic needs ordered categories.`,
      );
    }
  });
}

/**
 * Returns the indices of the anchors.
 *
 * @throws `DataError` when `fixed` is not an array of item ids.
 */
function checkFixed(fixed: readonly string[], itemIds: readonly string[]): ReadonlySet<number> {
  checkArray(fixed, "The option fixed must be an array of item ids");
  return new Set(
    fixed.map((id) => {
      const index = itemIds.indexOf(id);
      if (index < 0) throw new DataError(`The option fixed names ${describe(id)}, which is not an item id.`);
      return index;
    }),
  );
}

/** Where each estimated parameter goes in Δ. */
interface ParameterColumns {
  /** The column of each entry of each item's parameter vector; empty for an anchor. */
  readonly ofItems: readonly Int32Array[];
  /** The columns of the mean and the standard deviation of an estimated normal distribution, the last two; -1 otherwise. */
  readonly mean: number;
  readonly sd: number;
  /** The number of columns: the estimated parameters. */
  readonly count: number;
}

/** Assigns the columns of Δ, as calibration assigns its free parameters: one for what the items of a model share. */
function parameterColumns(items: readonly IrtItem[], fixed: ReadonlySet<number>, normal: boolean): ParameterColumns {
  const sharing = new Map<IrtModel, Map<string, number>>();
  let count = 0;
  const ofItems = items.map((item, index) => {
    if (fixed.has(index)) return new Int32Array(0);
    const { model, parameters } = item;
    const shared = sharing.get(model) ?? new Map<string, number>();
    sharing.set(model, shared);
    return Int32Array.from(model.parameters(model.categoryCount(parameters)), ({ name, shared: common }) => {
      const existing = common === true ? shared.get(name) : undefined;
      if (existing !== undefined) return existing;
      if (common === true) shared.set(name, count);
      return count++;
    });
  });
  return normal ? { ofItems, mean: count, sd: count + 1, count: count + 2 } : { ofItems, mean: -1, sd: -1, count };
}

/**
 * Checks the parameters against those a calibration counts in its AIC.
 *
 * @throws `DataError` asking for the anchors' ids when they differ.
 */
function checkParameterCount<TParameters>({ aic, logLikelihood }: Calibration<TParameters>, count: number): void {
  checkFinite(aic, "The AIC of the calibration");
  checkFinite(logLikelihood, "The log-likelihood of the calibration");
  const counted = Math.round((aic + 2 * logLikelihood) / 2);
  if (counted !== count) {
    throw new DataError(
      `The calibration estimated ${counted} parameters, but these items have ${count}; give the anchors' ids in the option fixed.`,
    );
  }
}

/** Returns the scores of the items at each node, and the first-order rows of the statistic. */
function momentTables(items: readonly IrtItem[], { nodes, weights }: Population, statistic: Statistic): MomentTables {
  const itemCount = items.length;
  const nodeCount = nodes.length;
  const means = new Float64Array(nodeCount * itemCount);
  const squares = new Float64Array(nodeCount * itemCount);
  const traces = items.map((item) => {
    const count = item.model.categoryCount(item.parameters);
    return { count, trace: itemTrace(item, count, nodes) };
  });
  traces.forEach(({ count, trace }, i) => {
    for (let q = 0; q < nodeCount; q++) {
      for (let k = 1; k < count; k++) {
        const share = trace[q * count + k] ?? 0;
        means[q * itemCount + i] = (means[q * itemCount + i] ?? 0) + k * share;
        squares[q * itemCount + i] = (squares[q * itemCount + i] ?? 0) + k * k * share;
      }
    }
  });
  const first =
    statistic === "c2"
      ? categoryRows(traces, nodeCount)
      : { items: Int32Array.from(items, (_, i) => i), means, scores: squares, squares };
  return { weights, itemCount, means, squares, first };
}

/** Returns the first-order rows of C2: the indicators of the categories above the lowest. */
function categoryRows(
  traces: readonly { readonly count: number; readonly trace: Float64Array }[],
  nodeCount: number,
): FirstOrderRows {
  const items = Int32Array.from(traces.flatMap(({ count }, i) => new Array<number>(count - 1).fill(i)));
  const means = new Float64Array(nodeCount * items.length);
  const scores = new Float64Array(nodeCount * items.length);
  let row = 0;
  for (const { count, trace } of traces) {
    for (let k = 1; k < count; k++, row++) {
      for (let q = 0; q < nodeCount; q++) {
        const share = trace[q * count + k] ?? 0;
        means[q * items.length + row] = share;
        scores[q * items.length + row] = k * share;
      }
    }
  }
  return { items, means, scores, squares: means };
}

/** Returns the observed moments less the expected ones. */
function residuals(
  complete: readonly (readonly (number | null)[])[],
  items: readonly IrtItem[],
  tables: MomentTables,
  sample: SampleMoments,
  expected: Float64Array,
  statistic: Statistic,
): Float64Array {
  const { itemCount, first } = tables;
  const result = new Float64Array(expected.length);
  if (statistic === "c2") result.set(categoryShares(complete, items, first.items.length));
  else result.set(sample.means);
  for (let i = 0; i < itemCount; i++) {
    for (let j = i + 1; j < itemCount; j++) {
      result[pairRow(tables, i, j)] = sample.products[i * itemCount + j] ?? 0;
    }
  }
  return result.map((value, row) => value - (expected[row] ?? 0));
}

/** Returns the share of the persons in each category above each item's lowest, in the order of the rows of C2. */
function categoryShares(
  complete: readonly (readonly (number | null)[])[],
  items: readonly IrtItem[],
  rows: number,
): Float64Array {
  const shares = new Float64Array(rows);
  let offset = 0;
  items.forEach((item, i) => {
    for (const row of complete) {
      const response = row[i] ?? 0;
      if (response > 0) shares[offset + response - 1] = (shares[offset + response - 1] ?? 0) + 1;
    }
    offset += item.model.categoryCount(item.parameters) - 1;
  });
  return shares.map((count) => count / complete.length);
}

/** Returns Δ, by rows. */
function jacobian(
  items: readonly IrtItem[],
  group: Population,
  tables: MomentTables,
  conditional: Float64Array,
  columns: ParameterColumns,
  statistic: Statistic,
): Float64Array {
  const rows = rowCount(tables);
  const width = columns.count;
  const delta = new Float64Array(rows * width);
  const { nodes, weights } = group;
  let firstRow = 0;
  items.forEach((item, i) => {
    const count = item.model.categoryCount(item.parameters);
    const derivatives = categoryDerivatives(item, count, nodes);
    columns.ofItems[i]?.forEach((column, entry) => {
      const entries = columns.ofItems[i]?.length ?? 0;
      const share = (k: number): number => categoryDerivative(derivatives, weights, count, entries, k, entry);
      if (statistic === "c2") {
        for (let k = 1; k < count; k++) delta[(firstRow + k - 1) * width + column] = share(k);
      }
      const score = scoreDerivative(derivatives, nodes.length, count, entries, entry);
      if (statistic !== "c2") delta[firstRow * width + column] = meanOf(score, weights);
      addPairDerivatives(tables, delta, width, column, i, score);
    });
    firstRow += statistic === "c2" ? count - 1 : 1;
  });
  if (group.normal) {
    const { mean, sd } = weightDerivatives(group);
    addWeightDerivatives(conditional, delta, rows, width, columns.mean, mean);
    addWeightDerivatives(conditional, delta, rows, width, columns.sd, sd);
  }
  return delta;
}

/** Returns dP(k | θ)/dβ of an item, by node, then category, then entry of its parameter vector. */
function categoryDerivatives(item: IrtItem, count: number, nodes: Float64Array): Float64Array {
  const { model, parameters } = item;
  const size = model.parameters(count).length;
  const row = new Float64Array(count * size);
  const table = new Float64Array(nodes.length * row.length);
  nodes.forEach((theta, q) => {
    model.parameterDerivatives(parameters, theta, row);
    table.set(row, q * row.length);
  });
  return table;
}

/** Returns Σ_q w_q dP(k | θ_q)/dβ for one entry. */
function categoryDerivative(
  table: Float64Array,
  weights: Float64Array,
  count: number,
  entries: number,
  k: number,
  entry: number,
): number {
  let sum = 0;
  for (let q = 0; q < weights.length; q++) sum += (weights[q] ?? 0) * (table[(q * count + k) * entries + entry] ?? 0);
  return sum;
}

/** Returns dE(x | θ)/dβ at each node for one entry. */
function scoreDerivative(
  table: Float64Array,
  nodes: number,
  count: number,
  entries: number,
  entry: number,
): Float64Array {
  const derivative = new Float64Array(nodes);
  for (let q = 0; q < nodes; q++) {
    let sum = 0;
    for (let k = 1; k < count; k++) sum += k * (table[(q * count + k) * entries + entry] ?? 0);
    derivative[q] = sum;
  }
  return derivative;
}

/** Returns Σ_q w_q v_q. */
function meanOf(values: Float64Array, weights: Float64Array): number {
  return values.reduce((sum, value, q) => sum + value * (weights[q] ?? 0), 0);
}

/** Returns dw_q/dμ and dw_q/dσ of a normal distribution's weights, proportional to exp(-(θ_q - μ)² / 2σ²) at the nodes. */
function weightDerivatives({ nodes, weights, mean, sd }: Population): {
  readonly mean: Float64Array;
  readonly sd: Float64Array;
} {
  const scores = nodes.map((theta) => (theta - mean) / (sd * sd));
  const squares = nodes.map((theta) => (theta - mean) ** 2 / sd ** 3);
  const expected = (values: Float64Array): number =>
    values.reduce((sum, value, q) => sum + value * (weights[q] ?? 0), 0);
  const centered = (values: Float64Array): Float64Array => {
    const center = expected(values);
    return values.map((value, q) => (weights[q] ?? 0) * (value - center));
  };
  return { mean: centered(scores), sd: centered(squares) };
}

/** Returns TLI and CFI against the independence model; both `null` when it is undefined or does not fit worse. */
function comparativeIndices(
  chiSquare: number,
  degreesOfFreedom: number,
  sample: SampleMoments,
  itemCount: number,
): { readonly tli: number | null; readonly cfi: number | null } {
  const independence = independenceStatistic(sample, itemCount);
  const independenceDegrees = (itemCount * (itemCount - 1)) / 2;
  if (independence === null || independenceDegrees === 0 || independence <= chiSquare) return { tli: null, cfi: null };
  const ratio = independence / independenceDegrees;
  const tli = (ratio - chiSquare / degreesOfFreedom) / (ratio - 1);
  const cfi = 1 - (chiSquare - degreesOfFreedom) / (independence - independenceDegrees);
  return Number.isFinite(tli) && Number.isFinite(cfi)
    ? { tli, cfi: Math.min(1, Math.max(0, cfi)) }
    : { tli: null, cfi: null };
}
