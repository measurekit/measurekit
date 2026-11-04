import {
  checkArray,
  checkBoolean,
  checkChoice,
  checkNonNegative,
  checkOptions,
  checkPositive,
  checkRecord,
  checkUnitInterval,
  DataError,
} from "@measurekit/core";
import type { BivariateDistribution } from "./bivariate.js";
import { neatTables } from "./bivariate.js";
import { penaltyBandwidth } from "./kernel-bandwidth.js";
import {
  type Continuized,
  cdfGradient,
  continuize,
  kernelCdf,
  kernelDensity,
  kernelQuantile,
  LINEAR_BANDWIDTH,
  percentRelativeError,
  scoreMoments,
} from "./kernel-continuization.js";
import {
  bivariateSmoothedTable,
  type Margin,
  mixedMargin,
  ownMargin,
  type SmoothedTable,
  syntheticMargin,
  tableMargin,
  univariateTable,
} from "./kernel-design.js";

/**
 * The data of a kernel equating, by design: score distributions, as the number of persons at each score from 0, and the
 * degrees of their log-linear presmoothing. A bivariate table is presmoothed by
 * {@link bivariateLogLinearPresmoothing} with `degrees` and `crossDegree`, as {@link frequencyEstimationEquating} takes
 * the tables of the anchor designs.
 */
export type KernelData =
  | {
      /** Two equivalent groups, each of which took one form. */
      readonly design: "equivalent-groups";
      /** The persons at each score on the form to equate, from 0. */
      readonly from: readonly number[];
      /** The persons at each score on the form to equate to, from 0. */
      readonly to: readonly number[];
      /** The degrees of the log-linear presmoothing of each distribution. */
      readonly degrees: readonly [from: number, to: number];
    }
  | {
      /** One group, which took both forms. */
      readonly design: "single-group";
      /** The persons at each pair of scores: by score on the form to equate, then score on the form to equate to. */
      readonly joint: BivariateDistribution;
      /** The degrees of the scores of each form in the bivariate presmoothing. */
      readonly degrees: readonly [from: number, to: number];
      /**
       * The highest power of each score in the cross products of the presmoothing.
       *
       * @defaultValue `1`
       */
      readonly crossDegree?: number;
    }
  | {
      /** Two groups, which took both forms in opposite orders. */
      readonly design: "counterbalanced";
      /** The group that took the form to equate first: persons by its score, then the other form's. */
      readonly first: BivariateDistribution;
      /** The group that took the form to equate second: persons by its score, then the other form's. */
      readonly second: BivariateDistribution;
      /** The degrees of the scores of each form in the bivariate presmoothing. */
      readonly degrees: readonly [from: number, to: number];
      /**
       * The highest power of each score in the cross products of the presmoothing.
       *
       * @defaultValue `1`
       */
      readonly crossDegree?: number;
    }
  | {
      /** Two groups, each of which took one form and a common anchor, linked through it or post-stratified on it. */
      readonly design: "chained" | "post-stratification";
      /** The persons of the group that took the form to equate, by total score, then anchor score. */
      readonly from: BivariateDistribution;
      /** The persons of the group that took the form to equate to, by total score, then anchor score. */
      readonly to: BivariateDistribution;
      /** The degrees of the total and the anchor scores in the bivariate presmoothing. */
      readonly degrees: readonly [total: number, anchor: number];
      /**
       * The highest power of each score in the cross products of the presmoothing.
       *
       * @defaultValue `1`
       */
      readonly crossDegree?: number;
    };

/** The bandwidths h of the continuized distributions. */
export interface KernelBandwidths {
  /** The bandwidth of the form to equate. */
  readonly from: number;
  /** The bandwidth of the form to equate to. */
  readonly to: number;
  /** In the chained design, the bandwidth of the anchor in the group that took the form to equate. */
  readonly fromAnchor?: number;
  /** In the chained design, the bandwidth of the anchor in the group that took the form to equate to. */
  readonly toAnchor?: number;
}

/** Options of {@link kernelEquating}. */
export interface KernelEquatingOptions {
  /**
   * Bandwidths, each a positive number; each omitted one minimizes the penalty, or is 1000 σ with `linear`.
   * `fromAnchor` and `toAnchor` belong to the chained design, which the other designs ignore.
   *
   * @defaultValue the penalty's minimum
   */
  readonly bandwidths?: Partial<KernelBandwidths>;
  /**
   * Linear equating: each bandwidth not given is 1000 times the standard deviation of its scores.
   *
   * @defaultValue `false`
   */
  readonly linear?: boolean;
  /**
   * The weight of the first group of `"counterbalanced"`, or of the population of the group that took the form to
   * equate in the synthetic population of `"post-stratification"`: a number from 0 to 1. The other designs ignore it.
   *
   * @defaultValue `0.5`
   */
  readonly weight?: number;
  /**
   * The weight K of the second term of the penalty, which counts the turns of the density near the scores; 0 turns it
   * off.
   *
   * @defaultValue `0`
   */
  readonly penaltyWeight?: number;
}

/** A kernel equating with its standard errors. */
export interface KernelEquating {
  /** The equated score e(x) of each score of the form to equate, from 0. */
  readonly equated: readonly number[];
  /** The standard error of each equated score, by the delta method, the bandwidths held fixed. */
  readonly standardErrors: readonly number[];
  /** The bandwidths used. */
  readonly bandwidths: KernelBandwidths;
  /**
   * The percent relative error of the first ten moments of the equated scores against those of the form to equate to;
   * in the chained design, of the link of the total scores to the anchor in the group that took the form to equate.
   */
  readonly percentRelativeError: readonly number[];
  /** In the chained design, the percent relative error of the link of the anchor to the form to equate to. */
  readonly anchorPercentRelativeError?: readonly number[];
  /** The vector behind each score's standard error, whose length is the standard error. */
  readonly errorVectors: readonly (readonly number[])[];
  /** Whether every presmoothing converged and no bandwidth's search stopped at its bound. */
  readonly converged: boolean;
}

const DESIGNS = ["equivalent-groups", "single-group", "counterbalanced", "chained", "post-stratification"];
const DEFAULT_WEIGHT = 0.5;
const DEFAULT_CROSS_DEGREE = 1;

/** The presmoothed tables of a design and the two margins that it equates, or the four of the chained design. */
interface DesignMargins {
  readonly tables: readonly SmoothedTable[];
  readonly from: Margin;
  readonly to: Margin;
  /** The anchor margins of the chained design, in the group of `from`, then of `to`. */
  readonly anchors: readonly [Margin, Margin] | null;
}

/**
 * Returns the kernel equating of the scores of a form to the scale of another, in any of five data collection designs,
 * with the standard error of each equated score.
 *
 * @remarks
 * Each score distribution is presmoothed by a log-linear model and continuized by a Gaussian kernel, F_h(t) = Σ_j r_j
 * Φ((t - a x_j - (1 - a) μ) / (a h)), a = √(σ² / (σ² + h²)), which keeps its mean μ and variance σ² (the variance
 * times n / (n - 1), as kequate). The equated score is e(x) = G_h⁻¹(F_h(x)); the chained design links the form to
 * equate to the anchor in its group and the anchor to the other form in the other group. Each bandwidth not given
 * minimizes the penalty Σ_j (r_j - f_h(x_j))² + K Σ_j A_j over (0, 4], A_j marking a turn of the density between x_j -
 * 1/4 and x_j + 1/4; a very large bandwidth gives linear equating. The standard errors come from the delta method: the
 * Jacobian of e(x) through the design function and the matrices C of the log-linear fits, Cov(p̂) = C Cᵀ, with the
 * bandwidths fixed, as kequate.
 *
 * @param data - The design, its score distributions and the degrees of their presmoothing.
 * @param options - Bandwidths, linear equating, the weight of a group and of the penalty's second term.
 * @returns The equated scores, their standard errors and error vectors, the bandwidths, the percent relative errors of
 *   the moments, and whether every step converged.
 * @throws `DataError` when the data or the options are invalid: a distribution as the equating and presmoothing
 * functions refuse it, a degree out of its range, fewer than two persons or scores without variance, tables of a design
 * that do not fit together, a centered log-linear basis that is singular, or bandwidths so small that an equated score
 * or its standard error is not finite.
 * @see {@link https://doi.org/10.1007/b97446 | von Davier, Holland and Thayer (2004)}
 * @see {@link https://doi.org/10.1007/978-3-319-51824-4 | González and Wiberg (2017)}
 */
export function kernelEquating(data: KernelData, options: KernelEquatingOptions = {}): KernelEquating {
  checkOptions(options);
  const { bandwidths = {}, linear = false, weight = DEFAULT_WEIGHT, penaltyWeight = 0 } = options;
  checkRecord(bandwidths, "The option bandwidths must be an object { from, to, fromAnchor, toAnchor }");
  for (const [name, value] of Object.entries(bandwidths)) {
    if (value !== undefined) checkPositive(value, `The bandwidth ${name}`);
  }
  checkBoolean(linear, "The option linear");
  checkUnitInterval(weight, "The option weight");
  checkNonNegative(penaltyWeight, "The option penaltyWeight");
  const margins = designMargins(data, weight);
  const choose = (margin: Margin, label: string, given: number | undefined): Choice =>
    chooseBandwidth(margin, label, given, linear, penaltyWeight);
  const tablesConverged = margins.tables.every((table) => table.converged);
  if (margins.anchors === null) {
    const from = choose(margins.from, "from", bandwidths.from);
    const to = choose(margins.to, "to", bandwidths.to);
    return {
      ...checkFinite(singleLink(margins, from.continuized, to.continuized)),
      bandwidths: { from: from.bandwidth, to: to.bandwidth },
      converged: tablesConverged && from.converged && to.converged,
    };
  }
  const chain: [Choice, Choice, Choice, Choice] = [
    choose(margins.from, "from", bandwidths.from),
    choose(margins.anchors[0], "from anchor", bandwidths.fromAnchor),
    choose(margins.anchors[1], "to anchor", bandwidths.toAnchor),
    choose(margins.to, "to", bandwidths.to),
  ];
  return {
    ...checkFinite(
      chainedLink(
        margins,
        chain.map(({ continuized }) => continuized),
      ),
    ),
    bandwidths: {
      from: chain[0].bandwidth,
      to: chain[3].bandwidth,
      fromAnchor: chain[1].bandwidth,
      toAnchor: chain[2].bandwidth,
    },
    converged: tablesConverged && chain.every((choice) => choice.converged),
  };
}

/**
 * Returns the standard error of the difference of two kernel equatings of one design and the same data, such as kernel
 * and linear, or chained and post-stratification: at each score, the length of the difference of their error vectors,
 * as kequate's `genseed`.
 *
 * @param first - One equating.
 * @param second - The other, of the same design and data.
 * @returns The standard error of e₁(x) - e₂(x) at each score of the form to equate.
 * @throws `DataError` when either is not a kernel equating, or their error vectors differ in number or length.
 */
export function kernelEquatingDifference(first: KernelEquating, second: KernelEquating): readonly number[] {
  checkRecord(first, "The first kernel equating must be an object { errorVectors, ... }");
  checkRecord(second, "The second kernel equating must be an object { errorVectors, ... }");
  checkArray(first.errorVectors, "The error vectors of the first kernel equating must be an array");
  checkArray(second.errorVectors, "The error vectors of the second kernel equating must be an array");
  if (first.errorVectors.length !== second.errorVectors.length) {
    throw new DataError(
      `The two kernel equatings need error vectors for the same scores, found ${first.errorVectors.length} and ${second.errorVectors.length}.`,
    );
  }
  return first.errorVectors.map((vector, x) => {
    const other = second.errorVectors[x];
    checkArray(vector, `Error vector ${x + 1} of the first kernel equating must be an array`);
    checkArray(other, `Error vector ${x + 1} of the second kernel equating must be an array`);
    if (vector.length !== other?.length) {
      throw new DataError(
        `The two kernel equatings need error vectors of one length, from one design and data, found ${vector.length} and ${other?.length} at the score ${x}.`,
      );
    }
    let sum = 0;
    vector.forEach((value, k) => {
      const difference = value - (other[k] ?? 0);
      sum += difference * difference;
    });
    if (!Number.isFinite(sum)) throw new DataError(`The error vectors at the score ${x} are not finite.`);
    return Math.sqrt(sum);
  });
}

/** A margin continuized with its bandwidth, and whether the bandwidth's search converged. */
interface Choice {
  readonly continuized: Continuized;
  readonly bandwidth: number;
  readonly converged: boolean;
}

function chooseBandwidth(
  margin: Margin,
  label: string,
  given: number | undefined,
  linear: boolean,
  penaltyWeight: number,
): Choice {
  const moments = scoreMoments(margin.probabilities, margin.persons, label);
  const choice =
    given !== undefined
      ? { bandwidth: given, converged: true }
      : linear
        ? { bandwidth: LINEAR_BANDWIDTH * Math.sqrt(moments[1]), converged: true }
        : penaltyBandwidth(margin.probabilities, moments, penaltyWeight);
  return { ...choice, continuized: continuize(margin.probabilities, moments, choice.bandwidth) };
}

/**
 * Checks the data and presmooths its tables into the margins of its design.
 *
 * @throws `DataError` when the data are invalid.
 */
function designMargins(data: KernelData, weight: number): DesignMargins {
  checkRecord(data, "Kernel equating data must be an object { design, degrees, ... } with the design's distributions");
  checkChoice(data.design, "The design", DESIGNS);
  checkArray(data.degrees, "The degrees must be an array of the degrees of the two scores");
  if (data.degrees.length !== 2) {
    throw new DataError(`The degrees must have the degrees of the two scores, found ${data.degrees.length}.`);
  }
  const degrees: readonly [number, number] = [data.degrees[0], data.degrees[1]];
  if (data.design === "equivalent-groups") {
    const tables: [SmoothedTable, SmoothedTable] = [
      univariateTable(data.from, degrees[0], "from"),
      univariateTable(data.to, degrees[1], "to"),
    ];
    const widths = tables.map((table) => table.width);
    return { tables, from: ownMargin(widths, tables[0], 0), to: ownMargin(widths, tables[1], 1), anchors: null };
  }
  const { crossDegree = DEFAULT_CROSS_DEGREE } = data;
  const smooth = (distribution: BivariateDistribution, label: string): SmoothedTable =>
    bivariateSmoothedTable(distribution, degrees, crossDegree, label);
  if (data.design === "single-group") {
    const tables: [SmoothedTable] = [smooth(data.joint, "joint")];
    return {
      tables,
      from: tableMargin([tables[0].width], tables[0], 0, "rows"),
      to: tableMargin([tables[0].width], tables[0], 0, "columns"),
      anchors: null,
    };
  }
  if (data.design === "counterbalanced") {
    const tables: [SmoothedTable, SmoothedTable] = [smooth(data.first, "first"), smooth(data.second, "second")];
    if (tables[0].probabilities.length !== tables[1].probabilities.length || tables[0].columns !== tables[1].columns) {
      throw new DataError("The first and second distributions of a counterbalanced design need the same scores.");
    }
    const widths = tables.map((table) => table.width);
    // kequate scales both variances by the persons of the first group.
    const persons = tables[0].persons;
    const margin = (side: "rows" | "columns"): Margin =>
      mixedMargin(tableMargin(widths, tables[0], 0, side), tableMargin(widths, tables[1], 1, side), weight, persons);
    return { tables, from: margin("rows"), to: margin("columns"), anchors: null };
  }
  neatTables(data.from, data.to);
  const tables: [SmoothedTable, SmoothedTable] = [smooth(data.from, "from"), smooth(data.to, "to")];
  const widths = tables.map((table) => table.width);
  if (data.design === "post-stratification") {
    const from = { table: tables[0], index: 0 };
    const to = { table: tables[1], index: 1 };
    return {
      tables,
      from: syntheticMargin(widths, from, to, weight),
      to: syntheticMargin(widths, to, from, 1 - weight),
      anchors: null,
    };
  }
  return {
    tables,
    from: tableMargin(widths, tables[0], 0, "rows"),
    to: tableMargin(widths, tables[1], 1, "rows"),
    anchors: [tableMargin(widths, tables[0], 0, "columns"), tableMargin(widths, tables[1], 1, "columns")],
  };
}

/** The equating of a design with one link, without its bandwidths and convergence. */
type Link = Omit<KernelEquating, "bandwidths" | "converged">;

/** Returns e(x) = G⁻¹(F(x)) at each score of `from`, with the error vectors (1 / g(e)) (∂F/∂r R - ∂G/∂s S). */
function singleLink(margins: DesignMargins, from: Continuized, to: Continuized): Link {
  const vectors = errorVectorBuilder(margins.tables);
  const equated: number[] = [];
  const errorVectors = Array.from(margins.from.probabilities, (_, x) => {
    const e = kernelQuantile(to, kernelCdf(from, x), x);
    equated.push(e);
    const vector = vectors.start();
    const g = kernelDensity(to, e);
    vectors.add(vector, 1 / g, from, x, margins.from);
    vectors.add(vector, -1 / g, to, e, margins.to);
    return Array.from(vector);
  });
  return {
    equated,
    standardErrors: errorVectors.map(length),
    percentRelativeError: percentRelativeError(equated, margins.from.probabilities, margins.to.probabilities),
    errorVectors,
  };
}

/**
 * Returns the chained equating e(x) = G⁻¹(H_Q(H_P⁻¹(F(x)))), `chain` holding F, H_P, H_Q and G, with its error vectors
 * (von Davier et al., 2004, section 7.4).
 */
function chainedLink(margins: DesignMargins, chain: readonly Continuized[]): Link {
  const [from, fromAnchor, toAnchor, to] = chain;
  const anchors = margins.anchors;
  if (
    from === undefined ||
    fromAnchor === undefined ||
    toAnchor === undefined ||
    to === undefined ||
    anchors === null
  ) {
    throw new DataError("A chained kernel equating needs four distributions.");
  }
  const vectors = errorVectorBuilder(margins.tables);
  const anchorEquated: number[] = [];
  const equated: number[] = [];
  const errorVectors = Array.from(margins.from.probabilities, (_, x) => {
    const anchor = kernelQuantile(fromAnchor, kernelCdf(from, x), fromAnchor.mean);
    const e = kernelQuantile(to, kernelCdf(toAnchor, anchor), x);
    anchorEquated.push(anchor);
    equated.push(e);
    const g = kernelDensity(to, e);
    const throughAnchor = kernelDensity(toAnchor, anchor) / (g * kernelDensity(fromAnchor, anchor));
    const vector = vectors.start();
    vectors.add(vector, throughAnchor, from, x, margins.from);
    vectors.add(vector, -throughAnchor, fromAnchor, anchor, anchors[0]);
    vectors.add(vector, 1 / g, toAnchor, anchor, anchors[1]);
    vectors.add(vector, -1 / g, to, e, margins.to);
    return Array.from(vector);
  });
  const anchorToForm = Array.from(anchors[1].probabilities, (_, a) =>
    kernelQuantile(to, kernelCdf(toAnchor, a), to.mean),
  );
  return {
    equated,
    standardErrors: errorVectors.map(length),
    percentRelativeError: percentRelativeError(anchorEquated, margins.from.probabilities, anchors[0].probabilities),
    anchorPercentRelativeError: percentRelativeError(anchorToForm, anchors[1].probabilities, margins.to.probabilities),
    errorVectors,
  };
}

/**
 * Returns a link whose equated scores and standard errors are finite.
 *
 * @throws `DataError` when one is not, as when a bandwidth so small that the density vanishes between the scores leaves
 * a standard error without a density to divide by.
 */
function checkFinite(link: Link): Link {
  const score = [...link.equated, ...link.standardErrors].findIndex((value) => !Number.isFinite(value));
  if (score >= 0) {
    throw new DataError(
      "Kernel equating gives a value that is not finite with these bandwidths: the continuized density vanishes at an equated score. Use larger bandwidths, or omit them to minimize the penalty.",
    );
  }
  return link;
}

/** Starts and adds to error vectors, with one entry for each column of C of each table, in the order of the tables. */
interface ErrorVectorBuilder {
  /** Returns a vector of zeros. */
  readonly start: () => Float64Array;
  /** Adds `coefficient` times the gradient of the continuized C(t) in the probabilities, through the margin's blocks. */
  readonly add: (vector: Float64Array, coefficient: number, c: Continuized, t: number, margin: Margin) => void;
}

function errorVectorBuilder(tables: readonly SmoothedTable[]): ErrorVectorBuilder {
  const offsets: number[] = [];
  let size = 0;
  for (const table of tables) {
    offsets.push(size);
    size += table.width;
  }
  const gradients = new Map<number, Float64Array>();
  return {
    start: () => new Float64Array(size),
    add: (vector, coefficient, c, t, margin) => {
      const scores = margin.probabilities.length;
      const gradient = gradients.get(scores) ?? new Float64Array(scores);
      gradients.set(scores, gradient);
      cdfGradient(c, t, kernelDensity(c, t), gradient);
      tables.forEach((table, f) => {
        const block = margin.blocks[f];
        const offset = offsets[f] ?? 0;
        if (block === undefined) return;
        for (let j = 0; j < scores; j++) {
          const term = coefficient * (gradient[j] ?? 0);
          for (let k = 0; k < table.width; k++) {
            vector[offset + k] = (vector[offset + k] ?? 0) + term * (block[j * table.width + k] ?? 0);
          }
        }
      });
    },
  };
}

/** Returns the Euclidean length of a vector. */
function length(vector: readonly number[]): number {
  return Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
}
