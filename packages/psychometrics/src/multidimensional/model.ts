import type { ModelParameter } from "../irt/model.js";

/**
 * A multidimensional item response model: the probability of each response category of an item as a function of a
 * person's vector of thetas, given the item's parameters.
 *
 * @remarks
 * Categories are 0 to `categoryCount - 1`. Calibration, scoring and information use only these members, so a model
 * implemented outside the package works in them as a built-in one does.
 *
 * Calibration works on a parameter vector: the item's parameters in the order of {@link parameters}, the slopes of the
 * dimensions first. `parameterDerivatives` differentiates with respect to that vector.
 *
 * @typeParam TParameters - The parameters of one item, such as `{ a, intercept, c, d }`.
 */
export interface MultidimensionalModel<TParameters = unknown> {
  /** A short name for messages, such as `"M2PL"`. */
  readonly name: string;
  /** The most response categories an item can have: 2 for a dichotomous model, `Infinity` without a limit. */
  readonly maxCategories: number;
  /** Returns the number of response categories of an item with these parameters. */
  categoryCount(parameters: TParameters): number;
  /** Returns the number of dimensions the item's slopes span. */
  dimensionCount(parameters: TParameters): number;
  /**
   * Returns the parameters of an item with `categoryCount` categories on `dimensions` dimensions, in vector order: the
   * slopes `a1` to `aD` first. A slope that the pattern of loadings fixes at 0 stays in the vector.
   */
  parameters(categoryCount: number, dimensions: number): readonly ModelParameter[];
  /**
   * Returns starting values for calibration, given the share of persons in each category of an item and its starting
   * slopes. Bounded parameters start strictly inside their bounds.
   */
  startingValues(proportions: Float64Array, slopes: Float64Array): TParameters;
  /**
   * Returns the parameter vector of valid parameters.
   *
   * @throws `DataError` when the parameters are not valid for the model.
   */
  toVector(parameters: TParameters): Float64Array;
  /** Returns the parameters of a parameter vector on `dimensions` dimensions. */
  fromVector(vector: Float64Array, dimensions: number): TParameters;
  /** Writes the probability of each category at `theta` into `out`. */
  probabilities(parameters: TParameters, theta: Float64Array, out: Float64Array): void;
  /**
   * Writes the derivative of each category's probability with respect to each dimension of theta: entry `d` of
   * category `k` goes to `out[k * dimensions + d]`.
   */
  thetaGradients(parameters: TParameters, theta: Float64Array, out: Float64Array): void;
  /**
   * Writes the derivative of each category's probability with respect to each entry of the parameter vector, by
   * category, then entry: entry `j` of category `k` goes to `out[k * vectorLength + j]`.
   */
  parameterDerivatives(parameters: TParameters, theta: Float64Array, out: Float64Array): void;
  /**
   * Writes the Fisher information matrix of the item at `theta` into `out`, by rows: the sum over categories of
   * ∇P ∇Pᵀ / P.
   */
  information(parameters: TParameters, theta: Float64Array, out: Float64Array): void;
}

/**
 * An item with known parameters, as scoring and information take it.
 *
 * @typeParam TParameters - The parameters of the item's model.
 */
export interface MultidimensionalItem<TParameters = unknown> {
  /** The item's model. */
  readonly model: MultidimensionalModel<TParameters>;
  /** The item's parameters. */
  readonly parameters: TParameters;
}

/**
 * An item with known parameters and an id, such as a calibration's item.
 *
 * @typeParam TParameters - The parameters of the item's model.
 */
export interface MultidimensionalTestItem<TParameters = unknown> extends MultidimensionalItem<TParameters> {
  /** The item's id. */
  readonly id: string;
}

/**
 * The dimensions each item loads on, in a confirmatory calibration: the names of an item's dimensions by item id, such
 * as `{ q1: ["verbal"], q2: ["verbal", "quantitative"] }`.
 */
export type LoadingPattern = Readonly<Record<string, readonly string[]>>;
