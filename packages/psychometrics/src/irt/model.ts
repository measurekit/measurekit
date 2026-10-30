/**
 * An item response model: the probability of each response category of an item as a function of theta, given the item's
 * parameters.
 *
 * @remarks
 * Categories are 0 to `categoryCount - 1`. Calibration, ability estimation and the other functions that take items of
 * any model use only these members, so a model implemented outside the package works in them as a built-in one does.
 *
 * Calibration works on a parameter vector: the item's estimated parameters, in the order of {@link parameters}.
 * `parameterDerivatives` differentiates with respect to that vector.
 *
 * @typeParam TParameters - The parameters of one item, such as `{ a, b, c, d }`.
 */
export interface IrtModel<TParameters = unknown> {
  /** A short name for messages, such as `"2PL"`. */
  readonly name: string;
  /** The most response categories an item can have: 2 for a dichotomous model, `Infinity` without a limit. */
  readonly maxCategories: number;
  /** Returns the number of response categories of an item with these parameters. */
  categoryCount(parameters: TParameters): number;
  /** Returns the estimated parameters of an item with `categoryCount` categories, in vector order. */
  parameters(categoryCount: number): readonly ModelParameter[];
  /**
   * Returns starting values for calibration, given the share of persons in each category of an item. Bounded
   * parameters start strictly inside their bounds.
   */
  startingValues(proportions: Float64Array): TParameters;
  /**
   * Returns the parameter vector of valid parameters.
   *
   * @throws `DataError` when the parameters are not valid for the model.
   */
  toVector(parameters: TParameters): Float64Array;
  /** Returns the parameters of a parameter vector. */
  fromVector(vector: Float64Array): TParameters;
  /** Writes the probability of each category at `theta` into `out`. */
  probabilities(parameters: TParameters, theta: number, out: Float64Array): void;
  /** Writes the first and second derivatives of each category's probability with respect to theta. */
  thetaDerivatives(parameters: TParameters, theta: number, first: Float64Array, second: Float64Array): void;
  /**
   * Writes the derivative of each category's probability with respect to each entry of the parameter vector, by
   * category, then entry: entry `j` of category `k` goes to `out[k * vectorLength + j]`.
   */
  parameterDerivatives(parameters: TParameters, theta: number, out: Float64Array): void;
  /** Returns the Fisher information of the item at `theta`: the sum over categories of P'(θ)² / P(θ). */
  information(parameters: TParameters, theta: number): number;
}

/** One estimated parameter of a model: its name, bounds and whether the items of the model share it. */
export interface ModelParameter {
  /** The parameter's name, such as `"a"`. */
  readonly name: string;
  /**
   * The lower bound, exclusive in estimation.
   *
   * @defaultValue `-Infinity`
   */
  readonly lower?: number;
  /**
   * The upper bound, exclusive in estimation.
   *
   * @defaultValue `Infinity`
   */
  readonly upper?: number;
  /**
   * Whether one value is estimated for all calibrated items of the model, as the 1PL's `a`.
   *
   * @defaultValue `false`
   */
  readonly shared?: boolean;
}

/**
 * An item with known parameters, as ability estimation and information take it.
 *
 * @typeParam TParameters - The parameters of the item's model.
 */
export interface IrtItem<TParameters = unknown> {
  /** The item's model. */
  readonly model: IrtModel<TParameters>;
  /** The item's parameters. */
  readonly parameters: TParameters;
}

/**
 * An item with known parameters and an id, such as a calibration's item, as adaptive and multistage tests and ordered
 * item booklets take it.
 *
 * @typeParam TParameters - The parameters of the item's model.
 */
export interface TestItem<TParameters = unknown> extends IrtItem<TParameters> {
  /** The item's id. */
  readonly id: string;
}
