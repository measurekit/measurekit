import { checkArray, checkRecord, DataError, describe, isRecord } from "@measurekit/core";
import { INTERCEPT } from "./features.js";
import type { ItemFeatureRegression } from "./item-feature-regression.js";

/** The prediction of an item parameter from the item's features. */
export interface ParameterPrediction {
  /** The predicted value, on the scale of the regression: the logarithm for a slope. */
  readonly mean: number;
  /** The standard deviation of the prediction: the residual between items and the uncertainty of the effects. */
  readonly sd: number;
}

/**
 * Returns the prediction of an item parameter for a new item from its features: the mean xᵀη̂ and the standard deviation
 * √(τ̂² + xᵀ Cov x).
 *
 * @remarks
 * The standard deviation is that of the parameter of an item drawn from the model: the residual τ̂² that the features
 * leave between items plus the variance of the estimated mean. On the regression's `scale`, a normal prior has this mean
 * and sd, `normalPrior(mean, sd)` for a difficulty; for a slope, which is regressed as its logarithm, `logNormalPrior(mean,
 * sd)`. Passed to `calibrate` as `itemPriors: { new1: { b: prior } }`, it makes a calibration of an item that few
 * persons answered start from the prediction.
 *
 * Valid input: a regression of {@link itemFeatureRegression} or one loaded from JSON, and finite values of every
 * feature the regression uses; other features are ignored.
 *
 * @example
 * ```ts
 * import { predictItemParameter } from "@measurekit/psychometrics/explanatory";
 *
 * // `regression` is the result of the example of itemFeatureRegression.
 * predictItemParameter(regression, { steps: 6 }); // { mean: 2.24, sd: 0.12 }
 * ```
 *
 * @param regression - The regression of the parameter on the features.
 * @param features - The new item's value of each feature the regression uses, by feature name.
 * @returns The mean and the standard deviation of the item's parameter, on the regression's scale.
 * @throws `DataError` when the regression is not a regression, or the features lack one it uses or give it a value that
 * is not finite.
 */
export function predictItemParameter(
  regression: ItemFeatureRegression,
  features: Readonly<Record<string, number>>,
): ParameterPrediction {
  checkRecord(regression, "The regression must be an object from itemFeatureRegression");
  checkArray(regression.effects, "The effects of the regression must be an array");
  checkArray(regression.covariance, "The covariance of the regression must be an array of rows");
  const size = regression.effects.length;
  if (size === 0 || regression.covariance.length !== size) {
    throw new DataError(
      `The regression needs a covariance with a row for each of its effects, found ${regression.covariance.length} rows for ${size} effects.`,
    );
  }
  checkRecord(features, "The features of the item must be an object of numbers by feature name");
  const x = regression.effects.map((effect, j) => {
    if (!(isRecord(effect) && typeof effect.name === "string" && Number.isFinite(effect.estimate))) {
      throw new DataError(
        `Effect ${j + 1} of the regression needs a name and a finite estimate, found ${describe(effect)}.`,
      );
    }
    const value: unknown = effect.name === INTERCEPT ? 1 : features[effect.name];
    if (!(typeof value === "number" && Number.isFinite(value))) {
      throw new DataError(
        `The features of the item need a finite number for "${effect.name}", found ${describe(value)}.`,
      );
    }
    return value;
  });
  const { residualSd } = regression;
  if (!(typeof residualSd === "number" && Number.isFinite(residualSd) && residualSd >= 0)) {
    throw new DataError(
      `The residual sd of the regression must be a non-negative number, found ${describe(residualSd)}.`,
    );
  }
  let mean = 0;
  let variance = residualSd ** 2;
  regression.effects.forEach((effect, j) => {
    mean += (x[j] ?? 0) * effect.estimate;
    const row: unknown = regression.covariance[j];
    if (!(Array.isArray(row) && row.length === size)) {
      throw new DataError(
        `Row ${j + 1} of the covariance of the regression needs ${size} entries, found ${describe(row)}.`,
      );
    }
    row.forEach((entry: unknown, k) => {
      if (!(typeof entry === "number" && Number.isFinite(entry))) {
        throw new DataError(`The covariance of the regression needs finite entries, found ${describe(entry)}.`);
      }
      variance += (x[j] ?? 0) * entry * (x[k] ?? 0);
    });
  });
  return { mean, sd: Math.sqrt(Math.max(0, variance)) };
}
