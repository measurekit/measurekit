import { DataError, isRecord } from "@measurekit/core";
import type { Fold } from "../data/split.js";
import type { LearnerModel } from "../model.js";
import { type CrossValidation, crossValidate } from "./cross-validation.js";

/** One model's row of a comparison: its name and its evaluation. */
export interface ModelEvaluation extends CrossValidation {
  /** The model's name, its key in the models compared. */
  readonly name: string;
}

/**
 * Evaluates several models on the same folds, as {@link crossValidate} does each, so that their metrics compare the
 * same predictions of the same events. The rows follow the order of `models`; the metrics of each fold allow paired
 * comparisons across folds.
 *
 * @throws `DataError` when there are no models, or in the cases listed for {@link crossValidate}.
 */
export function compareModels(
  models: Readonly<Record<string, LearnerModel>>,
  folds: readonly Fold[],
): ModelEvaluation[] {
  const entries = isRecord(models) ? Object.entries(models) : [];
  if (entries.length === 0) {
    throw new DataError("A comparison needs at least one model, as an object of models by name.");
  }
  return entries.map(([name, model]) => ({ name, ...crossValidate(model, folds) }));
}
