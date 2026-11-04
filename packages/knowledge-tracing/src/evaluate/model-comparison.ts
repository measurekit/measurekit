import { checkOptions, DataError, isRecord } from "@measurekit/core";
import type { Fold } from "../data/internal.js";
import type { LearnerModel } from "../model.js";
import {
  CROSS_VALIDATE_OPTIONS,
  type CrossValidateOptions,
  type CrossValidation,
  crossValidate,
} from "./cross-validation.js";

/** One model's row of a comparison: its name and its evaluation. */
export interface ModelEvaluation extends CrossValidation {
  /** The model's name, its key in the models compared. */
  readonly name: string;
}

/**
 * Evaluates several models on the same folds, as {@link crossValidate} does each, so that their metrics compare the
 * same predictions of the same events. The rows follow the order of `models`; the metrics of each fold allow paired
 * comparisons across folds. With `unseenSkills: "skip"`, every model has a `null` prediction for the same events, those
 * of a skill absent from the training log, so that the metrics still compare the same events.
 *
 * @param models - The models by name.
 * @param folds - The folds, the same for every model.
 * @param options - What to do with a test skill the training log lacks.
 * @returns The evaluation of each model with its name, in the order of the names.
 * @throws `DataError` when there are no models, or in the cases listed for {@link crossValidate}.
 */
export function compareModels(
  models: Readonly<Record<string, LearnerModel>>,
  folds: readonly Fold[],
  options: CrossValidateOptions = {},
): readonly ModelEvaluation[] {
  checkOptions(options, CROSS_VALIDATE_OPTIONS);
  const entries = isRecord(models) ? Object.entries(models) : [];
  if (entries.length === 0) {
    throw new DataError("A comparison needs at least one model, as an object of models by name.");
  }
  return entries.map(([name, model]) => ({ name, ...crossValidate(model, folds, options) }));
}
