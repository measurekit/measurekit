import { checkRecord, cholesky, DataError, describe, isRecord } from "@measurekit/core";
import { designCrossProducts } from "./cross-products.js";

/**
 * Features of items by item id, such as `{ q1: { fractions: 1, steps: 3 } }`: the number of fractions and of steps in
 * a task, say. A categorical feature is one 0/1 column for each of its categories but one.
 */
export type ItemFeatures = Readonly<Record<string, Readonly<Record<string, number>>>>;

/** The effect of one feature, or of the intercept, on an item parameter. */
export interface FeatureEffect {
  /** The feature's name; `"(intercept)"` for the intercept. */
  readonly name: string;
  /** The estimated effect: the change of the parameter for one unit of the feature. */
  readonly estimate: number;
  /** Its standard error; `null` where the model gives none. */
  readonly standardError: number | null;
}

/** The name of the intercept among the effects. */
export const INTERCEPT = "(intercept)";

/** The design of a regression on features: the effects' names and the items' values of them. */
export interface Design {
  /** The names of the effects, the intercept first when there is one. */
  readonly names: readonly string[];
  /** The values by item, then effect. */
  readonly matrix: Float64Array;
}

/**
 * Builds the design matrix of items from their features: an optional intercept, then the features, in the order of
 * their first appearance among the items.
 *
 * @param ids - The ids of the items, in the order of the rows.
 * @param features - The features of each item by item id.
 * @param intercept - Whether the first effect is an intercept, a column of 1.
 * @returns The names of the effects and the design matrix by item, then effect.
 * @throws `DataError` when the features are not an object of objects, an item has none or lacks a feature that another
 * has, a value is not finite, a feature is named like the intercept, or the design has no effect.
 */
export function designMatrix(ids: readonly string[], features: ItemFeatures, intercept: boolean): Design {
  checkRecord(features, "The features must be an object of features by item id");
  const records = ids.map((id) => {
    const record: unknown = Object.hasOwn(features, id) ? features[id] : undefined;
    if (!isRecord(record))
      throw new DataError(`Item "${id}" needs features, an object of numbers, found ${describe(record)}.`);
    return record;
  });
  const featureNames = [...new Set(records.flatMap((record) => Object.keys(record)))];
  if (featureNames.includes(INTERCEPT)) {
    throw new DataError(`A feature cannot be named "${INTERCEPT}", the name of the intercept.`);
  }
  const names = intercept ? [INTERCEPT, ...featureNames] : featureNames;
  if (names.length === 0) throw new DataError("The design needs an intercept or at least one feature.");
  const matrix = new Float64Array(ids.length * names.length);
  records.forEach((record, i) => {
    names.forEach((name, j) => {
      const value: unknown = name === INTERCEPT ? 1 : record[name];
      if (!(typeof value === "number" && Number.isFinite(value))) {
        throw new DataError(
          Object.hasOwn(record, name)
            ? `The feature "${name}" of item "${ids[i]}" must be a finite number, found ${describe(value)}.`
            : `Item "${ids[i]}" lacks the feature "${name}".`,
        );
      }
      matrix[i * names.length + j] = value;
    });
  });
  return { names, matrix };
}

/**
 * Checks that the columns of the design are linearly independent: that XᵀX of each leading set of columns is positive
 * definite.
 *
 * @remarks
 * A leading set factors when every smaller one does: the Cholesky factor of the smaller is the leading part of the
 * larger's, and the least pivot that passes grows with the size. So the first set that does not factor is found by
 * bisection, in time that grows as the cube of the effects, where trying every set took their fourth power.
 *
 * @param design - The design of the effects.
 * @throws `DataError` naming the first effect that is a linear combination of those before it.
 */
export function checkFullRank({ names, matrix }: Design): void {
  const columns = names.length;
  const rows = matrix.length / columns;
  const gram = designCrossProducts(matrix, rows, columns);
  const factors = (size: number): boolean => {
    const leading = new Float64Array(size * size);
    for (let j = 0; j < size; j++) {
      for (let k = 0; k < size; k++) leading[j * size + k] = gram[j * columns + k] ?? 0;
    }
    return cholesky(leading, size) !== null;
  };
  if (factors(columns)) return;
  let factored = 0;
  let failed = columns;
  while (failed - factored > 1) {
    const size = Math.floor((factored + failed) / 2);
    if (factors(size)) factored = size;
    else failed = size;
  }
  throw new DataError(
    `The effect "${names[failed - 1]}" is all zeros or a linear combination of the effects before it, so the effects have no unique estimates.`,
  );
}
