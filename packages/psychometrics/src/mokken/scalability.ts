import type { ResponseMatrix } from "../data/internal.js";
import { type Covariances, covariances, mokkenData } from "./scores.js";
import { coefficientErrors } from "./standard-errors.js";

/** A scalability coefficient with its standard error. */
export interface ScalabilityCoefficient {
  /** The coefficient; `null` where it is undefined, because an item involved has no variance. */
  readonly coefficient: number | null;
  /** The standard error of the coefficient; `null` with it. */
  readonly standardError: number | null;
}

/** The scalability coefficient of an item, H_i, with its standard error. */
export interface ItemScalabilityCoefficient extends ScalabilityCoefficient {
  /** The item id. */
  readonly id: string;
}

/** The scalability coefficients of a set of items. */
export interface Scalability {
  /** The scale's coefficient H. */
  readonly scale: ScalabilityCoefficient;
  /** Each item's coefficient H_i, in the order of the matrix's items. */
  readonly items: readonly ItemScalabilityCoefficient[];
  /** The coefficient H_ij of each pair of items by item, then item; `null` on the diagonal. */
  readonly pairs: readonly (readonly (ScalabilityCoefficient | null)[])[];
}

/**
 * Returns the scalability coefficients of items: H of the scale, H_i of each item and H_ij of each pair, with standard
 * errors, over the persons who answered every item.
 *
 * @remarks
 * H_ij = Cov(X_i, X_j) / Cov_max(X_i, X_j), where the largest covariance given the items' score distributions is that
 * of the two sorted columns of scores. H_i and H are the ratios of the sums of the covariances to the sums of the
 * largest covariances, over the item's pairs and over all pairs. Scores are integers from the lowest of all items, taken
 * as 0, up to 9, so items may be dichotomous or polytomous. The standard errors come from the delta method over the
 * numbers of persons with each response pattern, as the mokken package's `coefH`. An item without variance has no
 * coefficient, nor do its pairs, and the other coefficients leave it out.
 *
 * @param matrix - The responses of the persons.
 * @returns H of the scale, H_i of each item and H_ij of each pair, with their standard errors.
 * @throws `DataError` when the matrix is invalid, has fewer than two items, fewer than two persons answered every item,
 * or the scores span more than 10 categories.
 * @see {@link https://doi.org/10.1177/0081175013481958 | Kuijpers, van der Ark and Croon (2013)}
 * @see {@link https://doi.org/10.4135/9781412984676 | Sijtsma and Molenaar (2002)}
 */
export function scalability(matrix: ResponseMatrix): Scalability {
  const data = mokkenData(matrix, "Scalability");
  const moments = covariances(data);
  const { items } = data;
  const usable = usableItems(moments, items);
  const errors = coefficientErrors(data, usable);
  const coefficient = (observed: number, maximum: number): number | null => (maximum > 0 ? observed / maximum : null);
  const pairs = Array.from({ length: items }, (_, i) =>
    Array.from({ length: items }, (_, j) =>
      i === j
        ? null
        : {
            coefficient: coefficient(moments.observed[i * items + j] ?? 0, moments.maximum[i * items + j] ?? 0),
            standardError: errors.pairs[i]?.[j] ?? null,
          },
    ),
  );
  const itemCoefficients = itemScalability(moments, items, usable).map((coefficient, i) => ({
    id: data.itemIds[i] ?? "",
    coefficient,
    standardError: usable[i] ? (errors.items[i] ?? null) : null,
  }));
  const everyone = Array.from({ length: items }, (_, j) => j).filter((j) => usable[j] === true);
  return {
    scale: { coefficient: scaleCoefficient(moments, items, everyone), standardError: errors.scale },
    items: itemCoefficients,
    pairs,
  };
}

/** Returns whether each item has variance. */
export function usableItems(moments: Covariances, items: number): boolean[] {
  return Array.from({ length: items }, (_, item) => (moments.observed[item * items + item] ?? 0) > 0);
}

/** Returns each item's coefficient H_i over the other items with variance, `null` for an item without variance. */
export function itemScalability(moments: Covariances, items: number, usable: readonly boolean[]): (number | null)[] {
  const members = Array.from({ length: items }, (_, item) => item).filter((item) => usable[item] === true);
  return Array.from({ length: items }, (_, item) =>
    usable[item] ? subsetCoefficient(moments, items, [item], members) : null,
  );
}

/** Returns the ratio of the sums of the covariances and of the largest covariances of `rows` with `columns`. */
function subsetCoefficient(
  moments: Covariances,
  items: number,
  rows: readonly number[],
  columns: readonly number[],
): number | null {
  let observed = 0;
  let maximum = 0;
  for (const i of rows) {
    for (const j of columns) {
      if (i === j) continue;
      observed += moments.observed[i * items + j] ?? 0;
      maximum += moments.maximum[i * items + j] ?? 0;
    }
  }
  return maximum > 0 ? observed / maximum : null;
}

/** Returns H of the items in `members`, the ratio of the sums over their pairs. */
export function scaleCoefficient(moments: Covariances, items: number, members: readonly number[]): number | null {
  return subsetCoefficient(moments, items, members, members);
}
