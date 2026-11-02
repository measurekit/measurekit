/**
 * The elementary symmetric functions of a set of items in the Rasch family: γ_r, the sum over the response patterns of
 * total r of the product of the items' category weights e^(-η_ik), η_ik the sum of item i's thresholds up to k. They are
 * kept as `values[r] e^logScale`, the values at most 1, so that they stay finite for long tests.
 */
export interface SymmetricFunctions {
  /** γ_r over e^logScale, from r = 0 to the sum of the items' highest categories. */
  readonly values: Float64Array;
  /** The log of the common factor of the values. */
  readonly logScale: number;
}

/** The functions of no items: γ_0 = 1. */
export const NO_ITEMS: SymmetricFunctions = { values: Float64Array.of(1), logScale: 0 };

/**
 * Returns the functions of one item, a polynomial in the score with the coefficients e^(-η_k), η_0 = 0.
 *
 * @param categoryParameters - η_1 to η_m of the item: the sums of its thresholds up to each category.
 */
export function itemFunctions(categoryParameters: ArrayLike<number>): SymmetricFunctions {
  let largest = 0;
  for (let k = 0; k < categoryParameters.length; k++) largest = Math.max(largest, -(categoryParameters[k] ?? 0));
  const values = new Float64Array(categoryParameters.length + 1);
  values[0] = Math.exp(-largest);
  for (let k = 0; k < categoryParameters.length; k++) values[k + 1] = Math.exp(-(categoryParameters[k] ?? 0) - largest);
  return { values, logScale: largest };
}

/**
 * Returns the functions of the union of two disjoint sets of items, the product of their polynomials: the summation
 * algorithm, one item at a time when one set is an item.
 */
export function combine(a: SymmetricFunctions, b: SymmetricFunctions): SymmetricFunctions {
  const values = new Float64Array(a.values.length + b.values.length - 1);
  for (let i = 0; i < a.values.length; i++) {
    const left = a.values[i] ?? 0;
    if (left === 0) continue;
    for (let j = 0; j < b.values.length; j++) values[i + j] = (values[i + j] ?? 0) + left * (b.values[j] ?? 0);
  }
  let largest = 0;
  for (const value of values) largest = Math.max(largest, value);
  if (largest > 0) for (let r = 0; r < values.length; r++) values[r] = (values[r] ?? 0) / largest;
  return { values, logScale: a.logScale + b.logScale + (largest > 0 ? Math.log(largest) : 0) };
}

/** Returns ln γ_r, `-Infinity` where it is 0. */
export function logFunction({ values, logScale }: SymmetricFunctions, r: number): number {
  return Math.log(values[r] ?? 0) + logScale;
}

/**
 * The functions of a set of items with one or two of them left out, from the products of the items before and after
 * them, so that no function is divided by another.
 */
interface LeftOut {
  /** The functions of every item. */
  readonly all: SymmetricFunctions;
  /** Returns the functions of every item but the one at `i`. */
  without(i: number): SymmetricFunctions;
  /** Calls `visit` with each pair i < j and the functions of every item but the two. */
  forEachPair(visit: (i: number, j: number, functions: SymmetricFunctions) => void): void;
}

/** Returns the functions of a set of items, given each item's functions, with one or two left out. */
export function leaveOut(items: readonly SymmetricFunctions[]): LeftOut {
  const count = items.length;
  // before[i] holds the items below i, after[i] the items from i on.
  const before: SymmetricFunctions[] = [NO_ITEMS];
  for (let i = 0; i < count; i++) before.push(combine(before[i] ?? NO_ITEMS, items[i] ?? NO_ITEMS));
  const after: SymmetricFunctions[] = new Array<SymmetricFunctions>(count + 1).fill(NO_ITEMS);
  for (let i = count - 1; i >= 0; i--) after[i] = combine(items[i] ?? NO_ITEMS, after[i + 1] ?? NO_ITEMS);
  return {
    all: before[count] ?? NO_ITEMS,
    without: (i) => combine(before[i] ?? NO_ITEMS, after[i + 1] ?? NO_ITEMS),
    forEachPair: (visit) => {
      for (let i = 0; i < count; i++) {
        // The items below i and those between i and j, grown as j moves up.
        let left = before[i] ?? NO_ITEMS;
        for (let j = i + 1; j < count; j++) {
          visit(i, j, combine(left, after[j + 1] ?? NO_ITEMS));
          left = combine(left, items[j] ?? NO_ITEMS);
        }
      }
    },
  };
}
