import { checkOptions, DataError, type FieldNames, type ThetaDistribution } from "@measurekit/core";
import {
  checkThetaDistribution,
  expectedScore,
  indexItems,
  largestCategoryCount,
  type TestItem,
} from "../irt/internal.js";

/** Options of {@link differentialFunctioning}. */
export interface DifferentialFunctioningOptions {
  /**
   * The focal group's distribution of theta on the reference scale: nodes and weights, such as a calibration group's,
   * or the focal persons' estimates with equal weights.
   */
  readonly population: ThetaDistribution;
}

const DIFFERENTIAL_FUNCTIONING_OPTIONS: FieldNames<DifferentialFunctioningOptions> = { population: true };

/** The differential functioning of one item. */
export interface ItemFunctioning {
  /** The item id. */
  readonly id: string;
  /** NCDIF: E[(T_F - T_R)²], the mean squared difference of the item's expected scores over the population. */
  readonly ncdif: number;
  /** CDIF: E[d D], with d the item's difference of expected scores and D that of the test. */
  readonly cdif: number;
}

/** The differential functioning of the items and of the test they form. */
export interface TestFunctioning {
  /** Each item's functioning, in item order. */
  readonly items: readonly ItemFunctioning[];
  /** DTF, E[D²], the sum of the items' CDIF; over a bundle's items it is the bundle's DBF. */
  readonly dtf: number;
}

/**
 * Returns the noncompensatory and compensatory differential functioning of every item and the differential test
 * functioning, as DFIT's `Ncdif`, `Cdif` and `Dtf`, for items of any model on one scale.
 *
 * @remarks
 * At each node θ of the population, d_i(θ) = T_F,i(θ) - T_R,i(θ) is the difference of item i's expected scores under its
 * focal and its reference parameters, and D(θ) = Σ d_i(θ) that of the test. NCDIF_i = E[d_i²], CDIF_i = E[d_i D] and
 * DTF = E[D²] = Σ CDIF_i, each the weighted sum over the nodes. An item's expected score is Σ k P_k, so polytomous items
 * enter as Flowers, Oshima and Raju's. The two lists hold the same items in the same order, with the parameters each
 * group's calibration gives on the reference scale, such as the free items of a multiple-group calibration or the
 * focal estimates moved by a linking transformation. Items equal in both groups have NCDIF 0 and add 0 to the DTF.
 *
 * @param reference - The items with the reference group's parameters.
 * @param focal - The same items with the focal group's parameters, in the same order.
 * @param options - The focal group's distribution of theta.
 * @returns The functioning of each item and of the test.
 * @throws `DataError` when an item list is not an array of valid items with unique string ids, the lists differ in
 * their ids or their order, or the population is not a distribution with finite nodes and weights summing to 1.
 * @see {@link https://doi.org/10.1177/014662169501900405 | Raju, van der Linden and Fleer (1995)}
 * @see {@link https://doi.org/10.1177/01466219922031437 | Flowers, Oshima and Raju (1999)}
 */
export function differentialFunctioning(
  reference: readonly TestItem[],
  focal: readonly TestItem[],
  options: DifferentialFunctioningOptions,
): TestFunctioning {
  indexItems(reference);
  indexItems(focal);
  checkOptions(options, DIFFERENTIAL_FUNCTIONING_OPTIONS);
  if (focal.length !== reference.length) {
    throw new DataError(`Found ${focal.length} focal items for ${reference.length} reference items.`);
  }
  focal.forEach((item, i) => {
    if (item.id !== reference[i]?.id) {
      throw new DataError(
        `The focal items must be the reference items in the same order; item ${i + 1} is "${reference[i]?.id}" in the reference list and "${item.id}" in the focal list.`,
      );
    }
  });
  const { nodes, weights } = checkThetaDistribution(options.population, "population");
  const count = reference.length;
  const scratch = new Float64Array(Math.max(largestCategoryCount(reference), largestCategoryCount(focal)));
  const pairs = focal.map((item, i) => ({ focal: item, reference: reference[i] ?? item }));
  const differences = new Float64Array(count);
  const ncdif = new Float64Array(count);
  const cdif = new Float64Array(count);
  nodes.forEach((theta, q) => {
    const weight = weights[q] ?? 0;
    let test = 0;
    pairs.forEach((pair, i) => {
      const difference = expectedScore(pair.focal, theta, scratch) - expectedScore(pair.reference, theta, scratch);
      differences[i] = difference;
      test += difference;
    });
    for (let i = 0; i < count; i++) {
      const difference = differences[i] ?? 0;
      ncdif[i] = (ncdif[i] ?? 0) + weight * difference * difference;
      cdif[i] = (cdif[i] ?? 0) + weight * difference * test;
    }
  });
  return {
    items: reference.map(({ id }, i) => ({ id, ncdif: ncdif[i] ?? 0, cdif: cdif[i] ?? 0 })),
    dtf: cdif.reduce((sum, value) => sum + value, 0),
  };
}
