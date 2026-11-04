/**
 * Answer similarity: the selection of the pairs of persons who share a context, and the indices of copying and
 * collusion between two persons' responses, Wollack's ω, the generalized binomial test, Maynes' M4, and the K index
 * with its variants K1, K2, S1 and S2.
 *
 * @module @measurekit/psychometrics/similarity
 */
export { type GeneralizedBinomialTest, generalizedBinomialTest } from "./generalized-binomial.js";
export { type KIndex, type KIndexOptions, kIndex } from "./k-index.js";
export { type M4Index, type M4IndexOptions, m4Index } from "./m4.js";
export { type OmegaIndex, omegaIndex } from "./omega.js";
export {
  type PairSelection,
  type PersonPair,
  type SelectPairsOptions,
  type SkippedGroup,
  selectPairs,
} from "./pairs.js";
