import { type Random, shuffle } from "@measurekit/core";

/** Returns a uniformly random permutation of 0 to `size - 1`, drawn by Fisher-Yates from `random`. */
export function permutation(size: number, random: Random): Uint32Array {
  const order = Uint32Array.from({ length: size }, (_, index) => index);
  shuffle(order, random);
  return order;
}
