import type { Random } from "@measurekit/core";

/** Returns a uniformly random permutation of 0 to `size - 1`, drawn by Fisher-Yates from `random`. */
export function permutation(size: number, random: Random): Uint32Array {
  const order = Uint32Array.from({ length: size }, (_, index) => index);
  for (let i = size - 1; i > 0; i--) {
    const j = Math.floor(random.uniform() * (i + 1));
    const value = order[i] ?? 0;
    order[i] = order[j] ?? 0;
    order[j] = value;
  }
  return order;
}
