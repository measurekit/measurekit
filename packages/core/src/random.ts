import { checkSeed } from "./checks.js";
import { normalQuantile } from "./normal.js";

/** A source of random numbers. Functions that draw take one, so that a seeded source reproduces their results. */
export interface Random {
  /** Returns a uniform draw from [0, 1). */
  uniform(): number;
  /** Returns a draw from the standard normal distribution. */
  normal(): number;
}

const TWO_TO_26 = 2 ** 26;
const TWO_TO_53 = 2 ** 53;
/** 2³² divided by the golden ratio, which spaces the four seed words of the state. */
const GOLDEN_GAMMA = 0x9e3779b9;

/**
 * Creates a seeded random number generator: xoshiro128** seeded through the MurmurHash3 finalizer.
 *
 * @remarks
 * Equal seeds give equal sequences. Uniform draws have 53 random bits from integer arithmetic, identical in every
 * JavaScript runtime. Normal draws invert the normal distribution function at a uniform draw; they use `Math.log`,
 * whose last bit may differ between runtimes.
 *
 * @param seed - A safe integer.
 * @returns The generator, whose draws are the same for the same seed.
 * @throws `DataError` when `seed` is not a safe integer.
 *
 * @example
 * ```ts
 * import { createRandom } from "@measurekit/core";
 *
 * const random = createRandom(42);
 * random.uniform(); // 0.66
 * random.uniform(); // 0.11
 * createRandom(42).uniform(); // 0.66, the same seed gives the same draws
 * createRandom(42).normal(); // 0.41
 * ```
 */
export function createRandom(seed: number): Random {
  checkSeed(seed);
  const high = mix(Math.floor(seed / 2 ** 32) >>> 0);
  const state = new Uint32Array(4);
  for (let i = 0; i < 4; i++) state[i] = mix((seed >>> 0) + Math.imul(i + 1, GOLDEN_GAMMA) + high);
  const bits53 = (): number => (next(state) >>> 5) * TWO_TO_26 + (next(state) >>> 6);
  return {
    uniform: () => bits53() / TWO_TO_53,
    normal: () => normalQuantile((bits53() + 0.5) / TWO_TO_53),
  };
}

/**
 * Permutes the entries of `values` in place by Fisher-Yates, uniformly at random: one uniform draw for each entry but the
 * first.
 *
 * @param values - The entries to permute.
 * @param random - The source of the draws.
 */
export function shuffle(values: Int32Array | Uint32Array | Float64Array, random: Random): void {
  for (let i = values.length - 1; i > 0; i--) {
    const j = Math.floor(random.uniform() * (i + 1));
    const held = values[i] ?? 0;
    values[i] = values[j] ?? 0;
    values[j] = held;
  }
}

/** The MurmurHash3 32-bit finalizer, a bijection that spreads every input bit over the output. */
function mix(value: number): number {
  let h = value >>> 0;
  h = Math.imul(h ^ (h >>> 16), 0x85ebca6b);
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35);
  return (h ^ (h >>> 16)) >>> 0;
}

/** Advances xoshiro128** (Blackman and Vigna) and returns its next 32-bit output. */
function next(s: Uint32Array): number {
  const s0 = s[0] ?? 0;
  const s1 = s[1] ?? 0;
  const s2 = s[2] ?? 0;
  const s3 = s[3] ?? 0;
  const result = Math.imul(rotate(Math.imul(s1, 5), 7), 9) >>> 0;
  const t = s1 << 9;
  const n2 = s2 ^ s0;
  const n3 = s3 ^ s1;
  s[0] = s0 ^ n3;
  s[1] = s1 ^ n2;
  s[2] = n2 ^ t;
  s[3] = rotate(n3, 11);
  return result;
}

function rotate(x: number, k: number): number {
  return (x << k) | (x >>> (32 - k));
}
