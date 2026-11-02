/**
 * The highest response, and the highest score of a faceted score, that the analyses take: the largest 32-bit signed
 * integer, 2³¹ - 1, as they hold categories in 32-bit arrays.
 */
export const MAX_RESPONSE = 2_147_483_647;

/** Returns whether a value is a response: an integer from 0 to {@link MAX_RESPONSE}. */
export function isResponse(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value) && value >= 0 && value <= MAX_RESPONSE;
}
