import { DataError } from "@measurekit/core";

/**
 * The most elements, items or skills, that the analyses of the relations between elements take: IITA, the difference
 * test, the validity of a structure and the space of a surmise relation. Each holds tables with a cell for every pair of
 * elements, so that its memory grows with the square of their number and its time faster: at 1,000 elements a table of
 * counts takes 4 MB.
 */
const MAX_ELEMENTS = 1_000;

/**
 * Checks the number of elements of an analysis of the relations between them.
 *
 * @param subject - Starts the message: what takes the elements, such as `IITA`.
 * @param noun - What the elements are, `items` or `elements`.
 * @throws `DataError` when there are more than 1,000.
 */
export function checkElementCount(count: number, subject: string, noun: "items" | "elements"): void {
  if (count <= MAX_ELEMENTS) return;
  throw new DataError(
    `${subject} takes at most ${MAX_ELEMENTS} ${noun}, as its tables grow with the square of their number; found ${count}. Split the domain into units, such as chapters, and analyze each.`,
  );
}
