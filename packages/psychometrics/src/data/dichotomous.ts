import { DataError } from "@measurekit/core";
import type { ResponseMatrix } from "./response-matrix.js";

/**
 * Checks that every response of a checked matrix is 0, 1 or `null`.
 *
 * @param statistic - Names the statistic in the error message.
 * @throws `DataError` naming the first other response.
 */
export function checkDichotomous(matrix: ResponseMatrix, statistic: string): void {
  matrix.responses.forEach((row, person) => {
    const item = row.findIndex((response) => response !== null && response > 1);
    if (item >= 0) {
      throw new DataError(
        `${statistic} needs dichotomous items, but person "${matrix.personIds[person]}" scored ${row[item]} on item "${matrix.itemIds[item]}".`,
      );
    }
  });
}
