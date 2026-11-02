import { checkOptions, checkRecord, DataError } from "@measurekit/core";
import { type ScoredResponses, scoreResponses } from "../data/internal.js";
import { correlation } from "../math/internal.js";
import { completeCases } from "./complete-cases.js";
import { type GroupOptions, groupMean, scoreGroups } from "./groups.js";

/** How often one option of an item was chosen, and by whom. */
export interface OptionStatistics {
  /** The option label. */
  readonly label: string;
  /** Persons who chose the option. */
  readonly count: number;
  /** Share of the persons who answered the item that chose the option. */
  readonly proportion: number | null;
  /** Share of the upper group that chose the option. */
  readonly upper: number | null;
  /** Share of the lower group that chose the option. */
  readonly lower: number | null;
  /** Correlation of choosing the option with the total score. Negative for a working distractor. */
  readonly pointBiserial: number | null;
}

/** The options of one item. */
export interface OptionAnalysis {
  /** The item id. */
  readonly id: string;
  /** The keyed option label. */
  readonly key: string;
  /** Statistics of every option label, in label order. */
  readonly options: readonly OptionStatistics[];
}

/**
 * Returns statistics of every option of every item, computed from the choices and the key. `count` and `proportion`
 * use every person who answered the item; the groups and correlations use the persons who answered every item, by
 * total score. A statistic that is undefined for the data is `null`.
 *
 * @param scored - The choices of the persons and the key, as `scoreResponses` returns them.
 * @param options - The share of persons in the upper and lower groups.
 * @returns The statistics of every option of every item.
 * @throws `DataError` when the choices, key, labels or ids are invalid, or `groupFraction` is out of range.
 */
export function distractorAnalysis(scored: ScoredResponses, options: GroupOptions = {}): OptionAnalysis[] {
  // Scored responses built by hand or loaded from JSON may be another value or lack a field.
  checkRecord(scored, "Scored responses must be an object from scoreResponses");
  checkOptions(options);
  const { personIds, itemIds, choices, key, optionLabels } = scored;
  if (!(Array.isArray(choices) && Array.isArray(key) && Array.isArray(optionLabels))) {
    throw new DataError("Scored responses need the arrays choices, key and optionLabels.");
  }
  const matrix = scoreResponses(choices, key, { personIds, itemIds, optionLabels });
  const cases = completeCases(matrix);
  const groups = scoreGroups(cases.totals, options);
  return matrix.itemIds.map((id, item) => {
    const column = choices.map((row) => row[item] ?? null);
    const answered = column.filter((choice) => choice !== null).length;
    const completeColumn = cases.persons.map((person) => column[person]);
    return {
      id,
      key: key[item] ?? "",
      options: optionLabels.map((label) => {
        const count = column.filter((choice) => choice === label).length;
        const chosen = Float64Array.from(completeColumn, (choice) => (choice === label ? 1 : 0));
        return {
          label,
          count,
          proportion: answered > 0 ? count / answered : null,
          upper: groupMean(groups.upper, chosen),
          lower: groupMean(groups.lower, chosen),
          pointBiserial: correlation(chosen, cases.totals),
        };
      }),
    };
  });
}
