import { checkRecord } from "@measurekit/core";
import { checkRating } from "./state.js";

/** The parameters of an Elo model: plain data, which JSON serializes. */
export interface EloParameters {
  /** The items rated so far, by item id. An item without a rating has difficulty 0 and count 0. */
  readonly items: Readonly<Record<string, EloItem>>;
}

/** An item's rating. */
export interface EloItem {
  /** d, the item's difficulty on the logit scale. */
  readonly difficulty: number;
  /** n, the number of events that have updated `difficulty`. */
  readonly count: number;
}

/** Checks parameters, which may come from storage, throwing a `DataError` that names the first problem found. */
export function checkParameters(parameters: EloParameters): void {
  checkRecord(parameters, "Elo parameters must be an object");
  checkRecord(parameters.items, "Elo parameters need items, an object of item ratings by item id");
  for (const [item, rating] of Object.entries(parameters.items)) {
    checkRecord(rating, `Item "${item}" of the Elo parameters must be an object`);
    checkRating(rating.difficulty, rating.count, `Item "${item}" of the Elo parameters`, "a difficulty");
  }
}
