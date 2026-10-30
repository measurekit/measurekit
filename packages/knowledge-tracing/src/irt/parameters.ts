import { checkRecord } from "@measurekit/core";
import { checkItem, type IrtItem } from "./items.js";

/**
 * The parameters of an IRT model: item parameters from a calibration, as those of `@measurekit/psychometrics` or mirt's
 * with `IRTpars = TRUE`.
 */
export interface IrtParameters {
  /** Each item's parameters, by item id. */
  readonly items: Readonly<Record<string, IrtItem>>;
}

/** Checks parameters, throwing a `DataError` that names the first problem found. */
export function checkParameters(parameters: IrtParameters): void {
  checkRecord(parameters, "IRT parameters must be an object");
  checkRecord(parameters.items, "IRT parameters need items, item parameters by item id");
  for (const [id, item] of Object.entries(parameters.items)) checkItem(item, id);
}
