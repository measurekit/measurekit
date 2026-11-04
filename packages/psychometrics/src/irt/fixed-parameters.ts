import { checkFinite, checkRecord, checkRecordById, DataError } from "@measurekit/core";
import type { FixedParameters } from "./calibration-options.js";
import type { ModelParameter } from "./model.js";

/**
 * Checks the option `fixedParameters` against the items and returns its values by item id, then parameter name.
 *
 * @param itemIds - The ids of the items.
 * @param fixed - The ids of the items fixed whole, which hold no single parameter.
 * @throws `DataError` when the option is not an object of objects of finite numbers, an id is not an item id, or the
 * item is fixed whole.
 */
export function readFixedParameters(
  fixedParameters: FixedParameters,
  itemIds: readonly string[],
  fixed: ReadonlySet<string>,
): Map<string, ReadonlyMap<string, number>> {
  checkRecordById(
    fixedParameters,
    "The option fixedParameters must be an object of values by item id and parameter name",
  );
  const ids = new Set(itemIds);
  return new Map(
    Object.entries(fixedParameters).map(([id, byName]) => {
      if (!ids.has(id)) {
        throw new DataError(`The option fixedParameters holds parameters of "${id}", which is not an item id.`);
      }
      if (fixed.has(id)) {
        throw new DataError(
          `The option fixedParameters holds parameters of item "${id}", which the option fixed holds whole.`,
        );
      }
      checkRecord(byName, `The held parameters of item "${id}" must be an object of values by parameter name`);
      for (const [name, value] of Object.entries(byName))
        checkFinite(value, `The held value of "${name}" of item "${id}"`);
      return [id, new Map(Object.entries(byName))];
    }),
  );
}

/**
 * Checks that an item's model has the parameters `fixedParameters` holds, does not share them among items, and keeps
 * one free.
 *
 * @param specifications - The estimated parameters of the item's model, in vector order.
 * @throws `DataError` when the model has no parameter of a held name, shares one, or every parameter is held.
 */
export function checkHeldParameters(
  id: string,
  modelName: string,
  specifications: readonly ModelParameter[],
  held: ReadonlyMap<string, number>,
): void {
  const names = specifications.map(({ name }) => name);
  for (const name of held.keys()) {
    const specification = specifications.find((parameter) => parameter.name === name);
    if (specification === undefined) {
      throw new DataError(
        `The option fixedParameters holds "${name}" of item "${id}", whose ${modelName} model has ${names.join(", ")}.`,
      );
    }
    if (specification.shared === true) {
      throw new DataError(
        `The option fixedParameters holds "${name}" of item "${id}", whose ${modelName} model shares it among items.`,
      );
    }
  }
  if (names.every((name) => held.has(name))) {
    throw new DataError(
      `The option fixedParameters holds every parameter of item "${id}"; give its parameters in the option fixed.`,
    );
  }
}

/**
 * Writes the held values into an item's starting vector, and moves the starting values of its other ordered parameters
 * to fit them, so that a valid held value is not refused for the starts around it: an entry between two held ones goes
 * evenly between them, and one beyond them keeps its distance from the nearest. Other starts stay as they are.
 *
 * @param specifications - The estimated parameters of the item's model, in vector order.
 * @param held - The values `fixedParameters` holds, each of a parameter in `specifications`.
 * @param start - The item's starting vector, which receives the values.
 */
export function startAtHeldParameters(
  specifications: readonly ModelParameter[],
  held: ReadonlyMap<string, number>,
  start: Float64Array,
): void {
  const series = Array.from(specifications.keys()).filter((p) => specifications[p]?.ordered === true);
  const heldAt = (p: number): number | undefined => held.get(specifications[p]?.name ?? "");
  const pinned = series.filter((p) => heldAt(p) !== undefined);
  const before = start.slice();
  for (const p of series) {
    const lower = pinned.findLast((q) => q <= p);
    const upper = pinned.find((q) => q >= p);
    if (lower !== undefined && upper !== undefined && lower < upper) {
      const share = (series.indexOf(p) - series.indexOf(lower)) / (series.indexOf(upper) - series.indexOf(lower));
      start[p] = (heldAt(lower) ?? 0) + share * ((heldAt(upper) ?? 0) - (heldAt(lower) ?? 0));
    } else if (lower !== undefined || upper !== undefined) {
      const nearest = lower ?? upper ?? p;
      start[p] = (heldAt(nearest) ?? 0) + (before[p] ?? 0) - (before[nearest] ?? 0);
    }
  }
  specifications.forEach(({ name }, p) => {
    const value = held.get(name);
    if (value !== undefined) start[p] = value;
  });
}
