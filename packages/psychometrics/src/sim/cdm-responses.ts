import { checkArray, checkRandom, checkRecord, DataError, describe, type Random } from "@measurekit/core";
import {
  bitPattern,
  type CdmItem,
  checkItemAttributes,
  checkSuccessProbabilities,
  profileIndices,
} from "../cdm/internal.js";
import { indexIds, type ResponseMatrix, responseMatrix } from "../data/internal.js";

/** A cognitive diagnosis model to simulate from: its attributes and items, such as a calibration's. */
export interface CdmModelDescription {
  /** The attributes, in the order of the profiles. */
  readonly attributes: readonly string[];
  /** The items, with their attributes and success probabilities per latent group. */
  readonly items: readonly Pick<CdmItem, "id" | "attributes" | "probabilities">[];
}

/**
 * Simulates responses from a cognitive diagnosis model: one person per profile, each response correct with the
 * success probability of the latent group the profile falls in for the item, drawn by one uniform draw, persons in
 * order, each person's items in order.
 *
 * @param model - The items with their attributes and the probability of success of each latent group.
 * @param profiles - The profile of each person: a 0 or 1 for each attribute.
 * @param random - The generator of the responses.
 * @returns The responses, one row for each profile.
 * @throws `DataError` when the model, its attributes, its items or the profiles are not objects and arrays of their
 * shape, the attributes repeat, a profile is not 0s and 1s over them, an item has no attributes, repeats one or names
 * one that is not among them, an item does not have one probability from 0 to 1 per latent group, item ids repeat,
 * there are no items or no profiles, or `random` is not a random number generator.
 */
export function simulateCdmResponses(
  model: CdmModelDescription,
  profiles: readonly (readonly number[])[],
  random: Random,
): ResponseMatrix {
  checkRecord(model, "A cognitive diagnosis model must be an object { attributes, items }");
  const { attributes, items } = model;
  checkArray(attributes, "A cognitive diagnosis model needs an array of attributes");
  checkArray(items, "A cognitive diagnosis model needs an array of items");
  checkArray(profiles, "The profiles must be an array");
  checkRandom(random);
  const attributeIndices = indexIds("attribute", attributes);
  const groups = items.map((item, j) => {
    checkRecord(item, `Item ${j + 1} must be an object { id, attributes, probabilities }`);
    if (typeof item.id !== "string") {
      throw new DataError(`Item ${j + 1} needs a string id, found ${describe(item.id)}.`);
    }
    checkItemAttributes(item.id, item.attributes);
    const indices = item.attributes.map((attribute) => {
      const index = attributeIndices.get(attribute);
      if (index === undefined) {
        throw new DataError(`Item "${item.id}" has attribute "${attribute}", which is not an attribute.`);
      }
      return index;
    });
    checkSuccessProbabilities(item.id, item.probabilities, 2 ** indices.length);
    return { indices, groupOf: profileIndices(indices.length) };
  });
  const responses = profiles.map((profile, person) => {
    const valid = Array.isArray(profile) && profile.length === attributes.length;
    if (!(valid && profile.every((value) => value === 0 || value === 1))) {
      throw new DataError(`Profile ${person + 1} needs a 0 or 1 for each of the ${attributes.length} attributes.`);
    }
    return items.map((item, j) => {
      const group = groups[j];
      const latentGroup = group === undefined ? 0 : (group.groupOf[bitPattern(profile, group.indices)] ?? 0);
      return random.uniform() < (item.probabilities[latentGroup] ?? 0) ? 1 : 0;
    });
  });
  return responseMatrix(responses, { itemIds: items.map((item) => item.id) });
}
