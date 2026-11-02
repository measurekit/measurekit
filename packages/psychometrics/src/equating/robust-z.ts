import { DataError, mean } from "@measurekit/core";
import type { IrtItem } from "../irt/internal.js";
import { sampleQuantile } from "../math/internal.js";
import { checkCommonItems } from "./linking.js";
import { itemMoments } from "./models.js";

/** How far one common item's location moved between two calibrations, and the robust z that screens it. */
export interface AnchorDrift {
  /** The mean location of the item on the `to` scale less its mean location on the `from` scale. */
  readonly difference: number;
  /** The robust z of the difference among the common items. */
  readonly z: number;
  /** Whether the absolute robust z exceeds 1.645, the 95th percentile of the standard normal distribution. */
  readonly flagged: boolean;
}

/** Huynh and Meyer's critical value of the absolute robust z: the 95th percentile of the standard normal distribution. */
const ROBUST_Z_CRITERION = 1.645;
/** Huynh and Meyer's divisor of the interquartile range, which makes it estimate the standard deviation of a normal distribution. */
const IQR_TO_SD = 0.74;

/**
 * Returns the robust z that screens common items for drift between two calibrations, one entry per item.
 *
 * @remarks
 * An item's location is the difficulty b, or for a polytomous item the mean of its thresholds. With d the difference of
 * an item's location on the `to` scale and on the `from` scale, the robust z is (d - median d) / (0.74 IQR d) over the
 * common items, the median and the interquartile range being R's `median` and `IQR` (quantiles by linear
 * interpolation). An item is flagged when its absolute robust z exceeds 1.645. The differences of the locations are
 * those of a shift, so the screen suits Rasch and partial credit items, whose scales differ by a constant; for other
 * models, put the `from` items on the `to` scale first by a linking method. `from[j]` and `to[j]` are the same item,
 * with the same model and number of categories.
 *
 * @param from - The common items on the scale to move.
 * @param to - The same items on the scale to move to.
 * @returns For each item, the difference of its difficulties between the calibrations, the robust z of the difference
 *   and whether the item is flagged.
 * @throws `DataError` when the item lists differ in length or are empty, a pair has different models or numbers of
 * categories, an item is invalid, an NRM item or not one of the built-in models, or the interquartile range of the
 * differences is 0, as with one common item.
 * @see {@link https://pareonline.net/getvn.asp?v=15&n=2 | Huynh and Meyer (2010)}
 */
export function robustZ(from: readonly IrtItem[], to: readonly IrtItem[]): AnchorDrift[] {
  checkCommonItems(from, to);
  const locations = (items: readonly IrtItem[], scale: string): number[] =>
    items.map((item, j) => mean(itemMoments(item, `Common item ${j + 1} on the ${scale} scale`).locations));
  const before = locations(from, "from");
  const differences = Float64Array.from(locations(to, "to"), (location, j) => location - (before[j] ?? 0));
  const sorted = Float64Array.from(differences).sort();
  const median = sampleQuantile(sorted, 0.5);
  const interquartileRange = sampleQuantile(sorted, 0.75) - sampleQuantile(sorted, 0.25);
  if (!(interquartileRange > 0)) {
    throw new DataError("Robust z needs common items whose differences have an interquartile range above 0.");
  }
  return Array.from(differences, (difference) => {
    const z = (difference - median) / (IQR_TO_SD * interquartileRange);
    return { difference, z, flagged: Math.abs(z) > ROBUST_Z_CRITERION };
  });
}
