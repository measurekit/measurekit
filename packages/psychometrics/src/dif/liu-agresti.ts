import { normalCdf } from "@measurekit/core";
import type { ScoreGroups } from "./groups.js";

/** Liu and Agresti's cumulative common odds ratio of an item's categories across score groups, with its test. */
export interface LiuAgresti {
  /**
   * ψ̂, the odds of a response above each category in the reference group over those in the focal group, common to the
   * categories and the score groups: above 1 when the item favors the reference group. For a 0/1 item it is the
   * Mantel-Haenszel odds ratio.
   */
  readonly oddsRatio: number;
  /** The standard error of ln ψ̂, Liu and Agresti's dually consistent estimate; `null` when ψ̂ is 0. */
  readonly logStandardError: number | null;
  /** ln ψ̂ over its standard error. */
  readonly z: number | null;
  /** The two-sided probability of a z at least as large in size without DIF. */
  readonly significance: number | null;
}

/**
 * Returns Liu and Agresti's cumulative common odds ratio of the score groups' tables, as difR's `liu_agresti_ccor` with
 * the score groups as strata, or `null` where it is undefined, as when no score group holds both groups.
 *
 * @remarks
 * In score group k, with n_F and n_R the persons of each group, N their sum, and F_j and R_j the persons of each group
 * at or below category j: ψ̂ = Σ_k Σ_j F_j (n_R - R_j) / N / Σ_k Σ_j (n_F - F_j) R_j / N, over the categories j but the
 * highest that the item's persons used. The variance of ln ψ̂ is equation 3 of Liu and Agresti, which reduces to a sum
 * over the score groups of products of these counts and ψ̂. Each score group's cumulative counts are formed again for
 * the variance once ψ̂ is known, so that the memory does not grow with the score groups times the categories.
 *
 * @see {@link https://doi.org/10.2307/2532838 | Liu and Agresti (1996)}
 * @see {@link https://doi.org/10.1111/j.1745-3984.2003.tb01151.x | Penfield and Algina (2003)}
 */
export function liuAgresti(groups: ScoreGroups): LiuAgresti | null {
  let concordant = 0;
  let discordant = 0;
  forEachCumulativeTable(groups, ({ focal, reference, focalAtOrBelow, referenceAtOrBelow }) => {
    const size = focal + reference;
    focalAtOrBelow.forEach((focalBelow, j) => {
      const referenceBelow = referenceAtOrBelow[j] ?? 0;
      concordant += (focalBelow * (reference - referenceBelow)) / size;
      discordant += ((focal - focalBelow) * referenceBelow) / size;
    });
  });
  if (discordant === 0) return null;
  const oddsRatio = concordant / discordant;
  const untested = { oddsRatio, logStandardError: null, z: null, significance: null };
  if (oddsRatio === 0) return untested;
  let spread = 0;
  forEachCumulativeTable(groups, (table) => {
    spread += stratumSpread(table, oddsRatio);
  });
  const logStandardError = Math.sqrt(spread) / (oddsRatio * discordant);
  const z = Math.log(oddsRatio) / logStandardError;
  if (!(Number.isFinite(logStandardError) && Number.isFinite(z))) return untested;
  return { oddsRatio, logStandardError, z, significance: 2 * normalCdf(-Math.abs(z)) };
}

/** The cumulative counts of one score group with persons of both groups. */
interface CumulativeTable {
  /** The persons of the focal group. */
  readonly focal: number;
  /** The persons of the reference group. */
  readonly reference: number;
  /** The persons of the focal group at or below each category but the highest. */
  readonly focalAtOrBelow: Float64Array;
  /** The persons of the reference group at or below each category but the highest. */
  readonly referenceAtOrBelow: Float64Array;
}

/**
 * Calls `visit` with the cumulative counts of each score group that holds persons of both groups, in order; the others
 * add nothing. The arrays of the table are reused between calls.
 */
function forEachCumulativeTable(
  { categories, scoreCount, sizes, cellStarts, cellCategories, cellCounts }: ScoreGroups,
  visit: (table: CumulativeTable) => void,
): void {
  const focalAtOrBelow = new Float64Array(Math.max(categories - 1, 0));
  const referenceAtOrBelow = new Float64Array(Math.max(categories - 1, 0));
  const cumulate = (cell: number, atOrBelow: Float64Array): void => {
    let entry = cellStarts[cell] ?? 0;
    const end = cellStarts[cell + 1] ?? 0;
    let persons = 0;
    for (let j = 0; j < categories - 1; j++) {
      if (entry < end && cellCategories[entry] === j) persons += cellCounts[entry++] ?? 0;
      atOrBelow[j] = persons;
    }
  };
  for (let s = 0; s < scoreCount; s++) {
    const [reference = 0, focal = 0] = sizes.subarray(2 * s, 2 * s + 2);
    if (reference === 0 || focal === 0) continue;
    cumulate(2 * s, referenceAtOrBelow);
    cumulate(2 * s + 1, focalAtOrBelow);
    visit({ focal, reference, focalAtOrBelow, referenceAtOrBelow });
  }
}

/**
 * Returns ξ_k(ψ) of Liu and Agresti for one score group: Σ_j φ_jj + 2 Σ_(j < s) φ_js, with, for j ≤ s,
 * φ_js = n_F n_R / N² [ψ (n_F - F_s) R_j / n_F (1 + (ψ - 1) R_s / n_R) + F_j (n_R - R_s) / n_R (ψ - (ψ - 1) F_s / n_F)].
 * φ_js is the sum of R_j and F_j times factors of s alone, so the sum over j below s takes the running sums of R_j and F_j.
 */
function stratumSpread(
  { focal, reference, focalAtOrBelow, referenceAtOrBelow }: CumulativeTable,
  oddsRatio: number,
): number {
  const size = focal + reference;
  const scale = (focal * reference) / (size * size);
  let sum = 0;
  let referenceBelow = 0;
  let focalBelow = 0;
  for (let s = 0; s < focalAtOrBelow.length; s++) {
    const focalS = focalAtOrBelow[s] ?? 0;
    const referenceS = referenceAtOrBelow[s] ?? 0;
    const first = ((oddsRatio * (focal - focalS)) / focal) * (1 + ((oddsRatio - 1) * referenceS) / reference);
    const second = ((reference - referenceS) / reference) * (oddsRatio - ((oddsRatio - 1) * focalS) / focal);
    sum += scale * (first * (2 * referenceBelow + referenceS) + second * (2 * focalBelow + focalS));
    referenceBelow += referenceS;
    focalBelow += focalS;
  }
  return sum;
}
