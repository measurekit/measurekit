import type { ProfileEstimate } from "../cdm/internal.js";
import { logProbability } from "../irt/internal.js";
import type { DiagnosticTestItem } from "./diagnostic-items.js";
import type { AdaptiveTestContext, ItemSelectionRule } from "./selection.js";

/**
 * The Kullback-Leibler divergence of the response distribution at a probability of success q from that at p, as a
 * difference of logarithms with q and 1 - q at least `MIN_PROBABILITY`, so that it is finite where a probability is 0.
 */
function divergence(p: number, q: number): number {
  const right = p > 0 ? p * (Math.log(p) - logProbability(q)) : 0;
  const wrong = p < 1 ? (1 - p) * (Math.log(1 - p) - logProbability(1 - q)) : 0;
  return right + wrong;
}

/** Returns the index of the profile of largest posterior probability, the first on ties. */
function modeOf(posterior: readonly number[]): number {
  return posterior.reduce((top, p, c) => (p > (posterior[top] ?? 0) ? c : top), 0);
}

/** Writes to `masses` the posterior probability of each latent group of an item: the profiles in it summed. */
function groupMasses(item: DiagnosticTestItem, posterior: readonly number[], masses: Float64Array): void {
  masses.fill(0);
  item.groups.forEach((group, c) => {
    masses[group] = (masses[group] ?? 0) + (posterior[c] ?? 0);
  });
}

function largestGroupCount(pool: readonly DiagnosticTestItem[]): number {
  return pool.reduce((most, item) => Math.max(most, item.probabilities.length), 0);
}

/**
 * Builds a rule that scores each eligible item from the posterior probabilities of its latent groups and the item's
 * latent group at the MAP profile.
 */
function groupRule(
  score: (probabilities: readonly number[], masses: Float64Array, mapGroup: number) => number,
): ItemSelectionRule<DiagnosticTestItem, ProfileEstimate> {
  return {
    scores({ pool, eligible, estimate }: AdaptiveTestContext<DiagnosticTestItem, ProfileEstimate>, out) {
      const masses = new Float64Array(largestGroupCount(pool));
      const mode = modeOf(estimate.posterior);
      pool.forEach((item, index) => {
        if (eligible[index] !== 1) return;
        const view = masses.subarray(0, item.probabilities.length);
        groupMasses(item, estimate.posterior, view);
        out[index] = score(item.probabilities, view, item.groups[mode] ?? 0);
      });
    },
  };
}

/**
 * The Kullback-Leibler index (KL) of cognitive diagnosis: scores each item by the divergence Σ_c Σ_x P(x | α̂) ln(P(x |
 * α̂) / P(x | α_c)) of the response distribution at the MAP profile α̂ from that at every profile α_c, unweighted.
 *
 * @remarks
 * Profiles in one latent group of the item have the same response distribution, so the sum is over the latent groups,
 * each counted as often as it has profiles. It sums over every profile, also one the prior excludes.
 *
 * @see {@link https://doi.org/10.1007/s11336-009-9123-2 | Cheng (2009)}
 */
export const diagnosticKullbackLeibler: ItemSelectionRule<DiagnosticTestItem, ProfileEstimate> = {
  scores({ pool, eligible, estimate }, out) {
    const mode = modeOf(estimate.posterior);
    pool.forEach((item, index) => {
      if (eligible[index] !== 1) return;
      const atMode = item.probabilities[item.groups[mode] ?? 0] ?? 0;
      let sum = 0;
      for (const probability of item.probabilities) sum += divergence(atMode, probability);
      out[index] = (sum * item.groups.length) / item.probabilities.length;
    });
  },
};

/**
 * The posterior-weighted Kullback-Leibler index (PWKL): scores each item by the divergence of the response
 * distribution at the MAP profile from that at each profile, weighted by the posterior probability of the profile.
 *
 * @see {@link https://doi.org/10.1007/s11336-009-9123-2 | Cheng (2009)}
 */
export const posteriorWeightedKullbackLeibler: ItemSelectionRule<DiagnosticTestItem, ProfileEstimate> = groupRule(
  (probabilities, masses, mapGroup) => {
    const atMode = probabilities[mapGroup] ?? 0;
    return probabilities.reduce((sum, probability, l) => sum + (masses[l] ?? 0) * divergence(atMode, probability), 0);
  },
);

/**
 * The modified posterior-weighted Kullback-Leibler index (MPWKL): scores each item by the divergence between the
 * response distributions at every pair of profiles, weighted by the product of their posterior probabilities, so that
 * it does not depend on a point estimate.
 *
 * @see {@link https://doi.org/10.1177/0146621614554650 | Kaplan, de la Torre and Barrada (2015)}
 */
export const modifiedPosteriorWeightedKullbackLeibler: ItemSelectionRule<DiagnosticTestItem, ProfileEstimate> =
  groupRule((probabilities, masses) => {
    let sum = 0;
    probabilities.forEach((p, d) => {
      let inner = 0;
      probabilities.forEach((q, l) => {
        inner += (masses[l] ?? 0) * divergence(p, q);
      });
      sum += (masses[d] ?? 0) * inner;
    });
    return sum;
  });

/**
 * The G-DINA discrimination index (GDI): scores each item by the variance Σ_l π_l (P_l - P̄)² of its success
 * probability over its latent groups, P̄ the mean over the posterior, π_l the posterior probability of group l.
 *
 * @see {@link https://doi.org/10.1177/0146621614554650 | Kaplan, de la Torre and Barrada (2015)}
 */
export const gdinaDiscriminationIndex: ItemSelectionRule<DiagnosticTestItem, ProfileEstimate> = groupRule(
  (probabilities, masses) => {
    const mean = probabilities.reduce((sum, p, l) => sum + (masses[l] ?? 0) * p, 0);
    return probabilities.reduce((sum, p, l) => sum + (masses[l] ?? 0) * (p - mean) ** 2, 0);
  },
);

/**
 * The Shannon entropy rule (SHE): scores each item by minus the expected Shannon entropy of the posterior after the
 * response, the expectation over the item's response distribution at the current posterior.
 *
 * @see {@link https://doi.org/10.1111/1467-9868.00377 | Tatsuoka and Ferguson (2003)}
 */
export const shannonEntropy: ItemSelectionRule<DiagnosticTestItem, ProfileEstimate> = {
  scores({ pool, eligible, estimate }, out) {
    const { posterior } = estimate;
    const masses = new Float64Array(largestGroupCount(pool));
    const entropies = new Float64Array(largestGroupCount(pool));
    const entropyTerms = Float64Array.from(posterior, (p) => (p > 0 ? p * Math.log(p) : 0));
    pool.forEach((item, index) => {
      if (eligible[index] !== 1) return;
      masses.fill(0);
      entropies.fill(0);
      item.groups.forEach((group, c) => {
        masses[group] = (masses[group] ?? 0) + (posterior[c] ?? 0);
        entropies[group] = (entropies[group] ?? 0) + (entropyTerms[c] ?? 0);
      });
      // The expected entropy Σ_x P(x) H(π | x) with π_c(x) = π_c w_lx / P(x), w_lx the probability of x in group l:
      // Σ_x [P(x) ln P(x) - Σ_l w_lx (Σ_{c in l} π_c ln π_c + π*_l ln w_lx)].
      let expected = 0;
      for (const correct of [true, false]) {
        const weights = item.probabilities.map((p) => (correct ? p : 1 - p));
        const response = weights.reduce((sum, w, l) => sum + w * (masses[l] ?? 0), 0);
        if (response > 0) expected += response * Math.log(response);
        weights.forEach((w, l) => {
          if (w > 0) expected -= w * ((entropies[l] ?? 0) + (masses[l] ?? 0) * Math.log(w));
        });
      }
      out[index] = -expected;
    });
  },
};
