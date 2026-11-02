import { checkChoice, checkOptions } from "@measurekit/core";
import { checkFunction } from "../checks.js";
import {
  type AbilityEstimator,
  type IrtItem,
  largestCategoryCount,
  MIN_PROBABILITY,
  posteriorModeEstimate,
} from "../irt/internal.js";
import type { ItemSelectionRule } from "./selection.js";

/** The information of an item at an estimate: observed, of the response given, or Fisher's. */
export type ExpectedInformationKind = "observed" | "fisher";

const KINDS: readonly ExpectedInformationKind[] = ["observed", "fisher"];

/** Options of {@link expectedInformation}. */
export interface ExpectedInformationOptions {
  /**
   * Estimates theta from the responses so far and a possible response to the item scored. It must give an estimate for
   * every pattern of responses, as a posterior estimator does.
   *
   * @defaultValue `posteriorModeEstimate`
   */
  readonly estimator?: AbilityEstimator;
  /**
   * The item's information at each such estimate: `"observed"`, minus the second derivative of the log-probability of
   * the response given, or `"fisher"`, the item's Fisher information.
   *
   * @defaultValue `"observed"`, as catR's `infoType`
   */
  readonly information?: ExpectedInformationKind;
}

/**
 * Returns maximum expected information (MEI), catR's `MEI`: each item scored by its information at the estimate after
 * its response, expected over its categories at the current estimate.
 *
 * @remarks
 * The score is Σ_k P_k(θ̂) I_k(θ̂_k), where θ̂_k is the estimator's estimate from the responses so far and the response k
 * to the item, and I_k is the observed information (P'_k / P_k)² - P''_k / P_k of the response k or the Fisher
 * information there. It calls the estimator once for each category of each eligible item, so it costs as many
 * estimates.
 *
 * @param options - The estimator and the kind of information.
 * @returns The selection rule.
 * @throws `DataError` when the options are not an object, `estimator` is not a function or `information` is unknown.
 * @see {@link https://doi.org/10.1007/BF02294775 | van der Linden (1998)}
 * @see {@link https://doi.org/10.1007/978-0-387-85461-8_1 | van der Linden and Pashley (2010)}
 */
export function expectedInformation(options: ExpectedInformationOptions = {}): ItemSelectionRule {
  checkOptions(options);
  const { estimator = posteriorModeEstimate, information = "observed" } = options;
  checkFunction(
    estimator,
    "The option estimator must be a function of items and responses, such as posteriorModeEstimate",
  );
  checkChoice(information, "The option information", KINDS);
  return {
    scores({ pool, administered, responses, estimate, eligible }, out) {
      const size = largestCategoryCount(pool);
      const atEstimate = new Float64Array(size);
      const probabilities = new Float64Array(size);
      const first = new Float64Array(size);
      const second = new Float64Array(size);
      const given: IrtItem[] = administered.flatMap((index) => pool[index] ?? []);
      pool.forEach((item, index) => {
        if (eligible[index] !== 1) return;
        const { model, parameters } = item;
        const count = model.categoryCount(parameters);
        model.probabilities(parameters, estimate.theta, atEstimate);
        const items = [...given, item];
        let score = 0;
        for (let k = 0; k < count; k++) {
          const { theta } = estimator(items, [...responses, k]);
          if (information === "fisher") {
            score += (atEstimate[k] ?? 0) * model.information(parameters, theta);
            continue;
          }
          model.probabilities(parameters, theta, probabilities);
          model.thetaDerivatives(parameters, theta, first, second);
          const p = Math.max(probabilities[k] ?? 0, MIN_PROBABILITY);
          score += (atEstimate[k] ?? 0) * (((first[k] ?? 0) / p) ** 2 - (second[k] ?? 0) / p);
        }
        out[index] = score;
      });
    },
  };
}
