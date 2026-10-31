import { checkMethod, checkOptions, type Random } from "@measurekit/core";
import type { ProfileProportion } from "../cdm/calibration.js";
import { type ProfileEstimate, profileEstimate } from "../cdm/persons.js";
import { bindPool, type DiagnosticPoolItem, type DiagnosticTestItem } from "./diagnostic-items.js";
import { gdinaDiscriminationIndex } from "./diagnostic-selection.js";
import { type AdaptiveState, adaptiveEngine } from "./engine.js";
import { checkCategory } from "./items.js";
import { profilePosterior } from "./profile-posterior.js";
import { profileSpace } from "./profile-space.js";
import type { ItemSelectionRule } from "./selection.js";
import type { StoppingRule } from "./stopping.js";
import type { WithinTestlet } from "./testlets.js";

/** Options of {@link diagnosticAdaptiveTest}. */
export interface DiagnosticAdaptiveTestOptions {
  /** The item pool: calibrated items `{ id, attributes, probabilities }`, with `contentArea` and `exposure` optional. */
  readonly pool: readonly DiagnosticPoolItem[];
  /** The attributes, from 1 to 20, in the order of the profiles. Every attribute of an item is among them. */
  readonly attributes: readonly string[];
  /**
   * The prior probability of each profile, as a calibration's `profiles`: every profile once, matched by its pattern,
   * with a proportion from 0 to 1, not all 0. A proportion of 0 excludes the profile, as a hierarchy among the
   * attributes does. The proportions are normalized to sum to 1.
   *
   * @defaultValue uniform over the profiles
   */
  readonly prior?: readonly ProfileProportion[];
  /** The rule that ends the test. The test also ends when no item is left. */
  readonly stopping: StoppingRule<DiagnosticTestItem, ProfileEstimate>;
  /**
   * The rule that selects the next item.
   *
   * @defaultValue `gdinaDiscriminationIndex`
   */
  readonly selection?: ItemSelectionRule<DiagnosticTestItem, ProfileEstimate>;
  /**
   * Target proportions of the content areas, by area, for Kingsbury-Zara content balancing; they need not sum to 1.
   * Every pool item then needs one of these areas.
   *
   * @defaultValue none: no content balancing
   */
  readonly contentTargets?: Readonly<Record<string, number>>;
  /**
   * Enemy sets: ids of items that must not be administered in the same test. Once an item of a set is administered,
   * the others are not eligible.
   *
   * @defaultValue none
   */
  readonly enemies?: readonly (readonly string[])[];
  /**
   * How the items of a testlet are ordered once the testlet is selected: `"in-order"`, in pool order; `"adaptive"`,
   * each next item the one the selection rule scores highest among the testlet's remaining items. It applies to pool
   * items that have a `testlet`.
   *
   * @defaultValue `"in-order"`
   */
  readonly withinTestlet?: WithinTestlet;
}

/** The state of a diagnostic adaptive test: plain data, which JSON serializes, so that a test can resume from it. */
export interface DiagnosticAdaptiveTestState extends AdaptiveState<ProfileEstimate> {}

/** A diagnostic adaptive test: a pool with the rules that select its items and end the test. */
export interface DiagnosticAdaptiveTest {
  /** The item pool. */
  readonly pool: readonly DiagnosticPoolItem[];
  /** The attributes, in the order of the profiles. */
  readonly attributes: readonly string[];
  /**
   * Returns the state before the first response: the prior as the estimate, and the first item.
   *
   * @throws `DataError` when `random` is not a random number generator.
   */
  start(random: Random): DiagnosticAdaptiveTestState;
  /**
   * Returns the state after a response to the state's next item: the new estimate, and the next item unless the test
   * ends.
   *
   * @throws `DataError` when the test has ended, the state is malformed, names items the pool does not have or an item
   * twice, a response is not 0 or 1, or `random` is not a random number generator.
   */
  answer(state: DiagnosticAdaptiveTestState, response: number, random: Random): DiagnosticAdaptiveTestState;
}

/**
 * Returns an adaptive test of a pool of cognitive diagnosis items, which classifies a person's profile of attributes.
 *
 * @remarks
 * The estimate after each response is the posterior over the 2^K profiles of K attributes, the prior times the
 * probability of each response in the item's latent group of the profile, with the attribute mastery, MAP and EAP
 * profiles of {@link ProfileEstimate}. Any cognitive diagnosis model gives the items: the rules use only each item's
 * success probabilities by latent group. Before each item the test checks the stopping rule; unless the test ends,
 * content balancing restricts the eligible items to one content area, the selection rule selects among them, and
 * Sympson-Hetter exposure control administers or withholds the selected item. Items administered, withheld or
 * enemies of an administered item are not eligible. The test holds, for each item, an index of 2^K entries.
 *
 * @throws `DataError` when `stopping` or `selection` is not a rule, the attributes, the prior or the pool are
 * invalid, an item has an attribute that is not among the attributes, a content target is invalid or missing for an
 * item's area, an exposure parameter is outside [0, 1], or an enemy set does not name two or more distinct pool items.
 *
 * @example
 * ```ts
 * import { createRandom } from "@measurekit/psychometrics";
 * import { anyOf, diagnosticAdaptiveTest, posteriorThreshold, testLength } from "@measurekit/psychometrics/cat";
 *
 * const pool = [
 *   { id: "q1", attributes: ["add"], probabilities: [0.2, 0.9] },
 *   { id: "q2", attributes: ["subtract"], probabilities: [0.15, 0.85] },
 *   { id: "q3", attributes: ["add", "subtract"], probabilities: [0.1, 0.4, 0.4, 0.9] },
 * ];
 * const test = diagnosticAdaptiveTest({
 *   pool,
 *   attributes: ["add", "subtract"],
 *   stopping: anyOf(testLength(3), posteriorThreshold(0.8)),
 * });
 * const random = createRandom(1);
 * let state = test.start(random);
 * state.next; // the id of the first item
 * state = test.answer(state, 1, random); // the response to it
 * state.estimate.mastery; // the probability of mastering each attribute
 * ```
 */
export function diagnosticAdaptiveTest(options: DiagnosticAdaptiveTestOptions): DiagnosticAdaptiveTest {
  checkOptions(options);
  const {
    pool,
    attributes,
    prior,
    stopping,
    selection = gdinaDiscriminationIndex,
    contentTargets,
    enemies = [],
    withinTestlet,
  } = options;
  checkMethod(stopping, "stops", "The option stopping must be a stopping rule, such as testLength(20)");
  checkMethod(selection, "scores", "The option selection must be a selection rule, such as gdinaDiscriminationIndex");
  const space = profileSpace(attributes, prior);
  const { items, indices } = bindPool(pool, space);
  const engine = adaptiveEngine({
    pool: items,
    indices,
    stopping,
    selection,
    contentTargets,
    enemies,
    withinTestlet,
    estimate: (administered, responses) =>
      profileEstimate(profilePosterior(space.prior, administered, responses), space.profiles),
    checkResponse: (item, response) => {
      checkCategory(item.id, 2, response);
    },
  });
  return { pool, attributes, ...engine };
}
