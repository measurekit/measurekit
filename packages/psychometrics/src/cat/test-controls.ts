import type { WithinTestlet } from "./testlets.js";

/** The options of an adaptive test that restrict which items it may administer, the same for every kind of test. */
export interface TestControlOptions {
  /**
   * Target proportions of the content areas, by area, for Kingsbury-Zara content balancing; they need not sum to 1.
   * Every pool item then needs exactly one of these areas among its `contentAreas`.
   *
   * @defaultValue none: no content balancing
   */
  readonly contentTargets?: Readonly<Record<string, number>>;
  /**
   * Enemy sets: ids of items that must not be administered in the same test, such as items that give away each other's
   * answers. Once an item of a set is administered, the others are not eligible.
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
