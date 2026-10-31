import type { MultidimensionalTestItem } from "../multidimensional/model.js";
import type { PoolControls } from "./items.js";

/**
 * An item of a multidimensional adaptive test's pool: a multidimensional item with an id, such as an item of a
 * `MultidimensionalCalibration`, with the controls of every pool item.
 *
 * @typeParam TParameters - The parameters of the item's model.
 */
export interface MultidimensionalPoolItem<TParameters = unknown>
  extends MultidimensionalTestItem<TParameters>,
    PoolControls {}
