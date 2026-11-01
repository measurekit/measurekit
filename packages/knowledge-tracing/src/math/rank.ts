import { dot } from "@measurekit/core";

/**
 * How small a column may become, relative to its own norm, once what earlier columns span is removed, and still count:
 * the default `tol` of R's `qr`.
 */
const RANK_TOLERANCE = 1e-7;

/**
 * Returns the numerical rank of a matrix given by its columns, which are overwritten.
 *
 * @remarks
 * Columns are taken in order, as R's `qr` does by limited pivoting: a column joins when, after the part that earlier
 * columns span is removed, at least `RANK_TOLERANCE` of its own norm is left. The part is removed twice, so that the
 * basis stays orthogonal when columns are nearly dependent. A column of 0 has no rank. Each column is judged against
 * its own norm, so columns on different scales, such as derivatives with respect to a probability and to its logarithm,
 * need no rescaling.
 */
export function rank(columns: readonly Float64Array[]): number {
  const basis: Float64Array[] = [];
  for (const column of columns) {
    const original = Math.sqrt(dot(column, column));
    if (!(original > 0)) continue;
    for (let pass = 0; pass < 2; pass++) {
      for (const direction of basis) subtractProjection(column, direction);
    }
    const left = Math.sqrt(dot(column, column));
    if (left < RANK_TOLERANCE * original) continue;
    for (let i = 0; i < column.length; i++) column[i] = (column[i] ?? 0) / left;
    basis.push(column);
  }
  return basis.length;
}

/** Removes from `column` its component along the unit vector `direction`. */
function subtractProjection(column: Float64Array, direction: Float64Array): void {
  const along = dot(column, direction);
  for (let i = 0; i < column.length; i++) column[i] = (column[i] ?? 0) - along * (direction[i] ?? 0);
}
