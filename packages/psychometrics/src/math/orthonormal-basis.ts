import { dot } from "@measurekit/core";

/**
 * Returns the polynomials of degree 0 to `degree` of the points, orthonormal over them, by point, then degree:
 * Gram-Schmidt on the powers of the points centered on the middle of their range and scaled by half of it, at least 1,
 * twice for precision. The points must be distinct and more than `degree`.
 */
export function orthonormalBasis(points: Float64Array, degree: number): Float64Array {
  const terms = degree + 1;
  const count = points.length;
  let lowest = Number.POSITIVE_INFINITY;
  let highest = Number.NEGATIVE_INFINITY;
  for (const point of points) {
    lowest = Math.min(lowest, point);
    highest = Math.max(highest, point);
  }
  const center = (lowest + highest) / 2;
  const scale = Math.max((highest - lowest) / 2, 1);
  const columns = Array.from({ length: terms }, (_, j) =>
    Float64Array.from(points, (x) => ((x - center) / scale) ** j),
  );
  columns.forEach((column, j) => {
    for (let pass = 0; pass < 2; pass++)
      for (const previous of columns.slice(0, j)) subtractProjection(column, previous);
    const norm = Math.sqrt(dot(column, column));
    column.forEach((value, x) => {
      column[x] = value / norm;
    });
  });
  const basis = new Float64Array(count * terms);
  columns.forEach((column, j) => {
    column.forEach((value, x) => {
      basis[x * terms + j] = value;
    });
  });
  return basis;
}

/** Subtracts from `column` its projection on the unit vector `unit`. */
function subtractProjection(column: Float64Array, unit: Float64Array): void {
  const projection = dot(column, unit);
  column.forEach((value, x) => {
    column[x] = value - projection * (unit[x] ?? 0);
  });
}
