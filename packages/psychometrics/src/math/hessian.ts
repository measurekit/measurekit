import type { GradientFunction } from "./optimize.js";

/** The relative step of the central differences, about the cube root of the machine epsilon. */
const STEP = 1e-5;

/**
 * Returns the Hessian of a function at `point`, by rows, from central differences of its gradient with steps of
 * 1e-5 times the coordinate's magnitude, at least 1e-5. The result is symmetrized.
 */
export function numericalHessian(f: GradientFunction, point: Float64Array): Float64Array {
  const size = point.length;
  const hessian = new Float64Array(size * size);
  const x = Float64Array.from(point);
  const forward = new Float64Array(size);
  const backward = new Float64Array(size);
  for (let j = 0; j < size; j++) {
    const center = point[j] ?? 0;
    const step = STEP * Math.max(1, Math.abs(center));
    x[j] = center + step;
    f(x, forward);
    x[j] = center - step;
    f(x, backward);
    x[j] = center;
    for (let i = 0; i < size; i++) hessian[i * size + j] = ((forward[i] ?? 0) - (backward[i] ?? 0)) / (2 * step);
  }
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < i; j++) {
      const mean = ((hessian[i * size + j] ?? 0) + (hessian[j * size + i] ?? 0)) / 2;
      hessian[i * size + j] = mean;
      hessian[j * size + i] = mean;
    }
  }
  return hessian;
}
