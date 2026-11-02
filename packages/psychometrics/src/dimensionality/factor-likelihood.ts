import { type CommunalityObjective, principalAxes, projectorHessian, symmetricEigen } from "../math/internal.js";

/**
 * Returns Jöreskog's likelihood of a factor model concentrated on the uniquenesses, as an objective of the communalities:
 * the sum over the eigenvalues γ of Ψ^(-1/2) R Ψ^(-1/2) that the factors leave out of γ - log γ - 1, with Ψ the
 * uniquenesses 1 - h.
 *
 * @remarks
 * The loadings are Ψ^(1/2) times the leading eigenvectors scaled by √(γ - 1), as factanal's and psych's; the derivative
 * by a communality is (h_i - Σ_k λ_ik²) / ψ_i², and the Hessian at a perfect fit is the squares of the entries of the
 * projector onto the eigenvectors left out, each row and column divided by its uniqueness. `r` is a positive definite
 * correlation matrix by rows, and the communalities are at most 0.995, so that every uniqueness is positive.
 *
 * @see {@link https://doi.org/10.1007/BF02289658 | Jöreskog (1967)}
 */
export function likelihoodObjective(r: Float64Array, size: number, factors: number): CommunalityObjective {
  return (communalities) => {
    const uniqueness = Float64Array.from(communalities, (communality) => 1 - communality);
    const root = uniqueness.map(Math.sqrt);
    const scaled = new Float64Array(size * size);
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        scaled[i * size + j] = (r[i * size + j] ?? 0) / ((root[i] ?? 1) * (root[j] ?? 1));
      }
    }
    const { values, vectors } = symmetricEigen(scaled, size);
    let value = 0;
    for (let k = factors; k < size; k++) value += (values[k] ?? 1) - Math.log(values[k] ?? 1) - 1;
    const extent = values.map((eigenvalue) => eigenvalue - 1);
    const loadings = principalAxes(extent, vectors, size, factors, root);
    const gradient = Float64Array.from(communalities, (communality, i) => {
      let common = 0;
      for (let k = 0; k < factors; k++) common += (loadings[i * factors + k] ?? 0) ** 2;
      return (communality - common) / (uniqueness[i] ?? 1) ** 2;
    });
    const inverse = uniqueness.map((value) => 1 / value);
    return {
      value,
      gradient,
      hessian: (free) => projectorHessian(vectors, size, factors, free, inverse),
      loadings: () => loadings,
    };
  };
}
