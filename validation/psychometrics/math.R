# Numerics fixtures: normal, logistic and chi-square functions from base R, and Gauss-Hermite rules for the standard
# normal density by the Golub-Welsch eigenvalue method, an algorithm independent of measurekit's.

source("validation/psychometrics/common.R")

p <- c(1e-300, 1e-20, 1e-10, 1e-5, 0.001, 0.01, 0.025, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.975, 0.999,
       1 - 1e-10)
x <- c(-30, -10, -5, -2.5, -1, -0.3, 0, 0.3, 1, 2.5, 5, 10, 30)
# The normal distribution function across the branches of Cody's algorithm, at 0.67448975, sqrt(32) and 37.5193.
cumulative_x <- c(-40, -37.5, -20, -8, -5.7, -5.6, -3, -0.7, -0.6, -1e-20, 0, 1e-17, 0.2, 0.6744, 0.675, 2, 5.6, 5.7,
                  8, 9, 40)
chi_square <- expand.grid(x = c(0, 1e-8, 0.01, 0.5, 1, 2.5, 3.841459, 7, 15, 40, 100, 400), df = c(1, 2, 3, 5, 12, 31, 100))
# Near the center of a distribution with many degrees of freedom, where the series of the incomplete gamma function
# needs terms in proportion to the square root of the degrees of freedom.
chi_square_large <- data.frame(x = c(1e5, 99000, 6e5, 598000, 6e5 + 10, 1.8e6, 5e6, 4.99e6, 1e7), df = c(1e5, 1e5, 6e5, 6e5, 6e5, 1.8e6, 5e6, 5e6, 1e7))
write_fixture("math-functions", list(
  source = "Base R qnorm, dnorm, pnorm, plogis and pchisq(lower.tail = FALSE).",
  p = p,
  normalQuantile = qnorm(p),
  x = x,
  normalDensity = dnorm(x),
  cumulativeX = cumulative_x,
  normalDistribution = pnorm(cumulative_x),
  logistic = plogis(x),
  chiSquare = list(x = chi_square$x, df = chi_square$df,
                   upperTail = pchisq(chi_square$x, chi_square$df, lower.tail = FALSE)),
  chiSquareLarge = list(x = chi_square_large$x, df = chi_square_large$df,
                        upperTail = pchisq(chi_square_large$x, chi_square_large$df, lower.tail = FALSE))
))

gauss_hermite <- function(count) {
  if (count == 1) {
    return(list(count = 1, nodes = I(0), weights = I(1)))
  }
  jacobi <- matrix(0, count, count)
  off_diagonal <- sqrt(seq_len(count - 1))
  jacobi[cbind(seq_len(count - 1), 2:count)] <- off_diagonal
  jacobi[cbind(2:count, seq_len(count - 1))] <- off_diagonal
  decomposition <- eigen(jacobi, symmetric = TRUE)
  order <- order(decomposition$values)
  list(
    count = count,
    nodes = decomposition$values[order],
    weights = decomposition$vectors[1, order]^2
  )
}
# 200 and 500 points: Newton's method from asymptotic starting values, Numerical Recipes' gauher, returns the largest
# root again for every other root from about 195 points, and the rules must hold for any number of points.
write_fixture("math-gauss-hermite", list(
  source = "Golub-Welsch: eigen() of the Jacobi matrix of the probabilists' Hermite polynomials.",
  rules = lapply(c(1, 2, 5, 21, 61, 200, 500), gauss_hermite)
))

# Symmetric eigendecompositions by base R's eigen(symmetric = TRUE), which calls LAPACK dsyevr, an algorithm other than
# the package's Jacobi method. Eigenvectors are stored by rows, in the order of the eigenvalues, largest first; they are
# unique up to sign where the eigenvalues are distinct.
eigen_case <- function(name, matrix) {
  decomposition <- eigen(matrix, symmetric = TRUE)
  distinct <- all(diff(decomposition$values) < -1e-8)
  list(name = name, size = nrow(matrix), matrix = unname(matrix), values = I(decomposition$values),
       vectors = if (distinct) unname(t(decomposition$vectors)) else NULL)
}

set.seed(20261029)
random_symmetric <- crossprod(matrix(rnorm(25), 5)) - diag(c(0.5, 1, 1.5, 2, 2.5))
rotation <- qr.Q(qr(matrix(rnorm(9), 3)))
repeated <- rotation %*% diag(c(5, 2, 2)) %*% t(rotation)
repeated <- (repeated + t(repeated)) / 2
lsat7_correlation <- cor(mirt::expand.table(mirt::LSAT7))
bfi_correlation <- cor(psych::bfi[, 1:25], use = "pairwise.complete.obs")
write_fixture("math-eigen", list(
  source = paste("Base R eigen(symmetric = TRUE): a random symmetric matrix with negative eigenvalues, a matrix with a",
                 "repeated eigenvalue, mirt::LSAT7's correlations, psych::bfi's pairwise correlations (25 items), and",
                 "small matrices."),
  cases = list(
    eigen_case("a random symmetric matrix with negative eigenvalues", random_symmetric),
    eigen_case("a matrix with a repeated eigenvalue", repeated),
    eigen_case("the correlations of LSAT7", lsat7_correlation),
    eigen_case("the pairwise correlations of bfi items 1 to 25", bfi_correlation),
    eigen_case("a 2 by 2 matrix", matrix(c(2, 1, 1, 3), 2)),
    eigen_case("the identity matrix of order 3", diag(3)),
    eigen_case("one entry", matrix(-4, 1, 1))
  )
))
