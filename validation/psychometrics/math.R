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
# the package's Householder reduction and implicit QL. Eigenvectors are stored by rows, in the order of the eigenvalues,
# largest first. An eigenvector is unique up to sign where its eigenvalue is apart from the others, and any algorithm
# finds it to rounding over that gap: a row is null where the gap is below SEPARATION times the largest absolute
# eigenvalue, or 1, so that the tests compare only the vectors that rounding leaves exact to their tolerance.
SEPARATION <- 1e-3
eigen_case <- function(name, matrix) {
  decomposition <- eigen(matrix, symmetric = TRUE)
  values <- decomposition$values
  gaps <- vapply(seq_along(values), function(k) min(Inf, abs(values[k] - values[-k])), 0)
  separated <- gaps > SEPARATION * max(1, abs(values))
  list(name = name, size = nrow(matrix), matrix = unname(matrix), values = I(values),
       vectors = lapply(seq_along(values), function(k) if (separated[k]) I(decomposition$vectors[, k]) else NULL))
}

set.seed(20261029)
random_symmetric <- crossprod(matrix(rnorm(25), 5)) - diag(c(0.5, 1, 1.5, 2, 2.5))
rotation <- qr.Q(qr(matrix(rnorm(9), 3)))
repeated <- rotation %*% diag(c(5, 2, 2)) %*% t(rotation)
repeated <- (repeated + t(repeated)) / 2
lsat7_correlation <- cor(mirt::expand.table(mirt::LSAT7))
bfi_correlation <- cor(psych::bfi[, 1:25], use = "pairwise.complete.obs")
# A 40 by 40 matrix of normal entries made symmetric, whose eigenvalues spread from about -9 to 9.
wigner <- matrix(rnorm(40 * 40), 40)
wigner <- (wigner + t(wigner)) / 2
# The correlations of 40 normal variables over 30 persons: of rank 29, so that 11 eigenvalues are 0 to rounding.
few_persons <- cor(matrix(rnorm(30 * 40), 30))
# Two eigenvalues 1e-9 apart, one of 1e-10 and one of 0.
basis <- qr.Q(qr(matrix(rnorm(36), 6)))
clustered <- basis %*% diag(c(4, 2 + 1e-9, 2, 1, 1e-10, 0)) %*% t(basis)
clustered <- (clustered + t(clustered)) / 2

# The correlation matrices that the tests of dimensionality analyse, from the fixture of exploratory-factors.R.
factors <- jsonlite::fromJSON("packages/psychometrics/test/fixtures/dimensionality-factors.json", simplifyVector = FALSE)
as_matrix <- function(rows) matrix(unlist(rows), nrow = length(rows), byrow = TRUE)
factor_matrix <- function(name) as_matrix(Filter(function(entry) entry$name == name, factors$cases)[[1]]$matrix)
smoothing_matrix <- function(name) as_matrix(Filter(function(entry) entry$name == name, factors$smoothing)[[1]]$raw)

write_fixture("math-eigen", list(
  source = paste("Base R eigen(symmetric = TRUE): random symmetric matrices with negative eigenvalues, a matrix with a",
                 "repeated eigenvalue, one with two eigenvalues 1e-9 apart and two near 0, the correlations of 40",
                 "variables over 30 persons, mirt::LSAT7's correlations, psych::bfi's pairwise correlations (25 items),",
                 "the correlation matrices of dimensionality-factors.json, and small matrices."),
  cases = list(
    eigen_case("a random symmetric matrix with negative eigenvalues", random_symmetric),
    eigen_case("a matrix with a repeated eigenvalue", repeated),
    eigen_case("the correlations of LSAT7", lsat7_correlation),
    eigen_case("the pairwise correlations of bfi items 1 to 25", bfi_correlation),
    eigen_case("a random symmetric 40 by 40 matrix", wigner),
    eigen_case("the correlations of 40 variables over 30 persons, 11 eigenvalues of 0", few_persons),
    eigen_case("a matrix with two eigenvalues 1e-9 apart and two near 0", clustered),
    eigen_case("the tetrachoric correlations of 12 simulated items", factor_matrix("two factors, tetrachoric")),
    eigen_case("the polychoric correlations of bfi items 1 to 25", factor_matrix("bfi, five factors, polychoric")),
    eigen_case("the tetrachoric correlations of SAT12, with a negative eigenvalue", smoothing_matrix("SAT12, tetrachoric")),
    eigen_case("the tetrachoric correlations of SAT12, smoothed", factor_matrix("SAT12, two factors, tetrachoric, smoothed")),
    eigen_case("Harman's 24 psychological tests", factor_matrix("Harman's 24 tests, four factors")),
    eigen_case("a one-factor matrix with a loading of 0.99999", factor_matrix("one factor with a communality beyond its bound")),
    eigen_case("three correlations that are not positive definite", smoothing_matrix("three items")),
    eigen_case("a 2 by 2 matrix", matrix(c(2, 1, 1, 3), 2)),
    eigen_case("the identity matrix of order 3", diag(3)),
    eigen_case("one entry", matrix(-4, 1, 1))
  )
))
