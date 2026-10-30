# Numerics fixtures: normal, logistic and chi-square functions from base R, and Gauss-Hermite rules for the standard
# normal density by the Golub-Welsch eigenvalue method, an algorithm independent of the package's.

source("validation/psychometrics/common.R")

p <- c(1e-300, 1e-20, 1e-10, 1e-5, 0.001, 0.01, 0.025, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.975, 0.999,
       1 - 1e-10)
x <- c(-30, -10, -5, -2.5, -1, -0.3, 0, 0.3, 1, 2.5, 5, 10, 30)
# The normal distribution function across the branches of Cody's algorithm, at 0.67448975, sqrt(32) and 37.5193.
cumulative_x <- c(-40, -37.5, -20, -8, -5.7, -5.6, -3, -0.7, -0.6, -1e-20, 0, 1e-17, 0.2, 0.6744, 0.675, 2, 5.6, 5.7,
                  8, 9, 40)
chi_square <- expand.grid(x = c(0, 1e-8, 0.01, 0.5, 1, 2.5, 3.841459, 7, 15, 40, 100, 400), df = c(1, 2, 3, 5, 12, 31, 100))
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
                   upperTail = pchisq(chi_square$x, chi_square$df, lower.tail = FALSE))
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
write_fixture("math-gauss-hermite", list(
  source = "Golub-Welsch: eigen() of the Jacobi matrix of the probabilists' Hermite polynomials.",
  rules = lapply(c(1, 2, 5, 21, 61), gauss_hermite)
))
