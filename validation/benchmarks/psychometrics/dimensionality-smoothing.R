# The eigendecomposition of a correlation matrix of 25 to 1,000 items, the most that the analyses of dimensionality take:
# smoothing finds each matrix positive definite and returns it as it is, against psych's cor.smooth, which takes the
# eigendecomposition by base R's eigen, LAPACK's dsyevr. Each matrix holds the correlations of five factors whose loadings
# are multiples of 1/1024, so that their products and the sums of five are exact, and R and measurekit build the same
# matrix from the first rows of the loadings.

suppressPackageStartupMessages(library(psych))

SIZES <- c(25, 50, 100, 200, 1000)
FACTORS <- 5
set.seed(20261104)
loadings <- matrix(round(runif(max(SIZES) * FACTORS, 0, 0.4) * 1024) / 1024, ncol = FACTORS)

bench_fixture("psychometrics", "dimensionality-smoothing", list(
  source = paste("Loadings of 1,000 items on 5 factors, multiples of 1/1024 from 0 to 0.4 (set.seed(20261104)); a matrix",
                 "of n items holds the correlations of the first n, their products, with 1 on the diagonal."),
  sizes = SIZES,
  loadings = unname(loadings)
))

for (items in SIZES) {
  r <- tcrossprod(loadings[seq_len(items), , drop = FALSE])
  diag(r) <- 1
  stopifnot(min(eigen(r, symmetric = TRUE, only.values = TRUE)$values) > .Machine$double.eps)
  bench_task(sprintf("dimensionality.smoothCorrelations%d", items),
             "Smoothing of a positive definite correlation matrix: its eigendecomposition",
             sprintf("%s items", count(items)), reference("psych", "cor.smooth"), function() cor.smooth(r),
             note = if (items == max(SIZES)) paste("base R's eigen runs LAPACK's dsyevr on OpenBLAS, a reduction in blocks and MRRR in",
                                                   "native code, and measurekit Householder's reduction and the QL rotations in JavaScript"))
}
