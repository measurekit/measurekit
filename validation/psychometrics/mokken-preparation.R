# Two-way imputation and norms, steps 2 and 9 of a Mokken scale analysis (mokken-imputation.json and
# scaling-norm-statistics.json). References: mokken's twoway without its error term, run with rnorm returning 0, which
# leaves round(PM + IM - OM) within the bounds; and check.norms, whose estimates and standard errors are taken unrounded
# and whose intervals, at the constant 1.96, are recomputed at qnorm of the level. The standard errors of the z scores
# come from the delta method written out in base R, as check.norms subtracts its correction term twice for them.
#
# twoway draws its error with rnorm(N J, 0, var(residuals)): the variance where rnorm takes a standard deviation. The
# script asserts that a seeded twoway equals round(TW + rnorm(N J, 0, v)) with v the residual variance, so the package's
# error, of standard deviation √v as van Ginkel et al. (2007) define it, is checked against TW and v, which the fixture
# holds.
#
# Data: mokken's acl, its first 10 adjectives, with 10% of the responses missing at random; transreas with 5% missing;
# small matrices whose two-way values fall on a half and below every score; and total scores of acl, cavalini and a
# seeded set with negatives.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages(library(mokken))

SEED <- 20261037
LEVELS <- c(0.95, 0.9)
# check.norms' constant multiplier of a standard error, the 0.975 quantile of the standard normal to two decimals.
MOKKEN_Z <- 1.96

rows_of <- function(X) unname(lapply(seq_len(nrow(X)), function(i) as.numeric(X[i, ])))

# --- Two-way imputation -------------------------------------------------------------------------------------------------

two_way <- function(X) {
  outer(rowMeans(X, na.rm = TRUE), colMeans(X, na.rm = TRUE), "+") - mean(X, na.rm = TRUE)
}

# mokken's twoway with its error term set to 0.
twoway_without_error <- function(X, ...) {
  f <- twoway
  environment(f) <- list2env(list(rnorm = function(n, mean, sd) rep(0, n)), parent = asNamespace("mokken"))
  suppressWarnings(f(X, ...))
}

imputation_case <- function(name, X, minimum = NULL, maximum = NULL) {
  args <- list(X)
  if (!is.null(minimum)) args$minX <- minimum
  if (!is.null(maximum)) args$maxX <- maximum
  imputed <- do.call(twoway_without_error, args)
  observed <- !is.na(X)
  tw <- two_way(X)
  residual_variance <- var((X - tw)[observed])
  # twoway's error has the residual variance as its standard deviation.
  set.seed(SEED)
  drawn <- suppressWarnings(do.call(twoway, args))
  set.seed(SEED)
  low <- if (is.null(minimum)) min(X, na.rm = TRUE) else minimum
  high <- if (is.null(maximum)) max(X, na.rm = TRUE) else maximum
  written <- pmin(pmax(round(tw + matrix(rnorm(length(X), 0, residual_variance), nrow(X))), low), high)
  written[observed] <- X[observed]
  stopifnot(all(drawn == written))
  list(name = name, responses = rows_of(X), minimum = minimum, maximum = maximum, imputed = rows_of(imputed),
       twoWay = rows_of(tw), residualVariance = residual_variance)
}

set.seed(SEED)
data(acl, transreas, cavalini)
acl_items <- as.matrix(acl[, 1:10])
storage.mode(acl_items) <- "double"
acl_missing <- acl_items
acl_missing[sample(length(acl_items), round(0.1 * length(acl_items)))] <- NA
transreas_missing <- as.matrix(transreas[, 2:13])
storage.mode(transreas_missing) <- "double"
transreas_missing[sample(length(transreas_missing), round(0.05 * length(transreas_missing)))] <- NA
# A missing cell of person mean 1.5 and item mean 2, with the overall mean 1: its two-way value is exactly 2.5, which R's
# round takes to the even 2.
halves <- matrix(c(1, 2, NA, 0, 0, 1, 1, 0, 2, 0, 1, 3), 4, 3, byrow = TRUE)
stopifnot(two_way(halves)[1, 3] == 2.5)
# Scores from 1 to 5 whose missing cell has a two-way value of -0.82: the observed lowest score bounds it by default.
low <- matrix(c(1, 1, NA, 5, 5, 1, 5, 5, 1, 3, 3, 1), 4, 3, byrow = TRUE)

imputations <- list(
  imputation_case("acl, ten adjectives, 10% missing", acl_missing),
  imputation_case("transitive reasoning, 5% missing", transreas_missing),
  imputation_case("a two-way value of a half", halves),
  imputation_case("a cell below every score", low),
  imputation_case("a cell below every score, bounds from 0 to 5", low, minimum = 0, maximum = 5)
)

# --- Norms --------------------------------------------------------------------------------------------------------------

# The delta method over the counts of the distinct scores, written out: the variance of a statistic with gradient g by
# count is Σ n g² - (Σ n g)² / N.
delta_error <- function(n, g) sqrt(sum(n * g^2) - sum(n * g)^2 / sum(n))

norm_case <- function(name, y) {
  result <- check.norms(y, nice.output = FALSE)
  R <- sort(unique(y))
  n <- as.numeric(table(factor(y, levels = R)))
  N <- sum(n)
  m <- mean(y)
  s <- sd(y)
  of_mean <- (R - m) / N
  of_sd <- ((R - m)^2 - s^2) / (2 * s * (N - 1))
  z_errors <- sapply(seq_along(R), function(j) delta_error(n, -of_mean / s - (R[j] - m) * of_sd / s^2))
  # check.norms subtracts (Σ n g)² / N twice in the variance of a z score, and once elsewhere: its other standard errors
  # are the delta method's, and its z scores' are that variance less the term once more.
  doubled <- sapply(seq_along(R), function(j) {
    g <- -of_mean / s - (R[j] - m) * of_sd / s^2
    sqrt(sum(n * g^2) - 2 * sum(n * g)^2 / N)
  })
  stopifnot(
    abs(result$mean[1, 2] - delta_error(n, of_mean)) < 1e-12,
    abs(result$sd[1, 2] - delta_error(n, of_sd)) < 1e-12,
    max(abs(result$z[, 2] - doubled)) < 1e-12
  )
  for (table in result) stopifnot(max(abs(table[, 3] - (table[, 1] - MOKKEN_Z * table[, 2]))) < 1e-12)
  result$z[, 2] <- z_errors
  interval <- function(table, level) {
    z <- qnorm(1 - (1 - level) / 2)
    unname(lapply(seq_len(nrow(table)), function(k) {
      list(estimate = table[k, 1], standardError = table[k, 2], lower = table[k, 1] - z * table[k, 2],
           upper = table[k, 1] + z * table[k, 2])
    }))
  }
  by_level <- lapply(LEVELS, function(level) {
    list(level = level, mean = interval(result$mean, level)[[1]], sd = interval(result$sd, level)[[1]],
         scores = Map(function(score, z, percentile) list(score = score, z = z, percentile = percentile),
                      R, interval(result$z, level), interval(result$perc, level)),
         stanines = interval(result$sta9, level))
  })
  list(name = name, scores = y, mokkenZErrors = doubled, levels = unname(by_level))
}

norms <- list(
  norm_case("acl, total scores of ten adjectives", rowSums(acl_items)),
  norm_case("cavalini, total scores", rowSums(as.matrix(cavalini))),
  norm_case("seeded scores from -5 to 10", sample(-5:10, 300, replace = TRUE))
)

write_fixture("mokken-imputation", list(
  source = paste("mokken 3.1.2: twoway without its error term. Data: acl's first 10 adjectives with 10% of the responses",
                 "missing and transreas with 5% missing (set.seed(20261037)), and two 4 by 3 matrices."),
  complete = rows_of(acl_items),
  cases = imputations
))
write_fixture("scaling-norm-statistics", list(
  source = paste("mokken 3.1.2: check.norms, its intervals at qnorm of the level. Data: total scores of acl's first 10",
                 "adjectives and of cavalini, and 300 seeded scores from -5 to 10 (set.seed(20261037))."),
  cases = norms
))
