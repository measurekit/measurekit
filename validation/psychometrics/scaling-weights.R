# Writes scaling-weighted-norms.json: percentile ranks and norms of a norm sample with person weights.
#
# References: Hmisc's weighted empirical distribution function wtd.Ecdf, whose mean at a score and just below it, times
# 100, is the weighted percentile rank; wtd.mean for the mean; n / (n - 1) times wtd.var(method = "ML"), the weighted
# mean squared deviation, for the variance, as the survey package's svyvar defines it. The standard errors are the delta
# method over the weights, written out: each person's weight w_l times the derivative of the statistic with respect to
# it, u_l = w_l ∂f/∂w_l, which numDeriv's jacobian takes of the statistics computed with Hmisc, and Var f = Σ (u_l - ū)².
# With equal weights this is the multinomial delta method of mokken's check.norms. The script asserts that the
# estimates and the variances times n / (n - 1) are the survey package's svymean and svyvar, by linearization with one
# stage and no strata, for the mean, the variance and a percentile rank, the weighted mean of 100 for a score below and
# 50 for one at it.
#
# Data, seeded: 80 integer scores from 0 to 20 with lognormal weights; 40 scores of one decimal from two strata weighted
# 2.5 and 0.8; the first case's scores with every weight 7, whose norms equal the unweighted ones.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages({
  library(Hmisc)
  library(numDeriv)
  library(survey)
})

SEED <- 20261104
LEVEL <- 0.9
STANINES <- c(-1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75)

# The weighted percentile rank of each of `at`: 100 times the mean of wtd.Ecdf at the score and just below it.
percentile_ranks <- function(x, w, at) {
  e <- wtd.Ecdf(x, w, type = "i/n", normwt = FALSE)
  # wtd.Ecdf opens with the lowest score at 0; the rest are the distinct scores and the distribution function at each.
  xs <- e$x[-1]
  Fs <- e$ecdf[-1]
  sapply(at, function(a) {
    through <- Fs[xs <= a]
    below <- Fs[xs < a]
    100 * ((if (length(through)) max(through) else 0) + (if (length(below)) max(below) else 0)) / 2
  })
}

statistics <- function(w, x, R) {
  m <- wtd.mean(x, w)
  s <- sqrt(length(x) / (length(x) - 1) * wtd.var(x, w, method = "ML"))
  c(m, s, (R - m) / s, percentile_ranks(x, w, R), m + STANINES * s)
}

interval <- function(estimate, error) {
  z <- qnorm(1 - (1 - LEVEL) / 2)
  list(estimate = estimate, standardError = error, lower = estimate - z * error, upper = estimate + z * error)
}

norm_case <- function(name, x, w, ranked) {
  R <- sort(unique(x))
  K <- length(R)
  n <- length(x)
  estimates <- statistics(w, x, R)
  J <- jacobian(function(v) statistics(v, x, R), w)
  u <- sweep(J, 2, w, "*")
  errors <- sqrt(rowSums((u - rowMeans(u))^2))
  # survey's linearized variances are n / (n - 1) times the delta method's.
  design <- svydesign(ids = ~1, weights = ~w, data = data.frame(x = x, w = w, rank = 100 * ((x < R[2]) + (x == R[2]) / 2)))
  agree <- function(statistic, estimate, error) {
    stopifnot(abs(coef(statistic) - estimate) < 1e-10 * max(1, abs(estimate)))
    stopifnot(abs(SE(statistic) - error * sqrt(n / (n - 1))) < 1e-7 * error)
  }
  agree(svymean(~x, design), estimates[1], errors[1])
  agree(svyvar(~x, design), estimates[2]^2, 2 * estimates[2] * errors[2])
  agree(svymean(~rank, design), estimates[2 + K + 2], errors[2 + K + 2])
  at <- function(k) interval(estimates[k], errors[k])
  list(
    name = name,
    scores = x,
    personWeights = w,
    ranked = ranked,
    ranks = percentile_ranks(x, w, ranked),
    level = LEVEL,
    mean = at(1),
    sd = at(2),
    norms = lapply(seq_len(K), function(k) list(score = R[k], z = at(2 + k), percentile = at(2 + K + k))),
    stanines = lapply(seq_along(STANINES), function(k) at(2 + 2 * K + k))
  )
}

set.seed(SEED)
integers <- rbinom(80, 20, 0.6)
lognormal <- round(100 * exp(rnorm(80, 0, 0.5)), 2)
decimals <- round(c(rnorm(25, 500, 100), rnorm(15, 560, 80)), 1)
strata <- c(rep(2.5, 25), rep(0.8, 15))

cases <- list(
  norm_case("integer scores with lognormal weights", integers, lognormal, c(-1, 0, 7.5, 12, 12.5, 20, 21)),
  norm_case("scores of one decimal from two strata", decimals, strata, c(300, 500, decimals[1], 900)),
  norm_case("integer scores with equal weights", integers, rep(7, 80), c(-1, 12, 21))
)

write_fixture("scaling-weighted-norms", list(
  source = paste(
    "Hmisc 5.3.0's wtd.Ecdf, wtd.mean and wtd.var(method = 'ML') times n / (n - 1); standard errors by the delta method",
    "over the weights with numDeriv 2016.8-1.1's jacobian, checked against survey 4.5's svymean and svyvar; level 0.9."
  ),
  cases = cases
))
