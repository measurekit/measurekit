# Livingston and Lewis's classification accuracy and consistency (standard-setting-livingston-lewis.json). Reference:
# betafunctions 2.0.0, LL.CA(true.model = "4P") for one cut and LL.CA.MC for several, with its fail-safe two-parameter
# beta. The effective test length, its rounding, the observed cuts and the beta's parameters come from them. Their
# integrals keep R's default tolerance, so the script recomputes the indices in base R from those parameters, with
# integrate(rel.tol = 1e-12) over each category's true range, and writes both.
#
# Data, set.seed(20261102): totals and Cronbach's alpha of simulated 2PL responses, 3,000 persons and 40 items, with one
# and three cuts; 1,500 persons and 30 items, whose four-parameter beta is impermissible, so the fail-safe engages;
# 1,500 persons of two separate groups on 30 items, whose true-score beta has both shapes below 1; and sums of 12
# five-point ratings from 1 to 5, a scale from 12 to 60.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages(library(betafunctions))

INTEGRATION <- 1e-12
set.seed(20261102)

cronbach_alpha <- function(m) {
  k <- ncol(m)
  k / (k - 1) * (1 - sum(apply(m, 2, var)) / var(rowSums(m)))
}

simulate_2pl <- function(theta, items, slope = c(0.6, 1.8)) {
  a <- runif(items, slope[1], slope[2])
  b <- rnorm(items, -0.2, 0.9)
  sapply(seq_len(items), function(j) rbinom(length(theta), 1, plogis(a[j] * (theta - b[j]))))
}

# Ratings from 1 to 5 by a graded model: each item's thresholds spread around its location.
simulate_ratings <- function(theta, items) {
  sapply(seq_len(items), function(j) {
    thresholds <- rnorm(1, 0, 0.5) + c(-1.8, -0.6, 0.4, 1.5)
    above <- sapply(thresholds, function(t) plogis(1.3 * (theta - t)))
    1 + rowSums(matrix(runif(length(theta) * 4), ncol = 4) < above)
  })
}

# The integral of the beta density times f over [from, to] within the beta's support, by integrate on the beta's own
# scale t = (p - l) / (u - l), split at t = 1/2. Where a shape is below 1 the density is infinite at that end, so the
# half there is integrated in v = t^alpha, or w = (1 - t)^beta, in which the density's factor t^(alpha - 1) dt is dv /
# alpha: an exact change of variables that leaves a bounded integrand.
beta_integral <- function(f, params, from, to) {
  a <- params$alpha
  b <- params$beta
  width <- params$u - params$l
  at <- function(t) f(params$l + width * t)
  lo <- max(0, (from - params$l) / width)
  hi <- min(1, (to - params$l) / width)
  if (!(lo < hi)) return(0)
  log_beta <- lbeta(a, b)
  piece <- function(x0, x1) {
    if (!(x0 < x1)) return(0)
    if (x1 <= 0.5 && a < 1) {
      g <- function(v) { t <- v^(1 / a); exp((b - 1) * log1p(-t) - log_beta) / a * at(t) }
      return(integrate(g, x0^a, x1^a, rel.tol = INTEGRATION, subdivisions = 1000L)$value)
    }
    if (x0 >= 0.5 && b < 1) {
      g <- function(w) { t <- 1 - w^(1 / b); exp((a - 1) * log(t) - log_beta) / b * at(t) }
      return(integrate(g, (1 - x1)^b, (1 - x0)^b, rel.tol = INTEGRATION, subdivisions = 1000L)$value)
    }
    g <- function(t) exp((a - 1) * log(t) + (b - 1) * log1p(-t) - log_beta) * at(t)
    integrate(g, x0, x1, rel.tol = INTEGRATION, subdivisions = 1000L)$value
  }
  piece(lo, min(hi, 0.5)) + piece(max(lo, 0.5), hi)
}

# Accuracy and consistency from the beta's parameters, the rounded effective length N and the cuts on the proportion
# scale: the beta-weighted integral of the probability of each observed category over its true range, and of the sum
# of their squares over the support.
indices <- function(params, N, cuts) {
  thresholds <- ceiling(cuts * N - 1e-9 * pmax(1, abs(cuts * N)))
  K <- length(cuts) + 1
  category <- function(p, j) {
    upper <- if (j == K) 1 else pbinom(thresholds[j] - 1, N, p)
    lower <- if (j == 1) 0 else pbinom(thresholds[j - 1] - 1, N, p)
    upper - lower
  }
  bounds <- c(-Inf, cuts, Inf)
  accuracy <- sum(sapply(seq_len(K), function(j) {
    beta_integral(function(p) category(p, j), params, bounds[j], bounds[j + 1])
  }))
  consistency <- beta_integral(function(p) Reduce(`+`, lapply(seq_len(K), function(j) category(p, j)^2)), params,
                               -Inf, Inf)
  list(accuracy = accuracy, consistency = consistency)
}

case <- function(name, scores, reliability, cuts, minimum, maximum) {
  proportions <- (cuts - minimum) / (maximum - minimum)
  per_cut <- lapply(seq_along(cuts), function(k) {
    one <- suppressWarnings(LL.CA(scores, reliability, cuts[k], min = minimum, max = maximum, modelfit = NULL))
    written <- indices(one$parameters, one$parameters$etl_rounded, proportions[k])
    list(cut = cuts[k], accuracy = written$accuracy, consistency = written$consistency,
         package = list(accuracy = one$classification.accuracy$Accuracy,
                        consistency = one$classification.consistency$p))
  })
  several <- suppressWarnings(LL.CA.MC(scores, reliability, cuts, min = minimum, max = maximum, modelfit = NULL))
  params <- several$parameters
  overall <- indices(params, params$etl_rounded, proportions)
  engaged <- isTRUE(params$l == 0 && params$u == 1)
  list(name = name, scores = scores, reliability = reliability, cuts = I(cuts), minimum = minimum, maximum = maximum,
       effectiveLength = params$etl, roundedLength = params$etl_rounded,
       observedCuts = I(ceiling(proportions * params$etl_rounded - 1e-9 * pmax(1, abs(proportions * params$etl_rounded)))),
       trueScore = list(alpha = params$alpha, beta = params$beta, lower = params$l, upper = params$u),
       failSafe = engaged,
       perCut = per_cut, accuracy = overall$accuracy, consistency = overall$consistency,
       package = if (length(cuts) == 1) per_cut[[1]]$package else
         list(accuracy = several$accuracy$overall$accuracy, consistency = several$consistency$overall$statistics$p))
}

theta <- rnorm(3000)
responses <- simulate_2pl(theta, 40)
totals <- rowSums(responses)
reliability <- cronbach_alpha(responses)

failsafe_responses <- simulate_2pl(rnorm(1500), 30, slope = c(1.2, 1.2))
groups <- c(rnorm(800, 1.6, 0.4), rnorm(700, -1.6, 0.4))
separated_responses <- simulate_2pl(groups, 30, slope = c(1.5, 1.5))
ratings <- simulate_ratings(rnorm(1200), 12)

cases <- list(
  case("one cut", totals, reliability, 24, 0, 40),
  case("three cuts", totals, reliability, c(16, 24, 32), 0, 40),
  case("the fail-safe beta", rowSums(failsafe_responses), cronbach_alpha(failsafe_responses), c(12, 20), 0, 30),
  case("two separate groups", rowSums(separated_responses), cronbach_alpha(separated_responses), c(10, 20), 0, 30),
  case("ratings from 12 to 60", rowSums(ratings), cronbach_alpha(ratings), c(30, 42), 12, 60)
)
names(cases) <- sapply(cases, `[[`, "name")
stopifnot(cases[["the fail-safe beta"]]$failSafe, !cases[["one cut"]]$failSafe)
stopifnot(cases[["two separate groups"]]$trueScore$alpha < 1, cases[["two separate groups"]]$trueScore$beta < 1)

write_fixture("standard-setting-livingston-lewis", list(
  source = paste("betafunctions 2.0.0: LL.CA(true.model = \"4P\") per cut and LL.CA.MC, with the fail-safe; the indices",
                 "recomputed by integrate(rel.tol = 1e-12) from their parameters; simulated scores,",
                 "set.seed(20261102)."),
  cases = cases
))
