# Plausible values and Rubin's rules, the recipes of scaling.md: the moments of the plausible values that mirt's fscores
# and TAM's tam.pv draw, with and without a latent regression on a covariate, their expectation computed exactly in base
# R from mirt's posteriors on a fine grid, and Rubin's combining rules written out in base R, as mice::pool computes them
# for a statistic with an infinite complete-data degrees of freedom. The data are simulated with set.seed(20261102):
# 2,000 persons answer 20 two-parameter items, and theta is 0.6 x + 0.8 e for a centered covariate x with variance 1.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages({
  library(mirt)
  library(TAM)
})

TOL <- 1e-8 # mirt's EM tolerance, tight so that its estimates are the optimum
DRAWS <- 200 # plausible values drawn for each person to estimate a moment
NODES <- 2000 # points of the grid that tam.pv draws on: its default of 21 adds the cell's variance, 0.03, to each draw
POOLED <- 5 # plausible values that Rubin's rules combine
TAM_CALLS <- 5 # calls of tam.pv whose moments are averaged: a call's moments vary by 0.02 around their mean
GRID <- seq(-8, 8, length.out = 2001) # the grid of the exact expectation, whose cells add under 1e-5 to a variance

set.seed(20261102)
persons <- 2000
items <- 20
x <- as.vector(scale(rnorm(persons), scale = FALSE))
theta <- 0.6 * x + 0.8 * rnorm(persons)
data <- simdata(matrix(exp(rnorm(items, 0, 0.3))), rnorm(items), N = persons, itemtype = "dich", Theta = matrix(theta))
colnames(data) <- sprintf("item%02d", seq_len(items))
covariates <- data.frame(x = x)

# The variance of the plausible values of every person, the mean of those variances over the draws, and the slope of the
# plausible values on x averaged the same way. `draws` is a list of vectors, one for each draw.
moments <- function(draws) {
  list(
    mean = mean(sapply(draws, mean)),
    variance = mean(sapply(draws, var)),
    slope = mean(sapply(draws, function(draw) unname(coef(lm(draw ~ x))[2])))
  )
}

# The moments of the posterior means: their variance and their slope on x.
eap_moments <- function(eap) list(variance = var(eap), slope = unname(coef(lm(eap ~ x))[2]))

# The expectation of moments() over the draws, exactly: the persons' posteriors on GRID under the prior N(`prior_mean`,
# 1), the items of `model`. The plausible values of different persons are independent with mean m_i and variance v_i, so
# a draw's variance has the expectation (Σ (m_i - m)² + (1 - 1 / n) Σ v_i) / (n - 1) and its slope on x that of the m_i.
exact_moments <- function(model, prior_mean) {
  log_likelihood <- matrix(0, nrow(data), length(GRID))
  for (j in seq_len(ncol(data))) {
    trace <- probtrace(extract.item(model, j), GRID)
    log_likelihood <- log_likelihood + outer(data[, j], log(trace[, 2])) + outer(1 - data[, j], log(trace[, 1]))
  }
  log_posterior <- log_likelihood + outer(rep(prior_mean, length.out = nrow(data)), GRID, function(m, node) dnorm(node, m, 1, log = TRUE))
  posterior <- exp(log_posterior - apply(log_posterior, 1, max))
  posterior <- posterior / rowSums(posterior)
  m <- as.vector(posterior %*% GRID)
  v <- as.vector(posterior %*% GRID^2) - m^2
  n <- length(m)
  list(
    mean = mean(m),
    variance = (sum((m - mean(m))^2) + (1 - 1 / n) * sum(v)) / (n - 1),
    slope = unname(coef(lm(m ~ x))[2])
  )
}

# The moments of TAM's draws averaged over TAM_CALLS calls with seeds of their own.
tam_moments <- function(model) {
  calls <- lapply(seq_len(TAM_CALLS), function(call) {
    set.seed(100 + call)
    draws <- as.data.frame(tam.pv(model, nplausible = DRAWS, ntheta = NODES, normal.approx = FALSE, verbose = FALSE)$pv)
    moments(lapply(grep("^PV", colnames(draws)), function(j) draws[[j]]))
  })
  lapply(setNames(names(calls[[1]]), names(calls[[1]])), function(name) mean(sapply(calls, function(call) call[[name]])))
}

# Rubin's rules (1987) for m estimates q with variances u: the pooled estimate, the within-imputation variance, the
# between-imputation variance, the total variance and the degrees of freedom (m - 1) (1 + 1 / r)^2 with
# r = (1 + 1 / m) b / u, which is mice::pool's `df` when the complete-data degrees of freedom are infinite.
pool <- function(q, u) {
  m <- length(q)
  within <- mean(u)
  between <- var(q)
  total <- within + (1 + 1 / m) * between
  r <- (1 + 1 / m) * between / within
  list(estimate = mean(q), within = within, between = between, total = total, degreesOfFreedom = (m - 1) * (1 + 1 / r)^2)
}

# --- Without covariates: mirt's Metropolis-Hastings draws, TAM's -----------------------------------------------------

# Each draw starts from its own seed, so that no fit before it moves the generator's state.
free <- mirt(data, 1, itemtype = "2PL", TOL = TOL, verbose = FALSE)
free_eap <- fscores(free, method = "EAP")[, 1]
set.seed(1)
free_draws <- lapply(fscores(free, method = "EAP", plausible.draws = DRAWS, plausible.type = "MH"), function(draw) draw[, 1])
tam_free <- tam.mml.2pl(data, verbose = FALSE)

# --- With a latent regression on x: mirt's normal draws, as it has no sampler for the regression, and TAM's ------------

conditional <- mirt(data, 1, itemtype = "2PL", covdata = covariates, formula = ~x, TOL = TOL, verbose = FALSE)
conditional_values <- mod2values(conditional)
conditional_eap <- fscores(conditional, method = "EAP")[, 1]
set.seed(2)
conditional_draws <- lapply(
  fscores(conditional, method = "EAP", plausible.draws = DRAWS, plausible.type = "normal"), function(draw) draw[, 1]
)
tam_conditional <- tam.mml.2pl(data, Y = covariates, verbose = FALSE)

# --- Rubin's rules over the first POOLED plausible values of the regression model -----------------------------------

pooled_draws <- conditional_draws[seq_len(POOLED)]
mean_estimates <- sapply(pooled_draws, mean)
mean_variances <- sapply(pooled_draws, function(draw) var(draw) / persons)
fits <- lapply(pooled_draws, function(draw) summary(lm(draw ~ x))$coefficients["x", ])
slope_estimates <- sapply(fits, function(fit) fit[["Estimate"]])
slope_variances <- sapply(fits, function(fit) fit[["Std. Error"]]^2)

write_fixture("scaling-plausible-values", list(
  source = paste(
    "simdata (set.seed(20261102)): 2,000 persons, 20 two-parameter items, theta = 0.6 x + 0.8 e with x centered.",
    "mirt 1.48 draws", DRAWS, "plausible values for each person, and TAM 4.3-25", DRAWS, "in each of", TAM_CALLS, "calls;",
    "`exact` is the expectation of a draw's moments from mirt's posteriors on a grid of", length(GRID), "points."
  ),
  itemIds = colnames(data), responses = unname(data), covariate = x, draws = DRAWS,
  withoutCovariates = list(
    mirt = c(moments(free_draws), list(eap = eap_moments(free_eap))),
    tam = tam_moments(tam_free),
    exact = exact_moments(free, 0)
  ),
  withCovariates = list(
    coefficient = conditional_values$value[conditional_values$name == "F1_x"],
    mirt = c(moments(conditional_draws), list(eap = eap_moments(conditional_eap))),
    tam = c(tam_moments(tam_conditional), list(coefficient = unname(tam_conditional$beta[2, 1]))),
    exact = exact_moments(conditional, conditional_values$value[conditional_values$name == "F1_x"] * x)
  ),
  rubin = list(
    plausibleValues = unname(do.call(cbind, pooled_draws)),
    mean = list(estimates = mean_estimates, variances = mean_variances, pooled = pool(mean_estimates, mean_variances)),
    slope = list(estimates = slope_estimates, variances = slope_variances, pooled = pool(slope_estimates, slope_variances))
  )
))
