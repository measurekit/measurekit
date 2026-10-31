# Explanatory item response theory fixtures: the regression of item parameters on item features, estimated in two stages.
# The estimates and standard errors of a bank are mirt's 2PL calibration of simulated responses (SE.type = "Richardson",
# the delta method for b); the regression of them on the features by restricted maximum likelihood is the base R
# maximization of the restricted likelihood, and metafor's rma(method = "REML") agrees. The predictions of new items are
# the mean x' eta and the sd sqrt(tau^2 + x' Cov x), written out.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages({
  library(mirt)
  library(metafor)
  library(numDeriv)
})

POINTS <- 61 # mirt's default quadrature: 61 equally spaced nodes on [-6, 6]
TOL <- 1e-10 # mirt's EM tolerance, tight so that its estimates are the optimum

# --- Banks ------------------------------------------------------------------------------------------------------------
# Item i has the features x_i and the difficulty b_i = x_i' beta + e_i, e_i ~ N(0, tau^2), and the discrimination
# a_i = exp(0.1 + 0.15 x_i1 + u_i), u_i ~ N(0, 0.15^2). Responses follow the 2PL for theta ~ N(0, 1).

simulate_bank <- function(persons, items, beta, tau, seed) {
  set.seed(seed)
  features <- cbind(steps = round(runif(items, 1, 6)), density = round(rnorm(items), 2),
                    open = rbinom(items, 1, 0.4))[, seq_len(length(beta) - 1), drop = FALSE]
  b <- drop(cbind(1, features) %*% beta) + rnorm(items, 0, tau)
  a <- exp(0.1 + 0.15 * scale(features[, 1], scale = FALSE) + rnorm(items, 0, 0.15))
  theta <- rnorm(persons)
  p <- plogis(outer(theta, seq_len(items), function(t, i) a[i] * (t - b[i])))
  responses <- matrix(rbinom(persons * items, 1, p), persons, items)
  colnames(responses) <- sprintf("q%02d", seq_len(items))
  list(responses = responses, features = features, a = drop(a), b = b)
}

# The package's items of mirt's calibration: a and b with their standard errors.
calibrated_items <- function(responses) {
  fit <- mirt(as.data.frame(responses), 1, itemtype = "2PL", SE = TRUE, SE.type = "Richardson", TOL = TOL,
              quadpts = POINTS, verbose = FALSE, technical = list(NCYCLES = 1e5))
  stopifnot(extract.mirt(fit, "converged"))
  coefficients <- coef(fit, IRTpars = TRUE, printSE = TRUE)
  lapply(colnames(responses), function(id) {
    x <- coefficients[[id]]
    list(id = id, parameters = list(a = x["par", "a"], b = x["par", "b"], c = 0, d = 1),
         standardErrors = list(a = x["SE", "a"], b = x["SE", "b"]))
  })
}

# --- The regression ---------------------------------------------------------------------------------------------------
# The restricted log-likelihood of y = X eta + u + e, u ~ N(0, tau^2), e ~ N(0, v), with the constants of its density:
# -(n - p) / 2 ln 2 pi + ln |X'X| / 2 - sum ln(v + tau^2) / 2 - ln |X'WX| / 2 - (y - X eta)' W (y - X eta) / 2.

generalized_least_squares <- function(y, v, X, tau2) {
  w <- 1 / (v + tau2)
  information <- crossprod(X, X * w)
  covariance <- solve(information)
  beta <- drop(covariance %*% crossprod(X, w * y))
  residual <- y - drop(X %*% beta)
  list(beta = beta, covariance = covariance, information = information, w = w, residual = residual)
}

restricted_loglik <- function(tau2, y, v, X) {
  fit <- generalized_least_squares(y, v, X, tau2)
  n <- length(y)
  p <- ncol(X)
  -(n - p) / 2 * log(2 * pi) + as.numeric(determinant(crossprod(X))$modulus) / 2 - sum(log(v + tau2)) / 2 -
    as.numeric(determinant(fit$information)$modulus) / 2 - sum(fit$w * fit$residual^2) / 2
}

# The derivative of the restricted log-likelihood with respect to tau^2: -tr(P) / 2 + y' P P y / 2, with
# P = W - W X (X'WX)^-1 X'W.
restricted_score <- function(tau2, y, v, X) {
  fit <- generalized_least_squares(y, v, X, tau2)
  W <- diag(fit$w)
  P <- W - W %*% X %*% fit$covariance %*% t(X) %*% W
  -sum(diag(P)) / 2 + sum((fit$w * fit$residual)^2) / 2
}

# tau^2 by optimize over [0, upper], or 0 where the likelihood is highest there; an interior maximum is the root of the
# score, found by uniroot around the optimize result, which the central differences of the likelihood confirm.
reml_tau2 <- function(y, v, X) {
  objective <- function(t) restricted_loglik(t, y, v, X)
  upper <- 100 * (var(y) + max(v))
  tau2 <- optimize(objective, c(0, upper), maximum = TRUE, tol = 1e-14)$maximum
  if (objective(0) >= objective(tau2)) return(0)
  score <- function(t) restricted_score(t, y, v, X)
  width <- 1e-4 * max(tau2, 1e-3)
  bracket <- c(max(0, tau2 - width), tau2 + width)
  stopifnot(score(bracket[1]) > 0, score(bracket[2]) < 0)
  tau2 <- uniroot(score, bracket, tol = 1e-15)$root
  stopifnot(abs(score(tau2) - grad(objective, tau2, method.args = list(r = 6))) < 1e-6)
  tau2
}

# One case: the bank's estimates of a parameter regressed on a design, with the base R estimates and metafor's.
regression_case <- function(name, items, features, parameter, intercept, new_features) {
  ids <- vapply(items, `[[`, "", "id")
  estimate <- vapply(items, function(item) item$parameters[[parameter]], 0)
  error <- vapply(items, function(item) item$standardErrors[[parameter]], 0)
  # A slope is regressed as its logarithm, with the delta method variance (se / a)^2.
  log_scale <- parameter == "a"
  y <- if (log_scale) log(estimate) else estimate
  v <- if (log_scale) (error / estimate)^2 else error^2
  X <- cbind(if (intercept) 1, features)
  colnames(X) <- c(if (intercept) "(intercept)", colnames(features))
  tau2 <- reml_tau2(y, v, X)
  fit <- generalized_least_squares(y, v, X, tau2)
  loglik <- restricted_loglik(tau2, y, v, X)
  # metafor at the same tau^2 gives the same likelihood, coefficients and covariance; its own search finds the maximum
  # too, except for the design without an intercept, where it stops short of it.
  at_tau2 <- rma(yi = y, sei = sqrt(v), mods = X, intercept = FALSE, method = "REML", tau2 = tau2)
  stopifnot(max(abs(coef(at_tau2) - fit$beta)) < 1e-8, max(abs(vcov(at_tau2) - fit$covariance)) < 1e-8,
            abs(as.numeric(logLik(at_tau2)) - loglik) < 1e-8)
  searched <- rma(yi = y, sei = sqrt(v), mods = X, intercept = FALSE, method = "REML",
                  control = list(threshold = 1e-12, maxiter = 1000))
  stopifnot(loglik >= as.numeric(logLik(searched)) - 1e-8)
  message(name, ": tau^2 ", tau2, ", metafor's search ", searched$tau2)
  if (intercept) stopifnot(abs(searched$tau2 - tau2) < 1e-6)
  prediction <- lapply(new_features, function(x) {
    row <- c(if (intercept) 1, x)
    list(features = as.list(x), mean = sum(row * fit$beta),
         sd = sqrt(tau2 + drop(t(row) %*% fit$covariance %*% row)))
  })
  list(
    name = name, parameter = parameter, intercept = intercept,
    items = items,
    features = setNames(lapply(seq_along(ids), function(i) as.list(features[i, ])), ids),
    effects = lapply(seq_len(ncol(X)), function(j) {
      list(name = colnames(X)[j], estimate = fit$beta[j], standardError = sqrt(fit$covariance[j, j]))
    }),
    covariance = unname(fit$covariance),
    residualSd = sqrt(tau2),
    logLikelihood = loglik,
    predictions = prediction
  )
}

new_items <- list(c(steps = 2, density = 0.3, open = 1), c(steps = 5, density = -0.8, open = 0),
                  c(steps = 3, density = 0, open = 0))
subset_features <- function(x, count) lapply(new_items, function(row) row[seq_len(count)])

# 1000 persons, 20 items, three features, tau = 0.3: the regression of b and of ln a, with and without an intercept.
bank <- simulate_bank(1000, 20, c(-0.8, 0.45, 0.3, 0.6), 0.3, 20261029)
items <- calibrated_items(bank$responses)
# 1500 persons, 40 items, two features and no residual between items: tau^2 is at its boundary.
exact <- simulate_bank(1500, 40, c(-0.5, 0.4, 0.5), 0, 20261029)
exact_items <- calibrated_items(exact$responses)
# One feature for 12 items answered by 800 persons: the smallest design with an intercept and a residual.
small <- simulate_bank(800, 12, c(-0.3, 0.35), 0.25, 20261029)
small_items <- calibrated_items(small$responses)

cases <- list(
  regression_case("b on three features, with an intercept", items, bank$features, "b", TRUE, new_items),
  regression_case("ln a on three features", items, bank$features, "a", TRUE, new_items),
  regression_case("b on three features, without an intercept", items, bank$features, "b", FALSE, new_items),
  regression_case("b on two features with no residual between items", exact_items, exact$features, "b", TRUE,
                  subset_features(NULL, 2)),
  regression_case("b on one feature", small_items, small$features, "b", TRUE, subset_features(NULL, 1))
)

write_fixture("explanatory-regression", list(
  source = paste("2PL simulations (set.seed(20261029)) calibrated by mirt: 1000 persons and 20 items with three",
                 "features and tau = 0.3; 1500 persons and 40 items with two features and tau = 0; 800 persons and",
                 "12 items with one feature and tau = 0.25. The regression is the base R maximum of the restricted",
                 "likelihood, and metafor's rma(method = \"REML\") agrees."),
  cases = cases
))
