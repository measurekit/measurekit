# Explanatory item response theory fixtures. The regression of item parameters on item features, estimated in two
# stages: the estimates and standard errors of a bank are mirt's 2PL calibration of simulated responses (SE.type =
# "Richardson", the delta method for b); the regression of them on the features by restricted maximum likelihood is the
# base R maximization of the restricted likelihood, and metafor's rma(method = "REML") agrees. The predictions of new
# items are the mean x' eta and the sd sqrt(tau^2 + x' Cov x), written out. The linear logistic test model: mirt's Rasch
# model with an item design (SE.type = "Oakes", the only one mirt supports with an item design), compared with mirt's
# Rasch model; eRm's conditional maximum likelihood LLTM cross-checks the method.

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

# --- The linear logistic test model ---------------------------------------------------------------------------------
# The reference is the base R maximum of the marginal log-likelihood on the package's rule: theta ~ N(0, sigma^2) on 61
# rectangular points z sigma, an item's logit sigma z - b_j with b = X eta, by BFGS and Newton steps on the exact gradient,
# and the standard errors from the inverse Hessian, by central differences of the gradient. The Rasch model of the
# comparison is the same with X the identity. mirt's Rasch model with `itemdesign` and `item.formula` (SE.type =
# "Oakes", the only one it supports with an item design) cross-checks it: it keeps its nodes on [-6, 6] when it frees
# the variance, so its estimates differ by the quadrature, up to 2e-5 here.

rectangular_rule <- function(points) {
  z <- seq(-6, 6, length.out = points)
  list(z = z, weights = dnorm(z) / sum(dnorm(z)))
}

# The marginal log-likelihood and its gradient with respect to (eta, ln sigma), by Fisher's identity: the posterior mean
# of the complete-data gradient. `x` has NA for missing responses.
lltm_marginal <- function(par, x, design, rule = rectangular_rule(POINTS)) {
  effects <- par[seq_len(ncol(design))]
  sigma <- exp(par[ncol(design) + 1])
  b <- drop(design %*% effects)
  logit <- outer(sigma * rule$z, b, "-")
  correct <- ifelse(is.na(x), 0, x)
  answered <- ifelse(is.na(x), 0, 1)
  joint <- correct %*% t(plogis(logit, log.p = TRUE)) + (answered - correct) %*% t(plogis(-logit, log.p = TRUE)) +
    matrix(log(rule$weights), nrow(x), length(rule$z), byrow = TRUE)
  largest <- apply(joint, 1, max)
  posterior <- exp(joint - largest)
  totals <- rowSums(posterior)
  posterior <- posterior / totals
  residual <- t(posterior) %*% correct - (t(posterior) %*% answered) * plogis(logit)
  # d/d b_j = -sum over nodes of the residual; d/d sigma = sum of z times the residual.
  gradient_b <- -colSums(residual)
  list(logLikelihood = sum(largest + log(totals)),
       gradient = c(drop(crossprod(design, gradient_b)), sigma * sum(rule$z * residual)))
}

lltm_maximum <- function(x, design) {
  objective <- function(par) -lltm_marginal(par, x, design)$logLikelihood
  gradient <- function(par) -lltm_marginal(par, x, design)$gradient
  start <- c(rep(0, ncol(design)), 0)
  par <- optim(start, objective, gradient, method = "BFGS", control = list(reltol = 1e-16, maxit = 10000))$par
  # BFGS stops on the relative change of the objective; Newton steps take the gradient to its rounding level.
  for (step in 1:5) par <- par - solve(jacobian(gradient, par), gradient(par))
  stopifnot(max(abs(gradient(par))) < 1e-8)
  covariance <- solve(jacobian(gradient, par))
  list(par = par, covariance = covariance, logLikelihood = -objective(par))
}

lltm_case <- function(name, dataset, responses, design, intercept) {
  terms <- colnames(design)
  X <- cbind(if (intercept) 1, as.matrix(design))
  colnames(X) <- c(if (intercept) "(intercept)", terms)
  items <- ncol(responses)
  fit <- lltm_maximum(responses, X)
  rasch <- lltm_maximum(responses, diag(items))
  effect_count <- ncol(X)
  size <- effect_count + 1
  statistic <- 2 * (rasch$logLikelihood - fit$logLikelihood)
  persons <- nrow(responses)
  # mirt's Rasch model with the item design: its coefficient is minus a difficulty effect.
  formula <- as.formula(paste("~", if (intercept) "1 +" else "0 +", paste(terms, collapse = " + ")))
  mirt_fit <- mirt(as.data.frame(responses), 1, itemtype = "Rasch", item.formula = formula, itemdesign = design,
                   SE = TRUE, SE.type = "Oakes", TOL = TOL, quadpts = POINTS, verbose = FALSE,
                   technical = list(NCYCLES = 1e5))
  mirt_rasch <- mirt(as.data.frame(responses), 1, itemtype = "Rasch", TOL = TOL, quadpts = POINTS, verbose = FALSE)
  stopifnot(extract.mirt(mirt_fit, "converged"), extract.mirt(mirt_rasch, "converged"))
  mirt_effects <- -coef(mirt_fit)[[1]][1, seq_len(effect_count)]
  mirt_errors <- sqrt(diag(vcov(mirt_fit)))[seq_len(effect_count)]
  mirt_statistic <- 2 * (as.numeric(logLik(mirt_rasch)) - as.numeric(logLik(mirt_fit)))
  stopifnot(
    max(abs(mirt_effects - fit$par[seq_len(effect_count)])) < 1e-4,
    max(abs(mirt_errors / sqrt(diag(fit$covariance))[seq_len(effect_count)] - 1)) < 1e-3,
    abs(sqrt(coef(mirt_fit)$GroupPars[1, "COV_11"]) - exp(fit$par[size])) < 1e-4,
    abs(as.numeric(logLik(mirt_fit)) - fit$logLikelihood) < 5e-4,
    abs(mirt_statistic - statistic) < 1e-3
  )
  message(name, ": chi-square ", statistic, " on ", items - effect_count, " df; mirt ", mirt_statistic)
  list(
    name = name, dataset = dataset, intercept = intercept,
    effects = lapply(seq_len(effect_count), function(j) {
      list(name = colnames(X)[j], estimate = fit$par[j], standardError = sqrt(fit$covariance[j, j]))
    }),
    sd = exp(fit$par[size]),
    difficulties = drop(X %*% fit$par[seq_len(effect_count)]),
    logLikelihood = fit$logLikelihood, aic = -2 * fit$logLikelihood + 2 * size,
    bic = -2 * fit$logLikelihood + size * log(persons),
    raschLogLikelihood = rasch$logLikelihood, chiSquare = statistic, degreesOfFreedom = items - effect_count,
    significance = pchisq(statistic, items - effect_count, lower.tail = FALSE)
  )
}

lltm_features <- function(items, seed) {
  set.seed(seed)
  data.frame(steps = round(runif(items, 1, 6)), density = round(rnorm(items), 2), open = rbinom(items, 1, 0.4),
             row.names = sprintf("q%02d", seq_len(items)))
}

lltm_responses <- function(persons, difficulty, spread, seed) {
  set.seed(seed)
  theta <- rnorm(persons, 0, spread)
  x <- matrix(rbinom(persons * length(difficulty), 1, plogis(outer(theta, difficulty, "-"))), persons,
              length(difficulty))
  colnames(x) <- sprintf("q%02d", seq_along(difficulty))
  x
}

# 1000 persons, 20 items, three features: the LLTM holds exactly (tau = 0), and with a residual of 0.3 between items.
lltm_design <- lltm_features(20, 20261029)
exact_difficulty <- drop(as.matrix(cbind(1, lltm_design)) %*% c(-0.5, 0.35, 0.3, 0.5))
set.seed(20261030)
noisy_difficulty <- exact_difficulty + rnorm(20, 0, 0.3)
exact_responses <- lltm_responses(1000, exact_difficulty, 1.2, 20261029)
noisy_responses <- lltm_responses(1000, noisy_difficulty, 1.2, 20261029)
# 8% of the responses of the exact data missing at random.
set.seed(20261029)
missing_responses <- exact_responses
missing_responses[matrix(runif(length(exact_responses)) < 0.08, nrow(exact_responses))] <- NA

# eRm's lltmdat1: 100 persons answering 15 items at two time points, 30 virtual items, with a difficulty of each item
# and a shift at the second time point. The cross-check is eRm's conditional maximum likelihood estimate of the shift.
data(lltmdat1, package = "eRm")
erm_ids <- sprintf("%s.t%d", sprintf("I%02d", rep(1:15, 2)), rep(1:2, each = 15))
erm_responses <- as.matrix(lltmdat1)
colnames(erm_responses) <- erm_ids
erm_design <- as.data.frame(cbind(sapply(1:15, function(i) as.numeric(rep(1:15, 2) == i)), time2 = rep(0:1, each = 15)))
colnames(erm_design) <- c(sprintf("I%02d", 1:15), "time2")
rownames(erm_design) <- erm_ids
erm_lltm <- eRm::LLTM(lltmdat1, mpoints = 2)
erm_time <- unname(erm_lltm$etapar[15])

datasets <- list(
  list(name = "simulated exact LLTM", itemIds = colnames(exact_responses), responses = unname(exact_responses)),
  list(name = "simulated LLTM with a residual of 0.3", itemIds = colnames(noisy_responses),
       responses = unname(noisy_responses)),
  list(name = "simulated exact LLTM with 8% missing", itemIds = colnames(missing_responses),
       responses = unname(missing_responses)),
  list(name = "eRm lltmdat1", itemIds = erm_ids, responses = unname(erm_responses))
)
features_of <- function(design) setNames(lapply(seq_len(nrow(design)), function(i) as.list(design[i, ])), rownames(design))

cases <- list(
  lltm_case("exact LLTM, three features with an intercept", 0, exact_responses, lltm_design, TRUE),
  lltm_case("LLTM of difficulties with a residual", 1, noisy_responses, lltm_design, TRUE),
  lltm_case("exact LLTM, three features without an intercept", 0, exact_responses, lltm_design, FALSE),
  lltm_case("exact LLTM with missing responses", 2, missing_responses, lltm_design, TRUE),
  lltm_case("difficulty of each item and a shift at the second time point", 3, erm_responses, erm_design, FALSE)
)
for (i in c(1, 2, 3, 4)) cases[[i]]$features <- features_of(lltm_design)
cases[[5]]$features <- features_of(erm_design)
# eRm's conditional maximum likelihood shift has the opposite sign, as eRm's parameters are easiness; the cross-check is
# that it is within 0.05 of the marginal maximum likelihood estimate.
shift <- cases[[5]]$effects[[16]]$estimate
message("time shift: MML ", shift, ", eRm CML ", erm_time)
stopifnot(abs(shift + erm_time) < 0.05)

write_fixture("explanatory-lltm", list(
  source = paste("Simulated LLTMs (set.seed(20261029)): 1000 persons, 20 items, three features, theta ~ N(0, 1.2^2),",
                 "exact and with a residual of 0.3 between items, and with 8% of the responses missing; eRm's lltmdat1",
                 "as 30 items. Base R maxima of the marginal likelihood on the package's rule; mirt's Rasch model with an item",
                 "design (Oakes standard errors) and its Rasch model cross-check them."),
  datasets = datasets,
  cases = cases
))
