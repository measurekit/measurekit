# Latent regression: theta ~ N(x'b, s^2) with no intercept, estimated with the items. References: mirt (covdata and
# formula) and TAM (tam.mml with Y). With a free slope per item the scale is fixed by s = 1 in mirt, as in measurekit;
# with a Rasch model, which fixes the slope at 1, mirt and TAM estimate s. The covariates are centered, because the
# regression has no intercept.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages({
  library(mirt)
  library(numDeriv)
  library(TAM)
})

SEED <- 20261029
TOL <- 1e-6
NODES <- 61

as_responses <- function(data) unname(lapply(seq_len(nrow(data)), function(i) as.list(data[i, ])))

simulate_responses <- function(theta, a, b) {
  sapply(seq_along(a), function(i) rbinom(length(theta), 1, plogis(a[i] * (theta - b[i]))))
}

# Fits with mirt's EM at a tight tolerance on the nodes measurekit uses.
fit_lr <- function(data, itemtype, covdata, formula, tolerance = TOL, ...) {
  fit <- mirt(data, 1, itemtype = itemtype, covdata = covdata, formula = formula, TOL = tolerance, quadpts = NODES,
              verbose = FALSE, technical = list(NCYCLES = 1e4), ...)
  stopifnot(extract.mirt(fit, "converged"))
  fit
}

items_of <- function(fit) {
  items <- coef(fit, IRTpars = TRUE, simplify = TRUE)$items
  lapply(seq_len(nrow(items)), function(i) list(a = unname(items[i, "a"]), b = unname(items[i, "b"])))
}

# The coefficients of the covariates by name; the first row of lr.betas is the intercept, fixed at 0. With standard
# errors, coef() returns the estimates and their intervals in a list.
coefficients_of <- function(fit, names) {
  betas <- coef(fit)$lr.betas
  if (is.list(betas)) betas <- betas$betas
  setNames(as.list(unname(betas[names, 1])), names)
}

lr_case <- function(name, model, data, covdata, formula, fit, fixed = list(), extra = list(), items = items_of(fit)) {
  covariates <- as.list(covdata)
  c(list(name = name, model = model, itemIds = colnames(data), responses = unname(data), covariates = covariates,
         fixed = if (length(fixed) > 0) fixed else setNames(list(), character(0)),
         mirt = list(items = items, coefficients = coefficients_of(fit, colnames(covdata)),
                     residualVariance = unname(coef(fit, simplify = TRUE)$cov[1, 1]),
                     logLikelihood = extract.mirt(fit, "logLik"))),
    extra)
}

# --- Free slopes: 2PL, 15 items, 1000 persons, theta = 0.6 x + 0.3 z + e, e ~ N(0, 0.8^2) -----------------------------
set.seed(SEED)
persons <- 1000
n_items <- 15
a <- round(runif(n_items, 0.8, 1.8), 2)
b <- round(rnorm(n_items), 2)
x <- rnorm(persons)
z <- rbinom(persons, 1, 0.4)
x <- x - mean(x)
z <- z - mean(z)
theta <- 0.6 * x + 0.3 * z + rnorm(persons, 0, 0.8)
free_data <- simulate_responses(theta, a, b)
colnames(free_data) <- sprintf("q%02d", seq_len(n_items))
free_data[matrix(runif(length(free_data)) < 0.05, persons)] <- NA
free_covdata <- data.frame(x = x, z = z)
free_fit <- fit_lr(free_data, "2PL", free_covdata, ~ x + z)
free_case <- lr_case("2PL, a continuous and a binary covariate", "2pl", free_data, free_covdata, ~ x + z, free_fit)

# --- Discrete covariates only: two patterns of a binary covariate ---------------------------------------------------------
set.seed(SEED)
persons <- 800
group <- rbinom(persons, 1, 0.5)
group <- group - mean(group)
theta <- 0.8 * group + rnorm(persons)
discrete_data <- simulate_responses(theta, a[1:10], b[1:10])
colnames(discrete_data) <- sprintf("q%02d", 1:10)
discrete_covdata <- data.frame(group = group)
discrete_fit <- fit_lr(discrete_data, "2PL", discrete_covdata, ~ group)
# The standard error of the coefficient is the root of its diagonal entry in the inverse of the observed information:
# the Hessian of the marginal log-likelihood in the items and the coefficient, by numerical differentiation of base R's
# own marginal log-likelihood at mirt's estimates. mirt's covariance matrices (Oakes, Louis, SEM) give 0.0714, 0.0707 and
# 0.0700 for this coefficient, below the 0.0885 of the Hessian.
marginal_log_likelihood <- function(data, covariate, parameters) {
  items <- ncol(data)
  a <- parameters[seq_len(items)]
  b <- parameters[items + seq_len(items)]
  nodes <- seq(-6, 6, length.out = NODES)
  logit <- outer(nodes, b, "-") * rep(a, each = NODES)
  answered <- !is.na(data)
  correct <- (data == 1) & answered
  incorrect <- (data == 0) & answered
  joint <- (correct + 0) %*% t(plogis(logit, log.p = TRUE)) + (incorrect + 0) %*% t(plogis(-logit, log.p = TRUE))
  prior <- -0.5 * outer(parameters[2 * items + 1] * covariate, nodes, function(mean, node) (node - mean)^2)
  prior <- prior - log(rowSums(exp(prior)))
  joint <- joint + prior
  peak <- apply(joint, 1, max)
  sum(peak + log(rowSums(exp(joint - peak))))
}
discrete_items <- coef(discrete_fit, IRTpars = TRUE, simplify = TRUE)$items
discrete_estimates <- c(unname(discrete_items[, "a"]), unname(discrete_items[, "b"]),
                        unname(coefficients_of(discrete_fit, "group")$group))
stopifnot(abs(marginal_log_likelihood(discrete_data, discrete_covdata$group, discrete_estimates) -
                extract.mirt(discrete_fit, "logLik")) < 1e-6)
information <- -numDeriv::hessian(function(p) marginal_log_likelihood(discrete_data, discrete_covdata$group, p),
                                  discrete_estimates)
discrete_error <- sqrt(diag(solve(information)))[length(discrete_estimates)]
discrete_case <- lr_case("2PL, a binary covariate", "2pl", discrete_data, discrete_covdata, ~ group, discrete_fit,
                         extra = list(hessianStandardErrors = list(group = unname(discrete_error))))

# --- Common slope: the Rasch model, whose slope is fixed at 1, so that mirt and TAM estimate s -------------------------
set.seed(SEED)
persons <- 1000
x <- rnorm(persons)
x <- x - mean(x)
theta <- 0.6 * x + rnorm(persons, 0, 0.8)
rasch_data <- simulate_responses(theta, rep(1, n_items), b)
colnames(rasch_data) <- sprintf("q%02d", seq_len(n_items))
rasch_covdata <- data.frame(x = x)
rasch_fit <- fit_lr(rasch_data, "Rasch", rasch_covdata, ~ x, tolerance = 1e-5)
tam_fit <- tam.mml(rasch_data, Y = rasch_covdata, verbose = FALSE, control = list(convD = 1e-9, conv = 1e-9, maxiter = 1e4))
stopifnot(tam_fit$iter < 1e4)
rasch_case <- lr_case("Rasch, one covariate", "1pl", rasch_data, rasch_covdata, ~ x, rasch_fit,
                      extra = list(tam = list(difficulties = unname(tam_fit$xsi$xsi), coefficient = unname(tam_fit$beta[2, 1]),
                                              residualVariance = unname(tam_fit$variance[1, 1]))))
# The two packages agree on the Rasch model.
stopifnot(abs(tam_fit$beta[2, 1] - rasch_case$mirt$coefficients$x) < 2e-3,
          abs(tam_fit$variance[1, 1] - rasch_case$mirt$residualVariance) < 2e-3)

# --- Fixed items: 6 of the 15 2PL items of the first case are anchors, and s is estimated --------------------------------
anchors <- setNames(lapply(1:6, function(i) list(a = a[i], b = b[i], c = 0, d = 1)), colnames(free_data)[1:6])
values <- mirt(free_data, 1, itemtype = "2PL", covdata = free_covdata, formula = ~ x + z, pars = "values")
for (id in names(anchors)) {
  rows <- values$item == id
  values$value[rows & values$name == "a1"] <- anchors[[id]]$a
  values$value[rows & values$name == "d"] <- -anchors[[id]]$a * anchors[[id]]$b
  values$est[rows] <- FALSE
}
values$est[values$name == "COV_11"] <- TRUE
# mirt's EM with fixed items creeps near 1e-6 and stops at its cycle limit, so the tolerance is 1e-5.
anchored_fit <- fit_lr(free_data, "2PL", free_covdata, ~ x + z, tolerance = 1e-5, pars = values)
anchored_case <- lr_case("2PL, 6 fixed items and an estimated residual variance", "2pl", free_data, free_covdata, ~ x + z,
                         anchored_fit, fixed = anchors)

# --- Polytomous: the generalized partial credit model with three categories, free slopes, so that s = 1 -----------------

simulate_gpcm <- function(theta, a, steps) {
  sapply(seq_along(a), function(i) {
    logits <- cbind(0, t(apply(outer(theta, steps[i, ], "-") * a[i], 1, cumsum)))
    probabilities <- exp(logits - apply(logits, 1, max))
    probabilities <- probabilities / rowSums(probabilities)
    rowSums(runif(length(theta)) > t(apply(probabilities, 1, cumsum)))
  })
}

set.seed(SEED)
persons <- 800
gpcm_items <- 10
gpcm_a <- round(runif(gpcm_items, 0.8, 1.8), 2)
gpcm_steps <- t(sapply(round(rnorm(gpcm_items), 2), function(m) round(m + sort(rnorm(2, 0, 0.8)), 2)))
x <- rnorm(persons)
x <- x - mean(x)
theta <- 0.6 * x + rnorm(persons, 0, 0.8)
gpcm_data <- simulate_gpcm(theta, gpcm_a, gpcm_steps)
colnames(gpcm_data) <- sprintf("q%02d", seq_len(gpcm_items))
gpcm_covdata <- data.frame(x = x)
gpcm_fit <- fit_lr(gpcm_data, "gpcm", gpcm_covdata, ~ x)
gpcm_coefficients <- coef(gpcm_fit, IRTpars = TRUE, simplify = TRUE)$items
gpcm_thresholds <- lapply(seq_len(gpcm_items), function(i) {
  list(a = unname(gpcm_coefficients[i, "a"]), b = I(unname(gpcm_coefficients[i, c("b1", "b2")])))
})
gpcm_case <- lr_case("GPCM, one covariate", "gpcm", gpcm_data, gpcm_covdata, ~ x, gpcm_fit, items = gpcm_thresholds)

write_fixture("irt-latent-regression", list(
  source = paste("Simulations, set.seed(20261029), with mirt's EM on 61 nodes: 1000 persons and 15 2PL items,",
                 "theta = 0.6 x + 0.3 z + e with e ~ N(0, 0.64) and 5% of responses missing; 800 persons and 10 items",
                 "with a binary covariate, whose standard error is the numerical Hessian of the marginal",
                 "log-likelihood; 1000 persons and 15 Rasch items, checked against TAM's tam.mml(Y = x); the first",
                 "case with 6 items fixed at their generating values and COV_11 estimated; 800 persons and 10 GPCM items of three",
                 "categories with one covariate."),
  cases = list(free_case, discrete_case, rasch_case, anchored_case),
  polytomous = gpcm_case
))
