# Survey weights in calibration (van der Linden, 2017, Handbook of IRT vol. 3, chapter 14): each person's terms of the
# marginal likelihood count times the person's weight, the weights rescaled to sum to the persons, as mirt's
# documentation asks of survey.weights. References: mirt's survey.weights, given the rescaled weights; base R's maximum
# of the weighted likelihood in two groups, where mirt's EM stops short; TAM's pweights, which TAM rescales itself; and
# mirt on the data with each person repeated as often as an integer weight says, for the latent regression, where
# mirt's survey.weights do not reach the regression (base R shows it). The fixture holds the raw weights, which
# measurekit rescales.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages({
  library(mirt)
  library(TAM)
})

SEED <- 20261102
TOL <- 1e-8
NODES <- 61

rescaled <- function(weights) weights * length(weights) / sum(weights)

fit_mirt <- function(data, itemtype, weights = NULL, ...) {
  fit <- mirt(data, 1, itemtype = itemtype, survey.weights = weights, TOL = TOL, quadpts = NODES, verbose = FALSE,
              technical = list(NCYCLES = 1e5), ...)
  stopifnot(extract.mirt(fit, "converged"))
  fit
}

simulate_2pl <- function(theta, a, b) sapply(seq_along(a), function(i) rbinom(length(theta), 1, plogis(a[i] * (theta - b[i]))))

# 5% of the responses missing, while every person keeps at least one.
with_missing <- function(data) {
  missing <- matrix(runif(length(data)) < 0.05, nrow(data))
  missing[rowSums(!missing) == 0, 1] <- FALSE
  data[missing] <- NA
  data
}

# Base R: the weighted marginal log-likelihood of 2PL items, Σ_p w_p ln Σ_q π_pq Π_i P_i(x_pi | θ_q), on mirt's 61 nodes
# from -6 to 6, with `log_prior` the persons' log weights π_pq by person, then node.
weighted_log_likelihood <- function(data, weights, a, b, log_prior) {
  nodes <- seq(-6, 6, length.out = NODES)
  logit <- outer(nodes, b, "-") * rep(a, each = NODES)
  answered <- !is.na(data)
  joint <- ((data == 1) & answered) %*% t(plogis(logit, log.p = TRUE)) +
    ((data == 0) & answered) %*% t(plogis(-logit, log.p = TRUE)) + log_prior
  peak <- apply(joint, 1, max)
  sum(weights * (peak + log(rowSums(exp(joint - peak)))))
}

# Each person's log weights of N(mean, sd^2) on the nodes, normalized over them, by person, then node.
normal_log_prior <- function(means, sd) {
  nodes <- seq(-6, 6, length.out = NODES)
  prior <- -0.5 * outer(means, nodes, function(mean, node) ((node - mean) / sd)^2)
  prior - log(rowSums(exp(prior)))
}

# BFGS on the negative log-likelihood `objective` with its gradient by central differences, then Newton steps on
# optimHess, as irt.R finds the maximum of the testlet model.
maximum_of <- function(objective, start) {
  gradient <- function(par) {
    sapply(seq_along(par), function(j) {
      step <- 1e-4 * max(1, abs(par[j]))
      up <- par
      down <- par
      up[j] <- up[j] + step
      down[j] <- down[j] - step
      (objective(up) - objective(down)) / (2 * step)
    })
  }
  par <- optim(start, objective, gradient, method = "BFGS", control = list(reltol = 1e-15, maxit = 5000))$par
  for (step in 1:3) par <- par - solve(optimHess(par, objective, gradient), gradient(par))
  list(par = par, logLikelihood = -objective(par), gradient = gradient(par))
}

# The 2PL items of a fit in IRT form, with the standard errors of a and b by the delta method when it has them.
logistic_items <- function(fit) {
  coefficients <- coef(fit, IRTpars = TRUE, printSE = TRUE)
  unname(lapply(head(coefficients, -1), function(x) {
    item <- list(a = x["par", "a"], b = x["par", "b"])
    if ("SE" %in% rownames(x)) item$standardErrors <- list(a = x["SE", "a"], b = x["SE", "b"])
    item
  }))
}

# --- 2PL: 1,000 persons, 12 items, log-normal weights, both kinds of standard errors -----------------------------------
set.seed(SEED)
persons <- 1000
a <- round(runif(12, 0.8, 1.8), 2)
b <- round(rnorm(12), 2)
two_pl_data <- with_missing(simulate_2pl(rnorm(persons), a, b))
colnames(two_pl_data) <- sprintf("q%02d", seq_len(ncol(two_pl_data)))
two_pl_weights <- round(exp(rnorm(persons, 0, 0.6)), 4)
hessian_fit <- fit_mirt(two_pl_data, "2PL", rescaled(two_pl_weights), SE = TRUE, SE.type = "Richardson")
cross_product_fit <- fit_mirt(two_pl_data, "2PL", rescaled(two_pl_weights), SE = TRUE, SE.type = "crossprod")
two_pl <- list(
  description = "1000 persons, 12 2PL items, 5% missing, log-normal weights: mirt's Richardson and crossprod errors",
  itemIds = colnames(two_pl_data), responses = unname(two_pl_data), weights = two_pl_weights,
  items = lapply(logistic_items(hessian_fit), function(item) item[c("a", "b")]),
  hessianErrors = lapply(logistic_items(hessian_fit), `[[`, "standardErrors"),
  crossProductErrors = lapply(logistic_items(cross_product_fit), `[[`, "standardErrors"),
  logLikelihood = extract.mirt(hessian_fit, "logLik"), bic = extract.mirt(hessian_fit, "BIC")
)

# mirt honors survey.weights where it is the reference: integer weights give the fit of the repeated persons, its
# log-likelihood scaled by N / Σw and its standard errors by √(Σw / N).
integer_weights <- 1 + (seq_len(persons) %% 3)
repeated <- rep(seq_len(persons), integer_weights)
weighted_check <- fit_mirt(two_pl_data, "2PL", rescaled(integer_weights), SE = TRUE, SE.type = "crossprod")
expanded_check <- fit_mirt(two_pl_data[repeated, ], "2PL", SE = TRUE, SE.type = "crossprod")
stopifnot(max(abs(coef(weighted_check, simplify = TRUE)$items - coef(expanded_check, simplify = TRUE)$items)) < 1e-5,
          abs(extract.mirt(weighted_check, "logLik") * sum(integer_weights) / persons -
                extract.mirt(expanded_check, "logLik")) < 1e-4,
          max(abs(sqrt(diag(vcov(weighted_check))) / sqrt(diag(vcov(expanded_check))) /
                    sqrt(sum(integer_weights) / persons) - 1)) < 1e-4)

# --- Graded response model: 800 persons, 8 items of 4 categories --------------------------------------------------------
set.seed(SEED + 1)
persons <- 800
graded_a <- round(runif(8, 0.8, 2), 2)
graded_theta <- rnorm(persons)
graded_data <- sapply(graded_a, function(slope) findInterval(slope * graded_theta + rlogis(persons), c(-1.2, 0.1, 1.3)))
colnames(graded_data) <- sprintf("g%02d", seq_len(ncol(graded_data)))
graded_weights <- round(exp(rnorm(persons, 0, 0.6)), 4)
graded_fit <- fit_mirt(graded_data, "graded", rescaled(graded_weights))
graded_values <- coef(graded_fit, simplify = TRUE)$items
graded <- list(
  description = "800 persons, 8 graded items of 4 categories, log-normal weights",
  itemIds = colnames(graded_data), responses = unname(graded_data), weights = graded_weights,
  items = lapply(seq_len(nrow(graded_values)), function(i) {
    list(a = unname(graded_values[i, "a1"]), b = I(unname(-graded_values[i, c("d1", "d2", "d3")] / graded_values[i, "a1"])))
  }),
  logLikelihood = extract.mirt(graded_fit, "logLik")
)

# --- Two groups: the reference group N(0, 1), the focal group's mean and SD estimated -----------------------------------
set.seed(SEED + 2)
persons <- 1000
group <- factor(rep(c("reference", "focal"), length.out = persons), levels = c("reference", "focal"))
group_data <- with_missing(simulate_2pl(rnorm(persons, ifelse(group == "focal", -0.5, 0), ifelse(group == "focal", 1.3, 1)),
                                        a[1:10], b[1:10]))
colnames(group_data) <- sprintf("q%02d", seq_len(ncol(group_data)))
group_weights <- round(exp(rnorm(persons, ifelse(group == "focal", 0.3, 0), 0.6)), 4)
group_fit <- multipleGroup(group_data, 1, group, invariance = c("free_means", "free_var", colnames(group_data)),
                           survey.weights = rescaled(group_weights), TOL = TOL, quadpts = NODES, verbose = FALSE,
                           technical = list(NCYCLES = 1e5))
stopifnot(extract.mirt(group_fit, "converged"))
# mirt's EM stops short of the maximum in the focal group's SD, by 8e-4 here and by 1e-4 without the weights, so base R
# maximizes the weighted likelihood from mirt's estimates: the items' a and b, the focal mean and ln SD.
focal <- coef(group_fit, simplify = TRUE)$focal
mirt_items <- coef(group_fit, IRTpars = TRUE, simplify = TRUE)$reference$items
is_focal <- group == "focal"
group_objective <- function(par) {
  k <- ncol(group_data)
  log_prior <- normal_log_prior(ifelse(is_focal, par[2 * k + 1], 0), 1)
  log_prior[is_focal, ] <- normal_log_prior(rep(par[2 * k + 1], sum(is_focal)), exp(par[2 * k + 2]))
  -weighted_log_likelihood(group_data, rescaled(group_weights), par[1:k], par[k + 1:k], log_prior)
}
group_maximum <- maximum_of(group_objective, c(unname(mirt_items[, "a"]), unname(mirt_items[, "b"]),
                                               unname(focal$means[1]), log(sqrt(unname(focal$cov[1, 1])))))
stopifnot(max(abs(group_maximum$gradient)) < 1e-4,
          group_maximum$logLikelihood >= extract.mirt(group_fit, "logLik"),
          abs(group_maximum$logLikelihood - extract.mirt(group_fit, "logLik")) < 1e-3)
k <- ncol(group_data)
groups <- list(
  description = paste("1000 persons in two groups, 10 2PL items, 5% missing, weights heavier in the focal group: base R's",
                      "maximum of the weighted likelihood from mirt's multipleGroup with survey.weights"),
  itemIds = colnames(group_data), responses = unname(group_data), weights = group_weights, groups = as.character(group),
  items = lapply(seq_len(k), function(i) list(a = group_maximum$par[i], b = group_maximum$par[k + i])),
  focal = list(mean = group_maximum$par[2 * k + 1], sd = exp(group_maximum$par[2 * k + 2])),
  logLikelihood = group_maximum$logLikelihood,
  mirt = list(focal = list(mean = unname(focal$means[1]), sd = sqrt(unname(focal$cov[1, 1]))),
              logLikelihood = extract.mirt(group_fit, "logLik"))
)

# --- Rasch model with a latent regression: TAM's pweights ---------------------------------------------------------------
set.seed(SEED + 3)
persons <- 1000
x <- rnorm(persons)
x <- x - mean(x)
rasch_data <- simulate_2pl(0.6 * x + rnorm(persons, 0, 0.8), rep(1, 12), b)
colnames(rasch_data) <- sprintf("q%02d", seq_len(ncol(rasch_data)))
rasch_weights <- round(exp(rnorm(persons, 0, 0.6)), 4)
tam_fit <- tam.mml(rasch_data, Y = data.frame(x = x), pweights = rasch_weights, verbose = FALSE,
                   control = list(convD = 1e-9, conv = 1e-9, maxiter = 1e4))
stopifnot(tam_fit$iter < 1e4)
rasch <- list(
  description = "1000 persons, 12 Rasch items, theta = 0.6 x + e, log-normal weights: TAM's tam.mml(Y, pweights)",
  itemIds = colnames(rasch_data), responses = unname(rasch_data), weights = rasch_weights, covariates = list(x = x),
  tam = list(difficulties = unname(tam_fit$xsi$xsi), coefficient = unname(tam_fit$beta[2, 1]),
             residualVariance = unname(tam_fit$variance[1, 1]))
)

# --- 2PL with a latent regression: integer weights against mirt on the repeated persons --------------------------------
set.seed(SEED + 4)
persons <- 600
x <- rnorm(persons)
x <- x - mean(x)
regression_data <- with_missing(simulate_2pl(0.6 * x + rnorm(persons, 0, 0.8), a[1:10], b[1:10]))
colnames(regression_data) <- sprintf("q%02d", seq_len(ncol(regression_data)))
regression_weights <- sample(1:4, persons, replace = TRUE)
repeated <- rep(seq_len(persons), regression_weights)
expanded_fit <- fit_mirt(regression_data[repeated, ], "2PL", covdata = data.frame(x = x[repeated]), formula = ~ x)
weighted_fit <- fit_mirt(regression_data, "2PL", rescaled(regression_weights), covdata = data.frame(x = x), formula = ~ x)

# Base R's weighted log-likelihood with the prior N(β x_p, 1) of each person: the repeated persons' estimates reach a
# higher value than mirt's with survey.weights, so these do not maximize it: mirt leaves the weights out of the
# regression.
regression_estimates <- function(fit) {
  items <- coef(fit, IRTpars = TRUE, simplify = TRUE)$items
  list(a = unname(items[, "a"]), b = unname(items[, "b"]), beta = unname(coef(fit)$lr.betas["x", 1]))
}
at_expanded <- regression_estimates(expanded_fit)
at_weighted <- regression_estimates(weighted_fit)
weighted_value <- function(estimates) {
  weighted_log_likelihood(regression_data, rescaled(regression_weights), estimates$a, estimates$b,
                          normal_log_prior(estimates$beta * x, 1))
}
stopifnot(abs(weighted_value(at_expanded) - extract.mirt(expanded_fit, "logLik") * persons / sum(regression_weights)) < 1e-6,
          weighted_value(at_expanded) > weighted_value(at_weighted) + 1)
regression <- list(
  description = paste("600 persons, 10 2PL items, theta = 0.6 x + e, 5% missing, integer weights 1 to 4: mirt on the",
                      "persons repeated by their weights; base R's weighted log-likelihood at mirt's estimates with",
                      "survey.weights, which leave the weights out of the regression"),
  itemIds = colnames(regression_data), responses = unname(regression_data), weights = regression_weights,
  covariates = list(x = x),
  items = lapply(seq_along(at_expanded$a), function(i) list(a = at_expanded$a[i], b = at_expanded$b[i])),
  coefficient = at_expanded$beta,
  logLikelihood = weighted_value(at_expanded),
  mirtSurveyWeights = list(coefficient = at_weighted$beta, logLikelihood = weighted_value(at_weighted),
                           largestItemDifference = max(abs(c(at_weighted$a - at_expanded$a, at_weighted$b - at_expanded$b))))
)

write_fixture("irt-person-weights", list(
  source = paste("Simulations from set.seed(20261102) to set.seed(20261106), fitted on 61 rectangular nodes with mirt's",
                 "survey.weights rescaled to the persons, base R's weighted likelihood, TAM's pweights, and mirt on",
                 "persons repeated by integer weights. The weights are raw; measurekit rescales them."),
  cases = list(twoPl = two_pl, graded = graded, groups = groups, rasch = rasch, regression = regression)
))
