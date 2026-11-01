# The linear logistic test model of items with several categories: the partial credit and the rating scale model whose item
# locations are linear in the features. The reference is the base R maximum of the marginal likelihood on the package's
# rule, as for the dichotomous model in explanatory.R. TAM's tam.mml with a design array A, whose quadrature differs, is the
# package of record and cross-checks it: its design array writes the location effects and the thresholds out as the model
# defines them. mirt's item design takes dichotomous items only, and eRm's LPCM with a design matrix diverged.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages({
  library(TAM)
  library(numDeriv)
})

POINTS <- 61 # the package's default quadrature: 61 equally spaced nodes on [-6, 6] for a standard normal

rectangular_rule <- function(points) {
  z <- seq(-6, 6, length.out = points)
  list(z = z, weights = dnorm(z) / sum(dnorm(z)))
}

# --- Simulation ---------------------------------------------------------------------------------------------------------
# Item i has the location x_i' eta, with an intercept, and thresholds tau of its own, or one set for all items, that sum to
# 0; theta ~ N(0, spread^2). Categories follow the partial credit model.

simulate_responses <- function(persons, location, thresholds, spread, seed, missing = 0) {
  set.seed(seed)
  items <- length(location)
  theta <- rnorm(persons, 0, spread)
  x <- sapply(seq_len(items), function(i) {
    tau <- thresholds[[i]]
    vapply(theta, function(t) {
      z <- c(0, cumsum(t - location[i] - tau))
      p <- exp(z - max(z))
      sample(0:length(tau), 1, prob = p / sum(p))
    }, numeric(1))
  })
  colnames(x) <- sprintf("q%02d", seq_len(items))
  if (missing > 0) x[matrix(runif(length(x)) < missing, nrow(x))] <- NA
  x
}

features_of <- function(items, seed) {
  set.seed(seed)
  data.frame(steps = round(runif(items, 1, 6)), density = round(rnorm(items), 2), open = rbinom(items, 1, 0.4),
             row.names = sprintf("q%02d", seq_len(items)))
}

# --- The marginal likelihood in base R ----------------------------------------------------------------------------------
# The parameters are the effects of the design (the intercept first, if any), the free thresholds of each set and ln sigma.
# A set of K categories has K - 2 free thresholds; the last is minus their sum. `set_of` gives the set of each item.

unpack <- function(par, effect_count, categories, set_of) {
  free <- categories - 2
  start <- effect_count + c(0, cumsum(free))
  thresholds <- lapply(seq_along(categories), function(s) {
    tau <- if (free[s] > 0) par[start[s] + seq_len(free[s])] else numeric(0)
    c(tau, -sum(tau))[seq_len(categories[s] - 1)]
  })
  list(effects = par[seq_len(effect_count)], thresholds = thresholds, sigma = exp(par[length(par)]),
       tau_of = thresholds[set_of])
}

# Returns the cumulated step logits of item i at the nodes, one row for each node and one column for each category: the
# sum over the first k steps of (sigma z_q - location - tau_j), the first column 0.
step_logits <- function(location, sigma, tau, rule) {
  cbind(0, sapply(seq_along(tau), function(k) k * (sigma * rule$z - location) - sum(tau[seq_len(k)])))
}

# The marginal log-likelihood, and its gradient by Fisher's identity: the posterior mean of the gradient of the complete-data
# log-likelihood. `x` has NA for missing responses.
marginal <- function(par, x, design, categories, set_of, rule = rectangular_rule(POINTS)) {
  p <- unpack(par, ncol(design), categories, set_of)
  location <- drop(design %*% p$effects)
  items <- ncol(x)
  log_p <- lapply(seq_len(items), function(i) {
    logits <- step_logits(location[i], p$sigma, p$tau_of[[i]], rule)
    peak <- apply(logits, 1, max)
    logits - (peak + log(rowSums(exp(logits - peak))))
  })
  total <- matrix(log(rule$weights), nrow(x), length(rule$z), byrow = TRUE)
  for (i in seq_len(items)) {
    answered <- !is.na(x[, i])
    total[answered, ] <- total[answered, , drop = FALSE] + t(log_p[[i]][, x[answered, i] + 1, drop = FALSE])
  }
  largest <- apply(total, 1, max)
  posterior <- exp(total - largest)
  totals <- rowSums(posterior)
  posterior <- posterior / totals
  free <- categories - 2
  start <- ncol(design) + c(0, cumsum(free))
  gradient <- numeric(length(par))
  for (i in seq_len(items)) {
    answered <- !is.na(x[, i])
    s <- set_of[i]
    steps <- length(p$tau_of[[i]])
    # Expected counts W[q, c] of category c at node q, and the residual R[q] = sum_c c W[q, c] - N[q] E_q[k].
    counts <- sapply(0:steps, function(k) colSums(posterior[answered & x[, i] == k & !is.na(x[, i]), , drop = FALSE]))
    probabilities <- exp(log_p[[i]])
    residual <- drop(counts %*% (0:steps)) - rowSums(counts) * drop(probabilities %*% (0:steps))
    gradient[seq_len(ncol(design))] <- gradient[seq_len(ncol(design))] - design[i, ] * sum(residual)
    gradient[length(par)] <- gradient[length(par)] + p$sigma * sum(rule$z * residual)
    for (j in seq_len(free[s])) {
      # d_kj = 1 for j <= k <= K - 2: the free threshold j is in the sum of the first k, except for the last category.
      d <- as.numeric(j <= 0:steps & 0:steps <= categories[s] - 2)
      gradient[start[s] + j] <- gradient[start[s] + j] - sum(counts %*% d - rowSums(counts) * drop(probabilities %*% d))
    }
  }
  list(logLikelihood = sum(largest + log(totals)), gradient = gradient)
}

# Maximizes it by BFGS, then by Newton steps on the Jacobian of the gradient, to the rounding of the likelihood. A set shared
# by several items adds the gradients of all of them.
maximum <- function(x, design, categories, set_of) {
  size <- ncol(design) + sum(categories - 2) + 1
  objective <- function(par) -marginal(par, x, design, categories, set_of)$logLikelihood
  gradient <- function(par) -marginal(par, x, design, categories, set_of)$gradient
  par <- optim(rep(0, size), objective, gradient, method = "BFGS", control = list(reltol = 1e-16, maxit = 10000))$par
  for (step in 1:6) par <- par - solve(jacobian(gradient, par), gradient(par))
  stopifnot(max(abs(gradient(par))) < 1e-8)
  list(par = par, covariance = solve(jacobian(gradient, par)), logLikelihood = -objective(par))
}

# --- TAM -----------------------------------------------------------------------------------------------------------------
# tam.mml's design array A has, for item i and category k, minus the coefficients of the parameters whose sum is the
# cumulated step difficulty of k: k times the location, and the first k thresholds, the last threshold being minus the
# sum of the free ones, so that category K - 1 has all of them: none. TAM fixes the mean of theta at 0 and estimates its
# variance, as an intercept in the design does.

tam_design <- function(design, categories, set_of) {
  items <- nrow(design)
  effects <- ncol(design)
  free <- categories - 2
  start <- effects + c(0, cumsum(free))
  A <- array(0, c(items, max(categories), effects + sum(free)))
  for (i in seq_len(items)) {
    s <- set_of[i]
    # Categories that an item does not have are NA, as in TAM's own designMatrices.
    if (categories[s] < max(categories)) A[i, (categories[s] + 1):max(categories), ] <- NA
    for (k in seq_len(categories[s] - 1)) {
      A[i, k + 1, seq_len(effects)] <- -k * design[i, ]
      for (j in seq_len(free[s])) if (j <= k && k <= categories[s] - 2) A[i, k + 1, start[s] + j] <- -1
    }
  }
  dimnames(A) <- list(rownames(design), paste0("Cat", seq_len(max(categories)) - 1),
                      c(colnames(design), paste0("tau", rep(seq_along(free), free), ".", unlist(lapply(free, seq_len)))))
  A
}

tam_fit <- function(x, design, categories, set_of) {
  fit <- tam.mml(x, A = tam_design(design, categories, set_of), verbose = FALSE,
                 control = list(nodes = seq(-6, 6, length.out = POINTS), convD = 1e-10, conv = 1e-10, maxiter = 5000,
                                fac.oldxsi = 0))
  # TAM reports no flag: it stopped on its criteria if it took fewer iterations than it was allowed.
  stopifnot(fit$iter < fit$control$maxiter)
  # tam.mml's own errors come from the information of the M-step, which leaves out what the latent variable hides;
  # tam.se differentiates the marginal likelihood.
  errors <- tam.se(fit)$xsi$se
  list(estimates = fit$xsi$xsi, errors = errors, variance = drop(fit$variance), deviance = fit$deviance)
}

# --- A case ---------------------------------------------------------------------------------------------------------------

polytomous_case <- function(name, model, x, features, intercept) {
  items <- ncol(x)
  terms <- colnames(features)
  design <- cbind(if (intercept) 1, as.matrix(features))
  colnames(design) <- c(if (intercept) "(intercept)", terms)
  rownames(design) <- rownames(features)
  observed <- apply(x, 2, function(column) max(column, na.rm = TRUE) + 1)
  categories <- if (model == "rating-scale") max(observed) else observed
  set_of <- if (model == "rating-scale") rep(1, items) else seq_len(items)
  if (model == "rating-scale") stopifnot(all(observed == observed[1]))
  fit <- maximum(x, design, categories, set_of)
  rasch <- maximum(x, diag(items), categories, set_of)
  effect_count <- ncol(design)
  parameter_count <- effect_count + 1 + sum(categories - 2)
  p <- unpack(fit$par, effect_count, categories, set_of)
  statistic <- 2 * (rasch$logLikelihood - fit$logLikelihood)
  persons <- sum(rowSums(!is.na(x)) > 0)
  # TAM's estimates and standard errors of the effects and the standard deviation of theta cross-check the maximum.
  tam <- tam_fit(x, design, categories, set_of)
  # TAM's nodes are fixed on [-6, 6] and the package's follow sigma, so the likelihoods differ by the quadrature, up to 0.02.
  # tam.se reports for each parameter the reciprocal of the root of the diagonal of the information of the marginal
  # likelihood, as if the others were known, so its errors are below the ones that the whole covariance matrix gives, which
  # are the package's, as mirt's Oakes errors are.
  information <- solve(fit$covariance)
  alone <- 1 / sqrt(diag(information))
  stopifnot(
    length(tam$errors) >= effect_count, all(is.finite(tam$errors)),
    max(abs(tam$estimates[seq_len(effect_count)] - p$effects)) < 5e-4,
    max(abs(tam$errors[seq_len(effect_count)] / alone[seq_len(effect_count)] - 1)) < 2e-3,
    abs(sqrt(tam$variance) - p$sigma) < 5e-3,
    abs(tam$deviance / -2 - fit$logLikelihood) < 2e-2
  )
  message(name, ": chi-square ", statistic, " on ", items - effect_count, " df; TAM within ",
          signif(max(abs(tam$estimates[seq_len(effect_count)] - p$effects)), 3), ", errors within ",
          signif(max(abs(tam$errors[seq_len(effect_count)] / alone[seq_len(effect_count)] - 1)), 3))
  steps <- lapply(seq_len(items), function(i) drop(design[i, ] %*% p$effects) + p$tau_of[[i]])
  list(
    name = name, model = model, intercept = intercept,
    effects = lapply(seq_len(effect_count), function(j) {
      list(name = colnames(design)[j], estimate = p$effects[j], standardError = sqrt(fit$covariance[j, j]))
    }),
    sd = p$sigma, steps = lapply(steps, I),
    logLikelihood = fit$logLikelihood, aic = -2 * fit$logLikelihood + 2 * parameter_count,
    bic = -2 * fit$logLikelihood + parameter_count * log(persons),
    raschLogLikelihood = rasch$logLikelihood, chiSquare = statistic, degreesOfFreedom = items - effect_count,
    significance = pchisq(statistic, items - effect_count, lower.tail = FALSE),
    tamEstimates = tam$estimates[seq_len(effect_count)]
  )
}

as_features <- function(features) {
  setNames(lapply(seq_len(nrow(features)), function(i) as.list(features[i, ])), rownames(features))
}

# 2,000 persons answer 8 items of 4 categories whose thresholds differ by item, a partial credit model, with an intercept
# of -0.3, 0.25 a step, 0.3 a unit of density and 0.45 for an open response.
effects <- c(-0.3, 0.25, 0.3, 0.45)
design8 <- features_of(8, 20261029)
location8 <- drop(cbind(1, as.matrix(design8)) %*% effects)
set.seed(20261030)
own_thresholds <- lapply(1:8, function(i) { tau <- c(-0.8, 0.1, 0.7) + rnorm(3, 0, 0.15); tau - mean(tau) })
partial <- simulate_responses(2000, location8, own_thresholds, 1.1, 20261029)

# A rating scale of 5 categories, 10 items, one set of thresholds for all.
design10 <- features_of(10, 20261031)
location10 <- drop(cbind(1, as.matrix(design10)) %*% effects)
scale5 <- simulate_responses(2000, location10, rep(list(c(-1.2, -0.3, 0.4, 1.1)), 10), 1.0, 20261031)

# Partial credit with items of 3 and of 4 categories, 12% of the responses missing at random.
design9 <- features_of(9, 20261032)
location9 <- drop(cbind(1, as.matrix(design9)) %*% effects)
mixed_thresholds <- lapply(1:9, function(i) if (i %% 2 == 0) c(-0.6, 0.6) else c(-0.8, 0.1, 0.7) + c(0.1, -0.2, 0.1) * (i %% 3))
mixed_thresholds <- lapply(mixed_thresholds, function(tau) tau - mean(tau))
mixed <- simulate_responses(2000, location9, mixed_thresholds, 1.1, 20261032, missing = 0.12)

# Features that do not explain the locations, so that the comparison with the Rasch model rejects the LLTM: a residual of
# 0.4 between items.
set.seed(20261033)
noisy_location <- location8 + rnorm(8, 0, 0.4)
residual <- simulate_responses(2000, noisy_location, own_thresholds, 1.1, 20261033)

datasets <- list(
  list(name = "partial credit, 8 items of 4 categories", itemIds = colnames(partial), responses = unname(partial)),
  list(name = "rating scale, 10 items of 5 categories", itemIds = colnames(scale5), responses = unname(scale5)),
  list(name = "partial credit, items of 3 and 4 categories, 12% missing", itemIds = colnames(mixed),
       responses = unname(mixed)),
  list(name = "partial credit with a residual of 0.4 between items", itemIds = colnames(residual),
       responses = unname(residual))
)
cases <- list(
  polytomous_case("partial credit with an intercept", "partial-credit", partial, design8, TRUE),
  polytomous_case("partial credit without an intercept", "partial-credit", partial, design8, FALSE),
  polytomous_case("rating scale with an intercept", "rating-scale", scale5, design10, TRUE),
  polytomous_case("partial credit of items with 3 and 4 categories and missing responses", "partial-credit", mixed,
                  design9, TRUE),
  polytomous_case("partial credit with a residual between items", "partial-credit", residual, design8, TRUE)
)
dataset_of <- c(1, 1, 2, 3, 4)
feature_sets <- list(design8, design8, design10, design9, design8)
for (i in seq_along(cases)) {
  cases[[i]]$dataset <- dataset_of[i] - 1
  cases[[i]]$features <- as_features(feature_sets[[i]])
}

write_fixture("explanatory-lltm-polytomous", list(
  source = paste("Simulated partial credit and rating scale responses (set.seed(20261029) and the four seeds that follow):",
                 "2,000 persons, theta ~ N(0, 1.1^2), effects -0.3, 0.25, 0.3 and 0.45. Base R maxima of the marginal",
                 "likelihood on the package's rule; TAM's tam.mml with a design array cross-checks them. mirt's item design",
                 "takes dichotomous items only."),
  datasets = datasets,
  cases = cases
))
