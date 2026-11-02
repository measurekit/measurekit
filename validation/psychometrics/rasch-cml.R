# Conditional maximum likelihood fixtures (irt-conditional.json). References: eRm's RM, PCM and RSM with sum0 = TRUE,
# their parameters, conditional log-likelihoods and the covariance of their parameters, moved to measurekit's
# normalization; and base R's conditional likelihood of a small mixed test, by every response pattern, maximized by
# optim and differentiated by optimHess, which is the reference for polytomous items with missing responses, as eRm's
# PCM does not maximize that likelihood there. Data: eRm's raschdat1, complete and with 2% of the responses missing,
# pcmdat2, pcmdat and rsmdat, and a seeded simulation.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages(library(eRm))

set.seed(20261102)
MISSING_SHARE <- 0.02

rows_of <- function(data) unname(lapply(seq_len(nrow(data)), function(i) as.numeric(data[i, ])))

# measurekit's normalization: every threshold of every item sums to 0. CML identifies the thresholds up to a common
# shift, so measurekit's are eRm's less their mean, and their covariance is eRm's projected along that shift.
normalized <- function(thresholds, covariance) {
  n <- length(thresholds)
  projection <- diag(n) - matrix(1 / n, n, n)
  list(thresholds = drop(projection %*% thresholds), covariance = projection %*% covariance %*% t(projection))
}

# The thresholds of an eRm model and their covariance: betapar are the item-category easiness parameters, so the
# thresholds of an item are minus the differences of its betapar, from 0 for category 0; the covariance of betapar is
# W H⁻¹ Wᵀ, whose diagonal is eRm's se.beta.
erm_thresholds <- function(model, categories) {
  covariance <- model$W %*% solve(model$hessian) %*% t(model$W)
  stopifnot(max(abs(sqrt(diag(covariance)) - model$se.beta)) < 1e-12)
  n <- length(model$betapar)
  difference <- matrix(0, n, n)
  start <- 0
  for (m in categories) {
    for (k in seq_len(m)) {
      difference[start + k, start + k] <- -1
      if (k > 1) difference[start + k, start + k - 1] <- 1
    }
    start <- start + m
  }
  normalized(drop(difference %*% model$betapar), difference %*% covariance %*% t(difference))
}

# Splits a vector over the items, `categories` entries each.
by_item <- function(values, categories) unname(split(values, rep(seq_along(categories), categories)))

erm_case <- function(name, data, model) {
  categories <- apply(data, 2, max, na.rm = TRUE)
  fit <- switch(model, rasch = RM(data, sum0 = TRUE), "partial-credit" = PCM(data, sum0 = TRUE),
                "rating-scale" = RSM(data, sum0 = TRUE))
  result <- erm_thresholds(fit, categories)
  se <- sqrt(diag(result$covariance))
  entry <- list(name = name, model = model, responses = rows_of(data), logLikelihood = fit$loglik,
                parameters = fit$npar, thresholds = by_item(result$thresholds, categories),
                standardErrors = by_item(se, categories))
  if (model == "rating-scale") {
    # The location b_i is the mean of item i's thresholds and the steps are item 1's thresholds less its location:
    # linear in the thresholds, so their covariance follows.
    m <- categories[[1]]
    items <- length(categories)
    map <- matrix(0, items + m, items * m)
    for (i in seq_len(items)) map[i, ((i - 1) * m + 1):(i * m)] <- 1 / m
    for (k in seq_len(m)) map[items + k, ] <- -map[1, ]
    for (k in seq_len(m)) map[items + k, k] <- map[items + k, k] + 1
    values <- drop(map %*% result$thresholds)
    covariance <- map %*% result$covariance %*% t(map)
    entry$locations <- values[seq_len(items)]
    entry$steps <- values[items + seq_len(m)]
    entry$locationErrors <- sqrt(diag(covariance))[seq_len(items)]
    entry$stepErrors <- sqrt(diag(covariance))[items + seq_len(m)]
  }
  entry
}

# --- Base R: the conditional likelihood of a small test, by every pattern -------------------------------------------

# Items of 2, 2, 2, 3 and 4 categories; 400 persons; 5% of the responses missing.
small_categories <- c(1, 1, 1, 2, 3)
small_thresholds <- list(-1, 0, 0.8, c(-0.6, 0.4), c(-1, 0, 1.2))
theta <- rnorm(400)
small_data <- sapply(seq_along(small_categories), function(i) {
  tau <- small_thresholds[[i]]
  vapply(theta, function(t) {
    weights <- exp(c(0, cumsum(t - tau)))
    sample(0:length(tau), 1, prob = weights / sum(weights))
  }, 0)
})
small_data[matrix(runif(length(small_data)) < 0.05, nrow(small_data))] <- NA
colnames(small_data) <- paste0("item", seq_along(small_categories))

# -ln L of the free thresholds, the last threshold minus the sum of the others. Each person's responses are compared
# with every pattern of the items the person answered with the same total.
conditional_loglik <- function(free, data, categories) {
  thresholds <- c(free, -sum(free))
  eta <- by_item(thresholds, categories)
  eta <- lapply(eta, function(tau) c(0, cumsum(tau)))
  total <- 0
  for (p in seq_len(nrow(data))) {
    answered <- which(!is.na(data[p, ]))
    if (length(answered) < 2) next
    x <- data[p, answered]
    r <- sum(x)
    if (r == 0 || r == sum(categories[answered])) next
    patterns <- as.matrix(expand.grid(lapply(categories[answered], function(m) 0:m)))
    patterns <- patterns[rowSums(patterns) == r, , drop = FALSE]
    weight <- function(pattern) exp(-sum(vapply(seq_along(answered), function(j) eta[[answered[j]]][pattern[j] + 1], 0)))
    total <- total + log(weight(x)) - log(sum(apply(patterns, 1, weight)))
  }
  -total
}

base_case <- function() {
  n <- sum(small_categories)
  fit <- optim(rep(0, n - 1), conditional_loglik, data = small_data, categories = small_categories,
               method = "BFGS", control = list(reltol = 1e-15, maxit = 1000))
  hessian <- optimHess(fit$par, conditional_loglik, data = small_data, categories = small_categories)
  free_covariance <- solve(hessian)
  map <- rbind(diag(n - 1), rep(-1, n - 1))
  covariance <- map %*% free_covariance %*% t(map)
  thresholds <- c(fit$par, -sum(fit$par))
  # eRm's PCM agrees with the write-up on complete responses, and on missing ones of dichotomous items, but not on
  # missing ones of polytomous items: its estimates do not maximize the conditional likelihood of each person's answered
  # items, and the log-likelihood it reports is not that likelihood at them.
  complete <- small_data
  complete[is.na(complete)] <- 0
  complete_fit <- optim(rep(0, n - 1), conditional_loglik, data = complete, categories = small_categories,
                        method = "BFGS", control = list(reltol = 1e-15, maxit = 1000))
  stopifnot(abs(PCM(complete, sum0 = TRUE)$loglik + complete_fit$value) < 1e-6)
  erm <- PCM(small_data, sum0 = TRUE)
  erm_thresholds_sparse <- erm_thresholds(erm, small_categories)$thresholds
  at_erm <- -conditional_loglik(head(erm_thresholds_sparse, -1), small_data, small_categories)
  stopifnot(at_erm < -fit$value - 1, abs(erm$loglik - at_erm) > 1)
  message("Small sparse test: the conditional log-likelihood is ", -fit$value, " at its maximum and ", at_erm,
          " at eRm's estimates, where eRm reports ", erm$loglik)
  list(name = "small, sparse", model = "partial-credit", responses = rows_of(small_data), logLikelihood = -fit$value,
       parameters = n - 1, thresholds = by_item(thresholds, small_categories),
       standardErrors = by_item(sqrt(diag(covariance)), small_categories),
       erm = list(logLikelihood = erm$loglik, atItsEstimates = at_erm,
                  thresholds = by_item(erm_thresholds_sparse, small_categories)))
}

sparse_rasch <- as.matrix(raschdat1)
sparse_rasch[matrix(runif(length(sparse_rasch)) < MISSING_SHARE, nrow(sparse_rasch))] <- NA

write_fixture("irt-conditional", list(
  source = paste("eRm's RM, PCM and RSM with sum0 = TRUE, moved to thresholds that sum to 0, on raschdat1, pcmdat2,",
                 "pcmdat and rsmdat; base R's conditional likelihood of a small mixed test by every pattern."),
  cases = list(
    erm_case("raschdat1", as.matrix(raschdat1), "rasch"),
    erm_case("raschdat1, sparse", sparse_rasch, "rasch"),
    erm_case("pcmdat2", as.matrix(pcmdat2), "partial-credit"),
    erm_case("pcmdat", as.matrix(pcmdat), "partial-credit"),
    erm_case("rsmdat", as.matrix(rsmdat), "rating-scale"),
    base_case()
  )
))
