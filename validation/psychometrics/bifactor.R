# Bifactor and two-tier fixtures: calibration of the 2PL, 3PL and graded models by mirt's bfactor, with standard errors,
# items without a specific dimension, missing responses and long groups, and the truth of seeded simulations for the
# recovery of slopes. The data are simulated, each case with its own seed. mirt runs with a tight tolerance so that its
# estimates are the optimum; each case records the cycles it took, and whether it stopped before the limit. On flat
# likelihoods, such as the asymptotes of the 3PL, mirt's EM does not reach the tolerance in any number of cycles, and its
# estimates are then short of the optimum.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages({
  library(mirt)
  library(mvtnorm)
})

TOL <- 1e-10
NCYCLES <- 5000
item_ids <- function(n) paste0("item", seq_len(n))
SEED <- 20261101
# Case names on the command line run only those cases: `Rscript validation/psychometrics/bifactor.R m2pl`.
requested <- commandArgs(trailingOnly = TRUE)
selected <- function(name) length(requested) == 0 || name %in% requested

# --- Simulation ----------------------------------------------------------------------------------------------------------

# The thetas of `persons`: the primary dimensions, correlated by `rho` when there are two, then one column for each
# specific dimension, all standard normal.
simulate_thetas <- function(persons, primary, specific, rho) {
  general <- if (primary == 2) rmvnorm(persons, sigma = matrix(c(1, rho, rho, 1), 2)) else matrix(rnorm(persons), persons, 1)
  cbind(general, matrix(rnorm(persons * specific), persons, specific))
}

# Slopes of items by dimension: `primary_of` gives each item's primary dimension and `specific_of` its specific one, NA
# for none. The general slopes are in 0.8 to 1.8, the specific ones in 0.5 to 1.5, rounded to 3 digits.
simulate_slopes <- function(primary_of, specific_of, primary, specific) {
  slopes <- matrix(0, length(primary_of), primary + specific)
  for (i in seq_along(primary_of)) {
    slopes[i, primary_of[i]] <- round(runif(1, 0.8, 1.8), 3)
    if (!is.na(specific_of[i])) slopes[i, primary + specific_of[i]] <- round(runif(1, 0.5, 1.5), 3)
  }
  slopes
}

# Dichotomous responses of slopes and intercepts, P = c + (1 - c) logistic(theta a + d).
dichotomous <- function(theta, slopes, intercepts, guessing = 0) {
  eta <- theta %*% t(slopes) + rep(intercepts, each = nrow(theta))
  p <- rep(guessing, each = nrow(theta)) + (1 - rep(guessing, each = nrow(theta))) * plogis(eta)
  matrix(rbinom(length(p), 1, p), nrow(theta), nrow(slopes))
}

# Graded responses with three decreasing intercepts for each item around its location.
graded_responses <- function(theta, slopes, locations) {
  persons <- nrow(theta)
  intercepts <- t(sapply(locations, function(m) sort(round(m + c(-1, 0, 1) + rnorm(3, 0, 0.2), 3), decreasing = TRUE)))
  eta <- theta %*% t(slopes)
  uniform <- matrix(runif(persons * nrow(slopes)), persons, nrow(slopes))
  responses <- matrix(0L, persons, nrow(slopes))
  for (k in 1:3) responses <- responses + (uniform < plogis(eta + rep(intercepts[, k], each = persons)))
  list(responses = matrix(as.integer(responses), persons, nrow(slopes)), intercepts = intercepts)
}

# --- mirt's results as measurekit's --------------------------------------------------------------------------------------

# Standard errors of the estimated parameters of each item by name, as measurekit names them. The asymptote g is estimated
# on the logit scale, so its error is the delta method's.
mirt_standard_errors <- function(fit, dimensions, graded, guessing) {
  tables <- coef(fit, printSE = TRUE)
  values <- coef(fit, simplify = TRUE)$items
  lapply(rownames(values), function(id) {
    se <- tables[[id]]["SE", ]
    named <- list()
    for (k in seq_len(dimensions)) {
      if (!is.na(se[[paste0("a", k)]])) named[[paste0("a", k)]] <- se[[paste0("a", k)]]
    }
    if (graded) {
      for (name in grep("^d[0-9]+$", names(se), value = TRUE)) named[[sub("^d", "intercept", name)]] <- se[[name]]
    } else {
      named$intercept <- se[["d"]]
      if (guessing) {
        g <- values[id, "g"]
        named$c <- g * (1 - g) * se[["logit(g)"]]
      }
    }
    named
  })
}

# The items of a fit as measurekit's: the parameters of each and, with `errors`, the standard errors of the estimated ones.
mirt_items <- function(fit, dimensions, graded, guessing, errors) {
  values <- coef(fit, simplify = TRUE)$items
  standard_errors <- if (errors) mirt_standard_errors(fit, dimensions, graded, guessing) else NULL
  lapply(seq_len(nrow(values)), function(i) {
    v <- values[i, ]
    slopes <- unname(v[paste0("a", seq_len(dimensions))])
    parameters <- if (graded) {
      list(a = I(slopes), intercepts = I(unname(v[grep("^d[0-9]+$", names(v))])))
    } else {
      list(a = I(slopes), intercept = v[["d"]], c = if (guessing) v[["g"]] else 0, d = 1)
    }
    row <- list(id = rownames(values)[i], parameters = parameters)
    if (errors) row$standardErrors <- standard_errors[[i]]
    row
  })
}

# The correlations of all dimensions, with their standard errors where the fit has them and a primary correlation exists.
mirt_correlations <- function(fit, dimensions, primary, errors) {
  correlations <- unname(coef(fit, simplify = TRUE)$cov)
  result <- list(correlations = correlations)
  if (errors && primary == 2) {
    se <- matrix(0, dimensions, dimensions)
    se[1, 2] <- se[2, 1] <- coef(fit, printSE = TRUE)$GroupPars["SE", "COV_21"]
    result$correlationStandardErrors <- se
  }
  result
}

# mirt's standardized loadings and communalities, from summary(): loadings (a / 1.702) / sqrt(1 + sum((a / 1.702)^2)) and
# their sum of squares, for the dimensions in the order of the slopes.
mirt_standardized <- function(fit) {
  standardized <- summary(fit, verbose = FALSE)
  list(standardized = list(loadings = unname(standardized$rotF), communalities = as.numeric(standardized$h2)))
}

# A case of a fixture: the data and structure measurekit gets, and everything mirt returns for them.
case_of <- function(responses, itemtype, primary_of, specific_of, primary, specific, points, fit, graded = FALSE,
                    guessing = FALSE, errors = FALSE, extra = list()) {
  dimensions <- primary + specific
  item_names <- item_ids(ncol(responses))
  primary_names <- if (primary == 2) c("g1", "g2") else "general"
  specific_names <- paste0("s", seq_len(specific))
  c(list(
    model = itemtype,
    points = points,
    dimensions = c(primary_names, specific_names),
    itemIds = item_names,
    responses = unname(responses),
    specific = setNames(lapply(specific_of, function(s) if (is.na(s)) NULL else specific_names[s]), item_names),
    primary = setNames(lapply(primary_of, function(p) I(primary_names[p])), item_names),
    items = mirt_items(fit, dimensions, graded, guessing, errors),
    logLikelihood = fit@Fit$logLik,
    aic = extract.mirt(fit, "AIC"),
    bic = extract.mirt(fit, "BIC"),
    mirt = list(iterations = extract.mirt(fit, "iterations"), converged = extract.mirt(fit, "iterations") < NCYCLES)
  ), mirt_correlations(fit, dimensions, primary, errors), mirt_standardized(fit), extra)
}

# bfactor at the tight tolerance. `specific_of` is the mirt vector, NA for an item without a specific dimension.
fit_bifactor <- function(responses, specific_of, points, itemtype = "2PL", model2 = NULL, SE = FALSE, SE.type = "Oakes") {
  colnames(responses) <- item_ids(ncol(responses))
  args <- list(responses, specific_of, itemtype = itemtype, quadpts = points, TOL = TOL, SE = SE, SE.type = SE.type,
               verbose = FALSE, technical = list(NCYCLES = NCYCLES))
  if (!is.null(model2)) args$model2 <- model2
  do.call(bfactor, args)
}

# --- Cases ---------------------------------------------------------------------------------------------------------------

# A bifactor 2PL of `groups` specific dimensions of `per_group` items: the case of the fixture and the truth.
bifactor_case <- function(seed, persons, groups, per_group, with_errors) {
  set.seed(seed)
  items <- groups * per_group
  specific_of <- rep(seq_len(groups), each = per_group)
  slopes <- simulate_slopes(rep(1, items), specific_of, 1, groups)
  intercepts <- round(rnorm(items), 3)
  responses <- dichotomous(simulate_thetas(persons, 1, groups, 0), slopes, intercepts)
  fit <- fit_bifactor(responses, specific_of, 31, SE = with_errors)
  extra <- list(truth = list(slopes = slopes, intercepts = intercepts))
  if (with_errors) {
    cross_fit <- fit_bifactor(responses, specific_of, 31, SE = TRUE, SE.type = "crossprod")
    extra$crossProduct <- list(items = mirt_items(cross_fit, 1 + groups, FALSE, FALSE, TRUE))
  }
  case_of(responses, "m2pl", rep(1, items), specific_of, 1, groups, 31, fit, errors = with_errors, extra = extra)
}

if (selected("m2pl")) {
  write_fixture("bifactor-m2pl", list(
    source = "Simulations, set.seed(20261101 + 1 and + 2): bifactor 2PLs of 2,000 persons, 12 items in 3 specific dimensions and 40 in 4. The first has standard errors, Oakes' and the cross-product.",
    items12 = bifactor_case(SEED + 1, 2000, 3, 4, TRUE),
    items40 = bifactor_case(SEED + 2, 2000, 4, 10, FALSE)
  ))
}

if (selected("long")) {
  write_fixture("bifactor-long", list(
    source = "Simulation, set.seed(20261101 + 3): a bifactor 2PL of 1,500 persons and 40 items in 2 specific dimensions of 20 items.",
    long = bifactor_case(SEED + 3, 1500, 2, 20, FALSE)
  ))
}

if (selected("two-tier")) {
  set.seed(SEED + 4)
  persons <- 1000
  primary_of <- rep(1:2, each = 8)
  specific_of <- rep(1:4, each = 4)
  slopes <- simulate_slopes(primary_of, specific_of, 2, 4)
  intercepts <- round(rnorm(16), 3)
  responses <- dichotomous(simulate_thetas(persons, 2, 4, 0.5), slopes, intercepts)
  fit <- fit_bifactor(responses, specific_of, 21, model2 = mirt.model("G1 = 1-8\nG2 = 9-16\nCOV = G1*G2"), SE = TRUE)
  write_fixture("bifactor-two-tier", list(
    source = "Simulation, set.seed(20261101 + 4): a two-tier 2PL of 1,000 persons and 16 items, two primary dimensions of 8 items correlated 0.5, each with 2 specific dimensions of 4 items.",
    twoTier = case_of(responses, "m2pl", primary_of, specific_of, 2, 4, 21, fit, errors = TRUE,
                      extra = list(truth = list(slopes = slopes, intercepts = intercepts, correlation = 0.5)))
  ))
}

if (selected("m3pl")) {
  set.seed(SEED + 5)
  persons <- 4000
  specific_of <- rep(1:3, each = 4)
  slopes <- simulate_slopes(rep(1, 12), specific_of, 1, 3)
  intercepts <- round(rnorm(12), 3)
  guessing <- round(runif(12, 0.05, 0.2), 3)
  responses <- dichotomous(simulate_thetas(persons, 1, 3, 0), slopes, intercepts, guessing)
  # Without a prior the likelihood of the asymptotes is flat and mirt's EM does not stop before its limit of cycles.
  prior <- list(type = "logit-normal", mean = qlogis(0.15), sd = 0.5)
  model2 <- mirt.model(sprintf("G = 1-12\nPRIOR = (1-12, g, norm, %.17g, %.17g)", prior$mean, prior$sd))
  fit <- fit_bifactor(responses, specific_of, 31, itemtype = "3PL", model2 = model2, SE = TRUE)
  write_fixture("bifactor-m3pl", list(
    source = "Simulation, set.seed(20261101 + 5): a bifactor 3PL of 4,000 persons and 12 items in 3 specific dimensions, c in 0.05 to 0.2, with a normal prior on logit c.",
    m3pl = case_of(responses, "m3pl", rep(1, 12), specific_of, 1, 3, 31, fit, guessing = TRUE, errors = TRUE,
                   extra = list(priors = list(c = prior), truth = list(slopes = slopes, intercepts = intercepts, guessing = guessing)))
  ))
}

if (selected("graded")) {
  set.seed(SEED + 6)
  persons <- 2000
  specific_of <- rep(1:3, each = 4)
  slopes <- simulate_slopes(rep(1, 12), specific_of, 1, 3)
  simulated <- graded_responses(simulate_thetas(persons, 1, 3, 0), slopes, round(rnorm(12), 3))
  fit <- fit_bifactor(simulated$responses, specific_of, 31, itemtype = "graded", SE = TRUE)
  write_fixture("bifactor-graded", list(
    source = "Simulation, set.seed(20261101 + 6): a bifactor graded model of 2,000 persons and 12 items of 4 categories in 3 specific dimensions.",
    graded = case_of(simulated$responses, "mgrm", rep(1, 12), specific_of, 1, 3, 31, fit, graded = TRUE, errors = TRUE,
                     extra = list(truth = list(slopes = slopes, intercepts = simulated$intercepts)))
  ))
}

if (selected("incomplete")) {
  set.seed(SEED + 7)
  persons <- 1500
  # Items 5 and 10 have no specific dimension.
  specific_of <- c(1, 1, 1, 1, NA, 2, 2, 2, 2, NA, 3, 3, 3, 3)
  items <- length(specific_of)
  slopes <- simulate_slopes(rep(1, items), specific_of, 1, 3)
  intercepts <- round(rnorm(items), 3)
  complete <- dichotomous(simulate_thetas(persons, 1, 3, 0), slopes, intercepts)
  ungrouped <- fit_bifactor(complete, specific_of, 31)
  # One tenth of the responses are missing at random, and a fifth of the persons skip the items of a specific dimension.
  missing <- complete
  missing[matrix(runif(length(missing)) < 0.1, nrow(missing))] <- NA
  skipped <- sample(persons, persons / 5)
  missing[skipped, which(specific_of == 3)] <- NA
  incomplete <- fit_bifactor(missing, specific_of, 31)
  write_fixture("bifactor-incomplete", list(
    source = "Simulation, set.seed(20261101 + 7): a bifactor 2PL of 1,500 persons and 14 items in 3 specific dimensions of 4 items, two other items having none; and the same responses with a tenth missing at random and a fifth of the persons not answering the items of one specific dimension.",
    ungrouped = case_of(complete, "m2pl", rep(1, items), specific_of, 1, 3, 31, ungrouped),
    missing = case_of(missing, "m2pl", rep(1, items), specific_of, 1, 3, 31, incomplete)
  ))
}

if (selected("recovery")) {
  set.seed(SEED + 8)
  # 3,000 persons and 80 items: 40 on each of two primary dimensions correlated 0.5, in 16 specific dimensions of 5 items.
  persons <- 3000
  primary_of <- rep(1:2, each = 40)
  specific_of <- rep(1:16, each = 5)
  slopes <- simulate_slopes(primary_of, specific_of, 2, 16)
  intercepts <- round(rnorm(80), 3)
  responses <- dichotomous(simulate_thetas(persons, 2, 16, 0.5), slopes, intercepts)
  colnames(responses) <- item_ids(80)
  write_fixture("bifactor-recovery", list(
    source = "Simulation, set.seed(20261101 + 8): a two-tier 2PL of 3,000 persons and 80 items, 40 on each of two primary dimensions correlated 0.5, in 16 specific dimensions of 5 items. Only the truth: recovery is compared with it.",
    itemIds = item_ids(80),
    responses = unname(responses),
    specific = setNames(lapply(specific_of, function(s) paste0("s", s)), item_ids(80)),
    primary = setNames(lapply(primary_of, function(p) I(paste0("g", p))), item_ids(80)),
    truth = list(slopes = slopes, intercepts = intercepts, correlation = 0.5)
  ))
}
