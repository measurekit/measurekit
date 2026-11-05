# Limited-information fit fixtures. Reference: mirt::M2(type = "M2*", "C2"), quadpts = 61, on a 12-item dichotomous set
# and a 10-item polytomous set, each calibration at TOL = 1e-8. Where mirt's value cannot be a reference, the script
# writes the statistic out in base R, after checking that the same code reproduces mirt where mirt is right:
# - mirt's Δ leaves out the parameters of an estimated distribution of theta, while its df counts them;
# - mirt's dP for gpcm items has a wrong derivative with respect to the slope, so its M2 and C2 of the GPCM and the PCM
#   with estimated slopes are not the statistic: the script checks both this and its own derivative by central differences;
# - for M2* on polytomous items, mirt's Δ of the independence model has rank J below its columns, and the columns of Q
#   that M2 takes are then an arbitrary part of the complement: its statistic changes with the order of the items;
# - mirt's M2 of 3PL items whose lower asymptote est = FALSE holds is not the statistic that the written-out form gives,
#   which reproduces mirt's 2PL.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages(library(mirt))

TOL <- 1e-8
POINTS <- 61
THETA <- seq(-6, 6, length.out = POINTS)

set.seed(20261102)
ITEMS <- 12
PERSONS <- 2000
slopes <- runif(ITEMS, 0.8, 2)
intercepts <- -slopes * rnorm(ITEMS)
binary <- simdata(matrix(slopes), matrix(intercepts), PERSONS, itemtype = "2PL")
colnames(binary) <- paste0("item", seq_len(ITEMS))

# --- Written out in base R ----------------------------------------------------------------------------------------

# The rows of a statistic. A row lists, by item, the values of a function of the item's score at its categories: the
# score or, for C2, the indicator of a category, then the scores of each pair.
statistic_rows <- function(counts, type) {
  score <- function(i) seq_len(counts[i]) - 1
  row <- function(items, values) setNames(values, items)
  J <- length(counts)
  first <- lapply(seq_len(J), function(i) {
    if (type == "C2") lapply(seq_len(counts[i] - 1), function(k) row(i, list(as.numeric(score(i) == k))))
    else list(row(i, list(score(i))))
  })
  pairs <- lapply(seq_len(J - 1), function(i) lapply((i + 1):J, function(j) row(c(i, j), list(score(i), score(j)))))
  c(unlist(first, recursive = FALSE), unlist(pairs, recursive = FALSE))
}

# E(row | node) at each node; `probabilities` holds each item's categories by node.
conditional_row <- function(row, probabilities) {
  values <- mapply(function(item, h) probabilities[[as.integer(item)]] %*% h, names(row), row, SIMPLIFY = FALSE)
  Reduce(`*`, values)
}

# The moments of the rows and Ξ: two rows integrate the union of their items, each once.
model_moments <- function(rows, probabilities, weights) {
  conditional <- sapply(rows, conditional_row, probabilities = probabilities)
  conditional <- matrix(conditional, length(weights), length(rows))
  expected <- drop(crossprod(weights, conditional))
  size <- length(rows)
  covariance <- matrix(0, size, size)
  for (r in seq_len(size)) {
    for (s in r:size) {
      shared <- intersect(names(rows[[r]]), names(rows[[s]]))
      joint <- c(rows[[r]][setdiff(names(rows[[r]]), shared)], rows[[s]][setdiff(names(rows[[s]]), shared)])
      for (item in shared) joint[[item]] <- rows[[r]][[item]] * rows[[s]][[item]]
      covariance[r, s] <- covariance[s, r] <- sum(weights * conditional_row(joint, probabilities)) - expected[r] * expected[s]
    }
  }
  list(expected = expected, covariance = covariance, conditional = conditional)
}

# The column of Δ of a parameter of one item, given dP(k | node)/dβ of its categories by node.
item_column <- function(rows, probabilities, weights, item, derivative) {
  key <- as.character(item)
  sapply(rows, function(row) {
    if (!key %in% names(row)) return(0)
    others <- row[names(row) != key]
    rest <- if (length(others) == 0) rep(1, length(weights)) else conditional_row(others, probabilities)
    sum(weights * (derivative %*% row[[key]]) * rest)
  })
}

observed_moments <- function(rows, data) {
  sapply(rows, function(row) {
    mean(Reduce(`*`, mapply(function(item, h) h[data[, as.integer(item)] + 1], names(row), row, SIMPLIFY = FALSE)))
  })
}

quadratic_form <- function(residuals, covariance, jacobian, persons) {
  solved <- solve(covariance, residuals)
  directions <- solve(covariance, jacobian)
  projected <- crossprod(jacobian, solved)
  persons * abs(sum(residuals * solved) - sum(projected * solve(crossprod(jacobian, directions), projected)))
}

normal_weights <- function(mean, variance) {
  weights <- dnorm(THETA, mean, sqrt(variance))
  weights / sum(weights)
}

# dw/dμ and dw/dσ of the weights of a normal distribution at the nodes.
weight_derivatives <- function(mean, variance) {
  weights <- normal_weights(mean, variance)
  center <- function(values) weights * (values - sum(weights * values))
  list(mean = center((THETA - mean) / variance), sd = center((THETA - mean)^2 / variance^1.5))
}

# Items by slope and intercept. P(k) ∝ exp(k a θ + d_k), d_0 = 0: the 2PL has K = 2, and the GPCM any K.
exponential_probabilities <- function(slope, intercepts) {
  logits <- outer(THETA, seq_along(intercepts)) * slope
  logits <- cbind(0, sweep(logits, 2, intercepts, `+`))
  exp(logits) / rowSums(exp(logits))
}

# dP_k/da = θ P_k (k - E(x | θ)) and dP_k/dd_m = P_k ([k = m] - P_m).
slope_derivative <- function(probabilities) {
  k <- col(probabilities) - 1
  THETA * probabilities * (k - drop(probabilities %*% (seq_len(ncol(probabilities)) - 1)))
}
intercept_derivative <- function(probabilities, m) {
  probabilities * (as.numeric(col(probabilities) - 1 == m) - probabilities[, m + 1])
}

# The items of a fit of slopes and intercepts, with what the fit estimates of each.
exponential_items <- function(fit) {
  values <- mod2values(fit)
  values <- values[values$item != "GROUP", ]
  lapply(unique(values$item), function(id) {
    rows <- values[values$item == id, ]
    intercept_names <- grep("^d[1-9]?[0-9]*$", rows$name, value = TRUE)
    intercept_names <- setdiff(intercept_names, "d0")
    list(slope = rows$value[rows$name == "a1"], intercepts = rows$value[match(intercept_names, rows$name)],
         slope_free = rows$est[rows$name == "a1"], intercepts_free = rows$est[match(intercept_names, rows$name)])
  })
}

# Returns the parts of the statistic of such a fit: the rows' moments and the columns of Δ by the slope and the intercepts.
exponential_parts <- function(fit, data, type, weights) {
  items <- exponential_items(fit)
  probabilities <- lapply(items, function(item) exponential_probabilities(item$slope, item$intercepts))
  rows <- statistic_rows(sapply(probabilities, ncol), type)
  moments <- model_moments(rows, probabilities, weights)
  column <- function(i, derivative) item_column(rows, probabilities, weights, i, derivative)
  slopes <- lapply(seq_along(items), function(i) {
    if (items[[i]]$slope_free) column(i, slope_derivative(probabilities[[i]])) else NULL
  })
  intercepts <- lapply(seq_along(items), function(i) {
    free <- which(items[[i]]$intercepts_free)
    lapply(free, function(m) column(i, intercept_derivative(probabilities[[i]], m)))
  })
  list(residuals = observed_moments(rows, data) - moments$expected, moments = moments, slopes = slopes,
       intercepts = do.call(cbind, unlist(intercepts, recursive = FALSE)))
}

# --- What mirt reports --------------------------------------------------------------------------------------------

m2_of <- function(fit, type = "M2*", CI = 0.9) {
  result <- M2(fit, type = type, quadpts = POINTS, CI = CI)
  tails <- grep("^RMSEA_", names(result), value = TRUE)
  list(statistic = result$M2, degreesOfFreedom = result$df, significance = result$p, rmsea = result$RMSEA,
       rmseaInterval = c(result[[tails[1]]], result[[tails[2]]]), srmsr = result$SRMSR,
       tli = result$TLI, cfi = result$CFI)
}

fit_mirt <- function(data, model, itemtype, ...) {
  fit <- mirt(data, model, itemtype = itemtype, TOL = TOL, quadpts = POINTS, verbose = FALSE,
              technical = list(NCYCLES = 1e5), ...)
  stopifnot(extract.mirt(fit, "converged"))
  fit
}

equal_slopes <- function(J) mirt.model(sprintf("F = 1-%d\nCONSTRAIN = (1-%d, a1)", J, J))

# The independence model: N Σ r², on J (J - 1) / 2 degrees of freedom.
independence <- function(data) {
  correlations <- cor(data)
  list(statistic = nrow(data) * sum(correlations[lower.tri(correlations)]^2),
       degreesOfFreedom = ncol(data) * (ncol(data) - 1) / 2)
}

# mirt's result with the statistic of the written-out form, and what follows from it.
with_statistic <- function(result, statistic, null, persons) {
  df <- result$degreesOfFreedom
  ratio <- null$statistic / null$degreesOfFreedom
  result$statistic <- statistic
  result$significance <- pchisq(statistic, df, lower.tail = FALSE)
  result$rmsea <- mirt:::rmsea(statistic, df, persons)
  result$rmseaInterval <- mirt:::RMSEA.CI(statistic, df, persons)
  result$tli <- mirt:::tli(statistic, null$statistic, df, null$degreesOfFreedom)
  result$cfi <- mirt:::cfi(statistic, null$statistic, df, null$degreesOfFreedom)
  result
}

# --- The dichotomous set ------------------------------------------------------------------------------------------

two_pl <- fit_mirt(binary, 1, "2PL")
one_pl <- fit_mirt(binary, equal_slopes(ITEMS), "2PL")
rasch <- fit_mirt(binary, 1, "Rasch")
stopifnot(abs(logLik(rasch) - logLik(one_pl)) < 1e-3)

# Anchors at their true parameters, on the standard normal and with an estimated normal distribution.
ANCHORS <- 4
anchored_values <- function(free_group) {
  values <- mirt(binary, 1, itemtype = "2PL", pars = "values")
  for (i in seq_len(ANCHORS)) {
    id <- colnames(binary)[i]
    values$value[values$item == id & values$name == "a1"] <- slopes[i]
    values$value[values$item == id & values$name == "d"] <- intercepts[i]
    values$est[values$item == id] <- FALSE
  }
  if (free_group) values$est[values$name %in% c("MEAN_1", "COV_11")] <- TRUE
  values
}
anchored <- mirt(binary, 1, itemtype = "2PL", pars = anchored_values(FALSE), TOL = 1e-6, quadpts = POINTS, verbose = FALSE)
anchored_normal <- mirt(binary, 1, itemtype = "2PL", pars = anchored_values(TRUE), TOL = 1e-6, quadpts = POINTS,
                        verbose = FALSE, technical = list(NCYCLES = 1e5))

# The written-out form reproduces mirt on the 2PL, where mirt's parts are right.
two_pl_parts <- exponential_parts(two_pl, binary, "M2*", normal_weights(0, 1))
two_pl_written <- quadratic_form(two_pl_parts$residuals, two_pl_parts$moments$covariance,
                                 cbind(do.call(cbind, two_pl_parts$slopes), two_pl_parts$intercepts), PERSONS)
stopifnot(abs(two_pl_written - m2_of(two_pl)$statistic) < 1e-5)

# What mirt's M2 of an estimated distribution is, and the same with the distribution's columns.
with_distribution <- function(fit, mean, variance, distribution) {
  parts <- exponential_parts(fit, binary, "M2*", normal_weights(mean, variance))
  jacobian <- cbind(do.call(cbind, parts$slopes), parts$intercepts)
  derivatives <- weight_derivatives(mean, variance)
  group <- sapply(derivatives[distribution], function(d) drop(crossprod(d, parts$moments$conditional)))
  list(mirt = quadratic_form(parts$residuals, parts$moments$covariance, jacobian, PERSONS),
       written = quadratic_form(parts$residuals, parts$moments$covariance, cbind(jacobian, group), PERSONS))
}

rasch_variance <- mod2values(rasch)$value[mod2values(rasch)$name == "COV_11"]
rasch_written <- with_distribution(rasch, 0, rasch_variance, "sd")
normal_values <- mod2values(anchored_normal)
normal_written <- with_distribution(anchored_normal, normal_values$value[normal_values$name == "MEAN_1"],
                                    normal_values$value[normal_values$name == "COV_11"], c("mean", "sd"))

rasch_m2 <- m2_of(rasch)
normal_m2 <- m2_of(anchored_normal)
stopifnot(abs(rasch_written$mirt - rasch_m2$statistic) < 1e-4, abs(normal_written$mirt - normal_m2$statistic) < 1e-4)
stopifnot(rasch_written$written < rasch_m2$statistic, normal_written$written < normal_m2$statistic)
# The 1PL and the Rasch model with a free variance are one model, and with the column of the variance one statistic.
stopifnot(abs(rasch_written$written - m2_of(one_pl)$statistic) < 1e-3)

null_binary <- mirt(binary, 1, verbose = FALSE, quadpts = 3, technical = list(NULL.MODEL = TRUE))
null_m2 <- M2(null_binary, calcNull = FALSE, quadpts = 2)
binary_independence <- independence(binary)
stopifnot(abs(null_m2$M2 / binary_independence$statistic - 1) < 1e-3,
          null_m2$df == binary_independence$degreesOfFreedom)

anchors <- lapply(seq_len(ANCHORS), function(i) list(a = slopes[i], b = -intercepts[i] / slopes[i], c = 0, d = 1))
names(anchors) <- colnames(binary)[seq_len(ANCHORS)]

binary_cases <- list(
  list(name = "the 2PL", model = "2pl", expected = m2_of(two_pl)),
  list(name = "the 1PL with a common slope", model = "1pl", expected = m2_of(one_pl)),
  list(name = "the Rasch model with a free variance, as the 1PL", model = "1pl", mirtStatistic = rasch_m2$statistic,
       expected = with_statistic(rasch_m2, rasch_written$written, binary_independence, PERSONS)),
  list(name = "the 2PL with anchors", model = "2pl", fixed = anchors, expected = m2_of(anchored)),
  list(name = "the 2PL with anchors and an estimated normal distribution", model = "2pl", fixed = anchors,
       latent = "normal", mirtStatistic = normal_m2$statistic,
       expected = with_statistic(normal_m2, normal_written$written, binary_independence, PERSONS))
)

# --- The polytomous set -------------------------------------------------------------------------------------------

set.seed(20261103)
POLYTOMOUS_ITEMS <- 10
POLYTOMOUS_PERSONS <- 1500
graded_data <- simdata(
  matrix(runif(POLYTOMOUS_ITEMS, 0.8, 2)),
  t(apply(matrix(rnorm(POLYTOMOUS_ITEMS * 3), POLYTOMOUS_ITEMS), 1, sort, decreasing = TRUE)),
  POLYTOMOUS_PERSONS, itemtype = "graded"
)
colnames(graded_data) <- paste0("item", seq_len(POLYTOMOUS_ITEMS))

rsm_values <- mirt(graded_data, 1, itemtype = "rsm", pars = "values")
rsm_values$est[rsm_values$name == "a1"] <- TRUE
rsm_values$est[rsm_values$name == "COV_11"] <- FALSE
rsm_values$value[rsm_values$name == "COV_11"] <- 1
polytomous_fits <- list(
  grm = fit_mirt(graded_data, 1, "graded"),
  gpcm = fit_mirt(graded_data, 1, "gpcm"),
  pcm = fit_mirt(graded_data, equal_slopes(POLYTOMOUS_ITEMS), "gpcm"),
  rsm = fit_mirt(graded_data, 1, "rsm", pars = rsm_values, constrain = list(rsm_values$parnum[rsm_values$name == "a1"]))
)

# mirt's derivative with respect to a gpcm slope, against central differences of its own probabilities; ours agree.
gpcm_item <- extract.item(polytomous_fits$gpcm, 1)
gpcm_theta <- matrix(THETA)
expected_of <- function(item) expected.item(item, gpcm_theta, min = 0)
shifted <- function(step) {
  item <- gpcm_item
  item@par[1] <- item@par[1] + step
  item
}
central <- (expected_of(shifted(1e-5)) - expected_of(shifted(-1e-5))) / 2e-5
gpcm_coefficients <- exponential_items(polytomous_fits$gpcm)[[1]]
gpcm_probabilities <- exponential_probabilities(gpcm_coefficients$slope, gpcm_coefficients$intercepts)
stopifnot(max(abs(drop(slope_derivative(gpcm_probabilities) %*% 0:3) - central)) < 1e-6,
          max(abs(mirt:::dP(gpcm_item, gpcm_theta)[, 1] - central)) > 0.1)

# A statistic of the GPCM or the PCM, written out: the slopes each a column, or summed into one.
written_partial_credit <- function(fit, type, shared) {
  parts <- exponential_parts(fit, graded_data, type, normal_weights(0, 1))
  columns <- do.call(cbind, parts$slopes)
  jacobian <- cbind(if (shared) rowSums(columns) else columns, parts$intercepts)
  quadratic_form(parts$residuals, parts$moments$covariance, jacobian, POLYTOMOUS_PERSONS)
}

# mirt's M2* of the independence model is not a function of the data alone: the order of the items changes it.
null_polytomous <- function(data) {
  fit <- mirt(data, 1, verbose = FALSE, quadpts = 3, technical = list(NULL.MODEL = TRUE))
  M2(fit, calcNull = FALSE, type = "M2*", quadpts = 2)$M2
}
reordered <- rev(seq_len(POLYTOMOUS_ITEMS))
stopifnot(abs(null_polytomous(graded_data) - null_polytomous(graded_data[, reordered])) > 1)

# The whole complement of Δ, whose rank is J, gives the closed form. A category's shift of 1 stands for a shift of the mean.
polytomous_independence <- independence(graded_data)
shares <- lapply(seq_len(POLYTOMOUS_ITEMS), function(i) {
  matrix(table(factor(graded_data[, i], levels = 0:3)) / POLYTOMOUS_PERSONS, 1)
})
null_rows <- statistic_rows(rep(4, POLYTOMOUS_ITEMS), "M2*")
null_moments <- model_moments(null_rows, shares, 1)
null_jacobian <- sapply(seq_len(POLYTOMOUS_ITEMS), function(i) item_column(null_rows, shares, 1, i, matrix(c(0, 1, 0, 0), 1)))
null_written <- quadratic_form(observed_moments(null_rows, graded_data) - null_moments$expected, null_moments$covariance,
                               null_jacobian, POLYTOMOUS_PERSONS)
stopifnot(abs(null_written - polytomous_independence$statistic) < 1e-6)

polytomous_case <- function(model, type) {
  fit <- polytomous_fits[[model]]
  expected <- m2_of(fit, type)
  written <- model %in% c("gpcm", "pcm")
  statistic <- if (written) written_partial_credit(fit, type, model == "pcm") else expected$statistic
  if (written || type == "M2*") {
    expected <- with_statistic(expected, statistic, polytomous_independence, POLYTOMOUS_PERSONS)
  }
  list(name = paste(model, type), model = model, statistic = tolower(sub("[*]", "-star", type)), expected = expected)
}
polytomous_cases <- unlist(lapply(names(polytomous_fits), function(model) {
  list(polytomous_case(model, "M2*"), polytomous_case(model, "C2"))
}), recursive = FALSE)

# --- The rejections of a true model -------------------------------------------------------------------------------

SETS <- 30
SET_PERSONS <- 400
rejections <- lapply(seq_len(SETS), function(set) {
  data <- simdata(matrix(slopes), matrix(intercepts), SET_PERSONS, itemtype = "2PL")
  colnames(data) <- colnames(binary)
  fit <- mirt(data, 1, itemtype = "2PL", TOL = 1e-6, quadpts = POINTS, verbose = FALSE)
  list(responses = apply(data, 1, paste, collapse = ""), statistic = M2(fit, quadpts = POINTS, calcNull = FALSE)$M2)
})

# --- Held parameters ----------------------------------------------------------------------------------------------
# Eight 3PL items with the lower asymptote held at 1/4 or 1/5, three 2PL items, and a 2PL item with its slope held at 1,
# as mirt's est = FALSE; Δ has a column for each estimated parameter only.

# The statistic of logistic items written out, items by rows of a, b and c; `slopes` says which slopes are estimated.
# dP/da = (1 - c) L (1 - L) (θ - b) and dP/db = -(1 - c) L (1 - L) a, for L the logistic of a (θ - b).
logistic_written <- function(items, data, slopes) {
  probabilities <- lapply(seq_len(nrow(items)), function(j) {
    p <- items[j, "c"] + (1 - items[j, "c"]) * plogis(items[j, "a"] * (THETA - items[j, "b"]))
    cbind(1 - p, p)
  })
  weights <- normal_weights(0, 1)
  rows <- statistic_rows(rep(2, nrow(items)), "M2*")
  moments <- model_moments(rows, probabilities, weights)
  columns <- lapply(seq_len(nrow(items)), function(j) {
    logistic <- plogis(items[j, "a"] * (THETA - items[j, "b"]))
    slope <- (1 - items[j, "c"]) * logistic * (1 - logistic)
    location <- cbind(slope * items[j, "a"], -slope * items[j, "a"])
    b <- item_column(rows, probabilities, weights, j, location)
    if (!slopes[j]) return(b)
    a <- item_column(rows, probabilities, weights, j, cbind(-slope, slope) * (THETA - items[j, "b"]))
    cbind(a, b)
  })
  quadratic_form(observed_moments(rows, data) - moments$expected, moments$covariance, do.call(cbind, columns),
                 nrow(data))
}

two_pl_items <- coef(two_pl, IRTpars = TRUE, simplify = TRUE)$items
stopifnot(abs(logistic_written(cbind(two_pl_items[, c("a", "b")], c = 0), binary, rep(TRUE, ITEMS)) -
                m2_of(two_pl)$statistic) < 1e-5)

set.seed(20261104)
held_guessing <- c(rep(0.25, 6), rep(0.2, 2), rep(0, 4))
held_slopes <- round(rlnorm(ITEMS, 0.1, 0.25), 2)
held_slopes[ITEMS] <- 1
held_locations <- round(rnorm(ITEMS, 0, 0.8), 2)
held_theta <- rnorm(PERSONS)
held_probabilities <- sapply(seq_len(ITEMS), function(j) {
  held_guessing[j] + (1 - held_guessing[j]) * plogis(held_slopes[j] * (held_theta - held_locations[j]))
})
held_data <- (matrix(runif(PERSONS * ITEMS), PERSONS) < held_probabilities) * 1
colnames(held_data) <- colnames(binary)
held_types <- c(rep("3PL", 8), rep("2PL", 4))
held_values <- mirt(held_data, 1, itemtype = held_types, pars = "values")
for (j in 1:8) {
  rows <- held_values$item == colnames(held_data)[j] & held_values$name == "g"
  held_values$value[rows] <- held_guessing[j]
  held_values$est[rows] <- FALSE
}
held_values$value[held_values$item == colnames(held_data)[ITEMS] & held_values$name == "a1"] <- 1
held_values$est[held_values$item == colnames(held_data)[ITEMS] & held_values$name == "a1"] <- FALSE
held_fit <- fit_mirt(held_data, 1, held_types, pars = held_values)
held_items <- coef(held_fit, IRTpars = TRUE, simplify = TRUE)$items
colnames(held_items)[colnames(held_items) == "g"] <- "c"
held_m2 <- m2_of(held_fit)
held_written <- logistic_written(held_items, held_data, seq_len(ITEMS) != ITEMS)
held_case <- list(
  responses = unname(lapply(seq_len(PERSONS), function(i) as.numeric(held_data[i, ]))),
  models = I(c(rep("3pl", 8), rep("2pl", 4))),
  fixedParameters = c(setNames(lapply(held_guessing[1:8], function(g) list(c = g)), colnames(held_data)[1:8]),
                      setNames(list(list(a = 1)), colnames(held_data)[ITEMS])),
  mirtStatistic = held_m2$statistic,
  expected = with_statistic(held_m2, held_written, independence(held_data), PERSONS)
)

write_fixture("fit-limited-information", list(
  source = "mirt::M2(quadpts = 61) on calibrations at TOL = 1e-8; written out in base R where the script says.",
  binary = list(responses = unname(lapply(seq_len(PERSONS), function(i) as.numeric(binary[i, ]))),
                cases = binary_cases),
  polytomous = list(responses = unname(lapply(seq_len(POLYTOMOUS_PERSONS), function(i) as.numeric(graded_data[i, ]))),
                    cases = polytomous_cases),
  level = list(level = 0.95, expected = m2_of(one_pl, CI = 0.95)),
  rejections = list(sets = rejections, alpha = 0.05, degreesOfFreedom = 54),
  held = held_case
))

# --- The noncentral chi-square ------------------------------------------------------------------------------------

noncentral <- expand.grid(x = c(0.5, 5, 20, 54, 80, 200, 700, 3000, 30000, 120000), df = c(1, 3, 12, 54, 189, 1000),
                          noncentrality = c(0, 0.01, 1, 10, 50, 200, 1500, 12000, 100000))
write_fixture("math-noncentral-chi-square", list(
  source = "Base R pchisq(x, df, ncp).",
  x = noncentral$x, df = noncentral$df, noncentrality = noncentral$noncentrality,
  cdf = pchisq(noncentral$x, noncentral$df, ncp = noncentral$noncentrality)
))
