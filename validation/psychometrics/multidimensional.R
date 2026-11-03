# Multidimensional item response theory fixtures: confirmatory and exploratory calibration of the M2PL, the M3PL and the
# graded model with standard errors, rotation, EAP, MAP, ML and WLE scoring, and information matrices. References: mirt
# (mirt, fscores, probtrace, iteminfo), GPArotation (varimax, oblimin) and base R written out below: direct
# maximization of the marginal likelihood where mirt's EM stops short, the grid integral of the EAP, the information
# matrices from the derivatives of the category probabilities, and the roots of the score and of Warm's weighted score.
# The data are simulated with set.seed(20261029) or are psych::bfi and mirt::SAT12.

source("validation/psychometrics/common.R")
source("validation/psychometrics/multidimensional-items.R")
suppressPackageStartupMessages({
  library(mirt)
  library(GPArotation)
  library(mvtnorm)
  library(psych)
})

TOL <- 1e-10 # mirt's EM tolerance, tight so that its estimates are the optimum
SCALE <- 1.702
# Case names on the command line run only those cases: `Rscript validation/psychometrics/multidimensional.R m3pl`.
requested <- commandArgs(trailingOnly = TRUE)
selected <- function(name) length(requested) == 0 || name %in% requested

# --- Helpers -----------------------------------------------------------------------------------------------------------

# The standard errors of the estimated parameters of each item by name, as measurekit names them. The asymptote g is
# estimated on the logit scale, so its error is the delta method's.
mirt_standard_errors <- function(fit, dimensions, graded, guessing) {
  tables <- coef(fit, printSE = TRUE)
  lapply(rownames(coef(fit, simplify = TRUE)$items), function(id) {
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
        g <- coef(fit, simplify = TRUE)$items[id, "g"]
        named$c <- g * (1 - g) * se[["logit(g)"]]
      }
    }
    named
  })
}

# measurekit's items of mirt's coefficients: the parameters of each and, with `errors`, the standard errors of the
# estimated ones.
mirt_items <- function(fit, dimensions, graded, guessing, errors = TRUE) {
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

# The correlations between the dimensions and their standard errors with a zero diagonal, as measurekit's matrices.
mirt_correlations <- function(fit, dimensions, errors = TRUE) {
  correlations <- unname(coef(fit, simplify = TRUE)$cov)
  result <- list(correlations = correlations)
  if (errors) {
    se <- matrix(0, dimensions, dimensions)
    group <- coef(fit, printSE = TRUE)$GroupPars
    for (i in seq_len(dimensions)) {
      for (j in seq_len(dimensions)) {
        name <- sprintf("COV_%d%d", max(i, j), min(i, j))
        if (i != j && !is.na(group["SE", name])) se[i, j] <- group["SE", name]
      }
    }
    result$correlationStandardErrors <- se
  }
  result
}

# mirt's standardized loadings and communalities, from summary(): loadings (a / 1.702) / sqrt(1 + sum((a / 1.702)^2)) and
# their row sums of squares. They equal measurekit's, which take the correlations of the dimensions into account, for
# items that load on one dimension.
mirt_loadings <- function(fit) {
  standardized <- summary(fit, verbose = FALSE)
  list(standardized = list(loadings = unname(standardized$rotF), communalities = as.numeric(standardized$h2)))
}

mirt_statistics <- function(fit) {
  list(logLikelihood = extract.mirt(fit, "logLik"), aic = extract.mirt(fit, "AIC"), bic = extract.mirt(fit, "BIC"),
       iterations = extract.mirt(fit, "iterations"))
}

stopifnot_converged <- function(fit) stopifnot(extract.mirt(fit, "converged"))

# The information matrix of one item at theta, from the derivatives of the category probabilities written out:
# sum_k grad P_k grad P_k' / P_k.
base_information <- function(item, theta) {
  z <- sum(item[["a"]] * theta) + item[["intercept"]]
  if (is.null(item[["intercepts"]])) {
    probability <- item[["c"]] + (1 - item[["c"]]) * plogis(z)
    gradient <- (1 - item[["c"]]) * plogis(z) * plogis(-z) * item[["a"]]
    return(outer(gradient, gradient) * (1 / probability + 1 / (1 - probability)))
  }
  boundaries <- plogis(z + c(Inf, item[["intercepts"]], -Inf))
  slopes <- c(0, plogis(z + item[["intercepts"]]) * plogis(-(z + item[["intercepts"]])), 0)
  sum_matrices <- 0
  for (k in seq_len(length(item[["intercepts"]]) + 1)) {
    gradient <- (slopes[k] - slopes[k + 1]) * item[["a"]]
    sum_matrices <- sum_matrices + outer(gradient, gradient) / (boundaries[k] - boundaries[k + 1])
  }
  sum_matrices
}

# One response pattern per row of `patterns`, with `x` an NA for a missing response, scored by mirt with the fit's own
# prior: the EAP and MAP thetas with their standard errors. EAP on `points` rectangular nodes from -6 to 6. mirt's MAP
# standard error comes from the observed Hessian; measurekit's from the expected information, which base R computes
# here at mirt's MAP from the information matrices of the items.
mirt_scores <- function(fit, items, patterns, points) {
  eap <- fscores(fit, method = "EAP", response.pattern = patterns, quadpts = points, QMC = FALSE,
                 full.scores.SE = TRUE)
  map <- fscores(fit, method = "MAP", response.pattern = patterns, full.scores.SE = TRUE)
  dimensions <- extract.mirt(fit, "nfact")
  precision <- solve(coef(fit, simplify = TRUE)$cov)
  columns <- function(table) {
    theta <- table[, seq_len(dimensions), drop = FALSE]
    se <- table[, dimensions + seq_len(dimensions), drop = FALSE]
    lapply(seq_len(nrow(table)), function(p) list(theta = I(unname(theta[p, ])), standardErrors = I(unname(se[p, ]))))
  }
  expected_errors <- function(theta, pattern) {
    answered <- which(!is.na(pattern))
    information <- Reduce(`+`, lapply(answered, function(i) {
      parameters <- items[[i]]$parameters
      base_information(c(parameters, list(intercept = if (is.null(parameters[["intercept"]])) 0 else parameters[["intercept"]])), theta)
    }))
    sqrt(diag(solve(information + precision)))
  }
  map_theta <- columns(unname(map[, grep("^F|^SE_F", colnames(map))]))
  list(patterns = unname(patterns), points = points, eap = columns(unname(eap[, grep("^F|^SE_F", colnames(eap))])),
       map = lapply(seq_along(map_theta), function(p) {
         c(map_theta[[p]], list(expectedStandardErrors = I(expected_errors(map_theta[[p]]$theta, patterns[p, ]))))
       }))
}

# Response patterns for scoring: the first rows of the data, then rows with responses set missing.
scoring_patterns <- function(data, count = 24) {
  patterns <- as.matrix(data[seq_len(count), ])
  patterns[2, sample(ncol(patterns), 3)] <- NA
  patterns[5, seq(1, ncol(patterns), by = 2)] <- NA
  patterns[8, -1] <- NA
  patterns
}

confirmatory_pattern <- function(ids, groups) {
  setNames(lapply(seq_along(ids), function(i) I(groups[[i]])), ids)
}

simulate_slopes <- function(per_dimension, dimensions, meanlog = 0.1, sdlog = 0.3) {
  rows <- per_dimension * dimensions
  slopes <- matrix(0, rows, dimensions)
  for (d in seq_len(dimensions)) {
    block <- (d - 1) * per_dimension + seq_len(per_dimension)
    slopes[block, d] <- rlnorm(per_dimension, meanlog, sdlog)
  }
  slopes
}

two_groups <- c(rep("F1", 6), rep("F2", 6)) # the dimension of each of the 12 items of the 2D cases

# --- Confirmatory 2D M2PL, correlated -----------------------------------------------------------------------------------

set.seed(20261029)
m2pl_data <- simdata(simulate_slopes(6, 2), rnorm(12), 1500, itemtype = "dich", sigma = matrix(c(1, 0.4, 0.4, 1), 2))
colnames(m2pl_data) <- sprintf("q%02d", 1:12)

if (selected("m2pl")) {
  m2pl <- mirt(m2pl_data, mirt.model("F1 = 1-6\nF2 = 7-12\nCOV = F1*F2"), quadpts = 31, TOL = TOL, SE = TRUE,
               SE.type = "Oakes", verbose = FALSE, technical = list(NCYCLES = 5000))
  stopifnot_converged(m2pl)
  m2pl_items <- mirt_items(m2pl, 2, graded = FALSE, guessing = FALSE)
  # The cross-product standard errors, mirt's SE.type = "crossprod", from the same estimates.
  m2pl_cross <- mirt(m2pl_data, mirt.model("F1 = 1-6\nF2 = 7-12\nCOV = F1*F2"), quadpts = 31, TOL = TOL, SE = TRUE,
                     SE.type = "crossprod", verbose = FALSE, technical = list(NCYCLES = 5000))
  stopifnot_converged(m2pl_cross)

  write_fixture("multidimensional-m2pl", c(
    list(
      source = "simdata (set.seed(20261029)): 1500 persons, 12 dichotomous items, 6 on each of 2 dimensions correlated 0.4.",
      model = "m2pl", points = 31, itemIds = colnames(m2pl_data), responses = unname(m2pl_data),
      loadings = confirmatory_pattern(colnames(m2pl_data), two_groups), dimensionNames = c("F1", "F2"),
      items = m2pl_items
    ),
    mirt_correlations(m2pl, 2), mirt_statistics(m2pl), mirt_loadings(m2pl),
    list(
      crossProduct = c(
        list(standardErrors = mirt_standard_errors(m2pl_cross, 2, graded = FALSE, guessing = FALSE)),
        mirt_correlations(m2pl_cross, 2)["correlationStandardErrors"]
      ),
      scoring = mirt_scores(m2pl, m2pl_items, scoring_patterns(m2pl_data), 31)
    )
  ))
}

if (selected("missing")) {
  # The same simulation with 8% of the responses missing completely at random.
  set.seed(20261029)
  missing_data <- m2pl_data
  missing_data[matrix(runif(length(missing_data)) < 0.08, nrow(missing_data))] <- NA
  missing <- mirt(missing_data, mirt.model("F1 = 1-6\nF2 = 7-12\nCOV = F1*F2"), quadpts = 31, TOL = TOL, SE = TRUE,
                  SE.type = "Oakes", verbose = FALSE, technical = list(NCYCLES = 5000))
  stopifnot_converged(missing)

  write_fixture("multidimensional-missing", c(
    list(
      source = "The 2D M2PL simulation of multidimensional-m2pl with 8% of the responses missing completely at random.",
      model = "m2pl", points = 31, itemIds = colnames(missing_data), responses = unname(missing_data),
      loadings = confirmatory_pattern(colnames(missing_data), two_groups), dimensionNames = c("F1", "F2"),
      items = mirt_items(missing, 2, graded = FALSE, guessing = FALSE)
    ),
    mirt_correlations(missing, 2), mirt_statistics(missing)
  ))
}

# --- Confirmatory 2D M3PL with normal priors on logit(g) ---------------------------------------------------------------

set.seed(20261029)
m3pl_data <- simdata(simulate_slopes(6, 2), rnorm(12), 1000, itemtype = rep("3PL", 12), guess = rep(0.15, 12),
                     sigma = matrix(c(1, 0.4, 0.4, 1), 2))
colnames(m3pl_data) <- sprintf("q%02d", 1:12)

if (selected("m3pl")) {
  m3pl_prior <- list(type = "logit-normal", mean = qlogis(0.15), sd = 0.5)
  m3pl <- mirt(m3pl_data, mirt.model(sprintf(
    "F1 = 1-6\nF2 = 7-12\nCOV = F1*F2\nPRIOR = (1-12, g, norm, %.17g, %.17g)", m3pl_prior$mean, m3pl_prior$sd
  )), itemtype = "3PL", quadpts = 21, TOL = TOL, SE = TRUE, SE.type = "Oakes", verbose = FALSE,
  technical = list(NCYCLES = 20000))
  stopifnot_converged(m3pl)

  write_fixture("multidimensional-m3pl", c(
    list(
      source = "simdata (set.seed(20261029)): 1000 persons, 12 3PL items with g = 0.15, 6 on each of 2 dimensions correlated 0.4.",
      model = "m3pl", points = 21, itemIds = colnames(m3pl_data), responses = unname(m3pl_data),
      loadings = confirmatory_pattern(colnames(m3pl_data), two_groups), dimensionNames = c("F1", "F2"),
      priors = list(c = m3pl_prior), items = mirt_items(m3pl, 2, graded = FALSE, guessing = TRUE)
    ),
    mirt_correlations(m3pl, 2), mirt_statistics(m3pl)
  ))
}

# --- Confirmatory graded models: psych::bfi ----------------------------------------------------------------------------

bfi_items <- function(scales) {
  columns <- unlist(lapply(scales, function(scale) paste0(scale, 1:5)))
  data <- na.omit(psych::bfi[, columns]) - 1
  rownames(data) <- NULL
  data
}

if (selected("graded")) {
  graded_data <- bfi_items(c("A", "C"))
  graded <- mirt(graded_data, mirt.model("F1 = 1-5\nF2 = 6-10\nCOV = F1*F2"), itemtype = "graded", quadpts = 31,
                 TOL = TOL, SE = TRUE, SE.type = "Oakes", verbose = FALSE, technical = list(NCYCLES = 20000))
  stopifnot_converged(graded)
  graded_groups <- c(rep("A", 5), rep("C", 5))
  graded_items <- mirt_items(graded, 2, graded = TRUE, guessing = FALSE)

  write_fixture("multidimensional-graded", c(
    list(
      source = sprintf("psych::bfi items A1-A5 and C1-C5, complete cases (n = %d), coded 0-5, under the graded model.",
                      nrow(graded_data)),
      model = "mgrm", points = 31, itemIds = colnames(graded_data), responses = unname(as.matrix(graded_data)),
      loadings = confirmatory_pattern(colnames(graded_data), graded_groups), dimensionNames = c("A", "C"),
      items = graded_items
    ),
    mirt_correlations(graded, 2), mirt_statistics(graded), mirt_loadings(graded),
    list(scoring = {set.seed(20261029); mirt_scores(graded, graded_items, scoring_patterns(graded_data), 31)})
  ))
}

if (selected("graded3")) {
  graded3_data <- bfi_items(c("A", "C", "E"))[1:500, ]
  graded3 <- mirt(graded3_data, mirt.model("F1 = 1-5\nF2 = 6-10\nF3 = 11-15\nCOV = F1*F2, F1*F3, F2*F3"),
                  itemtype = "graded", quadpts = 15, TOL = TOL, verbose = FALSE, technical = list(NCYCLES = 20000))
  stopifnot_converged(graded3)

  write_fixture("multidimensional-graded3", c(
    list(
      source = "psych::bfi items A1-A5, C1-C5 and E1-E5, the first 500 complete cases, coded 0-5, under the graded model.",
      model = "mgrm", points = 15, itemIds = colnames(graded3_data), responses = unname(as.matrix(graded3_data)),
      loadings = confirmatory_pattern(colnames(graded3_data), c(rep("A", 5), rep("C", 5), rep("E", 5))),
      dimensionNames = c("A", "C", "E"), items = mirt_items(graded3, 3, graded = TRUE, guessing = FALSE, errors = FALSE)
    ),
    mirt_correlations(graded3, 3, errors = FALSE), mirt_statistics(graded3)
  ))
}

# --- MH-RM: psych::bfi, five dimensions --------------------------------------------------------------------------------

# mirt's MH-RM on all 25 items of psych::bfi under the graded model, the first 800 complete cases, on five correlated
# dimensions, with several seeds: mirt and measurekit are both stochastic, so the seeds give mirt's own Monte Carlo
# error, with which the test justifies its tolerance. mirt's default `SE.type = "MHRM"` gives the standard errors.
if (selected("mhrm")) {
  mhrm_data <- bfi_items(c("A", "C", "E", "N", "O"))[1:800, ]
  mhrm_groups <- rep(c("A", "C", "E", "N", "O"), each = 5)
  mhrm_model <- mirt.model(paste0(
    "A = 1-5\nC = 6-10\nE = 11-15\nN = 16-20\nO = 21-25\n",
    "COV = A*C, A*E, A*N, A*O, C*E, C*N, C*O, E*N, E*O, N*O"
  ))
  fits <- lapply(1:4, function(seed) {
    fit <- mirt(mhrm_data, mhrm_model, itemtype = "graded", method = "MHRM", SE = TRUE, verbose = FALSE,
                technical = list(set.seed = seed))
    stopifnot_converged(fit)
    fit
  })
  runs <- lapply(seq_along(fits), function(seed) {
    fit <- fits[[seed]]
    c(list(seed = seed, items = mirt_items(fit, 5, graded = TRUE, guessing = FALSE)),
      mirt_correlations(fit, 5), mirt_statistics(fit))
  })
  # mirt's complete-data score and information of two items at given thetas: the quantities of one MH-RM cycle, which
  # mirt's Deriv returns for the item's slopes a1 to a5 and intercepts d1 to d5. The thetas are the EAPs of the persons
  # plus noise, so that they are spread as the draws of a cycle are.
  set.seed(20261029)
  thetas <- fscores(fits[[1]], method = "EAP", quadpts = 5, QMC = TRUE, QMC_draws = 2000)[, 1:5]
  thetas <- thetas + matrix(rnorm(length(thetas), 0, 0.4), nrow(thetas))
  complete_data <- lapply(c(1, 23), function(i) {
    item <- extract.item(fits[[1]], i)
    derivative <- mirt:::Deriv(item, Theta = thetas, estHess = TRUE)
    list(item = colnames(mhrm_data)[i], parameters = unname(item@par), responses = unname(mhrm_data[, i]),
         gradient = derivative$grad, hessian = unname(derivative$hess))
  })
  write_fixture("multidimensional-mhrm", list(
    source = paste("psych::bfi items A1-E5, N1-N5 and O1-O5, the first 800 complete cases, coded 0-5, under the graded model:",
                   "mirt's MH-RM with seeds 1 to 4, and its complete-data score and information of items A1 and O3."),
    model = "mgrm", itemIds = colnames(mhrm_data), responses = unname(as.matrix(mhrm_data)),
    loadings = confirmatory_pattern(colnames(mhrm_data), mhrm_groups), dimensionNames = c("A", "C", "E", "N", "O"),
    runs = runs, thetas = unname(thetas), completeData = complete_data
  ))
}

# --- Exploratory 2D models -----------------------------------------------------------------------------------------------

# The model with the first D - 1 items' later slopes fixed at 0, which identifies the rotation, as measurekit fixes it.
exploratory_values <- function(data, dimensions) {
  values <- mirt(data, dimensions, pars = "values")
  for (j in seq_len(dimensions - 1)) {
    later <- values$item == colnames(data)[j] & values$name %in% paste0("a", seq(j + 1, dimensions))
    values$value[later] <- 0
    values$est[later] <- FALSE
  }
  values
}

# The marginal log-likelihood and its gradient of the dichotomous compensatory model on the rectangular grid, in base
# R: `slopes` is items by dimensions, `free` marks the estimated slopes, `parameters` holds them then the intercepts.
# The gradient is the expected complete-data gradient (Fisher's identity).
base_marginal <- function(parameters, responses, grid, log_weights, free) {
  items <- ncol(responses)
  slopes <- matrix(0, items, ncol(grid))
  slopes[free] <- parameters[seq_len(sum(free))]
  intercepts <- parameters[-seq_len(sum(free))]
  z <- grid %*% t(slopes) + rep(intercepts, each = nrow(grid))
  log_joint <- responses %*% t(plogis(z, log.p = TRUE)) + (1 - responses) %*% t(plogis(-z, log.p = TRUE)) +
    rep(log_weights, each = nrow(responses))
  largest <- apply(log_joint, 1, max)
  log_marginal <- largest + log(rowSums(exp(log_joint - largest)))
  posterior <- exp(log_joint - log_marginal)
  expected <- t(posterior) %*% responses - colSums(posterior) * plogis(z)
  list(logLikelihood = sum(log_marginal), gradient = c((t(expected) %*% grid)[free], colSums(expected)))
}

# The maximum of the marginal likelihood of the dichotomous compensatory model from the estimates of `fit`, by BFGS in
# base R: where mirt's EM stops short on a flat likelihood this is the reference.
base_maximum <- function(fit, responses, points) {
  nodes <- seq(-6, 6, length.out = points)
  grid <- as.matrix(expand.grid(nodes, nodes))
  log_weights <- dmvnorm(grid, c(0, 0), diag(2), log = TRUE)
  log_weights <- log_weights - log(sum(exp(log_weights)))
  values <- coef(fit, simplify = TRUE)$items
  free <- matrix(TRUE, nrow(values), 2)
  free[1, 2] <- FALSE
  start <- c(values[, c("a1", "a2")][free], values[, "d"])
  objective <- function(p) -base_marginal(p, responses, grid, log_weights, free)$logLikelihood
  gradient <- function(p) -base_marginal(p, responses, grid, log_weights, free)$gradient
  fitted <- optim(start, objective, gradient, method = "BFGS", control = list(maxit = 20000, reltol = 1e-15))
  stopifnot(fitted$convergence == 0)
  # Newton steps with the Hessian by central differences of the exact gradient finish what BFGS leaves.
  par <- fitted$par
  for (step in 1:5) par <- par - solve(optimHess(par, objective, gradient), gradient(par))
  stopifnot(max(abs(gradient(par))) < 1e-8)
  slopes <- matrix(0, nrow(values), 2)
  slopes[free] <- par[seq_len(sum(free))]
  list(slopes = slopes, intercepts = par[-seq_len(sum(free))], logLikelihood = -objective(par),
       gradient = max(abs(gradient(par))))
}

exploratory_case <- function(data, points, source_note, base = FALSE) {
  fit <- mirt(data, 2, pars = exploratory_values(data, 2), quadpts = points, TOL = TOL, verbose = FALSE,
              technical = list(NCYCLES = 50000))
  stopifnot_converged(fit)
  result <- c(
    list(source = source_note, model = "m2pl", points = points, dimensions = 2, itemIds = colnames(data),
         responses = unname(as.matrix(data)), items = mirt_items(fit, 2, graded = FALSE, guessing = FALSE, errors = FALSE)),
    mirt_statistics(fit)
  )
  if (base) {
    maximum <- base_maximum(fit, as.matrix(data), points)
    result$baseMaximum <- list(
      slopes = unname(maximum$slopes), intercepts = maximum$intercepts, logLikelihood = maximum$logLikelihood,
      gradient = maximum$gradient
    )
  }
  list(fit = fit, fixture = result)
}

# --- Rotation: GPArotation on the loadings of the maximum, mirt's sign convention ------------------------------------

standardized_loadings <- function(slopes) {
  scaled <- slopes / SCALE
  scaled / sqrt(1 + rowSums(scaled^2))
}

# GPArotation's default algorithm (Barzilai-Borwein steps with a nonmonotone line search) to a gradient of 1e-10, then
# mirt's sign convention: each rotated dimension's largest loading in absolute value is positive. The result is checked
# against mirt's own Rotate, which calls GPArotation the same way.
reference_rotation <- function(loadings, method, ...) {
  rotate <- if (method == "varimax") GPArotation::Varimax else GPArotation::oblimin
  rotated <- rotate(loadings, eps = 1e-10, maxit = 1000, ...)
  stopifnot(rotated$convergence)
  rotated_loadings <- unclass(rotated$loadings)
  signs <- apply(rotated_loadings, 2, function(x) sign(x[which.max(abs(x))]))
  correlations <- if (is.null(rotated$Phi)) diag(ncol(loadings)) else rotated$Phi
  rotation <- rotated$Th %*% diag(signs)
  result <- list(
    loadings = unname(t(signs * t(rotated_loadings))),
    correlations = unname(diag(signs) %*% correlations %*% diag(signs)),
    rotation = unname(rotation),
    iterations = nrow(rotated$Table)
  )
  mirt_rotated <- mirt:::Rotate(loadings, method, eps = 1e-10, maxit = 1000, ...)
  stopifnot(max(abs(unclass(mirt_rotated$loadings) - result$loadings)) < 1e-12)
  # The rotated slopes of mirt's rotateLambdas, a (T')^-1 for the same rotation.
  slopes <- mirt:::rotateLambdas(list(rotF = result$loadings, h2 = matrix(rowSums(loadings^2))))
  list(rotation = result, slopes = unname(slopes))
}

rotation_cases <- list(
  list(options = list(method = "varimax"), call = list("varimax")),
  list(options = list(method = "oblimin"), call = list("oblimin")),
  list(options = list(method = "oblimin", normalize = TRUE), call = list("oblimin", normalize = TRUE)),
  list(options = list(method = "oblimin", gamma = 0.5), call = list("oblimin", gam = 0.5))
)

# The patterns scored in the rotated dimensions by mirt's fscores(rotate = "oblimin"): mirt rotates with its default
# tolerance, 1e-5, and its default algorithm.
rotated_scores <- function(data, base, points) {
  coefficients <- data.frame(a1 = base$slopes[, 1], a2 = base$slopes[, 2], d = base$intercepts)
  fit <- set_multidimensional_items(data, 2, "2PL", coefficients)
  patterns <- as.matrix(data[c(1, 3, 4, 6, 7, 9, 10, 11), ])
  patterns[2, c(2, 5, 8)] <- NA
  eap <- fscores(fit, rotate = "oblimin", method = "EAP", response.pattern = patterns, quadpts = points,
                 QMC = FALSE, full.scores.SE = TRUE)
  list(patterns = unname(patterns), points = points,
       eap = lapply(seq_len(nrow(eap)), function(p) list(theta = I(unname(eap[p, c("F1", "F2")])),
                                                          standardErrors = I(unname(eap[p, c("SE_F1", "SE_F2")])))))
}

if (selected("exploratory")) {
  # A clear simple structure (the first 600 persons of the M2PL simulation) and SAT12 scored, whose second dimension is
  # weak and whose likelihood is flat. mirt's EM stops short of the maximum on both, so the reference is the base R
  # maximum started from mirt's estimates.
  clear <- exploratory_case(
    m2pl_data[1:600, ], 11,
    "The first 600 persons of the 2D M2PL simulation of multidimensional-m2pl, fitted as an exploratory model.",
    base = TRUE
  )
  sat12 <- key2binary(SAT12, SAT12_KEY)
  flat <- exploratory_case(
    sat12, 11, "mirt::SAT12 scored with its key, omissions wrong: 600 persons, 32 items, an exploratory 2D model.",
    base = TRUE
  )
  write_fixture("multidimensional-exploratory", list(clear = clear$fixture, flat = flat$fixture))
}

# --- Rotation of the exploratory solution -------------------------------------------------------------------------------

if (selected("rotation")) {
  # The base R maximum of the clear exploratory case, which the exploratory case wrote.
  clear <- jsonlite::read_json("packages/psychometrics/test/fixtures/multidimensional-exploratory.json")$clear
  base <- list(
    slopes = do.call(rbind, lapply(clear$baseMaximum$slopes, unlist)),
    intercepts = unlist(clear$baseMaximum$intercepts)
  )
  unrotated <- standardized_loadings(base$slopes)
  write_fixture("multidimensional-rotation", list(
    source = "GPArotation on the standardized loadings of the base R maximum of the clear exploratory case.",
    slopes = unname(base$slopes), intercepts = base$intercepts, loadings = unname(unrotated),
    rotations = lapply(rotation_cases, function(case) {
      reference <- do.call(reference_rotation, c(list(unrotated), case$call))
      c(list(options = case$options), reference$rotation, list(slopes = reference$slopes))
    }),
    scoring = rotated_scores(m2pl_data[1:600, ], base, 31)
  ))
}

# --- Information matrices ------------------------------------------------------------------------------------------------

information_cases <- list(
  list(name = "m2pl", dimensions = 2, itemtype = "2PL", categories = 2,
       coefficients = data.frame(a1 = c(1.2, 0.5, -0.8, 0.9, 1.4, -0.3), a2 = c(0.7, 1.1, 1.5, -0.6, 0.2, 0.9),
                                 d = c(0.3, -0.4, 1, -1.1, 0.6, 0), g = 0, u = 1)),
  list(name = "m3pl", dimensions = 2, itemtype = "3PL", categories = 2,
       coefficients = data.frame(a1 = c(1.2, 0.5, -0.8, 0.9, 1.4, -0.3), a2 = c(0.7, 1.1, 1.5, -0.6, 0.2, 0.9),
                                 d = c(0.3, -0.4, 1, -1.1, 0.6, 0), g = c(0.2, 0.1, 0.25, 0.15, 0.05, 0.3), u = 1)),
  list(name = "mgrm", dimensions = 2, itemtype = "graded", categories = 4,
       coefficients = data.frame(a1 = c(1.2, 0.5, -0.8, 1), a2 = c(0.7, 1.1, 1.5, 0.2), d1 = c(2, 1.5, 2.2, 1),
                                 d2 = c(0, 0.3, 0.5, -0.4), d3 = c(-2, -1.2, -1.9, -1.5))),
  list(name = "mgrm3", dimensions = 3, itemtype = "graded", categories = 3,
       coefficients = data.frame(a1 = c(1.2, 0.5, 0.9), a2 = c(0.7, 1.1, -0.4), a3 = c(0.3, 0.8, 1.3),
                                 d1 = c(1.5, 1, 0.8), d2 = c(-1, -0.5, -1.4)))
)

if (selected("information")) {
  directions <- list(c(45, 45), c(30, 60), c(45, 60, 60))
  cases <- lapply(information_cases, function(case) {
    items_count <- nrow(case$coefficients)
    data <- as.data.frame(sapply(seq_len(items_count), function(i) rep(0:(case$categories - 1), length.out = 12)))
    colnames(data) <- sprintf("q%d", seq_len(items_count))
    fit <- set_multidimensional_items(data, case$dimensions, case$itemtype, case$coefficients)
    thetas <- list(rep(0, case$dimensions), seq(-0.8, 1.1, length.out = case$dimensions), seq(1.9, -2.4, length.out = case$dimensions))
    values <- coef(fit, simplify = TRUE)$items
    items <- lapply(seq_len(items_count), function(i) {
      a <- unname(values[i, paste0("a", seq_len(case$dimensions))])
      if (case$itemtype == "graded") {
        list(a = a, intercept = 0, intercepts = unname(values[i, grep("^d[0-9]+$", colnames(values))]))
      } else {
        list(a = a, intercept = values[i, "d"], c = values[i, "g"])
      }
    })
    degrees <- directions[[case$dimensions]]
    direction <- cos(degrees * pi / 180)
    matrices <- lapply(seq_len(items_count), function(i) lapply(thetas, function(theta) {
      information <- base_information(items[[i]], theta)
      # The same item's probabilities and its information along a direction, by mirt.
      item <- extract.item(fit, i)
      stopifnot(max(abs(probtrace(item, rbind(theta))[1, ] - if (is.null(items[[i]]$intercepts)) {
        p <- items[[i]]$c + (1 - items[[i]]$c) * plogis(sum(items[[i]]$a * theta) + items[[i]]$intercept)
        c(1 - p, p)
      } else {
        z <- sum(items[[i]]$a * theta)
        boundaries <- plogis(z + c(Inf, items[[i]]$intercepts, -Inf))
        -diff(boundaries)
      })) < 1e-14)
      stopifnot(abs(drop(t(direction) %*% information %*% direction) - iteminfo(item, rbind(theta), degrees = degrees)) < 1e-10)
      unname(information)
    }))
    list(model = case$name, dimensions = case$dimensions,
         items = lapply(seq_len(items_count), function(i) {
           parameters <- items[[i]]
           if (is.null(parameters$intercepts)) c(list(a = I(parameters$a), intercept = parameters$intercept, c = parameters$c, d = 1))
           else list(a = I(parameters$a), intercepts = I(parameters$intercepts))
         }),
         thetas = lapply(thetas, I), matrices = matrices,
         # mirt's multidimensional discrimination and difficulty, one difficulty per boundary between categories.
         discrimination = unname(MDISC(fit)),
         difficulty = lapply(seq_len(items_count), function(i) I(unname(Filter(Negate(is.na), MDIFF(fit)[i, ])))))
  })
  write_fixture("multidimensional-information", list(
    source = "Items set in mirt; information from the derivatives of the category probabilities in base R, checked against probtrace and iteminfo; MDISC and MDIFF of mirt.",
    cases = cases
  ))
}

# --- Maximum and weighted likelihood estimates ---------------------------------------------------------------------------

# Items of 2 and 3 dimensions, some on one dimension and some on two, with the patterns to score: seeded random
# patterns, the lowest and the highest categories everywhere, the highest on the items of the first dimension alone with
# the lowest elsewhere, and patterns with missing responses.
likelihood_cases <- list(
  list(name = "m2pl", dimensions = 2, itemtype = "2PL", categories = 2,
       coefficients = data.frame(a1 = c(1.2, 0.9, 1.5, 0.7, 1.1, 0, 0, 0, 0.6, 0.8), a2 = c(0, 0, 0, 0, 0, 1.3, 0.8, 1.4, 0.9, 0.5),
                                 d = c(0.4, -0.6, 1, -1.2, 0.2, 0.8, -0.3, -1, 0.5, -0.4), g = 0, u = 1)),
  list(name = "m3pl", dimensions = 2, itemtype = "3PL", categories = 2,
       coefficients = data.frame(a1 = c(1.2, 0.9, 1.5, 0.7, 1.1, 0, 0, 0, 0.6, 0.8), a2 = c(0, 0, 0, 0, 0, 1.3, 0.8, 1.4, 0.9, 0.5),
                                 d = c(0.4, -0.6, 1, -1.2, 0.2, 0.8, -0.3, -1, 0.5, -0.4),
                                 g = c(0.2, 0.15, 0.1, 0.25, 0.2, 0.1, 0.2, 0.15, 0.1, 0.2), u = 1)),
  list(name = "mgrm", dimensions = 2, itemtype = "graded", categories = 4,
       coefficients = data.frame(a1 = c(1.2, 0.8, 1.4, 0, 0, 0.7), a2 = c(0, 0, 0, 1.1, 1.5, 0.9),
                                 d1 = c(2, 1.5, 2.2, 1.8, 2.4, 1.6), d2 = c(0, 0.3, -0.5, 0.2, 0, -0.2),
                                 d3 = c(-2, -1.2, -1.9, -1.5, -2.2, -1.7))),
  list(name = "mgrm3", dimensions = 3, itemtype = "graded", categories = 3,
       coefficients = data.frame(a1 = c(1.2, 0.9, 0, 0, 0, 0, 0.7), a2 = c(0, 0, 1.1, 1.4, 0, 0, 0.6), a3 = c(0, 0, 0, 0, 1.3, 0.8, 0.5),
                                 d1 = c(1.5, 1, 0.8, 1.2, 1.4, 0.9, 1.1), d2 = c(-1, -0.5, -1.4, -0.8, -1.2, -0.6, -0.9)))
)
LIKELIHOOD_RANGE <- c(-6, 6)

# The probabilities of an item's categories at z = a · theta and their first and second derivatives along z.
category_curves <- function(item, z) {
  if (is.null(item$intercepts)) {
    s <- plogis(z + item$intercept)
    p1 <- item$c + (1 - item$c) * s
    d1 <- (1 - item$c) * s * (1 - s)
    d2 <- d1 * (1 - 2 * s)
    list(p = c(1 - p1, p1), d1 = c(-d1, d1), d2 = c(-d2, d2))
  } else {
    s <- plogis(z + c(Inf, item$intercepts, -Inf))
    first <- s * (1 - s)
    second <- first * (1 - 2 * s)
    list(p = -diff(s), d1 = -diff(first), d2 = -diff(second))
  }
}

# The score, the information and Warm's correction of a pattern at theta, written out for compensatory items: with
# P_k = f_k(a · theta), ∇P_k = f_k' a and ∇²P_k = f_k'' a aᵀ, so that ½ tr(I⁻¹ J_r) = ½ Σ_i Σ_k f' f'' / f (aᵀ I⁻¹ a) a_r.
pattern_terms <- function(items, pattern, theta) {
  dims <- length(theta)
  score <- numeric(dims)
  information <- matrix(0, dims, dims)
  curvature <- list()
  log_likelihood <- 0
  for (i in seq_along(items)) {
    if (is.na(pattern[i])) next
    a <- items[[i]]$a
    curves <- category_curves(items[[i]], sum(a * theta))
    x <- pattern[i] + 1
    log_likelihood <- log_likelihood + log(curves$p[x])
    score <- score + curves$d1[x] / curves$p[x] * a
    information <- information + sum(curves$d1^2 / curves$p) * a %o% a
    curvature[[length(curvature) + 1]] <- list(a = a, j = sum(curves$d1 * curves$d2 / curves$p))
  }
  inverse <- solve(information)
  correction <- Reduce(`+`, lapply(curvature, function(term) 0.5 * term$j * drop(t(term$a) %*% inverse %*% term$a) * term$a))
  list(log_likelihood = log_likelihood, score = score, information = information, weighted = score + correction)
}

# The root of a field within the box by Newton's method on its coordinates inside the box, a coordinate at a bound staying
# there while the field pushes it out, with the Jacobian by numDeriv and steps halved until the field's length falls.
box_root <- function(field, start) {
  clamp <- function(x) pmin(LIKELIHOOD_RANGE[2], pmax(LIKELIHOOD_RANGE[1], x))
  free_of <- function(theta, g) !((theta <= LIKELIHOOD_RANGE[1] & g < 0) | (theta >= LIKELIHOOD_RANGE[2] & g > 0))
  size <- function(theta) { g <- field(theta); sqrt(sum(g[free_of(theta, g)]^2)) }
  theta <- clamp(start)
  for (iteration in 1:200) {
    g <- field(theta)
    free <- free_of(theta, g)
    if (all(abs(g[free]) < 1e-12)) break
    jacobian <- numDeriv::jacobian(field, theta)
    step <- numeric(length(theta))
    step[free] <- -solve(jacobian[free, free, drop = FALSE], g[free])
    current <- size(theta)
    length <- 1
    repeat {
      trial <- clamp(theta + length * step)
      if (size(trial) < current || length < 1e-12) break
      length <- length / 2
    }
    if (length < 1e-12) break
    theta <- trial
  }
  g <- field(theta)
  stopifnot(all(abs(g[free_of(theta, g)]) < 1e-9))
  theta
}

if (selected("likelihood")) {
  set.seed(20261104)
  cases <- lapply(likelihood_cases, function(case) {
    count <- nrow(case$coefficients)
    data <- as.data.frame(sapply(seq_len(count), function(i) rep(0:(case$categories - 1), length.out = 12)))
    colnames(data) <- sprintf("q%d", seq_len(count))
    # A confirmatory model of every item on every factor: mirt rotates the scores of an exploratory model by oblimin.
    model <- mirt.model(paste0("F", seq_len(case$dimensions), " = 1-", count, collapse = "\n"))
    fit <- set_multidimensional_items(data, model, case$itemtype, case$coefficients)
    values <- coef(fit, simplify = TRUE)$items
    items <- lapply(seq_len(count), function(i) {
      a <- unname(values[i, paste0("a", seq_len(case$dimensions))])
      if (case$itemtype == "graded") list(a = a, intercepts = unname(values[i, grep("^d[0-9]+$", colnames(values))]))
      else list(a = a, intercept = values[i, "d"], c = values[i, "g"])
    })
    first <- sapply(items, function(item) item$a[1] != 0 && all(item$a[-1] == 0))
    random <- t(replicate(10, sample(0:(case$categories - 1), count, replace = TRUE)))
    missing <- random[1:2, ]
    missing[1, c(2, 5)] <- NA
    missing[2, c(1, 6)] <- NA
    patterns <- rbind(random, rep(0, count), rep(case$categories - 1, count),
                      ifelse(first, case$categories - 1, 0), missing)
    colnames(patterns) <- colnames(data)
    mirt_ml <- fscores(fit, method = "ML", response.pattern = patterns)
    mirt_wle <- fscores(fit, method = "WLE", response.pattern = patterns)
    factors <- paste0("F", seq_len(case$dimensions))
    # mirt's WLE is the mode of L √det I, Jeffreys' prior: the gradient of its log vanishes at mirt's estimates, within
    # mirt's tolerance. That is Warm's estimator for the M2PL and the graded model, not for the M3PL.
    jeffreys <- function(pattern) function(theta) {
      terms <- pattern_terms(items, pattern, theta)
      terms$log_likelihood + 0.5 * log(det(terms$information))
    }
    for (p in seq_len(nrow(patterns))) {
      stopifnot(max(abs(numDeriv::grad(jeffreys(patterns[p, ]), unname(mirt_wle[p, factors])))) < 1e-4)
    }
    estimates <- lapply(seq_len(nrow(patterns)), function(p) {
      pattern <- patterns[p, ]
      start_ml <- pmin(6, pmax(-6, ifelse(is.finite(mirt_ml[p, factors]), mirt_ml[p, factors], sign(mirt_ml[p, factors]) * 6)))
      ml <- box_root(function(theta) pattern_terms(items, pattern, theta)$score, start_ml)
      wle <- box_root(function(theta) pattern_terms(items, pattern, theta)$weighted, pmin(6, pmax(-6, mirt_wle[p, factors])))
      se <- function(theta) sqrt(diag(solve(pattern_terms(items, pattern, theta)$information)))
      list(pattern = I(unname(pattern)),
           ml = I(ml), mlStandardErrors = I(se(ml)), wle = I(wle), wleStandardErrors = I(se(wle)),
           mirtMl = I(unname(mirt_ml[p, factors])), mirtWle = I(unname(mirt_wle[p, factors])))
    })
    list(model = case$name, dimensions = case$dimensions,
         items = lapply(items, function(item) {
           if (is.null(item$intercepts)) list(a = I(item$a), intercept = item$intercept, c = item$c, d = 1)
           else list(a = I(item$a), intercepts = I(item$intercepts))
         }),
         estimates = estimates)
  })
  write_fixture("multidimensional-likelihood", list(
    source = paste("Items set in mirt; base R maximum likelihood and Warm's weighted likelihood estimates in the box [-6, 6],",
                   "the roots of the score and of the score plus ½ tr(I⁻¹ J_r) by Newton's method; mirt's fscores ML and WLE,",
                   "whose WLE maximizes L √det I, Warm's estimator for the M2PL and the graded model and not for the M3PL;",
                   "standard errors from I⁻¹."),
    range = I(LIKELIHOOD_RANGE),
    cases = cases
  ))
}
