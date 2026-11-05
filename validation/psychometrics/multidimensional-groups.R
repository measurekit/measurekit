# Multidimensional item response theory with several groups: calibration of the M2PL and the graded model in groups with
# their own means and covariances, with and without fixed items, against mirt's multipleGroup, with items invariant across
# groups (invariance = c("slopes", "intercepts")) and the means and covariances of every group but the first free
# ("free_means", "free_var"). Fixed items are mirt's pars with est = FALSE, and the scale they set frees the means and
# covariance of the first group too. The data are simulated with set.seed(20261029). Two dimensions are on 21 nodes, where
# measurekit's calibrations agree with mirt's as closely as on 31 in less than half the time; three are on 15.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages({
  library(mirt)
  library(mvtnorm)
})

TOL <- 1e-10 # mirt's EM tolerance, tight so that its estimates are the optimum
INVARIANCE <- c("slopes", "intercepts", "free_means", "free_var")
cases <- list()

# --- Helpers -----------------------------------------------------------------------------------------------------------

# Slopes with simple structure, `per_dimension` items on each of `dimensions` dimensions.
simulate_slopes <- function(per_dimension, dimensions) {
  slopes <- matrix(0, per_dimension * dimensions, dimensions)
  for (d in seq_len(dimensions)) {
    slopes[(d - 1) * per_dimension + seq_len(per_dimension), d] <- round(rlnorm(per_dimension, 0.1, 0.3), 3)
  }
  slopes
}

# The mirt model of simple structure: dimension d is measured by its block of items; with `correlated`, COV frees the
# correlations.
simple_model <- function(per_dimension, dimensions, correlated) {
  names <- paste0("F", seq_len(dimensions))
  lines <- sprintf("%s = %d-%d", names, (seq_len(dimensions) - 1) * per_dimension + 1, seq_len(dimensions) * per_dimension)
  pairs <- if (correlated) paste(combn(names, 2, paste, collapse = "*"), collapse = ", ") else NULL
  mirt.model(paste(c(lines, if (correlated) paste("COV =", pairs)), collapse = "\n"))
}

# The data of persons from several groups: `simdata` draws each group with its own mean and covariance. `n` and `mu`
# have one entry per group, `sigma` a covariance matrix per group.
simulate_groups <- function(slopes, intercepts, n, mu, sigma, itemtype = "dich") {
  data <- do.call(rbind, lapply(seq_along(n), function(g) {
    simdata(slopes, intercepts, n[g], itemtype = itemtype, mu = mu[[g]], sigma = sigma[[g]])
  }))
  list(data = data, group = rep(LETTERS[seq_along(n)], times = n))
}

ids <- function(count) sprintf("q%02d", seq_len(count))

# Sets responses missing by design: `blocks` lists the items that each group does not answer.
leave_out <- function(data, group, blocks) {
  for (label in names(blocks)) data[group == label, blocks[[label]]] <- NA
  data
}

# measurekit's items of mirt's coefficients in the first group, whose items every group shares: the parameters of each
# and, with `errors`, the standard errors of the estimated entries by name. `fixed` lists the ids of items that are not
# estimated, which have none.
mirt_items <- function(fit, dimensions, graded, errors, fixed = character()) {
  values <- coef(fit, simplify = TRUE)[[1]]$items
  tables <- if (errors) coef(fit, printSE = TRUE)[[1]] else NULL
  lapply(seq_len(nrow(values)), function(i) {
    v <- values[i, ]
    id <- rownames(values)[i]
    slopes <- unname(v[paste0("a", seq_len(dimensions))])
    parameters <- if (graded) {
      list(a = I(slopes), intercepts = I(unname(v[grep("^d[0-9]+$", names(v))])))
    } else {
      list(a = I(slopes), intercept = v[["d"]], c = 0, d = 1)
    }
    row <- list(id = id, parameters = parameters)
    if (errors && !(id %in% fixed)) {
      se <- tables[[id]]["SE", ]
      named <- list()
      for (k in seq_len(dimensions)) {
        if (!is.na(se[[paste0("a", k)]])) named[[paste0("a", k)]] <- se[[paste0("a", k)]]
      }
      if (graded) {
        for (name in grep("^d[0-9]+$", names(se), value = TRUE)) named[[sub("^d", "intercept", name)]] <- se[[name]]
      } else {
        named$intercept <- se[["d"]]
      }
      row$standardErrors <- named
    }
    row
  })
}

# Each group's means and covariance, and the correlations of the first group; with `errors`, their standard errors, which
# mirt reports for the covariances, so that they are the correlations' only while the first group's variances are 1.
mirt_groups <- function(fit, errors) {
  moments <- coef(fit, simplify = TRUE)
  first <- moments[[1]]$cov
  dimensions <- nrow(first)
  result <- list(
    groupMoments = lapply(names(moments), function(label) {
      list(group = label, means = I(unname(moments[[label]]$means)), covariance = unname(moments[[label]]$cov))
    }),
    correlations = unname(cov2cor(first))
  )
  if (errors) {
    se <- matrix(0, dimensions, dimensions)
    table <- coef(fit, printSE = TRUE)[[1]]$GroupPars
    for (i in seq_len(dimensions)) {
      for (j in seq_len(dimensions)) {
        name <- sprintf("COV_%d%d", max(i, j), min(i, j))
        if (i != j && name %in% colnames(table) && !is.na(table["SE", name])) se[i, j] <- table["SE", name]
      }
    }
    result$correlationStandardErrors <- se
  }
  result
}

mirt_statistics <- function(fit) {
  list(logLikelihood = extract.mirt(fit, "logLik"), aic = extract.mirt(fit, "AIC"), bic = extract.mirt(fit, "BIC"),
       iterations = extract.mirt(fit, "iterations"))
}

# The loadings of each item: the names of the dimensions it loads on, as measurekit's `loadings` option.
loading_pattern <- function(per_dimension, dimensions) {
  setNames(lapply(seq_len(per_dimension * dimensions), function(i) I(paste0("F", ceiling(i / per_dimension)))),
           ids(per_dimension * dimensions))
}

calibrate <- function(data, model, group, itemtype, quadpts, errors) {
  multipleGroup(data, model, group = group, itemtype = itemtype, invariance = INVARIANCE, quadpts = quadpts, TOL = TOL,
                SE = errors, SE.type = "Oakes", verbose = FALSE, technical = list(NCYCLES = 5000))
}

stopifnot_converged <- function(fit) stopifnot(extract.mirt(fit, "converged"))

# --- Correlated dimensions, two groups -----------------------------------------------------------------------------------

set.seed(20261029)
two_slopes <- simulate_slopes(6, 2)
two_intercepts <- round(rnorm(12), 3)
two <- simulate_groups(two_slopes, two_intercepts, c(900, 700),
                       list(c(0, 0), c(0.4, 0.4)),
                       list(matrix(c(1, 0.4, 0.4, 1), 2), matrix(c(1.2, 0.5, 0.5, 0.8), 2)))
colnames(two$data) <- ids(12)
two_model <- simple_model(6, 2, TRUE)

cases$correlated <- local({
  fit <- calibrate(two$data, two_model, two$group, "2PL", 21, TRUE)
  stopifnot_converged(fit)
  c(
    list(
      source = "mirt::simdata (set.seed(20261029)): 900 persons of N(0, R) and 700 of N((0.4, 0.4), S), 12 dichotomous items, 6 on each of 2 dimensions, R and S correlated 0.4 and 0.52.",
      model = "m2pl", points = 21, itemIds = colnames(two$data), responses = unname(two$data),
      personGroups = two$group, loadings = loading_pattern(6, 2),
      items = mirt_items(fit, 2, graded = FALSE, errors = TRUE)
    ),
    mirt_groups(fit, TRUE), mirt_statistics(fit)
  )
})

# --- Fixed items ---------------------------------------------------------------------------------------------------------

cases$anchors <- local({
  # Items q01, q02, q07 and q08 are fixed at the parameters that generated them, which put the first group's mean and
  # covariance on the scale of the simulation.
  anchors <- c(1, 2, 7, 8)
  # The other items are equal across groups by name: mirt cannot constrain parameters that are not estimated.
  invariance <- c(colnames(two$data)[-anchors], "free_means", "free_var")
  values <- multipleGroup(two$data, two_model, group = two$group, invariance = invariance, pars = "values")
  for (i in anchors) {
    for (name in c("a1", "a2", "d")) {
      rows <- values$item == colnames(two$data)[i] & values$name == name
      values$value[rows] <- if (name == "d") two_intercepts[i] else two_slopes[i, as.integer(substr(name, 2, 2))]
      values$est[rows] <- FALSE
    }
  }
  first <- values$group == "A" & values$item == "GROUP"
  values$est[first & values$name %in% c("MEAN_1", "MEAN_2", "COV_11", "COV_22")] <- TRUE
  fit <- multipleGroup(two$data, two_model, group = two$group, invariance = invariance, pars = values, quadpts = 21,
                       TOL = TOL, SE = TRUE, SE.type = "Oakes", verbose = FALSE, technical = list(NCYCLES = 5000))
  stopifnot_converged(fit)
  fixed <- setNames(lapply(anchors, function(i) {
    list(a = I(two_slopes[i, ]), intercept = two_intercepts[i], c = 0, d = 1)
  }), colnames(two$data)[anchors])
  c(
    list(
      source = "The data of the correlated case, with items q01, q02, q07 and q08 fixed at their generating parameters.",
      model = "m2pl", points = 21, itemIds = colnames(two$data), responses = unname(two$data),
      personGroups = two$group, loadings = loading_pattern(6, 2), fixed = fixed,
      items = mirt_items(fit, 2, graded = FALSE, errors = TRUE, fixed = colnames(two$data)[anchors])
    ),
    mirt_groups(fit, FALSE), mirt_statistics(fit)
  )
})

# --- Uncorrelated dimensions, three groups, items missing by design ------------------------------------------------------

cases$uncorrelated <- local({
  set.seed(20261029)
  slopes <- simulate_slopes(5, 2)
  intercepts <- round(rnorm(10), 3)
  three <- simulate_groups(slopes, intercepts, c(500, 450, 400),
                           list(c(0, 0), c(0.3, -0.2), c(-0.4, 0.5)),
                           list(diag(2), diag(c(1.3, 0.7)), diag(c(0.8, 1.1))))
  colnames(three$data) <- ids(10)
  # The second group does not answer q01 and q06, the third q02 and q07: the groups share the other items.
  three$data <- leave_out(three$data, three$group, list(B = c(1, 6), C = c(2, 7)))
  fit <- calibrate(three$data, simple_model(5, 2, FALSE), three$group, "2PL", 21, TRUE)
  stopifnot_converged(fit)
  c(
    list(
      source = "mirt::simdata (set.seed(20261029)): 500, 450 and 400 persons of three groups with uncorrelated dimensions, 10 dichotomous items, 5 on each of 2 dimensions, 2 items left unanswered in each of the last two groups.",
      model = "m2pl", points = 21, itemIds = colnames(three$data), responses = unname(three$data),
      personGroups = three$group, loadings = loading_pattern(5, 2), correlated = FALSE,
      items = mirt_items(fit, 2, graded = FALSE, errors = TRUE)
    ),
    mirt_groups(fit, FALSE), mirt_statistics(fit)
  )
})

# --- The graded model, two groups ----------------------------------------------------------------------------------------

cases$graded <- local({
  set.seed(20261029)
  slopes <- simulate_slopes(4, 2)
  intercepts <- t(sapply(round(rnorm(8), 3), function(m) sort(round(m + c(-1, 0, 1) + rnorm(3, 0, 0.2), 3), decreasing = TRUE)))
  graded <- simulate_groups(slopes, intercepts, c(700, 600), list(c(0, 0), c(0.5, 0.2)),
                            list(matrix(c(1, 0.3, 0.3, 1), 2), matrix(c(0.7, 0.2, 0.2, 1.4), 2)), itemtype = "graded")
  colnames(graded$data) <- ids(8)
  fit <- calibrate(graded$data, simple_model(4, 2, TRUE), graded$group, "graded", 21, FALSE)
  stopifnot_converged(fit)
  c(
    list(
      source = "mirt::simdata (set.seed(20261029)): 700 and 600 persons of two groups, 8 items of 4 categories, 4 on each of 2 correlated dimensions.",
      model = "mgrm", points = 21, itemIds = colnames(graded$data), responses = unname(graded$data),
      personGroups = graded$group, loadings = loading_pattern(4, 2),
      items = mirt_items(fit, 2, graded = TRUE, errors = FALSE)
    ),
    mirt_groups(fit, FALSE), mirt_statistics(fit)
  )
})

# --- Three dimensions ---------------------------------------------------------------------------------------------------

# The second group's standard deviations are above the spacing of 15 nodes from -6 to 6, 0.857, which measurekit needs
# to call a distribution estimated.
cases$three <- local({
  set.seed(20261029)
  slopes <- simulate_slopes(6, 3)
  intercepts <- round(rnorm(18), 3)
  sigma <- matrix(c(1, 0.3, 0.2, 0.3, 1, 0.4, 0.2, 0.4, 1), 3)
  scale <- diag(c(1.15, 1.3, 1.1))
  three <- simulate_groups(slopes, intercepts, c(600, 500), list(rep(0, 3), c(0.4, 0.3, 0.5)),
                           list(sigma, scale %*% sigma %*% scale))
  colnames(three$data) <- ids(18)
  fit <- calibrate(three$data, simple_model(6, 3, TRUE), three$group, "2PL", 15, FALSE)
  stopifnot_converged(fit)
  c(
    list(
      source = "mirt::simdata (set.seed(20261029)): 600 and 500 persons of two groups, 18 dichotomous items, 6 on each of 3 correlated dimensions.",
      model = "m2pl", points = 15, itemIds = colnames(three$data), responses = unname(three$data),
      personGroups = three$group, loadings = loading_pattern(6, 3),
      items = mirt_items(fit, 3, graded = FALSE, errors = FALSE)
    ),
    mirt_groups(fit, FALSE), mirt_statistics(fit)
  )
})

# --- A wide group ---------------------------------------------------------------------------------------------------------

# mirt's log-likelihood at the estimates of `maximum`, evaluated with every parameter held and TOL = NaN, so that mirt
# judges the base R maximum by its own likelihood.
mirt_log_likelihood_at <- function(data, model, group, points, maximum) {
  values <- multipleGroup(data, model, group = group, quadpts = points, pars = "values")
  for (g in seq_along(maximum$groupMoments)) {
    moments <- maximum$groupMoments[[g]]
    label <- moments$group
    for (i in seq_along(maximum$items)) {
      item <- maximum$items[[i]]
      for (name in c("a1", "a2", "d")) {
        value <- if (name == "d") item$intercept else item$a[[as.integer(substr(name, 2, 2))]]
        values$value[values$group == label & values$item == colnames(data)[i] & values$name == name] <- value
      }
    }
    entries <- c(MEAN_1 = moments$means[[1]], MEAN_2 = moments$means[[2]], COV_11 = moments$covariance[1, 1],
                 COV_21 = moments$covariance[2, 1], COV_22 = moments$covariance[2, 2])
    for (name in names(entries)) values$value[values$group == label & values$item == "GROUP" & values$name == name] <- entries[[name]]
  }
  values$est <- FALSE
  extract.mirt(multipleGroup(data, model, group = group, quadpts = points, pars = values, TOL = NaN, verbose = FALSE), "logLik")
}

# The maximum of the marginal likelihood of two groups of the M2PL with simple structure on the rectangular grid of `points`
# nodes per dimension from -6 to 6, from mirt's estimates, by BFGS in base R: the slopes and intercepts, the first group's
# correlation, and the second group's means and covariance through its Cholesky factor. Each group's normal density at the
# nodes is normalized to 1, as measurekit and mirt weigh the nodes.
base_groups_maximum <- function(fit, data, group, per_dimension, points) {
  nodes <- seq(-6, 6, length.out = points)
  grid <- as.matrix(expand.grid(nodes, nodes))
  items <- ncol(data)
  dimension <- ceiling(seq_len(items) / per_dimension)
  labels <- unique(group)
  unpack <- function(x) {
    slopes <- matrix(0, items, 2)
    slopes[cbind(seq_len(items), dimension)] <- x[seq_len(items)]
    factor <- matrix(c(exp(x[2 * items + 4]), x[2 * items + 5], 0, exp(x[2 * items + 6])), 2)
    list(slopes = slopes, intercepts = x[items + seq_len(items)],
         moments = list(list(means = c(0, 0), covariance = matrix(c(1, tanh(x[2 * items + 1]), tanh(x[2 * items + 1]), 1), 2)),
                        list(means = x[2 * items + 2:3], covariance = factor %*% t(factor))))
  }
  log_likelihood <- function(x) {
    p <- unpack(x)
    z <- grid %*% t(p$slopes) + rep(p$intercepts, each = nrow(grid))
    total <- 0
    for (g in seq_along(labels)) {
      rows <- data[group == labels[g], , drop = FALSE]
      log_weights <- dmvnorm(grid, p$moments[[g]]$means, p$moments[[g]]$covariance, log = TRUE)
      log_weights <- log_weights - log(sum(exp(log_weights - max(log_weights)))) - max(log_weights)
      log_joint <- rows %*% t(plogis(z, log.p = TRUE)) + (1 - rows) %*% t(plogis(-z, log.p = TRUE)) +
        rep(log_weights, each = nrow(rows))
      largest <- apply(log_joint, 1, max)
      total <- total + sum(largest + log(rowSums(exp(log_joint - largest))))
    }
    total
  }
  values <- coef(fit, simplify = TRUE)
  first <- values[[1]]$items
  second <- values[[2]]$cov
  factor <- t(chol(second))
  start <- c(first[cbind(seq_len(items), dimension)], first[, "d"], atanh(cov2cor(values[[1]]$cov)[1, 2]),
             values[[2]]$means, log(factor[1, 1]), factor[2, 1], log(factor[2, 2]))
  found <- optim(start, function(x) -log_likelihood(x), method = "BFGS", control = list(reltol = 1e-15, maxit = 5000))
  stopifnot(found$convergence == 0)
  found$value <- -found$value
  gradient <- numDeriv::grad(log_likelihood, found$par)
  p <- unpack(found$par)
  list(
    items = lapply(seq_len(items), function(i) list(a = I(p$slopes[i, ]), intercept = p$intercepts[i])),
    groupMoments = lapply(seq_along(labels), function(g) {
      list(group = labels[g], means = I(p$moments[[g]]$means), covariance = p$moments[[g]]$covariance)
    }),
    logLikelihood = found$value, gradient = max(abs(gradient))
  )
}

# The second group's standard deviations are near 2 on the scale of the first. mirt's EM stops at estimates whose
# log-likelihood, by mirt's own evaluation, is below the maximum, with that group's variances smaller: the reference is the
# maximum of the same likelihood in base R, and the test holds measurekit's log-likelihood to at least mirt's.
cases$wide <- local({
  set.seed(20261106)
  slopes <- simulate_slopes(4, 2)
  intercepts <- round(rnorm(8), 3)
  spread <- diag(c(2, 1.9))
  wide <- simulate_groups(slopes, intercepts, c(400, 400), list(c(0, 0), c(0.5, -0.3)),
                          list(matrix(c(1, 0.4, 0.4, 1), 2), spread %*% matrix(c(1, 0.3, 0.3, 1), 2) %*% spread))
  colnames(wide$data) <- ids(8)
  fit <- calibrate(wide$data, simple_model(4, 2, TRUE), wide$group, "2PL", 21, FALSE)
  stopifnot_converged(fit)
  c(
    list(
      source = "mirt::simdata (set.seed(20261106)): 400 persons of N(0, R) and 400 of N((0.5, -0.3), S), S with standard deviations 2 and 1.9 and correlation 0.3, 8 dichotomous items, 4 on each of 2 dimensions.",
      model = "m2pl", points = 21, itemIds = colnames(wide$data), responses = unname(wide$data),
      personGroups = wide$group, loadings = loading_pattern(4, 2),
      items = mirt_items(fit, 2, graded = FALSE, errors = FALSE)
    ),
    mirt_groups(fit, FALSE), mirt_statistics(fit),
    local({
      maximum <- base_groups_maximum(fit, wide$data, wide$group, 4, 21)
      maximum$mirtLogLikelihood <- mirt_log_likelihood_at(wide$data, simple_model(4, 2, TRUE), wide$group, 21, maximum)
      list(baseMaximum = maximum)
    })
  )
})

write_fixture("multidimensional-groups", cases)
