# Helpers of longitudinal.R, longitudinal-recovery.R and the benchmark of calibrateLongitudinal: simulation of persons who
# answer the same items on several occasions, and mirt's model of them, with one dimension per occasion.

suppressPackageStartupMessages({
  library(mirt)
  library(mvtnorm)
})

TOL <- 1e-9 # mirt's EM tolerance, tight so that its estimates are the optimum

# --- Simulation ---------------------------------------------------------------------------------------------------------

# The correlation matrix of the occasions: `rho` between the first two and an autoregressive decay beyond.
autoregressive <- function(occasions, rho) rho^abs(outer(seq_len(occasions), seq_len(occasions), "-"))

# Abilities of `n` persons on the occasions, normal with the means and standard deviations and the correlations.
simulate_abilities <- function(n, means, sds, correlations) {
  rmvnorm(n, mean = means, sigma = diag(sds) %*% correlations %*% diag(sds))
}

# The items of a design: a master table of ids, slopes and difficulties, and the master rows each occasion answers.
# Dichotomous items have one difficulty; graded items, `categories - 1` increasing thresholds.
simulate_items <- function(count, categories = 2) {
  thresholds <- function(location) {
    if (categories == 2) return(location)
    sort(round(location + seq(-1, 1, length.out = categories - 1) + rnorm(categories - 1, 0, 0.15), 3))
  }
  list(
    ids = sprintf("q%02d", seq_len(count)),
    a = round(rlnorm(count, 0.1, 0.3), 3),
    b = do.call(rbind, lapply(round(rnorm(count, 0, 0.8), 3), thresholds)),
    categories = categories
  )
}

# Responses of every person to the items of one occasion, from the abilities of that occasion.
respond <- function(theta, items, rows) {
  sapply(rows, function(i) {
    cumulative <- sapply(items$b[i, ], function(b) plogis(items$a[i] * (theta - b)))
    cumulative <- matrix(cumulative, length(theta))
    u <- runif(length(theta))
    as.integer(rowSums(u < cumulative))
  })
}

# The responses of the persons to each occasion's items, as a list of person-by-item matrices with one row per person
# present: `absent` lists, for each occasion, the persons who did not take part.
simulate_responses <- function(thetas, items, occasion_items, absent) {
  lapply(seq_along(occasion_items), function(t) {
    data <- respond(thetas[, t], items, occasion_items[[t]])
    data <- matrix(data, nrow(thetas), length(occasion_items[[t]]))
    colnames(data) <- items$ids[occasion_items[[t]]]
    rownames(data) <- sprintf("p%04d", seq_len(nrow(thetas)))
    if (length(absent[[t]]) > 0) data <- data[-absent[[t]], , drop = FALSE]
    data
  })
}

# --- mirt ---------------------------------------------------------------------------------------------------------------

# The wide data of mirt: every occasion's items as columns "<id>_t<occasion>", persons absent from an occasion NA.
wide_data <- function(occasions) {
  persons <- sprintf("p%04d", seq_len(max(sapply(occasions, function(m) max(as.integer(sub("p", "", rownames(m))))))))
  blocks <- lapply(seq_along(occasions), function(t) {
    block <- matrix(NA_integer_, length(persons), ncol(occasions[[t]]),
                    dimnames = list(persons, paste0(colnames(occasions[[t]]), "_t", t)))
    block[rownames(occasions[[t]]), ] <- occasions[[t]]
    block
  })
  do.call(cbind, blocks)
}

# mirt's model of the wide data: occasion t is dimension F<t>; the first has mean 0 and variance 1, the others free means,
# variances and covariances.
wide_model <- function(widths) {
  last <- cumsum(widths)
  first <- last - widths + 1
  lines <- sprintf("F%d = %d-%d", seq_along(widths), first, last)
  names <- paste0("F", seq_along(widths))
  covariances <- c(combn(names, 2, paste, collapse = "*"), paste0(names[-1], "*", names[-1]))
  mirt.model(paste(c(lines, paste("COV =", paste(covariances, collapse = ", ")),
                     paste("MEAN =", paste(names[-1], collapse = ", "))), collapse = "\n"))
}

# The constraints that tie an item's parameters across the occasions that have it, and with `common_slope` every slope to
# one: lists of parameter numbers of mirt's table.
constraints <- function(values, occasions, common_slope) {
  parnum <- function(column, name) values$parnum[values$item == column & values$name == name]
  ids <- unique(unlist(lapply(occasions, colnames)))
  ties <- list()
  slopes <- integer()
  for (id in ids) {
    where <- which(sapply(occasions, function(m) id %in% colnames(m)))
    columns <- paste0(id, "_t", where)
    slopes <- c(slopes, mapply(function(column, t) parnum(column, paste0("a", t)), columns, where))
    if (length(where) > 1 && !common_slope) {
      ties[[length(ties) + 1]] <- unname(mapply(function(column, t) parnum(column, paste0("a", t)), columns, where))
    }
    if (length(where) > 1) {
      for (name in grep("^d[0-9]*$", unique(values$name[values$item == columns[1]]), value = TRUE)) {
        ties[[length(ties) + 1]] <- unname(sapply(columns, function(column) parnum(column, name)))
      }
    }
  }
  if (common_slope) ties[[length(ties) + 1]] <- unname(slopes)
  ties
}

fit_mirt <- function(occasions, itemtype, quadpts, common_slope = FALSE, errors = FALSE, tolerance = TOL) {
  data <- wide_data(occasions)
  model <- wide_model(sapply(occasions, ncol))
  values <- mirt(data, model, itemtype = itemtype, pars = "values")
  ties <- constraints(values, occasions, common_slope)
  fit <- mirt(data, model, itemtype = itemtype, constrain = ties, quadpts = quadpts, TOL = tolerance,
              SE = errors, SE.type = "Oakes", verbose = FALSE, technical = list(NCYCLES = 20000))
  stopifnot(extract.mirt(fit, "converged"))
  list(fit = fit, data = data, model = model, constraints = ties)
}

# The cross-product errors at the estimates of a fit: the same model evaluated where it is, not estimated again.
crossproduct_errors <- function(run, itemtype, quadpts) {
  values <- mod2values(run$fit)
  mirt(run$data, run$model, itemtype = itemtype, pars = values, constrain = run$constraints, quadpts = quadpts,
       TOL = NaN, SE = TRUE, SE.type = "crossprod", verbose = FALSE)
}

# The items of mirt's fit as measurekit's, by id of first appearance: the slope on the occasion's dimension and the
# difficulty b = -d / a, or the increasing thresholds of the graded model.
mirt_items <- function(run, occasions, graded) {
  values <- coef(run$fit, simplify = TRUE)$items
  ids <- unique(unlist(lapply(occasions, colnames)))
  lapply(ids, function(id) {
    t <- which(sapply(occasions, function(m) id %in% colnames(m)))[1]
    row <- values[paste0(id, "_t", t), ]
    a <- row[[paste0("a", t)]]
    parameters <- if (graded) {
      list(a = a, b = I(unname(-row[grep("^d[0-9]+$", names(row))] / a)))
    } else {
      list(a = a, b = -row[["d"]] / a, c = 0, d = 1)
    }
    list(id = id, parameters = parameters)
  })
}

# The change between consecutive occasions, as the contrast of the two traits: its mean, standard deviation and
# correlation with the trait of the earlier occasion, from the means and the covariance matrix. Occasions count from 0.
occasion_changes <- function(means, covariance) {
  lapply(seq_len(nrow(covariance) - 1), function(t) {
    contrast <- start <- numeric(nrow(covariance))
    contrast[t:(t + 1)] <- c(-1, 1)
    start[t] <- 1
    variance <- drop(t(contrast) %*% covariance %*% contrast)
    list(from = t - 1, to = t, mean = sum(contrast * means), sd = sqrt(variance),
         correlationWithStart = drop(t(start) %*% covariance %*% contrast) / sqrt(covariance[t, t] * variance))
  })
}

# The moments of the occasions: means, standard deviations, correlations, the covariance matrix and the changes.
mirt_moments <- function(run) {
  moments <- coef(run$fit, simplify = TRUE)
  covariance <- moments$cov
  list(means = I(unname(moments$means)), sds = I(unname(sqrt(diag(covariance)))),
       correlations = unname(cov2cor(covariance)), covariance = unname(covariance),
       changes = occasion_changes(unname(moments$means), unname(covariance)))
}

mirt_statistics <- function(run) {
  list(logLikelihood = extract.mirt(run$fit, "logLik"), iterations = extract.mirt(run$fit, "iterations"))
}

# What a case of the fixture holds for measurekit: the occasions' responses by person and item, with the ids.
occasion_records <- function(occasions) {
  lapply(occasions, function(m) list(personIds = rownames(m), itemIds = colnames(m), responses = unname(m)))
}

# The standard errors of a fit of two occasions that share all their items, in the order of measurekit's `standardErrors`:
# the slope and the difficulty of each item, the correlation and the standard deviation of the second occasion and its
# mean. mirt's covariance of the estimates is of the slopes, the intercepts d, the mean and the covariance entries, so
# the difficulty -d / a, the correlation c21 / sqrt(c22) and the standard deviation sqrt(c22) come by the delta method.
mirt_errors <- function(fit) {
  v <- vcov(fit)
  items <- (ncol(v) - 3) / 2
  stopifnot(startsWith(colnames(v)[1:2], c("a", "d")), startsWith(colnames(v)[ncol(v) - 2:0], c("MEAN_2", "COV_21", "COV_22")))
  values <- mod2values(fit)
  first <- unique(sub("_t[0-9]+$", "", values$item[values$est & values$name == "d"]))[seq_len(items)]
  a <- sapply(first, function(id) values$value[values$item == paste0(id, "_t1") & values$name == "a1"])
  d <- sapply(first, function(id) values$value[values$item == paste0(id, "_t1") & values$name == "d"])
  delta <- function(gradient, indices) sqrt(drop(t(gradient) %*% v[indices, indices] %*% gradient))
  errors <- unlist(lapply(seq_len(items), function(i) {
    at <- 2 * i - 1:0
    c(sqrt(v[at[1], at[1]]), delta(c(d[i] / a[i]^2, -1 / a[i]), at))
  }))
  covariance <- values$value[values$name == "COV_21"]
  variance <- values$value[values$name == "COV_22"]
  entries <- ncol(v) - 1:0
  c(errors, delta(c(1 / sqrt(variance), -covariance / (2 * variance^1.5)), entries),
    delta(c(0, 1 / (2 * sqrt(variance))), entries), sqrt(v[ncol(v) - 2, ncol(v) - 2]))
}
