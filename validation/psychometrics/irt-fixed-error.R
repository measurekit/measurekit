# The standard errors of a calibration with fixed items that carry the fixed items' sampling error
# (irt-fixed-error.json). Reference: the pseudo maximum likelihood variance of restricted recalibration (Liu, Yang and
# Maydeu-Olivares, 2019; Gong and Samaniego, 1981), V + J S J', written out in base R. The marginal log-likelihood of
# 2PL items on mirt's 61 nodes, in measurekit's parameters (a, b) and the mean and sd of a normal distribution of
# theta, has its gradient by Fisher's identity and its Hessian over the estimated and the fixed parameters together by
# central differences of that gradient: V = (-H22)^-1, J = V H21. A second route checks J: the derivative of the
# estimates, maximized again with each fixed parameter moved, by central differences. S is the covariance of the fixed
# items' estimates from a calibration of other persons, the inverse of the negative Hessian of that calibration.
#
# Two cases: the fixed items calibrated on 800 other persons, and 1,500 persons of N(0.3, 1.1^2) with 5% of their
# responses missing whose normal distribution is estimated with 3 pretest items; and the fixed items calibrated on the
# same 1,500 persons of N(0, 1), whose distribution stays standard normal while the 3 pretest items are estimated.

source("validation/psychometrics/common.R")

POINTS <- 61
NODES <- seq(-6, 6, length.out = POINTS)

normal_weights <- function(mean, sd) {
  w <- dnorm(NODES, mean, sd)
  w / sum(w)
}

# Each person's posterior over the nodes and the log-likelihood, for items of slopes `a` and difficulties `b`.
posteriors <- function(x, a, b, log_weights) {
  traces <- plogis(outer(NODES, b, function(theta, location) theta - location) * matrix(a, POINTS, length(a), byrow = TRUE))
  log_joint <- matrix(log_weights, nrow(x), POINTS, byrow = TRUE)
  for (i in seq_along(a)) {
    answered <- !is.na(x[, i])
    log_joint[answered, ] <- log_joint[answered, ] + outer(x[answered, i], log(traces[, i])) +
      outer(1 - x[answered, i], log(1 - traces[, i]))
  }
  largest <- apply(log_joint, 1, max)
  posterior <- exp(log_joint - largest)
  totals <- rowSums(posterior)
  list(posterior = posterior / totals, logLikelihood = sum(largest + log(totals)), traces = traces)
}

# The problem of a calibration: parameters `items` (indices) of a and b in that order, then the mean and sd when
# `normal`; every other item at `a` and `b`.
problem <- function(x, items, a, b, normal) {
  unpack <- function(p) {
    a[items] <- p[seq_along(items)]
    b[items] <- p[length(items) + seq_along(items)]
    list(a = a, b = b, mean = if (normal) p[2 * length(items) + 1] else 0, sd = if (normal) p[2 * length(items) + 2] else 1)
  }
  gradient <- function(p) {
    u <- unpack(p)
    e <- posteriors(x, u$a, u$b, log(normal_weights(u$mean, u$sd)))
    item_gradient <- sapply(items, function(i) {
      answered <- !is.na(x[, i])
      residual <- colSums(e$posterior[answered, , drop = FALSE] *
        (x[answered, i] - matrix(e$traces[, i], sum(answered), POINTS, byrow = TRUE)))
      c(sum(residual * (NODES - u$b[i])), -u$a[i] * sum(residual))
    })
    latent <- if (normal) {
      counts <- colSums(e$posterior)
      excess <- counts - sum(counts) * normal_weights(u$mean, u$sd)
      c(sum(excess * (NODES - u$mean)) / u$sd^2, sum(excess * (NODES - u$mean)^2) / u$sd^3)
    }
    c(item_gradient[1, ], item_gradient[2, ], latent)
  }
  logLikelihood <- function(p) {
    u <- unpack(p)
    if (any(u$a <= 0) || u$sd <= 0) return(-Inf)
    value <- posteriors(x, u$a, u$b, log(normal_weights(u$mean, u$sd)))$logLikelihood
    if (is.finite(value)) value else -Inf
  }
  list(unpack = unpack, gradient = gradient, logLikelihood = logLikelihood)
}

hessian <- function(gradient, p) {
  h <- sapply(seq_along(p), function(j) {
    step <- 1e-5 * max(1, abs(p[j]))
    up <- p
    down <- p
    up[j] <- up[j] + step
    down[j] <- down[j] - step
    (gradient(up) - gradient(down)) / (2 * step)
  })
  (h + t(h)) / 2
}

# Newton's method to a gradient below 1e-9, halving a step that lowers the log-likelihood or leaves the parameters'
# range; `curvature`, a Hessian near the maximum, replaces the Hessian of each step.
maximize <- function(problem, start, curvature = NULL) {
  p <- start
  for (iteration in 1:200) {
    g <- problem$gradient(p)
    if (max(abs(g)) < 1e-9) return(p)
    h <- if (is.null(curvature)) hessian(problem$gradient, p) else curvature
    direction <- if (all(eigen(h, symmetric = TRUE, only.values = TRUE)$values < 0)) -solve(h, g) else g
    current <- problem$logLikelihood(p)
    length <- 1
    # Rounding of a sum of thousands of terms: near the maximum a Newton step changes it by less.
    while (problem$logLikelihood(p + length * direction) < current - 1e-10 * abs(current) && length > 1e-10) {
      length <- length / 2
    }
    p <- p + length * direction
  }
  stop("Newton's method did not converge")
}

simulate <- function(persons, a, b, mean, sd, missing) {
  theta <- rnorm(persons, mean, sd)
  x <- sapply(seq_along(a), function(i) rbinom(persons, 1, plogis(a[i] * (theta - b[i]))))
  x[matrix(runif(length(x)) < missing, persons)] <- NA
  x
}

fixed_error_case <- function(name, calibration_data, x, operational, normal) {
  ids <- sprintf("q%02d", seq_len(ncol(x)))
  colnames(x) <- ids
  # The fixed items' calibration: every operational item estimated, theta standard normal.
  first <- problem(calibration_data, operational, rep(1, length(operational)), rep(0, length(operational)), FALSE)
  first_start <- c(rep(1, length(operational)), rep(0, length(operational)))
  estimates <- maximize(first, first_start)
  fixed_covariance <- solve(-hessian(first$gradient, estimates))
  fixed_a <- estimates[seq_along(operational)]
  fixed_b <- estimates[length(operational) + seq_along(operational)]
  pretest <- setdiff(seq_len(ncol(x)), operational)
  a <- rep(1, ncol(x))
  b <- rep(0, ncol(x))
  a[operational] <- fixed_a
  b[operational] <- fixed_b
  # The calibration with the operational items fixed, and the joint problem over its parameters and the fixed ones.
  second <- problem(x, pretest, a, b, normal)
  start <- c(rep(1, length(pretest)), rep(0, length(pretest)), if (normal) c(0, 1))
  estimated <- maximize(second, start)
  size <- length(estimated)
  joint <- problem(x, c(pretest, operational), a, b, normal)
  order <- c(seq_along(pretest), length(pretest) + length(operational) + seq_along(pretest), # pretest a, b
             if (normal) 2 * (length(pretest) + length(operational)) + 1:2, # mean, sd
             length(pretest) + seq_along(operational), # operational a
             2 * length(pretest) + length(operational) + seq_along(operational)) # operational b
  point <- c(estimated[seq_along(pretest)], fixed_a, estimated[length(pretest) + seq_along(pretest)], fixed_b,
             if (normal) estimated[2 * length(pretest) + 1:2])
  full <- hessian(joint$gradient, point)[order, order]
  h22 <- full[seq_len(size), seq_len(size)]
  h21 <- full[seq_len(size), -seq_len(size)]
  known <- solve(-h22)
  derivative <- known %*% h21
  # The second route: the estimates maximized again with each fixed parameter moved.
  moved <- sapply(seq_len(2 * length(operational)), function(m) {
    step <- 1e-4
    at <- function(shift) {
      a_moved <- a
      b_moved <- b
      k <- (m - 1) %% length(operational) + 1
      if (m <= length(operational)) a_moved[operational[k]] <- a[operational[k]] + shift
      else b_moved[operational[k]] <- b[operational[k]] + shift
      maximize(problem(x, pretest, a_moved, b_moved, normal), estimated, h22)
    }
    (at(step) - at(-step)) / (2 * step)
  })
  difference <- max(abs(moved - derivative)) / max(abs(derivative))
  message(name, ": the two routes to J differ by ", signif(difference, 3), " of its largest entry")
  stopifnot(difference < 1e-4)
  corrected <- known + derivative %*% fixed_covariance %*% t(derivative)
  # measurekit's order of the covariance: by item, a then b.
  item_order <- as.vector(rbind(seq_along(pretest), length(pretest) + seq_along(pretest)))
  labels <- function(indices) {
    unlist(lapply(indices, function(i) list(list(item = ids[i], name = "a"), list(item = ids[i], name = "b"))),
           recursive = FALSE)
  }
  fixed_order <- as.vector(rbind(seq_along(operational), length(operational) + seq_along(operational)))
  latent_errors <- function(covariance) {
    if (!normal) return(NULL)
    list(mean = sqrt(covariance[2 * length(pretest) + 1, 2 * length(pretest) + 1]),
         sd = sqrt(covariance[2 * length(pretest) + 2, 2 * length(pretest) + 2]))
  }
  list(
    name = name,
    itemIds = ids,
    responses = unname(x),
    latent = if (normal) "normal" else "standard-normal",
    fixed = setNames(lapply(seq_along(operational), function(k) list(a = fixed_a[k], b = fixed_b[k], c = 0, d = 1)),
                     ids[operational]),
    fixedCovariance = list(parameters = labels(operational),
                           matrix = unname(fixed_covariance[fixed_order, fixed_order])),
    items = lapply(seq_along(pretest), function(k) {
      list(id = ids[pretest[k]], a = estimated[k], b = estimated[length(pretest) + k])
    }),
    groups = if (normal) list(mean = estimated[2 * length(pretest) + 1], sd = estimated[2 * length(pretest) + 2]),
    covariance = list(parameters = labels(pretest), matrix = unname(corrected[item_order, item_order])),
    knownCovariance = unname(known[item_order, item_order]),
    groupErrors = latent_errors(corrected),
    knownGroupErrors = latent_errors(known)
  )
}

set.seed(20261104)
operational <- 1:8
a <- round(runif(11, 0.8, 2), 2)
b <- round(rnorm(11, 0, 1), 2)
other_persons <- simulate(800, a[operational], b[operational], 0, 1, 0)
independent <- fixed_error_case(
  "8 items fixed at a calibration of 800 other persons; 3 pretest items and a normal distribution estimated",
  other_persons, simulate(1500, a, b, 0.3, 1.1, 0.05), operational, TRUE
)
same <- simulate(1500, a, b, 0, 1, 0)
same_persons <- fixed_error_case(
  "8 items fixed at a calibration of the same persons; 3 pretest items estimated, theta standard normal",
  same[, operational], same, operational, FALSE
)

write_fixture("irt-fixed-error", list(cases = list(independent, same_persons)))
