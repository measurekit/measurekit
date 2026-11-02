# Kernel-smoothed item curves (fit-kernel-curves.json): KernSmoothIRT 6.6's ksIRT with format = 2 and a key, so that each
# response is its own option and its score, on the responses of fit-rmsd.json, which the test reads. Expected values
# only, by case name.
#
# ksIRT treats a missing response as an option of its own, -1, of score 0. Its curve of a category is the kernel average
# over all persons; measurekit's is over the persons who answered the item, which is ksIRT's curve over the sum of the
# curves of the item's categories. The script checks that a base R kernel average over those persons gives the same.
#
# ksIRT's standard error is sqrt(Σ_j w_j² P (1 - P)) with P the running sum of the weighted responses up to person j,
# not the curve: its loop adds the term for person j before the later persons enter P. The script checks that the
# running sum reproduces ksIRT's standard errors, and writes Ramsay's (1991) pointwise error with the curve's P.
# ksIRT's bandwidth takes "Silverman" or "CV" alone in 6.6, as match.arg refuses a number, so the case with a bandwidth
# given is the base R kernel average.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages(library(KernSmoothIRT))

TOLERANCE <- 1e-12
inputs <- fromJSON("packages/psychometrics/test/fixtures/fit-rmsd.json", simplifyVector = TRUE)$cases

# Silverman's rule for N(0, 1) scores, as ksIRT: 1.06 n^(-1/5).
silverman <- function(persons) 1.06 * persons^(-0.2)

# The normal scores of the persons with a response: ranks of their totals, ties in the order of the persons, as ksIRT.
normal_scores <- function(X) {
  totals <- rowSums(X, na.rm = TRUE)
  qnorm(rank(totals, ties.method = "first") / (nrow(X) + 1))
}

nodes_of <- function(persons, points) seq(qnorm(1 / (persons + 1)), qnorm(persons / (persons + 1)), length.out = points)

# Ramsay's kernel average over the persons who answered each item, by item, then category, then node, with the
# pointwise standard error sqrt(Σ_j (w_j / Σ w)² P (1 - P)).
kernel_curves <- function(X, nodes, bandwidth) {
  theta <- normal_scores(X)
  lapply(seq_len(ncol(X)), function(i) {
    answered <- !is.na(X[, i])
    weights <- sapply(nodes, function(t) exp(-((t - theta[answered]) / bandwidth)^2 / 2))
    shares <- sweep(weights, 2, colSums(weights), "/")
    categories <- 0:max(X[answered, i])
    probabilities <- t(sapply(categories, function(k) colSums(shares * (X[answered, i] == k))))
    # 1 - P as the share of the other categories, which keeps its digits where P is within rounding of 1.
    others <- t(sapply(categories, function(k) colSums(shares * (X[answered, i] != k))))
    errors <- sqrt(sweep(probabilities * others, 2, colSums(shares^2), "*"))
    list(probabilities = probabilities, standardErrors = errors)
  })
}

# ksIRT's curves of the categories, over the sum of the curves of the responses, by item, then category, then node.
kernsmooth_curves <- function(fit, X) {
  lapply(seq_len(ncol(X)), function(i) {
    rows <- fit$OCC[fit$OCC[, 1] == i & fit$OCC[, 2] != -1, , drop = FALSE]
    rows <- rows[order(rows[, 2]), , drop = FALSE]
    stopifnot(identical(as.numeric(rows[, 2]), as.numeric(0:max(X[, i], na.rm = TRUE))))
    curves <- rows[, -(1:3), drop = FALSE]
    sweep(curves, 2, colSums(curves), "/")
  })
}

# ksIRT's standard error of a response's curve from the running sum of P, by item, then category, then node.
running_errors <- function(fit, X) {
  theta <- fit$subjtheta
  h <- fit$bandwidth[1]
  lapply(seq_len(ncol(X)), function(i) {
    t(sapply(0:max(X[, i], na.rm = TRUE), function(k) {
      chosen <- X[, i] %in% k
      sapply(fit$evalpoints, function(t) {
        weights <- exp(-((t - theta) / h)^2 / 2)
        weights <- weights / sum(weights)
        running <- cumsum(weights * chosen)
        sqrt(sum(weights^2 * running * (1 - running)))
      })
    }))
  })
}

kernsmooth_errors <- function(fit, X) {
  lapply(seq_len(ncol(X)), function(i) {
    rows <- fit$stderrs[fit$stderrs[, 1] == i & fit$stderrs[, 2] != -1, , drop = FALSE]
    rows[order(rows[, 2]), -(1:3), drop = FALSE]
  })
}

curve_case <- function(name, data, points = NULL, bandwidth = NULL) {
  X <- inputs$responses[[which(inputs$name == data)]]
  colnames(X) <- paste0("item", seq_len(ncol(X)))
  stopifnot(all(rowSums(!is.na(X)) > 0))
  for (i in seq_len(ncol(X))) stopifnot(all(0:max(X[, i], na.rm = TRUE) %in% X[, i]))
  persons <- nrow(X)
  h <- if (is.null(bandwidth)) silverman(persons) else bandwidth
  nodes <- nodes_of(persons, if (is.null(points)) 51 else points)
  base <- kernel_curves(X, nodes, h)
  if (is.null(bandwidth)) {
    # ksIRT weighs each response by its value only when it has a key; for format = 2 any key does.
    fit <- suppressWarnings(ksIRT(X, key = apply(X, 2, max, na.rm = TRUE), format = 2, nevalpoints = length(nodes)))
    stopifnot(all(fit$subjscore == rowSums(X, na.rm = TRUE)))
    stopifnot(abs(fit$bandwidth[1] - h) < TOLERANCE, max(abs(fit$evalpoints - nodes)) < TOLERANCE)
    stopifnot(max(abs(fit$subjtheta - normal_scores(X))) < TOLERANCE)
    reference <- kernsmooth_curves(fit, X)
    for (i in seq_along(base)) stopifnot(max(abs(reference[[i]] - base[[i]]$probabilities)) < TOLERANCE)
    running <- running_errors(fit, X)
    reported <- kernsmooth_errors(fit, X)
    for (i in seq_along(base)) stopifnot(max(abs(running[[i]] - reported[[i]])) < TOLERANCE)
    probabilities <- reference
  } else {
    probabilities <- lapply(base, `[[`, "probabilities")
  }
  options <- setNames(list(), character(0))
  if (!is.null(points)) options$points <- points
  if (!is.null(bandwidth)) options$bandwidth <- bandwidth
  list(
    name = name, data = data, options = options, bandwidth = h, nodes = nodes,
    probabilities = lapply(probabilities, function(p) unname(lapply(seq_len(nrow(p)), function(k) p[k, ]))),
    standardErrors = lapply(base, function(b) unname(lapply(seq_len(nrow(b$standardErrors)), function(k) b$standardErrors[k, ])))
  )
}

write_fixture("fit-kernel-curves", list(
  source = paste(
    "KernSmoothIRT 6.6's ksIRT(format = 2) on the responses of fit-rmsd.json, its curves over the persons who answered",
    "each item; Ramsay's pointwise standard errors and the case with a bandwidth in base R."
  ),
  cases = list(
    curve_case("dichotomous, sparse", "2pl, sparse"),
    curve_case("polytomous", "generalized partial credit"),
    curve_case("points and bandwidth", "partial credit", points = 21, bandwidth = 0.3)
  )
))
