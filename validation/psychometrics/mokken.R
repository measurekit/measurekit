# Mokken scale analysis fixtures. Reference: the mokken package (van der Ark, 2007, 2012): coefH for the scalability
# coefficients and their standard errors, aisp for the automated item selection, check.monotonicity and check.iio
# (method MIIO) for the monotonicity and invariant item ordering checks. mokken's summaries round their results, so the
# unrounded values are taken from the raw results of the checks, aggregated as the summary methods do, and the summaries
# are asserted to agree. Two rounded values change a count or a criterion: the monotonicity check's critical z is the
# constant 1.6449, and the summary of the invariant ordering rounds H_i to three decimals. The fixture holds the exact
# values and, where they differ, mokken's own under mokkenRounded.
#
# mokken stops at missing responses and zero-variance items; its checks use persons who answered every item, so the
# complete cases of each data set are what the references run on.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages(library(mokken))
coefHTiny <- mokken:::coefHTiny

SEED <- 20261029
ALPHA_LEVELS <- c(0.05, 0.01)
LOWER_BOUNDS <- c(0.3, 0.4, 0.5)

# --- Data ------------------------------------------------------------------------------------------------------------

data(acl, cavalini, transreas, DS14)

# Reverse-scores the items whose names end in "*", as mokken's documentation does for the adjective checklist.
reverse_starred <- function(X) {
  starred <- grep("\\*$", colnames(X))
  if (length(starred) > 0) X <- recode(X, items = starred, values = min(X, na.rm = TRUE):max(X, na.rm = TRUE))
  X
}

as_items <- function(X) {
  X <- as.matrix(X)
  storage.mode(X) <- "integer"
  X
}

# Two latent dimensions of six items each, and two items of pure noise, 500 persons: the number of persons at which
# mokken's default minimum group size changes in check.monotonicity and check.iio.
set.seed(SEED)
two_dimensions <- local({
  n <- 500
  theta <- cbind(rnorm(n), rnorm(n))
  difficulty <- c(-1.2, -0.7, -0.2, 0.3, 0.8, 1.3)
  responses <- sapply(1:12, function(i) {
    dimension <- if (i <= 6) 1 else 2
    rbinom(n, 1, plogis(1.7 * (theta[, dimension] - difficulty[(i - 1) %% 6 + 1])))
  })
  X <- cbind(responses, rbinom(n, 1, 0.3), rbinom(n, 1, 0.6))
  colnames(X) <- paste0("i", 1:14)
  X
})

# One dimension of eight items; items 3 and 4 have the same number of persons passing them, so that their item steps tie.
tied <- local({
  n <- 300
  theta <- rnorm(n)
  X <- sapply(1:8, function(i) rbinom(n, 1, plogis(1.5 * (theta - seq(-1.4, 1.4, length.out = 8)[i]))))
  X[, 4] <- X[, 3]
  swapped <- sample(n, n %/% 2)
  X[swapped, 4] <- sample(X[swapped, 3])
  stopifnot(sum(X[, 3]) == sum(X[, 4]))
  colnames(X) <- paste0("t", 1:8)
  X
})

DS14_items <- reverse_starred(as.matrix(DS14[, 3:16]))

cases <- list(
  list(name = "acl communality", X = reverse_starred(as_items(acl[, 1:10]))),
  list(name = "acl four scales", X = reverse_starred(as_items(acl[, 1:40]))),
  list(name = "cavalini", X = as_items(cavalini)),
  list(name = "transitive reasoning", X = as_items(transreas[, 2:13])),
  list(name = "DS14", X = as_items(DS14_items)),
  list(name = "two dimensions", X = as_items(two_dimensions)),
  list(name = "tied item steps", X = as_items(tied))
)

# --- References ------------------------------------------------------------------------------------------------------

# mokken's criterion (Molenaar and Sijtsma, 2000) from the aggregated counts of an item, before rounding.
criterion <- function(item_h, ac, vi, maxvi, total, zmax, zsig) {
  crit <- 50 * (0.3 - item_h) + sqrt(vi) + 100 * (vi / ac) + 100 * maxvi + 10 * sqrt(total) + 1000 * (total / ac) +
    5 * zmax + 10 * sqrt(zsig) + 100 * zsig / ac
  if (vi == 0) 0 else crit
}

# The violations of an item whose z exceeds a critical value, counted from the table of rest score groups that
# check.monotonicity returns, with the formulas of its source. mokken's critical value is the constant 1.6449.
MOKKEN_CRITICAL <- 1.6449

count_significant <- function(groups, m, minvi, critical) {
  freq <- groups[, 5:(m + 4), drop = FALSE]
  total <- 0
  for (i in 1:(m - 1)) {
    p <- groups[, m + 5 + i]
    V <- outer(p, p, "-")
    V[row(V) <= col(V)] <- 0
    V[V >= -minvi] <- 0
    if (max(abs(V)) > minvi) {
      below <- rowSums(freq[, 1:i, drop = FALSE])
      above <- rowSums(freq[, (i + 1):m, drop = FALSE])
      Z <- abs(sign(-V) * 2 * (sqrt(outer(above + 1, below + 1)) - sqrt(outer(below, above))) /
                 sqrt(outer(above, below, "+") + outer(below, above, "+") - 1))
      total <- total + sum(Z > critical)
    }
  }
  total
}

monotonicity_reference <- function(X, minvi, minsize) {
  result <- suppressWarnings(
    if (is.null(minsize)) check.monotonicity(X, minvi = minvi) else check.monotonicity(X, minvi = minvi, minsize = minsize))
  m <- result$m
  rows <- lapply(seq_along(result$results), function(j) {
    total <- result$results[[j]][[3]][m, ]
    groups <- result$results[[j]][[2]]
    stopifnot(count_significant(groups, m, minvi, MOKKEN_CRITICAL) == total[10])
    exact <- count_significant(groups, m, minvi, qnorm(0.95))
    criterion_with <- function(significant) {
      floor(max(0, criterion(result$Hi[j], total[1], total[2], total[4], total[5], total[7], significant)))
    }
    row <- list(id = result$I.labels[j], activeComparisons = unname(total[1]), violations = unname(total[2]),
                maximumViolation = unname(total[4]), violationSum = unname(total[5]), maximumZ = unname(total[7]),
                significantViolations = exact, criterion = criterion_with(exact))
    # The fixture keeps what mokken reports where its rounded critical value changes the count.
    if (exact != total[10]) {
      row$mokkenRounded <- list(significantViolations = unname(total[10]), criterion = criterion_with(total[10]))
    }
    row
  })
  summarized <- summary(result)
  reported <- sapply(rows, function(r) if (is.null(r$mokkenRounded)) r$criterion else r$mokkenRounded$criterion)
  stopifnot(all(reported == summarized[, "crit"]))
  list(minimumViolation = minvi, minimumGroupSize = minsize, items = rows)
}

# The aggregation of check.iio's results by item, as summary.iio.class does it, unrounded.
iio_reference <- function(X, alpha, selection, minvi, minsize) {
  args <- list(X, method = "MIIO", alpha = alpha, item.selection = selection)
  if (!is.null(minvi)) args$minvi <- minvi
  if (!is.null(minsize)) args$minsize <- minsize
  set.seed(SEED)
  result <- suppressWarnings(do.call(check.iio, args))
  J <- ncol(X)
  labels <- dimnames(result$violations)[[1]]
  aggregated <- matrix(0, J, 6, dimnames = list(labels, c("ac", "vi", "sum", "tsig", "maxvi", "tmax")))
  k <- 0
  for (i in 1:(J - 1)) for (j in (i + 1):J) {
    k <- k + 1
    pair <- result$results[[k]][[3]][1, ]
    for (item in c(i, j)) {
      aggregated[item, c("ac", "vi", "sum", "tsig")] <- aggregated[item, c("ac", "vi", "sum", "tsig")] +
        pair[c(1, 2, 5, 8)]
      aggregated[item, "maxvi"] <- max(aggregated[item, "maxvi"], pair[4])
      aggregated[item, "tmax"] <- max(aggregated[item, "tmax"], pair[7])
    }
  }
  rows <- lapply(seq_len(J), function(item) {
    a <- aggregated[item, ]
    criterion_with <- function(item_h) {
      floor(criterion(item_h, a["ac"], a["vi"], a["maxvi"], a["sum"], a["tmax"], a["tsig"]))
    }
    row <- list(id = labels[item], activeComparisons = unname(a["ac"]), violations = unname(a["vi"]),
                maximumViolation = unname(a["maxvi"]), violationSum = unname(a["sum"]), maximumZ = unname(a["tmax"]),
                significantViolations = unname(a["tsig"]), criterion = criterion_with(result$Hi[item]))
    # summary.iio.class rounds the item's H to three decimals before it computes the criterion, which the fixture
    # keeps where rounding changes the criterion.
    rounded <- criterion_with(round(result$Hi[item], 3))
    if (rounded != row$criterion) row$mokkenRounded <- list(criterion = rounded)
    row
  })
  summarized <- summary(result)$item.summary
  reported <- sapply(rows, function(r) if (is.null(r$mokkenRounded)) r$criterion else r$mokkenRounded$criterion)
  stopifnot(all(reported == summarized[, "crit"]))
  # items.removed counts the columns in the order check.iio sorts them: by item mean, the highest first.
  sorted_labels <- colnames(X)[rev(order(colMeans(X)))]
  removed <- sorted_labels[result$items.removed]
  list(alpha = alpha, selection = selection, minimumViolation = minvi, minimumGroupSize = minsize, items = rows,
       removed = I(removed), coefficient = unname(result$HT))
}

analyse <- function(case) {
  X <- case$X
  complete <- X[complete.cases(X), , drop = FALSE]
  coefficients <- coefH(complete, se = TRUE, nice.output = FALSE)
  selections <- list()
  # aisp stops when the lower bound is above every H_ij, which no scale could reach.
  off_diagonal <- coefHTiny(complete)$Hij
  diag(off_diagonal) <- 0
  for (alpha in ALPHA_LEVELS) for (bound in LOWER_BOUNDS) {
    if (bound > max(off_diagonal)) next
    assigned <- suppressWarnings(aisp(complete, lowerbound = bound, alpha = alpha))
    selections[[length(selections) + 1]] <- list(lowerBound = bound, alpha = alpha, scales = as.integer(assigned[, 1]))
  }
  monotonicity <- list(monotonicity_reference(complete, 0.03, NULL), monotonicity_reference(complete, 0.05, 60))
  invariant <- list()
  if (ncol(complete) >= 3) {
    invariant <- list(iio_reference(complete, 0.05, TRUE, NULL, NULL), iio_reference(complete, 0.05, FALSE, NULL, NULL),
                      iio_reference(complete, 0.01, TRUE, NULL, 60))
  }
  list(
    name = case$name,
    itemIds = I(colnames(X)),
    responses = unname(X),
    scalability = list(
      pairs = unname(coefficients$Hij), pairErrors = unname(coefficients$se.Hij), items = as.numeric(coefficients$Hi),
      itemErrors = as.numeric(coefficients$se.Hi), scale = as.numeric(coefficients$H),
      scaleError = as.numeric(coefficients$se.H)),
    selections = selections,
    monotonicity = monotonicity,
    invariantOrdering = invariant
  )
}

write_fixture("mokken", list(
  source = paste("mokken's acl (the four scales of ten adjectives, reverse-scored where starred), cavalini, transreas,",
                 "DS14 (with its missing responses), and two simulations, set.seed(20261029): two dimensions and noise",
                 "items for 500 persons, and eight items with tied item steps for 300."),
  cases = lapply(cases, analyse)
))
