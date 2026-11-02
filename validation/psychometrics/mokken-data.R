# The data sets of the Mokken scale analysis fixtures: mokken's example data and seeded simulations. Sourced by
# mokken.R, mokken-reliability.R and mokken-steps.R, which run their references on the complete cases of each.

suppressPackageStartupMessages(library(mokken))

SEED <- 20261029

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
