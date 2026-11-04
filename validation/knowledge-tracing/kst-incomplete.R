# Writes kst-diff-test-incomplete.json: the test of a quasi order's fit on incomplete response patterns.
#
# DAKS takes complete patterns only, so the reference is written out in base R. Each pair of items counts the learners
# who answered both, each item's solution rate the learners who answered it, and the fit measure is the mean over the
# ordered pairs that some learner answered both of: the squared difference of the observed share of learners who fail
# one item and solve the other and the share IITA's corrected model expects, with the error rate of the minimized or the
# corrected version. The estimate is written as a function of a weight for each learner, and its variance is the delta
# method: the sum of the squared derivatives with respect to the weights, which numDeriv's grad takes, the minimized
# error rate found again at each weight. The script asserts that, with every response present, the estimate is DAKS's
# z_test estimate and the variance that of kst-diff-test.json's delta method over the patterns.
#
# Data: DAKS's pisa with 20% of the responses missing completely at random, with the alternative of IITA's quasi order
# without one covering pair, the test of one edge; seven simulated items answered by 300 learners in two booklets that
# share five items, so the two others are never answered together, with 15% more missing; and pisa complete.

source("validation/knowledge-tracing/common.R")
suppressPackageStartupMessages({
  library(DAKS)
  library(sets)
  library(numDeriv)
})

SEED <- 20261104
LEVEL <- 0.9

relation_of <- function(imp, m) {
  relation <- diag(m) == 1
  for (pair in lapply(as.list(imp), function(t) unlist(as.list(t)))) relation[pair[1], pair[2]] <- TRUE
  relation
}

# The fit measure over n², at learner weights w, of data with NA for a missing response.
estimate_of <- function(w, data, relation, version) {
  m <- ncol(data)
  seen <- !is.na(data)
  solved <- ifelse(seen, data, 0)
  rho <- colSums(w * solved) / colSums(w * seen)
  C <- matrix(0, m, m)
  observed <- matrix(FALSE, m, m)
  for (i in 1:m) for (j in 1:m) if (i != j) {
    both <- seen[, i] & seen[, j]
    if (any(both)) {
      observed[i, j] <- TRUE
      C[i, j] <- sum(w[both] * (data[both, i] == 0 & data[both, j] == 1)) / sum(w[both])
    }
  }
  pairs <- which(relation & !diag(m) & observed, arr.ind = TRUE)
  if (version == 1) {
    x <- rep(0, 4)
    for (i in 1:m) for (j in 1:m) if (i != j && observed[i, j]) {
      if (relation[i, j]) {
        x[2] <- x[2] - 2 * C[i, j] * rho[j]
        x[4] <- x[4] + 2 * rho[j]^2
      } else if (relation[j, i]) {
        x[1] <- x[1] - 2 * C[i, j] * rho[i] + 2 * rho[i] * rho[j] - 2 * rho[i]^2
        x[3] <- x[3] + 2 * rho[i]^2
      }
    }
    gamma <- -(x[1] + x[2]) / (x[3] + x[4])
  } else {
    gamma <- mean(C[pairs] / rho[pairs[, 2]])
  }
  total <- 0
  for (i in 1:m) for (j in 1:m) if (i != j && observed[i, j]) {
    expected <- if (relation[i, j]) gamma * rho[j]
                else if (relation[j, i]) rho[j] - rho[i] + gamma * rho[i]
                else (1 - rho[i]) * rho[j]
    total <- total + (C[i, j] - expected)^2
  }
  total / sum(observed)
}

delta_test <- function(data, relations, version) {
  value <- function(w) {
    estimates <- sapply(relations, function(relation) estimate_of(w, data, relation, version))
    estimates[1] - (if (length(estimates) > 1) estimates[2] else 0)
  }
  w <- rep(1, nrow(data))
  list(estimate = value(w), variance = sum(grad(value, w)^2))
}

tail_values <- function(estimate, variance, mu, level) {
  se <- sqrt(variance)
  z <- (estimate - mu) / se
  list(z = z,
       pValue = list(`two-sided` = 2 * min(pnorm(z), 1 - pnorm(z)), less = pnorm(z), greater = 1 - pnorm(z)),
       interval = list(`two-sided` = estimate + c(-1, 1) * qnorm(1 - (1 - level) / 2) * se,
                       less = estimate + qnorm(level) * se, greater = estimate - qnorm(level) * se))
}

prerequisites_of <- function(relation, items) {
  setNames(lapply(seq_along(items), function(j) I(items[which(relation[, j] & seq_along(items) != j)])), items)
}

# A pattern as a string of 0, 1 and "." for a missing response, with the number of learners who gave it.
patterns_of <- function(data) {
  key <- apply(data, 1, function(row) paste(ifelse(is.na(row), ".", row), collapse = ""))
  as.list(table(factor(key, levels = unique(key))))
}

# The relation without one of its covering pairs: i before j with no item between them.
without_cover <- function(relation) {
  m <- nrow(relation)
  for (i in 1:m) for (j in 1:m) if (i != j && relation[i, j]) {
    between <- any(sapply(setdiff(1:m, c(i, j)), function(k) relation[i, k] && relation[k, j]))
    if (!between) {
      relation[i, j] <- FALSE
      return(relation)
    }
  }
  stop("The relation has no covering pair.")
}

set.seed(SEED)
data(pisa)
complete <- as.matrix(pisa)
amputated <- complete
amputated[matrix(runif(length(complete)) < 0.2, nrow(complete))] <- NA

simulated <- as.matrix(simu(7, 300, 0.05, 0.05, delta = 0.2)$dataset)
booklets <- simulated
second_booklet <- runif(nrow(booklets)) < 0.5
booklets[second_booklet, 6] <- NA
booklets[!second_booklet, 7] <- NA
booklets[matrix(runif(length(booklets)) < 0.15, nrow(booklets))] <- NA
# Each item is solved by someone, so every solution rate is above 0.
stopifnot(all(colSums(booklets == 1, na.rm = TRUE) > 0), all(colSums(amputated == 1, na.rm = TRUE) > 0))

datasets <- list(
  list(name = "pisa with 20% missing", data = amputated, quasi = complete),
  list(name = "two booklets of seven items", data = booklets, quasi = simulated),
  list(name = "pisa complete", data = complete, quasi = complete)
)

# The delta method over the complete patterns of kst-diagnostics.R, which the weights must reproduce.
complete_cases <- jsonlite::fromJSON("packages/knowledge-tracing/test/fixtures/kst-diff-test.json", simplifyVector = FALSE)
cases <- list()
for (d in datasets) {
  items <- paste0("i", seq_len(ncol(d$data)))
  for (version in 1:2) {
    chosen <- relation_of(iita(d$quasi, v = version)$implications, ncol(d$data))
    for (alternative in list(NULL, without_cover(chosen))) {
      relations <- c(list(chosen), if (!is.null(alternative)) list(alternative))
      delta <- delta_test(d$data, relations, version)
      if (d$name == "pisa complete" && is.null(alternative)) {
        daks <- z_test(d$data, iita(d$quasi, v = version)$implications, v = version)
        stopifnot(abs(delta$estimate - daks$diff_value) < 1e-12 * delta$estimate)
        reference <- Filter(function(case) case$name == "pisa" && case$version == c("minimized", "corrected")[version] &&
                              is.null(case$alternative) && case$mean == 0, complete_cases)[[1]]
        stopifnot(abs(delta$variance - reference$variance / nrow(d$data)) < 1e-6 * delta$variance)
      }
      for (mu in c(0, 0.001)) {
        cases[[length(cases) + 1]] <- c(
          list(name = d$name, items = I(items), patterns = patterns_of(d$data),
               version = c("minimized", "corrected")[version], prerequisites = prerequisites_of(chosen, items),
               alternative = if (!is.null(alternative)) prerequisites_of(alternative, items),
               mean = mu, confidence = LEVEL, estimate = delta$estimate, variance = delta$variance),
          tail_values(delta$estimate, delta$variance, mu, LEVEL))
      }
    }
  }
}
write_fixture("kst-diff-test-incomplete", cases)
