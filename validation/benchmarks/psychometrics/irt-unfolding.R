# Calibration of the generalized graded unfolding model, 2,000 persons and 15 items of 4 categories simulated from it,
# against mirt's ggum item type at measurekit's quadrature and tolerance, 61 points on [-6, 6] and 1e-4. The GGUM
# package, the reference of the validation, needs Java through its import of xlsx and is not timed here.

suppressPackageStartupMessages(library(mirt))

set.seed(20261102)
persons <- 2000
item_count <- 15
highest <- 3

# --- Data ----------------------------------------------------------------------------------------------------------

# P(z | θ) of the GGUM, with thresholds t_1 to t_C and M = 2C + 1: the sum of its two subjective responses z and M - z.
ggum_probabilities <- function(a, b, t, theta) {
  z <- 0:highest
  cumulative <- matrix(cumsum(c(0, t)), length(theta), highest + 1, byrow = TRUE)
  weights <- exp(a * (outer(theta - b, z) - cumulative)) + exp(a * (outer(theta - b, 2 * highest + 1 - z) - cumulative))
  weights / rowSums(weights)
}

a <- round(runif(item_count, 0.7, 2), 3)
b <- round(seq(-2, 2, length.out = item_count), 3)
t <- lapply(seq_len(item_count), function(i) round(sort(runif(highest, -2, -0.4), decreasing = TRUE), 3))
theta <- rnorm(persons)
responses <- sapply(seq_len(item_count), function(i) {
  rowSums(runif(persons) > t(apply(ggum_probabilities(a[i], b[i], t[[i]], theta), 1, cumsum))[, seq_len(highest)])
})
dimnames(responses) <- list(NULL, ids(item_count))

bench_fixture("psychometrics", "irt-unfolding", list(
  source = "A simulation of the GGUM, set.seed(20261102): 2,000 persons, 15 items of 4 categories.",
  itemIds = ids(item_count),
  responses = unname(responses)
))

# --- References -----------------------------------------------------------------------------------------------------

bench_task("irt.calibrateUnfolding", "Calibration of the generalized graded unfolding model",
           sprintf("%s persons, %d items of 4 categories", count(persons), item_count),
           reference("mirt", "mirt(itemtype = \"ggum\")"),
           function() suppressWarnings(mirt(responses, 1, itemtype = "ggum", quadpts = 61, TOL = 1e-4,
                                            technical = list(NCYCLES = 2000), verbose = FALSE)),
           note = "measurekit fits in the GGUM package's three stages from its starts, and mirt from its own starts")
