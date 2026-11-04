# The parametric bootstrap of Yen's Q3 (Christensen, Makransky and Horton, 2017) with mirt, for fit-q3-bootstrap.json.
# The data: 500 persons and 10 2PL items, items 1 to 3 sharing a passage whose effect has the SD 1.2, 8% of the
# responses missing completely at random. mirt calibrates them at its defaults, which are measurekit's: 61 points on
# [-6, 6] and a tolerance of 1e-4. Each of 1000 replications draws standard normal thetas and responses from the fitted
# items, sets missing the responses that the data lack, draws again where an item lacks a category, calibrates them
# the same way, and takes Q3 at mirt's EAP of each person: residuals(type = "Q3", Theta = ). A pair's excess is its
# Q3 less the mean Q3 of the pairs, and each replication gives its largest. R's generator is not measurekit's, so the
# test compares the distribution of the largest excess, its mean and SD, within four Monte Carlo standard errors.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages(library(mirt))

set.seed(20261104)
POINTS <- 61
TOL <- 1e-4
PERSONS <- 500
ITEMS <- 10
PASSAGE <- 1:3
PASSAGE_SD <- 1.2
MISSING_SHARE <- 0.08
REPLICATIONS <- 1000

a <- round(runif(ITEMS, 0.8, 2), 2)
b <- round(rnorm(ITEMS), 2)
theta <- rnorm(PERSONS)
effect <- rnorm(PERSONS, 0, PASSAGE_SD)
responses <- sapply(seq_len(ITEMS), function(j) {
  rbinom(PERSONS, 1, plogis(a[j] * (theta + (if (j %in% PASSAGE) effect else 0) - b[j])))
})
responses[matrix(runif(PERSONS * ITEMS) < MISSING_SHARE, PERSONS, ITEMS)] <- NA
colnames(responses) <- sprintf("item%d", seq_len(ITEMS))

rows_of <- function(data) unname(lapply(seq_len(nrow(data)), function(i) as.numeric(data[i, ])))
calibrate_2pl <- function(data) mirt(data, 1, itemtype = "2PL", quadpts = POINTS, TOL = TOL, verbose = FALSE)

pair_excesses <- function(fit, data) {
  thetas <- fscores(fit, method = "EAP", quadpts = POINTS, theta_lim = c(-6, 6), verbose = FALSE)
  q3 <- residuals(fit, type = "Q3", Theta = thetas, verbose = FALSE)
  values <- q3[upper.tri(q3)]
  values <- values[!is.na(values)]
  list(mean = mean(values), excesses = values - mean(values))
}

observed_fit <- calibrate_2pl(responses)
stopifnot(extract.mirt(observed_fit, "converged"))
coefficients <- coef(observed_fit, IRTpars = TRUE, simplify = TRUE)$items
fitted_a <- unname(coefficients[, "a"])
fitted_b <- unname(coefficients[, "b"])
observed <- pair_excesses(observed_fit, responses)

missing <- is.na(responses)
largest <- numeric(0)
redrawn <- 0
while (length(largest) < REPLICATIONS) {
  draws <- rnorm(PERSONS)
  replicated <- sapply(seq_len(ITEMS), function(j) rbinom(PERSONS, 1, plogis(fitted_a[j] * (draws - fitted_b[j]))))
  replicated[missing] <- NA
  colnames(replicated) <- colnames(responses)
  if (any(apply(replicated, 2, function(column) length(unique(na.omit(column))) < 2))) {
    redrawn <- redrawn + 1
    next
  }
  largest <- c(largest, max(pair_excesses(calibrate_2pl(replicated), replicated)$excesses))
}
message("Q3 bootstrap: ", REPLICATIONS, " replications, ", redrawn, " redrawn; mean of the largest excess ",
        signif(mean(largest), 4), ", SD ", signif(sd(largest), 4))

write_fixture("fit-q3-bootstrap", list(
  source = paste("500 persons and 10 2PL items, items 1 to 3 in a passage of SD 1.2, 8% missing, set.seed(20261104);",
                 "mirt 1.48's 2PL at 61 points and TOL 1e-4, EAP thetas and residuals(type = 'Q3', Theta = ); 1000",
                 "parametric bootstrap replications from the fitted items, each item drawn again where it lacks a",
                 "category, the largest pair excess of each."),
  itemIds = colnames(responses),
  responses = rows_of(responses),
  items = lapply(seq_len(ITEMS), function(j) list(a = fitted_a[j], b = fitted_b[j])),
  observedMeanQ3 = observed$mean,
  observedExcesses = observed$excesses,
  replications = REPLICATIONS,
  largestMean = mean(largest),
  largestSd = sd(largest),
  largestQuantiles = list(median = unname(quantile(largest, 0.5)), upper = unname(quantile(largest, 0.95)))
))
