# SIBTEST fixtures (dif-sibtest.json). References: mirt's SIBTEST for each item and each bundle (suspect_set), matched on
# the anchors less the studied items, on the persons who answered them; difR's difSIBTEST(type = "udif"), which calls
# it, for the items and their purification. The recipes of dif.md: difR's difStd for STD P-DIF, and sirt's dif.variance
# for the DIF variance and the empirical Bayes estimates. Data: the cases of dif.json and dif-polytomous.json, so run
# dif.R first, and with 2% of their responses missing where a case has them.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages({
  library(difR)
  library(mirt)
  library(sirt)
})

ALPHA <- 0.05
# mirt's default for Jmin, the persons of each group a score group must exceed.
JMIN <- 5L
# The fewest anchors purification goes on with: the correction's coefficient alpha needs two matching items.
MINIMUM_ANCHORS <- 2L
# The chance level of a multiple-choice item with five options, and the guessing case's option.
GUESSING <- 0.2

# A response matrix from the rows of a fixture, with NA for null.
matrix_of <- function(rows) {
  do.call(rbind, lapply(rows, function(row) vapply(row, function(x) if (is.null(x)) NA_real_ else x, 0)))
}

# Beta, SE, chi-square and p of one set of studied items. mirt refuses any missing response, and reads only the studied
# and matching items and the number of columns; so the persons are those who answered these items, and any other
# column is set to 0. Beta and SE are undefined without a score group to use, where mirt gives 0 and a NaN chi-square.
sibtest_set <- function(data, focal, studied, anchors, guessing = 0, jmin = JMIN, correction = TRUE) {
  matching <- setdiff(anchors, studied)
  used <- complete.cases(data[, c(matching, studied), drop = FALSE])
  complete <- data[used, , drop = FALSE]
  complete[is.na(complete)] <- 0
  result <- suppressWarnings(SIBTEST(complete, focal[used], suspect_set = studied, match_set = matching,
                                     focal_name = 1, guess_correction = guessing, Jmin = jmin, correction = correction))
  row <- c(beta = result$beta[1], standardError = result$SE[1], chiSquare = result$X2[1], significance = result$p[1])
  if (is.finite(row[["chiSquare"]])) row else c(beta = NA, standardError = NA, chiSquare = NA, significance = NA)
}

# The statistics of every item, as columns.
sibtest_items <- function(data, focal, anchors = seq_len(ncol(data)), ...) {
  rows <- sapply(seq_len(ncol(data)), function(item) sibtest_set(data, focal, item, anchors, ...))
  as.list(as.data.frame(t(rows)))
}

# Purification as difR's: flag at 0.05, compute again on the items not flagged, until a round flags the items the
# previous one did, too few anchors are left, or 10 rounds have run.
purified <- function(data, focal, ...) {
  anchors <- seq_len(ncol(data))
  result <- sibtest_items(data, focal, anchors, ...)
  flagged <- function(statistics) !is.na(statistics$significance) & statistics$significance < ALPHA
  flags <- flagged(result)
  iterations <- 0
  converged <- !any(flags)
  while (!converged && iterations < 10 && sum(!flags) >= MINIMUM_ANCHORS) {
    iterations <- iterations + 1
    anchors <- which(!flags)
    result <- sibtest_items(data, focal, anchors, ...)
    following <- flagged(result)
    converged <- all(following == flags)
    flags <- following
  }
  c(result, list(anchors = I(anchors), iterations = iterations, converged = converged))
}

# On complete data the statistics are difSIBTEST's, purified and not.
check_difr <- function(data, focal, statistics, purified_statistics) {
  plain <- difSIBTEST(data, group = focal, focal.name = 1, type = "udif")
  stopifnot(max(abs(plain$Beta - statistics$beta), na.rm = TRUE) < 1e-12,
            max(abs(plain$X2 - statistics$chiSquare), na.rm = TRUE) < 1e-12)
  pure <- difSIBTEST(data, group = focal, focal.name = 1, type = "udif", purify = TRUE)
  stopifnot(max(abs(pure$Beta - purified_statistics$beta), na.rm = TRUE) < 1e-12,
            pure$nrPur == purified_statistics$iterations, pure$convergence == purified_statistics$converged)
}

# One bundle's name, items from 0 and statistics, matched on all items and on the anchors of purification.
bundle_entry <- function(data, focal, name, items, purified_anchors) {
  list(name = name, items = I(items - 1),
       default = as.list(sibtest_set(data, focal, items, seq_len(ncol(data)))),
       purified = as.list(sibtest_set(data, focal, items, purified_anchors)))
}

# bundles: the sets of studied items, by name, as 1-based columns.
sibtest_case <- function(name, data, focal, bundles) {
  statistics <- sibtest_items(data, focal)
  pure <- purified(data, focal)
  if (!anyNA(data)) check_difr(data, focal, statistics, pure)
  list(
    name = name,
    default = statistics,
    uncorrected = sibtest_items(data, focal, correction = FALSE),
    guessing = sibtest_items(data, focal, guessing = GUESSING),
    smallCells = sibtest_items(data, focal, jmin = 2L),
    largeCells = sibtest_items(data, focal, jmin = 20L),
    purified = pure,
    bundles = lapply(names(bundles), function(name) {
      bundle_entry(data, focal, name, bundles[[name]], pure$anchors)
    })
  )
}

dichotomous <- jsonlite::fromJSON("packages/psychometrics/test/fixtures/dif.json", simplifyVector = FALSE)$cases
polytomous <- jsonlite::fromJSON("packages/psychometrics/test/fixtures/dif-polytomous.json", simplifyVector = FALSE)$cases

fixture_case <- function(case, responses, name, bundles) {
  sibtest_case(name, matrix_of(responses), unlist(case$focal), bundles)
}

# verbal's items of cursing and its items of what a person would do, not want to do; in the simulations, the items with
# DIF and five clean ones; in bfi, whose five items leave few to match on, two items.
verbal_bundles <- list(curse = seq(1, 22, by = 3), do = 13:24)
simulated_bundles <- list(shifted = 1:2, clean = 5:9)
polytomous_bundles <- list(first = 1:2)

# --- The recipes of dif.md ---------------------------------------------------------------------------------------

# STD P-DIF with the focal group's weights: difStd's PDIF. The matching score holds the studied item, as mantelTest's.
standardization <- function(data, focal) {
  difStd(data, group = focal, focal.name = 1, stdWeight = "focal")$PDIF
}

# dif.variance of MH D-DIF. sirt returns lambda (d - mean) + (1 - lambda) mean as eb.dif, which is the shrinkage of d
# towards its mean only when the mean is 0; so it takes the centered values, and the mean is added back to the result.
dif_variance <- function(data, focal) {
  result <- difMH(data, group = focal, focal.name = 1)
  delta <- -2.35 * log(result$alphaMH)
  se <- 2.35 * sqrt(result$varLambda)
  centered <- dif.variance(delta - mean(delta), se, items = colnames(data))
  uncentered <- dif.variance(delta, se, items = colnames(data))
  stopifnot(abs(centered$weighted.DIFSD - uncentered$weighted.DIFSD) < 1e-12)
  list(
    standardDeviation = centered$weighted.DIFSD,
    shrunken = centered$eb.dif + mean(delta)
  )
}

recipes <- function(data, focal) {
  list(standardization = standardization(data, focal), variance = dif_variance(data, focal))
}

verbal <- matrix_of(dichotomous[[1]]$responses)
colnames(verbal) <- paste0("item", seq_len(ncol(verbal)))
simulated <- matrix_of(dichotomous[[2]]$responses)
colnames(simulated) <- paste0("item", seq_len(ncol(simulated)))

write_fixture("dif-sibtest", list(
  source = paste("mirt's SIBTEST and difR's difSIBTEST on the cases of dif.json and dif-polytomous.json, complete and",
                 "with 2% of the responses missing; difStd and sirt's dif.variance for the recipes of dif.md."),
  guessing = GUESSING,
  cases = list(
    fixture_case(dichotomous[[1]], dichotomous[[1]]$responses, "verbal", verbal_bundles),
    fixture_case(dichotomous[[1]], dichotomous[[1]]$sparse$responses, "verbal, sparse", verbal_bundles),
    fixture_case(dichotomous[[2]], dichotomous[[2]]$responses, "simulated", simulated_bundles),
    fixture_case(dichotomous[[2]], dichotomous[[2]]$sparse$responses, "simulated, sparse", simulated_bundles),
    fixture_case(polytomous[[1]], polytomous[[1]]$responses, "bfi", polytomous_bundles),
    fixture_case(polytomous[[1]], polytomous[[1]]$sparse$responses, "bfi, sparse", polytomous_bundles),
    fixture_case(polytomous[[2]], polytomous[[2]]$responses, "graded", simulated_bundles),
    fixture_case(polytomous[[2]], polytomous[[2]]$sparse$responses, "graded, sparse", simulated_bundles)
  ),
  recipes = list(verbal = recipes(verbal, unlist(dichotomous[[1]]$focal)),
                 simulated = recipes(simulated, unlist(dichotomous[[2]]$focal)))
))
