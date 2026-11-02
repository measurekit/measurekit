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
# mirt's default for Jmin: a score group needs more than this many persons of each group.
JMIN <- 5L
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
  if (!is.finite(row[["chiSquare"]])) row[] <- NA
  c(row, persons = sum(used))
}

# The statistics of every item, as columns.
sibtest_items <- function(data, focal, anchors = seq_len(ncol(data)), ...) {
  rows <- sapply(seq_len(ncol(data)), function(item) sibtest_set(data, focal, item, anchors, ...))
  as.list(as.data.frame(t(rows)))
}

# The fewest anchors at which each anchor is tested on the others: the studied anchor and two matching items with the
# correction, as coefficient alpha needs two; without it, two when an item is dichotomous, as a score group needs a
# score between the lowest and the highest, else one. An item's highest score is its highest response, at least 1.
minimum_anchors <- function(data, correction = TRUE) {
  highest <- pmax(1, suppressWarnings(apply(data, 2, max, na.rm = TRUE)))
  if (correction || any(highest < 2)) 3L else 2L
}

# Purification as difR's: flag at 0.05, compute again on the items not flagged, until a round flags the items the
# previous one did, too few anchors are left, or 10 rounds have run. `floored` says whether too few anchors stopped it,
# where difR, which has no floor, goes on.
purified <- function(data, focal, correction = TRUE, ...) {
  anchors <- seq_len(ncol(data))
  result <- sibtest_items(data, focal, anchors, correction = correction, ...)
  flagged <- function(statistics) !is.na(statistics$significance) & statistics$significance < ALPHA
  flags <- flagged(result)
  iterations <- 0
  converged <- !any(flags)
  floor <- minimum_anchors(data, correction)
  while (!converged && iterations < 10 && sum(!flags) >= floor) {
    iterations <- iterations + 1
    anchors <- which(!flags)
    result <- sibtest_items(data, focal, anchors, correction = correction, ...)
    following <- flagged(result)
    converged <- all(following == flags)
    flags <- following
  }
  c(result, list(anchors = I(anchors), iterations = iterations, converged = converged,
                 floored = !converged && iterations < 10))
}

# On complete data the statistics are difSIBTEST's, and so is the purification where the floor does not stop it.
check_difr <- function(data, focal, statistics, purified_statistics) {
  plain <- difSIBTEST(data, group = focal, focal.name = 1, type = "udif")
  stopifnot(max(abs(plain$Beta - statistics$beta), na.rm = TRUE) < 1e-12,
            max(abs(plain$X2 - statistics$chiSquare), na.rm = TRUE) < 1e-12)
  if (purified_statistics$floored) return(invisible())
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

# input: the fixture and the case whose responses these are, and whether they are its sparse responses. bundles: the
# sets of studied items, by name, as 1-based columns.
sibtest_case <- function(input, data, focal, bundles) {
  statistics <- sibtest_items(data, focal)
  pure <- purified(data, focal)
  if (!anyNA(data)) check_difr(data, focal, statistics, pure)
  pure$floored <- NULL
  uncorrected_pure <- purified(data, focal, correction = FALSE)
  uncorrected_pure$floored <- NULL
  list(
    input = input,
    default = statistics,
    uncorrected = sibtest_items(data, focal, correction = FALSE),
    guessing = sibtest_items(data, focal, guessing = GUESSING),
    smallCells = sibtest_items(data, focal, jmin = 2L),
    largeCells = sibtest_items(data, focal, jmin = 20L),
    purified = pure,
    purifiedUncorrected = uncorrected_pure,
    bundles = lapply(names(bundles), function(name) {
      bundle_entry(data, focal, name, bundles[[name]], pure$anchors)
    })
  )
}

dichotomous <- jsonlite::fromJSON("packages/psychometrics/test/fixtures/dif.json", simplifyVector = FALSE)$cases
polytomous <- jsonlite::fromJSON("packages/psychometrics/test/fixtures/dif-polytomous.json", simplifyVector = FALSE)$cases

case_named <- function(cases, name) Filter(function(case) case$name == name, cases)[[1]]

# The case `name` of a fixture, with its complete or its sparse responses.
fixture_case <- function(fixture, name, sparse, bundles) {
  case <- case_named(if (fixture == "dif") dichotomous else polytomous, name)
  responses <- if (sparse) case$sparse$responses else case$responses
  input <- list(fixture = fixture, case = name, sparse = sparse)
  sibtest_case(input, matrix_of(responses), unlist(case$focal), bundles)
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

verbal_case <- case_named(dichotomous, "verbal")
verbal <- matrix_of(verbal_case$responses)
colnames(verbal) <- paste0("item", seq_len(ncol(verbal)))
simulated_case <- case_named(dichotomous, "simulated")
simulated <- matrix_of(simulated_case$responses)
colnames(simulated) <- paste0("item", seq_len(ncol(simulated)))

write_fixture("dif-sibtest", list(
  source = paste("mirt's SIBTEST and difR's difSIBTEST on the cases of dif.json and dif-polytomous.json, complete and",
                 "with 2% of the responses missing; difStd and sirt's dif.variance for the recipes of dif.md."),
  guessing = GUESSING,
  cases = list(
    verbal = fixture_case("dif", "verbal", FALSE, verbal_bundles),
    "verbal, sparse" = fixture_case("dif", "verbal", TRUE, verbal_bundles),
    simulated = fixture_case("dif", "simulated", FALSE, simulated_bundles),
    "simulated, sparse" = fixture_case("dif", "simulated", TRUE, simulated_bundles),
    bfi = fixture_case("dif-polytomous", "bfi", FALSE, polytomous_bundles),
    "bfi, sparse" = fixture_case("dif-polytomous", "bfi", TRUE, polytomous_bundles),
    graded = fixture_case("dif-polytomous", "simulated", FALSE, simulated_bundles),
    "graded, sparse" = fixture_case("dif-polytomous", "simulated", TRUE, simulated_bundles)
  ),
  recipes = list(verbal = recipes(verbal, unlist(verbal_case$focal)),
                 simulated = recipes(simulated, unlist(simulated_case$focal)))
))
