# Rater agreement fixtures. References: irr (kappa2, kappam.fleiss, icc) and psych (cohen.kappa, ICC) as a check.
# Data: irr's anxiety (20 subjects, 3 raters, ratings 1 to 6), recoded 0 to 5, and diagnoses (30 subjects, 6 raters,
# 5 categories), coded 0 to 4; anxiety again with two ratings missing.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages({
  library(irr)
  library(psych)
})

data(anxiety, diagnoses, package = "irr")
anxiety <- as.matrix(anxiety) - 1
diagnoses <- sapply(diagnoses, function(x) as.integer(substr(as.character(x), 1, 1)) - 1)
sparse <- anxiety
sparse[3, 2] <- NA
sparse[11, 1] <- NA

icc_values <- function(x) {
  coefficient <- function(model, type, unit) icc(x, model = model, type = type, unit = unit)$value
  values <- list(
    oneWay = coefficient("oneway", "consistency", "single"),
    agreement = coefficient("twoway", "agreement", "single"),
    consistency = coefficient("twoway", "consistency", "single"),
    oneWayAverage = coefficient("oneway", "consistency", "average"),
    agreementAverage = coefficient("twoway", "agreement", "average"),
    consistencyAverage = coefficient("twoway", "consistency", "average")
  )
  reference <- suppressMessages(ICC(na.omit(x), lmer = FALSE))$results$ICC
  stopifnot(max(abs(unlist(values) - reference[c(1, 2, 3, 4, 5, 6)])) < 1e-12)
  values
}

kappa_values <- function(x) {
  values <- list(
    none = kappa2(x, "unweighted")$value,
    linear = kappa2(x, "equal")$value,
    quadratic = kappa2(x, "squared")$value
  )
  stopifnot(abs(values$none - cohen.kappa(na.omit(x))$kappa) < 1e-12,
            abs(values$quadratic - cohen.kappa(na.omit(x))$weighted.kappa) < 1e-12)
  values
}

ratings <- function(x) unname(lapply(seq_len(nrow(x)), function(i) as.numeric(x[i, ])))

write_fixture("rater", list(
  source = "irr's anxiety, recoded 0 to 5, also with two ratings missing; irr's diagnoses, coded 0 to 4.",
  cases = list(
    list(name = "anxiety", ratings = ratings(anxiety), cohen = kappa_values(anxiety[, 1:2]),
         fleiss = kappam.fleiss(anxiety)$value, icc = icc_values(anxiety)),
    list(name = "anxiety with missing ratings", ratings = ratings(sparse), cohen = kappa_values(sparse[, 1:2]),
         fleiss = kappam.fleiss(sparse)$value, icc = icc_values(sparse)),
    list(name = "diagnoses", ratings = ratings(diagnoses), cohen = kappa_values(diagnoses[, 1:2]),
         fleiss = kappam.fleiss(diagnoses)$value, icc = icc_values(diagnoses))
  )
))
