# Nested-model chi-squares, the recipe of irt.md: the likelihood ratio of two calibrations of the same responses, as
# mirt's anova gives it. Two pairs: the 1PL, which is the 2PL with one slope for all items, against the 2PL; and the
# partial credit model, which is the generalized partial credit model with one slope, against that model. The data are
# simulated with set.seed(20261102): 1,500 persons and 12 dichotomous items with unequal slopes for the first pair, 1,500
# persons and 8 items of three categories with one slope for the second, which the partial credit model holds.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages(library(mirt))

TOL <- 1e-10 # mirt's EM tolerance, tight so that its estimates are the optimum
POINTS <- 61 # mirt's default quadrature, which measurekit's default is

fit_mirt <- function(data, itemtype, ...) {
  fit <- mirt(data, 1, itemtype = itemtype, TOL = TOL, quadpts = POINTS, verbose = FALSE,
              technical = list(NCYCLES = 1e5), ...)
  stopifnot(extract.mirt(fit, "converged"))
  fit
}

# What the recipe compares: the log-likelihood and the number of estimated parameters of each fit, and mirt's anova.
pair <- function(source, data, restricted, general) {
  comparison <- anova(restricted, general)
  list(
    source = source,
    itemIds = colnames(data), responses = unname(data),
    restricted = list(logLikelihood = extract.mirt(restricted, "logLik"), parameters = extract.mirt(restricted, "nest")),
    general = list(logLikelihood = extract.mirt(general, "logLik"), parameters = extract.mirt(general, "nest")),
    chiSquare = comparison$X2[2], degreesOfFreedom = comparison$df[2], pValue = comparison$p[2]
  )
}

set.seed(20261102)
items <- 12
slopes <- simdata(a = matrix(runif(items, 0.5, 2)), d = rnorm(items), N = 1500, itemtype = "dich")
colnames(slopes) <- sprintf("item%02d", seq_len(items))
values <- mirt(slopes, 1, itemtype = "2PL", pars = "values")
equal <- fit_mirt(slopes, "2PL", constrain = list(values$parnum[values$name == "a1"]))
free <- fit_mirt(slopes, "2PL")

steps <- 8
credit <- simdata(a = matrix(rep(1.1, steps)), d = cbind(0, rnorm(steps, 0.4), rnorm(steps, -0.4)), N = 1500,
                  itemtype = "gpcm")
colnames(credit) <- sprintf("item%02d", seq_len(steps))
credit_values <- mirt(credit, 1, itemtype = "gpcm", pars = "values")

write_fixture("irt-nested-models", list(
  slopes = pair("simdata: 1,500 persons, 12 items with slopes from 0.5 to 2. The 1PL is mirt's 2PL with one slope.",
                slopes, equal, free),
  credit = pair("simdata: 1,500 persons, 8 items of three categories with one slope. The PCM is mirt's gpcm with one slope.",
                credit, fit_mirt(credit, "gpcm", constrain = list(credit_values$parnum[credit_values$name == "a1"])),
                fit_mirt(credit, "gpcm"))
))
