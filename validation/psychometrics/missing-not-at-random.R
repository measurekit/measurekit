# Responses missing not at random, the recipe of multidimensional.md (Holman and Glas, 2005): the indicator of whether
# each item was answered is an item of a second dimension, the propensity to respond, which correlates with the ability.
# The reference is mirt's confirmatory two-dimensional 2PL of the same combined responses. The data are simulated with
# set.seed(20261102): 1,500 persons, 8 two-parameter items, and 8 indicators, whose propensity correlates 0.5 with the
# ability, so that the persons of low ability omit more.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages({
  library(mirt)
  library(mvtnorm)
})

TOL <- 1e-10 # mirt's EM tolerance, tight so that its estimates are the optimum
POINTS <- 31 # mirt's default quadrature for two dimensions
persons <- 1500
items <- 8

set.seed(20261102)
a <- runif(items, 0.8, 1.6)
b <- seq(-1.4, 1.4, length.out = items)
slopes <- runif(items, 0.8, 1.8)
intercepts <- seq(-0.5, 1.5, length.out = items)
traits <- rmvnorm(persons, sigma = matrix(c(1, 0.5, 0.5, 1), 2))
correct <- matrix(rbinom(persons * items, 1, plogis(sweep(outer(traits[, 1], b, "-"), 2, a, "*"))), persons, items)
answered <- matrix(rbinom(persons * items, 1, plogis(outer(traits[, 2], slopes) + rep(intercepts, each = persons))),
                   persons, items)
responses <- correct
responses[answered == 0] <- NA
combined <- cbind(responses, answered)
colnames(combined) <- c(sprintf("q%02d", seq_len(items)), sprintf("answered:q%02d", seq_len(items)))

fit <- mirt(combined, mirt.model(sprintf("ability = 1-%d\npropensity = %d-%d\nCOV = ability*propensity", items,
                                         items + 1, 2 * items)),
            itemtype = "2PL", quadpts = POINTS, TOL = TOL, verbose = FALSE, technical = list(NCYCLES = 5000))
stopifnot(extract.mirt(fit, "converged"))
values <- coef(fit, simplify = TRUE)

write_fixture("multidimensional-not-at-random", list(
  source = paste(
    "Simulated (set.seed(20261102)): 1,500 persons, 8 two-parameter items with a propensity to answer each, correlated 0.5",
    "with the ability. The missingness of an item is the response of an indicator item on the propensity."
  ),
  model = "m2pl", points = POINTS, itemIds = colnames(combined), responses = unname(combined),
  loadings = setNames(lapply(seq_len(2 * items), function(i) I(if (i <= items) "ability" else "propensity")),
                      colnames(combined)),
  dimensionNames = c("ability", "propensity"),
  items = lapply(seq_len(2 * items), function(i) list(
    id = colnames(combined)[i],
    parameters = list(a = I(unname(values$items[i, c("a1", "a2")])), intercept = unname(values$items[i, "d"]), c = 0, d = 1)
  )),
  correlations = unname(values$cov),
  logLikelihood = extract.mirt(fit, "logLik"), aic = extract.mirt(fit, "AIC"), bic = extract.mirt(fit, "BIC"),
  truth = list(correlation = 0.5, share = mean(answered), slopes = a, intercepts = -a * b)
))
