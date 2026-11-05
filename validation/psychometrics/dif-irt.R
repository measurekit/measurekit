# IRT DIF fixtures (dif-irt.json). References: mirt's multipleGroup for the likelihood-ratio tests, the reference group
# N(0, 1) and the focal group's mean and variance free, with mirt's DIF(scheme = "drop") as a check; difR's difRaju on
# ltm's 2PL calibrations of each group, run to convergence, for Raju's areas, with base R's integrate as a check of the
# unsigned area; DFIT's Ncdif, Cdif and Dtf for differential functioning. Data: the cases of dif.json and
# dif-polytomous.json, so run dif.R first, and a seeded set of items for DFIT.

source("validation/psychometrics/common.R")
source("validation/psychometrics/ltm-items.R")
suppressPackageStartupMessages({
  library(DFIT)
  library(difR)
  library(mirt)
})

ALPHA <- 0.05
# mirt's default rectangular rule, which measurekit's default quadrature reproduces.
QUADRATURE_POINTS <- 61
# EM is run well past mirt's default 1e-4, so that the chi-squares are exact to about 1e-5: on the few hundred persons of the
# reduced cases the likelihood is flat enough that 1e-7 leaves the two EM runs 4e-4 apart in a chi-square.
TOLERANCE <- 1e-9
MAX_ROUNDS <- 10
# Each anchor is tested against the other anchors, of which one links the focal group's mean and variance.
MINIMUM_ANCHORS <- 2

# A response matrix from the rows of a fixture, with NA for null.
matrix_of <- function(rows) {
  data <- do.call(rbind, lapply(rows, function(row) vapply(row, function(x) if (is.null(x)) NA_real_ else x, 0)))
  colnames(data) <- paste0("item", seq_len(ncol(data)))
  data
}

# --- Likelihood ratio --------------------------------------------------------------------------------------------

# The multiple-group model with the items `anchors` equal across groups and the others free in each.
fit_groups <- function(data, groups, itemtype, anchors) {
  invariance <- c(colnames(data)[anchors], "free_means", "free_var")
  model <- multipleGroup(data, 1, group = groups, itemtype = itemtype, invariance = invariance,
                         quadpts = QUADRATURE_POINTS, TOL = TOLERANCE, optimizer = "NR",
                         technical = list(NCYCLES = 20000), verbose = FALSE)
  list(logLik = extract.mirt(model, "logLik"), converged = extract.mirt(model, "converged"), model = model)
}

# Whether both groups answered every category of an item from 0 to its highest response.
categories_in_both <- function(data, focal, item) {
  highest <- max(data[, item], na.rm = TRUE)
  all(vapply(0:1, function(g) all(0:highest %in% data[focal == g, item]), TRUE))
}

# The test of every item with the items `anchors` equal across groups: an anchor against the model that frees it, an
# item not an anchor against the model that constrains it too.
lr_items <- function(data, focal, groups, itemtype, anchors) {
  constrained <- fit_groups(data, groups, itemtype, anchors)
  tests <- lapply(seq_len(ncol(data)), function(item) {
    degrees <- sum(extract.item(extract.group(constrained$model, 1), item)@est)
    if (!categories_in_both(data, focal, item)) {
      return(c(chiSquare = NA, degreesOfFreedom = degrees, significance = NA))
    }
    other <- fit_groups(data, groups, itemtype, if (item %in% anchors) setdiff(anchors, item) else c(anchors, item))
    larger <- if (item %in% anchors) other else constrained
    smaller <- if (item %in% anchors) constrained else other
    if (!(larger$converged && smaller$converged)) {
      return(c(chiSquare = NA, degreesOfFreedom = degrees, significance = NA))
    }
    chi <- max(0, 2 * (larger$logLik - smaller$logLik))
    c(chiSquare = chi, degreesOfFreedom = degrees, significance = pchisq(chi, degrees, lower.tail = FALSE))
  })
  list(tests = as.list(as.data.frame(do.call(rbind, tests))), constrained = constrained)
}

# Purification as difR's: flag at 0.05, test again with the items flagged free, until a round flags the items the
# previous one did, fewer than MINIMUM_ANCHORS items are left unflagged, or 10 rounds have run.
purified_lr <- function(data, focal, groups, itemtype) {
  anchors <- seq_len(ncol(data))
  result <- lr_items(data, focal, groups, itemtype, anchors)
  flagged <- function(tests) !is.na(tests$significance) & tests$significance < ALPHA
  flags <- flagged(result$tests)
  iterations <- 0
  converged <- !any(flags)
  while (!converged && iterations < MAX_ROUNDS && sum(!flags) >= MINIMUM_ANCHORS) {
    iterations <- iterations + 1
    anchors <- which(!flags)
    result <- lr_items(data, focal, groups, itemtype, anchors)
    following <- flagged(result$tests)
    converged <- all(following == flags)
    flags <- following
  }
  list(result = result, anchors = anchors, iterations = iterations, converged = converged)
}

# The focal group's mean and standard deviation in a multiple-group model.
focal_distribution <- function(model) {
  parameters <- coef(model)$focal$GroupPars
  list(mean = parameters["par", "MEAN_1"], sd = sqrt(parameters["par", "COV_11"]))
}

# The case `name` of a fixture with its complete or sparse responses, tested under `model`, measurekit's name. The tests
# of every item take as many fits of the multiple-group model as the case has items, each as long as its persons and items,
# so a case is reduced to every `every`-th person, which keeps both groups and the pattern of the missing responses, and to
# the `items` at these positions, by seconds in the tests: each case records what it keeps.
lr_case <- function(fixture, name, sparse, model, purify, every, items) {
  case <- case_named(if (fixture == "dif") dichotomous else polytomous, name)
  data <- matrix_of(if (sparse) case$sparse$responses else case$responses)
  focal <- unlist(case$focal)
  kept <- seq(1, nrow(data), by = every)
  data <- data[kept, items, drop = FALSE]
  focal <- focal[kept]
  groups <- factor(ifelse(focal == 1, "focal", "reference"), levels = c("reference", "focal"))
  itemtype <- switch(model, "2pl" = "2PL", grm = "graded", gpcm = "gpcm")
  default <- lr_items(data, focal, groups, itemtype, seq_len(ncol(data)))
  # mirt's own DIF, on the model with every item constrained, frees each item in turn: the same tests.
  which_par <- grep("^(a1|d[0-9]*)$", names(coef(default$constrained$model)[[1]][[1]][1, ]), value = TRUE)
  check <- DIF(default$constrained$model, which.par = which_par, scheme = "drop", verbose = FALSE,
               technical = list(NCYCLES = 20000), TOL = TOLERANCE)
  stopifnot(max(abs(check$X2 - default$tests$chiSquare), na.rm = TRUE) < 1e-2)
  entry <- list(
    input = list(fixture = fixture, case = name, sparse = sparse, every = every, items = I(items)),
    model = model,
    default = default$tests,
    logLikelihood = default$constrained$logLik,
    focal = focal_distribution(default$constrained$model)
  )
  if (purify) {
    pure <- purified_lr(data, focal, groups, itemtype)
    entry$purified <- c(pure$result$tests, list(anchors = I(pure$anchors), iterations = pure$iterations,
                                                converged = pure$converged,
                                                logLikelihood = pure$result$constrained$logLik))
  }
  entry
}

# --- Raju's areas ------------------------------------------------------------------------------------------------

# The unsigned area between two 2PL curves by numerical integration.
integrated_area <- function(reference, focal) {
  difference <- function(theta) abs(plogis(focal[1] * (theta - focal[2])) - plogis(reference[1] * (theta - reference[2])))
  integrate(difference, -Inf, Inf, rel.tol = 1e-12, subdivisions = 1000)$value
}

# difRaju's signed and unsigned statistics on the converged ltm calibrations of each group, linked over `anchors`.
raju_statistics <- function(reference, focal_parameters, anchors) {
  linked <- itemRescale(reference, focal_parameters, items = anchors)
  signed <- RajuZ(reference, linked, signed = TRUE)$res
  unsigned <- RajuZ(reference, linked, signed = FALSE)$res
  # difR's unsigned area is negative where the focal slope is the lower: the area is its absolute value.
  areas <- vapply(seq_len(nrow(reference)), function(i) integrated_area(reference[i, 1:2], linked[i, 1:2]), 0)
  stopifnot(max(abs(abs(unsigned[, "H"]) - areas)) < 1e-6)
  list(signedArea = signed[, "H"], signedZ = signed[, "Z"], unsignedArea = abs(unsigned[, "H"]),
       unsignedZ = abs(unsigned[, "Z"]), significance = 2 * pnorm(-abs(unsigned[, "Z"])),
       difrUnsignedArea = unsigned[, "H"], difrUnsignedZ = unsigned[, "Z"],
       difrSignificance = 2 * (1 - pnorm(unsigned[, "Z"])))
}

raju_case <- function(name) {
  case <- case_named(dichotomous, name)
  data <- matrix_of(case$responses)
  focal <- unlist(case$focal)
  reference <- item_parameters_2pl(data[focal == 0, ])
  focal_parameters <- item_parameters_2pl(data[focal == 1, ])
  parameters <- rbind(reference, focal_parameters)
  default <- difRaju(data, group = focal, focal.name = 1, model = "2PL", irtParam = parameters, same.scale = FALSE)
  statistics <- raju_statistics(reference, focal_parameters, seq_len(ncol(data)))
  stopifnot(max(abs(default$RajuZ - statistics$difrUnsignedZ)) < 1e-12)
  pure <- difRaju(data, group = focal, focal.name = 1, model = "2PL", irtParam = parameters, same.scale = FALSE,
                  purify = TRUE, alpha = ALPHA)
  anchors <- if (pure$nrPur == 0) seq_len(ncol(data)) else which(pure$difPur[pure$nrPur, ] == 0)
  purified <- raju_statistics(reference, focal_parameters, anchors)
  stopifnot(max(abs(pure$RajuZ - purified$difrUnsignedZ)) < 1e-12)
  list(input = list(fixture = "dif", case = name, sparse = FALSE), default = statistics,
       purified = c(purified, list(anchors = I(anchors), iterations = pure$nrPur, converged = pure$convergence)))
}

# --- Differential functioning ------------------------------------------------------------------------------------

set.seed(20261102)
DFIT_ITEMS <- 8
DFIT_PERSONS <- 500
# The focal group's thetas, sorted: the population's nodes, each of weight 1 / 500.
focal_thetas <- sort(rnorm(DFIT_PERSONS, -0.5, 1))
two_pl <- cbind(a = rlnorm(DFIT_ITEMS, 0, 0.3), b = rnorm(DFIT_ITEMS))
graded <- cbind(a = rlnorm(DFIT_ITEMS, 0, 0.3), t(apply(matrix(rnorm(3 * DFIT_ITEMS), DFIT_ITEMS), 1, sort)))
colnames(graded) <- c("a", "b1", "b2", "b3")
# Item 1 is harder for the focal group, item 2 discriminates less for it, the others are equal.
with_dif <- function(parameters) {
  shifted <- parameters
  shifted[1, -1] <- shifted[1, -1] + 0.5
  shifted[2, 1] <- shifted[2, 1] * 0.6
  shifted
}

dfit_case <- function(model, reference) {
  irt_model <- switch(model, "2pl" = "2pl", grm = "grm")
  parameters <- list(focal = with_dif(reference), reference = reference)
  ncdif <- Ncdif(parameters, irtModel = irt_model, focalAbilities = focal_thetas, logistic = TRUE)
  cdif <- Cdif(parameters, irtModel = irt_model, focalAbilities = focal_thetas, logistic = TRUE)
  dtf <- Dtf(cdif = cdif)
  stopifnot(all(ncdif[3:DFIT_ITEMS] == 0), abs(dtf - sum(cdif)) < 1e-12)
  item <- function(row) {
    if (model == "2pl") list(a = row[["a"]], b = row[["b"]], c = 0, d = 1) else list(a = row[["a"]], b = I(unname(row[-1])))
  }
  list(model = model,
       reference = lapply(seq_len(DFIT_ITEMS), function(i) item(reference[i, ])),
       focal = lapply(seq_len(DFIT_ITEMS), function(i) item(parameters$focal[i, ])),
       ncdif = ncdif, cdif = cdif, dtf = dtf)
}

dichotomous <- jsonlite::fromJSON("packages/psychometrics/test/fixtures/dif.json", simplifyVector = FALSE)$cases
polytomous <- jsonlite::fromJSON("packages/psychometrics/test/fixtures/dif-polytomous.json", simplifyVector = FALSE)$cases
case_named <- function(cases, name) Filter(function(case) case$name == name, cases)[[1]]

write_fixture("dif-irt", list(
  source = paste("mirt's multipleGroup likelihood ratios, checked against its DIF(scheme = \"drop\"); difR's difRaju on",
                 "ltm calibrations run to convergence, checked against integrate; DFIT's Ncdif, Cdif and Dtf."),
  tolerance = TOLERANCE,
  likelihoodRatio = list(
    "verbal, 2PL" = lr_case("dif", "verbal", FALSE, "2pl", TRUE, 1, c(4, 6, 9, 16, 19, 20)),
    "simulated, sparse, 2PL" = lr_case("dif", "simulated", TRUE, "2pl", FALSE, 8, c(1, 2, 3, 5, 8, 12, 15, 20)),
    "bfi, graded" = lr_case("dif-polytomous", "bfi", FALSE, "grm", TRUE, 10, 1:5),
    "graded, sparse, generalized partial credit" = lr_case("dif-polytomous", "simulated", TRUE, "gpcm", FALSE, 8,
                                                           c(1, 2, 3, 4, 7, 10, 11, 12))
  ),
  raju = list(verbal = raju_case("verbal"), simulated = raju_case("simulated")),
  population = list(nodes = focal_thetas, weights = rep(1 / DFIT_PERSONS, DFIT_PERSONS)),
  functioning = list("2PL" = dfit_case("2pl", two_pl), graded = dfit_case("grm", graded))
))
