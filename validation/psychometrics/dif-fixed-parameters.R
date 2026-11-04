# DIF of 3PL items whose lower asymptote is held (dif-fixed-parameters.json). References: difR's difLord and difRaju
# with the guessing parameters fixed (`c`), given the calibrations of each group by mirt with est = FALSE on g, run to
# the optimum with Richardson standard errors, and checked against LordChi2, RajuZ and itemRescale; the unsigned area
# against integrate over the 3PL curves; difLord(c = ) on its own ltm calibrations as a check; mirt's multipleGroup for
# the likelihood-ratio tests with g held in both groups, checked against its DIF(scheme = "drop"). Data: a 3PL
# simulation of 12 items, 1000 persons per group, the focal group's theta N(-0.5, 1), c of 0.2 and 0.25 held at the
# values that generated the responses. Item 1 is 0.7 harder for the focal group; item 2 discriminates half as well in it.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages({
  library(difR)
  library(mirt)
})

SEED <- 20261104
ITEMS <- 12
PERSONS <- 1000
ALPHA <- 0.05
# mirt's default rectangular rule, which measurekit's default quadrature reproduces.
QUADRATURE_POINTS <- 61
# The separate calibrations run to the optimum, so that their Richardson standard errors are exact to about 1e-6.
CALIBRATION_TOLERANCE <- 1e-10
# The likelihood-ratio fits stop at the tolerance of dif-irt.R, so that the chi-squares are exact to about 1e-4.
LIKELIHOOD_RATIO_TOLERANCE <- 1e-7

set.seed(SEED)
a <- round(runif(ITEMS, 0.8, 2), 2)
b <- round(rnorm(ITEMS), 2)
guessing <- rep(c(0.2, 0.25), length.out = ITEMS)
focal_a <- a
focal_a[2] <- a[2] / 2
focal_b <- b
focal_b[1] <- b[1] + 0.7
simulate <- function(theta, a, b) {
  probability <- sapply(seq_len(ITEMS), function(j) guessing[j] + (1 - guessing[j]) * plogis(a[j] * (theta - b[j])))
  (matrix(runif(length(theta) * ITEMS), length(theta)) < probability) * 1
}
data <- rbind(simulate(rnorm(PERSONS), a, b), simulate(rnorm(PERSONS, -0.5), focal_a, focal_b))
colnames(data) <- paste0("item", seq_len(ITEMS))
focal <- rep(0:1, each = PERSONS)
held <- setNames(lapply(guessing, function(g) list(c = g)), colnames(data))

# Sets g of every item to its held value with est = FALSE in a table of mirt's parameters.
hold_guessing <- function(values) {
  for (j in seq_len(ITEMS)) {
    rows <- values$item == colnames(data)[j] & values$name == "g"
    values$value[rows] <- guessing[j]
    values$est[rows] <- FALSE
  }
  values
}

# --- Separate calibrations, Lord and Raju ----------------------------------------------------------------------------

# One group's 3PL calibration with g held, in difR's layout of the 3PL with fixed c: a, b, se(a), se(b), cov(a,b), c.
# The delta method from mirt's slope and intercept (a1, d) to a and b = -d / a is itemPar3PLconst's, from ltm's.
# Newton's M-step reaches the optimum: with the default BFGS, EM stops 3e-9 below it in the focal group's
# log-likelihood, 8e-5 short in the slope of its hardest item.
item_parameters_3pl <- function(responses) {
  values <- hold_guessing(mirt(responses, 1, itemtype = "3PL", pars = "values"))
  fit <- mirt(responses, 1, pars = values, SE = TRUE, SE.type = "Richardson", TOL = CALIBRATION_TOLERANCE,
              quadpts = QUADRATURE_POINTS, optimizer = "NR", verbose = FALSE, technical = list(NCYCLES = 1e5))
  stopifnot(extract.mirt(fit, "converged"))
  covariance <- vcov(fit)
  table <- mod2values(fit)
  rows <- t(sapply(colnames(responses), function(id) {
    own <- table[table$item == id, ]
    slope <- own$value[own$name == "a1"]
    intercept <- own$value[own$name == "d"]
    label <- function(name) paste0(name, ".", own$parnum[own$name == name])
    var_a <- covariance[label("a1"), label("a1")]
    var_d <- covariance[label("d"), label("d")]
    cov_ad <- covariance[label("a1"), label("d")]
    c(a = slope, b = -intercept / slope, "se(a)" = sqrt(var_a),
      "se(b)" = sqrt(var_d / slope^2 + intercept^2 * var_a / slope^4 - 2 * intercept * cov_ad / slope^3),
      "cov(a,b)" = intercept * var_a / slope^2 - cov_ad / slope, c = own$value[own$name == "g"])
  }))
  rows
}

reference <- item_parameters_3pl(data[focal == 0, ])
focal_parameters <- item_parameters_3pl(data[focal == 1, ])
parameters <- rbind(reference, focal_parameters)

lord <- difLord(data, group = focal, focal.name = 1, model = "3PL", c = guessing, irtParam = parameters,
                same.scale = FALSE)
linked <- itemRescale(reference, focal_parameters)
stopifnot(max(abs(LordChi2(reference, linked) - lord$LordChi)) < 1e-12)
slope <- sd(reference[, "b"]) / sd(focal_parameters[, "b"])
# difLord on its own calibrations by ltm's tpm, whose optim stops at a relative change of 1.5e-8 in the
# log-likelihood, about 1e-4 here, short of the optimum.
default <- difLord(data, group = focal, focal.name = 1, model = "3PL", c = guessing)
stopifnot(max(abs(default$LordChi - lord$LordChi) / pmax(1, lord$LordChi)) < 0.05)

# The unsigned area between two 3PL curves of the same lower asymptote by numerical integration.
integrated_area <- function(reference, focal, c) {
  difference <- function(theta) {
    abs((1 - c) * (plogis(focal[1] * (theta - focal[2])) - plogis(reference[1] * (theta - reference[2]))))
  }
  integrate(difference, -Inf, Inf, rel.tol = 1e-12, subdivisions = 1000)$value
}

raju <- difRaju(data, group = focal, focal.name = 1, model = "3PL", c = guessing, irtParam = parameters,
                same.scale = FALSE)
signed <- RajuZ(reference, linked, signed = TRUE)$res
unsigned <- RajuZ(reference, linked, signed = FALSE)$res
stopifnot(max(abs(raju$RajuZ - unsigned[, "Z"])) < 1e-12)
areas <- vapply(seq_len(ITEMS), function(j) integrated_area(reference[j, 1:2], linked[j, 1:2], guessing[j]), 0)
stopifnot(max(abs(abs(unsigned[, "H"]) - areas)) < 1e-6)

# --- Likelihood ratio --------------------------------------------------------------------------------------------

groups <- factor(ifelse(focal == 1, "focal", "reference"), levels = c("reference", "focal"))

# The multiple-group model with g held in both groups, a1 and d of the items `anchors` equal across groups and the
# others free. Naming the anchors in `invariance` would constrain their held g as well, which mirt refuses.
fit_groups <- function(anchors) {
  items <- paste(anchors, collapse = ",")
  syntax <- mirt.model(sprintf("F = 1-%d\nCONSTRAINB = (%s, a1), (%s, d)", ITEMS, items, items))
  invariance <- c("free_means", "free_var")
  values <- hold_guessing(multipleGroup(data, syntax, group = groups, itemtype = "3PL", invariance = invariance,
                                        pars = "values"))
  model <- multipleGroup(data, syntax, group = groups, itemtype = "3PL", invariance = invariance, pars = values,
                         quadpts = QUADRATURE_POINTS, TOL = LIKELIHOOD_RATIO_TOLERANCE,
                         technical = list(NCYCLES = 20000), verbose = FALSE)
  list(logLik = extract.mirt(model, "logLik"), converged = extract.mirt(model, "converged"), model = model)
}

constrained <- fit_groups(seq_len(ITEMS))
stopifnot(constrained$converged)
tests <- t(sapply(seq_len(ITEMS), function(item) {
  degrees <- sum(extract.item(extract.group(constrained$model, 1), item)@est)
  other <- fit_groups(setdiff(seq_len(ITEMS), item))
  stopifnot(other$converged)
  chi <- max(0, 2 * (other$logLik - constrained$logLik))
  c(chiSquare = chi, degreesOfFreedom = degrees, significance = pchisq(chi, degrees, lower.tail = FALSE))
}))
check <- DIF(constrained$model, which.par = c("a1", "d"), scheme = "drop", verbose = FALSE,
             technical = list(NCYCLES = 20000), TOL = LIKELIHOOD_RATIO_TOLERANCE)
stopifnot(max(abs(check$X2 - tests[, "chiSquare"])) < 1e-2)
group_parameters <- coef(constrained$model)$focal$GroupPars

write_fixture("dif-fixed-parameters", list(
  source = paste("A 3PL simulation of 12 items with c held at 0.2 and 0.25, set.seed(20261104): difLord and difRaju",
                 "with c on mirt 1.48 calibrations of each group with est = FALSE on g, optimizer NR, TOL = 1e-10, 61",
                 "points, SE.type Richardson; mirt's multipleGroup likelihood ratios with g held, TOL = 1e-7."),
  itemIds = colnames(data),
  responses = unname(lapply(seq_len(nrow(data)), function(i) unname(data[i, ]))),
  focal = focal,
  fixedParameters = held,
  calibrationTolerance = CALIBRATION_TOLERANCE,
  lord = list(
    chiSquare = lord$LordChi,
    degreesOfFreedom = 2,
    significance = pchisq(lord$LordChi, 2, lower.tail = FALSE),
    reference = list(a = unname(reference[, "a"]), b = unname(reference[, "b"])),
    focal = list(a = unname(focal_parameters[, "a"]), b = unname(focal_parameters[, "b"])),
    transformation = list(slope = slope, intercept = mean(reference[, "b"]) - slope * mean(focal_parameters[, "b"])),
    difrDefaultChiSquare = default$LordChi
  ),
  raju = list(
    signedArea = unname(signed[, "H"]),
    signedZ = unname(signed[, "Z"]),
    unsignedArea = unname(abs(unsigned[, "H"])),
    unsignedZ = unname(abs(unsigned[, "Z"])),
    significance = unname(2 * pnorm(-abs(unsigned[, "Z"])))
  ),
  likelihoodRatioTolerance = LIKELIHOOD_RATIO_TOLERANCE,
  likelihoodRatio = list(
    chiSquare = unname(tests[, "chiSquare"]),
    degreesOfFreedom = unname(tests[, "degreesOfFreedom"]),
    significance = unname(tests[, "significance"]),
    logLikelihood = constrained$logLik,
    focal = list(mean = group_parameters["par", "MEAN_1"], sd = sqrt(group_parameters["par", "COV_11"]))
  )
))
