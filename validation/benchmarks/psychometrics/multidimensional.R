# Multidimensional item response theory: confirmatory and exploratory calibration by EM, calibration of five dimensions by
# Metropolis-Hastings Robbins-Monro, ability estimation, and rotation of the loadings, against mirt's mirt and fscores and
# GPArotation. mirt runs at its defaults: a tolerance of 1e-4 and a limit of 500 cycles, and 31 quadrature points per
# dimension, as measurekit does; its MH-RM, at its default tolerance, with one seed.

suppressPackageStartupMessages({
  library(mirt)
  library(GPArotation)
  library(mvtnorm)
  library(psych)
})

set.seed(20261029)
TOL <- 1e-4
POINTS <- 31
persons <- 1000

# --- Data ----------------------------------------------------------------------------------------------------------

correlated_thetas <- function(n, rho) rmvnorm(n, sigma = matrix(c(1, rho, rho, 1), 2))

simple_structure <- function(per_dimension) {
  loadings <- matrix(0, 2 * per_dimension, 2)
  loadings[seq_len(per_dimension), 1] <- round(runif(per_dimension, 0.8, 1.8), 3)
  loadings[per_dimension + seq_len(per_dimension), 2] <- round(runif(per_dimension, 0.8, 1.8), 3)
  loadings
}

# Dichotomous responses of slopes and intercepts: P = logistic(theta a + d).
dichotomous <- function(theta, slopes, intercepts) {
  eta <- theta %*% t(slopes) + rep(intercepts, each = nrow(theta))
  matrix(rbinom(length(eta), 1, plogis(eta)), nrow(theta), nrow(slopes))
}

confirmatory_slopes <- simple_structure(12)
confirmatory <- dichotomous(correlated_thetas(persons, 0.5), confirmatory_slopes, round(rnorm(24), 3))
colnames(confirmatory) <- ids(24)

graded_slopes <- simple_structure(10)
graded_theta <- correlated_thetas(persons, 0.5)
graded_eta <- graded_theta %*% t(graded_slopes)
graded_intercepts <- t(sapply(round(rnorm(20), 3), function(m) sort(round(m + c(-1, 0, 1) + rnorm(3, 0, 0.2), 3), decreasing = TRUE)))
uniform <- matrix(runif(persons * 20), persons, 20)
graded <- matrix(0L, persons, 20)
for (k in 1:3) graded <- graded + (uniform < plogis(graded_eta + rep(graded_intercepts[, k], each = persons)))
graded <- matrix(as.integer(graded), persons, 20, dimnames = list(NULL, ids(20)))

exploratory_slopes <- simple_structure(10)
exploratory_slopes[1:10, 2] <- round(runif(10, 0, 0.4), 3)
exploratory_slopes[11:20, 1] <- round(runif(10, 0, 0.4), 3)
exploratory <- dichotomous(correlated_thetas(persons, 0.3), exploratory_slopes, round(rnorm(20), 3))
colnames(exploratory) <- ids(20)

# psych's bfi, the 25 items of its five scales for the first 800 persons who answered all of them, scored from 0, on five
# correlated dimensions under the graded model.
data(bfi, package = "psych")
bfi_items <- as.matrix(bfi[, 1:25])
bfi_items <- (bfi_items[complete.cases(bfi_items), ][1:800, ] - 1L)
colnames(bfi_items) <- ids(25)
bfi_scales <- rep(c("A", "C", "E", "N", "O"), each = 5)

# --- References -----------------------------------------------------------------------------------------------------

confirmatory_model <- mirt.model("F1 = 1-12\nF2 = 13-24\nCOV = F1*F2")
graded_model <- mirt.model("F1 = 1-10\nF2 = 11-20\nCOV = F1*F2")

calibrate_confirmatory <- function() mirt(confirmatory, confirmatory_model, quadpts = POINTS, TOL = TOL, verbose = FALSE)
calibrate_graded <- function() mirt(graded, graded_model, itemtype = "graded", quadpts = POINTS, TOL = TOL, verbose = FALSE)
calibrate_exploratory <- function() mirt(exploratory, 2, quadpts = POINTS, TOL = TOL, verbose = FALSE)

confirmatory_fit <- calibrate_confirmatory()
exploratory_fit <- calibrate_exploratory()
loadings <- summary(exploratory_fit, rotate = "none", verbose = FALSE)$rotF

bench_fixture("psychometrics", "multidimensional", list(
  source = "Simulations, set.seed(20261029): a confirmatory 2PL and a graded model with simple structure over two correlated dimensions, and an exploratory 2PL with small cross-loadings.",
  points = POINTS,
  confirmatory = list(itemIds = ids(24), responses = unname(confirmatory),
                      loadings = setNames(lapply(1:24, function(i) I(if (i <= 12) "F1" else "F2")), ids(24))),
  graded = list(itemIds = ids(20), responses = unname(graded),
                loadings = setNames(lapply(1:20, function(i) I(if (i <= 10) "F1" else "F2")), ids(20))),
  exploratory = list(itemIds = ids(20), responses = unname(exploratory)),
  mhrm = list(itemIds = ids(25), responses = unname(bfi_items), loadings = setNames(lapply(bfi_scales, I), ids(25)))
))

bench_task("multidimensional.calibrateConfirmatory", "Calibration of a confirmatory two-dimensional 2PL by EM",
           sprintf("%s persons, 24 items, 2 correlated dimensions", count(persons)),
           reference("mirt", "mirt(model, quadpts = 31)"), calibrate_confirmatory)
bench_task("multidimensional.calibrateGraded", "Calibration of a confirmatory two-dimensional graded model by EM",
           sprintf("%s persons, 20 items of 4 categories, 2 correlated dimensions", count(persons)),
           reference("mirt", "mirt(itemtype = \"graded\", quadpts = 31)"), calibrate_graded)
bench_task("multidimensional.calibrateExploratory", "Calibration of an exploratory two-dimensional 2PL by EM",
           sprintf("%s persons, 20 items, 2 dimensions", count(persons)), reference("mirt", "mirt(data, 2, quadpts = 31)"),
           calibrate_exploratory, note = "mirt stops at its limit of 500 cycles when it has not converged")
bench_task("multidimensional.calibrateMhrm", "Calibration of a five-dimensional graded model by Metropolis-Hastings Robbins-Monro",
           sprintf("%d persons, 25 items of 6 categories, 5 correlated dimensions", nrow(bfi_items)),
           reference("mirt", "mirt(method = \"MHRM\")"),
           function() mirt(bfi_items, mirt.model(paste0("A = 1-5\nC = 6-10\nE = 11-15\nN = 16-20\nO = 21-25\n",
                                                          "COV = A*C, A*E, A*N, A*O, C*E, C*N, C*O, E*N, E*O, N*O")),
                           itemtype = "graded", method = "MHRM", verbose = FALSE, technical = list(set.seed = 1)),
           note = "both draw the abilities, so the time depends on the seed; mirt runs with one")
bench_task("multidimensional.estimateMode", "Abilities by the posterior mode, in two dimensions",
           sprintf("%s persons, 24 items", count(persons)), reference("mirt", "fscores(method = \"MAP\")"),
           function() fscores(confirmatory_fit, method = "MAP", quadpts = POINTS, verbose = FALSE))
bench_task("multidimensional.estimateMean", "Abilities by the posterior mean, in two dimensions",
           sprintf("%s persons, 24 items", count(persons)), reference("mirt", "fscores(method = \"EAP\")"),
           function() fscores(confirmatory_fit, method = "EAP", quadpts = POINTS, verbose = FALSE))
bench_task("multidimensional.rotateOblimin", "Oblimin rotation of the loadings of a calibration",
           "20 items, 2 dimensions", reference("GPArotation", "oblimin"), function() oblimin(loadings))
bench_task("multidimensional.rotateVarimax", "Varimax rotation of the loadings of a calibration",
           "20 items, 2 dimensions", reference("GPArotation", "Varimax"), function() Varimax(loadings))
