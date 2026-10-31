# Item response theory: calibration of the logistic, graded and generalized partial credit models and of testlets by mirt,
# ability estimation by mirt's fscores, joint maximum likelihood by TAM's tam.jml, and test information by mirt's testinfo.
# mirt runs at its defaults, which are measurekit's: 61 points on [-6, 6] and a tolerance of 1e-4 on the change of a
# parameter. The calibrations of measurekit and mirt differ in the iterations they need, which the note of each row gives.

suppressPackageStartupMessages({
  library(mirt)
  library(TAM)
})

source("validation/benchmarks/simulation.R")

set.seed(20261029)
TOL <- 1e-4
POINTS <- 61
TESTLET_POINTS <- 21
# The size of the calibration with standard errors from the Hessian, which takes minutes on the full data in mirt.
PILOT_PERSONS <- 500
PILOT_ITEMS <- 20

# --- Data ----------------------------------------------------------------------------------------------------------

persons <- 2000
item_count <- 40
simulated <- simulate_2pl(persons, item_count)
b <- simulated$b
theta <- simulated$theta
binary <- simulated$responses

guessing <- round(runif(item_count, 0.05, 0.25), 3)
a_three <- round(runif(item_count, 0.8, 2), 3)
three <- matrix(rbinom(persons * item_count, 1,
                       guessing + (1 - guessing) * plogis(sweep(outer(theta, b, "-"), 2, a_three, "*"))),
                persons, item_count, dimnames = list(NULL, ids(item_count)))

graded_persons <- 1500
graded_items <- 20
a_graded <- round(rlnorm(graded_items, 0, 0.3), 3)
thresholds <- t(sapply(round(rnorm(graded_items), 3), function(m) round(m + sort(rnorm(4, 0, 1)), 3)))
theta_graded <- rnorm(graded_persons)
uniform <- matrix(runif(graded_persons * graded_items), graded_persons, graded_items)
graded <- matrix(0L, graded_persons, graded_items, dimnames = list(NULL, ids(graded_items)))
for (k in 1:4) {
  graded <- graded + (uniform < plogis(sweep(outer(theta_graded, thresholds[, k], "-"), 2, a_graded, "*")))
}
graded <- matrix(as.integer(graded), graded_persons, graded_items, dimnames = list(NULL, ids(graded_items)))

testlet_persons <- 1000
testlet_count <- 6
testlet_size <- 5
testlet_of <- rep(seq_len(testlet_count), each = testlet_size)
a_testlet <- round(runif(length(testlet_of), 0.8, 1.8), 3)
b_testlet <- round(rnorm(length(testlet_of)), 3)
effect_sd <- seq(0.4, 1, length.out = testlet_count)
theta_testlet <- rnorm(testlet_persons)
effects <- sapply(effect_sd, function(s) rnorm(testlet_persons, 0, s))
testlet_data <- sapply(seq_along(testlet_of), function(j) {
  rbinom(testlet_persons, 1, plogis(a_testlet[j] * (theta_testlet + effects[, testlet_of[j]] - b_testlet[j])))
})
colnames(testlet_data) <- ids(length(testlet_of))

# --- References -----------------------------------------------------------------------------------------------------

calibrate <- function(data, itemtype, ...) {
  mirt(data, 1, itemtype = itemtype, TOL = TOL, quadpts = POINTS, verbose = FALSE, ...)
}

testlet_fit <- function(data, testlet, ...) {
  values <- bfactor(data, testlet, pars = "values")
  specific <- grep("^COV_([0-9]+)\\1$", values$name, value = TRUE)
  values$est[values$item == "GROUP" & values$name %in% setdiff(specific, "COV_11")] <- TRUE
  slopes <- lapply(colnames(data), function(id) values$parnum[values$item == id & grepl("^a", values$name) & values$value != 0])
  bfactor(data, testlet, pars = values, constrain = slopes, TOL = TOL, quadpts = TESTLET_POINTS, verbose = FALSE, ...)
}

fit_2pl <- calibrate(binary, "2PL")
stopifnot(extract.mirt(fit_2pl, "converged"))
coefficients <- coef(fit_2pl, IRTpars = TRUE, simplify = TRUE)$items

bench_fixture("psychometrics", "irt", list(
  source = "Simulations, set.seed(20261029): 2PL, 3PL, graded and testlet responses; the 2PL items are mirt's estimates.",
  binary = list(itemIds = ids(item_count), responses = unname(binary)),
  three = list(itemIds = ids(item_count), responses = unname(three)),
  graded = list(itemIds = ids(graded_items), responses = unname(graded)),
  testlet = list(itemIds = colnames(testlet_data), responses = unname(testlet_data),
                 testlets = setNames(lapply(seq_len(testlet_count), function(t) I(colnames(testlet_data)[testlet_of == t])),
                                     paste0("t", seq_len(testlet_count))),
                 points = TESTLET_POINTS),
  items = list(a = I(unname(coefficients[, "a"])), b = I(unname(coefficients[, "b"])))
))

sizes <- list(
  binary = sprintf("%s persons, %d dichotomous items", count(persons), item_count),
  graded = sprintf("%s persons, %d items of 5 categories", count(graded_persons), graded_items),
  testlet = sprintf("%s persons, %d testlets of %d items", count(testlet_persons), testlet_count, testlet_size)
)

bench_task("irt.calibrate1pl", "Calibration of the one-parameter model by EM", sizes$binary,
           reference("mirt", "mirt(itemtype = \"Rasch\")"), function() calibrate(binary, "Rasch"))
bench_task("irt.calibrate2pl", "Calibration of the two-parameter model by EM", sizes$binary,
           reference("mirt", "mirt(itemtype = \"2PL\")"), function() calibrate(binary, "2PL"))
bench_task("irt.calibrate3pl", "Calibration of the three-parameter model by EM", sizes$binary,
           reference("mirt", "mirt(itemtype = \"3PL\")"), function() calibrate(three, "3PL"))
bench_task("irt.calibrate2plCrossProduct", "The two-parameter model with cross-product standard errors", sizes$binary,
           reference("mirt", "mirt(SE.type = \"crossprod\")"),
           function() calibrate(binary, "2PL", SE = TRUE, SE.type = "crossprod"))
pilot <- binary[1:PILOT_PERSONS, 1:PILOT_ITEMS]
bench_task("irt.calibrate2plHessian", "The two-parameter model with standard errors from the Hessian",
           sprintf("%s persons, %d items", count(PILOT_PERSONS), PILOT_ITEMS),
           reference("mirt", "mirt(SE.type = \"Richardson\")"),
           function() calibrate(pilot, "2PL", SE = TRUE, SE.type = "Richardson"))
bench_task("irt.calibrateGraded", "Calibration of the graded response model by EM", sizes$graded,
           reference("mirt", "mirt(itemtype = \"graded\")"), function() calibrate(graded, "graded"))
bench_task("irt.calibrateGeneralizedPartialCredit", "Calibration of the generalized partial credit model by EM",
           sizes$graded, reference("mirt", "mirt(itemtype = \"gpcm\")"), function() calibrate(graded, "gpcm"))
bench_task("irt.calibrateTestlets", "Calibration of the testlet model by EM", sizes$testlet,
           reference("mirt", "bfactor"), function() testlet_fit(testlet_data, testlet_of))

for (method in c("EAP", "MAP", "ML", "WLE")) {
  local({
    estimate <- method
    key <- c(EAP = "irt.abilityPosteriorMean", MAP = "irt.abilityPosteriorMode", ML = "irt.abilityMaximumLikelihood",
             WLE = "irt.abilityWeightedLikelihood")[[estimate]]
    task <- c(EAP = "Ability by the posterior mean", MAP = "Ability by the posterior mode",
              ML = "Ability by maximum likelihood", WLE = "Ability by weighted likelihood")[[estimate]]
    bench_task(key, task, sprintf("%s persons, the %d items of the calibration", count(persons), item_count),
               reference("mirt", sprintf("fscores(method = \"%s\")", estimate)),
               function() fscores(fit_2pl, method = estimate, theta_lim = c(-6, 6), quadpts = POINTS, verbose = FALSE))
  })
}

bench_task("irt.jointMaximumLikelihood", "Joint maximum likelihood calibration of the Rasch model",
           sprintf("%s persons, %d items", count(persons), item_count), reference("TAM", "tam.jml"),
           function() tam.jml(binary, verbose = FALSE))

bench_task("irt.summedScoreTable", "The posterior mean of the ability given each summed score", sprintf("%d items", item_count))

grid <- matrix(seq(-6, 6, length.out = 121))
bench_task("irt.testInformation", "Information of a test at 121 values of theta", sprintf("%d items", item_count),
           reference("mirt", "testinfo"), function() testinfo(fit_2pl, grid))
