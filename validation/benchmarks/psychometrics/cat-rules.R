# Item selection and stopping rules of adaptive tests on the pool of cat.R, 500 3PL items. One selection by each rule
# beyond maximum information, after 15 items of a test, against catR's nextItem, which scores every item left and picks
# one, as measurekit's rules score them. Simulations of 200 tests of at most 30 items that stop once a classification at
# a cut of 0 is decided: by the confidence interval of the estimate, against catR's randomCAT with its classification
# rule, and by the sequential probability ratio test, against mirtCAT's SPRT at its error rates of 0.05, as catR has none.

suppressPackageStartupMessages({
  library(catR)
  library(mirt)
  library(mirtCAT)
})

cat_fixture <- fromJSON("packages/psychometrics/test/fixtures/bench-cat.json")
rules_bank <- cbind(a = cat_fixture$pool$a, b = cat_fixture$pool$b, c = cat_fixture$pool$c, d = 1)
rules_thetas <- cat_fixture$thetas
rules_max <- cat_fixture$length

# --- Selection ------------------------------------------------------------------------------------------------------
# The first 15 items of a test of 30 by maximum information and the posterior mean, for a person of theta 0.5.

set.seed(20261103)
rules_given <- 15
rules_test <- randomCAT(0.5, rules_bank, start = list(nrItems = 1, startSelect = "MFI"),
                   test = list(method = "EAP", itemSelect = "MFI"), stop = list(rule = "length", thr = rules_given),
                   final = list(method = "EAP"), genSeed = 20261103)
rules_out <- rules_test$testItems
rules_x <- rules_test$pattern
rules_theta <- rules_test$thFinal
rules_error <- rules_test$seFinal

bench_fixture("psychometrics", "cat-rules", list(
  source = "catR's randomCAT, genSeed 20261103: the first 15 items of a test of a person of theta 0.5 on bench-cat's pool, by maximum information and the posterior mean.",
  administered = I(rules_out - 1),
  responses = I(rules_x),
  theta = rules_theta,
  standardError = rules_error
))

selection_size <- sprintf("a pool of %d 3PL items, after %d of %d items", nrow(rules_bank), rules_given, rules_max)
rules_selection <- function(criterion) {
  function() nextItem(rules_bank, theta = rules_theta, out = rules_out, x = rules_x, criterion = criterion,
                      rule = "length", thr = rules_max)
}
selection_rules <- list(
  c("cat.likelihoodWeightedInformation", "MLWI", "likelihood-weighted information"),
  c("cat.posteriorWeightedInformation", "MPWI", "posterior-weighted information"),
  c("cat.expectedInformation", "MEI", "maximum expected information"),
  c("cat.expectedPosteriorVariance", "MEPV", "minimum expected posterior variance"),
  c("cat.kullbackLeibler", "KL", "Kullback-Leibler information"),
  c("cat.posteriorKullbackLeibler", "KLP", "posterior Kullback-Leibler information"),
  c("cat.discriminationIndex", "GDI", "the global discrimination index"),
  c("cat.posteriorDiscriminationIndex", "GDIP", "the posterior global discrimination index"),
  c("cat.closestDifficulty", "bOpt", "the difficulty closest to the estimate"),
  c("cat.progressive", "progressive", "the progressive method"),
  c("cat.proportional", "proportional", "the proportional method")
)
for (selection_rule in selection_rules) {
  bench_task(selection_rule[[1]], sprintf("One item selection by %s", selection_rule[[3]]), selection_size,
             reference("catR", sprintf("nextItem(criterion = \"%s\")", selection_rule[[2]])),
             rules_selection(selection_rule[[2]]))
}

# --- Stopping -------------------------------------------------------------------------------------------------------

stopping_size <- sprintf("%d persons, a pool of %d 3PL items, at most %d items each", length(rules_thetas), nrow(rules_bank),
                         rules_max)

classify_interval <- function() {
  for (theta in rules_thetas) {
    randomCAT(theta, rules_bank, start = list(nrItems = 1, startSelect = "MFI"),
              test = list(method = "EAP", itemSelect = "MFI"),
              stop = list(rule = c("classification", "length"), thr = c(0, rules_max), alpha = 0.05),
              final = list(method = "EAP"))
  }
}

# mirtCAT answers from a full pattern of each person, drawn here from the 3PL.
set.seed(20261104)
sprt_model <- generate.mirt_object(data.frame(a1 = rules_bank[, "a"], d = -rules_bank[, "a"] * rules_bank[, "b"],
                                              g = rules_bank[, "c"]), "3PL")
sprt_patterns <- simdata(model = sprt_model, Theta = matrix(rules_thetas))
classify_sprt <- function() {
  mirtCAT(mo = sprt_model, criteria = "MI", start_item = "MI", method = "EAP", local_pattern = sprt_patterns,
          design = list(max_items = rules_max, sprt_lower = -0.3, sprt_upper = 0.3))
}

bench_task("cat.simulateClassification",
           "Simulation of adaptive tests that stop when the 95% interval of the estimate clears a cut", stopping_size,
           reference("catR", "randomCAT(stop = list(rule = \"classification\"))"), classify_interval)
bench_task("cat.simulateSprt",
           "Simulation of adaptive tests that stop when the sequential probability ratio test decides at a cut", stopping_size,
           reference("mirtCAT", "mirtCAT(design = list(sprt_lower, sprt_upper))"), classify_sprt)
