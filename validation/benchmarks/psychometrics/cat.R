# Simulation of adaptive tests: unidimensional by catR's randomCAT, diagnostic by cdcatR's cdcat, multidimensional by
# mirtCAT. Each test is a full one for every simulated person, from the start to the stopping rule. The unidimensional test
# selects by maximum information and estimates by the posterior mean; the diagnostic test selects by the G-DINA
# discrimination index; the multidimensional one by the determinant of the information, and estimates by the posterior
# mode. Testlets have no reference. The routing cuts of a multistage panel are derived by uniroot on the information curves
# of adjacent modules, between their peaks.

suppressPackageStartupMessages({
  library(catR)
  library(cdcatR)
  library(GDINA)
  library(mirt)
  library(mirtCAT)
})

source("validation/benchmarks/simulation.R")

set.seed(20261029)

# --- Data ----------------------------------------------------------------------------------------------------------

pool_size <- 500
persons <- 200
length_of_test <- 30
bank <- cbind(a = round(rlnorm(pool_size, 0, 0.3), 3), b = round(rnorm(pool_size), 3), c = round(runif(pool_size, 0.05, 0.25), 3), d = 1)
thetas <- rnorm(persons)
# The first 400 items of the pool are in 100 testlets of four items, the others alone.
testlet_items <- 400
testlet_of <- ifelse(seq_len(pool_size) <= testlet_items, paste0("t", ceiling(seq_len(pool_size) / 4)), NA)

multi_pool <- 200
multi_persons <- 100
multi_length <- 20
cross <- sample(c(TRUE, FALSE), multi_pool, replace = TRUE, prob = c(0.3, 0.7))
first <- sample(c(TRUE, FALSE), multi_pool, replace = TRUE)
slopes <- cbind(a1 = ifelse(first | cross, round(runif(multi_pool, 0.8, 1.8), 3), 0),
                a2 = ifelse(!first | cross, round(runif(multi_pool, 0.8, 1.8), 3), 0))
intercepts <- round(rnorm(multi_pool), 3)
multi_thetas <- matrix(rnorm(multi_persons * 2), multi_persons, 2)
multi_responses <- matrix(rbinom(multi_persons * multi_pool, 1, plogis(multi_thetas %*% t(slopes) + rep(intercepts, each = multi_persons))),
                          multi_persons, multi_pool, dimnames = list(NULL, ids(multi_pool)))

diagnostic_persons <- 200
diagnostic_length <- 10
simulated <- simulate_dina(1000, 20, 4)
diagnostic_fit <- GDINA(simulated$responses, simulated$Q, model = "GDINA", control = list(conv.crit = 1e-4, maxitr = 2000), verbose = 0)
diagnostic_responses <- simulated$responses[seq_len(diagnostic_persons), ]

# A panel of 1-3-3-3 modules of 20 items: each module takes the unused items of the pool whose difficulties are nearest its
# location, the routing module first.
module_size <- 20
module_locations <- list(0, c(-1, 0, 1), c(-1.2, 0, 1.2), c(-1.5, 0, 1.5))
unused <- seq_len(pool_size)
panel <- lapply(module_locations, function(locations) {
  lapply(locations, function(location) {
    chosen <- unused[order(abs(bank[unused, "b"] - location))[seq_len(module_size)]]
    unused <<- setdiff(unused, chosen)
    chosen
  })
})

bench_fixture("psychometrics", "cat", list(
  source = "Simulations, set.seed(20261029): a 3PL pool, and a two-dimensional pool of M2PL items; the diagnostic pool is that of bench-cdm.",
  length = length_of_test,
  pool = list(a = I(bank[, "a"]), b = I(bank[, "b"]), c = I(bank[, "c"])),
  thetas = I(thetas),
  testlets = I(testlet_of),
  multidimensional = list(length = multi_length, slopes = unname(slopes), intercepts = I(intercepts),
                          thetas = unname(multi_thetas)),
  diagnostic = list(length = diagnostic_length, persons = diagnostic_persons),
  multistage = list(stages = lapply(panel, function(modules) lapply(modules, function(items) I(items - 1))))
))

# --- References -----------------------------------------------------------------------------------------------------

simulate_unidimensional <- function() {
  for (theta in thetas) {
    randomCAT(theta, bank, start = list(nrItems = 1, startSelect = "MFI"), test = list(method = "EAP", itemSelect = "MFI"),
              stop = list(rule = "length", thr = length_of_test), final = list(method = "EAP"))
  }
}

multi_model <- generate.mirt_object(cbind(slopes, d = intercepts), itemtype = "2PL")
simulate_multidimensional <- function() {
  mirtCAT(mo = multi_model, criteria = "Drule", start_item = "Drule", method = "MAP", local_pattern = multi_responses,
          design = list(max_items = multi_length), verbose = FALSE)
}

simulate_diagnostic <- function() {
  cdcat(fit = diagnostic_fit, dat = diagnostic_responses, itemSelect = "GDI", MAXJ = diagnostic_length, FIXED.LENGTH = TRUE,
        n.cores = 1, seed = 1, print.progress = FALSE)
}

# The information of a module at a vector of thetas: the 3PL's, a^2 (Q / P) ((P - c) / (1 - c))^2.
module_information <- function(items) {
  a <- bank[items, "a"]
  b <- bank[items, "b"]
  c <- bank[items, "c"]
  function(theta) {
    n <- length(theta)
    slope <- rep(a, each = n)
    guess <- rep(c, each = n)
    p <- guess + (1 - guess) * plogis(slope * (theta - rep(b, each = n)))
    rowSums(matrix(slope^2 * (1 - p) / p * ((p - guess) / (1 - guess))^2, n))
  }
}
derive_cuts <- function() {
  grid <- seq(-6, 6, by = 0.01)
  for (modules in panel[-1]) {
    curves <- lapply(modules, module_information)
    peaks <- sapply(curves, function(curve) grid[which.max(curve(grid))])
    for (j in seq_len(length(modules) - 1)) {
      uniroot(function(theta) curves[[j + 1]](theta) - curves[[j]](theta), peaks[c(j, j + 1)], tol = 1e-12)
    }
  }
}

bench_task("cat.simulateAdaptive", "Simulation of adaptive tests by maximum information and the posterior mean",
           sprintf("%d persons, a pool of %d 3PL items, %d items each", persons, pool_size, length_of_test),
           reference("catR", "randomCAT"), simulate_unidimensional)
bench_task("cat.simulateTestlets", "Simulation of adaptive tests that select testlets as units, by their mean information",
           sprintf("%d persons, a pool of %d 3PL items with %d in testlets of 4, %d items each", persons, pool_size, testlet_items, length_of_test),
           note = "catR has no testlets, so no reference does this task")
bench_task("cat.simulateDiagnostic", "Simulation of diagnostic adaptive tests by the G-DINA discrimination index",
           sprintf("%d persons, a pool of 20 items over 4 attributes, %d items each", diagnostic_persons, diagnostic_length),
           reference("cdcatR", "cdcat"), simulate_diagnostic)
bench_task("cat.simulateMultidimensional", "Simulation of multidimensional adaptive tests by the determinant of the information",
           sprintf("%d persons, a pool of %d two-dimensional items, %d items each", multi_persons, multi_pool, multi_length),
           reference("mirtCAT", "mirtCAT(criteria = \"Drule\")"), simulate_multidimensional)
bench_task("cat.multistageCuts", "Routing cuts of a multistage panel by maximum information",
           sprintf("a panel of 1-3-3-3 modules of %d items, from a pool of %d 3PL items", module_size, pool_size),
           reference("stats", "uniroot on the information curves"), derive_cuts)
