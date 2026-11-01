# Simulation of adaptive tests: unidimensional by catR's randomCAT, diagnostic by cdcatR's cdcat, multidimensional by
# mirtCAT. Each test is a full one for every simulated person, from the start to the stopping rule. The unidimensional test
# selects by maximum information and estimates by the posterior mean; the diagnostic test selects by the G-DINA
# discrimination index; the multidimensional one by the determinant of the information, and estimates by the posterior
# mode. Testlets have no reference. The routing cuts of a multistage panel are derived by uniroot on the information curves
# of adjacent modules, between their peaks. A step of a test under content bounds is timed against mirtCAT's shadow test,
# and the assembly of parallel forms against highs.

suppressPackageStartupMessages({
  library(catR)
  library(cdcatR)
  library(GDINA)
  library(mirt)
  library(mirtCAT)
  library(highs)
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

# --- Content bounds ---------------------------------------------------------------------------------------------------
# One step of a test of 30 items on a pool of 1,000 3PL items in 12 content areas (two facets of six), after 12 items of a
# hidden feasible form. The reference is the step of a shadow test: mirtCAT's information of every item, and the integer
# program of lpSolve through findNextItem that maximizes it subject to the length, the bounds of every area and the items
# given. measurekit scores every item by the information, the priority index and the filter.

set.seed(20261030)
bounds_pool <- 1000
bounds_length <- 30
bounds_given <- 12
bounds_bank <- cbind(a = round(rlnorm(bounds_pool, 0, 0.3), 3), b = round(rnorm(bounds_pool), 3),
                     c = round(runif(bounds_pool, 0.05, 0.25), 3), d = 1)
bounds_areas <- cbind(paste0("first", sample(6, bounds_pool, replace = TRUE)), paste0("second", sample(6, bounds_pool, replace = TRUE)))
bounds_names <- sort(unique(as.vector(bounds_areas)))
bounds_membership <- sapply(bounds_names, function(area) as.numeric(rowSums(bounds_areas == area) > 0))
bounds_hidden <- sample(bounds_pool, bounds_length)
bounds_counts <- colSums(bounds_membership[bounds_hidden, , drop = FALSE])
bounds_minimum <- pmax(0, bounds_counts - sample(0:1, length(bounds_names), replace = TRUE))
bounds_maximum <- bounds_counts + sample(0:1, length(bounds_names), replace = TRUE)
bounds_taken <- bounds_hidden[seq_len(bounds_given)]
bounds_responses <- rbinom(bounds_given, 1, 0.6)

bounds_model <- generate.mirt_object(data.frame(a1 = bounds_bank[, "a"], d = -bounds_bank[, "a"] * bounds_bank[, "b"], g = bounds_bank[, "c"]), "3PL")
bounds_constr_fun <- function(design, person, test) {
  lhs <- rbind(rep(1, bounds_pool), t(bounds_membership), t(bounds_membership))
  data.frame(lhs, c("==", rep(">=", length(bounds_names)), rep("<=", length(bounds_names))),
             c(bounds_length, bounds_minimum, bounds_maximum))
}
bounds_design <- mirtCAT(mo = bounds_model, design_elements = TRUE, design = list(constr_fun = bounds_constr_fun))
for (k in seq_len(bounds_given)) {
  bounds_design <- updateDesign(bounds_design, new_item = bounds_taken[k], new_response = bounds_responses[k])
}
bounds_theta <- as.numeric(extract.mirtCAT(bounds_design$person, "thetas"))
bounds_error <- as.numeric(extract.mirtCAT(bounds_design$person, "thetas_SE_history")[bounds_given + 1, ])

bench_fixture("psychometrics", "cat-bounds", list(
  source = "Simulations, set.seed(20261030): a 3PL pool of 1,000 items in 12 content areas and the first 12 items of a hidden feasible form of 30.",
  pool = list(a = I(bounds_bank[, "a"]), b = I(bounds_bank[, "b"]), c = I(bounds_bank[, "c"])),
  areas = unname(lapply(seq_len(bounds_pool), function(i) I(bounds_areas[i, ]))),
  constraints = lapply(seq_along(bounds_names), function(k) list(area = bounds_names[k], minimum = bounds_minimum[k], maximum = bounds_maximum[k])),
  length = bounds_length,
  given = I(bounds_taken - 1),
  responses = I(bounds_responses),
  theta = bounds_theta,
  standardError = bounds_error
))

shadow_step <- function() {
  findNextItem(bounds_design, objective = computeCriteria(bounds_design, criteria = "MI"))
}

bench_task("cat.contentBoundsStep", "One item selection under content bounds, by the information at the estimate",
           sprintf("a pool of %d 3PL items in %d content areas, a test of %d items after %d", bounds_pool, length(bounds_names), bounds_length, bounds_given),
           reference("mirtCAT", "findNextItem(objective = computeCriteria(criteria = \"MI\"))` with lpSolve's `lp"), shadow_step)

# --- Form assembly ---------------------------------------------------------------------------------------------------
# Four forms of 40 items from a pool of 1,000 3PL items in 5 content areas, with 30 pairs of enemies and the information at
# three thetas as the target, from a hidden feasible form. The reference is highs, which finds the four forms whose largest
# gap from the target is smallest (minimax) with no item on two forms, within a limit of 10 seconds; measurekit takes the
# forms by weighted deviations in rotation.

set.seed(20261031)
forms_pool <- 1000
forms_count <- 4
forms_length <- 40
forms_bank <- cbind(a = round(rlnorm(forms_pool, 0, 0.3), 3), b = round(rnorm(forms_pool), 3), c = round(runif(forms_pool, 0.05, 0.25), 3), d = 1)
forms_areas <- paste0("area", sample(5, forms_pool, replace = TRUE, prob = c(0.3, 0.25, 0.2, 0.15, 0.1)))
forms_names <- sort(unique(forms_areas))
forms_membership <- sapply(forms_names, function(area) as.numeric(forms_areas == area))
forms_hidden <- sample(forms_pool, forms_length)
forms_counts <- colSums(forms_membership[forms_hidden, , drop = FALSE])
forms_minimum <- pmax(0, forms_counts - 1)
forms_maximum <- forms_counts + 1
forms_thetas <- c(-1, 0, 1)
forms_information <- sapply(forms_thetas, function(theta) sum(Ii(theta, forms_bank)$Ii[forms_hidden]))
forms_enemies <- t(replicate(30, sample(forms_pool, 2)))

bench_fixture("psychometrics", "cat-forms", list(
  source = "Simulations, set.seed(20261031): a 3PL pool of 1,000 items in 5 content areas, 30 pairs of enemies and the bounds and information of a hidden feasible form of 40.",
  pool = list(a = I(forms_bank[, "a"]), b = I(forms_bank[, "b"]), c = I(forms_bank[, "c"])),
  areas = I(forms_areas),
  constraints = lapply(seq_along(forms_names), function(k) list(area = forms_names[k], minimum = forms_minimum[k], maximum = forms_maximum[k])),
  target = lapply(seq_along(forms_thetas), function(k) list(theta = forms_thetas[k], information = forms_information[k])),
  enemies = unname(lapply(seq_len(nrow(forms_enemies)), function(k) I(forms_enemies[k, ] - 1))),
  forms = forms_count,
  length = forms_length
))

forms_model <- local({
  n <- forms_pool
  variables <- forms_count * n + 1
  index <- function(f, i) (f - 1) * n + i
  rows <- list()
  lhs <- numeric(0)
  rhs <- numeric(0)
  add <- function(row, low, high) {
    rows[[length(rows) + 1]] <<- row
    lhs <<- c(lhs, low)
    rhs <<- c(rhs, high)
  }
  for (i in seq_len(n)) {
    row <- numeric(variables)
    row[index(seq_len(forms_count), i)] <- 1
    add(row, -Inf, 1)
  }
  info <- sapply(forms_thetas, function(theta) Ii(theta, forms_bank)$Ii)
  for (f in seq_len(forms_count)) {
    row <- numeric(variables)
    row[index(f, seq_len(n))] <- 1
    add(row, forms_length, forms_length)
    for (k in seq_along(forms_names)) {
      row <- numeric(variables)
      row[index(f, seq_len(n))] <- forms_membership[, k]
      add(row, forms_minimum[k], forms_maximum[k])
    }
    for (pair in seq_len(nrow(forms_enemies))) {
      row <- numeric(variables)
      row[index(f, forms_enemies[pair, ])] <- 1
      add(row, -Inf, 1)
    }
    for (k in seq_along(forms_thetas)) {
      row <- numeric(variables)
      row[index(f, seq_len(n))] <- info[, k]
      row[variables] <- -forms_information[k]
      add(row, -Inf, forms_information[k])
      row[variables] <- forms_information[k]
      add(row, forms_information[k], Inf)
    }
  }
  list(A = do.call(rbind, rows), lhs = lhs, rhs = rhs, variables = variables)
})

solve_forms <- function() {
  highs_solve(L = c(numeric(forms_model$variables - 1), 1), lower = rep(0, forms_model$variables), upper = rep(1, forms_model$variables),
              A = forms_model$A, lhs = forms_model$lhs, rhs = forms_model$rhs,
              types = c(rep("I", forms_model$variables - 1), "C"), control = highs_control(time_limit = 10))
}

bench_task("cat.assembleForms", "Assembly of parallel forms with enemies, by weighted deviations in rotation",
           sprintf("%d forms of %d items from a pool of %d 3PL items in %d content areas, with %d pairs of enemies and 3 targets", forms_count, forms_length, forms_pool, length(forms_names), nrow(forms_enemies)),
           reference("highs", "highs_solve"), solve_forms,
           note = "highs finds the forms whose largest gap from the target is smallest (minimax), to its limit of 10 seconds.")
