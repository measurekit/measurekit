# Online item response theory: the posterior of a learner's ability on a grid, updated after each response, against mirt's
# EAP of the final response patterns, which scores the learner once. The items are mirt's 2PL estimates of the data of
# the irt benchmark of psychometrics, which is the same simulation.

suppressPackageStartupMessages(library(mirt))

source("validation/benchmarks/simulation.R")

set.seed(20261029)
persons <- 2000
item_count <- 40
data <- simulate_2pl(persons, item_count)$responses
fit <- mirt(data, 1, itemtype = "2PL", TOL = 1e-4, quadpts = 61, verbose = FALSE)
coefficients <- coef(fit, IRTpars = TRUE, simplify = TRUE)$items

bench_fixture("knowledge-tracing", "irt", list(
  source = "The 2PL simulation of the irt benchmark of psychometrics and mirt's estimates of its items.",
  responses = unname(data),
  items = list(a = I(unname(coefficients[, "a"])), b = I(unname(coefficients[, "b"])))
))

bench_task("irt.replay", "Replay of the responses of learners, the grid posterior of the ability updated after each",
           sprintf("%s learners, %d responses each", count(persons), item_count), "mirt 1.48 `fscores(method = \"EAP\")`",
           function() fscores(fit, method = "EAP", theta_lim = c(-6, 6), quadpts = 61, verbose = FALSE),
           note = "mirt scores each pattern once; measurekit updates the posterior after every response")
