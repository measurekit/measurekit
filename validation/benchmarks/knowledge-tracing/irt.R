# Online item response theory: the posterior of a learner's ability on a grid, updated after each response, against mirt's
# EAP. mirt scores a response pattern from its start, so the reference scores the patterns of the first k responses of every
# learner for each k, which gives the posterior after each response, as the replay does. The items are mirt's 2PL estimates
# of the data of the irt benchmark of psychometrics, which is the same simulation.

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

# The patterns of the first k responses of every learner, the others missing, for k from 1 to the number of items.
prefixes <- lapply(seq_len(item_count), function(k) {
  pattern <- data
  if (k < item_count) pattern[, (k + 1):item_count] <- NA
  pattern
})

bench_task("irt.replay", "Replay of the responses of learners, the grid posterior of the ability updated after each",
           sprintf("%s learners, %d responses each", count(persons), item_count), reference("mirt", "fscores(method = \"EAP\")"),
           function() {
             for (pattern in prefixes) {
               fscores(fit, method = "EAP", response.pattern = pattern, theta_lim = c(-6, 6), quadpts = 61, verbose = FALSE)
             }
           },
           note = paste("mirt scores each of the 40 prefixes of the responses from the first; measurekit updates the posterior",
                        "by the one response and predicts it. A scoring of the final patterns alone takes a fortieth of the time"))
