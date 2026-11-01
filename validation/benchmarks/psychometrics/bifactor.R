# Bifactor calibration against mirt's bfactor at its defaults, a tolerance of 1e-4 and 31 quadrature points, as
# measurekit runs: two bifactor 2PLs of 2,000 persons, 12 items in 3 specific dimensions and 40 items in 4.

suppressPackageStartupMessages(library(mirt))

set.seed(20261101)

# --- Data ----------------------------------------------------------------------------------------------------------

# Responses of a bifactor 2PL of `groups` specific dimensions of `per_group` items: general slopes in 0.8 to 1.8, specific
# slopes in 0.5 to 1.5 and standard normal intercepts, rounded to 3 digits. Returns the responses with item names and the
# specific dimension of each item as mirt's vector of numbers.
bifactor_data <- function(persons, groups, per_group) {
  items <- groups * per_group
  specific <- rep(seq_len(groups), each = per_group)
  slopes <- matrix(0, items, 1 + groups)
  slopes[, 1] <- round(runif(items, 0.8, 1.8), 3)
  slopes[cbind(seq_len(items), 1 + specific)] <- round(runif(items, 0.5, 1.5), 3)
  intercepts <- round(rnorm(items), 3)
  theta <- matrix(rnorm(persons * (1 + groups)), persons, 1 + groups)
  eta <- theta %*% t(slopes) + rep(intercepts, each = persons)
  responses <- matrix(rbinom(length(eta), 1, plogis(eta)), persons, items, dimnames = list(NULL, ids(items)))
  list(responses = responses, specific = specific)
}

twelve <- bifactor_data(2000, 3, 4)
forty <- bifactor_data(2000, 4, 10)

# measurekit takes the specific dimension of each item by item id, as names.
bench_case <- function(data) {
  names <- colnames(data$responses)
  list(itemIds = names, responses = unname(data$responses), specific = setNames(as.list(paste0("s", data$specific)), names))
}
bench_fixture("psychometrics", "bifactor", list(items12 = bench_case(twelve), items40 = bench_case(forty)))

# --- Calibration ---------------------------------------------------------------------------------------------------

bench_task("multidimensional.calibrateBifactor12", "Calibration of a bifactor 2PL by EM with dimension reduction",
           sprintf("%s persons, 12 items, 3 specific dimensions", count(nrow(twelve$responses))),
           reference("mirt", "bfactor(quadpts = 31)"),
           function() bfactor(twelve$responses, twelve$specific, quadpts = 31, verbose = FALSE))
bench_task("multidimensional.calibrateBifactor40", "Calibration of a bifactor 2PL by EM with dimension reduction",
           sprintf("%s persons, 40 items, 4 specific dimensions", count(nrow(forty$responses))),
           reference("mirt", "bfactor(quadpts = 31)"),
           function() bfactor(forty$responses, forty$specific, quadpts = 31, verbose = FALSE))
