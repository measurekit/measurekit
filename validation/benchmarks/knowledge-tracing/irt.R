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

# Nominal response items of 4 categories, centered as mirt reports them, and responses drawn from them at N(0, 1)
# abilities: drawn after the 2PL data, so that those stay as they were.
set.seed(20261104)
nominal_slopes <- t(replicate(item_count, {
  slopes <- sort(rnorm(4, 0, 0.8))
  slopes - mean(slopes)
}))
nominal_intercepts <- t(replicate(item_count, {
  intercepts <- rnorm(4, 0, 0.6)
  intercepts - mean(intercepts)
}))
nominal_theta <- rnorm(persons)
nominal_data <- sapply(seq_len(item_count), function(j) {
  z <- outer(nominal_theta, nominal_slopes[j, ]) + matrix(nominal_intercepts[j, ], persons, 4, byrow = TRUE)
  p <- exp(z) / rowSums(exp(z))
  rowSums(runif(persons) > t(apply(p, 1, cumsum))[, 1:3])
})

bench_fixture("knowledge-tracing", "irt", list(
  source = paste("The 2PL simulation of the irt benchmark of psychometrics and mirt's estimates of its items; nominal items",
                 "of 4 categories and responses drawn from them, set.seed(20261104)."),
  responses = unname(data),
  items = list(a = I(unname(coefficients[, "a"])), b = I(unname(coefficients[, "b"]))),
  nominal = list(responses = unname(nominal_data), a = unname(nominal_slopes), c = unname(nominal_intercepts))
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

# The same replay of the responses to the nominal items, by mirt's nominal item type with the items' parameters fixed in
# its slope-intercept form: a1 = (a_4 - a_1) / 3, ak_k = (a_k - a_1) / a1 and d_k = c_k - c_1.
colnames(nominal_data) <- ids(item_count)
nominal_values <- mirt(nominal_data, 1, itemtype = "nominal", pars = "values")
for (j in seq_len(item_count)) {
  id <- colnames(nominal_data)[j]
  set <- function(name, value) nominal_values$value[nominal_values$item == id & nominal_values$name == name] <<- value
  a1 <- (nominal_slopes[j, 4] - nominal_slopes[j, 1]) / 3
  set("a1", a1)
  for (k in 1:4) {
    set(paste0("ak", k - 1), (nominal_slopes[j, k] - nominal_slopes[j, 1]) / a1)
    set(paste0("d", k - 1), nominal_intercepts[j, k] - nominal_intercepts[j, 1])
  }
}
nominal_values$est <- FALSE
nominal_fit <- mirt(nominal_data, 1, itemtype = "nominal", pars = nominal_values, TOL = NaN, verbose = FALSE)
nominal_prefixes <- lapply(seq_len(item_count), function(k) {
  pattern <- nominal_data
  if (k < item_count) pattern[, (k + 1):item_count] <- NA
  pattern
})
bench_task("irt.nominal", "Replay of the responses of learners to nominal items, the grid posterior of the ability updated after each",
           sprintf("%s learners, %d responses each, 4 categories", count(persons), item_count),
           reference("mirt", "fscores(method = \"EAP\")"),
           function() {
             for (pattern in nominal_prefixes) {
               fscores(nominal_fit, method = "EAP", response.pattern = pattern, theta_lim = c(-6, 6), quadpts = 61,
                       verbose = FALSE)
             }
           },
           note = "As for the 2PL items, mirt scores each of the 40 prefixes of the responses from the first.")

# The responses of the first 200 learners a minute apart, with a drift of variance 0.02 a week, on 241 points: the
# forward pass in logs of validation/knowledge-tracing/irt.R, with the kernel e^-t I_d(t) of besselI folded at the
# grid's ends, which is the same for every gap. besselI warns where the kernel falls below the range of doubles, and
# reads 0 there.
drift_learners <- 200
drift_points <- 241
drift_jumps <- 0.02 / (7 * 24 * 60) / (12 / (drift_points - 1))^2
drift_replay <- function() {
  z <- seq(-6, 6, length.out = drift_points)
  kernel <- suppressWarnings(besselI(drift_jumps, 0:(4 * drift_points), expon.scaled = TRUE))
  transition <- matrix(0, drift_points, drift_points)
  from <- 0:(drift_points - 1)
  for (offset in -(4 * drift_points):(4 * drift_points)) {
    folded <- (from + offset) %% (2 * drift_points)
    index <- cbind(from + 1, ifelse(folded < drift_points, folded, 2 * drift_points - 1 - folded) + 1)
    transition[index] <- transition[index] + kernel[abs(offset) + 1]
  }
  transition <- log(transition)
  for (person in seq_len(drift_learners)) {
    log_density <- -z^2 / 2
    for (item in seq_len(item_count)) {
      if (item > 1) {
        terms <- transition + log_density
        top <- apply(terms, 2, max)
        log_density <- top + log(colSums(exp(terms - rep(top, each = drift_points))))
      }
      weights <- exp(log_density - max(log_density))
      p <- plogis(coefficients[item, "a"] * (z - coefficients[item, "b"]))
      prediction <- sum(weights * p) / sum(weights)
      log_density <- log_density + log(if (data[person, item] == 1) p else 1 - p)
    }
  }
}
bench_task("irt.drift",
           "Replay of the responses of learners, the posterior advanced by drift and updated after each",
           sprintf("%s learners, %d responses each, 241 points", count(drift_learners), item_count),
           "base R, the forward pass in logs", drift_replay)
