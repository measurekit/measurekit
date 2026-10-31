# Response times: the lognormal model of the times, and the joint model of speed and accuracy. The lognormal model has a
# reference in nlme's lme, which estimates the intensities of the items and the variance of the speed, not the
# discriminations that measurekit also estimates. The joint model has no reference of comparable cost: LNIRT estimates it by
# MCMC.

suppressPackageStartupMessages({
  library(nlme)
  library(mirt)
})

source("validation/benchmarks/simulation.R")

set.seed(20261029)
persons <- 1000
item_count <- 20

speed <- rnorm(persons)
intensity <- round(rnorm(item_count, 4, 0.5), 3)
discrimination <- round(runif(item_count, 1, 2), 3)
log_times <- matrix(rnorm(persons * item_count), persons, item_count) / rep(discrimination, each = persons) +
  rep(intensity, each = persons) - speed
times <- exp(log_times)
theta <- 0.4 * speed + sqrt(1 - 0.16) * rnorm(persons)
b <- round(rnorm(item_count), 3)
responses <- matrix(rbinom(persons * item_count, 1, plogis(outer(theta, b, "-"))), persons, item_count)

bench_fixture("psychometrics", "response-time", list(
  source = "A simulation, set.seed(20261029): lognormal times with a speed correlated 0.4 with the ability of the 2PL responses.",
  itemIds = ids(item_count),
  responses = unname(responses),
  times = unname(times)
))

long <- data.frame(logtime = c(log_times), item = factor(rep(seq_len(item_count), each = persons)),
                   person = factor(rep(seq_len(persons), times = item_count)))
size <- sprintf("%s persons, %d items", count(persons), item_count)
bench_task("responseTime.calibrateResponseTimes", "The lognormal model of the response times", size,
           reference("nlme", "lme(logtime ~ 0 + item, random = ~ 1 | person)"),
           function() lme(logtime ~ 0 + item, random = ~ 1 | person, data = long, method = "REML"),
           note = "measurekit also estimates the item discriminations")
bench_task("responseTime.calibrateSpeedAccuracy", "The joint model of speed and accuracy, with the 2PL", size)
bench_task("responseTime.speedEstimates", "Speed of each person given the times of the items", size)
