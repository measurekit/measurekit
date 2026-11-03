# Livingston and Lewis's accuracy and consistency of a classification by one cut, from the total scores of 3,000 persons
# on 40 items of a 2PL simulation and their KR-20, against betafunctions' LL.CA with its four-parameter beta true-score
# distribution, without its test of model fit.

suppressPackageStartupMessages(library(betafunctions))

source("validation/benchmarks/simulation.R")

set.seed(20261102)
persons <- 3000
item_count <- 40
CUT <- 24

responses <- simulate_2pl(persons, item_count)$responses
scores <- rowSums(responses)
p <- colMeans(responses)
reliability <- item_count / (item_count - 1) * (1 - sum(p * (1 - p)) / var(scores))

bench_fixture("psychometrics", "standard-setting", list(
  source = "A 2PL simulation, set.seed(20261102): total scores of 40 items and their KR-20.",
  scores = I(unname(scores)),
  reliability = reliability,
  maximum = item_count,
  cut = CUT
))

bench_task("standardSetting.livingstonLewis", "Livingston and Lewis's accuracy and consistency by one cut",
           sprintf("%s scores of %d items", count(persons), item_count), reference("betafunctions", "LL.CA"),
           function() suppressWarnings(LL.CA(scores, reliability, CUT, min = 0, max = item_count, modelfit = NULL)),
           note = paste("LL.CA integrates by integrate's default relative tolerance, about 1.2e-4, and measurekit to 1e-12;",
                        "at LL.CA's tolerance measurekit takes about a quarter of LL.CA's time"))
