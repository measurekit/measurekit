# The linear logistic model of change (LLRA) of two occasions: the trend and the effect of a group on each of 8 items,
# by conditional maximum likelihood, against eRm's LLRA. The responses of 2,000 persons are simulated with a trait for each
# person and item, which the model conditions away.

suppressPackageStartupMessages(library(eRm))

set.seed(20261029)
persons <- 2000
change_items <- 8
group <- rep(c("ctl", "trt"), each = persons / 2)
trend <- c(0.5, 0.2, -0.3, 0.8, -0.6, 0.1, 0.4, -0.1)
effect <- c(0.7, 0, -0.4, 0.3, 0.5, -0.2, 0, 0.6)
before <- matrix(rbinom(persons * change_items, 1, plogis(matrix(rnorm(persons * change_items), persons))), persons)
after <- matrix(rbinom(persons * change_items, 1,
                       plogis(matrix(rnorm(persons * change_items), persons) + outer(rep(1, persons), trend) +
                                outer(group == "trt", effect))), persons)
colnames(before) <- colnames(after) <- ids(change_items)

bench_fixture("psychometrics", "explanatory-change", list(
  source = "A simulation, set.seed(20261029): 2,000 persons in two groups answer 8 items twice; a trait for each person and item.",
  itemIds = ids(change_items),
  groups = I(group),
  occasions = list(unname(before), unname(after))
))

erm_responses <- cbind(before, after)
colnames(erm_responses) <- paste0(rep(paste0("I", seq_len(change_items)), 2), ".t", rep(1:2, each = change_items))
bench_task("explanatory.linearChange", "The linear logistic model of change: a trend and a group effect on each item",
           sprintf("%s persons, %d items, 2 occasions, 2 groups", count(persons), change_items),
           reference("eRm", "LLRA"), function() LLRA(erm_responses, mpoints = 2, groups = group, baseline = "ctl"))
