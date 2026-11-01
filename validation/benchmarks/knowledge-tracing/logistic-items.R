# Logistic models with an intercept for each skill and each item, a proficiency for each learner and a ridge, as Best-LR of
# Gervet et al. (2020) is fitted: the log-counts of a learner's earlier successes and failures at each skill, and at all
# skills together. The fit from the events to the coefficients, against LiblineaR's L2-regularized logistic regression on
# the sparse design, at 1,000 and at 50,000 items. LiblineaR stops at epsilon 1e-4, LKT's default; measurekit at a gradient
# of 1e-8.

suppressPackageStartupMessages({
  library(LiblineaR)
  library(Matrix)
  library(SparseM)
})

# Events at random learners and items, in time order. The outcome follows the learner, the skill and the item.
simulate_items <- function(learners, skill_count, item_count, events) {
  learner <- sample.int(learners, events, replace = TRUE)
  item <- sample.int(item_count, events, replace = TRUE)
  skill_of <- sample.int(skill_count, item_count, replace = TRUE)
  logit <- rnorm(learners, 0, 0.7)[learner] + rnorm(skill_count, -0.2, 0.3)[skill_of[item]] + rnorm(item_count, 0, 0.5)[item]
  item_ids <- sort(unique(item))
  list(learner = learner, item = match(item, item_ids), skill = skill_of[item], outcome = rbinom(events, 1, plogis(logit)),
       learners = learners, skills = skill_count, items = length(item_ids))
}

# The design of Best-LR in compressed sparse rows, from the events: the columns are the learners, skills and items, the
# log-counts of the learner's earlier successes and failures at the skill, and at any skill.
best_design <- function(d) {
  n <- length(d$outcome)
  before <- function(group) {
    successes <- ave(d$outcome, group, FUN = cumsum) - d$outcome
    list(successes = successes, failures = ave(rep(1, n), group, FUN = cumsum) - 1 - successes)
  }
  at_skill <- before(interaction(d$learner, d$skill, drop = TRUE))
  overall <- before(d$learner)
  rows <- seq_len(n)
  skill_slopes <- d$learners + d$skills + d$items
  parts <- list(
    list(rows, d$learner, 1), list(rows, d$learners + d$skill, 1), list(rows, d$learners + d$skills + d$item, 1),
    list(rows, skill_slopes + d$skill, log1p(at_skill$successes)),
    list(rows, skill_slopes + d$skills + d$skill, log1p(at_skill$failures)),
    list(rows, rep(skill_slopes + 2 * d$skills + 1, n), log1p(overall$successes)),
    list(rows, rep(skill_slopes + 2 * d$skills + 2, n), log1p(overall$failures))
  )
  design <- sparseMatrix(i = unlist(lapply(parts, `[[`, 1)), j = unlist(lapply(parts, `[[`, 2)),
                         x = unlist(lapply(parts, function(p) rep_len(p[[3]], length(p[[1]])))),
                         dims = c(n, skill_slopes + 2 * d$skills + 2))
  rows_form <- as(design, "RsparseMatrix")
  new("matrix.csr", ra = rows_form@x, ja = rows_form@j + 1L, ia = rows_form@p + 1L, dimension = dim(rows_form))
}

fit_ridge <- function(d) LiblineaR(best_design(d), d$outcome, type = 0, cost = 1, bias = -1, epsilon = 1e-4)

set.seed(20261101)
small <- simulate_items(200, 20, 1000, 20000)
large <- simulate_items(3000, 100, 52700, 150000)

# Learner, item and skill are codes; the benchmark of measurekit gives them their ids.
bench_fixture("knowledge-tracing", "logistic-items", list(
  source = "A simulation, set.seed(20261101): events at random learners and items of 20 and of 100 skills; outcomes by learner, skill and item.",
  small = lapply(small[c("learner", "item", "skill", "outcome")], I),
  large = lapply(large[c("learner", "item", "skill", "outcome")], I)
))

bench_task("logistic.fitItems", "Fit with an intercept for each skill and item, a proficiency for each learner and a ridge of 1",
           sprintf("%s events of %d learners at %s items", count(length(small$outcome)), small$learners, count(small$items)),
           reference("LiblineaR", "LiblineaR(type = 0)"), function() fit_ridge(small),
           note = "the time of the reference includes the features and the design from the events, as measurekit's does")
bench_task("logistic.fitManyItems", "Fit with an intercept for each skill and item, a proficiency for each learner and a ridge of 1",
           sprintf("%s events of %s learners at %s items", count(length(large$outcome)), count(large$learners), count(large$items)),
           reference("LiblineaR", "LiblineaR(type = 0)"), function() fit_ridge(large),
           note = "the time of the reference includes the features and the design from the events, as measurekit's does")
