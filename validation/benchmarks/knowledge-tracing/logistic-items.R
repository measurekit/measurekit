# Logistic models with an intercept for each skill and each item, a proficiency for each learner and a ridge, as Best-LR of
# Gervet et al. (2020) is fitted: the log-counts of a learner's earlier successes and failures at each skill, and at all
# skills together. The fit from the events to the coefficients, against LiblineaR's L2-regularized logistic regression on
# the sparse design, at 1,000 and at 50,000 items.
#
# LiblineaR stops at epsilon 1e-4, LKT's default, at a gradient of the objective (its largest component) that the script
# measures; measurekit stops at that gradient, which the fixture holds, so that both end at the same distance from the
# minimum. At it measurekit takes 3.0 and 2.1 times as long (the medians of six runs of each), and the time goes to:
#   - the replay of the log for the predictions and the states of the learners, which LiblineaR does not give: 24% and 31%;
#   - the features, which run through states that copy their records: 1.1 and 1.3 times the time of LiblineaR's features
#     and design;
#   - the solver, which takes 2.3 times as long at 1,000 items and as long at 50,000. It needs 5 Newton iterations and 102
#     and 60 Hessian-vector products against LiblineaR's 5 and 14 and 84 and 90, which `LiblineaR(verbose = TRUE)` prints.
# At measurekit's default gradient of 1e-8 the fit takes 1.4 and 1.3 times as long as at LiblineaR's.

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

# The design of Best-LR from the events: the columns are the learners, skills and items, the log-counts of the learner's
# earlier successes and failures at the skill, and at any skill.
best_matrix <- function(d) {
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
  sparseMatrix(i = unlist(lapply(parts, `[[`, 1)), j = unlist(lapply(parts, `[[`, 2)),
               x = unlist(lapply(parts, function(p) rep_len(p[[3]], length(p[[1]])))),
               dims = c(n, skill_slopes + 2 * d$skills + 2))
}

# The same in compressed sparse rows, the form LiblineaR takes.
best_design <- function(d) {
  rows_form <- as(best_matrix(d), "RsparseMatrix")
  new("matrix.csr", ra = rows_form@x, ja = rows_form@j + 1L, ia = rows_form@p + 1L, dimension = dim(rows_form))
}

EPSILON <- 1e-4
fit_ridge <- function(d) LiblineaR(best_design(d), d$outcome, type = 0, cost = 1, bias = -1, epsilon = EPSILON)

# The largest component of the gradient of -log-likelihood + |w|² / 2 at the weights LiblineaR returns.
reached_gradient <- function(d) {
  fit <- fit_ridge(d)
  X <- best_matrix(d)
  w <- fit$W[1, seq_len(ncol(X))]
  # The first class name is the one that LiblineaR's weights score as positive.
  if (fit$ClassNames[1] == 0) w <- -w
  max(abs(as.numeric(crossprod(X, plogis(as.numeric(X %*% w)) - d$outcome)) + w))
}

set.seed(20261101)
small <- simulate_items(200, 20, 1000, 20000)
large <- simulate_items(3000, 100, 52700, 150000)
gradients <- c(small = reached_gradient(small), large = reached_gradient(large))

# Learner, item and skill are codes; the benchmark of measurekit gives them their ids. `gradient` is the tolerance at which
# measurekit stops, the one that LiblineaR's solution reaches.
bench_fixture("knowledge-tracing", "logistic-items", list(
  source = "A simulation, set.seed(20261101): events at random learners and items of 20 and of 100 skills; outcomes by learner, skill and item.",
  small = c(lapply(small[c("learner", "item", "skill", "outcome")], I), list(gradient = unname(gradients[["small"]]))),
  large = c(lapply(large[c("learner", "item", "skill", "outcome")], I), list(gradient = unname(gradients[["large"]])))
))

# Both tasks have one task text, so the note of the first covers both. The ratios and shares are those of the header.
note <- sprintf(paste("the time of the reference includes the features and the design from the events, as measurekit's does;",
                      "both stop at the gradient that LiblineaR reaches at epsilon %g (%.2g at 1,000 items, %.2g at 50,000),",
                      "where measurekit takes 3.0 and 2.1 times as long: it also replays the log for the predictions and states,",
                      "which the reference does not give (24%% and 31%% of its time), its features take 1.1 and 1.3 times as long as",
                      "the reference's, and its solver 2.3 times as long at 1,000 items and as long at 50,000; at its default",
                      "gradient of 1e-8 it takes 1.4 and 1.3 times as long as at these"),
                EPSILON, gradients[["small"]], gradients[["large"]])
bench_task("logistic.fitItems", "Fit with an intercept for each skill and item, a proficiency for each learner and a ridge of 1",
           sprintf("%s events of %d learners at %s items", count(length(small$outcome)), small$learners, count(small$items)),
           reference("LiblineaR", "LiblineaR(type = 0)"), function() fit_ridge(small), note = note)
bench_task("logistic.fitManyItems", "Fit with an intercept for each skill and item, a proficiency for each learner and a ridge of 1",
           sprintf("%s events of %s learners at %s items", count(length(large$outcome)), count(large$learners), count(large$items)),
           reference("LiblineaR", "LiblineaR(type = 0)"), function() fit_ridge(large))
