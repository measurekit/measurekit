# Elo ratings of learners and items: a replay of a log, predicting each attempt and then updating the learner and the item,
# against the update of the learner-item model written out in base R: P = logistic(theta - d), every estimate moving by
# U(n) (y - P) with U(n) = a / (1 + b n), measurekit's defaults of a = 1 and b = 0.05.

set.seed(20261029)
learners <- 2000
items <- 400
attempts <- 50

ability <- rnorm(learners)
difficulty <- rnorm(items)
learner <- rep(seq_len(learners), each = attempts)
item <- sample(items, learners * attempts, replace = TRUE)
outcome <- rbinom(length(learner), 1, plogis(ability[learner] - difficulty[item]))

bench_fixture("knowledge-tracing", "elo", list(
  source = "A simulation, set.seed(20261029): Rasch outcomes of 2,000 learners with 50 attempts each at 400 items.",
  learner = I(paste0("l", learner)), item = I(paste0("q", item)), time = I(rep(seq_len(attempts), learners)),
  outcome = I(outcome)
))

elo_replay <- function() {
  theta <- numeric(learners)
  learner_count <- integer(learners)
  d <- numeric(items)
  item_count <- integer(items)
  predictions <- numeric(length(learner))
  for (i in seq_along(learner)) {
    l <- learner[i]
    j <- item[i]
    p <- 1 / (1 + exp(-(theta[l] - d[j])))
    error <- outcome[i] - p
    theta[l] <- theta[l] + error / (1 + 0.05 * learner_count[l])
    d[j] <- d[j] - error / (1 + 0.05 * item_count[j])
    learner_count[l] <- learner_count[l] + 1L
    item_count[j] <- item_count[j] + 1L
    predictions[i] <- p
  }
  predictions
}

bench_task("elo.replay", "Replay of a log through the learner-item Elo model",
           sprintf("%s attempts of %s learners at %d items", count(length(learner)), count(learners), items),
           "base R, a loop over the attempts", elo_replay)
