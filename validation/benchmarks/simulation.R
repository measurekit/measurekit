# Simulations that more than one module of the benchmarks uses. A module sets its seed, then calls them.

# DINA responses over correlated attributes: a Q-matrix whose first items measure one attribute each, so that the
# attributes appear in the order of their names, and whose others measure one to three; a guessing of 0.2 and a slip of
# 0.1. Returns the Q-matrix, the attribute names, the profiles and the responses.
simulate_dina <- function(persons, item_count, attribute_count) {
  Q <- rbind(diag(attribute_count), t(replicate(item_count - attribute_count, {
    row <- integer(attribute_count)
    row[sample(attribute_count, sample(3, 1))] <- 1L
    row
  })))
  attributes <- paste0("a", seq_len(attribute_count))
  colnames(Q) <- attributes
  rownames(Q) <- ids(item_count)
  ability <- rnorm(persons)
  profiles <- sapply(seq_len(attribute_count), function(k) {
    as.integer(ability + rnorm(persons, 0, 0.7) > qnorm(k / 5 - 0.1))
  })
  mastered <- (profiles %*% t(Q)) == rep(rowSums(Q), each = persons)
  responses <- matrix(rbinom(persons * item_count, 1, ifelse(mastered, 0.9, 0.2)), persons, item_count,
                      dimnames = list(NULL, ids(item_count)))
  list(Q = Q, attributes = attributes, profiles = profiles, responses = responses)
}

# 2PL responses: slopes from a log-normal distribution, difficulties and abilities from the standard normal. Returns the
# parameters, the abilities and the responses.
simulate_2pl <- function(persons, item_count) {
  a <- round(rlnorm(item_count, 0, 0.3), 3)
  b <- round(rnorm(item_count), 3)
  theta <- rnorm(persons)
  responses <- matrix(rbinom(persons * item_count, 1, plogis(sweep(outer(theta, b, "-"), 2, a, "*"))), persons, item_count,
                      dimnames = list(NULL, ids(item_count)))
  list(a = a, b = b, theta = theta, responses = responses)
}

# Attempts of learners at skills that follow Bayesian knowledge tracing: a prior of 0.3, a learning rate of 0.15, a slip
# of 0.1 and a guess of 0.2. A learner practices each skill for `attempts` attempts on one of four items of the skill.
# Returns the events in the order of learners, skills and attempts.
simulate_bkt <- function(learners, skill_count, attempts) {
  per_learner <- skill_count * attempts
  rows <- learners * per_learner
  known <- logical(rows)
  state <- runif(learners * skill_count) < 0.3
  outcome <- integer(rows)
  for (chain in seq_len(learners * skill_count)) {
    current <- state[chain]
    for (a in seq_len(attempts)) {
      row <- (chain - 1) * attempts + a
      outcome[row] <- as.integer(runif(1) < ifelse(current, 0.9, 0.2))
      if (!current) current <- runif(1) < 0.15
    }
  }
  chain <- (seq_len(rows) - 1) %/% attempts
  data.frame(learner = paste0("l", chain %/% skill_count + 1), skill = paste0("s", chain %% skill_count + 1),
             item = paste0("s", chain %% skill_count + 1, "q", (seq_len(rows) - 1) %% 4 + 1),
             time = (seq_len(rows) - 1) %% per_learner + 1, outcome = outcome)
}
