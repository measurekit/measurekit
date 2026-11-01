# Mastery criteria over a log, replayed in base R: no package has them. A seeded log of 15 learners at 4 skills, with items
# of one or two skills, outcomes of 0 and 1, partial credit, and lessons with no outcome. Each criterion's table gives, by
# learner and skill, the opportunity after which it was first met, 1 being the first event with an outcome at the skill,
# or null. The predicted criterion uses a performance factors analysis with parameters written out: the probability of
# the attempt after its own outcome is plogis of the sum over its skills of the skill's intercept, 0.06 per success and 0.01
# per failure so far.

source("validation/knowledge-tracing/common.R")

set.seed(20261029)
learners <- sprintf("l%02d", 1:15)
skills <- c("s1", "s2", "s3", "s4")
items <- list(a = "s1", b = "s2", c = "s3", d = "s4", e = c("s1", "s2"), f = c("s3", "s4"), g = c("s2", "s3"))
n <- 1500
learner <- sample(learners, n, replace = TRUE)
item <- sample(names(items), n, replace = TRUE)
event_skills <- items[item]
time <- seq_len(n)
level <- new.env()
outcome <- rep(NA_real_, n)
for (i in seq_len(n)) {
  seen <- vapply(event_skills[[i]], function(s) get0(paste(learner[i], s), level, ifnotfound = 0), 0)
  for (s in event_skills[[i]]) assign(paste(learner[i], s), get0(paste(learner[i], s), level, ifnotfound = 0) + 1, level)
  u <- runif(1)
  outcome[i] <- if (u < 0.08) NA else if (u < 0.18) 0.5 else as.numeric(runif(1) < plogis(-0.8 + 0.1 * mean(seen)))
}

# The first opportunity of each learner and skill after which a criterion holds. `update(i)` takes event i, updates the state
# of the criterion and returns a function of a skill that says whether the criterion holds there.
first_opportunities <- function(update) {
  counts <- new.env()
  first <- new.env()
  for (i in seq_len(n)) {
    after <- update(i)
    if (is.na(outcome[i])) next
    for (s in event_skills[[i]]) {
      key <- paste(learner[i], s)
      count <- get0(key, counts, ifnotfound = 0) + 1
      assign(key, count, counts)
      reached <- after(s)
      recorded <- get0(key, first, ifnotfound = NA_integer_)
      assign(key, if (is.na(recorded) && reached) count else recorded, first)
    }
  }
  table <- list()
  for (key in ls(counts)) {
    parts <- strsplit(key, " ", fixed = TRUE)[[1]]
    table[[parts[1]]][[parts[2]]] <- get(key, first)
  }
  # NA becomes null; learners and skills in order.
  table <- table[order(names(table))]
  lapply(table, function(row) row[order(names(row))])
}

moving_average <- function(decay, threshold) {
  state <- new.env()
  first_opportunities(function(i) {
    function(s) {
      key <- paste(learner[i], s)
      m <- decay * get0(key, state, ifnotfound = 0) + (1 - decay) * outcome[i]
      assign(key, m, state)
      m >= threshold
    }
  })
}

window_criterion <- function(correct, window) {
  state <- new.env()
  first_opportunities(function(i) {
    function(s) {
      key <- paste(learner[i], s)
      recent <- tail(c(get0(key, state, ifnotfound = numeric(0)), as.numeric(outcome[i] == 1)), window)
      assign(key, recent, state)
      sum(recent) >= correct
    }
  })
}

streak <- function(length) {
  state <- new.env()
  first_opportunities(function(i) {
    function(s) {
      key <- paste(learner[i], s)
      run <- if (outcome[i] == 1) get0(key, state, ifnotfound = 0) + 1 else 0
      assign(key, run, state)
      run >= length
    }
  })
}

# Performance factors analysis: successes and failures of each skill before the event, updated by events with an outcome.
intercepts <- c(s1 = -0.3, s2 = 0.1, s3 = -0.2, s4 = 0.4)
SUCCESS <- 0.06
FAILURE <- 0.01
predicted <- function(threshold) {
  state <- new.env()
  first_opportunities(function(i) {
    p <- NA_real_
    if (!is.na(outcome[i])) {
      for (s in event_skills[[i]]) {
        key <- paste(learner[i], s)
        h <- get0(key, state, ifnotfound = c(0, 0))
        assign(key, h + c(outcome[i], 1 - outcome[i]), state)
      }
      z <- sum(vapply(event_skills[[i]], function(s) {
        h <- get(paste(learner[i], s), state)
        intercepts[[s]] + SUCCESS * h[1] + FAILURE * h[2]
      }, 0))
      p <- plogis(z)
    }
    function(s) p >= threshold
  })
}

events <- events_of(learner, item, event_skills, time, outcome)
cases <- function(name, builder, values) {
  lapply(values, function(v) list(criterion = name, options = v, table = do.call(builder, unname(v))))
}
write_fixture("practice-mastery", list(
  source = paste("A seeded log (set.seed(20261029)): 1,500 events of 15 learners at 4 skills, items of 1 or 2 skills, 8% lessons",
                 "with no outcome and 10% outcomes of 0.5. Each criterion replayed in base R."),
  events = events,
  pfa = list(intercepts = as.list(intercepts), successes = SUCCESS, failures = FAILURE),
  tables = c(
    cases("streak", streak, list(list(length = 3), list(length = 5))),
    cases("movingAverage", moving_average,
          list(list(decay = 0.7, threshold = 0.9), list(decay = 0.5, threshold = 0.95), list(decay = 0.9, threshold = 0.6))),
    cases("window", window_criterion,
          list(list(correct = 3, window = 4), list(correct = 4, window = 5), list(correct = 5, window = 5), list(correct = 1, window = 1))),
    cases("predicted", predicted, list(list(threshold = 0.8), list(threshold = 0.9), list(threshold = 0.95)))
  )
))
