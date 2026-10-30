# Elo fixtures. No R package of record implements the Elo learner models, so the updates are written out here from
# their equations (Pelánek, 2016; Nižnan, Pelánek and Řihák, 2015) and applied to small worked logs; the evaluation of
# held-out events follows the package's documented order; and a seeded simulation checks that a replay recovers the abilities
# and difficulties it simulated.

source("validation/knowledge-tracing/common.R")

# Recovery thresholds, fixed before the simulation was first run: the Pearson correlation of the replayed estimates
# with the simulated values.
ABILITY_THRESHOLD <- 0.85
DIFFICULTY_THRESHOLD <- 0.95

unrated <- list(difficulty = 0, count = 0)
initial_learner <- function() list(ability = 0, count = 0, skills = setNames(list(), character(0)))
skill_of <- function(learner, skill) learner$skills[[skill]] %||% list(ability = 0, count = 0)

# One event: P = logistic(theta + mean of the skill abilities - d); every estimate moves by U(n) (y - P), U(n) = a / (1
# + b n) with n its earlier updates, skill abilities by gamma U(n_k) (y - P), the difficulty the other way.
elo_step <- function(settings, learner, item, skills, outcome) {
  skill_term <- 0
  if (settings$skills && length(skills) > 0) {
    for (skill in skills) skill_term <- skill_term + skill_of(learner, skill)$ability
    skill_term <- skill_term / length(skills)
  }
  p <- logistic(learner$ability + skill_term - item$difficulty)
  error <- outcome - p
  change <- function(n) (settings$a / (1 + settings$b * n)) * error
  updated <- learner
  updated$ability <- learner$ability + change(learner$count)
  updated$count <- learner$count + 1
  if (settings$skills) {
    for (skill in skills) {
      s <- skill_of(learner, skill)
      updated$skills[[skill]] <- list(ability = s$ability + settings$gamma * change(s$count), count = s$count + 1)
    }
  }
  list(prediction = p, learner = updated, item = list(difficulty = item$difficulty - change(item$count),
                                                       count = item$count + 1))
}

outcome_log_likelihood <- function(p, y) ifelse(y > 0, y * log(p), 0) + ifelse(y < 1, (1 - y) * log1p(-p), 0)

# Replays events in order, updating learners and items.
elo_replay <- function(settings, events, items = list()) {
  learners <- list()
  predictions <- numeric(length(events))
  for (i in seq_along(events)) {
    e <- events[[i]]
    s <- elo_step(settings, learners[[e$learner]] %||% initial_learner(), items[[e$item]] %||% unrated, e$skills,
                  e$outcome)
    learners[[e$learner]] <- s$learner
    items[[e$item]] <- s$item
    predictions[i] <- s$prediction
  }
  outcomes <- sapply(events, `[[`, "outcome")
  list(predictions = predictions, learners = learners, items = items,
       logLikelihood = sum(outcome_log_likelihood(predictions, outcomes)))
}

# Predicts held-out events with the item ratings of a replay of the training events: each test learner starts from
# the initial state and goes through their training and test events in time order, training events first at equal
# times; test events are predicted, then update the learner; items stay fixed.
elo_holdout <- function(settings, train, test) {
  items <- elo_replay(settings, train)$items
  states <- list()
  done <- list()
  sapply(test, function(e) {
    state <- states[[e$learner]] %||% initial_learner()
    history <- Filter(function(x) x$learner == e$learner, train)
    k <- done[[e$learner]] %||% 0
    while (k < length(history) && history[[k + 1]]$time <= e$time) {
      h <- history[[k + 1]]
      state <- elo_step(settings, state, items[[h$item]] %||% unrated, h$skills, h$outcome)$learner
      k <- k + 1
    }
    done[[e$learner]] <<- k
    s <- elo_step(settings, state, items[[e$item]] %||% unrated, e$skills, e$outcome)
    states[[e$learner]] <<- s$learner
    s$prediction
  })
}

settings_of <- function(a = 1, b = 0.05, skills = FALSE, gamma = 1) list(a = a, b = b, skills = skills, gamma = gamma)
options_of <- function(settings) {
  options <- list(uncertainty = list(a = settings$a, b = settings$b), skills = settings$skills)
  if (settings$skills) options$skillWeight <- settings$gamma
  options
}
states_of <- function(learners) {
  lapply(setNames(names(learners), names(learners)), function(id) {
    l <- learners[[id]]
    list(learner = id, ability = l$ability, count = l$count, skills = l$skills)
  })
}

worked_case <- function(name, settings, events, items = setNames(list(), character(0))) {
  replay <- elo_replay(settings, events, items)
  list(name = name, options = options_of(settings), parameters = list(items = items), events = events,
       predictions = replay$predictions, states = states_of(replay$learners), items = replay$items,
       logLikelihood = replay$logLikelihood)
}

basic_events <- events_of(
  learner = c("ana", "budi", "ana", "ana", "budi", "budi", "ana", "budi"),
  item = c("q1", "q1", "q2", "q1", "q2", "q3", "q3", "q1"),
  skills = rep(list(character(0)), 8),
  time = 1:8,
  outcome = c(1, 0, 1, 0, 1, 1, 0, 1)
)
basic <- worked_case("learner-item", settings_of(), basic_events)
# The first update by hand: P = logistic(0 - 0) = 0.5 and U(0) = 1, so ana's ability rises by 0.5 and q1 falls by 0.5.
first <- elo_replay(settings_of(), basic_events[1])
stopifnot(first$predictions == 0.5, first$learners$ana$ability == 0.5, first$items$q1$difficulty == -0.5)

skill_events <- events_of(
  learner = c("ana", "ana", "budi", "ana", "budi", "ana", "budi", "ana"),
  item = c("q1", "q2", "q2", "q3", "q1", "q4", "q3", "q2"),
  skills = list("fractions", c("fractions", "decimals"), c("fractions", "decimals"), "decimals", "fractions",
                character(0), "decimals", c("fractions", "decimals")),
  time = c(10, 20, 25, 30, 40, 50, 60, 70),
  outcome = c(1, 0, 0.5, 1, 1, 0, 0, 1)
)
write_fixture("elo-worked", list(
  source = "The Elo equations written out in base R and replayed event by event.",
  cases = list(
    basic,
    worked_case("constant K", settings_of(a = 0.4, b = 0), basic_events),
    worked_case("skills, partial credit, starting ratings", settings_of(a = 0.8, b = 0.1, skills = TRUE, gamma = 0.8),
                skill_events, list(q1 = list(difficulty = 0.3, count = 4), q3 = list(difficulty = -0.2, count = 1)))
  )
))

set.seed(20261029)
# Six learners, four items with skills; times with ties.
holdout_learner <- sprintf("L%d", rep(1:6, each = 6))
holdout_item <- sample(sprintf("q%d", 1:4), 36, replace = TRUE)
item_skills <- list(q1 = "x", q2 = "x", q3 = c("x", "y"), q4 = "y")
holdout_time <- sample(1:16, 36, replace = TRUE)
holdout_outcome <- rbinom(36, 1, 0.6)
order <- order(holdout_time)
holdout <- events_of(holdout_learner[order], holdout_item[order], unname(item_skills[holdout_item[order]]),
                     holdout_time[order], holdout_outcome[order])
times <- sapply(holdout, `[[`, "time")

# Forward chaining by time: blocks of about n / 3 events, each boundary moved past events of equal time.
bounds <- 0
for (block in 1:2) {
  bound <- max(floor(block * length(holdout) / 3), bounds[length(bounds)] + 1)
  while (bound < length(holdout) && times[bound + 1] == times[bound]) bound <- bound + 1
  bounds <- c(bounds, bound)
}
bounds <- c(bounds, length(holdout))
# The ties move at least one boundary.
stopifnot(any(bounds[2:3] != floor(1:2 * length(holdout) / 3)))
fold <- function(name, test) {
  list(name = name, train = which(!test) - 1, test = which(test) - 1)
}
learners_of <- sapply(holdout, `[[`, "learner")
items_of <- sapply(holdout, `[[`, "item")
folds <- list(
  fold("learners L5 and L6", learners_of %in% c("L5", "L6")),
  fold("item q3", items_of == "q3"),
  fold("time block 2", seq_along(holdout) > bounds[2] & seq_along(holdout) <= bounds[3]),
  fold("time block 3", seq_along(holdout) > bounds[3])
)
# Time folds train on every earlier block only.
folds[[4]]$train <- seq_len(bounds[3]) - 1
folds[[3]]$train <- seq_len(bounds[2]) - 1
holdout_settings <- list(basic = settings_of(), skills = settings_of(skills = TRUE))
write_fixture("elo-holdout", list(
  source = "Base R: replay of the training events, then the test learners' events in time order with items fixed.",
  events = holdout,
  timeBounds = bounds,
  folds = lapply(folds, function(f) {
    c(f, list(predictions = lapply(holdout_settings, function(settings) {
      elo_holdout(settings, holdout[f$train + 1], holdout[f$test + 1])
    })))
  })
))

# Recovery: 400 learners with abilities from N(0, 1) answer all of 60 Rasch items with difficulties from N(0, 1), in
# random order; learners start at random times, so their events interleave.
learner_count <- 400
item_count <- 60
true_ability <- rnorm(learner_count)
true_difficulty <- rnorm(item_count)
start <- runif(learner_count, 0, 1000)
sim <- do.call(rbind, lapply(seq_len(learner_count), function(s) {
  data.frame(learner = s, item = sample(item_count), time = start[s] + seq_len(item_count))
}))
sim <- sim[order(sim$time), ]
sim$outcome <- rbinom(nrow(sim), 1, plogis(true_ability[sim$learner] - true_difficulty[sim$item]))
sim_events <- events_of(sprintf("s%d", sim$learner), sprintf("i%d", sim$item), rep(list(character(0)), nrow(sim)),
                        seq_len(nrow(sim)), sim$outcome)
replay <- elo_replay(settings_of(), sim_events)
ability <- sapply(sprintf("s%d", seq_len(learner_count)), function(id) replay$learners[[id]]$ability)
difficulty <- sapply(sprintf("i%d", seq_len(item_count)), function(id) replay$items[[id]]$difficulty)
correlations <- c(ability = cor(ability, true_ability), difficulty = cor(difficulty, true_difficulty))
message(sprintf("Recovery correlations: ability %.4f, difficulty %.4f", correlations[["ability"]],
                correlations[["difficulty"]]))
stopifnot(correlations[["ability"]] >= ABILITY_THRESHOLD, correlations[["difficulty"]] >= DIFFICULTY_THRESHOLD)
write_fixture("elo-recovery", list(
  source = "set.seed(20261029): Rasch simulation, replayed through Elo with U(n) = 1 / (1 + 0.05 n) in base R.",
  thresholds = list(ability = ABILITY_THRESHOLD, difficulty = DIFFICULTY_THRESHOLD),
  learners = sim$learner - 1,
  items = sim$item - 1,
  outcomes = sim$outcome,
  trueAbility = true_ability,
  trueDifficulty = true_difficulty,
  ability = unname(ability),
  difficulty = unname(difficulty),
  correlations = as.list(correlations)
))
