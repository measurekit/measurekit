# Evaluation of models, in base R with irr and pROC: R² and Cohen's kappa of predictions above 0.5 (irr::kappa2); the paired
# t test of two models' losses over learners (t.test(paired = TRUE)); a curriculum-ordered log cross-validated by time with
# test events of skills the training log has no outcome of left out; metrics by opportunity; power laws of learning curves
# (lm of log error rate on log opportunity, weighted by count) and DataShop's categories, written out from its help page
# (version 11.4.0).

source("validation/knowledge-tracing/common.R")
suppressPackageStartupMessages({
  library(irr)
  library(pROC)
})

# Predictions clipped to [eps, 1 - eps] with eps the machine epsilon, as scikit-learn's log_loss and measurekit clip them.
outcome_log_likelihood <- function(p, y) {
  q <- pmin(pmax(p, .Machine$double.eps), 1 - .Machine$double.eps)
  ifelse(y > 0, y * log(q), 0) + ifelse(y < 1, (1 - y) * log1p(-q), 0)
}

r_squared <- function(p, y) if (length(unique(y)) < 2) NA else 1 - sum((y - p)^2) / sum((y - mean(y))^2)

# Cohen's kappa of the predictions above 0.5 and 0/1 outcomes: NaN, written as null, where chance agreement is 1.
kappa_of <- function(p, y) {
  if (!all(y %in% c(0, 1))) return(NA)
  suppressWarnings(kappa2(cbind(factor(as.numeric(p > 0.5), levels = 0:1), factor(y, levels = 0:1)))$value)
}

metrics_of <- function(p, y) {
  binary <- all(y %in% c(0, 1))
  list(
    count = length(p),
    auc = if (binary && length(unique(y)) == 2) as.numeric(auc(roc(response = y, predictor = p, levels = c(0, 1),
                                                                   direction = "<", quiet = TRUE))) else NA,
    rmse = sqrt(mean((p - y)^2)),
    logLoss = -mean(outcome_log_likelihood(p, y)),
    accuracy = if (binary) mean((p > 0.5) == (y == 1)) else NA,
    rSquared = r_squared(p, y),
    kappa = kappa_of(p, y)
  )
}

# NULL where no event has a prediction.
metrics_or_null <- function(p, y) if (length(p) == 0) NULL else metrics_of(p, y)

set.seed(20261029)

# R² and kappa on the cases of evaluate-metrics, and on cases that are undefined or extreme.
binary_p <- round(runif(400, 0.01, 0.99), 2)
binary_y <- rbinom(400, 1, binary_p)
partial_p <- runif(150, 0.02, 0.98)
partial_y <- round(runif(150), 1)
agreement_case <- function(name, p, y) c(list(name = name, predictions = p, outcomes = y), metrics_of(p, y))
write_fixture("evaluate-agreement", list(
  source = "R² in base R; kappa by irr::kappa2 of the predictions above 0.5 and the outcomes.",
  cases = list(
    agreement_case("textbook", c(0.1, 0.4, 0.35, 0.8), c(0, 0, 1, 1)),
    agreement_case("binary", binary_p, binary_y),
    agreement_case("partial credit", partial_p, partial_y),
    agreement_case("worse than the mean outcome", c(0.9, 0.9, 0.1, 0.1), c(0, 0, 1, 1)),
    agreement_case("all outcomes equal", c(0.2, 0.9, 0.6), c(1, 1, 1)),
    agreement_case("all predicted correct", c(0.9, 0.8, 0.7, 0.6), c(1, 0, 1, 1)),
    agreement_case("every prediction and outcome correct", c(0.9, 0.8), c(1, 1)),
    agreement_case("predictions on 0.5", c(0.5, 0.5, 0.5, 0.51), c(1, 0, 0, 1))
  )
))

# The test between models. 40 learners of 8 to 12 events at 5 items; some events are lessons, which no model predicts.
# Two models' predictions are arbitrary inputs: the second is the first plus noise and a bias, so the learners differ.
learner_count <- 40
per_learner <- sample(8:12, learner_count, replace = TRUE)
events <- data.frame(learner = rep(sprintf("l%02d", seq_len(learner_count)), per_learner))
events$time <- sample(nrow(events))
events <- events[order(events$time), ]
events$time <- seq_len(nrow(events))
events$item <- sample(sprintf("q%d", 1:5), nrow(events), replace = TRUE)
events$outcome <- ifelse(runif(nrow(events)) < 0.07, NA, rbinom(nrow(events), 1, 0.65))
events$first <- ifelse(is.na(events$outcome), NA, round(pmin(pmax(rbeta(nrow(events), 3, 2), 0.02), 0.98), 3))
events$second <- ifelse(is.na(events$outcome), NA,
                        round(pmin(pmax(events$first + rnorm(nrow(events), -0.03, 0.12), 0.01), 0.99), 3))
event_list <- lapply(seq_len(nrow(events)), function(i) {
  list(learner = events$learner[i], item = events$item[i], skills = I(character(0)), time = events$time[i],
       outcome = events$outcome[i])
})

# Folds by learner (every learner in one test log) and by item, of which only the first two are tested (a learner in
# several test logs, with events of the third untested).
learner_fold <- (as.integer(factor(events$learner)) - 1) %% 3
item_fold <- (as.integer(factor(events$item)) - 1) %% 3
folds_of <- function(key) lapply(0:2, function(k) list(train = which(key != k) - 1, test = which(key == k) - 1))

paired_test <- function(folds, loss) {
  test <- unlist(lapply(folds, function(f) f$test)) + 1
  rows <- events[test, ]
  rows <- rows[!is.na(rows$outcome), ]
  lost <- function(p) if (loss == "log") -outcome_log_likelihood(p, rows$outcome) else (p - rows$outcome)^2
  by_learner <- aggregate(cbind(first = lost(rows$first), second = lost(rows$second)) ~ learner, rows, mean)
  tested <- t.test(by_learner$first, by_learner$second, paired = TRUE)
  list(loss = loss, learners = nrow(by_learner), meanDifference = unname(tested$estimate),
       standardError = unname(tested$stderr), statistic = unname(tested$statistic),
       degreesOfFreedom = unname(tested$parameter), pValue = tested$p.value)
}
by_learner_folds <- folds_of(learner_fold)
by_item_folds <- folds_of(item_fold)[1:2]
write_fixture("evaluate-comparison", list(
  source = paste("t.test(paired = TRUE) of the two models' mean losses per learner, over the test events with an",
                 "outcome. The models' predictions are inputs, indexed by the time of the event."),
  events = event_list,
  predictions = list(first = events$first, second = events$second),
  cases = list(
    list(name = "folds by learner", folds = by_learner_folds,
         tests = list(paired_test(by_learner_folds, "log"), paired_test(by_learner_folds, "squared"))),
    list(name = "folds by item", folds = by_item_folds,
         tests = list(paired_test(by_item_folds, "log"), paired_test(by_item_folds, "squared")))
  )
))

# A curriculum: 24 learners meet skills a to d in turn, 5 to 8 events per skill, at items of the new skill, of the new and
# the previous skill, or of an earlier skill; 5% of the events are lessons. Folds by time train on the earlier blocks, so a
# test block holds events of skills that its training log has no outcome of. The learner model is written outside the
# package: P(correct) = (1 + correct so far) / (2 + answered so far), the learner's own events with an outcome.
skill_names <- c("a", "b", "c", "d")
curriculum <- do.call(rbind, lapply(1:24, function(l) {
  do.call(rbind, lapply(1:4, function(phase) {
    n <- sample(5:8, 1)
    data.frame(learner = sprintf("l%02d", l), phase = phase, time = (phase - 1) * 1000 + l * 10 + seq_len(n))
  }))
}))
curriculum <- curriculum[order(curriculum$time), ]
curriculum$skills <- lapply(curriculum$phase, function(phase) {
  options <- list(skill_names[phase])
  if (phase > 1) options <- c(options, list(skill_names[c(phase - 1, phase)], skill_names[phase - 1]),
                             list(skill_names[sample(phase - 1, 1)]))
  options[[sample(length(options), 1)]]
})
curriculum$item <- sapply(curriculum$skills, function(s) paste0("item-", paste(s, collapse = "")))
curriculum$outcome <- ifelse(runif(nrow(curriculum)) < 0.05, NA, rbinom(nrow(curriculum), 1, 0.65))

# Time blocks as measurekit forms them: n / (folds + 1) events with an outcome, each bound moved past equal times.
time_bounds <- function(log, folds) {
  observed <- which(!is.na(log$outcome))
  blocks <- folds + 1
  bounds <- 0
  previous <- 0
  for (block in 1:(blocks - 1)) {
    bound <- max(floor(block * length(observed) / blocks), previous + 1)
    while (bound < length(observed) && log$time[observed[bound + 1]] == log$time[observed[bound]]) bound <- bound + 1
    previous <- bound
    position <- observed[bound + 1]
    while (position > 1 && log$time[position - 1] == log$time[position]) position <- position - 1
    bounds <- c(bounds, position - 1)
  }
  c(bounds, nrow(log))
}
bounds <- time_bounds(curriculum, 4)

# The model's predictions of a fold's test events; with `skip`, events with a skill that the training events with an
# outcome do not have are not predicted and do not update their learner.
laplace_fold <- function(log, train, test, skip) {
  seen <- unique(unlist(log$skills[train[!is.na(log$outcome[train])]]))
  state <- list()
  used <- list()
  skipped <- 0
  predictions <- sapply(test, function(i) {
    key <- log$learner[i]
    s <- state[[key]] %||% c(0, 0)
    for (j in train) {
      if (log$learner[j] == key && log$time[j] <= log$time[i] && !(j %in% used[[key]])) {
        used[[key]] <<- c(used[[key]], j)
        if (!is.na(log$outcome[j])) s <- s + c(1, log$outcome[j])
      }
    }
    state[[key]] <<- s
    if (is.na(log$outcome[i])) return(NA_real_)
    if (skip && !all(log$skills[[i]] %in% seen)) {
      skipped <<- skipped + 1
      return(NA_real_)
    }
    state[[key]] <<- s + c(1, log$outcome[i])
    (1 + s[2]) / (2 + s[1])
  })
  list(predictions = predictions, skipped = skipped)
}
`%||%` <- function(a, b) if (is.null(a)) b else a

curriculum_folds <- lapply(1:4, function(k) list(train = seq_len(bounds[k + 1]) - 1,
                                                  test = which(seq_len(nrow(curriculum)) > bounds[k + 1] &
                                                               seq_len(nrow(curriculum)) <= bounds[k + 2]) - 1))
evaluate_folds <- function(skip) {
  fits <- lapply(curriculum_folds, function(f) laplace_fold(curriculum, f$train + 1, f$test + 1, skip))
  pooled_p <- unlist(lapply(fits, function(f) f$predictions))
  pooled_y <- unlist(lapply(seq_along(fits), function(k) curriculum$outcome[curriculum_folds[[k]]$test + 1]))
  kept <- !is.na(pooled_p) & !is.na(pooled_y)
  list(
    folds = lapply(seq_along(fits), function(k) {
      y <- curriculum$outcome[curriculum_folds[[k]]$test + 1]
      p <- fits[[k]]$predictions
      kept <- !is.na(p) & !is.na(y)
      list(predictions = p, skipped = fits[[k]]$skipped, metrics = metrics_or_null(p[kept], y[kept]))
    }),
    metrics = metrics_of(pooled_p[kept], pooled_y[kept])
  )
}
stopifnot(sum(sapply(evaluate_folds(TRUE)$folds, function(f) f$skipped)) > 20)
write_fixture("evaluate-unseen", list(
  source = paste("Base R: the Laplace-rate model replayed over the training events of each test learner, then the test",
                 "events in time order; with skip, events of skills without an outcome in the training log are left out."),
  events = lapply(seq_len(nrow(curriculum)), function(i) {
    list(learner = curriculum$learner[i], item = curriculum$item[i], skills = I(curriculum$skills[[i]]),
         time = curriculum$time[i], outcome = curriculum$outcome[i])
  }),
  folds = curriculum_folds,
  throw = evaluate_folds(FALSE),
  skip = evaluate_folds(TRUE)
))

# Metrics by opportunity: 40 learners of 8 to 14 events at 12 items of 4 skills, some of two; the outcome improves with
# practice; 6% of the events are lessons and 5% of those with an outcome have no prediction, as under skip.
skills_of <- list(c("a"), c("a"), c("a", "b"), c("b"), c("b"), c("b", "c"), c("c"), c("c"), c("c", "d"), c("d"), c("d"),
                  c("a", "d"))
learner_n <- sample(8:14, 40, replace = TRUE)
log <- data.frame(learner = rep(sprintf("learner%02d", 1:40), learner_n))
log$time <- sample(nrow(log))
log <- log[order(log$time), ]
log$time <- seq_len(nrow(log))
log$item <- sample(12, nrow(log), replace = TRUE)
log$skills <- skills_of[log$item]
log$earlier <- ave(seq_along(log$learner), log$learner, FUN = seq_along) - 1
log$outcome <- ifelse(runif(nrow(log)) < 0.06, NA, rbinom(nrow(log), 1, plogis(-1 + 0.3 * log$earlier)))
log$prediction <- ifelse(is.na(log$outcome) | runif(nrow(log)) < 0.05, NA, round(runif(nrow(log), 0.02, 0.98), 4))

# One row per observed event and skill, with the opportunity: the rank among the learner's observed events at the skill.
rows <- do.call(rbind, lapply(which(!is.na(log$outcome)), function(i) {
  data.frame(event = i, learner = log$learner[i], skill = log$skills[[i]])
}))
rows$opportunity <- ave(rows$event, rows$learner, rows$skill, FUN = seq_along)
rows$practice <- ave(rows$event, rows$learner, rows$skill, FUN = length)
rows$outcome <- log$outcome[rows$event]
rows$prediction <- log$prediction[rows$event]
points_of <- function(rows, min) {
  rows <- rows[rows$practice >= min, ]
  lapply(sort(unique(rows$opportunity)), function(k) {
    at <- rows[rows$opportunity == k, ]
    given <- !is.na(at$prediction)
    list(opportunity = k, count = nrow(at), errorRate = 1 - mean(at$outcome),
         predictedErrorRate = if (any(given)) 1 - mean(at$prediction[given]) else NA,
         metrics = metrics_or_null(at$prediction[given], at$outcome[given]))
  })
}
curve_of <- function(min) c(list(overall = points_of(rows, min)),
                            list(skills = sapply(sort(unique(rows$skill[rows$practice >= min])),
                                                 function(s) points_of(rows[rows$skill == s, ], min), simplify = FALSE)))
stopifnot(any(sapply(curve_of(1)$overall, function(p) is.null(p$metrics$auc) || is.na(p$metrics$auc))))
write_fixture("evaluate-opportunity", list(
  source = paste("Base R and pROC: opportunities by ave(seq_along) over the observed events of a learner and skill; metrics",
                 "of the events with a prediction by opportunity; learner and skill pairs below the minimum left out."),
  events = lapply(seq_len(nrow(log)), function(i) {
    list(learner = log$learner[i], item = sprintf("q%02d", log$item[i]), skills = I(log$skills[[i]]), time = log$time[i],
         outcome = log$outcome[i])
  }),
  predictions = log$prediction,
  curves = list(list(minOpportunities = 1, curve = curve_of(1)), list(minOpportunities = 6, curve = curve_of(6)))
))

# Power laws and DataShop's categories of curves. The first curves are those above; the others are built to the cases.
power_curve <- function(counts, errors) Map(function(n, e, k) list(opportunity = k, count = n, errorRate = e),
                                            counts, errors, seq_along(counts))
curves <- c(
  list(overall = curve_of(1)$overall),
  setNames(curve_of(1)$skills, paste("skill", names(curve_of(1)$skills))),
  list(
    "exact power law" = power_curve(rep(40, 8), 0.5 * (1:8)^-0.7),
    "weights decide" = power_curve(c(60, 40, 8, 5, 3, 2), c(0.50, 0.38, 0.30, 0.33, 0.12, 0.40)),
    "zero errors left out" = power_curve(rep(20, 6), c(0.6, 0.4, 0.25, 0.1, 0, 0)),
    "two points" = power_curve(c(30, 25), c(0.5, 0.3)),
    "one point with errors" = power_curve(c(30, 25, 10), c(0.5, 0, 0)),
    "constant" = power_curve(rep(20, 5), rep(0.3, 5)),
    "rising" = power_curve(rep(20, 5), c(0.1, 0.2, 0.3, 0.45, 0.5)),
    "few learners at the end" = power_curve(c(50, 40, 30, 20, 3, 2), c(0.7, 0.55, 0.45, 0.35, 0.05, 0.9))
  )
)
power_law <- function(points) {
  points <- Filter(function(p) p$errorRate > 0, points)
  if (length(points) < 2) return(NULL)
  n <- sapply(points, function(p) p$opportunity)
  e <- sapply(points, function(p) p$errorRate)
  w <- sapply(points, function(p) p$count)
  fit <- lm(log(e) ~ log(n), weights = w)
  list(scale = unname(exp(coef(fit)[1])), exponent = unname(-coef(fit)[2]),
       rSquared = if (length(unique(log(e))) < 2) NA else summary(fit)$r.squared)
}
# DataShop's rules in its order: discard points of fewer learners than the student threshold; too few points left is too
# little data; a point beneath the low error threshold is low and flat; a last point above the high error threshold is
# still high; a slope beneath the AFM slope threshold shows no learning; the rest is good.
categorize_curve <- function(points, minLearners, minPoints, slope, minSlope, lowError = 0.2, highError = 0.4) {
  kept <- Filter(function(p) p$count >= minLearners, points)
  if (length(kept) < minPoints) return("too little data")
  errors <- sapply(kept, function(p) p$errorRate)
  if (any(errors < lowError)) return("low and flat")
  if (errors[length(errors)] > highError) return("still high")
  if (slope < minSlope) return("no learning")
  "good"
}
option_sets <- list(
  list(minLearners = 5, minPoints = 3, slope = 0.1, minSlope = 0.05),
  list(minLearners = 10, minPoints = 3, slope = 0.1, minSlope = 0.05),
  list(minLearners = 5, minPoints = 8, slope = 0.1, minSlope = 0.05),
  list(minLearners = 5, minPoints = 3, slope = 0.01, minSlope = 0.05),
  list(minLearners = 5, minPoints = 3, slope = 0.1, minSlope = 0.05, lowError = 0.45, highError = 0.9),
  list(minLearners = 5, minPoints = 3, slope = 0.1, minSlope = 0.05, lowError = 0.02, highError = 0.2),
  list(minLearners = 1, minPoints = 2, slope = -0.3, minSlope = -0.2, lowError = 0, highError = 1),
  list(minLearners = 20, minPoints = 5, slope = 0.3, minSlope = 0.3, lowError = 0.05, highError = 0.5)
)
cases <- lapply(names(curves), function(name) {
  points <- curves[[name]]
  points <- lapply(points, function(p) p[c("opportunity", "count", "errorRate")])
  list(name = name, points = points, powerLaw = power_law(points),
       categories = lapply(option_sets, function(o) list(options = o, category = do.call(categorize_curve, c(list(points), o)))))
})
found <- unique(unlist(lapply(cases, function(c) lapply(c$categories, function(k) k$category))))
stopifnot(setequal(found, c("too little data", "low and flat", "still high", "no learning", "good")))
write_fixture("evaluate-curves", list(
  source = "lm(log(errorRate) ~ log(opportunity), weights = count); DataShop's categorization rules, version 11.4.0.",
  cases = cases
))
