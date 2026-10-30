# Evaluation fixtures: AUC by pROC; RMSE, log-loss, accuracy, calibration bins and learning curves in base R.

source("validation/knowledge-tracing/common.R")
suppressPackageStartupMessages(library(pROC))

calibration_bins <- function(p, y, bins) {
  # Bounds k / K as measurekit computes them, bins closed on the left and the last one closed on both sides.
  breaks <- (0:bins) / bins
  bin <- cut(p, breaks = breaks, right = FALSE, include.lowest = TRUE, labels = FALSE)
  lapply(seq_len(bins), function(k) {
    inside <- bin == k
    list(
      lower = breaks[k],
      upper = breaks[k + 1],
      count = sum(inside),
      meanPrediction = if (any(inside)) mean(p[inside]) else NA,
      meanOutcome = if (any(inside)) mean(y[inside]) else NA
    )
  })
}

metric_case <- function(name, p, y) {
  binary <- all(y %in% c(0, 1))
  list(
    name = name,
    predictions = p,
    outcomes = y,
    auc = if (binary) as.numeric(auc(roc(response = y, predictor = p, levels = c(0, 1), direction = "<",
                                          quiet = TRUE))) else NA,
    rmse = sqrt(mean((p - y)^2)),
    logLoss = -mean(ifelse(y > 0, y * log(p), 0) + ifelse(y < 1, (1 - y) * log(1 - p), 0)),
    accuracy = if (binary) mean((p > 0.5) == (y == 1)) else NA,
    calibration = calibration_bins(p, y, 10),
    calibration4 = calibration_bins(p, y, 4)
  )
}

set.seed(20261029)
# Predictions rounded to two decimals tie often, and some are exactly 0.5 or a bin bound.
binary_p <- round(runif(400, 0.01, 0.99), 2)
binary_y <- rbinom(400, 1, binary_p)
partial_p <- runif(150, 0.02, 0.98)
partial_y <- round(runif(150), 1)
write_fixture("evaluate-metrics", list(
  source = "pROC::auc of roc(direction = \"<\"); RMSE, log-loss, accuracy (p > 0.5) and cut() bins in base R.",
  cases = list(
    metric_case("textbook", c(0.1, 0.4, 0.35, 0.8), c(0, 0, 1, 1)),
    metric_case("binary", binary_p, binary_y),
    metric_case("partial credit", partial_p, partial_y)
  )
))

# A log of 30 learners practicing 12 items on 4 skills, some items with two skills, with outcomes that improve with
# practice; the predictions are arbitrary inputs.
skills_of <- list(c("a"), c("a"), c("a", "b"), c("b"), c("b"), c("b", "c"), c("c"), c("c"), c("c", "d"), c("d"),
                  c("d"), c("a", "d"))
learners <- rep(sprintf("learner%02d", 1:30), each = 15)
items <- unlist(lapply(1:30, function(i) sample(12, 15, replace = TRUE)))
time <- unlist(lapply(1:30, function(i) i * 100 + 1:15))
log <- data.frame(learner = learners, item = items, time = time)
log <- log[order(log$time), ]
skill_lists <- skills_of[log$item]
# The outcome improves with the learner's earlier events.
earlier <- ave(seq_along(log$learner), log$learner, FUN = seq_along) - 1
log$outcome <- rbinom(nrow(log), 1, plogis(-1 + 0.25 * earlier))
predictions <- runif(nrow(log))

# One row per learner, skill and event, with the opportunity: the event's rank among the learner's events with
# the skill.
pairs <- do.call(rbind, lapply(seq_len(nrow(log)), function(i) {
  data.frame(event = i, learner = log$learner[i], skill = skill_lists[[i]])
}))
pairs$opportunity <- ave(pairs$event, pairs$learner, pairs$skill, FUN = seq_along)
pairs$outcome <- log$outcome[pairs$event]
pairs$prediction <- predictions[pairs$event]
curve <- function(rows) {
  lapply(sort(unique(rows$opportunity)), function(k) {
    at <- rows[rows$opportunity == k, ]
    list(opportunity = k, count = nrow(at), errorRate = 1 - mean(at$outcome), predictedErrorRate = 1 - mean(at$prediction))
  })
}
write_fixture("evaluate-learning-curve", list(
  source = "Base R: opportunities by ave(seq_along) per learner and skill, error rates by opportunity.",
  events = events_of(log$learner, sprintf("q%02d", log$item), skill_lists, log$time, log$outcome),
  predictions = predictions,
  overall = curve(pairs),
  skills = sapply(c("a", "b", "c", "d"), function(s) curve(pairs[pairs$skill == s, ]), simplify = FALSE)
))
