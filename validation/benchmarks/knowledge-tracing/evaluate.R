# The area under the ROC curve of predictions: pROC's auc of roc, with the direction of the predictions given; and the
# metrics of predictions by opportunity: pROC's auc by group, the other metrics in base R.

suppressPackageStartupMessages(library(pROC))

set.seed(20261029)
count_of_predictions <- 100000
predictions <- runif(count_of_predictions)
outcomes <- rbinom(count_of_predictions, 1, predictions)

bench_fixture("knowledge-tracing", "evaluate", list(
  source = "A simulation, set.seed(20261029): outcomes of calibrated predictions.",
  predictions = I(predictions),
  outcomes = I(outcomes)
))

size <- sprintf("%s predictions", count(count_of_predictions))
bench_task("evaluate.auc", "The area under the ROC curve", size, reference("pROC", "auc(roc(direction = \"<\"))"),
           function() as.numeric(auc(roc(response = outcomes, predictor = predictions, levels = c(0, 1), direction = "<",
                                         quiet = TRUE))))
bench_task("evaluate.metrics", "The AUC, RMSE, log-loss and accuracy of the predictions", size)

# Metrics by opportunity of a log of 2,000 learners with 25 events each, at 5 skills, one skill per event; 6% of the events are
# lessons, which have no outcome and no prediction. The drawing comes after that of the task above, whose fixture it leaves
# as it was.
learner_count <- 2000
events_per_learner <- 25
event_count <- learner_count * events_per_learner
learner <- rep(sprintf("l%04d", seq_len(learner_count)), each = events_per_learner)
skill <- sprintf("s%d", sample(5, event_count, replace = TRUE))
earlier <- rep(seq_len(events_per_learner) - 1, learner_count)
lesson <- runif(event_count) < 0.06
outcome <- ifelse(lesson, NA, rbinom(event_count, 1, plogis(-1 + 0.1 * earlier)))
prediction <- ifelse(lesson, NA, round(plogis(-1 + 0.1 * earlier + rnorm(event_count, 0, 0.5)), 4))

bench_fixture("knowledge-tracing", "evaluate-opportunity", list(
  source = "A simulation, set.seed(20261029): 2,000 learners with 25 events each at 5 skills; 6% lessons.",
  learner = I(learner), skill = I(skill), time = I(seq_len(event_count)), outcome = I(outcome), prediction = I(prediction)
))

metrics_of <- function(p, y) {
  chosen <- p > 0.5
  agreement <- mean(chosen == (y == 1))
  chance <- mean(chosen) * mean(y == 1) + mean(!chosen) * mean(y == 0)
  c(count = length(p),
    auc = if (length(unique(y)) == 2) as.numeric(auc(roc(response = y, predictor = p, levels = c(0, 1), direction = "<",
                                                          quiet = TRUE))) else NA,
    rmse = sqrt(mean((p - y)^2)),
    logLoss = -mean(y * log(p) + (1 - y) * log1p(-p)),
    accuracy = agreement,
    rSquared = if (length(unique(y)) == 2) 1 - sum((y - p)^2) / sum((y - mean(y))^2) else NA,
    kappa = if (chance < 1) (agreement - chance) / (1 - chance) else NA)
}
by_opportunity <- function() {
  kept <- which(!is.na(outcome))
  rows <- data.frame(learner = learner[kept], skill = skill[kept], outcome = outcome[kept], prediction = prediction[kept])
  rows$opportunity <- ave(seq_len(nrow(rows)), rows$learner, rows$skill, FUN = seq_along)
  c(lapply(split(rows, rows$opportunity), function(at) metrics_of(at$prediction, at$outcome)),
    lapply(split(rows, list(rows$skill, rows$opportunity), drop = TRUE), function(at) metrics_of(at$prediction, at$outcome)))
}
bench_task("evaluate.metricsByOpportunity", "The metrics of predictions by opportunity, over all skills and for each skill",
           sprintf("%s events of %s learners at 5 skills", count(event_count), count(learner_count)),
           reference("pROC", "auc(roc(direction = \"<\"))"), by_opportunity)
